import asyncio
import copy
import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from peft import PeftModel
from transformers import LlamaConfig, LlamaForCausalLM

import tinyrl.workers.model
from test.helpers import RedisTest, new_id, tiny
from test.test_api import proto_body
from tinyrl.api import wire
from tinyrl.bus import Op, Run
from tinyrl.storage.store import Store
from tinyrl.workers.model import LOSSES, Model, batches, target_logprobs, token_budget
from tinyrl.workers.trainer import Stop, Trainer

LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


def counting(n: int, start: int = 0, **inputs) -> wire.Datum:
  """A sequence that counts up from `start`: each token's target is the next one."""
  tokens = np.arange(start, start + n + 1) % 96
  return wire.Datum(tokens[:-1].astype(np.int32), {"target_tokens": tokens[1:].astype(np.int64), **inputs})


def train(model: Model, run: str, steps: int = 8, lr: float = 1e-2) -> list[float]:
  losses = []
  for _ in range(steps):
    _, metrics = model.forward_backward(run, [counting(12), counting(9, start=40)], "cross_entropy", {}, False)
    model.optim_step(run, {"learning_rate": lr})
    losses.append(metrics["loss:sum"])
  return losses


def lora_runs(n: int = 1) -> tuple[Model, list[str]]:
  model, runs = Model(tiny), [new_id() for _ in range(n)]
  for run in runs:
    model.create_run(run, "tiny", 4, LORA)
  return model, runs


def same(a, b) -> bool:
  av = a.values() if isinstance(a, dict) else a
  bv = b.values() if isinstance(b, dict) else b
  return all(torch.equal(x, y) for x, y in zip(av, bv))


class TestForwardPass(unittest.TestCase):
  def test_batches(self):
    self.assertEqual(batches([5, 5, 5], 100), [[0, 1, 2]])
    self.assertEqual(batches([5, 5, 5], 10), [[0, 1], [2]])
    self.assertEqual(batches([3, 1, 2], 0), [[1], [2], [0]])  # no budget means one sequence per pass
    # a long sequence gets a pass of its own instead of padding the short ones
    self.assertEqual(batches([10] * 6 + [50], 1000), [[0, 1, 2, 3, 4, 5], [6]])

    lengths = [int(n) for n in np.random.default_rng(0).integers(1, 64, 200)]
    passes = batches(lengths, 256)
    self.assertEqual(sorted(i for p in passes for i in p), list(range(200)))
    self.assertTrue(all(len(p) == 1 or max(lengths[i] for i in p) * len(p) <= 256 for p in passes))

  def test_the_token_budget_fits_the_free_gpu_memory(self):
    gpu = SimpleNamespace(
      device=torch.device("cuda", 0), dtype=torch.bfloat16, config=LlamaConfig(num_hidden_layers=32, hidden_size=4096)
    )

    def budget(free_gib: int, ckpt: bool = True) -> int:
      gpu.is_gradient_checkpointing = ckpt
      with (
        patch.object(torch.cuda, "mem_get_info", return_value=(free_gib * 2**30, 0)),
        patch.object(torch.cuda, "memory_reserved", return_value=2**30),  # 1 GiB cached by torch but unused
        patch.object(torch.cuda, "memory_allocated", return_value=0),
      ):
        return token_budget(gpu)

    self.assertEqual(budget(7, True), 8 * 2**30 // 2 // (6 * 32 * 4096 * 2))
    self.assertEqual(budget(7, False), 8 * 2**30 // 2 // (64 * 32 * 4096 * 2))
    self.assertEqual(budget(80), tinyrl.workers.model.MAX_TOKEN_BUDGET)
    self.assertEqual(token_budget(tiny()), tinyrl.workers.model.MAX_TOKEN_BUDGET)  # on the cpu
    with patch.object(tinyrl.workers.model, "TOKEN_BUDGET", 512):
      self.assertEqual(token_budget(gpu), 512)

  @patch("os.environ", {})
  def test_should_checkpoint_layers(self):
    from tinyrl.workers.model import should_checkpoint_layers

    class MockProps:
      def __init__(self, mem):
        self.total_memory = mem

    def mock_get_device_properties(dev):
      if str(dev) == "cuda:0":
        return MockProps(80 * 1024**3)
      return MockProps(40 * 1024**3)

    gpu_80 = SimpleNamespace(device=torch.device("cuda", 0), config=SimpleNamespace())
    gpu_40 = SimpleNamespace(device=torch.device("cuda", 1), config=SimpleNamespace())

    with patch("torch.cuda.get_device_properties", mock_get_device_properties):
      import tinyrl.workers.model

      old_grad = tinyrl.workers.model.GRAD_CKPT
      tinyrl.workers.model.GRAD_CKPT = 1
      self.assertFalse(should_checkpoint_layers(gpu_80, lora=True))
      self.assertTrue(should_checkpoint_layers(gpu_80, lora=False))
      self.assertTrue(should_checkpoint_layers(gpu_40, lora=True))

      os.environ["GRAD_CKPT"] = "1"
      self.assertTrue(should_checkpoint_layers(gpu_80, lora=True))

      os.environ["GRAD_CKPT"] = "0"
      self.assertFalse(should_checkpoint_layers(gpu_80, lora=True))
      self.assertFalse(should_checkpoint_layers(gpu_80, lora=False))

      del os.environ["GRAD_CKPT"]
      tinyrl.workers.model.GRAD_CKPT = 0
      self.assertFalse(should_checkpoint_layers(gpu_80, lora=True))
      tinyrl.workers.model.GRAD_CKPT = old_grad

  @patch("os.environ", {"GRAD_CKPT": "0"})
  def test_cpu_lora_without_gradient_checkpointing(self):
    model, runs = lora_runs(1)
    run = runs[0]
    weights_before = [p.clone() for p in model.lm.parameters()]
    _, metrics = model.forward_backward(run, [counting(12)], "cross_entropy", {}, False)
    step_metrics = model.optim_step(run, {"learning_rate": 1e-2})

    self.assertGreater(step_metrics["grad_norm:mean"], 0)
    weights_after = [p.clone() for p in model.lm.parameters()]
    changed = any(not torch.equal(b, a) for b, a in zip(weights_before, weights_after))
    self.assertTrue(changed)

  def test_chunked_logprobs_are_the_full_softmax(self):
    lm = tiny().eval()
    ids, targets = torch.randint(0, 96, (3, 11)), torch.randint(0, 96, (3, 11))
    mask = torch.ones_like(ids)
    mask[2, 7:] = 0

    with patch.object(tinyrl.workers.model, "CHUNK", 4), torch.no_grad():
      got = target_logprobs(lm, ids, mask, targets)
    with torch.no_grad():
      want = lm(ids, attention_mask=mask).logits.log_softmax(-1).gather(-1, targets[..., None])[..., 0]
    torch.testing.assert_close(got[mask.bool()], want[mask.bool()])

  def test_losses_are_their_definitions(self):
    g = torch.Generator().manual_seed(0)
    logprobs, sampler_logprobs = -torch.rand(4, 9, generator=g), -torch.rand(4, 9, generator=g)
    advantages, weights = torch.randn(4, 9, generator=g), torch.rand(4, 9, generator=g)
    ratio = torch.exp(logprobs - sampler_logprobs)

    def loss(name: str, **config) -> torch.Tensor:
      return LOSSES[name](logprobs, weights, sampler_logprobs, advantages, config)

    torch.testing.assert_close(loss("cross_entropy"), -logprobs * weights)
    torch.testing.assert_close(loss("importance_sampling"), -ratio * advantages * weights)
    clipped = torch.min(ratio * advantages, ratio.clamp(0.9, 1.1) * advantages)
    torch.testing.assert_close(loss("ppo", clip_range=0.1), -clipped * weights)
    kl = (ratio - 1) - (logprobs - sampler_logprobs)
    torch.testing.assert_close(loss("ppo", kl_coeff=0.5) - loss("ppo"), 0.5 * kl * weights)


class CudaMock(torch.Tensor):
  @property
  def is_cuda(self):
    return getattr(self, "_is_cuda", False)

  def is_pinned(self):
    return getattr(self, "_is_pinned", False)

  def to(self, *args, **kwargs):
    device = args[0] if args else kwargs.get("device")
    if device == "cuda":
      ret = torch.Tensor.to(self, "cpu", **kwargs).as_subclass(CudaMock)
      ret._is_cuda = True
      ret._is_pinned = False
      return ret
    return super().to(*args, **kwargs)


class TestModel(unittest.TestCase):
  @patch("torch.cuda.is_available", return_value=True)
  @patch("torch.cuda.synchronize")
  @patch("torch.cuda.empty_cache")
  @patch("torch.empty")
  def test_move_buffer_reuse_and_cleanup(self, mock_empty, mock_empty_cache, mock_sync, mock_is_avail):
    def fake_empty(*args, **kwargs):
      kwargs.pop("pin_memory", None)
      return torch.zeros(*args, **kwargs)

    mock_empty.side_effect = fake_empty

    model = Model(tiny)
    model.create_run("run", "tiny", 4, LORA)

    t = torch.randn(10, 10).as_subclass(CudaMock)
    t._is_cuda = True
    t._is_pinned = False

    id_t = id(t)
    original_tensors = model.gpu_tensors
    model.gpu_tensors = lambda optim=True: iter([t])
    model.move_tensors([t], "cpu")
    self.assertIn(id_t, model.pinned)
    buf_id = id(model.pinned[id_t])

    t.data = t.data.as_subclass(CudaMock)
    t._is_pinned = True
    t.data._is_pinned = True
    t.data._is_cuda = False
    model.move_tensors([t], "cuda")

    t.data = t.data.as_subclass(CudaMock)
    t.data._is_cuda = True
    t.data._is_pinned = False
    t._is_cuda = True
    model.move_tensors([t], "cpu")
    self.assertEqual(id(model.pinned[id_t]), buf_id)

    def fake_tensors(optim=True):
      return iter([])

    model.gpu_tensors = fake_tensors

    model.move_tensors([], "cpu")
    self.assertNotIn(id_t, model.pinned)
    model.gpu_tensors = original_tensors
    model.pinned["fake"] = torch.zeros(1)
    t_opt = torch.randn(5, 5)
    model.optimizers["run"] = torch.optim.Adam([torch.nn.Parameter(t_opt)])
    model.optimizers["run"].state[model.optimizers["run"].param_groups[0]["params"][0]] = {"step": t_opt}

    meta = {"optimizer": True}
    state = model.training_state("run")[0]
    model.restore_run("run", state, meta, True)
    self.assertNotIn("fake", model.pinned)

    model.pinned["fake"] = torch.zeros(1)
    model.delete_run("run")
    self.assertNotIn("fake", model.pinned)

  def test_lora_runs_share_a_frozen_base_and_train_apart(self):
    model, (a, b) = lora_runs(2)
    base = {name: p.detach().clone() for name, p in model.lm.named_parameters() if "lora" not in name}
    b_before = [p.detach().clone() for p in model.params[b].values()]

    losses = train(model, a)
    self.assertLess(losses[-1], losses[0] * 0.7)
    self.assertTrue(same(b_before, model.params[b]))
    self.assertTrue(all(torch.equal(base[name], p) for name, p in model.lm.named_parameters() if "lora" not in name))

  def test_interleaved_runs_keep_their_own_gradients(self):
    model, (a, b) = lora_runs(2)
    model.forward_backward(a, [counting(8)], "cross_entropy", {}, False)
    model.forward_backward(b, [counting(8, start=5)], "cross_entropy", {}, False)
    b_grads = [p.grad.clone() for p in model.params[b].values()]

    model.optim_step(a, {"learning_rate": 1e-2})
    self.assertTrue(all(p.grad is None for p in model.params[a].values()))
    self.assertTrue(same(b_grads, [p.grad for p in model.params[b].values()]))

  def test_forward_only_leaves_no_gradient(self):
    model, (a,) = lora_runs()
    logprobs, metrics = model.forward_backward(a, [counting(8), counting(5)], "cross_entropy", {}, True)
    self.assertTrue(all(p.grad is None for p in model.params[a].values()))
    self.assertEqual([lp.shape for lp in logprobs], [(8,), (5,)])
    self.assertAlmostEqual(metrics["loss:sum"], -sum(float(lp.sum()) for lp in logprobs), places=4)

  def test_rl_losses_move_logprobs_with_the_advantage(self):
    for loss in ("importance_sampling", "ppo"):
      model, (a,) = lora_runs()
      datum = counting(10)
      before = model.forward_backward(a, [datum], "cross_entropy", {}, True)[0][0]

      inputs = {**datum.inputs, "logprobs": before, "advantages": np.ones(10, np.float32)}
      for _ in range(5):
        model.forward_backward(a, [wire.Datum(datum.tokens, inputs)], loss, {}, False)
        model.optim_step(a, {"learning_rate": 1e-2})

      after = model.forward_backward(a, [datum], "cross_entropy", {}, True)[0][0]
      self.assertGreater(after.sum(), before.sum(), loss)

  def test_full_fine_tune(self):
    model, run = Model(tiny), new_id()
    model.create_run(run, "tiny", 0, {})
    losses = train(model, run)
    self.assertLess(losses[-1], losses[0] * 0.7)

    with self.assertRaises(ValueError):  # a full fine-tune has its trainer to itself
      model.create_run(new_id(), "tiny", 0, {})
    with self.assertRaises(ValueError):
      model.create_run(new_id(), "tiny", 4, LORA)

  def test_the_optimizer_updates_fp32_master_weights_and_casts_sampler_weights_to_bf16(self):
    def bf16(name: str) -> LlamaForCausalLM:  # how a base model loads on the gpu
      return tiny(name).to(torch.bfloat16)

    for rank in (0, 4):
      model, run = Model(bf16), new_id()
      model.create_run(run, "tiny", rank, {"seed": 0})
      model.forward_backward(run, [counting(12)], "cross_entropy", {}, False)
      grads = [p.grad for p in model.params[run].values()]
      model.optim_step(run, {"learning_rate": 1e-2})
      moments = [m for state in model.optimizers[run].state.values() for m in (state["exp_avg"], state["exp_avg_sq"])]
      self.assertEqual(
        {t.dtype for t in [*model.params[run].values(), *grads, *moments]}, {torch.float32}, f"rank {rank}"
      )
      self.assertEqual({t.dtype for t in model.sampler_weights(run)[0].values()}, {torch.bfloat16}, f"rank {rank}")
      if rank:  # a LoRA run's base model is frozen, so it stays bf16
        self.assertEqual(model.lm.get_input_embeddings().weight.dtype, torch.bfloat16)

  def test_a_run_restored_from_its_checkpoint_trains_on_exactly_as_before(self):
    store = Store(root=tempfile.mkdtemp())
    for rank in (0, 4):
      model, run = Model(tiny), new_id()
      model.create_run(run, "tiny", rank, LORA)
      train(model, run, steps=3)
      uri = f"tinker://{run}/weights/step-3"
      store.put(uri, *model.training_state(run))

      restored = Model(tiny)  # a new trainer, say on another node
      restored.create_run(run, "tiny", rank, LORA)
      restored.restore_run(run, *store.get(uri), optimizer=True)

      self.assertEqual(train(model, run, steps=3), train(restored, run, steps=3), f"rank {rank}")
      self.assertTrue(same(model.params[run].values(), restored.params[run].values()), f"rank {rank}")

  def test_save_and_load_bring_back_weights_and_optimizer(self):
    store = Store(root=tempfile.mkdtemp())
    model, (a, b) = lora_runs(2)
    train(model, a, steps=3)
    uri = f"tinker://{a}/weights/step-3"
    store.put(uri, *model.training_state(a))
    saved_weights = [p.detach().clone() for p in model.params[a].values()]
    saved_optimizer = copy.deepcopy(model.optimizers[a].state_dict())

    train(model, a, steps=3)
    model.restore_run(a, *store.get(uri), optimizer=True)
    self.assertTrue(same(saved_weights, model.params[a]))
    torch.testing.assert_close(model.optimizers[a].state_dict()["state"], saved_optimizer["state"])

    model.restore_run(b, *store.get(uri), optimizer=False)  # a run can start from another run's checkpoint
    self.assertTrue(same(saved_weights, model.params[b]))

  def test_loading_a_checkpoint_that_does_not_fit_fails(self):
    store = Store(root=tempfile.mkdtemp())
    model, (a,) = lora_runs()
    uri = f"tinker://{a}/sampler_weights/step-0"
    store.put(uri, *model.sampler_weights(a))

    no_mlp = new_id()
    model.create_run(no_mlp, "tiny", 4, {**LORA, "train_mlp": False})
    with self.assertRaises(ValueError):
      model.restore_run(no_mlp, *store.get(uri), optimizer=False)
    with self.assertRaises(ValueError):
      model.restore_run(a, *store.get(uri), optimizer=True)  # it was saved without the optimizer
    with self.assertRaises(ValueError):
      store.get(f"tinker://{a}/weights/missing")

  def test_sampler_weights_load_in_peft_and_transformers(self):
    store = Store(root=tempfile.mkdtemp())
    ids = torch.randint(0, 96, (2, 8))

    model, (lora_run,) = lora_runs()
    train(model, lora_run, steps=3)
    lora_dir = store.put(f"tinker://{lora_run}/sampler_weights/final", *model.sampler_weights(lora_run))
    with torch.no_grad():
      torch.testing.assert_close(
        PeftModel.from_pretrained(tiny(), lora_dir).eval()(ids).logits,
        model.activate_run(lora_run).eval()(ids).logits,
      )

    fft, fft_run = Model(tiny), new_id()
    fft.create_run(fft_run, "tiny", 0, {})
    train(fft, fft_run, steps=3)
    fft_dir = store.put(f"tinker://{fft_run}/sampler_weights/final", *fft.sampler_weights(fft_run))
    with torch.no_grad():
      torch.testing.assert_close(
        LlamaForCausalLM.from_pretrained(fft_dir).eval()(ids).logits,
        fft.activate_run(fft_run).eval()(ids).logits,
      )

  def test_mistakes_are_the_users(self):
    model, (a,) = lora_runs()
    with self.assertRaises(KeyError):
      model.forward_backward(new_id(), [counting(4)], "cross_entropy", {}, False)
    with self.assertRaises(ValueError):
      model.forward_backward(a, [counting(4)], "dpo", {}, False)
    with self.assertRaises(ValueError):
      model.create_run(new_id(), "other", 4, LORA)

    model.delete_run(a)
    with self.assertRaises(KeyError):
      model.optim_step(a, {})


class TestTrainer(RedisTest):
  async def submit(self, run: str, *cmds: tuple[Op, dict | bytes]) -> list[str]:
    request_ids = []
    for op, body in cmds:
      request_ids.append(
        await self.bus.submit_command(run, op, body if isinstance(body, bytes) else json.dumps(body).encode())
      )
    return request_ids

  async def test_commands_flow_through_and_delete_drops_the_run(self):
    run = new_id()
    await self.bus.create_run(Run(run, "p", "tiny", 4))
    request_ids = await self.submit(
      run,
      (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA}),
      (Op.FORWARD_BACKWARD, b"not protobuf"),
      (Op.OPTIM_STEP, {"adam_params": {}}),
      (Op.DELETE_MODEL, {}),
    )

    t = Trainer(self.bus, "p", Model(tiny))
    await self.own(t)
    while await t.step(block_ms=10):
      pass

    replies = [await self.bus.wait_for_reply(request_id, 0) for request_id in request_ids]
    bodies = [json.loads(reply.body) for reply in replies]
    self.assertEqual([reply.ok for reply in replies], [True, False, True, True])
    self.assertEqual([body.get("type") for body in bodies], ["create_model", None, "optim_step", "delete_model"])
    self.assertEqual(bodies[1]["category"], "user")

    self.assertEqual((await self.bus.get_run(run))["steps"], 1)
    self.assertIsNone(await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}"))  # deleted
    self.assertIsNone(t.stopping)  # a LoRA trainer keeps serving its worker

  async def test_a_trainer_told_to_stop_during_a_batch_that_deletes_a_run_still_stops(self):
    run = new_id()
    await self.bus.create_run(Run(run, "p", "tiny", 4, lora=LORA))
    await self.submit(run, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA}), (Op.DELETE_MODEL, {}))
    t = Trainer(self.bus, "p", Model(tiny))
    await self.own(t)
    answer = t.answer_command

    async def sigterm_then_answer(cmd):
      t.stop()  # SIGTERM lands while the batch runs
      return await answer(cmd)

    t.answer_command = sigterm_then_answer
    await t.step(block_ms=10)
    self.assertEqual(t.stopping, Stop.PARK)

  async def test_a_new_trainer_fails_what_a_dead_one_left_in_flight(self):
    run = new_id()
    await self.bus.create_run(Run(run, "p", "tiny", 4))
    [request_id] = await self.submit(run, (Op.OPTIM_STEP, {"adam_params": {}}))
    await self.bus.take_commands("p", "dead", block_ms=10)  # taken, and never answered

    t = Trainer(self.bus, "p", Model(tiny), owner="new")
    t.stopping = Stop.EXIT  # so serve() stops right after reclaiming
    await t.serve()

    reply = await self.bus.wait_for_reply(request_id, 1)
    self.assertFalse(reply.ok)
    self.assertEqual(json.loads(reply.body)["category"], "server")
    self.assertEqual((await self.bus.queued_and_in_flight(run))["in_flight"], [])

  async def test_a_full_fine_tune_trainer_exits_with_its_run(self):
    run = new_id()
    await self.bus.create_run(Run(run, run, "tiny", 0))
    await self.submit(run, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 0, "lora": {}}), (Op.DELETE_MODEL, {}))

    t = Trainer(self.bus, run, Model(tiny))
    await asyncio.wait_for(t.serve(), 60)
    self.assertEqual(t.stopping, Stop.EXIT)
    self.assertEqual(await self.bus.runs_in_worker(run), [])

  async def test_a_run_moves_between_trainers_through_the_store(self):
    store = Store(root=tempfile.mkdtemp())
    for rank, lora in ((4, LORA), (0, {})):
      ctrl, r1, r2 = new_id(), new_id(), new_id()
      for r, worker in ((ctrl, f"ctrl-{rank}"), (r1, f"p1-{rank}"), (r2, f"p2-{rank}")):
        await self.bus.create_run(Run(r, worker, "tiny", rank))

      tc = Trainer(self.bus, f"ctrl-{rank}", Model(tiny), store=store)
      t1 = Trainer(self.bus, f"p1-{rank}", Model(tiny), store=store)
      t2 = Trainer(self.bus, f"p2-{rank}", Model(tiny), store=store)
      await asyncio.gather(self.own(tc), self.own(t1), self.own(t2))

      for r in (ctrl, r1):
        await self.submit(r, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": rank, "lora": lora}))

      async def step_run(tr: Trainer, r: str) -> float:
        fb, _ = await self.submit(
          r, (Op.FORWARD_BACKWARD, proto_body(r)), (Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}})
        )
        while await tr.step(block_ms=10):
          pass
        return wire.output_json((await self.bus.wait_for_reply(fb, 1)).body)["metrics"]["loss:sum"]

      losses_ctrl = [await step_run(tc, ctrl) for _ in range(2)]
      losses_moved = [await step_run(t1, r1) for _ in range(2)]

      # save + delete on trainer 1, then create + load(optimizer=True) on trainer 2
      uri = f"tinker://{r1}/weights/handoff"
      await self.submit(r1, (Op.SAVE_WEIGHTS, {"uri": uri}), (Op.DELETE_MODEL, {}))
      while await t1.step(block_ms=10):
        pass
      await self.submit(
        r2,
        (Op.CREATE_MODEL, {"base_model": "tiny", "rank": rank, "lora": lora}),
        (Op.LOAD_WEIGHTS, {"uri": uri, "optimizer": True}),
      )
      while await t2.step(block_ms=10):
        pass

      for _ in range(3):
        losses_ctrl.append(await step_run(tc, ctrl))
        losses_moved.append(await step_run(t2, r2))

      self.assertEqual(losses_moved, losses_ctrl, f"rank {rank}")
      self.assertTrue(same(tc.model.params[ctrl], t2.model.params[r2]), f"rank {rank}")

  async def test_a_stopped_trainer_saves_its_runs_and_the_next_one_trains_on_exactly(self):
    store = Store(root=tempfile.mkdtemp())
    for rank, lora in ((4, LORA), (0, {})):
      ctrl, run = new_id(), new_id()
      await self.bus.create_run(Run(ctrl, f"ctrl-{rank}", "tiny", rank, lora=lora))
      await self.bus.create_run(Run(run, f"p-{rank}", "tiny", rank, lora=lora))
      tc, first = (Trainer(self.bus, worker, Model(tiny), store=store) for worker in (f"ctrl-{rank}", f"p-{rank}"))
      serving = [asyncio.create_task(tc.serve()), asyncio.create_task(first.serve())]

      async def call(r: str, op: Op, body: dict | bytes):
        [request_id] = await self.submit(r, (op, body))
        return await self.bus.wait_for_reply(request_id, 60)

      async def step(r: str) -> float:
        fb = await call(r, Op.FORWARD_BACKWARD, proto_body(r))
        self.assertTrue((await call(r, Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}})).ok)
        return wire.output_json(fb.body)["metrics"]["loss:sum"]

      for r in (ctrl, run):
        self.assertTrue((await call(r, Op.CREATE_MODEL, {"base_model": "tiny", "rank": rank, "lora": lora})).ok)
      losses_ctrl = [await step(ctrl) for _ in range(2)]
      losses = [await step(run) for _ in range(2)]

      first.stop()  # what SIGTERM does
      await serving.pop()
      self.assertIsNotNone(await self.bus.parked_checkpoint(run))

      second = Trainer(self.bus, f"p-{rank}", Model(tiny), store=store)
      serving.append(asyncio.create_task(second.serve()))
      for _ in range(3):
        losses_ctrl.append(await step(ctrl))
        losses.append(await step(run))
      self.assertEqual(losses, losses_ctrl, f"rank {rank}")
      # restored once, so a later crash can't bring back old state
      self.assertIsNone(await self.bus.parked_checkpoint(run))

      tc.stopping = second.stopping = Stop.EXIT
      await asyncio.gather(*serving)

  @patch("tinyrl.workers.trainer.LEASE_MS", 200)
  async def test_a_trainer_takes_its_lease_again_when_redis_loses_it(self):
    t = Trainer(self.bus, "p", Model(tiny))
    await self.own(t)
    await self.bus.r.delete("tinyrl:owner:p")  # what a Redis restart longer than the lease does
    await asyncio.sleep(0.2)
    self.assertEqual((await self.bus.lease_owner("p"), t.stopping), (t.owner, None))

    t.model.params["r"] = {}  # but not once another trainer has served the worker and parked its run
    await self.bus.r.set("tinyrl:parked:r", "tinker://r/parked/1")
    await self.bus.r.delete("tinyrl:owner:p")
    await asyncio.sleep(0.2)
    self.assertEqual((await self.bus.lease_owner("p"), t.stopping), (None, Stop.EXIT))

  async def test_a_run_whose_trainer_died_unsaved_fails_until_it_loads_a_checkpoint(self):
    store = Store(root=tempfile.mkdtemp())
    run, uri = new_id(), "tinker://saved/weights/step-1"
    await self.bus.create_run(Run(run, "p", "tiny", 4, lora=LORA))
    await self.submit(
      run, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA}), (Op.SAVE_WEIGHTS, {"uri": uri})
    )
    first = Trainer(self.bus, "p", Model(tiny), store=store)
    await self.own(first)
    while await first.step(block_ms=10):
      pass
    for task in self.renewal_tasks:  # it dies without saving
      task.cancel()
    await self.bus.release_lease("p", first.owner)

    second = Trainer(self.bus, "p", Model(tiny), store=store)
    await self.own(second)
    ids = await self.submit(
      run,
      (Op.OPTIM_STEP, {"adam_params": {}}),
      (Op.LOAD_WEIGHTS, {"uri": uri, "optimizer": True}),
      (Op.OPTIM_STEP, {"adam_params": {}}),
    )
    while await second.step(block_ms=10):
      pass
    replies = [await self.bus.wait_for_reply(request_id, 1) for request_id in ids]
    self.assertEqual([reply.ok for reply in replies], [False, True, True])
    self.assertIn("lost its unsaved steps", json.loads(replies[0].body)["error"])

  async def test_a_run_whose_parked_state_is_gone_still_deletes(self):
    run = new_id()
    await self.bus.create_run(Run(run, "p", "tiny", 4, lora=LORA))
    await self.bus.r.set(f"tinyrl:parked:{run}", "tinker://gone/parked/1")  # its node lost the files
    [request_id] = await self.submit(run, (Op.DELETE_MODEL, {}))
    t = Trainer(self.bus, "p", Model(tiny), store=Store(root=tempfile.mkdtemp()))
    await self.own(t)
    while await t.step(block_ms=10):
      pass
    self.assertTrue((await self.bus.wait_for_reply(request_id, 1)).ok)
    self.assertIsNone(await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}"))  # deleted

  async def test_sampler_save_is_processed_before_other_runs_finish(self):
    store = Store(root=tempfile.mkdtemp())
    t = Trainer(self.bus, "p", Model(tiny), store=store)
    await self.own(t)

    runs = [new_id() for _ in range(3)]
    for r in runs:
      await self.bus.create_run(Run(r, "p", "tiny", 4, lora=LORA))
      await self.submit(r, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA}))
    await t.step(count=16, block_ms=10)

    opt_ids = {}
    for r in runs:
      ids = await self.submit(r, (Op.FORWARD_BACKWARD, proto_body(r)), (Op.OPTIM_STEP, {"adam_params": {}}))
      opt_ids[r] = ids[-1]

    done_at: dict[str, float] = {}

    async def watch(name: str, req_id: str):
      reply = await self.bus.wait_for_reply(req_id, 5)
      done_at[name] = time.monotonic()
      return reply

    opt_tasks = {r: asyncio.create_task(watch(r, req_id)) for r, req_id in opt_ids.items()}

    async def submit_first_save():
      finished, _ = await asyncio.wait(opt_tasks.values(), return_when=asyncio.FIRST_COMPLETED)
      first_run = next(r for r, task in opt_tasks.items() if task in finished)
      [save_id] = await self.submit(first_run, (Op.SAVE_WEIGHTS_FOR_SAMPLER, {"uri": "tinker://r/sampler_weights/s0"}))
      reply = await watch("save", save_id)
      return first_run, reply

    save_task = asyncio.create_task(submit_first_save())
    n = await t.step(count=16, block_ms=10)
    first_run, save_reply = await save_task
    for task in opt_tasks.values():
      await task

    self.assertEqual(n, 7)
    self.assertTrue(save_reply.ok)
    other_times = [ts for name, ts in done_at.items() if name not in (first_run, "save")]
    self.assertLess(done_at["save"], max(other_times))

    causal = runs[0]
    pipelined = await self.submit(
      causal,
      (Op.FORWARD_BACKWARD, proto_body(causal)),
      (Op.OPTIM_STEP, {"adam_params": {"learning_rate": 0.1}}),
      (Op.SAVE_WEIGHTS_FOR_SAMPLER, {"uri": f"tinker://{causal}/sampler_weights/s1"}),
    )
    self.assertEqual(await t.step(count=16, block_ms=10), 3)
    self.assertTrue(all([(await self.bus.wait_for_reply(rid, 1)).ok for rid in pipelined]))
    self.assertEqual((await self.bus.get_run(causal))["steps"], 2)


class TestCheckpointNames(unittest.TestCase):
  def test_a_multimodal_checkpoint_trains_under_its_own_parameter_names(self):
    """Qwen3.5's checkpoint and vLLM name its layers model.language_model.*, its causal LM class model.layers.*. The trainer loads the class the checkpoint names, so a saved adapter's weights carry names vLLM maps."""
    from safetensors import safe_open
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    text = {
      "hidden_size": 32,
      "intermediate_size": 64,
      "num_hidden_layers": 2,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 16,
      "linear_num_key_heads": 1,
      "linear_num_value_heads": 2,
      "linear_key_head_dim": 16,
      "linear_value_head_dim": 16,
      "vocab_size": 128,
      "layer_types": ["linear_attention", "full_attention"],
    }
    vision = {"hidden_size": 16, "intermediate_size": 32, "depth": 1, "num_heads": 2, "out_hidden_size": 32}
    path = tempfile.mkdtemp()
    Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vision)).save_pretrained(path)
    with safe_open(f"{path}/model.safetensors", "pt") as f:
      names = set(f.keys())

    model, run = Model(tinyrl.workers.model.load), new_id()
    model.create_run(run, path, 4, LORA)
    train(model, run, steps=1)
    tensors, meta = model.sampler_weights(run)
    for key in tensors:  # base_model.model.<module>.lora_A.<...>: the module is one the checkpoint has
      module = key.removeprefix("base_model.model.").split(".lora_")[0]
      self.assertIn(f"{module}.weight", names)
    self.assertTrue(any(".language_model.layers." in key for key in tensors))

  def test_gemma4_lora_skips_clippable_vision_and_audio_wrappers(self):
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    cfg = Gemma4Config(
      text_config={
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "vocab_size": 128,
        "num_kv_shared_layers": 1,
        "layer_types": ["sliding_attention", "full_attention"],
        "hidden_size_per_layer_input": 0,
      },
      vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2},
      audio_config={"hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2, "conf_conv_kernel_size": 3},
    )
    path = tempfile.mkdtemp()
    Gemma4ForConditionalGeneration(cfg).save_pretrained(path)

    model, a, b = Model(tinyrl.workers.model.load), new_id(), new_id()
    model.create_run(a, path, 4, LORA)
    model.create_run(b, path, 4, LORA)
    tensors, _ = model.sampler_weights(b)
    self.assertTrue(tensors)
    self.assertTrue(all(".language_model." in k for k in tensors))


if __name__ == "__main__":
  unittest.main()
