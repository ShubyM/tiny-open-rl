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
from test.execution import TestNode, gate, watch
from test.helpers import RedisTest, new_id, tiny
from test.test_api import proto_body
from tinyrl.api import wire
from tinyrl.bus import Cmd, Ev, Op, Run
from tinyrl.storage.store import Store
from tinyrl.workers.model import LOSSES, Model, batches, target_logprobs, token_budget
from tinyrl.workers.trainer import Trainer

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
  def test_pad_rounds_width_on_cuda(self):
    from tinyrl.workers.model import pad

    rows, lengths = [np.ones(5), np.ones(130)], [5, 130]
    self.assertEqual(pad(rows, lengths, np.float32, torch.device("cpu")).shape, (2, 130))
    with patch("torch.Tensor.to", lambda t, _: t):
      out = pad(rows, lengths, np.float32, torch.device("cuda"))
      self.assertEqual(out.shape, (2, 256))
      self.assertEqual(out[1, 129].item(), 1.0)
      self.assertEqual(out[1, 130].item(), 0.0)

  def test_liger_kernel_fallback(self):
    lm = tiny()
    ids, mask, targets = (
      torch.zeros((1, 5), dtype=torch.long),
      torch.ones((1, 5), dtype=torch.long),
      torch.zeros((1, 5), dtype=torch.long),
    )
    expected = target_logprobs(lm, ids, mask, targets)
    with (
      patch.object(
        type(lm), "device", new_callable=unittest.mock.PropertyMock, return_value=SimpleNamespace(type="cuda")
      ),
      patch("tinyrl.workers.model.LigerFusedLinearCrossEntropyLoss", None),
    ):
      actual = target_logprobs(lm, ids, mask, targets)
    self.assertTrue(torch.allclose(actual, expected))

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
  def test_full_ft_delete_run_drops_model_lm(self):
    m = Model(tiny)
    m.create_run("test-full-ft", "tiny", 0, {})
    m.activate_run("test-full-ft")
    self.assertIsNotNone(m.lm)
    m.delete_run("test-full-ft")
    self.assertIsNone(m.lm)

  def test_a1_and_a2_gradients(self):
    model, run_id = Model(tiny), new_id()
    model.create_run(run_id, "tiny", 0, {})
    model.activate_run(run_id)
    for p in model.params[run_id].values():
      p.grad = torch.ones_like(p.data) * 0.5

    tensors, meta = model.training_state(run_id)
    self.assertTrue(any("grad" in name for name in tensors.keys()))

    model.restore_run(run_id, tensors, meta, True, gradients=True)
    self.assertTrue(any(p.grad is not None for p in model.params[run_id].values()))

    model.restore_run(run_id, tensors, meta, True, gradients=False)
    self.assertFalse(any(p.grad is not None for p in model.params[run_id].values()))

  def test_a4_mixed_state(self):
    model, run_id = Model(tiny), new_id()
    model.create_run(run_id, "tiny", 0, {})
    model.activate_run(run_id)
    opt = model.adam_for_run(run_id)
    for p in model.params[run_id].values():
      opt.state[p] = {"exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p), "step": torch.tensor(1.0)}

    tensors, meta = model.training_state(run_id)
    bad_tensors = tensors.copy()
    for k in list(bad_tensors.keys()):
      if k.endswith(".step"):
        del bad_tensors[k]
        break

    for p in model.params[run_id].values():
      p.data.add_(1.0)
    original_params = {k: p.clone() for k, p in model.params[run_id].items()}

    with self.assertRaisesRegex(ValueError, "bad:"):
      model.restore_run(run_id, bad_tensors, meta, True)

    for k, p in model.params[run_id].items():
      self.assertTrue(torch.equal(p, original_params[k]))

  def test_a5_partial_gradients(self):
    model, run_id = Model(tiny), new_id()
    model.create_run(run_id, "tiny", 0, {})

    import tinyrl.workers.model

    old_budget = tinyrl.workers.model.TOKEN_BUDGET
    tinyrl.workers.model.TOKEN_BUDGET = 1

    calls = [0]
    original = tinyrl.workers.model.target_logprobs

    def fake_target_logprobs(*args, **kwargs):
      calls[0] += 1
      if calls[0] == 2:
        raise RuntimeError("OOM")
      return original(*args, **kwargs)

    tinyrl.workers.model.target_logprobs = fake_target_logprobs

    data = [
      wire.Datum(tokens=np.array([1, 2], dtype=np.int32), inputs={"target_tokens": np.array([1, 2], dtype=np.int64)}),
      wire.Datum(tokens=np.array([3, 4], dtype=np.int32), inputs={"target_tokens": np.array([3, 4], dtype=np.int64)}),
    ]

    with self.assertRaises(RuntimeError):
      model.forward_backward(run_id, data, "cross_entropy", {}, False)

    has_grad = any(p.grad is not None for p in model.params[run_id].values())
    self.assertFalse(has_grad)

    tinyrl.workers.model.target_logprobs = original

    model.forward_backward(run_id, data, "cross_entropy", {}, False)

    with self.assertRaisesRegex(RuntimeError, "a forward_backward failed after it began"):
      model.optim_step(run_id, {"learning_rate": 1e-2})

    model.forward_backward(run_id, data, "cross_entropy", {}, False)
    model.optim_step(run_id, {"learning_rate": 1e-2})

    tinyrl.workers.model.TOKEN_BUDGET = old_budget

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
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.store = Store(root=tempfile.mkdtemp())
    self.node = TestNode(self.bus, store=self.store)

  async def asyncTearDown(self):
    await self.node.aclose()
    await super().asyncTearDown()

  async def submit(self, run: str, *cmds: tuple[Op, dict | bytes]) -> list[str]:
    return [
      await self.bus.submit_command(run, op, body if isinstance(body, bytes) else json.dumps(body).encode())
      for op, body in cmds
    ]

  async def replies(self, ids: list[str], timeout: float = 5) -> list:
    return [await self.bus.wait_for_reply(request_id, timeout) for request_id in ids]

  def model(self, worker):
    return self.node.local[worker].engine.process.trainer.model

  async def test_commands_flow_through_the_node_and_delete_drops_the_run(self):
    run = new_id()
    await self.bus.create_run(Run(run, "p", "tiny", 4))
    request_ids = await self.submit(
      run,
      (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA}),
      (Op.FORWARD_BACKWARD, b"not protobuf"),
      (Op.OPTIM_STEP, {"adam_params": {}}),
      (Op.DELETE_MODEL, {}),
    )
    await self.node.drain(run)
    replies = await self.replies(request_ids, 0)
    bodies = [json.loads(reply.body) for reply in replies]
    self.assertEqual([reply.ok for reply in replies], [True, False, True, True])
    self.assertEqual([body.get("type") for body in bodies], ["create_model", None, "optim_step", "delete_model"])
    self.assertEqual(bodies[1]["category"], "user")
    self.assertEqual((await self.bus.get_run(run))["steps"], 1)
    self.assertIsNone(await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}"))

  async def test_a_stop_finishes_the_running_command_and_leaves_the_rest_queued(self):
    run = new_id()
    await self.bus.create_run(Run(run, "p", "tiny", 4, lora=LORA))
    ids = await self.submit(
      run,
      (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA}),
      *[(Op.OPTIM_STEP, {"adam_params": {}}) for _ in range(5)],
    )
    lw = await self.node.add("p")
    entered, go = gate(lw.engine.process, "execute")
    await asyncio.wait_for(entered.wait(), 5)
    self.assertEqual(len((await self.bus.queued_and_in_flight(run))["in_flight"]), 1)
    self.node.note(Ev.UNPLACED, worker="p")
    go.set()
    await asyncio.wait_for(lw.task, 10)
    self.assertTrue((await self.bus.wait_for_reply(ids[0], 0)).ok)
    self.assertEqual(await self.bus.queued_and_in_flight(run), {"live": True, "queued": 5, "in_flight": []})
    self.assertIsNotNone(await self.bus.parked_checkpoint(run))
    self.assertIsNone(await self.bus.lease_owner("p"))
    await self.node.add("p")
    self.assertTrue(all([reply.ok for reply in await self.replies(ids[1:], 10)]))
    self.assertEqual((await self.bus.get_run(run))["steps"], 5)

  async def test_a_revoked_trainer_finishes_the_running_command_then_parks_and_takes_its_next_turn(self):
    run = new_id()
    await self.bus.create_run(Run(run, "p", "tiny", 4, lora=LORA))
    ids = await self.submit(
      run,
      (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA}),
      *[(Op.OPTIM_STEP, {"adam_params": {}}) for _ in range(5)],
    )
    lw = await self.node.add("p")
    calls = watch(lw.engine.process)
    entered, go = gate(lw.engine.process, "execute")
    await asyncio.wait_for(entered.wait(), 5)
    self.node.note(Ev.REVOKED, worker="p")
    go.set()
    self.assertTrue(all([reply.ok for reply in await self.replies(ids, 10)]))
    methods = [method for _, method, _ in calls]
    self.assertEqual(methods[:5], ["resume", "execute", "park", "resume", "execute"])
    self.assertEqual(methods.count("execute"), 6)

  async def test_a_revoked_busy_trainer_parks_after_at_most_one_more_command(self):
    await self.node.aclose()
    self.node = TestNode(self.bus, store=self.store, gpus=1)
    self.node.state.quantum = 0.01
    create = (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA})
    a, b = new_id(), new_id()
    for run, worker in ((a, "a"), (b, "b")):
      await self.bus.create_run(Run(run, worker, "tiny", 4, lora=LORA))
    [created] = await self.submit(a, create)
    lw = await self.node.add("a")
    entered, go = gate(lw.engine.process, "execute")
    await asyncio.wait_for(entered.wait(), 5)
    calls = watch(lw.engine.process, {"execute": 0.1})
    steps = await self.submit(a, *[(Op.OPTIM_STEP, {"adam_params": {}}) for _ in range(16)])
    go.set()
    self.assertTrue((await self.bus.wait_for_reply(created, 10)).ok)
    await asyncio.sleep(0.15)
    asked = time.time()
    await self.node.add("b")
    [created] = await self.submit(b, create)
    self.assertTrue((await self.bus.wait_for_reply(created, 10)).ok)
    parked = next(t for t, method, _ in calls if method == "park" and t > asked)
    sent = [t for t, method, _ in calls if method == "execute" and asked < t < parked]
    self.assertLessEqual(len(sent), 2, f"{len(sent)} commands ran after another trainer asked for the GPU")
    self.assertTrue(all([reply.ok for reply in await self.replies(steps, 30)]))

  async def test_a_command_s_busy_ms_is_its_own_running_time(self):
    run = new_id()
    await self.bus.create_run(Run(run, "p", "tiny", 4, lora=LORA))
    [created] = await self.submit(run, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA}))
    lw = await self.node.add("p")
    entered, go = gate(lw.engine.process, "execute")
    await asyncio.wait_for(entered.wait(), 5)
    watch(lw.engine.process, {"execute": 0.1})
    steps = await self.submit(run, *[(Op.OPTIM_STEP, {"adam_params": {}}) for _ in range(10)])
    go.set()
    self.assertTrue((await self.bus.wait_for_reply(created, 10)).ok)
    before = (await self.bus.get_run(run))["busy_ms"]
    self.assertTrue(all([reply.ok for reply in await self.replies(steps, 30)]))
    busy = (await self.bus.get_run(run))["busy_ms"] - before
    self.assertGreaterEqual(busy, 1000)
    self.assertLess(busy, 2000, f"10 commands of 0.1 s counted {busy} ms busy")

  async def test_a_new_node_owner_fails_what_a_dead_one_left_in_flight(self):
    run = new_id()
    await self.bus.create_run(Run(run, "p", "tiny", 4))
    [request_id] = await self.submit(run, (Op.OPTIM_STEP, {"adam_params": {}}))
    await self.bus.take_commands([run], "dead")
    await self.node.add("p")
    reply = await self.bus.wait_for_reply(request_id, 5)
    self.assertFalse(reply.ok)
    self.assertEqual(json.loads(reply.body)["category"], "server")
    self.assertEqual((await self.bus.queued_and_in_flight(run))["in_flight"], [])

  async def test_a_full_fine_tune_process_closes_with_its_run(self):
    run = new_id()
    await self.bus.create_run(Run(run, run, "tiny", 0))
    await self.submit(run, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 0, "lora": {}}), (Op.DELETE_MODEL, {}))
    lw = await self.node.add(run)
    await asyncio.wait_for(lw.task, 10)
    self.assertTrue(lw.engine.process.closed)
    self.assertIsNone(await self.bus.lease_owner(run))

  async def test_a_run_moves_between_node_owned_models_through_the_store(self):
    for rank, lora in ((4, LORA), (0, {})):
      ctrl, r1, r2 = new_id(), new_id(), new_id()
      for r, worker in ((ctrl, f"ctrl-{rank}"), (r1, f"p1-{rank}"), (r2, f"p2-{rank}")):
        await self.bus.create_run(Run(r, worker, "tiny", rank, lora=lora))
      for r in (ctrl, r1):
        await self.submit(r, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": rank, "lora": lora}))

      async def step_run(r: str) -> float:
        fb, _ = await self.submit(
          r, (Op.FORWARD_BACKWARD, proto_body(r)), (Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}})
        )
        await self.node.drain(r)
        return wire.output_json((await self.bus.wait_for_reply(fb, 1)).body)["metrics"]["loss:sum"]

      losses_ctrl = [await step_run(ctrl) for _ in range(2)]
      losses_moved = [await step_run(r1) for _ in range(2)]
      uri = f"tinker://{r1}/weights/handoff"
      await self.submit(r1, (Op.SAVE_WEIGHTS, {"uri": uri}), (Op.DELETE_MODEL, {}))
      await self.node.drain(r1)
      registered = await self.registry.get_checkpoint(uri)
      self.assertEqual(registered["tinker_path"], uri)
      self.assertGreater(registered["size_bytes"], 0)
      await self.submit(
        r2,
        (Op.CREATE_MODEL, {"base_model": "tiny", "rank": rank, "lora": lora}),
        (Op.LOAD_WEIGHTS, {"uri": uri, "optimizer": True}),
      )
      await self.node.drain(r2)
      for _ in range(3):
        losses_ctrl.append(await step_run(ctrl))
        losses_moved.append(await step_run(r2))
      self.assertEqual(losses_moved, losses_ctrl, f"rank {rank}")
      self.assertTrue(same(self.model(f"ctrl-{rank}").params[ctrl], self.model(f"p2-{rank}").params[r2]))

  async def test_graceful_replacement_registers_and_restores_weights_optimizer_and_pending_gradients(self):
    for rank, lora in ((4, LORA), (0, {})):
      ctrl, run = new_id(), new_id()
      for r, worker in ((ctrl, f"ctrl-{rank}"), (run, f"p-{rank}")):
        await self.bus.create_run(Run(r, worker, "tiny", rank, lora=lora))
        await self.submit(r, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": rank, "lora": lora}))
        for _ in range(2):
          await self.submit(
            r, (Op.FORWARD_BACKWARD, proto_body(r)), (Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}})
          )
        await self.submit(r, (Op.FORWARD_BACKWARD, proto_body(r)))  # pause between accumulation and optimizer step
        await self.node.drain(r)
      leaving = self.node.local[f"p-{rank}"]
      self.node.note(Ev.UNPLACED, worker=f"p-{rank}")
      await asyncio.wait_for(leaving.task, 10)
      parked = await self.bus.parked_checkpoint(run)
      self.assertIsNotNone(parked)
      self.assertIsNotNone(await self.registry.get_checkpoint(parked))
      for r in (ctrl, run):
        [request] = await self.submit(r, (Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}}))
        await self.node.drain(r)
        self.assertTrue((await self.bus.wait_for_reply(request, 1)).ok)
      self.assertTrue(same(self.model(f"ctrl-{rank}").params[ctrl], self.model(f"p-{rank}").params[run]))
      self.assertIsNone(await self.bus.parked_checkpoint(run))
      self.assertIsNone(await self.registry.get_checkpoint(parked))  # recovery cannot replay old state after a crash

  async def test_the_node_recovers_an_uncontested_lease_but_rejects_an_intervening_owner(self):
    run = new_id()
    await self.bus.create_run(Run(run, "p", "tiny", 4))
    lw = await self.node.add("p")
    await self.bus.r.delete("tinyrl:owner:p")
    lw.lease.renewed = 0
    await self.node.renew()
    self.assertEqual(await self.bus.lease_owner("p"), lw.lease.owner)
    self.assertFalse(lw.lease.lost)
    await self.bus.r.set("tinyrl:last_owner:p", "another")
    await self.bus.r.delete("tinyrl:owner:p")
    lw.lease.renewed = 0
    await self.node.renew()
    self.assertTrue(lw.lease.lost)
    await asyncio.wait_for(asyncio.gather(lw.task, return_exceptions=True), 5)
    self.assertTrue(lw.task.cancelled())
    self.assertTrue(lw.engine.process.closed)
    self.assertIsNone(await self.bus.lease_owner("p"))

  async def test_a_run_whose_process_died_unsaved_fails_until_it_loads_a_checkpoint(self):
    run, uri = new_id(), "tinker://saved/weights/step-1"
    await self.bus.create_run(Run(run, "p", "tiny", 4, lora=LORA))
    await self.submit(
      run, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA}), (Op.SAVE_WEIGHTS, {"uri": uri})
    )
    lw = await self.node.drain(run)
    lw.engine.process.kill()
    await self.node.tick()
    await asyncio.wait_for(asyncio.gather(lw.task, return_exceptions=True), 5)
    ids = await self.submit(
      run,
      (Op.OPTIM_STEP, {"adam_params": {}}),
      (Op.LOAD_WEIGHTS, {"uri": uri, "optimizer": True}),
      (Op.OPTIM_STEP, {"adam_params": {}}),
    )
    await self.node.drain(run)
    replies = await self.replies(ids, 1)
    self.assertEqual([reply.ok for reply in replies], [False, True, True])
    self.assertIn("lost its unsaved steps", json.loads(replies[0].body)["error"])

  async def test_a_run_whose_parked_state_is_gone_still_deletes(self):
    run = new_id()
    await self.bus.create_run(Run(run, "p", "tiny", 4, lora=LORA))
    await self.bus.r.set(f"tinyrl:parked:{run}", "tinker://gone/parked/1")
    [request_id] = await self.submit(run, (Op.DELETE_MODEL, {}))
    await self.node.drain(run)
    self.assertTrue((await self.bus.wait_for_reply(request_id, 1)).ok)
    self.assertIsNone(await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}"))

  async def test_a_sampler_save_passes_other_runs_but_never_its_own_earlier_commands(self):
    runs = [new_id() for _ in range(3)]
    for run in runs:
      await self.bus.create_run(Run(run, "p", "tiny", 4, lora=LORA))
      await self.submit(run, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA}))
      await self.node.drain(run)
    process = self.node.local["p"].engine.process
    calls = watch(process)
    entered, go = gate(process, "execute")
    fb, save = Op.FORWARD_BACKWARD, Op.SAVE_WEIGHTS_FOR_SAMPLER
    ids = await self.submit(runs[0], (fb, proto_body(runs[0])))
    await asyncio.wait_for(entered.wait(), 5)
    for run, op in ((runs[1], fb), (runs[2], fb), (runs[0], save), (runs[1], save)):
      body = proto_body(run) if op == fb else {"uri": f"tinker://{run}/sampler_weights/s0"}
      ids += await self.submit(run, (op, body))
      await asyncio.sleep(0.002)  # entry ids order runs against each other at millisecond resolution
    go.set()
    self.assertTrue(all([reply.ok for reply in await self.replies(ids, 10)]))
    executed = [(args[0].run, args[0].op) for _, method, args in calls if method == "execute"]
    self.assertEqual(executed, [(runs[0], fb), (runs[0], save), (runs[1], fb), (runs[1], save), (runs[2], fb)])
    causal = runs[0]
    pipelined = await self.submit(
      causal,
      (fb, proto_body(causal)),
      (Op.OPTIM_STEP, {"adam_params": {"learning_rate": 0.1}}),
      (save, {"uri": f"tinker://{causal}/sampler_weights/s1"}),
    )
    self.assertTrue(all([reply.ok for reply in await self.replies(pipelined, 10)]))
    executed = [(args[0].run, args[0].op) for _, method, args in calls if method == "execute"]
    self.assertEqual(executed[5:], [(causal, fb), (causal, Op.OPTIM_STEP), (causal, save)])
    self.assertEqual((await self.bus.get_run(causal))["steps"], 1)

  async def test_a_sampler_save_sent_while_the_trainer_waits_for_its_turn_passes_other_runs_not_its_own(self):
    await self.node.aclose()
    self.node = TestNode(self.bus, store=self.store, gpus=1)
    create = (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA})
    a, b, other = new_id(), new_id(), new_id()
    for run, worker in ((a, "p"), (b, "p"), (other, "q")):
      await self.bus.create_run(Run(run, worker, "tiny", 4, lora=LORA))
    p = await self.node.add("p")
    created = await self.submit(a, create) + await self.submit(b, create)
    self.assertTrue(all([reply.ok for reply in await self.replies(created, 10)]))
    q = await self.node.add("q")
    entered, go = gate(q.engine.process, "execute")
    await self.submit(other, create)
    await asyncio.wait_for(entered.wait(), 10)
    calls = watch(p.engine.process)
    step, save = Op.OPTIM_STEP, Op.SAVE_WEIGHTS_FOR_SAMPLER
    ids = await self.submit(b, (step, {"adam_params": {}}))
    for _ in range(100):
      if self.node.state.workers["p"].asked:
        break
      await asyncio.sleep(0.02)
    self.assertTrue(self.node.state.workers["p"].asked, "p never peeked b's optim_step to wait for its turn")
    for run in (b, a):
      ids += await self.submit(run, (save, {"uri": f"tinker://{run}/sampler_weights/s0"}))
    go.set()
    self.assertTrue(all([reply.ok for reply in await self.replies(ids, 10)]))
    executed = [(args[0].run, args[0].op) for _, method, args in calls if method == "execute"]
    self.assertEqual(executed, [(a, save), (b, step), (b, save)])


class TestDistributedTrainer(unittest.IsolatedAsyncioTestCase):
  async def test_two_rank_fsdp_matches_a_single_model_process(self):
    import sys

    from test.test_process import require_unix
    from tinyrl.control.process import Process

    require_unix()
    base = tempfile.mkdtemp()
    tiny().save_pretrained(base)
    for rank, lora in ((0, {"seed": 0}), (4, LORA)):
      ckpt, r1, r2 = tempfile.mkdtemp(), new_id(), new_id()
      store = Store(root=ckpt)
      single = Trainer(store=store)
      env = {**os.environ, "CKPT": ckpt, "CUDA_VISIBLE_DEVICES": ""}
      args = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]
      multi = Process([*args, "-m", "tinyrl.workers.trainer"], env)
      try:
        for run in (r1, r2):
          commands = [
            (Op.CREATE_MODEL, {"base_model": base, "rank": rank, "lora": lora}),
            (Op.FORWARD_BACKWARD, proto_body(run)),
            (Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}}),
            (Op.FORWARD_BACKWARD, proto_body(run)),
            (Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}}),
            (Op.SAVE_WEIGHTS_FOR_SAMPLER, {"uri": f"tinker://{run}/sampler_weights/s1"}),
          ]
          for i, (op, body) in enumerate(commands):
            cmd = Cmd(run, str(i), op, body if isinstance(body, bytes) else json.dumps(body).encode())
            reply, _ = single.execute(cmd) if run == r1 else await asyncio.wait_for(multi.call("execute", cmd), 90)
            self.assertTrue(reply.ok, reply.body)
        expected, actual = (store.get(f"tinker://{run}/sampler_weights/s1")[0] for run in (r1, r2))
        self.assertEqual(expected.keys(), actual.keys(), f"rank {rank}")
        for name in expected:
          torch.testing.assert_close(actual[name], expected[name], atol=1e-5, rtol=1e-5, msg=f"rank {rank} {name}")
      finally:
        await multi.close()


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


class TestG1PPO(unittest.TestCase):
  def test_ppo_clip_and_reject(self):
    logprobs = torch.tensor([1.0])
    weights = torch.tensor([1.0])
    sampler_logprobs = torch.tensor([0.0])
    advantages = torch.tensor([1.0])

    config = {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}
    loss_actual = tinyrl.workers.model.ppo(logprobs, weights, sampler_logprobs, advantages, config)
    self.assertAlmostEqual(loss_actual.item(), -1.1, places=3)

    with self.assertRaises(ValueError):
      tinyrl.workers.model.ppo(logprobs, weights, sampler_logprobs, advantages, {"unknown": 1.0})
