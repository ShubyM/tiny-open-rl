import asyncio
import copy
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from peft import PeftModel
from transformers import LlamaConfig, LlamaForCausalLM

from test.helpers import RedisTest, new_id, tiny
from test.test_api import proto_body
from tinyrl import trainer, wire
from tinyrl.bus import Op, Run
from tinyrl.store import Store
from tinyrl.trainer import LOSSES, Model, Trainer, batches, target_logprobs, token_budget

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
    model.create(run, "tiny", 4, LORA)
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
    bytes_per_token = 6 * 32 * 4096 * 2

    def budget(free_gib: int) -> int:
      with (
        patch.object(torch.cuda, "mem_get_info", return_value=(free_gib * 2**30, 0)),
        patch.object(torch.cuda, "memory_reserved", return_value=2**30),  # 1 GiB cached by torch but unused
        patch.object(torch.cuda, "memory_allocated", return_value=0),
      ):
        return token_budget(gpu)

    self.assertEqual(budget(7), 8 * 2**30 // 2 // bytes_per_token)
    self.assertEqual(budget(80), trainer.MAX_TOKEN_BUDGET)
    self.assertEqual(token_budget(tiny()), trainer.MAX_TOKEN_BUDGET)  # on the cpu
    with patch.object(trainer, "TOKEN_BUDGET", 512):
      self.assertEqual(token_budget(gpu), 512)

  def test_chunked_logprobs_are_the_full_softmax(self):
    lm = tiny().eval()
    ids, targets = torch.randint(0, 96, (3, 11)), torch.randint(0, 96, (3, 11))
    mask = torch.ones_like(ids)
    mask[2, 7:] = 0

    with patch.object(trainer, "CHUNK", 4), torch.no_grad():
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


class TestModel(unittest.TestCase):
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
    model.create(run, "tiny", 0, {})
    losses = train(model, run)
    self.assertLess(losses[-1], losses[0] * 0.7)

    with self.assertRaises(ValueError):  # a full fine-tune has its trainer to itself
      model.create(new_id(), "tiny", 0, {})
    with self.assertRaises(ValueError):
      model.create(new_id(), "tiny", 4, LORA)

  def test_the_optimizer_updates_fp32_master_weights_and_casts_sampler_weights_to_bf16(self):
    def bf16(name: str) -> LlamaForCausalLM:  # how a base model loads on the gpu
      return tiny(name).to(torch.bfloat16)

    for rank in (0, 4):
      model, run = Model(bf16), new_id()
      model.create(run, "tiny", rank, {"seed": 0})
      model.forward_backward(run, [counting(12)], "cross_entropy", {}, False)
      grads = [p.grad for p in model.params[run].values()]
      model.optim_step(run, {"learning_rate": 1e-2})
      moments = [m for state in model.optimizers[run].state.values() for m in (state["exp_avg"], state["exp_avg_sq"])]
      self.assertEqual(
        {t.dtype for t in [*model.params[run].values(), *grads, *moments]}, {torch.float32}, f"rank {rank}"
      )
      self.assertEqual({t.dtype for t in model.weights(run)[0].values()}, {torch.bfloat16}, f"rank {rank}")
      if rank:  # a LoRA run's base model is frozen, so it stays bf16
        self.assertEqual(model.lm.get_input_embeddings().weight.dtype, torch.bfloat16)

  def test_a_run_restored_from_its_checkpoint_trains_on_exactly_as_before(self):
    store = Store(root=tempfile.mkdtemp())
    for rank in (0, 4):
      model, run = Model(tiny), new_id()
      model.create(run, "tiny", rank, LORA)
      train(model, run, steps=3)
      uri = f"tinker://{run}/weights/step-3"
      store.put(uri, *model.state(run))

      restored = Model(tiny)  # a new trainer, say on another node
      restored.create(run, "tiny", rank, LORA)
      restored.restore(run, *store.get(uri), optimizer=True)

      self.assertEqual(train(model, run, steps=3), train(restored, run, steps=3), f"rank {rank}")
      self.assertTrue(same(model.params[run].values(), restored.params[run].values()), f"rank {rank}")

  def test_save_and_load_bring_back_weights_and_optimizer(self):
    store = Store(root=tempfile.mkdtemp())
    model, (a, b) = lora_runs(2)
    train(model, a, steps=3)
    uri = f"tinker://{a}/weights/step-3"
    store.put(uri, *model.state(a))
    saved_weights = [p.detach().clone() for p in model.params[a].values()]
    saved_optimizer = copy.deepcopy(model.optimizers[a].state_dict())

    train(model, a, steps=3)
    model.restore(a, *store.get(uri), optimizer=True)
    self.assertTrue(same(saved_weights, model.params[a]))
    torch.testing.assert_close(model.optimizers[a].state_dict()["state"], saved_optimizer["state"])

    model.restore(b, *store.get(uri), optimizer=False)  # a run can start from another run's checkpoint
    self.assertTrue(same(saved_weights, model.params[b]))

  def test_loading_a_checkpoint_that_does_not_fit_fails(self):
    store = Store(root=tempfile.mkdtemp())
    model, (a,) = lora_runs()
    uri = f"tinker://{a}/sampler_weights/step-0"
    store.put(uri, *model.weights(a))

    no_mlp = new_id()
    model.create(no_mlp, "tiny", 4, {**LORA, "train_mlp": False})
    with self.assertRaises(ValueError):
      model.restore(no_mlp, *store.get(uri), optimizer=False)
    with self.assertRaises(ValueError):
      model.restore(a, *store.get(uri), optimizer=True)  # it was saved without the optimizer
    with self.assertRaises(ValueError):
      store.get(f"tinker://{a}/weights/missing")

  def test_sampler_weights_load_in_peft_and_transformers(self):
    store = Store(root=tempfile.mkdtemp())
    ids = torch.randint(0, 96, (2, 8))

    model, (lora_run,) = lora_runs()
    train(model, lora_run, steps=3)
    lora_dir = store.put(f"tinker://{lora_run}/sampler_weights/final", *model.weights(lora_run))
    with torch.no_grad():
      torch.testing.assert_close(
        PeftModel.from_pretrained(tiny(), lora_dir).eval()(ids).logits,
        model.use(lora_run).eval()(ids).logits,
      )

    fft, fft_run = Model(tiny), new_id()
    fft.create(fft_run, "tiny", 0, {})
    train(fft, fft_run, steps=3)
    fft_dir = store.put(f"tinker://{fft_run}/sampler_weights/final", *fft.weights(fft_run))
    with torch.no_grad():
      torch.testing.assert_close(
        LlamaForCausalLM.from_pretrained(fft_dir).eval()(ids).logits,
        fft.use(fft_run).eval()(ids).logits,
      )

  def test_mistakes_are_the_users(self):
    model, (a,) = lora_runs()
    with self.assertRaises(KeyError):
      model.forward_backward(new_id(), [counting(4)], "cross_entropy", {}, False)
    with self.assertRaises(ValueError):
      model.forward_backward(a, [counting(4)], "dpo", {}, False)
    with self.assertRaises(ValueError):
      model.create(new_id(), "other", 4, LORA)

    model.delete(a)
    with self.assertRaises(KeyError):
      model.optim_step(a, {})


class TestTrainer(RedisTest):
  async def submit(self, run: str, *cmds: tuple[Op, dict | bytes]) -> list[str]:
    request_ids = []
    for op, body in cmds:
      request_ids.append(await self.bus.submit(run, op, body if isinstance(body, bytes) else json.dumps(body).encode()))
    return request_ids

  async def test_commands_flow_through_and_delete_drops_the_run(self):
    run = new_id()
    await self.bus.create(Run(run, "p", "tiny", 4))
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

    replies = [await self.bus.result(request_id, 0) for request_id in request_ids]
    bodies = [json.loads(reply.body) for reply in replies]
    self.assertEqual([reply.ok for reply in replies], [True, False, True, True])
    self.assertEqual([body.get("type") for body in bodies], ["create_model", None, "optim_step", "delete_model"])
    self.assertEqual(bodies[1]["category"], "user")

    self.assertEqual((await self.bus.run(run))["steps"], 1)
    self.assertIsNone(await self.bus.submit(run, Op.OPTIM_STEP, b"{}"))  # deleted
    self.assertFalse(t.done)  # a LoRA trainer keeps serving its pool

  async def test_a_new_trainer_fails_what_a_dead_one_left_in_flight(self):
    run = new_id()
    await self.bus.create(Run(run, "p", "tiny", 4))
    [request_id] = await self.submit(run, (Op.OPTIM_STEP, {"adam_params": {}}))
    await self.bus.take("p", "dead", block_ms=10)  # taken, and never answered

    t = Trainer(self.bus, "p", Model(tiny), worker="new")
    t.done = True  # so serve() stops right after reclaiming
    await t.serve()

    reply = await self.bus.result(request_id, 1)
    self.assertFalse(reply.ok)
    self.assertEqual(json.loads(reply.body)["category"], "server")
    self.assertEqual((await self.bus.queue(run))["in_flight"], [])

  async def test_a_full_fine_tune_trainer_exits_with_its_run(self):
    run = new_id()
    await self.bus.create(Run(run, run, "tiny", 0))
    await self.submit(run, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 0, "lora": {}}), (Op.DELETE_MODEL, {}))

    t = Trainer(self.bus, run, Model(tiny))
    await asyncio.wait_for(t.serve(), 60)
    self.assertTrue(t.done)
    self.assertEqual(await self.bus.runs(run), [])

  async def test_a_run_moves_between_trainers_through_the_store(self):
    store = Store(root=tempfile.mkdtemp())
    for rank, lora in ((4, LORA), (0, {})):
      ctrl, r1, r2 = new_id(), new_id(), new_id()
      for r, pool in ((ctrl, f"ctrl-{rank}"), (r1, f"p1-{rank}"), (r2, f"p2-{rank}")):
        await self.bus.create(Run(r, pool, "tiny", rank))

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
        return wire.output_json((await self.bus.result(fb, 1)).body)["metrics"]["loss:sum"]

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
      await self.bus.create(Run(ctrl, f"ctrl-{rank}", "tiny", rank, lora=lora))
      await self.bus.create(Run(run, f"p-{rank}", "tiny", rank, lora=lora))
      tc, first = (Trainer(self.bus, pool, Model(tiny), store=store) for pool in (f"ctrl-{rank}", f"p-{rank}"))
      serving = [asyncio.create_task(tc.serve()), asyncio.create_task(first.serve())]

      async def call(r: str, op: Op, body: dict | bytes):
        [request_id] = await self.submit(r, (op, body))
        return await self.bus.result(request_id, 60)

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
      self.assertIsNotNone(await self.bus.parked(run))

      second = Trainer(self.bus, f"p-{rank}", Model(tiny), store=store)
      serving.append(asyncio.create_task(second.serve()))
      for _ in range(3):
        losses_ctrl.append(await step(ctrl))
        losses.append(await step(run))
      self.assertEqual(losses, losses_ctrl, f"rank {rank}")
      self.assertIsNone(await self.bus.parked(run))  # restored once, so a later crash can't bring back old state

      tc.done = second.done = True
      await asyncio.gather(*serving)

  @patch("tinyrl.trainer.LEASE_MS", 200)
  async def test_a_trainer_takes_its_lease_again_when_redis_loses_it(self):
    t = Trainer(self.bus, "p", Model(tiny))
    await self.own(t)
    await self.bus.r.delete("tinyrl:owner:p")  # what a Redis restart longer than the lease does
    await asyncio.sleep(0.2)
    self.assertEqual((await self.bus.owner("p"), t.done), (t.worker, False))

    t.model.params["r"] = {}  # but not once another trainer has served the pool and parked its run
    await self.bus.r.set("tinyrl:parked:r", "tinker://r/parked/1")
    await self.bus.r.delete("tinyrl:owner:p")
    await asyncio.sleep(0.2)
    self.assertEqual((await self.bus.owner("p"), t.done), (None, True))

  async def test_a_run_whose_trainer_died_unsaved_fails_until_it_loads_a_checkpoint(self):
    store = Store(root=tempfile.mkdtemp())
    run, uri = new_id(), "tinker://saved/weights/step-1"
    await self.bus.create(Run(run, "p", "tiny", 4, lora=LORA))
    await self.submit(
      run, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA}), (Op.SAVE_WEIGHTS, {"uri": uri})
    )
    first = Trainer(self.bus, "p", Model(tiny), store=store)
    await self.own(first)
    while await first.step(block_ms=10):
      pass
    for task in self.renewal_tasks:  # it dies without saving
      task.cancel()
    await self.bus.release("p", first.worker)

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
    replies = [await self.bus.result(request_id, 1) for request_id in ids]
    self.assertEqual([reply.ok for reply in replies], [False, True, True])
    self.assertIn("lost its unsaved steps", json.loads(replies[0].body)["error"])


if __name__ == "__main__":
  unittest.main()
