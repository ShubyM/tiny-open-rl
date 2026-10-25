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

from test.helpers import CoordinatorTest, new_id, tiny
from test.test_api import proto_body
from tinyrl import trainer, wire
from tinyrl.bus import Cmd, Op, Run
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

  def test_save_and_load_bring_back_weights_and_optimizer(self):
    store = Store(root=tempfile.mkdtemp())
    model, (a, b) = lora_runs(2)
    train(model, a, steps=3)
    ref = store.put(*model.state(a))
    saved_weights = [p.detach().clone() for p in model.params[a].values()]
    saved_optimizer = copy.deepcopy(model.optimizers[a].state_dict())

    train(model, a, steps=3)
    model.restore(a, *store.get(ref), optimizer=True)
    self.assertTrue(same(saved_weights, model.params[a]))
    torch.testing.assert_close(model.optimizers[a].state_dict()["state"], saved_optimizer["state"])

    model.restore(b, *store.get(ref), optimizer=False)  # a run can start from another run's checkpoint
    self.assertTrue(same(saved_weights, model.params[b]))

  def test_loading_a_checkpoint_that_does_not_fit_fails(self):
    store = Store(root=tempfile.mkdtemp())
    model, (a,) = lora_runs()
    ref = store.put(*model.weights(a))

    no_mlp = new_id()
    model.create(no_mlp, "tiny", 4, {**LORA, "train_mlp": False})
    with self.assertRaises(ValueError):
      model.restore(no_mlp, *store.get(ref), optimizer=False)
    with self.assertRaises(ValueError):
      model.restore(a, *store.get(ref), optimizer=True)  # it was saved without the optimizer
    with self.assertRaises(ValueError):
      store.get(f"tinker://{a}/weights/missing")

  def test_sampler_weights_load_in_peft_and_transformers(self):
    store = Store(root=tempfile.mkdtemp())
    ids = torch.randint(0, 96, (2, 8))

    model, (lora_run,) = lora_runs()
    train(model, lora_run, steps=3)
    lora_dir = store.export(store.put(*model.weights(lora_run)))
    with torch.no_grad():
      torch.testing.assert_close(
        PeftModel.from_pretrained(tiny(), lora_dir).eval()(ids).logits,
        model.use(lora_run).eval()(ids).logits,
      )

    fft, fft_run = Model(tiny), new_id()
    fft.create(fft_run, "tiny", 0, {})
    train(fft, fft_run, steps=3)
    fft_dir = store.export(store.put(*fft.weights(fft_run)))
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

  def test_restore_rejects_shapes_before_changing_weights_gradients_or_optimizer(self):
    model, (source, target) = lora_runs(2)
    train(model, source, steps=2)
    train(model, target, steps=1)
    model.forward_backward(target, [counting(8)], "cross_entropy", {}, False)
    weights = [p.clone() for p in model.params[target].values()]
    gradients = [p.grad.clone() for p in model.params[target].values()]
    optimizer = copy.deepcopy(model.optimizers[target].state_dict())
    tensors, meta = model.state(source)
    last = list(model.params[source])[-1]
    for key in (last, f"{last}.exp_avg", f"__grad.{last}"):
      broken, info = dict(tensors), copy.deepcopy(meta)
      if key.startswith("__grad"):
        info["grads"].append(last)
      broken[key] = torch.zeros(1)
      with self.assertRaises(ValueError):
        model.restore(target, broken, info, optimizer=True)
      self.assertTrue(same(weights, model.params[target]))
      self.assertTrue(same(gradients, [p.grad for p in model.params[target].values()]))
      torch.testing.assert_close(model.optimizers[target].state_dict(), optimizer)

  def test_snapshot_preserves_none_gradients_and_private_dropout_rng(self):
    config = {**LORA, "lora_dropout": 0.25}
    source, run = Model(tiny), new_id()
    source.create(run, "tiny", 4, config)
    train(source, run, steps=2)
    source.forward_backward(run, [counting(8)], "cross_entropy", {}, False)
    first = next(iter(source.params[run]))
    source.params[run][first].grad = None
    store = Store(root=tempfile.mkdtemp())
    ref = store.put(*source.state(run))
    restored, other = Model(tiny), new_id()
    restored.create(other, "tiny", 4, config)
    restored.restore(other, *store.get(ref), optimizer=True)
    self.assertIsNone(restored.params[other][first].grad)
    source.optim_step(run, {"learning_rate": 0.01})
    restored.optim_step(other, {"learning_rate": 0.01})
    self.assertTrue(same(source.params[run], restored.params[other]))
    noise = new_id()
    source.create(noise, "tiny", 4, {**config, "seed": 87})
    train(source, noise, steps=2)
    self.assertEqual(train(source, run, steps=2), train(restored, other, steps=2))
    self.assertTrue(same(source.params[run], restored.params[other]))


class TestTrainer(CoordinatorTest):
  async def submit(self, run: str, op: Op, body: dict | bytes):
    return await self.bus.submit(run, op, body if isinstance(body, bytes) else json.dumps(body).encode())

  async def create_run(self, pool="p", rank=4, lora=None):
    run = new_id()
    lora = LORA if lora is None else lora
    await self.bus.create(Run(run, pool, "tiny", rank), body={"base_model": "tiny", "rank": rank, "lora": lora})
    return run

  async def drain(self, worker):
    while await worker.step(block_ms=0):
      pass

  async def test_commands_and_failed_parse_preserve_order(self):
    run = await self.create_run()
    bad = await self.submit(run, Op.FORWARD_BACKWARD, b"not protobuf")
    step = await self.submit(run, Op.OPTIM_STEP, {"adam_params": {}})
    delete = await self.submit(run, Op.DELETE_MODEL, {})
    worker = Trainer(self.bus, "p", Model(tiny))
    await self.drain(worker)
    self.assertFalse((await self.bus.result(bad, 0)).ok)
    self.assertTrue((await self.bus.result(step, 0)).ok)
    self.assertTrue((await self.bus.result(delete, 0)).ok)
    self.assertEqual((await self.bus.run(run))["phase"], "deleted")

  async def test_lost_completion_response_does_not_execute_optimizer_twice(self):
    run = await self.create_run()
    worker = Trainer(self.bus, "p", Model(tiny))
    await self.drain(worker)
    await self.submit(run, Op.FORWARD_BACKWARD, proto_body(run))
    await self.drain(worker)
    request = await self.submit(run, Op.OPTIM_STEP, {"adam_params": {"learning_rate": 0.01}})
    finish = self.bus.finish
    dropped = False

    async def drop_response(cmd, reply):
      nonlocal dropped
      result = await finish(cmd, reply)
      if not dropped:
        dropped = True
        raise ConnectionError("response disappeared after durable commit")
      return result

    with (
      patch.object(self.bus, "finish", side_effect=drop_response),
      patch.object(worker.model, "optim_step", wraps=worker.model.optim_step) as optim,
    ):
      await worker.step(block_ms=0)
    self.assertEqual(optim.call_count, 1)
    self.assertTrue((await self.bus.result(request, 0)).ok)
    self.assertEqual(worker.pending, {})

  async def test_concurrent_step_callers_do_not_execute_one_attempt_twice(self):
    run = await self.create_run()
    worker = Trainer(self.bus, "p", Model(tiny))
    await self.drain(worker)
    await self.submit(run, Op.OPTIM_STEP, {"adam_params": {}})
    with patch.object(worker.model, "optim_step", wraps=worker.model.optim_step) as step:
      await asyncio.gather(worker.step(block_ms=0), worker.step(block_ms=0))
    self.assertEqual(step.call_count, 1)

  async def test_migration_keeps_pending_gradients_rng_and_run_identity(self):
    for rank in (0, 4):
      config = {**LORA, "lora_dropout": 0.2} if rank else {"seed": 0}
      pool_a, pool_b = f"a{rank}", f"b{rank}"
      run = await self.create_run(pool_a, rank, config)
      store = Store(root=f"{self.tmpdir.name}/artifacts-{rank}")
      source = Trainer(self.bus, pool_a, Model(tiny), store=store)
      target = Trainer(self.bus, pool_b, Model(tiny), store=store)
      await self.drain(source)
      control, ctrl = Model(tiny), new_id()
      control.create(ctrl, "tiny", rank, config)
      data = wire.datums(wire.parse(proto_body(run)))
      control.forward_backward(ctrl, data, "cross_entropy", {}, False)
      await self.submit(run, Op.FORWARD_BACKWARD, proto_body(run))
      await self.drain(source)
      before = [p.clone() for p in source.model.params[run].values()]
      await self.bus.request_move(run, pool_b)
      await source.step(block_ms=0)  # snapshot
      await target.step(block_ms=0)  # restore inactive
      self.assertIn(run, target.model.inactive)
      with self.assertRaises(ValueError):
        target.model.optim_step(run, {})
      await target.step(block_ms=0)  # activate
      await self.drain(source)  # detach old generation
      self.assertNotIn(run, source.model.params)
      self.assertEqual((await self.bus.run(run))["pool"], pool_b)
      control.optim_step(ctrl, {"learning_rate": 0.01})
      await self.submit(run, Op.OPTIM_STEP, {"adam_params": {"learning_rate": 0.01}})
      await self.drain(target)
      self.assertFalse(same(before, target.model.params[run]))
      self.assertTrue(same(control.params[ctrl], target.model.params[run]))
      # Further stochastic training must continue from the same private RNG state.
      self.assertEqual(train(control, ctrl, steps=2), train(target.model, run, steps=2))
      self.assertTrue(same(control.params[ctrl], target.model.params[run]))

  async def test_partial_backward_failure_requires_recovery(self):
    run = await self.create_run()
    worker = Trainer(self.bus, "p", Model(tiny), store=Store(root=f"{self.tmpdir.name}/artifacts"))
    await self.drain(worker)
    saved = worker.store.put(*worker.model.state(run))
    body = {"model_id": run, "forward_backward_input": {"loss_fn": "cross_entropy", "data": []}}
    for n in (8, 9):
      body["forward_backward_input"]["data"].append(
        {
          "model_input": {"chunks": [{"tokens": list(range(n))}]},
          "loss_fn_inputs": {"target_tokens": list(range(1, n + 1)), "weights": [1.0] * (n if n == 8 else 2)},
        }
      )
    failed = await self.submit(run, Op.FORWARD_BACKWARD, wire.from_json(body))
    await self.submit(run, Op.OPTIM_STEP, {"adam_params": {}})
    with patch.object(trainer, "TOKEN_BUDGET", 8):
      await self.drain(worker)
    self.assertTrue((await self.bus.result(failed, 0)).recover)
    self.assertEqual((await self.bus.run(run))["phase"], "recovery")
    await self.bus.recover(run, saved, "p")
    await self.drain(worker)
    self.assertEqual((await self.bus.run(run))["phase"], "active")
    self.assertTrue(all(p.grad is None for p in worker.model.params[run].values()))

  def test_local_parking_retains_live_model_and_gradients_without_store_io(self):
    model, (run,) = lora_runs()
    model.forward_backward(run, [counting(8)], "cross_entropy", {}, False)
    params, lm = model.params[run], model.lm
    gradients = [p.grad.clone() for p in params.values()]
    worker = Trainer(self.bus, "p", model)
    with patch.object(worker.store, "put", side_effect=AssertionError("local parking serialized a checkpoint")):
      worker.park()
      worker.resume()
    self.assertIs(worker.model.lm, lm)
    self.assertTrue(all(a is b for a, b in zip(params.values(), worker.model.params[run].values())))
    self.assertTrue(same(gradients, [p.grad for p in params.values()]))

  def test_stale_detach_does_not_remove_returned_generation(self):
    model, (run,) = lora_runs()
    worker = Trainer(self.bus, "p", model)
    worker.generations[run] = 3
    reply = worker.run(Cmd(run, "old-detach", Op.DETACH, b"{}", generation=1))
    self.assertTrue(reply.ok)
    self.assertIn(run, model.params)

  def test_restore_checks_coordinator_spec_before_removing_old_generation(self):
    model, (run,) = lora_runs()
    store = Store(root=f"{self.tmpdir.name}/artifacts")
    tensors, meta = model.state(run)
    ref = store.put(tensors, meta)
    worker = Trainer(self.bus, "p", model, store=store)
    worker.generations[run] = 1
    before = [p.clone() for p in model.params[run].values()]
    for wrong in (
      {"base": "other", "rank": 4, "lora": LORA},
      {"base": "tiny", "rank": 1, "lora": LORA},
      {"base": "tiny", "rank": 4, "lora": {**LORA, "train_mlp": False}},
    ):
      body = json.dumps({"ref": ref, "recovery": True, "spec": wrong}).encode()
      reply = worker.run(Cmd(run, "restore", Op.RESTORE, body, generation=2))
      self.assertFalse(reply.ok)
      self.assertEqual(worker.generations[run], 1)
      self.assertTrue(same(before, model.params[run]))

  def test_full_run_detach_frees_base_without_stopping_worker(self):
    model, run = Model(tiny), new_id()
    model.create(run, "tiny", 0, {})
    worker = Trainer(self.bus, "p", model)
    worker.generations[run] = 1
    self.assertTrue(worker.run(Cmd(run, "detach", Op.DETACH, b"{}", generation=1)).ok)
    self.assertIsNone(model.lm)
    self.assertFalse(worker.done)
    model.create(new_id(), "tiny", 0, {})


if __name__ == "__main__":
  unittest.main()
