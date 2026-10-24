import asyncio
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from peft import PeftModel

from test.helpers import RedisTest, new_id, tiny
from tinyrl import trainer, wire
from tinyrl.bus import Run
from tinyrl.trainer import LOSSES, Model, Trainer, batches, target_logprobs

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


def same(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
  return all(torch.equal(x, y) for x, y in zip(a, b))


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
    b_before = [p.detach().clone() for p in model.params[b]]

    losses = train(model, a)
    self.assertLess(losses[-1], losses[0] * 0.7)
    self.assertTrue(same(b_before, model.params[b]))
    self.assertTrue(all(torch.equal(base[name], p) for name, p in model.lm.named_parameters() if "lora" not in name))

  def test_interleaved_runs_keep_their_own_gradients(self):
    model, (a, b) = lora_runs(2)
    model.forward_backward(a, [counting(8)], "cross_entropy", {}, False)
    model.forward_backward(b, [counting(8, start=5)], "cross_entropy", {}, False)
    b_grads = [p.grad.clone() for p in model.params[b]]

    model.optim_step(a, {"learning_rate": 1e-2})
    self.assertTrue(all(p.grad is None for p in model.params[a]))
    self.assertTrue(same(b_grads, [p.grad for p in model.params[b]]))

  def test_forward_only_leaves_no_gradient(self):
    model, (a,) = lora_runs()
    logprobs, metrics = model.forward_backward(a, [counting(8), counting(5)], "cross_entropy", {}, True)
    self.assertTrue(all(p.grad is None for p in model.params[a]))
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

  def test_save_and_load_bring_back_weights_and_optimizer(self):
    model, (a, b) = lora_runs(2)
    train(model, a, steps=3)
    path = tempfile.mkdtemp()
    model.save(a, path, optimizer=True)
    saved_weights = [p.detach().clone() for p in model.params[a]]
    saved_optimizer = copy.deepcopy(model.optimizers[a].state_dict())

    train(model, a, steps=3)
    model.load_weights(a, path, optimizer=True)
    self.assertTrue(same(saved_weights, model.params[a]))
    torch.testing.assert_close(model.optimizers[a].state_dict()["state"], saved_optimizer["state"])

    model.load_weights(b, path, optimizer=False)  # a run can start from another run's checkpoint
    self.assertTrue(same(saved_weights, model.params[b]))

  def test_loading_a_checkpoint_that_does_not_fit_fails(self):
    model, (a,) = lora_runs()
    path = tempfile.mkdtemp()
    model.save(a, path, optimizer=False)

    no_mlp = new_id()
    model.create(no_mlp, "tiny", 4, {**LORA, "train_mlp": False})
    with self.assertRaises(ValueError):
      model.load_weights(no_mlp, path, optimizer=False)
    with self.assertRaises(ValueError):
      model.load_weights(a, path, optimizer=True)  # it was saved without the optimizer
    with self.assertRaises(ValueError):
      model.load_weights(a, tempfile.mkdtemp(), optimizer=False)

  def test_sampler_weights_are_a_peft_adapter(self):
    model, (a,) = lora_runs()
    train(model, a, steps=3)
    path = tempfile.mkdtemp()
    model.save(a, path, optimizer=False)

    ids = torch.randint(0, 96, (2, 8))
    with torch.no_grad():
      want = model.use(a).eval()(ids).logits
      got = PeftModel.from_pretrained(tiny(), path).eval()(ids).logits
    torch.testing.assert_close(got, want)

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
  async def submit(self, run: str, *cmds: tuple[str, dict | bytes]) -> list[str]:
    request_ids = []
    for op, body in cmds:
      request_ids.append(await self.bus.submit(run, op, body if isinstance(body, bytes) else json.dumps(body).encode()))
    return request_ids

  async def test_commands_flow_through_and_delete_drops_the_run(self):
    run = new_id()
    await self.bus.create(Run(run, "p", "tiny", 4))
    request_ids = await self.submit(
      run,
      ("create_model", {"base_model": "tiny", "rank": 4, "lora": LORA}),
      ("forward_backward", b"not protobuf"),
      ("optim_step", {"adam_params": {}}),
      ("delete_model", {}),
    )

    t = Trainer(self.bus, "p", Model(tiny))
    while await t.step(block_ms=10):
      pass

    replies = [await self.bus.result(request_id, 0) for request_id in request_ids]
    bodies = [json.loads(reply.body) for reply in replies]
    self.assertEqual([reply.ok for reply in replies], [True, False, True, True])
    self.assertEqual([body.get("type") for body in bodies], ["create_model", None, "optim_step", "delete_model"])
    self.assertEqual(bodies[1]["category"], "user")

    self.assertEqual((await self.bus.run(run))["steps"], 1)
    self.assertIsNone(await self.bus.submit(run, "optim_step", b"{}"))  # deleted
    self.assertFalse(t.done)  # a LoRA trainer keeps serving its pool

  async def test_a_new_trainer_fails_what_a_dead_one_left_in_flight(self):
    run = new_id()
    await self.bus.create(Run(run, "p", "tiny", 4))
    [request_id] = await self.submit(run, ("optim_step", {"adam_params": {}}))
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
    await self.submit(run, ("create_model", {"base_model": "tiny", "rank": 0, "lora": {}}), ("delete_model", {}))

    t = Trainer(self.bus, run, Model(tiny))
    await asyncio.wait_for(t.serve(), 60)
    self.assertTrue(t.done)
    self.assertEqual(await self.bus.runs(run), [])


if __name__ == "__main__":
  unittest.main()
