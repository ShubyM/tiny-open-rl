import asyncio
import contextlib
import json
import unittest
from unittest.mock import patch

import torch

from test.helpers import RedisTest, new_id, tiny
from test.test_api import proto_body
from tinyrl import trainer, wire
from tinyrl.bus import Op, Run
from tinyrl.trainer import Model, Trainer

LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


class TestTakingTurns(RedisTest):
  """Two trainers whose pools the node daemon put on one GPU: only the one holding the GPU's lease runs commands."""

  async def trainer(self, pool: str) -> tuple[Trainer, str]:
    t = Trainer(self.bus, pool, Model(tiny), worker=pool)
    await t.own()
    run = new_id()
    await self.bus.create(Run(run, pool, "tiny", 4))
    await self.submit(run, Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA})
    return t, run

  async def submit(self, run: str, op: Op, body: dict) -> str:
    return await self.bus.submit(run, op, json.dumps(body).encode())

  async def test_an_idle_trainer_yields_its_gpu_to_one_that_waits(self):
    with patch.object(trainer, "GPU", "node-a:0"):
      a, _ = await self.trainer("a")
      await a.step(block_ms=10)  # takes the GPU, creates its model
      b, run_b = await self.trainer("b")
      waiting = asyncio.create_task(b.step(block_ms=10))
      await asyncio.sleep(0.3)
      self.assertFalse(waiting.done())  # b took its command, but a holds the GPU
      self.assertEqual(len((await self.bus.queue(run_b))["in_flight"]), 1)

      await a.step(block_ms=10)  # a has nothing to do, and b waits
      self.assertEqual(await waiting, 1)
      self.assertTrue(a.model.parked)
      self.assertEqual((a.seated > 0, b.seated > 0), (False, True))

  async def test_a_busy_trainer_keeps_its_gpu_for_hold_s(self):
    with patch.object(trainer, "GPU", "node-a:0"), patch.object(trainer, "HOLD_S", 1.0):
      a, run_a = await self.trainer("a")
      await a.step(block_ms=10)
      b, _ = await self.trainer("b")
      waiting = asyncio.create_task(b.step(block_ms=10))
      await asyncio.sleep(0.2)

      await self.submit(run_a, Op.OPTIM_STEP, {"adam_params": {}})
      await a.step(block_ms=10)  # busy, and it has held the GPU under HOLD_S
      self.assertFalse(waiting.done())

      await asyncio.sleep(1.0)
      await self.submit(run_a, Op.OPTIM_STEP, {"adam_params": {}})
      await a.step(block_ms=10)  # past HOLD_S, so it yields after this batch
      self.assertEqual(await waiting, 1)
      self.assertTrue(a.model.parked)

  async def test_full_finetune_releases_gpu_on_exit(self):
    with patch.object(trainer, "GPU", "node-a:0"):
      t = Trainer(self.bus, "a", Model(tiny), worker="a")
      run = new_id()
      await self.bus.create(Run(run, "a", "tiny", 0))
      await self.submit(run, Op.CREATE_MODEL, {"base_model": "tiny", "rank": 0, "lora": {}})
      await self.submit(run, Op.DELETE_MODEL, {})

      await t.serve()

      self.assertIsNone(await self.bus.owner("gpu:node-a:0"))

  async def test_cancelled_step_withdraws_gpu_request(self):
    with patch.object(trainer, "GPU", "node-a:0"):
      a, _ = await self.trainer("a")
      await a.step(block_ms=10)

      b, _ = await self.trainer("b")
      waiting = asyncio.create_task(b.step(block_ms=10))
      await asyncio.sleep(0.3)

      self.assertTrue(await self.bus.others_requested("node-a:0", "a"))

      waiting.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await waiting

      self.assertFalse(await self.bus.others_requested("node-a:0", "a"))


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestParking(unittest.TestCase):
  def test_parking_frees_the_gpu_and_training_continues_exactly(self):
    parked, resident = Model(lambda _: tiny().cuda()), Model(lambda _: tiny().cuda())
    data = wire.datums(wire.parse(proto_body("x")))
    losses = {id(parked): [], id(resident): []}
    for step in range(3):
      for model in (parked, resident):
        if step == 0:
          model.create("x", "tiny", 4, LORA)
        _, metrics = model.forward_backward("x", data, "cross_entropy", {}, False)
        losses[id(model)].append(metrics["loss:sum"])
        if model is parked:  # between forward_backward and optim_step, so gradients move too
          # what the allocator holds for them: each tensor's bytes, rounded up to its 512 byte blocks
          on_gpu = sum(-(-t.numel() * t.element_size() // 512) * 512 for t in model.tensors() if t.is_cuda)
          before = torch.cuda.memory_allocated()
          model.park()
          self.assertFalse(any(t.is_cuda for t in model.tensors()))
          self.assertEqual(before - torch.cuda.memory_allocated(), on_gpu)  # all of it freed, nothing else
          model.resume()
          self.assertTrue(all(t.is_cuda or not t.is_pinned() for t in model.tensors()))
        model.optim_step("x", {"learning_rate": 1e-2})
    self.assertEqual(losses[id(parked)], losses[id(resident)])


if __name__ == "__main__":
  unittest.main()
