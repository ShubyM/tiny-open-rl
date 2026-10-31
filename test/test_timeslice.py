import asyncio
import json
import unittest
from unittest.mock import patch

import torch

from test.helpers import RedisTest, new_id, tiny
from test.test_api import proto_body
from tinyrl.api import wire
from tinyrl.bus import Op, Run
from tinyrl.control.turns import GpuTurn
from tinyrl.workers import trainer
from tinyrl.workers.model import Model
from tinyrl.workers.trainer import Trainer

LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


class TestTakingTurns(RedisTest):
  """Two trainers whose workers the node daemon put on one GPU: the scheduler decides whose turn it is."""

  async def trainer(self, worker: str) -> tuple[Trainer, str]:
    t = Trainer(self.bus, worker, Model(tiny), owner=worker)
    await t.own()
    run = new_id()
    await self.bus.create(Run(run, worker, "tiny", 4))
    await self.submit(run, Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA})
    return t, run

  async def submit(self, run: str, op: Op, body: dict) -> str:
    return await self.bus.submit(run, op, json.dumps(body).encode())

  async def until_it_yields(self, t: Trainer) -> None:
    for _ in range(100):
      await t.step(block_ms=10)
      if not t.turn.held:
        return
    self.fail(f"{t.worker} kept its GPU")

  async def test_an_idle_trainer_yields_its_gpu_to_one_that_waits(self):
    with patch.object(trainer, "GPU", "node-a:0"):
      await self.run_scheduler({"a": [0], "b": [0]})
      a, _ = await self.trainer("a")
      await a.step(block_ms=10)
      b, run_b = await self.trainer("b")
      waiting = asyncio.create_task(b.step(block_ms=10))
      await asyncio.sleep(0.3)
      self.assertFalse(waiting.done())  # b took its command, but a holds the GPU
      self.assertEqual(len((await self.bus.queue(run_b))["in_flight"]), 1)

      await self.until_it_yields(a)
      self.assertEqual(await asyncio.wait_for(waiting, 5), 1)
      self.assertTrue(a.model.parked)
      self.assertEqual((a.turn.held, b.turn.held), (False, True))

  async def test_a_busy_trainer_keeps_its_gpu_for_its_quantum(self):
    with patch.object(trainer, "GPU", "node-a:0"):
      await self.run_scheduler({"a": [0], "b": [0]}, quantum=1.0)
      a, run_a = await self.trainer("a")
      await a.step(block_ms=10)
      b, _ = await self.trainer("b")
      waiting = asyncio.create_task(b.step(block_ms=10))

      for _ in range(3):  # busy for well under its quantum
        await self.submit(run_a, Op.OPTIM_STEP, {"adam_params": {}})
        await a.step(block_ms=10)
        await asyncio.sleep(0.1)
      self.assertFalse(waiting.done())

      await asyncio.sleep(1.0)
      await self.submit(run_a, Op.OPTIM_STEP, {"adam_params": {}})
      await a.step(block_ms=10)  # its quantum ran out, so it yields after this batch
      self.assertEqual(await asyncio.wait_for(waiting, 5), 1)
      self.assertTrue(a.model.parked)

  async def test_a_stopping_trainer_parks_before_it_gives_its_gpu_back(self):
    with patch.object(trainer, "GPU", "node-a:0"):
      await self.run_scheduler({"a": [0], "b": [0]})
      a, _ = await self.trainer("a")
      await a.step(block_ms=10)
      b = GpuTurn(self.bus, "b", shared=True)
      waiting = asyncio.create_task(b.take(1))
      await asyncio.sleep(0.3)
      self.assertFalse(waiting.done())
      a.stop()  # what SIGTERM calls, after which serve drains and leaves
      await a.drain()
      await a.leave()
      self.assertTrue(await asyncio.wait_for(waiting, 5))
      self.assertTrue(a.model.parked)

  async def test_a_trainer_that_exits_gives_its_turn_to_the_next(self):
    with patch.object(trainer, "GPU", "node-a:0"):
      await self.run_scheduler({"a": [0], "b": [0]})
      t = Trainer(self.bus, "a", Model(tiny), owner="a")
      run = new_id()
      await self.bus.create(Run(run, "a", "tiny", 0))
      await self.submit(run, Op.CREATE_MODEL, {"base_model": "tiny", "rank": 0, "lora": {}})
      await self.submit(run, Op.DELETE_MODEL, {})
      b, _ = await self.trainer("b")

      await t.serve()
      self.assertEqual(await asyncio.wait_for(b.step(block_ms=10), 5), 1)


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
