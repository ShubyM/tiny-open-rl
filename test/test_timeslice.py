import asyncio
import json
import unittest

import torch

from test.execution import TestNode
from test.helpers import RedisTest, new_id, tiny
from test.test_api import proto_body
from tinyrl.api import wire
from tinyrl.bus import Ev, Op, Run
from tinyrl.control import scheduler
from tinyrl.workers.model import Model

LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


class TestTakingTurns(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.node, self.tasks = TestNode(self.bus, gpus=1, node="n"), []

  async def asyncTearDown(self):
    for task in self.tasks:
      task.cancel()
    await asyncio.gather(*self.tasks, return_exceptions=True)
    await self.node.aclose()
    await super().asyncTearDown()

  def start(self, coroutine):
    self.tasks.append(task := asyncio.create_task(coroutine))
    return task

  async def trainer(self, worker, gpus=None):
    run = new_id()
    await self.bus.create_run(Run(run, worker, "tiny", 4, lora=LORA, gpus=len(gpus or [0])))
    await self.node.add(worker, gpus=gpus)
    await self.submit(run, Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA})
    return run

  async def submit(self, run, op, body):
    return await self.bus.submit_command(run, op, json.dumps(body).encode())

  async def test_an_idle_trainer_yields_its_gpu_to_one_that_waits(self):
    await self.trainer("a")
    await self.node.step("a", block_ms=10)
    run_b = await self.trainer("b")
    waiting = self.start(self.node.step("b", block_ms=10))
    await asyncio.sleep(0.3)
    self.assertFalse(waiting.done())
    self.assertEqual(len((await self.bus.queued_and_in_flight(run_b))["in_flight"]), 1)
    await self.node.step("a", block_ms=10)
    self.assertEqual(await asyncio.wait_for(waiting, 5), 1)
    self.assertTrue(self.node.running["a"].model.trainer.model.parked)
    self.assertEqual((self.node.running["a"].awake, self.node.running["b"].awake), (False, True))
    self.assertFalse(any(e.kind in (Ev.GRANTED, Ev.REVOKED, Ev.WANTS_GPU) for e in await self.bus.read_events()))

  async def test_a_busy_trainer_keeps_its_gpu_for_its_quantum(self):
    self.node.state.quantum = 1.0
    run = await self.trainer("a")
    await self.node.step("a", block_ms=10)
    await self.trainer("b")
    waiting = self.start(self.node.step("b", block_ms=10))
    for _ in range(3):
      await self.submit(run, Op.OPTIM_STEP, {"adam_params": {}})
      await self.node.step("a", block_ms=10)
      await asyncio.sleep(0.03)
    self.assertFalse(waiting.done())
    await asyncio.sleep(1.0)
    await self.node.step("a", block_ms=10)
    self.assertEqual(await asyncio.wait_for(waiting, 5), 1)
    self.assertTrue(self.node.running["a"].model.trainer.model.parked)

  async def test_parking_must_acknowledge_before_another_worker_is_granted(self):
    await self.trainer("a")
    await self.node.step("a", block_ms=10)
    await self.trainer("b")
    waiting = self.start(self.node.step("b", block_ms=10))
    await asyncio.sleep(0.05)
    parked, finish = asyncio.Event(), asyncio.Event()
    child = self.node.running["a"].model
    original = child.call

    async def delayed(method, *args):
      if method == "park":
        parked.set()
        await finish.wait()
      return await original(method, *args)

    child.call = delayed
    releasing = self.start(self.node.step("a", block_ms=10))
    await asyncio.wait_for(parked.wait(), 5)
    self.assertFalse(waiting.done())
    self.assertEqual(scheduler.holders(self.node.state, ("n", 0)), ["a"])
    finish.set()
    await releasing
    self.assertEqual(await asyncio.wait_for(waiting, 5), 1)

  async def test_process_exit_must_acknowledge_before_the_gpu_is_released(self):
    await self.trainer("a")
    await self.node.step("a", block_ms=10)
    await self.trainer("b")
    waiting = self.start(self.node.step("b", block_ms=10))
    closing, finish = asyncio.Event(), asyncio.Event()
    child = self.node.running["a"].model
    original = child.close

    async def delayed():
      closing.set()
      await finish.wait()
      await original()

    child.close = delayed
    cleanup = self.start(self.node.close_worker("a"))
    await asyncio.wait_for(closing.wait(), 5)
    self.assertEqual(scheduler.holders(self.node.state, ("n", 0)), ["a"])
    self.assertFalse(waiting.done())
    finish.set()
    await cleanup
    self.assertEqual(await asyncio.wait_for(waiting, 5), 1)

  async def test_a_waiting_lora_trainer_moves_before_its_next_grant_without_restarting(self):
    self.node.gpus = 2
    self.node.state.nodes["n"] = (2, 0)
    self.node.note(Ev.RUN_CREATED, run="reserved", worker="s", role="sampler", rank=4)
    self.node.note(Ev.PLACED, worker="s", node="n", gpus=[1])
    await self.trainer("a", [0])
    await self.node.step("a", block_ms=10)
    await self.trainer("b", [0])
    child = self.node.running["b"].model
    waiting = self.start(self.node.step("b", block_ms=10))
    await asyncio.sleep(0.3)
    self.assertFalse(waiting.done())
    self.node.note(Ev.PLACED, worker="b", node="n", gpus=[1])
    self.assertEqual(await asyncio.wait_for(waiting, 5), 1)
    self.assertIs(self.node.running["b"].model, child)
    self.assertEqual(self.node.state.workers["b"].on, (("n", 1),))
    self.assertEqual(scheduler.holders(self.node.state, ("n", 0)), ["a"])
    self.assertEqual(scheduler.holders(self.node.state, ("n", 1)), ["b"])

  async def test_a_multi_gpu_worker_waits_for_all_its_gpus_at_once(self):
    self.node.gpus = 2
    self.node.state.nodes["n"] = (2, 0)
    await self.trainer("a", [1])
    await self.node.step("a", block_ms=10)
    await self.trainer("wide", [0, 1])
    waiting = self.start(self.node.step("wide", block_ms=10))
    await asyncio.sleep(0.3)
    self.assertFalse(waiting.done())
    self.assertEqual(scheduler.holders(self.node.state, ("n", 0)), [])
    self.assertEqual(scheduler.holders(self.node.state, ("n", 1)), ["a"])
    await self.node.step("a", block_ms=10)
    self.assertEqual(await asyncio.wait_for(waiting, 5), 1)
    self.assertEqual(scheduler.holders(self.node.state, ("n", 0)), ["wide"])
    self.assertEqual(scheduler.holders(self.node.state, ("n", 1)), ["wide"])

  async def test_an_idle_holder_that_resumes_is_not_preempted_as_idle(self):
    run = await self.trainer("a")
    await self.node.step("a", block_ms=10)
    await self.node.step("a", block_ms=10)
    self.assertTrue(self.node.state.workers["a"].idle)
    await self.submit(run, Op.OPTIM_STEP, {"adam_params": {}})
    await self.node.step("a", block_ms=10)
    self.assertFalse(self.node.state.workers["a"].idle)
    await self.trainer("b")
    waiting = self.start(self.node.step("b", block_ms=10))
    await asyncio.sleep(0.3)
    self.assertFalse(waiting.done())
    self.assertFalse(self.node.state.workers["a"].revoked)
    await self.node.step("a", block_ms=10)
    await asyncio.wait_for(waiting, 5)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestParking(unittest.TestCase):
  def test_parking_frees_the_gpu_and_training_continues_exactly(self):
    parked, resident = Model(lambda _: tiny().cuda()), Model(lambda _: tiny().cuda())
    data = wire.datums(wire.parse(proto_body("x")))
    losses = {id(parked): [], id(resident): []}
    for step in range(3):
      for model in (parked, resident):
        if step == 0:
          model.create_run("x", "tiny", 4, LORA)
        _, metrics = model.forward_backward("x", data, "cross_entropy", {}, False)
        losses[id(model)].append(metrics["loss:sum"])
        if model is parked:  # between forward_backward and optim_step, so gradients move too
          # what the allocator holds for them: each tensor's bytes, rounded up to its 512 byte blocks
          on_gpu = sum(-(-t.numel() * t.element_size() // 512) * 512 for t in model.gpu_tensors() if t.is_cuda)
          before = torch.cuda.memory_allocated()
          model.park_on_host()
          self.assertFalse(any(t.is_cuda for t in model.gpu_tensors()))
          self.assertEqual(before - torch.cuda.memory_allocated(), on_gpu)  # all of it freed, nothing else
          model.resume_on_gpu()
          self.assertTrue(all(t.is_cuda or not t.is_pinned() for t in model.gpu_tensors()))
        model.optim_step("x", {"learning_rate": 1e-2})
    self.assertEqual(losses[id(parked)], losses[id(resident)])


if __name__ == "__main__":
  unittest.main()
