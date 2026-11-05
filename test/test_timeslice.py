import asyncio
import json
import unittest

import torch

from test.execution import TestNode
from test.helpers import RedisTest, new_id, tiny
from test.test_api import proto_body
from tinyrl.api import wire
from tinyrl.bus import Op, Run
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

  async def until(self, done):
    async with asyncio.timeout(5):
      while not done():
        self.node.dispatch()
        await self.node.wait(0.01)

  async def trainer(self, worker, gpus=None):
    run = new_id()
    await self.bus.create_run(Run(run, worker, "tiny", 4, lora=LORA, gpus=len(gpus or [0])))
    await self.node.add(worker, gpus=gpus)
    await self.submit(run, Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA})
    return run

  async def submit(self, run, op, body):
    return await self.bus.submit_command(run, op, json.dumps(body).encode())

  async def test_an_idle_trainer_stays_resident_until_another_worker_waits(self):
    await self.trainer("a")
    self.assertEqual(await self.node.step("a"), 1)
    self.assertEqual(await self.node.step("a"), 0)
    a = self.node.running["a"]
    self.assertIsNotNone(a.since)
    self.assertFalse(a.model.trainer.model.parked)
    await self.trainer("b")
    self.assertEqual(await self.node.step("b"), 1)
    self.assertTrue(a.model.trainer.model.parked)
    self.assertIsNone(a.since)
    self.assertIsNotNone(self.node.running["b"].since)
    local = {"granted", "revoked", "wants_gpu", "released", "batch_done"}
    self.assertFalse(any(e.kind in local for e in await self.bus.read_events()))

  async def test_a_busy_trainer_yields_after_its_current_command_when_its_quantum_expires(self):
    self.node.state.quantum = 0.3
    run = await self.trainer("a")
    await self.node.step("a")
    first, finish, order = asyncio.Event(), asyncio.Event(), []
    child, original = self.node.running["a"].model, self.node.running["a"].model.call

    async def held(method, *args):
      result = await original(method, *args)
      if method == "execute":
        order.append("a")
        if len(order) == 1:
          first.set()
          await finish.wait()
      return result

    child.call = held
    for _ in range(2):
      await self.submit(run, Op.OPTIM_STEP, {"adam_params": {}})
    work = self.start(self.node.step("a"))
    try:
      await self.until(first.is_set)
      await self.trainer("b")
      other = self.node.running["b"].model
      execute = other.call

      async def observed(method, *args):
        if method == "execute":
          order.append("b")
        return await execute(method, *args)

      other.call = observed
      waiting = self.start(self.node.step("b"))
      await self.until(lambda: self.node.running["b"].ticket > 0)
      self.assertFalse(waiting.done())
      await asyncio.sleep(0.35)
      self.node.dispatch()
      self.assertFalse(waiting.done())  # the outstanding command still owns its GPU after the deadline
      self.assertIsNotNone(self.node.running["a"].since)
      finish.set()
      await asyncio.gather(work, waiting)
      await self.node.step("a")
      self.assertEqual(order, ["a", "b", "a"])
    finally:
      finish.set()

  async def test_parking_must_acknowledge_before_another_worker_is_granted(self):
    await self.trainer("a")
    await self.node.step("a")
    parked, finish = asyncio.Event(), asyncio.Event()
    child = self.node.running["a"].model
    original = child.call

    async def delayed(method, *args):
      if method == "park":
        parked.set()
        await finish.wait()
      return await original(method, *args)

    child.call = delayed
    try:
      await self.trainer("b")
      waiting = self.start(self.node.step("b"))
      await self.until(parked.is_set)
      self.assertFalse(waiting.done())
      self.assertIsNotNone(self.node.running["a"].since)
      self.assertIsNone(self.node.running["b"].since)
      finish.set()
      self.assertEqual(await waiting, 1)
      self.assertIsNone(self.node.running["a"].since)
    finally:
      finish.set()

  async def test_process_exit_must_acknowledge_before_the_gpu_is_released(self):
    await self.trainer("a")
    await self.node.step("a")
    closing, finish = asyncio.Event(), asyncio.Event()
    child, original = self.node.running["a"].model, self.node.running["a"].model.close

    async def delayed():
      closing.set()
      await finish.wait()
      await original()

    child.close = delayed
    self.node.running["a"].cancel()
    try:
      await self.trainer("b")
      waiting = self.start(self.node.step("b"))
      await self.until(closing.is_set)
      self.assertIsNotNone(self.node.running["a"].since)
      self.assertIsNone(self.node.running["b"].since)
      self.assertFalse(waiting.done())
      finish.set()
      self.assertEqual(await waiting, 1)
      self.assertTrue(child.closed)
    finally:
      finish.set()

  async def test_a_waiting_lora_trainer_moves_before_its_next_reservation_without_restarting(self):
    self.node.gpus = 2
    self.node.state.nodes["n"] = (2, 0)
    await self.bus.create_run(Run("reserved", "s", "tiny", 4, role="sampler"))
    await self.node.add("s", gpus=[1])
    run = await self.trainer("a", [0])
    await self.node.step("a")
    entered, finish = asyncio.Event(), asyncio.Event()
    original = self.node.running["a"].model.call

    async def held(method, *args):
      if method == "execute":
        entered.set()
        await finish.wait()
      return await original(method, *args)

    self.node.running["a"].model.call = held
    await self.submit(run, Op.OPTIM_STEP, {"adam_params": {}})
    work = self.start(self.node.step("a"))
    try:
      await self.until(entered.is_set)
      await self.trainer("b", [0])
      child = self.node.running["b"].model
      waiting = self.start(self.node.step("b"))
      await self.until(lambda: self.node.running["b"].ticket > 0)
      self.assertFalse(waiting.done())
      self.node.state.workers["b"].gpus = (1,)
      self.assertEqual(await waiting, 1)
      self.assertIs(self.node.running["b"].model, child)
      self.assertEqual(self.node.running["b"].gpus, (1,))
      self.assertIsNotNone(self.node.running["a"].since)
      self.assertIsNotNone(self.node.running["b"].since)
    finally:
      finish.set()
      await work

  async def test_a_multi_gpu_worker_waits_for_all_its_gpus_at_once(self):
    self.node.gpus = 2
    self.node.state.nodes["n"] = (2, 0)
    await self.trainer("a", [1])
    await self.node.step("a")
    entered, finish = asyncio.Event(), asyncio.Event()
    original = self.node.running["a"].model.call

    async def held(method, *args):
      if method == "park":
        entered.set()
        await finish.wait()
      return await original(method, *args)

    self.node.running["a"].model.call = held
    try:
      await self.trainer("wide", [0, 1])
      waiting = self.start(self.node.step("wide"))
      await self.until(entered.is_set)
      self.assertFalse(waiting.done())
      self.assertIsNone(self.node.running["wide"].since)
      self.assertEqual([r.gpus for r in self.node.running.values() if r.since is not None], [(1,)])
      finish.set()
      self.assertEqual(await waiting, 1)
      self.assertEqual(self.node.running["wide"].gpus, (0, 1))
      self.assertIsNotNone(self.node.running["wide"].since)
    finally:
      finish.set()

  async def test_an_idle_holder_that_resumes_keeps_its_quantum_for_queued_work(self):
    run = await self.trainer("a")
    await self.node.step("a")
    await self.node.step("a")
    entered, finish, order = asyncio.Event(), asyncio.Event(), []
    original = self.node.running["a"].model.call

    async def held(method, *args):
      result = await original(method, *args)
      if method == "execute":
        order.append("a")
        if len(order) == 1:
          entered.set()
          await finish.wait()
      return result

    self.node.running["a"].model.call = held
    for _ in range(2):
      await self.submit(run, Op.OPTIM_STEP, {"adam_params": {}})
    work = self.start(self.node.step("a"))
    try:
      await self.until(entered.is_set)
      await self.trainer("b")
      original_b = self.node.running["b"].model.call

      async def observed(method, *args):
        if method == "execute":
          order.append("b")
        return await original_b(method, *args)

      self.node.running["b"].model.call = observed
      waiting = self.start(self.node.step("b"))
      await self.until(lambda: self.node.running["b"].ticket > 0)
      self.assertFalse(waiting.done())
      finish.set()
      await asyncio.gather(work, waiting)
      self.assertEqual(order, ["a", "a", "b"])
    finally:
      finish.set()


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
