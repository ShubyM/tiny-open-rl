import asyncio
from unittest.mock import patch

from test.helpers import RedisTest
from tinyrl.control import scheduler
from tinyrl.control.turns import GpuTurn


class TestScheduler(RedisTest):
  async def start(self, *workers: str, quantum: float = 30.0) -> asyncio.Task:
    await self.running({worker: [0] for worker in workers})
    return asyncio.create_task(scheduler.run(self.bus, quantum))

  async def decisions(self) -> list[tuple[str, str, str]]:
    return [
      (e.kind, e.body["worker"], e.body["rule"]) for e in await self.bus.events() if e.kind in ("granted", "revoked")
    ]

  async def test_a_worker_gets_its_turn_from_the_log_and_gives_it_back_when_revoked(self):
    task = await self.start("a", "b")
    a, b = GpuTurn(self.bus, "a", shared=True), GpuTurn(self.bus, "b", shared=True)
    try:
      self.assertTrue(await asyncio.wait_for(a.take(1), 5))
      waiting = asyncio.create_task(b.take(1))
      await asyncio.sleep(0.3)
      self.assertFalse(waiting.done())
      await a.went_idle()
      for _ in range(50):
        if await a.revoked():
          break
        await asyncio.sleep(0.05)
      await a.give_back()
      self.assertTrue(await asyncio.wait_for(waiting, 5))
      self.assertEqual(
        await self.decisions(),
        [
          ("granted", "a", "turn.granted"),
          ("revoked", "a", "turn.idle_holder_yields"),
          ("granted", "b", "turn.granted"),
        ],
      )
    finally:
      task.cancel()

  async def test_a_restarted_scheduler_decides_nothing_twice(self):
    task = await self.start("a", "b")
    a = GpuTurn(self.bus, "a", shared=True)
    self.assertTrue(await asyncio.wait_for(a.take(1), 5))
    await self.bus.emit("wants_gpu", worker="b", queued=1)
    task.cancel()
    task = asyncio.create_task(scheduler.run(self.bus))
    await asyncio.sleep(0.5)
    task.cancel()
    self.assertEqual(await self.decisions(), [("granted", "a", "turn.granted")])

  async def test_a_busy_holder_is_revoked_once_its_quantum_runs_out(self):
    task = await self.start("a", "b", quantum=0.2)
    a = GpuTurn(self.bus, "a", shared=True)
    try:
      self.assertTrue(await asyncio.wait_for(a.take(1), 5))
      await self.bus.emit("wants_gpu", worker="b", queued=1)
      await asyncio.sleep(0.5)
      self.assertTrue(await a.revoked())
      self.assertEqual(
        await self.decisions(), [("granted", "a", "turn.granted"), ("revoked", "a", "turn.quantum_expired")]
      )
    finally:
      task.cancel()

  @patch("tinyrl.control.turns.LEASE_MS", 200, create=True)
  async def test_a_waiting_worker_emits_wants_gpu_once_and_can_take_a_second_turn(self):
    task = await self.start("a", "b")
    a = GpuTurn(self.bus, "a", shared=True)
    b = GpuTurn(self.bus, "b", shared=True)
    try:
      self.assertTrue(await asyncio.wait_for(a.take(1), 5))
      waiting = asyncio.create_task(b.take(1))
      for _ in range(5):
        await asyncio.sleep(0.05)
        await self.bus.emit("random_event")
      self.assertFalse(waiting.done())
      wants = [e for e in await self.bus.events() if e.kind == "wants_gpu" and e.body.get("worker") == "b"]
      self.assertEqual(len(wants), 1)

      await a.went_idle()
      while not await a.revoked():
        await asyncio.sleep(0.02)
      await a.give_back()
      self.assertTrue(await asyncio.wait_for(waiting, 5))

      second = asyncio.create_task(a.take(1))
      await b.went_idle()
      while not await b.revoked():
        await asyncio.sleep(0.02)
      await b.give_back()
      self.assertTrue(await asyncio.wait_for(second, 5))
    finally:
      task.cancel()
