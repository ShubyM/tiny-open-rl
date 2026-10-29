import asyncio

from test.helpers import RedisTest
from tinyrl import control, scheduler
from tinyrl.turns import GpuTurn


class TestScheduler(RedisTest):
  async def start(self, *pools: str, quantum: float = 30.0) -> asyncio.Task:
    await self.running({pool: [0] for pool in pools})
    return asyncio.create_task(scheduler.run(self.bus, control.Policy(quantum=quantum)))

  async def decisions(self) -> list[tuple[str, str, str]]:
    return [
      (e.kind, e.body["pool"], e.body["rule"]) for e in await self.bus.events() if e.kind in ("granted", "revoked")
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
    await self.bus.emit("wants_gpu", pool="b", queued=1)
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
      await self.bus.emit("wants_gpu", pool="b", queued=1)
      await asyncio.sleep(0.5)
      self.assertTrue(await a.revoked())
      self.assertEqual(
        await self.decisions(), [("granted", "a", "turn.granted"), ("revoked", "a", "turn.quantum_expired")]
      )
    finally:
      task.cancel()
