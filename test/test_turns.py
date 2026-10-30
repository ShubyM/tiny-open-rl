import asyncio

from test.helpers import RedisTest
from tinyrl import scheduler
from tinyrl.turns import GpuTurn


class TestScheduler(RedisTest):
  async def start(self, *pools: str, quantum: float = 30.0) -> asyncio.Task:
    await self.running({pool: [0] for pool in pools})
    return asyncio.create_task(scheduler.run(self.bus, quantum))

  async def decisions(self) -> list[tuple[str, str]]:
    return [(e.kind, e.body["worker"]) for e in await self.bus.events() if e.kind in ("granted", "revoked")]

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
          ("granted", "a"),
          ("revoked", "a"),
          ("granted", "b"),
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
    self.assertEqual(await self.decisions(), [("granted", "a")])

  async def test_a_busy_holder_is_revoked_once_its_quantum_runs_out(self):
    task = await self.start("a", "b", quantum=0.2)
    a = GpuTurn(self.bus, "a", shared=True)
    try:
      self.assertTrue(await asyncio.wait_for(a.take(1), 5))
      await self.bus.emit("wants_gpu", worker="b", queued=1)
      await asyncio.sleep(0.5)
      self.assertTrue(await a.revoked())
      self.assertEqual(await self.decisions(), [("granted", "a"), ("revoked", "a")])
    finally:
      task.cancel()

  async def test_old_process_and_old_turn_decisions_cannot_revoke_a_new_turn(self):
    turn = GpuTurn(self.bus, "current", shared=True)
    turn.after = await self.bus.emit("wants_gpu", worker="current", queued=1)
    await self.bus.emit("granted", worker="old-process", turn="old")
    await self.bus.emit("granted", worker="current", turn="first")
    self.assertEqual(await turn.decision(0), "granted")
    turn.held = True
    await turn.give_back()
    await self.bus.emit("granted", worker="current", turn="second")
    self.assertEqual(await turn.decision(0), "granted")
    turn.held = True
    await self.bus.emit("revoked", worker="current", turn="first")
    await self.bus.emit("revoked", worker="old-process", turn="second")
    self.assertFalse(await turn.revoked())
    await self.bus.emit("revoked", worker="current", turn="second")
    self.assertTrue(await turn.revoked())
