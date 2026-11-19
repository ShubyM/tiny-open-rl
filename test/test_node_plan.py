import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

from test.helpers import RedisTest
from tinyrl.bus import Bus, connect
from tinyrl.control import scheduler
from tinyrl.control.accounting import Charge
from tinyrl.control.node_daemon import NodeDaemon
from tinyrl.types import Op, Reply, Role, Run


class TestPublishedPlan(RedisTest):
  def daemon(self, bus):
    return NodeDaemon(bus, node="node", gpus=2, gpu_mib=0, store=Mock())

  async def test_scheduler_loads_its_plan_once_per_process_and_resumes_worker_numbering(self):
    observer = self.daemon(self.bus)
    await self.bus.node_heartbeat(observer.id, 2)
    await self.bus.write_plan({"made": 40})

    @asynccontextmanager
    async def running_scheduler():
      bus = Bus(connect(self.url))
      self.addAsyncCleanup(bus.aclose)
      with patch.object(bus, "read_plan", new=AsyncMock(wraps=bus.read_plan)) as reads:
        task = asyncio.create_task(scheduler.run(bus))
        try:
          yield reads
        finally:
          task.cancel()
          with self.assertRaises(asyncio.CancelledError):
            await task

    async def wait_for_homes(expected):
      async with asyncio.timeout(5):
        while (await observer.read_state()).home != expected:
          await asyncio.sleep(0.01)

    await self.bus.create_run(Run("keep", "tiny", 0))
    async with running_scheduler() as reads:
      await wait_for_homes({"keep": "p41"})
      await self.bus.create_run(Run("temporary", "tiny", 0))
      await wait_for_homes({"keep": "p41", "temporary": "p42"})
      await self.bus.delete_run("temporary")
      await wait_for_homes({"keep": "p41"})
      reads.assert_awaited_once()

    await self.bus.create_run(Run("new", "tiny", 0))
    async with running_scheduler() as reads:
      await wait_for_homes({"keep": "p41", "new": "p43"})
      reads.assert_awaited_once()

  async def test_restarted_node_recovers_assignments_and_accounts_for_work_since_publication(self):
    await self.bus.node_heartbeat("a", 1)
    await self.bus.node_heartbeat("node", 1)
    await self.bus.create_run(Run("trainer", "tiny", 4, account="team"))
    sampler = Run("sampler", "tiny", 4, role=Role.SAMPLER, train_run="trainer", model_path="tinker://trainer/sampler/a")
    await self.bus.create_run(sampler)
    await scheduler.tick(self.bus)
    await self.bus.submit_command("sampler", Op.SAMPLE, b"{}")
    [cmd] = await self.bus.take_commands(["sampler"], "replica")
    self.assertTrue(await self.bus.commit_sample(cmd, Reply(True, "application/json", b"{}"), "team", Charge(gpu_ms=8)))

    fresh = Bus(connect(self.url))
    self.addAsyncCleanup(fresh.aclose)
    with patch.object(fresh, "observations", side_effect=AssertionError("node requested full cluster state")):
      state = await self.daemon(fresh).read_state()
    [pool] = state.procs_of("sampler")
    self.assertEqual(state.procs[pool].node, "node")
    self.assertEqual(state.procs[state.home["trainer"]].node, "a")
    self.assertEqual(state.runs(state.home["trainer"]), ["trainer"])
    self.assertEqual(state.jobs["sampler"].record["model_path"], sampler.model_path)
    self.assertEqual(state.account("sampler"), "team")
    self.assertEqual(state.ready_lag("sampler"), -8)

  async def test_pool_membership_changes_and_wakes_nodes_at_publication(self):
    node = self.daemon(self.bus)
    await self.bus.node_heartbeat(node.id, 2)
    await self.bus.create_run(Run("first", "tiny", 4, role=Role.SAMPLER))
    await scheduler.tick(self.bus)
    [pool] = (await node.read_state()).procs_of("first")
    cursor = await self.bus.wait_wake("0-0", 0, "plan_wake")

    await self.bus.create_run(Run("second", "tiny", 4, role=Role.SAMPLER))
    self.assertEqual((await node.read_state()).runs(pool), ["first"])
    self.assertEqual(await self.bus.wait_wake(cursor, 0, "plan_wake"), cursor)
    await scheduler.tick(self.bus)
    published = await self.bus.wait_wake(cursor, 0, "plan_wake")
    self.assertNotEqual(published, cursor)
    self.assertEqual((await node.read_state()).runs(pool), ["first", "second"])

    await self.bus.delete_run("first")
    self.assertEqual((await node.read_state()).runs(pool), ["first", "second"])
    self.assertEqual(await self.bus.wait_wake(published, 0, "plan_wake"), published)
    await scheduler.tick(self.bus)
    self.assertNotEqual(await self.bus.wait_wake(published, 0, "plan_wake"), published)
    self.assertEqual((await node.read_state()).runs(pool), ["second"])

  async def test_legacy_plan_waits_for_scheduler_to_publish_its_run_records(self):
    node = self.daemon(self.bus)
    await self.bus.node_heartbeat(node.id, 2)
    await self.bus.create_run(Run("r", "tiny", 4))
    await self.bus.write_plan(
      {
        "procs": {"p1": {"spec": {"role": "trainer", "base": "tiny", "lora": True}, "node": node.id}},
        "home": {"r": ["p1"]},
      }
    )
    self.assertEqual((await node.read_state()).runs("p1"), [])
    await scheduler.tick(self.bus)
    restored = await node.read_state()
    self.assertEqual(restored.runs("p1"), ["r"])
    self.assertEqual(restored.home, {"r": "p1"})
