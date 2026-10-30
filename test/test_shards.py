import asyncio
from unittest.mock import patch

from test.helpers import RedisTest, new_id
from tinyrl.bus import Run


class TestAssignments(RedisTest):
  async def test_runs_wait_for_scheduler_assignment_without_inventing_a_worker(self):
    runs = [new_id() for _ in range(4)]
    for run in runs:
      await self.bus.create(Run(run, "", "tiny", 4))
    self.assertEqual(await self.bus.pools(), [])
    for run in runs:
      self.assertEqual((await self.bus.run(run))["pool"], "")
    for run, worker in zip(runs, ["w1", "w2", "w1", "w2"]):
      self.assertTrue(await self.bus.assign(run, worker, rule="test"))
    self.assertEqual(sorted(await self.bus.runs("w1")), sorted(runs[::2]))
    self.assertEqual(sorted(await self.bus.runs("w2")), sorted(runs[1::2]))

  async def test_assignment_retry_does_not_duplicate_facts_or_move_a_live_run(self):
    run = new_id()
    await self.bus.create(Run(run, "", "tiny", 4))
    self.assertTrue(await self.bus.assign(run, "w1"))
    self.assertTrue(await self.bus.assign(run, "w1"))
    self.assertFalse(await self.bus.assign(run, "w2"))
    self.assertEqual([e.body["worker"] for e in await self.bus.events() if e.kind == "run_assigned"], ["w1"])

  async def test_a_dead_node_daemons_gpus_stop_counting(self):
    await self.bus.heartbeat("node-a", 2)
    with patch("tinyrl.bus.NODE_MS", 5):
      await self.bus.heartbeat("node-b", 8)
    await asyncio.sleep(0.02)
    self.assertEqual(await self.bus.gpus(), 2)

  async def test_heartbeat_renewal_survives_the_old_expiry_and_ignores_host_clock(self):
    with patch("tinyrl.bus.NODE_MS", 5):
      await self.bus.heartbeat("node", 2)
    with patch("tinyrl.bus.time.time", return_value=-1e12):
      await self.bus.heartbeat("node", 4)
    await asyncio.sleep(0.02)
    with patch("tinyrl.bus.time.time", return_value=1e12):
      self.assertEqual(await self.bus.nodes(), {"node": 4})
