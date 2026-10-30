import asyncio
from unittest.mock import patch

from test.helpers import RedisTest, new_id
from tinyrl import trainer
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

  async def test_a_move_token_cannot_be_changed_while_source_is_draining(self):
    run = new_id()
    await self.bus.create(Run(run, "w1", "tiny", 4))
    await self.bus.own("w1", "process1")
    self.assertTrue(await self.bus.request_move(run, "w2", "one"))
    self.assertTrue(await self.bus.request_move(run, "w2", "one"))
    self.assertFalse(await self.bus.request_move(run, "w3", "two"))
    self.assertEqual(await self.bus.moves("w1", "other-process"), {})
    self.assertFalse(await self.bus.release_run(run, "w1", "process1", "two"))

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

  async def test_a_trainer_reports_how_busy_it_was(self):
    t = trainer.Trainer(self.bus, "lora:tiny:0", trainer.Model())
    await t.own()
    with patch.object(trainer, "LEASE_MS", 40), patch.object(trainer, "UTIL_MS", 100):
      keeping = asyncio.create_task(t.keep())
      t.busy = 0.05  # 50 ms of commands in a 100 ms window
      while (await self.bus.loads(["lora:tiny:0"]))[0][0] is None:
        await asyncio.sleep(0.01)
      keeping.cancel()  # keep() renews until serve() cancels it
    util = (await self.bus.loads(["lora:tiny:0"]))[0][0]
    self.assertTrue(0.2 < util <= 0.5, util)
