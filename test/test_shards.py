import asyncio
from unittest.mock import patch

from test.helpers import RedisTest, new_id
from tinyrl import trainer
from tinyrl.bus import Run
from tinyrl.gateway import pool_for


class TestShards(RedisTest):
  async def place(self, n: int = 1) -> list[str]:
    pools = []
    for _ in range(n):
      run = new_id()
      pools.append(await pool_for(self.bus, run, "tiny", 4, {}))
      await self.bus.create(Run(run, pools[-1], "tiny", 4))
    return pools

  async def test_without_node_daemons_a_base_keeps_one_shard(self):
    await self.place()
    await self.bus.report("lora:tiny:0", 1.0)
    self.assertEqual(await self.place(2), ["lora:tiny:0", "lora:tiny:0"])

  async def test_a_busy_shard_opens_another_up_to_one_per_gpu(self):
    await self.bus.heartbeat("node-a", 2)
    await self.place()
    await self.bus.report("lora:tiny:0", 0.9)
    self.assertEqual(await self.place(), ["lora:tiny:1"])
    await self.bus.report("lora:tiny:1", 0.9)
    self.assertEqual(await self.place(), ["lora:tiny:0"])  # no third GPU, so the less loaded of the two

  async def test_a_burst_of_runs_spreads_before_any_shard_reports_again(self):
    await self.bus.heartbeat("node-a", 2)
    await self.place()
    await self.bus.report("lora:tiny:0", 0.5)  # one run keeps it half busy
    pools = await self.place(4)
    self.assertEqual(sorted(pools), ["lora:tiny:0", "lora:tiny:0", "lora:tiny:1", "lora:tiny:1"])

  async def test_a_report_replaces_the_estimate_for_runs_placed_before_it(self):
    await self.bus.heartbeat("node-a", 2)
    await self.place(3)
    self.assertEqual((await self.bus.loads(["lora:tiny:0"]))[0][1], 3)
    await self.bus.report("lora:tiny:0", 0.2)
    self.assertEqual(await self.bus.loads(["lora:tiny:0"]), [(0.2, 0, 3)])

  async def test_a_dead_node_daemons_gpus_stop_counting(self):
    await self.bus.heartbeat("node-a", 2)
    await self.bus.r.hset("tinyrl:nodes", "node-b", '{"gpus": 8, "until": 0}')
    self.assertEqual(await self.bus.gpus(), 2)

  async def test_a_trainer_reports_how_busy_it_was(self):
    t = trainer.Trainer(self.bus, "lora:tiny:0", trainer.Model())
    await t.own()
    with patch.object(trainer, "LEASE_MS", 40), patch.object(trainer, "UTIL_MS", 100):
      keeping = asyncio.create_task(t.keep())
      t.busy = 0.05  # 50 ms of commands in a 100 ms window
      while (await self.bus.loads(["lora:tiny:0"]))[0][0] is None:
        await asyncio.sleep(0.01)
      t.done = True
      await keeping
    util = (await self.bus.loads(["lora:tiny:0"]))[0][0]
    self.assertTrue(0.2 < util <= 0.5, util)
