import asyncio
import json
import os
import socket
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from test.helpers import RedisTest, new_id, tiny
from tinyrl.bus import Op, Run
from tinyrl.node_daemon import NodeDaemon

LEASE_MS = 2000
LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


class TestNodeDaemon(RedisTest):
  async def test_the_node_daemon_starts_a_trainer_for_a_pool_with_runs_and_again_when_it_dies(self):
    base = f"{tempfile.mkdtemp()}/tiny"
    tiny().save_pretrained(base)
    pool, run = f"lora:{base}", new_id()
    await self.bus.create(Run(run, pool, base, 4))
    await self.bus.create(Run(new_id(), "empty", base, 4))
    await self.bus.drop((await self.bus.runs("empty"))[0], "empty")

    with patch.dict(os.environ, {"REDIS_URL": self.url, "LEASE_MS": str(LEASE_MS)}):
      daemon = NodeDaemon(self.bus, gpus=2)
      try:
        await daemon.tick()  # it takes the scheduler lease
        await daemon.tick()  # and places the pool on itself
        self.assertEqual(list(daemon.trainers), [pool])  # not the pool whose runs are all gone
        request_id = await self.bus.submit(
          run, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 4, "lora": LORA}).encode()
        )
        self.assertTrue((await self.bus.result(request_id, 60)).ok)

        gpu, first = daemon.trainers[pool]
        await daemon.tick()
        self.assertIs(daemon.trainers[pool][1], first)  # one trainer per pool

        first.kill()
        await asyncio.to_thread(first.wait)
        while await self.bus.owner(pool):  # its lease runs out
          await asyncio.sleep(0.1)
        await daemon.tick()
        self.assertIsNot(daemon.trainers[pool][1], first)
        request_id = await self.bus.submit(
          run, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 4, "lora": LORA}).encode()
        )
        self.assertTrue((await self.bus.result(request_id, 60)).ok)
      finally:
        for _, proc in daemon.trainers.values():
          proc.terminate()
          proc.wait()

  @patch("tinyrl.bus.NODE_MS", 600)
  @patch("tinyrl.bus.LEASE_MS", 600)
  async def test_pools_spread_over_nodes_and_move_off_a_dead_one(self):
    for run in ("r1", "r2"):
      await self.bus.create(Run(run, f"pool-{run}", "base", 4))
    a, b = NodeDaemon(self.bus, gpus=1, node="a"), NodeDaemon(self.bus, gpus=1, node="b")
    with patch("subprocess.Popen", side_effect=lambda *_, **__: MagicMock(**{"poll.return_value": None})):
      await b.bus.heartbeat("b", 1)
      await a.tick()  # a takes the scheduler lease
      await a.tick()  # and places one pool on each live node daemon
      await b.tick()
      self.assertEqual(sorted(await self.bus.placements()), ["pool-r1", "pool-r2"])
      self.assertEqual(sorted((await self.bus.placements()).values()), ["a", "b"])
      self.assertEqual(len(a.trainers) + len(b.trainers), 2)

      for _ in range(30):  # a dies, so b takes the scheduler lease and a's pool once their leases run out
        await asyncio.sleep(0.1)
        await b.tick()
      self.assertEqual(set((await self.bus.placements()).values()), {"b"})
      self.assertEqual(sorted(b.trainers), ["pool-r1", "pool-r2"])

      proc = b.trainers["pool-r1"][1]
      await self.bus.drop("r1", "pool-r1")
      await b.tick()  # an empty pool's process is stopped
      self.assertEqual(sorted(b.trainers), ["pool-r2"])
      proc.terminate.assert_called()

  def test_each_trainer_gets_the_least_used_gpu(self):
    daemon = NodeDaemon(None, gpus=2)
    with patch("subprocess.Popen") as popen:
      daemon.trainers = {"a": daemon.start("a")}
      daemon.trainers["b"] = daemon.start("b")
      daemon.trainers["c"] = daemon.start("c")
    gpus = [call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for call in popen.call_args_list]
    self.assertEqual(gpus, ["0", "1", "0"])

  def test_a_wide_pool_gets_an_aligned_block_of_the_least_used_gpus(self):
    daemon = NodeDaemon(None, gpus=4)
    with patch("subprocess.Popen") as popen:
      daemon.trainers = {"a": daemon.start("a")}  # one GPU: 0
      daemon.trainers["tp"] = daemon.start("tp", width=2)  # 0-1 is busier than 2-3
      daemon.trainers["b"] = daemon.start("b")
    envs = [call.kwargs["env"] for call in popen.call_args_list]
    self.assertEqual([env["CUDA_VISIBLE_DEVICES"] for env in envs], ["0", "2,3", "1"])
    self.assertEqual(envs[1]["GPU"].split(","), [f"{socket.gethostname()}:2", f"{socket.gethostname()}:3"])


if __name__ == "__main__":
  unittest.main()
