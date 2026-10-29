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
    daemon = NodeDaemon(None, gpus=8)
    with patch("subprocess.Popen") as popen:
      daemon.trainers = {"lora:a:0": daemon.start("lora:a:0")}
      daemon.trainers["lora:b:0"] = daemon.start("lora:b:0")
      daemon.trainers["lora:c:0"] = daemon.start("lora:c:0")
    gpus = [call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for call in popen.call_args_list]
    self.assertEqual(gpus, ["0", "1", "0"])

  def test_a_wide_pool_gets_an_aligned_block_of_the_least_used_gpus(self):
    daemon = NodeDaemon(None, gpus=16)
    with patch("subprocess.Popen") as popen:
      daemon.trainers = {"lora:a:0": daemon.start("lora:a:0", width=2)}
      daemon.trainers["lora:tp:0"] = daemon.start("lora:tp:0", width=2)
      daemon.trainers["lora:b:0"] = daemon.start("lora:b:0", width=2)
    envs = [call.kwargs["env"] for call in popen.call_args_list]
    self.assertEqual([env["CUDA_VISIBLE_DEVICES"] for env in envs], ["0,1", "2,3", "0,1"])
    self.assertEqual(envs[1]["GPU"].split(","), [f"{socket.gethostname()}:2", f"{socket.gethostname()}:3"])

  @patch("tinyrl.node_daemon.subprocess.Popen")
  def test_start_sets_cuda_alloc_conf_on_trainer(self, mock_popen):
    import os

    daemon = NodeDaemon(self.bus, 2)
    daemon.start("lora:base:0", 1)
    env1 = mock_popen.call_args[1]["env"]
    self.assertEqual(env1["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")

    daemon.start("sampler:base", 1)
    env2 = mock_popen.call_args[1]["env"]
    self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", env2)

    with patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}):
      daemon.start("lora:base:1", 1)
      env3 = mock_popen.call_args[1]["env"]
      self.assertEqual(env3["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:512")

  @patch("tinyrl.node_daemon.subprocess.Popen")
  def test_sampler_gpu_memory_and_lease_slots(self, mock_popen):
    daemon = NodeDaemon(None, gpus=8, gpu_mib=81559)
    host = socket.gethostname()

    daemon.trainers["sampler:fft:Qwen"] = daemon.start("sampler:fft:Qwen")
    env1 = mock_popen.call_args[1]["env"]
    self.assertEqual(env1["GPU"], f"{host}:6:0")
    self.assertEqual(env1["TINYRL_GPU_MEM"], "0.42")
    self.assertEqual(env1["TINYRL_KV_BYTES"], str(16 * 1024**3))

    daemon.trainers["sampler:fft:Gemma"] = daemon.start("sampler:fft:Gemma")
    env2 = mock_popen.call_args[1]["env"]
    self.assertEqual(env2["GPU"], f"{host}:6:1")
    self.assertEqual(env2["TINYRL_GPU_MEM"], "0.42")
    self.assertEqual(env2["TINYRL_KV_BYTES"], str(16 * 1024**3))

    del daemon.trainers["sampler:fft:Qwen"]
    daemon.trainers["sampler:fft:Mistral"] = daemon.start("sampler:fft:Mistral")
    env3 = mock_popen.call_args[1]["env"]
    self.assertEqual(env3["GPU"], f"{host}:6:0")

    daemon.trainers["sampler:Qwen"] = daemon.start("sampler:Qwen")
    env4 = mock_popen.call_args[1]["env"]
    self.assertEqual(env4["GPU"], f"{host}:4")
    self.assertNotIn("TINYRL_GPU_MEM", env4)
    self.assertEqual(env4["TINYRL_KV_BYTES"], str(24 * 1024**3))

    daemon.trainers["sampler:fft:Llama"] = daemon.start("sampler:fft:Llama", exclusive={"sampler:fft:Llama"})
    env5 = mock_popen.call_args[1]["env"]
    self.assertEqual(env5["GPU"], f"{host}:7")
    self.assertNotIn("TINYRL_GPU_MEM", env5)
    self.assertEqual(env5["TINYRL_KV_BYTES"], str(24 * 1024**3))

    small = NodeDaemon(None, gpus=8, gpu_mib=23034)
    small.trainers["sampler:fft:Qwen"] = small.start("sampler:fft:Qwen")
    env6 = mock_popen.call_args[1]["env"]
    self.assertEqual(env6["GPU"], f"{host}:6")
    self.assertNotIn("TINYRL_GPU_MEM", env6)
    self.assertNotIn("TINYRL_KV_BYTES", env6)


if __name__ == "__main__":
  unittest.main()
