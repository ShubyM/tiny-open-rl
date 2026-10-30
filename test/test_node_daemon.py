import asyncio
import json
import os
import socket
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from test.helpers import RedisTest, new_id, tiny
from tinyrl import scheduler
from tinyrl.bus import Op, Run
from tinyrl.node_daemon import NodeDaemon

LEASE_MS = 2000
LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


class TestNodeDaemon(RedisTest):
  async def test_the_node_daemon_starts_a_trainer_for_a_worker_with_runs_and_again_when_it_dies(self):
    base = f"{tempfile.mkdtemp()}/tiny"
    tiny().save_pretrained(base)
    worker, run = f"lora:{base}", new_id()
    await self.bus.create(Run(run, worker, base, 4))
    await self.bus.create(Run(new_id(), "empty", base, 4))
    await self.bus.drop((await self.bus.runs("empty"))[0], "empty")

    with patch.dict(os.environ, {"REDIS_URL": self.url, "LEASE_MS": str(LEASE_MS)}):
      daemon = NodeDaemon(self.bus, gpus=2)
      await daemon.up()
      scheduling = asyncio.create_task(scheduler.run(self.bus))  # it places the worker and grants GPU turns
      try:
        await self.until(lambda: daemon.procs, daemon)
        self.assertEqual(list(daemon.procs), [worker])  # not the worker whose runs are all gone
        request_id = await self.bus.submit(
          run, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 4, "lora": LORA}).encode()
        )
        self.assertTrue((await self.bus.result(request_id, 60)).ok)

        gpu, first = daemon.procs[worker]
        await daemon.tick()
        self.assertIs(daemon.procs[worker][1], first)  # one trainer per worker

        first.kill()
        await asyncio.to_thread(first.wait)
        while await self.bus.owner(worker):  # its lease runs out
          await asyncio.sleep(0.1)
        await daemon.tick()
        self.assertIsNot(daemon.procs[worker][1], first)
        request_id = await self.bus.submit(
          run, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 4, "lora": LORA}).encode()
        )
        self.assertTrue((await self.bus.result(request_id, 60)).ok)
      finally:
        scheduling.cancel()
        for _, proc in daemon.procs.values():
          proc.terminate()
          proc.wait()

  async def until(self, done, *daemons: NodeDaemon, tries: int = 100) -> None:
    """Tick the node daemons until `done` says yes."""
    for _ in range(tries):
      for daemon in daemons:
        await daemon.tick()
      if done():
        return
      await asyncio.sleep(0.05)
    self.fail("never happened")

  @patch("tinyrl.bus.NODE_MS", 600)
  async def test_workers_spread_over_nodes_and_move_off_a_dead_one(self):
    for run in ("r1", "r2"):
      await self.bus.create(Run(run, f"worker-{run}", "base", 4))
    a, b = NodeDaemon(self.bus, gpus=1, node="a"), NodeDaemon(self.bus, gpus=1, node="b")
    await a.up()
    await b.up()
    scheduling = asyncio.create_task(scheduler.run(self.bus))
    try:
      with patch("subprocess.Popen", side_effect=lambda *_, **__: MagicMock(**{"poll.return_value": None})):
        await self.until(lambda: len(a.procs) == len(b.procs) == 1, a, b)  # one worker on each node

        await self.until(lambda: sorted(b.procs) == ["worker-r1", "worker-r2"], b)  # a stops ticking, so it dies

        proc = b.procs["worker-r1"][1]
        await self.bus.drop("r1", "worker-r1")
        await self.until(lambda: sorted(b.procs) == ["worker-r2"], b)  # an empty worker's process is stopped
        proc.terminate.assert_called()
    finally:
      scheduling.cancel()

  async def test_a_stopped_process_keeps_its_turn_until_it_exits(self):
    await self.bus.create(Run("ra", "a", "base", 4))
    daemon, proc = NodeDaemon(self.bus, gpus=1, node="n", gpu_mib=0), MagicMock(**{"poll.return_value": None})
    await daemon.up()
    scheduling = asyncio.create_task(scheduler.run(self.bus))
    try:
      with patch("subprocess.Popen", return_value=proc):
        await self.until(lambda: daemon.procs, daemon)
        await self.bus.drop("ra", "a")
        await self.until(lambda: proc.terminate.called, daemon)
        for _ in range(5):
          await daemon.tick()
        self.assertNotIn("exited", [e.kind for e in await self.bus.events()])  # it may still be saving on its GPU
        proc.poll.return_value = 0
        await daemon.tick()
        self.assertIn("exited", [e.kind for e in await self.bus.events()])
    finally:
      scheduling.cancel()

  @patch("tinyrl.node_daemon.subprocess.Popen")
  def test_a_process_starts_on_its_placed_gpus_with_its_placed_settings(self, popen):
    daemon = NodeDaemon(None, gpus=8, node="n")
    daemon.gpu_mib = 81559
    daemon.start("sampler:fft:a", scheduler.Worker(scheduler.Kind("sampler", lora=False), gpus=(6,)))
    env = popen.call_args[1]["env"]
    self.assertEqual((env["CUDA_VISIBLE_DEVICES"], env["GPU"]), ("6", f"{socket.gethostname()}:6"))
    self.assertEqual(env["TINYRL_GPU_MEM"], "0.42")
    self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", env)

  @patch("tinyrl.node_daemon.subprocess.Popen")
  def test_start_sets_cuda_alloc_conf_on_trainer(self, popen):
    daemon = NodeDaemon(None, gpus=2, node="n")
    daemon.start("lora:base:0", scheduler.Worker(gpus=(0,)))
    self.assertEqual(popen.call_args[1]["env"]["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
    with patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}):
      daemon.start("lora:base:1", scheduler.Worker(gpus=(0,)))
      self.assertEqual(popen.call_args[1]["env"]["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:512")


if __name__ == "__main__":
  unittest.main()
