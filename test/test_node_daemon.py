import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from test.helpers import RedisTest
from tinyrl import scheduler
from tinyrl.node_daemon import NodeDaemon, group_alive


class TestNodeDaemon(RedisTest):
  async def placed(self, daemon, worker="w1", **body):
    await self.bus.emit("worker_created", worker=worker, node=daemon.id, gpus=[0], **body)

  async def test_a_process_id_is_started_once_and_replacement_gets_a_new_id(self):
    daemon = NodeDaemon(self.bus, node="n", gpu_mib=0)
    first, second = (MagicMock(pid=2**30, **{"poll.return_value": None}) for _ in range(2))
    await daemon.up()
    await self.placed(daemon)
    with patch("tinyrl.node_daemon.subprocess.Popen", side_effect=[first, second]) as spawn:
      await daemon.tick()
      await daemon.tick()
      self.assertEqual(spawn.call_count, 1)
      first.poll.return_value = 1
      await daemon.tick()
      await daemon.tick()
      self.assertEqual(spawn.call_count, 1)
      self.assertIn("w1", daemon.finished)
      await self.placed(daemon, "w2")
      await daemon.tick()
      self.assertEqual(list(daemon.trainers), ["w2"])
      self.assertIs(daemon.trainers["w2"][1], second)
      facts = [(e.kind, e.body["worker"]) for e in await self.bus.events() if e.kind in ("started", "exited")]
      self.assertEqual(facts, [("started", "w1"), ("exited", "w1"), ("started", "w2")])

  async def test_parent_death_kills_engine_descendants_before_reporting_exit(self):
    daemon = NodeDaemon(self.bus, node="n", gpu_mib=0)
    await daemon.up()
    await self.placed(daemon)
    with tempfile.TemporaryDirectory() as folder:
      pidfile = f"{folder}/child"
      code = (
        "import subprocess, sys, time; from pathlib import Path; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
      )
      parent = subprocess.Popen([sys.executable, "-c", code, pidfile], start_new_session=True)
      try:
        for _ in range(100):
          if Path(pidfile).exists():
            break
          await asyncio.sleep(0.01)
        child = int(Path(pidfile).read_text())
        os.kill(child, 0)
        daemon.trainers["w1"] = [0], parent
        parent.kill()
        await asyncio.to_thread(parent.wait)
        self.assertTrue(group_alive(parent.pid))  # the engine outlives its parent until the agent intervenes
        for _ in range(100):
          await daemon.tick()
          if "w1" in daemon.finished:
            break
          await asyncio.sleep(0.01)
        self.assertIn("w1", daemon.finished)
        self.assertFalse(group_alive(parent.pid))
        self.assertEqual([e.body["worker"] for e in await self.bus.events() if e.kind == "exited"], ["w1"])
      finally:
        daemon.kill(parent)
        await asyncio.to_thread(parent.wait)

  async def test_stopping_is_not_exited_until_the_process_really_exits(self):
    daemon = NodeDaemon(self.bus, node="n", gpu_mib=0)
    proc = MagicMock(pid=2**30, **{"poll.return_value": None})
    await daemon.up()
    await self.placed(daemon)
    with patch("tinyrl.node_daemon.subprocess.Popen", return_value=proc):
      await daemon.tick()
      await self.bus.emit("worker_stopped", worker="w1")
      await daemon.tick()
      proc.terminate.assert_called_once()
      await daemon.tick()
      self.assertNotIn("exited", [e.kind for e in await self.bus.events()])
      proc.poll.return_value = 0
      await daemon.tick()
      exited = [e.body for e in await self.bus.events() if e.kind == "exited"]
      self.assertEqual(exited, [{"worker": "w1", "node": "n"}])

  async def test_a_stuck_child_is_killed_after_its_drain_deadline(self):
    daemon = NodeDaemon(self.bus, node="n", gpu_mib=0)
    proc = MagicMock(pid=2**30, **{"poll.return_value": None})
    daemon.stopping["old"] = proc, 0
    with patch.object(daemon, "kill") as kill:
      await daemon.tick()
      kill.assert_called_once_with(proc)
    self.assertIn("old", daemon.stopping)  # no exit fact until poll confirms it

  async def test_disconnection_stops_children_including_those_already_draining(self):
    daemon = NodeDaemon(self.bus, node="n", gpu_mib=0)
    first, stopping = (MagicMock(pid=2**30, **{"poll.return_value": None}) for _ in range(2))
    daemon.trainers["w1"] = [0], first
    daemon.stopping["w2"] = stopping, 0

    async def disconnected():
      await asyncio.Future()

    with patch.object(daemon, "up", AsyncMock()), patch.object(daemon, "tick", disconnected):
      with patch("tinyrl.node_daemon.LEASE_MS", 10):
        with self.assertRaises(TimeoutError):
          await daemon.serve()
    first.terminate.assert_called_once()
    stopping.terminate.assert_called_once()
    first.wait.assert_called_once()
    stopping.wait.assert_called_once()
    self.assertEqual({e.body["worker"] for e in await self.bus.events() if e.kind == "exited"}, {"w1", "w2"})

  def test_a_restarted_daemon_has_a_fresh_incarnation_even_with_the_same_pid(self):
    self.assertNotEqual(NodeDaemon(None, gpu_mib=0).id, NodeDaemon(None, gpu_mib=0).id)

  @patch("tinyrl.node_daemon.subprocess.Popen")
  def test_start_uses_explicit_role_identity_and_scheduler_resources(self, popen):
    daemon = NodeDaemon(None, gpus=8, node="n", gpu_mib=0)
    daemon.start(
      "opaque-worker", scheduler.Worker(gpus=(6,), role="sampler", lora=True, env={"TINYRL_GPU_MEM": "0.42"})
    )
    env = popen.call_args.kwargs["env"]
    self.assertEqual((env["CUDA_VISIBLE_DEVICES"], env["GPU"]), ("6", f"{socket.gethostname()}:6"))
    self.assertEqual(
      (env["TINYRL_WORKER_ID"], env["TINYRL_ROLE"], env["TINYRL_LORA"]), ("opaque-worker", "sampler", "1")
    )
    self.assertEqual(env["TINYRL_GPU_MEM"], "0.42")
    self.assertEqual(popen.call_args.args[0][-2:], ["tinyrl.sampler", "opaque-worker"])
    self.assertTrue(popen.call_args.kwargs["start_new_session"])

  @patch("tinyrl.node_daemon.subprocess.Popen")
  def test_start_sets_cuda_alloc_conf_on_trainer(self, popen):
    daemon = NodeDaemon(None, gpus=2, node="n", gpu_mib=0)
    with patch.dict(os.environ, {}, clear=True):
      daemon.start("w1", scheduler.Worker(gpus=(0,)))
      self.assertEqual(popen.call_args.kwargs["env"]["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
    with patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}):
      daemon.start("w2", scheduler.Worker(gpus=(0,)))
      self.assertEqual(popen.call_args.kwargs["env"]["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:512")


if __name__ == "__main__":
  unittest.main()
