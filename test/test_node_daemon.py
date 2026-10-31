import asyncio
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from unittest.mock import MagicMock, patch

import torch

from test.helpers import RedisTest, new_id, tiny
from tinyrl.bus import Op, Run
from tinyrl.control import scheduler
from tinyrl.control.node_daemon import SWEEP_GRACE_S, NodeDaemon, reap
from tinyrl.storage.store import Store

LEASE_MS = 2000
LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


class TestNodeDaemon(RedisTest):
  async def test_the_node_daemon_starts_a_trainer_for_a_worker_with_runs_and_again_when_it_dies(self):
    base = f"{tempfile.mkdtemp()}/tiny"
    tiny().save_pretrained(base)
    worker, run = f"lora:{base}", new_id()
    await self.bus.create_run(Run(run, worker, base, 4))
    await self.bus.create_run(Run(new_id(), "empty", base, 4))
    await self.bus.delete_run((await self.bus.runs_in_worker("empty"))[0], "empty")

    with patch.dict(os.environ, {"REDIS_URL": self.url, "LEASE_MS": str(LEASE_MS)}):
      daemon = NodeDaemon(self.bus, gpus=2)
      await daemon.announce_node()
      scheduling = asyncio.create_task(scheduler.run(self.bus))  # it places the worker and grants GPU turns
      try:
        await self.until(lambda: daemon.procs, daemon)
        self.assertEqual(list(daemon.procs), [worker])  # not the worker whose runs are all gone
        request_id = await self.bus.submit_command(
          run, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 4, "lora": LORA}).encode()
        )
        self.assertTrue((await self.bus.wait_for_reply(request_id, 60)).ok)

        gpu, first = daemon.procs[worker]
        await daemon.tick()
        self.assertIs(daemon.procs[worker][1], first)  # one trainer per worker

        first.kill()
        await asyncio.to_thread(first.wait)
        while await self.bus.lease_owner(worker):  # its lease runs out
          await asyncio.sleep(0.1)
        await daemon.tick()
        self.assertIsNot(daemon.procs[worker][1], first)
        request_id = await self.bus.submit_command(
          run, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 4, "lora": LORA}).encode()
        )
        self.assertTrue((await self.bus.wait_for_reply(request_id, 60)).ok)
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
      await self.bus.create_run(Run(run, f"worker-{run}", "base", 4))
    a, b = NodeDaemon(self.bus, gpus=1, node="a"), NodeDaemon(self.bus, gpus=1, node="b")
    await a.announce_node()
    await b.announce_node()
    scheduling = asyncio.create_task(scheduler.run(self.bus))
    try:
      with patch("subprocess.Popen", side_effect=lambda *_, **__: MagicMock(**{"poll.return_value": None})):
        await self.until(lambda: len(a.procs) == len(b.procs) == 1, a, b)  # one worker on each node

        await self.until(lambda: sorted(b.procs) == ["worker-r1", "worker-r2"], b)  # a stops ticking, so it dies

        proc = b.procs["worker-r1"][1]
        await self.bus.delete_run("r1", "worker-r1")
        await self.until(lambda: sorted(b.procs) == ["worker-r2"], b)  # an empty worker's process is stopped
        proc.terminate.assert_called()
    finally:
      scheduling.cancel()

  async def test_a_stopped_process_keeps_its_turn_until_it_exits(self):
    await self.bus.create_run(Run("ra", "a", "base", 4))
    daemon, proc = NodeDaemon(self.bus, gpus=1, node="n", gpu_mib=0), MagicMock(**{"poll.return_value": None})
    await daemon.announce_node()
    scheduling = asyncio.create_task(scheduler.run(self.bus))
    try:
      with patch("subprocess.Popen", return_value=proc):
        await self.until(lambda: daemon.procs, daemon)
        await self.bus.delete_run("ra", "a")
        await self.until(lambda: proc.terminate.called, daemon)
        for _ in range(5):
          await daemon.tick()
        self.assertNotIn("exited", [e.kind for e in await self.bus.read_events()])  # it may still be saving on its GPU
        proc.poll.return_value = 0
        await daemon.tick()
        self.assertIn("exited", [e.kind for e in await self.bus.read_events()])
    finally:
      scheduling.cancel()

  def test_reaping_a_process_kills_what_it_started(self):
    spawn = "import subprocess, sys; print(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']).pid)"
    proc = subprocess.Popen([sys.executable, "-c", spawn], stdout=subprocess.PIPE, text=True, start_new_session=True)
    child = int(proc.stdout.readline())
    proc.wait()
    reap(proc)
    for _ in range(100):
      try:
        if pathlib.Path(f"/proc/{child}/stat").read_text().rsplit(") ", 1)[1][0] == "Z":
          break
      except FileNotFoundError:
        break
      time.sleep(0.01)
    else:
      self.fail("the process a worker started outlived it")

  @patch("tinyrl.control.node_daemon.subprocess.Popen")
  def test_a_process_starts_on_its_placed_gpus_with_its_placed_settings(self, popen):
    daemon = NodeDaemon(None, gpus=8, node="n")
    daemon.gpu_mib = 81559
    daemon.start_worker_process("sampler:fft:a", scheduler.Worker(scheduler.Kind("sampler", lora=False), gpus=(6,)))
    env = popen.call_args[1]["env"]
    self.assertEqual((env["CUDA_VISIBLE_DEVICES"], env["GPU"]), ("6", f"{socket.gethostname()}:6"))
    self.assertEqual(env["TINYRL_GPU_MEM"], "0.42")
    self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", env)

  @patch("tinyrl.control.node_daemon.subprocess.Popen")
  def test_start_sets_cuda_alloc_conf_on_trainer(self, popen):
    daemon = NodeDaemon(None, gpus=2, node="n")
    daemon.start_worker_process("lora:base:0", scheduler.Worker(gpus=(0,)))
    self.assertEqual(popen.call_args[1]["env"]["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
    with patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}):
      daemon.start_worker_process("lora:base:1", scheduler.Worker(gpus=(0,)))
      self.assertEqual(popen.call_args[1]["env"]["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:512")

  async def test_started_is_in_the_log_before_the_process_spawns(self):
    await self.bus.create_run(Run("r1", "worker", "base", 4))
    daemon = NodeDaemon(self.bus, gpus=1, node="n", gpu_mib=0)
    await daemon.announce_node()
    await self.bus.append_event("placed", worker="worker", node="n", gpus=[0], rule="test")
    import redis

    seen = []

    def spawn(*_, **__):
      with redis.Redis.from_url(self.url) as r:
        seen.extend(f[b"kind"].decode() for _, entries in r.xread({"tinyrl:events": "0-0"}) for _, f in entries)
      return MagicMock(**{"poll.return_value": None})

    with patch("subprocess.Popen", side_effect=spawn):
      await daemon.tick()
    self.assertIn("started", seen)

  async def test_a_slow_sweep_never_holds_up_the_heartbeat(self):
    daemon, sweeps, done = NodeDaemon(self.bus, node="n", gpu_mib=0), [], asyncio.Event()

    async def slow_sweep():
      sweeps.append(time.time())
      await done.wait()  # deleting a wave's checkpoints from a bucket took 85 s on the cluster

    daemon.sweep_checkpoints = slow_sweep
    with patch("tinyrl.control.node_daemon.SWEEP_S", 0):
      await asyncio.wait_for(daemon.tick(), 5)
      await asyncio.wait_for(daemon.tick(), 5)  # and heartbeats again while the sweep runs
    self.assertEqual(len(sweeps), 1)  # one sweep at a time
    done.set()

  async def test_a_node_the_scheduler_found_lost_announces_itself_again(self):
    daemon = NodeDaemon(self.bus, gpus=1, node="n", gpu_mib=0)
    await daemon.announce_node()
    await daemon.tick()
    await self.bus.append_event("node_lost", node="n")  # its heartbeat ran out while it was still running
    await daemon.tick()
    await daemon.tick()
    self.assertEqual([e.kind for e in await self.bus.read_events()], ["node_up", "node_lost", "node_up"])
    self.assertIn("n", daemon.state.nodes)

  async def test_a_sweep_deletes_only_what_the_registry_lets_go(self):
    store = Store(root=tempfile.mkdtemp())
    old = time.time() - SWEEP_GRACE_S - 60
    for name in ("before", "kept", "deleted", "young"):
      store.put(f"tinker://r/weights/{name}", {"w": torch.ones(2)})
    os.utime(store.path("tinker://r/weights/before"), (old - 3600, old - 3600), follow_symlinks=False)
    await self.registry.register_checkpoint("tinker://r/weights/kept", 8)  # the registry starts here
    await self.registry.register_checkpoint("tinker://r/weights/deleted", 8)
    await self.registry.unregister_checkpoints("tinker://r/weights/deleted")
    await self.bus.r.set("tinyrl:ckpt_since", old - 1800)
    for name in ("kept", "deleted"):
      os.utime(store.path(f"tinker://r/weights/{name}"), (old, old), follow_symlinks=False)
    await NodeDaemon(self.bus, node="n", gpu_mib=0, store=store).sweep_checkpoints()
    names = sorted(u.rsplit("/", 1)[1] for u in store.list_checkpoints())
    self.assertEqual(names, ["before", "kept", "young"])  # older than the registry, registered, still being saved

  async def test_the_node_daemon_uploads_what_its_node_saved(self):
    store = Store(f"memory://tinyrl-test/{uuid.uuid4()}", cache=tempfile.mkdtemp())
    store.put("tinker://r/weights/w", {"w": torch.ones(2)})  # a trainer exits right after, leaving it to the daemon
    daemon = NodeDaemon(self.bus, node="n", gpu_mib=0, store=store)
    await daemon.tick()
    await daemon.uploading
    self.assertEqual(store.list_checkpoints(), ["tinker://r/weights/w"])

  async def test_a_sweep_keeps_the_chain_links_a_registered_checkpoint_builds_on(self):
    store, v = Store(f"memory://tinyrl-test/{uuid.uuid4()}", cache=tempfile.mkdtemp()), []
    for step in range(3):
      store.put(
        f"tinker://r/sampler_weights/{step}", {"w": torch.full((64,), float(step))}, {"config": {"model_type": "x"}}
      )
      v.append(store.versions[f"tinker://r/sampler_weights/{step}"])
    store.upload_pending_saves()
    await self.registry.register_checkpoint("tinker://r/sampler_weights/1", 128, version=v[1])
    await self.bus.r.set("tinyrl:ckpt_since", 0)
    with patch("tinyrl.control.node_daemon.SWEEP_GRACE_S", -60):
      await NodeDaemon(self.bus, node="n", gpu_mib=0, store=store).sweep_checkpoints()
    self.assertEqual(sorted(p.rsplit("/", 1)[1] for p in store.fs.glob(f"{store.base}/r/_versions/*")), v[:2])


if __name__ == "__main__":
  unittest.main()
