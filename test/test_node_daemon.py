import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch

import torch

from test.execution import InlineProcess, TestNode
from test.helpers import RedisTest, new_id, tiny
from test.test_process import require_unix
from tinyrl.bus import Ev, Op, Role, Run
from tinyrl.control import scheduler
from tinyrl.control.node_daemon import SWEEP_GRACE_S, NodeDaemon
from tinyrl.storage.store import Store
from tinyrl.workers.model import Model
from tinyrl.workers.trainer import Trainer

LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


class TestNodeDaemon(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.nodes, self.tasks = [], []

  def node(self, gpus=1, node="n", store=None, real=False, gpu_mib=0):
    if store is None:
      folder = tempfile.TemporaryDirectory()
      self.addCleanup(folder.cleanup)
      store = Store(root=folder.name)
    daemon = (
      NodeDaemon(self.bus, gpus=gpus, node=node, store=store, gpu_mib=gpu_mib)
      if real
      else TestNode(self.bus, gpus=gpus, node=node, store=store)
    )
    self.nodes.append(daemon)
    return daemon

  def background(self, coro):
    task = asyncio.create_task(coro)
    self.tasks.append(task)
    return task

  async def asyncTearDown(self):
    for task in self.tasks:
      task.cancel()
    await asyncio.gather(*self.tasks, return_exceptions=True)
    for node in self.nodes:
      tasks = []
      for running in node.running.values():
        running.stopping = running.lost = True
        if running.task:
          running.task.cancel()
          tasks.append(running.task)
      await asyncio.gather(*tasks, return_exceptions=True)
      for worker, running in list(node.running.items()):
        if running.task is None:
          await node.close_worker(worker)
      node.running.clear()
      for task in (node.uploading, node.sweeping):
        if task:
          task.cancel()
          await asyncio.gather(task, return_exceptions=True)
    await super().asyncTearDown()

  async def until(self, done, *nodes, tries=500):
    for _ in range(tries):
      for node in nodes:
        await node.tick()
      if done():
        return
      await asyncio.sleep(0.02)
    self.fail("node never reached the expected state")

  async def create(self, run, base="tiny", rank=4):
    request = await self.bus.submit_command(
      run, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": rank, "lora": LORA}).encode()
    )
    reply = await self.bus.wait_for_reply(request, 60)
    self.assertIsNotNone(reply)
    self.assertTrue(reply.ok, reply.body)
    return request

  async def test_a_node_starts_only_nonempty_workers_and_restarts_a_dead_child(self):
    run, empty = new_id(), new_id()
    await self.bus.create_run(Run(run, "worker", "tiny", 4, lora=LORA))
    await self.bus.create_run(Run(empty, "empty", "tiny", 4))
    await self.bus.delete_run(empty, "empty")
    node = self.node(gpus=2)
    await node.announce_node()
    self.background(scheduler.run(self.bus))
    await self.until(lambda: bool(node.running), node)
    self.assertEqual(set(node.running), {"worker"})
    await self.create(run)
    first = node.running["worker"].model
    await node.tick()
    self.assertIs(node.running["worker"].model, first)
    first.kill()
    await self.until(lambda: "worker" in node.running and node.running["worker"].model is not first, node)
    self.assertTrue(first.closed)
    await self.create(run)

  @patch("tinyrl.bus.NODE_MS", 600)
  @patch("tinyrl.bus.LEASE_MS", 400)
  @patch("tinyrl.control.node_daemon.LEASE_MS", 400)
  async def test_workers_spread_over_nodes_and_move_off_a_dead_one(self):
    for run in ("r1", "r2"):
      await self.bus.create_run(Run(run, f"worker-{run}", "tiny", 4))
    a, b = self.node(node="a"), self.node(node="b")
    await a.announce_node()
    await b.announce_node()
    self.background(scheduler.run(self.bus))
    await self.until(lambda: len(a.running) == len(b.running) == 1, a, b)
    await self.until(lambda: set(b.running) == {"worker-r1", "worker-r2"}, b)
    child = b.running["worker-r1"].model
    await self.bus.delete_run("r1", "worker-r1")
    await self.until(lambda: set(b.running) == {"worker-r2"}, b)
    self.assertTrue(child.closed)

  async def test_a_lora_move_parks_and_resumes_the_same_model_process(self):
    run = new_id()
    await self.bus.create_run(Run(run, "worker", "tiny", 4, lora=LORA))
    node = self.node(gpus=2)
    await self.bus.submit_command(
      run, Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode()
    )
    await node.drain(run)
    running = node.running["worker"]
    first = running.model
    weights = first.trainer.model.params[run]
    node.note(Ev.PLACED, worker="worker", node=node.id, gpus=[1], rule="test")
    with patch.object(first, "call", wraps=first.call) as calls:
      self.assertTrue(await asyncio.wait_for(node.acquire("worker"), 5))
    self.assertIs(node.running["worker"].model, first)
    self.assertIs(first.trainer.model.params[run], weights)
    self.assertEqual(running.gpus, (1,))
    self.assertEqual(node.state.workers["worker"].on, ((node.id, 1),))
    self.assertEqual([c.args for c in calls.call_args_list], [("park",), ("resume", 1)])

  async def test_a_gpu_is_not_granted_until_park_acknowledges(self):
    node = self.node()
    for run, worker in (("ra", "a"), ("rb", "b")):
      await self.bus.create_run(Run(run, worker, "tiny", 4))
      await node.add(worker, gpus=[0])
    await node.acquire("a")
    node.note(Ev.IDLE, worker="a")
    entered, finish = asyncio.Event(), asyncio.Event()
    model = node.running["a"].model
    call = model.call

    async def slow_park(method, *args):
      if method == "park":
        entered.set()
        await finish.wait()
      return await call(method, *args)

    try:
      with patch.object(model, "call", side_effect=slow_park):
        waiting = self.background(node.acquire("b"))
        await self.until(lambda: node.state.workers["a"].revoked)
        releasing = self.background(node.release("a"))
        await asyncio.wait_for(entered.wait(), 5)
        self.assertFalse(waiting.done())
        self.assertEqual(scheduler.holders(node.state, (node.id, 0)), ["a"])
        finish.set()
        await releasing
        self.assertTrue(await asyncio.wait_for(waiting, 5))
        self.assertEqual(scheduler.holders(node.state, (node.id, 0)), ["b"])
    finally:
      finish.set()

  async def test_a_closing_process_keeps_its_gpu_until_exit(self):
    node = self.node()
    for run, worker in (("ra", "a"), ("rb", "b")):
      await self.bus.create_run(Run(run, worker, "tiny", 4))
      await node.add(worker, gpus=[0])
    await node.acquire("a")
    node.running["a"].stopping = node.state.workers["a"].closing = True
    node.note(Ev.UNPLACED, worker="a")
    entered, finish = asyncio.Event(), asyncio.Event()
    child = node.running["a"].model
    close = child.close

    async def slow_close():
      entered.set()
      await finish.wait()
      await close()

    try:
      with patch.object(child, "close", side_effect=slow_close):
        closing = self.background(node.close_worker("a"))
        await asyncio.wait_for(entered.wait(), 5)
        waiting = self.background(node.acquire("b"))
        await asyncio.sleep(0.05)
        self.assertFalse(waiting.done())
        self.assertEqual(scheduler.holders(node.state, (node.id, 0)), ["a"])
        finish.set()
        await closing
        self.assertTrue(await asyncio.wait_for(waiting, 5))
    finally:
      finish.set()

  async def test_cancelling_node_cleanup_keeps_ownership_until_the_child_is_closed(self):
    node = self.node()
    await self.bus.create_run(Run("run", "worker", "tiny", 4))
    running = await node.add("worker", gpus=[0])
    await node.acquire("worker")
    entered, finish = asyncio.Event(), asyncio.Event()
    close = running.model.close

    async def slow_close():
      entered.set()
      await finish.wait()
      await close()

    try:
      with patch.object(running.model, "close", side_effect=slow_close):
        closing = self.background(node.close_worker("worker"))
        await asyncio.wait_for(entered.wait(), 5)
        closing.cancel()
        await asyncio.sleep(0.02)
        self.assertFalse(closing.done())
        self.assertFalse(running.closed)
        self.assertEqual(await self.bus.lease_owner("worker"), running.owner)
        self.assertEqual(scheduler.holders(node.state, (node.id, 0)), ["worker"])
        finish.set()
        await closing
      self.assertTrue(running.closed)
      self.assertEqual(scheduler.holders(node.state, (node.id, 0)), [])
      self.assertIsNone(await self.bus.lease_owner("worker"))
    finally:
      finish.set()

  async def test_failed_cleanup_keeps_the_gpu_and_lease_for_a_retry(self):
    node = self.node()
    await self.bus.create_run(Run("run", "worker", "tiny", 4))
    running = await node.add("worker", gpus=[0])
    await node.acquire("worker")
    with patch.object(running.model, "close", side_effect=RuntimeError("could not reap")):
      with self.assertRaisesRegex(RuntimeError, "could not reap"):
        await node.close_worker("worker")
    self.assertFalse(running.closed)
    self.assertEqual(scheduler.holders(node.state, (node.id, 0)), ["worker"])
    self.assertEqual(await self.bus.lease_owner("worker"), running.owner)
    await node.close_worker("worker")
    self.assertTrue(running.closed)
    self.assertEqual(scheduler.holders(node.state, (node.id, 0)), [])
    self.assertIsNone(await self.bus.lease_owner("worker"))

  async def test_restarting_the_cluster_scheduler_does_not_restart_execution(self):
    run = new_id()
    await self.bus.create_run(Run(run, "worker", "tiny", 4, lora=LORA))
    node = self.node()
    await node.announce_node()
    first_scheduler = self.background(scheduler.run(self.bus))
    await self.until(lambda: "worker" in node.running, node)
    await self.create(run)
    first = node.running["worker"]
    first_scheduler.cancel()
    await asyncio.gather(first_scheduler, return_exceptions=True)
    self.background(scheduler.run(self.bus))
    for _ in range(5):
      await node.tick()
      await asyncio.sleep(0.03)
    self.assertIs(node.running["worker"], first)
    self.assertEqual(await self.bus.lease_owner("worker"), first.owner)
    request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
    self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
    local = {Ev.STARTED, Ev.WANTS_GPU, Ev.GRANTED, Ev.REVOKED, Ev.BATCH_DONE, Ev.IDLE, Ev.RELEASED, Ev.EXITED}
    self.assertFalse(local.intersection(e.kind for e in await self.bus.read_events()))

  async def test_launch_settings_are_per_worker_and_multi_gpu_uses_torchrun(self):
    node = self.node(gpus=8, real=True, gpu_mib=81559)
    node.note(Ev.NODE_UP, node=node.id, gpus=8, memory=node.gpu_mib * 2**20)
    kinds = {
      "sample": scheduler.Kind(Role.SAMPLER, base="tiny", lora=False),
      "lora": scheduler.Kind(base="tiny", lora=True),
      "fft": scheduler.Kind(base="tiny", lora=False),
      "multi": scheduler.Kind(base="tiny", lora=False, gpus=2),
    }
    for name, kind in kinds.items():
      node.state.workers[name] = scheduler.Worker(kind, node=node.id, gpus=(2, 3) if name == "multi" else (6,))
    with (
      patch.dict(os.environ, {}, clear=True),
      patch(
        "tinyrl.control.node_daemon.Process", side_effect=lambda *_: InlineProcess(Trainer(Model(tiny), node.store))
      ) as process,
    ):
      sample = await node.open_worker("sample")
      self.assertEqual(sample.sampler.env["CUDA_VISIBLE_DEVICES"], "6")
      self.assertEqual(sample.sampler.env["TINYRL_GPU_MEM"], "0.42")
      self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", sample.sampler.env)
      await node.open_worker("lora")
      cmd, env = process.call_args.args
      self.assertEqual(cmd, [sys.executable, "-m", "tinyrl.workers.trainer"])
      self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
      self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
      with patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}):
        await node.open_worker("fft")
      self.assertEqual(process.call_args.args[1]["CUDA_VISIBLE_DEVICES"], "6")
      self.assertEqual(process.call_args.args[1]["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:512")
      self.assertNotIn("TINYRL_NODE", process.call_args.args[1])
      await node.open_worker("multi")
      cmd, env = process.call_args.args
      self.assertEqual(
        cmd,
        [
          sys.executable,
          "-m",
          "torch.distributed.run",
          "--standalone",
          "--nproc-per-node=2",
          "-m",
          "tinyrl.workers.trainer",
        ],
      )
      self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,3")
    self.assertNotIn("TINYRL_GPU_MEM", os.environ)

  async def test_a_node_does_not_start_a_child_before_it_owns_the_worker(self):
    node = self.node(real=True)
    node.state.workers["worker"] = scheduler.Worker(scheduler.Kind(base="tiny"), node=node.id, gpus=(0,))
    self.assertTrue(await self.bus.take_lease("worker", "another-node"))
    with patch("tinyrl.control.node_daemon.Process") as process:
      self.assertIsNone(await node.open_worker("worker"))
    process.assert_not_called()
    self.assertNotIn("worker", node.running)

  async def test_a_lost_node_reannounces_only_after_its_model_has_closed(self):
    await self.bus.create_run(Run("run", "worker", "tiny", 4))
    node = self.node()
    await node.announce_node()
    self.background(scheduler.run(self.bus))
    await self.until(lambda: "worker" in node.running, node)
    child = node.running["worker"].model
    close, entered, finish = child.close, asyncio.Event(), asyncio.Event()

    async def slow_close():
      entered.set()
      await finish.wait()
      await close()

    try:
      with patch.object(child, "close", side_effect=slow_close):
        await self.bus.append_event(Ev.NODE_LOST, node=node.id)
        await node.tick()
        await asyncio.wait_for(entered.wait(), 5)
        for _ in range(3):
          await node.tick()
        self.assertEqual([e.kind for e in await self.bus.read_events()].count(Ev.NODE_UP), 1)
        self.assertNotIn(node.id, node.state.nodes)
        finish.set()
        await self.until(lambda: node.id in node.state.nodes, node)
        self.assertTrue(child.closed)
        self.assertEqual([e.kind for e in await self.bus.read_events()].count(Ev.NODE_UP), 2)
    finally:
      finish.set()

  async def test_a_slow_sweep_never_holds_up_the_heartbeat(self):
    node, sweeps, finish = self.node(), [], asyncio.Event()

    async def slow_sweep():
      sweeps.append(time.time())
      await finish.wait()

    node.sweep_checkpoints = slow_sweep
    try:
      with patch("tinyrl.control.node_daemon.SWEEP_S", 0):
        await asyncio.wait_for(node.tick(), 5)
        await asyncio.wait_for(node.tick(), 5)
      self.assertEqual(len(sweeps), 1)
      self.assertIn(node.id, await self.bus.live_nodes())
    finally:
      finish.set()

  async def test_real_model_process_serves_moves_and_restarts(self):
    require_unix()
    with tempfile.TemporaryDirectory() as folder:
      base = f"{folder}/base"
      tiny().save_pretrained(base)
      run = new_id()
      await self.bus.create_run(Run(run, "worker", base, 4, lora=LORA))
      node = self.node(gpus=2, real=True)
      await node.announce_node()
      self.background(scheduler.run(self.bus))
      await self.until(lambda: "worker" in node.running, node)
      await self.create(run, base)
      first = node.running["worker"].model.proc
      node.note(Ev.PLACED, worker="worker", node=node.id, gpus=[1], rule="test")
      request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
      reply = await self.bus.wait_for_reply(request, 60)
      self.assertTrue(reply.ok, reply.body)
      self.assertIs(node.running["worker"].model.proc, first)
      self.assertEqual(node.running["worker"].gpus, (1,))
      first.kill()
      await asyncio.to_thread(first.wait)
      await self.until(lambda: "worker" in node.running and node.running["worker"].model.proc is not first, node)
      await self.create(run, base)

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
    daemon = self.node(store=store)
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

  async def test_a_sweep_moves_on_past_a_checkpoint_it_cannot_delete(self):
    store = Store(f"memory://tinyrl-test/{uuid.uuid4()}", cache=tempfile.mkdtemp())
    for name in ("a", "b"):
      store.put(f"tinker://r/weights/{name}", {"w": torch.ones(2)})
    store.upload_pending_saves()
    await self.bus.r.set("tinyrl:ckpt_since", 0)
    delete = store.delete
    store.delete = lambda uri: (_ for _ in ()).throw(OSError("503")) if uri.endswith("/a") else delete(uri)
    with patch("tinyrl.control.node_daemon.SWEEP_GRACE_S", -60):
      await NodeDaemon(self.bus, node="n", gpu_mib=0, store=store).sweep_checkpoints()
    self.assertEqual(store.list_checkpoints(), ["tinker://r/weights/a"])

  async def test_a_sweep_deletes_a_save_whose_upload_never_finished(self):
    store = Store(f"memory://tinyrl-test/{uuid.uuid4()}", cache=tempfile.mkdtemp())
    store.put("tinker://r/weights/w", {"w": torch.ones(2)})
    pipe = store.fs.pipe_file
    with patch.object(
      store.fs, "pipe_file", lambda p, d: (_ for _ in ()).throw(OSError("429")) if p.endswith("_latest") else pipe(p, d)
    ):
      store.upload_pending_saves()  # its files went up, but not the _latest that names them
    self.assertTrue(store.fs.find(store.path("tinker://r/weights/w")))
    await self.bus.r.set("tinyrl:ckpt_since", 0)
    with patch("tinyrl.control.node_daemon.SWEEP_GRACE_S", -60):
      await NodeDaemon(self.bus, node="n", gpu_mib=0, store=store).sweep_checkpoints()
    self.assertFalse(store.fs.exists(store.path("tinker://r/weights/w")))


if __name__ == "__main__":
  unittest.main()
