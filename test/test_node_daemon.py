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

from test.execution import InlineProcess, TestNode, gate
from test.helpers import RedisTest, new_id, tiny
from test.test_process import require_unix
from tinyrl.bus import Ev, Op, Role, Run
from tinyrl.control import scheduler
from tinyrl.control.node_daemon import SWEEP_GRACE_S, Lease, NodeDaemon, TrainerEngine
from tinyrl.storage.store import Store
from tinyrl.workers.model import Model
from tinyrl.workers.sampler import Sampler
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
      for lw in node.local.values():
        lw.cancel(lost=True)
      await asyncio.gather(*(lw.task for lw in node.local.values()), return_exceptions=True)
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

  async def served(self, node, run, worker, gpus):
    await self.bus.create_run(Run(run, worker, "tiny", 4))
    await node.add(worker, gpus=gpus)
    await self.create(run)
    return node.local[worker]

  async def test_a_node_starts_only_nonempty_workers_and_restarts_a_dead_child(self):
    run, empty = new_id(), new_id()
    await self.bus.create_run(Run(run, "worker", "tiny", 4, lora=LORA))
    await self.bus.create_run(Run(empty, "empty", "tiny", 4))
    await self.bus.delete_run(empty, "empty")
    node = self.node(gpus=2)
    await node.announce_node()
    self.background(scheduler.run(self.bus))
    await self.until(lambda: bool(node.local), node)
    self.assertEqual(set(node.local), {"worker"})
    await self.create(run)
    first = node.local["worker"].engine.process
    await node.tick()
    self.assertIs(node.local["worker"].engine.process, first)
    first.kill()
    await self.until(lambda: "worker" in node.local and node.local["worker"].engine.process is not first, node)
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
    await self.until(lambda: len(a.local) == len(b.local) == 1, a, b)
    await self.until(lambda: set(b.local) == {"worker-r1", "worker-r2"}, b)
    child = b.local["worker-r1"].engine.process
    await self.bus.delete_run("r1", "worker-r1")
    await self.until(lambda: set(b.local) == {"worker-r2"}, b)
    self.assertTrue(child.closed)

  async def test_a_lora_move_parks_and_resumes_the_same_model_process(self):
    run = new_id()
    await self.bus.create_run(Run(run, "worker", "tiny", 4, lora=LORA))
    node = self.node(gpus=2)
    await self.bus.submit_command(
      run, Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode()
    )
    await node.drain(run)
    first = node.local["worker"].engine.process
    weights = first.trainer.model.params[run]
    node.note(Ev.PLACED, worker="worker", node=node.id, gpus=[1], rule="test")
    with patch.object(first, "call", wraps=first.call) as calls:
      request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
      self.assertTrue((await self.bus.wait_for_reply(request, 10)).ok)
    self.assertIs(node.local["worker"].engine.process, first)
    self.assertIs(first.trainer.model.params[run], weights)
    self.assertEqual(node.state.workers["worker"].on, ((node.id, 1),))
    self.assertEqual([c.args[0] for c in calls.call_args_list], ["park", "resume", "execute"])
    self.assertEqual(calls.call_args_list[1].args, ("resume", 1))

  async def test_a_gpu_is_not_granted_until_park_acknowledges(self):
    node = self.node()
    a = await self.served(node, "ra", "a", [0])
    entered, finish = gate(a.engine.process, "park")
    waiting = self.background(self.served(node, "rb", "b", [0]))
    await asyncio.wait_for(entered.wait(), 5)
    self.assertFalse(waiting.done())
    self.assertEqual(scheduler.holders(node.state, (node.id, 0)), ["a"])
    finish.set()
    await asyncio.wait_for(waiting, 10)
    self.assertEqual(scheduler.holders(node.state, (node.id, 0)), ["b"])

  async def test_a_closing_process_keeps_its_gpu_until_exit(self):
    node = self.node()
    a = await self.served(node, "ra", "a", [0])
    entered, finish = asyncio.Event(), asyncio.Event()
    close = a.engine.process.close

    async def slow_close():
      entered.set()
      await finish.wait()
      await close()

    try:
      with patch.object(a.engine.process, "close", side_effect=slow_close):
        node.note(Ev.UNPLACED, worker="a")
        await asyncio.wait_for(entered.wait(), 5)
        self.assertEqual(node.state.workers["a"].on, ((node.id, 0),))
        self.assertEqual(await self.bus.lease_owner("a"), a.lease.owner)
        finish.set()
        await asyncio.wait_for(a.task, 5)
      self.assertEqual(node.state.workers["a"].on, ())
      self.assertIsNone(await self.bus.lease_owner("a"))
    finally:
      finish.set()

  async def test_cancelling_node_cleanup_keeps_ownership_until_the_child_is_closed(self):
    node = self.node()
    lw = await self.served(node, "run", "worker", [0])
    entered, finish = asyncio.Event(), asyncio.Event()
    close = lw.engine.process.close

    async def slow_close():
      entered.set()
      await finish.wait()
      await close()

    try:
      with patch.object(lw.engine.process, "close", side_effect=slow_close):
        node.note(Ev.UNPLACED, worker="worker")
        await asyncio.wait_for(entered.wait(), 5)
        lw.cancel()
        await asyncio.sleep(0.02)
        self.assertFalse(lw.task.done())
        self.assertEqual(await self.bus.lease_owner("worker"), lw.lease.owner)
        self.assertEqual(node.state.workers["worker"].on, ((node.id, 0),))
        finish.set()
        await asyncio.gather(lw.task, return_exceptions=True)
      self.assertTrue(lw.engine.process.closed)
      self.assertEqual(node.state.workers["worker"].on, ())
      self.assertIsNone(await self.bus.lease_owner("worker"))
    finally:
      finish.set()

  async def test_failed_cleanup_keeps_the_gpu_and_lease_for_a_retry(self):
    node = self.node()
    lw = await self.served(node, "run", "worker", [0])
    close, failures = lw.engine.process.close, []

    async def failing_once():
      if not failures:
        failures.append(RuntimeError("could not reap"))
        raise failures[0]
      await close()

    with patch.object(lw.engine.process, "close", side_effect=failing_once):
      node.note(Ev.UNPLACED, worker="worker")
      await self.until(lambda: bool(failures))
      self.assertFalse(lw.task.done())
      self.assertEqual(node.state.workers["worker"].on, ((node.id, 0),))
      self.assertEqual(await self.bus.lease_owner("worker"), lw.lease.owner)
      await asyncio.wait_for(lw.task, 5)
    self.assertTrue(lw.engine.process.closed)
    self.assertEqual(node.state.workers["worker"].on, ())
    self.assertIsNone(await self.bus.lease_owner("worker"))

  async def test_restarting_the_cluster_scheduler_does_not_restart_execution(self):
    run = new_id()
    await self.bus.create_run(Run(run, "worker", "tiny", 4, lora=LORA))
    node = self.node()
    await node.announce_node()
    first_scheduler = self.background(scheduler.run(self.bus))
    await self.until(lambda: "worker" in node.local, node)
    await self.create(run)
    first = node.local["worker"]
    first_scheduler.cancel()
    await asyncio.gather(first_scheduler, return_exceptions=True)
    self.background(scheduler.run(self.bus))
    for _ in range(5):
      await node.tick()
      await asyncio.sleep(0.03)
    self.assertIs(node.local["worker"], first)
    self.assertEqual(await self.bus.lease_owner("worker"), first.lease.owner)
    request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
    self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
    local = {Ev.STARTED, Ev.WANTS_GPU, Ev.GRANTED, Ev.REVOKED, Ev.BATCH_DONE, Ev.IDLE, Ev.RELEASED, Ev.EXITED}
    self.assertFalse(local.intersection(e.kind for e in await self.bus.read_events()))

  async def test_a_moved_multi_gpu_trainer_parks_its_runs_on_its_own_gpus(self):
    node = self.node(gpus=2)
    await self.bus.create_run(Run("rw", "wide", "tiny", 4, lora=LORA, gpus=2))
    await node.add("wide", gpus=[0, 1])
    await self.create("rw")
    x = await self.served(node, "rx", "x", [0])
    wide = node.local["wide"]
    self.assertIsNone(node.state.workers["wide"].since)
    entered, finish = gate(wide.engine.process, "park_all")
    calls = []
    try:
      with patch.object(x.engine.process, "call", wraps=x.engine.process.call) as x_calls:
        node.note(Ev.PLACED, worker="wide", node="elsewhere")
        await asyncio.wait_for(entered.wait(), 10)
        for gpu in (0, 1):
          self.assertEqual(scheduler.holders(node.state, (node.id, gpu)), ["wide"])
        self.assertTrue(wide.engine.awake)
        calls = [c.args[0] for c in x_calls.call_args_list]
    finally:
      finish.set()
    self.assertIn("park", calls)
    await asyncio.wait_for(wide.task, 10)
    self.assertIsNotNone(await self.bus.parked_checkpoint("rw"))
    self.assertIsNone(await self.bus.lease_owner("wide"))

  async def wide_granted_while_asleep_in_acquire(self):
    node = self.node(gpus=2)
    await self.bus.create_run(Run("rw", "wide", "tiny", 4, lora=LORA, gpus=2))
    await node.add("wide", gpus=[0, 1])
    await self.create("rw")
    x = await self.served(node, "rx", "x", [0])
    wide = node.local["wide"]
    x_entered, x_go = gate(x.engine.process, "execute")
    await self.bus.submit_command("rx", Op.OPTIM_STEP, b'{"adam_params":{}}')
    await asyncio.wait_for(x_entered.wait(), 5)
    await self.bus.submit_command("rw", Op.OPTIM_STEP, b'{"adam_params":{}}')
    await self.until(lambda: node.state.workers["wide"].asked)
    x_go.set()
    await self.until(lambda: node.state.workers["wide"].since is not None)
    self.assertFalse(wide.engine.awake)
    return node, wide

  async def leaves_resumed_after_a_grant_landed_in_its_sleep(self, leave):
    node, wide = await self.wide_granted_while_asleep_in_acquire()
    entered, finish = gate(wide.engine.process, "park_all")
    leave(node)
    try:
      await asyncio.wait_for(entered.wait(), 10)
      self.assertTrue(wide.engine.awake, "park_all ran with the FSDP trainer still parked on host")
      self.assertFalse(wide.engine.process.trainer.model.parked)
    finally:
      finish.set()
    await asyncio.wait_for(wide.task, 10)
    self.assertIsNotNone(await self.bus.parked_checkpoint("rw"))

  @patch("tinyrl.control.node_daemon.TICK", 1.5)
  async def test_a_multi_gpu_trainer_granted_while_it_slept_resumes_before_parking_on_shutdown(self):
    await self.leaves_resumed_after_a_grant_landed_in_its_sleep(lambda node: setattr(node, "stopping", True))

  @patch("tinyrl.control.node_daemon.TICK", 1.5)
  async def test_a_multi_gpu_trainer_granted_while_it_slept_resumes_before_parking_on_a_move(self):
    await self.leaves_resumed_after_a_grant_landed_in_its_sleep(
      lambda node: node.note(Ev.PLACED, worker="wide", node="elsewhere")
    )

  def test_both_engines_have_the_same_members(self):
    engine = ["batch", "top_up", "warm", "proc", "started", "start", "resume", "park", "park_runs", "close", "execute"]
    lease = Lease(self.bus, "w", "o")
    trainer = TrainerEngine(InlineProcess(Trainer(Model(tiny), Store(root=tempfile.mkdtemp()))), lease, self.registry)
    sampler = Sampler("base", True, {})

    def shape(e):
      return [(name, callable(getattr(e, name))) for name in engine]

    self.assertEqual(shape(trainer), shape(sampler))

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
      sample = await node.open("sample")
      self.assertEqual(sample.engine.env["CUDA_VISIBLE_DEVICES"], "6")
      self.assertEqual(sample.engine.env["TINYRL_GPU_MEM"], "0.42")
      self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", sample.engine.env)
      await node.open("lora")
      cmd, env = process.call_args.args
      self.assertEqual(cmd, [sys.executable, "-m", "tinyrl.workers.trainer"])
      self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
      self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
      with patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}):
        await node.open("fft")
      self.assertEqual(process.call_args.args[1]["CUDA_VISIBLE_DEVICES"], "6")
      self.assertEqual(process.call_args.args[1]["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:512")
      self.assertNotIn("TINYRL_NODE", process.call_args.args[1])
      multi = await node.open("multi")
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
      await asyncio.wait_for(multi.task, 5)
    self.assertNotIn("TINYRL_GPU_MEM", os.environ)

  async def test_a_node_does_not_start_a_child_before_it_owns_the_worker(self):
    node = self.node(real=True)
    node.state.workers["worker"] = scheduler.Worker(scheduler.Kind(base="tiny"), node=node.id, gpus=(0,))
    self.assertTrue(await self.bus.take_lease("worker", "another-node"))
    with patch("tinyrl.control.node_daemon.Process") as process:
      self.assertIsNone(await node.open("worker"))
    process.assert_not_called()
    self.assertNotIn("worker", node.local)

  async def test_a_lost_node_reannounces_only_after_its_model_has_closed(self):
    await self.bus.create_run(Run("run", "worker", "tiny", 4))
    node = self.node()
    await node.announce_node()
    self.background(scheduler.run(self.bus))
    await self.until(lambda: "worker" in node.local, node)
    child = node.local["worker"].engine.process
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
      await self.until(lambda: "worker" in node.local, node)
      await self.create(run, base)
      first = node.local["worker"].engine.proc
      node.note(Ev.PLACED, worker="worker", node=node.id, gpus=[1], rule="test")
      request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
      reply = await self.bus.wait_for_reply(request, 60)
      self.assertTrue(reply.ok, reply.body)
      self.assertIs(node.local["worker"].engine.proc, first)
      self.assertEqual(node.state.workers["worker"].on, ((node.id, 1),))
      first.kill()
      await asyncio.to_thread(first.wait)
      await self.until(lambda: "worker" in node.local and node.local["worker"].engine.proc is not first, node)
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
