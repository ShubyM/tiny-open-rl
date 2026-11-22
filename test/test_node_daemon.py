import asyncio
import contextlib
import json
import os
import signal
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from test.execution import TestNode
from test.helpers import BusTest, new_id, tiny
from test.test_api import proto_body
from test.test_process import require_unix
from test.test_sampler import JOB
from tinyrl.control.node_daemon import NodeDaemon, WorkerHandle
from tinyrl.storage.store import Store
from tinyrl.types import Op, Role, Run, WorkerSpec

LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


class TestNodeDaemon(BusTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.nodes, self.tasks = [], []

  def node(self, gpus=1, store=None, real=False):
    if store is None:
      folder = tempfile.TemporaryDirectory()
      self.addCleanup(folder.cleanup)
      store = Store(root=folder.name)
    daemon = (NodeDaemon if real else TestNode)(self.bus, gpus=gpus, node="node", store=store, gpu_mib=0)
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
      await node.close()
    await super().asyncTearDown()

  async def until(self, done, node):
    async with asyncio.timeout(60):
      while not done():
        await node.step_once(wait_ms=0)
        await asyncio.sleep(0.01)

  async def reply(self, request, node):
    waiting = self.background(self.bus.wait_for_reply(request, 60))
    await self.until(waiting.done, node)
    reply = waiting.result()
    self.assertIsNotNone(reply)
    self.assertTrue(reply.ok, reply.body)
    return reply

  async def create(self, run, base, node):
    request = self.bus.submit_command(
      run, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 4, "lora": LORA}).encode()
    )
    await self.reply(request, node)

  async def test_a_node_makes_its_checkpoint_folders_before_any_sampler_restores(self):
    with tempfile.TemporaryDirectory() as folder:
      self.node(store=Store(f"file://{folder}/bucket", cache=f"{folder}/cache"))
      self.node(store=Store(f"{folder}/local"))
      self.assertTrue(os.path.isdir(f"{folder}/cache") and os.path.isdir(f"{folder}/local"))

  async def test_a_backlogged_run_yields_to_a_ready_sibling(self):
    node, a, b = self.node(), new_id(), new_id()
    creates = {}
    for run in (a, b):
      self.bus.create_run(Run(run, "tiny", 4, lora=LORA))
      creates[run] = self.bus.submit_command(
        run, Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode()
      )
    for _ in range(12):
      self.bus.submit_command(a, Op.OPTIM_STEP, b'{"adam_params":{}}')
    await node.bind("worker", a, b)
    await self.reply(creates[b], node)
    child = node.processes["worker"]
    self.assertLessEqual((self.bus.get_run(a))["steps"], 4)
    self.assertGreater((self.bus.queued_and_in_flight(a))["queued"], 0)
    await node.drain(a)
    self.assertEqual((self.bus.get_run(a))["steps"], 12)
    self.assertIs(node.processes["worker"], child)

  async def test_a_slow_checkpoint_read_does_not_block_ready_work_or_gpu_handoff(self):
    with tempfile.TemporaryDirectory() as folder:
      writer = Store(f"file://{folder}/bucket", cache=f"{folder}/writer")
      node = self.node(store=Store(writer.root, cache=f"{folder}/reader"))
      started, release, read = threading.Event(), threading.Event(), node.store.local_folder

      def slow_read(*args):  # a big copy still running off-GPU
        started.set()
        release.wait()
        return read(*args)

      node.store.local_folder = slow_read
      uri = "tinker://training/sampler_weights/slow"
      version = writer.put({}, {"base": "tiny", "lora": True, "config": {}})
      self.seed_checkpoint(uri, 0, version=version)
      self.bus.create_run(Run("slow", "tiny", 4, role=Role.SAMPLER, model_path=uri))
      self.bus.create_run(Run("ready", "tiny", 4, role=Role.SAMPLER))
      slow = self.bus.submit_command("slow", Op.SAMPLE, json.dumps(JOB).encode())
      try:
        ready = self.bus.submit_command("ready", Op.SAMPLE, json.dumps(JOB).encode())
        await self.reply(ready, node)
        resident = self.bus.plan_state.procs_of("slow")[0]
        await self.until(started.is_set, node)
        # Another base-model sampler takes the GPU from `resident` while `slow` is still downloading off-GPU.
        self.bus.create_run(Run("other", "other-base", 4, role=Role.SAMPLER))
        request = self.bus.submit_command("other", Op.SAMPLE, json.dumps(JOB).encode())
        await self.reply(request, node)
        contender = self.bus.plan_state.procs_of("other")[0]
        self.assertEqual(node.workers[resident].resident, ())
        self.assertEqual(node.workers[contender].resident, (0,))
        self.assertIsNone(await self.bus.wait_for_reply(slow, 0))
        self.assertEqual(self.bus.queued_and_in_flight("slow"), {"live": True, "queued": 1, "in_flight": []})
      finally:
        await asyncio.to_thread(writer.upload_pending_saves)
        release.set()
      await self.reply(slow, node)

  async def test_a_checkpoint_whose_node_left_before_it_uploaded_fails_fast(self):
    with tempfile.TemporaryDirectory() as folder:
      writer = Store(f"file://{folder}/bucket", cache=f"{folder}/writer")
      writer.peer = "127.0.0.1:9"  # a node that no longer answers
      node = self.node(store=Store(writer.root, cache=f"{folder}/reader"))
      uri = "tinker://training/sampler_weights/lost"
      version = writer.put({}, {"base": "tiny", "lora": True, "config": {}})  # never uploaded
      self.seed_checkpoint(uri, 0, version=version)
      self.bus.create_run(Run("lost", "tiny", 4, role=Role.SAMPLER, model_path=uri))
      request = self.bus.submit_command("lost", Op.SAMPLE, json.dumps(JOB).encode())
      await node.bind("sampler", "lost")
      waiting = self.background(self.bus.wait_for_reply(request, 10))
      await self.until(waiting.done, node)
      self.assertFalse(waiting.result().ok)
      self.assertIn("has not landed, and 127.0.0.1:9 failed to copy", waiting.result().body.decode())

  async def test_trainer_leaves_next_command_unclaimed_during_execution(self):
    node, a, b = self.node(), new_id(), new_id()
    for r in (a, b):
      self.bus.create_run(Run(r, "tiny", 4, lora=LORA))
    await node.bind("worker", a, b)
    for r in (a, b):
      await self.create(r, "tiny", node)
    child = node.processes["worker"]
    entered, release = asyncio.Event(), asyncio.Event()
    call = child.call

    async def hold_step(method, *args):
      if method == "execute" and args[0].run == a and args[0].op == Op.OPTIM_STEP:
        entered.set()
        await release.wait()
      return await call(method, *args)

    child.call = hold_step
    first = self.bus.submit_command(a, Op.OPTIM_STEP, b'{"adam_params":{}}')
    try:
      await self.until(entered.is_set, node)
      second = self.bus.submit_command(a, Op.FORWARD_BACKWARD, proto_body(a, 6))
      other = self.bus.submit_command(b, Op.OPTIM_STEP, b'{"adam_params":{}}')
      await node.tick()
      state = self.bus.queued_and_in_flight(a)
      self.assertEqual((state["queued"], len(state["in_flight"])), (1, 1))
    finally:
      release.set()
    for request in (first, second, other):
      await self.reply(request, node)

  async def test_dead_trainer_cleanup_fails_only_the_command_it_claimed(self):
    node, run = self.node(), new_id()
    self.bus.create_run(Run(run, "tiny", 4, lora=LORA))
    await node.bind("worker", run)
    await self.create(run, "tiny", node)
    child = node.processes["worker"]
    entered, stopped = asyncio.Event(), asyncio.Event()
    call = child.call

    async def die_during_execute(method, *args):
      if method == "execute":
        entered.set()
        await stopped.wait()
        child.kill()
      return await call(method, *args)

    child.call = die_during_execute
    first = self.bus.submit_command(run, Op.OPTIM_STEP, b"{}")
    await self.until(entered.is_set, node)
    next_request = self.bus.submit_command(run, Op.CREATE_MODEL, b'{"base_model":"tiny","rank":4,"lora":{}}')
    stopped.set()
    await self.until(lambda: not node.busy(), node)
    self.assertFalse((await self.bus.wait_for_reply(first, 0)).ok)
    await self.reply(next_request, node)

  async def test_failed_park_keeps_gpu_until_child_closes(self):
    node, a, b = self.node(), new_id(), new_id()
    for run, base in ((a, "a"), (b, "b")):
      self.bus.create_run(Run(run, base, 4, lora=LORA))
    await node.bind("holder", a)
    await self.create(a, "a", node)
    closing, release = asyncio.Event(), asyncio.Event()
    child = node.workers["holder"].model
    close = child.close

    async def fail_park():
      raise RuntimeError("park failed")

    async def slow_close():
      closing.set()
      await release.wait()
      await close()

    child.park, child.close = fail_park, slow_close
    await node.bind("contender", b)
    request = self.bus.submit_command(
      b, Op.CREATE_MODEL, json.dumps({"base_model": "b", "rank": 4, "lora": LORA}).encode()
    )
    try:
      await self.until(closing.is_set, node)
      self.assertEqual(node.workers["holder"].resident, (0,))
      self.assertNotIn("contender", node.workers)
    finally:
      release.set()
    await self.reply(request, node)
    self.assertTrue(child.closed)
    self.assertEqual({k: h.resident for k, h in node.workers.items() if h.resident}, {"contender": (0,)})

  async def test_admission_waits_for_the_previous_model_process_to_park(self):
    require_unix()
    node = self.node(real=True)
    base = f"{node.store.root}/base"
    tiny().save_pretrained(base)
    a, b = new_id(), new_id()
    self.bus.create_run(Run(a, base, 0))
    self.bus.create_run(Run(b, base, 0))
    req_a = self.bus.submit_command(
      a, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 0, "lora": {}}).encode()
    )
    await self.reply(req_a, node)
    pa = self.bus.plan_state.home[a]
    holder = node.workers[pa]
    pid = holder.model.proc.pid
    os.killpg(pid, signal.SIGSTOP)
    try:
      request = self.bus.submit_command(
        b, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 0, "lora": {}}).encode()
      )
      await self.until(lambda: node.workers[pa].op_task is not None, node)
      pb = self.bus.plan_state.home[b]
      self.assertEqual(holder.resident, (0,))
      self.assertNotIn(pb, node.workers)  # Rule R1: never enters GPU 0 until holder finishes parking
      before = self.bus.nodes[node.id]["until"]
      await node.step_once(wait_ms=0)
      self.assertGreater(self.bus.nodes[node.id]["until"], before)
    finally:
      with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGCONT)
    await self.reply(request, node)
    self.assertEqual(holder.resident, ())

  async def test_real_model_process_serves_moves_and_restarts(self):
    require_unix()
    node = self.node(gpus=2, real=True)
    base = f"{node.store.root}/base"
    tiny().save_pretrained(base)
    run = new_id()
    self.bus.create_run(Run(run, base, 4, lora=LORA))
    await self.create(run, base, node)
    proc = self.bus.plan_state.home[run]
    first = node.workers[proc].model.proc.pid
    # When GPU 0 is occupied, the parked 1-GPU trainer hops to GPU 1 without restarting its subprocess.
    handle = node.workers[proc]
    await handle.park()
    other_spec = WorkerSpec(Role.TRAINER, base, False, True, None, 1)
    node.workers["other"] = WorkerHandle("other", other_spec, handle.model, resident=(0,))
    try:
      await handle.ensure_gpu((1,))
      self.assertEqual(handle.model.proc.pid, first)
      self.assertEqual(handle.resident, (1,))
    finally:
      node.workers.pop("other", None)
    os.killpg(first, signal.SIGKILL)
    await self.until(lambda: proc not in node.workers, node)
    await self.create(run, base, node)
    self.assertNotEqual(node.workers[proc].model.proc.pid, first)

  async def test_the_node_uploads_what_its_model_saved(self):
    with tempfile.TemporaryDirectory() as folder:
      store = Store(f"file://{folder}/bucket", cache=f"{folder}/cache")
      version = store.put({"w": torch.ones(2)})
      node = self.node(store=store)
      await self.until(lambda: store.fs.exists(f"{store.base}/objects/{version.partition('@')[0]}/manifest.json"), node)

  async def test_synchronous_clock_cycles_hold_quantum_across_step_gaps_without_parking(self):
    node = self.node(gpus=1)
    a, b = new_id(), new_id()
    self.bus.create_run(Run(a, "tiny-a", 4, account="acct-a", lora=LORA))
    self.bus.create_run(Run(b, "tiny-b", 4, account="acct-b", lora=LORA))
    await self.create(a, "tiny", node)
    await self.create(b, "tiny", node)
    parks = []
    for w in node.processes.values():
      orig = w.trainer.park
      w.trainer.park = lambda p=orig: (parks.append(1), p())[1]
    parks.clear()

    # Waiter `b` queues work while `a` runs 4 synchronous SDK-style steps with a 15ms HTTP round-trip gap between steps.
    req_b = self.bus.submit_command(b, Op.OPTIM_STEP, b'{"adam_params":{}}')
    for _ in range(4):
      fb = self.bus.submit_command(a, Op.FORWARD_BACKWARD, proto_body(a, 6))
      opt = self.bus.submit_command(a, Op.OPTIM_STEP, b'{"adam_params":{}}')
      await self.reply(fb, node)
      await self.reply(opt, node)
      await asyncio.sleep(0.015)
    await self.reply(req_b, node)
    self.assertLessEqual(len(parks), 2)

  async def test_contending_sequential_clients_do_not_swap_the_gpu_on_every_command(self):
    node = self.node(gpus=1)
    a, b = new_id(), new_id()
    self.bus.create_run(Run(a, "tiny-a", 4, account="acct-a", lora=LORA))
    self.bus.create_run(Run(b, "tiny-b", 4, account="acct-b", lora=LORA))
    await self.create(a, "tiny", node)
    await self.create(b, "tiny", node)
    parks = []
    for w in node.processes.values():
      orig = w.trainer.park
      w.trainer.park = lambda p=orig: (parks.append(1), p())[1]

    async def client(run):
      for _ in range(4):
        for op, body in ((Op.FORWARD_BACKWARD, proto_body(run, 6)), (Op.OPTIM_STEP, b'{"adam_params":{}}')):
          reply = await self.bus.wait_for_reply(self.bus.submit_command(run, op, body), 60)
          self.assertTrue(reply is not None and reply.ok)
          await asyncio.sleep(0.015)

    clients = [self.background(client(a)), self.background(client(b))]
    await self.until(lambda: all(c.done() for c in clients), node)
    for c in clients:
      c.result()
    self.assertLessEqual(len(parks), 3)

  async def test_a_swap_waits_for_an_open_training_step_to_finish(self):
    self.enterContext(patch("tinyrl.control.node_daemon.QUANTUM_MS", 40))
    node = self.node(gpus=1)
    a, b = new_id(), new_id()
    self.bus.create_run(Run(a, "tiny-a", 4, account="acct-a", lora=LORA))
    self.bus.create_run(Run(b, "tiny-b", 4, account="acct-b", lora=LORA))
    await self.create(a, "tiny", node)
    await self.create(b, "tiny", node)
    parks = []  # the last command each process ran before it parked

    def watch(w):
      call, last = w.call, []

      async def watched(method, *args):
        if method == "execute":
          last[:] = [args[0].op]
        if method == "park":
          parks.extend(last)
        return await call(method, *args)

      w.call = watched

    for w in node.processes.values():
      watch(w)

    async def client(run):
      for _ in range(8):
        for op, body in ((Op.FORWARD_BACKWARD, proto_body(run, 6)), (Op.OPTIM_STEP, b'{"adam_params":{}}')):
          reply = await self.bus.wait_for_reply(self.bus.submit_command(run, op, body), 60)
          self.assertTrue(reply is not None and reply.ok)
          await asyncio.sleep(0.015)

    clients = [self.background(client(a)), self.background(client(b))]
    await self.until(lambda: all(c.done() for c in clients), node)
    for c in clients:
      c.result()
    self.assertGreater(len(parks), 2)
    self.assertNotIn(Op.FORWARD_BACKWARD, parks)

  async def test_reset_reply_fences_and_closes_workers_with_fresh_boot_id(self):
    node, run = self.node(), new_id()
    self.bus.create_run(Run(run, "tiny", 4, lora=LORA))
    await node.bind("worker", run)
    await self.create(run, "tiny", node)
    old_boot = node.boot_id
    child = node.workers["worker"].model
    self.bus.expire_node(node.id, old_boot)
    ok = await node.step_once(wait_ms=0)
    self.assertFalse(ok)
    self.assertTrue(child.closed)
    self.assertEqual(node.workers, {})
    self.assertNotEqual(node.boot_id, old_boot)

  async def test_snapshot_waits_for_concurrent_park_without_colliding(self):
    node, a, b = self.node(gpus=1), new_id(), new_id()
    for run, base in ((a, "a"), (b, "b")):
      self.bus.create_run(Run(run, base, 4, lora=LORA))
    await node.bind("holder", a)
    await self.create(a, "a", node)
    holder = node.workers["holder"]
    child, call = holder.model, holder.model.call
    in_flight, overlap = [0], []
    parking, release = asyncio.Event(), asyncio.Event()

    async def checked(method, *args):
      in_flight[0] += 1
      if in_flight[0] > 1:
        overlap.append(method)
      try:
        if method == "park":
          parking.set()
          await release.wait()
        return await call(method, *args)
      finally:
        in_flight[0] -= 1

    child.call = checked
    await node.bind("contender", b)
    req = self.bus.submit_command(b, Op.CREATE_MODEL, json.dumps({"base_model": "b", "rank": 4, "lora": LORA}).encode())
    try:
      await self.until(parking.is_set, node)
      snap = self.background(holder.detach_run(a))
      for _ in range(5):
        await asyncio.sleep(0)
      self.assertFalse(snap.done())
      self.assertEqual(overlap, [])
    finally:
      release.set()
    saved = await snap
    self.assertIsNotNone(saved)
    self.assertEqual(overlap, [])
    await self.reply(req, node)


if __name__ == "__main__":
  unittest.main()
