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
from test.helpers import RedisTest, new_id, tiny
from test.test_api import proto_body
from test.test_process import require_unix
from test.test_sampler import JOB
from tinyrl.bus import GROUP, LEASE_MS, Op, Role, Run, key
from tinyrl.control import scheduler
from tinyrl.control.fairness import Demand
from tinyrl.control.node_daemon import NodeDaemon
from tinyrl.storage.store import Store

LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


class TestNodeDaemon(RedisTest):
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
      await node.close(graceful=False)
      if task := node.sweeping:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    await super().asyncTearDown()

  async def until(self, done, node):
    async with asyncio.timeout(60):
      while not done():
        await node.tick()
        await asyncio.sleep(0.01)

  async def reply(self, request, node):
    waiting = self.background(self.bus.wait_for_reply(request, 60))
    await self.until(waiting.done, node)
    reply = waiting.result()
    self.assertIsNotNone(reply)
    self.assertTrue(reply.ok, reply.body)
    return reply

  async def create(self, run, base, node):
    request = await self.bus.submit_command(
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
      await self.bus.create_run(Run(run, "tiny", 4, lora=LORA))
      creates[run] = await self.bus.submit_command(
        run, Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode()
      )
    for _ in range(12):
      await self.bus.submit_command(a, Op.OPTIM_STEP, b'{"adam_params":{}}')
    await node.bind("worker", a, b)
    child = node.processes["worker"]
    self.assertTrue((await self.bus.wait_for_reply(creates[b], 5)).ok)
    self.assertLessEqual((await self.bus.get_run(a))["steps"], 4)
    self.assertGreater((await self.bus.queued_and_in_flight(a))["queued"], 0)
    await node.drain(a)
    self.assertEqual((await self.bus.get_run(a))["steps"], 12)
    self.assertIs(node.processes["worker"], child)

  async def test_a_slow_checkpoint_read_does_not_block_ready_work_or_gpu_handoff(self):
    with tempfile.TemporaryDirectory() as folder:
      writer = Store(f"file://{folder}/bucket", cache=f"{folder}/writer")
      node = self.node(store=Store(writer.root, cache=f"{folder}/reader"))
      await node.announce_node()
      self.background(scheduler.run(self.bus))
      started, release, read = threading.Event(), threading.Event(), node.store.local_folder

      def slow_read(*args):  # a big copy still running off-GPU
        started.set()
        release.wait()
        return read(*args)

      node.store.local_folder = slow_read
      uri = "tinker://training/sampler_weights/slow"
      version = writer.put({}, {"base": "tiny", "lora": True, "config": {}})
      await self.seed_checkpoint(uri, 0, version=version)
      await self.bus.create_run(Run("slow", "tiny", 4, role=Role.SAMPLER, model_path=uri))
      await self.bus.create_run(Run("ready", "tiny", 4, role=Role.SAMPLER))
      slow = await self.bus.submit_command("slow", Op.SAMPLE, json.dumps(JOB).encode())
      try:
        ready = await self.bus.submit_command("ready", Op.SAMPLE, json.dumps(JOB).encode())
        await self.reply(ready, node)
        resident = node.workers[node.state.assignments["slow"][0]]
        await self.until(started.is_set, node)
        # Another base-model sampler yields the GPU from `resident` while `slow` is still downloading off-GPU.
        await self.bus.create_run(Run("other", "other-base", 4, role=Role.SAMPLER))
        request = await self.bus.submit_command("other", Op.SAMPLE, json.dumps(JOB).encode())
        await self.reply(request, node)
        contender = node.workers[node.state.assignments["other"][0]]
        self.assertFalse(resident.has_gpus)
        self.assertTrue(contender.has_gpus)
        self.assertIsNone(await self.bus.wait_for_reply(slow, 0))
        self.assertEqual(await self.bus.queued_and_in_flight("slow"), {"live": True, "queued": 1, "in_flight": []})
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
      await self.seed_checkpoint(uri, 0, version=version)
      await self.bus.create_run(Run("lost", "tiny", 4, role=Role.SAMPLER, model_path=uri))
      request = await self.bus.submit_command("lost", Op.SAMPLE, json.dumps(JOB).encode())
      [cmd] = await self.bus.claim_samples(["lost"], "dead", 1)
      await self.bus.raw.xclaim(key("q", "lost"), GROUP, "dead", 0, [cmd.entry_id], idle=LEASE_MS)
      await node.bind("sampler", "lost")
      waiting = self.background(self.bus.wait_for_reply(request, 10))
      await self.until(waiting.done, node)
      self.assertFalse(waiting.result().ok)
      self.assertIn("has not landed, and 127.0.0.1:9 failed to copy", waiting.result().body.decode())

  async def test_dead_trainer_cleanup_fails_only_the_command_it_claimed(self):
    node, run = self.node(), new_id()
    await self.bus.create_run(Run(run, "tiny", 4, lora=LORA))
    await node.bind("worker", run)
    await self.create(run, "tiny", node)
    child, worker_task = node.processes["worker"], node.tasks["worker"]
    entered, stopped = asyncio.Event(), asyncio.Event()
    call = child.call

    async def die_during_execute(method, *args):
      if method == "execute":
        entered.set()
        await stopped.wait()
        child.kill()
      return await call(method, *args)

    child.call = die_during_execute
    first = await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}")
    await asyncio.wait_for(entered.wait(), 5)
    next_request = await self.bus.submit_command(run, Op.CREATE_MODEL, b'{"base_model":"tiny","rank":4,"lora":{}}')
    stopped.set()
    try:
      await asyncio.wait_for(worker_task, 5)
    except* EOFError:
      pass  # A task group may wrap the child's disconnect.
    self.assertFalse((await self.bus.wait_for_reply(first, 0)).ok)
    self.assertEqual(await self.bus.queued_and_in_flight(run), {"live": True, "queued": 1, "in_flight": []})
    await self.reply(next_request, node)

  async def test_admission_waits_for_the_previous_model_process_to_park(self):
    require_unix()
    node = self.node(real=True)
    base = f"{node.store.root}/base"
    tiny().save_pretrained(base)
    a, b = new_id(), new_id()
    await self.bus.create_run(Run(a, base, 0))
    await self.bus.create_run(Run(b, base, 0))
    await node.announce_node()
    self.background(scheduler.run(self.bus))
    req_a = await self.bus.submit_command(
      a, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 0, "lora": {}}).encode()
    )
    await self.reply(req_a, node)
    pa = node.state.assignments[a][0]
    holder = node.workers[pa]
    pid = holder.model.proc.pid
    os.killpg(pid, signal.SIGSTOP)
    try:
      request = await self.bus.submit_command(
        b, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 0, "lora": {}}).encode()
      )
      # The holder keeps its turn for CYCLE_GRACE_MS after its command, then the newcomer is planned onto its GPU.
      await self.until(
        lambda: (pb := node.state.assignments.get(b, [""])[0]) in node.workers and node.select_gpus(pb), node
      )
      pb = node.state.assignments[b][0]
      self.assertNotEqual(node.workers[pb].demand, Demand.IDLE)
      self.assertEqual(node.select_gpus(pb), (0,))
      self.assertEqual(node.gpu_allocations.get(holder.worker_id), (0,))
      self.assertFalse(node.workers[pb].has_gpus)  # Rule R1: never enters GPU 0 until holder finishes parking
      self.assertEqual(await self.bus.queued_and_in_flight(b), {"live": True, "queued": 1, "in_flight": []})
      heartbeat = (await self.bus.live_nodes(details=True))[node.id]["until"]
      await node.tick()
      self.assertGreater((await self.bus.live_nodes(details=True))[node.id]["until"], heartbeat)
    finally:
      with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGCONT)
    await self.reply(request, node)
    self.assertFalse(holder.has_gpus)

  async def test_real_model_process_serves_moves_and_restarts(self):
    require_unix()
    node = self.node(gpus=2, real=True)
    base = f"{node.store.root}/base"
    tiny().save_pretrained(base)
    run, other, blocker = new_id(), new_id(), new_id()
    await self.bus.create_run(Run(run, base, 4, lora=LORA))
    await self.bus.create_run(Run(other, base, 0))
    await self.bus.create_run(Run(blocker, base, 0, exclusive=True))
    await node.announce_node()
    self.background(scheduler.run(self.bus))
    await self.create(run, base, node)
    proc = node.state.assignments[run][0]
    first = node.workers[proc].model.proc.pid
    # Fill GPU 1, preempt GPU 0 with an exclusive run, then free GPU 1: `proc` hops to GPU 1 without restarting.
    create_fft = json.dumps({"base_model": base, "rank": 0, "lora": {}}).encode()
    await self.reply(await self.bus.submit_command(other, Op.CREATE_MODEL, create_fft), node)
    await self.reply(await self.bus.submit_command(blocker, Op.CREATE_MODEL, create_fft), node)
    await self.reply(await self.bus.submit_command(other, Op.DELETE_MODEL, b"{}"), node)
    await self.reply(await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}'), node)
    self.assertEqual(node.workers[proc].model.proc.pid, first)
    self.assertEqual(node.gpu_allocations[node.workers[proc].worker_id], (1,))
    await self.reply(await self.bus.submit_command(blocker, Op.DELETE_MODEL, b"{}"), node)
    os.killpg(first, signal.SIGKILL)
    await self.until(lambda: proc in node.workers and node.workers[proc].model.proc.pid != first, node)
    await self.create(run, base, node)

  async def test_the_node_uploads_what_its_model_saved(self):
    with tempfile.TemporaryDirectory() as folder:
      store = Store(f"file://{folder}/bucket", cache=f"{folder}/cache")
      version = store.put({"w": torch.ones(2)})
      node = self.node(store=store)
      await self.until(lambda: store.fs.exists(f"{store.base}/objects/{version.partition('@')[0]}/manifest.json"), node)

  async def test_synchronous_clock_cycles_hold_quantum_across_step_gaps_without_parking(self):
    node = self.node(gpus=1)
    await node.announce_node()
    self.background(scheduler.run(self.bus))
    a, b = new_id(), new_id()
    await self.bus.create_run(Run(a, "tiny-a", 4, account="acct-a", lora=LORA))
    await self.bus.create_run(Run(b, "tiny-b", 4, account="acct-b", lora=LORA))
    await self.create(a, "tiny", node)
    await self.create(b, "tiny", node)
    parks = []
    for w in node.processes.values():
      orig = w.trainer.park
      w.trainer.park = lambda p=orig: (parks.append(1), p())[1]
    parks.clear()

    # Waiter `b` queues work while `a` runs 4 synchronous SDK-style steps with a 15ms HTTP round-trip gap between steps.
    req_b = await self.bus.submit_command(b, Op.OPTIM_STEP, b'{"adam_params":{}}')
    for _ in range(4):
      fb = await self.bus.submit_command(a, Op.FORWARD_BACKWARD, proto_body(a, 6))
      opt = await self.bus.submit_command(a, Op.OPTIM_STEP, b'{"adam_params":{}}')
      await self.reply(fb, node)
      await self.reply(opt, node)
      await asyncio.sleep(0.015)
    await self.reply(req_b, node)
    self.assertLessEqual(len(parks), 2)  # `a` holds its quantum across all 4 step gaps instead of parking every step!

  async def test_contending_sequential_clients_do_not_swap_the_gpu_on_every_command(self):
    node = self.node(gpus=1)
    await node.announce_node()
    self.background(scheduler.run(self.bus))
    a, b = new_id(), new_id()
    await self.bus.create_run(Run(a, "tiny-a", 4, account="acct-a", lora=LORA))
    await self.bus.create_run(Run(b, "tiny-b", 4, account="acct-b", lora=LORA))
    await self.create(a, "tiny", node)
    await self.create(b, "tiny", node)
    parks = []
    for w in node.processes.values():
      orig = w.trainer.park
      w.trainer.park = lambda p=orig: (parks.append(1), p())[1]

    async def client(run):
      for _ in range(4):
        for op, body in ((Op.FORWARD_BACKWARD, proto_body(run, 6)), (Op.OPTIM_STEP, b'{"adam_params":{}}')):
          reply = await self.bus.wait_for_reply(await self.bus.submit_command(run, op, body), 60)
          self.assertTrue(reply is not None and reply.ok)
          await asyncio.sleep(0.015)  # the client's round trip, during which its trainer has nothing ready

    clients = [self.background(client(a)), self.background(client(b))]
    await self.until(lambda: all(c.done() for c in clients), node)
    for c in clients:
      c.result()
    self.assertLessEqual(len(parks), 3)  # a trainer that gave up the GPU in every gap would park 16 times

  async def test_a_swap_waits_for_an_open_training_step_to_finish(self):
    self.enterContext(patch("tinyrl.control.node_daemon.QUANTUM_MS", 40))  # a turn often ends between two commands
    node = self.node(gpus=1)
    await node.announce_node()
    self.background(scheduler.run(self.bus))
    a, b = new_id(), new_id()
    await self.bus.create_run(Run(a, "tiny-a", 4, account="acct-a", lora=LORA))
    await self.bus.create_run(Run(b, "tiny-b", 4, account="acct-b", lora=LORA))
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
          reply = await self.bus.wait_for_reply(await self.bus.submit_command(run, op, body), 60)
          self.assertTrue(reply is not None and reply.ok)
          await asyncio.sleep(0.015)

    clients = [self.background(client(a)), self.background(client(b))]
    await self.until(lambda: all(c.done() for c in clients), node)
    for c in clients:
      c.result()
    self.assertGreater(len(parks), 2)
    self.assertNotIn(Op.FORWARD_BACKWARD, parks)  # parking with gradients open moves them too


if __name__ == "__main__":
  unittest.main()
