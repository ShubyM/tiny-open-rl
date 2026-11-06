import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import uuid
from unittest.mock import patch

import torch

from test.execution import InlineProcess, TestNode
from test.helpers import RedisTest, new_id, tiny
from test.test_process import require_unix
from test.test_sampler import JOB
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
      else TestNode(self.bus, gpus=gpus, node=node, store=store, gpu_mib=gpu_mib)
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
      await node.close(graceful=False)
      for task in (node.uploading, node.sweeping):
        if task:
          task.cancel()
          await asyncio.gather(task, return_exceptions=True)
    await super().asyncTearDown()

  async def until(self, done, *nodes, timeout=10):
    async with asyncio.timeout(timeout):
      while not done():
        for node in nodes:
          await node.tick()
        await asyncio.sleep(0.01)

  async def reply(self, request, *nodes):
    waiting = self.background(self.bus.wait_for_reply(request, 60))
    await self.until(waiting.done, *(nodes or self.nodes), timeout=60)
    reply = waiting.result()
    self.assertIsNotNone(reply)
    self.assertTrue(reply.ok, reply.body)
    return reply

  async def create(self, run, base="tiny", rank=4):
    request = await self.bus.submit_command(
      run, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": rank, "lora": LORA}).encode()
    )
    await self.reply(request)
    return request

  async def prepared(self, node, worker="worker", gpus=(0,)):
    run = new_id()
    await self.bus.create_run(Run(run, worker, "tiny", 4, lora=LORA, gpus=len(gpus)))
    running = await node.add(worker, gpus=gpus)
    await self.create(run)
    return run, running

  async def test_a_backlogged_run_yields_to_a_sibling_within_the_command_budget(self):
    node = self.node()
    a, _ = await self.prepared(node)
    b = new_id()
    await self.bus.create_run(Run(b, "worker", "tiny", 4, lora=LORA))
    await self.create(b)
    child, executed = node.processes["worker"], []
    entered, finish = asyncio.Event(), asyncio.Event()
    original = child.call

    async def execute(method, *args):
      if method == "execute" and args[0].op == Op.OPTIM_STEP:
        executed.append(args[0].run)
        if len(executed) == 1:
          entered.set()
          await finish.wait()
      return await original(method, *args)

    try:
      with (
        patch.object(child, "call", side_effect=execute),
        patch("tinyrl.control.node_daemon.RUN_COMMANDS", 4),
        patch("tinyrl.control.node_daemon.RUN_QUANTUM", 60.0),
        patch("tinyrl.control.node_daemon.SAVE_WAIT_S", 0),
      ):
        queued = [await self.bus.submit_command(a, Op.OPTIM_STEP, b'{"adam_params":{}}') for _ in range(12)]
        await self.until(entered.is_set, node)
        other = await self.bus.submit_command(b, Op.OPTIM_STEP, b'{"adam_params":{}}')
        queue = await self.bus.queued_and_in_flight(a)
        self.assertEqual((queue["queued"], len(queue["in_flight"])), (11, 1))
        finish.set()
        await self.reply(other, node)
        self.assertEqual(executed[0], a)
        self.assertLessEqual(executed.index(b), 4)
        await node.drain(a)
        self.assertTrue(all([(await self.bus.wait_for_reply(request, 0)).ok for request in queued]))
      self.assertIs(node.processes["worker"], child)
    finally:
      finish.set()

  async def test_commands_are_not_claimed_while_waiting_for_gpu_admission(self):
    node = self.node()
    run, holder = await self.prepared(node, "holder")
    await self.bus.create_run(Run("waiting", "waiting", "tiny", 4, lora=LORA))
    waiter = await node.add("waiting", gpus=[0])
    entered, finish = asyncio.Event(), asyncio.Event()
    child = node.processes["holder"]
    original = child.call

    async def execute(method, *args):
      if method == "execute":
        entered.set()
        await finish.wait()
      return await original(method, *args)

    try:
      with patch.object(child, "call", side_effect=execute):
        await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
        await self.until(entered.is_set, node)
        first = await self.bus.submit_command(
          "waiting", Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode()
        )
        later = await self.bus.submit_command("waiting", Op.OPTIM_STEP, b'{"adam_params":{}}')
        await self.until(lambda: waiter.ticket > 0, node)
        self.assertIsNotNone(holder.since)
        self.assertIsNone(waiter.since)
        self.assertEqual(await self.bus.queued_and_in_flight("waiting"), {"live": True, "queued": 2, "in_flight": []})
        finish.set()
        await self.reply(first, node)
        await self.reply(later, node)
    finally:
      finish.set()

  async def test_a_run_does_not_bank_service_credit_while_its_preparation_is_blocked(self):
    node = self.node()
    a, _ = await self.prepared(node)
    b = new_id()
    await self.bus.create_run(Run(b, "worker", "tiny", 4, lora=LORA))
    await self.create(b)
    child, executed = node.processes["worker"], []
    preparing, finish = asyncio.Event(), asyncio.Event()
    original, get_run = child.call, self.bus.get_run

    async def prepare(run):
      if run == b and not finish.is_set():
        preparing.set()
        await finish.wait()
      return await get_run(run)

    async def execute(method, *args):
      if method == "execute" and args[0].op == Op.OPTIM_STEP:
        executed.append(args[0].run)
        if executed.count(a) == 20:
          finish.set()
        await asyncio.sleep(0.004)
      return await original(method, *args)

    def a_resumed():
      return b in executed and a in executed[executed.index(b) + 1 :]

    try:
      with (
        patch.object(self.bus, "get_run", side_effect=prepare),
        patch.object(child, "call", side_effect=execute),
        patch("tinyrl.control.node_daemon.RUN_COMMANDS", 4),
        patch("tinyrl.control.node_daemon.RUN_QUANTUM", 60.0),
        patch("tinyrl.control.node_daemon.SAVE_WAIT_S", 0),
      ):
        for _ in range(50):
          await self.bus.submit_command(b, Op.OPTIM_STEP, b'{"adam_params":{}}')
        await self.until(preparing.is_set, node)
        for _ in range(50):
          await self.bus.submit_command(a, Op.OPTIM_STEP, b'{"adam_params":{}}')
        await self.until(a_resumed, node)
        first = executed.index(b)
        resumed = executed.index(a, first)
        self.assertGreaterEqual(first, 20)
        self.assertLessEqual(resumed - first, 4, "blocked preparation accumulated service credit")
        await node.drain(a)
    finally:
      finish.set()

  async def test_a_ready_run_executes_while_its_siblings_checkpoint_is_loading(self):
    node = self.node()
    uri = "tinker://training/sampler_weights/slow"
    node.store.put(uri, {}, {"base": "tiny", "lora": True, "config": {}})
    await self.bus.create_run(Run("slow", "sampler", "tiny", 4, role=Role.SAMPLER, model_path=uri))
    await self.bus.create_run(Run("ready", "sampler", "tiny", 4, role=Role.SAMPLER))
    await node.add("sampler")
    preparing, finish = asyncio.Event(), threading.Event()
    local, loop = node.store.local_folder, asyncio.get_running_loop()

    def blocked(uri, version):
      loop.call_soon_threadsafe(preparing.set)
      if not finish.wait(10):
        raise TimeoutError("checkpoint preparation was never released")
      return local(uri, version)

    try:
      with patch.object(node.store, "local_folder", side_effect=blocked):
        slow = await self.bus.submit_command("slow", Op.SAMPLE, json.dumps(JOB).encode())
        await self.until(preparing.is_set, node)
        ready = await self.bus.submit_command("ready", Op.SAMPLE, json.dumps(JOB).encode())
        reply = await self.bus.wait_for_reply(ready, 3)
        self.assertIsNotNone(reply, "one run's checkpoint download blocked its ready sibling")
        self.assertTrue(reply.ok, reply.body)
        self.assertFalse(finish.is_set())
        self.assertIsNone(await self.bus.wait_for_reply(slow, 0))
        self.assertEqual(await self.bus.queued_and_in_flight("slow"), {"live": True, "queued": 1, "in_flight": []})
        finish.set()
        await self.reply(slow, node)
    finally:
      finish.set()

  async def test_checkpoint_preparation_does_not_prevent_a_resident_worker_from_yielding(self):
    node = self.node()
    uri = "tinker://training/sampler_weights/slow"
    node.store.put(uri, {}, {"base": "tiny", "lora": True, "config": {}})
    await self.bus.create_run(Run("base", "sampler", "tiny", 4, role=Role.SAMPLER))
    await self.bus.create_run(Run("slow", "sampler", "tiny", 4, role=Role.SAMPLER, model_path=uri))
    resident = await node.add("sampler")
    initial = await self.bus.submit_command("base", Op.SAMPLE, json.dumps(JOB).encode())
    await self.reply(initial, node)
    self.assertIsNotNone(resident.since)
    preparing, finish = asyncio.Event(), threading.Event()
    local, loop = node.store.local_folder, asyncio.get_running_loop()

    def blocked(uri, version):
      loop.call_soon_threadsafe(preparing.set)
      if not finish.wait(10):
        raise TimeoutError("checkpoint preparation was never released")
      return local(uri, version)

    try:
      with patch.object(node.store, "local_folder", side_effect=blocked):
        slow = await self.bus.submit_command("slow", Op.SAMPLE, json.dumps(JOB).encode())
        await self.until(preparing.is_set, node)
        await self.bus.create_run(Run("other", "other", "tiny", 4, role=Role.SAMPLER))
        contender = await node.add("other", gpus=[0])
        request = await self.bus.submit_command("other", Op.SAMPLE, json.dumps(JOB).encode())
        reply = await self.bus.wait_for_reply(request, 3)
        self.assertIsNotNone(reply, "checkpoint preparation prevented the resident worker from yielding")
        self.assertTrue(reply.ok, reply.body)
        self.assertFalse(finish.is_set())
        self.assertIsNone(resident.since)
        self.assertIsNotNone(contender.since)
        self.assertFalse(node.tasks["sampler"].done())
        self.assertEqual(await self.bus.queued_and_in_flight("slow"), {"live": True, "queued": 1, "in_flight": []})
        finish.set()
        await self.reply(slow, node)
    finally:
      finish.set()

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
    first = node.processes["worker"]
    await node.tick()
    self.assertIs(node.processes["worker"], first)
    first.kill()
    await self.until(lambda: "worker" in node.running and node.processes["worker"] is not first, node)
    self.assertTrue(first.closed)
    await self.create(run)

  async def test_a_child_that_dies_before_claiming_answers_each_waiting_run(self):
    node = self.node()
    a, b = new_id(), new_id()
    waiting = []
    for run in (a, b):
      await self.bus.create_run(Run(run, "worker", "tiny", 4, lora=LORA))
      waiting.append(
        await self.bus.submit_command(
          run, Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode()
        )
      )
    later = await self.bus.submit_command(a, Op.OPTIM_STEP, b'{"adam_params":{}}')
    factory = node.open_trainer

    def failed_start(worker, env):
      child = factory(worker, env)
      child.kill()
      return child

    with patch.object(node, "open_trainer", side_effect=failed_start):
      await node.add("worker")
      await asyncio.wait_for(asyncio.gather(node.tasks["worker"], return_exceptions=True), 5)
    for request in waiting:
      reply = await self.bus.wait_for_reply(request, 0)
      self.assertIsNotNone(reply, "a child that died before claiming left its client waiting")
      self.assertFalse(reply.ok)
      self.assertEqual(json.loads(reply.body)["category"], "server")
    self.assertTrue(node.processes["worker"].closed)
    self.assertIsNone(await self.bus.lease_owner("worker"))
    self.assertIsNone(await self.bus.wait_for_reply(later, 0))
    self.assertEqual(await self.bus.queued_and_in_flight(a), {"live": True, "queued": 1, "in_flight": []})

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
    child = b.processes["worker-r1"]
    await self.bus.delete_run("r1", "worker-r1")
    await self.until(lambda: set(b.running) == {"worker-r2"}, b)
    self.assertTrue(child.closed)

  async def test_a_lora_move_parks_and_resumes_the_same_model_process(self):
    node = self.node(gpus=2)
    run, running = await self.prepared(node)
    first = node.processes["worker"]
    weights = first.trainer.model.params[run]
    node.state.workers["worker"].gpus = (1,)
    with patch.object(first, "call", wraps=first.call) as calls:
      request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
      await self.reply(request, node)
    self.assertIs(node.processes["worker"], first)
    self.assertIs(first.trainer.model.params[run], weights)
    self.assertEqual(running.gpus, (1,))
    self.assertIsNotNone(running.since)
    self.assertEqual([c.args for c in calls.call_args_list[:2]], [("park",), ("resume", 1)])

  async def test_a_blocked_park_keeps_its_gpu_but_other_gpus_and_heartbeats_progress(self):
    node = self.node(gpus=2)
    _, a = await self.prepared(node, "a", (0,))
    for run, worker, gpu in (("rc", "c", 1), ("rb", "b", 0)):
      await self.bus.create_run(Run(run, worker, "tiny", 4, lora=LORA))
      await node.add(worker, gpus=[gpu])
    entered, finish = asyncio.Event(), asyncio.Event()
    child = node.processes["a"]
    original = child.call

    async def slow_park(method, *args):
      if method == "park":
        entered.set()
        await finish.wait()
      return await original(method, *args)

    try:
      with patch.object(child, "call", side_effect=slow_park):
        waiting = await self.bus.submit_command(
          "rb", Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode()
        )
        await self.until(entered.is_set, node)
        heartbeat = (await self.bus.live_nodes(details=True))[node.id]["until"]
        other = await self.bus.submit_command(
          "rc", Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode()
        )
        await self.reply(other, node)
        self.assertGreater((await self.bus.live_nodes(details=True))[node.id]["until"], heartbeat)
        self.assertIsNotNone(a.since)
        self.assertIsNone(node.running["b"].since)
        self.assertIsNone(await self.bus.wait_for_reply(waiting, 0))
        self.assertFalse(finish.is_set())
        finish.set()
        await self.reply(waiting, node)
        self.assertIsNone(a.since)
    finally:
      finish.set()

  async def test_unplacement_cancels_preparation_and_leaves_unclaimed_work_queued(self):
    node = self.node()
    uri = "tinker://training/sampler_weights/ready"
    node.store.put(uri, {}, {"base": "tiny", "lora": True, "config": {}})
    await self.bus.create_run(Run("base", "sampler", "tiny", 4, role=Role.SAMPLER))
    await self.bus.create_run(Run("sample", "sampler", "tiny", 4, role=Role.SAMPLER, model_path=uri))
    preparing, finish = asyncio.Event(), threading.Event()
    local, loop = node.store.local_folder, asyncio.get_running_loop()

    def blocked(uri, version):
      loop.call_soon_threadsafe(preparing.set)
      if not finish.wait(10):
        raise TimeoutError("checkpoint preparation was never released")
      return local(uri, version)

    try:
      with (
        patch.object(node.store, "local_folder", side_effect=blocked),
        patch.object(self.bus, "take_commands", wraps=self.bus.take_commands) as reads,
      ):
        running = await node.add("sampler")
        owner, stopping = await self.bus.lease_owner("sampler"), node.tasks["sampler"]
        initial = await self.bus.submit_command("base", Op.SAMPLE, json.dumps(JOB).encode())
        await self.reply(initial, node)
        self.assertIsNotNone(running.since)
        first = await self.bus.submit_command("sample", Op.SAMPLE, json.dumps(JOB).encode())
        await asyncio.wait_for(preparing.wait(), 5)
        count = len([call for call in reads.call_args_list if call.args[1] == owner])
        await self.bus.append_event(Ev.UNPLACED, worker="sampler")
        await node.tick()
        later = await self.bus.submit_command("sample", Op.SAMPLE, json.dumps(JOB).encode())
        await self.bus.create_run(Run("other", "other", "tiny", 4, role=Role.SAMPLER))
        await node.add("other", gpus=[0])
        contender = await self.bus.submit_command("other", Op.SAMPLE, json.dumps(JOB).encode())
        reply = await self.bus.wait_for_reply(contender, 3)
        self.assertIsNotNone(reply, "checkpoint preparation held the GPU after unplacement")
        self.assertTrue(reply.ok, reply.body)
        self.assertFalse(finish.is_set())
        await asyncio.wait_for(asyncio.shield(stopping), 5)
        self.assertIsNone(running.since)
        self.assertIsNone(await self.bus.wait_for_reply(first, 0))
        self.assertEqual(len([call for call in reads.call_args_list if call.args[1] == owner]), count)
      self.assertIsNone(await self.bus.wait_for_reply(first, 0))
      self.assertIsNone(await self.bus.wait_for_reply(later, 0))
      self.assertEqual(await self.bus.queued_and_in_flight("sample"), {"live": True, "queued": 2, "in_flight": []})
      node.dispatch()
      self.assertNotIn("sampler", node.tasks)
      self.assertIsNone(await self.bus.lease_owner("sampler"))
    finally:
      finish.set()

  async def test_a_retiring_wide_worker_keeps_its_ticket_and_intake_closed_when_placed_again(self):
    node = self.node(gpus=2)
    wide_run, wide = await self.prepared(node, "wide", (0, 1))
    holder_run, _ = await self.prepared(node, "holder", (0, 1))
    self.assertIsNone(wide.since)
    child, task = node.processes["wide"], node.tasks["wide"]
    owner = await self.bus.lease_owner("wide")
    executing, finish_work, saving, finish_save = (asyncio.Event() for _ in range(4))
    execute, save = node.processes["holder"].call, child.call

    async def blocked_work(method, *args):
      if method == "execute":
        executing.set()
        await finish_work.wait()
      return await execute(method, *args)

    async def blocked_save(method, *args):
      if method == "park_all":
        saving.set()
        await finish_save.wait()
      return await save(method, *args)

    try:
      with (
        patch.object(node.processes["holder"], "call", side_effect=blocked_work),
        patch.object(child, "call", side_effect=blocked_save),
      ):
        await self.bus.submit_command(holder_run, Op.OPTIM_STEP, b'{"adam_params":{}}')
        await self.until(executing.is_set, node)
        await self.bus.append_event(Ev.UNPLACED, worker="wide")
        await self.until(lambda: wide.ticket > 0, node)
        ticket = wide.ticket
        await self.bus.create_run(Run("later", "later", "tiny", 4, lora=LORA))
        later = await node.add("later", gpus=[0])
        waiting = await self.bus.submit_command(
          "later", Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode()
        )
        await self.until(lambda: later.ticket > 0, node)
        self.assertGreater(later.ticket, ticket)

        with patch.object(self.bus, "take_commands", wraps=self.bus.take_commands) as reads:
          await self.bus.append_event(Ev.PLACED, worker="wide", node=node.id)
          after_stop = await self.bus.submit_command(wide_run, Op.OPTIM_STEP, b'{"adam_params":{}}')
          for _ in range(5):
            await node.tick()
            await asyncio.sleep(0.01)
            self.assertEqual(wide.ticket, ticket)
            self.assertIsNone(wide.since)
          finish_work.set()
          await self.until(saving.is_set, node)
          self.assertEqual(wide.gpus, (0, 1))
          self.assertIsNotNone(wide.since)
          self.assertEqual(wide.ticket, 0)
          self.assertIsNone(later.since)
          self.assertIsNone(await self.bus.wait_for_reply(waiting, 0))
          self.assertIsNone(await self.bus.wait_for_reply(after_stop, 0))
          finish_save.set()
          await asyncio.wait_for(asyncio.shield(task), 5)
          self.assertFalse([call for call in reads.call_args_list if call.args[1] == owner])
        self.assertTrue(child.closed)
        self.assertIsNone(await self.bus.lease_owner("wide"))
        self.assertEqual(await self.bus.queued_and_in_flight(wide_run), {"live": True, "queued": 1, "in_flight": []})
        self.assertIsNone(await self.bus.wait_for_reply(after_stop, 0))
        uri = await self.bus.parked_checkpoint(wide_run)
        self.assertIsNotNone(uri)
        self.assertIsNotNone(await self.registry.get_checkpoint(uri))
        self.assertTrue((await self.bus.wait_for_reply(waiting, 5)).ok)
    finally:
      finish_work.set()
      finish_save.set()

  async def test_a_blocked_close_keeps_its_gpu_and_lease_while_other_gpus_progress(self):
    node = self.node(gpus=2)
    _, a = await self.prepared(node, "a", (0,))
    for run, worker, gpu in (("rc", "c", 1), ("rb", "b", 0)):
      await self.bus.create_run(Run(run, worker, "tiny", 4, lora=LORA))
      await node.add(worker, gpus=[gpu])
    entered, finish = asyncio.Event(), asyncio.Event()
    child = node.processes["a"]
    owner = await self.bus.lease_owner("a")
    original = child.close

    async def slow_close():
      entered.set()
      await finish.wait()
      await original()

    try:
      with patch.object(child, "close", side_effect=slow_close):
        await self.bus.append_event(Ev.UNPLACED, worker="a")
        await self.until(entered.is_set, node)
        waiting = await self.bus.submit_command(
          "rb", Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode()
        )
        other = await self.bus.submit_command(
          "rc", Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode()
        )
        await self.reply(other, node)
        self.assertIsNotNone(a.since)
        self.assertIsNone(node.running["b"].since)
        self.assertEqual(await self.bus.lease_owner("a"), owner)
        self.assertIsNone(await self.bus.wait_for_reply(waiting, 0))
        finish.set()
        await self.reply(waiting, node)
        self.assertTrue(child.closed)
        self.assertIsNone(await self.bus.lease_owner("a"))
    finally:
      finish.set()

  async def test_cancelling_node_cleanup_keeps_ownership_until_the_child_is_closed(self):
    node = self.node()
    _, running = await self.prepared(node)
    entered, finish = asyncio.Event(), asyncio.Event()
    child = node.processes["worker"]
    owner = await self.bus.lease_owner("worker")
    close = child.close

    async def slow_close():
      entered.set()
      await finish.wait()
      await close()

    try:
      with patch.object(child, "close", side_effect=slow_close):
        closing = node.tasks["worker"]
        await self.bus.append_event(Ev.UNPLACED, worker="worker")
        await self.until(entered.is_set, node)
        closing.cancel()
        await asyncio.sleep(0.02)
        self.assertFalse(closing.done())
        self.assertFalse(child.closed)
        self.assertEqual(await self.bus.lease_owner("worker"), owner)
        self.assertIsNotNone(running.since)
        finish.set()
        await asyncio.gather(closing, return_exceptions=True)
      self.assertTrue(child.closed)
      self.assertIsNone(running.since)
      self.assertIsNone(await self.bus.lease_owner("worker"))
    finally:
      finish.set()

  async def test_failed_cleanup_keeps_the_gpu_and_lease_for_a_retry(self):
    node = self.node()
    _, running = await self.prepared(node)
    entered, finish, attempts = asyncio.Event(), asyncio.Event(), []
    child = node.processes["worker"]
    owner = await self.bus.lease_owner("worker")
    close = child.close

    async def retry_close():
      attempts.append(True)
      if len(attempts) == 1:
        raise RuntimeError("could not reap")
      entered.set()
      await finish.wait()
      await close()

    try:
      with patch.object(child, "close", side_effect=retry_close):
        await self.bus.append_event(Ev.UNPLACED, worker="worker")
        await self.until(entered.is_set, node)
        self.assertEqual(len(attempts), 2)
        self.assertFalse(child.closed)
        self.assertIsNotNone(running.since)
        self.assertEqual(await self.bus.lease_owner("worker"), owner)
        finish.set()
        await self.until(lambda: "worker" not in node.running, node)
      self.assertTrue(child.closed)
      self.assertIsNone(await self.bus.lease_owner("worker"))
    finally:
      finish.set()

  async def test_a_cancelled_child_close_retries_without_blocking_the_node(self):
    node = self.node()
    _, running = await self.prepared(node)
    child = node.processes["worker"]
    owner = await self.bus.lease_owner("worker")
    entered, finish, attempts = asyncio.Event(), asyncio.Event(), []
    close = child.close

    async def retry_close():
      attempts.append(True)
      if len(attempts) == 1:
        raise asyncio.CancelledError
      entered.set()
      await finish.wait()
      await close()

    try:
      with patch.object(child, "close", side_effect=retry_close):
        await self.bus.append_event(Ev.UNPLACED, worker="worker")
        await self.until(entered.is_set, node)
        heartbeat = (await self.bus.live_nodes(details=True))[node.id]["until"]
        await node.tick()
        self.assertGreater((await self.bus.live_nodes(details=True))[node.id]["until"], heartbeat)
        self.assertEqual(len(attempts), 2)
        self.assertFalse(child.closed)
        self.assertIsNotNone(running.since)
        self.assertEqual(await self.bus.lease_owner("worker"), owner)
        finish.set()
        await self.until(lambda: "worker" not in node.running, node)
      self.assertTrue(child.closed)
      self.assertIsNone(await self.bus.lease_owner("worker"))
    finally:
      finish.set()

  @patch("tinyrl.bus.LEASE_MS", 800)
  @patch("tinyrl.control.node_daemon.LEASE_MS", 800)
  async def test_uncertain_cleanup_keeps_its_gpu_and_lease_past_the_lease_timeout(self):
    node = self.node()
    _, running = await self.prepared(node)
    child, task = node.processes["worker"], node.tasks["worker"]
    owner = await self.bus.lease_owner("worker")
    entered, finish, attempts = asyncio.Event(), asyncio.Event(), []
    close = child.close

    async def uncertain_close():
      attempts.append(True)
      if len(attempts) == 1:
        raise TimeoutError("child exit has not been confirmed")
      entered.set()
      await finish.wait()
      await close()

    try:
      with patch.object(child, "close", side_effect=uncertain_close):
        task.cancel()
        await self.until(entered.is_set, node)
        await asyncio.sleep(1.7)  # pass two lease deadlines without advancing the control loop
        self.assertFalse(task.done())
        self.assertFalse(child.closed)
        self.assertIsNotNone(running.since)
        self.assertEqual(await self.bus.lease_owner("worker"), owner)
        self.assertFalse(await self.bus.take_lease("worker", "replacement"))
        finish.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
      self.assertTrue(child.closed)
      self.assertIsNone(running.since)
      self.assertIsNone(await self.bus.lease_owner("worker"))
    finally:
      finish.set()

  async def test_restarting_the_cluster_scheduler_does_not_restart_execution(self):
    run = new_id()
    await self.bus.create_run(Run(run, "worker", "tiny", 4, lora=LORA))
    node = self.node()
    await node.announce_node()
    first_scheduler = self.background(scheduler.run(self.bus))
    await self.until(lambda: "worker" in node.running, node)
    await self.create(run)
    first = node.tasks["worker"]
    child = node.processes["worker"]
    owner = await self.bus.lease_owner("worker")
    first_scheduler.cancel()
    await asyncio.gather(first_scheduler, return_exceptions=True)
    self.background(scheduler.run(self.bus))
    for _ in range(5):
      await node.tick()
      await asyncio.sleep(0.03)
    self.assertIs(node.tasks["worker"], first)
    self.assertIs(node.processes["worker"], child)
    self.assertEqual(await self.bus.lease_owner("worker"), owner)
    request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
    await self.reply(request, node)
    local = {"started", "wants_gpu", "granted", "revoked", "batch_done", "idle", "released", "exited"}
    self.assertFalse(local.intersection(e.kind for e in await self.bus.read_events()))

  async def test_launch_settings_are_per_worker_and_multi_gpu_uses_torchrun(self):
    node = self.node(gpus=8, real=True, gpu_mib=81559)
    node.state.nodes[node.id] = (8, node.gpu_mib * 2**20)
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
      patch.object(node, "open_sampler", wraps=node.open_sampler) as sampler,
    ):
      node.start("sample")
      await self.until(lambda: sampler.called)
      env = sampler.call_args.args[1]
      self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "6")
      self.assertEqual(env["TINYRL_GPU_MEM"], "0.42")
      self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", env)
      node.start("lora")
      await self.until(lambda: process.call_count == 1)
      cmd, env = process.call_args.args
      self.assertEqual(cmd, [sys.executable, "-m", "tinyrl.workers.trainer"])
      self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
      self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
      with patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}):
        node.start("fft")
        await self.until(lambda: process.call_count == 2)
      self.assertEqual(process.call_args.args[1]["CUDA_VISIBLE_DEVICES"], "6")
      self.assertEqual(process.call_args.args[1]["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:512")
      self.assertNotIn("TINYRL_NODE", process.call_args.args[1])
      node.start("multi")
      await self.until(lambda: process.call_count == 3)
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
      self.assertIsNone(await node.start("worker"))
    process.assert_not_called()
    self.assertNotIn("worker", node.running)

  async def test_startup_rechecks_ownership_after_stranded_command_cleanup(self):
    node = self.node(real=True)
    node.state.workers["worker"] = scheduler.Worker(scheduler.Kind(base="tiny"), node=node.id, gpus=(0,))
    entered, finish = asyncio.Event(), asyncio.Event()
    cleanup = self.bus.fail_stranded_commands

    async def delayed(*args):
      entered.set()
      await finish.wait()
      return await cleanup(*args)

    try:
      with (
        patch.object(self.bus, "fail_stranded_commands", side_effect=delayed),
        patch("tinyrl.control.node_daemon.Process") as process,
      ):
        starting = node.start("worker")
        await asyncio.wait_for(entered.wait(), 5)
        self.assertIsNotNone(await self.bus.lease_owner("worker"))
        self.assertNotIn("worker", node.running)
        process.assert_not_called()
        await self.bus.r.pexpire("tinyrl:owner:worker", 0)
        self.assertTrue(await self.bus.take_lease("worker", "new-owner"))
        finish.set()
        self.assertIsNone(await starting)
        process.assert_not_called()
      self.assertNotIn("worker", node.running)
      self.assertEqual(await self.bus.lease_owner("worker"), "new-owner")
    finally:
      finish.set()

  async def test_cancelled_startup_never_leaves_a_child_or_lease(self):
    for stage in ("before_start", "after_lease"):
      with self.subTest(stage=stage):
        node = self.node(node=stage)
        await self.bus.create_run(Run(stage, stage, "tiny", 4, lora=LORA))
        await node.sync()
        node.state.workers[stage].node = node.id
        node.state.workers[stage].gpus = (0,)
        entered = asyncio.Event()
        cleanup = self.bus.fail_stranded_commands

        async def held_cleanup(*args, entered=entered, cleanup=cleanup):
          if not entered.is_set():
            entered.set()
            await asyncio.Event().wait()
          return await cleanup(*args)

        with patch.object(self.bus, "fail_stranded_commands", side_effect=held_cleanup):
          starting = node.start(stage)
          if stage == "after_lease":
            await asyncio.wait_for(entered.wait(), 5)
            self.assertIsNotNone(await self.bus.lease_owner(stage))
          starting.cancel()
          [result] = await asyncio.gather(starting, return_exceptions=True)
        self.assertIsInstance(result, asyncio.CancelledError)
        node.dispatch()
        self.assertNotIn(stage, node.tasks)
        self.assertNotIn(stage, node.running)
        self.assertNotIn(stage, node.processes)
        self.assertIsNone(await self.bus.lease_owner(stage))

  async def test_a_lost_node_reannounces_only_after_its_model_has_closed(self):
    await self.bus.create_run(Run("run", "worker", "tiny", 4))
    node = self.node()
    await node.announce_node()
    self.background(scheduler.run(self.bus))
    await self.until(lambda: "worker" in node.running, node)
    child = node.processes["worker"]
    close, entered, finish = child.close, asyncio.Event(), asyncio.Event()

    async def slow_close():
      entered.set()
      await finish.wait()
      await close()

    try:
      with patch.object(child, "close", side_effect=slow_close):
        await self.bus.append_event(Ev.NODE_LOST, node=node.id)
        await self.until(entered.is_set, node)
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

    async def slow_sweep(_store, _grace):
      sweeps.append(time.time())
      await finish.wait()

    node.registry.sweep = slow_sweep
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
      launched, factory = [], node.open_trainer

      def capture(worker, env):
        launched.append(process := factory(worker, env))
        return process

      with patch.object(node, "open_trainer", side_effect=capture):
        await node.announce_node()
        self.background(scheduler.run(self.bus))
        await self.until(lambda: "worker" in node.running, node)
        await self.create(run, base)
        first = launched[-1].proc
        self.assertEqual(node.running["worker"].pid, first.pid)
        node.state.workers["worker"].gpus = (1,)
        request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
        await self.reply(request, node)
        self.assertEqual(len(launched), 1)
        self.assertEqual(node.running["worker"].gpus, (1,))
        first.kill()
        await asyncio.to_thread(first.wait)
        await self.until(lambda: len(launched) == 2, node)
        self.assertNotEqual(node.running["worker"].pid, first.pid)
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
    await self.registry.sweep(store, SWEEP_GRACE_S)
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
    await self.registry.sweep(store, -60)
    self.assertEqual(sorted(p.rsplit("/", 1)[1] for p in store.fs.glob(f"{store.base}/r/_versions/*")), v[:2])

  async def test_a_sweep_moves_on_past_a_checkpoint_it_cannot_delete(self):
    store = Store(f"memory://tinyrl-test/{uuid.uuid4()}", cache=tempfile.mkdtemp())
    for name in ("a", "b"):
      store.put(f"tinker://r/weights/{name}", {"w": torch.ones(2)})
    store.upload_pending_saves()
    await self.bus.r.set("tinyrl:ckpt_since", 0)
    delete = store.delete
    store.delete = lambda uri: (_ for _ in ()).throw(OSError("503")) if uri.endswith("/a") else delete(uri)
    await self.registry.sweep(store, -60)
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
    await self.registry.sweep(store, -60)
    self.assertFalse(store.fs.exists(store.path("tinker://r/weights/w")))


if __name__ == "__main__":
  unittest.main()
