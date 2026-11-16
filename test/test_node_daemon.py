import asyncio
import contextlib
import json
import os
import signal
import tempfile
import threading
import unittest
from unittest import mock

import torch

from test.execution import TestNode
from test.helpers import RedisTest, new_id, tiny
from test.test_process import require_unix
from test.test_sampler import JOB
from tinyrl.bus import Op, Role, Run, key
from tinyrl.control import node_daemon, scheduler, turns
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
      await self.bus.create_run(Run(run, "worker", "tiny", 4, lora=LORA))
      creates[run] = await self.bus.submit_command(
        run, Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode()
      )
    for _ in range(12):
      await self.bus.submit_command(a, Op.OPTIM_STEP, b'{"adam_params":{}}')
    await node.add("worker")
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
      release, read = threading.Event(), node.store.local_folder

      def slow_read(*args):  # a big copy still running
        release.wait()
        return read(*args)

      node.store.local_folder = slow_read
      uri = "tinker://training/sampler_weights/slow"
      version = writer.put(uri, {}, {"base": "tiny", "lora": True, "config": {}})
      await self.registry.register_checkpoint(uri, 0, version=version)
      await self.bus.create_run(Run("slow", "sampler", "tiny", 4, role=Role.SAMPLER, model_path=uri))
      await self.bus.create_run(Run("ready", "sampler", "tiny", 4, role=Role.SAMPLER))
      resident = await node.add("sampler")
      slow = await self.bus.submit_command("slow", Op.SAMPLE, json.dumps(JOB).encode())
      try:
        ready = await self.bus.submit_command("ready", Op.SAMPLE, json.dumps(JOB).encode())
        await self.reply(ready, node)
        await self.bus.create_run(Run("other", "other", "tiny", 4, role=Role.SAMPLER))
        contender = await node.add("other", gpus=[0])
        request = await self.bus.submit_command("other", Op.SAMPLE, json.dumps(JOB).encode())
        await self.reply(request, node)
        self.assertIsNone(resident.served)
        self.assertIsNotNone(contender.served)
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
      version = writer.put(uri, {}, {"base": "tiny", "lora": True, "config": {}})  # never uploaded
      await self.registry.register_checkpoint(uri, 0, version=version)
      await self.bus.create_run(Run("lost", "sampler", "tiny", 4, role=Role.SAMPLER, model_path=uri))
      await node.add("sampler")
      request = await self.bus.submit_command("lost", Op.SAMPLE, json.dumps(JOB).encode())
      waiting = self.background(self.bus.wait_for_reply(request, 10))
      await self.until(waiting.done, node)
      self.assertFalse(waiting.result().ok)
      self.assertIn("has not landed, and 127.0.0.1:9 failed to copy", waiting.result().body.decode())

  async def test_admission_waits_for_the_previous_model_process_to_park(self):
    require_unix()
    node = self.node(real=True)
    base = f"{node.store.root}/base"
    tiny().save_pretrained(base)
    a, b = new_id(), new_id()
    for run, worker in ((a, "a"), (b, "b")):
      await self.bus.create_run(Run(run, worker, base, 4, lora=LORA))
    await node.announce_node()
    self.background(scheduler.run(self.bus))
    await self.create(a, base, node)
    holder = node.running["a"]
    os.killpg(holder.pid, signal.SIGSTOP)
    try:
      request = await self.bus.submit_command(
        b, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 4, "lora": LORA}).encode()
      )
      await self.until(lambda: "b" in node.running and node.running["b"].vtime is not None, node)
      self.assertIsNotNone(holder.served)
      self.assertIsNone(node.running["b"].served)
      self.assertEqual(await self.bus.queued_and_in_flight(b), {"live": True, "queued": 1, "in_flight": []})
      heartbeat = (await self.bus.live_nodes(details=True))[node.id]["until"]
      await node.tick()
      self.assertGreater((await self.bus.live_nodes(details=True))[node.id]["until"], heartbeat)
    finally:
      with contextlib.suppress(ProcessLookupError):
        os.killpg(holder.pid, signal.SIGCONT)
    await self.reply(request, node)
    self.assertIsNone(holder.served)

  async def test_real_model_process_serves_moves_and_restarts(self):
    require_unix()
    node = self.node(gpus=2, real=True)
    base = f"{node.store.root}/base"
    tiny().save_pretrained(base)
    run = new_id()
    await self.bus.create_run(Run(run, "worker", base, 4, lora=LORA))
    await node.announce_node()
    self.background(scheduler.run(self.bus))
    await self.create(run, base, node)
    first = node.running["worker"].pid
    node.state.workers["worker"].gpus = (1,)
    request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
    await self.reply(request, node)
    self.assertEqual(node.running["worker"].pid, first)
    self.assertEqual(node.running["worker"].gpus, (1,))
    os.killpg(first, signal.SIGKILL)
    await self.until(lambda: "worker" in node.running and node.running["worker"].pid != first, node)
    await self.create(run, base, node)

  async def test_the_node_uploads_what_its_model_saved(self):
    with tempfile.TemporaryDirectory() as folder:
      store = Store(f"file://{folder}/bucket", cache=f"{folder}/cache")
      store.put("tinker://r/weights/w", {"w": torch.ones(2)})
      node = self.node(store=store)
      await self.until(lambda: store.list_checkpoints() == ["tinker://r/weights/w"], node)

  async def test_busy_holder_does_not_park_for_a_more_served_waiter_until_caught_up(self):
    node = self.node(gpus=1)
    for run, worker in (("more", "w-more"), ("less", "w-less")):
      await self.bus.create_run(Run(run, worker, "tiny", 4, lora=LORA))
      await node.add(worker, gpus=[0])
      await self.create(run, "tiny", node)
    await self.bus.r.set(key("vtime", "more"), 500)
    await self.bus.r.set(key("vtime", "less"), 10)
    parks = []
    orig_park = node.processes["w-less"].trainer.park

    def count_park():
      parks.append(1)
      return orig_park()

    node.processes["w-less"].trainer.park = count_park
    with mock.patch.object(turns, "TURN", 10):
      # get w-less onto GPU 0 first, then queue w-more (vtime=500) alongside 3 commands of w-less (10 -> 310 < 500)
      first = await self.bus.submit_command("less", Op.OPTIM_STEP, b'{"adam_params":{}}', cost=100)
      await self.reply(first, node)
      parks.clear()
      less_reqs = [
        await self.bus.submit_command("less", Op.OPTIM_STEP, b'{"adam_params":{}}', cost=100) for _ in range(3)
      ]
      more_req = await self.bus.submit_command("more", Op.OPTIM_STEP, b'{"adam_params":{}}', cost=10)
      for req in less_reqs:
        await self.reply(req, node)
      self.assertEqual(len(parks), 0)  # zero wasted self-handoffs while w-less (vtime <= 410) < w-more (500)!
      await self.reply(more_req, node)  # once w-less is idle, w-more takes the GPU

  async def test_heavy_sampling_in_an_account_prioritizes_another_accounts_training(self):
    node = self.node(gpus=2)
    await self.bus.create_run(Run("train-a", "trainer", "tiny", 4, session="sess-a", lora=LORA))
    await self.bus.create_run(Run("train-b", "trainer", "tiny", 4, session="sess-b", lora=LORA))
    await self.bus.create_run(
      Run("samp-a", "sampler", "tiny", 4, session="sess-a", role=Role.SAMPLER, train_run="train-a")
    )
    await node.add("trainer", gpus=[0])
    await node.add("sampler", gpus=[1])
    await self.create("train-a", "tiny", node)
    await self.create("train-b", "tiny", node)

    entered, release, order = asyncio.Event(), asyncio.Event(), []
    orig_call = node.processes["trainer"].call

    async def gated_call(method, *args):
      if method == "execute":
        cmd = args[0]
        if cmd.op == Op.FORWARD_BACKWARD:
          entered.set()
          await release.wait()
        elif cmd.op == Op.OPTIM_STEP:
          order.append(cmd.run)
      return await orig_call(method, *args)

    node.processes["trainer"].call = gated_call
    with mock.patch.object(node_daemon, "RUN_COMMANDS", 1):
      gate = await self.bus.submit_command("train-b", Op.FORWARD_BACKWARD, b"", cost=1)
      await self.until(entered.is_set, node)
      reqs_a = [
        await self.bus.submit_command("train-a", Op.OPTIM_STEP, b'{"adam_params":{}}', cost=100) for _ in range(3)
      ]
      reqs_b = [
        await self.bus.submit_command("train-b", Op.OPTIM_STEP, b'{"adam_params":{}}', cost=100) for _ in range(3)
      ]
      # While trainer is mid-command (and background peek prepares train-a), sess-a burns 5,000 sample tokens.
      samp_req = await self.bus.submit_command("samp-a", Op.SAMPLE, json.dumps(JOB).encode(), cost=5000)
      await self.reply(samp_req, node)
      release.set()
      await self.bus.wait_for_reply(gate, 5)
      for req in [*reqs_b, *reqs_a]:
        await self.reply(req, node)
    self.assertEqual(order, ["train-b", "train-b", "train-b", "train-a", "train-a", "train-a"])


if __name__ == "__main__":
  unittest.main()
