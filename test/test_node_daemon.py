import asyncio
import contextlib
import json
import os
import signal
import tempfile
import unittest

import torch

from test.execution import TestNode
from test.helpers import RedisTest, new_id, tiny
from test.test_process import require_unix
from test.test_sampler import JOB
from tinyrl.bus import Op, Role, Run
from tinyrl.control import scheduler
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
      for task in (node.uploading, node.sweeping):
        if task:
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

  async def test_an_unpublished_checkpoint_does_not_block_ready_work_or_gpu_handoff(self):
    with tempfile.TemporaryDirectory() as folder:
      writer = Store(f"file://{folder}/bucket", cache=f"{folder}/writer")
      node = self.node(store=Store(writer.root, cache=f"{folder}/reader"))
      uri = "tinker://training/sampler_weights/slow"
      writer.put(uri, {}, {"base": "tiny", "lora": True, "config": {}})
      await self.registry.register_checkpoint(uri, 0, version=writer.versions[uri], node=node.id)  # live, so it waits
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
        self.assertIsNone(resident.since)
        self.assertIsNotNone(contender.since)
        self.assertIsNone(await self.bus.wait_for_reply(slow, 0))
        self.assertEqual(await self.bus.queued_and_in_flight("slow"), {"live": True, "queued": 1, "in_flight": []})
      finally:
        await asyncio.to_thread(writer.upload_pending_saves)
      await self.reply(slow, node)

  async def test_a_checkpoint_whose_node_left_before_it_uploaded_fails_fast(self):
    with tempfile.TemporaryDirectory() as folder:
      writer = Store(f"file://{folder}/bucket", cache=f"{folder}/writer")
      node = self.node(store=Store(writer.root, cache=f"{folder}/reader"))
      uri = "tinker://training/sampler_weights/lost"
      writer.put(uri, {}, {"base": "tiny", "lora": True, "config": {}})  # never uploaded
      await self.registry.register_checkpoint(uri, 0, version=writer.versions[uri], node="gone")
      await self.bus.create_run(Run("lost", "sampler", "tiny", 4, role=Role.SAMPLER, model_path=uri))
      await node.add("sampler")
      request = await self.bus.submit_command("lost", Op.SAMPLE, json.dumps(JOB).encode())
      waiting = self.background(self.bus.wait_for_reply(request, 10))
      await self.until(waiting.done, node)
      self.assertFalse(waiting.result().ok)
      self.assertIn("was lost with node gone", waiting.result().body.decode())

  async def test_a_newer_save_from_a_live_node_is_waited_for_even_after_an_older_saver_left(self):
    with tempfile.TemporaryDirectory() as folder:
      writer = Store(f"file://{folder}/bucket", cache=f"{folder}/writer")
      node = self.node(store=Store(writer.root, cache=f"{folder}/reader"))
      uri, meta = "tinker://training/sampler_weights/w", {"base": "tiny", "lora": True, "config": {}}
      writer.put(uri, {}, meta)
      await self.registry.register_checkpoint(uri, 0, version=writer.versions[uri], node="gone")
      reading = self.background(node.landed(uri))
      await asyncio.sleep(0.2)  # it has seen "gone" leave
      writer.put(uri, {}, meta)
      await self.bus.node_heartbeat("live", 1)
      await self.registry.register_checkpoint(uri, 0, version=writer.versions[uri], node="live")
      await asyncio.sleep(1)
      self.assertFalse(reading.done())
      await asyncio.to_thread(writer.upload_pending_saves)
      self.assertEqual(await asyncio.wait_for(reading, 5), writer.versions[uri])

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
      await self.until(lambda: "b" in node.running and node.running["b"].ticket > 0, node)
      self.assertIsNotNone(holder.since)
      self.assertIsNone(node.running["b"].since)
      self.assertEqual(await self.bus.queued_and_in_flight(b), {"live": True, "queued": 1, "in_flight": []})
      heartbeat = (await self.bus.live_nodes(details=True))[node.id]["until"]
      await node.tick()
      self.assertGreater((await self.bus.live_nodes(details=True))[node.id]["until"], heartbeat)
    finally:
      with contextlib.suppress(ProcessLookupError):
        os.killpg(holder.pid, signal.SIGCONT)
    await self.reply(request, node)
    self.assertIsNone(holder.since)

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
      await node.tick()
      await node.uploading
      self.assertEqual(store.list_checkpoints(), ["tinker://r/weights/w"])


if __name__ == "__main__":
  unittest.main()
