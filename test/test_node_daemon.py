import asyncio
import contextlib
import fcntl
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
from tinyrl.storage.store import Store, file_lock

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

  async def test_a_checkpoint_whose_node_left_before_it_uploaded_fails_fast(self):
    with tempfile.TemporaryDirectory() as folder:
      writer = Store(f"file://{folder}/bucket", cache=f"{folder}/writer")
      writer.peer = "127.0.0.1:9"  # a node that no longer answers
      node = self.node(store=Store(writer.root, cache=f"{folder}/reader"))
      uri = "tinker://training/sampler_weights/lost"
      version = writer.put(uri, {}, {"base": "tiny", "lora": True, "config": {}})  # never uploaded
      await self.seed_checkpoint(uri, 0, version=version)
      await self.bus.create_run(Run("lost", "sampler", "tiny", 4, role=Role.SAMPLER, model_path=uri))
      await node.add("sampler")
      request = await self.bus.submit_command("lost", Op.SAMPLE, json.dumps(JOB).encode())
      waiting = self.background(self.bus.wait_for_reply(request, 10))
      await self.until(waiting.done, node)
      self.assertFalse(waiting.result().ok)
      self.assertIn("has not landed, and 127.0.0.1:9 failed to copy", waiting.result().body.decode())

  async def test_staging_one_job_does_not_block_a_ready_sibling(self):
    node = self.node()
    uri = "tinker://training/sampler_weights/slow"
    version = node.store.put(uri, {}, {"base": "tiny", "lora": True})
    await self.seed_checkpoint(uri, 0, version=version)
    await self.bus.create_run(Run("slow", "sampler", "tiny", 4, role=Role.SAMPLER, model_path=uri, version=version))
    await self.bus.create_run(Run("ready", "sampler", "tiny", 4, role=Role.SAMPLER))
    with file_lock(node.store.folder(version), fcntl.LOCK_EX):
      slow = await self.bus.submit_command("slow", Op.SAMPLE, json.dumps(JOB).encode())
      ready = await self.bus.submit_command("ready", Op.SAMPLE, json.dumps(JOB).encode())
      await self.reply(ready, node)
      self.assertIsNone(await self.bus.wait_for_reply(slow, 0))
      self.assertEqual((await self.bus.queued_and_in_flight("slow"))["queued"], 1)
    await self.reply(slow, node)

  async def test_admission_waits_for_the_previous_model_process_to_park(self):
    require_unix()
    node = self.node(real=True)
    base = f"{node.store.root}/base"
    tiny().save_pretrained(base)
    for run in ("a", "b"):
      await self.bus.create_run(Run(run, run, base, 4, lora=LORA))
    await node.tick()
    self.background(scheduler.run(self.bus))
    await self.create("a", base, node)
    holder = node.running["a"]["pid"]
    await self.until(lambda: node.running["a"]["released"], node)
    os.killpg(holder, signal.SIGSTOP)
    try:
      await self.bus.submit_command("a", Op.OPTIM_STEP, b'{"adam_params":{}}')
      await self.until(lambda: not node.running["a"]["released"], node)
      request = await self.bus.submit_command(
        "b", Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 4, "lora": LORA}).encode()
      )
      await self.until(lambda: "b" in node.running and node.running["b"]["ready"], node)
      self.assertIsNone(node.running["b"]["pid"])
      self.assertIsNone(await self.bus.wait_for_reply(request, 0))
      heartbeat = (await self.bus.read_table())["nodes"][node.id]["until"]
      await node.tick()
      self.assertGreater((await self.bus.read_table())["nodes"][node.id]["until"], heartbeat)
    finally:
      with contextlib.suppress(ProcessLookupError):
        os.killpg(holder, signal.SIGCONT)
    await self.reply(request, node)
    self.assertTrue(node.running["a"]["released"])

  async def test_an_idle_process_death_is_detected_and_initial_state_reconstructed(self):
    require_unix()
    node = self.node(real=True)
    base = f"{node.store.root}/base"
    tiny().save_pretrained(base)
    run = new_id()
    await self.bus.create_run(Run(run, "worker", base, 4, lora=LORA))
    await node.tick()
    self.background(scheduler.run(self.bus))
    await self.create(run, base, node)
    first = node.running["worker"]["pid"]
    await self.until(lambda: node.running["worker"]["released"], node)
    os.killpg(first, signal.SIGKILL)
    await self.until(lambda: "worker" not in node.running or node.running["worker"]["pid"] != first, node)
    request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
    await self.reply(request, node)
    self.assertNotEqual(node.running["worker"]["pid"], first)

  async def test_the_node_uploads_what_its_model_saved(self):
    with tempfile.TemporaryDirectory() as folder:
      store = Store(f"file://{folder}/bucket", cache=f"{folder}/cache")
      uri = "tinker://r/weights/w"
      version = store.put(uri, {"w": torch.ones(2)})
      node = self.node(store=store)
      await self.until(lambda: store.is_durable(uri, version), node)
      self.assertTrue(store.is_durable(uri, version))


if __name__ == "__main__":
  unittest.main()
