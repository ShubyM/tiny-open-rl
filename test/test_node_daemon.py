import asyncio
import contextlib
import copy
import fcntl
import io
import json
import os
import signal
import tempfile
import unittest

import torch

from test.execution import TestNode
from test.helpers import RedisTest, new_id, tiny
from test.test_api import proto_body
from test.test_process import require_unix
from test.test_sampler import JOB
from tinyrl.bus import Op, Role, Run, key
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
    record = await self.bus.get_run(run)
    request = await self.bus.submit_command(
      run, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": record["rank"], "lora": record["lora"]}).encode()
    )
    await self.reply(request, node)

  async def test_deleting_a_job_releases_a_turn_the_engine_never_observed(self):
    node, run = self.node(), new_id()
    await self.bus.create_run(Run(run, "tiny", 4, lora=LORA))
    await self.create(run, "tiny", node)
    engine = await node.worker_for(run)
    await self.until(lambda: not node.state["turns"], node)
    node.driver.cancel()
    await asyncio.gather(node.driver, return_exceptions=True)
    await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
    async with asyncio.timeout(5):
      while True:
        old = await self.bus.read_table()
        if run in old["observations"][engine]["ready"]:
          break
        await asyncio.sleep(0.001)
    await scheduler.commit(self.bus, old, scheduler.decide(old))
    self.assertTrue((await self.bus.read_table())["turns"])
    self.assertFalse(node.state["turns"])
    await self.bus.delete_run(run)
    await node.tick()
    await asyncio.gather(*node.tasks.values(), return_exceptions=True)
    self.assertFalse((await self.bus.read_table())["turns"])
    self.assertTrue(node.processes[engine].closed)

  async def test_a_checkpoint_whose_node_left_before_it_uploaded_fails_fast(self):
    with tempfile.TemporaryDirectory() as folder:
      writer = Store(f"file://{folder}/bucket", cache=f"{folder}/writer")
      writer.peer = "127.0.0.1:9"  # a node that no longer answers
      node = self.node(store=Store(writer.root, cache=f"{folder}/reader"))
      uri = "tinker://training/sampler_weights/lost"
      version = writer.put(uri, {}, {"base": "tiny", "lora": True, "config": {}})  # never uploaded
      await self.seed_checkpoint(uri, 0, version=version)
      await self.bus.create_run(Run("lost", "tiny", 4, role=Role.SAMPLER, model_path=uri, version=version))
      await node.add(await node.worker_for("lost"))
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
    await self.bus.create_run(Run("slow", "tiny", 4, role=Role.SAMPLER, model_path=uri, version=version))
    await self.bus.create_run(Run("ready", "tiny", 4, role=Role.SAMPLER))
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
      await self.bus.create_run(Run(run, base, 0))
    await node.tick()
    self.background(scheduler.run(self.bus))
    await self.create("a", base, node)
    engine = (await self.bus.read_table())["claims"]["a"]["engine"]
    await self.until(lambda: not node.state["turns"] and node.state["observations"][engine]["pid"], node)
    holder = node.state["observations"][engine]["pid"]
    os.killpg(holder, signal.SIGSTOP)
    try:
      await self.bus.submit_command("a", Op.OPTIM_STEP, b'{"adam_params":{}}')
      await self.until(lambda: any(turn["engine"] == engine for turn in node.state["turns"].values()), node)
      request = await self.bus.submit_command(
        "b", Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 0, "lora": {}}).encode()
      )
      sibling = (await self.bus.read_table())["claims"]["b"]["engine"]
      await self.until(lambda: node.state["observations"].get(sibling, {}).get("ready"), node)
      self.assertIsNone(node.state["observations"][sibling]["pid"])
      self.assertIsNone(await self.bus.wait_for_reply(request, 0))
    finally:
      with contextlib.suppress(ProcessLookupError):
        os.killpg(holder, signal.SIGCONT)
    await self.reply(request, node)
    self.assertNotIn(engine, {turn["engine"] for turn in node.state["turns"].values()})

  async def test_an_idle_process_death_is_detected_and_initial_state_reconstructed(self):
    require_unix()
    node = self.node(real=True)
    base = f"{node.store.root}/base"
    tiny().save_pretrained(base)
    run = new_id()
    await self.bus.create_run(Run(run, base, 4, lora=LORA))
    await node.tick()
    self.background(scheduler.run(self.bus))
    await self.create(run, base, node)
    engine = (await self.bus.read_table())["claims"][run]["engine"]
    await self.until(lambda: not node.state["turns"] and node.state["observations"][engine]["pid"], node)
    first = node.state["observations"][engine]["pid"]
    os.killpg(first, signal.SIGKILL)
    await self.until(lambda: node.state["claims"].get(run, {}).get("engine") not in (None, engine), node)
    request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
    await self.reply(request, node)
    current = node.state["claims"][run]["engine"]
    await self.until(lambda: node.state["observations"][current]["pid"], node)
    self.assertNotEqual(node.state["observations"][current]["pid"], first)

  async def test_the_node_uploads_what_its_model_saved(self):
    with tempfile.TemporaryDirectory() as folder:
      store = Store(f"file://{folder}/bucket", cache=f"{folder}/cache")
      uri = "tinker://r/weights/w"
      version = store.put(uri, {"w": torch.ones(2)})
      node = self.node(store=store)
      await self.until(lambda: store.is_durable(uri, version), node)
      self.assertTrue(store.is_durable(uri, version))

  async def test_one_job_moves_without_discarding_its_shared_trainer(self):
    source = self.node()
    moving, sibling = new_id(), new_id()
    for r in (moving, sibling):
      await self.bus.create_run(Run(r, "tiny", 4, lora=LORA))
      await self.create(r, "tiny", source)
    await self.reply(await self.bus.submit_command(moving, Op.FORWARD_BACKWARD, proto_body(moving)), source)
    original = (await self.bus.read_table())["claims"]
    engine = original[moving]["engine"]
    self.assertEqual(engine, original[sibling]["engine"])
    model = source.processes[engine].trainer.model
    expected = copy.deepcopy(model.training_state(moving)[0])
    self.assertIsNone((await self.bus.get_run(moving))["state"]["working"])  # ordinary turns did not checkpoint
    destination = TestNode(self.bus, store=Store(source.store.root), gpus=1, node="destination")
    self.nodes.append(destination)
    await destination.tick()
    cache, blocked = source.store.cache, f"{source.store.root}/not-a-directory"
    with open(blocked, "w"):
      pass
    try:
      source.store.cache = blocked
      with contextlib.redirect_stderr(io.StringIO()) as errors:
        await self.bus.r.hset(key("moves"), moving, json.dumps(destination.id))
        await self.until(lambda: "keeping its live state" in errors.getvalue(), source)
      self.assertEqual((await self.bus.read_table())["claims"][moving], original[moving])
      self.assertIn(moving, model.params)
    finally:
      source.store.cache = cache
    await self.until(lambda: source.state["claims"].get(moving, {}).get("node") == destination.id, source)
    self.assertEqual(source.state["claims"][sibling], original[sibling])
    self.assertNotEqual(source.state["claims"][moving]["token"], original[moving]["token"])
    uri = f"tinker://{moving}/weights/after-move"
    await self.reply(
      await self.bus.submit_command(moving, Op.SAVE_WEIGHTS, json.dumps({"uri": uri}).encode()), destination
    )
    tensors, _ = destination.store.get(uri, await self.registry.version(uri))
    torch.testing.assert_close(tensors, expected, atol=0, rtol=0)
    self.assertIn(sibling, model.params)
    self.assertNotIn(moving, model.params)


if __name__ == "__main__":
  unittest.main()
