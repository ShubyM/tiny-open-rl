import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test.execution import TestNode
from test.helpers import RedisTest, new_id, tiny
from test.test_process import require_unix
from tinyrl.bus import GROUP, Cmd, Ev, Op, Reply, Run, key
from tinyrl.control import scheduler
from tinyrl.storage.store import Store

LEASE_MS = 2000
LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


class TestLease(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.nodes, self.tasks, self.processes = [], [], []
    self.folder = tempfile.TemporaryDirectory()
    self.addCleanup(self.folder.cleanup)
    self.store = Store(root=self.folder.name)

  async def asyncTearDown(self):
    for proc in self.processes:
      if proc.poll() is None:
        proc.send_signal(signal.SIGCONT)
        proc.terminate()
    for proc in self.processes:
      try:
        await asyncio.to_thread(proc.wait, 30)
      except subprocess.TimeoutExpired:
        proc.kill()
        await asyncio.to_thread(proc.wait)
    for task in self.tasks:
      task.cancel()
    await asyncio.gather(*self.tasks, return_exceptions=True)
    for node in self.nodes:
      await node.aclose()
    await super().asyncTearDown()

  def node(self, name):
    node = TestNode(self.bus, store=self.store, gpus=1, node=name)
    self.nodes.append(node)
    return node

  async def prepared(self):
    run = new_id()
    await self.bus.create_run(Run(run, "worker", "tiny", 4, lora=LORA))
    first = self.node("first")
    await first.add("worker")
    await self.bus.submit_command(
      run, Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode()
    )
    await first.drain(run)
    return first, run

  async def takeover(self):
    await self.bus.r.delete(key("owner", "worker"))  # its lease expired while the node was paused
    second = self.node("second")
    await second.add("worker")
    return second

  async def test_a_stale_parent_checks_ownership_before_dispatching_to_its_child(self):
    first, run = await self.prepared()
    owner, child = await self.bus.lease_owner("worker"), first.processes["worker"]
    # This metadata read follows the claim, so takeover must fail it before the old child can execute it.
    entered, finish, parked_checkpoint = asyncio.Event(), asyncio.Event(), self.bus.parked_checkpoint

    async def delayed(name):
      result = await parked_checkpoint(name)
      if name == run and not entered.is_set():
        entered.set()
        await finish.wait()
      return result

    try:
      with (
        patch.object(self.bus, "parked_checkpoint", side_effect=delayed),
        patch.object(child, "call", wraps=child.call) as call,
      ):
        request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
        await asyncio.wait_for(entered.wait(), 5)
        await self.takeover()
        new_owner = await self.bus.lease_owner("worker")
        expected = await self.bus.wait_for_reply(request, 0)
        self.assertFalse(expected.ok)
        finish.set()
        [result] = await asyncio.gather(first.tasks["worker"], return_exceptions=True)
        self.assertIsInstance(result, BaseException)
      call.assert_not_called()
      self.assertTrue(child.closed)
      self.assertEqual(await self.bus.wait_for_reply(request, 0), expected)
      cmd = Cmd(run, request.split("/", 1)[1], Op.OPTIM_STEP, b'{"adam_params":{}}')
      self.assertFalse(await self.bus.write_reply(cmd, Reply(True, "application/json", b"{}"), "worker", owner))
      self.assertEqual(await self.bus.lease_owner("worker"), new_owner)
    finally:
      finish.set()

  @patch("tinyrl.control.node_daemon.LEASE_MS", 800)
  async def test_a_renewal_racing_with_close_cannot_reacquire_the_released_lease(self):
    first, _ = await self.prepared()
    child = first.processes["worker"]
    entered, finish, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    renew = self.bus.renew_lease

    async def delayed(*args):
      entered.set()
      try:
        await finish.wait()
        return await renew(*args)
      except asyncio.CancelledError:
        cancelled.set()
        raise

    try:
      with (
        patch.object(self.bus, "renew_lease", side_effect=delayed),
        patch.object(self.bus, "take_lease", wraps=self.bus.take_lease) as acquire,
      ):
        await asyncio.wait_for(entered.wait(), 5)
        await first.close(graceful=False)
        self.assertTrue(cancelled.is_set())
        self.assertTrue(child.closed)
        self.assertIsNone(await self.bus.lease_owner("worker"))
        finish.set()
        await asyncio.sleep(0)
        acquire.assert_not_called()
      self.assertIsNone(await self.bus.lease_owner("worker"))
    finally:
      finish.set()

  async def test_a_completed_stale_result_cannot_replace_the_takeover_failure(self):
    first, run = await self.prepared()
    entered, finish = asyncio.Event(), asyncio.Event()
    child = first.processes["worker"]
    execute = child.call

    async def held_reply(method, *args):
      result = await execute(method, *args)
      if method == "execute":
        entered.set()
        await finish.wait()
      return result

    try:
      with patch.object(child, "call", side_effect=held_reply):
        request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
        await asyncio.wait_for(entered.wait(), 5)
        second = await self.takeover()
        expected = await self.bus.wait_for_reply(request, 0)
        self.assertFalse(expected.ok)
        finish.set()
        [result] = await asyncio.gather(first.tasks["worker"], return_exceptions=True)
        self.assertIsInstance(result, BaseException)
      self.assertEqual(await self.bus.wait_for_reply(request, 0), expected)
      self.assertEqual((await self.bus.queued_and_in_flight(run))["in_flight"], [])
      self.assertNotIn(run, second.processes["worker"].trainer.model.params)
      self.assertEqual((await self.bus.get_run(run))["steps"], 0)
    finally:
      finish.set()

  async def test_a_stale_checkpoint_result_is_not_registered(self):
    first, run = await self.prepared()
    uri = f"tinker://{run}/weights/stale"
    entered, finish = asyncio.Event(), asyncio.Event()
    child = first.processes["worker"]
    execute = child.call

    async def held_save(method, *args):
      result = await execute(method, *args)
      if method == "execute":
        entered.set()
        await finish.wait()
      return result

    try:
      with patch.object(child, "call", side_effect=held_save):
        request = await self.bus.submit_command(run, Op.SAVE_WEIGHTS, json.dumps({"uri": uri}).encode())
        await asyncio.wait_for(entered.wait(), 5)
        await self.takeover()
        finish.set()
        [result] = await asyncio.gather(first.tasks["worker"], return_exceptions=True)
        self.assertIsInstance(result, BaseException)
      self.assertIsNone(await self.registry.get_checkpoint(uri))
      self.assertFalse((await self.bus.wait_for_reply(request, 0)).ok)
    finally:
      finish.set()

  async def test_an_intervening_owner_prevents_recovery_even_after_it_released_the_lease(self):
    first, run = await self.prepared()
    child = first.processes["worker"]
    entered, finish, parked_checkpoint = asyncio.Event(), asyncio.Event(), self.bus.parked_checkpoint

    async def delayed(name):
      result = await parked_checkpoint(name)
      if name == run and not entered.is_set():
        entered.set()
        await finish.wait()
      return result

    try:
      with (
        patch.object(self.bus, "parked_checkpoint", side_effect=delayed),
        patch.object(child, "call", wraps=child.call) as call,
      ):
        request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
        await asyncio.wait_for(entered.wait(), 5)
        second = await self.takeover()
        await second.close(graceful=False)
        self.assertIsNone(await self.bus.lease_owner("worker"))
        finish.set()
        [result] = await asyncio.gather(first.tasks["worker"], return_exceptions=True)
        self.assertIsInstance(result, BaseException)
      call.assert_not_called()
      self.assertTrue(child.closed)
      self.assertIsNone(await self.bus.lease_owner("worker"))
      self.assertFalse((await self.bus.wait_for_reply(request, 0)).ok)
    finally:
      finish.set()

  def daemon(self):
    env = {
      **os.environ,
      "REDIS_URL": self.url,
      "LEASE_MS": str(LEASE_MS),
      "GPUS": "1",
      "GPU_MEMORY_MIB": "0",
      "TICK": "0.05",
      "CKPT": self.folder.name,
    }
    proc = subprocess.Popen([sys.executable, "-m", "tinyrl.control.node_daemon"], env=env)
    self.processes.append(proc)
    return proc

  async def until(self, predicate, timeout=60):
    async def wait():
      while not await predicate():
        await asyncio.sleep(0.02)

    await asyncio.wait_for(wait(), timeout)

  async def call(self, run: str, op: Op, body: dict) -> Reply:
    request = await self.bus.submit_command(run, op, json.dumps(body).encode())
    reply = await self.bus.wait_for_reply(request, 60)
    self.assertIsNotNone(reply)
    return reply

  async def test_a_paused_daemon_cannot_answer_after_another_node_takes_its_worker(self):
    require_unix()
    base = f"{self.folder.name}/base"
    tiny().save_pretrained(base)
    run = new_id()
    await self.bus.create_run(Run(run, "worker", base, 4, lora=LORA))
    first = self.daemon()
    scheduling = asyncio.create_task(scheduler.run(self.bus))
    self.tasks.append(scheduling)
    self.assertTrue((await self.call(run, Op.CREATE_MODEL, {"base_model": base, "rank": 4, "lora": LORA})).ok)
    old_owner = await self.bus.lease_owner("worker")
    [first_node] = await self.bus.live_nodes()
    second = self.daemon()

    async def both_up():
      return len(await self.bus.live_nodes()) == 2

    await self.until(both_up)
    await asyncio.sleep(1.5 * LEASE_MS / 1000)
    self.assertEqual(await self.bus.lease_owner("worker"), old_owner)
    self.assertEqual(len(await self.bus.r.xinfo_consumers(key("q", run), GROUP)), 1)

    async def reported():
      node = (await self.bus.live_nodes(details=True)).get(first_node, {})
      return node.get("workers", {}).get("worker", {}).get("pid")

    await self.until(reported)
    model_pid = await reported()
    os.killpg(model_pid, signal.SIGSTOP)
    try:
      request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')

      async def pending():
        return bool((await self.bus.queued_and_in_flight(run))["in_flight"])

      await self.until(pending)
      first.send_signal(signal.SIGSTOP)
      os.killpg(model_pid, signal.SIGCONT)  # the child may run while its parent cannot publish a result
      await self.bus.append_event(Ev.NODE_LOST, node=first_node)
      expected = await self.bus.wait_for_reply(request, 30)
      self.assertFalse(expected.ok)
      self.assertIn("lost the worker", json.loads(expected.body)["error"])
      new_owner = await self.bus.lease_owner("worker")
      self.assertNotEqual(new_owner, old_owner)
    finally:
      first.send_signal(signal.SIGCONT)
      with contextlib.suppress(ProcessLookupError):
        os.killpg(model_pid, signal.SIGCONT)

    async def first_reaped():
      nodes = await self.bus.live_nodes(details=True)
      return first_node in nodes and not nodes[first_node]["workers"]

    await self.until(first_reaped)
    self.assertEqual(await self.bus.wait_for_reply(request, 0), expected)
    self.assertEqual((await self.bus.queued_and_in_flight(run))["in_flight"], [])
    other = new_id()
    await self.bus.create_run(Run(other, "worker", base, 4, lora=LORA))
    self.assertTrue((await self.call(other, Op.CREATE_MODEL, {"base_model": base, "rank": 4, "lora": LORA})).ok)
    self.assertIsNone(second.poll())

  async def test_sigterm_parks_runs_before_the_node_releases_ownership(self):
    require_unix()
    base = f"{self.folder.name}/base"
    tiny().save_pretrained(base)
    run = new_id()
    await self.bus.create_run(Run(run, "worker", base, 4, lora=LORA))
    proc = self.daemon()
    self.tasks.append(asyncio.create_task(scheduler.run(self.bus)))
    self.assertTrue((await self.call(run, Op.CREATE_MODEL, {"base_model": base, "rank": 4, "lora": LORA})).ok)
    proc.terminate()
    self.assertEqual(await asyncio.to_thread(proc.wait, 30), 0)
    uri = await self.bus.parked_checkpoint(run)
    self.assertIsNotNone(uri)
    tensors, meta = self.store.get(uri)
    self.assertTrue(tensors)
    self.assertTrue(meta["optimizer"])
    self.assertIsNotNone(await self.registry.get_checkpoint(uri))
    self.assertIsNone(await self.bus.lease_owner("worker"))


if __name__ == "__main__":
  unittest.main()
