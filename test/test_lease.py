import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test.execution import TestNode, gate
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
    lw = first.local["worker"]
    await self.bus.r.delete(key("owner", "worker"))
    self.assertTrue(await self.bus.take_lease("worker", "new-owner"))
    with patch.object(lw.engine.process, "call", wraps=lw.engine.process.call) as call:
      request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
      await asyncio.wait_for(asyncio.gather(lw.task, return_exceptions=True), 10)
    self.assertNotIn("execute", [c.args[0] for c in call.call_args_list])
    self.assertTrue(lw.lease.lost)
    self.assertEqual(await self.bus.queued_and_in_flight(run), {"live": True, "queued": 1, "in_flight": []})
    self.assertIsNone(await self.bus.wait_for_reply(request, 0))
    cmd = Cmd(run, request.split("/")[1], Op.OPTIM_STEP, b"{}")
    self.assertFalse(await self.bus.write_reply(cmd, Reply(True, "application/json", b"{}"), "worker", lw.lease.owner))
    self.assertEqual(await self.bus.lease_owner("worker"), "new-owner")

  async def test_a_renewal_racing_with_close_cannot_reacquire_the_released_lease(self):
    first, _ = await self.prepared()
    lw = first.local["worker"]
    lw.lease.renewed = 0
    entered, finish = asyncio.Event(), asyncio.Event()
    renew = self.bus.renew_lease

    async def delayed(*args):
      entered.set()
      await finish.wait()
      return await renew(*args)

    with patch.object(self.bus, "renew_lease", side_effect=delayed):
      renewing = asyncio.create_task(first.renew())
      self.tasks.append(renewing)
      await entered.wait()
      lw.cancel()
      await asyncio.gather(lw.task, return_exceptions=True)
      self.assertIsNone(await self.bus.lease_owner("worker"))
      finish.set()
      await renewing
    self.assertIsNone(await self.bus.lease_owner("worker"))
    self.assertTrue(lw.engine.process.closed)

  async def test_a_completed_stale_result_cannot_replace_the_takeover_failure(self):
    first, run = await self.prepared()
    lw = first.local["worker"]
    entered, finish = gate(lw.engine.process, "execute", after=True)
    try:
      request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
      await asyncio.wait_for(entered.wait(), 5)
      second = await self.takeover()
      await self.bus.fail_stranded_commands([run], "worker", second.local["worker"].lease.owner)
      expected = await self.bus.wait_for_reply(request, 0)
      self.assertFalse(expected.ok)
      finish.set()
      await asyncio.wait_for(asyncio.gather(lw.task, return_exceptions=True), 10)
      self.assertIn("lost worker", repr(lw.task.exception()))
      self.assertEqual(await self.bus.wait_for_reply(request, 0), expected)
      self.assertEqual((await self.bus.queued_and_in_flight(run))["in_flight"], [])
      self.assertNotIn(run, second.local["worker"].engine.process.trainer.model.params)
      self.assertEqual((await self.bus.get_run(run))["steps"], 0)
    finally:
      finish.set()

  async def test_a_stale_checkpoint_result_is_not_registered(self):
    first, run = await self.prepared()
    lw = first.local["worker"]
    uri = f"tinker://{run}/weights/stale"
    entered, finish = gate(lw.engine.process, "execute", after=True)
    try:
      request = await self.bus.submit_command(run, Op.SAVE_WEIGHTS, json.dumps({"uri": uri}).encode())
      await asyncio.wait_for(entered.wait(), 5)
      second = await self.takeover()
      await self.bus.fail_stranded_commands([run], "worker", second.local["worker"].lease.owner)
      finish.set()
      await asyncio.wait_for(asyncio.gather(lw.task, return_exceptions=True), 10)
      self.assertIn("lost worker", repr(lw.task.exception()))
      self.assertIsNone(await self.registry.get_checkpoint(uri))
      self.assertFalse((await self.bus.wait_for_reply(request, 0)).ok)
    finally:
      finish.set()

  async def test_an_intervening_owner_prevents_recovery_even_after_it_released_the_lease(self):
    first, run = await self.prepared()
    lw = first.local["worker"]
    await self.bus.r.delete(key("owner", "worker"))
    self.assertTrue(await self.bus.take_lease("worker", "new-owner"))
    await self.bus.release_lease("worker", "new-owner")
    self.assertIsNone(await self.bus.lease_owner("worker"))
    with patch.object(lw.engine.process, "call", wraps=lw.engine.process.call) as call:
      await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
      await asyncio.wait_for(asyncio.gather(lw.task, return_exceptions=True), 10)
    self.assertNotIn("execute", [c.args[0] for c in call.call_args_list])
    self.assertTrue(lw.lease.lost)
    self.assertIsNone(await self.bus.lease_owner("worker"))

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
    first.send_signal(signal.SIGSTOP)
    request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
    await self.bus.append_event(Ev.NODE_LOST, node=first_node)  # what the heartbeat timeout decides
    expected = await self.bus.wait_for_reply(request, 30)
    self.assertFalse(expected.ok)
    self.assertIn("lost its unsaved steps", json.loads(expected.body)["error"])
    new_owner = await self.bus.lease_owner("worker")
    self.assertNotEqual(new_owner, old_owner)
    first.send_signal(signal.SIGCONT)

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
