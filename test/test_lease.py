import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest

from test.helpers import RedisTest, new_id, tiny
from test.test_process import require_unix
from tinyrl.bus import Op, Reply, Run, key, unpack_snap
from tinyrl.control import scheduler
from tinyrl.storage.store import Store

LEASE_MS = 2000
LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


class TestLease(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.tasks, self.processes = [], []
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
    await super().asyncTearDown()

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

  async def test_a_paused_daemon_cannot_answer_after_another_node_takes_its_run(self):
    require_unix()
    base = f"{self.folder.name}/base"
    tiny().save_pretrained(base)
    run = new_id()
    await self.bus.create_run(Run(run, base, 4, lora=LORA))
    first = self.daemon()
    self.tasks.append(asyncio.create_task(scheduler.run(self.bus)))
    self.assertTrue((await self.call(run, Op.CREATE_MODEL, {"base_model": base, "rank": 4, "lora": LORA})).ok)
    old_owner = (await self.bus.get_run(run))["owner"]
    proc_name = old_owner.split("@", 1)[0]
    [first_node] = await self.bus.live_nodes()
    second = self.daemon()

    async def both_up():
      return len(await self.bus.live_nodes()) == 2

    await self.until(both_up)
    await asyncio.sleep(1.5 * LEASE_MS / 1000)
    self.assertEqual((await self.bus.get_run(run))["owner"], old_owner)
    self.assertEqual((await self.bus.queued_and_in_flight(run))["in_flight"], [])

    async def reported():
      node = (await self.bus.live_nodes(details=True)).get(first_node, {})
      return node.get("pids", {}).get(proc_name)

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
      await self.bus.r.hdel(key("nodes"), first_node)  # its heartbeat ran out
      expected = await self.bus.wait_for_reply(request, 30)
      self.assertFalse(expected.ok)
      self.assertIn("lost ownership", json.loads(expected.body)["error"])
      new_owner = (await self.bus.get_run(run))["owner"]
      self.assertNotEqual(new_owner, old_owner)
    finally:
      first.send_signal(signal.SIGCONT)
      with contextlib.suppress(ProcessLookupError):
        os.killpg(model_pid, signal.SIGCONT)

    async def first_reaped():
      nodes = await self.bus.live_nodes(details=True)
      return first_node in nodes and not nodes[first_node]["pids"] and not nodes[first_node]["on"]

    await self.until(first_reaped)
    self.assertEqual(await self.bus.wait_for_reply(request, 0), expected)
    self.assertEqual((await self.bus.queued_and_in_flight(run))["in_flight"], [])
    other = new_id()
    await self.bus.create_run(Run(other, base, 4, lora=LORA))
    self.assertTrue((await self.call(other, Op.CREATE_MODEL, {"base_model": base, "rank": 4, "lora": LORA})).ok)
    self.assertIsNone(second.poll())

  async def test_sigterm_detaches_runs_before_the_node_releases_ownership(self):
    require_unix()
    base = f"{self.folder.name}/base"
    tiny().save_pretrained(base)
    run = new_id()
    await self.bus.create_run(Run(run, base, 4, lora=LORA))
    proc = self.daemon()
    self.tasks.append(asyncio.create_task(scheduler.run(self.bus)))
    self.assertTrue((await self.call(run, Op.CREATE_MODEL, {"base_model": base, "rank": 4, "lora": LORA})).ok)
    proc.terminate()
    self.assertEqual(await asyncio.to_thread(proc.wait, 30), 0)
    info = await self.bus.get_run(run)
    self.assertIsNone(info.get("owner"))
    uri, version = unpack_snap(info["snap"])
    tensors, meta = self.store.get(version)
    self.assertTrue(tensors)
    self.assertTrue(meta["optimizer"])
    self.assertIsNotNone(await self.bus.get_checkpoint(uri))


if __name__ == "__main__":
  unittest.main()
