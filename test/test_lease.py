import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from test.helpers import BusTest, new_id, serve_control, tiny
from test.test_process import require_unix
from tinyrl.control.node_daemon import NodeDaemon
from tinyrl.storage.store import Store
from tinyrl.types import Op, Reply, Run

LEASE_MS = 2000
LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


class TestLease(BusTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.runner, self.url = await serve_control(self.bus, host="127.0.0.1", port=0)
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
    await self.runner.cleanup()
    await super().asyncTearDown()

  def daemon(self):
    env = {
      **os.environ,
      "CONTROL_URL": self.url,
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
    request = self.bus.submit_command(run, op, json.dumps(body).encode())
    reply = await self.bus.wait_for_reply(request, 60)
    self.assertIsNotNone(reply)
    return reply

  async def test_a_paused_daemon_cannot_answer_after_another_node_takes_its_run(self):
    require_unix()
    base = f"{self.folder.name}/base"
    tiny().save_pretrained(base)
    run = new_id()
    self.bus.create_run(Run(run, base, 4, lora=LORA))
    first = self.daemon()
    self.start_maintenance()
    self.assertTrue((await self.call(run, Op.CREATE_MODEL, {"base_model": base, "rank": 4, "lora": LORA})).ok)
    old_owner = (self.bus.get_run(run))["owner"]
    proc_name = old_owner[0]
    [first_node] = self.bus.live_nodes()
    second = self.daemon()

    async def both_up():
      return len(self.bus.live_nodes()) == 2

    await self.until(both_up)
    await asyncio.sleep(1.5 * LEASE_MS / 1000)
    self.assertEqual((self.bus.get_run(run))["owner"], old_owner)

    async def reported():
      node = self.bus.nodes.get(first_node)
      return w.pid if node and (w := node.workers.get(proc_name)) else None

    await self.until(reported)
    model_pid = await reported()
    os.killpg(model_pid, signal.SIGSTOP)
    try:
      request = self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')

      async def pending():
        return bool((self.bus.queued_and_in_flight(run))["in_flight"])

      await self.until(pending)
      first.send_signal(signal.SIGSTOP)
      os.killpg(model_pid, signal.SIGCONT)  # the child may run while its parent cannot publish a result
      self.bus.nodes[first_node].until = 0.0  # its heartbeat ran out
      expected = await self.bus.wait_for_reply(request, 30)
      self.assertFalse(expected.ok)
      self.assertIn("lost ownership", json.loads(expected.body)["error"])
      self.assertNotEqual((self.bus.get_run(run)).get("owner"), old_owner)
    finally:
      first.send_signal(signal.SIGCONT)
      with contextlib.suppress(ProcessLookupError):
        os.killpg(model_pid, signal.SIGCONT)

    async def first_reaped():
      self.bus.expire_nodes()
      node = self.bus.nodes.get(first_node)
      return node is not None and not node.workers

    await self.until(first_reaped)
    self.assertEqual(await self.bus.wait_for_reply(request, 0), expected)
    self.assertEqual((self.bus.queued_and_in_flight(run))["in_flight"], [])
    other = new_id()
    self.bus.create_run(Run(other, base, 4, lora=LORA))
    self.assertTrue((await self.call(other, Op.CREATE_MODEL, {"base_model": base, "rank": 4, "lora": LORA})).ok)
    self.assertIsNone(second.poll())

  async def test_sigterm_detaches_runs_before_the_node_releases_ownership(self):
    require_unix()
    base = f"{self.folder.name}/base"
    tiny().save_pretrained(base)
    run = new_id()
    self.bus.create_run(Run(run, base, 4, lora=LORA))
    proc = self.daemon()
    self.start_maintenance()
    self.assertTrue((await self.call(run, Op.CREATE_MODEL, {"base_model": base, "rank": 4, "lora": LORA})).ok)
    proc.terminate()
    self.assertEqual(await asyncio.to_thread(proc.wait, 30), 0)
    info = self.bus.get_run(run)
    self.assertIsNone(info.get("owner"))
    version = info["snapshot"]
    tensors, meta = self.store.get(version)
    self.assertTrue(tensors)
    self.assertTrue(meta["optimizer"])
    self.assertIn(version.partition("@")[0], self.bus.live_versions())

  async def test_silent_open_ws_self_fences_on_lease_expiry(self):
    from aiohttp import WSMsgType

    class SilentWS:
      closed = False

      def __init__(self):
        self.blocker = asyncio.Event()

      async def send_bytes(self, data: bytes) -> None:
        pass

      async def close(self) -> None:
        self.closed = True
        self.blocker.set()

      def __aiter__(self):
        return self

      async def __anext__(self):
        await self.blocker.wait()
        return SimpleNamespace(type=WSMsgType.CLOSE, data=b"")

    ws = SilentWS()
    daemon = NodeDaemon(SimpleNamespace(), gpus=1, node="n1", gpu_mib=0, store=self.store)
    self.addAsyncCleanup(daemon.close)
    with mock.patch("tinyrl.control.node_daemon.LEASE_MS", 120):
      with self.assertRaisesRegex(RuntimeError, "lease expired"):
        await asyncio.wait_for(daemon.serve_socket(ws, 0.03, None), 2.0)


if __name__ == "__main__":
  unittest.main()
