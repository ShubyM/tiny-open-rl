import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest

from test.helpers import RedisTest, new_id, tiny
from tinyrl.bus import GROUP, Op, Reply, Run, key

LEASE_MS = 2000
LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


class TestLease(RedisTest):
  """Two trainer processes and one worker. The lease decides which one serves it, even after the first comes back from a
  pause long enough for the second to take over."""

  def trainer(self, worker: str) -> subprocess.Popen:
    env = {**os.environ, "REDIS_URL": self.url, "LEASE_MS": str(LEASE_MS)}
    return subprocess.Popen([sys.executable, "-m", "tinyrl.workers.trainer", worker], env=env)

  async def call(self, run: str, op: Op, body: dict) -> Reply:
    request_id = await self.bus.submit(run, op, json.dumps(body).encode())
    return await self.bus.result(request_id, 60)

  async def test_a_paused_trainer_cannot_answer_once_another_owns_its_worker(self):
    base = f"{tempfile.mkdtemp()}/tiny"
    tiny().save_pretrained(base)
    worker, run = f"lora:{base}", new_id()
    await self.bus.create(Run(run, worker, base, 4))

    first = self.trainer(worker)
    second = None
    try:
      self.assertTrue((await self.call(run, Op.CREATE_MODEL, {"base_model": base, "rank": 4, "lora": LORA})).ok)

      second = self.trainer(worker)  # it starts while the first owns the worker, so it waits
      await asyncio.sleep(3 * LEASE_MS / 1000)
      self.assertTrue((await self.call(run, Op.OPTIM_STEP, {"adam_params": {}})).ok)
      consumers = await self.bus.r.xinfo_consumers(key("q", run), GROUP)
      self.assertEqual(len(consumers), 1)  # the second never read from the worker

      # pause the first while it's blocked reading the worker, so Redis hands it the next command, and past its lease
      while not any(c["cmd"] == "xreadgroup" and "b" in c["flags"] for c in await self.bus.r.client_list()):
        await asyncio.sleep(0.01)
      first.send_signal(signal.SIGSTOP)
      request_id = await self.bus.submit(run, Op.OPTIM_STEP, b'{"adam_params": {}}')
      await asyncio.sleep(0.2)
      [stranded] = (await self.bus.queue(run))["in_flight"]
      self.assertEqual(stranded["request_id"], request_id)

      reply = await self.bus.result(request_id, 30)  # the second takes the worker over and fails the command
      self.assertFalse(reply.ok)
      self.assertIn("lost the worker", json.loads(reply.body)["error"])

      first.send_signal(signal.SIGCONT)
      await asyncio.to_thread(first.wait, 30)  # it finds its lease gone, and exits
      self.assertEqual(await self.bus.result(request_id, 0), reply)  # and nothing it did reached the client
      self.assertEqual((await self.bus.queue(run))["in_flight"], [])

      other = new_id()
      await self.bus.create(Run(other, worker, base, 4))
      self.assertTrue((await self.call(other, Op.CREATE_MODEL, {"base_model": base, "rank": 4, "lora": LORA})).ok)
      self.assertIsNone(second.poll())  # the second serves the worker from now on
    finally:
      for proc in (first, second):
        if proc is not None and proc.poll() is None:
          proc.send_signal(signal.SIGCONT)
          proc.terminate()
          proc.wait()

  async def test_sigterm_saves_the_runs_before_the_trainer_exits(self):
    base, ckpt = f"{tempfile.mkdtemp()}/tiny", tempfile.mkdtemp()
    tiny().save_pretrained(base)
    worker, run = f"lora:{base}", new_id()
    await self.bus.create(Run(run, worker, base, 4, lora=LORA))
    env = {**os.environ, "REDIS_URL": self.url, "LEASE_MS": str(LEASE_MS), "CKPT": ckpt}
    proc = subprocess.Popen([sys.executable, "-m", "tinyrl.workers.trainer", worker], env=env)
    try:
      self.assertTrue((await self.call(run, Op.CREATE_MODEL, {"base_model": base, "rank": 4, "lora": LORA})).ok)
      proc.terminate()
      self.assertEqual(await asyncio.to_thread(proc.wait, 30), 0)
      uri = await self.bus.parked(run)
      self.assertTrue(os.path.exists(f"{ckpt}/{uri.removeprefix('tinker://')}/state.safetensors"))
      self.assertIsNone(await self.bus.owner(worker))  # released, so the next trainer starts at once
    finally:
      if proc.poll() is None:
        proc.kill()


if __name__ == "__main__":
  unittest.main()
