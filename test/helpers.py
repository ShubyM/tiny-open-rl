import asyncio
import os
import subprocess
import tempfile
import time
import unittest
import uuid

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tinyrl.bus import Bus, connect
from tinyrl.storage.registry import Registry


def tiny(name: str = "") -> LlamaForCausalLM:
  """A two layer Llama with random weights: the same every time, and nothing to download."""
  torch.manual_seed(0)
  config = LlamaConfig(
    vocab_size=96,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
  )
  return LlamaForCausalLM(config)


def new_id() -> str:
  return str(uuid.uuid4())


def start_redis(socket_path: str) -> subprocess.Popen:
  args = ["--port", "0", "--unixsocket", socket_path, "--save", "", "--appendonly", "no"]
  proc = subprocess.Popen([os.getenv("REDIS_SERVER", "redis-server"), *args], stdout=subprocess.DEVNULL)
  for _ in range(500):
    if os.path.exists(socket_path):
      break
    time.sleep(0.01)
  return proc


class RedisTest(unittest.IsolatedAsyncioTestCase):
  """Starts a throwaway redis-server for the test class, or uses REDIS_URL. Each test gets an empty database."""

  proc: subprocess.Popen | None = None

  @classmethod
  def setUpClass(cls):
    if os.getenv("REDIS_URL"):
      cls.url = os.environ["REDIS_URL"]
      return
    socket_path = f"{tempfile.mkdtemp()}/redis.sock"
    cls.proc = start_redis(socket_path)
    cls.url = f"unix://{socket_path}"

  @classmethod
  def tearDownClass(cls):
    if cls.proc:
      cls.proc.terminate()
      cls.proc.wait(timeout=5)

  async def asyncSetUp(self):
    self.renewal_tasks = []
    self.bus = Bus(connect(self.url))
    self.registry = Registry(self.bus)
    await self.bus.r.flushall()

  async def worker(self, run: str) -> str:
    await assign_new_runs(self.bus)
    return (await self.bus.get_run(run))["worker"]

  async def asyncTearDown(self):
    for task in self.renewal_tasks:
      task.cancel()
    await asyncio.gather(*self.renewal_tasks, return_exceptions=True)
    await self.bus.r.aclose()


async def assign_new_runs(bus) -> None:
  """Assign new runs through the real scheduler decision logic."""
  from tinyrl.control import scheduler

  s = scheduler.from_table(await bus.read_table())
  new, changes, gen_bumps = scheduler.decide(s)
  await scheduler.commit(bus, s, new, changes, gen_bumps)


class FakeEngine:
  """A CPU sampling engine implementing vLLM's sample, adapter load/unload, pause/resume, abort and sleep contracts."""

  def __init__(self):
    self.loaded: set[str] = set()
    self.lora = True
    self.paused = False
    self.sleeping = False
    self.aborted: set[str] = set()
    self.artifact_key = "fake-artifact"
    self.startup_mode = "cached"
    self.startup_s = 0.001

  @classmethod
  async def start(cls, base: str, lora: bool, env: dict[str, str] | None = None) -> "FakeEngine":
    engine = cls()
    engine.lora = lora
    return engine

  async def load_checkpoint(self, name: str, path: str) -> None:
    self.loaded = self.loaded | {name} if self.lora else {name}

  async def unload_checkpoint(self, name: str) -> None:
    self.loaded.discard(name)

  async def pause(self, mode: str = "wait") -> None:
    self.paused = True

  async def resume(self) -> None:
    self.paused = False

  async def abort(self, request_ids: list[str]) -> None:
    self.aborted.update(request_ids)

  async def sleep(self) -> None:
    self.sleeping = True

  async def wake(self) -> None:
    self.sleeping = False

  async def close(self) -> None:
    pass

  async def sample(self, job: dict, name: str | None, request_id: str | None = None):
    from tinyrl.workers.sampler import Out, Seq

    if request_id and request_id in self.aborted:
      raise asyncio.CancelledError
    tokens = [(job["prompt"][-1] + 1 + i) % 32 for i in range(job["max_tokens"])]
    top = [{t: -1.0, 0: -2.0} for t in tokens]
    seqs = [Seq(tokens, [-1.0] * len(tokens), top, False) for _ in range(job["num_samples"])]
    return Out(seqs, [None] + [{t: -1.0, 0: -2.0} for t in job["prompt"][1:]])
