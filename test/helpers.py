import asyncio
import json
import os
import subprocess
import tempfile
import time
import unittest
import uuid

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tinyrl.bus import Bus, connect, key, write_checkpoint
from tinyrl.storage.registry import Registry
from tinyrl.storage.store import split_version


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
  # run ids are uuids, since PEFT finds an adapter's weights by its name
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

  async def asyncSetUp(self):
    self.renewal_tasks = []
    self.bus = Bus(connect(self.url))
    self.registry = Registry(self.bus)
    await self.bus.r.flushall()

  async def seed_checkpoint(self, uri, size=0, ttl=None, owned=False, *, version):
    """Seed an existing checkpoint before a scenario; runtime publication is tested through write_reply."""
    p = self.bus.r.pipeline()
    p.set(
      key("ver", split_version(version)[0]),
      json.dumps({"uri": uri, "version": version, "time": time.time(), "state": "live"}),
    )
    write_checkpoint(p, {"uri": uri, "size": size, "ttl": ttl, "owned": owned, "version": version})
    await p.execute()

  async def worker(self, run: str) -> str:
    """The run's worker, once every new run is assigned the way the scheduler assigns it."""
    await assign_new_runs(self.bus)
    return (await self.bus.get_run(run))["worker"]

  async def asyncTearDown(self):
    for task in self.renewal_tasks:
      task.cancel()
    await asyncio.gather(*self.renewal_tasks, return_exceptions=True)
    await self.bus.r.aclose()


async def assign_new_runs(bus) -> None:
  """Assign new runs the way the scheduler would, without placing anything."""
  from tinyrl.control import scheduler

  s = scheduler.from_table(await bus.read_table())
  await scheduler.commit(bus, s, scheduler.decide(s))


class FakeEngine:
  """A CPU sampling engine: tokens count up from the last prompt token, with logprob -1."""

  def __init__(self):
    self.loaded: set[str] = set()
    self.lora = True
    self.active = {}

  @classmethod
  async def start(cls, base: str, lora: bool, env: dict[str, str] | None = None) -> "FakeEngine":
    engine = cls()
    engine.lora = lora
    return engine

  async def load_checkpoint(self, name: str, path: str) -> None:
    self.loaded = self.loaded | {name} if self.lora else {name}

  async def unload_checkpoint(self, name: str) -> None:
    if self.lora:
      self.loaded.discard(name)

  async def abort(self, request_ids: list[str]) -> None:
    tasks = [self.active[r] for r in request_ids if r in self.active]
    for task in tasks:
      task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

  async def sleep(self) -> None:
    pass

  async def wake(self) -> None:
    pass

  async def close(self) -> None:
    await self.abort(list(self.active))

  async def sample(self, job: dict, name: str | None, request_id: str):
    from tinyrl.workers.sampler import Out, Seq

    self.active[request_id] = asyncio.current_task()
    try:
      await asyncio.sleep(0)
      tokens = [(job["prompt"][-1] + 1 + i) % 32 for i in range(job["max_tokens"])]
      top = [{t: -1.0, 0: -2.0} for t in tokens]
      seqs = [Seq(tokens, [-1.0] * len(tokens), top, False) for _ in range(job["num_samples"])]
      return Out(seqs, [None] + [{t: -1.0, 0: -2.0} for t in job["prompt"][1:]])
    finally:
      self.active.pop(request_id, None)
