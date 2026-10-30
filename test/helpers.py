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
    await self.bus.r.flushall()

  async def own(self, trainer):
    """Renew the lease as Trainer.serve does."""
    await trainer.own()
    self.renewal_tasks.append(asyncio.create_task(trainer.keep()))

  async def pool(self, run: str) -> str:
    """The run's pool, once every new run is assigned the way the scheduler assigns it."""
    from tinyrl import scheduler

    s = scheduler.State()
    for e in await self.bus.events():
      s = scheduler.fold(s, e)
    for kind, body in scheduler.decide(s, 0.0):
      if kind == "assigned":
        await self.bus.assign(body["run"], body["pool"])
    return (await self.bus.run(run))["pool"]

  async def running(self, placed: dict[str, list[int]]) -> None:
    """Node n is up with these pools placed on these GPUs and their processes started, each kept by a held run."""
    gpus = 1 + max(i for pool_gpus in placed.values() for i in pool_gpus)
    await self.bus.heartbeat("n", gpus)
    await self.bus.emit("node_up", node="n", gpus=gpus, memory=0)
    for pool, pool_gpus in placed.items():
      await self.bus.emit("run_created", run=f"{pool}-held", pool=pool)
      await self.bus.emit("placed", pool=pool, node="n", gpus=pool_gpus, memory=0, env={}, rule="test")
      await self.bus.emit("started", pool=pool, node="n")

  async def run_scheduler(self, placed: dict[str, list[int]], quantum: float = 30.0) -> None:
    """Run the scheduler over these running pools, so workers that share a GPU get turns."""
    from tinyrl import scheduler

    await self.running(placed)
    self.renewal_tasks.append(asyncio.create_task(scheduler.run(self.bus, quantum)))

  async def asyncTearDown(self):
    for task in self.renewal_tasks:
      task.cancel()
    await self.bus.r.aclose()


class FakeEngine:
  """A sampling engine that needs no GPU: each sample counts up from the last prompt token, every logprob is -1, and
  `calls` records what the sampler asked of it."""

  def __init__(self):
    self.calls = []

  @classmethod
  async def start(cls, base: str, lora: bool) -> "FakeEngine":
    engine = cls()
    engine.calls.append(("start", base, lora))
    return engine

  async def load(self, name: str, path: str) -> None:
    self.calls.append(("load", name, path))

  async def sleep(self) -> None:
    self.calls.append(("sleep",))

  async def wake(self) -> None:
    self.calls.append(("wake",))

  async def close(self) -> None:
    self.calls.append(("close",))

  async def sample(self, job: dict, name: str | None):
    from tinyrl.sampler import Out, Seq

    self.calls.append(("sample", name))
    tokens = [(job["prompt"][-1] + 1 + i) % 32 for i in range(job["max_tokens"])]
    top = [{t: -1.0, 0: -2.0} for t in tokens]
    seqs = [Seq(tokens, [-1.0] * len(tokens), top, False) for _ in range(job["num_samples"])]
    return Out(seqs, [None] + [{t: -1.0, 0: -2.0} for t in job["prompt"][1:]])
