import asyncio
import tempfile
import unittest
import uuid

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tinyrl.bus import Bus, make_checkpoint
from tinyrl.types import TurnResult


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


def finished(turn, reply=None, charge=None, **effects) -> TurnResult:
  """What a node reports when `turn` ends with `reply` for each command it carried."""
  return TurnResult(turn.id, True, reply if turn.cmd else None, charge, **effects)


def new_id() -> str:
  # run ids are uuids, since PEFT finds an adapter's weights by its name
  return str(uuid.uuid4())


async def serve_control(bus: Bus, host: str = "127.0.0.1", port: int = 0):
  """Run the FastAPI gateway (including `/internal/exchange`) on a TCP port for tests."""
  import socket

  import uvicorn

  from tinyrl.api import gateway

  sock = socket.socket()
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  sock.bind((host, port))
  sock.listen(128)
  actual = sock.getsockname()[1]
  server = uvicorn.Server(uvicorn.Config(gateway.app(bus), host=host, port=actual, log_level="error"))
  task = asyncio.create_task(server.serve(sockets=[sock]))
  while not server.started:
    await asyncio.sleep(0.005)

  class Handle:
    async def cleanup(self):
      server.should_exit = True
      await task
      sock.close()

  return Handle(), f"http://{host}:{actual}"


class BusTest(unittest.IsolatedAsyncioTestCase):
  """Starts an isolated SQLite Bus per test."""

  async def seed_checkpoint(self, uri, size, *, ttl=None, owned=False, version=None):
    with self.bus.db:
      self.bus.checkpoints[uri] = make_checkpoint(uri, size, version=version or uuid.uuid4().hex, ttl=ttl, owned=owned)

  async def asyncSetUp(self):
    self.renewal_tasks = []
    self._db_dir = tempfile.TemporaryDirectory()
    self.db_path = f"{self._db_dir.name}/tinyrl.db"
    self.bus = Bus(self.db_path)

  def start_maintenance(self, period: float = 0.02) -> asyncio.Task:
    async def loop():
      while True:
        await self.bus.maintain()
        await asyncio.sleep(period)

    task = asyncio.create_task(loop())
    self.renewal_tasks.append(task)
    return task

  async def worker(self, run: str) -> str:
    """The run's primary process, once every new run is assigned the way the scheduler assigns it."""
    self.bus.reconcile()
    return self.bus.state().procs_of(run)[0]

  async def asyncTearDown(self):
    for task in self.renewal_tasks:
      task.cancel()
    await asyncio.gather(*self.renewal_tasks, return_exceptions=True)
    await self.bus.aclose()
    self._db_dir.cleanup()


class FakeEngine:
  """A CPU sampling engine: tokens count up from the last prompt token, with logprob -1."""

  def __init__(self):
    self.loaded: set[str] = set()
    self.lora = True

  @classmethod
  async def start(cls, base: str, lora: bool, env: dict[str, str] | None = None) -> "FakeEngine":
    engine = cls()
    engine.lora = lora
    return engine

  async def load_checkpoint(self, name: str, path: str) -> None:
    self.loaded = self.loaded | {name} if self.lora else {name}

  async def sleep(self) -> None:
    pass

  async def wake(self) -> None:
    pass

  async def close(self) -> None:
    pass

  async def sample(self, job: dict, name: str | None):
    from tinyrl.workers.sampler import Out, Seq

    tokens = [(job["prompt"][-1] + 1 + i) % 32 for i in range(job["max_tokens"])]
    top = [{t: -1.0, 0: -2.0} for t in tokens]
    seqs = [Seq(tokens, [-1.0] * len(tokens), top, False) for _ in range(job["num_samples"])]
    return Out(seqs, [None] + [{t: -1.0, 0: -2.0} for t in job["prompt"][1:]])
