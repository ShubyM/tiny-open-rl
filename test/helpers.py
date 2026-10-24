import os
import subprocess
import tempfile
import time
import unittest
import uuid

import redis.asyncio as redis
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tinyrl.bus import Bus


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
    self.bus = Bus(redis.from_url(self.url))
    await self.bus.r.flushall()

  async def asyncTearDown(self):
    await self.bus.r.aclose()
