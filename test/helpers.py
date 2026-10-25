import os
import pathlib
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import uuid

import redis
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


class RedisServer:
  """An isolated real Redis process with persistent, synchronously flushed writes."""

  def __init__(self):
    self.tmpdir = tempfile.TemporaryDirectory()
    with socket.socket() as sock:
      sock.bind(("127.0.0.1", 0))
      self.port = sock.getsockname()[1]
    self.url = f"redis://127.0.0.1:{self.port}/0"
    self.client = redis.Redis.from_url(self.url)
    self.proc = None

  def start(self):
    binary = os.getenv("REDIS_SERVER") or shutil.which("redis-server")
    if not binary:
      raise RuntimeError("tests require redis-server; set REDIS_SERVER to its executable")
    self.proc = subprocess.Popen(
      [
        binary,
        "--bind",
        "127.0.0.1",
        "--port",
        str(self.port),
        "--dir",
        self.tmpdir.name,
        "--appendonly",
        "yes",
        "--appendfsync",
        "always",
        "--save",
        "",
        "--logfile",
        f"{self.tmpdir.name}/redis.log",
      ],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and self.proc.poll() is None:
      try:
        if self.client.ping():
          return self
      except redis.ConnectionError:
        time.sleep(0.02)
    log = pathlib.Path(self.tmpdir.name, "redis.log")
    raise RuntimeError(f"Redis did not start: {log.read_text() if log.exists() else binary}")

  def stop(self, *, kill=False):
    if self.proc is not None and self.proc.poll() is None:
      self.proc.kill() if kill else self.proc.terminate()
      try:
        self.proc.wait(timeout=10)
      except subprocess.TimeoutExpired:
        self.proc.kill()
        self.proc.wait()

  def close(self):
    self.stop()
    self.client.close()
    self.tmpdir.cleanup()

  def __enter__(self):
    return self.start()

  def __exit__(self, *_):
    self.close()


class CoordinatorTest(unittest.IsolatedAsyncioTestCase):
  """Each test owns a coordinator namespace in a real, isolated Redis server."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.redis_server = RedisServer()
    cls.addClassCleanup(cls.redis_server.close)
    cls.redis_server.start()

  async def asyncSetUp(self):
    self.tmpdir = tempfile.TemporaryDirectory()
    self.url = self.redis_server.url
    self.namespace = f"test-{uuid.uuid4().hex}"
    self.bus = Bus(url=self.url, namespace=self.namespace)

  async def asyncTearDown(self):
    await self.bus.close()
    keys = list(self.redis_server.client.scan_iter(f"{self.bus.prefix}*"))
    if keys:
      self.redis_server.client.delete(*keys)
    self.tmpdir.cleanup()
