import tempfile
import unittest
import uuid

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


class CoordinatorTest(unittest.IsolatedAsyncioTestCase):
  """Each test owns a real coordinator and a private SQLite journal."""

  async def asyncSetUp(self):
    self.tmpdir = tempfile.TemporaryDirectory()
    self.path = f"{self.tmpdir.name}/journal.sqlite3"
    self.bus = Bus(self.path)

  async def asyncTearDown(self):
    await self.bus.close()
    self.tmpdir.cleanup()
