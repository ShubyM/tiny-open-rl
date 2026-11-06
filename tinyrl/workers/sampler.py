"""The node's sampling driver. It loads staged checkpoints into an engine; the node owns storage, queues and GPU turns.

ENGINE names the engine's class, vLLM (tinyrl/workers/vllm.py) unless set. Engines own loaded checkpoint names and make load_checkpoint idempotent.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
from dataclasses import dataclass

import numpy as np

from tinyrl.api import wire
from tinyrl.bus import Cmd, Reply

MASK_LOGPROB = -99999.0  # the SDK's logprob for a top-k slot that wasn't scored


@dataclass(frozen=True)
class Seq:
  tokens: list[int]
  logprobs: list[float]
  top: list[dict[int, float]]  # per token: the top-k alternatives and the token itself
  stop: bool


@dataclass(frozen=True)
class Out:
  seqs: list[Seq]
  prompt: list[dict[int, float] | None]  # per prompt token, like Seq.top. Empty unless prompt logprobs were asked for


def topk(rows: list[dict[int, float] | None], k: int) -> tuple[np.ndarray, np.ndarray]:
  pairs = [sorted(r.items(), key=lambda kv: -kv[1])[:k] if r else [] for r in rows]
  pairs = [p + [(0, MASK_LOGPROB)] * (k - len(p)) for p in pairs]
  ids = np.array([[t for t, _ in p] for p in pairs], np.int32).reshape(len(rows), k)
  return ids, np.array([[lp for _, lp in p] for p in pairs], np.float32).reshape(len(rows), k)


def encode(job: dict, out: Out) -> bytes:
  """The SampleResponse protobuf for an engine's output, with prompt rows before prompt_logprobs_last_n unscored."""
  prompt, last_n = job["prompt"], job["prompt_logprobs_last_n"]
  rows = [r if i >= (len(prompt) - last_n if last_n else 1) else None for i, r in enumerate(out.prompt)]
  prompt_lp = (
    np.array([r[t] if r else np.nan for r, t in zip(rows, prompt)], np.float32) if job["prompt_logprobs"] else None
  )
  k = job["topk_sample"]
  seqs = [
    wire.SampleOut(
      s.stop, np.array(s.tokens, np.int32), np.array(s.logprobs, np.float32), topk(s.top, k) if k else None
    )
    for s in out.seqs
  ]
  return wire.sample_output(seqs, prompt_lp, topk(rows, job["topk_prompt"]) if job["topk_prompt"] else None)


class Sampler:
  def __init__(self, base: str, lora: bool, env=None, engine=None):
    self.env = dict(os.environ if env is None else env)
    module, name = self.env.get("ENGINE", "tinyrl.workers.vllm:VLLM").split(":")
    self.engine_type = engine or getattr(importlib.import_module(module), name)
    self.base, self.lora = base, lora
    self.running = None
    self.loading = asyncio.Lock()

  async def start(self) -> None:
    """Start once; the node excludes this boot time from the worker's first quantum."""
    async with self.loading:
      if self.running is None:
        self.running = await self.engine_type.start(self.base, self.lora, self.env)

  async def execute(self, cmd: Cmd, run: dict | None, checkpoint: tuple[str, str] | None) -> Reply:
    try:
      if run is None:
        raise ValueError(f"no sampling session {cmd.run}")
      job, uri, name = json.loads(cmd.body), run.get("model_path"), None
      await self.start()
      async with self.loading:
        if checkpoint:
          path, version = checkpoint
          name = hashlib.sha256(f"{uri}\0{version}".encode()).hexdigest()
          await self.running.load_checkpoint(name, path)
      return Reply(True, wire.PROTO, encode(job, await self.running.sample(job, name)))
    except Exception as e:
      return Reply.error(e)

  async def resume(self, gpu: int | None = None) -> None:
    if self.running:
      await self.running.wake()

  async def park(self) -> None:
    if self.running:
      await self.running.sleep()

  async def close(self) -> None:
    if self.running:
      await self.running.close()
      self.running = None
