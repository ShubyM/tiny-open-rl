"""The node's sampling driver. It loads staged checkpoints into an engine; the node owns storage, queues and GPU turns.

ENGINE names the engine's class, vLLM (tinyrl/workers/vllm.py) unless set. Engines own loaded checkpoint names and make load_checkpoint idempotent.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib
import inspect
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
  top: list[dict[int, float]]
  stop: bool


@dataclass(frozen=True)
class Out:
  seqs: list[Seq]
  prompt: list[dict[int, float] | None]


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
    self.paused = self.parked = False
    self.loading = asyncio.Lock()
    self.inflight: dict[str, dict[str, asyncio.Task]] = {}
    self.adapters: dict[str, str] = {}
    self.fenced: set[str] = set()

  @property
  def proc(self):
    return getattr(self.running, "server", None)

  def startup_info(self) -> dict:
    if not self.running:
      return {}
    return {
      k: getattr(self.running, k)
      for k in ("artifact_key", "startup_mode", "startup_s")
      if getattr(self.running, k, None) is not None
    }

  async def start(self) -> None:
    async with self.loading:
      if self.running is None:
        self.running = await self.engine_type.start(self.base, self.lora, self.env)
        self.paused = self.parked = False

  async def load_adapter(self, run: str, uri: str, path: str, version: str) -> str:
    name = hashlib.sha256(f"{uri}\0{version}".encode()).hexdigest()
    async with self.loading:
      await self.running.load_checkpoint(name, path)
      self.adapters[run] = name
    return name

  async def execute(self, cmd: Cmd, run: dict | None, checkpoint: tuple[str, str] | None) -> tuple[Reply, int]:
    req_id, task = cmd.request_id, asyncio.current_task()
    try:
      if cmd.run in self.fenced:
        raise asyncio.CancelledError
      if run is None:
        raise ValueError(f"no sampling session {cmd.run}")
      job, uri, name = json.loads(cmd.body), run.get("model_path"), None
      self.inflight.setdefault(cmd.run, {})[req_id] = task
      await self.start()
      if checkpoint:
        name = await self.load_adapter(cmd.run, uri, *checkpoint)
      if cmd.run in self.fenced:
        raise asyncio.CancelledError
      sample = self.running.sample
      takes_id = "request_id" in inspect.signature(sample).parameters
      out = await (sample(job, name, request_id=req_id) if takes_id else sample(job, name))
      tokens = len(job["prompt"]) + sum(len(s.tokens) for s in out.seqs)
      return Reply(True, wire.PROTO, await asyncio.to_thread(encode, job, out)), tokens
    except asyncio.CancelledError:
      raise
    except Exception as e:
      return Reply.error(e), 0
    finally:
      if (active := self.inflight.get(cmd.run)) and active.pop(req_id, None) and not active:
        self.inflight.pop(cmd.run, None)

  async def cancel_run(self, run: str) -> None:
    """Stop and abort one job on a shared engine without disturbing sibling jobs."""
    self.fenced.add(run)
    active = dict(self.inflight.get(run, {}))
    if self.running and active and hasattr(self.running, "abort"):
      with contextlib.suppress(Exception):
        await self.running.abort(sorted(active))
    for t in active.values():
      if t is not asyncio.current_task() and not t.done():
        t.cancel()
    if active:
      await asyncio.gather(*(t for t in active.values() if t is not asyncio.current_task()), return_exceptions=True)
    if (name := self.adapters.pop(run, None)) and self.lora and name not in self.adapters.values():
      if self.running and hasattr(self.running, "unload_checkpoint"):
        async with self.loading:
          with contextlib.suppress(Exception):
            await self.running.unload_checkpoint(name)

  def unfence_run(self, run: str) -> None:
    self.fenced.discard(run)

  async def pause(self, mode: str = "wait") -> None:
    if self.running and not self.paused:
      if hasattr(self.running, "pause"):
        await self.running.pause(mode)
      self.paused = True

  async def unpause(self) -> None:
    if self.running and self.paused:
      if hasattr(self.running, "resume"):
        await self.running.resume()
      self.paused = False

  async def resume(self, gpu: int | None = None) -> None:
    if self.running:
      if self.parked:
        await self.running.wake()
        self.parked = False
      await self.unpause()

  async def park(self, mode: str = "wait") -> None:
    if self.running and not self.parked:
      await self.pause(mode)
      await self.running.sleep()
      self.parked = True

  async def close(self) -> None:
    for run in list(self.inflight):
      await self.cancel_run(run)
    if self.running:
      await self.running.close()
      self.running = None
      self.paused = self.parked = False
