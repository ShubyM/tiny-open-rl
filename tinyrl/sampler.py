"""Sampling. A sampler serves a sampling pool the way a trainer serves a training pool: it holds the pool's lease, runs
its commands in order, and takes turns on the GPU.

It samples through an Engine, which knows nothing of Redis, pools or checkpoint URIs. ENGINE names the engine's class,
vLLM (tinyrl/vllm.py) unless it's set.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tinyrl import wire
from tinyrl.bus import LEASE_MS, Bus, Reply
from tinyrl.helpers import getenv
from tinyrl.store import Store
from tinyrl.trainer import GPU, HOLD_S, STRANDED_MS, fail

ENGINE = getenv("ENGINE", "tinyrl.vllm:VLLM")
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


class Engine(Protocol):
  @classmethod
  async def start(cls, base: str, lora: bool) -> Engine: ...
  async def load_lora(self, name: str, path: str) -> None: ...
  async def load_weights(self, path: str) -> None: ...
  async def sample(self, job: dict, adapter: str | None) -> Out: ...
  async def sleep(self) -> None: ...
  async def wake(self) -> None: ...
  async def close(self) -> None: ...


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
  def __init__(
    self, bus: Bus, pool: str, engine: type[Engine] | None = None, store: Store | None = None, worker: str = ""
  ):
    module, name = ENGINE.split(":")
    self.bus = bus
    self.pool = pool
    self.engine_type = engine or getattr(importlib.import_module(module), name)
    self.engine: Engine | None = None  # started by the first command, which says which base model
    self.store = store if store is not None else Store()
    self.lora = not pool.startswith("sampler:fft:")
    self.worker = worker or f"{socket.gethostname()}-{os.getpid()}-{os.urandom(3).hex()}"
    self.adapters: dict[tuple[str, int], str] = {}
    self.loaded: tuple[str, int] | None = None
    self.done = False
    self.seated = 0.0  # when this sampler took its GPU's lease, or 0 while it doesn't hold it

  async def serve(self) -> None:
    await self.own()
    renewing = asyncio.create_task(self.keep())
    try:
      await self.fail_stranded(0)
      while not self.done:
        await self.step()
        await self.fail_stranded(STRANDED_MS)
    finally:
      renewing.cancel()
      if self.engine:
        await self.engine.close()
      if GPU:
        await self.bus.release(f"gpu:{GPU}", self.worker)
        self.seated = 0.0
      await self.bus.release(self.pool, self.worker)

  async def own(self) -> None:
    while not await self.bus.own(self.pool, self.worker):
      await asyncio.sleep(LEASE_MS / 4000)

  async def keep(self) -> None:
    while not self.done:
      await asyncio.sleep(LEASE_MS / 4000)
      if not await self.bus.renew(self.pool, self.worker):
        self.done = True
        return
      if self.seated and not await self.bus.renew(f"gpu:{GPU}", self.worker):
        self.seated = 0.0

  async def fail_stranded(self, min_idle_ms: int) -> None:
    for cmd in await self.bus.reclaim(self.pool, self.worker, min_idle_ms):
      await self.bus.finish(cmd, fail(RuntimeError("its sampler died, or lost the pool")), self.pool, self.worker)

  async def step(self, count: int = 16, block_ms: int = 1000) -> int:
    if self.done:
      return 0
    cmds = await self.bus.take(self.pool, self.worker, count, block_ms)
    if not cmds:
      await self.yield_gpu(idle=True)
      return 0
    async with self.acquire():
      for cmd in cmds:
        start = time.time()
        first = await self.bus.first_reply(cmd)
        reply = first or await self.run(cmd.run, json.loads(cmd.body))
        if not await self.bus.finish(cmd, reply, self.pool, self.worker, repeat=first is not None, started=start):
          self.done = True
          break
    return len(cmds)

  async def run(self, session: str, job: dict) -> Reply:
    try:
      run = await self.bus.run(session)
      if run is None:
        raise ValueError(f"no sampling session {session}")
      if self.engine is None:
        self.engine = await self.engine_type.start(run["base"], self.lora)
      out = await self.engine.sample(job, await self.load(run["model_path"]))
      return Reply(True, wire.PROTO, encode(job, out))
    except Exception as e:
      return fail(e)

  async def load(self, uri: str | None) -> str | None:
    """Get a checkpoint into the engine: an adapter, whose name is returned, or a full fine-tune's weights."""
    if not uri:
      return None
    key = (uri, os.stat(self.store.file(uri)).st_mtime_ns)  # a checkpoint name can be saved again
    if self.lora:
      if key not in self.adapters:
        name = f"a{len(self.adapters)}"
        await self.engine.load_lora(name, self.store.path(uri))
        self.adapters[key] = name
      return self.adapters[key]
    if key != self.loaded:
      await self.engine.load_weights(self.store.path(uri))
      self.loaded = key
    return None

  @contextlib.asynccontextmanager
  async def acquire(self):
    if GPU and not self.seated:
      try:
        while not await self.bus.own(f"gpu:{GPU}", self.worker):
          await self.bus.request_gpu(GPU, self.worker)
          await asyncio.sleep(0.05)
      finally:
        await self.bus.withdraw_request(GPU, self.worker)
      if self.engine:
        await self.engine.wake()
      self.seated = time.time()
    try:
      yield
    finally:
      await self.yield_gpu(idle=False)

  async def yield_gpu(self, idle: bool) -> None:
    held = self.seated and (idle or time.time() - self.seated >= HOLD_S)
    if held and await self.bus.others_requested(GPU, self.worker):
      if self.engine:
        await self.engine.sleep()
      await self.bus.release(f"gpu:{GPU}", self.worker)
      self.seated = 0.0


if __name__ == "__main__":
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # so serve's finally closes the engine
  asyncio.run(Sampler(Bus(), sys.argv[1]).serve())
