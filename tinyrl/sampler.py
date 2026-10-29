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

GPUS = GPU.split(",") if GPU else []  # the leases of every GPU this sampler runs on, as tensor parallel ranks

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
  async def load(self, name: str, path: str) -> None: ...  # an adapter, or a full fine-tune's weights in place
  async def sample(self, job: dict, name: str | None) -> Out: ...  # name is a loaded checkpoint, or None for the base
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
    self.names: dict[tuple[str, str], str] = {}  # (uri, version) -> the engine's name for that checkpoint
    self.loading = asyncio.Lock()
    self.done, self.seated = False, 0.0  # seated: when this sampler took its GPU's lease, or 0 while unheld
    self.loaded: str | None = None

  async def warm(self) -> None:
    if self.engine is not None:
      return
    base = (await self.bus.warm_samplers()).get(self.pool)
    if not base:
      for run_id in await self.bus.runs(self.pool):
        if (run := await self.bus.run(run_id)) and run.get("base"):
          base = run["base"]
          break
    if not base:
      return
    async with self.acquire():
      async with self.loading:
        if self.engine is None:
          with contextlib.suppress(Exception):
            self.engine = await self.engine_type.start(base, self.lora)
            if self.seated:
              self.seated = time.time()

  async def serve(self) -> None:
    await self.own()
    renewing = asyncio.create_task(self.keep())
    try:
      await self.fail_stranded(0)
      if os.environ.get("TINYRL_WARM_SAMPLER") == "1":
        await self.warm()
      while not self.done:
        await self.step()
        await self.fail_stranded(STRANDED_MS)
    finally:
      renewing.cancel()
      if self.engine:
        await self.engine.close()
      await self.release_gpus(GPUS)
      self.seated = 0.0
      await self.bus.release(self.pool, self.worker)

  async def own(self) -> None:
    while not await self.bus.own(self.pool, self.worker):
      await asyncio.sleep(LEASE_MS / 4000)

  async def keep(self) -> None:
    while not self.done:
      await asyncio.sleep(LEASE_MS / 4000)
      if not await self.bus.renew(self.pool, self.worker) and not await self.bus.own(self.pool, self.worker):
        self.done = (
          True  # another sampler took the pool. An expired lease nobody took, as after a Redis restart, is taken again
        )
        return
      if self.seated and not await self.renew_gpus(GPUS):
        self.seated = 0.0

  async def ensure_engine(self, session: str) -> None:
    if self.engine is None:
      run = await self.bus.run(session)
      if run:
        async with self.loading:
          if self.engine is None:
            self.engine = await self.engine_type.start(run["base"], self.lora)
            if self.seated:
              self.seated = time.time()

  async def fail_stranded(self, min_idle_ms: int) -> None:
    for cmd in await self.bus.reclaim(self.pool, self.worker, min_idle_ms):
      await self.bus.finish(cmd, fail(RuntimeError("its sampler died, or lost the pool")), self.pool, self.worker)

  async def step(self, count: int = 128, block_ms: int = 1000) -> int:
    if self.done:
      return 0
    take_block = 150 if self.seated and block_ms > 150 else block_ms
    cmds = await self.bus.take(self.pool, self.worker, count, take_block)
    if not cmds:
      await self.yield_gpu(idle=True)
      return 0

    async def one(cmd) -> None:
      start = time.time()
      first = await self.bus.first_reply(cmd)
      reply = first or await self.run(cmd.run, json.loads(cmd.body))
      if not await self.bus.finish(cmd, reply, self.pool, self.worker, repeat=first is not None, started=start):
        self.done = True

    async with self.acquire():
      if self.engine is None:
        with contextlib.suppress(Exception):
          await self.ensure_engine(cmds[0].run)
      if block_ms:
        await asyncio.sleep(0.02)
        cmds += await self.bus.take(self.pool, self.worker, count, 0)
      if self.lora:
        batches = [cmds]
      else:
        by_run: dict[str, list] = {}
        [by_run.setdefault(c.run, []).append(c) for c in cmds]
        batches = list(by_run.values())
      for batch in batches:
        if not self.done:
          await asyncio.gather(*(one(c) for c in batch))
    return len(cmds)

  async def run(self, session: str, job: dict) -> Reply:
    try:
      run = await self.bus.run(session)
      if run is None:
        raise ValueError(f"no sampling session {session}")
      async with self.loading:
        if self.engine is None:
          self.engine = await self.engine_type.start(run["base"], self.lora)
        name = await self.load(run["model_path"])
      return Reply(True, wire.PROTO, encode(job, await self.engine.sample(job, name)))
    except Exception as e:
      return fail(e)

  async def load(self, uri: str | None) -> str | None:
    """The engine's name for a checkpoint, which the engine loads unless it already holds it."""
    if not uri:
      return None
    path, version = await asyncio.to_thread(self.store.local, uri)  # a checkpoint name can be saved again
    if (uri, version) not in self.names:
      name = f"c{len(self.names)}"
      await self.engine.load(name, path)
      self.names[(uri, version)] = name
      self.loaded = name
    elif not self.lora and self.loaded != self.names[(uri, version)]:
      await self.engine.load(self.names[(uri, version)], path)
      self.loaded = self.names[(uri, version)]
    return self.names[(uri, version)]

  @contextlib.asynccontextmanager
  async def acquire(self):
    if GPUS and not self.seated:
      held: list[str] = []
      try:  # take every GPU in index order, so two samplers that share GPUs never wait on each other
        while len(held) < len(GPUS):
          gpu = GPUS[len(held)]
          if await self.bus.take_gpu(gpu, self.worker):
            held.append(gpu)
            continue
          if not await self.renew_gpus(held):  # one expired while it waited for the next: start over
            await self.release_gpus(held)
            held = []
          await asyncio.sleep(0.05)
      finally:
        for gpu in GPUS:
          await self.bus.withdraw_request(gpu, self.worker)
      if self.engine:
        await self.engine.wake()
      self.seated = time.time()
    try:
      yield
    finally:
      await self.yield_gpu(idle=False)

  async def yield_gpu(self, idle: bool) -> None:
    held = self.seated and (idle or time.time() - self.seated >= HOLD_S)
    if held and any([await self.bus.others_requested(gpu, self.worker) for gpu in GPUS]):
      if self.engine:
        await self.engine.sleep()
      await self.release_gpus(GPUS)
      self.seated = 0.0

  async def renew_gpus(self, gpus: list[str]) -> bool:
    return all([await self.bus.renew(f"gpu:{gpu}", self.worker) for gpu in gpus])

  async def release_gpus(self, gpus: list[str]) -> None:
    for gpu in gpus:
      await self.bus.release(f"gpu:{gpu}", self.worker)


if __name__ == "__main__":
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # so serve's finally closes the engine
  asyncio.run(Sampler(Bus(), sys.argv[1]).serve())
