"""Sampling. A sampler serves sampling sessions the way a trainer serves training runs: it holds its worker's lease,
runs their commands in order, and takes turns on the GPU.

It samples through an Engine, which knows nothing of Redis, workers or checkpoint URIs. ENGINE names the engine's class,
vLLM (tinyrl/workers/vllm.py) unless it's set.
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
import traceback
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tinyrl.api import wire
from tinyrl.bus import LEASE_MS, Bus, Reply
from tinyrl.control.turns import IDLE_MS, GpuTurn
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry
from tinyrl.storage.store import Store

STRANDED_MS = 5000
GPU = getenv("GPU", "")  # the GPUs this sampler takes turns on, set by the node daemon. Unset, it never takes turns


def fail(error: Exception) -> Reply:
  category = "user" if isinstance(error, (ValueError, KeyError)) else "server"
  body = {"error": f"{type(error).__name__}: {error}", "category": category}
  return Reply(False, wire.JSON, json.dumps(body).encode())


ENGINE = getenv("ENGINE", "tinyrl.workers.vllm:VLLM")
WARM = getenv("TINYRL_WARM_SAMPLER", 0)  # start the engine before the first command, from the scheduler's placement
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
  async def load_checkpoint(
    self, name: str, path: str
  ) -> None: ...  # an adapter, or a full fine-tune's weights in place
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
    self, bus: Bus, worker: str, engine: type[Engine] | None = None, store: Store | None = None, owner: str = ""
  ):
    module, name = ENGINE.split(":")
    self.bus = bus
    self.registry = Registry(bus)
    self.worker = worker
    self.engine_type = engine or getattr(importlib.import_module(module), name)
    self.engine: Engine | None = None  # started by the first command, which says which base model
    self.store = store if store is not None else Store()
    self.lora = os.getenv("TINYRL_LORA", "1") != "0"
    self.owner = owner or f"{socket.gethostname()}-{os.getpid()}-{os.urandom(3).hex()}"
    self.names: dict[tuple[str, str], str] = {}  # (uri, version) -> the engine's name for that checkpoint
    self.loading = asyncio.Lock()
    self.done = False
    self.turn = GpuTurn(bus, worker, bool(GPU))
    self.loaded: str | None = None

  async def warm_engine(self) -> None:
    if self.engine is not None:
      return
    base = os.getenv("TINYRL_BASE")
    if not base:
      return
    async with self.hold_gpu_for_batch(0):
      async with self.loading:
        if self.engine is None:
          try:
            start = time.time()
            self.engine = await self.engine_type.start(base, self.lora)
            boot = time.time() - start
            await self.turn.report_batch_done(queued=0, seconds=boot, boot=boot, ops=[])
          except Exception:  # its first command starts it again and replies with the error
            print(f"{self.worker}: the engine failed to start ahead of its commands", file=sys.stderr, flush=True)
            traceback.print_exc()

  async def serve(self) -> None:
    await self.wait_for_lease()
    renewing = asyncio.create_task(self.keep_renewing_lease())
    try:
      await self.fail_stranded_commands(0)
      if WARM:
        await self.warm_engine()
      while not self.done:
        await self.step()
        await self.fail_stranded_commands(STRANDED_MS)
    finally:
      renewing.cancel()
      if self.engine:
        await self.engine.close()
      await self.turn.give_turn_back()
      await self.bus.release_lease(self.worker, self.owner)

  async def wait_for_lease(self) -> None:
    while not await self.bus.take_lease(self.worker, self.owner):
      await asyncio.sleep(LEASE_MS / 4000)

  async def keep_renewing_lease(self) -> None:
    while not self.done:
      await asyncio.sleep(LEASE_MS / 4000)
      if not await self.bus.renew_lease(self.worker, self.owner) and not await self.bus.take_lease(
        self.worker, self.owner
      ):
        self.done = (
          True  # another sampler took the worker. A lease that expired untaken, as in a Redis restart, is taken again
        )
        return

  async def start_engine_for_session(self, session: str) -> None:
    if self.engine is None:
      run = await self.bus.get_run(session)
      if run:
        async with self.loading:
          if self.engine is None:
            self.engine = await self.engine_type.start(run["base"], self.lora)

  async def fail_stranded_commands(self, min_idle_ms: int) -> None:
    for cmd in await self.bus.reclaim_stranded_commands(self.worker, self.owner, min_idle_ms):
      await self.bus.write_reply(
        cmd, fail(RuntimeError("its sampler died, or lost the worker")), self.worker, self.owner
      )

  async def step(self, count: int = 128, block_ms: int = 1000) -> int:
    if self.done:
      return 0
    take_block = min(block_ms, IDLE_MS) if self.turn.held and GPU else block_ms
    cmds = await self.bus.take_commands(self.worker, self.owner, count, take_block)
    if not cmds:
      await self.turn.report_idle()
      await self.yield_gpu()
      return 0

    async def one(cmd) -> list:
      start = time.time()
      first = await self.bus.reply_to_repeated_command(cmd)
      reply = first or await self.run_sample(cmd.run, json.loads(cmd.body))
      ms = (time.time() - start) * 1000
      if not await self.bus.write_reply(cmd, reply, self.worker, self.owner, repeat=first is not None, started=start):
        self.done = True
      return [cmd.op, ms]

    async with self.hold_gpu_for_batch(len(cmds)):
      batch_start = time.time()
      boot = 0.0
      if self.engine is None:
        try:
          await self.start_engine_for_session(cmds[0].run)
          boot = time.time() - batch_start
        except Exception:  # each command starts it again and replies with the error
          print(f"{self.worker}: the engine failed to start for a batch", file=sys.stderr, flush=True)
          traceback.print_exc()
      if block_ms:
        await asyncio.sleep(0.02)
        cmds += await self.bus.take_commands(self.worker, self.owner, count, 0)
      if self.lora:
        batches = [cmds]
      else:
        by_run: dict[str, list] = {}
        [by_run.setdefault(c.run, []).append(c) for c in cmds]
        batches = list(by_run.values())
      ops = []
      for batch in batches:
        if not self.done:
          ops.extend(await asyncio.gather(*(one(c) for c in batch)))

      await self.turn.report_batch_done(
        queued=len(cmds) - len(ops), seconds=time.time() - batch_start, boot=boot, ops=ops
      )
    return len(cmds)

  async def run_sample(self, session: str, job: dict) -> Reply:
    try:
      run = await self.bus.get_run(session)
      if run is None:
        raise ValueError(f"no sampling session {session}")
      async with self.loading:
        if self.engine is None:
          self.engine = await self.engine_type.start(run["base"], self.lora)
        name = await self.load_checkpoint(run["model_path"])
      return Reply(True, wire.PROTO, encode(job, await self.engine.sample(job, name)))
    except Exception as e:
      return fail(e)

  async def load_checkpoint(self, uri: str | None) -> str | None:
    """The engine's name for a checkpoint, which the engine loads unless it already holds it."""
    if not uri:
      return None
    saved = (await self.registry.get_checkpoint(uri) or {}).get("version")  # another node's may still be uploading
    path, version = await asyncio.to_thread(self.store.local_folder, uri, saved)  # a checkpoint name can be saved again
    if (uri, version) not in self.names:
      name = f"c{len(self.names)}"
      await self.engine.load_checkpoint(name, path)
      self.names[(uri, version)] = name
      self.loaded = name
    elif not self.lora and self.loaded != self.names[(uri, version)]:
      await self.engine.load_checkpoint(self.names[(uri, version)], path)
      self.loaded = self.names[(uri, version)]
    return self.names[(uri, version)]

  @contextlib.asynccontextmanager
  async def hold_gpu_for_batch(self, queued: int):
    """Hold the GPU for one batch. Waits for this sampler's turn and wakes its engine, then gives the turn back after
    the batch if the scheduler revoked it."""
    if await self.turn.wait_for_turn(queued) and self.engine:
      await self.engine.wake()
    try:
      yield
    finally:
      await self.yield_gpu()

  async def yield_gpu(self) -> None:
    if await self.turn.was_revoked():
      if self.engine:
        await self.engine.sleep()
      await self.turn.give_turn_back()


if __name__ == "__main__":
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # so serve's finally closes the engine
  asyncio.run(Sampler(Bus(), sys.argv[1]).serve())
