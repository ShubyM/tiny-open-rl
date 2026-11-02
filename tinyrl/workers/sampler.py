"""Sampling. A sampler serves sampling sessions the way a trainer serves training runs: it holds its worker's lease, runs their commands in order, and takes turns on the GPU.

It samples through an Engine, which knows nothing of Redis, workers or checkpoint URIs. ENGINE names the engine's class, vLLM (tinyrl/workers/vllm.py) unless it's set.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
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
  loaded: set[str]  # adapters resident in the engine, or its current full checkpoint

  @classmethod
  async def start(cls, base: str, lora: bool) -> Engine: ...
  # an adapter, or a full fine-tune's weights in place
  async def load_checkpoint(self, name: str, path: str) -> None: ...
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
    self.running: Engine | None = None  # started by the first command, which says which base model
    self.store = store if store is not None else Store()
    self.lora = os.getenv("TINYRL_LORA", "1") != "0"
    self.owner = owner or f"{socket.gethostname()}-{os.getpid()}-{os.urandom(3).hex()}"
    self.loading = asyncio.Lock()
    self.done = False
    self.turn = GpuTurn(bus, worker, bool(GPU))

  async def warm_engine(self) -> None:
    if self.running is not None:
      return
    base = os.getenv("TINYRL_BASE")
    if not base:
      return
    async with self.hold_gpu_for_batch(0):
      async with self.loading:
        if self.running is None:
          try:
            start = time.time()
            self.running = await self.engine_type.start(base, self.lora)
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
      if self.running:
        await self.running.close()
      await self.turn.give_turn_back()
      await self.bus.release_lease(self.worker, self.owner)

  async def wait_for_lease(self) -> None:
    while not await self.bus.take_lease(self.worker, self.owner):
      await asyncio.sleep(LEASE_MS / 4000)

  async def keep_renewing_lease(self) -> None:
    while not self.done:
      await asyncio.sleep(LEASE_MS / 4000)
      if not await self.bus.renew_lease(self.worker, self.owner) and not await self.bus.take_lease(
        self.worker, self.owner, recover=True
      ):
        # another sampler took the worker. A lease that expired untaken, as in a Redis restart, is taken again
        self.done = True
        return

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
    runs = {run: await self.bus.get_run(run) for run in {cmd.run for cmd in cmds}}
    with contextlib.ExitStack() as files:

      async def prepare(run):
        if run is None:
          raise ValueError("no sampling session")
        checkpoint = None
        if uri := run["model_path"]:
          version = (await self.registry.get_checkpoint(uri) or {}).get("version")
          name = hashlib.sha256(f"{uri}/{version}".encode()).hexdigest()
          if self.lora and self.running and name in self.running.loaded:
            return run, ("", name)
          async with self.loading:
            await self.yield_gpu(force=True)  # release a retained turn before checkpoint I/O, too
          hold = self.store.hold(uri, version)
          path, version = await asyncio.to_thread(hold.__enter__)
          files.push(hold)
          checkpoint = path, hashlib.sha256(f"{uri}/{version}".encode()).hexdigest()
        return run, checkpoint

      prepared = dict(zip(runs, await asyncio.gather(*(prepare(run) for run in runs.values()), return_exceptions=True)))

      async def one(cmd) -> list:
        start, ready = time.time(), prepared[cmd.run]
        reply = fail(ready) if isinstance(ready, Exception) else await self.run_sample(*ready, cmd.body)
        ms = (time.time() - start) * 1000
        if not await self.bus.write_reply(cmd, reply, self.worker, self.owner, started=start):
          self.done = True
        return [cmd.op, ms]

      bad = [cmd for cmd in cmds if isinstance(prepared[cmd.run], Exception)]
      await asyncio.gather(*(one(cmd) for cmd in bad))
      ready = [cmd for cmd in cmds if not isinstance(prepared[cmd.run], Exception)]
      if ready and not self.done:
        async with self.hold_gpu_for_batch(len(ready)):
          batch_start = time.time()
          by_run: dict[str, list] = {}
          for cmd in ready:
            by_run.setdefault("" if self.lora else cmd.run, []).append(cmd)
          ops = []
          for batch in by_run.values():
            ops.extend(await asyncio.gather(*(one(cmd) for cmd in batch)))
          await self.turn.report_batch_done(queued=0, seconds=time.time() - batch_start, ops=ops)
    return len(cmds)

  async def run_sample(self, run: dict, checkpoint: tuple[str, str] | None, body: bytes) -> Reply:
    try:
      job, name = json.loads(body), checkpoint[1] if checkpoint else None
      async with self.loading:
        if self.running is None:
          self.running = await self.engine_type.start(run["base"], self.lora)
        if checkpoint and name not in self.running.loaded:
          await self.running.load_checkpoint(name, checkpoint[0])
      return Reply(True, wire.PROTO, encode(job, await self.running.sample(job, name)))
    except Exception as e:
      return fail(e)

  @contextlib.asynccontextmanager
  async def hold_gpu_for_batch(self, queued: int):
    """Hold the GPU for one batch. Waits for this sampler's turn and wakes its engine, then gives the turn back after the batch if the scheduler revoked it."""
    if await self.turn.wait_for_turn(queued) and self.running:
      await self.running.wake()
    try:
      yield
    finally:
      await self.yield_gpu()

  async def yield_gpu(self, force: bool = False) -> None:
    if self.turn.held and self.turn.shared and (force or await self.turn.was_revoked()):
      if self.running:
        await self.running.sleep()
      await self.turn.give_turn_back()


if __name__ == "__main__":
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # so serve's finally closes the engine
  asyncio.run(Sampler(Bus(), sys.argv[1]).serve())
