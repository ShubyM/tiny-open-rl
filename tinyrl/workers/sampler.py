"""Sampling driver: coalesced checkpoint loading, SampleResponse encoding, and Store-backed stability blobs."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from dataclasses import dataclass

import numpy as np

from tinyrl.api import wire
from tinyrl.control.accounting import Charge, Wave
from tinyrl.types import Cmd, Reply, StabSlice


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
  pairs = [(sorted(r.items(), key=lambda kv: -kv[1])[:k] if r else []) for r in rows]
  padded = [p + [(0, wire.MASK_LOGPROB)] * (k - len(p)) for p in pairs]
  ids = np.array([[t for t, _ in p] for p in padded], np.int32).reshape(len(rows), k)
  return ids, np.array([[lp for _, lp in p] for p in padded], np.float32).reshape(len(rows), k)


def encode(job: dict, out: Out) -> bytes:
  prompt, last_n, k = job["prompt"], job["prompt_logprobs_last_n"], job["topk_sample"]
  rows = [r if i >= (len(prompt) - last_n if last_n else 1) else None for i, r in enumerate(out.prompt)]
  plp = np.array([r[t] if r else np.nan for r, t in zip(rows, prompt)], np.float32) if job["prompt_logprobs"] else None
  seqs = [
    wire.SampleOut(
      s.stop, np.array(s.tokens, np.int32), np.array(s.logprobs, np.float32), topk(s.top, k) if k else None
    )
    for s in out.seqs
  ]
  return wire.sample_output(seqs, plp, topk(rows, job["topk_prompt"]) if job["topk_prompt"] else None)


@dataclass
class SharedLoad:
  task: asyncio.Task
  cost: Charge
  waiters: int = 0


def pack_stability_blob(request_id: str, out: Out, store) -> dict[str, StabSlice]:
  k = max((len(row) for seq in out.seqs for row in seq.top if row), default=1)
  chunks = [wire.pack_stability(*topk(seq.top, k)) for seq in out.seqs]
  ref = store.put_blob(b"".join(chunks))
  slices: dict[str, StabSlice] = {}
  offset = 0
  for index, chunk in enumerate(chunks):
    slices[f"{request_id}:{index}"] = (ref, offset, len(chunk))
    offset += len(chunk)
  return slices


class Sampler:
  def __init__(self, base: str, lora: bool, env=None, engine=None, store=None):
    self.env = dict(os.environ if env is None else env)
    module, name = self.env.get("ENGINE", "tinyrl.workers.vllm:VLLM").split(":")
    self.engine_type = engine or getattr(importlib.import_module(module), name)
    self.base = base
    self.lora = lora
    self.store = store
    self.running = None
    self.loading = asyncio.Lock()
    self.loads: dict[tuple[str, str], SharedLoad] = {}

  @property
  def proc(self):
    return getattr(self.running, "server", None)

  async def start(self) -> None:
    async with self.loading:
      if self.running is None:
        self.running = await self.engine_type.start(self.base, self.lora, self.env)

  async def load(self, checkpoint: tuple[str, str]) -> None:
    path, name = checkpoint
    await self.start()
    async with self.loading:
      await self.running.load_checkpoint(name, path)

  async def load_shared(self, checkpoint: tuple[str, str], wave: Wave, charge: Charge) -> None:
    entry = self.loads.get(checkpoint)
    if entry is None:
      cost = Charge()
      task = asyncio.create_task(wave.run(self.load(checkpoint), cost))
      entry = SharedLoad(task=task, cost=cost)
      self.loads[checkpoint] = entry
    entry.waiters += 1
    try:
      await entry.task
    finally:
      if self.loads.get(checkpoint) is entry:
        del self.loads[checkpoint]
      charge.started_ms = entry.cost.started_ms
      charge.busy_ms += entry.cost.busy_ms
      charge.gpu_ms += entry.cost.gpu_ms / entry.waiters

  async def execute(
    self,
    cmd: Cmd,
    checkpoint: tuple[str, str] | None = None,
    *,
    wave: Wave | None = None,
    charge: Charge | None = None,
  ) -> Reply:
    try:
      if checkpoint is None:
        await self.start()
      elif wave and charge:
        await self.load_shared(checkpoint, wave, charge)
      else:
        await self.load(checkpoint)

      job = json.loads(cmd.body)
      ckpt_name = checkpoint[1] if checkpoint else None
      call = self.running.sample(job, ckpt_name)
      out = await wave.run(call, charge) if wave and charge else await call

      def encode_reply() -> Reply:
        stability = None
        if job.get("record_stability_info") and self.store:
          stability = pack_stability_blob(cmd.request_id, out, self.store)
        return Reply(True, wire.PROTO, encode(job, out), stability)

      return await asyncio.to_thread(encode_reply)
    except Exception as exc:
      return Reply.error(exc)

  async def resume(self, gpus: tuple[int, ...]) -> None:
    if self.running is None:
      self.env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
      await self.start()
    else:
      await self.running.wake()

  async def park(self) -> None:
    if self.running:
      await self.running.sleep()

  async def close(self) -> None:
    if self.running:
      await self.running.close()
      self.running = None
