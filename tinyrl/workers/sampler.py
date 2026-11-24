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


def pack_stability_blob(request_id: str, out: Out, store) -> dict[str, StabSlice]:
  k = max((len(r) for s in out.seqs for r in s.top if r), default=1)
  chunks = [wire.pack_stability(*topk(s.top, k)) for s in out.seqs]
  ref, at, out_map = store.put_blob(b"".join(chunks)), 0, {}
  for i, ch in enumerate(chunks):
    out_map[f"{request_id}:{i}"] = (ref, at, len(ch))
    at += len(ch)
  return out_map


class Sampler:
  def __init__(self, base: str, lora: bool, env=None, engine=None, store=None):
    self.env = dict(os.environ if env is None else env)
    module, name = self.env.get("ENGINE", "tinyrl.workers.vllm:VLLM").split(":")
    self.engine_type = engine or getattr(importlib.import_module(module), name)
    self.base, self.lora, self.store = base, lora, store
    self.running, self.loading, self.loads = None, asyncio.Lock(), {}

  @property
  def proc(self):
    return getattr(self.running, "server", None)

  async def start(self) -> None:
    async with self.loading:
      if self.running is None:
        self.running = await self.engine_type.start(self.base, self.lora, self.env)

  async def load(self, checkpoint: tuple[str, str]) -> None:
    await self.start()
    async with self.loading:
      await self.running.load_checkpoint(checkpoint[1], checkpoint[0])

  async def load_shared(self, ckpt: tuple[str, str], wave: Wave, charge: Charge) -> None:
    if (entry := self.loads.get(ckpt)) is None:
      cost = Charge()
      self.loads[ckpt] = entry = [asyncio.create_task(wave.run(self.load(ckpt), cost)), cost, 0]
    entry[2] += 1
    try:
      await entry[0]
    finally:
      if self.loads.get(ckpt) is entry:
        del self.loads[ckpt]
      charge.started_ms, charge.busy_ms = entry[1].started_ms, charge.busy_ms + entry[1].busy_ms
      charge.gpu_ms += entry[1].gpu_ms / entry[2]

  async def execute(
    self, cmd: Cmd, ckpt: tuple[str, str] | None = None, *, wave: Wave | None = None, charge: Charge | None = None
  ) -> Reply:
    try:
      if ckpt is not None:
        await (self.load_shared(ckpt, wave, charge) if wave and charge else self.load(ckpt))
      else:
        await self.start()
      job = json.loads(cmd.body)
      call = self.running.sample(job, ckpt[1] if ckpt else None)
      out = await (wave.run(call, charge) if wave and charge else call)

      def pack() -> Reply:
        stab = (
          pack_stability_blob(cmd.request_id, out, self.store)
          if job.get("record_stability_info") and self.store
          else None
        )
        return Reply(True, wire.PROTO, encode(job, out), stab)

      return await asyncio.to_thread(pack)
    except Exception as e:
      return Reply.error(e)

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
