"""Sampling. A sampler serves a sampling pool the way a trainer serves a training pool: it holds the pool's lease, runs
its commands in order, and takes turns on the GPU.

It samples through an Engine. SAMPLER=vllm runs `vllm serve` on a local port and drives it over HTTP, so any server
with the same endpoints can stand in. Otherwise it samples with transformers in process, which is what the tests use.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinyrl import wire
from tinyrl.bus import LEASE_MS, Bus, Reply
from tinyrl.helpers import getenv
from tinyrl.store import Store
from tinyrl.trainer import GPU, HOLD_S, STRANDED_MS, fail

SAMPLER = getenv("SAMPLER", "hf")
VLLM_ARGS = getenv("VLLM_ARGS", "")  # more `vllm serve` flags, like --gpu-memory-utilization 0.4
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
  async def load_lora(self, name: str, path: str) -> None: ...
  async def load_weights(self, path: str) -> None: ...
  async def sample(self, job: dict, adapter: str | None) -> Out: ...
  async def sleep(self) -> None: ...
  async def wake(self) -> None: ...


def wants_prompt(job: dict) -> bool:
  return job["prompt_logprobs"] or job["topk_prompt"] > 0


def split_stop(stop: str | list | None) -> tuple[list[str], list[int]]:
  stop = [stop] if isinstance(stop, str) else stop or []
  return [s for s in stop if isinstance(s, str)], [s for s in stop if isinstance(s, int)]


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


class VLLMServer:
  def __init__(self, url: str, base: str, client: httpx.AsyncClient | None = None):
    self.http = client or httpx.AsyncClient(base_url=url, timeout=None)
    self.base = base

  async def post(self, path: str, **kwargs) -> httpx.Response:
    res = await self.http.post(path, **kwargs)
    res.raise_for_status()
    return res

  async def load_lora(self, name: str, path: str) -> None:
    await self.post("/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path})

  async def load_weights(self, path: str) -> None:
    await self.post("/collective_rpc", json={"method": "reload_weights", "kwargs": {"weights_path": path}})
    await self.post("/reset_prefix_cache")

  async def sleep(self) -> None:
    await self.post("/sleep", params={"level": 1})

  async def wake(self) -> None:
    await self.post("/wake_up")

  async def sample(self, job: dict, adapter: str | None) -> Out:
    strs, ids = split_stop(job["stop"])
    req = {
      "model": adapter or self.base,
      "prompt": job["prompt"],
      "n": job["num_samples"],
      "max_tokens": job["max_tokens"],
      "temperature": job["temperature"],
      "top_k": job["top_k"],
      "top_p": job["top_p"],
      "seed": job["seed"],
      "stop": strs,
      "stop_token_ids": ids,
      "logprobs": max(1, job["topk_sample"]),
      "prompt_logprobs": max(1, job["topk_prompt"]) if wants_prompt(job) else None,
      "return_token_ids": True,
      "return_tokens_as_token_ids": True,  # so logprobs are keyed "token_id:<id>", not by text two tokens can share
    }
    choices = sorted((await self.post("/v1/completions", json=req)).json()["choices"], key=lambda c: c["index"])

    def by_id(d: dict) -> dict[int, float]:
      return {int(k.removeprefix("token_id:")): v["logprob"] if isinstance(v, dict) else v for k, v in d.items()}

    seqs = [
      Seq(
        c["token_ids"],
        c["logprobs"]["token_logprobs"],
        [by_id(d) for d in c["logprobs"]["top_logprobs"]],
        c["finish_reason"] == "stop",
      )
      for c in choices
    ]
    return Out(seqs, [by_id(d) if d else None for d in choices[0].get("prompt_logprobs") or []])


def top(lps: torch.Tensor, k: int, tokens: list[int]) -> list[dict[int, float]]:
  """Seq.top rows from [len(tokens), vocab] logprobs."""
  values, ids = lps.topk(k, dim=-1)
  chosen = lps[torch.arange(len(tokens)), tokens].tolist()
  return [{**dict(zip(i.tolist(), v.tolist())), t: c} for v, i, t, c in zip(values, ids, tokens, chosen)]


class HFEngine:
  """transformers and PEFT in process. Slow, but it needs no server, so tests and CPU machines use it."""

  def __init__(self, base: str):
    self.base = base
    self.lm = AutoModelForCausalLM.from_pretrained(base).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    self.device = self.lm.device

  async def load_lora(self, name: str, path: str) -> None:
    import peft

    if isinstance(self.lm, peft.PeftModel):
      await asyncio.to_thread(self.lm.load_adapter, path, adapter_name=name)
    else:
      self.lm = await asyncio.to_thread(peft.PeftModel.from_pretrained, self.lm, path, adapter_name=name)

  async def load_weights(self, path: str) -> None:
    import safetensors.torch

    await asyncio.to_thread(self.lm.load_state_dict, safetensors.torch.load_file(f"{path}/model.safetensors"), False)

  async def sleep(self) -> None:
    await asyncio.to_thread(self.lm.to, "cpu")

  async def wake(self) -> None:
    await asyncio.to_thread(self.lm.to, self.device)

  async def sample(self, job: dict, adapter: str | None) -> Out:
    return await asyncio.to_thread(self.generate, job, adapter)

  def generate(self, job: dict, adapter: str | None) -> Out:
    import peft

    lm, prompt = self.lm, job["prompt"]
    lora = isinstance(lm, peft.PeftModel)
    if adapter:
      lm.set_adapter(adapter)
    strs, stop_ids = split_stop(job["stop"])
    eos = lm.generation_config.eos_token_id
    eos = [*([eos] if isinstance(eos, int) else eos or []), *stop_ids]
    if job["temperature"] > 0:
      how = {"do_sample": True, "temperature": job["temperature"], "top_k": max(0, job["top_k"]), "top_p": job["top_p"]}
    else:
      how = {"do_sample": False}
    if strs:
      how |= {"stop_strings": strs, "tokenizer": AutoTokenizer.from_pretrained(self.base)}
    ids = torch.tensor([prompt], device=self.device)

    with lm.disable_adapter() if lora and not adapter else contextlib.nullcontext(), torch.no_grad():
      rows = []
      if wants_prompt(job):
        rows = [None, *top(lm(ids).logits[0, :-1].float().log_softmax(-1), max(1, job["topk_prompt"]), prompt[1:])]
      if job["seed"] is not None:
        torch.manual_seed(job["seed"])
      seqs = []
      for _ in range(job["num_samples"]):  # one at a time, so a finished sample isn't padded out to the longest
        out = lm.generate(
          ids,
          max_new_tokens=job["max_tokens"],
          eos_token_id=eos,
          pad_token_id=eos[0] if eos else 0,
          output_logits=True,
          return_dict_in_generate=True,
          **how,
        )
        tokens = out.sequences[0, len(prompt) :].tolist()
        tops = top(torch.stack(out.logits)[:, 0].float().log_softmax(-1), max(1, job["topk_sample"]), tokens)
        stop = len(tokens) < job["max_tokens"] or tokens[-1] in eos
        seqs.append(Seq(tokens, [d[t] for d, t in zip(tops, tokens)], tops, stop))
    return Out(seqs, rows)


class Sampler:
  def __init__(self, bus: Bus, pool: str, engine: Engine | None = None, store: Store | None = None, worker: str = ""):
    self.bus = bus
    self.pool = pool
    self.engine = engine
    self.store = store if store is not None else Store()
    self.lora = not pool.startswith("sampler:fft:")
    self.worker = worker or f"{socket.gethostname()}-{os.getpid()}-{os.urandom(3).hex()}"
    self.adapters: dict[tuple[str, int], str] = {}
    self.loaded: tuple[str, int] | None = None
    self.server: subprocess.Popen | None = None
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
      if self.server:
        self.server.terminate()
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
        self.engine = await self.start(run["base"])
      out = await self.engine.sample(job, await self.load(run["model_path"]))
      return Reply(True, wire.PROTO, encode(job, out))
    except Exception as e:
      return fail(e)

  async def start(self, base: str) -> Engine:
    if SAMPLER != "vllm":
      return await asyncio.to_thread(HFEngine, base)
    with socket.socket() as s:
      s.bind(("127.0.0.1", 0))
      port = s.getsockname()[1]
    lora = ["--enable-lora", "--max-lora-rank", "64"] if self.lora else []
    args = ["vllm", "serve", base, "--host", "127.0.0.1", "--port", str(port), "--enable-sleep-mode", *lora]
    env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
    self.server = subprocess.Popen([*args, *shlex.split(VLLM_ARGS)], env=env)
    engine = VLLMServer(f"http://127.0.0.1:{port}", base)
    while self.server.poll() is None:
      with contextlib.suppress(httpx.HTTPError):
        if (await engine.http.get("/health")).is_success:
          return engine
      await asyncio.sleep(1)
    raise RuntimeError(f"vllm serve exited with {self.server.returncode}")

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
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # so serve's finally stops vllm serve
  asyncio.run(Sampler(Bus(), sys.argv[1]).serve())
