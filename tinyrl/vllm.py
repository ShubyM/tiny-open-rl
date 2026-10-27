"""vLLM as a sampling engine: `vllm serve` on a local port, driven through its OpenAI completions API and its dev-mode
endpoints for adapters, weight reloads and sleep. Any server with the same endpoints can stand in.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import socket
import subprocess

import httpx

from tinyrl.helpers import getenv
from tinyrl.sampler import Out, Seq

VLLM_ARGS = getenv("VLLM_ARGS", "")  # more `vllm serve` flags, like --gpu-memory-utilization 0.4


def split_stop(stop: str | list | None) -> tuple[list[str], list[int]]:
  stop = [stop] if isinstance(stop, str) else stop or []
  return [s for s in stop if isinstance(s, str)], [s for s in stop if isinstance(s, int)]


def by_id(logprobs: dict) -> dict[int, float]:
  return {int(k.removeprefix("token_id:")): v["logprob"] if isinstance(v, dict) else v for k, v in logprobs.items()}


class VLLM:
  def __init__(
    self,
    url: str,
    base: str,
    lora: bool,
    server: subprocess.Popen | None = None,
    client: httpx.AsyncClient | None = None,
  ):
    self.http = client or httpx.AsyncClient(base_url=url, timeout=None)
    self.base = base
    self.lora = lora
    self.server = server
    self.loaded: set[str] = set()  # adapters, or the one full fine-tune checkpoint whose weights are in place

  @classmethod
  async def start(cls, base: str, lora: bool) -> VLLM:
    with socket.socket() as s:
      s.bind(("127.0.0.1", 0))
      port = s.getsockname()[1]
    args = ["vllm", "serve", base, "--host", "127.0.0.1", "--port", str(port), "--enable-sleep-mode"]
    args += ["--enable-lora", "--max-lora-rank", "64"] if lora else []
    env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
    server = subprocess.Popen([*args, *shlex.split(VLLM_ARGS)], env=env)
    engine = cls(f"http://127.0.0.1:{port}", base, lora, server)
    while server.poll() is None:
      with contextlib.suppress(httpx.HTTPError):
        if (await engine.http.get("/health")).is_success:
          return engine
      await asyncio.sleep(1)
    raise RuntimeError(f"vllm serve exited with {server.returncode}")

  async def close(self) -> None:
    if self.server:
      self.server.terminate()
      try:
        await asyncio.to_thread(self.server.wait, 30)
      except subprocess.TimeoutExpired:
        self.server.kill()
    await self.http.aclose()

  async def post(self, path: str, **kwargs) -> httpx.Response:
    res = await self.http.post(path, **kwargs)
    res.raise_for_status()
    return res

  async def load(self, name: str, path: str) -> None:
    if name in self.loaded:
      return
    if self.lora:
      await self.post("/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path})
      self.loaded.add(name)
    else:
      await self.post("/collective_rpc", json={"method": "reload_weights", "kwargs": {"weights_path": path}})
      await self.post("/reset_prefix_cache")
      self.loaded = {name}

  async def sleep(self) -> None:
    await self.post("/sleep", params={"level": 1})

  async def wake(self) -> None:
    await self.post("/wake_up")

  async def sample(self, job: dict, name: str | None) -> Out:
    strs, ids = split_stop(job["stop"])
    scored = job["prompt_logprobs"] or job["topk_prompt"] > 0
    req = {
      "model": name if self.lora and name else self.base,
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
      "prompt_logprobs": max(1, job["topk_prompt"]) if scored else None,
      "return_token_ids": True,
      "return_tokens_as_token_ids": True,  # so logprobs are keyed "token_id:<id>", not by text two tokens can share
    }
    choices = sorted((await self.post("/v1/completions", json=req)).json()["choices"], key=lambda c: c["index"])
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
