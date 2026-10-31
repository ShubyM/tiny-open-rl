"""vLLM as a sampling engine: `vllm serve` on a local port, driven through its OpenAI completions API and its dev-mode
endpoints for adapters, weight reloads and sleep. Any server with the same endpoints can stand in.
"""

from __future__ import annotations

import asyncio
import contextlib
import glob
import json
import os
import shlex
import socket
import struct
import subprocess

import httpx

from tinyrl.helpers import getenv
from tinyrl.sampler import Out, Seq

VLLM_ARGS = getenv("VLLM_ARGS", "")  # more `vllm serve` flags, like --gpu-memory-utilization 0.4
GPU_MEM = getenv("TINYRL_GPU_MEM", "")  # the share of GPU memory vLLM may use, from the scheduler's placement
KV_BYTES = getenv("TINYRL_KV_BYTES", "")  # bytes for vLLM's KV cache, from the scheduler's placement
CUDAGRAPH_CONFIG = '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256]}'
KERNEL_CONFIG = '{"enable_flashinfer_autotune":false}'


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
    limits = httpx.Limits(max_connections=1024, max_keepalive_connections=256, keepalive_expiry=2.0)
    self.http = client or httpx.AsyncClient(base_url=url, timeout=None, limits=limits)
    self.base = base
    self.lora = lora
    self.server = server
    self.loaded: set[str] = set()  # adapters, or the one full fine-tune checkpoint whose weights are in place
    self.clean: bool = True

  @classmethod
  async def start(cls, base: str, lora: bool) -> VLLM:
    with socket.socket() as s:
      s.bind(("127.0.0.1", 0))
      port = s.getsockname()[1]
    args = ["vllm", "serve", base, "--host", "127.0.0.1", "--port", str(port), "--enable-sleep-mode"]
    args += ["--language-model-only"] + (
      ["--enable-lora", "--max-lora-rank", "64", "--max-loras", "16", "--max-cpu-loras", "64"] if lora else []
    )
    if (tp := len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))) > 1:
      args += ["--tensor-parallel-size", str(tp)]
    env = {
      "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",  # its FlashAttention 4 kernels compile at each start unless cached
      "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": os.path.expanduser("~/.cache/flash-attention-cute"),
      **{k: v for k, v in os.environ.items() if k not in {"TINYRL_GPU_MEM", "TINYRL_KV_BYTES", "TINYRL_WARM_SAMPLER"}},
      "VLLM_SERVER_DEV_MODE": "1",
      "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
    }
    extra = shlex.split(VLLM_ARGS)
    if not any(
      a == "--compilation-config" or a.startswith("--compilation-config=") or a.startswith("-O") for a in extra
    ):
      extra += ["--compilation-config", CUDAGRAPH_CONFIG]
    if not any(a == "--kernel-config" or a.startswith("--kernel-config=") for a in extra):
      extra += ["--kernel-config", KERNEL_CONFIG]
    if GPU_MEM:
      extra += ["--gpu-memory-utilization", GPU_MEM]
    if KV_BYTES and not any(a == "--kv-cache-memory-bytes" or a.startswith("--kv-cache-memory-bytes=") for a in extra):
      extra += ["--kv-cache-memory-bytes", KV_BYTES]
    server = subprocess.Popen([*args, *extra], env=env)
    engine = cls(f"http://127.0.0.1:{port}", base, lora, server)
    while server.poll() is None:
      with contextlib.suppress(httpx.HTTPError):
        if (await engine.http.get("/health")).is_success:
          return engine
      await asyncio.sleep(0.2)
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
    for attempt in range(3):
      try:
        res = await self.http.post(path, **kwargs)
        res.raise_for_status()
        return res
      except httpx.TransportError:
        if attempt == 2 or (self.server and self.server.poll() is not None):
          raise
    raise AssertionError

  async def load(self, name: str, path: str) -> None:
    if name in self.loaded:
      return
    if not self.lora and self.clean and os.path.isdir(path):
      files = sorted(glob.glob(f"{path}/*.safetensors"))
      if files:
        with open(files[0], "rb") as f:
          n = struct.unpack("<Q", f.read(8))[0]
          meta = json.loads(json.loads(f.read(n)).get("__metadata__", {}).get("meta", "{}"))
        if meta.get("untouched"):
          self.loaded = {name}
          return
    if self.lora:
      await self.post("/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path})
      self.loaded.add(name)
    else:
      await self.post("/collective_rpc", json={"method": "reload_weights", "kwargs": {"weights_path": path}})
      await self.post("/reset_prefix_cache")
      self.clean = False
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
