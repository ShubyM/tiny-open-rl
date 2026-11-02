"""vLLM as a sampling engine: `vllm serve` on a local port, driven through its OpenAI completions API and its dev-mode endpoints for adapters, weight reloads and sleep. Any server with the same endpoints can stand in."""

from __future__ import annotations

import asyncio
import contextlib
import glob
import json
import os
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import traceback

import httpx

from tinyrl.helpers import getenv
from tinyrl.storage import models
from tinyrl.workers import snapshot
from tinyrl.workers.sampler import Out, Seq

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


def unix_client(sock: str) -> httpx.AsyncClient:
  limits = httpx.Limits(max_connections=1024, max_keepalive_connections=256, keepalive_expiry=2.0)
  return httpx.AsyncClient(
    base_url="http://vllm", timeout=None, transport=httpx.AsyncHTTPTransport(uds=sock, limits=limits)
  )


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
    path = await asyncio.to_thread(models.local, base)  # served under the reference's name, which requests use
    flags = ["--enable-sleep-mode", "--language-model-only"]
    flags += ["--enable-lora", "--max-lora-rank", "64", "--max-loras", "16", "--max-cpu-loras", "64"] if lora else []
    if (tp := len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))) > 1:
      flags += ["--tensor-parallel-size", str(tp)]
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
    flags += extra

    def serve(*listen: str) -> list[str]:
      return ["vllm", "serve", path, "--served-model-name", base, *listen, *flags]

    async def boot() -> VLLM:
      with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
      server = subprocess.Popen(serve("--host", "127.0.0.1", "--port", str(port)), env=env)
      return await cls(f"http://127.0.0.1:{port}", base, lora, server).ready()

    # a multi-GPU server's NCCL communicators must be suspended before a snapshot, which isn't done yet
    if not snapshot.SNAPSHOTS or tp > 1:
      return await boot()

    folder = os.path.join(snapshot.SNAPSHOTS, await asyncio.to_thread(snapshot.key, serve()))
    if os.path.isdir(folder):
      try:
        return await cls.restore(folder, base, lora)
      except Exception:
        print(f"the snapshot in {folder} did not restore, so vLLM boots", file=sys.stderr, flush=True)
        traceback.print_exc()
        shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(snapshot.RUN, exist_ok=True)
    name = f"{snapshot.RUN}/{os.urandom(4).hex()}"
    saving = (
      env | snapshot.ENV | {"PYTHONPATH": os.pathsep.join([snapshot.site(), *filter(None, [env.get("PYTHONPATH")])])}
    )
    # its own stdin and log: a snapshot can't hold files from outside it
    with open(f"{name}.log", "ab") as log:
      server = subprocess.Popen(
        serve("--uds", f"{name}.sock"), env=saving, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
      )
    engine = await cls("http://vllm", base, lora, server, unix_client(f"{name}.sock")).ready()
    try:
      await engine.post("/sleep", params={"level": 2})
      await engine.http.aclose()  # a connection into the server would reach outside the snapshot
      await asyncio.to_thread(snapshot.save, server.pid, folder, name)
      await asyncio.to_thread(server.wait)
      return await cls.restore(folder, base, lora)
    except Exception:
      print(f"vLLM could not be saved to {folder} and restored, so it boots again", file=sys.stderr, flush=True)
      traceback.print_exc()
      await engine.close()
      return await boot()

  @classmethod
  async def restore(cls, folder: str, base: str, lora: bool) -> VLLM:
    """Restore a snapshot of vLLM asleep at level 2, then wake it with its weights read again from the model files."""
    namespaces, sock = await asyncio.to_thread(snapshot.restore, folder)
    engine = cls("http://vllm", base, lora, namespaces, unix_client(sock))
    try:
      await engine.ready()
      await engine.post("/wake_up", params={"tags": "weights"})
      await engine.post("/collective_rpc", json={"method": "reload_weights"})
      await engine.post("/wake_up", params={"tags": "kv_cache"})
      await engine.post("/reset_prefix_cache")
      return engine
    except Exception:
      await engine.close()
      raise

  async def ready(self) -> VLLM:
    while self.server.poll() is None:
      with contextlib.suppress(httpx.HTTPError):
        if (await self.http.get("/health")).is_success:
          return self
      await asyncio.sleep(0.2)
    raise RuntimeError(f"vllm serve exited with {self.server.returncode}")

  async def close(self) -> None:
    if self.server:
      if self.server.args[0] == "unshare":
        self.server.kill()  # a restored server's namespaces end with their root, which blocks SIGTERM
      else:
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

  async def load_checkpoint(self, name: str, path: str) -> None:
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
