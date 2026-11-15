"""vLLM as a sampling engine: `vllm serve` on a local port, driven through its OpenAI completions API and its dev-mode endpoints for pause/resume, request aborts, adapters, weight reloads and sleep. Any server with the same endpoints can stand in."""

from __future__ import annotations

import asyncio
import contextlib
import glob
import hashlib
import importlib.metadata
import json
import os
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import time
import traceback

import aiohttp

from tinyrl.control.process import launch, reap, target_pid
from tinyrl.helpers import getenv
from tinyrl.storage import models
from tinyrl.trace import span
from tinyrl.workers import snapshot
from tinyrl.workers.sampler import Out, Seq

CUDAGRAPH_CONFIG = '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256]}'
KERNEL_CONFIG = '{"enable_flashinfer_autotune":false}'
ARTIFACT_CACHE = getenv("ARTIFACT_CACHE", os.path.expanduser("~/.cache/tinyrl-artifacts"))
STARTUP_GATE_P95_S = getenv("STARTUP_GATE_P95_S", 45.0)
MAX_LORA_RANK = 64


def pkg_version(name: str) -> str:
  with contextlib.suppress(importlib.metadata.PackageNotFoundError):
    return importlib.metadata.version(name)
  return "unknown"


def artifact_key(base: str, lora: bool, tp: int, flags: list[str], gpu_class: str = "") -> str:
  """Compatibility key for compiled engine/compiler caches; independent of job ID and mutable checkpoint version."""
  runtime_only = {
    "--gpu-memory-utilization",
    "--kv-cache-memory-bytes",
    "--host",
    "--port",
    "--uds",
    "--served-model-name",
  }
  kept, skip = [], False
  for f in flags:
    if skip:
      skip = False
    elif f in runtime_only:
      skip = True
    elif not any(f.startswith(f"{r}=") for r in runtime_only):
      kept.append(f)
  spec = {"base": base, "lora": bool(lora), "max_lora_rank": MAX_LORA_RANK if lora else 0, "tp": int(tp)}
  spec |= {
    "gpu_class": gpu_class or "generic",
    "flags": kept,
    "vllm": pkg_version("vllm"),
    "torch": pkg_version("torch"),
  }
  return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:24]


def startup_gate(samples_s: list[float], target_p95_s: float = STARTUP_GATE_P95_S, min_samples: int = 5) -> dict:
  """Evaluate whether ordinary startup with warm compiled artifacts passes the latency gate to retire snapshot.py."""
  if not samples_s:
    return {"count": 0, "p50_s": None, "p95_s": None, "max_s": None, "target_p95_s": target_p95_s, "passed": False}
  ordered = sorted(float(x) for x in samples_s)
  p50 = ordered[int(0.50 * (len(ordered) - 1))]
  p95 = ordered[min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999999) - 1))]
  out = {"count": len(ordered), "p50_s": round(p50, 3), "p95_s": round(p95, 3), "max_s": round(ordered[-1], 3)}
  return out | {"target_p95_s": target_p95_s, "passed": len(ordered) >= min_samples and p95 <= target_p95_s}


def split_stop(stop: str | list | None) -> tuple[list[str], list[int]]:
  stop = [stop] if isinstance(stop, str) else stop or []
  return [s for s in stop if isinstance(s, str)], [s for s in stop if isinstance(s, int)]


def by_id(logprobs: dict) -> dict[int, float]:
  return {int(k.removeprefix("token_id:")): v["logprob"] if isinstance(v, dict) else v for k, v in logprobs.items()}


def session(url: str, connector: aiohttp.BaseConnector) -> aiohttp.ClientSession:
  return aiohttp.ClientSession(url, connector=connector, timeout=aiohttp.ClientTimeout(total=None))


def unix_client(sock: str) -> aiohttp.ClientSession:
  return session("http://vllm", aiohttp.UnixConnector(path=sock, limit=0, keepalive_timeout=2))


def parse(body: bytes) -> Out:
  choices = sorted(json.loads(body)["choices"], key=lambda c: c["index"])
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


class VLLM:
  def __init__(
    self,
    url: str,
    base: str,
    lora: bool,
    server: subprocess.Popen | None = None,
    client: aiohttp.ClientSession | None = None,
  ):
    self.http = client or session(url, aiohttp.TCPConnector(limit=0, keepalive_timeout=2))
    self.base, self.lora, self.server = base, lora, server
    self.loaded: set[str] = set()
    self.clean = True
    self.artifact_key, self.startup_mode, self.startup_s = "", "cold", 0.0

  @classmethod
  async def start(cls, base: str, lora: bool, env: dict[str, str] | None = None) -> VLLM:
    t0 = time.monotonic()
    options = dict(os.environ if env is None else env)
    path = await asyncio.to_thread(models.local, base)
    flags = ["--enable-sleep-mode", "--language-model-only"]
    flags += (
      ["--enable-lora", "--max-lora-rank", str(MAX_LORA_RANK), "--max-loras", "16", "--max-cpu-loras", "64"]
      if lora
      else []
    )
    visible = options.get("CUDA_VISIBLE_DEVICES", "0")
    if (tp := len([g for g in visible.split(",") if g != ""])) > 1:
      flags += ["--tensor-parallel-size", str(tp)]
    extra = shlex.split(options.get("VLLM_ARGS", ""))
    if not any(a == "--compilation-config" or a.startswith(("--compilation-config=", "-O")) for a in extra):
      extra += ["--compilation-config", CUDAGRAPH_CONFIG]
    if not any(a == "--kernel-config" or a.startswith("--kernel-config=") for a in extra):
      extra += ["--kernel-config", KERNEL_CONFIG]
    if memory := options.get("TINYRL_GPU_MEM"):
      extra += ["--gpu-memory-utilization", memory]
    if (kv := options.get("TINYRL_KV_BYTES")) and not any(
      a == "--kv-cache-memory-bytes" or a.startswith("--kv-cache-memory-bytes=") for a in extra
    ):
      extra += ["--kv-cache-memory-bytes", kv]
    flags += extra

    akey = artifact_key(base, lora, tp, flags, options.get("TINYRL_GPU_CLASS", ""))
    acache = os.path.join(options.get("ARTIFACT_CACHE", ARTIFACT_CACHE), akey)
    warm_cache = os.path.isdir(acache) and bool(os.listdir(acache))
    os.makedirs(acache, exist_ok=True)
    strip = {"TINYRL_GPU_MEM", "TINYRL_KV_BYTES", "TINYRL_WARM_SAMPLER", "TINYRL_GPU_CLASS"}
    env = {
      "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
      "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": os.path.expanduser("~/.cache/flash-attention-cute"),
      "VLLM_CACHE_ROOT": os.path.join(acache, "vllm"),
      "TORCHINDUCTOR_CACHE_DIR": os.path.join(acache, "inductor"),
      **{k: v for k, v in options.items() if k not in strip},
      "VLLM_SERVER_DEV_MODE": "1",
      "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
    }

    def serve(*listen: str) -> list[str]:
      return ["vllm", "serve", path, "--served-model-name", base, *listen, *flags]

    def finish(engine: VLLM, mode: str) -> VLLM:
      engine.artifact_key, engine.startup_mode, engine.startup_s = akey, mode, round(time.monotonic() - t0, 3)
      return engine

    async def boot() -> VLLM:
      with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
      server = launch(serve("--host", "127.0.0.1", "--port", str(port)), env=env)
      return finish(
        await cls(f"http://127.0.0.1:{port}", base, lora, server).ready(), "cached" if warm_cache else "cold"
      )

    # CRIU snapshots remain gated behind TINYRL_SNAPSHOTS until GPU hardware validates startup_gate() and retires snapshot.py.
    if not (snapshots := options.get("TINYRL_SNAPSHOTS")) or tp > 1:
      return await boot()

    folder = os.path.join(snapshots, await asyncio.to_thread(snapshot.key, serve(), visible))
    if os.path.isdir(folder):
      try:
        return finish(await cls.restore(folder, base, lora, env), "snapshot")
      except Exception:
        print(f"the snapshot in {folder} did not restore, so vLLM boots", file=sys.stderr, flush=True)
        traceback.print_exc()
        shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(snapshot.RUN, exist_ok=True)
    name = f"{snapshot.RUN}/{os.urandom(4).hex()}"
    saving = (
      env | snapshot.ENV | {"PYTHONPATH": os.pathsep.join([snapshot.site(), *filter(None, [env.get("PYTHONPATH")])])}
    )
    with open(f"{name}.log", "ab") as log:
      server = launch(
        serve("--uds", f"{name}.sock"),
        env=saving,
        checkpointable=True,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
      )
    engine = await cls("http://vllm", base, lora, server, unix_client(f"{name}.sock")).ready()
    try:
      await engine.post("/sleep", params={"level": 2})
      await engine.http.close()
      await asyncio.to_thread(snapshot.save, await asyncio.to_thread(target_pid, server), folder, name, visible)
      await asyncio.to_thread(server.wait)
      return finish(await cls.restore(folder, base, lora, env), "snapshot")
    except Exception:
      print(f"vLLM could not be saved to {folder} and restored, so it boots again", file=sys.stderr, flush=True)
      traceback.print_exc()
      await engine.close()
      return await boot()
    finally:
      await engine.close()

  @classmethod
  async def restore(cls, folder: str, base: str, lora: bool, env: dict[str, str]) -> VLLM:
    namespaces, sock = await snapshot.restore(folder, env.get("CUDA_VISIBLE_DEVICES", "0"), env)
    engine = cls("http://vllm", base, lora, namespaces, unix_client(sock))
    try:
      await engine.ready()
      await engine.post("/wake_up", params={"tags": "weights"})
      await engine.post("/collective_rpc", json={"method": "reload_weights"})
      await engine.post("/wake_up", params={"tags": "kv_cache"})
      await engine.post("/reset_prefix_cache")
      return engine
    except BaseException:
      await engine.close()
      raise

  async def ready(self) -> VLLM:
    try:
      while self.server.poll() is None:
        with contextlib.suppress(aiohttp.ClientError, TimeoutError):
          async with self.http.get("/health", timeout=aiohttp.ClientTimeout(total=1)) as res:
            if res.ok:
              return self
        await asyncio.sleep(0.2)
      raise RuntimeError(f"vllm serve exited with {self.server.returncode}")
    except BaseException:
      await self.close()
      raise

  async def close(self) -> None:
    try:
      if self.server:
        reap(self.server)
        await asyncio.to_thread(self.server.wait)
        self.server = None
    finally:
      await self.http.close()

  async def post(self, path: str, **kwargs) -> bytes:
    for attempt in range(3):
      try:
        with span(f"vllm{path}"):
          async with self.http.post(path, **kwargs) as res:
            res.raise_for_status()
            return await res.read()
      except aiohttp.ClientConnectionError:
        if attempt == 2 or (self.server and self.server.poll() is not None):
          raise
    raise AssertionError

  async def load_checkpoint(self, name: str, path: str) -> None:
    if name in self.loaded:
      return
    if not self.lora and self.clean and os.path.isdir(path):
      if files := sorted(glob.glob(f"{path}/*.safetensors")):
        with open(files[0], "rb") as f:
          meta = json.loads(
            json.loads(f.read(struct.unpack("<Q", f.read(8))[0])).get("__metadata__", {}).get("meta", "{}")
          )
        if meta.get("untouched"):
          self.loaded = {name}
          return
    if self.lora:
      await self.post("/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path})
      self.loaded.add(name)
    else:
      await self.post("/collective_rpc", json={"method": "reload_weights", "kwargs": {"weights_path": path}})
      await self.post("/reset_prefix_cache")
      self.clean, self.loaded = False, {name}

  async def unload_checkpoint(self, name: str) -> None:
    if self.lora and name in self.loaded:
      await self.post("/v1/unload_lora_adapter", json={"lora_name": name})
      self.loaded.discard(name)

  async def pause(self, mode: str = "wait") -> None:
    await self.post("/pause", params={"mode": mode})

  async def resume(self) -> None:
    await self.post("/resume")

  async def abort(self, request_ids: list[str]) -> None:
    if request_ids:
      await self.post("/abort_requests", json={"request_ids": request_ids})

  async def sleep(self) -> None:
    await self.post("/sleep", params={"level": 1})

  async def wake(self) -> None:
    await self.post("/wake_up")

  async def sample(self, job: dict, name: str | None, request_id: str | None = None) -> Out:
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
      "return_tokens_as_token_ids": True,
    }
    if request_id:
      req["request_id"] = request_id
    return await asyncio.to_thread(parse, await self.post("/v1/completions", json=req))
