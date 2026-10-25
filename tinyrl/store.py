"""Fast checkpoint storage across local shared memory (/dev/shm), peer HTTP, and shared disk."""

from __future__ import annotations

import atexit
import functools
import glob
import hashlib
import os
import shutil
import tempfile
import threading
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import redis
import safetensors
import safetensors.torch
import torch

from tinyrl.helpers import getenv

SHM = getenv("SHM", "")  # shared memory directory on an agent; unset, each process gets its own under /dev/shm
LOC_TTL = getenv("LOC_TTL", 3600)


def temp_shm() -> str:
  path = tempfile.mkdtemp(prefix="tinyrl-", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
  atexit.register(shutil.rmtree, path, True)
  return path


def save_atomic(src: str | dict[str, torch.Tensor], dst: str, meta: dict[str, str] | None = None) -> None:
  os.makedirs(os.path.dirname(dst), exist_ok=True)
  tmp = f"{dst}.{os.getpid()}.{threading.get_ident()}.tmp"
  if isinstance(src, str):
    shutil.copyfile(src, tmp)
  else:
    safetensors.torch.save_file({k: v.contiguous() for k, v in src.items()}, tmp, metadata=meta)
  os.replace(tmp, dst)


class QuietHandler(SimpleHTTPRequestHandler):
  def log_message(self, *args: Any) -> None:
    pass


class Store:
  """Moves named tensor bundles through local /dev/shm, peer HTTP between nodes, and durable disk."""

  def __init__(self, shm: str = "", r: redis.Redis | None = None):
    self.shm = shm or SHM or temp_shm()
    os.makedirs(self.shm, exist_ok=True)
    self.r = r
    self.url = ""
    self.server: ThreadingHTTPServer | None = None

  def serve(self, host: str = "127.0.0.1", port: int = 0) -> str:
    """Serve this store's /dev/shm directory over HTTP so peer trainers and samplers can pull straight from RAM."""
    self.server = ThreadingHTTPServer((host, port), functools.partial(QuietHandler, directory=self.shm))
    self.url = f"http://{host}:{self.server.server_port}"
    threading.Thread(target=self.server.serve_forever, daemon=True).start()
    return self.url

  def close(self) -> None:
    if self.server is not None:
      self.server.shutdown()
      self.server.server_close()
      self.server = None

  def file(self, key: str) -> tuple[str, str]:
    name = f"{hashlib.sha256(key.encode()).hexdigest()}.safetensors"
    return name, f"{self.shm}/{name}"

  def put(self, key: str, tensors: dict[str, torch.Tensor], meta: dict[str, str] | None = None, disk: str = "") -> str:
    """Stage `tensors` into local /dev/shm, advertise its peer URL in Redis, and optionally write through to `disk`."""
    name, shm_file = self.file(key)
    save_atomic(tensors, shm_file, meta)
    if disk:
      save_atomic(shm_file, disk)
    if self.r is not None and self.url:
      self.r.set(f"tinyrl:loc:{name}", f"{self.url}/{name}", ex=LOC_TTL)
    return shm_file

  def get(self, key: str, device: str = "cpu") -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Load `key` from local /dev/shm, else stream from a peer's /dev/shm over HTTP, else read from `key` on disk."""
    name, shm_file = self.file(key)
    if not os.path.exists(shm_file) and self.r is not None:
      url = self.r.get(f"tinyrl:loc:{name}")
      if url and url != f"{self.url}/{name}":
        try:
          tmp = f"{shm_file}.{os.getpid()}.{threading.get_ident()}.tmp"
          urllib.request.urlretrieve(url, tmp)
          os.replace(tmp, shm_file)
        except OSError:
          pass  # the peer is gone; fall back to durable storage

    files = [shm_file] if os.path.exists(shm_file) else sorted(glob.glob(f"{key}/*.safetensors"))
    tensors, meta = {}, {}
    for file in files:
      with safetensors.safe_open(file, framework="pt", device=device) as f:
        meta.update(f.metadata() or {})
        tensors.update({k: f.get_tensor(k) for k in f.keys()})
    return tensors, meta

  def drop(self, key: str) -> None:
    name, shm_file = self.file(key)
    if os.path.exists(shm_file):
      os.remove(shm_file)
    if self.r is not None:
      self.r.delete(f"tinyrl:loc:{name}")
