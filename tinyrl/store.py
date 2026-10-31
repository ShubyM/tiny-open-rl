"""Checkpoint storage behind put(uri) and get(uri).

Tensors are stored in safetensors with JSON metadata in the header, never torch.save pickle. Sampler checkpoints write
the layout PEFT and vLLM load directly (adapter_model.safetensors + adapter_config.json, or model.safetensors +
config.json, sharded with a model.safetensors.index.json when large).

CKPT is a local directory, or a URL like gs://bucket/prefix that every node can reach. With a URL, a checkpoint is
written locally and uploaded, and read by downloading it once into CKPT_CACHE, since engines load from a local path.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import glob
import hashlib
import json
import os
import shutil
import tempfile
import threading
import uuid
from typing import TYPE_CHECKING

import fsspec

if TYPE_CHECKING:
  import torch

from tinyrl.helpers import getenv

CKPT = getenv("CKPT", "/tmp/tinyrl")
CKPT_CACHE = getenv("CKPT_CACHE", "/tmp/tinyrl-cache")
SHARD_BYTES = getenv("SHARD_BYTES", 256 * 1024 * 1024)  # a full checkpoint this big is written as parallel shards
MAX_SHARDS = 16
MAX_PIN_BYTES = getenv("MAX_PIN_BYTES", 10 * 1024 * 1024 * 1024)  # the most pinned host memory a save holds


def nbytes(t: torch.Tensor) -> int:
  return t.numel() * t.element_size()


def aligned(n: int) -> int:
  return (n + 7) & ~7  # so a view of the pinned buffer can start any dtype


def as_bytes(t: torch.Tensor) -> torch.Tensor:
  import torch

  return t.detach().contiguous().reshape(-1).view(torch.uint8)


def header(path: str) -> dict:
  import safetensors

  with safetensors.safe_open(path, framework="pt", device="cpu") as f:
    return json.loads((f.metadata() or {}).get("meta", "{}"))


def parallel(fn, *xs):
  if len(xs[0]) <= 1:
    return [fn(*x) for x in zip(*xs)]
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(xs[0])) as pool:
    return list(pool.map(fn, *xs))


class Store:
  def __init__(self, root: str | None = None, cache: str | None = None):
    self.root = (CKPT if root is None else root).rstrip("/")
    self.remote = "://" in self.root
    self.fs, self.base = fsspec.core.url_to_fs(self.root, auto_mkdir=True)  # a file:// URL makes folders as it writes
    if not self.remote:
      self.base = os.path.realpath(self.base)
    self.cache = CKPT_CACHE if cache is None else cache
    self.pinned: torch.Tensor | None = None  # reused by every save, since pinning host memory is slow
    self.lock = threading.RLock()  # a save's tensors can be views of self.pinned until it is written

  def path(self, uri: str) -> str:
    return f"{self.base}/{uri.removeprefix('tinker://')}"

  # ***** put *****

  def stage(self, tensors: dict[str, torch.Tensor], is_cuda=lambda t: t.is_cuda) -> dict[str, torch.Tensor]:
    """The tensors on the CPU. GPU tensors go through one pinned buffer, so the copy runs at DMA speed. When they all
    fit, they are views of it, else each streams through it in pieces into its own host tensor."""
    import torch

    with self.lock:
      gpu = {k for k, v in tensors.items() if is_cuda(v)}
      out = {k: v.detach().cpu().contiguous() for k, v in tensors.items() if k not in gpu}
      total = sum(aligned(nbytes(tensors[k])) for k in gpu)
      if not total:
        return out
      size = min(total, MAX_PIN_BYTES)
      if self.pinned is None or self.pinned.numel() < size:
        self.pinned = None  # free the old buffer before pinning a bigger one
        self.pinned = torch.empty(size, dtype=torch.uint8, pin_memory=torch.cuda.is_available())
      sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
      at = 0
      for k in gpu:
        src, v = as_bytes(tensors[k]), tensors[k]
        if total <= MAX_PIN_BYTES:
          self.pinned[at : at + len(src)].copy_(src, non_blocking=True)
          out[k] = self.pinned[at : at + len(src)].view(v.dtype).view(v.shape)
          at += aligned(len(src))
          continue
        dst = torch.empty(len(src), dtype=torch.uint8)
        for i in range(0, len(src), size):
          piece = src[i : i + size]
          self.pinned[: len(piece)].copy_(piece, non_blocking=True)
          sync()
          dst[i : i + len(piece)].copy_(self.pinned[: len(piece)])
        out[k] = dst.view(v.dtype).view(v.shape)
      sync()
      return {k: out[k] for k in tensors}

  def write(self, folder: str, tensors: dict[str, torch.Tensor], info: dict) -> list[str]:
    """Write a checkpoint into folder, and return its file names, with the weights last."""
    import safetensors.torch

    lora, cfg, total = info.get("lora"), info.get("config"), sum(map(nbytes, tensors.values()))
    n = max(1, min(MAX_SHARDS, len(tensors))) if not lora and total >= SHARD_BYTES else 1
    stem = ("adapter_model" if lora else "model") if cfg else "state"
    weights = [f"{stem}.safetensors"] if n == 1 else [f"{stem}-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]
    shards = [dict(list(tensors.items())[i::n]) for i in range(n)]
    files = {}
    if cfg:
      files["adapter_config.json" if lora else "config.json"] = cfg
    if cfg and n > 1:
      weight_map = {k: w for w, shard in zip(weights, shards) for k in shard}
      files["model.safetensors.index.json"] = {"metadata": {"total_size": total}, "weight_map": weight_map}
    for name, obj in files.items():
      with open(f"{folder}/{name}", "w") as f:
        json.dump(obj, f)
    meta = {"meta": json.dumps(info)}
    parallel(lambda name, shard: safetensors.torch.save_file(shard, f"{folder}/{name}", metadata=meta), weights, shards)
    return [*files, *weights]

  def put(self, uri: str, tensors: dict[str, torch.Tensor], meta: dict | None = None) -> str:
    dest = self.path(uri)
    if self.remote:
      os.makedirs(self.cache, exist_ok=True)
      folder, version = tempfile.mkdtemp(dir=self.cache), uuid.uuid4().hex[:16]
      try:
        with self.lock:
          names = self.write(folder, self.stage(tensors), dict(meta or {}))
        old = {}
        with contextlib.suppress(FileNotFoundError):
          old = json.loads(self.fs.cat_file(f"{dest}/_latest"))
        try:
          parallel(lambda n: self.fs.put_file(f"{folder}/{n}", f"{dest}/{n}"), names)
        except Exception:
          if not old:
            with contextlib.suppress(Exception):
              self.fs.rm(dest, recursive=True)
          raise
        self.fs.pipe_file(f"{dest}/_latest", json.dumps({"version": version, "files": names}).encode())
        for stale in set(old.get("files", [])) - set(names):
          with contextlib.suppress(FileNotFoundError):
            self.fs.rm_file(f"{dest}/{stale}")
      finally:
        shutil.rmtree(folder, ignore_errors=True)
      return dest
    parent, base = os.path.dirname(dest), os.path.basename(dest)
    os.makedirs(parent, exist_ok=True)
    folder = tempfile.mkdtemp(dir=parent, prefix=f".{base}.")
    link, prev = f"{folder}.link", f"{parent}/.{base}.prev"
    try:
      with self.lock:
        self.write(folder, self.stage(tensors), dict(meta or {}))
        cur = os.readlink(dest) if os.path.islink(dest) else None
        stale = os.readlink(prev) if os.path.islink(prev) else None
        os.symlink(os.path.basename(folder), link)
        if os.path.isdir(dest) and not os.path.islink(dest):
          os.rename(dest, f"{folder}.old")
          os.replace(link, dest)
          shutil.rmtree(f"{folder}.old", ignore_errors=True)
        else:
          os.replace(link, dest)
        if cur:
          os.symlink(cur, link)
          os.replace(link, prev)
        folder = ""
      if stale and stale != cur:
        shutil.rmtree(f"{parent}/{stale}", ignore_errors=True)
    finally:
      if folder:
        shutil.rmtree(folder, ignore_errors=True)
      with contextlib.suppress(FileNotFoundError):
        os.unlink(link)
    return dest

  # ***** get *****

  def local(self, uri: str) -> tuple[str, str]:
    """A local directory holding the checkpoint, and its version, which changes when the checkpoint is saved again."""
    dest = self.path(uri)
    if not self.remote:
      real = dest
      with contextlib.suppress(OSError):
        real = os.path.join(os.path.dirname(dest), os.readlink(dest))
      if not os.path.isdir(real):
        raise ValueError(f"no checkpoint at {uri}")
      with os.scandir(real) as it:
        stats = sorted((e.name, s.st_size, s.st_mtime_ns) for e in it for s in [e.stat()])
      if not any(n.endswith(".safetensors") for n, _, _ in stats):
        raise ValueError(f"no checkpoint at {uri}")
      return dest, hashlib.blake2b(repr((real, stats)).encode(), digest_size=8).hexdigest()
    try:
      info = json.loads(self.fs.cat_file(f"{dest}/_latest"))
    except FileNotFoundError:
      raise ValueError(f"no checkpoint at {uri}") from None
    version, names = info["version"], info["files"]
    if not any(n.endswith(".safetensors") for n in names):
      raise ValueError(f"no checkpoint at {uri}")
    folder = f"{self.cache}/{version}"
    if not os.path.isdir(folder):
      os.makedirs(self.cache, exist_ok=True)
      download = tempfile.mkdtemp(dir=self.cache)
      try:
        parallel(lambda n: self.fs.get_file(f"{dest}/{n}", f"{download}/{n}"), names)
        try:
          os.rename(download, folder)
        except OSError:
          if not os.path.isdir(folder):
            raise
      finally:
        shutil.rmtree(download, ignore_errors=True)
    return folder, version

  def weights(self, uri: str) -> list[str]:
    folder = self.local(uri)[0]
    with contextlib.suppress(OSError):
      folder = os.path.join(os.path.dirname(folder), os.readlink(folder))
    return sorted(glob.glob(f"{folder}/*.safetensors"))

  def get(self, uri: str) -> tuple[dict[str, torch.Tensor], dict]:
    import safetensors

    def load(path: str) -> dict[str, torch.Tensor]:
      with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}

    paths = self.weights(uri)
    return {k: v for s in parallel(load, paths) for k, v in s.items()}, header(paths[0])

  def meta(self, uri: str) -> dict:
    return header(self.weights(uri)[0])

  # ***** delete *****

  def delete(self, uri: str) -> None:
    """Delete a checkpoint and every version of it this store kept."""
    dest = self.path(uri)
    if self.remote:
      with contextlib.suppress(FileNotFoundError):
        self.fs.rm(dest, recursive=True)
      return
    parent, base = os.path.split(dest)
    tail = (
      len(parent) + len(base) + 3
    )  # the random suffix of a version folder has no dot, so step-1 never takes step-1.5's
    versions = [p for p in glob.glob(f"{glob.escape(parent)}/.{glob.escape(base)}.*") if "." not in p[tail:]]
    for p in [dest, *versions]:  # the link to the live version, the .prev link, and the folders they name
      if os.path.islink(p):
        os.unlink(p)
      else:
        shutil.rmtree(p, ignore_errors=True)

  def stored(self) -> list[str]:
    """Every checkpoint in the store."""
    if self.remote:
      return [
        f"tinker://{p[len(self.base) + 1 : -len('/_latest')]}" for p in self.fs.glob(f"{self.base}/*/*/*/_latest")
      ]
    found = glob.glob(f"{glob.escape(self.base)}/*/*/*")  # glob skips the dot folders a save writes into
    return [f"tinker://{os.path.relpath(p, self.base)}" for p in found]

  def saved_at(self, uri: str) -> float:
    """When a checkpoint was last saved."""
    if self.remote:
      return self.fs.modified(f"{self.path(uri)}/_latest").timestamp()
    return os.lstat(self.path(uri)).st_mtime
