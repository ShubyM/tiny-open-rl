"""Checkpoint storage behind put(uri) and get(uri).

Tensors are stored in safetensors with JSON metadata in the header, never torch.save pickle. Sampler checkpoints write
the layout PEFT and vLLM load directly (adapter_model.safetensors + adapter_config.json, or model.safetensors +
config.json, sharded with a model.safetensors.index.json when large).

CKPT is a local directory, or a URL like gs://bucket/prefix that every node can reach. With a URL, a checkpoint is
written locally and uploaded, and read by downloading it once into CKPT_CACHE, since engines load from a local path.
"""

from __future__ import annotations

import concurrent.futures
import glob
import json
import os
import shutil
import tempfile
import threading
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


class Store:
  def __init__(self, root: str | None = None, cache: str | None = None):
    self.root = (CKPT if root is None else root).rstrip("/")
    self.remote = "://" in self.root
    self.fs, self.base = fsspec.core.url_to_fs(self.root, auto_mkdir=True)  # a file:// URL makes folders as it writes
    self.cache = CKPT_CACHE if cache is None else cache
    self.pinned: torch.Tensor | None = None  # reused by every save, since pinning host memory is slow
    self.lock = threading.Lock()  # a save's tensors can be views of self.pinned until it is written

  def path(self, uri: str) -> str:
    return f"{self.base}/{uri.removeprefix('tinker://')}"

  # ***** put *****

  def stage(self, tensors: dict[str, torch.Tensor], is_cuda=lambda t: t.is_cuda) -> dict[str, torch.Tensor]:
    """The tensors on the CPU. GPU tensors go through one pinned buffer, so the copy runs at DMA speed. When they all
    fit, they are views of it, else each streams through it in pieces into its own host tensor."""
    import torch

    gpu = {k for k, v in tensors.items() if is_cuda(v)}
    out = {k: v.detach().contiguous() for k, v in tensors.items() if k not in gpu}
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
    n = min(MAX_SHARDS, len(tensors)) if not lora and total >= SHARD_BYTES else 1
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

    def save(name: str, shard: dict[str, torch.Tensor]) -> None:
      safetensors.torch.save_file(shard, f"{folder}/{name}", metadata={"meta": json.dumps(info)})

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
      list(pool.map(save, weights, shards))
    return [*files, *weights]

  def put(self, uri: str, tensors: dict[str, torch.Tensor], meta: dict | None = None) -> str:
    dest = self.path(uri)
    parent = self.cache if self.remote else os.path.dirname(dest)
    os.makedirs(parent, exist_ok=True)
    folder = tempfile.mkdtemp(dir=parent)
    with self.lock:
      names = self.write(folder, self.stage(tensors), dict(meta or {}))
    if self.remote:
      try:
        stale = {f["name"].rsplit("/", 1)[-1]: f["name"] for f in self.fs.ls(dest, detail=True, refresh=True)}
      except FileNotFoundError:
        stale = {}
      for name in names:  # the weights last, since a reader finds a checkpoint by its weights
        self.fs.put_file(f"{folder}/{name}", f"{dest}/{name}")
        stale.pop(name, None)
      for path in stale.values():
        self.fs.rm_file(path)
      shutil.rmtree(folder)
    elif os.path.exists(dest):
      os.rename(dest, f"{folder}.old")
      os.rename(folder, dest)
      shutil.rmtree(f"{folder}.old")
    else:
      os.rename(folder, dest)
    return dest

  # ***** get *****

  def local(self, uri: str) -> tuple[str, str]:
    """A local directory holding the checkpoint, and its version, which changes when the checkpoint is saved again."""
    try:
      files = {f["name"].rsplit("/", 1)[-1]: f for f in self.fs.ls(self.path(uri), detail=True, refresh=True)}
    except FileNotFoundError:
      files = {}
    weights = sorted(name for name in files if name.endswith(".safetensors"))
    if not weights:
      raise ValueError(f"no checkpoint at {uri}")
    version = fsspec.utils.tokenize([files[w] for w in weights] if len(weights) > 1 else files[weights[0]])
    if not self.remote:
      return self.path(uri), version
    folder = f"{self.cache}/{version}"
    if not os.path.isdir(folder):
      os.makedirs(self.cache, exist_ok=True)
      download = tempfile.mkdtemp(dir=self.cache)
      for name, f in files.items():
        self.fs.get_file(f["name"], f"{download}/{name}")
      try:
        os.rename(download, folder)  # whole, so another process on this node never reads half of it
      except OSError:
        shutil.rmtree(download)  # another process got it here first
    return folder, version

  def weights(self, uri: str) -> list[str]:
    return sorted(glob.glob(f"{self.local(uri)[0]}/*.safetensors"))

  def get(self, uri: str) -> tuple[dict[str, torch.Tensor], dict]:
    import safetensors

    def load(path: str) -> dict[str, torch.Tensor]:
      with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}

    paths = self.weights(uri)
    with concurrent.futures.ThreadPoolExecutor() as pool:
      tensors = {k: v for shard in pool.map(load, paths) for k, v in shard.items()}
    return tensors, header(paths[0])

  def meta(self, uri: str) -> dict:
    return header(self.weights(uri)[0])
