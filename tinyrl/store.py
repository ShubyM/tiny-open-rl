"""Checkpoint storage behind put(uri) and get(uri).

Tensors are stored in safetensors with JSON metadata in the header, never torch.save pickle. Sampler checkpoints write
the layout PEFT and vLLM load directly (adapter_model.safetensors + adapter_config.json, or model.safetensors +
config.json).

CKPT is a local directory, or a URL like gs://bucket/prefix that every node can reach. With a URL, a checkpoint is
written locally and uploaded, and read by downloading it once into CKPT_CACHE, since engines load from a local path.
"""

from __future__ import annotations

import concurrent.futures
import glob
import json
import os
import pathlib
import shutil
import tempfile
import threading
from typing import TYPE_CHECKING, Any

import fsspec

if TYPE_CHECKING:
  import torch

from tinyrl.helpers import getenv

CKPT = getenv("CKPT", "/tmp/tinyrl")
CKPT_CACHE = getenv("CKPT_CACHE", "/tmp/tinyrl-cache")
SHARD_BYTES = getenv("SHARD_BYTES", 256 * 1024 * 1024)
MAX_SHARDS = 16
MAX_PIN_BYTES = getenv("MAX_PIN_BYTES", 10 * 1024 * 1024 * 1024)


class Store:
  def __init__(self, root: str | None = None, cache: str | None = None):
    self.root = (CKPT if root is None else root).rstrip("/")
    self.remote = "://" in self.root
    self.fs, self.base = fsspec.core.url_to_fs(self.root, auto_mkdir=True)  # a file:// URL makes folders as it writes
    self.cache = CKPT_CACHE if cache is None else cache
    self._pin: Any = None
    self._pin_lock = threading.Lock()

  def _stage(
    self, tensors: dict[str, torch.Tensor], is_cuda=lambda t: t.device.type == "cuda", sync=None
  ) -> dict[str, torch.Tensor]:
    import torch

    if sync is None:
      sync = torch.cuda.synchronize

    cuda_bytes = sum(((v.numel() * v.element_size() + 7) & ~7) for v in tensors.values() if is_cuda(v))
    if cuda_bytes == 0:
      return {
        k: v.detach() if v.device.type == "cpu" and v.is_contiguous() else v.detach().cpu().contiguous()
        for k, v in tensors.items()
      }

    has_cuda = torch.cuda.is_available()
    if 0 < cuda_bytes <= MAX_PIN_BYTES:
      if self._pin is None or self._pin.numel() < cuda_bytes:
        self._pin = None
        self._pin = torch.empty(cuda_bytes, dtype=torch.uint8, pin_memory=has_cuda)
      cpu = {}
      off = 0
      for k, v in tensors.items():
        if is_cuda(v):
          src = v.detach().contiguous()
          nb = src.numel() * src.element_size()
          dst = self._pin[off : off + nb].view(src.dtype).reshape(src.shape)
          dst.copy_(src, non_blocking=True)
          cpu[k] = dst
          off = (off + nb + 7) & ~7
        else:
          cpu[k] = v.detach() if v.is_contiguous() else v.detach().contiguous()
      if has_cuda:
        sync()
      return cpu

    max_elem_bytes = max((v.element_size() for v in tensors.values() if is_cuda(v)), default=1)
    pin_bytes = max(MAX_PIN_BYTES, max_elem_bytes)
    if self._pin is None or self._pin.numel() < pin_bytes:
      self._pin = None
      self._pin = torch.empty(pin_bytes, dtype=torch.uint8, pin_memory=has_cuda)

    cpu = {}
    for k, v in tensors.items():
      if not is_cuda(v):
        cpu[k] = v.detach() if v.is_contiguous() else v.detach().contiguous()
      else:
        src = v.detach().contiguous()
        dst = torch.empty(src.shape, dtype=src.dtype)
        src_flat = src.view(-1)
        dst_flat = dst.view(-1)
        chunk_elems = max(1, MAX_PIN_BYTES // src.element_size())

        for start in range(0, src.numel(), chunk_elems):
          end = min(src.numel(), start + chunk_elems)
          chunk = src_flat[start:end]
          nb = chunk.numel() * src.element_size()
          pin_view = self._pin[:nb].view(src.dtype)
          pin_view.copy_(chunk, non_blocking=True)
          if has_cuda:
            sync()
          dst_flat[start:end].copy_(pin_view)
        cpu[k] = dst
    return cpu

  def path(self, uri: str) -> str:
    return f"{self.base}/{uri.removeprefix('tinker://')}"

  def put(self, uri: str, tensors: dict[str, torch.Tensor], meta: dict | None = None) -> str:
    import safetensors
    import safetensors.torch

    info = dict(meta or {})
    dest = self.path(uri)
    folder = tempfile.mkdtemp(dir=self._mkdir(self.cache if self.remote else os.path.dirname(dest)))
    with self._pin_lock:
      cpu = self._stage(tensors)
      total_bytes = sum(v.numel() * v.element_size() for v in cpu.values())
      meta_hdr = {"meta": json.dumps(info)}

      if not info.get("lora") and total_bytes >= SHARD_BYTES and len(cpu) > 1:
        stem = "model" if info.get("config") else "state"
        n_shards = min(MAX_SHARDS, len(cpu))
        items = list(cpu.items())
        shards = [dict(items[s::n_shards]) for s in range(n_shards)]
        shard_names = [f"{stem}-{s + 1:05d}-of-{n_shards:05d}.safetensors" for s in range(n_shards)]
        names = []
        if cfg := info.get("config"):
          pathlib.Path(f"{folder}/config.json").write_text(json.dumps(cfg))
          index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": {k: shard_names[s] for s, sh in enumerate(shards) for k in sh},
          }
          pathlib.Path(f"{folder}/model.safetensors.index.json").write_text(json.dumps(index))
          names += ["config.json", "model.safetensors.index.json"]
        names += shard_names
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_shards) as ex:
          list(
            ex.map(
              lambda s: safetensors.torch.save_file(shards[s], f"{folder}/{shard_names[s]}", metadata=meta_hdr),
              range(n_shards),
            )
          )
      else:
        names = ["state.safetensors"]
        if cfg := info.get("config"):
          prefix = "adapter_" if info.get("lora") else ""
          names = [f"{prefix}config.json", f"{prefix}model.safetensors"]
          pathlib.Path(f"{folder}/{names[0]}").write_text(json.dumps(cfg))
        safetensors.torch.save_file(cpu, f"{folder}/{names[-1]}", metadata=meta_hdr)
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
      old = f"{folder}.old"
      os.rename(dest, old)
      os.rename(folder, dest)
      shutil.rmtree(old)
    else:
      os.rename(folder, dest)
    if not self.remote or self.fs.protocol == "file" or "file" in self.fs.protocol:
      # A handoff publishes its pointer only after put returns. Persist files and directory entries first.
      for name in names:
        with open(f"{dest}/{name}", "rb") as f:
          os.fsync(f.fileno())
      folder_path = pathlib.Path(dest).resolve()
      for directory in (folder_path, *folder_path.parents):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
          os.fsync(fd)
        finally:
          os.close(fd)
    return dest

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
      download = tempfile.mkdtemp(dir=self._mkdir(self.cache))
      for name, f in files.items():
        self.fs.get_file(f["name"], f"{download}/{name}")
      try:
        os.rename(download, folder)  # whole, so another process on this node never reads half of it
      except OSError:
        shutil.rmtree(download)  # another process got it here first
    return folder, version

  def get(self, uri: str, *, device: str = "cpu") -> tuple[dict[str, torch.Tensor], dict]:
    import safetensors

    paths = sorted(glob.glob(f"{self.local(uri)[0]}/*.safetensors"))
    if not paths:
      raise ValueError(f"no checkpoint at {uri}")

    def load_shard(p):
      with safetensors.safe_open(p, framework="pt", device=device) as f:
        return {k: f.get_tensor(k) for k in f.keys()}

    tensors = {}
    if len(paths) > 1 and device == "cpu":
      with concurrent.futures.ThreadPoolExecutor() as ex:
        for t in ex.map(load_shard, paths):
          tensors.update(t)
    else:
      for p in paths:
        tensors.update(load_shard(p))

    with safetensors.safe_open(paths[0], framework="pt", device=device) as f:
      meta = json.loads((f.metadata() or {}).get("meta", "{}"))
    return tensors, meta

  def meta(self, uri: str) -> dict:
    import safetensors

    with safetensors.safe_open(self._weights(uri), framework="pt", device="cpu") as f:
      return json.loads((f.metadata() or {}).get("meta", "{}"))

  def _weights(self, uri: str) -> str:
    return sorted(glob.glob(f"{self.local(uri)[0]}/*.safetensors"))[0]

  @staticmethod
  def _mkdir(folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    return folder
