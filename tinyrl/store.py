"""Checkpoint storage behind put(uri) and get(uri).

Tensors are stored in safetensors with JSON metadata in the header, never torch.save pickle. Sampler checkpoints write
the layout PEFT and vLLM load directly (adapter_model.safetensors + adapter_config.json, or model.safetensors +
config.json). Setting SHM caches blobs in local tmpfs (/dev/shm) by an immutable tag in the header so same-node reads
never pull tensor bytes across a network filesystem.
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import shutil
import uuid

import safetensors
import safetensors.torch
import torch

from tinyrl.helpers import getenv

CKPT = getenv("CKPT", "/tmp/tinyrl")
SHM = getenv("SHM", "")


def _relpath(uri: str) -> str:
  parts = uri.removeprefix("tinker://").split("/")
  if len(parts) < 3 or any(p in ("", ".", "..") for p in parts):
    raise ValueError(f"bad checkpoint uri {uri}")
  return "/".join(parts)


def _atomic_write(path: str, write_fn) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
  try:
    write_fn(tmp)
    os.replace(tmp, path)
  except BaseException:
    pathlib.Path(tmp).unlink(missing_ok=True)
    raise


class Store:
  def __init__(self, root: str | None = None, shm: str | None = None):
    self.root = CKPT if root is None else root
    self.shm = SHM if shm is None else shm

  def path(self, uri: str) -> str:
    return f"{self.root}/{_relpath(uri)}"

  def put(self, uri: str, tensors: dict[str, torch.Tensor], meta: dict | None = None) -> str:
    folder, info, tag = self.path(uri), dict(meta or {}), uuid.uuid4().hex
    cpu = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    header = {"meta": json.dumps(info), "tag": tag}
    if cfg := info.get("config"):
      lora = bool(info.get("lora"))
      weights_file = f"{folder}/{'adapter_model' if lora else 'model'}.safetensors"
      cfg_file = f"{folder}/{'adapter_config' if lora else 'config'}.json"
      _atomic_write(cfg_file, lambda tmp: pathlib.Path(tmp).write_text(json.dumps(cfg)))
    else:
      weights_file = f"{folder}/state.safetensors"

    if self.shm:
      cached = f"{self.shm}/{tag}.safetensors"
      _atomic_write(cached, lambda tmp: safetensors.torch.save_file(cpu, tmp, metadata=header))
      _atomic_write(weights_file, lambda tmp: shutil.copyfile(cached, tmp))
    else:
      _atomic_write(weights_file, lambda tmp: safetensors.torch.save_file(cpu, tmp, metadata=header))
    for old in glob.glob(f"{folder}/*.safetensors"):
      if old != weights_file:
        pathlib.Path(old).unlink(missing_ok=True)
    return folder

  def get(self, uri: str, *, device: str = "cpu") -> tuple[dict[str, torch.Tensor], dict]:
    folder = self.path(uri)
    files = sorted(glob.glob(f"{folder}/*.safetensors"))
    if not files:
      raise ValueError(f"no checkpoint at {uri}")
    src = files[0]
    if self.shm:
      with safetensors.safe_open(src, framework="pt", device="cpu") as f:
        tag = (f.metadata() or {}).get("tag", "")
      if tag and os.path.isfile(cached := f"{self.shm}/{tag}.safetensors"):
        src = cached
    with safetensors.safe_open(src, framework="pt", device=device) as f:
      meta = json.loads((f.metadata() or {}).get("meta", "{}"))
      return {k: f.get_tensor(k) for k in f.keys()}, meta
