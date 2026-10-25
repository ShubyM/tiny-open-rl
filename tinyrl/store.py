"""Immutable safetensors artifacts; only the coordinator publishes logical names."""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import shutil
import tempfile
import uuid

import safetensors
import safetensors.torch
import torch

from tinyrl.helpers import atomic_link, getenv, is_hex, make_directory, sync_directory

CKPT = getenv("CKPT", "/tmp/tinyrl")
SHM = getenv("SHM", "")


def _write_new(path: str, data: bytes) -> None:
  with open(path, "xb") as out:
    out.write(data)
    out.flush()
    os.fsync(out.fileno())


class Store:
  def __init__(self, root: str | None = None, shm: str | None = None):
    self.root = CKPT if root is None else root
    self.shm = SHM if shm is None else shm
    make_directory(self.root)

  def path(self, ref: str) -> str:
    if not is_hex(ref, 32): raise ValueError(f"bad artifact ref {ref}")
    return f"{self.root}/{ref}.safetensors"

  def put(self, tensors: dict[str, torch.Tensor], meta: dict | None = None, *, ref: str | None = None) -> str:
    ref = ref or uuid.uuid4().hex
    path = self.path(ref)
    cpu = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    header = {"meta": json.dumps(meta or {})}
    atomic_link(path, lambda tmp: safetensors.torch.save_file(cpu, tmp, metadata=header), ".artifact-")
    if self.shm:
      tmp = ""
      with contextlib.suppress(OSError):  # cache exhaustion cannot fail a durable checkpoint
        try:
          make_directory(self.shm)
          fd, tmp = tempfile.mkstemp(prefix=".cache-", dir=self.shm)
          os.close(fd)
          shutil.copyfile(path, tmp)
          os.replace(tmp, f"{self.shm}/{ref}.safetensors")
        finally:
          if tmp: pathlib.Path(tmp).unlink(missing_ok=True)
    return ref

  def get(self, ref: str, *, device: str = "cpu") -> tuple[dict[str, torch.Tensor], dict]:
    durable = self.path(ref)
    for path in ([f"{self.shm}/{ref}.safetensors", durable] if self.shm else [durable]):
      try:
        with safetensors.safe_open(path, framework="pt", device=device) as blob:
          meta = json.loads((blob.metadata() or {}).get("meta", "{}"))
          return {name: blob.get_tensor(name) for name in blob.keys()}, meta
      except (OSError, ValueError, safetensors.SafetensorError):
        if path == durable: raise

  def export(self, ref: str, folder: str | None = None) -> str:
    """Materialize an immutable HF/PEFT directory without publishing a logical URI."""
    blob = self.path(ref)
    with safetensors.safe_open(blob, framework="pt") as src:
      meta = json.loads((src.metadata() or {}).get("meta", "{}"))
    if "config" not in meta or meta.get("optimizer"): raise ValueError("only sampler artifacts can be exported")
    target = folder or f"{make_directory(f'{self.root}/exports')}/{ref}"
    if os.path.exists(target):
      marker = pathlib.Path(target, "artifact.json")
      if marker.is_file() and json.loads(marker.read_text()) == {"ref": ref}: return target
      raise FileExistsError(f"export destination already exists: {target}")
    parent = make_directory(os.path.dirname(os.path.abspath(target)))
    tmp = tempfile.mkdtemp(prefix=".export-", dir=parent)
    try:
      stem, cfg = ("adapter_model", "adapter_config") if meta.get("lora") else ("model", "config")
      os.symlink(os.path.abspath(blob), f"{tmp}/{stem}.safetensors")
      for name, obj in ((f"{cfg}.json", meta["config"]), ("artifact.json", {"ref": ref})):
        _write_new(f"{tmp}/{name}", json.dumps(obj).encode())
      sync_directory(tmp)
      os.rename(tmp, target)
      sync_directory(parent)
    finally:
      if os.path.isdir(tmp): shutil.rmtree(tmp)
    return target
