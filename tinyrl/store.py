"""Immutable artifacts; only the coordinator publishes their logical names.

put returns after the blob and directory are fsynced. An old worker can leave an
unreferenced artifact, but cannot overwrite another operation's checkpoint.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
import uuid

import safetensors
import safetensors.torch
import torch

from tinyrl.helpers import getenv, make_directory, sync_directory

CKPT = getenv("CKPT", "/tmp/tinyrl")
SHM = getenv("SHM", "")


def write_file(path: str, contents: bytes) -> None:
  with open(path, "xb") as out:
    out.write(contents)
    out.flush()
    os.fsync(out.fileno())


class Store:
  def __init__(self, root: str | None = None, shm: str | None = None):
    self.root = CKPT if root is None else root
    self.shm = SHM if shm is None else shm
    make_directory(self.root)

  def path(self, ref: str) -> str:
    if len(ref) != 32 or any(c not in "0123456789abcdef" for c in ref):
      raise ValueError(f"bad artifact ref {ref}")
    return f"{self.root}/{ref}.safetensors"

  def put(self, tensors: dict[str, torch.Tensor], meta: dict | None = None, *, ref: str | None = None) -> str:
    ref = ref or uuid.uuid4().hex
    path = self.path(ref)
    fd, temporary = tempfile.mkstemp(prefix=".artifact-", dir=self.root)
    os.close(fd)
    try:
      cpu = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
      safetensors.torch.save_file(cpu, temporary, metadata={"meta": json.dumps(meta or {})})
      with open(temporary, "rb") as src:
        os.fsync(src.fileno())
      os.link(temporary, path)  # atomic; refuses to overwrite an existing ref
      os.unlink(temporary)
      sync_directory(self.root)
    finally:
      pathlib.Path(temporary).unlink(missing_ok=True)
    if self.shm:
      cached_tmp = None
      try:
        make_directory(self.shm)
        fd, cached_tmp = tempfile.mkstemp(prefix=".cache-", dir=self.shm)
        os.close(fd)
        shutil.copyfile(path, cached_tmp)
        os.replace(cached_tmp, f"{self.shm}/{ref}.safetensors")
      except OSError:
        pass  # cache exhaustion cannot fail a checkpoint already durable on disk
      finally:
        if cached_tmp:
          pathlib.Path(cached_tmp).unlink(missing_ok=True)
    return ref

  def get(self, ref: str, *, device: str = "cpu") -> tuple[dict[str, torch.Tensor], dict]:
    durable = self.path(ref)
    paths = [f"{self.shm}/{ref}.safetensors", durable] if self.shm else [durable]
    for path in paths:
      try:
        with safetensors.safe_open(path, framework="pt", device=device) as blob:
          meta = json.loads((blob.metadata() or {}).get("meta", "{}"))
          return {name: blob.get_tensor(name) for name in blob.keys()}, meta
      except (OSError, ValueError, safetensors.SafetensorError):
        if path == durable:
          raise

  def export(self, ref: str, folder: str | None = None) -> str:
    """Materialize an immutable sampler directory, without publishing a URI."""
    blob = self.path(ref)
    with safetensors.safe_open(blob, framework="pt") as source:
      meta = json.loads((source.metadata() or {}).get("meta", "{}"))
    if "config" not in meta or meta.get("optimizer"):
      raise ValueError("only sampler artifacts can be exported")
    parent = f"{self.root}/exports"
    make_directory(parent)
    target = folder or f"{parent}/{ref}"
    if os.path.exists(target):
      marker = pathlib.Path(target, "artifact.json")
      if marker.is_file() and json.loads(marker.read_text()) == {"ref": ref}:
        return target
      raise FileExistsError(f"export destination already exists: {target}")
    target_parent = os.path.dirname(os.path.abspath(target))
    make_directory(target_parent)
    temporary = tempfile.mkdtemp(prefix=".export-", dir=target_parent)
    try:
      name = "adapter_model" if meta.get("lora") else "model"
      config = "adapter_config" if meta.get("lora") else "config"
      os.symlink(os.path.abspath(blob), f"{temporary}/{name}.safetensors")
      write_file(f"{temporary}/{config}.json", json.dumps(meta["config"]).encode())
      write_file(f"{temporary}/artifact.json", json.dumps({"ref": ref}).encode())
      sync_directory(temporary)
      os.rename(temporary, target)
      sync_directory(target_parent)
    finally:
      if os.path.isdir(temporary):
        shutil.rmtree(temporary)
    return target
