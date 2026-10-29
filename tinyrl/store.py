"""Checkpoint storage behind put(uri) and get(uri).

Tensors are stored in safetensors with JSON metadata in the header, never torch.save pickle. Sampler checkpoints write
the layout PEFT and vLLM load directly (adapter_model.safetensors + adapter_config.json, or model.safetensors +
config.json).

CKPT is a local directory, or a URL like gs://bucket/prefix that every node can reach. With a URL, a checkpoint is
written locally and uploaded, and read by downloading it once into CKPT_CACHE, since engines load from a local path.
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import shutil
import tempfile

import fsspec
import safetensors
import safetensors.torch
import torch

from tinyrl.helpers import getenv

CKPT = getenv("CKPT", "/tmp/tinyrl")
CKPT_CACHE = getenv("CKPT_CACHE", "/tmp/tinyrl-cache")


class Store:
  def __init__(self, root: str | None = None, cache: str | None = None):
    self.root = (CKPT if root is None else root).rstrip("/")
    self.remote = "://" in self.root
    self.fs, self.base = fsspec.core.url_to_fs(self.root, auto_mkdir=True)  # a file:// URL makes folders as it writes
    self.cache = CKPT_CACHE if cache is None else cache

  def path(self, uri: str) -> str:
    return f"{self.base}/{uri.removeprefix('tinker://')}"

  def put(self, uri: str, tensors: dict[str, torch.Tensor], meta: dict | None = None) -> str:
    info = dict(meta or {})
    folder = tempfile.mkdtemp(dir=self._mkdir(self.cache)) if self.remote else self._mkdir(self.path(uri))
    cpu = {
      k: v.detach() if v.device.type == "cpu" and v.is_contiguous() else v.detach().cpu().contiguous()
      for k, v in tensors.items()
    }
    names = ["state.safetensors"]
    if cfg := info.get("config"):
      prefix = "adapter_" if info.get("lora") else ""
      names = [f"{prefix}config.json", f"{prefix}model.safetensors"]
      pathlib.Path(f"{folder}/{names[0]}").write_text(json.dumps(cfg))
    safetensors.torch.save_file(cpu, f"{folder}/{names[-1]}", metadata={"meta": json.dumps(info)})
    if self.remote:
      for name in names:  # the weights last, since a reader finds a checkpoint by its weights
        self.fs.put_file(f"{folder}/{name}", f"{self.path(uri)}/{name}")
      shutil.rmtree(folder)
    return self.path(uri)

  def local(self, uri: str) -> tuple[str, str]:
    """A local directory holding the checkpoint, and its version, which changes when the checkpoint is saved again."""
    try:
      files = {f["name"].rsplit("/", 1)[-1]: f for f in self.fs.ls(self.path(uri), detail=True, refresh=True)}
    except FileNotFoundError:
      files = {}
    weights = sorted(name for name in files if name.endswith(".safetensors"))
    if not weights:
      raise ValueError(f"no checkpoint at {uri}")
    version = fsspec.utils.tokenize(files[weights[0]])
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
    with safetensors.safe_open(self._weights(uri), framework="pt", device=device) as f:
      return {k: f.get_tensor(k) for k in f.keys()}, json.loads((f.metadata() or {}).get("meta", "{}"))

  def meta(self, uri: str) -> dict:
    with safetensors.safe_open(self._weights(uri), framework="pt", device="cpu") as f:
      return json.loads((f.metadata() or {}).get("meta", "{}"))

  def _weights(self, uri: str) -> str:
    return sorted(glob.glob(f"{self.local(uri)[0]}/*.safetensors"))[0]

  @staticmethod
  def _mkdir(folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    return folder
