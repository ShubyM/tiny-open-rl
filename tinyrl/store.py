"""Checkpoint storage behind put(uri) and get(uri).

Tensors are stored in safetensors with JSON metadata in the header, never torch.save pickle. Sampler checkpoints write
the layout PEFT and vLLM load directly (adapter_model.safetensors + adapter_config.json, or model.safetensors +
config.json).
"""

from __future__ import annotations

import glob
import json
import os
import pathlib

import safetensors
import safetensors.torch
import torch

from tinyrl.helpers import getenv

CKPT = getenv("CKPT", "/tmp/tinyrl")


class Store:
  def __init__(self, root: str | None = None):
    self.root = CKPT if root is None else root

  def path(self, uri: str) -> str:
    return f"{self.root}/{uri.removeprefix('tinker://')}"

  def file(self, uri: str) -> str:
    folder = self.path(uri)
    files = sorted(glob.glob(f"{folder}/*.safetensors"))
    if not files:
      raise ValueError(f"no checkpoint at {uri}")
    return files[0]

  def put(self, uri: str, tensors: dict[str, torch.Tensor], meta: dict | None = None) -> str:
    folder, info = self.path(uri), dict(meta or {})
    cpu = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    header = {"meta": json.dumps(info)}
    os.makedirs(folder, exist_ok=True)
    if cfg := info.get("config"):
      lora = bool(info.get("lora"))
      weights_file = f"{folder}/{'adapter_model' if lora else 'model'}.safetensors"
      cfg_file = f"{folder}/{'adapter_config' if lora else 'config'}.json"
      pathlib.Path(cfg_file).write_text(json.dumps(cfg))
    else:
      weights_file = f"{folder}/state.safetensors"

    safetensors.torch.save_file(cpu, weights_file, metadata=header)
    return folder

  def get(self, uri: str, *, device: str = "cpu") -> tuple[dict[str, torch.Tensor], dict]:
    src = self.file(uri)
    with safetensors.safe_open(src, framework="pt", device=device) as f:
      meta = json.loads((f.metadata() or {}).get("meta", "{}"))
      return {k: f.get_tensor(k) for k in f.keys()}, meta

  def meta(self, uri: str) -> dict:
    src = self.file(uri)
    with safetensors.safe_open(src, framework="pt", device="cpu") as f:
      return json.loads((f.metadata() or {}).get("meta", "{}"))
