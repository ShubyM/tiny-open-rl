"""Where base models come from. A model reference is what a client passes as base_model:

  Qwen/Qwen3-0.6B or hf://Qwen/Qwen3-0.6B   the Hugging Face hub, through HF_HOME as before
  gs://bucket/models/qwen3, s3://..., any fsspec URL   a folder holding config.json and *.safetensors
  /mnt/models/qwen3 or file:///mnt/models/qwen3   a folder on every node

local(ref) turns one into something transformers and vLLM can load, fetching a URL once per node into MODEL_CACHE. A reference names fixed weights: publish new weights under a new path. With MODEL_MIRROR set, a hub model is read from MODEL_MIRROR/<org>/<name> when it's there, so nodes need no internet. `python -m tinyrl.storage.models mirror <id>` puts it there.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile

import fsspec

from tinyrl.helpers import getenv

MODEL_CACHE = getenv("MODEL_CACHE", "/tmp/tinyrl-models")
MODEL_MIRROR = getenv("MODEL_MIRROR", "")


def exists(url: str) -> bool:
  fs, path = fsspec.core.url_to_fs(url)
  return fs.exists(path)


def local(ref: str) -> str:
  """A model id or local folder for a model reference, fetched into MODEL_CACHE if it's a URL."""
  ref = ref.removeprefix("hf://")
  if ref.startswith("file://"):
    return ref.removeprefix("file://")
  if "://" not in ref:
    mirrored = f"{MODEL_MIRROR.rstrip('/')}/{ref}"
    if not MODEL_MIRROR or os.path.isabs(ref) or not exists(f"{mirrored}/config.json"):
      return ref  # a hub id or a local folder, which transformers and vLLM load themselves
    ref = mirrored
  folder = f"{MODEL_CACHE}/{hashlib.sha256(ref.encode()).hexdigest()[:16]}"
  if not os.path.isdir(folder):
    os.makedirs(MODEL_CACHE, exist_ok=True)
    fs, path = fsspec.core.url_to_fs(ref)
    download = tempfile.mkdtemp(dir=MODEL_CACHE)
    try:
      fs.get(f"{path.rstrip('/')}/", f"{download}/", recursive=True)
      if not os.path.exists(f"{download}/config.json"):
        raise ValueError(f"no model at {ref}: it has no config.json")
      try:
        os.rename(download, folder)
      except OSError:  # another worker on this node fetched it first
        if not os.path.isdir(folder):
          raise
    finally:
      shutil.rmtree(download, ignore_errors=True)
  return folder


def mirror(model_id: str) -> str:
  """Copy a hub model into MODEL_MIRROR, for nodes without internet. Returns where it went."""
  from huggingface_hub import snapshot_download

  if not MODEL_MIRROR:
    raise ValueError("set MODEL_MIRROR to the folder or bucket to copy models into")
  dest = f"{MODEL_MIRROR.rstrip('/')}/{model_id}"
  fs, path = fsspec.core.url_to_fs(dest)
  fs.put(
    f"{snapshot_download(model_id, allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'])}/",
    f"{path}/",
    recursive=True,
  )
  return dest


if __name__ == "__main__":
  if sys.argv[1:2] != ["mirror"] or len(sys.argv) != 3:
    raise SystemExit("usage: python -m tinyrl.storage.models mirror <hub model id>")
  print(mirror(sys.argv[2]))
