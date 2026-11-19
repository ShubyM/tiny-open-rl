"""Estimates memory needs and GPU width before placement without importing PyTorch or transformers.

Activations are left to the dynamic token_budget in model.py. No static reserve is added here.
"""

import contextlib
import functools
import json
import os
import re

from huggingface_hub import get_safetensors_metadata, hf_hub_download

OVERHEAD = int(0.75 * 1024**3)  # measured: 529 MiB CUDA context + ~210 MiB cuBLAS, H100


def estimate(
  repo: str,
  mode: str,
  budget: int = 80 * 1024**3,
  revision: str = "main",
  max_model_len: int = 4096,
) -> tuple[int, int | None]:
  meta = get_safetensors_metadata(repo, revision=revision).parameter_count
  if mode == "full":
    tot = sum(meta.values()) * 16 + OVERHEAD
  elif mode == "vllm":
    base = sum(c * (1 if k == "BOOL" else int(re.search(r"\d+", k).group()) // 8) for k, c in meta.items())
    with open(hf_hub_download(repo, "config.json", revision=revision)) as f:
      raw = json.load(f)
    cfg = raw.get("text_config") or raw
    dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    heads = cfg.get("num_key_value_heads") or cfg["num_attention_heads"]
    tot = base + cfg["num_hidden_layers"] * 2 * max_model_len * heads * dim * 2 + OVERHEAD
  else:
    raise ValueError(f"unknown mode {mode}")
  return tot, next((n for n in (1, 2, 4, 8) if (tot - OVERHEAD) / n + OVERHEAD <= budget), None)


@functools.lru_cache
def gpus(repo: str, mode: str) -> int:
  with contextlib.suppress(Exception):
    if "/" in repo and "://" not in repo and not os.path.exists(repo):
      return estimate(repo, mode)[1] or 1
  return 1
