"""Estimates model memory and GPU width for full fine-tunes and vLLM before placement."""

import contextlib
import functools
import os
import re

OVERHEAD = int(0.75 * 1024**3)  # measured: 529 MiB CUDA context + ~210 MiB cuBLAS, H100


def estimate(
  repo: str, mode: str, budget: int = 80 * 1024**3, revision: str = "main", max_model_len: int = 4096
) -> tuple[int, int | None]:
  from huggingface_hub import get_safetensors_metadata

  meta = get_safetensors_metadata(repo, revision=revision).parameter_count
  if mode == "full":
    body = sum(meta.values()) * 16
  elif mode == "vllm":
    from transformers import AutoConfig

    base = sum(c * (1 if k == "BOOL" else int(re.search(r"\d+", k).group()) // 8) for k, c in meta.items())
    cfg = AutoConfig.from_pretrained(repo, revision=revision).get_text_config()
    dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    body = base + cfg.num_hidden_layers * 2 * max_model_len * heads * dim * 2
  else:
    raise ValueError(f"unknown mode {mode!r}")
  return body + OVERHEAD, next((n for n in (1, 2, 4, 8) if body / n + OVERHEAD <= budget), None)


@functools.lru_cache
def gpus(repo: str, mode: str) -> int:
  with contextlib.suppress(Exception):
    if "/" in repo and "://" not in repo and not os.path.exists(repo):
      return estimate(repo, mode)[1] or 1
  return 1
