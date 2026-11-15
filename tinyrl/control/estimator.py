"""Default GPU width for full training or sampling; an explicit openrl.gpus overrides it."""

import contextlib
import functools
import os
import re

from huggingface_hub import get_safetensors_metadata
from transformers import AutoConfig

OVERHEAD = int(0.75 * 1024**3)  # measured CUDA context + cuBLAS, H100


@functools.cache
def gpus(repo: str, mode: str) -> int:
  with contextlib.suppress(Exception):
    if "/" in repo and "://" not in repo and not os.path.exists(repo):
      meta = get_safetensors_metadata(repo).parameter_count
      if mode == "full":
        size = sum(meta.values()) * 16
      else:
        cfg = AutoConfig.from_pretrained(repo).get_text_config()
        dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        size = sum(c * (1 if k == "BOOL" else int(re.search(r"\d+", k).group()) // 8) for k, c in meta.items())
        size += cfg.num_hidden_layers * 2 * 4096 * heads * dim * 2
      return next((n for n in (1, 2, 4, 8) if size / n + OVERHEAD <= 80 * 1024**3), 1)
  return 1
