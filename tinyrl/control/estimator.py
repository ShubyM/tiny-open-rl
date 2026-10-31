"""Estimates memory needs per engine and GPU width before placement, ensuring admissibility.

Activations are left to the dynamic token_budget in trainer.py. No static reserve is added here.
"""

import re

import torch
from huggingface_hub import get_safetensors_metadata
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from tinyrl.workers.trainer import lora_targets

OVERHEAD = int(0.75 * 1024**3)  # measured: 529 MiB CUDA context + ~210 MiB cuBLAS, H100


def estimate(
  repo: str,
  mode: str,
  runs: list[dict] | None = None,
  budget: int = 80 * 1024**3,
  revision: str = "main",
  max_model_len: int = 4096,
) -> tuple[int, int | None]:
  meta = get_safetensors_metadata(repo, revision=revision).parameter_count
  base = sum(c * (1 if k == "BOOL" else int(re.search(r"\d+", k).group()) // 8) for k, c in meta.items())
  cfg = AutoConfig.from_pretrained(repo, revision=revision)
  if mode == "vllm":
    dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    run_bytes = (
      cfg.num_hidden_layers * 2 * max_model_len * getattr(cfg, "num_key_value_heads", cfg.num_attention_heads) * dim * 2
    )
  elif mode == "full":
    base, run_bytes = 0, sum(meta.values()) * 16
  else:
    run_bytes = 0
    with torch.device("meta"):
      lm = AutoModelForCausalLM.from_config(cfg)
    for r in runs or []:
      p = get_peft_model(lm, LoraConfig(r=r.get("rank", 8), target_modules=lora_targets(lm, r)))
      run_bytes += sum(t.numel() for t in p.parameters() if t.requires_grad) * 16
      lm = p.unload()

  tot = base + run_bytes + OVERHEAD
  return tot, next((n for n in (1, 2, 4, 8) if (tot - OVERHEAD) / n + OVERHEAD <= budget), None)
