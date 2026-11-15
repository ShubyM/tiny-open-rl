"""Execution eligibility, memory envelopes (active GPU, offloaded CUDA context, parked host RAM), and GPU service/makespan rates."""

from __future__ import annotations

import contextlib
import functools
import os
import re

import torch
from huggingface_hub import get_safetensors_metadata
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from tinyrl.workers.model import lora_targets
from tinyrl.workers.vllm import MAX_LORA_RANK

OVERHEAD = int(0.75 * 1024**3)  # measured: 529 MiB CUDA context + ~210 MiB cuBLAS, H100
CLASS_RATE = {"b200": 1.8, "gb200": 1.8, "h200": 1.15, "h100": 1.0, "a100": 0.5, "l40s": 0.45, "l4": 0.2, "cpu": 1.0}


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
  cfg = AutoConfig.from_pretrained(repo, revision=revision).get_text_config()
  if mode == "vllm":
    dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    run_bytes = cfg.num_hidden_layers * 2 * max_model_len * heads * dim * 2
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


@functools.lru_cache
def gpus(repo: str, mode: str) -> int:
  with contextlib.suppress(Exception):
    if "/" in repo and "://" not in repo and not os.path.exists(repo):
      return estimate(repo, mode)[1] or 1
  return 1


def class_rate(gpu_class: str, custom: float = 0.0) -> float:
  if custom > 0:
    return float(custom)
  low = (gpu_class or "").lower()
  return next((v for k, v in CLASS_RATE.items() if k in low), 1.0)


def service_rate(width: int, gpu_class: str = "", custom: float = 0.0) -> float:
  """Fair-service burn rate in normalized GPU-seconds per elapsed second."""
  return max(1, width) * class_rate(gpu_class, custom)


def eligible_bundles(
  gpu_ids: tuple[int, ...], width: int, explicit: list[list[int]] | None = None
) -> list[tuple[int, ...]]:
  """Admissible intra-node GPU bundles of size `width`."""
  if width < 1 or len(gpu_ids) < width:
    return []
  if explicit:
    live = set(gpu_ids)
    exact = [tuple(b) for b in explicit if len(b) == width and set(b) <= live]
    if exact:
      return exact
    return [tuple(b[:width]) for b in explicit if len(b) > width and set(b[:width]) <= live]
  if width == 1:
    return [(g,) for g in gpu_ids]
  aligned = [gpu_ids[i : i + width] for i in range(0, len(gpu_ids) - width + 1, width)]
  return aligned or [gpu_ids[:width]]


def compatible(profile: dict, gpu_class: str, gpu_ids: tuple[int, ...], bundles: list[list[int]] | None = None) -> bool:
  """Whether a node satisfies a job's execution profile (GPU class allowlist, adapter limits, and intra-node bundle width)."""
  allowed = [c.lower() for c in profile.get("gpu_classes") or ()]
  if allowed and (gpu_class or "").lower() not in allowed:
    return False
  if profile.get("role") == "sampler" and profile.get("lora") and int(profile.get("rank", 0)) > MAX_LORA_RANK:
    return False
  return bool(eligible_bundles(gpu_ids, int(profile.get("gpus", 1)), bundles))


def transfer_seconds(
  version_info: dict | None, peer: str, cached: set[str], peer_bps: float, bucket_bps: float
) -> float:
  """Estimated time to stage a version onto a node from its cheapest reachable source."""
  if not version_info or version_info.get("version") in cached:
    return 0.0
  size = float(version_info.get("size_bytes") or 0)
  locs = [p for p in version_info.get("locations") or () if p and p != peer]
  if locs:
    return size / max(1.0, peer_bps)
  if version_info.get("durable"):
    return size / max(1.0, bucket_bps)
  return 1e9  # unreachable non-durable version


def exec_seconds(tokens: int, width: int, gpu_class: str = "", custom: float = 0.0) -> float:
  tps = 16_000.0 * service_rate(width, gpu_class, custom)
  return max(0.05, float(max(1, tokens)) / tps)
