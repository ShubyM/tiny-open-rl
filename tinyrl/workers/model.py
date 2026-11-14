"""The model a trainer serves: the losses, token budget batching, and one base model with its LoRA adapters or full fine-tune, which park in host memory between turns."""

from __future__ import annotations

import contextlib
import glob
import hashlib
import logging
import math
import os
from collections.abc import Callable

import numpy as np
import torch
import torch.distributed as dist
import transformers
from peft import LoraConfig
from peft.helpers import disable_input_dtype_casting
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from tinyrl.api import wire
from tinyrl.helpers import getenv
from tinyrl.storage import models
from tinyrl.storage.store import aligned, header, nbytes, pinned_bytes, read_manifest

LigerFusedLinearCrossEntropyLoss = apply_liger = None
with contextlib.suppress(ImportError):
  logging.getLogger("liger_kernel").setLevel(logging.ERROR)
  from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
  from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance as apply_liger

TOKEN_BUDGET = getenv("TOKEN_BUDGET", 0)
MAX_TOKEN_BUDGET = 16384
CHUNK = getenv("CHUNK", 1024)
GRAD_CKPT = getenv("GRAD_CKPT", 1)

# ***** losses *****


def cross_entropy(logprobs, weights, sampler_logprobs, advantages, config):
  return -logprobs * weights


def importance_sampling(logprobs, weights, sampler_logprobs, advantages, config):
  ratio = (logprobs - sampler_logprobs).clamp(-20, 20).exp()
  return torch.nan_to_num(-ratio * advantages * weights, nan=0.0, posinf=0.0, neginf=0.0)


def ppo(logprobs, weights, sampler_logprobs, advantages, config):
  if bad := set(config) - {"clip_range", "kl_coeff", "clip_low_threshold", "clip_high_threshold"}:
    raise ValueError(f"unknown config keys: {bad}")
  c, kl = config.get("clip_range", 0.2), config.get("kl_coeff", 0.0)
  low, high = config.get("clip_low_threshold", 1 - c), config.get("clip_high_threshold", 1 + c)
  log_ratio = (logprobs - sampler_logprobs).clamp(-20, 20)
  ratio = log_ratio.exp()
  objective = torch.min(ratio * advantages, ratio.clamp(low, high) * advantages)
  return -(objective - kl * ((ratio - 1) - log_ratio)) * weights


LOSSES = {"cross_entropy": cross_entropy, "importance_sampling": importance_sampling, "ppo": ppo}


# ***** forward pass *****


def should_checkpoint_layers(lm: PreTrainedModel, lora: bool) -> bool:
  if "GRAD_CKPT" in os.environ:
    return bool(int(os.environ["GRAD_CKPT"]))
  if not GRAD_CKPT:
    return False
  if lora and lm.device.type == "cuda" and torch.cuda.get_device_properties(lm.device).total_memory >= 64 * 1024**3:
    return False
  return True


def token_budget(lm: PreTrainedModel) -> int:
  if TOKEN_BUDGET:
    return TOKEN_BUDGET
  if lm.device.type != "cuda":
    return MAX_TOKEN_BUDGET
  free, _ = torch.cuda.mem_get_info(lm.device)
  free += torch.cuda.memory_reserved(lm.device) - torch.cuda.memory_allocated(lm.device)
  config = lm.config.get_text_config()
  scale = 6 if getattr(lm, "is_gradient_checkpointing", False) else 64
  bytes_per_token = scale * config.num_hidden_layers * config.hidden_size * lm.dtype.itemsize
  return min(MAX_TOKEN_BUDGET, int(free / 2 / bytes_per_token))


def batches(lengths: list[int], budget: int) -> list[list[int]]:
  passes: list[list[int]] = []
  for i in sorted(range(len(lengths)), key=lambda i: lengths[i]):
    if passes:
      current = passes[-1]
      padded = lengths[i] * (len(current) + 1)
      real = lengths[i] + sum(lengths[j] for j in current)
      if padded <= budget and padded - real <= padded / 4:
        current.append(i)
        continue
    passes.append([i])
  return passes


def pad(rows: list[np.ndarray], lengths: list[int], dtype, device: torch.device) -> torch.Tensor:
  width = -(-max(lengths) // 128) * 128 if device.type == "cuda" else max(lengths)
  out = np.zeros((len(rows), width), dtype)
  for i, (row, n) in enumerate(zip(rows, lengths)):
    out[i, :n] = row[:n]
  return torch.from_numpy(out).to(device)


def batch_tensors(data: list[wire.Datum], lengths: list[int], device: torch.device) -> dict[str, torch.Tensor]:
  def column(name: str, default: float, dtype) -> torch.Tensor:
    return pad([d.inputs.get(name, np.full(n, default)) for d, n in zip(data, lengths)], lengths, dtype, device)

  return {
    "ids": pad([d.tokens for d in data], lengths, np.int64, device),
    "mask": pad([np.ones(n) for n in lengths], lengths, np.int64, device),
    "targets": column("target_tokens", 0, np.int64),
    "weights": column("weights", 1.0, np.float32),
    "sampler_logprobs": column("logprobs", 0.0, np.float32),
    "advantages": column("advantages", 0.0, np.float32),
  }


def target_logprobs(lm: PreTrainedModel, ids: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
  out = lm(ids, attention_mask=mask, use_cache=False, logits_to_keep=1, output_hidden_states=True)
  hidden = out.hidden_states[-1].flatten(0, 1)
  head = lm.get_output_embeddings()
  softcap = getattr(lm.config.get_text_config(), "final_logit_softcapping", None)

  if lm.device.type == "cuda" and type(head) is torch.nn.Linear and LigerFusedLinearCrossEntropyLoss:
    w, b = head.weight.to(hidden.dtype), head.bias.to(hidden.dtype) if head.bias is not None else None
    loss = LigerFusedLinearCrossEntropyLoss(reduction="none", softcap=softcap)(w, hidden, targets.flatten(), b)
    return -loss.view(targets.shape)

  def chunk_logprobs(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    logits = head(h)
    if softcap:
      logits = torch.tanh(logits / softcap) * softcap
    logits = logits.float()
    return logits.gather(-1, t[:, None])[:, 0] - logits.logsumexp(-1)

  chunks = zip(hidden.split(CHUNK), targets.flatten().split(CHUNK))
  return torch.cat([checkpoint(chunk_logprobs, h, t, use_reentrant=False) for h, t in chunks]).view(targets.shape)


# ***** model *****


def local_tensor(t: torch.Tensor) -> torch.Tensor:
  return t._local_tensor if isinstance(t, DTensor) else t


def full_tensor(t: torch.Tensor, cpu_group=None) -> torch.Tensor:
  """Materialize a DTensor on rank 0 without requiring GPU when its local shard is parked on CPU."""
  if not isinstance(t, DTensor):
    return t
  local = t._local_tensor
  if local.is_cuda or not dist.is_initialized():
    return t.full_tensor()
  placement = t.placements[0]
  if not hasattr(placement, "dim"):
    return local.detach()
  world, rank, dim = dist.get_world_size(), dist.get_rank(), placement.dim
  local = local.detach().contiguous()
  shapes: list = [None] * world
  dist.all_gather_object(shapes, tuple(local.shape), group=cpu_group)
  if rank == 0:
    parts = [local]
    for r in range(1, world):
      buf = torch.empty(shapes[r], dtype=local.dtype)
      dist.recv(buf, src=r, group=cpu_group)
      parts.append(buf)
    return torch.cat(parts, dim=dim)
  dist.send(local, dst=0, group=cpu_group)
  return local


def like_param(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
  return distribute_tensor(t.to(p.device), p.device_mesh, p.placements) if isinstance(p, DTensor) else t.to(p.device)


def slice_for_param(slc, full_shape: tuple[int, ...], p: torch.Tensor) -> torch.Tensor:
  """Read only this rank's local shard directly from a safetensors slice object."""
  if not isinstance(p, DTensor):
    return slc[:]
  placement = p.placements[0]
  if not hasattr(placement, "dim"):
    return slc[:]
  dim = placement.dim
  world, rank = dist.get_world_size(), dist.get_rank()
  size, offset = placement._local_shard_size_and_offset(full_shape[dim], world, rank)
  idx = [slice(None)] * len(full_shape)
  idx[dim] = slice(offset, offset + size)
  return slc[tuple(idx)]


def load(name: str) -> PreTrainedModel:
  name = models.local(name)
  arch = (AutoConfig.from_pretrained(name).architectures or [""])[0]
  cls = getattr(transformers, arch, AutoModelForCausalLM)
  if torch.cuda.is_available() and not dist.is_initialized():
    return cls.from_pretrained(name, dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()})
  return cls.from_pretrained(name, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)


def lora_targets(lm: PreTrainedModel, config: dict) -> list[str]:
  if explicit := config.get("target_modules"):
    mods = {
      n for n, m in ((n, getattr(m, "base_layer", m)) for n, m in lm.named_modules()) if isinstance(m, torch.nn.Linear)
    }
    if all(t in mods for t in explicit):
      return list(explicit)
  targets = ["q_proj", "k_proj", "v_proj", "o_proj"] if config.get("train_attn", True) else []
  targets += ["gate_proj", "up_proj", "down_proj"] if config.get("train_mlp", True) else []
  tied = getattr(lm.config.get_text_config(), "tie_word_embeddings", False)
  if config.get("train_unembed", True) and not tied:
    targets.append("lm_head")
  mods = ((n, getattr(m, "base_layer", m)) for n, m in lm.named_modules())
  return [n for n, m in mods if n.rsplit(".", 1)[-1] in targets and isinstance(m, torch.nn.Linear)]


def _seeded_rng_state(seed: int) -> dict[str, torch.Tensor]:
  prev_cpu = torch.get_rng_state()
  prev_cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
  try:
    torch.manual_seed(int(seed) & ((1 << 63) - 1))
    state = {"cpu": torch.get_rng_state().clone()}
    if torch.cuda.is_available():
      state["cuda"] = torch.cuda.get_rng_state().clone()
    return state
  finally:
    torch.set_rng_state(prev_cpu)
    if prev_cuda is not None:
      torch.cuda.set_rng_state(prev_cuda)


class Model:
  """A base model and the runs training on it: a LoRA adapter per run, or the whole model for one full fine-tune."""

  def __init__(self, load: Callable[[str], PreTrainedModel] = load):
    self.load, self.lm, self.base, self.lora, self.dtype = load, None, "", True, torch.float32
    self.params: dict[str, dict[str, torch.nn.Parameter]] = {}
    self.optimizers: dict[str, torch.optim.AdamW] = {}
    self.configs: dict[str, dict] = {}
    self.rng: dict[str, dict[str, torch.Tensor]] = {}
    self.pinned: dict[int, torch.Tensor] = {}
    self.parked, self.dirty, self.needs_recovery = False, set(), set()
    self._cpu_group = None

  def _ensure_cpu_group(self):
    if dist.is_initialized() and dist.get_backend() != "gloo" and self._cpu_group is None:
      self._cpu_group = dist.new_group(backend="gloo")
    return self._cpu_group

  @contextlib.contextmanager
  def _run_rng(self, run: str):
    """Isolate a job's CPU/CUDA random stream so interleaving another job on a shared trainer cannot perturb it."""
    prev_cpu = torch.get_rng_state()
    prev_cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    if state := self.rng.get(run):
      torch.set_rng_state(state["cpu"])
      if prev_cuda is not None and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])
    try:
      yield
    finally:
      saved = {"cpu": torch.get_rng_state().clone()}
      if prev_cuda is not None:
        saved["cuda"] = torch.cuda.get_rng_state().clone()
      self.rng[run] = saved
      torch.set_rng_state(prev_cpu)
      if prev_cuda is not None:
        torch.cuda.set_rng_state(prev_cuda)

  def create_run(self, run: str, base: str, rank: int, config: dict) -> None:
    if self.lm is None:
      self.lm, self.base, self.lora = self.load(base), base, rank > 0
      if hasattr(inner := getattr(self.lm, "model", None), "language_model"):
        for child in [c for c, _ in inner.named_children() if c != "language_model"]:
          setattr(inner, child, None)
      self.dtype = self.lm.dtype
      if not self.lora:
        self.lm.float()
      if should_checkpoint_layers(self.lm, self.lora):
        self.lm.gradient_checkpointing_enable()
        if self.lora:
          self.lm.enable_input_require_grads()
      if dist.is_initialized():
        self._ensure_cpu_group()
        mp = MixedPrecisionPolicy(reduce_dtype=torch.float32)
        for layer in [m for m in self.lm.modules() if type(m).__name__ in (self.lm._no_split_modules or ())]:
          fully_shard(layer, mp_policy=mp)
        fully_shard(self.lm, mp_policy=mp)
      if self.lm.device.type == "cuda" and apply_liger:
        apply_liger(model=self.lm, cross_entropy=False, fused_linear_cross_entropy=False)

    if base != self.base or (rank > 0) != self.lora:
      raise ValueError(f"this trainer does {'lora' if self.lora else 'full'} fine-tuning of {self.base}")

    cfg = dict(config or {})
    self.configs[run] = cfg
    if cfg.get("seed") is not None:
      self.rng[run] = _seeded_rng_state(int(cfg["seed"]))
    else:
      self.rng[run] = _seeded_rng_state(int(torch.randint(0, 2**62, (1,)).item()))

    self.dirty.discard(run)
    self.needs_recovery.discard(run)
    if not self.lora:
      if self.params:
        raise ValueError(f"a full fine-tune has its trainer to itself, and {next(iter(self.params))} is on this one")
      self.params[run] = {name: p.requires_grad_() for name, p in self.lm.named_parameters()}
      return

    alpha, drop = cfg.get("lora_alpha", 16), cfg.get("lora_dropout", 0.05)
    adapter = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=drop, target_modules=lora_targets(self.lm, cfg))
    with self._run_rng(run), torch.device(self.lm.device):
      self.lm.add_adapter(adapter, adapter_name=run)
    infix = f".{run}."
    self.params[run] = {name.replace(infix, "."): p for name, p in self.lm.named_parameters() if infix in name}
    for p in self.params[run].values():
      p.data = p.data.float()
      if dist.is_initialized():
        dist.broadcast(p.data, src=0)

  def activate_run(self, run: str) -> PreTrainedModel:
    if self.lm is None or run not in self.params:
      raise KeyError(f"model {run} is not on this trainer, it was deleted or its trainer restarted")
    if self.lora:
      self.lm.set_adapter(run)
    return self.lm

  def forward_backward(self, run: str, data: list[wire.Datum], loss_fn: str, config: dict, forward_only: bool):
    if loss_fn not in LOSSES:
      raise ValueError(f"unknown loss_fn {loss_fn}, this trainer has {', '.join(LOSSES)}")

    lm = self.activate_run(run).train(not forward_only)
    lengths = [min(len(d.tokens), len(d.inputs["target_tokens"])) for d in data]
    logprobs: list[np.ndarray] = [np.zeros(0, np.float32)] * len(data)
    total, accumulating = 0.0, False
    world, rank = (dist.get_world_size(), dist.get_rank()) if dist.is_initialized() else (1, 0)
    mine = sorted(range(len(lengths)), key=lambda i: lengths[i])[rank::world]
    passes = [[mine[j] for j in p] for p in batches([lengths[i] for i in mine], token_budget(lm))]
    if world > 1:
      dist.all_gather_object(counts := [0] * world, len(passes))
    steps = max(counts) if world > 1 else len(passes)

    gpu = lm.device.type == "cuda"
    try:
      with self._run_rng(run), torch.set_grad_enabled(not forward_only), disable_input_dtype_casting(lm, active=gpu):
        for step in range(steps):
          batch = passes[step] if step < len(passes) else []
          seq = [data[i] for i in batch] or [wire.Datum(np.zeros(1, np.int32), {"weights": np.zeros(1, np.float32)})]
          x = batch_tensors(seq, [lengths[i] for i in batch] or [1], lm.device)
          with torch.autocast(lm.device.type, dtype=torch.bfloat16, enabled=gpu):
            lp = target_logprobs(lm, x["ids"], x["mask"], x["targets"])
          loss = LOSSES[loss_fn](lp, x["weights"], x["sampler_logprobs"], x["advantages"], config).sum()
          if not forward_only:
            accumulating = True
            (loss * world).backward()
          total += loss.item()

          for row, i in enumerate(batch):
            values = lp[row, : lengths[i]].detach().float()
            logprobs[i] = torch.nan_to_num(values, nan=0.0).clamp(-9999, 9999).cpu().numpy()

      if world > 1:
        for p in self.params[run].values() if self.lora and not forward_only else ():
          if p.grad is not None:
            dist.all_reduce(p.grad.div_(world))
        dist.all_gather_object(gathered := [None] * world, (total, [(i, logprobs[i]) for i in mine]))
        total = sum(t for t, _ in gathered)
        for i, lp_i in (x for _, items in gathered for x in items):
          logprobs[i] = lp_i
      return logprobs, {"loss:sum": total, "loss:mean": total / max(1, len(data))}
    except Exception:
      if accumulating:
        self.needs_recovery.add(run)
        for p in self.params[run].values():
          p.grad = None
      raise

  def adam_for_run(self, run: str) -> torch.optim.AdamW:
    if run not in self.optimizers:
      params = list(self.params[run].values())
      self.optimizers[run] = torch.optim.AdamW(params, fused=bool(params and params[0].is_cuda))
    return self.optimizers[run]

  def optim_step(self, run: str, adam: dict) -> dict[str, float]:
    if run in self.needs_recovery:
      self.needs_recovery.remove(run)
      raise RuntimeError("a forward_backward failed after it began; accumulated gradients were discarded")
    self.activate_run(run)
    optimizer = self.adam_for_run(run)

    for g in optimizer.param_groups:
      g["lr"], g["eps"] = adam.get("learning_rate", 1e-4), adam.get("eps", 1e-12)
      g["weight_decay"], g["betas"] = adam.get("weight_decay", 0.0), (adam.get("beta1", 0.9), adam.get("beta2", 0.95))

    max_norm = adam.get("grad_clip_norm") or math.inf
    with self._run_rng(run):
      grad_norm = torch.nn.utils.clip_grad_norm_(self.params[run].values(), max_norm).item()
      if not self.lora and torch.cuda.is_available() and any(not t.is_cuda for t in self.adam_moments()):
        torch.cuda.empty_cache()
        self.move_tensors(self.adam_moments(), "cuda")
      optimizer.step()
      optimizer.zero_grad()
      if not self.lora and not self.moments_fit():
        self.move_tensors(self.adam_moments(), "cpu")
    self.dirty.add(run)
    return {"grad_norm:mean": grad_norm}

  def _run_metadata(self, run: str, optimizer: bool) -> dict:
    adapter = self.lm.peft_config[run] if self.lora else None
    cfg = self.configs.get(run, {})
    target_modules = sorted(adapter.target_modules) if adapter else []
    targets = sorted({t.rsplit(".", 1)[-1] for t in target_modules}) if adapter else []
    world = dist.get_world_size() if dist.is_initialized() else 1
    return {
      "optimizer": optimizer,
      "base": self.base,
      "lora": self.lora,
      "rank": adapter.r if adapter else 0,
      "lora_alpha": adapter.lora_alpha if adapter else 0,
      "lora_dropout": float(adapter.lora_dropout) if adapter else 0.0,
      "target_modules": target_modules,
      "targets": targets,
      "train_attn": bool(cfg.get("train_attn", True)) if self.lora else False,
      "train_mlp": bool(cfg.get("train_mlp", True)) if self.lora else False,
      "train_unembed": bool(cfg.get("train_unembed", True)) if self.lora else False,
      "seed": cfg.get("seed"),
      "needs_recovery": run in self.needs_recovery,
      "profile": {
        "dtype": str(self.dtype).removeprefix("torch."),
        "world_size": world,
        "layout": "fsdp2" if dist.is_initialized() else "single",
        "lora": self.lora,
      },
    }

  def training_state(self, run: str) -> tuple[dict[str, torch.Tensor], dict]:
    """A run's portable training checkpoint: fp32 master weights, AdamW state, accumulated grads, and per-rank RNG."""
    self.activate_run(run)
    cpu_group = self._cpu_group if self.parked else None
    tensors = {name: full_tensor(p.detach(), cpu_group) for name, p in self.params[run].items()}
    optim = self.optimizers.get(run)
    for name, p in self.params[run].items():
      if p.grad is not None:
        tensors[f"{name}.grad"] = full_tensor(p.grad.detach(), cpu_group)
      if optim and p in optim.state:
        for k in ("exp_avg", "exp_avg_sq", "step"):
          tensors[f"{name}.{k}"] = full_tensor(optim.state[p][k].detach(), cpu_group)

    # Save per-rank RNG state as uint8 safetensors (no pickle)
    local_rng = self.rng.get(run)
    rng_payload = {k: v.to(dtype=torch.uint8, device="cpu").tolist() for k, v in (local_rng or {}).items()}
    if dist.is_initialized():
      gathered: list = [None] * dist.get_world_size()
      dist.all_gather_object(gathered, rng_payload, group=cpu_group)
    else:
      gathered = [rng_payload]
    for r, entry in enumerate(gathered):
      for dev_k, raw in (entry or {}).items():
        tensors[f"rng.{dev_k}.{r}"] = torch.tensor(raw, dtype=torch.uint8)

    return tensors, self._run_metadata(run, optimizer=True)

  def sampler_weights(self, run: str, source_version: str | None = None) -> tuple[dict[str, torch.Tensor], dict]:
    """Sampler weights in the base model's compute dtype (bf16 on GPU), plus PEFT/vLLM config and provenance."""
    lm = self.activate_run(run)
    cpu_group = self._cpu_group if self.parked else None
    prefix = "base_model.model." if self.lora else ""
    tensors = {
      f"{prefix}{name}": full_tensor(p.detach().to(self.dtype), cpu_group) for name, p in self.params[run].items()
    }
    if self.lora:
      cfg = {k: sorted(v) if isinstance(v, set) else v for k, v in lm.peft_config[run].to_dict().items()}
    else:
      cfg = lm.config.to_dict()
    untouched = not self.lora and run not in self.dirty
    meta = self._run_metadata(run, optimizer=False) | {
      "untouched": untouched,
      "config": cfg,
      "source_weights_version": source_version,
      "export_profile": {"dtype": str(self.dtype).removeprefix("torch."), "lora": self.lora},
    }
    return tensors, meta

  def _restore_rng(
    self, run: str, meta: dict, load_tensor: Callable[[str], torch.Tensor | None], restore: bool
  ) -> None:
    if not restore:
      return
    world, rank = (dist.get_world_size(), dist.get_rank()) if dist.is_initialized() else (1, 0)
    src_world = (meta.get("profile") or {}).get("world_size", 1)
    cpu_t = load_tensor(f"rng.cpu.{rank}")
    if cpu_t is not None and src_world == world:
      state = {"cpu": cpu_t.to(dtype=torch.uint8, device="cpu").clone()}
      if torch.cuda.is_available() and (cuda_t := load_tensor(f"rng.cuda.{rank}")) is not None:
        state["cuda"] = cuda_t.to(dtype=torch.uint8, device="cpu").clone()
      self.rng[run] = state
    elif src_world != world:
      # Deterministic rank-layout remapping when destination world_size differs from checkpoint profile.
      digest = hashlib.sha256(f"{meta.get('seed', 0)}:{rank}:{meta.get('completed_id', '')}".encode()).digest()
      self.rng[run] = _seeded_rng_state(int.from_bytes(digest[:8], "little"))

  def _apply_restore(
    self,
    run: str,
    meta: dict,
    shapes: dict[str, tuple[int, ...]],
    optimizer: bool,
    gradients: bool,
    load_full: Callable[[str], torch.Tensor | None],
    copy_weight: Callable[[str, torch.Tensor], None],
    load_like: Callable[[str, torch.Tensor, bool], torch.Tensor],
  ) -> None:
    self.activate_run(run)
    if optimizer and not meta.get("optimizer"):
      raise ValueError("checkpoint has no optimizer state")
    suffixes = (".exp_avg", ".exp_avg_sq", ".step", ".grad")
    weights = {
      k.removeprefix("base_model.model."): k for k in shapes if not k.startswith("rng.") and not k.endswith(suffixes)
    }
    if not self.lora and "lm_head.weight" not in self.params[run]:
      weights.pop("lm_head.weight", None)
    if weights.keys() != self.params[run].keys():
      raise ValueError("checkpoint does not fit this model")
    req_opt = optimizer and any(k.endswith(".step") for k in shapes)
    for n, p in self.params[run].items():
      if shapes[weights[n]] != tuple(p.shape) or (
        req_opt and not all(f"{n}.{s}" in shapes for s in ("exp_avg", "exp_avg_sq", "step"))
      ):
        raise ValueError(f"bad: {n}")

    for name, p in self.params[run].items():
      copy_weight(weights[name], p)
      p.grad = load_like(f"{name}.grad", p, False) if gradients and f"{name}.grad" in shapes else None

    self.needs_recovery.add(run) if gradients and meta.get("needs_recovery") else self.needs_recovery.discard(run)
    self._restore_rng(run, meta, load_full, restore=gradients or optimizer)
    self.pinned.clear()
    self.optimizers.pop(run, None)
    if req_opt:
      opt = self.adam_for_run(run)
      for name, p in self.params[run].items():
        step_t = load_full(f"{name}.step")
        opt.state[p] = {
          "exp_avg": load_like(f"{name}.exp_avg", p, True),
          "exp_avg_sq": load_like(f"{name}.exp_avg_sq", p, True),
          "step": step_t.to(p.device if opt.defaults.get("fused") else "cpu", torch.float32),
        }
      if not self.lora:
        self.move_tensors(self.adam_moments(), "cpu")
    self.dirty.add(run)

  def restore_run(
    self, run: str, tensors: dict[str, torch.Tensor], meta: dict, optimizer: bool, gradients: bool = False
  ) -> None:
    self._apply_restore(
      run,
      meta,
      {k: tuple(v.shape) for k, v in tensors.items()},
      optimizer,
      gradients,
      tensors.get,
      lambda k, p: p.data.copy_(like_param(tensors[k], p)),
      lambda k, p, flt: like_param(tensors[k].float() if flt else tensors[k], p),
    )

  def restore_from_folder(self, run: str, folder: str, optimizer: bool, gradients: bool = False) -> None:
    """Zero-pickle multi-GPU restore: every rank opens local safetensors files and slices only its own DTensor shard."""
    import safetensors

    paths = sorted(glob.glob(f"{folder}/*.safetensors"))
    if not paths:
      raise ValueError(f"no safetensors files in {folder}")
    manifest = read_manifest(folder)
    meta = manifest["meta"] if manifest is not None else header(paths[0])
    opened = {path: safetensors.safe_open(path, framework="pt", device="cpu") for path in paths}
    try:
      loc = {k: (path, tuple(f.get_slice(k).get_shape())) for path, f in opened.items() for k in f.keys()}

      def slice_local(k: str, p: torch.Tensor) -> torch.Tensor:
        path, shape = loc[k]
        return slice_for_param(opened[path].get_slice(k), shape, p)

      def load_like(k: str, p: torch.Tensor, cast_float: bool) -> torch.Tensor:
        slc = slice_local(k, p)
        slc = slc.float() if cast_float else slc
        return (
          DTensor.from_local(slc.to(p.device), p.device_mesh, p.placements, run_check=False)
          if isinstance(p, DTensor)
          else slc.to(p.device)
        )

      self._apply_restore(
        run,
        meta,
        {k: shp for k, (_, shp) in loc.items()},
        optimizer,
        gradients,
        lambda k: opened[loc[k][0]].get_tensor(k) if k in loc else None,
        lambda k, p: local_tensor(p).data.copy_(slice_local(k, p).to(p.device)),
        load_like,
      )
    finally:
      for f in opened.values():
        f.__exit__(None, None, None)

  def moments_fit(self) -> bool:
    if not torch.cuda.is_available():
      return True
    size = sum(t.numel() * t.element_size() for t in self.adam_moments() if t.is_cuda)
    free = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    return free >= size

  def adam_moments(self):
    states = (s for o in self.optimizers.values() for s in o.state.values())
    return (local_tensor(v) for s in states for v in s.values() if torch.is_tensor(v))

  def gpu_tensors(self, optim: bool = True):
    if self.lm is None:
      return
    yield from (local_tensor(p) for p in self.lm.parameters())
    yield from (local_tensor(b) for b in self.lm.buffers())
    yield from (local_tensor(p.grad) for p in self.lm.parameters() if p.grad is not None)
    if optim:
      yield from self.adam_moments()

  def host_ram_bytes(self) -> int:
    return sum(nbytes(t) for t in self.pinned.values()) if self.parked else 0

  def move_tensors(self, tensors, device: str) -> None:
    tensors = list(tensors)
    if device == "cpu":
      cuda = [t for t in tensors if t.is_cuda]
      if need := [t for t in cuda if (b := self.pinned.get(id(t))) is None or (b.shape, b.dtype) != (t.shape, t.dtype)]:
        slab, at = pinned_bytes(sum(aligned(nbytes(t)) for t in need)), 0
        for t in need:
          self.pinned[id(t)] = slab[at : at + nbytes(t)].view(t.dtype).view(t.shape)
          at += aligned(nbytes(t))
    for t in tensors:
      if device == "cpu" and t.is_cuda:
        buf = self.pinned[id(t)]
        buf.copy_(t, non_blocking=True)
        t.data = buf
      elif device == "cuda" and (t.is_pinned() or id(t) in self.pinned):
        t.data = t.data.to("cuda", non_blocking=True)
    if self.lm is not None and dist.is_initialized():
      self.lm._apply(lambda t: t)

    if torch.cuda.is_available():
      torch.cuda.synchronize()
      if device == "cpu":
        torch.cuda.empty_cache()
        for k in set(self.pinned) - {id(x) for x in self.gpu_tensors()}:
          del self.pinned[k]

  def park_on_host(self) -> None:
    """Copy every GPU tensor to pinned host memory and synchronize all ranks before reporting GPU release."""
    self.move_tensors(self.gpu_tensors(), "cpu")
    if dist.is_initialized():
      dist.barrier()
    self.parked = True

  def resume_on_gpu(self) -> None:
    self.move_tensors(self.gpu_tensors(optim=self.lora), "cuda")
    if dist.is_initialized():
      dist.barrier()
    self.parked = False

  def delete_run(self, run: str) -> None:
    lm = self.activate_run(run)
    self.dirty.discard(run)
    self.needs_recovery.discard(run)
    self.configs.pop(run, None)
    self.rng.pop(run, None)
    if self.lora:
      lm.delete_adapter(run)
    else:
      self.lm = None
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del self.params[run]
    self.pinned.clear()
    self.optimizers.pop(run, None)
