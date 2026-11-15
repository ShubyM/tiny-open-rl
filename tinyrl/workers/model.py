"""The model a trainer serves: the losses, token budget batching, and one base model with its LoRA adapters or full fine-tune, which park in host memory between turns and restore cleanly with per-job optimizer, gradient and RNG state."""

from __future__ import annotations

import contextlib
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
from tinyrl.storage.store import aligned, nbytes, pinned_bytes

LigerFusedLinearCrossEntropyLoss = apply_liger = None
with contextlib.suppress(ImportError):
  logging.getLogger("liger_kernel").setLevel(logging.ERROR)
  from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
  from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance as apply_liger

TOKEN_BUDGET = getenv("TOKEN_BUDGET", 0)  # padded tokens per forward/backward pass. 0 fits it to free GPU memory
MAX_TOKEN_BUDGET = 16384  # past this, bigger passes stop getting faster and only use more memory
CHUNK = getenv("CHUNK", 1024)  # positions per output head chunk when computing logprobs
GRAD_CKPT = getenv("GRAD_CKPT", 1)  # gradient checkpointing


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


# ***** randomness *****


def stream_seed(seed: int, step: int, rank: int, tag: str) -> int:
  raw = hashlib.blake2b(f"{seed}:{step}:{rank}:{tag}".encode(), digest_size=8).digest()
  return int.from_bytes(raw, "little") & ((1 << 63) - 1)


def capture_rng() -> tuple[torch.Tensor, torch.Tensor | None]:
  return torch.get_rng_state(), (torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def apply_rng(state: tuple[torch.Tensor, torch.Tensor | None]) -> None:
  torch.set_rng_state(state[0])
  if state[1] is not None and torch.cuda.is_available():
    torch.cuda.set_rng_state(state[1])


def derived_rng(seed: int, step: int, rank: int) -> tuple[torch.Tensor, torch.Tensor | None]:
  """Deterministic rank RNG state used when restoring a checkpoint onto a different rank width or device."""
  saved = capture_rng()
  try:
    torch.manual_seed(stream_seed(seed, step, rank, "cpu"))
    cpu = torch.get_rng_state()
    cuda = None
    if torch.cuda.is_available():
      torch.cuda.manual_seed(stream_seed(seed, step, rank, "cuda"))
      cuda = torch.cuda.get_rng_state()
    return cpu, cuda
  finally:
    apply_rng(saved)


def normalize_lora_config(rank: int, config: dict, seed: int) -> dict:
  if rank <= 0:
    return {"rank": 0, "seed": seed}
  flags = {k: bool(config.get(k, True)) for k in ("train_attn", "train_mlp", "train_unembed")}
  return {
    "rank": int(rank),
    "seed": seed,
    "lora_alpha": config.get("lora_alpha", 16),
    "lora_dropout": float(config.get("lora_dropout", 0.05)),
    **flags,
  }


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
  """How many padded tokens one forward/backward pass can hold: TOKEN_BUDGET if set, else half of free GPU memory."""
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
  """Group sequences into passes, shortest first so a pass pads little."""
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
  """The logprob of each target token, chunked over the output head under checkpointing."""
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


def full_tensor(t: torch.Tensor) -> torch.Tensor:
  return t.full_tensor() if isinstance(t, DTensor) else t


def like_param(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
  return distribute_tensor(t.to(p.device), p.device_mesh, p.placements) if isinstance(p, DTensor) else t.to(p.device)


def load(name: str) -> PreTrainedModel:
  """Load the checkpoint's exact architecture class so saved parameter and adapter names match vLLM."""
  name = models.local(name)
  arch = (AutoConfig.from_pretrained(name).architectures or [""])[0]
  cls = getattr(transformers, arch, AutoModelForCausalLM)
  if torch.cuda.is_available() and not dist.is_initialized():
    return cls.from_pretrained(name, dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()})
  return cls.from_pretrained(name, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)


def lora_targets(lm: PreTrainedModel, config: dict) -> list[str]:
  targets = ["q_proj", "k_proj", "v_proj", "o_proj"] if config.get("train_attn", True) else []
  targets += ["gate_proj", "up_proj", "down_proj"] if config.get("train_mlp", True) else []
  tied = getattr(lm.config.get_text_config(), "tie_word_embeddings", False)
  if config.get("train_unembed", True) and not tied:
    targets.append("lm_head")
  mods = ((n, getattr(m, "base_layer", m)) for n, m in lm.named_modules())
  return [n for n, m in mods if n.rsplit(".", 1)[-1] in targets and isinstance(m, torch.nn.Linear)]


class Model:
  """A base model and the runs training on it: a LoRA adapter per run, or the whole model for one full fine-tune."""

  def __init__(self, load: Callable[[str], PreTrainedModel] = load):
    self.load, self.lm, self.base, self.lora, self.dtype = load, None, "", True, torch.float32
    self.params: dict[str, dict[str, torch.nn.Parameter]] = {}
    self.optimizers: dict[str, torch.optim.AdamW] = {}
    self.configs: dict[str, dict] = {}
    self.rngs: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
    self.completed: dict[str, int] = {}
    self.pinned: dict[int, torch.Tensor] = {}
    self.parked, self.dirty, self.needs_recovery = False, set(), set()

  @contextlib.contextmanager
  def job_rng(self, run: str, *, advance: bool = True):
    """Run one job operation under that job's private RNG stream so sibling adapters never perturb it."""
    saved = capture_rng()
    apply_rng(self.rngs[run])
    try:
      yield
      if advance:
        self.rngs[run] = capture_rng()
    finally:
      apply_rng(saved)

  def profile(self) -> dict:
    world = dist.get_world_size() if dist.is_initialized() else 1
    dev = (
      "cuda"
      if torch.cuda.is_available() and not self.parked
      else (self.lm.device.type if self.lm is not None else "cpu")
    )
    return {
      "dtype": str(self.dtype).removeprefix("torch."),
      "master_dtype": "float32",
      "world_size": world,
      "layout": "lora" if self.lora else ("fsdp" if world > 1 else "single"),
      "device": dev,
    }

  def create_run(self, run: str, base: str, rank: int, config: dict) -> None:
    if self.lm is None:
      saved = capture_rng()
      try:
        self.lm, self.base, self.lora = self.load(base), base, rank > 0
      finally:
        apply_rng(saved)
      if hasattr(inner := getattr(self.lm, "model", None), "language_model"):
        for child in [c for c, _ in inner.named_children() if c != "language_model"]:
          setattr(inner, child, None)
      self.dtype = self.lm.dtype
      if not self.lora:
        self.lm.float()  # fp32 master weights; forward runs in bf16 under autocast on CUDA
      if should_checkpoint_layers(self.lm, self.lora):
        self.lm.gradient_checkpointing_enable()
        if self.lora:
          self.lm.enable_input_require_grads()
      if dist.is_initialized():
        mp = MixedPrecisionPolicy(reduce_dtype=torch.float32)
        for layer in [m for m in self.lm.modules() if type(m).__name__ in (self.lm._no_split_modules or ())]:
          fully_shard(layer, mp_policy=mp)
        fully_shard(self.lm, mp_policy=mp)
      if self.lm.device.type == "cuda" and apply_liger:
        apply_liger(model=self.lm, cross_entropy=False, fused_linear_cross_entropy=False)

    if base != self.base or (rank > 0) != self.lora:
      raise ValueError(f"this trainer does {'lora' if self.lora else 'full'} fine-tuning of {self.base}")
    if not self.lora and self.params:
      raise ValueError(f"a full fine-tune has its trainer to itself, and {next(iter(self.params))} is on this one")

    world, my_rank = (dist.get_world_size(), dist.get_rank()) if dist.is_initialized() else (1, 0)
    if config.get("seed") is not None:
      seed = int(config["seed"])
    else:
      picked = [int.from_bytes(os.urandom(4), "little") & 0x7FFF_FFFF if my_rank == 0 else 0]
      if world > 1:
        dist.broadcast_object_list(picked, src=0)
      seed = picked[0]
    cfg = normalize_lora_config(rank, config, seed)

    saved = capture_rng()
    try:
      torch.manual_seed(seed)
      self.dirty.discard(run)
      self.needs_recovery.discard(run)
      if not self.lora:
        self.params[run] = {name: p.requires_grad_() for name, p in self.lm.named_parameters()}
      else:
        targets = lora_targets(self.lm, cfg)
        if not targets:
          raise ValueError("lora_config selects no trainable modules")
        adapter = LoraConfig(
          r=cfg["rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"], target_modules=targets
        )
        with torch.device(self.lm.device):
          self.lm.add_adapter(adapter, adapter_name=run)
        infix = f".{run}."
        self.params[run] = {name.replace(infix, "."): p for name, p in self.lm.named_parameters() if infix in name}
        for p in self.params[run].values():
          p.data = p.data.float()
          if world > 1:
            dist.broadcast(p.data, src=0)
      if my_rank > 0:
        torch.manual_seed(stream_seed(seed, 0, my_rank, "init"))
      self.rngs[run] = capture_rng()
    finally:
      apply_rng(saved)
    self.configs[run] = cfg
    self.completed[run] = 0

  def activate_run(self, run: str) -> PreTrainedModel:
    if self.lm is None or run not in self.params:
      raise KeyError(f"model {run} is not on this trainer, it was deleted or its trainer restarted")
    if self.lora:
      self.lm.set_adapter(run)
    return self.lm

  def forward_backward(self, run: str, data: list[wire.Datum], loss_fn: str, config: dict, forward_only: bool):
    """Run the loss over every datum, accumulating gradients unless forward_only, under the job's own RNG stream."""
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
      with self.job_rng(run, advance=not forward_only):
        with torch.set_grad_enabled(not forward_only), disable_input_dtype_casting(lm, active=gpu):
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
      if not forward_only:
        self.completed[run] = self.completed.get(run, 0) + 1
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
    grad_norm = torch.nn.utils.clip_grad_norm_(self.params[run].values(), max_norm).item()
    if not self.lora and torch.cuda.is_available() and any(not t.is_cuda for t in self.adam_moments()):
      torch.cuda.empty_cache()
      self.move_tensors(self.adam_moments(), "cuda")
    optimizer.step()
    optimizer.zero_grad()
    if not self.lora and not self.moments_fit():
      self.move_tensors(self.adam_moments(), "cpu")
    self.dirty.add(run)
    self.completed[run] = self.completed.get(run, 0) + 1
    return {"grad_norm:mean": grad_norm}

  def training_state(self, run: str, boundary: str = "") -> tuple[dict[str, torch.Tensor], dict]:
    """Portable FP32 training recovery state: master weights, accumulated gradients, AdamW state, per-rank RNG and profile."""
    self.activate_run(run)
    if run in self.needs_recovery:
      raise RuntimeError("cannot publish training state after a partially failed forward_backward")
    tensors = {name: full_tensor(p.detach()).float() for name, p in self.params[run].items()}
    optim = self.optimizers.get(run)
    has_grad = False
    for name, p in self.params[run].items():
      if p.grad is not None:
        has_grad = True
        tensors[f"{name}.grad"] = full_tensor(p.grad.detach()).float()
      if optim and p in optim.state:
        for k in ("exp_avg", "exp_avg_sq", "step"):
          tensors[f"{name}.{k}"] = full_tensor(optim.state[p][k].detach()).float()
    cpu_rng, cuda_rng = self.rngs[run]
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world > 1:
      dist.all_gather_object(cpus := [None] * world, cpu_rng.cpu())
      dist.all_gather_object(cudas := [None] * world, cuda_rng.cpu() if cuda_rng is not None else None)
    else:
      cpus, cudas = [cpu_rng.cpu()], [cuda_rng.cpu() if cuda_rng is not None else None]
    for i, s in enumerate(cpus):
      tensors[f"__rng.cpu.{i}"] = s.clone()
    for i, s in enumerate(cudas):
      if s is not None:
        tensors[f"__rng.cuda.{i}"] = s.clone()
    cfg = dict(self.configs[run])
    targets = sorted({t.rsplit(".", 1)[-1] for t in self.lm.peft_config[run].target_modules}) if self.lora else []
    head = {
      "kind": "training",
      "optimizer": True,
      "base": self.base,
      "lora": self.lora,
      "rank": cfg["rank"],
      "lora_config": cfg,
    }
    tail = {"targets": targets, "seed": cfg["seed"], "completed_cmds": self.completed.get(run, 0), "boundary": boundary}
    return tensors, head | tail | {"dirty": run in self.dirty, "has_gradients": has_grad, "profile": self.profile()}

  def sampler_weights(
    self, run: str, source_version: str | None = None, source_boundary: str = ""
  ) -> tuple[dict[str, torch.Tensor], dict]:
    """Sampler export in compute dtype (never treated as an optimizer-bearing recovery checkpoint)."""
    lm = self.activate_run(run)
    if run in self.needs_recovery:
      raise RuntimeError("cannot export sampler weights after a partially failed forward_backward")
    prefix = "base_model.model." if self.lora else ""
    tensors = {f"{prefix}{name}": full_tensor(p.detach().to(self.dtype)) for name, p in self.params[run].items()}
    cfg = (
      {k: sorted(v) if isinstance(v, set) else v for k, v in lm.peft_config[run].to_dict().items()}
      if self.lora
      else lm.config.to_dict()
    )
    head = {
      "kind": "sampler",
      "base": self.base,
      "lora": self.lora,
      "rank": self.configs[run]["rank"],
      "optimizer": False,
    }
    tail = {
      "untouched": not self.lora and run not in self.dirty,
      "source_version": source_version,
      "source_boundary": source_boundary,
    }
    return tensors, head | tail | {"export_profile": self.profile(), "config": cfg}

  def restore_run(
    self, run: str, tensors: dict[str, torch.Tensor], meta: dict, optimizer: bool, gradients: bool = False
  ) -> dict:
    self.activate_run(run)
    if optimizer and meta.get("kind") == "sampler":
      raise ValueError("checkpoint is a sampler export and has no optimizer state")
    if optimizer and not meta.get("optimizer"):
      raise ValueError("checkpoint has no optimizer state")
    if (src_lora := meta.get("lora")) is not None and bool(src_lora) != self.lora:
      raise ValueError("checkpoint mode does not match this trainer")
    if (src_cfg := meta.get("lora_config")) and self.lora and optimizer:
      cur_cfg = self.configs[run]
      if any(src_cfg.get(k) != cur_cfg.get(k) for k in ("rank", "lora_alpha", "lora_dropout")):
        raise ValueError("checkpoint adapter configuration does not fit this model")
    suffixes = (".exp_avg", ".exp_avg_sq", ".step", ".grad")
    weights = {
      k.removeprefix("base_model.model."): v
      for k, v in tensors.items()
      if not k.startswith("__rng.") and not k.endswith(suffixes)
    }
    if not self.lora and "lm_head.weight" not in self.params[run]:
      weights.pop("lm_head.weight", None)
    if weights.keys() != self.params[run].keys():
      raise ValueError("checkpoint does not fit this model")
    req_opt = optimizer and any(k.endswith(".step") for k in tensors)
    reqs = ("exp_avg", "exp_avg_sq", "step")
    for n, p in self.params[run].items():
      if weights[n].shape != p.shape or (req_opt and not all(f"{n}.{s}" in tensors for s in reqs)):
        raise ValueError(f"bad: {n}")
      if gradients and f"{n}.grad" in tensors and tensors[f"{n}.grad"].shape != p.shape:
        raise ValueError(f"bad: {n}.grad")

    for name, p in self.params[run].items():
      p.data.copy_(like_param(weights[name].float(), p))
      has_grad = gradients and f"{name}.grad" in tensors
      p.grad = like_param(tensors[f"{name}.grad"].float(), p) if has_grad else None

    self.needs_recovery.discard(run)
    self.pinned.clear()
    self.optimizers.pop(run, None)
    if req_opt:
      opt = self.adam_for_run(run)
      for name, p in self.params[run].items():
        opt.state[p] = {
          "exp_avg": like_param(tensors[f"{name}.exp_avg"].float(), p),
          "exp_avg_sq": like_param(tensors[f"{name}.exp_avg_sq"].float(), p),
          "step": tensors[f"{name}.step"].to(p.device if opt.defaults.get("fused") else "cpu", torch.float32),
        }
      if not self.lora and not self.moments_fit():
        self.move_tensors(self.adam_moments(), "cpu")

    rng_remapped = False
    if optimizer:
      world, my_rank = (dist.get_world_size(), dist.get_rank()) if dist.is_initialized() else (1, 0)
      same_world = int((meta.get("profile") or {}).get("world_size", 1)) == world
      seed = int(meta.get("seed", self.configs[run]["seed"]))
      done = int(meta.get("completed_cmds", 0))
      fallback_cpu, fallback_cuda = derived_rng(seed, done, my_rank)
      cpu_t = tensors.get(f"__rng.cpu.{my_rank}")
      if same_world and cpu_t is not None and cpu_t.shape == fallback_cpu.shape and cpu_t.dtype == torch.uint8:
        cpu_rng = cpu_t.to(device="cpu", dtype=torch.uint8).clone()
      else:
        cpu_rng, rng_remapped = fallback_cpu, True
      cuda_rng = None
      if torch.cuda.is_available() and fallback_cuda is not None:
        cuda_t = tensors.get(f"__rng.cuda.{my_rank}")
        if same_world and cuda_t is not None and cuda_t.shape == fallback_cuda.shape and cuda_t.dtype == torch.uint8:
          cuda_rng = cuda_t.to(device="cpu", dtype=torch.uint8).clone()
        else:
          cuda_rng, rng_remapped = fallback_cuda, True
      self.rngs[run] = (cpu_rng, cuda_rng)
      self.completed[run] = done
      self.configs[run]["seed"] = seed

    if meta.get("dirty", True):
      self.dirty.add(run)
    else:
      self.dirty.discard(run)
    return {"boundary": meta.get("boundary", ""), "rng_remapped": rng_remapped}

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

    if torch.cuda.is_available():
      torch.cuda.synchronize()
      if device == "cpu":
        torch.cuda.empty_cache()
        for k in set(self.pinned) - {id(x) for x in self.gpu_tensors()}:
          del self.pinned[k]

  def move_model(self, device: str, optim: bool) -> None:
    if self.lm is not None and dist.is_initialized():
      # DTensor wrappers track their own device; moving via _apply updates FSDP shard wrappers coherently across ranks.
      if torch.cuda.is_available() or device == "cpu":
        self.lm._apply(lambda t: t.to(device, non_blocking=True))
        self.move_tensors(self.adam_moments() if optim else (), device)
        if torch.cuda.is_available():
          torch.cuda.synchronize()
          if device == "cpu":
            torch.cuda.empty_cache()
    else:
      self.move_tensors(self.gpu_tensors(optim), device)
    self.parked = device == "cpu"

  def park_on_host(self) -> dict:
    """Copy every GPU tensor to host memory and free model allocations; all ranks reach the boundary together."""
    self.move_model("cpu", True)
    if dist.is_initialized():
      dist.barrier()
    return self.memory_state()

  def resume_on_gpu(self) -> dict:
    self.move_model("cuda", self.lora)
    if dist.is_initialized():
      dist.barrier()
    return self.memory_state()

  def memory_state(self) -> dict:
    tensors = list(self.gpu_tensors())
    host = sum(nbytes(t) for t in tensors if not t.is_cuda)
    gpu = int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
    return {"parked": self.parked, "gpu_bytes": gpu, "host_bytes": host, "runs": sorted(self.params)}

  def delete_run(self, run: str) -> None:
    lm = self.activate_run(run)
    self.dirty.discard(run)
    self.needs_recovery.discard(run)
    self.configs.pop(run, None)
    self.rngs.pop(run, None)
    self.completed.pop(run, None)
    if self.lora:
      lm.delete_adapter(run)
    else:
      self.lm = None
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del self.params[run]
    self.pinned.clear()
    self.optimizers.pop(run, None)
