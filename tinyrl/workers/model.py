"""The model a trainer serves: the losses, token budget batching, and one base model with its LoRA adapters or full fine-tune, which park in host memory between turns."""

from __future__ import annotations

import math
import os
from collections.abc import Callable

import numpy as np
import torch
import transformers
from peft import LoraConfig
from peft.helpers import disable_input_dtype_casting
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from tinyrl.api import wire
from tinyrl.helpers import getenv
from tinyrl.storage import models

TOKEN_BUDGET = getenv("TOKEN_BUDGET", 0)  # padded tokens per forward/backward pass. 0 fits it to free GPU memory
MAX_TOKEN_BUDGET = 16384  # past this, bigger passes stop getting faster and only use more memory
CHUNK = getenv("CHUNK", 1024)  # positions per output head chunk when computing logprobs
GRAD_CKPT = getenv("GRAD_CKPT", 1)  # gradient checkpointing

# ***** losses *****
#
# Each loss is elementwise over the target tokens, and the trainer sums it. The arguments are
#   logprobs          the policy's logprob of each target token
#   weights           per-token weights, 1 unless the client sends them
#   sampler_logprobs  the logprob the sampler gave each token, for the RL losses
#   advantages        per-token advantages, for the RL losses


def cross_entropy(logprobs, weights, sampler_logprobs, advantages, config):
  return -logprobs * weights


def importance_sampling(logprobs, weights, sampler_logprobs, advantages, config):
  ratio = (logprobs - sampler_logprobs).clamp(-20, 20).exp()
  return torch.nan_to_num(-ratio * advantages * weights, nan=0.0, posinf=0.0, neginf=0.0)


def ppo(logprobs, weights, sampler_logprobs, advantages, config):
  if bad := set(config) - {"clip_range", "kl_coeff", "clip_low_threshold", "clip_high_threshold"}:
    raise ValueError(f"unknown PPO config keys: {bad}")
  if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in config.values()):
    raise ValueError("PPO config values must be finite numbers")
  clip = config.get("clip_range", 0.2)
  low, high = config.get("clip_low_threshold", 1 - clip), config.get("clip_high_threshold", 1 + clip)
  kl_coeff = config.get("kl_coeff", 0.0)
  if not 0 <= low <= high or kl_coeff < 0:
    raise ValueError("PPO clipping thresholds must be finite and ordered, and kl_coeff nonnegative")
  log_ratio = (logprobs - sampler_logprobs).clamp(-20, 20)
  ratio = log_ratio.exp()
  objective = torch.min(ratio * advantages, ratio.clamp(low, high) * advantages)
  kl = (ratio - 1) - log_ratio
  return -(objective - kl_coeff * kl) * weights


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
  """How many padded tokens one forward/backward pass can hold: TOKEN_BUDGET if it's set, otherwise as many as fit in half of the free GPU memory. It's measured on every call, since memory use changes as runs come and go, and an optimizer allocates its state on its first step."""
  if TOKEN_BUDGET:
    return TOKEN_BUDGET
  if lm.device.type != "cuda":
    return MAX_TOKEN_BUDGET

  free, _ = torch.cuda.mem_get_info(lm.device)
  free += torch.cuda.memory_reserved(lm.device) - torch.cuda.memory_allocated(lm.device)  # cached by torch, but unused

  # with gradient checkpointing, a token keeps about one hidden vector per layer. 6x that covers the layer being recomputed in the backward pass, and leaves a margin.
  config = lm.config.get_text_config()
  bytes_per_token = (
    (6 if getattr(lm, "is_gradient_checkpointing", False) else 64)
    * config.num_hidden_layers
    * config.hidden_size
    * lm.dtype.itemsize
  )
  return min(MAX_TOKEN_BUDGET, int(free / 2 / bytes_per_token))


def batches(lengths: list[int], budget: int) -> list[list[int]]:
  """Group sequences into passes, shortest first so a pass pads little. A sequence starts a new pass if joining the current one would go over the token budget, or make the pass more than 25% padding."""
  passes: list[list[int]] = []
  for i in sorted(range(len(lengths)), key=lambda i: lengths[i]):
    if passes:
      current = passes[-1]
      padded = lengths[i] * (len(current) + 1)  # sequence i is the pass's longest, since they're sorted
      real = lengths[i] + sum(lengths[j] for j in current)
      if padded <= budget and padded - real <= padded / 4:
        current.append(i)
        continue
    passes.append([i])
  return passes


def pad(rows: list[np.ndarray], lengths: list[int], dtype, device: torch.device) -> torch.Tensor:
  """Right-pad rows with zeros into one [rows, max(lengths)] tensor, keeping the first lengths[i] values of row i."""
  out = np.zeros((len(rows), max(lengths)), dtype)
  for i, (row, n) in enumerate(zip(rows, lengths)):
    out[i, :n] = row[:n]
  return torch.from_numpy(out).to(device)


def batch_tensors(data: list[wire.Datum], lengths: list[int], device: torch.device) -> dict[str, torch.Tensor]:
  """Pad one pass's sequences into the tensors the model and the losses take."""

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
  """The logprob of each target token.

  The model returns its hidden states instead of logits. The output head and the logsumexp then run CHUNK positions at a time under checkpointing, so the [tokens, vocab] logits never exist all at once."""
  out = lm(ids, attention_mask=mask, use_cache=False, logits_to_keep=1, output_hidden_states=True)
  hidden = out.hidden_states[-1].flatten(0, 1)
  head = lm.get_output_embeddings()
  softcap = getattr(lm.config.get_text_config(), "final_logit_softcapping", None)  # gemma caps its logits

  def chunk_logprobs(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    logits = head(h)
    if softcap:
      logits = torch.tanh(logits / softcap) * softcap
    logits = logits.float()
    return logits.gather(-1, t[:, None])[:, 0] - logits.logsumexp(-1)

  chunks = zip(hidden.split(CHUNK), targets.flatten().split(CHUNK))
  return torch.cat([checkpoint(chunk_logprobs, h, t, use_reentrant=False) for h, t in chunks]).view(targets.shape)


# ***** model *****


def load(name: str) -> PreTrainedModel:
  """The class the checkpoint names in its config, which is the one vLLM serves, so the weights and adapters a trainer saves carry the checkpoint's parameter names and vLLM maps them like the checkpoint's own. A multimodal checkpoint's causal LM class names its layers differently (Qwen3.5: model.layers.* for model.language_model.*), and vLLM drops an adapter whose names it can't map, without an error."""
  name = models.local(name)
  arch = (AutoConfig.from_pretrained(name).architectures or [""])[0]
  cls = getattr(transformers, arch, AutoModelForCausalLM)
  # bf16 on the gpu, where a LoRA run's frozen base model stays. A full fine-tune upcasts to fp32 in Model.create_run
  if torch.cuda.is_available():
    return cls.from_pretrained(name, dtype=torch.bfloat16, device_map="auto")
  return cls.from_pretrained(name, dtype=torch.float32)


def lora_targets(lm: PreTrainedModel, config: dict) -> list[str]:
  targets = []
  if config.get("train_attn", True):
    targets += ["q_proj", "k_proj", "v_proj", "o_proj"]
  if config.get("train_mlp", True):
    targets += ["gate_proj", "up_proj", "down_proj"]

  # vLLM can't load an adapter on an output head that is tied to the embeddings
  tied = getattr(lm.config.get_text_config(), "tie_word_embeddings", False)
  if config.get("train_unembed", True) and not tied:
    targets.append("lm_head")
  return [
    n
    for n, m in lm.named_modules()
    if n.rsplit(".", 1)[-1] in targets and isinstance(getattr(m, "base_layer", m), torch.nn.Linear)
  ]


class Model:
  """A base model and the runs training on it: a LoRA adapter per run, or the whole model for one full fine-tune."""

  def __init__(self, load: Callable[[str], PreTrainedModel] = load):
    self.load = load
    self.lm: PreTrainedModel | None = None
    self.base = ""
    self.lora = True
    self.dtype = torch.float32
    self.params: dict[str, dict[str, torch.nn.Parameter]] = {}  # canonical name -> trainable parameter
    self.optimizers: dict[str, torch.optim.AdamW] = {}
    self.pinned: dict[int, torch.Tensor] = {}
    self.parked = False
    self.dirty: set[str] = set()

  def create_run(self, run: str, base: str, rank: int, config: dict) -> None:
    if self.lm is None:  # the first run loads the base model
      self.lm, self.base, self.lora = self.load(base), base, rank > 0
      if hasattr(inner := getattr(self.lm, "model", None), "language_model"):
        for child, _ in list(inner.named_children()):
          if child != "language_model":
            setattr(inner, child, None)
      self.dtype = self.lm.dtype
      if not self.lora:
        self.lm.float()  # a full fine-tune updates fp32 master weights. Its forward pass runs in bf16, under autocast
      if should_checkpoint_layers(self.lm, self.lora):
        self.lm.gradient_checkpointing_enable()
        if self.lora:
          self.lm.enable_input_require_grads()  # with the embeddings frozen, checkpointed layers need a gradient path

    if base != self.base or (rank > 0) != self.lora:
      raise ValueError(f"this trainer does {'lora' if self.lora else 'full'} fine-tuning of {self.base}")
    if config.get("seed") is not None:
      torch.manual_seed(config["seed"])

    self.dirty.discard(run)
    if not self.lora:
      if self.params:
        raise ValueError(f"a full fine-tune has its trainer to itself, and {next(iter(self.params))} is on this one")
      self.params[run] = {name: p.requires_grad_() for name, p in self.lm.named_parameters()}
      return

    adapter = LoraConfig(
      r=rank,
      lora_alpha=config.get("lora_alpha", 16),
      lora_dropout=config.get("lora_dropout", 0.05),
      target_modules=lora_targets(self.lm, config),
    )
    self.lm.add_adapter(adapter, adapter_name=run)
    infix = f".{run}."
    self.params[run] = {name.replace(infix, "."): p for name, p in self.lm.named_parameters() if infix in name}
    for p in self.params[run].values():
      p.data = p.data.float()  # PEFT gives an adapter its base model's dtype, but master weights are fp32

  def activate_run(self, run: str) -> PreTrainedModel:
    """The model, with the run's adapter active."""
    if self.lm is None or run not in self.params:
      raise KeyError(f"model {run} is not on this trainer, it was deleted or its trainer restarted")
    if self.lora:
      self.lm.set_adapter(run)
    return self.lm

  def forward_backward(self, run: str, data: list[wire.Datum], loss_fn: str, config: dict, forward_only: bool):
    """Run the loss over every datum, accumulating gradients unless forward_only. Returns each datum's target logprobs, and the loss metrics."""
    if loss_fn not in LOSSES:
      raise ValueError(f"unknown loss_fn {loss_fn}, this trainer has {', '.join(LOSSES)}")

    lm = self.activate_run(run).train(not forward_only)
    lengths = [len(d.tokens) for d in data]
    for d, n in zip(data, lengths):
      if not n or d.tokens.shape != (n,) or "target_tokens" not in d.inputs:
        raise ValueError("each datum needs a nonempty token sequence and target_tokens")
      if set(d.inputs) - {"target_tokens", "weights", "logprobs", "advantages"}:
        raise ValueError("unsupported loss inputs")
      if any(x.shape != (n,) or not np.isfinite(x).all() for x in d.inputs.values()):
        raise ValueError("loss inputs must be finite 1-D arrays with one value per input token")
      if any(
        x.dtype.kind not in "iu" or np.any(x < 0) or np.any(x >= lm.config.get_text_config().vocab_size)
        for x in (d.tokens, d.inputs["target_tokens"])
      ):
        raise ValueError("input and target tokens must be integer vocabulary indices")
    if loss_fn != "ppo" and config:
      raise ValueError(f"{loss_fn} has no loss config")
    logprobs: list[np.ndarray] = [np.zeros(0, np.float32)] * len(data)
    total = 0.0

    # on the gpu autocast runs the adapters' matmuls in bf16 anyway, so PEFT's upcast of each layer's input to the fp32 adapter only adds casts
    gpu = lm.device.type == "cuda"
    accumulating = False
    try:
      with torch.set_grad_enabled(not forward_only), disable_input_dtype_casting(lm, active=gpu):
        for batch in batches(lengths, token_budget(lm)):
          x = batch_tensors([data[i] for i in batch], [lengths[i] for i in batch], lm.device)
          with torch.autocast(lm.device.type, dtype=torch.bfloat16, enabled=gpu):
            lp = target_logprobs(lm, x["ids"], x["mask"], x["targets"])
          loss = LOSSES[loss_fn](lp, x["weights"], x["sampler_logprobs"], x["advantages"], config).sum()
          if not forward_only:
            accumulating = True
            loss.backward()
          total += loss.item()

          for row, i in enumerate(batch):
            values = lp[row, : lengths[i]].detach().float()
            # keep them finite, so a JSON client can carry them
            logprobs[i] = torch.nan_to_num(values, nan=0.0).clamp(-9999, 9999).cpu().numpy()
    except Exception as e:
      if accumulating:
        self.delete_run(run)
        raise RuntimeError(
          "backward failed after mutation; this run lost its unsaved state, load_state to recover"
        ) from e
      raise

    return logprobs, {"loss:sum": total, "loss:mean": total / max(1, len(data))}

  def adam_for_run(self, run: str, **config) -> torch.optim.AdamW:
    params = list(self.params[run].values())
    optimizer = torch.optim.AdamW(params, fused=bool(params and params[0].is_cuda), **config)
    if run in self.optimizers:
      optimizer.state = self.optimizers[run].state
    self.optimizers[run] = optimizer
    return optimizer

  def optim_step(self, run: str, adam: dict) -> dict[str, float]:
    self.activate_run(run)
    if set(adam) - {"learning_rate", "beta1", "beta2", "eps", "weight_decay", "grad_clip_norm"}:
      raise ValueError("unknown Adam parameters")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in adam.values()):
      raise ValueError("Adam parameters must be finite numbers")
    max_norm = adam.get("grad_clip_norm") or math.inf  # 0 means don't clip
    if max_norm < 0:
      raise ValueError("grad_clip_norm must be nonnegative")
    optimizer = self.adam_for_run(
      run,
      lr=adam.get("learning_rate", 1e-4),
      betas=(adam.get("beta1", 0.9), adam.get("beta2", 0.95)),
      eps=adam.get("eps", 1e-12),
      weight_decay=adam.get("weight_decay", 0.0),
    )  # Adam validates each command's hyperparameters before any gradients or state change.

    try:
      grad_norm = torch.nn.utils.clip_grad_norm_(self.params[run].values(), max_norm, error_if_nonfinite=True).item()
      # moments on the host come back to the GPU, after freeing torch's cache to make room
      if not self.lora and torch.cuda.is_available() and any(not t.is_cuda for t in self.adam_moments()):
        torch.cuda.empty_cache()
        self.move_tensors(self.adam_moments(), "cuda")
      optimizer.step()
      optimizer.zero_grad()
      if not self.lora and not self.moments_fit():
        self.move_tensors(self.adam_moments(), "cpu")
      self.dirty.add(run)
      return {"grad_norm:mean": grad_norm}
    except Exception as e:
      self.delete_run(run)
      raise RuntimeError(
        "optimizer failed after mutation; this run lost its unsaved state, load_state to recover"
      ) from e

  def training_state(self, run: str) -> tuple[dict[str, torch.Tensor], dict]:
    """A run's portable training checkpoint: fp32 master weights and named AdamW state, with no pickle."""
    self.activate_run(run)
    tensors = {name: p.detach() for name, p in self.params[run].items()}
    optim = self.optimizers.get(run)
    for name, p in self.params[run].items():
      if p.grad is not None:
        tensors[f"{name}.grad"] = p.grad.detach()
      if optim:
        tensors |= {f"{name}.{key}": t.detach() for key, t in optim.state.get(p, {}).items()}
    return tensors, {"optimizer": True}

  def sampler_weights(self, run: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Sampler weights in the base model's compute dtype (bf16 on GPU), plus the config vLLM or PEFT reads."""
    lm = self.activate_run(run)
    prefix = "base_model.model." if self.lora else ""
    tensors = {f"{prefix}{name}": p.detach().to(self.dtype) for name, p in self.params[run].items()}
    if self.lora:
      cfg = {k: sorted(v) if isinstance(v, set) else v for k, v in lm.peft_config[run].to_dict().items()}
    else:
      cfg = lm.config.to_dict()
    return tensors, {
      "base": self.base,
      "lora": self.lora,
      "optimizer": False,
      "untouched": not self.lora and run not in self.dirty,
      "config": cfg,
    }

  def restore_run(
    self, run: str, tensors: dict[str, torch.Tensor], meta: dict, optimizer: bool, gradients: bool = False
  ) -> None:
    self.activate_run(run)
    if optimizer and not meta.get("optimizer"):
      raise ValueError("checkpoint has no optimizer state")
    suffixes = (".exp_avg", ".exp_avg_sq", ".step", ".grad")
    tensors = {k.removeprefix("base_model.model."): v for k, v in tensors.items()}
    if not self.lora and "lm_head.weight" not in self.params[run]:
      tensors.pop("lm_head.weight", None)  # tied to embed_tokens
    allowed = {name + s for name in self.params[run] for s in ("", *suffixes)}
    if not self.params[run].keys() <= tensors.keys() or tensors.keys() - allowed:
      raise ValueError("checkpoint does not fit this model")
    for name, p in self.params[run].items():
      for suffix in ("", *suffixes):
        if (t := tensors.get(name + suffix)) is not None:
          if not t.is_floating_point() or t.shape != (() if suffix == ".step" else p.shape):
            raise ValueError(f"checkpoint shape or dtype mismatch for {name + suffix}")
          if suffix == ".step" and (not math.isfinite(t.item()) or t.item() < 0 or t.item() % 1):
            raise ValueError(f"invalid optimizer step for {name}")
      if (
        optimizer
        and any(name + s in tensors for s in suffixes[:3])
        and not all(name + s in tensors for s in suffixes[:3])
      ):
        raise ValueError(f"incomplete optimizer state for {name}")
    try:
      for name, p in self.params[run].items():
        p.data.copy_(tensors[name])
        p.grad = tensors[f"{name}.grad"].to(p, copy=True) if gradients and f"{name}.grad" in tensors else None
      self.pinned.clear()
      self.optimizers.pop(run, None)
      if optimizer and any(k.endswith(".step") for k in tensors):
        opt = self.adam_for_run(run)
        for name, p in self.params[run].items():
          if f"{name}.step" in tensors:
            opt.state[p] = {
              s: tensors[f"{name}.{s}"].to(
                p.device if s != "step" or opt.defaults.get("fused") else "cpu", torch.float32, copy=True
              )
              for s in ("exp_avg", "exp_avg_sq", "step")
            }
        if not self.lora:
          self.move_tensors(self.adam_moments(), "cpu")
      self.dirty.add(run)
    except Exception:
      self.delete_run(run)
      raise

  def moments_fit(self) -> bool:
    """Whether a full fine-tune's AdamW moments stay on the GPU between steps: when what's left free is at least as big as they are, so forward_backward keeps room. Otherwise they cross PCIe twice a step, 2.7 of a 2.8 s step for Gemma E2B."""
    if not torch.cuda.is_available():
      return True
    size = sum(t.numel() * t.element_size() for t in self.adam_moments() if t.is_cuda)
    free = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    return free >= size

  def adam_moments(self):
    return (v for opt in self.optimizers.values() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))

  def gpu_tensors(self, optim: bool = True):
    """Every tensor on the GPU: weights, buffers, gradients between forward_backward and optim_step, AdamW moments."""
    if self.lm is None:
      return
    yield from self.lm.parameters()
    yield from self.lm.buffers()
    yield from (p.grad for p in self.lm.parameters() if p.grad is not None)
    if optim:
      yield from self.adam_moments()

  def move_tensors(self, tensors, device: str) -> None:
    for t in tensors:
      if device == "cpu" and t.is_cuda:
        buf = self.pinned.get(id(t))
        if buf is None or buf.shape != t.shape or buf.dtype != t.dtype:
          buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
          self.pinned[id(t)] = buf
        buf.copy_(t, non_blocking=True)
        t.data = buf
      elif device == "cuda" and (t.is_pinned() or id(t) in self.pinned):
        t.data = t.data.to("cuda", non_blocking=True)

    if torch.cuda.is_available():
      torch.cuda.synchronize()
      if device == "cpu":
        torch.cuda.empty_cache()
        active = {id(x) for x in self.gpu_tensors()}
        for k in list(self.pinned.keys()):
          if k not in active:
            del self.pinned[k]

  def park_on_host(self) -> None:
    """Copy every GPU tensor to pinned host memory and free the GPU, for another trainer's turn. resume_on_gpu() undoes it."""
    self.move_tensors(self.gpu_tensors(), "cpu")
    self.parked = True

  def resume_on_gpu(self) -> None:
    self.move_tensors(self.gpu_tensors(optim=self.lora), "cuda")
    self.parked = False

  def delete_run(self, run: str) -> None:
    lm = self.activate_run(run)
    self.dirty.discard(run)
    if self.lora:
      lm.delete_adapter(run)
    del self.params[run]
    self.pinned.clear()
    self.optimizers.pop(run, None)
