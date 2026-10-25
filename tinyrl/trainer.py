"""A trainer serves one pool: the LoRA adapters that share a base model, or a single full fine-tune.

It executes coordinator-issued commands over HTTP and retains results until the coordinator acknowledges them.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import socket
import sys
import uuid
from collections.abc import Callable
from contextlib import contextmanager

import numpy as np
import torch
from peft import LoraConfig
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, PreTrainedModel

from tinyrl import wire
from tinyrl.bus import Cmd, Op, Reply
from tinyrl.helpers import DEBUG, getenv
from tinyrl.store import Store

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
  clip, kl_coeff = config.get("clip_range", 0.2), config.get("kl_coeff", 0.0)
  log_ratio = (logprobs - sampler_logprobs).clamp(-20, 20)
  ratio = log_ratio.exp()
  objective = torch.min(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)
  kl = (ratio - 1) - log_ratio
  return -(objective - kl_coeff * kl) * weights


LOSSES = {"cross_entropy": cross_entropy, "importance_sampling": importance_sampling, "ppo": ppo}


# ***** forward pass *****


def token_budget(lm: PreTrainedModel) -> int:
  """How many padded tokens one forward/backward pass can hold: TOKEN_BUDGET if it's set, otherwise as many as fit in
  half of the free GPU memory. It's measured on every call, since memory use changes as runs come and go, and an
  optimizer allocates its state on its first step."""
  if TOKEN_BUDGET:
    return TOKEN_BUDGET
  if lm.device.type != "cuda":
    return MAX_TOKEN_BUDGET

  free, _ = torch.cuda.mem_get_info(lm.device)
  free += torch.cuda.memory_reserved(lm.device) - torch.cuda.memory_allocated(lm.device)  # cached by torch, but unused

  # with gradient checkpointing, a token keeps about one hidden vector per layer. 6x that covers the layer being
  # recomputed in the backward pass, and leaves a margin.
  config = lm.config.get_text_config()
  bytes_per_token = 6 * config.num_hidden_layers * config.hidden_size * lm.dtype.itemsize
  return min(MAX_TOKEN_BUDGET, int(free / 2 / bytes_per_token))


def batches(lengths: list[int], budget: int) -> list[list[int]]:
  """Group sequences into passes, shortest first so a pass pads little. A sequence starts a new pass if joining the
  current one would go over the token budget, or make the pass more than 25% padding."""
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

  The model returns its hidden states instead of logits. The output head and the logsumexp then run CHUNK positions at a
  time under checkpointing, so the [tokens, vocab] logits never exist all at once."""
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


def capture_rng() -> dict:
  return {
    "cpu": torch.get_rng_state(),
    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    "python": random.getstate(),
    "numpy": np.random.get_state(),
  }


def restore_rng(state: dict) -> None:
  devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
  if len(state["cuda"]) != devices:
    raise ValueError("checkpoint RNG device topology does not fit this worker")
  torch.set_rng_state(state["cpu"].cpu())
  if state["cuda"]:
    torch.cuda.set_rng_state_all(state["cuda"])
  random.setstate(state["python"])
  np.random.set_state(state["numpy"])


def tuples(value):
  return tuple(tuples(v) for v in value) if isinstance(value, (list, tuple)) else value


def structural_spec(spec: dict) -> tuple:
  config, rank = spec.get("config", spec.get("lora", {})), spec.get("rank")
  options = (
    ("train_attn", True),
    ("train_mlp", True),
    ("train_unembed", True),
    ("lora_alpha", 16),
    ("lora_dropout", 0.05),
  )
  return spec.get("base"), rank, tuple(config.get(k, v) for k, v in options) if rank else ()


# ***** model *****


def load(name: str) -> PreTrainedModel:
  # bf16 on the gpu, where a LoRA run's frozen base model stays. A full fine-tune upcasts to fp32 in Model.create
  if torch.cuda.is_available():
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto")
  return AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)


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
  return targets


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
    self.specs: dict[str, dict] = {}
    self.rng: dict[str, dict] = {}
    self.inactive: set[str] = set()

  def create(self, run: str, base: str, rank: int, config: dict) -> None:
    if run in self.params:
      raise ValueError(f"model {run} already exists")
    previous = capture_rng()
    seed = config.get("seed")
    seed = int.from_bytes(os.urandom(8), "little") if seed is None else seed
    config = {**config, "seed": seed}
    try:
      random.seed(seed)
      np.random.seed(seed % 2**32)
      self._create(run, base, rank, config)
      self.specs[run] = {"base": base, "rank": rank, "config": config}
      self.rng[run] = capture_rng()
    finally:
      restore_rng(previous)

  def _create(self, run: str, base: str, rank: int, config: dict) -> None:
    if self.lm is None:  # the first run loads the base model
      self.lm, self.base, self.lora = self.load(base), base, rank > 0
      self.dtype = self.lm.dtype
      if not self.lora:
        self.lm.float()  # a full fine-tune updates fp32 master weights. Its forward pass runs in bf16, under autocast
      if GRAD_CKPT:
        self.lm.gradient_checkpointing_enable()
        if self.lora:
          self.lm.enable_input_require_grads()  # with the embeddings frozen, checkpointed layers need a gradient path

    if base != self.base or (rank > 0) != self.lora:
      raise ValueError(f"this trainer does {'lora' if self.lora else 'full'} fine-tuning of {self.base}")
    if config.get("seed") is not None:
      torch.manual_seed(config["seed"])

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

  def use(self, run: str, *, allow_inactive: bool = False) -> PreTrainedModel:
    """The model, with the run's adapter active."""
    if self.lm is None or run not in self.params:
      raise KeyError(f"model {run} is not on this trainer, it was deleted or its trainer restarted")
    if run in self.inactive and not allow_inactive:
      raise ValueError(f"model {run} is prepared but not active")
    if self.lora:
      self.lm.set_adapter(run)
    return self.lm

  @contextmanager
  def random_state(self, run: str):
    self.use(run)
    previous = capture_rng()
    try:
      restore_rng(self.rng[run])
      yield
    finally:
      self.rng[run] = capture_rng()
      restore_rng(previous)

  def forward_backward(self, run: str, data: list[wire.Datum], loss_fn: str, config: dict, forward_only: bool):
    with self.random_state(run):
      return self._forward_backward(run, data, loss_fn, config, forward_only)

  def _forward_backward(self, run: str, data: list[wire.Datum], loss_fn: str, config: dict, forward_only: bool):
    """Run the loss over every datum, accumulating gradients unless forward_only. Returns each datum's target
    logprobs, and the loss metrics."""
    if loss_fn not in LOSSES:
      raise ValueError(f"unknown loss_fn {loss_fn}, this trainer has {', '.join(LOSSES)}")

    lm = self.use(run).train(not forward_only)
    lengths = [min(len(d.tokens), len(d.inputs["target_tokens"])) for d in data]
    logprobs: list[np.ndarray] = [np.zeros(0, np.float32)] * len(data)
    total = 0.0

    with torch.set_grad_enabled(not forward_only):
      for batch in batches(lengths, token_budget(lm)):
        x = batch_tensors([data[i] for i in batch], [lengths[i] for i in batch], lm.device)
        with torch.autocast(lm.device.type, dtype=torch.bfloat16, enabled=lm.device.type == "cuda"):
          lp = target_logprobs(lm, x["ids"], x["mask"], x["targets"])
        loss = LOSSES[loss_fn](lp, x["weights"], x["sampler_logprobs"], x["advantages"], config).sum()
        if not forward_only:
          loss.backward()
        total += loss.item()

        for row, i in enumerate(batch):
          values = lp[row, : lengths[i]].detach().float()
          # keep them finite, so a JSON client can carry them
          logprobs[i] = torch.nan_to_num(values, nan=0.0).clamp(-9999, 9999).cpu().numpy()

    return logprobs, {"loss:sum": total, "loss:mean": total / max(1, len(data))}

  def optimizer(self, run: str) -> torch.optim.AdamW:
    if run not in self.optimizers:
      self.optimizers[run] = torch.optim.AdamW(self.params[run].values())
    return self.optimizers[run]

  def optim_step(self, run: str, adam: dict) -> dict[str, float]:
    self.use(run)
    optimizer = self.optimizer(run)

    # every step brings its own hyperparameters, so a learning rate schedule just works
    for group in optimizer.param_groups:
      group["lr"] = adam.get("learning_rate", 1e-4)
      group["betas"] = (adam.get("beta1", 0.9), adam.get("beta2", 0.95))
      group["eps"] = adam.get("eps", 1e-12)
      group["weight_decay"] = adam.get("weight_decay", 0.0)

    max_norm = adam.get("grad_clip_norm") or math.inf  # 0 means don't clip
    grad_norm = torch.nn.utils.clip_grad_norm_(self.params[run].values(), max_norm).item()
    optimizer.step()
    optimizer.zero_grad()
    return {"grad_norm:mean": grad_norm}

  def state(self, run: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Complete numerical state, including pending gradients and private RNG state."""
    self.use(run)
    params, opt, rng = self.params[run], self.optimizers.get(run), self.rng[run]
    tensors = {name: p.detach() for name, p in params.items()}
    grads, moments = [], []
    for name, p in params.items():
      if p.grad is not None:
        grads.append(name)
        tensors[f"__grad.{name}"] = p.grad.detach()
      if opt and p in opt.state and opt.state[p]:
        moments.append(name)
        for field in ("exp_avg", "exp_avg_sq", "step"):
          tensors[f"{name}.{field}"] = opt.state[p][field].detach()
    tensors["__rng.cpu"] = rng["cpu"]
    tensors.update({f"__rng.cuda.{i}": value for i, value in enumerate(rng["cuda"])})
    numpy = rng["numpy"]
    meta = {
      "schema": 1,
      "optimizer": True,
      "spec": self.specs[run],
      "grads": grads,
      "moments": moments,
      "optimizer_group": {k: v for k, v in opt.param_groups[0].items() if k != "params"} if opt else None,
      "rng": {
        "cuda": len(rng["cuda"]),
        "python": rng["python"],
        "numpy": [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]],
      },
    }
    return tensors, meta

  def weights(self, run: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Sampler weights and the config vLLM or PEFT reads."""
    lm = self.use(run)
    prefix = "base_model.model." if self.lora else ""
    tensors = {f"{prefix}{name}": p.detach().to(self.dtype) for name, p in self.params[run].items()}
    if self.lora:
      cfg = {k: sorted(v) if isinstance(v, set) else v for k, v in lm.peft_config[run].to_dict().items()}
    else:
      cfg = lm.config.to_dict()
    return tensors, {"schema": 1, "spec": self.specs[run], "lora": self.lora, "optimizer": False, "config": cfg}

  def restore(self, run: str, tensors: dict[str, torch.Tensor], meta: dict, optimizer: bool) -> None:
    """Validate and stage the whole snapshot before changing any live parameter."""
    self.use(run, allow_inactive=True)
    params, spec = self.params[run], meta.get("spec", {})
    if meta.get("schema") != 1 or structural_spec(spec) != structural_spec(self.specs[run]):
      raise ValueError("checkpoint specification does not fit this model")
    if optimizer and not meta.get("optimizer"):
      raise ValueError("checkpoint has no optimizer state")
    suffixes = (".exp_avg", ".exp_avg_sq", ".step")
    weights = {
      k.removeprefix("base_model.model."): v
      for k, v in tensors.items()
      if not k.startswith("__") and not k.endswith(suffixes)
    }
    if weights.keys() != params.keys():
      raise ValueError("checkpoint parameter names do not fit this model")

    def parameter_tensor(value, p):
      if value.shape != p.shape or not value.is_floating_point():
        raise ValueError("checkpoint tensor shape or dtype does not fit this model")
      return value.to(p.device, p.dtype, copy=True)

    staged = {name: parameter_tensor(weights[name], p) for name, p in params.items()}
    grads, moments = set(meta.get("grads", [])), set(meta.get("moments", []))
    if not grads <= params.keys() or not moments <= params.keys():
      raise ValueError("checkpoint gradient or optimizer names do not fit this model")
    staged_grads = {name: parameter_tensor(tensors[f"__grad.{name}"], params[name]) for name in grads}
    staged_moments = {}
    for name in moments:
      step = tensors[f"{name}.step"]
      if step.ndim != 0 or not step.is_floating_point():
        raise ValueError("checkpoint optimizer step must be a scalar")
      staged_moments[name] = {
        "step": step.to("cpu", torch.float32, copy=True),
        **{field: parameter_tensor(tensors[f"{name}.{field}"], params[name]) for field in ("exp_avg", "exp_avg_sq")},
      }
    rng = self.rng[run]
    if meta.get("optimizer"):
      info = meta["rng"]
      numpy = info["numpy"]
      rng = {
        "cpu": tensors["__rng.cpu"].clone(),
        "cuda": [tensors[f"__rng.cuda.{i}"].clone() for i in range(info["cuda"])],
        "python": tuples(info["python"]),
        "numpy": (numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]),
      }
      previous = capture_rng()
      try:
        restore_rng(rng)  # validate RNG encoding before touching weights
      finally:
        restore_rng(previous)
    new_opt = None
    if optimizer and meta.get("optimizer_group") is not None:
      group = dict(meta["optimizer_group"])
      if not group.pop("decoupled_weight_decay", True) or group.get("amsgrad"):
        raise ValueError("checkpoint has unsupported optimizer settings")
      group["betas"] = tuple(group["betas"])
      try:
        new_opt = torch.optim.AdamW(params.values(), **group)
      except (TypeError, ValueError) as error:
        raise ValueError(f"invalid checkpoint optimizer settings: {error}") from error
      for name, values in staged_moments.items():
        new_opt.state[params[name]] = values
    with torch.no_grad():
      for name, p in params.items():
        p.copy_(staged[name])
        p.grad = staged_grads.get(name) if optimizer else None
    self.optimizers.pop(run, None)
    if new_opt is not None:
      self.optimizers[run] = new_opt
    self.rng[run] = rng

  def move(self, device: torch.device | str) -> None:
    """Move a live pool, preserving parameter identities, gradients, and RNG."""
    if self.lm is None:
      return
    self.lm.to(device)
    for optimizer in self.optimizers.values():
      for state in optimizer.state.values():
        for name, value in state.items():
          if isinstance(value, torch.Tensor):
            state[name] = value.to("cpu" if name == "step" else device)

  def delete(self, run: str) -> None:
    lm = self.use(run, allow_inactive=True)
    full = not self.lora
    if self.lora:
      lm.delete_adapter(run)
    del self.params[run], self.specs[run], self.rng[run]
    self.inactive.discard(run)
    self.optimizers.pop(run, None)
    if full:
      self.lm, self.base, self.lora, self.dtype = None, "", True, torch.float32


# ***** trainer *****


def ok(body: dict) -> Reply:
  return Reply(True, wire.JSON, json.dumps(body).encode())


def fail(error: Exception) -> Reply:
  # a request the trainer can't take, like bad input or an unknown model, is the user's error. Anything else is ours.
  category = "user" if isinstance(error, (ValueError, KeyError)) else "server"
  body = {"error": f"{type(error).__name__}: {error}", "category": category}
  return Reply(False, wire.JSON, json.dumps(body).encode())


class Trainer:
  def __init__(self, bus, pool: str, model: Model | None = None, worker: str = "", store: Store | None = None):
    self.bus, self.pool = bus, pool
    self.model = model if model is not None else Model()
    self.store = store if store is not None else Store()
    self.worker = worker or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}"
    self.done = self.registered = False
    self.generations: dict[str, int] = {}
    self.pending: dict[tuple[str, str, int, str], Reply] = {}  # retained until completion is acknowledged
    self.execution = asyncio.Lock()
    self.resume_device: torch.device | None = None

  async def register(self) -> None:
    if not self.registered:
      result = await self.bus.register(self.worker, self.pool)
      if not result.get("registered"):
        raise RuntimeError("coordinator rejected this worker incarnation")
      self.registered = True

  async def keep(self) -> None:
    while not self.done:
      try:
        if not await self.bus.heartbeat(self.worker):
          self.done = True
          return
      except (ConnectionError, OSError, TimeoutError):
        pass  # commands/results remain pinned while the same coordinator reconnects
      await asyncio.sleep(1)

  async def serve(self) -> None:
    await self.register()
    while not self.done:
      await self.step()

  def park(self) -> None:
    """At an execution boundary, retain the live pool in CPU RAM, including gradients."""
    if self.model.lm is not None and self.resume_device is None:
      self.resume_device = self.model.lm.device
      self.model.move("cpu")
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

  def resume(self) -> None:
    if self.resume_device is not None:
      self.model.move(self.resume_device)
      self.resume_device = None

  async def step(self, count: int = 16, block_ms: int = 1000) -> int:
    async with self.execution:
      return await self._step(count, block_ms)

  async def _step(self, count: int, block_ms: int) -> int:
    await self.register()
    renewing = asyncio.create_task(self.keep())
    try:
      cmds = await self.bus.take(self.pool, self.worker, count, block_ms)
      for cmd in cmds:
        if self.done:
          break
        identity = (cmd.run, cmd.entry_id, cmd.generation, cmd.token)
        if identity not in self.pending:
          # Cancelling an asyncio waiter cannot stop a numerical thread. Hold
          # the lock and preserve its outcome before allowing another step.
          execution = asyncio.get_running_loop().run_in_executor(None, self.run, cmd)
          cancelled = False
          while True:
            try:
              self.pending[identity] = await asyncio.shield(execution)
              break
            except asyncio.CancelledError:
              cancelled = True
          if cancelled:
            raise asyncio.CancelledError
        while not self.done:
          try:
            status = await self.bus.finish(cmd, self.pending[identity])
            break
          except (ConnectionError, OSError, TimeoutError):
            await asyncio.sleep(0.25)
        else:
          break
        del self.pending[identity]
        if status == "stale":
          self.forget(cmd.run, cmd.generation)
        if DEBUG:
          print(f"{self.pool} {cmd.request_id} {cmd.op} {status}")
      return len(cmds)
    finally:
      renewing.cancel()
      try:
        await renewing
      except asyncio.CancelledError:
        pass

  def forget(self, run: str, generation: int) -> None:
    if self.generations.get(run) == generation:
      if run in self.model.params:
        self.model.delete(run)
      self.generations.pop(run, None)

  def run(self, cmd: Cmd) -> Reply:
    model, mutating = self.model, False
    try:
      if self.resume_device is not None and cmd.op in (Op.FORWARD_BACKWARD, Op.OPTIM_STEP, Op.CREATE_MODEL):
        raise ValueError("pool is parked; the agent must resume it before dispatch")
      generation = self.generations.get(cmd.run)
      if cmd.op not in (Op.CREATE_MODEL, Op.RESTORE, Op.DETACH) and generation != cmd.generation:
        raise ValueError("command generation does not own this local model")
      if cmd.op == Op.FORWARD_BACKWARD:
        req = wire.parse(cmd.body)
        data, config = wire.datums(req), wire.loss_config(req)
        mutating = not req.forward_only
        logprobs, metrics = model.forward_backward(
          cmd.run, data, req.loss_fn or "cross_entropy", config, req.forward_only
        )
        return Reply(True, wire.PROTO, wire.output(logprobs, metrics))

      args, artifact = json.loads(cmd.body), ""
      match cmd.op:
        case Op.CREATE_MODEL:
          model.create(cmd.run, args["base_model"], args["rank"], args["lora"])
          self.generations[cmd.run] = cmd.generation
        case Op.OPTIM_STEP:
          mutating = True
          return ok({"type": cmd.op, "metrics": model.optim_step(cmd.run, args["adam_params"])})
        case Op.SAVE_WEIGHTS | Op.SNAPSHOT:
          tensors, meta = model.state(cmd.run)
          meta["boundary"] = args.get("boundary", {"generation": cmd.generation, "command": cmd.request_id})
          artifact = self.store.put(tensors, meta)
        case Op.SAVE_WEIGHTS_FOR_SAMPLER:
          artifact = self.store.put(*model.weights(cmd.run))
        case Op.LOAD_WEIGHTS:
          mutating = True
          model.restore(cmd.run, *self.store.get(args["ref"]), args["optimizer"])
        case Op.RESTORE:
          tensors, meta = self.store.get(args["ref"])
          if structural_spec(meta.get("spec", {})) != structural_spec(args.get("spec", {})):
            raise ValueError("checkpoint specification differs from the coordinator's run")
          if not args.get("recovery") and meta.get("boundary") != args.get("boundary"):
            raise ValueError("snapshot does not match the migration boundary")
          if generation is not None:
            if generation >= cmd.generation:
              raise ValueError("a current model already occupies this run on the destination")
            self.forget(cmd.run, generation)
          spec = meta["spec"]
          model.create(cmd.run, spec["base"], spec["rank"], spec["config"])
          model.inactive.add(cmd.run)
          try:
            model.restore(cmd.run, tensors, meta, optimizer=True)
          except Exception:
            model.delete(cmd.run)
            raise
          self.generations[cmd.run] = cmd.generation
        case Op.ACTIVATE:
          model.use(cmd.run, allow_inactive=True)
          model.inactive.discard(cmd.run)
        case Op.DELETE_MODEL | Op.DETACH:
          self.forget(cmd.run, cmd.generation)
        case _:
          raise ValueError(f"unknown op {cmd.op}")
      body = {"type": cmd.op, "model_id": cmd.run}
      if "uri" in args:
        body["path"] = args["uri"]
      return Reply(True, wire.JSON, json.dumps(body).encode(), artifact=artifact)
    except Exception as error:
      reply = fail(error)
      validation = cmd.op in (Op.LOAD_WEIGHTS, Op.RESTORE) and isinstance(error, (ValueError, KeyError))
      return Reply(reply.ok, reply.content_type, reply.body, recover=mutating and not validation)


if __name__ == "__main__":
  from tinyrl.client import WorkerClient

  async def main():
    client = WorkerClient()
    try:
      await Trainer(client, sys.argv[1]).serve()
    finally:
      await client.close()

  asyncio.run(main())
