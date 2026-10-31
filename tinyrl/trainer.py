"""A trainer serves one worker: the LoRA adapters that share a base model, or a single full fine-tune.

It takes commands from its worker's streams on the bus, runs each one on the model, and writes the reply back.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import math
import os
import signal
import socket
import sys
import time
from collections.abc import Callable

import numpy as np
import torch
import transformers
from peft import LoraConfig
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from tinyrl import models, wire
from tinyrl.bus import LEASE_MS, Bus, Cmd, Op, Reply
from tinyrl.helpers import DEBUG, getenv
from tinyrl.store import Store, nbytes
from tinyrl.turns import IDLE_MS, GpuTurn

TOKEN_BUDGET = getenv("TOKEN_BUDGET", 0)  # padded tokens per forward/backward pass. 0 fits it to free GPU memory
MAX_TOKEN_BUDGET = 16384  # past this, bigger passes stop getting faster and only use more memory
CHUNK = getenv("CHUNK", 1024)  # positions per output head chunk when computing logprobs
GRAD_CKPT = getenv("GRAD_CKPT", 1)  # gradient checkpointing
STRANDED_MS = 5000  # a command in flight this long while its worker's owner has nothing in flight was stranded
GPU = getenv("GPU", "")  # the GPU this trainer takes turns on, set by the node daemon. Unset, it never takes turns

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


def should_checkpoint_layers(lm: PreTrainedModel, lora: bool) -> bool:
  if "GRAD_CKPT" in os.environ:
    return bool(int(os.environ["GRAD_CKPT"]))
  if not GRAD_CKPT:
    return False
  if lora and lm.device.type == "cuda" and torch.cuda.get_device_properties(lm.device).total_memory >= 64 * 1024**3:
    return False
  return True


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
  bytes_per_token = (
    (6 if getattr(lm, "is_gradient_checkpointing", False) else 64)
    * config.num_hidden_layers
    * config.hidden_size
    * lm.dtype.itemsize
  )
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


# ***** model *****


def load(name: str) -> PreTrainedModel:
  """The class the checkpoint names in its config, which is the one vLLM serves, so the weights and adapters a
  trainer saves carry the checkpoint's parameter names and vLLM maps them like the checkpoint's own. A multimodal
  checkpoint's causal LM class names its layers differently (Qwen3.5: model.layers.* for model.language_model.*), and
  vLLM drops an adapter whose names it can't map, without an error."""
  name = models.local(name)
  arch = (AutoConfig.from_pretrained(name).architectures or [""])[0]
  cls = getattr(transformers, arch, AutoModelForCausalLM)
  # bf16 on the gpu, where a LoRA run's frozen base model stays. A full fine-tune upcasts to fp32 in Model.create
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

  def create(self, run: str, base: str, rank: int, config: dict) -> None:
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

  def use(self, run: str) -> PreTrainedModel:
    """The model, with the run's adapter active."""
    if self.lm is None or run not in self.params:
      raise KeyError(f"model {run} is not on this trainer, it was deleted or its trainer restarted")
    if self.lora:
      self.lm.set_adapter(run)
    return self.lm

  def forward_backward(self, run: str, data: list[wire.Datum], loss_fn: str, config: dict, forward_only: bool):
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
      params = list(self.params[run].values())
      self.optimizers[run] = torch.optim.AdamW(params, fused=bool(params and params[0].is_cuda))
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
    if not self.lora:
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      self.move(self.moments(), "cuda")
    optimizer.step()
    optimizer.zero_grad()
    if not self.lora:
      self.move(self.moments(), "cpu")
    self.dirty.add(run)
    return {"grad_norm:mean": grad_norm}

  def state(self, run: str) -> tuple[dict[str, torch.Tensor], dict]:
    """A run's portable training checkpoint: fp32 master weights and named AdamW state, with no pickle."""
    self.use(run)
    tensors = {name: p.detach() for name, p in self.params[run].items()}
    optim = self.optimizers.get(run)
    for name, p in self.params[run].items():
      if optim and p in optim.state:
        s = optim.state[p]
        tensors[f"{name}.exp_avg"] = s["exp_avg"].detach()
        tensors[f"{name}.exp_avg_sq"] = s["exp_avg_sq"].detach()
        tensors[f"{name}.step"] = s["step"].detach()
    return tensors, {"optimizer": True}

  def weights(self, run: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Sampler weights in the base model's compute dtype (bf16 on GPU), plus the config vLLM or PEFT reads."""
    lm = self.use(run)
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

  def restore(self, run: str, tensors: dict[str, torch.Tensor], meta: dict, optimizer: bool) -> None:
    self.use(run)
    if optimizer and not meta.get("optimizer"):
      raise ValueError("checkpoint has no optimizer state")
    suffixes = (".exp_avg", ".exp_avg_sq", ".step")
    weights = {k.removeprefix("base_model.model."): v for k, v in tensors.items() if not k.endswith(suffixes)}
    if not self.lora and "lm_head.weight" not in self.params[run]:
      weights.pop("lm_head.weight", None)  # tied to embed_tokens
    if weights.keys() != self.params[run].keys():
      raise ValueError("checkpoint does not fit this model")
    for name, p in self.params[run].items():
      p.data.copy_(weights[name])

    self.pinned.clear()
    self.optimizers.pop(run, None)
    if optimizer and any(k.endswith(".step") for k in tensors):
      opt = self.optimizer(run)
      for name, p in self.params[run].items():
        opt.state[p] = {
          "exp_avg": tensors[f"{name}.exp_avg"].to(p.device, torch.float32),
          "exp_avg_sq": tensors[f"{name}.exp_avg_sq"].to(p.device, torch.float32),
          "step": tensors[f"{name}.step"].to(p.device if opt.defaults.get("fused") else "cpu", torch.float32),
        }
      if not self.lora:
        self.move(self.moments(), "cpu")
    self.dirty.add(run)

  def moments(self):
    return (v for opt in self.optimizers.values() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))

  def tensors(self, optim: bool = True):
    """Every tensor on the GPU: weights, buffers, gradients between forward_backward and optim_step, AdamW moments."""
    if self.lm is None:
      return
    yield from self.lm.parameters()
    yield from self.lm.buffers()
    yield from (p.grad for p in self.lm.parameters() if p.grad is not None)
    if optim:
      yield from self.moments()

  def move(self, tensors, device: str) -> None:
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
        active = {id(x) for x in self.tensors()}
        for k in list(self.pinned.keys()):
          if k not in active:
            del self.pinned[k]

  def park(self) -> None:
    """Copy every GPU tensor to pinned host memory and free the GPU, for another trainer's turn. resume() undoes it."""
    self.move(self.tensors(), "cpu")
    self.parked = True

  def resume(self) -> None:
    self.move(self.tensors(optim=self.lora), "cuda")
    self.parked = False

  def delete(self, run: str) -> None:
    lm = self.use(run)
    self.dirty.discard(run)
    if self.lora:
      lm.delete_adapter(run)
    del self.params[run]
    self.pinned.clear()
    self.optimizers.pop(run, None)


# ***** trainer *****


def ok(body: dict) -> Reply:
  return Reply(True, wire.JSON, json.dumps(body).encode())


def fail(error: Exception) -> Reply:
  # a request the trainer can't take, like bad input or an unknown model, is the user's error. Anything else is ours.
  category = "user" if isinstance(error, (ValueError, KeyError)) else "server"
  body = {"error": f"{type(error).__name__}: {error}", "category": category}
  return Reply(False, wire.JSON, json.dumps(body).encode())


def enqueue_cmds(queue: collections.deque[Cmd], cmds: list[Cmd], lora: bool) -> None:
  if not lora:
    queue.extend(cmds)
    return
  blocked = {x.run for x in queue}
  front, back = [], []
  for c in cmds:
    if c.op == Op.SAVE_WEIGHTS_FOR_SAMPLER and c.run not in blocked:
      front.append(c)
    else:
      blocked.add(c.run)
      back.append(c)
  queue.extendleft(reversed(front))
  queue.extend(back)


class Trainer:
  def __init__(self, bus: Bus, worker: str, model: Model | None = None, owner: str = "", store: Store | None = None):
    self.bus = bus
    self.worker = worker
    self.model = model if model is not None else Model()
    self.store = store if store is not None else Store()
    self.owner = owner or f"{socket.gethostname()}-{os.getpid()}-{os.urandom(3).hex()}"
    self.done = False
    self.draining = False
    self.sizes: dict[str, int] = {}  # bytes each save just wrote, by request, for the checkpoint registry
    self.wanted: dict[str, str | None] = {}  # the version each load waits for, by request
    self.turn = GpuTurn(bus, worker, bool(GPU))

  async def serve(self) -> None:
    await self.own()
    renewing = asyncio.create_task(self.keep())
    try:
      await self.fail_stranded(0)  # anything in flight now was taken by a trainer that died or lost the worker
      while not self.done:
        await self.step()
        await self.fail_stranded(STRANDED_MS)
      if self.draining:
        await self.drain()
    finally:
      renewing.cancel()
      await self.leave()

  async def leave(self) -> None:
    """Hand the GPU back with nothing of ours left on it, then the worker to its next trainer."""
    if self.turn.held and self.turn.shared:
      await asyncio.to_thread(self.model.park)
    await self.turn.give_back()
    await self.bus.release(self.worker, self.owner)

  async def own(self) -> None:
    """Wait until this trainer owns its worker. Only the owner's replies land, see Bus.finish."""
    while not await self.bus.own(self.worker, self.owner):
      await asyncio.sleep(LEASE_MS / 4000)

  async def keep(self) -> None:
    """Renew the worker's lease until another trainer has it."""
    while True:  # until serve() cancels it, so the lease holds while a stopping trainer saves its runs
      await asyncio.sleep(LEASE_MS / 4000)
      if not await self.bus.renew(self.worker, self.owner) and not await self.reown():
        self.done = True
        return

  async def reown(self) -> bool:
    """Take the worker's lease again if it ran out with nobody taking it, as it does while Redis restarts. Not if
    another trainer has parked one of this trainer's runs since, which it only does after serving the worker."""
    if not await self.bus.own(self.worker, self.owner):
      return False
    if any([await self.bus.parked(run) for run in self.model.params]):
      await self.bus.release(self.worker, self.owner)
      return False
    return True

  def stop(self) -> None:
    """Finish the batch in hand, save every run for the worker's next trainer, and exit. SIGTERM calls it."""
    self.done = self.draining = True

  async def drain(self) -> None:
    for run in list(self.model.params):
      uri, (tensors, meta) = f"tinker://{run}/parked/{time.time_ns()}", self.model.state(run)
      await asyncio.to_thread(self.store.put, uri, tensors, meta)
      await self.bus.saved(uri, sum(map(nbytes, tensors.values())), owned=True, version=self.store.versions.get(uri))
      await self.bus.park(run, uri, self.worker, self.owner)

  async def recover(self, cmd: Cmd) -> None:
    """Bring back a run this trainer doesn't hold: from the state its last trainer saved when it stopped, or empty, for
    load_weights to fill. A run whose trainer died without saving it can't go on from where it was."""
    run, uri = await self.bus.run(cmd.run), await self.bus.parked(cmd.run)
    if not uri and cmd.op not in (Op.LOAD_WEIGHTS, Op.DELETE_MODEL):
      raise KeyError(
        f"model {cmd.run} lost its unsaved steps when its trainer stopped. load_state to go on from a checkpoint"
      )
    await asyncio.to_thread(self.model.create, cmd.run, run["base"], run["rank"], run.get("lora", {}))
    if uri:
      version = (await self.bus.checkpoint(uri) or {}).get("version")
      tensors, meta = await asyncio.to_thread(self.store.get, uri, version)
      await asyncio.to_thread(self.model.restore, cmd.run, tensors, meta, True)
      if await self.bus.park(cmd.run, None, self.worker, self.owner):
        await self.bus.forget(uri)  # the node daemons' sweeps delete it

  async def answer(self, cmd: Cmd) -> Reply:
    if cmd.op == Op.DELETE_MODEL and cmd.run not in self.model.params:  # nothing here to free, and its state not needed
      return ok({"type": cmd.op, "model_id": cmd.run})
    try:
      if cmd.op != Op.CREATE_MODEL and cmd.run not in self.model.params:
        await self.recover(cmd)
    except Exception as e:
      return fail(e)
    if cmd.op == Op.LOAD_WEIGHTS:  # the version saved last, which may be another node's, still uploading
      uri = json.loads(cmd.body)["uri"]
      self.wanted[cmd.request_id] = (await self.bus.checkpoint(uri) or {}).get("version")
    reply = await asyncio.to_thread(self.run, cmd)  # off the event loop, so redis stays responsive
    if reply.ok and cmd.op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER):  # registered before the reply lands
      args = json.loads(cmd.body)
      size = self.sizes.pop(cmd.request_id)
      version = self.store.versions.get(args["uri"])
      await self.bus.saved(args["uri"], size, args.get("ttl_seconds"), args.get("owned", False), version)
    return reply

  async def fail_stranded(self, min_idle_ms: int) -> None:
    for cmd in await self.bus.reclaim(self.worker, self.owner, min_idle_ms):
      reply = fail(RuntimeError(f"its trainer died, or lost the worker, during {cmd.op}"))
      await self.bus.finish(cmd, reply, self.worker, self.owner)

  async def step(self, count: int = 16, block_ms: int = 1000) -> int:
    """Take the next commands from the worker and run them. Returns how many there were."""
    if self.done:
      return 0
    take_block = min(block_ms, IDLE_MS) if self.turn.held and GPU else block_ms
    cmds = await self.bus.take(self.worker, self.owner, count, take_block)
    if not cmds:
      await self.turn.went_idle()
      await self.yield_gpu()
      return 0
    queue: collections.deque[Cmd] = collections.deque()
    enqueue_cmds(queue, cmds, self.model.lora)
    processed = 0
    async with self.acquire(len(cmds)):
      batch_start = time.time()
      ops = []
      while queue:
        cmd = queue.popleft()
        start = time.time()
        first = await self.bus.first_reply(cmd)  # no lock needed, the worker's owner is the only writer of its seq keys
        reply = first or await self.answer(cmd)
        ms = (time.time() - start) * 1000
        ops.append([cmd.op, ms])
        if not await self.bus.finish(cmd, reply, self.worker, self.owner, repeat=first is not None, started=start):
          self.done = True  # another trainer owns the worker now
          break

        if cmd.op == Op.DELETE_MODEL and reply.ok:
          await self.bus.drop(cmd.run, self.worker)
          await self.bus.forget(*await self.bus.owned(cmd.run))
          self.done = not self.model.lora  # a full fine-tune's trainer exits with its run

        if DEBUG:
          error = "" if reply.ok else f"  {reply.body.decode()}"
          print(f"{self.worker} {cmd.request_id} {cmd.op:24s} {ms:9.2f} ms{error}")

        processed += 1
        if self.model.lora and queue and (cmd.op == Op.OPTIM_STEP or queue[0].run != cmd.run):
          if cmd.op == Op.OPTIM_STEP:
            await asyncio.sleep(0.015)
          more = await self.bus.take(self.worker, self.owner, count, 0)
          enqueue_cmds(queue, more, True)
      await self.turn.batch_done(queued=len(queue), seconds=time.time() - batch_start, ops=ops)
    return processed

  @contextlib.asynccontextmanager
  async def acquire(self, queued: int):
    """Hold the GPU for one batch. Waits for this trainer's turn and brings its tensors back, then gives the turn back
    after the batch if the scheduler revoked it."""
    if await self.turn.take(queued):
      await asyncio.to_thread(self.model.resume)
    try:
      yield
    finally:
      await self.yield_gpu()

  async def yield_gpu(self) -> None:
    if await self.turn.revoked():
      await asyncio.to_thread(self.model.park)
      await self.turn.give_back()

  def run(self, cmd: Cmd) -> Reply:
    """Run one command on the model. A failure becomes the reply, never an exception."""
    model = self.model
    try:
      if cmd.op == Op.FORWARD_BACKWARD:
        req = wire.parse(cmd.body)
        data, loss_fn, config = wire.datums(req), req.loss_fn or "cross_entropy", wire.loss_config(req)
        logprobs, metrics = model.forward_backward(cmd.run, data, loss_fn, config, req.forward_only)
        return Reply(True, wire.PROTO, wire.output(logprobs, metrics))

      args = json.loads(cmd.body)
      match cmd.op:
        case Op.CREATE_MODEL:
          model.create(cmd.run, args["base_model"], args["rank"], args["lora"])
        case Op.OPTIM_STEP:
          return ok({"type": cmd.op, "metrics": model.optim_step(cmd.run, args["adam_params"])})
        case Op.SAVE_WEIGHTS | Op.SAVE_WEIGHTS_FOR_SAMPLER:
          tensors, meta = model.state(cmd.run) if cmd.op == Op.SAVE_WEIGHTS else model.weights(cmd.run)
          self.store.put(args["uri"], tensors, meta)
          self.sizes[cmd.request_id] = sum(map(nbytes, tensors.values()))
        case Op.LOAD_WEIGHTS:
          model.restore(cmd.run, *self.store.get(args["uri"], self.wanted.pop(cmd.request_id, None)), args["optimizer"])
        case Op.DELETE_MODEL:
          model.delete(cmd.run)
        case _:
          raise ValueError(f"unknown op {cmd.op}")

      reply = {"type": cmd.op, "model_id": cmd.run}
      if "uri" in args:
        reply["path"] = args["uri"]
      if "sampling_session_id" in args:
        reply["sampling_session_id"] = args["sampling_session_id"]
      return ok(reply)
    except Exception as e:
      return fail(e)


async def main(worker: str) -> None:
  trainer = Trainer(Bus(), worker)
  asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, trainer.stop)
  await trainer.serve()


if __name__ == "__main__":
  asyncio.run(main(sys.argv[1]))
