"""A trainer serves one pool: the LoRA adapters that share a base model, or a single full fine-tune.

It takes commands from its pool's streams on the bus, runs each one on the model, and writes the reply back.
"""

from __future__ import annotations

import asyncio
import glob
import json
import math
import os
import socket
import sys
import time
from collections.abc import Callable

import numpy as np
import safetensors.torch
import torch
from peft import LoraConfig, set_peft_model_state_dict
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, PreTrainedModel

from tinyrl import wire
from tinyrl.bus import Bus, Cmd, Op, Reply
from tinyrl.helpers import DEBUG, getenv

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


# ***** model *****


def load(name: str) -> PreTrainedModel:
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
    self.params: dict[str, list[torch.nn.Parameter]] = {}  # each run's trainable parameters
    self.optimizers: dict[str, torch.optim.AdamW] = {}

  def create(self, run: str, base: str, rank: int, config: dict) -> None:
    if self.lm is None:  # the first run loads the base model
      self.lm, self.base, self.lora = self.load(base), base, rank > 0
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
      self.params[run] = [p.requires_grad_() for p in self.lm.parameters()]
      return

    adapter = LoraConfig(
      r=rank,
      lora_alpha=config.get("lora_alpha", 16),
      lora_dropout=config.get("lora_dropout", 0.05),
      target_modules=lora_targets(self.lm, config),
    )
    self.lm.add_adapter(adapter, adapter_name=run)
    self.params[run] = [p for name, p in self.lm.named_parameters() if f".{run}." in name]

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
      self.optimizers[run] = torch.optim.AdamW(self.params[run])
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
    grad_norm = torch.nn.utils.clip_grad_norm_(self.params[run], max_norm).item()
    optimizer.step()
    optimizer.zero_grad()
    return {"grad_norm:mean": grad_norm}

  def save(self, run: str, path: str, optimizer: bool) -> None:
    # a LoRA run saves its adapter in PEFT format and a full fine-tune the whole model, either way what vLLM loads
    self.use(run).save_pretrained(path)
    if optimizer:
      torch.save(self.optimizer(run).state_dict(), f"{path}/optimizer.pt")

  def load_weights(self, run: str, path: str, optimizer: bool) -> None:
    lm = self.use(run)
    state = {}
    for file in glob.glob(f"{path}/*.safetensors"):
      for name, tensor in safetensors.torch.load_file(file).items():
        state[name.removeprefix("base_model.model.")] = tensor  # PEFT saves adapter weights under this prefix
    if not state:
      raise ValueError(f"no weights at {path}")
    if optimizer and not os.path.exists(f"{path}/optimizer.pt"):
      raise ValueError(f"no optimizer state at {path}")

    if self.lora:
      result = set_peft_model_state_dict(lm, state, adapter_name=run)
    else:
      result = lm.load_state_dict(state, strict=False)  # not strict: a tied output head isn't saved
    if result.unexpected_keys:
      raise ValueError(f"{path} does not fit this model: {', '.join(result.unexpected_keys[:3])}")

    self.optimizers.pop(run, None)
    if optimizer:
      self.optimizer(run).load_state_dict(torch.load(f"{path}/optimizer.pt", map_location="cpu"))

  def delete(self, run: str) -> None:
    lm = self.use(run)
    if self.lora:
      lm.delete_adapter(run)
    del self.params[run]
    self.optimizers.pop(run, None)


# ***** trainer *****


def ok(body: dict) -> Reply:
  return Reply(True, wire.JSON, json.dumps(body).encode())


def fail(error: Exception) -> Reply:
  # a request the trainer can't take, like bad input or an unknown model, is the user's error. Anything else is ours.
  category = "user" if isinstance(error, (ValueError, KeyError)) else "server"
  body = {"error": f"{type(error).__name__}: {error}", "category": category}
  return Reply(False, wire.JSON, json.dumps(body).encode())


class Trainer:
  def __init__(self, bus: Bus, pool: str, model: Model | None = None, worker: str = ""):
    self.bus = bus
    self.pool = pool
    self.model = model if model is not None else Model()
    self.worker = worker or f"{socket.gethostname()}-{os.getpid()}"
    self.done = False

  async def serve(self) -> None:
    # anything still in flight in the pool was taken by a trainer that died
    for cmd in await self.bus.reclaim(self.pool, self.worker):
      await self.bus.finish(cmd, fail(RuntimeError(f"its trainer died during {cmd.op}")))

    while not self.done:
      await self.step()

  async def step(self, count: int = 16, block_ms: int = 1000) -> int:
    """Take the next commands from the pool and run them. Returns how many there were."""
    cmds = await self.bus.take(self.pool, self.worker, count, block_ms)
    for cmd in cmds:
      start = time.perf_counter()
      reply = await asyncio.to_thread(self.run, cmd)  # off the event loop, so redis stays responsive
      await self.bus.finish(cmd, reply)

      if cmd.op == Op.DELETE_MODEL and reply.ok:
        await self.bus.drop(cmd.run, self.pool)
        self.done = not self.model.lora  # a full fine-tune's trainer exits with its run

      if DEBUG:
        ms = (time.perf_counter() - start) * 1000
        error = "" if reply.ok else f"  {reply.body.decode()}"
        print(f"{self.pool} {cmd.request_id} {cmd.op:24s} {ms:9.2f} ms{error}")
    return len(cmds)

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
          model.save(cmd.run, args["path"], optimizer=cmd.op == Op.SAVE_WEIGHTS)
        case Op.LOAD_WEIGHTS:
          model.load_weights(cmd.run, args["path"], args["optimizer"])
        case Op.DELETE_MODEL:
          model.delete(cmd.run)
        case _:
          raise ValueError(f"unknown op {cmd.op}")

      reply = {"type": cmd.op, "model_id": cmd.run}
      if "uri" in args:
        reply["path"] = args["uri"]
      return ok(reply)
    except Exception as e:
      return fail(e)


if __name__ == "__main__":
  asyncio.run(Trainer(Bus(), sys.argv[1]).serve())
