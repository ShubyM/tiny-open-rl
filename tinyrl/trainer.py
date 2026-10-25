"""A trainer serves one pool: the LoRA adapters that share a base model, or a single full fine-tune.

It executes coordinator-issued commands over HTTP and retains results until the coordinator acknowledges them.
"""

from __future__ import annotations

import asyncio
import contextlib
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
  return -(objective - kl_coeff * ((ratio - 1) - log_ratio)) * weights


LOSSES = {"cross_entropy": cross_entropy, "importance_sampling": importance_sampling, "ppo": ppo}
LORA_DEFAULTS = (
  ("train_attn", True), ("train_mlp", True), ("train_unembed", True), ("lora_alpha", 16), ("lora_dropout", 0.05),
)


# ***** forward pass *****


def token_budget(lm: PreTrainedModel) -> int:
  """How many padded tokens one forward/backward pass can hold: TOKEN_BUDGET if set, else half of free GPU memory."""
  if TOKEN_BUDGET: return TOKEN_BUDGET
  if lm.device.type != "cuda": return MAX_TOKEN_BUDGET
  free, _ = torch.cuda.mem_get_info(lm.device)
  free += torch.cuda.memory_reserved(lm.device) - torch.cuda.memory_allocated(lm.device)
  cfg = lm.config.get_text_config()
  return min(MAX_TOKEN_BUDGET, int(free / 2 / (6 * cfg.num_hidden_layers * cfg.hidden_size * lm.dtype.itemsize)))


def batches(lengths: list[int], budget: int) -> list[list[int]]:
  """Group sequences into passes, shortest first so a pass pads little (<= 25% padding, within budget)."""
  passes: list[list[int]] = []
  for i in sorted(range(len(lengths)), key=lengths.__getitem__):
    if passes:
      cur = passes[-1]
      padded, real = lengths[i] * (len(cur) + 1), lengths[i] + sum(lengths[j] for j in cur)
      if padded <= budget and padded - real <= padded / 4:
        cur.append(i)
        continue
    passes.append([i])
  return passes


def pad(rows: list[np.ndarray], lengths: list[int], dtype, device: torch.device) -> torch.Tensor:
  out = np.zeros((len(rows), max(lengths)), dtype)
  for i, (row, n) in enumerate(zip(rows, lengths)): out[i, :n] = row[:n]
  return torch.from_numpy(out).to(device)


def batch_tensors(data: list[wire.Datum], lengths: list[int], device: torch.device) -> dict[str, torch.Tensor]:
  def col(name: str, default: float, dtype) -> torch.Tensor:
    return pad([d.inputs.get(name, np.full(n, default)) for d, n in zip(data, lengths)], lengths, dtype, device)

  return {
    "ids": pad([d.tokens for d in data], lengths, np.int64, device),
    "mask": pad([np.ones(n) for n in lengths], lengths, np.int64, device),
    "targets": col("target_tokens", 0, np.int64),
    "weights": col("weights", 1.0, np.float32),
    "sampler_logprobs": col("logprobs", 0.0, np.float32),
    "advantages": col("advantages", 0.0, np.float32),
  }


def target_logprobs(lm: PreTrainedModel, ids: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
  """Compute target-token logprobs CHUNK positions at a time under checkpointing without materializing full logits."""
  out = lm(ids, attention_mask=mask, use_cache=False, logits_to_keep=1, output_hidden_states=True)
  hidden, head = out.hidden_states[-1].flatten(0, 1), lm.get_output_embeddings()
  softcap = getattr(lm.config.get_text_config(), "final_logit_softcapping", None)

  def chunk_lp(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    logits = head(h)
    if softcap: logits = torch.tanh(logits / softcap) * softcap
    logits = logits.float()
    return logits.gather(-1, t[:, None])[:, 0] - logits.logsumexp(-1)

  chunks = zip(hidden.split(CHUNK), targets.flatten().split(CHUNK))
  return torch.cat([checkpoint(chunk_lp, h, t, use_reentrant=False) for h, t in chunks]).view(targets.shape)


def capture_rng() -> dict:
  return {
    "cpu": torch.get_rng_state(),
    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    "python": random.getstate(),
    "numpy": np.random.get_state(),
  }


def restore_rng(state: dict) -> None:
  if len(state["cuda"]) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
    raise ValueError("checkpoint RNG device topology does not fit this worker")
  torch.set_rng_state(state["cpu"].cpu())
  if state["cuda"]: torch.cuda.set_rng_state_all(state["cuda"])
  random.setstate(state["python"])
  np.random.set_state(state["numpy"])


def _tuples(v): return tuple(_tuples(x) for x in v) if isinstance(v, (list, tuple)) else v


def structural_spec(spec: dict) -> tuple:
  cfg, rank = spec.get("config", spec.get("lora", {})), spec.get("rank")
  return spec.get("base"), rank, tuple(cfg.get(k, v) for k, v in LORA_DEFAULTS) if rank else ()


# ***** model *****


def load(name: str) -> PreTrainedModel:
  if torch.cuda.is_available():
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto")
  return AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)


def lora_targets(lm: PreTrainedModel, config: dict) -> list[str]:
  targets = []
  if config.get("train_attn", True): targets += ["q_proj", "k_proj", "v_proj", "o_proj"]
  if config.get("train_mlp", True): targets += ["gate_proj", "up_proj", "down_proj"]
  if config.get("train_unembed", True) and not getattr(lm.config.get_text_config(), "tie_word_embeddings", False):
    targets.append("lm_head")
  return targets


class Model:
  """A base model and the runs training on it: a LoRA adapter per run, or the whole model for one full fine-tune."""

  def __init__(self, load: Callable[[str], PreTrainedModel] = load):
    self.load = load
    self.lm: PreTrainedModel | None = None
    self.base, self.lora, self.dtype = "", True, torch.float32
    self.params: dict[str, dict[str, torch.nn.Parameter]] = {}
    self.optimizers: dict[str, torch.optim.AdamW] = {}
    self.specs: dict[str, dict] = {}
    self.rng: dict[str, dict] = {}
    self.inactive: set[str] = set()

  def create(self, run: str, base: str, rank: int, config: dict) -> None:
    if run in self.params: raise ValueError(f"model {run} already exists")
    prev = capture_rng()
    seed = int.from_bytes(os.urandom(8), "little") if config.get("seed") is None else config["seed"]
    config = {**config, "seed": seed}
    try:
      random.seed(seed)
      np.random.seed(seed % 2**32)
      self._create(run, base, rank, config)
      self.specs[run] = {"base": base, "rank": rank, "config": config}
      self.rng[run] = capture_rng()
    finally: restore_rng(prev)

  def _create(self, run: str, base: str, rank: int, config: dict) -> None:
    if self.lm is None:
      self.lm, self.base, self.lora = self.load(base), base, rank > 0
      self.dtype = self.lm.dtype
      if not self.lora: self.lm.float()  # fp32 master weights; GPU forward pass runs in bf16 under autocast
      if GRAD_CKPT:
        self.lm.gradient_checkpointing_enable()
        if self.lora: self.lm.enable_input_require_grads()
    if base != self.base or (rank > 0) != self.lora:
      raise ValueError(f"this trainer does {'lora' if self.lora else 'full'} fine-tuning of {self.base}")
    if config.get("seed") is not None: torch.manual_seed(config["seed"])
    if not self.lora:
      if self.params:
        raise ValueError(f"a full fine-tune has its trainer to itself, and {next(iter(self.params))} is on this one")
      self.params[run] = {name: p.requires_grad_() for name, p in self.lm.named_parameters()}
      return
    adapter = LoraConfig(r=rank, lora_alpha=config.get("lora_alpha", 16),
                         lora_dropout=config.get("lora_dropout", 0.05), target_modules=lora_targets(self.lm, config))
    self.lm.add_adapter(adapter, adapter_name=run)
    infix = f".{run}."
    self.params[run] = {name.replace(infix, "."): p for name, p in self.lm.named_parameters() if infix in name}
    for p in self.params[run].values(): p.data = p.data.float()

  def use(self, run: str, *, allow_inactive: bool = False) -> PreTrainedModel:
    if self.lm is None or run not in self.params:
      raise KeyError(f"model {run} is not on this trainer, it was deleted or its trainer restarted")
    if run in self.inactive and not allow_inactive: raise ValueError(f"model {run} is prepared but not active")
    if self.lora: self.lm.set_adapter(run)
    return self.lm

  @contextmanager
  def random_state(self, run: str):
    self.use(run)
    prev = capture_rng()
    try:
      restore_rng(self.rng[run])
      yield
    finally:
      self.rng[run] = capture_rng()
      restore_rng(prev)

  def forward_backward(self, run: str, data: list[wire.Datum], loss_fn: str, config: dict, forward_only: bool):
    if loss_fn not in LOSSES: raise ValueError(f"unknown loss_fn {loss_fn}, this trainer has {', '.join(LOSSES)}")
    with self.random_state(run):
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
          if not forward_only: loss.backward()
          total += loss.item()
          for row, i in enumerate(batch):
            vals = lp[row, :lengths[i]].detach().float()
            logprobs[i] = torch.nan_to_num(vals, nan=0.0).clamp(-9999, 9999).cpu().numpy()
      return logprobs, {"loss:sum": total, "loss:mean": total / max(1, len(data))}

  def optimizer(self, run: str) -> torch.optim.AdamW:
    if run not in self.optimizers: self.optimizers[run] = torch.optim.AdamW(self.params[run].values())
    return self.optimizers[run]

  def optim_step(self, run: str, adam: dict) -> dict[str, float]:
    self.use(run)
    opt = self.optimizer(run)
    for g in opt.param_groups:
      g["lr"], g["betas"] = adam.get("learning_rate", 1e-4), (adam.get("beta1", 0.9), adam.get("beta2", 0.95))
      g["eps"], g["weight_decay"] = adam.get("eps", 1e-12), adam.get("weight_decay", 0.0)
    grad_norm = torch.nn.utils.clip_grad_norm_(self.params[run].values(), adam.get("grad_clip_norm") or math.inf).item()
    opt.step()
    opt.zero_grad()
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
        for field in ("exp_avg", "exp_avg_sq", "step"): tensors[f"{name}.{field}"] = opt.state[p][field].detach()
    tensors["__rng.cpu"] = rng["cpu"]
    tensors.update({f"__rng.cuda.{i}": v for i, v in enumerate(rng["cuda"])})
    np_rng = rng["numpy"]
    meta = {
      "schema": 1, "optimizer": True, "spec": self.specs[run], "grads": grads, "moments": moments,
      "optimizer_group": {k: v for k, v in opt.param_groups[0].items() if k != "params"} if opt else None,
      "rng": {"cuda": len(rng["cuda"]), "python": rng["python"], "numpy": [np_rng[0], np_rng[1].tolist(), *np_rng[2:]]},
    }
    return tensors, meta

  def weights(self, run: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Sampler weights and the config vLLM or PEFT reads."""
    lm = self.use(run)
    prefix = "base_model.model." if self.lora else ""
    tensors = {f"{prefix}{name}": p.detach().to(self.dtype) for name, p in self.params[run].items()}
    cfg = ({k: sorted(v) if isinstance(v, set) else v for k, v in lm.peft_config[run].to_dict().items()}
           if self.lora else lm.config.to_dict())
    return tensors, {"schema": 1, "spec": self.specs[run], "lora": self.lora, "optimizer": False, "config": cfg}

  def restore(self, run: str, tensors: dict[str, torch.Tensor], meta: dict, optimizer: bool) -> None:
    """Validate and stage the whole snapshot before changing any live parameter."""
    self.use(run, allow_inactive=True)
    params = self.params[run]
    if meta.get("schema") != 1 or structural_spec(meta.get("spec", {})) != structural_spec(self.specs[run]):
      raise ValueError("checkpoint specification does not fit this model")
    if optimizer and not meta.get("optimizer"): raise ValueError("checkpoint has no optimizer state")

    def cast(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
      if t.shape != like.shape or not t.is_floating_point():
        raise ValueError("checkpoint tensor shape or dtype does not fit this model")
      return t.to(like.device, like.dtype, copy=True)

    weights = {k.removeprefix("base_model.model."): v for k, v in tensors.items()
               if not k.startswith("__") and not k.endswith((".exp_avg", ".exp_avg_sq", ".step"))}
    if weights.keys() != params.keys(): raise ValueError("checkpoint parameter names do not fit this model")
    staged = {k: cast(weights[k], p) for k, p in params.items()}

    grads, moments = set(meta.get("grads", [])), set(meta.get("moments", []))
    if not (grads <= params.keys() and moments <= params.keys()):
      raise ValueError("checkpoint gradient or optimizer names do not fit this model")
    staged_grads = {k: cast(tensors[f"__grad.{k}"], params[k]) for k in grads}
    staged_moments = {}
    for k in moments:
      step = tensors[f"{k}.step"]
      if step.ndim != 0 or not step.is_floating_point(): raise ValueError("checkpoint optimizer step must be a scalar")
      staged_moments[k] = {
        "step": step.to("cpu", torch.float32, copy=True),
        "exp_avg": cast(tensors[f"{k}.exp_avg"], params[k]),
        "exp_avg_sq": cast(tensors[f"{k}.exp_avg_sq"], params[k]),
      }

    rng = self.rng[run]
    if meta.get("optimizer"):
      info, np_rng = meta["rng"], meta["rng"]["numpy"]
      rng = {
        "cpu": tensors["__rng.cpu"].clone(),
        "cuda": [tensors[f"__rng.cuda.{i}"].clone() for i in range(info["cuda"])],
        "python": _tuples(info["python"]),
        "numpy": (np_rng[0], np.array(np_rng[1], dtype=np.uint32), *np_rng[2:]),
      }
      prev = capture_rng()
      try: restore_rng(rng)
      finally: restore_rng(prev)

    new_opt = None
    if optimizer and meta.get("optimizer_group") is not None:
      group = {**meta["optimizer_group"], "betas": tuple(meta["optimizer_group"]["betas"])}
      if not group.pop("decoupled_weight_decay", True) or group.get("amsgrad"):
        raise ValueError("checkpoint has unsupported optimizer settings")
      try: new_opt = torch.optim.AdamW(params.values(), **group)
      except (TypeError, ValueError) as e: raise ValueError(f"invalid checkpoint optimizer settings: {e}") from e
      for k, state in staged_moments.items(): new_opt.state[params[k]] = state

    with torch.no_grad():
      for k, p in params.items():
        p.copy_(staged[k])
        p.grad = staged_grads.get(k) if optimizer else None
    self.optimizers.pop(run, None)
    if new_opt is not None: self.optimizers[run] = new_opt
    self.rng[run] = rng

  def move(self, device: torch.device | str) -> None:
    if self.lm is None: return
    self.lm.to(device)
    for opt in self.optimizers.values():
      for state in opt.state.values():
        for k, v in state.items():
          if isinstance(v, torch.Tensor): state[k] = v.to("cpu" if k == "step" else device)

  def delete(self, run: str) -> None:
    lm = self.use(run, allow_inactive=True)
    if self.lora: lm.delete_adapter(run)
    del self.params[run], self.specs[run], self.rng[run]
    self.inactive.discard(run)
    self.optimizers.pop(run, None)
    if not self.lora: self.lm, self.base, self.lora, self.dtype = None, "", True, torch.float32


# ***** trainer *****


def ok(body: dict) -> Reply: return Reply(True, wire.JSON, json.dumps(body).encode())


def fail(error: Exception) -> Reply:
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
    self.pending: dict[tuple[str, str, int, str], Reply] = {}
    self.execution = asyncio.Lock()
    self.resume_device: torch.device | None = None

  async def register(self) -> None:
    if not self.registered:
      if not (await self.bus.register(self.worker, self.pool)).get("registered"):
        raise RuntimeError("coordinator rejected this worker incarnation")
      self.registered = True

  async def keep(self) -> None:
    while not self.done:
      with contextlib.suppress(ConnectionError, OSError, TimeoutError):
        if not await self.bus.heartbeat(self.worker):
          self.done = True
          return
      await asyncio.sleep(1)

  async def serve(self) -> None:
    await self.register()
    while not self.done: await self.step()

  def park(self) -> None:
    if self.model.lm is not None and self.resume_device is None:
      self.resume_device = self.model.lm.device
      self.model.move("cpu")
      if torch.cuda.is_available(): torch.cuda.empty_cache()

  def resume(self) -> None:
    if self.resume_device is not None:
      self.model.move(self.resume_device)
      self.resume_device = None

  async def _execute(self, identity: tuple, cmd: Cmd) -> None:
    if identity in self.pending: return
    fut = asyncio.get_running_loop().run_in_executor(None, self.run, cmd)
    cancelled = False
    while True:
      try:
        self.pending[identity] = await asyncio.shield(fut)
        break
      except asyncio.CancelledError: cancelled = True
    if cancelled: raise asyncio.CancelledError

  async def step(self, count: int = 16, block_ms: int = 1000) -> int:
    async with self.execution:
      await self.register()
      renewing = asyncio.create_task(self.keep())
      try:
        cmds = await self.bus.take(self.pool, self.worker, count, block_ms)
        for cmd in cmds:
          if self.done: break
          identity = (cmd.run, cmd.entry_id, cmd.generation, cmd.token)
          await self._execute(identity, cmd)
          while not self.done:
            try:
              status = await self.bus.finish(cmd, self.pending[identity])
              break
            except (ConnectionError, OSError, TimeoutError): await asyncio.sleep(0.25)
          else: break
          del self.pending[identity]
          if status == "stale": self.forget(cmd.run, cmd.generation)
          if DEBUG: print(f"{self.pool} {cmd.request_id} {cmd.op} {status}")
        return len(cmds)
      finally:
        renewing.cancel()
        with contextlib.suppress(asyncio.CancelledError): await renewing

  def forget(self, run: str, generation: int) -> None:
    if self.generations.get(run) == generation:
      if run in self.model.params: self.model.delete(run)
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
        lp, metrics = model.forward_backward(cmd.run, data, req.loss_fn or "cross_entropy", config, req.forward_only)
        return Reply(True, wire.PROTO, wire.output(lp, metrics))

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
          try: model.restore(cmd.run, tensors, meta, optimizer=True)
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
      body = {"type": cmd.op, "model_id": cmd.run, **({"path": args["uri"]} if "uri" in args else {})}
      return Reply(True, wire.JSON, json.dumps(body).encode(), artifact=artifact)
    except Exception as error:
      reply = fail(error)
      validation = cmd.op == Op.LOAD_WEIGHTS and isinstance(error, (ValueError, KeyError))
      return Reply(reply.ok, reply.content_type, reply.body, recover=mutating and not validation)


if __name__ == "__main__":
  from tinyrl.client import WorkerClient

  async def main():
    client = WorkerClient()
    try: await Trainer(client, sys.argv[1]).serve()
    finally: await client.close()

  asyncio.run(main())
