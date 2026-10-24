"""Unified TrainerRunner and ModelBackend protocol.

Replaces `training_requests_processor.py` (494 LOC), `trainer_worker.py` (427 LOC),
`lora_trainer_worker.py` (350 LOC), `fft_trainer_worker.py` (553 LOC), and
`automodel_worker.py` (630 LOC) with:
- One `TrainerRunner` owning Redis Stream consumption (`XREADGROUP`/`XACK`),
  startup PEL orphan failure (`XAUTOCLAIM`), `torchrun` rank-0 broadcast,
  token-budget microbatching, chunked logprob math, and RL loss functions.
- A 5-method `ModelBackend` protocol implemented by `TorchCausalLMBackend`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn

from tiny_open_rl import distributed, losses
from tiny_open_rl.bus import RuntimeBus, StreamEntry
from tiny_open_rl.codec import TensorDatum, decode_forward_backward_proto, encode_forward_backward_proto

MAX_PADDING_SHARE = 0.25
LOGPROB_CHUNK_TOKENS = 1024


def make_training_batches(
  data: list[TensorDatum],
  token_budget: int | None,
) -> list[list[tuple[int, TensorDatum]]]:
  """Group sequences into microbatches respecting `token_budget` and `MAX_PADDING_SHARE`."""
  if not data:
    return []
  if not token_budget or token_budget <= 0:
    return [[(i, d)] for i, d in enumerate(data)]

  batches: list[list[tuple[int, TensorDatum]]] = []
  current: list[tuple[int, TensorDatum]] = []
  max_len = 0
  sum_len = 0

  for idx, datum in enumerate(data):
    seq_len = max(1, int(datum.model_input.numel()))
    new_max = max(max_len, seq_len)
    new_count = len(current) + 1
    padded_tokens = new_max * new_count
    new_sum = sum_len + seq_len
    padding_share = (padded_tokens - new_sum) / padded_tokens if padded_tokens > 0 else 0.0

    if current and (padded_tokens > token_budget or padding_share > MAX_PADDING_SHARE):
      batches.append(current)
      current = [(idx, datum)]
      max_len = seq_len
      sum_len = seq_len
    else:
      current.append((idx, datum))
      max_len = new_max
      sum_len = new_sum

  if current:
    batches.append(current)
  return batches


def chunk_target_logprobs(
  lm_head: nn.Module,
  hidden: torch.Tensor,
  target_token_ids: torch.Tensor,
  softcap: float | None = None,
) -> torch.Tensor:
  """Compute `logit[target] - logsumexp(logits)` in chunks of `LOGPROB_CHUNK_TOKENS` rows."""
  batch_size, seq_len, hidden_dim = hidden.shape
  flat_hidden = hidden.reshape(-1, hidden_dim)
  flat_targets = target_token_ids.reshape(-1)
  out_chunks: list[torch.Tensor] = []

  for start in range(0, flat_hidden.shape[0], LOGPROB_CHUNK_TOKENS):
    end = start + LOGPROB_CHUNK_TOKENS
    logits = lm_head(flat_hidden[start:end])
    if softcap is not None:
      logits = torch.tanh(logits / softcap) * softcap
    logits = logits.float()
    target_Slice = flat_targets[start:end].clamp(min=0, max=logits.shape[-1] - 1)
    chosen = logits.gather(dim=-1, index=target_Slice.unsqueeze(-1)).squeeze(-1)
    out_chunks.append(chosen - torch.logsumexp(logits, dim=-1))

  return torch.cat(out_chunks, dim=0).reshape(batch_size, seq_len)


class ModelBackend(Protocol):
  """The 5 model-specific methods a training backend implements."""

  device: torch.device
  token_budget: int

  def create_or_load(self, run_id: str, base_model: str, config: dict[str, Any]) -> dict[str, Any]: ...

  def forward_hidden_and_head(
    self,
    run_id: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    forward_only: bool,
  ) -> tuple[torch.Tensor, nn.Module, float | None]: ...

  def optim_step(self, run_id: str, adam_params: dict[str, Any]) -> dict[str, float]: ...

  def save_checkpoint(self, run_id: str, path: str, *, for_sampler: bool) -> str: ...

  def delete(self, run_id: str) -> None: ...


class _TinyCausalLM(nn.Module):
  """Fast, deterministic in-memory causal LM for unit tests and local RL smoke runs."""

  def __init__(self, vocab_size: int = 256, hidden_size: int = 64) -> None:
    super().__init__()
    self.embed = nn.Embedding(vocab_size, hidden_size)
    self.block = nn.Sequential(
      nn.Linear(hidden_size, hidden_size),
      nn.GELU(),
      nn.LayerNorm(hidden_size),
    )
    self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

  def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
    return self.block(self.embed(input_ids))


class TorchCausalLMBackend:
  """Multi-run causal-LM backend supporting in-memory `_TinyCausalLM` and HuggingFace models."""

  def __init__(self, device: str | torch.device | None = None, token_budget: int = 4096) -> None:
    if device is not None:
      self.device = torch.device(device)
    elif torch.cuda.is_available():
      self.device = torch.device("cuda", distributed.local_rank())
    else:
      self.device = torch.device("cpu")
    self.token_budget = token_budget
    self.models: dict[str, nn.Module] = {}
    self.optimizers: dict[str, torch.optim.Optimizer] = {}

  def create_or_load(self, run_id: str, base_model: str, config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    vocab_size = int(config.get("vocab_size", 256))
    hidden_size = int(config.get("hidden_size", 64))
    model = _TinyCausalLM(vocab_size=vocab_size, hidden_size=hidden_size).to(self.device)
    self.models[run_id] = model
    self.optimizers.pop(run_id, None)
    return {
      "type": "model_created",
      "model_id": run_id,
      "base_model": base_model,
      "fine_tuning_type": config.get("fine_tuning_type", "lora"),
    }

  def forward_hidden_and_head(
    self,
    run_id: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    forward_only: bool,
  ) -> tuple[torch.Tensor, nn.Module, float | None]:
    model = self.models.get(run_id)
    if model is None:
      self.create_or_load(run_id, "tiny-gpt", {})
      model = self.models[run_id]
    model.eval() if forward_only else model.train()
    hidden = model.forward_hidden(input_ids)
    return hidden, model.lm_head, None

  def optim_step(self, run_id: str, adam_params: dict[str, Any]) -> dict[str, float]:
    model = self.models.get(run_id)
    if model is None:
      raise ValueError(f"Model {run_id!r} is not loaded")
    lr = float(adam_params.get("learning_rate", 1e-3))
    betas = (float(adam_params.get("beta1", 0.9)), float(adam_params.get("beta2", 0.999)))
    eps = float(adam_params.get("eps", 1e-8))
    weight_decay = float(adam_params.get("weight_decay", 0.0))

    opt = self.optimizers.get(run_id)
    if opt is None:
      opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
      self.optimizers[run_id] = opt
    else:
      for group in opt.param_groups:
        group["lr"] = lr
        group["betas"] = betas
        group["eps"] = eps
        group["weight_decay"] = weight_decay

    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(adam_params.get("grad_clip_norm") or 1.0)).item())
    opt.step()
    opt.zero_grad(set_to_none=True)
    return {"grad_norm": grad_norm, "learning_rate": lr}

  def save_checkpoint(self, run_id: str, path: str, *, for_sampler: bool) -> str:
    model = self.models.get(run_id)
    if model is None:
      raise ValueError(f"Model {run_id!r} is not loaded")
    out_dir = Path(path or f"/tmp/tiny-open-rl/{run_id}/{'sampler' if for_sampler else 'ckpt'}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = out_dir / "weights.pt"
    torch.save(model.state_dict(), ckpt_file)
    return str(out_dir)

  def delete(self, run_id: str) -> None:
    self.models.pop(run_id, None)
    self.optimizers.pop(run_id, None)


class TrainerRunner:
  """Single trainer execution loop across all model backends and `torchrun` ranks."""

  def __init__(
    self,
    bus: RuntimeBus,
    pool_id: str,
    worker_id: str | None = None,
    backend: ModelBackend | None = None,
  ) -> None:
    self.bus = bus
    self.pool_id = pool_id
    self.worker_id = worker_id or os.environ.get("POD_NAME", f"trainer-{pool_id}")
    self.backend = backend if backend is not None else TorchCausalLMBackend()
    self._Stopping = False

  def stop(self) -> None:
    self._Stopping = True

  async def claim_orphans(self, *, min_idle_ms: int = 0) -> int:
    """Fast-fail any unacknowledged PEL entries left by a prior crashed worker pod."""
    if not distributed.is_primary():
      return 0
    return await self.bus.fail_orphans(self.pool_id, self.worker_id, min_idle_ms=min_idle_ms)

  async def step_once(self, *, max_count: int = 64, block_ms: int = 100) -> int:
    """Read and execute one batch from `RuntimeBus`, returning the number of completed entries."""
    raw_batch = (
      await self.bus.read_batch(self.pool_id, self.worker_id, max_count=max_count, block_ms=block_ms) if distributed.is_primary() else []
    )
    entries: list[StreamEntry] = distributed.broadcast_object(raw_batch)
    if not entries:
      return 0

    for entry in entries:
      t0 = time.monotonic()
      try:
        wire, result, metrics, is_step = await self._execute_entry(entry)
        if distributed.is_primary():
          await self.bus.complete(
            entry,
            result,
            wire=wire,
            is_error=False,
            is_optim_step=is_step,
            elapsed_seconds=time.monotonic() - t0,
            metrics=metrics,
          )
      except Exception as exc:
        if distributed.is_primary():
          err = {"type": "RequestFailedResponse", "error_message": str(exc)}
          await self.bus.complete(entry, err, wire="json", is_error=True, elapsed_seconds=time.monotonic() - t0)
    return len(entries)

  async def serve(self, *, block_ms: int = 200) -> None:
    """Claim orphaned PEL entries on startup, then process batches until `stop()` is called."""
    await self.claim_orphans()
    while not self._Stopping:
      await self.step_once(block_ms=block_ms)

  async def _execute_entry(self, entry: StreamEntry) -> tuple[str, bytes | dict[str, Any], dict[str, float], bool]:
    if entry.op == "forward_backward":
      if entry.wire == "proto":
        fb = decode_forward_backward_proto(entry.body)
        outputs, metrics = self._run_forward_backward(entry.run_id, fb.data, fb.loss_fn, fb.loss_config, forward_only=fb.forward_only)
        return "proto", encode_forward_backward_proto(outputs, metrics), metrics, False
      payload = json.loads(entry.body.decode("utf-8"))
      datums = self._datums_from_json(payload.get("forward_backward_input", {}).get("data", []))
      fb_in = payload.get("forward_backward_input", {})
      outputs, metrics = self._run_forward_backward(
        entry.run_id,
        datums,
        fb_in.get("loss_fn", "cross_entropy"),
        fb_in.get("loss_fn_config") or {},
        forward_only=bool(payload.get("forward_only", False)),
      )
      json_outputs = [{k: v.detach().cpu().tolist() for k, v in row.items()} for row in outputs]
      return (
        "json",
        {"type": "forward_backward_completed", "loss_fn_output_type": "ArrayRecord", "loss_fn_outputs": json_outputs, "metrics": metrics},
        metrics,
        False,
      )

    payload = json.loads(entry.body.decode("utf-8")) if entry.body else {}
    match entry.op:
      case "create_model":
        res = self.backend.create_or_load(entry.run_id, payload.get("base_model", "tiny-gpt"), payload)
        return "json", res, {}, False
      case "optim_step":
        metrics = self.backend.optim_step(entry.run_id, payload.get("adam_params") or {})
        return "json", {"type": "optim_step_completed", "metrics": metrics}, metrics, True
      case "save_weights_for_sampler":
        saved = self.backend.save_checkpoint(entry.run_id, payload.get("path", ""), for_sampler=True)
        return "json", {"type": "sampler_weights_saved", "path": saved}, {}, False
      case "save_state":
        saved = self.backend.save_checkpoint(entry.run_id, payload.get("state_path", ""), for_sampler=False)
        return "json", {"type": "state_saved", "path": saved}, {}, False
      case "delete_model":
        self.backend.delete(entry.run_id)
        return "json", {"type": "model_deleted", "status": "ok"}, {}, False
      case _:
        raise NotImplementedError(f"Unsupported trainer operation: {entry.op!r}")

  def _run_forward_backward(
    self,
    run_id: str,
    data: list[TensorDatum],
    loss_fn: str,
    loss_config: dict[str, Any],
    *,
    forward_only: bool,
  ) -> tuple[list[dict[str, torch.Tensor]], dict[str, float]]:
    if not data:
      return [], {"loss:sum": 0.0}

    device = self.backend.device
    per_datum_outputs: list[dict[str, torch.Tensor]] = [{} for _ in data]
    microbatches = make_training_batches(data, self.backend.token_budget)
    total_loss = 0.0
    total_weight = 0.0

    with torch.set_grad_enabled(not forward_only):
      for mb in microbatches:
        indices = [idx for idx, _ in mb]
        datums = [d for _, d in mb]
        max_len = max(int(d.model_input.numel()) for d in datums)
        bsz = len(datums)

        input_ids = torch.zeros((bsz, max_len), dtype=torch.int64, device=device)
        attn_mask = torch.zeros((bsz, max_len), dtype=torch.int64, device=device)
        targets = torch.zeros((bsz, max_len), dtype=torch.int64, device=device)
        weights = torch.zeros((bsz, max_len), dtype=torch.float32, device=device)
        lengths: list[int] = []

        for row, d in enumerate(datums):
          seq_len = int(d.model_input.numel())
          lengths.append(seq_len)
          input_ids[row, :seq_len] = d.model_input.to(device=device, dtype=torch.int64)
          attn_mask[row, :seq_len] = 1
          if "target_tokens" in d.loss_fn_inputs:
            t_tok = d.loss_fn_inputs["target_tokens"].to(device=device, dtype=torch.int64).view(-1)
            targets[row, : min(seq_len, t_tok.numel())] = t_tok[:seq_len]
          if "weights" in d.loss_fn_inputs:
            w_tok = d.loss_fn_inputs["weights"].to(device=device, dtype=torch.float32).view(-1)
            weights[row, : min(seq_len, w_tok.numel())] = w_tok[:seq_len]
          else:
            weights[row, :seq_len] = 1.0

        hidden, lm_head, softcap = self.backend.forward_hidden_and_head(run_id, input_ids, attn_mask, forward_only=forward_only)
        target_logprobs = chunk_target_logprobs(lm_head, hidden, targets, softcap=softcap)

        match loss_fn:
          case "cross_entropy":
            elem_loss = losses.cross_entropy_loss(target_logprobs, weights)
          case "importance_sampling":
            ref_lp = self._pad_extra(datums, "logprobs", bsz, max_len, device)
            adv = self._pad_extra(datums, "advantages", bsz, max_len, device)
            elem_loss = losses.importance_sampling_loss(target_logprobs, ref_lp, adv)
          case "ppo":
            ref_lp = self._pad_extra(datums, "logprobs", bsz, max_len, device)
            adv = self._pad_extra(datums, "advantages", bsz, max_len, device)
            elem_loss = losses.ppo_loss(target_logprobs, ref_lp, adv, loss_config)
          case "cispo":
            ref_lp = self._pad_extra(datums, "logprobs", bsz, max_len, device)
            adv = self._pad_extra(datums, "advantages", bsz, max_len, device)
            elem_loss = losses.cispo_loss(target_logprobs, ref_lp, adv, loss_config)
          case "dro":
            ref_lp = self._pad_extra(datums, "logprobs", bsz, max_len, device)
            adv = self._pad_extra(datums, "advantages", bsz, max_len, device)
            elem_loss = losses.dro_loss(target_logprobs, ref_lp, adv, loss_config)
          case _:
            raise ValueError(f"Unknown loss_fn: {loss_fn!r}")

        mb_loss = elem_loss.sum()
        if not forward_only and mb_loss.requires_grad:
          mb_loss.backward()

        total_loss += float(mb_loss.detach().item())
        total_weight += float(weights.sum().detach().item())
        for row, orig_idx in enumerate(indices):
          seq_len = lengths[row]
          per_datum_outputs[orig_idx] = {"logprobs": target_logprobs[row, :seq_len].detach().cpu()}

    metrics = {
      "loss:sum": total_loss,
      "loss:mean": total_loss / max(1.0, total_weight),
    }
    return per_datum_outputs, metrics

  def _pad_extra(self, datums: list[TensorDatum], key: str, bsz: int, max_len: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros((bsz, max_len), dtype=torch.float32, device=device)
    for row, d in enumerate(datums):
      if key in d.loss_fn_inputs:
        t = d.loss_fn_inputs[key].to(device=device, dtype=torch.float32).view(-1)
        out[row, : min(max_len, t.numel())] = t[:max_len]
    return out

  def _datums_from_json(self, raw_data: list[dict[str, Any]]) -> list[TensorDatum]:
    datums: list[TensorDatum] = []
    for item in raw_data:
      chunks = item.get("model_input", {}).get("chunks", [])
      tokens = [tok for c in chunks for tok in c.get("tokens", [])]
      loss_in: dict[str, torch.Tensor] = {}
      for k, v in (item.get("loss_fn_inputs") or {}).items():
        vals = v.get("data", v) if isinstance(v, dict) else v
        dtype = torch.int64 if k == "target_tokens" else torch.float32
        loss_in[k] = torch.tensor(vals, dtype=dtype)
      datums.append(TensorDatum(model_input=torch.tensor(tokens, dtype=torch.int64), loss_fn_inputs=loss_in))
    return datums
