"""Unified FastAPI Gateway & Inspection API for Tiny-Open-RL.

Replaces `api_server.py` (1,399 LOC), `openrl_status.py` (139 LOC), and `dashboard/router.py` (339 LOC)
with a thin ~190-line HTTP facade over `RuntimeBus`:
- Zero-decode Protobuf pass-through on `/api/v1/forward_backward`
- Zero-encode Protobuf pass-through + `BLMOVE` wakeup on `/api/v1/retrieve_future`
- Direct `XINFO GROUPS` / `XPENDING` queue & in-flight inspection on `/api/v1/inspect/runs/{run_id}/summary`
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from tiny_open_rl.bus import RuntimeBus
from tiny_open_rl.codec import (
  PROTO_CONTENT_TYPE,
  decode_forward_backward_output_proto,
  encode_forward_backward_proto,
  peek_forward_backward_header,
)

_LABEL_UNSAFE = re.compile(r"[^a-z0-9-]+")


def pool_id_for(run_id: str, base_model: str, fine_tuning_type: str, *, exclusive: bool = False) -> str:
  """Compute the pool ID once at `create_model` time (replacing `runtime_of`, `owner_of`, and `active_set_id`)."""
  if fine_tuning_type == "full" or exclusive:
    return run_id
  stem = _LABEL_UNSAFE.sub("-", base_model.lower()).strip("-") or "default"
  return f"lora-{stem[:50]}"


class CreateSessionRequest(BaseModel):
  user_metadata: dict[str, Any] = Field(default_factory=dict)


class SessionHeartbeatRequest(BaseModel):
  session_id: str


class CreateModelRequest(BaseModel):
  base_model: str = "tiny-gpt"
  session_id: str | None = None
  fine_tuning_type: str = "lora"
  user_metadata: dict[str, Any] = Field(default_factory=dict)
  lora_config: dict[str, Any] = Field(default_factory=dict)


class OptimStepRequest(BaseModel):
  model_id: str
  session_id: str | None = None
  seq_id: int | None = None
  adam_params: dict[str, Any] = Field(default_factory=dict)


class SaveWeightsRequest(BaseModel):
  model_id: str
  path: str = ""
  session_id: str | None = None


class DeleteModelRequest(BaseModel):
  model_id: str


class RetrieveFutureRequest(BaseModel):
  request_id: str
  timeout: float = 5.0


def create_app(bus: RuntimeBus | None = None) -> FastAPI:
  app = FastAPI(title="Tiny-Open-RL Gateway")
  app.state.bus = bus if bus is not None else RuntimeBus()

  def _bus() -> RuntimeBus:
    return app.state.bus

  @app.post("/api/v1/create_session")
  async def create_session(req: CreateSessionRequest) -> dict[str, Any]:
    session_id = f"sess-{uuid.uuid4().hex[:12]}"
    await _bus().upsert_session(session_id, metadata=req.user_metadata)
    return {"session_id": session_id, "type": "create_session"}

  @app.post("/api/v1/session_heartbeat")
  async def session_heartbeat(req: SessionHeartbeatRequest) -> dict[str, str]:
    if req.session_id:
      await _bus().upsert_session(req.session_id)
    return {"type": "session_heartbeat"}

  @app.post("/api/v1/create_model")
  async def create_model(req: CreateModelRequest) -> dict[str, Any]:
    ft_type = req.user_metadata.get("openrl.fine_tuning_type", req.fine_tuning_type)
    exclusive = str(req.user_metadata.get("openrl.exclusive", "false")).lower() == "true"
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    pool_id = pool_id_for(run_id, req.base_model, ft_type, exclusive=exclusive)
    cfg = {"fine_tuning_type": ft_type, "exclusive": exclusive, **req.lora_config}

    await _bus().create_run(
      run_id=run_id,
      pool_id=pool_id,
      base_model=req.base_model,
      fine_tuning_type=ft_type,
      session_id=req.session_id,
      config=cfg,
    )
    await _bus().enqueue(
      run_id=run_id,
      req_id=run_id,
      op="create_model",
      body={"base_model": req.base_model, "fine_tuning_type": ft_type, **req.lora_config},
      wire="json",
    )
    return {"request_id": run_id, "model_id": run_id, "pool_id": pool_id, "type": "create_model"}

  @app.post("/api/v1/forward_backward")
  async def forward_backward(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    req_id = f"req-{uuid.uuid4().hex[:12]}"
    if PROTO_CONTENT_TYPE in content_type:
      raw_proto = await request.body()
      try:
        model_id, _seq_id = peek_forward_backward_header(raw_proto)
      except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
      if await _bus().get_run(model_id) is None:
        raise HTTPException(status_code=404, detail=f"No run {model_id}")
      # Zero-decode pass-through: raw Protobuf bytes go straight into the Redis Stream
      await _bus().enqueue(run_id=model_id, req_id=req_id, op="forward_backward", body=raw_proto, wire="proto")
      return {"request_id": req_id, "model_id": model_id}

    body = await request.json()
    model_id = body.get("model_id", "")
    if await _bus().get_run(model_id) is None:
      raise HTTPException(status_code=404, detail=f"No run {model_id}")
    if session_id := body.get("session_id"):
      await _bus().upsert_session(session_id, run_id=model_id)
    await _bus().enqueue(run_id=model_id, req_id=req_id, op="forward_backward", body=body, wire="json")
    return {"request_id": req_id, "model_id": model_id}

  @app.post("/api/v1/optim_step")
  async def optim_step(req: OptimStepRequest) -> dict[str, Any]:
    if await _bus().get_run(req.model_id) is None:
      raise HTTPException(status_code=404, detail=f"No run {req.model_id}")
    if req.session_id:
      await _bus().upsert_session(req.session_id, run_id=req.model_id)
    req_id = f"req-{uuid.uuid4().hex[:12]}"
    await _bus().enqueue(run_id=req.model_id, req_id=req_id, op="optim_step", body=req.model_dump(), wire="json")
    return {"request_id": req_id, "model_id": req.model_id}

  @app.post("/api/v1/save_weights_for_sampler")
  async def save_weights_for_sampler(req: SaveWeightsRequest) -> dict[str, Any]:
    if await _bus().get_run(req.model_id) is None:
      raise HTTPException(status_code=404, detail=f"No run {req.model_id}")
    req_id = f"req-{uuid.uuid4().hex[:12]}"
    await _bus().enqueue(run_id=req.model_id, req_id=req_id, op="save_weights_for_sampler", body=req.model_dump(), wire="json")
    return {"request_id": req_id, "model_id": req.model_id}

  @app.post("/api/v1/delete_model")
  async def delete_model(req: DeleteModelRequest) -> dict[str, Any]:
    await _bus().delete_run(req.model_id)
    return {"status": "deleted", "model_id": req.model_id}

  @app.post("/api/v1/retrieve_future")
  async def retrieve_future(req: RetrieveFutureRequest, request: Request) -> Any:
    wants_proto = PROTO_CONTENT_TYPE in request.headers.get("accept", "")
    wire, payload = await _bus().wait_result(req.request_id, timeout=req.timeout)
    if wire == "proto":
      if wants_proto:
        # Zero-encode pass-through: return raw Protobuf bytes directly from Redis
        return Response(content=payload, media_type=PROTO_CONTENT_TYPE)
      return decode_forward_backward_output_proto(payload)
    if wants_proto and isinstance(payload, dict) and payload.get("type") == "forward_backward_completed":
      import torch

      outputs = [{k: torch.tensor(v) for k, v in row.items()} for row in payload.get("loss_fn_outputs", [])]
      raw = encode_forward_backward_proto(outputs, payload.get("metrics") or {})
      return Response(content=raw, media_type=PROTO_CONTENT_TYPE)
    return payload

  # -- Unified Inspection & Status API (/api/v1/inspect/* + /api/v1/models/*) --

  @app.get("/api/v1/inspect/runs/{run_id}/summary")
  async def run_summary(run_id: str) -> dict[str, Any]:
    run, candidates = await _bus().resolve_run(run_id)
    if run is None:
      if len(candidates) > 1:
        ids = ", ".join(candidates)
        raise HTTPException(
          status_code=409,
          detail=f"Session {run_id} has multiple runs ({ids}); pass one run_id or read /api/v1/inspect/sessions/{run_id}/summary",
        )
      raise HTTPException(status_code=404, detail=f"No run {run_id}")

    canonical_id = run["run_id"]
    stream_stat, ops = await _bus().stream_status(canonical_id), await _bus().recent_ops(canonical_id, last=10)
    return {
      **run,
      "queue": {
        "queued": stream_stat.queued,
        "in_flight": [
          {"req_id": op.req_id, "op": op.op, "worker_id": op.worker_id, "idle_seconds": op.idle_seconds} for op in stream_stat.in_flight
        ],
      },
      "last_op": ops[-1] if ops else None,
    }

  @app.get("/api/v1/models/{model_id}/status")
  async def model_status(model_id: str) -> dict[str, Any]:
    """Thin alias over `/api/v1/inspect/runs/{model_id}/summary` (replaces `openrl_status.py`)."""
    return await run_summary(model_id)

  @app.get("/api/v1/inspect/sessions/{session_id}/summary")
  async def session_summary(session_id: str) -> dict[str, Any]:
    sess = await _bus().get_session(session_id)
    if sess is None:
      raise HTTPException(status_code=404, detail=f"No session {session_id}")
    models = [await _bus().get_run(mid) for mid in sess["model_ids"]]
    return {**sess, "models": [m for m in models if m is not None]}

  return app
