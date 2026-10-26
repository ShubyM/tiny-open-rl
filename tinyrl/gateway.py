"""The Tinker HTTP API.

Every training call becomes a command on its run's stream, and retrieve_future waits for the trainer's reply.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from tinyrl import wire
from tinyrl.bus import Bus, Op, Run
from tinyrl.helpers import getenv

WAIT = getenv("WAIT", 30.0)  # how long retrieve_future waits for a reply. The SDK gives up after 45s.


# ***** requests *****


class SessionRequest(BaseModel):
  session_id: str


class CreateModelRequest(BaseModel):
  base_model: str
  session_id: str = ""
  lora_config: dict[str, Any] | None = None
  user_metadata: dict[str, Any] | None = None


class ModelRequest(BaseModel):
  model_id: str
  seq_id: int | None = None


class OptimStepRequest(ModelRequest):
  adam_params: dict[str, Any] = {}


class SaveRequest(ModelRequest):
  path: str | None = None
  sampling_session_seq_id: int | None = None


class LoadRequest(ModelRequest):
  path: str
  optimizer: bool = False


class FutureRequest(BaseModel):
  request_id: str


def checkpoint_uri(run: str, kind: str, name: str) -> str:
  """The logical tinker://<run>/<kind>/<name> URI for a checkpoint."""
  if name.startswith("tinker://"):  # a checkpoint by URI, maybe another run's to resume from
    run, kind, name = name.removeprefix("tinker://").split("/", 2)
  if not name or name.startswith("/") or ".." in f"{run}/{kind}/{name}".split("/"):
    raise ValueError(f"bad checkpoint name {name}")
  return f"tinker://{run}/{kind}/{name}"


def pool_for(run: str, base_model: str, rank: int, metadata: dict) -> str:
  """LoRA runs on a base model share its trainer unless one asks for its own. A full fine-tune always gets its own."""
  exclusive = str(metadata.get("openrl.exclusive")).lower() == "true"
  if rank == 0 or exclusive:
    return run
  return f"lora:{base_model}"


def app(bus: Bus | None = None) -> FastAPI:
  bus = bus if bus is not None else Bus()
  api = FastAPI(title="tiny-open-rl")

  def bad_request(_: Request, e: Exception) -> JSONResponse:
    return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)

  api.add_exception_handler(ValueError, bad_request)
  api.add_exception_handler(KeyError, bad_request)

  async def submit(run: str, op: Op, body: bytes | dict, seq_id: int | None = None) -> dict[str, str]:
    if isinstance(body, dict):
      body = json.dumps(body).encode()
    request_id = await bus.submit(run, op, body, seq_id)
    if request_id is None:
      raise HTTPException(404, f"no model {run}")
    return {"request_id": request_id, "model_id": run}

  # ***** service *****

  @api.get("/api/v1/healthz")
  async def healthz():
    return {"status": "ok"}

  @api.post("/api/v1/client/config")
  @api.post("/api/v1/client/dynamic_config")
  async def client_config():
    return {}  # the SDK's defaults

  @api.get("/api/v1/get_server_capabilities")
  async def get_server_capabilities():
    return {"supported_models": []}

  @api.post("/api/v1/create_session")
  async def create_session():
    session_id = f"sess-{uuid.uuid4().hex[:12]}"
    await bus.session(session_id)
    return {"type": "create_session", "session_id": session_id}

  @api.post("/api/v1/session_heartbeat")
  async def session_heartbeat(req: SessionRequest):
    await bus.session(req.session_id)
    return {"type": "session_heartbeat"}

  @api.post("/api/v1/create_model")
  async def create_model(req: CreateModelRequest):
    metadata, lora = req.user_metadata or {}, req.lora_config or {}
    rank = 0 if metadata.get("openrl.fine_tuning_type") == "full" else int(lora.get("rank", 16))
    run = str(uuid.uuid4())
    await bus.create(Run(run, pool_for(run, req.base_model, rank, metadata), req.base_model, rank, req.session_id))
    return await submit(run, Op.CREATE_MODEL, {"base_model": req.base_model, "rank": rank, "lora": lora})

  @api.post("/api/v1/get_info")
  async def get_info(req: ModelRequest):
    run = await bus.run(req.model_id)
    if run is None:
      raise HTTPException(404, f"no model {req.model_id}")

    # the SDK reads LoRA shaped metadata, even for a full fine-tune
    return {
      "type": "get_info",
      "model_id": req.model_id,
      "model_name": run["base"],
      "is_lora": True,
      "lora_rank": run["rank"] or 16,
      "model_data": {"model_name": run["base"], "tokenizer_id": run["base"]},
    }

  # ***** training *****

  @api.post("/api/v1/forward_backward")
  async def forward_backward(request: Request):
    body = await request.body()
    if request.headers.get("content-type", "").startswith(wire.PROTO):
      parsed = wire.parse(body)
      return await submit(parsed.model_id, Op.FORWARD_BACKWARD, body, parsed.seq_id or None)

    data = json.loads(body)
    return await submit(data["model_id"], Op.FORWARD_BACKWARD, wire.from_json(data), data.get("seq_id"))

  @api.post("/api/v1/forward")  # SDKs before 0.25
  async def forward(request: Request):
    data = await request.json()
    body = wire.from_json(data, forward_only=True)
    return await submit(data["model_id"], Op.FORWARD_BACKWARD, body, data.get("seq_id"))

  @api.post("/api/v1/optim_step")
  async def optim_step(req: OptimStepRequest):
    return await submit(req.model_id, Op.OPTIM_STEP, {"adam_params": req.adam_params}, req.seq_id)

  async def save(req: SaveRequest, op: Op, kind: str):
    seq = req.sampling_session_seq_id if req.sampling_session_seq_id is not None else req.seq_id
    name = req.path or f"step-{seq if seq is not None else time.time_ns()}"
    return await submit(req.model_id, op, {"uri": checkpoint_uri(req.model_id, kind, name)}, req.seq_id)

  @api.post("/api/v1/save_weights")  # save_state, so the optimizer is saved too
  async def save_weights(req: SaveRequest):
    return await save(req, Op.SAVE_WEIGHTS, "weights")

  @api.post("/api/v1/save_weights_for_sampler")
  async def save_weights_for_sampler(req: SaveRequest):
    return await save(req, Op.SAVE_WEIGHTS_FOR_SAMPLER, "sampler_weights")

  @api.post("/api/v1/load_weights")
  async def load_weights(req: LoadRequest):
    uri = checkpoint_uri(req.model_id, "weights", req.path)
    return await submit(req.model_id, Op.LOAD_WEIGHTS, {"uri": uri, "optimizer": req.optimizer}, req.seq_id)

  @api.post("/api/v1/delete_model")
  async def delete_model(req: ModelRequest):
    await submit(req.model_id, Op.DELETE_MODEL, {}, req.seq_id)  # queued behind the run's work, which still finishes
    return {"status": "ok"}

  @api.post("/api/v1/retrieve_future")
  async def retrieve_future(req: FutureRequest, accept: str = Header("")):
    reply = await bus.result(req.request_id, WAIT)
    if reply is None:
      return {"type": "try_again", "request_id": req.request_id, "queue_state": "active"}
    if reply.content_type == wire.PROTO and wire.PROTO not in accept:
      return wire.output_json(reply.body)  # an older client, which reads JSON
    return Response(reply.body, media_type=reply.content_type)

  # ***** inspection *****

  @api.get("/api/v1/runs/{run_id}")
  async def run_info(run_id: str):
    run = await bus.run(run_id)
    if run is None:
      raise HTTPException(404, f"no model {run_id}")
    return {**run, **await bus.queue(run_id)}

  @api.get("/api/v1/sessions/{session_id}")
  async def session_info(session_id: str):
    session = await bus.session_info(session_id)
    if session is None:
      raise HTTPException(404, f"no session {session_id}")
    return session

  @api.get("/metrics")
  async def metrics():
    """Every run's counters for Prometheus. README.md has the queries for utilization, queueing delay and fairness."""
    lines = []
    for (pool, run), counters in (await bus.stats()).items():
      for name, n in counters.items():
        name, n = (f"{name[:-3]}_seconds", n / 1000) if name.endswith("_ms") else (name, n)
        lines.append(f'tinyrl_{name}_total{{pool="{pool}",run="{run}"}} {n}')
    return PlainTextResponse("\n".join(lines) + "\n")

  return api


if __name__ == "__main__":
  uvicorn.run(app(), host="0.0.0.0", port=getenv("PORT", 8000))
