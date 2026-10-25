"""Tinker API and the Redis coordinator's worker endpoints."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, Field
from redis.exceptions import RedisError

from tinyrl import wire
from tinyrl.bus import MAX_COMMAND_BYTES, Bus, CoordinatorUnavailable, Op, Run, pack, unpack_cmd, unpack_reply
from tinyrl.helpers import getenv
from tinyrl.payloads import INLINE_BYTES, PayloadStore

WAIT = getenv("WAIT", 30.0)  # how long retrieve_future waits for a reply. The SDK gives up after 45s.


class SessionRequest(BaseModel):
  session_id: str


class CreateModelRequest(BaseModel):
  base_model: str
  session_id: str = ""
  model_seq_id: int | None = Field(default=None, ge=0)
  lora_config: dict[str, Any] | None = None
  user_metadata: dict[str, Any] | None = None
  optimizer_config: dict[str, Any] | None = None


class ModelRequest(BaseModel):
  model_id: str
  seq_id: int | None = Field(default=None, ge=1)


class OptimStepRequest(ModelRequest):
  adam_params: dict[str, Any] = Field(
    default_factory=dict, validation_alias=AliasChoices("adam_params", "optim_params", "optimizer_params")
  )


class SaveRequest(ModelRequest):
  path: str | None = None
  sampling_session_seq_id: int | None = None


class LoadRequest(ModelRequest):
  path: str
  optimizer: bool = False


class FutureRequest(BaseModel):
  request_id: str


def checkpoint_uri(run: str, kind: str, name: str) -> str:
  if name.startswith("tinker://"): run, kind, name = name.removeprefix("tinker://").split("/", 2)
  if not name or name.startswith("/") or ".." in f"{run}/{kind}/{name}".split("/"):
    raise ValueError(f"bad checkpoint name {name}")
  return f"tinker://{run}/{kind}/{name}"


def pool_for(run: str, base_model: str, rank: int, metadata: dict) -> str:
  if rank == 0 or str(metadata.get("openrl.exclusive")).lower() == "true": return run
  return f"lora:{base_model}"


def app(bus: Bus | None = None, worker_token: str | None = None, payloads: PayloadStore | None = None) -> FastAPI:
  owns_bus = bus is None
  bus = bus if bus is not None else Bus()
  worker_token = os.getenv("COORDINATOR_TOKEN", "") if worker_token is None else worker_token
  payloads = payloads if payloads is not None else PayloadStore()

  @contextlib.asynccontextmanager
  async def lifespan(_: FastAPI):
    await bus.start()

    async def reap():
      while True:
        with contextlib.suppress(RedisError, CoordinatorUnavailable): await bus.expire_workers()
        await asyncio.sleep(1)

    task = asyncio.create_task(reap())
    try: yield
    finally:
      task.cancel()
      with contextlib.suppress(asyncio.CancelledError): await task
      if owns_bus: await bus.close()

  api = FastAPI(title="tiny-open-rl", lifespan=lifespan)
  for exc in (ValueError, KeyError):
    api.add_exception_handler(exc, lambda _, e: JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400))
  unavail = JSONResponse({"error": "coordinator unavailable; retry the same request"}, status_code=503)
  for exc in (RedisError, CoordinatorUnavailable):
    api.add_exception_handler(exc, lambda _, _e: unavail)

  async def submit(run: str, op: Op, body: bytes | dict, seq: int | None = None,
                   key: str | None = None) -> dict[str, str]:
    if isinstance(body, dict): body = json.dumps(body, sort_keys=True).encode()
    if seq is not None and (type(seq) is not int or seq < 1): raise ValueError("seq_id must be a positive integer")
    if len(body) > MAX_COMMAND_BYTES: raise ValueError("command exceeds MAX_COMMAND_BYTES")
    size, ref = len(body), ""
    if size > INLINE_BYTES:
      if op != Op.FORWARD_BACKWARD: raise ValueError("command metadata exceeds the inline size limit")
      ref, body = await asyncio.to_thread(payloads.put, body), b""
    req_id = await bus.submit(run, op, body, seq_id=None if seq is None else seq - 1,
                              idempotency_key=key, body_ref=ref, body_size=size)
    if req_id is None: raise HTTPException(404, f"no model {run}")
    return {"request_id": req_id, "model_id": run}

  # ***** service *****

  @api.get("/api/v1/healthz")
  async def healthz():
    await bus.check_owner()
    return {"status": "ok"}

  @api.post("/api/v1/client/config")
  @api.post("/api/v1/client/dynamic_config")
  async def client_config(): return {}

  @api.get("/api/v1/get_server_capabilities")
  async def get_server_capabilities(): return {"supported_models": []}

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
  async def create_model(req: CreateModelRequest, idempotency_key: str | None = Header(None)):
    if (req.optimizer_config or {}).get("type", "adamw") != "adamw":
      raise ValueError("only the adamw optimizer is supported")
    metadata, lora = req.user_metadata or {}, req.lora_config or {}
    rank = 0 if metadata.get("openrl.fine_tuning_type") == "full" else int(lora.get("rank", 16))
    if req.model_seq_id is not None and not req.session_id: raise ValueError("model_seq_id requires session_id")
    token = req.model_seq_id if req.model_seq_id is not None else idempotency_key
    key = json.dumps([req.session_id, token]) if token is not None else None
    run = str(uuid.uuid5(uuid.NAMESPACE_URL, f"tinyrl:create:{key}")) if key else str(uuid.uuid4())
    body = json.dumps({"base_model": req.base_model, "rank": rank, "lora": lora}, sort_keys=True).encode()
    pool = pool_for(run, req.base_model, rank, metadata)
    rec = await bus.create(Run(run, pool, req.base_model, rank, req.session_id), body=body, idempotency_key=key)
    return {"request_id": rec["request_id"], "model_id": rec["id"]}

  @api.post("/api/v1/get_info")
  async def get_info(req: ModelRequest):
    run = await bus.run(req.model_id)
    if run is None: raise HTTPException(404, f"no model {req.model_id}")
    return {
      "type": "get_info", "model_id": req.model_id, "model_name": run["base"], "is_lora": True,
      "lora_rank": run["rank"] or 16, "model_data": {"model_name": run["base"], "tokenizer_id": run["base"]},
    }

  # ***** training *****

  @api.post("/api/v1/forward_backward")
  async def forward_backward(request: Request, idempotency_key: str | None = Header(None)):
    body = await request.body()
    if request.headers.get("content-type", "").startswith(wire.PROTO):
      parsed = wire.parse(body)
      return await submit(parsed.model_id, Op.FORWARD_BACKWARD, body, parsed.seq_id or None, idempotency_key)
    data = json.loads(body)
    return await submit(data["model_id"], Op.FORWARD_BACKWARD, wire.from_json(data),
                        data.get("seq_id"), idempotency_key)

  @api.post("/api/v1/forward")
  async def forward(request: Request, idempotency_key: str | None = Header(None)):
    data = await request.json()
    return await submit(data["model_id"], Op.FORWARD_BACKWARD, wire.from_json(data, forward_only=True),
                        data.get("seq_id"), idempotency_key)

  @api.post("/api/v1/optim_step")
  async def optim_step(req: OptimStepRequest, idempotency_key: str | None = Header(None)):
    if req.adam_params.get("type", "adamw") != "adamw": raise ValueError("only the adamw optimizer is supported")
    return await submit(req.model_id, Op.OPTIM_STEP, {"adam_params": req.adam_params}, req.seq_id, idempotency_key)

  async def save(req: SaveRequest, op: Op, kind: str, idem: str | None):
    seq = req.sampling_session_seq_id if req.sampling_session_seq_id is not None else req.seq_id
    suffix = seq if seq is not None else hashlib.sha256(idem.encode()).hexdigest() if idem else time.time_ns()
    uri = checkpoint_uri(req.model_id, kind, req.path or f"step-{suffix}")
    return await submit(req.model_id, op, {"uri": uri}, req.seq_id, idem)

  @api.post("/api/v1/save_weights")
  async def save_weights(req: SaveRequest, idempotency_key: str | None = Header(None)):
    return await save(req, Op.SAVE_WEIGHTS, "weights", idempotency_key)

  @api.post("/api/v1/save_weights_for_sampler")
  async def save_weights_for_sampler(req: SaveRequest, idempotency_key: str | None = Header(None)):
    return await save(req, Op.SAVE_WEIGHTS_FOR_SAMPLER, "sampler_weights", idempotency_key)

  @api.post("/api/v1/load_weights")
  async def load_weights(req: LoadRequest, idempotency_key: str | None = Header(None)):
    body = {"uri": checkpoint_uri(req.model_id, "weights", req.path), "optimizer": req.optimizer}
    return await submit(req.model_id, Op.LOAD_WEIGHTS, body, req.seq_id, idempotency_key)

  @api.post("/api/v1/delete_model")
  async def delete_model(req: ModelRequest, idempotency_key: str | None = Header(None)):
    await submit(req.model_id, Op.DELETE_MODEL, {}, req.seq_id, idempotency_key)
    return {"status": "ok"}

  @api.post("/api/v1/retrieve_future")
  async def retrieve_future(req: FutureRequest, accept: str = Header("")):
    res = await bus.result(req.request_id, WAIT)
    if res is None: return {"type": "try_again", "request_id": req.request_id, "queue_state": "active"}
    body = res.body
    if res.body_ref:
      try: body = await asyncio.to_thread(payloads.get, res.body_ref, res.body_size)
      except (OSError, ValueError) as e:
        raise HTTPException(503, "result payload unavailable; retry this future") from e
    if res.content_type == wire.PROTO and wire.PROTO not in accept: return wire.output_json(body)
    return Response(body, media_type=res.content_type)

  # ***** inspection & worker protocol *****

  @api.get("/api/v1/runs/{run_id}")
  async def run_info(run_id: str):
    run = await bus.run(run_id)
    if run is None: raise HTTPException(404, f"no model {run_id}")
    return {**run, **await bus.queue(run_id)}

  @api.get("/api/v1/sessions/{session_id}")
  async def session_info(session_id: str):
    session = await bus.session_info(session_id)
    if session is None: raise HTTPException(404, f"no session {session_id}")
    return session

  @api.post("/api/v1/resolve_checkpoint")
  async def resolve_checkpoint(request: Request):
    uri = (await request.json())["path"]
    ref = await bus.ckpt(uri)
    if ref is None: raise HTTPException(404, f"no checkpoint {uri}")
    return {"ref": ref}

  def authorized(authorization: str = Header("")):
    if not worker_token or not hmac.compare_digest(authorization, f"Bearer {worker_token}"):
      raise HTTPException(401, "coordinator worker token required")

  internal = APIRouter(prefix="/internal", dependencies=[Depends(authorized)])

  @internal.post("/register")
  async def register(request: Request):
    args = await request.json()
    return await bus.register(args["worker"], args["pool"])

  @internal.post("/heartbeat")
  async def heartbeat(request: Request):
    return {"live": await bus.heartbeat((await request.json())["worker"])}

  @internal.post("/take")
  async def take(request: Request):
    args = await request.json()
    count, block = int(args.get("count", 16)), int(args.get("block_ms", 1000))
    if not 1 <= count <= 256 or not 0 <= block <= 30000: raise ValueError("invalid poll bounds")
    return {"commands": [pack(c) for c in await bus.take(args["pool"], args["worker"], count=count, block_ms=block)]}

  @internal.post("/finish")
  async def finish(request: Request):
    args = await request.json()
    return {"status": await bus.finish(unpack_cmd(args["command"]), unpack_reply(args["reply"]))}

  @internal.post("/move")
  async def move(request: Request):
    args = await request.json()
    return await bus.request_move(args["run_id"], args["target_pool"])

  @internal.post("/recover")
  async def recover(request: Request):
    args = await request.json()
    if ("path" in args) == ("ref" in args):
      raise ValueError("provide either a checkpoint path or an immutable snapshot ref")
    ref = await bus.ckpt(args["path"]) if "path" in args else args["ref"]
    if ref is None: raise HTTPException(404, "no recovery checkpoint")
    return await bus.recover(args["run_id"], ref, args.get("target_pool"))

  api.include_router(internal)
  return api


if __name__ == "__main__":
  uvicorn.run(app(), host="0.0.0.0", port=getenv("PORT", 8000))
