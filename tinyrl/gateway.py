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

from tinyrl import schedule, wire
from tinyrl.bus import Bus, Op, Run
from tinyrl.helpers import getenv
from tinyrl.store import Store

WAIT = getenv("WAIT", 30.0)  # how long retrieve_future waits for a reply. The SDK gives up after 45s.
TARGET_UTIL = getenv("TARGET_UTIL", 0.7)  # a LoRA shard busier than this gets no new runs while another can open
NEW_RUN_UTIL = float(getenv("NEW_RUN_UTIL", 0.04))  # the load a new run adds before any shard on its base has reported


# ***** requests *****


class SessionRequest(BaseModel):
  session_id: str


class CreateSessionRequest(BaseModel):
  tags: list[str] = []  # TINKER_TAGS: open-rl clients ask for full fine-tuning with openrl.fine_tuning_type=full
  user_metadata: dict[str, Any] | None = None


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


async def pool_for(bus: Bus, run: str, base_model: str, rank: int, metadata: dict) -> str:
  exclusive = str(metadata.get("openrl.exclusive")).lower() == "true"
  shards = [p for p in await bus.pools() if p.startswith(f"lora:{base_model}:")]
  loads = await bus.loads(shards)
  shard_loads = [schedule.ShardLoad(p, u, placed, runs) for p, (u, placed, runs) in zip(shards, loads)]
  gpus = await bus.gpus()
  dec = schedule.choose_pool(run, base_model, rank, exclusive, shard_loads, gpus, TARGET_UTIL, NEW_RUN_UTIL)
  await bus.decide("run", run, dec.as_dict())
  if dec.choice != run:
    await bus.placed(dec.choice)
  return dec.choice


def app(bus: Bus | None = None, store: Store | None = None) -> FastAPI:
  bus = bus if bus is not None else Bus()
  store = store if store is not None else Store()
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

  async def open_sampler(base_model: str | None, model_path: str | None, session_id: str = "") -> str:
    if model_path:
      if not model_path.startswith("tinker://"):
        raise ValueError(f"bad model_path {model_path}")
      uri = checkpoint_uri("x", "sampler_weights", model_path)
      train_run = uri.removeprefix("tinker://").split("/", 1)[0]
      train = await bus.run(train_run)
      if train and train["base"]:
        base, lora = train["base"], train["rank"] > 0
      else:
        meta = store.meta(uri)
        base, lora = meta.get("base") or base_model, bool(meta.get("lora"))
      if not base:
        raise ValueError("could not resolve base_model")
    else:
      if not base_model:
        raise ValueError("base_model required when model_path is None")
      base, lora, uri, train_run = base_model, True, None, None

    exclusive = train_run in await bus.exclusive() if train_run else False
    dec = schedule.choose_sampler_pool(base, lora, train_run, exclusive)
    await bus.warm_sampler(dec.choice, base, exclusive=exclusive)
    sid = f"samp-{uuid.uuid4().hex[:12]}"
    await bus.decide("run", sid, dec.as_dict())
    await bus.create(Run(sid, dec.choice, base, int(lora), session_id, model_path=uri, exclusive=exclusive))
    return sid

  # ***** service *****

  @api.get("/api/v1/scheduling")
  async def scheduling():
    return {
      "nodes": await bus.nodes(),
      "placements": await bus.placements(),
      "exclusive": sorted(await bus.exclusive()),
      "gpu_leases": await bus.gpu_fairness(),
      "decisions": await bus.decisions(),
    }

  @api.get("/api/v1/healthz")
  @api.post("/api/v1/sessions/{session_id}/finish")
  async def healthz(session_id: str = ""):
    return {"status": "ok"}

  @api.post("/api/v1/telemetry")
  async def telemetry():
    return {"status": "accepted"}

  @api.post("/api/v1/client/config")
  @api.post("/api/v1/client/dynamic_config")
  async def client_config():
    return {}  # the SDK's defaults

  @api.get("/api/v1/get_server_capabilities")
  async def get_server_capabilities():
    return {"supported_models": []}

  @api.post("/api/v1/create_session")
  async def create_session(req: CreateSessionRequest | None = None):
    session_id = f"sess-{uuid.uuid4().hex[:12]}"
    req = req or CreateSessionRequest()
    await bus.session(session_id, dict(t.partition("=")[::2] for t in req.tags) | (req.user_metadata or {}))
    return {"type": "create_session", "session_id": session_id}

  @api.post("/api/v1/session_heartbeat")
  async def session_heartbeat(req: SessionRequest):
    await bus.session(req.session_id)
    return {"type": "session_heartbeat"}

  @api.post("/api/v1/create_model")
  async def create_model(req: CreateModelRequest):
    session = await bus.session_info(req.session_id) or {}
    metadata, lora = session.get("metadata", {}) | (req.user_metadata or {}), req.lora_config or {}
    rank = 0 if metadata.get("openrl.fine_tuning_type") == "full" else int(lora.get("rank", 16))
    exclusive = str(metadata.get("openrl.exclusive")).lower() == "true"
    run = str(uuid.uuid4())
    pool = await pool_for(bus, run, req.base_model, rank, metadata)
    await bus.create(Run(run, pool, req.base_model, rank, req.session_id, lora=lora, exclusive=exclusive))
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
    train = await bus.run(req.model_id)
    if not train:
      raise HTTPException(404, f"no model {req.model_id}")
    seq = req.sampling_session_seq_id if req.sampling_session_seq_id is not None else req.seq_id
    name = req.path or f"step-{seq if seq is not None else time.time_ns()}"
    uri = checkpoint_uri(req.model_id, "sampler_weights", name)
    if req.path is None or req.sampling_session_seq_id is not None:
      sid = await open_sampler(train["base"], uri, train.get("session", ""))
      return await submit(
        req.model_id, Op.SAVE_WEIGHTS_FOR_SAMPLER, {"uri": uri, "sampling_session_id": sid}, req.seq_id
      )
    if train["base"]:
      excl = req.model_id in await bus.exclusive()
      dec = schedule.choose_sampler_pool(train["base"], train["rank"] > 0, req.model_id, excl)
      await bus.warm_sampler(dec.choice, train["base"], exclusive=excl)
    return await submit(req.model_id, Op.SAVE_WEIGHTS_FOR_SAMPLER, {"uri": uri}, req.seq_id)

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
      if req.request_id.startswith("samp-"):
        return wire.sample_json(reply.body)
      return wire.output_json(reply.body)  # an older client, which reads JSON
    return Response(reply.body, media_type=reply.content_type)

  # ***** sampling *****

  @api.post("/api/v1/create_sampling_session")
  async def create_sampling_session(req: Request):
    data = await req.json()
    sid = await open_sampler(data.get("base_model"), data.get("model_path"), data.get("session_id", ""))
    return {"type": "create_sampling_session", "sampling_session_id": sid}

  @api.post("/api/v1/join_sampling_session")
  async def join_sampling_session(req: Request):
    data = await req.json()
    sid = data.get("sampling_session_id")
    if not await bus.run(sid):
      raise HTTPException(404, f"no session {sid}")
    return {"type": "join_sampling_session", "client_counter": await bus.join(sid)}

  @api.get("/api/v1/samplers/{sampler_id}")
  async def get_sampler(sampler_id: str):
    run = await bus.run(sampler_id)
    if not run:
      raise HTTPException(404, f"no sampler {sampler_id}")
    return {"sampler_id": sampler_id, "base_model": run["base"], "model_path": run.get("model_path")}

  @api.post("/api/v1/asample")
  async def asample(req: Request):
    data = await req.json()
    prompt = data.get("prompt", {})
    chunks = prompt.get("chunks", [])
    if not all(c.get("type") == "encoded_text" for c in chunks):
      raise ValueError("only encoded_text chunks are supported")
    tokens = [int(t) for c in chunks for t in c.get("tokens", [])]
    if not tokens:
      raise ValueError("prompt is empty")

    if data.get("target_prompt_logprobs") is not None or data.get("prompt_alt_tokens_k", 0) > 0:
      raise ValueError("target_prompt_logprobs and prompt_alt_tokens are not supported")

    last_n = data.get("prompt_logprobs_last_n")
    if last_n is not None:
      if not data.get("prompt_logprobs"):
        raise ValueError("prompt_logprobs must be true if prompt_logprobs_last_n is set")
      if not (1 <= last_n <= len(tokens) - 1):
        raise ValueError("prompt_logprobs_last_n must be between 1 and len(tokens) - 1")

    sid = data.get("sampling_session_id") or await open_sampler(data.get("base_model"), data.get("model_path"))

    num_samples = int(data.get("num_samples", 1))
    params = data.get("sampling_params", {})
    job = {
      "prompt": tokens,
      "num_samples": num_samples,
      "max_tokens": params.get("max_tokens") or 128,
      "temperature": float(params.get("temperature", 1.0)),
      "top_k": int(params.get("top_k", -1)),
      "top_p": float(params.get("top_p", 1.0)),
      "seed": params.get("seed"),
      "stop": params.get("stop"),
      "prompt_logprobs": bool(data.get("prompt_logprobs")),
      "prompt_logprobs_last_n": last_n,
      "topk_prompt": int(data.get("topk_prompt_logprobs", 0)),
      "topk_sample": int(data.get("topk_sample_logprobs", 0)),
    }

    res = await submit(sid, Op.SAMPLE, job, data.get("seq_id"))
    return {
      "request_id": res["request_id"],
      "sample_sequence_ids": [f"{res['request_id']}:{i}" for i in range(num_samples)],
    }

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
