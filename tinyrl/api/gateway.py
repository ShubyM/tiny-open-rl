"""The Tinker HTTP API.

Every training call becomes a command on its run's stream, and retrieve_future waits for the trainer's reply.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import json
import time
import uuid
from typing import Any

import msgspec
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from tinyrl.api import wire
from tinyrl.bus import BackpressureError, Bus, PromiseGoneError
from tinyrl.control import estimator, scheduler
from tinyrl.helpers import getenv
from tinyrl.storage.store import Store
from tinyrl.types import NodeReport, Op, Role, Run, parse_uri

WAIT = getenv("WAIT", 30.0)  # how long retrieve_future waits for a reply. The SDK gives up after 45s.
CKPT_TTL = getenv("CKPT_TTL", 0)  # the TTL of a checkpoint saved without one, at least 3600. 0 keeps it until deleted


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
  model_seq_id: int | None = None


class ModelRequest(BaseModel):
  model_id: str
  seq_id: int | None = None


class OptimStepRequest(ModelRequest):
  adam_params: dict[str, Any] = {}


class SaveRequest(ModelRequest):
  path: str | None = None
  sampling_session_seq_id: int | None = None
  ttl_seconds: int | None = None

  def checkpoint(self, kind: str, *, owned: bool = False) -> dict:
    seq = self.sampling_session_seq_id if self.sampling_session_seq_id is not None else self.seq_id
    name = self.path or f"step-{seq if seq is not None else time.time_ns()}"
    seconds = self.ttl_seconds if self.ttl_seconds is not None or owned else CKPT_TTL or None
    return {"uri": checkpoint_uri(self.model_id, kind, name), "ttl_seconds": ttl(seconds)}


class TTLRequest(BaseModel):
  ttl_seconds: int | None = None


class LoadRequest(ModelRequest):
  path: str
  optimizer: bool = False


class FutureRequest(BaseModel):
  request_id: str


class WeightsInfoRequest(BaseModel):
  tinker_path: str


def ttl(seconds: int | None) -> int | None:
  """A checkpoint's TTL, which the Tinker API bounds between an hour and ten years."""
  if seconds is not None and not 3600 <= seconds <= 10 * 365 * 24 * 3600:
    raise HTTPException(400, "ttl_seconds must be between 3600 and 10 years")
  return seconds


def iso(t: float | None) -> str | None:
  return datetime.datetime.fromtimestamp(t, datetime.UTC).isoformat() if t else None


def public(c: dict) -> dict:
  """A registry entry as the Tinker API's Checkpoint."""
  keep = {k: c[k] for k in ("checkpoint_id", "checkpoint_type", "tinker_path", "size_bytes")}
  return keep | {"time": iso(c["time"]), "expires_at": iso(c["expires_at"])}


def checkpoint_uri(run: str, kind: str, name: str) -> str:
  """The logical tinker://<run>/<kind>/<name> URI for a checkpoint."""
  if name.startswith("tinker://"):  # a checkpoint by URI, maybe another run's to resume from
    run, cid = parse_uri(name)
    kind, _, name = cid.partition("/")
  if not name or name.startswith("/") or ".." in f"{run}/{kind}/{name}".split("/"):
    raise ValueError(f"bad checkpoint name {name}")
  return f"tinker://{run}/{kind}/{name}"


def app(bus: Bus | None = None, store: Store | None = None, *, control: bool = False) -> FastAPI:
  bus = bus if bus is not None else Bus()
  store = store if store is not None else Store()

  @contextlib.asynccontextmanager
  async def lifespan(_: FastAPI):
    async def maintain_loop():
      while True:
        await bus.maintain()
        await asyncio.sleep(scheduler.TICK_MS / 1000)

    maint = asyncio.create_task(maintain_loop())
    try:
      yield
    finally:
      maint.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await maint
      await bus.aclose()

  api = FastAPI(title="tiny-open-rl", lifespan=lifespan if control else None)

  @api.post("/internal/exchange")
  async def exchange(request: Request, wait_ms: int = 0) -> Response:
    report = msgspec.msgpack.decode(await request.body(), type=NodeReport)
    return Response(msgspec.msgpack.encode(await bus.exchange(report, wait_ms)), media_type="application/msgpack")

  def bad_request(_: Request, e: Exception) -> JSONResponse:
    return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)

  api.add_exception_handler(ValueError, bad_request)
  api.add_exception_handler(KeyError, bad_request)
  api.add_exception_handler(PromiseGoneError, lambda _, e: JSONResponse({"error": str(e)}, status_code=410))
  api.add_exception_handler(BackpressureError, lambda _, e: JSONResponse({"error": str(e)}, status_code=429))

  async def require_run(run_id: str, kind: str = "model") -> dict:
    if (record := await bus.get_run(run_id)) is None:
      raise HTTPException(404, f"no {kind} {run_id}")
    return record

  async def submit(run: str, op: Op, body: bytes | dict, seq_id: int | None = None, cost: int = 1) -> dict[str, str]:
    if isinstance(body, dict):
      body = json.dumps(body).encode()
    request_id = await bus.submit_command(run, op, body, seq_id, cost)
    if request_id is None:
      raise HTTPException(404, f"no model {run}")
    return {"request_id": request_id, "model_id": run}

  async def open_sampler(base_model: str | None, model_path: str | None, session_id: str = "") -> str:
    model_path = model_path or None
    train_run = parse_uri(model_path)[0] if model_path else None
    train = (await bus.get_run(train_run) or {}) if train_run else {}
    base, lora = base_model, True
    if train.get("base"):
      base, lora = train["base"], train["rank"] > 0
    elif model_path:
      meta = await asyncio.to_thread(store.read_meta, await bus.checkpoint_version(model_path))
      base, lora = meta.get("base") or base_model, bool(meta.get("lora"))
    if not base:
      raise ValueError("could not resolve base_model" if model_path else "base_model required when model_path is None")

    exclusive = bool(train.get("exclusive"))
    sess_meta = ((await bus.get_session(session_id)) or {}).get("metadata", {}) if session_id else {}
    account = str(train.get("account") or sess_meta.get("openrl.account") or "")
    gpus = await asyncio.to_thread(estimator.gpus, base, "vllm")
    sid = f"samp-{uuid.uuid4().hex[:12]}"
    await bus.create_run(
      Run(sid, base, int(lora), session_id, model_path, {}, exclusive, Role.SAMPLER, train_run, gpus, account)
    )
    return sid

  # ***** service *****

  @api.get("/api/v1/scheduling")
  async def scheduling():
    """The cluster's plan, account lags, and each process's actual GPUs and PID from its node's last heartbeat."""
    raw = await bus.read_state()
    state = dataclasses.asdict(scheduler.from_state(raw))
    for proc, placed in state["procs"].items():
      heartbeat = raw["nodes"].get(placed["node"], {})
      for src, field in (("pids", "pid"), ("on", "gpus")):
        if proc in heartbeat.get(src, {}):
          placed[field] = heartbeat[src][proc]
    return {"nodes": {n: v["gpus"] for n, v in raw["nodes"].items()}, "state": state}

  @api.get("/api/v1/healthz")
  async def healthz():
    return {"status": "ok"}

  @api.post("/api/v1/sessions/{session_id}/finish")
  async def finish_session(session_id: str):
    await bus.finish_session(session_id)  # the SDK's close(), and the end of a `with ServiceClient()` block
    return {"status": "ok"}

  @api.post("/api/v1/telemetry")
  async def telemetry():
    return {"status": "accepted"}

  @api.post("/api/v1/client/config")
  async def client_config():
    return {"parallel_fwdbwd_chunks": False}

  @api.post("/api/v1/client/dynamic_config")
  async def client_dynamic_config():
    return {}  # the SDK's defaults

  @api.get("/api/v1/get_server_capabilities")
  async def get_server_capabilities():
    return {"supported_models": []}

  @api.post("/api/v1/create_session")
  async def create_session(req: CreateSessionRequest | None = None):
    session_id = f"sess-{uuid.uuid4().hex[:12]}"
    req = req or CreateSessionRequest()
    await bus.heartbeat_session(session_id, dict(t.partition("=")[::2] for t in req.tags) | (req.user_metadata or {}))
    return {"type": "create_session", "session_id": session_id}

  @api.post("/api/v1/session_heartbeat")
  async def session_heartbeat(req: SessionRequest):
    await bus.heartbeat_session(req.session_id)
    return {"type": "session_heartbeat"}

  @api.post("/api/v1/create_model")
  async def create_model(req: CreateModelRequest):
    session = await bus.get_session(req.session_id) or {}
    metadata, lora = session.get("metadata", {}) | (req.user_metadata or {}), req.lora_config or {}
    rank = 0 if metadata.get("openrl.fine_tuning_type") == "full" else int(lora.get("rank", 16))
    exclusive = str(metadata.get("openrl.exclusive")).lower() == "true"

    seq = req.model_seq_id
    run = f"{req.session_id}-{seq}" if req.session_id and seq is not None else str(uuid.uuid4())

    base, account = req.base_model, str(metadata.get("openrl.account") or "")
    raw_gpus = metadata.get("openrl.gpus")
    gpus = (1 if rank else await asyncio.to_thread(estimator.gpus, base, "full")) if raw_gpus is None else int(raw_gpus)
    if gpus < 1:
      raise ValueError(f"openrl.gpus must be at least 1, got {gpus}")
    if not await bus.get_run(run):
      await bus.create_run(Run(run, base, rank, req.session_id, None, lora, exclusive, gpus=gpus, account=account))
    return await submit(run, Op.CREATE_MODEL, {"base_model": base, "rank": rank, "lora": lora}, seq)

  @api.post("/api/v1/get_info")
  async def get_info(req: ModelRequest):
    run = await require_run(req.model_id)

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

  async def submit_fb(data: dict, forward_only: bool = False) -> dict[str, str]:
    req = wire.from_json(data, forward_only=forward_only)
    return await submit(
      data["model_id"], Op.FORWARD_BACKWARD, req.SerializeToString(), data.get("seq_id"), wire.cost(req)
    )

  @api.post("/api/v1/forward_backward")
  async def forward_backward(request: Request):
    body = await request.body()
    if request.headers.get("content-type", "").startswith(wire.PROTO):
      parsed = wire.parse(body)
      return await submit(parsed.model_id, Op.FORWARD_BACKWARD, body, parsed.seq_id or None, wire.cost(parsed))
    return await submit_fb(json.loads(body))

  @api.post("/api/v1/forward")  # SDKs before 0.25
  async def forward(request: Request):
    return await submit_fb(await request.json(), forward_only=True)

  @api.post("/api/v1/optim_step")
  async def optim_step(req: OptimStepRequest):
    return await submit(req.model_id, Op.OPTIM_STEP, {"adam_params": req.adam_params}, req.seq_id)

  @api.post("/api/v1/save_weights")  # save_state, so the optimizer is saved too
  async def save_weights(req: SaveRequest):
    return await submit(req.model_id, Op.SAVE_WEIGHTS, req.checkpoint("weights"), req.seq_id)

  @api.post("/api/v1/save_weights_for_sampler")
  async def save_weights_for_sampler(req: SaveRequest):
    if request_id := await bus.seen_request(req.model_id, req.seq_id):
      return {"request_id": request_id, "model_id": req.model_id}

    train = await require_run(req.model_id)
    # a sampling session's weights, which its run owns
    owned = req.path is None or req.sampling_session_seq_id is not None
    args = req.checkpoint("sampler_weights", owned=owned)
    if owned:
      sid = await open_sampler(train["base"], args["uri"], train.get("session", ""))
      args.update(sampling_session_id=sid, owned=True)
    return await submit(req.model_id, Op.SAVE_WEIGHTS_FOR_SAMPLER, args, req.seq_id)

  @api.post("/api/v1/load_weights")
  async def load_weights(req: LoadRequest):
    uri = checkpoint_uri(req.model_id, "weights", req.path)
    return await submit(req.model_id, Op.LOAD_WEIGHTS, {"uri": uri, "optimizer": req.optimizer}, req.seq_id)

  @api.post("/api/v1/weights_info")
  async def weights_info(req: WeightsInfoRequest):
    """What the SDK asks before it resumes from a checkpoint: how to create a run like the one that saved it. The checkpoint
    says so itself, as its run is deleted soon after its client stops."""
    try:
      meta = await asyncio.to_thread(store.read_meta, await bus.checkpoint_version(req.tinker_path))
    except (OSError, IndexError, ValueError):
      meta = {}
    if "base" not in meta:
      raise HTTPException(404, f"no checkpoint {req.tinker_path} that names its base model")
    made = meta["targets"]
    trained = {"train_unembed": "lm_head" in made, "train_mlp": "gate_proj" in made, "train_attn": "q_proj" in made}
    # the SDK reads LoRA shaped metadata, even for a full fine-tune
    return {"base_model": meta["base"], "is_lora": True, "lora_rank": meta["rank"] or 16} | (
      trained if meta["rank"] else {}
    )

  @api.post("/api/v1/delete_model")
  async def delete_model(req: ModelRequest):
    await submit(req.model_id, Op.DELETE_MODEL, {}, req.seq_id)  # queued behind the run's work, which still finishes
    return {"status": "ok"}

  @api.post("/api/v1/retrieve_future")
  async def retrieve_future(req: FutureRequest, accept: str = Header("")):
    reply = await bus.wait_for_reply(req.request_id, WAIT)
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
    await require_run(sid, "session")
    return {"type": "join_sampling_session", "client_counter": await bus.join_sampling_session(sid)}

  @api.get("/api/v1/samplers/{sampler_id}")
  async def get_sampler(sampler_id: str):
    run = await require_run(sampler_id, "sampler")
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

    res = await submit(sid, Op.SAMPLE, job, data.get("seq_id"), len(tokens) + num_samples * job["max_tokens"])
    return {
      "request_id": res["request_id"],
      "sample_sequence_ids": [f"{res['request_id']}:{i}" for i in range(num_samples)],
    }

  # ***** checkpoints *****

  @api.get("/api/v1/training_runs/{run}/checkpoints")
  async def list_checkpoints(run: str):
    return {"checkpoints": [public(c) for c in await bus.checkpoints_of_run(run)]}

  @api.get("/api/v1/checkpoints")
  async def list_user_checkpoints(limit: int = 100, offset: int = 0):
    found, total = await bus.page_of_checkpoints(limit, offset)
    return {
      "checkpoints": [public(c) for c in found],
      "cursor": {"offset": offset, "limit": limit, "total_count": total},
    }

  @api.delete("/api/v1/training_runs/{run}/checkpoints/{checkpoint_id:path}")
  async def delete_checkpoint(run: str, checkpoint_id: str):
    if not await bus.unregister_checkpoints(f"tinker://{run}/{checkpoint_id}"):  # its files go with the next sweep
      raise HTTPException(404, f"no checkpoint {checkpoint_id} in {run}")

  @api.put("/api/v1/training_runs/{run}/checkpoints/{checkpoint_id:path}/ttl")
  async def set_checkpoint_ttl(run: str, checkpoint_id: str, req: TTLRequest):
    if not await bus.set_checkpoint_ttl(f"tinker://{run}/{checkpoint_id}", ttl(req.ttl_seconds)):
      raise HTTPException(404, f"no checkpoint {checkpoint_id} in {run}")

  # ***** inspection *****

  @api.get("/api/v1/runs/{run_id}")
  async def run_info(run_id: str):
    run = await require_run(run_id)
    procs = scheduler.from_state(await bus.read_state(backlogs=False)).procs_of(run_id)
    extra = {"procs": procs, **({"worker": procs[0]} if procs else {})}
    return {**run, **extra, **await bus.queued_and_in_flight(run_id)}

  @api.get("/api/v1/sessions/{session_id}")
  async def session_info(session_id: str):
    session = await bus.get_session(session_id)
    if session is None:
      raise HTTPException(404, f"no session {session_id}")
    return session

  @api.get("/metrics")
  async def metrics():
    """Every run's counters, account service and node loop lag for Prometheus."""
    raw = await bus.read_state(backlogs=False, counters=True)
    state = scheduler.from_state(raw)
    lines = []
    for run, info in raw["runs"].items():
      if ps := state.procs_of(run):
        for name, n in info["counters"].items():
          name, n = (f"{name[:-3]}_seconds", n / 1000) if name.endswith("_ms") else (name, n)
          lines.append(f'tinyrl_{name}_total{{worker="{ps[0]}",run="{run}"}} {n}')
    for account, total in sorted(raw["service"].items()):
      lines.append(f'tinyrl_account_service_total{{account="{account}"}} {float(total)}')
    for node, beat in raw["nodes"].items():
      lines.append(f'tinyrl_node_loop_lag_seconds{{node="{node}"}} {beat.get("lag", 0.0)}')
    return PlainTextResponse("\n".join(lines) + "\n")

  return api


if __name__ == "__main__":
  uvicorn.run(app(control=True), host="0.0.0.0", port=getenv("PORT", 8000))
