"""The Tinker HTTP API and the controller's typed `/internal/{method}` node RPC."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import inspect
import json
import sys
import time
import typing
import uuid
from typing import Any

import msgspec
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from tinyrl.api import wire
from tinyrl.control import estimator
from tinyrl.control.ledger import COUNTERS, TICK_MS, Ledger
from tinyrl.control.samples import BackpressureError, PromiseGoneError
from tinyrl.helpers import getenv
from tinyrl.types import FencedError, Op, Role, Run, parse_uri

WAIT = getenv("WAIT", 30.0)  # how long retrieve_future waits for a reply; the SDK gives up after 45s
CKPT_TTL = getenv("CKPT_TTL", 0)  # default TTL for unowned saves (>= 3600s; 0 keeps until deleted)
ERRORS = {ValueError: 400, FencedError: 409, PromiseGoneError: 410, BackpressureError: 429}


# ***** request models *****


class SessionRequest(BaseModel):
  session_id: str


class CreateSessionRequest(BaseModel):
  tags: list[str] = []
  user_metadata: dict[str, Any] | None = None


class CreateModelRequest(BaseModel):
  base_model: str
  session_id: str | None = None
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


class SamplingSessionRequest(BaseModel):
  sampling_session_id: str | None = None
  base_model: str | None = None
  model_path: str | None = None
  session_id: str | None = None


def ttl(seconds: int | None) -> int | None:
  if seconds is not None and not 3600 <= seconds <= 10 * 365 * 24 * 3600:
    raise HTTPException(400, "ttl_seconds must be between 3600 and 10 years")
  return seconds


def iso(t: float | None) -> str | None:
  return datetime.datetime.fromtimestamp(t, datetime.UTC).isoformat() if t else None


def public(c: dict) -> dict:
  keep = {k: c[k] for k in ("checkpoint_id", "checkpoint_type", "tinker_path", "size_bytes")}
  return keep | {"time": iso(c["time"]), "expires_at": iso(c["expires_at"])}


def checkpoint_uri(run: str, kind: str, name: str) -> str:
  uri = name if name.startswith("tinker://") else f"tinker://{run}/{kind}/{name}"
  owner, cid = parse_uri(uri)
  if not cid.partition("/")[2]:
    raise ValueError(f"bad checkpoint name {name}")
  return f"tinker://{owner}/{cid}"


def app(ledger: Ledger | None = None, *, control: bool = False) -> FastAPI:
  ledger = ledger if ledger is not None else Ledger()

  @contextlib.asynccontextmanager
  async def lifespan(_: FastAPI):
    async def maintain_loop():
      while True:
        try:
          ledger.maintain()
        except Exception as exc:
          print(f"ledger maintain failed: {exc!r}", file=sys.stderr, flush=True)
        await asyncio.sleep(TICK_MS / 1000)

    maint = asyncio.create_task(maintain_loop())
    try:
      yield
    finally:
      maint.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await maint
      await ledger.aclose()

  api = FastAPI(title="tiny-open-rl", lifespan=lifespan if control else None)
  rpc = {
    name: (fn := getattr(ledger, name), inspect.signature(fn), typing.get_type_hints(fn))
    for name in ("heartbeat", "take", "finish", "detach")
  }
  for exc, code in ERRORS.items():
    api.add_exception_handler(exc, lambda _, err, c=code: JSONResponse({"error": f"{type(err).__name__}: {err}"}, c))

  @api.post("/internal/{method}")
  async def internal(method: str, request: Request) -> Response:
    if (entry := rpc.get(method)) is None:
      raise HTTPException(404)
    fn, sig, hints = entry
    args, kw = msgspec.msgpack.decode(await request.body())
    bound = sig.bind(*args, **kw)
    for k, v in bound.arguments.items():
      if k in hints:
        bound.arguments[k] = msgspec.convert(v, hints[k])
    res = fn(*bound.args, **bound.kwargs)
    out = await res if asyncio.iscoroutine(res) else res
    return Response(msgspec.msgpack.encode(out), media_type="application/msgpack")

  def require_run(run_id: str | None) -> dict:
    if not run_id or (record := ledger.get_run(run_id)) is None:
      raise HTTPException(404, f"no model {run_id}")
    return record

  def submit(
    run: str, op: Op, body: bytes | dict, seq_id: int | None = None, cost: int = 1, uri: str | None = None
  ) -> dict[str, str]:
    raw = json.dumps(body).encode() if isinstance(body, dict) else body
    if (request_id := ledger.submit_command(run, op, raw, seq_id, cost, uri)) is None:
      raise HTTPException(404, f"no model {run}")
    return {"request_id": request_id, "model_id": run}

  async def make_sampler(base_model: str | None, model_path: str | None, session_id: str = "") -> Run:
    mp = model_path or None
    train_run = parse_uri(mp)[0] if mp else None
    train = (ledger.get_run(train_run) or {}) if train_run else {}
    base, lora = base_model, True
    if train.get("base"):
      base, lora = train["base"], train["rank"] > 0
    elif mp:
      meta = (ledger.get_checkpoint(mp) or {}).get("meta", {})
      base, lora = meta.get("base") or base_model, bool(meta.get("lora"))
    if not base:
      raise ValueError("could not resolve base_model" if mp else "base_model required when model_path is None")
    smeta = (ledger.get_session(session_id) or {}).get("metadata", {}) if session_id else {}
    acct, excl = str(train.get("account") or smeta.get("openrl.account") or ""), bool(train.get("exclusive"))
    gpus = await asyncio.to_thread(estimator.gpus, base, "vllm")
    sid = f"samp-{uuid.uuid4().hex[:12]}"
    return Run(sid, base, int(lora), session_id, mp, {}, excl, Role.SAMPLER, train_run, gpus, acct)

  async def open_sampler(base_model: str | None, model_path: str | None, session_id: str = "") -> str:
    ledger.create_run(run := await make_sampler(base_model, model_path, session_id))
    return run.id

  # ***** service & sessions *****

  @api.get("/api/v1/scheduling")
  async def scheduling():
    nodes, state = ledger.live_nodes(details=True), dataclasses.asdict(ledger.state())
    for proc, placed in state["procs"].items():
      beat = nodes.get(placed["node"], {})
      placed |= {dst: beat[src][proc] for src, dst in (("pids", "pid"), ("on", "gpus")) if proc in beat.get(src, {})}
    return {"nodes": {n: v["gpus"] for n, v in nodes.items()}, "state": state}

  api.get("/api/v1/healthz")(lambda: {"status": "ok"})
  api.post("/api/v1/telemetry")(lambda: {"status": "accepted"})
  api.post("/api/v1/client/config")(lambda: {"parallel_fwdbwd_chunks": False})
  api.post("/api/v1/client/dynamic_config")(lambda: {})
  api.get("/api/v1/get_server_capabilities")(lambda: {"supported_models": []})

  @api.post("/api/v1/sessions/{session_id}/finish")
  async def finish_session(session_id: str):
    ledger.heartbeat_session(session_id, finish=True)
    return {"status": "ok"}

  @api.post("/api/v1/create_session")
  async def create_session(req: CreateSessionRequest | None = None):
    sid = f"sess-{uuid.uuid4().hex[:12]}"
    meta = dict(t.partition("=")[::2] for t in (req.tags if req else ())) | ((req.user_metadata if req else None) or {})
    ledger.heartbeat_session(sid, meta)
    return {"type": "create_session", "session_id": sid}

  @api.post("/api/v1/session_heartbeat")
  async def session_heartbeat(req: SessionRequest):
    ledger.heartbeat_session(req.session_id)
    return {"type": "session_heartbeat"}

  @api.post("/api/v1/create_model")
  async def create_model(req: CreateModelRequest):
    sess = req.session_id or ""
    meta = (ledger.get_session(sess) or {}).get("metadata", {}) | (req.user_metadata or {})
    lora = req.lora_config or {}
    rank = 0 if meta.get("openrl.fine_tuning_type") == "full" else int(lora.get("rank", 16))
    excl, acct = str(meta.get("openrl.exclusive")).lower() == "true", str(meta.get("openrl.account") or "")
    run = f"{sess}-{req.model_seq_id}" if sess and req.model_seq_id is not None else str(uuid.uuid4())
    raw = meta.get("openrl.gpus")
    gpus = (1 if rank else await asyncio.to_thread(estimator.gpus, req.base_model, "full")) if raw is None else int(raw)
    if gpus < 1:
      raise ValueError(f"openrl.gpus must be at least 1, got {gpus}")
    if not ledger.get_run(run):
      ledger.create_run(Run(run, req.base_model, rank, sess, None, lora, excl, gpus=gpus, account=acct))
    return submit(run, Op.CREATE_MODEL, {"base_model": req.base_model, "rank": rank, "lora": lora}, req.model_seq_id)

  @api.post("/api/v1/get_info")
  async def get_info(req: ModelRequest):
    record = require_run(req.model_id)
    base, rank = record["base"], record["rank"] or 16
    res = {"type": "get_info", "model_id": req.model_id, "model_name": base, "is_lora": True, "lora_rank": rank}
    return res | {"model_data": {"model_name": base, "tokenizer_id": base}}

  # ***** training *****

  @api.post("/api/v1/forward_backward")
  async def forward_backward(request: Request):
    parsed = wire.parse(body := await request.body())
    return submit(parsed.model_id, Op.FORWARD_BACKWARD, body, parsed.seq_id or None, wire.cost(parsed))

  @api.post("/api/v1/optim_step")
  async def optim_step(req: OptimStepRequest):
    return submit(req.model_id, Op.OPTIM_STEP, {"adam_params": req.adam_params}, req.seq_id)

  @api.post("/api/v1/save_weights")
  async def save_weights(req: SaveRequest):
    return submit(req.model_id, Op.SAVE_WEIGHTS, req.checkpoint("weights"), req.seq_id)

  @api.post("/api/v1/save_weights_for_sampler")
  async def save_weights_for_sampler(req: SaveRequest):
    if request_id := ledger.seen_request(req.model_id, req.seq_id):
      return {"request_id": request_id, "model_id": req.model_id}
    train = require_run(req.model_id)
    owned = req.path is None or req.sampling_session_seq_id is not None
    args = req.checkpoint("sampler_weights", owned=owned)
    if owned:
      samp = await make_sampler(train["base"], args["uri"], train.get("session") or "")
      if request_id := ledger.seen_request(req.model_id, req.seq_id):
        return {"request_id": request_id, "model_id": req.model_id}
      ledger.create_run(samp)
      args |= {"sampling_session_id": samp.id, "owned": True}
    return submit(req.model_id, Op.SAVE_WEIGHTS_FOR_SAMPLER, args, req.seq_id)

  @api.post("/api/v1/load_weights")
  async def load_weights(req: LoadRequest):
    uri = checkpoint_uri(req.model_id, "weights", req.path)
    return submit(req.model_id, Op.LOAD_WEIGHTS, {"uri": uri, "optimizer": req.optimizer}, req.seq_id, uri=uri)

  @api.post("/api/v1/weights_info")
  async def weights_info(req: WeightsInfoRequest):
    meta = (ledger.get_checkpoint(req.tinker_path) or {}).get("meta", {})
    if "base" not in meta:
      raise HTTPException(404, f"no checkpoint {req.tinker_path} that names its base model")
    rank = int(meta.get("rank") or 0)
    return {"base_model": meta["base"], "is_lora": True, "lora_rank": rank or 16} | (
      meta.get("flags", {}) if rank else {}
    )

  @api.post("/api/v1/delete_model")
  async def delete_model(req: ModelRequest):
    submit(req.model_id, Op.DELETE_MODEL, {}, req.seq_id)
    return {"status": "ok"}

  @api.post("/api/v1/retrieve_future")
  async def retrieve_future(req: FutureRequest):
    if (reply := await ledger.wait_for_reply(req.request_id, WAIT)) is None:
      return {"type": "try_again", "request_id": req.request_id, "queue_state": "active"}
    return Response(reply.body, media_type=reply.content_type)

  # ***** sampling *****

  @api.post("/api/v1/create_sampling_session")
  async def create_sampling_session(req: SamplingSessionRequest):
    sid = req.sampling_session_id
    if not (sid and ledger.get_run(sid)):
      sid = await open_sampler(req.base_model, req.model_path, req.session_id or "")
    return {"type": "create_sampling_session", "sampling_session_id": sid}

  @api.post("/api/v1/join_sampling_session")
  async def join_sampling_session(req: SamplingSessionRequest):
    require_run(req.sampling_session_id)
    return {"type": "join_sampling_session", "client_counter": ledger.join_sampling_session(req.sampling_session_id)}

  @api.get("/api/v1/samplers/{sampler_id}")
  async def get_sampler(sampler_id: str):
    run = require_run(sampler_id)
    return {"sampler_id": sampler_id, "base_model": run["base"], "model_path": run.get("model_path")}

  @api.post("/api/v1/asample")
  async def asample(request: Request):
    data = await request.json()
    chunks = (data.get("prompt") or {}).get("chunks", [])
    if any(c.get("type", "encoded_text") != "encoded_text" for c in chunks):
      raise ValueError("only encoded_text prompt chunks are supported")
    if not (tokens := [int(t) for c in chunks for t in c.get("tokens", [])]):
      raise ValueError("empty prompt is not supported")
    if data.get("target_prompt_logprobs") is not None or (data.get("prompt_alt_tokens_k") or 0) > 0:
      raise ValueError("target_prompt_logprobs and prompt_alt_tokens are not supported")
    if (last_n := data.get("prompt_logprobs_last_n")) is not None:
      if not data.get("prompt_logprobs") or not 1 <= last_n <= len(tokens) - 1:
        raise ValueError("invalid prompt_logprobs_last_n")
    sid = data.get("sampling_session_id") or await open_sampler(data.get("base_model"), data.get("model_path"))
    n, p = int(data.get("num_samples", 1)), data.get("sampling_params") or {}
    job = {
      "prompt": tokens,
      "num_samples": n,
      "max_tokens": (mt := p.get("max_tokens") or 128),
      "seed": p.get("seed"),
      "stop": p.get("stop"),
      "temperature": float(p.get("temperature", 1.0)),
      "top_k": int(p.get("top_k", -1)),
      "top_p": float(p.get("top_p", 1.0)),
      "prompt_logprobs": bool(data.get("prompt_logprobs")),
      "prompt_logprobs_last_n": last_n,
      "topk_prompt": int(data.get("topk_prompt_logprobs", 0)),
      "topk_sample": int(data.get("topk_sample_logprobs", 0)),
      "record_stability_info": bool(data.get("record_stability_info")),
    }
    rid = submit(sid, Op.SAMPLE, job, data.get("seq_id"), len(tokens) + n * mt)["request_id"]
    return {"request_id": rid, "sample_sequence_ids": [f"{rid}:{i}" for i in range(n)]}

  # ***** checkpoints & inspection *****

  @api.get("/api/v1/training_runs/{run}/checkpoints")
  async def list_checkpoints(run: str):
    return {"checkpoints": [public(c) for c in ledger.list_checkpoints(run)]}

  @api.get("/api/v1/checkpoints")
  async def list_user_checkpoints(limit: int = 100, offset: int = 0):
    vis = ledger.list_checkpoints()
    page = [public(c) for c in (vis[offset : offset + limit] if limit > 0 else [])]
    return {"checkpoints": page, "cursor": {"offset": offset, "limit": limit, "total_count": len(vis)}}

  @api.delete("/api/v1/training_runs/{run}/checkpoints/{cid:path}")
  async def delete_checkpoint(run: str, cid: str):
    if not ledger.delete_checkpoint(f"tinker://{run}/{cid}"):
      raise HTTPException(404, f"no checkpoint {cid} in {run}")

  @api.put("/api/v1/training_runs/{run}/checkpoints/{cid:path}/ttl")
  async def set_checkpoint_ttl(run: str, cid: str, req: TTLRequest):
    if not ledger.set_checkpoint_ttl(f"tinker://{run}/{cid}", ttl(req.ttl_seconds)):
      raise HTTPException(404, f"no checkpoint {cid} in {run}")

  @api.get("/api/v1/runs/{run_id}")
  async def run_info(run_id: str):
    run, ps = require_run(run_id), ledger.state().procs_of(run_id)
    return {**run, "procs": ps, **({"worker": ps[0]} if ps else {}), **ledger.queued_and_in_flight(run_id)}

  @api.get("/api/v1/sessions/{session_id}")
  async def session_info(session_id: str):
    if (s := ledger.get_session(session_id)) is None:
      raise HTTPException(404, f"no session {session_id}")
    return s

  @api.get("/metrics")
  async def metrics():
    nodes, state, lines = ledger.live_nodes(details=True), ledger.state(), []
    for run in state.jobs:
      if ps := state.procs_of(run):
        info = ledger.get_run(run)
        for k in COUNTERS:
          lbl, v = (f"{k[:-3]}_seconds", info[k] / 1000) if k.endswith("_ms") else (k, info[k])
          lines.append(f'tinyrl_{lbl}_total{{worker="{ps[0]}",run="{run}"}} {v}')
    lines += [f'tinyrl_account_service_total{{account="{a}"}} {float(v)}' for a, v in sorted(ledger.service.items())]
    lines += [f'tinyrl_node_loop_lag_seconds{{node="{n}"}} {b.get("lag", 0.0)}' for n, b in nodes.items()]
    return PlainTextResponse("\n".join(lines) + "\n")

  return api


if __name__ == "__main__":
  uvicorn.run(app(control=True), host="0.0.0.0", port=getenv("PORT", 8000))
