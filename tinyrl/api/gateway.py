"""The Tinker HTTP API and the controller's `/internal/{method}` node RPC."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import json
import time
import uuid

import msgspec
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from tinyrl.api import wire
from tinyrl.control import estimator, scheduler
from tinyrl.control.accounting import Charge
from tinyrl.control.ledger import COUNTERS, Ledger
from tinyrl.control.samples import BackpressureError, PromiseGoneError
from tinyrl.helpers import getenv
from tinyrl.types import FencedError, Op, Reply, Role, Run, parse_uri

WAIT, CKPT_TTL = getenv("WAIT", 30.0), getenv("CKPT_TTL", 0)
ERRORS = {ValueError: 400, KeyError: 400, FencedError: 409, PromiseGoneError: 410, BackpressureError: 429}


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
  if name.startswith("tinker://"):
    run, cid = parse_uri(name)
    kind, _, name = cid.partition("/")
  if not name or name.startswith("/") or ".." in f"{run}/{kind}/{name}".split("/"):
    raise ValueError(f"bad checkpoint name {name}")
  return f"tinker://{run}/{kind}/{name}"


def save_spec(req: dict, kind: str, *, owned: bool = False) -> dict:
  seq = req["sampling_session_seq_id"] if req.get("sampling_session_seq_id") is not None else req.get("seq_id")
  name = req.get("path") or f"step-{seq if seq is not None else time.time_ns()}"
  seconds = req.get("ttl_seconds") if req.get("ttl_seconds") is not None or owned else CKPT_TTL or None
  return {"uri": checkpoint_uri(req["model_id"], kind, name), "ttl_seconds": ttl(seconds)}


def app(ledger: Ledger | None = None, *, control: bool = False) -> FastAPI:
  ledger = ledger if ledger is not None else Ledger()

  @contextlib.asynccontextmanager
  async def lifespan(_: FastAPI):
    async def loop():
      while True:
        ledger.maintain()
        await asyncio.sleep(scheduler.TICK_MS / 1000)

    maint = asyncio.create_task(loop())
    try:
      yield
    finally:
      maint.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await maint
      await ledger.aclose()

  api = FastAPI(title="tiny-open-rl", lifespan=lifespan if control else None)
  rpc = {"heartbeat": ledger.heartbeat, "take": ledger.take, "finish": ledger.finish, "detach": ledger.detach}
  for exc, code in ERRORS.items():
    api.add_exception_handler(exc, lambda _, e, c=code: JSONResponse({"error": f"{type(e).__name__}: {e}"}, c))

  @api.post("/internal/{method}")
  async def internal(method: str, request: Request) -> Response:
    if (fn := rpc.get(method)) is None:
      raise HTTPException(404)
    raw_args, kw = msgspec.msgpack.decode(await request.body())
    args = list(raw_args)
    if method == "finish":
      args[4] = msgspec.convert(args[4], Reply)
      if len(args) > 5 and args[5] is not None:
        args[5] = msgspec.convert(args[5], Charge)
    res = fn(*args, **kw)
    out = await res if asyncio.iscoroutine(res) else res
    return Response(msgspec.msgpack.encode(out), media_type="application/msgpack")

  def require_run(run_id: str) -> dict:
    if (record := ledger.get_run(run_id)) is None:
      raise HTTPException(404, f"no model {run_id}")
    return record

  def submit(run: str, op: Op, body: bytes | dict, seq_id: int | None = None, cost: int = 1) -> dict[str, str]:
    raw = json.dumps(body).encode() if isinstance(body, dict) else body
    if (request_id := ledger.submit_command(run, op, raw, seq_id, cost)) is None:
      raise HTTPException(404, f"no model {run}")
    return {"request_id": request_id, "model_id": run}

  async def open_sampler(base_model: str | None, model_path: str | None, session_id: str = "") -> str:
    model_path = model_path or None
    train_run = parse_uri(model_path)[0] if model_path else None
    train = (ledger.get_run(train_run) or {}) if train_run else {}
    base, lora = base_model, True
    if train.get("base"):
      base, lora = train["base"], train["rank"] > 0
    elif model_path:
      meta = (ledger.get_checkpoint(model_path) or {}).get("meta", {})
      base, lora = meta.get("base") or base_model, bool(meta.get("lora"))
    if not base:
      raise ValueError("could not resolve base_model" if model_path else "base_model required when model_path is None")
    smeta = (ledger.get_session(session_id) or {}).get("metadata", {}) if session_id else {}
    acct, excl = str(train.get("account") or smeta.get("openrl.account") or ""), bool(train.get("exclusive"))
    gpus, sid = await asyncio.to_thread(estimator.gpus, base, "vllm"), f"samp-{uuid.uuid4().hex[:12]}"
    ledger.create_run(Run(sid, base, int(lora), session_id, model_path, {}, excl, Role.SAMPLER, train_run, gpus, acct))
    return sid

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
    ledger.finish_session(session_id)
    return {"status": "ok"}

  @api.post("/api/v1/create_session")
  async def create_session(req: dict | None = Body(None)):
    sid, data = f"sess-{uuid.uuid4().hex[:12]}", req or {}
    meta = dict(t.partition("=")[::2] for t in data.get("tags") or ()) | (data.get("user_metadata") or {})
    ledger.heartbeat_session(sid, meta)
    return {"type": "create_session", "session_id": sid}

  @api.post("/api/v1/session_heartbeat")
  async def session_heartbeat(req: dict = Body(...)):
    ledger.heartbeat_session(req["session_id"])
    return {"type": "session_heartbeat"}

  @api.post("/api/v1/create_model")
  async def create_model(req: dict = Body(...)):
    base, sess, seq = req["base_model"], req.get("session_id", ""), req.get("model_seq_id")
    meta = (ledger.get_session(sess) or {}).get("metadata", {}) | (req.get("user_metadata") or {})
    lora = req.get("lora_config") or {}
    rank = 0 if meta.get("openrl.fine_tuning_type") == "full" else int(lora.get("rank", 16))
    excl, acct = str(meta.get("openrl.exclusive")).lower() == "true", str(meta.get("openrl.account") or "")
    run, raw = (f"{sess}-{seq}" if sess and seq is not None else str(uuid.uuid4())), meta.get("openrl.gpus")
    gpus = (1 if rank else await asyncio.to_thread(estimator.gpus, base, "full")) if raw is None else int(raw)
    if gpus < 1:
      raise ValueError(f"openrl.gpus must be at least 1, got {gpus}")
    if not ledger.get_run(run):
      ledger.create_run(Run(run, base, rank, sess, None, lora, excl, gpus=gpus, account=acct))
    return submit(run, Op.CREATE_MODEL, {"base_model": base, "rank": rank, "lora": lora}, seq)

  @api.post("/api/v1/get_info")
  async def get_info(req: dict = Body(...)):
    b, rank = (r := require_run(mid := req["model_id"]))["base"], r["rank"] or 16
    res = {"type": "get_info", "model_id": mid, "model_name": b, "is_lora": True, "lora_rank": rank}
    return res | {"model_data": {"model_name": b, "tokenizer_id": b}}

  # ***** training *****

  @api.post("/api/v1/forward_backward")
  async def forward_backward(request: Request):
    parsed = wire.parse(body := await request.body())
    return submit(parsed.model_id, Op.FORWARD_BACKWARD, body, parsed.seq_id or None, wire.cost(parsed))

  @api.post("/api/v1/optim_step")
  async def optim_step(req: dict = Body(...)):
    return submit(req["model_id"], Op.OPTIM_STEP, {"adam_params": req.get("adam_params") or {}}, req.get("seq_id"))

  @api.post("/api/v1/save_weights")
  async def save_weights(req: dict = Body(...)):
    return submit(req["model_id"], Op.SAVE_WEIGHTS, save_spec(req, "weights"), req.get("seq_id"))

  @api.post("/api/v1/save_weights_for_sampler")
  async def save_weights_for_sampler(req: dict = Body(...)):
    mid, seq = req["model_id"], req.get("seq_id")
    if request_id := ledger.seen_request(mid, seq):
      return {"request_id": request_id, "model_id": mid}
    train = require_run(mid)
    owned = req.get("path") is None or req.get("sampling_session_seq_id") is not None
    args = save_spec(req, "sampler_weights", owned=owned)
    if owned:
      sid = await open_sampler(train["base"], args["uri"], train.get("session", ""))
      args |= {"sampling_session_id": sid, "owned": True}
    return submit(mid, Op.SAVE_WEIGHTS_FOR_SAMPLER, args, seq)

  @api.post("/api/v1/load_weights")
  async def load_weights(req: dict = Body(...)):
    uri = checkpoint_uri(mid := req["model_id"], "weights", req["path"])
    return submit(mid, Op.LOAD_WEIGHTS, {"uri": uri, "optimizer": bool(req.get("optimizer"))}, req.get("seq_id"))

  @api.post("/api/v1/weights_info")
  async def weights_info(req: dict = Body(...)):
    meta = (ledger.get_checkpoint(path := req["tinker_path"]) or {}).get("meta", {})
    if "base" not in meta:
      raise HTTPException(404, f"no checkpoint {path} that names its base model")
    made, r = meta.get("targets", ()), meta["rank"]
    flags = {"train_unembed": "lm_head" in made, "train_mlp": "gate_proj" in made, "train_attn": "q_proj" in made}
    return {"base_model": meta["base"], "is_lora": True, "lora_rank": r or 16} | (flags if r else {})

  @api.post("/api/v1/delete_model")
  async def delete_model(req: dict = Body(...)):
    submit(req["model_id"], Op.DELETE_MODEL, {}, req.get("seq_id"))
    return {"status": "ok"}

  @api.post("/api/v1/retrieve_future")
  async def retrieve_future(req: dict = Body(...)):
    if (reply := await ledger.wait_for_reply(rid := req["request_id"], WAIT)) is None:
      return {"type": "try_again", "request_id": rid, "queue_state": "active"}
    return Response(reply.body, media_type=reply.content_type)

  # ***** sampling *****

  @api.post("/api/v1/create_sampling_session")
  async def create_sampling_session(data: dict = Body(...)):
    sid = data.get("sampling_session_id")
    if not (sid and ledger.get_run(sid)):
      sid = await open_sampler(data.get("base_model"), data.get("model_path"), data.get("session_id", ""))
    return {"type": "create_sampling_session", "sampling_session_id": sid}

  @api.post("/api/v1/asample")
  async def asample(data: dict = Body(...)):
    chunks = (data.get("prompt") or {}).get("chunks", [])
    if any(c.get("type", "encoded_text") != "encoded_text" for c in chunks):
      raise ValueError("only encoded_text prompt chunks are supported")
    if not (tokens := [int(t) for c in chunks for t in c.get("tokens", [])]):
      raise ValueError("empty prompt is not supported")
    if (last_n := data.get("prompt_logprobs_last_n")) is not None:
      if not data.get("prompt_logprobs") or not 1 <= last_n <= len(tokens) - 1:
        raise ValueError("invalid prompt_logprobs_last_n")
    sid = data.get("sampling_session_id") or await open_sampler(data.get("base_model"), data.get("model_path"))
    n, p, mt = int(data.get("num_samples", 1)), data.get("sampling_params") or {}, 0
    mt = p.get("max_tokens") or 128
    job = {"prompt": tokens, "num_samples": n, "max_tokens": mt, "seed": p.get("seed"), "stop": p.get("stop")}
    job |= {
      "temperature": float(p.get("temperature", 1.0)),
      "top_k": int(p.get("top_k", -1)),
      "top_p": float(p.get("top_p", 1.0)),
    }
    job |= {"prompt_logprobs": bool(data.get("prompt_logprobs")), "prompt_logprobs_last_n": last_n}
    job |= {
      "topk_prompt": int(data.get("topk_prompt_logprobs", 0)),
      "topk_sample": int(data.get("topk_sample_logprobs", 0)),
    }
    job["record_stability_info"] = bool(data.get("record_stability_info"))
    rid = submit(sid, Op.SAMPLE, job, data.get("seq_id"), len(tokens) + n * mt)["request_id"]
    return {"request_id": rid, "sample_sequence_ids": [f"{rid}:{i}" for i in range(n)]}

  # ***** checkpoints & inspection *****

  @api.get("/api/v1/training_runs/{run}/checkpoints")
  async def list_checkpoints(run: str):
    cs = [c for u, c in ledger.checkpoints.items() if parse_uri(u)[0] == run and c["checkpoint_type"] != "parked"]
    return {"checkpoints": [public(c) for c in sorted(cs, key=lambda c: c["time"])]}

  @api.get("/api/v1/checkpoints")
  async def list_user_checkpoints(limit: int = 100, offset: int = 0):
    vis = sorted((c for c in ledger.checkpoints.values() if c["checkpoint_type"] != "parked"), key=lambda c: -c["time"])
    page = [public(c) for c in (vis[offset : offset + limit] if limit > 0 else [])]
    return {"checkpoints": page, "cursor": {"offset": offset, "limit": limit, "total_count": len(vis)}}

  @api.delete("/api/v1/training_runs/{run}/checkpoints/{cid:path}")
  async def delete_checkpoint(run: str, cid: str):
    if not ledger.delete_checkpoint(f"tinker://{run}/{cid}"):
      raise HTTPException(404, f"no checkpoint {cid} in {run}")

  @api.put("/api/v1/training_runs/{run}/checkpoints/{cid:path}/ttl")
  async def set_checkpoint_ttl(run: str, cid: str, req: dict = Body(...)):
    if not ledger.set_checkpoint_ttl(f"tinker://{run}/{cid}", ttl(req.get("ttl_seconds"))):
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
