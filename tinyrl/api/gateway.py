"""Tinker HTTP API and full-duplex `/internal/ws` node channel (zero Store, fsspec, or ML imports)."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import sys
import time
import uuid
from typing import Any

import msgspec
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from tinyrl import types as T
from tinyrl.api import wire
from tinyrl.control import estimator
from tinyrl.control.controller import COUNTERS, TICK_MS, Controller
from tinyrl.helpers import getenv

WAIT, CKPT_TTL = getenv("WAIT", 30.0), getenv("CKPT_TTL", 0)
ERRORS = {ValueError: 400, KeyError: 400, T.FencedError: 409, T.RequestGoneError: 410, T.QueueFullError: 429}


class ModelReq(BaseModel):
  model_id: str
  seq_id: int | None = None


class SaveReq(ModelReq):
  path: str | None = None
  sampling_session_seq_id: int | None = None
  ttl_seconds: int | None = None

  def checkpoint(self, kind: str, *, owned: bool = False) -> dict:
    seq = self.sampling_session_seq_id if self.sampling_session_seq_id is not None else self.seq_id
    name = self.path or f"step-{seq if seq is not None else time.time_ns()}"
    sec = self.ttl_seconds if self.ttl_seconds is not None or owned else CKPT_TTL or None
    return {"uri": ckpt_uri(self.model_id, kind, name), "ttl_seconds": check_ttl(sec)}


class SamplingReq(BaseModel):
  sampling_session_id: str | None = None
  base_model: str | None = None
  model_path: str | None = None
  session_id: str | None = None


def check_ttl(seconds: int | None) -> int | None:
  if seconds is not None and not 3600 <= seconds <= 10 * 365 * 24 * 3600:
    raise HTTPException(400, "ttl_seconds must be between 3600 and 10 years")
  return seconds


def iso(t: float | None) -> str | None:
  return datetime.datetime.fromtimestamp(t, datetime.UTC).isoformat() if t else None


def public(c: T.Checkpoint) -> dict:
  raw = (cid := T.parse_uri(c.uri)[1]).split("/", 1)[0]
  kind = {"weights": "training", "sampler_weights": "sampler"}.get(raw, raw)
  out = {"checkpoint_id": cid, "checkpoint_type": kind, "tinker_path": c.uri, "size_bytes": c.size}
  return out | {"time": iso(c.time), "expires_at": iso(c.expires_at)}


def ckpt_uri(run: str, kind: str, name: str) -> str:
  owner, cid = T.parse_uri(name if name.startswith("tinker://") else f"tinker://{run}/{kind}/{name}")
  if not cid.partition("/")[2]:
    raise ValueError(f"bad checkpoint name {name}")
  return f"tinker://{owner}/{cid}"


def app(ctrl: Controller | None = None, *, control: bool = False) -> FastAPI:
  ctrl = ctrl if ctrl is not None else Controller()

  @contextlib.asynccontextmanager
  async def lifespan(_: FastAPI):
    async def loop():
      while True:
        try:
          ctrl.maintain()
        except Exception as exc:
          print(f"controller maintain failed: {exc!r}", file=sys.stderr, flush=True)
        due = ctrl.next_wake - time.monotonic() if ctrl.next_wake is not None else TICK_MS / 1000
        await asyncio.sleep(min(TICK_MS / 1000, max(0.001, due)))

    maint = asyncio.create_task(loop())
    try:
      yield
    finally:
      maint.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await maint
      await ctrl.aclose()

  api = FastAPI(title="tiny-open-rl", lifespan=lifespan if control else None)
  for exc, code in ERRORS.items():
    api.add_exception_handler(exc, lambda _, err, c=code: JSONResponse({"error": f"{type(err).__name__}: {err}"}, c))

  @api.websocket("/internal/ws")
  async def node_ws(ws: WebSocket) -> None:
    await ws.accept()
    key, pusher = None, None

    async def push(nid: str, bid: str) -> None:
      while (ns := ctrl.nodes.get(nid)) and ns.boot_id == bid:
        await ns.push_ev.wait()
        if (ns := ctrl.nodes.get(nid)) and ns.boot_id == bid:
          out = ns.drain_push()
          if out.tasks or out.cancel or out.stage or out.sweep or out.ack:
            await ws.send_bytes(msgspec.msgpack.encode(out))

    try:
      while True:
        msg = msgspec.msgpack.decode(await ws.receive_bytes(), type=T.NodeMsg)
        await ws.send_bytes(msgspec.msgpack.encode(reply := ctrl.handle(msg)))
        if reply.reset:
          return await ws.close()
        if key != (msg.node, msg.boot_id):
          if pusher:
            pusher.cancel()
          key = (msg.node, msg.boot_id)
          pusher = asyncio.create_task(push(*key))
    except (WebSocketDisconnect, RuntimeError):
      pass
    finally:
      if pusher:
        pusher.cancel()
        with contextlib.suppress(BaseException):
          await pusher

  def require_run(run_id: str | None) -> dict:
    if not run_id or (rec := ctrl.get_run(run_id)) is None:
      raise HTTPException(404, f"no model {run_id}")
    return rec

  def submit(run: str, op: T.Op, body: bytes | dict, seq_id=None, cost=1, uri=None, refs=None) -> dict[str, str]:
    raw = json.dumps(body).encode() if isinstance(body, dict) else body
    if (rid := ctrl.submit_command(run, op, raw, seq_id, cost, uri, refs)) is None:
      raise HTTPException(404, f"no model {run}")
    return {"request_id": rid, "model_id": run}

  async def make_sampler(
    base_model: str | None, model_path: str | None, sid: str = "", *, create: bool = True
  ) -> T.Run:
    tr_id = T.parse_uri(mp)[0] if (mp := model_path or None) else None
    tr = (ctrl.get_run(tr_id) or {}) if tr_id else {}
    meta = (ck.meta if (ck := ctrl.get_checkpoint(mp)) else {}) if mp and not tr.get("base") else {}
    base = tr.get("base") or meta.get("base") or base_model
    lora = tr["rank"] > 0 if tr.get("base") else (bool(meta.get("lora")) if mp else True)
    if not base:
      raise ValueError("could not resolve base_model" if mp else "base_model required when model_path is None")
    smeta = (ctrl.get_session(sid) or {}).get("metadata", {}) if sid else {}
    acct, excl = str(tr.get("account") or smeta.get("openrl.account") or ""), bool(tr.get("exclusive"))
    gpus = await asyncio.to_thread(estimator.gpus, base, "vllm")
    run = T.Run(f"samp-{uuid.uuid4().hex[:12]}", base, int(lora), sid, mp, {}, excl, T.Role.SAMPLER, tr_id, gpus, acct)
    if create:
      ctrl.create_run(run)
    return run

  api.get("/api/v1/scheduling")(ctrl.snapshot_scheduling)
  api.get("/api/v1/healthz")(lambda: {"status": "ok"})
  api.post("/api/v1/telemetry")(lambda: {"status": "accepted"})
  api.post("/api/v1/client/config")(lambda: {"parallel_fwdbwd_chunks": False})
  api.post("/api/v1/client/dynamic_config")(lambda: {})
  api.get("/api/v1/get_server_capabilities")(lambda: {"supported_models": []})

  @api.post("/api/v1/sessions/{session_id}/finish")
  async def finish_session(session_id: str):
    ctrl.heartbeat_session(session_id, finish=True)
    return {"status": "ok"}

  @api.post("/api/v1/create_session")
  async def create_session(req: dict[str, Any] | None = Body(None)):
    sid, d = f"sess-{uuid.uuid4().hex[:12]}", req or {}
    ctrl.heartbeat_session(
      sid, dict(t.partition("=")[::2] for t in d.get("tags") or ()) | (d.get("user_metadata") or {})
    )
    return {"type": "create_session", "session_id": sid}

  @api.post("/api/v1/session_heartbeat")
  async def session_heartbeat(session_id: str = Body(embed=True)):
    ctrl.heartbeat_session(session_id)
    return {"type": "session_heartbeat"}

  @api.post("/api/v1/create_model")
  async def create_model(
    base_model: str = Body(),
    session_id: str | None = Body(None),
    lora_config: dict | None = Body(None),
    user_metadata: dict | None = Body(None),
    model_seq_id: int | None = Body(None),
  ):
    sess, lora = session_id or "", lora_config or {}
    meta = (ctrl.get_session(sess) or {}).get("metadata", {}) | (user_metadata or {})
    rank = 0 if meta.get("openrl.fine_tuning_type") == "full" else int(lora.get("rank", 16))
    run = f"{sess}-{model_seq_id}" if sess and model_seq_id is not None else str(uuid.uuid4())
    raw = meta.get("openrl.gpus")
    gpus = (1 if rank else await asyncio.to_thread(estimator.gpus, base_model, "full")) if raw is None else int(raw)
    if gpus < 1:
      raise ValueError(f"openrl.gpus must be at least 1, got {gpus}")
    if not ctrl.get_run(run):
      excl, acct = str(meta.get("openrl.exclusive")).lower() == "true", str(meta.get("openrl.account") or "")
      ctrl.create_run(T.Run(run, base_model, rank, sess, None, lora, excl, gpus=gpus, account=acct))
    return submit(run, T.Op.CREATE_MODEL, {"base_model": base_model, "rank": rank, "lora": lora}, model_seq_id)

  @api.post("/api/v1/get_info")
  async def get_info(req: ModelReq):
    base, rank = (rec := require_run(req.model_id))["base"], rec["rank"] or 16
    info = {"type": "get_info", "model_id": req.model_id, "model_name": base, "is_lora": True, "lora_rank": rank}
    return info | {"model_data": {"model_name": base, "tokenizer_id": base}}

  @api.post("/api/v1/forward_backward")
  async def forward_backward(request: Request):
    p = wire.parse(body := await request.body())
    refs = wire.referenced_sequences(p) if p.loss_fn == "stable_reinforce" else None
    return submit(p.model_id, T.Op.FORWARD_BACKWARD, body, p.seq_id or None, wire.cost(p), refs=refs)

  @api.post("/api/v1/optim_step")
  async def optim_step(model_id: str = Body(), seq_id: int | None = Body(None), adam_params: dict[str, Any] = Body({})):
    return submit(model_id, T.Op.OPTIM_STEP, {"adam_params": adam_params}, seq_id)

  @api.post("/api/v1/save_weights")
  async def save_weights(req: SaveReq):
    return submit(req.model_id, T.Op.SAVE_WEIGHTS, req.checkpoint("weights"), req.seq_id)

  @api.post("/api/v1/save_weights_for_sampler")
  async def save_weights_for_sampler(req: SaveReq):
    if rid := ctrl.seen_request(req.model_id, req.seq_id):
      return {"request_id": rid, "model_id": req.model_id}
    tr = require_run(req.model_id)
    args = req.checkpoint(
      "sampler_weights", owned=(owned := req.path is None or req.sampling_session_seq_id is not None)
    )
    if owned:
      samp = await make_sampler(tr["base"], args["uri"], tr.get("session") or "", create=False)
      if rid := ctrl.seen_request(req.model_id, req.seq_id):
        return {"request_id": rid, "model_id": req.model_id}
      ctrl.create_run(samp)
      args |= {"sampling_session_id": samp.id, "owned": True}
    return submit(req.model_id, T.Op.SAVE_WEIGHTS_FOR_SAMPLER, args, req.seq_id)

  @api.post("/api/v1/load_weights")
  async def load_weights(
    model_id: str = Body(), path: str = Body(), seq_id: int | None = Body(None), optimizer: bool = Body(False)
  ):
    uri = ckpt_uri(model_id, "weights", path)
    return submit(model_id, T.Op.LOAD_WEIGHTS, {"uri": uri, "optimizer": optimizer}, seq_id, uri=uri)

  @api.post("/api/v1/weights_info")
  async def weights_info(tinker_path: str = Body(embed=True)):
    meta = ck.meta if (ck := ctrl.get_checkpoint(tinker_path)) else {}
    if "base" not in meta:
      raise HTTPException(404, f"no checkpoint {tinker_path} that names its base model")
    rank = int(meta.get("rank") or 0)
    return {"base_model": meta["base"], "is_lora": True, "lora_rank": rank or 16} | (
      meta.get("flags", {}) if rank else {}
    )

  @api.post("/api/v1/delete_model")
  async def delete_model(req: ModelReq):
    submit(req.model_id, T.Op.DELETE_MODEL, {}, req.seq_id)
    return {"status": "ok"}

  @api.post("/api/v1/retrieve_future")
  async def retrieve_future(request_id: str = Body(embed=True)):
    if (reply := await ctrl.wait_for_reply(request_id, WAIT)) is None:
      return {"type": "try_again", "request_id": request_id, "queue_state": "active"}
    return Response(reply.body, media_type=reply.content_type)

  @api.post("/api/v1/create_sampling_session")
  async def create_sampling_session(req: SamplingReq):
    sid = req.sampling_session_id
    if not (sid and ctrl.get_run(sid)):
      sid = (await make_sampler(req.base_model, req.model_path, req.session_id or "")).id
    return {"type": "create_sampling_session", "sampling_session_id": sid}

  @api.post("/api/v1/join_sampling_session")
  async def join_sampling_session(req: SamplingReq):
    require_run(req.sampling_session_id)
    return {"type": "join_sampling_session", "client_counter": ctrl.join_sampling_session(req.sampling_session_id)}

  @api.get("/api/v1/samplers/{sampler_id}")
  async def get_sampler(sampler_id: str):
    return {
      "sampler_id": sampler_id,
      "base_model": (r := require_run(sampler_id))["base"],
      "model_path": r.get("model_path"),
    }

  @api.post("/api/v1/asample")
  async def asample(request: Request):
    d = await request.json()
    chunks = (d.get("prompt") or {}).get("chunks", [])
    if any(c.get("type", "encoded_text") != "encoded_text" for c in chunks):
      raise ValueError("only encoded_text prompt chunks are supported")
    if not (toks := [int(t) for c in chunks for t in c.get("tokens", [])]):
      raise ValueError("empty prompt is not supported")
    if d.get("target_prompt_logprobs") is not None or (d.get("prompt_alt_tokens_k") or 0) > 0:
      raise ValueError("target_prompt_logprobs and prompt_alt_tokens are not supported")
    last_n = d.get("prompt_logprobs_last_n")
    if last_n is not None and (not d.get("prompt_logprobs") or not 1 <= last_n <= len(toks) - 1):
      raise ValueError("invalid prompt_logprobs_last_n")
    sid = d.get("sampling_session_id") or (await make_sampler(d.get("base_model"), d.get("model_path"))).id
    n, p = int(d.get("num_samples", 1)), d.get("sampling_params") or {}
    mt = p.get("max_tokens") or 128
    job = {"prompt": toks, "num_samples": n, "max_tokens": mt, "seed": p.get("seed"), "stop": p.get("stop")}
    job |= {
      "temperature": float(p.get("temperature", 1.0)),
      "top_k": int(p.get("top_k", -1)),
      "top_p": float(p.get("top_p", 1.0)),
    }
    job |= {"prompt_logprobs": bool(d.get("prompt_logprobs")), "prompt_logprobs_last_n": last_n}
    job |= {"topk_prompt": int(d.get("topk_prompt_logprobs", 0)), "topk_sample": int(d.get("topk_sample_logprobs", 0))}
    job["record_stability_info"] = bool(d.get("record_stability_info"))
    rid = submit(sid, T.Op.SAMPLE, job, d.get("seq_id"), len(toks) + n * mt)["request_id"]
    return {"request_id": rid, "sample_sequence_ids": [f"{rid}:{i}" for i in range(n)]}

  @api.get("/api/v1/training_runs/{run}/checkpoints")
  async def list_checkpoints(run: str):
    return {"checkpoints": [public(c) for c in ctrl.list_checkpoints(run)]}

  @api.get("/api/v1/checkpoints")
  async def list_user_checkpoints(limit: int = 100, offset: int = 0):
    vis = ctrl.list_checkpoints()
    page = [public(c) for c in (vis[offset : offset + limit] if limit > 0 else [])]
    return {"checkpoints": page, "cursor": {"offset": offset, "limit": limit, "total_count": len(vis)}}

  @api.delete("/api/v1/training_runs/{run}/checkpoints/{cid:path}")
  async def delete_checkpoint(run: str, cid: str):
    if not ctrl.delete_checkpoint(f"tinker://{run}/{cid}"):
      raise HTTPException(404, f"no checkpoint {cid} in {run}")

  @api.put("/api/v1/training_runs/{run}/checkpoints/{cid:path}/ttl")
  async def set_checkpoint_ttl(run: str, cid: str, ttl_seconds: int | None = Body(None, embed=True)):
    if not ctrl.set_checkpoint_ttl(f"tinker://{run}/{cid}", check_ttl(ttl_seconds)):
      raise HTTPException(404, f"no checkpoint {cid} in {run}")

  @api.get("/api/v1/runs/{run_id}")
  async def run_info(run_id: str):
    run, ws = require_run(run_id), ctrl.workers_of(run_id)
    return {**run, "procs": ws, **({"worker": ws[0]} if ws else {}), **ctrl.queued_and_in_flight(run_id)}

  @api.get("/api/v1/sessions/{session_id}")
  async def session_info(session_id: str):
    if (s := ctrl.get_session(session_id)) is None:
      raise HTTPException(404, f"no session {session_id}")
    return s

  @api.get("/metrics")
  async def metrics():
    ctrl.expire_nodes()
    lines = []
    for rid, rs in sorted(ctrl.runs.items()):
      if rs.live and (ws := ctrl.workers_of(rid)):
        for k in COUNTERS:
          lbl, v = (f"{k[:-3]}_seconds", rs.counters[k] / 1000) if k.endswith("_ms") else (k, rs.counters[k])
          lines.append(f'tinyrl_{lbl}_total{{worker="{ws[0]}",run="{rid}"}} {v}')
    lines += [f'tinyrl_account_service_total{{account="{a}"}} {float(v)}' for a, v in sorted(ctrl.service.items())]
    lines += [f'tinyrl_node_loop_lag_seconds{{node="{n}"}} {ns.lag}' for n, ns in sorted(ctrl.nodes.items()) if ns.gpus]
    return PlainTextResponse("\n".join(lines) + "\n")

  return api


if __name__ == "__main__":
  uvicorn.run(app(control=True), host="0.0.0.0", port=getenv("PORT", 8000))
