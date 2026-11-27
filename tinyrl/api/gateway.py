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

from tinyrl.api import wire
from tinyrl.control import estimator
from tinyrl.control.controller import COUNTERS, TICK_MS, Controller
from tinyrl.helpers import getenv
from tinyrl.types import Checkpoint, FencedError, NodeMsg, Op, QueueFullError, RequestGoneError, Role, Run, parse_uri

WAIT, CKPT_TTL, MAX_TTL_SECONDS = getenv("WAIT", 30.0), getenv("CKPT_TTL", 0), 10 * 365 * 24 * 3600
CHECKPOINT_KINDS = {"weights": "training", "sampler_weights": "sampler"}
ERROR_CODES = {ValueError: 400, KeyError: 400, FencedError: 409, RequestGoneError: 410, QueueFullError: 429}


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
    ttl = self.ttl_seconds if (self.ttl_seconds is not None or owned) else (CKPT_TTL or None)
    return {"uri": ckpt_uri(self.model_id, kind, name), "ttl_seconds": check_ttl(ttl)}


class SamplingReq(BaseModel):
  sampling_session_id: str | None = None
  base_model: str | None = None
  model_path: str | None = None
  session_id: str | None = None


def check_ttl(seconds: int | None) -> int | None:
  if seconds is not None and not 3600 <= seconds <= MAX_TTL_SECONDS:
    raise HTTPException(400, "ttl_seconds must be between 3600 and 10 years")
  return seconds


def iso(timestamp: float | None) -> str | None:
  return datetime.datetime.fromtimestamp(timestamp, datetime.UTC).isoformat() if timestamp else None


def public(ckpt: Checkpoint) -> dict:
  _, cid = parse_uri(ckpt.uri)
  prefix = cid.split("/", 1)[0]
  return {
    "checkpoint_id": cid,
    "checkpoint_type": CHECKPOINT_KINDS.get(prefix, prefix),
    "tinker_path": ckpt.uri,
    "size_bytes": ckpt.size,
    "time": iso(ckpt.time),
    "expires_at": iso(ckpt.expires_at),
  }


def ckpt_uri(run: str, kind: str, name: str) -> str:
  owner, cid = parse_uri(name if name.startswith("tinker://") else f"tinker://{run}/{kind}/{name}")
  if not cid.partition("/")[2]:
    raise ValueError(f"bad checkpoint name {name}")
  return f"tinker://{owner}/{cid}"


def app(ctrl: Controller | None = None, *, control: bool = False) -> FastAPI:
  ctrl = Controller() if ctrl is None else ctrl

  @contextlib.asynccontextmanager
  async def lifespan(_: FastAPI):
    async def maintain_loop() -> None:
      tick_s = TICK_MS / 1000
      while True:
        try:
          ctrl.maintain()
        except Exception as exc:
          print(f"controller maintain failed: {exc!r}", file=sys.stderr, flush=True)
        due = ctrl.next_wake - time.monotonic() if ctrl.next_wake is not None else tick_s
        await asyncio.sleep(min(tick_s, max(0.001, due)))

    maint_task = asyncio.create_task(maintain_loop())
    try:
      yield
    finally:
      maint_task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await maint_task
      await ctrl.aclose()

  api = FastAPI(title="tiny-open-rl", lifespan=lifespan if control else None)
  for exc_type, status_code in ERROR_CODES.items():
    api.add_exception_handler(
      exc_type, lambda _, err, code=status_code: JSONResponse({"error": f"{type(err).__name__}: {err}"}, code)
    )

  live_ws: dict[str, tuple[WebSocket, asyncio.Task | None]] = {}
  closing: set[asyncio.Task] = set()

  def close_later(old: WebSocket) -> None:
    async def _close() -> None:
      with contextlib.suppress(BaseException):
        await old.close()

    t = asyncio.create_task(_close())
    closing.add(t)
    t.add_done_callback(closing.discard)

  @api.websocket("/internal/ws")
  async def node_ws(ws: WebSocket) -> None:
    await ws.accept()
    bound_node: tuple[str, str] | None = None
    pusher: asyncio.Task | None = None
    send_lock = asyncio.Lock()

    async def push_loop(node_id: str, boot_id: str) -> None:
      while (node := ctrl.nodes.get(node_id)) is not None and node.boot_id == boot_id:
        await node.push_ev.wait()
        async with send_lock:
          node = ctrl.nodes.get(node_id)
          if live_ws.get(node_id, (None, None))[0] is not ws or node is None or node.boot_id != boot_id:
            return
          out = node.drain_push()
          if out.tasks or out.cancel or out.stage or out.sweep is not None or out.ack:
            await ws.send_bytes(msgspec.msgpack.encode(out))

    try:
      while True:
        msg = msgspec.msgpack.decode(await ws.receive_bytes(), type=NodeMsg)
        async with send_lock:
          if bound_node is not None and live_ws.get(bound_node[0], (None, None))[0] is not ws:
            await ws.close()
            return
          reply = ctrl.handle(msg)
          if not reply.reset:
            old_ws, old_pusher = live_ws.get(msg.node, (None, None))
            if old_ws is not None and old_ws is not ws:
              if old_pusher is not None:
                old_pusher.cancel()
              close_later(old_ws)
            live_ws[msg.node] = (ws, pusher if bound_node == (msg.node, msg.boot_id) else None)
          await ws.send_bytes(msgspec.msgpack.encode(reply))
          if reply.reset or live_ws.get(msg.node, (None, None))[0] is not ws:
            await ws.close()
            return
          if bound_node != (msg.node, msg.boot_id):
            if pusher is not None:
              pusher.cancel()
            bound_node = (msg.node, msg.boot_id)
            pusher = asyncio.create_task(push_loop(*bound_node))
            live_ws[msg.node] = (ws, pusher)
    except (WebSocketDisconnect, RuntimeError):
      pass
    finally:
      if pusher is not None:
        pusher.cancel()
        with contextlib.suppress(BaseException):
          await pusher
      if bound_node is not None and live_ws.get(bound_node[0], (None, None))[0] is ws:
        live_ws.pop(bound_node[0], None)

  def require_run(run_id: str | None) -> dict:
    if not run_id or (record := ctrl.get_run(run_id)) is None:
      raise HTTPException(404, f"no model {run_id}")
    return record

  def submit(
    run: str,
    op: Op,
    body: bytes | dict,
    seq_id: int | None = None,
    cost: int = 1,
    uri: str | None = None,
    refs: set[str] | None = None,
  ) -> dict[str, str]:
    payload = json.dumps(body).encode() if isinstance(body, dict) else body
    if (req_id := ctrl.submit_command(run, op, payload, seq_id, cost, uri, refs)) is None:
      raise HTTPException(404, f"no model {run}")
    return {"request_id": req_id, "model_id": run}

  async def make_sampler(
    base_model: str | None, model_path: str | None, session_id: str = "", *, create: bool = True
  ) -> Run:
    path = model_path or None
    train_run_id = parse_uri(path)[0] if path else None
    train_run = (ctrl.get_run(train_run_id) or {}) if train_run_id else {}
    ckpt_meta = (ckpt.meta if (ckpt := ctrl.get_checkpoint(path)) else {}) if path and not train_run.get("base") else {}

    base = train_run.get("base") or ckpt_meta.get("base") or base_model
    if not base:
      raise ValueError("could not resolve base_model" if path else "base_model required when model_path is None")
    lora = (train_run["rank"] > 0) if train_run.get("base") else (bool(ckpt_meta.get("lora")) if path else True)
    session_meta = (ctrl.get_session(session_id) or {}).get("metadata", {}) if session_id else {}
    account = str(train_run.get("account") or session_meta.get("openrl.account") or "")
    gpus = await asyncio.to_thread(estimator.gpus, base, "vllm")
    run = Run(
      f"samp-{uuid.uuid4().hex[:12]}",
      base,
      int(lora),
      session_id,
      path,
      exclusive=bool(train_run.get("exclusive")),
      role=Role.SAMPLER,
      train_run=train_run_id,
      gpus=gpus,
      account=account,
    )
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
    session_id, body = f"sess-{uuid.uuid4().hex[:12]}", req or {}
    tag_meta = dict(tag.partition("=")[::2] for tag in (body.get("tags") or ()))
    ctrl.heartbeat_session(session_id, tag_meta | (body.get("user_metadata") or {}))
    return {"type": "create_session", "session_id": session_id}

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
    session = session_id or ""
    if "/" in session:
      raise ValueError("session_id must not contain '/'")
    lora = lora_config or {}
    meta = (ctrl.get_session(session) or {}).get("metadata", {}) | (user_metadata or {})
    rank = 0 if meta.get("openrl.fine_tuning_type") == "full" else int(lora.get("rank", 16))
    run_id = f"{session}-{model_seq_id}" if session and model_seq_id is not None else str(uuid.uuid4())

    raw_gpus = meta.get("openrl.gpus")
    gpus = (
      int(raw_gpus)
      if raw_gpus is not None
      else (1 if rank else await asyncio.to_thread(estimator.gpus, base_model, "full"))
    )
    if gpus < 1:
      raise ValueError(f"openrl.gpus must be at least 1, got {gpus}")
    if not ctrl.get_run(run_id):
      exclusive, account = str(meta.get("openrl.exclusive")).lower() == "true", str(meta.get("openrl.account") or "")
      ctrl.create_run(Run(run_id, base_model, rank, session, None, lora, exclusive, gpus=gpus, account=account))
    dedup_seq = -1 if (session and model_seq_id is not None) else None
    return submit(run_id, Op.CREATE_MODEL, {"base_model": base_model, "rank": rank, "lora": lora}, dedup_seq)

  @api.post("/api/v1/get_info")
  async def get_info(req: ModelReq):
    rec = require_run(req.model_id)
    base = rec["base"]
    return {
      "type": "get_info",
      "model_id": req.model_id,
      "model_name": base,
      "is_lora": True,
      "lora_rank": rec["rank"] or 16,
      "model_data": {"model_name": base, "tokenizer_id": base},
    }

  @api.post("/api/v1/forward_backward")
  async def forward_backward(request: Request):
    body = await request.body()
    parsed = wire.parse(body)
    refs = wire.referenced_sequences(parsed) if parsed.loss_fn == "stable_reinforce" else None
    return submit(parsed.model_id, Op.FORWARD_BACKWARD, body, parsed.seq_id or None, wire.cost(parsed), refs=refs)

  @api.post("/api/v1/optim_step")
  async def optim_step(model_id: str = Body(), seq_id: int | None = Body(None), adam_params: dict[str, Any] = Body({})):
    return submit(model_id, Op.OPTIM_STEP, {"adam_params": adam_params}, seq_id)

  @api.post("/api/v1/save_weights")
  async def save_weights(req: SaveReq):
    return submit(req.model_id, Op.SAVE_WEIGHTS, req.checkpoint("weights"), req.seq_id)

  @api.post("/api/v1/save_weights_for_sampler")
  async def save_weights_for_sampler(req: SaveReq):
    if existing := ctrl.seen_request(req.model_id, req.seq_id):
      return {"request_id": existing, "model_id": req.model_id}
    train_run = require_run(req.model_id)
    owned = req.path is None or req.sampling_session_seq_id is not None
    args = req.checkpoint("sampler_weights", owned=owned)
    if owned:
      sampler = await make_sampler(train_run["base"], args["uri"], train_run.get("session") or "", create=False)
      if existing := ctrl.seen_request(req.model_id, req.seq_id):
        return {"request_id": existing, "model_id": req.model_id}
      if (state := ctrl.runs.get(req.model_id)) is None or not state.live:
        raise HTTPException(404, f"no model {req.model_id}")
      ctrl.create_run(sampler)
      args |= {"sampling_session_id": sampler.id, "owned": True}
    return submit(req.model_id, Op.SAVE_WEIGHTS_FOR_SAMPLER, args, req.seq_id)

  @api.post("/api/v1/load_weights")
  async def load_weights(
    model_id: str = Body(), path: str = Body(), seq_id: int | None = Body(None), optimizer: bool = Body(False)
  ):
    uri = ckpt_uri(model_id, "weights", path)
    return submit(model_id, Op.LOAD_WEIGHTS, {"uri": uri, "optimizer": optimizer}, seq_id, uri=uri)

  @api.post("/api/v1/weights_info")
  async def weights_info(tinker_path: str = Body(embed=True)):
    meta = ckpt.meta if (ckpt := ctrl.get_checkpoint(tinker_path)) else {}
    if "base" not in meta:
      raise HTTPException(404, f"no checkpoint {tinker_path} that names its base model")
    rank = int(meta.get("rank") or 0)
    return {"base_model": meta["base"], "is_lora": True, "lora_rank": rank or 16} | (
      meta.get("flags", {}) if rank else {}
    )

  @api.post("/api/v1/delete_model")
  async def delete_model(req: ModelReq):
    submit(req.model_id, Op.DELETE_MODEL, {}, req.seq_id)
    return {"status": "ok"}

  @api.post("/api/v1/retrieve_future")
  async def retrieve_future(request_id: str = Body(embed=True)):
    if (reply := await ctrl.wait_for_reply(request_id, WAIT)) is None:
      return {"type": "try_again", "request_id": request_id, "queue_state": "active"}
    return Response(reply.body, media_type=reply.content_type)

  @api.post("/api/v1/create_sampling_session")
  async def create_sampling_session(req: SamplingReq):
    sampler_id = req.sampling_session_id
    if not (sampler_id and ctrl.get_run(sampler_id)):
      sampler_id = (await make_sampler(req.base_model, req.model_path, req.session_id or "")).id
    return {"type": "create_sampling_session", "sampling_session_id": sampler_id}

  @api.post("/api/v1/join_sampling_session")
  async def join_sampling_session(req: SamplingReq):
    require_run(req.sampling_session_id)
    return {"type": "join_sampling_session", "client_counter": ctrl.join_sampling_session(req.sampling_session_id)}

  @api.get("/api/v1/samplers/{sampler_id}")
  async def get_sampler(sampler_id: str):
    rec = require_run(sampler_id)
    return {"sampler_id": sampler_id, "base_model": rec["base"], "model_path": rec.get("model_path")}

  @api.post("/api/v1/asample")
  async def asample(request: Request):
    body = await request.json()
    chunks = (body.get("prompt") or {}).get("chunks", [])
    if any(c.get("type", "encoded_text") != "encoded_text" for c in chunks):
      raise ValueError("only encoded_text prompt chunks are supported")
    tokens = [int(tok) for c in chunks for tok in c.get("tokens", [])]
    if not tokens:
      raise ValueError("empty prompt is not supported")
    if body.get("target_prompt_logprobs") is not None or (body.get("prompt_alt_tokens_k") or 0) > 0:
      raise ValueError("target_prompt_logprobs and prompt_alt_tokens are not supported")
    last_n = body.get("prompt_logprobs_last_n")
    if last_n is not None and (not body.get("prompt_logprobs") or not 1 <= last_n <= len(tokens) - 1):
      raise ValueError("invalid prompt_logprobs_last_n")

    sampler_id = (
      body.get("sampling_session_id") or (await make_sampler(body.get("base_model"), body.get("model_path"))).id
    )
    num_samples, params = int(body.get("num_samples", 1)), body.get("sampling_params") or {}
    max_tokens = params.get("max_tokens") or 128
    job = {
      "prompt": tokens,
      "num_samples": num_samples,
      "max_tokens": max_tokens,
      "seed": params.get("seed"),
      "stop": params.get("stop"),
      "temperature": float(params.get("temperature", 1.0)),
      "top_k": int(params.get("top_k", -1)),
      "top_p": float(params.get("top_p", 1.0)),
      "prompt_logprobs": bool(body.get("prompt_logprobs")),
      "prompt_logprobs_last_n": last_n,
      "topk_prompt": int(body.get("topk_prompt_logprobs", 0)),
      "topk_sample": int(body.get("topk_sample_logprobs", 0)),
      "record_stability_info": bool(body.get("record_stability_info")),
    }
    req_id = submit(sampler_id, Op.SAMPLE, job, body.get("seq_id"), len(tokens) + num_samples * max_tokens)[
      "request_id"
    ]
    return {"request_id": req_id, "sample_sequence_ids": [f"{req_id}:{i}" for i in range(num_samples)]}

  @api.get("/api/v1/training_runs/{run}/checkpoints")
  async def list_checkpoints(run: str):
    return {"checkpoints": [public(c) for c in ctrl.list_checkpoints(run)]}

  @api.get("/api/v1/checkpoints")
  async def list_user_checkpoints(limit: int = 100, offset: int = 0):
    visible = ctrl.list_checkpoints()
    window = visible[offset : offset + limit] if limit > 0 else []
    return {
      "checkpoints": [public(c) for c in window],
      "cursor": {"offset": offset, "limit": limit, "total_count": len(visible)},
    }

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
    rec, workers = require_run(run_id), ctrl.workers_of(run_id)
    return {**rec, "procs": workers, **ctrl.queued_and_in_flight(run_id), **({"worker": workers[0]} if workers else {})}

  @api.get("/api/v1/sessions/{session_id}")
  async def session_info(session_id: str):
    if (session := ctrl.get_session(session_id)) is None:
      raise HTTPException(404, f"no session {session_id}")
    return session

  @api.get("/metrics")
  async def metrics():
    ctrl.expire_nodes()
    sampler_worker: dict[Any, str] = {}
    for n in ctrl.nodes.values():
      if n.gpus:
        for w in n.workers.values():
          if not w.stopping and (w.spec not in sampler_worker or w.id < sampler_worker[w.spec]):
            sampler_worker[w.spec] = w.id
    lines: list[str] = []
    for run_id, state in sorted(ctrl.runs.items()):
      if not state.live:
        continue
      wid = (
        (state.owner[0] if state.owner else None)
        if state.run.role == Role.TRAINER
        else sampler_worker.get(state.run.spec)
      )
      if wid:
        for key in COUNTERS:
          label = f"{key[:-3]}_seconds" if key.endswith("_ms") else key
          val = state.counters[key] / 1000 if key.endswith("_ms") else state.counters[key]
          lines.append(f'tinyrl_{label}_total{{worker="{wid}",run="{run_id}"}} {val}')
    lines += [f'tinyrl_account_service_total{{account="{a}"}} {float(t)}' for a, t in sorted(ctrl.service.items())]
    lines += [
      f'tinyrl_node_loop_lag_seconds{{node="{nid}"}} {n.lag}' for nid, n in sorted(ctrl.nodes.items()) if n.gpus
    ]
    return PlainTextResponse("\n".join(lines) + "\n")

  return api


if __name__ == "__main__":
  uvicorn.run(
    app(control=True), host="0.0.0.0", port=getenv("PORT", 8000), ws_max_size=getenv("WS_MAX_BYTES", 64 * 2**20)
  )
