"""The Tinker HTTP API.

Every training or sampling call becomes a command on its run's stream, and retrieve_future waits for the reply. Internal checkpoint versions, ownership generations and budgets stay inside internal records.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import os
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from tinyrl.api import wire
from tinyrl.bus import Bus, Op, Role, Run, key
from tinyrl.control import estimator, scheduler
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry
from tinyrl.storage.store import Store

WAIT = getenv("WAIT", 30.0)
CKPT_TTL = getenv("CKPT_TTL", 0)
MAX_CLUSTER_GPUS = getenv("MAX_CLUSTER_GPUS", 0)


# ***** requests *****


class SessionRequest(BaseModel):
  session_id: str


class CreateSessionRequest(BaseModel):
  tags: list[str] = []
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


class TTLRequest(BaseModel):
  ttl_seconds: int | None = None


class LoadRequest(BaseModel):
  model_id: str | None = None
  seq_id: int | None = None
  session_id: str | None = None
  model_seq_id: int | None = None
  base_model: str | None = None
  user_metadata: dict[str, Any] | None = None
  path: str
  optimizer: bool = False
  optimizer_config: dict[str, Any] | None = None
  weights_access_token: str | None = None


class FutureRequest(BaseModel):
  request_id: str


class WeightsInfoRequest(BaseModel):
  tinker_path: str


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
    run, kind, name = name.removeprefix("tinker://").split("/", 2)
  if not name or name.startswith("/") or ".." in f"{run}/{kind}/{name}".split("/"):
    raise ValueError(f"bad checkpoint name {name}")
  return f"tinker://{run}/{kind}/{name}"


def parse_classes(raw: Any) -> list[str]:
  if isinstance(raw, str):
    return [c.strip().lower() for c in raw.split(",") if c.strip()]
  return [str(c).strip().lower() for c in (raw or ()) if str(c).strip()]


def reconstruct_meta(meta: dict) -> dict:
  """Derive exact base model, full-FT vs LoRA mode, and full adapter configuration from checkpoint metadata."""
  rank = int(meta.get("rank") or 0)
  is_lora = bool(meta["lora"]) if "lora" in meta else rank > 0
  if not is_lora:
    return {"base_model": meta["base"], "is_lora": False, "lora_rank": None, "fine_tuning_type": "full"}
  if cfg := meta.get("lora_config"):
    trained = {
      "train_unembed": bool(cfg.get("train_unembed", True)),
      "train_mlp": bool(cfg.get("train_mlp", True)),
      "train_attn": bool(cfg.get("train_attn", True)),
      "lora_alpha": cfg.get("lora_alpha", 16),
      "lora_dropout": float(cfg.get("lora_dropout", 0.05)),
    }
  else:
    made = meta.get("targets") or ()
    trained = {"train_unembed": "lm_head" in made, "train_mlp": "gate_proj" in made, "train_attn": "q_proj" in made}
  return {"base_model": meta["base"], "is_lora": True, "lora_rank": rank, **trained}


def app(bus: Bus | None = None, store: Store | None = None) -> FastAPI:
  bus = bus if bus is not None else Bus()
  registry = Registry(bus)
  store = store if store is not None else Store()
  api = FastAPI(title="tiny-open-rl")

  def bad_request(_: Request, e: Exception) -> JSONResponse:
    return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)

  api.add_exception_handler(ValueError, bad_request)
  api.add_exception_handler(KeyError, bad_request)

  async def submit(run: str, op: Op, body: bytes | dict, seq_id: int | None = None, cost: int = 1) -> dict[str, str]:
    if isinstance(body, dict):
      body = json.dumps(body).encode()
    if (request_id := await bus.submit_command(run, op, body, seq_id, cost)) is None:
      raise HTTPException(404, f"no model {run}")
    return {"request_id": request_id, "model_id": run}

  async def check_admission(gpus: int, gpu_classes: list[str], exclusive: bool, rank: int, role: Role) -> None:
    if gpus < 1 or (MAX_CLUSTER_GPUS and gpus > MAX_CLUSTER_GPUS):
      raise ValueError(f"infeasible GPU width {gpus}")
    nodes = await bus.live_nodes(details=True)
    if not nodes:
      return
    profile = {"gpus": gpus, "gpu_classes": gpu_classes, "lora": rank > 0, "rank": rank, "role": str(role)}
    if gpu_classes or exclusive:
      ok = any(
        estimator.compatible(
          profile, v.get("gpu_class", ""), tuple(v.get("gpu_ids") or range(v["gpus"])), v.get("bundles")
        )
        for v in nodes.values()
      )
      if not ok:
        raise ValueError(f"infeasible request: no eligible {gpus}-GPU bundle matches {gpu_classes or 'cluster'}")

  async def read_checkpoint_meta(uri: str) -> tuple[dict, str | None]:
    ver, peers, _, vinfo = await registry.resolve(uri)
    if vinfo and (meta := vinfo.get("meta")) and "base" in meta:
      return meta, ver
    meta = await asyncio.to_thread(store.read_meta, uri, ver, peers)
    return meta, ver

  async def open_sampler(
    base_model: str | None, model_path: str | None, session_id: str = "", *, pending_save: bool = False
  ) -> str:
    pinned_ver, lora_rank = None, 0
    if model_path:
      if not model_path.startswith("tinker://"):
        raise ValueError(f"bad model_path {model_path}")
      uri = checkpoint_uri("x", "sampler_weights", model_path)
      train_run = uri.removeprefix("tinker://").split("/", 1)[0]
      train = await bus.get_run(train_run) or {}
      if not pending_save:
        pinned_ver, peers, _, vinfo = await registry.resolve(uri)
      if train.get("base"):
        base, lora, lora_rank = train["base"], train["rank"] > 0, int(train["rank"])
      else:
        meta = (vinfo or {}).get("meta") or await asyncio.to_thread(store.read_meta, uri, pinned_ver, peers)
        base, lora, lora_rank = meta.get("base") or base_model, bool(meta.get("lora")), int(meta.get("rank") or 0)
        if pinned_ver is None and not pending_save:
          pinned_ver = (await asyncio.to_thread(store.discover_version, uri))[0]
      if not base:
        raise ValueError("could not resolve base_model")
    else:
      if not base_model:
        raise ValueError("base_model required when model_path is None")
      base, lora, uri, train_run, train = base_model, True, None, None, {}

    exclusive = bool(train.get("exclusive")) if train_run else False
    gpu_classes = list((train.get("lora") or {}).get("gpu_classes") or ())
    gpus = await asyncio.to_thread(estimator.gpus, base, "vllm")
    await check_admission(gpus, gpu_classes, exclusive, lora_rank, Role.SAMPLER)
    extra = ({"weights_version": pinned_ver} if pinned_ver else {}) | (
      {"gpu_classes": gpu_classes} if gpu_classes else {}
    )
    sid = f"samp-{uuid.uuid4().hex[:12]}"
    await bus.create_run(
      Run(sid, "", base, int(lora), session_id, uri, extra, exclusive, Role.SAMPLER, train_run, gpus)
    )
    return sid

  # ***** service *****

  @api.get("/api/v1/scheduling")
  async def scheduling():
    table = await bus.read_table()
    s = scheduler.from_table(table)
    state = {
      "workers_made": s.workers_made,
      "nodes": {
        n: dataclasses.asdict(v) | {"cached": sorted(v.cached), "models": sorted(v.models)} for n, v in s.nodes.items()
      },
      "workers": {n: dataclasses.asdict(w) for n, w in s.workers.items()},
      "jobs": {r: dataclasses.asdict(j) for r, j in s.jobs.items()},
      "observations": {n: dataclasses.asdict(o) for n, o in s.observations.items()},
      "service": s.service,
      "reserved": s.reserved,
    }
    for node, info in table["nodes"].items():
      for worker, local in info.get("workers", {}).items():
        if state["workers"].get(worker, {}).get("node") == node:
          state["workers"][worker] |= local
    return {"nodes": {n: v["gpus"] for n, v in table["nodes"].items()}, "state": state}

  @api.get("/api/v1/healthz")
  async def healthz():
    return {"status": "ok"}

  @api.post("/api/v1/sessions/{session_id}/finish")
  async def finish_session(session_id: str):
    await bus.finish_session(session_id)
    return {"status": "ok"}

  @api.post("/api/v1/telemetry")
  async def telemetry():
    return {"status": "accepted"}

  @api.post("/api/v1/client/config")
  @api.post("/api/v1/client/dynamic_config")
  async def client_config():
    return {}

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
    metadata = session.get("metadata", {}) | (req.user_metadata or {})
    lora = dict(req.lora_config or {})
    if lora.get("seed") is None:
      lora["seed"] = int.from_bytes(os.urandom(4), "little") & 0x7FFF_FFFF
    if gpu_classes := parse_classes(metadata.get("openrl.gpu_classes")):
      lora["gpu_classes"] = gpu_classes
    if dtype := metadata.get("openrl.dtype"):
      lora["dtype"] = str(dtype)
    rank = 0 if metadata.get("openrl.fine_tuning_type") == "full" else int(lora.get("rank", 16))
    exclusive = str(metadata.get("openrl.exclusive")).lower() == "true"
    run = (
      f"{req.session_id}-{req.model_seq_id}" if req.session_id and req.model_seq_id is not None else str(uuid.uuid4())
    )
    base = req.base_model
    gpus = int(metadata.get("openrl.gpus") or (1 if rank else await asyncio.to_thread(estimator.gpus, base, "full")))
    await check_admission(gpus, gpu_classes, exclusive, rank, Role.TRAINER)
    if not await bus.get_run(run):
      await bus.create_run(Run(run, "", base, rank, req.session_id, None, lora, exclusive, gpus=gpus))
    return await submit(run, Op.CREATE_MODEL, {"base_model": base, "rank": rank, "lora": lora}, req.model_seq_id)

  @api.post("/api/v1/get_info")
  async def get_info(req: ModelRequest):
    if (run := await bus.get_run(req.model_id)) is None:
      raise HTTPException(404, f"no model {req.model_id}")
    is_lora = run["rank"] > 0
    return {
      "type": "get_info",
      "model_id": req.model_id,
      "model_name": run["base"],
      "is_lora": is_lora,
      "lora_rank": run["rank"] if is_lora else None,
      "model_data": {"model_name": run["base"], "tokenizer_id": run["base"]},
    }

  # ***** training *****

  @api.post("/api/v1/forward_backward")
  async def forward_backward(request: Request):
    body = await request.body()
    if request.headers.get("content-type", "").startswith(wire.PROTO):
      parsed = wire.parse(body)
      return await submit(parsed.model_id, Op.FORWARD_BACKWARD, body, parsed.seq_id or None, wire.cost(parsed))
    data = json.loads(body)
    body = wire.from_json(data)
    return await submit(data["model_id"], Op.FORWARD_BACKWARD, body, data.get("seq_id"), wire.cost(wire.parse(body)))

  @api.post("/api/v1/forward")
  async def forward(request: Request):
    data = await request.json()
    body = wire.from_json(data, forward_only=True)
    return await submit(data["model_id"], Op.FORWARD_BACKWARD, body, data.get("seq_id"), wire.cost(wire.parse(body)))

  @api.post("/api/v1/optim_step")
  async def optim_step(req: OptimStepRequest):
    return await submit(req.model_id, Op.OPTIM_STEP, {"adam_params": req.adam_params}, req.seq_id)

  async def save(req: SaveRequest, op: Op, kind: str):
    seq = req.sampling_session_seq_id if req.sampling_session_seq_id is not None else req.seq_id
    name = req.path or f"step-{seq if seq is not None else time.time_ns()}"
    kept = req.ttl_seconds if req.ttl_seconds is not None else CKPT_TTL or None
    return await submit(
      req.model_id, op, {"uri": checkpoint_uri(req.model_id, kind, name), "ttl_seconds": ttl(kept)}, req.seq_id
    )

  @api.post("/api/v1/save_weights")
  async def save_weights(req: SaveRequest):
    return await save(req, Op.SAVE_WEIGHTS, "weights")

  @api.post("/api/v1/save_weights_for_sampler")
  async def save_weights_for_sampler(req: SaveRequest):
    if req.seq_id is not None and (request_id := await bus.r.get(key("seq", req.model_id, str(req.seq_id)))):
      return {"request_id": request_id.decode(), "model_id": req.model_id}
    if not (train := await bus.get_run(req.model_id)):
      raise HTTPException(404, f"no model {req.model_id}")
    seq = req.sampling_session_seq_id if req.sampling_session_seq_id is not None else req.seq_id
    uri = checkpoint_uri(
      req.model_id, "sampler_weights", req.path or f"step-{seq if seq is not None else time.time_ns()}"
    )
    owned = req.path is None or req.sampling_session_seq_id is not None
    kept = req.ttl_seconds if req.ttl_seconds is not None or owned else CKPT_TTL or None
    args: dict[str, Any] = {"uri": uri, "ttl_seconds": ttl(kept)}
    if owned:
      sid = await open_sampler(train["base"], uri, train.get("session", ""), pending_save=True)
      args |= {"sampling_session_id": sid, "owned": True}
    return await submit(req.model_id, Op.SAVE_WEIGHTS_FOR_SAMPLER, args, req.seq_id)

  @api.post("/api/v1/load_weights")
  async def load_weights(req: LoadRequest):
    if req.model_id is None:
      if not req.session_id or req.model_seq_id is None:
        raise ValueError("load_weights requires model_id or (session_id, model_seq_id)")
      uri = checkpoint_uri("", "weights", req.path)
      try:
        meta, ver = await read_checkpoint_meta(uri)
      except (OSError, IndexError, ValueError) as e:
        raise HTTPException(404, f"no checkpoint {req.path}: {e}") from e
      if "base" not in meta:
        raise HTTPException(404, f"checkpoint {req.path} has no base model metadata")
      if req.optimizer and (meta.get("kind") == "sampler" or not meta.get("optimizer")):
        raise ValueError("checkpoint has no optimizer state")
      info = reconstruct_meta(meta)
      session = await bus.get_session(req.session_id) or {}
      metadata = session.get("metadata", {}) | (req.user_metadata or {})
      base = req.base_model or info["base_model"]
      rank = int(info["lora_rank"] or 0) if info["is_lora"] else 0
      lora = dict(meta.get("lora_config") or {})
      if info["is_lora"]:
        for k in ("train_attn", "train_mlp", "train_unembed", "lora_alpha", "lora_dropout"):
          if k in info:
            lora.setdefault(k, info[k])
        lora["rank"] = rank
      lora.setdefault("seed", int(meta.get("seed") or 0))
      if gpu_classes := parse_classes(metadata.get("openrl.gpu_classes")):
        lora["gpu_classes"] = gpu_classes
      exclusive = str(metadata.get("openrl.exclusive")).lower() == "true"
      src_gpus = int((meta.get("profile") or {}).get("world_size") or 1)
      default_gpus = (
        1 if rank else (src_gpus if src_gpus > 1 else await asyncio.to_thread(estimator.gpus, base, "full"))
      )
      gpus = int(metadata.get("openrl.gpus") or default_gpus)
      await check_admission(gpus, gpu_classes, exclusive, rank, Role.TRAINER)
      model_id = f"{req.session_id}:train:{req.model_seq_id}"
      if not await bus.get_run(model_id):
        await bus.create_run(Run(model_id, "", base, rank, req.session_id, None, lora, exclusive, gpus=gpus))
      return await submit(model_id, Op.LOAD_WEIGHTS, {"uri": uri, "optimizer": req.optimizer, "version": ver}, 0)

    uri = checkpoint_uri(req.model_id, "weights", req.path)
    ver, _, _, _ = await registry.resolve(uri)
    body = {"uri": uri, "optimizer": req.optimizer, **({"version": ver} if ver else {})}
    return await submit(req.model_id, Op.LOAD_WEIGHTS, body, req.seq_id)

  @api.post("/api/v1/weights_info")
  async def weights_info(req: WeightsInfoRequest):
    """Answer checkpoint reconstruction queries directly from the exact registered version's metadata."""
    try:
      meta, _ = await read_checkpoint_meta(checkpoint_uri("", "", req.tinker_path))
    except (OSError, IndexError, ValueError):
      meta = {}
    if "base" not in meta:
      raise HTTPException(404, f"no checkpoint {req.tinker_path} that names its base model")
    return reconstruct_meta(meta)

  @api.post("/api/v1/delete_model")
  async def delete_model(req: ModelRequest):
    await submit(req.model_id, Op.DELETE_MODEL, {}, req.seq_id)
    return {"status": "ok"}

  @api.post("/api/v1/retrieve_future")
  async def retrieve_future(req: FutureRequest, accept: str = Header("")):
    if (reply := await bus.wait_for_reply(req.request_id, WAIT)) is None:
      return {"type": "try_again", "request_id": req.request_id, "queue_state": "active"}
    if reply.content_type == wire.PROTO and wire.PROTO not in accept:
      return wire.sample_json(reply.body) if req.request_id.startswith("samp-") else wire.output_json(reply.body)
    return Response(reply.body, media_type=reply.content_type)

  # ***** sampling *****

  @api.post("/api/v1/create_sampling_session")
  async def create_sampling_session(req: Request):
    data = await req.json()
    sid = await open_sampler(data.get("base_model"), data.get("model_path"), data.get("session_id", ""))
    return {"type": "create_sampling_session", "sampling_session_id": sid}

  @api.post("/api/v1/join_sampling_session")
  async def join_sampling_session(req: Request):
    sid = (await req.json()).get("sampling_session_id")
    if not await bus.get_run(sid):
      raise HTTPException(404, f"no session {sid}")
    return {"type": "join_sampling_session", "client_counter": await bus.join_sampling_session(sid)}

  @api.get("/api/v1/samplers/{sampler_id}")
  async def get_sampler(sampler_id: str):
    if not (run := await bus.get_run(sampler_id)):
      raise HTTPException(404, f"no sampler {sampler_id}")
    return {"sampler_id": sampler_id, "base_model": run["base"], "model_path": run.get("model_path")}

  @api.post("/api/v1/asample")
  async def asample(req: Request):
    data = await req.json()
    chunks = data.get("prompt", {}).get("chunks", [])
    if not all(c.get("type") == "encoded_text" for c in chunks):
      raise ValueError("only encoded_text chunks are supported")
    if not (tokens := [int(t) for c in chunks for t in c.get("tokens", [])]):
      raise ValueError("prompt is empty")
    if data.get("target_prompt_logprobs") is not None or data.get("prompt_alt_tokens_k", 0) > 0:
      raise ValueError("target_prompt_logprobs and prompt_alt_tokens are not supported")
    if (last_n := data.get("prompt_logprobs_last_n")) is not None:
      if not data.get("prompt_logprobs"):
        raise ValueError("prompt_logprobs must be true if prompt_logprobs_last_n is set")
      if not (1 <= last_n <= len(tokens) - 1):
        raise ValueError("prompt_logprobs_last_n must be between 1 and len(tokens) - 1")

    sid = data.get("sampling_session_id") or await open_sampler(data.get("base_model"), data.get("model_path"))
    num_samples, params = int(data.get("num_samples", 1)), data.get("sampling_params", {})
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
    return {"checkpoints": [public(c) for c in await registry.checkpoints_of_run(run)]}

  @api.get("/api/v1/checkpoints")
  async def list_user_checkpoints(limit: int = 100, offset: int = 0):
    found, total = await registry.page_of_checkpoints(limit, offset)
    return {
      "checkpoints": [public(c) for c in found],
      "cursor": {"offset": offset, "limit": limit, "total_count": total},
    }

  @api.delete("/api/v1/training_runs/{run}/checkpoints/{checkpoint_id:path}")
  async def delete_checkpoint(run: str, checkpoint_id: str):
    if not await registry.unregister_checkpoints(f"tinker://{run}/{checkpoint_id}"):
      raise HTTPException(404, f"no checkpoint {checkpoint_id} in {run}")

  @api.put("/api/v1/training_runs/{run}/checkpoints/{checkpoint_id:path}/ttl")
  async def set_checkpoint_ttl(run: str, checkpoint_id: str, req: TTLRequest):
    if not await registry.set_checkpoint_ttl(f"tinker://{run}/{checkpoint_id}", ttl(req.ttl_seconds)):
      raise HTTPException(404, f"no checkpoint {checkpoint_id} in {run}")

  # ***** inspection *****

  @api.get("/api/v1/runs/{run_id}")
  async def run_info(run_id: str):
    if (run := await bus.get_run(run_id)) is None:
      raise HTTPException(404, f"no model {run_id}")
    return {**run, **await bus.queued_and_in_flight(run_id)}

  @api.get("/api/v1/sessions/{session_id}")
  async def session_info(session_id: str):
    if (session := await bus.get_session(session_id)) is None:
      raise HTTPException(404, f"no session {session_id}")
    return session

  @api.get("/metrics")
  async def metrics():
    lines = []
    for (worker, run), counters in (await bus.run_counters()).items():
      for name, n in counters.items():
        name, n = (f"{name[:-3]}_seconds", n / 1000) if name.endswith("_ms") else (name, n)
        lines.append(f'tinyrl_{name}_total{{worker="{worker}",run="{run}"}} {n}')
    for node, beat in (await bus.live_nodes(details=True)).items():
      lines.append(f'tinyrl_node_loop_lag_seconds{{node="{node}"}} {beat.get("lag", 0.0)}')
    return PlainTextResponse("\n".join(lines) + "\n")

  return api


if __name__ == "__main__":
  uvicorn.run(app(), host="0.0.0.0", port=getenv("PORT", 8000))
