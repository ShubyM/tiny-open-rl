# the tinker http api. every training call becomes an entry on its run's stream, and retrieve_future replays what the trainer wrote.
from __future__ import annotations
import json, time, uuid
from typing import Any
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tinyrl import wire
from tinyrl.bus import Bus, Run
from tinyrl.helpers import getenv

CKPT, WAIT = getenv("CKPT", "/tmp/tinyrl"), getenv("WAIT", 30.0)  # the sdk gives retrieve_future 45s

class Session(BaseModel): session_id:str
class Create(BaseModel): base_model:str; session_id:str = ""; lora_config:dict[str, Any]|None = None; user_metadata:dict[str, Any]|None = None # noqa: E702
class Model(BaseModel): model_id:str
class Optim(Model): adam_params:dict[str, Any] = {}
class Save(Model): path:str|None = None; seq_id:int|None = None; sampling_session_seq_id:int|None = None # noqa: E702
class Load(Model): path:str; optimizer:bool = False # noqa: E702
class Future(BaseModel): request_id:str

def checkpoint(run:str, kind:str, name:str) -> tuple[str, str]:
  # tinker://<run>/<kind>/<name> lives at CKPT/<run>/<kind>/<name>, on a disk the trainers and samplers share
  if name.startswith("tinker://"): run, kind, name = name.removeprefix("tinker://").split("/", 2)
  if ".." in f"{run}/{kind}/{name}".split("/") or name.startswith("/"): raise ValueError(f"bad checkpoint name {name}")
  return f"{CKPT}/{run}/{kind}/{name}", f"tinker://{run}/{kind}/{name}"

def app(bus:Bus|None=None) -> FastAPI:
  api, bus = FastAPI(title="tiny-open-rl"), bus or Bus()
  for exc in (ValueError, KeyError): api.add_exception_handler(exc, lambda _, e: JSONResponse({"error": f"{type(e).__name__}: {e}"}, 400))

  async def submit(run:str, op:str, body:bytes|dict) -> dict[str, str]:
    rid = await bus.submit(run, op, body if isinstance(body, bytes) else json.dumps(body).encode())
    if rid is None: raise HTTPException(404, f"no model {run}")
    return {"request_id": rid, "model_id": run}

  @api.get("/api/v1/healthz")
  async def healthz(): return {"status": "ok"}

  @api.post("/api/v1/client/config")
  @api.post("/api/v1/client/dynamic_config")
  async def client_config(): return {}  # the sdk's defaults

  @api.get("/api/v1/get_server_capabilities")
  async def get_server_capabilities(): return {"supported_models": []}

  @api.post("/api/v1/create_session")
  async def create_session():
    await bus.session(sid:=f"sess-{uuid.uuid4().hex[:12]}")
    return {"type": "create_session", "session_id": sid}

  @api.post("/api/v1/session_heartbeat")
  async def session_heartbeat(r:Session):
    await bus.session(r.session_id)
    return {"type": "session_heartbeat"}

  @api.post("/api/v1/create_model")
  async def create_model(r:Create):
    # a lora run shares its base model's trainer unless it asks for its own. a full fine-tune always gets its own
    meta, lora, run = r.user_metadata or {}, r.lora_config or {}, str(uuid.uuid4())
    rank = 0 if meta.get("openrl.fine_tuning_type") == "full" else int(lora.get("rank", 16))
    pool = run if rank == 0 or str(meta.get("openrl.exclusive")).lower() == "true" else f"lora:{r.base_model}"
    await bus.create(Run(run, pool, r.base_model, rank, r.session_id))
    return await submit(run, "create_model", {"base_model": r.base_model, "rank": rank, "lora": lora})

  @api.post("/api/v1/forward_backward")
  async def forward_backward(req:Request):
    body = await req.body()
    if req.headers.get("content-type", "").startswith(wire.PROTO): return await submit(wire.parse(body).model_id, "forward_backward", body)
    return await submit((j:=json.loads(body))["model_id"], "forward_backward", wire.from_json(j))

  @api.post("/api/v1/forward")  # sdks before 0.25
  async def forward(req:Request): return await submit((j:=await req.json())["model_id"], "forward_backward", wire.from_json(j, forward_only=True))

  @api.post("/api/v1/optim_step")
  async def optim_step(r:Optim): return await submit(r.model_id, "optim_step", {"adam_params": r.adam_params})

  async def save(r:Save, op:str, kind:str):
    seq = next(s for s in (r.sampling_session_seq_id, r.seq_id, time.time_ns()) if s is not None)
    path, uri = checkpoint(r.model_id, kind, r.path or f"step-{seq}")
    return await submit(r.model_id, op, {"path": path, "uri": uri})

  @api.post("/api/v1/save_weights")  # save_state, so the optimizer comes too
  async def save_weights(r:Save): return await save(r, "save_weights", "weights")

  @api.post("/api/v1/save_weights_for_sampler")
  async def save_weights_for_sampler(r:Save): return await save(r, "save_weights_for_sampler", "sampler_weights")

  @api.post("/api/v1/load_weights")
  async def load_weights(r:Load):
    path, uri = checkpoint(r.model_id, "weights", r.path)
    return await submit(r.model_id, "load_weights", {"path": path, "uri": uri, "optimizer": r.optimizer})

  @api.post("/api/v1/delete_model")
  async def delete_model(r:Model):
    await submit(r.model_id, "delete_model", {})  # behind the run's queued work, which still finishes
    return {"status": "ok"}

  @api.post("/api/v1/retrieve_future")
  async def retrieve_future(r:Future, accept:str=Header("")):
    if (rep:=await bus.result(r.request_id, WAIT)) is None: return {"type": "try_again", "request_id": r.request_id, "queue_state": "active"}
    return wire.output_json(rep.body) if rep.ct == wire.PROTO and wire.PROTO not in accept else Response(rep.body, media_type=rep.ct)

  @api.post("/api/v1/get_info")
  async def get_info(r:Model):
    if (run:=await bus.run(r.model_id)) is None: raise HTTPException(404, f"no model {r.model_id}")
    # the sdk reads lora shaped metadata, even for a full fine-tune
    return {"type": "get_info", "model_id": r.model_id, "model_name": run["base"], "is_lora": True, "lora_rank": run["rank"] or 16,
            "model_data": {"model_name": run["base"], "tokenizer_id": run["base"]}}

  @api.get("/api/v1/runs/{run}")
  async def run_info(run:str):
    if (rec:=await bus.run(run)) is None: raise HTTPException(404, f"no model {run}")
    return rec | await bus.queue(run)

  @api.get("/api/v1/sessions/{sid}")
  async def session_info(sid:str):
    if (s:=await bus.session_info(sid)) is None: raise HTTPException(404, f"no session {sid}")
    return s

  return api

if __name__ == "__main__": uvicorn.run(app(), host="0.0.0.0", port=getenv("PORT", 8000))
