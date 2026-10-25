"""Coordinator-owned Redis request processing. Workers never mutate Redis directly.

Short WATCH/MULTI/EXEC transactions fence every state change with the coordinator lease.
Redis holds bounded metadata or small inline bytes; bulk payloads live in PayloadStore.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum

import redis.asyncio as redis
from redis.exceptions import RedisError, ResponseError, WatchError

from tinyrl.helpers import getenv, is_hex
from tinyrl.payloads import INLINE_BYTES

log = logging.getLogger(__name__)

LIVE_SECONDS = 120
REPLY_TTL = getenv("REPLY_TTL", 3600)
SESSION_TTL = getenv("SESSION_TTL", 7 * 24 * 3600)
WORKER_TTL = getenv("WORKER_TTL", 30.0)
COORDINATOR_TTL = getenv("COORDINATOR_TTL", 10.0)
MAX_PENDING_COMMANDS = getenv("MAX_PENDING_COMMANDS", 1024)
MAX_COMMAND_BYTES = getenv("MAX_COMMAND_BYTES", 64 << 20)
MAX_REPLY_BYTES = getenv("MAX_REPLY_BYTES", 64 << 20)
MAX_PENDING_BYTES = getenv("MAX_PENDING_BYTES", 256 << 20)


class CoordinatorUnavailable(RuntimeError):
  """No mutation was acknowledged; retry the same request identity."""


class Op(StrEnum):
  CREATE_MODEL = "create_model"
  FORWARD_BACKWARD = "forward_backward"
  OPTIM_STEP = "optim_step"
  SAVE_WEIGHTS = "save_weights"
  SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"
  LOAD_WEIGHTS = "load_weights"
  DELETE_MODEL = "delete_model"
  SNAPSHOT = "snapshot"
  RESTORE = "restore"
  ACTIVATE = "activate"
  DETACH = "detach"


CONTROL = {Op.SNAPSHOT, Op.RESTORE, Op.ACTIVATE, Op.DETACH}


@dataclass(frozen=True)
class Run:
  id: str
  pool: str
  base: str
  rank: int
  session: str = ""


@dataclass(frozen=True)
class Cmd:
  run: str
  entry_id: str
  op: Op
  body: bytes
  worker: str = ""
  generation: int = 0
  token: str = ""
  body_ref: str = ""
  body_size: int = 0

  @property
  def request_id(self) -> str: return f"{self.run}/{self.entry_id}"


@dataclass(frozen=True)
class Reply:
  ok: bool
  content_type: str
  body: bytes
  artifact: str = ""
  recover: bool = False
  body_ref: str = ""
  body_size: int = 0


def b64(data: bytes) -> str: return base64.b64encode(data).decode()
def unb64(text: str) -> bytes: return base64.b64decode(text, validate=True)
def encode(value) -> bytes: return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
def digest(value) -> str: return hashlib.sha256(encode(value)).hexdigest()
def error(msg: str, category: str = "server") -> Reply:
  return Reply(False, "application/json", encode({"error": msg, "category": category}))
def pack(value: Cmd | Reply) -> dict: return {**asdict(value), "body": "" if value.body_ref else b64(value.body)}
def unpack_cmd(d: dict) -> Cmd: return Cmd(**{**d, "op": Op(d["op"]), "body": unb64(d["body"])})
def unpack_reply(d: dict) -> Reply: return Reply(**{**d, "body": unb64(d["body"])})


def payload(body: bytes, ref: str = "", size: int = 0) -> tuple[str, int]:
  """Canonical content identity is independent of inline vs blob transport."""
  if ref:
    if not is_hex(ref, 64) or type(size) is not int or size < 0: raise ValueError("invalid payload reference or size")
    if body and (len(body) != size or hashlib.sha256(body).hexdigest() != ref):
      raise ValueError("payload content does not match its reference")
    return ref, size
  if len(body) > INLINE_BYTES: raise ValueError("large payload must be stored before admission")
  if size not in (0, len(body)): raise ValueError("payload size does not match inline bytes")
  return hashlib.sha256(body).hexdigest(), len(body)


class _Transaction:
  def __init__(self, pipe):
    self.pipe, self.records, self.dirty, self.ops = pipe, {}, {}, []

  async def record(self, key: str) -> dict | None:
    if key not in self.records:
      raw = await self.pipe.get(key)
      self.records[key] = json.loads(raw) if raw else None
    return self.records[key]

  def save(self, key: str, record: dict, *, ex: int | None = None) -> None:
    self.records[key] = record
    self.dirty[key] = ex

  def write(self, method: str, *args, **kwargs) -> None: self.ops.append((method, args, kwargs))

  def apply(self) -> None:
    for key, ex in self.dirty.items(): self.pipe.set(key, encode(self.records[key]), ex=ex)
    for method, args, kwargs in self.ops: getattr(self.pipe, method)(*args, **kwargs)


class Bus:
  def __init__(self, url: str | None = None, namespace: str | None = None, *, lease_seconds: float | None = None):
    self.url = url or getenv("REDIS_URL", "redis://localhost:6379/0")
    self.namespace = namespace or getenv("REDIS_NAMESPACE", "tinyrl")
    if not self.namespace or "{" in self.namespace or "}" in self.namespace:
      raise ValueError("Redis namespace cannot be empty or contain braces")
    self.prefix = f"tinyrl:{{{self.namespace}}}:"
    self.r = redis.Redis.from_url(self.url, socket_connect_timeout=5, socket_timeout=35)
    self.lease_seconds = COORDINATOR_TTL if lease_seconds is None else lease_seconds
    if self.lease_seconds <= 0: raise ValueError("coordinator lease must be positive")
    self.token, self.lock = uuid.uuid4().hex, asyncio.Lock()
    self.closed = self.lost = self.acquired = False
    self._started = time.time()
    self._wake = asyncio.Event()

  def key(self, kind: str, identity: str = "") -> str: return f"{self.prefix}{kind}:{identity}"

  def _notify(self) -> None:
    self._wake.set()
    self._wake = asyncio.Event()

  async def _wait(self, timeout: float) -> None:
    if timeout > 0:
      with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(self._wake.wait(), min(timeout, 0.5))

  async def start(self) -> None:
    async with self.lock:
      if self.closed or self.lost: raise CoordinatorUnavailable("coordinator no longer owns this namespace")
      if not self.acquired:
        try:
          await self.r.set(self.key("coordinator"), self.token, nx=True, px=math.ceil(self.lease_seconds * 1000))
          if await self.r.get(self.key("coordinator")) != self.token.encode():
            raise CoordinatorUnavailable("another coordinator owns this namespace")
        except RedisError as exc:
          raise CoordinatorUnavailable("Redis unavailable while acquiring coordinator ownership") from exc
        self.acquired, self._started = True, time.time()

  async def check_owner(self) -> bool:
    async with self._transaction(): return True

  @contextlib.asynccontextmanager
  async def _transaction(self):
    await self.start()
    async with self.lock, self.r.pipeline(transaction=True) as pipe:
      try:
        await pipe.watch(self.key("coordinator"))
        if self.lost or await pipe.get(self.key("coordinator")) != self.token.encode():
          self.lost = True
          raise CoordinatorUnavailable("coordinator lease was lost; restart this process")
        tx = _Transaction(pipe)
        yield tx
        pipe.multi()
        pipe.pexpire(self.key("coordinator"), math.ceil(self.lease_seconds * 1000))
        tx.apply()
        await pipe.execute()
        if tx.dirty or tx.ops: self._notify()
      except WatchError as exc:
        self.lost = True
        raise CoordinatorUnavailable("coordinator lease changed during commit") from exc
      except ResponseError as exc:
        self.lost = True
        log.error("Redis transaction failed in %s; repair this namespace before restarting: %s", self.prefix, exc)
        raise CoordinatorUnavailable("Redis transaction failed; inspect and repair before restart") from exc
      except RedisError as exc:
        raise CoordinatorUnavailable("Redis command outcome unavailable; retry the same identity") from exc

  async def close(self) -> None:
    async with self.lock:
      if self.closed: return
      self.closed = True
      try:
        async with self.r.pipeline(transaction=True) as pipe:
          await pipe.watch(self.key("coordinator"))
          if await pipe.get(self.key("coordinator")) == self.token.encode():
            pipe.multi()
            pipe.delete(self.key("coordinator"))
            await pipe.execute()
      except RedisError: pass
      finally: await self.r.aclose()

  async def _record(self, kind: str, identity: str) -> dict | None:
    raw = await self.r.get(self.key(kind, identity))
    return json.loads(raw) if raw else None

  def _cmd(self, c: dict) -> Cmd:
    body = unb64(c["body"])
    return Cmd(c["run"], c["id"], Op(c["op"]), body, c["worker"], c["generation"], c["token"],
               c["body_ref"], c["body_size"] if c["body_ref"] else len(body))

  def _ready(self, tx: _Transaction, r: dict) -> None:
    if r["phase"] == "active" and r["pending_count"] and not r["inflight"] and not r["ready"]:
      tx.write("rpush", self.key("ready", r["pool"]), r["id"])
      r["ready"] = True
    tx.save(self.key("run", r["id"]), r)

  def _unready(self, tx: _Transaction, r: dict) -> None:
    if r["ready"]:
      tx.write("lrem", self.key("ready", r["pool"]), 0, r["id"])
      r["ready"] = False
    tx.save(self.key("run", r["id"]), r)

  def _insert(self, tx: _Transaction, r: dict, op: Op, body: bytes, lane: str = "user", seq: int | None = None,
              idem: str | None = None, *, body_ref: str = "", body_size: int = 0, **fields) -> dict:
    cid = uuid.uuid4().hex
    content_hash, size = payload(body, body_ref, body_size)
    c = {
      "id": cid, "run": r["id"], "op": str(op), "body": "" if body_ref else b64(body),
      "body_ref": body_ref, "body_size": size, "lane": lane, "seq": seq, "idem": idem,
      "digest": digest([str(op), content_hash, size]), "status": "queued", "worker": "", "pool": "",
      "generation": 0, "token": "", "move_id": "", "created": time.time(), "claimed": None,
      "completed": None, "outcome": "", **fields,
    }
    tx.save(self.key("cmd", cid), c)
    tx.write("sadd", self.key("pending", r["id"]), cid)
    if seq is not None: tx.write("hset", self.key("sequence", r["id"]), str(seq), cid)
    if idem: tx.write("hset", self.key("identity", r["id"]), idem, cid)
    if lane == "control": tx.write("rpush", self.key("controls", c["pool"]), cid)
    else:
      r["pending_count"] += 1
      r["pending_bytes"] += size
      self._ready(tx, r)
    return c

  async def _control(self, tx: _Transaction, r: dict, op: Op, body: dict, *, worker: str = "", pool: str = "",
                     generation: int = 0) -> dict:
    if op == Op.DETACH:
      ids = [i.decode() for i in await tx.pipe.smembers(self.key("pending", r["id"]))]
      seen = [await tx.record(self.key("cmd", i)) for i in ids] + list(tx.records.values())
      for c in seen:
        if c and c.get("run") == r["id"] and c.get("op") == op and c["status"] in ("queued", "executing"):
          if (c["worker"], c["generation"]) == (worker, generation): return c
    move_id = "" if op == Op.DETACH else r["move_id"]
    return self._insert(tx, r, op, encode(body), "control",
                        worker=worker, pool=pool, generation=generation, move_id=move_id)

  async def session(self, session_id: str) -> None:
    async with self._transaction() as tx:
      s = await tx.record(self.key("session", session_id)) or {"created": time.time()}
      tx.save(self.key("session", session_id), {**s, "seen": time.time()}, ex=SESSION_TTL)

  async def session_info(self, session_id: str) -> dict | None:
    s = await self._record("session", session_id)
    if not s: return None
    runs = sorted(i.decode() for i in await self.r.smembers(self.key("session_runs", session_id)))
    return {"session_id": session_id, **s, "live": time.time() - s["seen"] < LIVE_SECONDS, "runs": runs}

  async def create(self, run: Run, *, body: bytes | dict | None = None, idempotency_key: str | None = None) -> dict:
    if isinstance(body, dict): body = encode(body)
    payload(body or b"")
    spec = {**asdict(run), "lora": json.loads(body).get("lora", {}) if body else {}}
    creation_digest = hashlib.sha256(encode(spec) + (body or b"")).hexdigest()
    async with self._transaction() as tx:
      prior = await tx.pipe.hget(self.key("creations"), idempotency_key) if idempotency_key else None
      r = await tx.record(self.key("run", prior.decode() if prior else run.id))
      if r:
        if r["creation_digest"] != creation_digest:
          raise ValueError("creation identity was already used with different model configuration")
      else:
        r = {
          "id": run.id, "spec": spec, "created": time.time(), "creation_digest": creation_digest,
          "pool": run.pool, "owner": "", "generation": 1, "phase": "active", "initialized": False,
          "cursor": 0, "next_seq": 0, "closing": False, "deletion": "", "steps": 0, "failed": 0,
          "snapshot": "", "move_id": "", "target_pool": "", "residents": {}, "bootstrap": "",
          "inflight": "", "ready": False, "pending_count": 0, "pending_bytes": 0,
        }
        if body is not None: r["bootstrap"] = self._insert(tx, r, Op.CREATE_MODEL, body, "bootstrap", -1)["id"]
        tx.save(self.key("run", run.id), r)
        tx.write("sadd", self.key("pool_runs", run.pool), run.id)
        if run.session: tx.write("sadd", self.key("session_runs", run.session), run.id)
        if idempotency_key: tx.write("hset", self.key("creations"), idempotency_key, run.id)
      info = self._info(r)
      if r["bootstrap"]: info["request_id"] = f"{r['id']}/{r['bootstrap']}"
      return info

  def _info(self, r: dict | None) -> dict | None:
    if not r: return None
    fields = "created pool owner generation phase steps failed cursor snapshot target_pool closing".split()
    return {**r["spec"], **{k: r[k] for k in fields}}

  async def run(self, run_id: str) -> dict | None: return self._info(await self._record("run", run_id))

  async def runs(self, pool: str) -> list[str]:
    return sorted(i.decode() for i in await self.r.smembers(self.key("pool_runs", pool)))

  async def submit(self, run_id: str, op: Op | str, body: bytes, *, seq_id: int | None = None,
                   idempotency_key: str | None = None, body_ref: str = "", body_size: int = 0) -> str | None:
    op = Op(op)
    content_hash, size = payload(body, body_ref, body_size)
    if size > MAX_COMMAND_BYTES: raise ValueError("command exceeds MAX_COMMAND_BYTES")
    if op in CONTROL or (body_ref and op != Op.FORWARD_BACKWARD):
      raise ValueError("control and routing metadata must be small inline requests")
    if seq_id is not None and (type(seq_id) is not int or not 0 <= seq_id < (1 << 63) - 1):
      raise ValueError("seq_id must be a nonnegative 63-bit integer")
    lane, seq = ("bootstrap", -1) if op == Op.CREATE_MODEL else ("user", seq_id)
    cmd_digest = digest([str(op), content_hash, size])
    async with self._transaction() as tx:
      r = await tx.record(self.key("run", run_id))
      if not r: return None
      cid = await tx.pipe.hget(self.key("identity", run_id), idempotency_key) if idempotency_key else None
      if not cid and seq is not None: cid = await tx.pipe.hget(self.key("sequence", run_id), str(seq))
      if cid:
        c = await tx.record(self.key("cmd", cid.decode()))
        if c["digest"] != cmd_digest or (seq is not None and c["seq"] != seq):
          raise ValueError("same command identity has different content")
        if idempotency_key and c["idem"] != idempotency_key:
          if c["idem"]: raise ValueError("sequence already has a different idempotency key")
          c["idem"] = idempotency_key
          tx.save(self.key("cmd", c["id"]), c)
          tx.write("hset", self.key("identity", run_id), idempotency_key, c["id"])
        return f"{run_id}/{c['id']}"
      if r["phase"] == "deleted": return None
      if r["closing"]:
        deletion = await tx.record(self.key("cmd", r["deletion"]))
        if op == Op.DELETE_MODEL: return f"{run_id}/{deletion['id']}"
        if seq is None or seq >= deletion["seq"]: raise ValueError("run is closing")
      if r["phase"] == "recovery" and op != Op.DELETE_MODEL:
        raise ValueError("run requires explicit checkpoint recovery")
      if lane == "user":
        seq = r["next_seq"] if seq is None else seq
        if seq < r["cursor"]: raise ValueError("sequence precedes the completed or failed command boundary")
      if op != Op.DELETE_MODEL or r["phase"] not in ("parked", "recovery"):
        if r["pending_count"] >= MAX_PENDING_COMMANDS or r["pending_bytes"] + size > MAX_PENDING_BYTES:
          raise ValueError("run command queue capacity exceeded")
        if lane == "user" and seq > r["cursor"] and (
          r["pending_count"] >= MAX_PENDING_COMMANDS - 1
          or r["pending_bytes"] + size > MAX_PENDING_BYTES - MAX_COMMAND_BYTES
        ):
          raise ValueError("run command queue capacity reserved for its next sequence")
      c = self._insert(tx, r, op, body, lane, seq, idempotency_key, body_ref=body_ref, body_size=size)
      if lane == "user": r["next_seq"] = max(r["next_seq"], seq + 1)
      if op == Op.DELETE_MODEL:
        r["closing"], r["deletion"] = True, c["id"]
        if r["phase"] in ("parked", "recovery"):
          await self._delete(tx, r)
          self._complete(tx, r, c, Reply(True, "application/json", encode({"type": op, "model_id": run_id})))
      tx.save(self.key("run", run_id), r)
      return f"{run_id}/{c['id']}"

  async def register(self, worker: str, pool: str) -> dict:
    if not worker or not pool: raise ValueError("worker incarnation and pool are required")
    async with self._transaction() as tx:
      w = await tx.record(self.key("worker", worker))
      if w:
        if w["pool"] != pool: raise ValueError("worker incarnation cannot change pools")
        if not w["live"]: return {"registered": False, "reason": "worker incarnation was lost"}
      else:
        cur = await tx.pipe.get(self.key("pool_owner", pool))
        if cur: return {"registered": False, "reason": "pool already has a worker", "owner": cur.decode()}
      now = time.time()
      tx.save(self.key("worker", worker), {"pool": pool, "live": True, "seen": now})
      tx.write("set", self.key("pool_owner", pool), worker)
      tx.write("zadd", self.key("workers"), {worker: now})
      return {"registered": True, "worker": worker, "pool": pool}

  async def heartbeat(self, worker: str) -> bool:
    async with self._transaction() as tx:
      w = await tx.record(self.key("worker", worker))
      if not w or not w["live"]: return False
      w["seen"] = time.time()
      tx.save(self.key("worker", worker), w)
      tx.write("zadd", self.key("workers"), {worker: w["seen"]})
      return True

  async def take(self, pool: str, worker: str, count: int = 16, block_ms: int = 1000) -> list[Cmd]:
    deadline, count = time.monotonic() + max(0, block_ms) / 1000, max(1, min(int(count), 64))
    while True:
      async with self._transaction() as tx:
        w = await tx.record(self.key("worker", worker))
        if not w or not w["live"] or w["pool"] != pool: return []
        eids = await tx.pipe.smembers(self.key("executing", worker))
        active = [await tx.record(self.key("cmd", i.decode())) for i in eids]
        if active: return [self._cmd(c) for c in sorted(active, key=lambda c: c["created"])[:count]]
        taken, busy = [], set()
        for cid in await tx.pipe.lrange(self.key("controls", pool), 0, -1):
          c = await tx.record(self.key("cmd", cid.decode()))
          r = await tx.record(self.key("run", c["run"]))
          if c["worker"] and c["worker"] != worker: continue
          if c["op"] != Op.DETACH and c["move_id"] != r["move_id"]:
            self._complete(tx, r, c, error("obsolete move"), "cancelled")
            continue
          if c["run"] in busy or (c["op"] != Op.DETACH and r["inflight"]): continue
          taken.append(await self._claim(tx, r, c, worker, c["generation"]))
          busy.add(c["run"])
          if len(taken) == count: break
        if len(taken) < count:
          for rid in await tx.pipe.lrange(self.key("ready", pool), 0, -1):
            r = await tx.record(self.key("run", rid.decode()))
            if r["id"] in busy or r["phase"] != "active" or r["inflight"] or r["owner"] not in ("", worker): continue
            c = await tx.record(self.key("cmd", r["bootstrap"])) if r["bootstrap"] else None
            if not c or c["status"] != "queued":
              cid = await tx.pipe.hget(self.key("sequence", r["id"]), str(r["cursor"]))
              c = await tx.record(self.key("cmd", cid.decode())) if cid else None
            if c and c["status"] == "queued":
              r["owner"] = worker
              r["residents"][worker] = [r["pool"], r["generation"]]
              taken.append(await self._claim(tx, r, c, worker, r["generation"]))
            if len(taken) == count: break
        if taken: return taken
      remaining = deadline - time.monotonic()
      if remaining <= 0: return []
      await self._wait(remaining)

  async def _claim(self, tx: _Transaction, r: dict, c: dict, worker: str, generation: int) -> Cmd:
    if c["op"] == Op.SNAPSHOT: c["body"] = b64(encode({"boundary": r["cursor"]}))
    elif c["op"] == Op.LOAD_WEIGHTS:
      args = json.loads(unb64(c["body"]))
      ref = await tx.pipe.hget(self.key("artifacts"), args["uri"])
      c["body"] = b64(encode({**args, "ref": ref.decode() if ref else ""}))
    c.update(status="executing", worker=worker, generation=generation, token=uuid.uuid4().hex, claimed=time.time())
    tx.save(self.key("cmd", c["id"]), c)
    tx.write("sadd", self.key("executing", worker), c["id"])
    if c["lane"] == "control": tx.write("lrem", self.key("controls", c["pool"]), 0, c["id"])
    if c["op"] != Op.DETACH:
      r["inflight"] = c["id"]
      self._unready(tx, r)
    return self._cmd(c)

  def _complete(self, tx: _Transaction, r: dict, c: dict, reply: Reply, status: str = "done") -> None:
    if c["status"] not in ("queued", "executing"): return
    if c["lane"] == "control": tx.write("lrem", self.key("controls", c["pool"]), 0, c["id"])
    else:
      r["pending_count"] -= 1
      r["pending_bytes"] -= c["body_size"]
    tx.write("srem", self.key("pending", r["id"]), c["id"])
    if c["worker"]: tx.write("srem", self.key("executing", c["worker"]), c["id"])
    if r["inflight"] == c["id"]: r["inflight"] = ""
    content_hash, size = payload(reply.body, reply.body_ref, reply.body_size)
    outcome = digest([reply.ok, reply.content_type, reply.artifact, reply.recover, content_hash, size])
    c.update(status=status, completed=time.time(), body="", body_ref="", body_size=0, outcome=outcome)
    tx.save(self.key("cmd", c["id"]), c)
    tx.save(self.key("run", r["id"]), r)
    tx.write("set", self.key("reply", c["id"]), encode(pack(reply)), px=max(1, math.ceil(REPLY_TTL * 1000)))

  async def finish(self, cmd: Cmd, reply: Reply) -> str:
    content_hash, size = payload(reply.body, reply.body_ref, reply.body_size)
    if size > MAX_REPLY_BYTES: raise ValueError("result exceeds MAX_REPLY_BYTES")
    outcome = digest([reply.ok, reply.content_type, reply.artifact, reply.recover, content_hash, size])
    async with self._transaction() as tx:
      c = await tx.record(self.key("cmd", cmd.entry_id))
      claim = (cmd.worker, cmd.generation, cmd.token)
      if not c or c["run"] != cmd.run or (c["worker"], c["generation"], c["token"]) != claim: return "stale"
      if c["status"] == "done":
        if c["outcome"] != outcome: raise ValueError("conflicting duplicate completion")
        return "already_committed"
      if c["status"] != "executing": return "stale"
      r = await tx.record(self.key("run", cmd.run))
      if c["lane"] == "control":
        if c["op"] != Op.DETACH and (c["move_id"] != r["move_id"] or r["phase"] not in ("moving", "activating")):
          return "stale"
      elif (r["owner"], r["generation"]) != (cmd.worker, cmd.generation) or r["phase"] in ("recovery", "deleted"):
        return "stale"
      if reply.ok and c["op"] in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER, Op.SNAPSHOT):
        if not is_hex(reply.artifact, 32):
          raise ValueError("successful snapshot completion requires an immutable artifact ref")
        if c["op"] != Op.SNAPSHOT:
          tx.write("hset", self.key("artifacts"), json.loads(unb64(c["body"]))["uri"], reply.artifact)
      self._complete(tx, r, c, reply)
      if c["lane"] == "control": await self._finish_control(tx, r, c, reply)
      else:
        if c["lane"] == "user": r["cursor"] = c["seq"] + 1
        r["failed"] += not reply.ok
        r["steps"] += reply.ok and c["op"] == Op.OPTIM_STEP
        if c["op"] == Op.CREATE_MODEL and reply.ok: r["initialized"] = True
        if c["op"] == Op.DELETE_MODEL and not reply.ok: r["closing"] = False
        if reply.recover or (c["op"] == Op.CREATE_MODEL and not reply.ok):
          await self._fail_run(tx, r, "run requires checkpoint recovery after a failed mutation")
        elif c["op"] == Op.DELETE_MODEL and reply.ok:
          r["residents"].pop(c["worker"], None)
          await self._delete(tx, r)
      self._ready(tx, r)
      return "committed"

  async def _terminal(self, tx: _Transaction, r: dict, message: str) -> None:
    for cid in await tx.pipe.smembers(self.key("pending", r["id"])):
      c = await tx.record(self.key("cmd", cid.decode()))
      if c["op"] != Op.DETACH:
        self._complete(tx, r, c, error(message), "unknown" if c["status"] == "executing" else "cancelled")

  async def _fail_run(self, tx: _Transaction, r: dict, message: str) -> None:
    await self._terminal(tx, r, message)
    self._unready(tx, r)
    r.update(phase="recovery", owner="", generation=r["generation"] + 1,
             cursor=r["next_seq"], move_id="", closing=False)
    tx.save(self.key("run", r["id"]), r)

  async def _detach_one(self, tx: _Transaction, r: dict, worker: str, pool: str, generation: int) -> bool:
    w = await tx.record(self.key("worker", worker)) if worker else None
    if not w or not w["live"]: return False
    await self._control(tx, r, Op.DETACH, {}, worker=worker, pool=pool, generation=generation)
    return True

  async def _detach_residents(self, tx: _Transaction, r: dict, *, keep: str = "") -> None:
    for worker, (pool, gen) in list(r["residents"].items()):
      if worker != keep and not await self._detach_one(tx, r, worker, pool, gen): del r["residents"][worker]

  def _begin_move(self, tx: _Transaction, r: dict, target: str) -> None:
    self._unready(tx, r)
    r.update(phase="moving", move_id=uuid.uuid4().hex, target_pool=target)
    tx.save(self.key("run", r["id"]), r)

  async def _restore_control(self, tx: _Transaction, r: dict, ref: str, *, recovery: bool = False) -> None:
    body = {"ref": ref, "boundary": r["cursor"], "spec": r["spec"], "recovery": recovery}
    await self._control(tx, r, Op.RESTORE, body, pool=r["target_pool"], generation=r["generation"] + 1)

  async def request_move(self, run: str, target_pool: str) -> dict:
    async with self._transaction() as tx:
      r = await tx.record(self.key("run", run))
      if not r: raise KeyError(run)
      if r["phase"] in ("moving", "activating") and r["target_pool"] == target_pool: return self._info(r)
      if r["closing"] or r["phase"] not in ("active", "parked"): raise ValueError("run is not available for movement")
      if r["pool"] == target_pool: return self._info(r)
      if r["phase"] == "active" and not r["initialized"]:
        raise ValueError("run must finish model creation before movement")
      parked = r["phase"] == "parked"
      self._begin_move(tx, r, target_pool)
      if parked: await self._restore_control(tx, r, r["snapshot"])
      else: await self._control(tx, r, Op.SNAPSHOT, {}, worker=r["owner"], pool=r["pool"], generation=r["generation"])
      return self._info(r)

  async def _finish_control(self, tx: _Transaction, r: dict, c: dict, reply: Reply) -> None:
    op = Op(c["op"])
    if op == Op.DETACH:
      if reply.ok and r["residents"].get(c["worker"]) == [c["pool"], c["generation"]]: del r["residents"][c["worker"]]
      return
    if not reply.ok:
      if op == Op.ACTIVATE or reply.recover or not r["owner"]:
        await self._fail_run(tx, r, "checkpoint restore or activation failed; explicit recovery required")
      else:
        if op == Op.RESTORE: await self._detach_one(tx, r, c["worker"], r["target_pool"], c["generation"])
        r.update(phase="active", move_id="", target_pool="")
      return
    if op == Op.SNAPSHOT:
      r["snapshot"] = reply.artifact
      if not r["target_pool"]:
        tx.write("srem", self.key("pool_runs", r["pool"]), r["id"])
        r.update(phase="parked", pool="", owner="", generation=r["generation"] + 1, move_id="")
        await self._detach_residents(tx, r)
      else: await self._restore_control(tx, r, reply.artifact)
    elif op == Op.RESTORE:
      if r["pool"]: tx.write("srem", self.key("pool_runs", r["pool"]), r["id"])
      r.update(owner=c["worker"], pool=r["target_pool"], generation=c["generation"], phase="activating")
      r["residents"][c["worker"]] = [r["pool"], c["generation"]]
      tx.write("sadd", self.key("pool_runs", r["pool"]), r["id"])
      await self._control(tx, r, Op.ACTIVATE, {"boundary": r["cursor"]},
                          worker=c["worker"], pool=r["pool"], generation=c["generation"])
    elif op == Op.ACTIVATE:
      r.update(phase="active", initialized=True, move_id="")
      await self._detach_residents(tx, r, keep=c["worker"])

  async def recover(self, run: str, checkpoint_ref: str, target_pool: str | None = None) -> dict:
    if not is_hex(checkpoint_ref, 32): raise ValueError("recovery requires a durable checkpoint ref")
    async with self._transaction() as tx:
      r = await tx.record(self.key("run", run))
      if not r: raise KeyError(run)
      if r["phase"] != "recovery": raise ValueError("run is not awaiting recovery")
      target = target_pool or r["pool"] or r["spec"]["pool"]
      if not target: raise ValueError("recovery requires a target pool")
      await self._detach_residents(tx, r)
      self._begin_move(tx, r, target)
      r.update(snapshot=checkpoint_ref, closing=False)
      await self._restore_control(tx, r, checkpoint_ref, recovery=True)
      return self._info(r)

  async def ckpt(self, uri: str) -> str | None:
    ref = await self.r.hget(self.key("artifacts"), uri)
    return ref.decode() if ref else None

  async def _lose_worker(self, tx: _Transaction, worker: str) -> int:
    w = await tx.record(self.key("worker", worker))
    if not w or not w["live"]: return 0
    w["live"] = False
    tx.save(self.key("worker", worker), w)
    tx.write("zrem", self.key("workers"), worker)
    tx.write("delete", self.key("pool_owner", w["pool"]))
    affected = 0
    for rid in await tx.pipe.smembers(self.key("pool_runs", w["pool"])):
      r = await tx.record(self.key("run", rid.decode()))
      r["residents"].pop(worker, None)
      tx.save(self.key("run", r["id"]), r)
      if r["owner"] == worker and r["phase"] not in ("recovery", "deleted", "parked"):
        await self._fail_run(tx, r, "worker was lost; execution outcome is uncertain, recover from a checkpoint")
        affected += 1
    control_ids = {*(await tx.pipe.smembers(self.key("executing", worker))),
                   *(await tx.pipe.lrange(self.key("controls", w["pool"]), 0, -1))}
    for cid in control_ids:
      c = await tx.record(self.key("cmd", cid.decode()))
      if c["lane"] != "control" or c["worker"] != worker or c["status"] not in ("queued", "executing"): continue
      r = await tx.record(self.key("run", c["run"]))
      r["residents"].pop(worker, None)
      self._complete(tx, r, c, error("worker was lost"), "cancelled")
      if c["op"] == Op.RESTORE and r["phase"] == "moving" and r["move_id"] == c["move_id"]:
        if r["owner"]:
          r.update(phase="active", move_id="", target_pool="")
          self._ready(tx, r)
        else: await self._fail_run(tx, r, "restore worker was lost")
    return affected

  async def worker_lost(self, worker: str) -> int:
    async with self._transaction() as tx: return await self._lose_worker(tx, worker)

  async def lose_worker(self, run: str, owner: str, generation: int) -> bool:
    async with self._transaction() as tx:
      r = await tx.record(self.key("run", run))
      if not r or (r["owner"], r["generation"]) != (owner, generation): return False
      await self._lose_worker(tx, owner)
      return True

  async def expire_workers(self) -> int:
    async with self._transaction() as tx:
      if time.time() - self._started < WORKER_TTL: return 0
      stale = await tx.pipe.zrangebyscore(self.key("workers"), "-inf", time.time() - WORKER_TTL)
      return sum([await self._lose_worker(tx, w.decode()) for w in stale])

  async def result(self, request_id: str, timeout: float) -> Reply | None:
    try: run, cid = request_id.rsplit("/", 1)
    except ValueError: return error("unknown future", "user")
    deadline = time.monotonic() + max(0, timeout)
    while True:
      c = await self._record("cmd", cid)
      if not c or c["run"] != run: return error("unknown future", "user")
      if c["completed"] is not None:
        raw = await self.r.get(self.key("reply", cid))
        if raw is None or time.time() - c["completed"] >= REPLY_TTL: return error("future result expired", "user")
        return unpack_reply(json.loads(raw))
      remaining = deadline - time.monotonic()
      if remaining <= 0: return None
      await self._wait(remaining)

  async def queue(self, run_id: str) -> dict:
    r = await self._record("run", run_id)
    if not r or r["phase"] == "deleted": return {"live": False}
    records = [await self._record("cmd", i.decode()) for i in await self.r.smembers(self.key("pending", run_id))]
    inflight = [
      {"request_id": f"{run_id}/{c['id']}", "trainer": c["worker"], "seconds": max(0, time.time() - c["claimed"]),
       "deliveries": 1, "generation": c["generation"]}
      for c in records if c["status"] == "executing"
    ]
    queued = sum(c["status"] == "queued" for c in records)
    return {"live": True, "queued": queued, "in_flight": inflight, "phase": r["phase"]}

  async def _delete(self, tx: _Transaction, r: dict) -> None:
    await self._terminal(tx, r, "run was deleted")
    self._unready(tx, r)
    await self._detach_residents(tx, r)
    if r["pool"]: tx.write("srem", self.key("pool_runs", r["pool"]), r["id"])
    r.update(phase="deleted", owner="", closing=True, generation=r["generation"] + 1, move_id="")
    tx.save(self.key("run", r["id"]), r)

  async def drop(self, run_id: str, pool: str = "") -> None:
    async with self._transaction() as tx:
      r = await tx.record(self.key("run", run_id))
      if r: await self._delete(tx, r)
