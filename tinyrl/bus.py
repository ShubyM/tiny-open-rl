"""Coordinator-owned Redis request processing. Workers never mutate Redis directly.

Short WATCH/MULTI/EXEC transactions fence every state change with the coordinator
lease. Redis holds bounded request metadata or small inline bytes; bulk bodies
are immutable objects referenced by content hash. GPU work and Store transfers
always happen outside these transactions. Unknown GPU effects are never replayed.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum

import redis.asyncio as redis
from redis.exceptions import RedisError, ResponseError, WatchError

from tinyrl.helpers import getenv
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
  def request_id(self):
    return f"{self.run}/{self.entry_id}"


@dataclass(frozen=True)
class Reply:
  ok: bool
  content_type: str
  body: bytes
  artifact: str = ""
  recover: bool = False
  body_ref: str = ""
  body_size: int = 0


def encode(value):
  return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def error(message, category="server"):
  return Reply(False, "application/json", encode({"error": message, "category": category}))


def payload(body, ref="", size=0):
  """Canonical content identity is independent of inline/object transport."""
  if ref:
    if not re.fullmatch(r"[0-9a-f]{64}", ref) or not isinstance(size, int) or isinstance(size, bool) or size < 0:
      raise ValueError("invalid payload reference or size")
    if body and (len(body) != size or hashlib.sha256(body).hexdigest() != ref):
      raise ValueError("payload content does not match its reference")
    return ref, size
  if len(body) > INLINE_BYTES:
    raise ValueError("large payload must be stored before admission")
  if size not in (0, len(body)):
    raise ValueError("payload size does not match inline bytes")
  return hashlib.sha256(body).hexdigest(), len(body)


def packed_reply(reply):
  return {**asdict(reply), "body": "" if reply.body_ref else base64.b64encode(reply.body).decode()}


def unpack_reply(value):
  return Reply(**{**value, "body": base64.b64decode(value["body"])})


class _Transaction:
  """Only records touched by this operation are cached, until EXEC or failure."""

  def __init__(self, pipe):
    self.pipe, self.records, self.dirty, self.operations = pipe, {}, set(), []

  async def record(self, key):
    if key not in self.records:
      fields = await self.pipe.hgetall(key)
      self.records[key] = {k.decode(): json.loads(v) for k, v in fields.items()} if fields else None
    return self.records[key]

  def save(self, key, record):
    self.records[key] = record
    self.dirty.add(key)

  def write(self, method, *args, **kwargs):
    self.operations.append((method, args, kwargs))

  def apply(self):
    for key in self.dirty:
      self.pipe.hset(key, mapping={k: encode(v) for k, v in self.records[key].items()})
    for method, args, kwargs in self.operations:
      getattr(self.pipe, method)(*args, **kwargs)


class Bus:
  def __init__(self, url=None, namespace=None, *, lease_seconds=None):
    self.url = url or getenv("REDIS_URL", "redis://localhost:6379/0")
    self.namespace = namespace or getenv("REDIS_NAMESPACE", "tinyrl")
    if not self.namespace or "{" in self.namespace or "}" in self.namespace:
      raise ValueError("Redis namespace cannot be empty or contain braces")
    self.prefix = f"tinyrl:{{{self.namespace}}}:"
    self.r = redis.Redis.from_url(self.url, socket_connect_timeout=5, socket_timeout=35)
    self.lease_seconds = COORDINATOR_TTL if lease_seconds is None else lease_seconds
    if self.lease_seconds <= 0:
      raise ValueError("coordinator lease must be positive")
    self.token, self.lock = uuid.uuid4().hex, asyncio.Lock()
    self.closed = self.lost = self.acquired = False
    self._started = time.time()

  def key(self, kind, identity=""):
    return f"{self.prefix}{kind}:{identity}"

  async def start(self):
    async with self.lock:
      if self.closed or self.lost:
        raise CoordinatorUnavailable("coordinator no longer owns this namespace")
      if not self.acquired:
        try:
          await self.r.set(self.key("coordinator"), self.token, nx=True, px=math.ceil(self.lease_seconds * 1000))
          if await self.r.get(self.key("coordinator")) != self.token.encode():
            raise CoordinatorUnavailable("another coordinator owns this namespace")
        except RedisError as exc:
          raise CoordinatorUnavailable("Redis unavailable while acquiring coordinator ownership") from exc
        self.acquired, self._started = True, time.time()

  async def check_owner(self):
    async with self._transaction():
      return True

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
        if tx.dirty or tx.operations:
          pipe.lpush(self.key("wake"), "1")
          pipe.ltrim(self.key("wake"), 0, 0)
        await pipe.execute()
      except WatchError as exc:
        self.lost = True
        raise CoordinatorUnavailable("coordinator lease changed during commit") from exc
      except ResponseError as exc:
        # MULTI/EXEC does not undo commands when another command has a runtime
        # error. A schema error violates the private, exclusively owned namespace
        # contract; do not treat possibly partial state as a transport retry.
        self.lost = True
        log.error(
          "Redis transaction failed in %s; inspect and repair this namespace before restarting: %s", self.prefix, exc
        )
        raise CoordinatorUnavailable(
          "Redis transaction failed; inspect and repair the namespace before restart"
        ) from exc
      except RedisError as exc:
        # EXEC's outcome may be unknown. Discard this transaction's cache; the
        # next attempt reads durable identities/results before deciding anything.
        raise CoordinatorUnavailable("Redis command outcome unavailable; retry the same identity") from exc

  async def close(self):
    async with self.lock:
      if self.closed:
        return
      self.closed = True
      try:
        async with self.r.pipeline(transaction=True) as pipe:
          await pipe.watch(self.key("coordinator"))
          if await pipe.get(self.key("coordinator")) == self.token.encode():
            pipe.multi()
            pipe.delete(self.key("coordinator"))
            await pipe.execute()
      except RedisError:
        pass  # an unreachable lease expires; never delete another owner's token
      finally:
        await self.r.aclose()

  async def _record(self, kind, identity):
    fields = await self.r.hgetall(self.key(kind, identity))
    return {k.decode(): json.loads(v) for k, v in fields.items()} if fields else None

  def _cmd(self, c):
    body = base64.b64decode(c["dispatched_body"] or c["body"])
    return Cmd(
      c["run"],
      c["id"],
      Op(c["op"]),
      body,
      c["worker"],
      c["generation"],
      c["token"],
      c["body_ref"],
      c["body_size"] if c["body_ref"] else len(body),
    )

  def _ready(self, tx, r):
    if r["phase"] == "active" and r["pending_count"] and not r["inflight"] and not r["ready"]:
      tx.write("rpush", self.key("ready", r["pool"]), r["id"])
      r["ready"] = True
    tx.save(self.key("run", r["id"]), r)

  def _unready(self, tx, r):
    if r["ready"]:
      tx.write("lrem", self.key("ready", r["pool"]), 0, r["id"])
      r["ready"] = False
    tx.save(self.key("run", r["id"]), r)

  def _insert(self, tx, r, op, body, lane="user", seq=None, idem=None, *, body_ref="", body_size=0, **fields):
    identifier = uuid.uuid4().hex
    content_hash, size = payload(body, body_ref, body_size)
    c = {
      "id": identifier,
      "run": r["id"],
      "op": str(op),
      "body": "" if body_ref else base64.b64encode(body).decode(),
      "body_ref": body_ref,
      "body_size": size,
      "dispatched_body": "",
      "lane": lane,
      "seq": seq,
      "idem": idem,
      "digest": hashlib.sha256(encode([str(op), content_hash, size])).hexdigest(),
      "status": "queued",
      "worker": "",
      "pool": "",
      "generation": 0,
      "token": "",
      "move_id": "",
      "created": time.time(),
      "claimed": None,
      "completed": None,
      "outcome_digest": "",
      **fields,
    }
    tx.save(self.key("cmd", identifier), c)
    tx.write("sadd", self.key("pending", r["id"]), identifier)
    if seq is not None:
      tx.write("hset", self.key("sequence", r["id"]), str(seq), identifier)
    if idem:
      tx.write("hset", self.key("identity", r["id"]), idem, identifier)
    if lane != "control":
      r["pending_count"] += 1
      r["pending_bytes"] += size
      self._ready(tx, r)
    else:
      tx.write("rpush", self.key("controls", c["pool"]), identifier)
      if c["worker"]:
        tx.write("sadd", self.key("worker_controls", c["worker"]), identifier)
    return c

  async def _control(self, tx, r, op, body, *, worker="", pool="", generation=0):
    if op == Op.DETACH:
      ids = await tx.pipe.smembers(self.key("pending", r["id"]))
      candidates = [await tx.record(self.key("cmd", i.decode())) for i in ids]
      candidates += list(tx.records.values())
      for c in candidates:
        if c and c.get("run") == r["id"] and c.get("op") == op and c["status"] in ("queued", "executing"):
          if (c["worker"], c["generation"]) == (worker, generation):
            return c
    return self._insert(
      tx,
      r,
      op,
      encode(body),
      "control",
      worker=worker,
      pool=pool,
      generation=generation,
      move_id=r["move_id"] if op != Op.DETACH else "",
    )

  async def session(self, session_id):
    async with self._transaction() as tx:
      s = await tx.record(self.key("session", session_id)) or {"created": time.time()}
      tx.save(self.key("session", session_id), {**s, "seen": time.time()})
      tx.write("expire", self.key("session", session_id), SESSION_TTL)

  async def session_info(self, session_id):
    s = await self._record("session", session_id)
    if not s:
      return None
    runs = sorted(i.decode() for i in await self.r.smembers(self.key("session_runs", session_id)))
    return {"session_id": session_id, **s, "live": time.time() - s["seen"] < LIVE_SECONDS, "runs": runs}

  async def create(self, run, *, body=None, idempotency_key=None):
    if isinstance(body, dict):
      body = encode(body)
    payload(body or b"")
    spec = {**asdict(run), "lora": json.loads(body).get("lora", {}) if body else {}}
    digest = hashlib.sha256(encode(spec) + (body or b"")).hexdigest()
    async with self._transaction() as tx:
      prior_id = await tx.pipe.hget(self.key("creations"), idempotency_key) if idempotency_key else None
      r = await tx.record(self.key("run", prior_id.decode() if prior_id else run.id))
      if r:
        if r["creation_digest"] != digest:
          raise ValueError("creation identity was already used with different model configuration")
      else:
        r = {
          "id": run.id,
          "spec": spec,
          "created": time.time(),
          "creation_digest": digest,
          "pool": run.pool,
          "owner": "",
          "generation": 1,
          "phase": "active",
          "initialized": False,
          "cursor": 0,
          "next_seq": 0,
          "closing": False,
          "steps": 0,
          "failed": 0,
          "snapshot": "",
          "move_id": "",
          "target_pool": "",
          "source_owner": "",
          "source_pool": "",
          "source_generation": 0,
          "owner_generation": 0,
          "bootstrap": "",
          "inflight": "",
          "ready": False,
          "pending_count": 0,
          "pending_bytes": 0,
        }
        if body is not None:
          r["bootstrap"] = self._insert(tx, r, Op.CREATE_MODEL, body, "bootstrap", -1)["id"]
        tx.save(self.key("run", run.id), r)
        tx.write("sadd", self.key("pool_runs", run.pool), run.id)
        if run.session:
          tx.write("sadd", self.key("session_runs", run.session), run.id)
        if idempotency_key:
          tx.write("hset", self.key("creations"), idempotency_key, run.id)
      info = self._info(r)
      if r["bootstrap"]:
        info["request_id"] = f"{r['id']}/{r['bootstrap']}"
      return info

  def _info(self, r):
    if not r:
      return None
    fields = "created pool owner generation phase steps failed cursor snapshot target_pool closing".split()
    return {**r["spec"], **{k: r[k] for k in fields}}

  async def run(self, run_id):
    return self._info(await self._record("run", run_id))

  async def runs(self, pool):
    return sorted(i.decode() for i in await self.r.smembers(self.key("pool_runs", pool)))

  async def submit(self, run_id, op, body, *, seq_id=None, idempotency_key=None, body_ref="", body_size=0):
    op = Op(op)
    content_hash, size = payload(body, body_ref, body_size)
    if size > MAX_COMMAND_BYTES:
      raise ValueError("command exceeds MAX_COMMAND_BYTES")
    if op in CONTROL or (body_ref and op != Op.FORWARD_BACKWARD):
      raise ValueError("control and routing metadata must be small inline requests")
    if seq_id is not None and (
      isinstance(seq_id, bool) or not isinstance(seq_id, int) or not 0 <= seq_id < (1 << 63) - 1
    ):
      raise ValueError("seq_id must be a nonnegative 63-bit integer")
    lane, seq = ("bootstrap", -1) if op == Op.CREATE_MODEL else ("user", seq_id)
    digest = hashlib.sha256(encode([str(op), content_hash, size])).hexdigest()
    async with self._transaction() as tx:
      r = await tx.record(self.key("run", run_id))
      if not r:
        return None
      identifier = await tx.pipe.hget(self.key("identity", run_id), idempotency_key) if idempotency_key else None
      if not identifier and seq is not None:
        identifier = await tx.pipe.hget(self.key("sequence", run_id), str(seq))
      if identifier:
        c = await tx.record(self.key("cmd", identifier.decode()))
        if c["digest"] != digest or (seq is not None and c["seq"] != seq):
          raise ValueError("same command identity has different content")
        if idempotency_key and c["idem"] != idempotency_key:
          if c["idem"]:
            raise ValueError("sequence already has a different idempotency key")
          c["idem"] = idempotency_key
          tx.save(self.key("cmd", c["id"]), c)
          tx.write("hset", self.key("identity", run_id), idempotency_key, c["id"])
        return f"{run_id}/{c['id']}"
      if r["phase"] == "deleted":
        return None
      if r["closing"]:
        deletion = await tx.record(self.key("cmd", r["deletion"]))
        if op == Op.DELETE_MODEL:
          return f"{run_id}/{deletion['id']}"
        if seq is None or seq >= deletion["seq"]:
          raise ValueError("run is closing")
      if r["phase"] == "recovery" and op != Op.DELETE_MODEL:
        raise ValueError("run requires explicit checkpoint recovery")
      if lane == "user":
        seq = r["next_seq"] if seq is None else seq
        if seq < r["cursor"]:
          raise ValueError("sequence precedes the completed or failed command boundary")
      if op != Op.DELETE_MODEL or r["phase"] not in ("parked", "recovery"):
        if r["pending_count"] >= MAX_PENDING_COMMANDS or r["pending_bytes"] + size > MAX_PENDING_BYTES:
          raise ValueError("run command queue capacity exceeded")
        if (
          lane == "user"
          and seq > r["cursor"]
          and (
            r["pending_count"] >= MAX_PENDING_COMMANDS - 1
            or r["pending_bytes"] + size > MAX_PENDING_BYTES - MAX_COMMAND_BYTES
          )
        ):
          raise ValueError("run command queue capacity reserved for its next sequence")
      c = self._insert(tx, r, op, body, lane, seq, idempotency_key, body_ref=body_ref, body_size=size)
      if lane == "user":
        r["next_seq"] = max(r["next_seq"], seq + 1)
      if op == Op.DELETE_MODEL:
        r["closing"], r["deletion"] = True, c["id"]
        if r["phase"] in ("parked", "recovery"):
          await self._delete(tx, r)
          self._complete(tx, r, c, Reply(True, "application/json", encode({"type": op, "model_id": run_id})))
      tx.save(self.key("run", run_id), r)
      return f"{run_id}/{c['id']}"

  async def register(self, worker, pool):
    if not worker or not pool:
      raise ValueError("worker incarnation and pool are required")
    async with self._transaction() as tx:
      w = await tx.record(self.key("worker", worker))
      if w:
        if w["pool"] != pool:
          raise ValueError("worker incarnation cannot change pools")
        if not w["live"]:
          return {"registered": False, "reason": "worker incarnation was lost"}
      else:
        current = await tx.pipe.get(self.key("pool_owner", pool))
        if current:
          return {"registered": False, "reason": "pool already has a worker", "owner": current.decode()}
      tx.save(self.key("worker", worker), {"pool": pool, "live": True, "seen": time.time()})
      tx.write("set", self.key("pool_owner", pool), worker)
      tx.write("zadd", self.key("workers"), {worker: time.time()})
      return {"registered": True, "worker": worker, "pool": pool}

  async def heartbeat(self, worker):
    async with self._transaction() as tx:
      w = await tx.record(self.key("worker", worker))
      if not w or not w["live"]:
        return False
      w["seen"] = time.time()
      tx.save(self.key("worker", worker), w)
      tx.write("zadd", self.key("workers"), {worker: w["seen"]})
      return True

  async def take(self, pool, worker, count=16, block_ms=1000):
    deadline, count = time.monotonic() + max(0, block_ms) / 1000, max(1, min(int(count), 64))
    while True:
      async with self._transaction() as tx:
        w = await tx.record(self.key("worker", worker))
        if not w or not w["live"] or w["pool"] != pool:
          return []
        pending = [
          await tx.record(self.key("cmd", i.decode())) for i in await tx.pipe.smembers(self.key("executing", worker))
        ]
        if pending:
          return [self._cmd(c) for c in sorted(pending, key=lambda c: c["created"])[:count]]
        taken, busy = [], set()
        for identifier in await tx.pipe.lrange(self.key("controls", pool), 0, -1):
          c = await tx.record(self.key("cmd", identifier.decode()))
          r = await tx.record(self.key("run", c["run"]))
          if c["worker"] and c["worker"] != worker:
            continue
          if c["op"] != Op.DETACH and c["move_id"] != r["move_id"]:
            self._complete(tx, r, c, error("obsolete move"), "cancelled")
            continue
          if c["run"] in busy or (c["op"] != Op.DETACH and r["inflight"]):
            continue
          taken.append(await self._claim(tx, r, c, worker, c["generation"]))
          busy.add(c["run"])
          if len(taken) == count:
            break
        if len(taken) < count:
          for run_id in await tx.pipe.lrange(self.key("ready", pool), 0, -1):
            r = await tx.record(self.key("run", run_id.decode()))
            if r["id"] in busy or r["phase"] != "active" or r["inflight"] or r["owner"] not in ("", worker):
              continue
            c = await tx.record(self.key("cmd", r["bootstrap"])) if r["bootstrap"] else None
            if not c or c["status"] != "queued":
              identifier = await tx.pipe.hget(self.key("sequence", r["id"]), str(r["cursor"]))
              c = await tx.record(self.key("cmd", identifier.decode())) if identifier else None
            if c and c["status"] == "queued":
              r["owner"], r["owner_generation"] = worker, r["generation"]
              tx.write("sadd", self.key("owned", worker), r["id"])
              taken.append(await self._claim(tx, r, c, worker, r["generation"]))
            if len(taken) == count:
              break
        if taken:
          return taken
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        return []
      # Wakeups are bounded, disposable hints. Ready queues and records decide
      # dispatch; missed wakeups simply wait for the next bounded poll.
      await self.r.blpop(self.key("wake"), timeout=min(remaining, 1))

  async def _claim(self, tx, r, c, worker, generation):
    body = base64.b64decode(c["body"])
    if c["op"] == Op.SNAPSHOT:
      body = encode({"boundary": r["cursor"]})
    elif c["op"] == Op.LOAD_WEIGHTS:
      args = json.loads(body)
      ref = await tx.pipe.hget(self.key("artifacts"), args["uri"])
      body = encode({**args, "ref": ref.decode() if ref else ""})
    c.update(
      status="executing",
      worker=worker,
      generation=generation,
      token=uuid.uuid4().hex,
      claimed=time.time(),
      dispatched_body=base64.b64encode(body).decode(),
    )
    tx.save(self.key("cmd", c["id"]), c)
    tx.write("sadd", self.key("executing", worker), c["id"])
    if c["lane"] == "control":
      tx.write("lrem", self.key("controls", c["pool"]), 0, c["id"])
      tx.write("sadd", self.key("worker_controls", worker), c["id"])
    if c["op"] != Op.DETACH:
      r["inflight"] = c["id"]
      self._unready(tx, r)
    return self._cmd(c)

  def _complete(self, tx, r, c, reply, status="done"):
    if c["status"] not in ("queued", "executing"):
      return
    if c["lane"] != "control":
      r["pending_count"] -= 1
      r["pending_bytes"] -= c["body_size"]
    else:
      tx.write("lrem", self.key("controls", c["pool"]), 0, c["id"])
      if c["worker"]:
        tx.write("srem", self.key("worker_controls", c["worker"]), c["id"])
    tx.write("srem", self.key("pending", r["id"]), c["id"])
    if c["worker"]:
      tx.write("srem", self.key("executing", c["worker"]), c["id"])
    if r["inflight"] == c["id"]:
      r["inflight"] = ""
    content_hash, size = payload(reply.body, reply.body_ref, reply.body_size)
    c.update(
      status=status,
      completed=time.time(),
      body="",
      dispatched_body="",
      body_ref="",
      body_size=0,
      outcome_digest=hashlib.sha256(
        encode([reply.ok, reply.content_type, reply.artifact, reply.recover, content_hash, size])
      ).hexdigest(),
    )
    tx.save(self.key("cmd", c["id"]), c)
    tx.save(self.key("run", r["id"]), r)
    tx.write("set", self.key("reply", c["id"]), encode(packed_reply(reply)), px=max(1, math.ceil(REPLY_TTL * 1000)))

  async def finish(self, cmd, reply):
    content_hash, size = payload(reply.body, reply.body_ref, reply.body_size)
    if size > MAX_REPLY_BYTES:
      raise ValueError("result exceeds MAX_REPLY_BYTES")
    outcome = hashlib.sha256(
      encode([reply.ok, reply.content_type, reply.artifact, reply.recover, content_hash, size])
    ).hexdigest()
    async with self._transaction() as tx:
      c = await tx.record(self.key("cmd", cmd.entry_id))
      if (
        not c
        or c["run"] != cmd.run
        or (c["worker"], c["generation"], c["token"]) != (cmd.worker, cmd.generation, cmd.token)
      ):
        return "stale"
      if c["status"] == "done":
        if c["outcome_digest"] != outcome:
          raise ValueError("conflicting duplicate completion")
        return "already_committed"
      if c["status"] != "executing":
        return "stale"
      r = await tx.record(self.key("run", cmd.run))
      if c["lane"] == "control":
        if c["op"] != Op.DETACH and (c["move_id"] != r["move_id"] or r["phase"] not in ("moving", "activating")):
          return "stale"
      elif (r["owner"], r["generation"]) != (cmd.worker, cmd.generation) or r["phase"] in ("recovery", "deleted"):
        return "stale"
      if reply.ok and c["op"] in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER, Op.SNAPSHOT):
        if not re.fullmatch(r"[0-9a-f]{32}", reply.artifact):
          raise ValueError("successful snapshot completion requires an immutable artifact ref")
        if c["op"] != Op.SNAPSHOT:
          tx.write("hset", self.key("artifacts"), json.loads(base64.b64decode(c["body"]))["uri"], reply.artifact)
      self._complete(tx, r, c, reply)
      if c["lane"] == "control":
        await self._finish_control(tx, r, c, reply)
      else:
        if c["lane"] == "user":
          r["cursor"] = c["seq"] + 1
        r["failed"] += not reply.ok
        r["steps"] += reply.ok and c["op"] == Op.OPTIM_STEP
        if c["op"] == Op.CREATE_MODEL and reply.ok:
          r["initialized"] = True
        if c["op"] == Op.DELETE_MODEL and not reply.ok:
          r["closing"] = False
        if reply.recover or (c["op"] == Op.CREATE_MODEL and not reply.ok):
          await self._fail_run(tx, r, "run requires checkpoint recovery after a failed mutation")
        elif c["op"] == Op.DELETE_MODEL and reply.ok:
          await self._delete(tx, r, detach_owner=False)
      self._ready(tx, r)
      return "committed"

  async def _terminal(self, tx, r, message):
    for identifier in await tx.pipe.smembers(self.key("pending", r["id"])):
      c = await tx.record(self.key("cmd", identifier.decode()))
      if c["op"] != Op.DETACH:
        self._complete(tx, r, c, error(message), "unknown" if c["status"] == "executing" else "cancelled")

  async def _fail_run(self, tx, r, message):
    await self._terminal(tx, r, message)
    self._unready(tx, r)
    r.update(phase="recovery", generation=r["generation"] + 1, cursor=r["next_seq"], move_id="", closing=False)
    tx.write("srem", self.key("owned", r["owner"]), r["id"])
    tx.save(self.key("run", r["id"]), r)

  async def request_move(self, run, target_pool):
    async with self._transaction() as tx:
      r = await tx.record(self.key("run", run))
      if not r:
        raise KeyError(run)
      if r["phase"] in ("moving", "activating") and r["target_pool"] == target_pool:
        return self._info(r)
      if r["closing"] or r["phase"] not in ("active", "parked"):
        raise ValueError("run is not available for movement")
      if r["pool"] == target_pool:
        return self._info(r)
      if r["phase"] == "active" and not r["initialized"]:
        raise ValueError("run must finish model creation before movement")
      parked = r["phase"] == "parked"
      self._begin_move(tx, r, target_pool)
      if parked:
        await self._restore_control(tx, r, r["snapshot"])
      else:
        await self._control(tx, r, Op.SNAPSHOT, {}, worker=r["owner"], pool=r["pool"], generation=r["generation"])
      return self._info(r)

  def _begin_move(self, tx, r, target, *, recovery=False):
    self._unready(tx, r)
    r.update(
      phase="moving",
      move_id=uuid.uuid4().hex,
      target_pool=target,
      source_owner="" if recovery else r["owner"],
      source_pool=r["pool"],
      source_generation=r["owner_generation"],
    )
    tx.save(self.key("run", r["id"]), r)

  async def _restore_control(self, tx, r, ref, *, recovery=False):
    await self._control(
      tx,
      r,
      Op.RESTORE,
      {"ref": ref, "boundary": r["cursor"], "spec": r["spec"], "recovery": recovery},
      pool=r["target_pool"],
      generation=r["generation"] + 1,
    )

  async def _detach(self, tx, r, worker, pool, generation):
    w = await tx.record(self.key("worker", worker)) if worker else None
    if w and w["live"]:
      await self._control(tx, r, Op.DETACH, {}, worker=worker, pool=pool, generation=generation)

  async def _finish_control(self, tx, r, c, reply):
    op = Op(c["op"])
    if op == Op.DETACH:
      if reply.ok and (r["source_owner"], r["source_generation"]) == (c["worker"], c["generation"]):
        r.update(source_owner="", source_pool="")
      return
    if not reply.ok:
      if op == Op.ACTIVATE or reply.recover or not r["source_owner"]:
        await self._fail_run(tx, r, "checkpoint restore or activation failed; explicit recovery required")
      else:
        if op == Op.RESTORE:
          await self._detach(tx, r, c["worker"], r["target_pool"], c["generation"])
        r.update(phase="active", move_id="", target_pool="")
      return
    if op == Op.SNAPSHOT:
      r["snapshot"] = reply.artifact
      if not r["target_pool"]:
        tx.write("srem", self.key("pool_runs", r["pool"]), r["id"])
        tx.write("srem", self.key("owned", r["owner"]), r["id"])
        r.update(phase="parked", pool="", owner="", generation=r["generation"] + 1, move_id="")
        await self._detach(tx, r, r["source_owner"], r["source_pool"], r["source_generation"])
      else:
        await self._restore_control(tx, r, reply.artifact)
    elif op == Op.RESTORE:
      tx.write("srem", self.key("pool_runs", r["pool"]), r["id"])
      tx.write("srem", self.key("owned", r["owner"]), r["id"])
      r.update(
        owner=c["worker"],
        owner_generation=c["generation"],
        pool=r["target_pool"],
        generation=c["generation"],
        phase="activating",
      )
      tx.write("sadd", self.key("pool_runs", r["pool"]), r["id"])
      tx.write("sadd", self.key("owned", r["owner"]), r["id"])
      await self._control(
        tx, r, Op.ACTIVATE, {"boundary": r["cursor"]}, worker=c["worker"], pool=r["pool"], generation=c["generation"]
      )
    elif op == Op.ACTIVATE:
      r.update(phase="active", initialized=True, move_id="")
      await self._detach(tx, r, r["source_owner"], r["source_pool"], r["source_generation"])

  async def recover(self, run, checkpoint_ref, target_pool=None):
    if not re.fullmatch(r"[0-9a-f]{32}", checkpoint_ref):
      raise ValueError("recovery requires a durable checkpoint ref")
    async with self._transaction() as tx:
      r = await tx.record(self.key("run", run))
      if not r:
        raise KeyError(run)
      if r["phase"] != "recovery":
        raise ValueError("run is not awaiting recovery")
      target = target_pool or r["pool"] or r["spec"]["pool"]
      if not target:
        raise ValueError("recovery requires a target pool")
      await self._detach(tx, r, r["owner"], r["pool"], r["owner_generation"])
      await self._detach(tx, r, r["source_owner"], r["source_pool"], r["source_generation"])
      self._begin_move(tx, r, target, recovery=True)
      r.update(snapshot=checkpoint_ref, closing=False)
      await self._restore_control(tx, r, checkpoint_ref, recovery=True)
      return self._info(r)

  async def ckpt(self, uri):
    ref = await self.r.hget(self.key("artifacts"), uri)
    return ref.decode() if ref else None

  async def _lose_worker(self, tx, worker):
    w = await tx.record(self.key("worker", worker))
    if not w or not w["live"]:
      return 0
    w["live"] = False
    tx.save(self.key("worker", worker), w)
    tx.write("zrem", self.key("workers"), worker)
    tx.write("delete", self.key("pool_owner", w["pool"]))
    affected = 0
    for identifier in await tx.pipe.smembers(self.key("owned", worker)):
      r = await tx.record(self.key("run", identifier.decode()))
      if r["owner"] == worker and r["phase"] not in ("recovery", "deleted", "parked"):
        await self._fail_run(tx, r, "worker was lost; execution outcome is uncertain, recover from a checkpoint")
        affected += 1
    for identifier in await tx.pipe.smembers(self.key("worker_controls", worker)):
      c = await tx.record(self.key("cmd", identifier.decode()))
      if c["status"] not in ("queued", "executing"):
        continue
      r = await tx.record(self.key("run", c["run"]))
      self._complete(tx, r, c, error("worker was lost"), "cancelled")
      if c["op"] == Op.RESTORE and r["phase"] == "moving" and r["move_id"] == c["move_id"]:
        if r["source_owner"]:
          r.update(phase="active", move_id="", target_pool="")
          self._ready(tx, r)
        else:
          await self._fail_run(tx, r, "restore worker was lost")
    return affected

  async def worker_lost(self, worker):
    async with self._transaction() as tx:
      return await self._lose_worker(tx, worker)

  async def lose_worker(self, run, owner, generation):
    async with self._transaction() as tx:
      r = await tx.record(self.key("run", run))
      if not r or (r["owner"], r["generation"]) != (owner, generation):
        return False
      await self._lose_worker(tx, owner)
      return True

  async def expire_workers(self):
    async with self._transaction() as tx:
      lost = 0
      if time.time() - self._started >= WORKER_TTL:
        for worker in await tx.pipe.zrangebyscore(self.key("workers"), "-inf", time.time() - WORKER_TTL):
          lost += await self._lose_worker(tx, worker.decode())
      return lost

  async def result(self, request_id, timeout):
    try:
      run, identifier = request_id.rsplit("/", 1)
    except ValueError:
      return error("unknown future", "user")
    deadline = time.monotonic() + max(0, timeout)
    while True:
      c = await self._record("cmd", identifier)
      if not c or c["run"] != run:
        return error("unknown future", "user")
      if c["completed"] is not None:
        value = await self.r.get(self.key("reply", identifier))
        if value is None or time.time() - c["completed"] >= REPLY_TTL:
          return error("future result expired", "user")
        return unpack_reply(json.loads(value))
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        return None
      await asyncio.sleep(min(0.05, remaining))

  async def queue(self, run_id):
    r = await self._record("run", run_id)
    if not r or r["phase"] == "deleted":
      return {"live": False}
    records = [await self._record("cmd", i.decode()) for i in await self.r.smembers(self.key("pending", run_id))]
    inflight = [
      {
        "request_id": f"{run_id}/{c['id']}",
        "trainer": c["worker"],
        "seconds": max(0, time.time() - c["claimed"]),
        "deliveries": 1,
        "generation": c["generation"],
      }
      for c in records
      if c["status"] == "executing"
    ]
    return {
      "live": True,
      "queued": sum(c["status"] == "queued" for c in records),
      "in_flight": inflight,
      "phase": r["phase"],
    }

  async def _delete(self, tx, r, *, detach_owner=True):
    await self._terminal(tx, r, "run was deleted")
    self._unready(tx, r)
    if detach_owner:
      await self._detach(tx, r, r["owner"], r["pool"], r["owner_generation"])
    await self._detach(tx, r, r["source_owner"], r["source_pool"], r["source_generation"])
    tx.write("srem", self.key("pool_runs", r["pool"]), r["id"])
    tx.write("srem", self.key("owned", r["owner"]), r["id"])
    r.update(phase="deleted", owner="", closing=True, generation=r["generation"] + 1, move_id="")
    tx.save(self.key("run", r["id"]), r)

  async def drop(self, run_id, pool=""):
    async with self._transaction() as tx:
      r = await tx.record(self.key("run", run_id))
      if r:
        await self._delete(tx, r)
