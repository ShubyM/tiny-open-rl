"""Runs, sessions, commands and replies, all on Redis.

Every training call is a command on its run's Redis stream, tinyrl:q:<run>. The node owning a worker reads its runs' streams through one consumer group, so Redis itself knows:

  what is queued       XINFO GROUPS, the group's lag
  what is in flight    XPENDING: which owner took each command, and how long ago
  what a dead trainer  XAUTOCLAIM, so the next trainer can fail it instead of leaving a client waiting
  left unfinished

A reply is a stream with a single entry, tinyrl:f:<request_id>. XREAD from id 0 blocks until it is written and never consumes it, so a client can retry retrieve_future, or many clients can wait on one, and every one gets the same answer.

A worker has one owner, the node holding its lease, tinyrl:owner:<worker>. Replies are written in a WATCH/MULTI/EXEC transaction on the lease key, so a node that lost its worker, say after a long pause, can't answer for it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum

import redis.asyncio as redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff

from tinyrl.helpers import getenv
from tinyrl.storage.store import split_version

GROUP = "trainer"  # the consumer group used by node-owned readers (name retained for stream compatibility)
LIVE_SECONDS = 30  # a session is live this long after its last heartbeat, which the SDK sends every 10 s
SESSION_GRACE = getenv("SESSION_GRACE", 60)  # and its runs are deleted this long after it
REPLY_TTL = getenv("REPLY_TTL", 3600)
SESSION_TTL = getenv("SESSION_TTL", 7 * 24 * 3600)
LEASE_MS = getenv("LEASE_MS", 10000)  # a node owns its worker this long after its last renewal
NODE_MS = 30000  # a node daemon is live this long after its last heartbeat


def key(*parts: str) -> str:
  return ":".join(("tinyrl", *parts))


class Op(StrEnum):
  CREATE_MODEL = "create_model"
  FORWARD_BACKWARD = "forward_backward"
  OPTIM_STEP = "optim_step"
  SAVE_WEIGHTS = "save_weights"
  SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"
  LOAD_WEIGHTS = "load_weights"
  DELETE_MODEL = "delete_model"
  SAMPLE = "sample"


class Role(StrEnum):
  TRAINER = "trainer"
  SAMPLER = "sampler"


@dataclass(frozen=True)
class Run:
  id: str
  worker: str  # "" until the scheduler assigns one
  base: str
  rank: int
  session: str = ""
  model_path: str | None = None
  lora: dict = field(default_factory=dict)  # the LoRA config it was created with, to create it again elsewhere
  exclusive: bool = False
  role: Role = Role.TRAINER
  train_run: str | None = None  # the training run a sampling session samples
  gpus: int = 1
  version: str | None = None


@dataclass(frozen=True)
class Cmd:
  run: str
  entry_id: str
  op: Op
  body: bytes
  cost: int = 1  # the tokens it works on, which is what the node charges its run
  generation: int | None = None

  @property
  def request_id(self) -> str:
    return f"{self.run}/{self.entry_id}"


@dataclass(frozen=True)
class Reply:
  ok: bool
  content_type: str
  body: bytes

  @classmethod
  def error(cls, exc: Exception) -> Reply:
    body = {
      "error": f"{type(exc).__name__}: {exc}",
      "category": "user" if isinstance(exc, (ValueError, KeyError)) else "server",
    }
    return cls(False, "application/json", json.dumps(body).encode())


def parse_cmd(run: str, entry_id: bytes, fields: dict[bytes, bytes]) -> Cmd:
  return Cmd(run, entry_id.decode(), Op(fields[b"op"].decode()), fields[b"body"], int(fields.get(b"cost", 1)))


def connect(url: str) -> redis.Redis:
  # a reply is waited for with a blocking read of up to 30s, and redis-py gives up on a read after 5s by default. Retrying a dropped connection for about a minute rides out a Redis restart, which keeps its data in its AOF file. Each request the gateway waits on holds a connection for that read, and redis-py's connection pool stops at 100 ("Too many connections" at 400 concurrent samples): allow as many as Redis itself does (maxclients 10000)
  retry = Retry(ExponentialBackoff(cap=2.0, base=0.1), 30)
  return redis.from_url(url, socket_timeout=None, retry=retry, max_connections=10_000)


def entry_key(entry: str) -> tuple[int, int]:
  return tuple(map(int, entry.split("-"))) if entry else (0, 0)


def write_future(p, request: str, reply: Reply) -> None:
  p.xadd(
    key("f", request),
    {"ok": int(reply.ok), "content_type": reply.content_type, "body": reply.body},
    maxlen=1,
    approximate=False,
  )
  p.expire(key("f", request), REPLY_TTL)


def write_checkpoint(p, saved: dict) -> None:
  uri, version, ttl = saved["uri"], saved["version"], saved.get("ttl")
  run, cid = uri.removeprefix("tinker://").split("/", 1)
  now, kind = time.time(), cid.split("/", 1)[0]
  info = {
    "checkpoint_id": cid,
    "checkpoint_type": {"weights": "training", "sampler_weights": "sampler"}.get(kind, kind),
    "time": now,
  }
  info |= {
    "tinker_path": uri,
    "size_bytes": saved.get("size", 0),
    "expires_at": now + ttl if ttl else None,
    "owned": saved.get("owned", False),
    "version": version,
    "boundary": saved.get("boundary", ""),
  }
  p.hset(key("ckpt", run), cid, json.dumps(info))
  p.hset(key("refs"), f"ckpt:{uri}", split_version(version)[0])
  if kind != "parked":
    p.zadd(key("ckpts"), {uri: now})
  p.zadd(key("expiry"), {uri: now + ttl}) if ttl else p.zrem(key("expiry"), uri)


class Bus:
  def __init__(self, r: redis.Redis | None = None):
    self.r = r if r is not None else connect(getenv("REDIS_URL", "redis://localhost:6379"))

  async def tx(self, keys, step, *, raise_on_error: bool = True):
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(*keys)
          write, result = await step(p)
          values = await p.execute(raise_on_error=raise_on_error) if write else []
          return result(values) if callable(result) else result
      except redis.WatchError:
        continue

  # ***** sessions *****

  async def heartbeat_session(self, session_id: str, metadata: dict | None = None) -> None:
    """Open a session, or record a heartbeat on one. `metadata` applies to every model the session creates."""
    k, now = key("sess", session_id), time.time()
    p = self.r.pipeline()
    p.hsetnx(k, "created", now)
    if metadata:
      p.hset(k, "metadata", json.dumps(metadata))
    p.hset(k, "seen", now)
    p.zadd(key("seen"), {session_id: now})
    p.expire(k, SESSION_TTL)
    p.expire(f"{k}:runs", SESSION_TTL)
    await p.execute()

  async def get_session(self, session_id: str) -> dict | None:
    p = self.r.pipeline()
    p.hgetall(key("sess", session_id))
    p.smembers(key("sess", session_id, "runs"))
    record, runs = await p.execute()
    if not record:
      return None

    seen = float(record[b"seen"])
    return {
      "session_id": session_id,
      "created": float(record[b"created"]),
      "seen": seen,
      "live": time.time() - seen < LIVE_SECONDS,
      "runs": sorted(r.decode() for r in runs),
      "metadata": json.loads(record.get(b"metadata") or "{}"),
    }

  async def expired_sessions(self) -> list[str]:
    """The sessions whose client stopped heartbeating SESSION_GRACE seconds ago."""
    return [s.decode() for s in await self.r.zrangebyscore(key("seen"), 0, time.time() - SESSION_GRACE)]

  async def finish_session(self, session_id: str) -> None:
    """A client finished its session: end it on the scheduler's next pass, without waiting out SESSION_GRACE."""
    await self.r.zadd(key("seen"), {session_id: 0}, xx=True)

  async def end_session(self, session_id: str) -> None:
    """Delete an expired session's runs: a training run once its queued work is done, a sampling session now. A session with a training run left is ended again SESSION_GRACE later, since the next trainer fails a delete one died holding."""
    left = False
    for run_id in (await self.get_session(session_id) or {}).get("runs", []):
      run = await self.get_run(run_id) or {}
      if not await self.r.sismember(key("runs"), run_id):
        continue
      if run.get("role") == Role.SAMPLER:
        await self.delete_run(run_id, run.get("worker", ""))
      elif run.get("role") == Role.TRAINER:
        left = bool(await self.submit_command(run_id, Op.DELETE_MODEL, b"{}")) or left
    if left:
      await self.r.zadd(key("seen"), {session_id: time.time()})
    else:
      await self.r.zrem(key("seen"), session_id)

  # ***** runs *****

  async def create_run(self, run: Run) -> None:
    rkey, v = key("run", run.id), split_version(run.version)[0]

    async def step(p):
      if await p.hexists(rkey, "record"):
        return False, None
      if v and (not (raw := await p.get(key("ver", v))) or json.loads(raw)["state"] != "live"):
        raise ValueError("checkpoint is missing or being deleted")
      p.multi()
      p.xgroup_create(key("q", run.id), GROUP, id="0", mkstream=True)
      p.sadd(key("runs"), run.id)
      state = {"generation": 0, "mutation": "", "created": "", "working": None, "durable": None}
      p.hset(
        rkey,
        mapping={
          "record": json.dumps(asdict(run) | {"created": time.time()}),
          "worker": run.worker,
          "state": json.dumps(state),
          "steps": 0,
          "failed": 0,
          "cmds": 0,
          "busy_ms": 0,
          "wait_ms": 0,
        },
      )
      if v:
        p.hset(key("refs"), f"job:{run.id}:weights", v)
      if run.session:
        p.sadd(key("sess", run.session, "runs"), run.id)
        p.expire(key("sess", run.session, "runs"), SESSION_TTL)
      return True, None

    await self.tx([rkey, *([key("ver", v)] if v else [])], step)

  # ***** the cluster's assignment *****

  async def read_table(self) -> dict:
    async def snapshot(p):
      runs = sorted(r.decode() for r in await p.smembers(key("runs")))
      workers = {n.decode(): json.loads(v) for n, v in (await p.hgetall(key("workers"))).items()}
      p.multi()
      p.hgetall(key("nodes"))
      for run in runs:
        p.hmget(key("run", run), "record", "worker", "state")
      for worker in workers:
        p.get(key("owner", worker))
      for name in ("obs", "service", "settled"):
        p.hgetall(key(name))
      return True, lambda rows: (runs, workers, rows)

    runs, workers, rows = await self.tx((key("runs"), key("workers")), snapshot, raise_on_error=False)
    nodes = {n.decode(): json.loads(v) for n, v in rows[0].items()}
    records, owners = rows[1 : 1 + len(runs)], rows[1 + len(runs) : 1 + len(runs) + len(workers)]
    obs, service, settled = rows[-3:]
    return {
      "nodes": {n: v for n, v in nodes.items() if v["until"] >= time.time()},
      "workers": workers,
      "runs": {
        r: (json.loads(rec), (w or b"").decode(), json.loads(st or "{}"))
        for r, (rec, w, st) in zip(runs, records)
        if rec
      },
      "owners": {n: (o or b"").decode() for n, o in zip(workers, owners)},
      "observations": {n.decode(): json.loads(v) for n, v in obs.items()},
      "service": {n.decode(): float(v) for n, v in service.items()},
      "settled": {n.decode(): json.loads(v) for n, v in settled.items()},
    }

  async def write_observation(self, worker: str, owner: str, obs: dict, *, starting: bool = False) -> bool:
    async def step(p):
      w = json.loads(raw) if (raw := await p.hget(key("workers"), worker)) else None
      if await p.get(key("owner", worker)) != owner.encode() or not w or w["node"] != obs["node"]:
        return False, False
      if starting and (
        w.get("owner") != owner or w["revision"] != obs["revision"] or w["gpus"] != obs["gpus"] or not w["budgets"]
      ):
        return False, False
      p.multi()
      current = {"revision": w["revision"]} if obs.get("closed") else {}
      p.hset(key("obs"), worker, json.dumps(obs | current | {"owner": owner, "at": time.time()}))
      return True, True

    return await self.tx((key("owner", worker), key("workers")), step)

  async def claim_scheduler(self, holder: str) -> bool:
    return await self.renew_lease("scheduler", holder) or await self.take_lease("scheduler", holder)

  async def get_run(self, run_id: str) -> dict | None:
    fields = await self.r.hgetall(key("run", run_id))
    if b"record" not in fields:
      return None
    record = json.loads(fields.pop(b"record"))
    state = json.loads(fields.pop(b"state", b"{}"))
    worker = fields.pop(b"worker", b"").decode()
    return record | {"worker": worker, "state": state} | {k.decode(): int(v) for k, v in fields.items()}

  async def join_sampling_session(self, run_id: str) -> int:
    return int(await self.r.incr(key("clients", run_id)))

  async def run_counters(self) -> dict[tuple[str, str], dict]:
    out = {}
    for r in await self.r.smembers(key("runs")):
      if (run := await self.get_run(r.decode())) and run["worker"]:
        out[(run["worker"], r.decode())] = {k: run.get(k, 0) for k in ("steps", "failed", "cmds", "busy_ms", "wait_ms")}
    return out

  async def delete_run(self, run_id: str, worker: str) -> None:
    async def step(p):
      await self._remove_run(p, run_id)
      return True, None

    await self.tx((key("run", run_id),), step)

  async def _remove_run(self, p, run: str, completed: str = "") -> None:
    q, ckpts, saving, refs = key("q", run), key("ckpt", run), key("saving", run), key("refs")
    await p.watch(q, ckpts, saving)
    owned = [c for raw in (await p.hvals(ckpts)) if (c := json.loads(raw))["owned"]]
    entries = await p.xrange(q)
    p.multi()
    p.hdel(refs, *(f"job:{run}:{name}" for name in ("working", "durable", "weights")))
    for eid, _ in entries:
      p.hdel(refs, f"cmd:{run}/{eid.decode()}", f"save:{run}/{eid.decode()}")
      if eid.decode() != completed:
        write_future(p, f"{run}/{eid.decode()}", Reply.error(RuntimeError("model deleted before command completed")))
    for c in owned:
      p.hdel(refs, f"ckpt:{c['tinker_path']}")
      p.hdel(ckpts, c["checkpoint_id"])
      p.zrem(key("ckpts"), c["tinker_path"])
      p.zrem(key("expiry"), c["tinker_path"])
    p.srem(key("runs"), run)
    p.delete(q, saving)
    p.hdel(key("run", run), "state")

  async def pin_weights(self, run_id: str, version: str) -> None:
    v, rkey = split_version(version)[0], key("run", run_id)

    async def step(p):
      if not (raw := await p.hget(rkey, "record")) or not await p.hexists(rkey, "state"):
        return False, None
      record = json.loads(raw)
      if not (raw := await p.get(key("ver", v))) or json.loads(raw)["state"] != "live":
        raise ValueError("checkpoint is missing or being deleted")
      p.multi()
      p.hset(rkey, "record", json.dumps(record | {"version": version}))
      p.hset(key("refs"), f"job:{run_id}:weights", v)
      return True, None

    await self.tx((rkey, key("ver", v)), step)

  # ***** commands *****

  async def submit_command(
    self, run_id: str, op: Op, body: bytes, seq_id: int | None = None, cost: int = 1
  ) -> str | None:
    q, skey = key("q", run_id), key("seq", run_id, str(seq_id))
    v = split_version(json.loads(body).get("version"))[0] if op == Op.LOAD_WEIGHTS else None

    async def step(p):
      if seq_id is not None and (raw := await p.get(skey)):
        return False, raw.decode()
      if not await p.exists(q):
        return False, None
      if v and (not (raw := await p.get(key("ver", v))) or json.loads(raw)["state"] != "live"):
        raise ValueError("checkpoint is missing or being deleted")
      # Choosing the stream ID under WATCH lets the queue entry, deduplication and input reference commit together.
      ms, n = map(int, (await p.xinfo_stream(q))["last-generated-id"].decode().split("-"))
      now = max(ms, time.time_ns() // 1_000_000)
      entry = f"{now}-{n + 1 if now == ms else 0}"
      request = f"{run_id}/{entry}"
      p.multi()
      p.xadd(q, {"op": op, "body": body, "cost": cost}, id=entry, nomkstream=True)
      if seq_id is not None:
        p.set(skey, request, ex=SESSION_TTL)
      if v:
        p.hset(key("refs"), f"cmd:{request}", v)
      return True, request

    return await self.tx([q, skey, *([key("ver", v)] if v else [])], step)

  async def head(self, run: str, block_ms: int = 0) -> Cmd | None:
    """Look at the next command without claiming it or changing its delivery cursor."""
    q = key("q", run)
    try:
      group = next(g for g in await self.r.xinfo_groups(q) if g["name"] == GROUP.encode())
    except redis.ResponseError as exc:
      if "no such key" not in str(exc):
        raise
      return None
    got = await self.r.xread({q: group["last-delivered-id"]}, count=1, block=block_ms or None)
    return parse_cmd(run, *got[0][1][0]) if got else None

  async def _claim(
    self, run: str, generation: int, owner: str, count: int, cursor: str | None = None, limit: int | None = None
  ):
    rkey, qkey = key("run", run), key("q", run)

    async def step(p):
      raw, worker = await p.hmget(rkey, "state", "worker")
      if not raw or json.loads(raw)["generation"] != generation or not worker:
        return False, ([], None)
      okey = key("owner", worker.decode())
      await p.watch(okey)
      if await p.get(okey) != owner.encode():
        return False, ([], None)
      take = count
      if limit is not None:
        group = next(g for g in await p.xinfo_groups(qkey) if g["name"] == GROUP.encode())
        take, cost = 0, 0
        for _, fields in await p.xrange(qkey, min=f"({group['last-delivered-id'].decode()}", count=count):
          n = int(fields.get(b"cost", 1))
          if limit <= 0 or (take and cost + n > limit):
            break
          take, cost = take + 1, cost + n
        if not take:
          return False, ([], None)
      p.multi()
      if cursor is None:
        p.xreadgroup(GROUP, owner, {qkey: ">"}, count=take)
      else:
        p.xautoclaim(qkey, GROUP, owner, min_idle_time=0, start_id=cursor, count=take)

      def unpack(values):
        nxt, entries = (
          (None, [entry for _, entries in values[0] for entry in entries]) if cursor is None else values[0][:2]
        )
        return [replace(parse_cmd(run, eid, fs), generation=generation) for eid, fs in entries], (
          nxt.decode() if nxt and nxt != b"0-0" else None
        )

      return True, unpack

    return await self.tx((rkey, qkey), step)

  async def take_commands(
    self, runs: dict[str, int], owner: str, count: int = 16, limits: dict[str, int] | None = None
  ) -> list[Cmd]:
    out = []
    for run, gen in runs.items():
      cmds, _ = await self._claim(run, gen, owner, count, limit=(limits or {}).get(run))
      out.extend(cmds)
    return out

  async def write_reply(
    self,
    cmd: Cmd,
    reply: Reply | None,
    worker: str,
    owner: str,
    started: float | None = None,
    *,
    checkpoint: dict | None = None,
    mutated: bool = False,
    defer: bool = False,
  ) -> bool:
    """Commit a snapshot or finish its pending command under the process lease and job generation."""
    rkey, okey, qkey = key("run", cmd.run), key("owner", worker), key("q", cmd.run)
    v, deleting = (
      split_version((checkpoint or {}).get("version"))[0],
      bool(reply and reply.ok and cmd.op == Op.DELETE_MODEL),
    )

    named = bool(
      checkpoint and reply and reply.ok and not defer and cmd.op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER)
    )
    alias = checkpoint["uri"].removeprefix("tinker://").split("/", 1) if named else None

    async def step(p):
      raw, assigned = await p.hmget(rkey, "state", "worker")
      if not raw or assigned != worker.encode() or await p.get(okey) != owner.encode():
        return False, False
      state = json.loads(raw)
      if state["generation"] != cmd.generation or (
        (reply or defer) and not await p.xpending_range(qkey, GROUP, cmd.entry_id, cmd.entry_id, count=1)
      ):
        return False, False
      if v:
        raw = await p.get(key("ver", v))
        if (raw and json.loads(raw)["state"] != "live") or (not raw and cmd.op == Op.LOAD_WEIGHTS):
          return False, False
      if named and cmd.op == Op.SAVE_WEIGHTS and not checkpoint.get("durable"):
        return False, False
      previous = json.loads(raw) if alias and (raw := await p.hget(key("ckpt", alias[0]), alias[1])) else {}
      changed = {}
      if checkpoint and checkpoint.get("meta", {}).get("optimizer", checkpoint.get("optimizer", False)):
        for name in ("working", "durable") if checkpoint.get("durable") else ("working",):
          if entry_key(checkpoint["boundary"]) >= entry_key((state.get(name) or {}).get("boundary", "")):
            state[name], changed[f"job:{cmd.run}:{name}"] = checkpoint, v
      if reply and reply.ok and mutated:
        state["mutation"] = cmd.entry_id
      if reply and reply.ok and cmd.op == Op.CREATE_MODEL:
        state["created"] = cmd.entry_id
      if deleting:
        await self._remove_run(p, cmd.run, cmd.entry_id)
      else:
        p.multi()
        p.hset(rkey, "state", json.dumps(state))
        if v:
          info = {name: checkpoint[name] for name in ("uri", "version")} | {"time": time.time(), "state": "live"}
          p.setnx(key("ver", v), json.dumps(info))
        if named and entry_key(checkpoint["boundary"]) >= entry_key(previous.get("boundary", "")):
          write_checkpoint(p, checkpoint)
        if changed:
          p.hset(key("refs"), mapping=changed)
      if defer:
        p.hset(key("saving", cmd.run), cmd.entry_id, json.dumps(checkpoint))
        p.hset(key("refs"), f"save:{cmd.request_id}", v)
      elif reply:
        write_future(p, cmd.request_id, reply)
        p.xack(qkey, GROUP, cmd.entry_id)
        p.xdel(qkey, cmd.entry_id)
        p.hdel(key("saving", cmd.run), cmd.entry_id)
        p.hdel(key("refs"), f"save:{cmd.request_id}", f"cmd:{cmd.request_id}")
        p.hincrby(rkey, "cmds", 1)
        if not reply.ok or cmd.op == Op.OPTIM_STEP:
          p.hincrby(rkey, "failed" if not reply.ok else "steps", 1)
      if started is not None:
        p.hincrby(rkey, "busy_ms", int((time.time() - started) * 1000))
        p.hincrby(rkey, "wait_ms", max(0, int(started * 1000) - int(cmd.entry_id.split("-")[0])))
      return True, True

    return await self.tx(
      [okey, rkey, qkey, *([key("ver", v)] if v else []), *([key("ckpt", alias[0])] if alias else [])], step
    )

  async def reclaim_commands(self, runs: dict[str, int], worker: str, owner: str) -> list[tuple[Cmd, dict | None]]:
    pending = []
    for run, gen in runs.items():
      cursor = "0-0"
      while cursor is not None:
        commands, cursor = await self._claim(run, gen, owner, 10000, cursor)
        for cmd in commands:
          saved = await self.r.hget(key("saving", run), cmd.entry_id)
          if saved or cmd.op == Op.SAMPLE:
            pending.append((cmd, json.loads(saved) if saved else None))
          else:
            await self.write_reply(cmd, Reply.error(RuntimeError(f"worker lost during {cmd.op}")), worker, owner)
    return pending

  # ***** worker leases *****

  async def take_lease(self, worker: str, owner: str) -> bool:
    return bool(await self.r.set(key("owner", worker), owner, nx=True, px=LEASE_MS))

  async def renew_lease(self, worker: str, owner: str) -> bool:
    """Hold the worker's lease for another LEASE_MS. False if `owner` doesn't hold it anymore."""
    return bool(await self.r.set(key("owner", worker), owner, ifeq=owner, px=LEASE_MS))

  async def release_lease(self, worker: str, owner: str) -> None:
    await self.r.delex(key("owner", worker), ifeq=owner)

  # ***** nodes *****

  async def node_heartbeat(self, node: str, gpus: int, memory: int = 0, **extra) -> None:
    beat = {"gpus": gpus, "memory": memory, "until": time.time() + NODE_MS / 1000} | extra
    await self.r.hset(key("nodes"), node, json.dumps(beat))

  # ***** replies *****

  async def wait_for_reply(self, request_id: str, timeout: float) -> Reply | None:
    """Wait up to `timeout` seconds for a request's reply. Reading a reply never consumes it."""
    block_ms = int(timeout * 1000) or None  # None means don't wait at all
    got = await self.r.xread({key("f", request_id): "0-0"}, count=1, block=block_ms)
    if not got:
      return None

    fields = got[0][1][0][1]
    return Reply(fields[b"ok"] == b"1", fields[b"content_type"].decode(), fields[b"body"])

  async def queued_and_in_flight(self, run_id: str) -> dict:
    """What a run has queued and in flight, read straight from its stream."""
    stream = key("q", run_id)
    p = self.r.pipeline()
    p.xinfo_groups(stream)
    p.xpending_range(stream, GROUP, min="-", max="+", count=100)
    groups, pending = await p.execute(raise_on_error=False)
    if isinstance(groups, Exception):  # the stream is gone, so the run was deleted
      return {"live": False}

    in_flight = [
      {
        "request_id": f"{run_id}/{entry['message_id'].decode()}",
        "trainer": entry["consumer"].decode(),
        "seconds": entry["time_since_delivered"] / 1000,
        "deliveries": entry["times_delivered"],
      }
      for entry in pending
    ]
    return {"live": True, "queued": groups[0]["lag"], "in_flight": in_flight}
