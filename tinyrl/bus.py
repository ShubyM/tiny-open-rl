"""Runs, sessions, commands and replies, all on Redis.

Every training call is a command on its run's Redis stream, tinyrl:q:<run>. The scheduler assigns runs to workers;
the legacy `pool` field holds that dispatch identity. Workers read assigned streams through one consumer group:

  what is queued       XINFO GROUPS, the group's lag
  what is in flight    XPENDING: which trainer took each command, and how long ago
  what a dead trainer  XAUTOCLAIM, so the next trainer can fail it instead of leaving a client waiting
  left unfinished

A reply is a stream with a single entry, tinyrl:f:<request_id>. XREAD from id 0 blocks until it is written and never
consumes it, so a client can retry retrieve_future, or many clients can wait on one, and every one gets the same answer.

A dispatch identity has one owner, holding tinyrl:owner:<pool>. Replies check both ownership and assignment.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import redis.asyncio as redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff

from tinyrl.helpers import getenv

GROUP = "trainer"  # the consumer group every pool's trainer reads through
LIVE_SECONDS = 120  # a session is live this long after its last heartbeat
REPLY_TTL = getenv("REPLY_TTL", 3600)
SESSION_TTL = getenv("SESSION_TTL", 7 * 24 * 3600)
LEASE_MS = getenv("LEASE_MS", 10000)  # a trainer owns its pool this long after its last renewal
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


@dataclass(frozen=True)
class Run:
  id: str
  pool: str
  base: str
  rank: int
  session: str = ""
  model_path: str | None = None
  lora: dict = field(default_factory=dict)  # the LoRA config it was created with, to create it again elsewhere
  exclusive: bool = False
  role: str = "trainer"
  train_run: str | None = None


@dataclass(frozen=True)
class Cmd:
  run: str
  entry_id: str
  op: Op
  body: bytes
  seq: int | None = None

  @property
  def request_id(self) -> str:
    return f"{self.run}/{self.entry_id}"


@dataclass(frozen=True)
class Reply:
  ok: bool
  content_type: str
  body: bytes


def parse_cmd(run: str, entry_id: bytes, fields: dict[bytes, bytes]) -> Cmd:
  seq = int(fields[b"seq"]) if b"seq" in fields else None
  return Cmd(run, entry_id.decode(), Op(fields[b"op"].decode()), fields[b"body"], seq=seq)


def connect(url: str) -> redis.Redis:
  # a reply is waited for with a blocking read of up to 30s, and redis-py gives up on a read after 5s by default.
  # Retrying a dropped connection for about a minute rides out a Redis restart, which keeps its data in its AOF file.
  # Each request the gateway waits on holds a connection for that read, and redis-py's pool stops at 100 by default
  # ("Too many connections" at 400 concurrent samples): allow as many as Redis itself does (maxclients 10000)
  retry = Retry(ExponentialBackoff(cap=2.0, base=0.1), 30)
  return redis.from_url(url, socket_timeout=None, retry=retry, max_connections=10_000)


@dataclass(frozen=True)
class Event:
  """An entry of the log, tinyrl:events. See scheduler.py."""

  id: str  # the stream id, <ms>-<n>
  t: float  # seconds, from the ms part of the id
  kind: str
  body: dict


class Bus:
  def __init__(self, r: redis.Redis | None = None):
    self.r = r if r is not None else connect(getenv("REDIS_URL", "redis://localhost:6379"))

  @staticmethod
  def event(p, kind: str, **body) -> None:
    # Never trim a replay log without a checkpoint and its corresponding cursor.
    p.xadd(key("events"), {"kind": kind, "body": json.dumps(body)})

  async def at_tail(self, p, after: str | None) -> bool:
    if after is None:
      return True
    await p.watch(key("events"))
    tail = await p.xrevrange(key("events"), count=1)
    return (tail[0][0].decode() if tail else "0-0") == after

  async def emit_decision(self, kind: str, after: str, **body) -> bool:
    """A decision applies only to the exact log prefix from which it was computed."""
    try:
      async with self.r.pipeline() as p:
        if not await self.at_tail(p, after):
          return False
        p.multi()
        self.event(p, kind, **body)
        await p.execute()
        return True
    except redis.WatchError:
      return False

  # ***** sessions *****

  async def session(self, session_id: str, metadata: dict | None = None) -> None:
    """Open a session, or record a heartbeat on one. `metadata` applies to every model the session creates."""
    k, now = key("sess", session_id), time.time()
    p = self.r.pipeline()
    p.hsetnx(k, "created", now)
    if metadata:
      p.hset(k, "metadata", json.dumps(metadata))
    p.hset(k, "seen", now)
    p.expire(k, SESSION_TTL)
    p.expire(f"{k}:runs", SESSION_TTL)
    await p.execute()

  async def session_info(self, session_id: str) -> dict | None:
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

  # ***** runs *****

  async def create(self, run: Run) -> None:
    if run.role not in ("trainer", "sampler"):
      raise ValueError("run role must be trainer or sampler")
    # one transaction, so a run is in its pool exactly when its stream and consumer group exist
    record = json.dumps({**asdict(run), "created": time.time()})
    p = self.r.pipeline(transaction=True)
    p.xgroup_create(key("q", run.id), GROUP, id="0", mkstream=True)
    if run.pool:
      p.sadd(key("pools"), run.pool)
      p.sadd(key("pool", run.pool), run.id)
    p.hset(
      key("run", run.id), mapping={"record": record, "steps": 0, "failed": 0, "cmds": 0, "busy_ms": 0, "wait_ms": 0}
    )
    if run.session:
      p.sadd(key("sess", run.session, "runs"), run.id)
      p.expire(key("sess", run.session, "runs"), SESSION_TTL)
    if run.exclusive:
      p.sadd(key("excl"), run.id)
    requirements = asdict(run)
    requirements.pop("id")
    self.event(p, "run_created", run=run.id, **requirements)
    await p.execute()

  async def run(self, run_id: str) -> dict | None:
    fields = await self.r.hgetall(key("run", run_id))
    if not fields:
      return None

    record = json.loads(fields.pop(b"record"))
    counters = {name.decode(): int(value) for name, value in fields.items()}  # steps, failed, cmds, busy_ms, wait_ms
    return {**record, **counters}

  async def runs(self, pool: str) -> list[str]:
    return [run_id.decode() for run_id in await self.r.smembers(key("pool", pool))]

  async def assign(
    self, run_id: str, pool: str, after: str | None = None, kind: str = "run_assigned", **decision
  ) -> bool:
    """Project the scheduler's assignment and its log entry, run_assigned or worker_created, together."""
    record_key, stream = key("run", run_id), key("q", run_id)
    while True:
      try:
        async with self.r.pipeline() as p:
          if not await self.at_tail(p, after):
            return False
          await p.watch(record_key, stream)
          raw = await p.hget(record_key, "record")
          if not raw or not await p.exists(stream):
            return False
          record = json.loads(raw)
          if record["pool"]:
            return record["pool"] == pool
          record["pool"] = pool
          p.multi()
          p.hset(record_key, "record", json.dumps(record))
          p.sadd(key("pools"), pool)
          p.sadd(key("pool", pool), run_id)
          p.hset(key("decide", "run"), run_id, json.dumps({"choice": pool, **decision}))
          self.event(p, kind, run=run_id, worker=pool, **decision)
          await p.execute()
          return True
      except redis.WatchError:
        continue

  async def unassign_worker(self, pool: str) -> None:
    """Scheduler only, after confirmed process exit. Unfinished commands stay pending and are failed, never replayed."""
    for run_id in await self.runs(pool):
      record_key = key("run", run_id)
      while True:
        try:
          async with self.r.pipeline() as p:
            await p.watch(record_key)
            raw = await p.hget(record_key, "record")
            if not raw or (record := json.loads(raw))["pool"] != pool:
              break
            record["pool"] = ""
            p.multi()
            p.hset(record_key, "record", json.dumps(record))
            p.srem(key("pool", pool), run_id)
            self.event(p, "run_released", run=run_id, source=pool, token="", uri=None, reason="exited")
            await p.execute()
            break
        except redis.WatchError:
          continue

  async def join(self, run_id: str) -> int:
    return int(await self.r.incr(key("clients", run_id)))

  async def pools(self) -> list[str]:
    return sorted(pool.decode() for pool in await self.r.smembers(key("pools")))

  async def stats(self) -> dict[tuple[str, str], dict]:
    """The counters of every run still in a pool, by (pool, run)."""
    runs = [
      (pool.decode(), run_id)
      for pool in await self.r.smembers(key("pools"))
      for run_id in await self.runs(pool.decode())
    ]
    p = self.r.pipeline()
    for _, run_id in runs:
      p.hgetall(key("run", run_id))
    return {at: {k.decode(): int(v) for k, v in f.items() if k != b"record"} for at, f in zip(runs, await p.execute())}

  async def drop(self, run_id: str, pool: str) -> None:
    """Take a deleted run out of its pool and delete its stream, in one transaction."""
    p = self.r.pipeline(transaction=True)
    p.srem(key("pool", pool), run_id)
    p.delete(key("q", run_id), key("parked", run_id))
    p.hdel(key("decide", "run"), run_id)
    self.event(p, "run_deleted", run=run_id, pool=pool)
    await p.execute()

  async def exclusive(self) -> set[str]:
    return {v.decode() for v in await self.r.smembers(key("excl"))}

  async def decide(self, kind: str, subject: str, decision: dict) -> None:
    await self.r.hset(key("decide", kind), subject, json.dumps(decision))

  async def decisions(self) -> dict[str, dict[str, dict]]:
    """The scheduler's current worker assignment for each run."""
    raw = await self.r.hgetall(key("decide", "run"))
    return {"run": {k.decode(): json.loads(v) for k, v in raw.items()}}

  # ***** commands *****

  async def submit(self, run_id: str, op: Op, body: bytes, seq_id: int | None = None) -> str | None:
    """Queue a command on a run's stream. Returns its request id, or None if there is no such run."""
    # NOMKSTREAM: an unknown or deleted run has no stream, so the write fails instead of queueing work nobody will read
    fields = {"op": op, "body": body}
    if seq_id is not None:
      fields["seq"] = str(seq_id)
    entry_id = await self.r.xadd(key("q", run_id), fields, nomkstream=True)
    if entry_id is None:
      return None
    return f"{run_id}/{entry_id.decode()}"

  async def take(self, pool: str, worker: str, count: int = 16, block_ms: int = 1000) -> list[Cmd]:
    """The next commands for a pool's trainer: up to `count` from each run that has work."""
    runs = await self.runs(pool)
    if not runs:
      await asyncio.sleep(block_ms / 1000)
      return []

    random.shuffle(runs)  # so no run always goes first
    streams = {key("q", run_id): ">" for run_id in runs}
    got = await self.r.xreadgroup(GROUP, worker, streams, count=count, block=block_ms or None)

    cmds = []
    for stream, entries in got:
      run_id = stream.decode().removeprefix(key("q", ""))
      cmds += [parse_cmd(run_id, entry_id, fields) for entry_id, fields in entries]
    return cmds

  async def finish(
    self,
    cmd: Cmd,
    reply: Reply,
    pool: str,
    worker: str,
    repeat: bool = False,
    started: float | None = None,
  ) -> bool:
    """Write a command's reply, acknowledge the command, and count it, all at once, if `worker` still owns the pool.
    Returns False, having written nothing, if it doesn't: another trainer took the pool over and fails the command.
    A command run at time.time() `started` also counts its wait since it was queued, and how long it ran."""
    counts = {}
    if not repeat:
      if not reply.ok:
        counts["failed"] = 1
      elif cmd.op == Op.OPTIM_STEP:
        counts["steps"] = 1
      if started is not None:
        queued_ms = int(cmd.entry_id.split("-")[0])  # a stream entry id starts with redis' clock when it was added
        counts |= {
          "cmds": 1,
          "wait_ms": max(0, int(started * 1000) - queued_ms),
          "busy_ms": int((time.time() - started) * 1000),
        }

    k = key("owner", pool)
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          stream, record_key = key("q", cmd.run), key("run", cmd.run)
          await p.watch(k, stream, record_key)
          if await p.get(k) != worker.encode():
            await p.unwatch()
            return False
          record = await p.hget(record_key, "record")
          if not record or json.loads(record)["pool"] != pool:
            return False
          if not await p.xrange(stream, cmd.entry_id, cmd.entry_id):
            return bool(await p.exists(key("f", cmd.request_id)))
          p.multi()
          fields = {"ok": int(reply.ok), "content_type": reply.content_type, "body": reply.body}
          p.xadd(key("f", cmd.request_id), fields, maxlen=1, approximate=False)
          p.expire(key("f", cmd.request_id), REPLY_TTL)
          p.xack(key("q", cmd.run), GROUP, cmd.entry_id)
          p.xdel(key("q", cmd.run), cmd.entry_id)
          for name, n in counts.items():
            p.hincrby(key("run", cmd.run), name, n)
          if not repeat and cmd.seq is not None:
            p.set(key("seq", cmd.run, str(cmd.seq)), cmd.request_id, ex=REPLY_TTL)
          await p.execute()
          return True
      except redis.WatchError:
        continue

  async def first_reply(self, cmd: Cmd) -> Reply | None:
    if cmd.seq is None:
      return None
    request_id = await self.r.get(key("seq", cmd.run, str(cmd.seq)))
    if not request_id:
      return None
    got = await self.r.xrange(key("f", request_id.decode()), "-", "+")
    if not got:
      return None
    fields = got[0][1]
    return Reply(fields[b"ok"] == b"1", fields[b"content_type"].decode(), fields[b"body"])

  async def reclaim(self, pool: str, worker: str, min_idle_ms: int = 0) -> list[Cmd]:
    """Claim the pool's commands that have been in flight for at least min_idle_ms, on any trainer. Called by the pool's
    owner with nothing in flight itself, everything it claims was left by a trainer that died or lost the pool."""
    while True:
      try:
        async with self.r.pipeline() as p:
          await p.watch(key("owner", pool), key("pool", pool))
          if await p.get(key("owner", pool)) != worker.encode():
            return []
          runs = [r.decode() for r in await p.smembers(key("pool", pool))]
          if not runs:
            return []
          p.multi()
          for run_id in runs:
            p.xautoclaim(key("q", run_id), GROUP, worker, min_idle_time=min_idle_ms, count=10000)
          got = await p.execute()
          return [
            parse_cmd(run_id, entry_id, fields)
            for run_id, (_, entries, _) in zip(runs, got)
            for entry_id, fields in entries
          ]
      except redis.WatchError:
        continue

  # ***** parked runs: the state a stopping trainer saved, for the next trainer to restore *****

  async def park(self, run_id: str, uri: str | None, pool: str, worker: str) -> bool:
    """Record where a run's state was saved, or clear it with None, if `worker` still owns the run's pool."""
    k = key("owner", pool)
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          record_key = key("run", run_id)
          await p.watch(k, record_key)
          record = await p.hget(record_key, "record")
          if await p.get(k) != worker.encode() or not record or json.loads(record)["pool"] != pool:
            await p.unwatch()
            return False
          p.multi()
          p.set(key("parked", run_id), uri) if uri else p.delete(key("parked", run_id))
          await p.execute()
          return True
      except redis.WatchError:  # the lease was renewed, maybe by this worker
        continue

  async def parked(self, run_id: str) -> str | None:
    uri = await self.r.get(key("parked", run_id))
    return uri.decode() if uri else None

  # ***** pool leases *****

  async def own(self, pool: str, worker: str) -> bool:
    """Take the pool's lease if no trainer holds it."""
    owner, retired = key("owner", pool), key("retired", pool)
    while True:
      try:
        async with self.r.pipeline() as p:
          await p.watch(owner, retired)
          if await p.exists(retired) or await p.exists(owner):
            return False
          p.multi()
          p.set(owner, worker, nx=True, px=LEASE_MS)
          return bool((await p.execute())[0])
      except redis.WatchError:
        continue

  async def fence_worker(self, worker: str, after: str) -> bool:
    """Retire a lost node's process identity after its lease expires; this does not assert physical exit."""
    owner, retired = key("owner", worker), key("retired", worker)
    try:
      async with self.r.pipeline() as p:
        if not await self.at_tail(p, after):
          return False
        await p.watch(owner, retired)
        if await p.exists(owner):
          return False
        if await p.exists(retired):
          return True
        p.multi()
        p.set(retired, 1)
        self.event(p, "worker_fenced", worker=worker)
        await p.execute()
        return True
    except redis.WatchError:
      return False

  async def renew(self, pool: str, worker: str) -> bool:
    """Hold the pool's lease for another LEASE_MS. False if `worker` doesn't hold it anymore."""
    return bool(await self.r.set(key("owner", pool), worker, ifeq=worker, px=LEASE_MS))

  async def release(self, pool: str, worker: str) -> None:
    await self.r.delex(key("owner", pool), ifeq=worker)

  async def owner(self, pool: str) -> str | None:
    worker = await self.r.get(key("owner", pool))
    return worker.decode() if worker else None

  async def heartbeat(self, node: str, gpus: int) -> None:
    p = self.r.pipeline()
    p.hset(key("nodes"), node, gpus)
    p.hpexpire(key("nodes"), NODE_MS, node)
    await p.execute()

  async def nodes(self) -> dict[str, int]:
    """The GPU count of every live node daemon, by id."""
    return {k.decode(): int(v) for k, v in (await self.r.hgetall(key("nodes"))).items()}

  async def gpus(self) -> int:
    return sum((await self.nodes()).values())

  # ***** replies *****

  async def result(self, request_id: str, timeout: float) -> Reply | None:
    """Wait up to `timeout` seconds for a request's reply. Reading a reply never consumes it."""
    block_ms = int(timeout * 1000) or None  # None means don't wait at all
    got = await self.r.xread({key("f", request_id): "0-0"}, count=1, block=block_ms)
    if not got:
      return None

    fields = got[0][1][0][1]
    return Reply(fields[b"ok"] == b"1", fields[b"content_type"].decode(), fields[b"body"])

  async def queue(self, run_id: str) -> dict:
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

  # ***** events *****

  async def emit(self, _kind: str, **body) -> str:
    entry_id = await self.r.xadd(key("events"), {"kind": _kind, "body": json.dumps(body)})
    return entry_id.decode()

  async def log_size(self) -> int:
    """How many entries were ever appended to the log. It drops below what a reader folded only if Redis was flushed."""
    try:
      return (await self.r.xinfo_stream(key("events")))["entries-added"]
    except redis.ResponseError:  # no log yet
      return 0

  async def events(self, after: str = "0-0", count: int = 10_000, block_ms: int = 0) -> list[Event]:
    """The entries after stream id `after`, waiting up to block_ms for one, or not at all if it is 0."""
    got = await self.r.xread({key("events"): after}, count=count, block=block_ms or None)
    out = []
    for entry_id, fields in got[0][1] if got else []:
      entry_id = entry_id.decode()
      t = int(entry_id.split("-")[0]) / 1000
      out.append(Event(entry_id, t, fields[b"kind"].decode(), json.loads(fields[b"body"])))
    return out
