"""Runs, sessions, commands and replies, all on Redis.

Every training call is a command on its run's Redis stream, tinyrl:q:<run>. The trainer for a worker reads its runs'
streams through one consumer group, so Redis itself knows:

  what is queued       XINFO GROUPS, the group's lag
  what is in flight    XPENDING: which trainer took each command, and how long ago
  what a dead trainer  XAUTOCLAIM, so the next trainer can fail it instead of leaving a client waiting
  left unfinished

A reply is a stream with a single entry, tinyrl:f:<request_id>. XREAD from id 0 blocks until it is written and never
consumes it, so a client can retry retrieve_future, or many clients can wait on one, and every one gets the same answer.

A worker has one owner, the trainer holding its lease, tinyrl:owner:<worker>. Replies are written in a WATCH/MULTI/EXEC
transaction on the lease key, so a trainer that lost its worker, say after a long pause, can't answer for it.
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

GROUP = "trainer"  # the consumer group every worker's trainer reads through
LIVE_SECONDS = 120  # a session is live this long after its last heartbeat
REPLY_TTL = getenv("REPLY_TTL", 3600)
SESSION_TTL = getenv("SESSION_TTL", 7 * 24 * 3600)
LEASE_MS = getenv("LEASE_MS", 10000)  # a trainer owns its worker this long after its last renewal
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
  worker: str  # "" until the scheduler assigns one
  base: str
  rank: int
  session: str = ""
  model_path: str | None = None
  lora: dict = field(default_factory=dict)  # the LoRA config it was created with, to create it again elsewhere
  exclusive: bool = False
  role: str = "trainer"
  train_run: str | None = None  # the training run a sampling session samples


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
  # Each request the gateway waits on holds a connection for that read, and redis-py's connection pool stops at 100
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
    # one transaction, so a run is in its worker exactly when its stream and consumer group exist
    record = json.dumps({**asdict(run), "created": time.time()})
    p = self.r.pipeline(transaction=True)
    p.xgroup_create(key("q", run.id), GROUP, id="0", mkstream=True)
    p.sadd(key("runs"), run.id)
    mapping = {"record": record, "steps": 0, "failed": 0, "cmds": 0, "busy_ms": 0, "wait_ms": 0}
    if run.worker:
      p.sadd(key("worker", run.worker), run.id)
      mapping["worker"] = run.worker
    p.hset(key("run", run.id), mapping=mapping)
    if run.session:
      p.sadd(key("sess", run.session, "runs"), run.id)
      p.expire(key("sess", run.session, "runs"), SESSION_TTL)
    body = {"run": run.id, "role": run.role, "base": run.base, "rank": run.rank, "exclusive": run.exclusive}
    body |= {"train_run": run.train_run} | ({"worker": run.worker} if run.worker else {})
    p.xadd(key("events"), {"kind": "run_created", "body": json.dumps(body)})
    await p.execute()

  async def assign(self, run: str, worker: str) -> None:
    """The scheduler's assignment of a run to a worker, with its log entry, at once. Only the scheduler writes it."""
    p = self.r.pipeline(transaction=True)
    p.hset(key("run", run), "worker", worker)
    p.sadd(key("worker", worker), run)
    p.xadd(key("events"), {"kind": "assigned", "body": json.dumps({"run": run, "worker": worker})})
    await p.execute()

  async def run(self, run_id: str) -> dict | None:
    fields = await self.r.hgetall(key("run", run_id))
    if not fields:
      return None

    record = json.loads(fields.pop(b"record"))
    if worker := fields.pop(b"worker", None):
      record["worker"] = worker.decode()
    counters = {name.decode(): int(value) for name, value in fields.items()}  # steps, failed, cmds, busy_ms, wait_ms
    return {**record, **counters}

  async def runs(self, worker: str) -> list[str]:
    return [run_id.decode() for run_id in await self.r.smembers(key("worker", worker))]

  async def join(self, run_id: str) -> int:
    return int(await self.r.incr(key("clients", run_id)))

  async def stats(self) -> dict[tuple[str, str], dict]:
    """The counters of every run still in a worker, by (worker, run)."""
    runs = sorted(r.decode() for r in await self.r.smembers(key("runs")))
    p = self.r.pipeline()
    for run_id in runs:
      p.hgetall(key("run", run_id))
    return {
      (f[b"worker"].decode(), run_id): {k.decode(): int(v) for k, v in f.items() if k not in (b"record", b"worker")}
      for run_id, f in zip(runs, await p.execute())
      if b"worker" in f
    }

  async def drop(self, run_id: str, worker: str) -> None:
    """Take a deleted run out of its worker and delete its stream, in one transaction."""
    p = self.r.pipeline(transaction=True)
    p.srem(key("worker", worker), run_id)
    p.srem(key("runs"), run_id)
    p.delete(key("q", run_id), key("parked", run_id))
    await p.execute()
    await self.emit("run_deleted", run=run_id, worker=worker)

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

  async def take(self, worker: str, owner: str, count: int = 16, block_ms: int = 1000) -> list[Cmd]:
    """The next commands for a worker's trainer: up to `count` from each run that has work."""
    runs = await self.runs(worker)
    if not runs:
      await asyncio.sleep(block_ms / 1000)
      return []

    random.shuffle(runs)  # so no run always goes first
    streams = {key("q", run_id): ">" for run_id in runs}
    got = await self.r.xreadgroup(GROUP, owner, streams, count=count, block=block_ms or None)

    cmds = []
    for stream, entries in got:
      run_id = stream.decode().removeprefix(key("q", ""))
      cmds += [parse_cmd(run_id, entry_id, fields) for entry_id, fields in entries]
    return cmds

  async def finish(
    self, cmd: Cmd, reply: Reply, worker: str, owner: str, repeat: bool = False, started: float | None = None
  ) -> bool:
    """Write a command's reply, acknowledge the command, and count it, all at once, if `owner` still owns the worker.
    Returns False, having written nothing, if it doesn't: another trainer took the worker over and fails the command.
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

    k = key("owner", worker)
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(k)
          if await p.get(k) != owner.encode():
            await p.unwatch()
            return False
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

  async def reclaim(self, worker: str, owner: str, min_idle_ms: int = 0) -> list[Cmd]:
    """Claim the worker's commands that have been in flight for at least min_idle_ms, on any trainer. Called by its
    owner with nothing in flight itself, everything it claims was left by a trainer that died or lost the worker."""
    cmds = []
    for run_id in await self.runs(worker):
      stream = key("q", run_id)
      _, entries, _ = await self.r.xautoclaim(stream, GROUP, owner, min_idle_time=min_idle_ms, count=10000)
      cmds += [parse_cmd(run_id, entry_id, fields) for entry_id, fields in entries]
    return cmds

  # ***** parked runs: the state a stopping trainer saved, for the next trainer to restore *****

  async def park(self, run_id: str, uri: str | None, worker: str, owner: str) -> bool:
    """Record where a run's state was saved, or clear it with None, if `owner` still owns the run's worker."""
    k = key("owner", worker)
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(k)
          if await p.get(k) != owner.encode():
            await p.unwatch()
            return False
          p.multi()
          p.set(key("parked", run_id), uri) if uri else p.delete(key("parked", run_id))
          await p.execute()
          return True
      except redis.WatchError:  # the lease was renewed, maybe by this owner
        continue

  async def parked(self, run_id: str) -> str | None:
    uri = await self.r.get(key("parked", run_id))
    return uri.decode() if uri else None

  # ***** worker leases *****

  async def own(self, worker: str, owner: str) -> bool:
    """Take the worker's lease if no trainer holds it."""
    return bool(await self.r.set(key("owner", worker), owner, nx=True, px=LEASE_MS))

  async def renew(self, worker: str, owner: str) -> bool:
    """Hold the worker's lease for another LEASE_MS. False if `owner` doesn't hold it anymore."""
    return bool(await self.r.set(key("owner", worker), owner, ifeq=owner, px=LEASE_MS))

  async def release(self, worker: str, owner: str) -> None:
    await self.r.delex(key("owner", worker), ifeq=owner)

  async def owner(self, worker: str) -> str | None:
    owner = await self.r.get(key("owner", worker))
    return owner.decode() if owner else None

  # ***** nodes *****

  async def heartbeat(self, node: str, gpus: int) -> None:
    await self.r.hset(key("nodes"), node, json.dumps({"gpus": gpus, "until": time.time() + NODE_MS / 1000}))

  async def nodes(self) -> dict[str, int]:
    """The GPU count of every live node daemon, by id."""
    nodes = {k.decode(): json.loads(v) for k, v in (await self.r.hgetall(key("nodes"))).items()}
    if dead := [k for k, v in nodes.items() if v["until"] < time.time()]:
      await self.r.hdel(key("nodes"), *dead)
    return {k: v["gpus"] for k, v in nodes.items() if k not in dead}

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
    entry_id = await self.r.xadd(
      key("events"), {"kind": _kind, "body": json.dumps(body)}, maxlen=1_000_000, approximate=True
    )
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
