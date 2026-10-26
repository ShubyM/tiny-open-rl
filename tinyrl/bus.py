"""Runs, sessions, commands and replies, all on Redis.

Every training call is a command on its run's Redis stream, tinyrl:q:<run>. The trainer for a pool reads its runs'
streams through one consumer group, so Redis itself knows:

  what is queued       XINFO GROUPS, the group's lag
  what is in flight    XPENDING: which trainer took each command, and how long ago
  what a dead trainer  XAUTOCLAIM, so the next trainer can fail it instead of leaving a client waiting
  left unfinished

A reply is a stream with a single entry, tinyrl:f:<request_id>. XREAD from id 0 blocks until it is written and never
consumes it, so a client can retry retrieve_future, or many clients can wait on one, and every one gets the same answer.

A pool has one owner, the trainer holding its lease, tinyrl:owner:<pool>. Replies are written in a WATCH/MULTI/EXEC
transaction on the lease key, so a trainer that lost its pool, say after a long pause, can't answer for it.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from enum import StrEnum

import redis.asyncio as redis

from tinyrl.helpers import getenv

GROUP = "trainer"  # the consumer group every pool's trainer reads through
LIVE_SECONDS = 120  # a session is live this long after its last heartbeat
REPLY_TTL = getenv("REPLY_TTL", 3600)
SESSION_TTL = getenv("SESSION_TTL", 7 * 24 * 3600)
LEASE_MS = getenv("LEASE_MS", 10000)  # a trainer owns its pool this long after its last renewal


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
  # a reply is waited for with a blocking read of up to 30s, and redis-py gives up on a read after 5s by default
  return redis.from_url(url, socket_timeout=None)


class Bus:
  def __init__(self, r: redis.Redis | None = None):
    self.r = r if r is not None else connect(getenv("REDIS_URL", "redis://localhost:6379"))

  # ***** sessions *****

  async def session(self, session_id: str) -> None:
    """Open a session, or record a heartbeat on one."""
    k, now = key("sess", session_id), time.time()
    p = self.r.pipeline()
    p.hsetnx(k, "created", now)
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
    }

  # ***** runs *****

  async def create(self, run: Run) -> None:
    # one transaction, so a run is in its pool exactly when its stream and consumer group exist
    record = json.dumps({**asdict(run), "created": time.time()})
    p = self.r.pipeline(transaction=True)
    p.xgroup_create(key("q", run.id), GROUP, id="0", mkstream=True)
    p.sadd(key("pools"), run.pool)
    p.sadd(key("pool", run.pool), run.id)
    p.hset(
      key("run", run.id), mapping={"record": record, "steps": 0, "failed": 0, "cmds": 0, "busy_ms": 0, "wait_ms": 0}
    )
    if run.session:
      p.sadd(key("sess", run.session, "runs"), run.id)
      p.expire(key("sess", run.session, "runs"), SESSION_TTL)
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
    p.delete(key("q", run_id))
    await p.execute()

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
    got = await self.r.xreadgroup(GROUP, worker, streams, count=count, block=block_ms)

    cmds = []
    for stream, entries in got:
      run_id = stream.decode().removeprefix(key("q", ""))
      cmds += [parse_cmd(run_id, entry_id, fields) for entry_id, fields in entries]
    return cmds

  async def finish(
    self, cmd: Cmd, reply: Reply, pool: str, worker: str, repeat: bool = False, started: float | None = None
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
          await p.watch(k)
          if await p.get(k) != worker.encode():
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

  async def reclaim(self, pool: str, worker: str, min_idle_ms: int = 0) -> list[Cmd]:
    """Claim the pool's commands that have been in flight for at least min_idle_ms, on any trainer. Called by the pool's
    owner with nothing in flight itself, everything it claims was left by a trainer that died or lost the pool."""
    cmds = []
    for run_id in await self.runs(pool):
      stream = key("q", run_id)
      _, entries, _ = await self.r.xautoclaim(stream, GROUP, worker, min_idle_time=min_idle_ms, count=10000)
      cmds += [parse_cmd(run_id, entry_id, fields) for entry_id, fields in entries]
    return cmds

  # ***** pool leases *****

  async def own(self, pool: str, worker: str) -> bool:
    """Take the pool's lease if no trainer holds it."""
    return bool(await self.r.set(key("owner", pool), worker, nx=True, px=LEASE_MS))

  async def renew(self, pool: str, worker: str) -> bool:
    """Hold the pool's lease for another LEASE_MS. False if `worker` doesn't hold it anymore."""
    return bool(await self.r.set(key("owner", pool), worker, ifeq=worker, px=LEASE_MS))

  async def release(self, pool: str, worker: str) -> None:
    await self.r.delex(key("owner", pool), ifeq=worker)

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
