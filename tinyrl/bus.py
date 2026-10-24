"""Runs, sessions, commands and replies, all on Redis.

Every training call is a command on its run's Redis stream, tinyrl:q:<run>. The trainer for a pool reads its runs'
streams through one consumer group, so Redis itself knows:

  what is queued       XINFO GROUPS, the group's lag
  what is in flight    XPENDING: which trainer took each command, and how long ago
  what a dead trainer  XAUTOCLAIM, so the next trainer can fail it instead of leaving a client waiting
  left unfinished

A reply is a stream with a single entry, tinyrl:f:<request_id>. XREAD from id 0 blocks until it is written and never
consumes it, so a client can retry retrieve_future, or many clients can wait on one, and every one gets the same answer.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass

import redis.asyncio as redis

from tinyrl.helpers import getenv

GROUP = "trainer"  # the consumer group every pool's trainer reads through
LIVE_SECONDS = 120  # a session is live this long after its last heartbeat
REPLY_TTL = getenv("REPLY_TTL", 3600)
SESSION_TTL = getenv("SESSION_TTL", 7 * 24 * 3600)


def key(*parts: str) -> str:
  return ":".join(("tinyrl", *parts))


@dataclass(frozen=True)
class Run:
  """A model being trained. A rank of 0 means a full fine-tune."""

  id: str
  pool: str
  base: str
  rank: int
  session: str = ""


@dataclass(frozen=True)
class Cmd:
  """A command a trainer took from a run's stream."""

  run: str
  id: str  # the stream entry id
  op: str
  body: bytes

  @property
  def request_id(self) -> str:
    return f"{self.run}/{self.id}"


@dataclass(frozen=True)
class Reply:
  ok: bool
  content_type: str
  body: bytes


def parse_cmd(run: str, entry_id: bytes, fields: dict[bytes, bytes]) -> Cmd:
  return Cmd(run, entry_id.decode(), fields[b"op"].decode(), fields[b"body"])


class Bus:
  def __init__(self, r: redis.Redis | None = None):
    self.r = r if r is not None else redis.from_url(getenv("REDIS_URL", "redis://localhost:6379"))

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
    p.sadd(key("pool", run.pool), run.id)
    p.hset(key("run", run.id), mapping={"record": record, "steps": 0, "failed": 0})
    if run.session:
      p.sadd(key("sess", run.session, "runs"), run.id)
      p.expire(key("sess", run.session, "runs"), SESSION_TTL)
    await p.execute()

  async def run(self, run_id: str) -> dict | None:
    fields = await self.r.hgetall(key("run", run_id))
    if not fields:
      return None

    record = json.loads(fields.pop(b"record"))
    counters = {name.decode(): int(value) for name, value in fields.items()}  # steps and failed
    return {**record, **counters}

  async def runs(self, pool: str) -> list[str]:
    return [run_id.decode() for run_id in await self.r.smembers(key("pool", pool))]

  async def drop(self, run_id: str, pool: str) -> None:
    """Take a deleted run out of its pool and delete its stream, in one transaction."""
    p = self.r.pipeline(transaction=True)
    p.srem(key("pool", pool), run_id)
    p.delete(key("q", run_id))
    await p.execute()

  # ***** commands *****

  async def submit(self, run_id: str, op: str, body: bytes) -> str | None:
    """Queue a command on a run's stream. Returns its request id, or None if there is no such run."""
    # NOMKSTREAM: an unknown or deleted run has no stream, so the write fails instead of queueing work nobody will read
    entry_id = await self.r.xadd(key("q", run_id), {"op": op, "body": body}, nomkstream=True)
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

  async def finish(self, cmd: Cmd, reply: Reply) -> None:
    """Write a command's reply, acknowledge the command, and count it, all in one transaction."""
    reply_key, stream = key("f", cmd.request_id), key("q", cmd.run)
    p = self.r.pipeline(transaction=True)
    p.xadd(reply_key, {"ok": int(reply.ok), "content_type": reply.content_type, "body": reply.body}, maxlen=1)
    p.expire(reply_key, REPLY_TTL)
    p.xack(stream, GROUP, cmd.id)
    p.xdel(stream, cmd.id)
    if not reply.ok:
      p.hincrby(key("run", cmd.run), "failed", 1)
    elif cmd.op == "optim_step":
      p.hincrby(key("run", cmd.run), "steps", 1)
    await p.execute()

  async def reclaim(self, pool: str, worker: str) -> list[Cmd]:
    """Claim every command still in flight in a pool. A pool has one trainer, so when a trainer starts, any command in
    flight was taken by one that died."""
    cmds = []
    for run_id in await self.runs(pool):
      # a trainer has one take in flight at most, so a single claim gets everything
      _, entries, _ = await self.r.xautoclaim(key("q", run_id), GROUP, worker, min_idle_time=0, count=10000)
      cmds += [parse_cmd(run_id, entry_id, fields) for entry_id, fields in entries]
    return cmds

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
