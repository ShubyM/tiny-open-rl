"""Runs, sessions, commands and replies, all on Redis.

Every call is a command on its run's Redis stream, tinyrl:q:<run>, read through one consumer group (`trainer`).
A reply is a stream with one entry, tinyrl:f:<request_id>, read from 0-0 without consuming it so retries see the same answer.

Keys that guard correctness:
  tinyrl:lease:<process>  TTL key renewed by each live worker process.
  tinyrl:owner:<run>      The one process holding a training run's live state; fences training replies and handoffs.
  tinyrl:run:<run>        Record, counters, and exact recovery state (`acked` stream id and `checkpoint` ref).
  tinyrl:reap:<run>       Monotonic counter watched when samplers reclaim pending prompts from a dead process's lease.
  tinyrl:vtime:<account>  Shared virtual time (tokens served) charged across an account's trainers and samplers.
  tinyrl:floor            Monotonic cluster virtual-time floor that idle accounts lift to on return.
  tinyrl:ready            Live worker processes' ready account sets, holding `tinyrl:floor` at the least-served waiter.

Sampling sessions have no single owner: any live sampler of matching spec pulls from tinyrl:q:<session> via XREADGROUP,
and surviving samplers reclaim and re-run prompts left pending by a sampler whose lease expired.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import redis.asyncio as redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff

from tinyrl.helpers import getenv

GROUP = "trainer"  # the consumer group used by node-owned readers (name retained for stream compatibility)
LIVE_SECONDS = 30  # a session is live this long after its last heartbeat, which the SDK sends every 10 s
SESSION_GRACE = getenv("SESSION_GRACE", 60)  # and its runs are deleted this long after it
REPLY_TTL = getenv("REPLY_TTL", 3600)
SESSION_TTL = getenv("SESSION_TTL", 7 * 24 * 3600)
LEASE_MS = getenv("LEASE_MS", 10000)  # a worker process is live this long after its last lease renewal
NODE_MS = 30000  # a node daemon is live this long after its last heartbeat
COUNTERS = ("steps", "failed", "cmds", "busy_ms", "wait_ms")


def key(*parts: str) -> str:
  return ":".join(("tinyrl", *parts))


def account_of(run: dict | None, run_id: str) -> str:
  """The fairness account a job charges: its client session, or a sampling session's training run, or itself."""
  return (run and (run.get("session") or run.get("train_run"))) or run_id


class Op(StrEnum):
  CREATE_MODEL = "create_model"
  FORWARD_BACKWARD = "forward_backward"
  OPTIM_STEP = "optim_step"
  SAVE_WEIGHTS = "save_weights"
  SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"
  LOAD_WEIGHTS = "load_weights"
  DELETE_MODEL = "delete_model"
  SAMPLE = "sample"


class Ev(StrEnum):
  """What changed in the cluster: runs and nodes coming and going, and the scheduler's decisions. A trace, not state."""

  RUN_CREATED = "run_created"
  ASSIGNED = "assigned"
  RUN_DELETED = "run_deleted"
  NODE_UP = "node_up"
  NODE_LOST = "node_lost"
  PLACED = "placed"
  UNPLACED = "unplaced"


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


@dataclass(frozen=True)
class Cmd:
  run: str
  entry_id: str
  op: Op
  body: bytes
  cost: int = 1  # the tokens it works on, which is what the node charges its run

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


@dataclass(frozen=True)
class Event:
  """An entry of the log, tinyrl:events. See scheduler.py."""

  id: str  # the stream id, <ms>-<n>
  t: float  # seconds, from the ms part of the id
  kind: str  # an Ev, or a kind this version doesn't know, which fold ignores
  body: dict


def trace(r, _kind: Ev, **body):
  """Add an entry to the event log, which keeps its last million. `r` is Redis, or a pipeline to queue it on."""
  return r.xadd(key("events"), {"kind": Ev(_kind), "body": json.dumps(body)}, maxlen=1_000_000, approximate=True)


def counts(cmd: Cmd, reply: Reply, started: float | None) -> dict[str, int]:
  """What a reply adds to its run's counters. A command run at time.time() `started` also counts its wait since it was queued, and how long it ran."""
  out = {"failed": 1} if not reply.ok else {"steps": 1} if cmd.op == Op.OPTIM_STEP else {}
  if started is None:
    return out
  queued_ms = int(cmd.entry_id.split("-")[0])  # a stream entry id starts with Redis' clock when it was added
  waited, ran = max(0, int(started * 1000) - queued_ms), int((time.time() - started) * 1000)
  return out | {"cmds": 1, "wait_ms": waited, "busy_ms": ran}


class Bus:
  def __init__(self, r: redis.Redis | None = None):
    self.r = r if r is not None else connect(getenv("REDIS_URL", "redis://localhost:6379"))

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
    # one transaction creates the run record, stream and consumer group
    record = json.dumps({**asdict(run), "created": time.time()})
    p = self.r.pipeline(transaction=True)
    p.xgroup_create(key("q", run.id), GROUP, id="0", mkstream=True)
    # how long XADD IDMP recognizes a retried call: Redis's longest window, a day, and its most ids
    p.execute_command("XCFGSET", key("q", run.id), "IDMP-DURATION", 86400, "IDMP-MAXSIZE", 10000)
    p.sadd(key("runs"), run.id)
    mapping = {"record": record} | dict.fromkeys(COUNTERS, 0)
    if run.worker:
      mapping["worker"] = run.worker
    p.hset(key("run", run.id), mapping=mapping)
    if run.session:
      p.sadd(key("sess", run.session, "runs"), run.id)
      p.expire(key("sess", run.session, "runs"), SESSION_TTL)
    body = {"run": run.id, "role": run.role, "base": run.base, "rank": run.rank, "exclusive": run.exclusive}
    body |= {"train_run": run.train_run, "gpus": run.gpus} | ({"worker": run.worker} if run.worker else {})
    trace(p, Ev.RUN_CREATED, **body)
    await p.execute()

  # ***** the cluster's assignment *****

  async def read_table(self) -> dict:
    """What the scheduler places from, read together: live nodes, each job's record and worker, and each worker's spec and node."""
    nodes, runs = await self.live_nodes(details=True), sorted(r.decode() for r in await self.r.smembers(key("runs")))
    p = self.r.pipeline(transaction=True)
    for run in runs:
      p.hmget(key("run", run), "record", "worker")
    p.hgetall(key("workers"))
    p.get(key("workers_made"))
    *records, workers, workers_made = await p.execute()
    return {
      "nodes": nodes,
      "runs": {r: (json.loads(rec), w and w.decode()) for r, (rec, w) in zip(runs, records) if rec},
      "workers": {k.decode(): json.loads(v) for k, v in workers.items()},
      "workers_made": int(workers_made or 0),
    }

  async def write_table(
    self, assigned: dict[str, str], workers: dict[str, dict], gone: list[str], workers_made: int, changes: list
  ) -> None:
    """The scheduler's new assignment, and a trace entry for each change, at once. Only the scheduler writes it."""
    p = self.r.pipeline(transaction=True)
    for run, worker in assigned.items():
      p.hset(key("run", run), "worker", worker)
    if workers:
      p.hset(key("workers"), mapping={n: json.dumps(w) for n, w in workers.items()})
    if gone:
      p.hdel(key("workers"), *gone)
    p.set(key("workers_made"), workers_made)
    for kind, body in changes:
      trace(p, kind, **body)
    await p.execute()

  async def get_run(self, run_id: str) -> dict | None:
    fields = await self.r.hgetall(key("run", run_id))
    if not fields:
      return None

    record = json.loads(fields[b"record"]) | {c: int(fields.get(c.encode(), 0)) for c in COUNTERS}
    for name in ("worker", "acked"):
      if name.encode() in fields:
        record[name] = fields[name.encode()].decode()
    if b"checkpoint" in fields:
      record["checkpoint"] = json.loads(fields[b"checkpoint"])
    return record

  async def join_sampling_session(self, run_id: str) -> int:
    return int(await self.r.incr(key("clients", run_id)))

  async def run_counters(self) -> dict[tuple[str, str], dict]:
    """The counters of every run still in a worker, by (worker, run)."""
    runs = sorted(r.decode() for r in await self.r.smembers(key("runs")))
    p = self.r.pipeline()
    for run_id in runs:
      p.hgetall(key("run", run_id))
    return {
      (f.get(b"worker", b"sampler").decode(), run_id): {c: int(f.get(c.encode(), 0)) for c in COUNTERS}
      for run_id, f in zip(runs, await p.execute())
      if f and (b"worker" in f or json.loads(f[b"record"]).get("role") == Role.SAMPLER)
    }

  async def delete_run(self, run_id: str, worker: str) -> None:
    """Take a deleted run out of its worker and delete its stream, in one transaction."""
    p = self.r.pipeline(transaction=True)
    p.srem(key("runs"), run_id)
    p.delete(key("q", run_id), key("owner", run_id), key("reap", run_id))
    trace(p, Ev.RUN_DELETED, run=run_id, worker=worker)
    await p.execute()

  # ***** commands *****

  async def submit_command(
    self, run_id: str, op: Op, body: bytes, seq_id: int | None = None, cost: int = 1
  ) -> str | None:
    """Queue a command on a run's stream. Returns its request id, or None if there is no such run. A call retried with the same seq_id gets the first call's request id and is queued once."""
    if seq_id is not None and (first := await self.r.get(key("seq", run_id, str(seq_id)))):
      return first.decode()  # a retry, even of a deleted run's command, a pre-IDMP one, or one older than IDMP's window
    seq = [] if seq_id is None else ["IDMP", run_id, seq_id]  # and two racing retries still queue once
    # NOMKSTREAM: an unknown or deleted run has no stream, so the write fails instead of queueing work nobody will read
    fields = ["op", op, "body", body, "cost", cost, *(["seq", seq_id] if seq else [])]
    entry_id = await self.r.execute_command("XADD", key("q", run_id), "NOMKSTREAM", *seq, "*", *fields)
    if entry_id is None:
      return None
    request_id = f"{run_id}/{entry_id.decode()}"
    if seq:  # the gateway and a repeated command look the first call's request id up by its seq_id
      await self.r.set(key("seq", run_id, str(seq_id)), request_id, ex=SESSION_TTL)
    return request_id

  async def dead_consumers(self, runs: list[str]) -> dict[str, list[str]]:
    """For each run, the stream consumers holding pending entries whose process lease is gone."""
    if not runs:
      return {}
    p, per_run, names = self.r.pipeline(transaction=False), {}, set()
    for run in runs:
      p.xinfo_consumers(key("q", run), GROUP)
    for run, info in zip(runs, await p.execute(raise_on_error=False)):
      if isinstance(info, Exception):
        if "NOGROUP" not in str(info) and "no such key" not in str(info):
          raise info
      elif holders := [c["name"].decode() for c in info if c["pending"] > 0]:
        per_run[run], names = holders, names | set(holders)
    if not names:
      return {}
    ordered, lp = sorted(names), self.r.pipeline(transaction=False)
    for name in ordered:
      lp.exists(key("lease", name))
    dead = {name for name, live in zip(ordered, await lp.execute()) if not live}
    return {r: [n for n in h if n in dead] for r, h in per_run.items() if dead.intersection(h)}

  async def reclaim_from_dead(self, run: str, dead: str, owner: str, count: int) -> list[Cmd]:
    """Atomically move up to `count` pending entries from `dead` to `owner` if `dead`'s lease is still gone."""
    stream, lease_key, reap_key = key("q", run), key("lease", dead), key("reap", run)
    while True:
      with contextlib.suppress(redis.WatchError):
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(lease_key, reap_key)
          if await p.exists(lease_key):
            return []
          try:
            pending = await p.xpending_range(stream, GROUP, min="-", max="+", count=count, consumername=dead)
          except redis.ResponseError as e:
            if "NOGROUP" in str(e):
              return []
            raise
          p.multi()
          if not pending:
            p.xgroup_delconsumer(stream, GROUP, dead)
            await p.execute()
            return []
          p.incr(reap_key)
          p.expire(reap_key, 60)
          p.xclaim(stream, GROUP, owner, 0, [e["message_id"] for e in pending])
          if len(pending) < count:
            p.xgroup_delconsumer(stream, GROUP, dead)
          return [parse_cmd(run, *entry) for entry in (await p.execute())[2] if entry]

  async def peek_commands(self, runs: list[str], block_ms: int = 0, reclaim: bool = False) -> list[Cmd]:
    """Unclaimed heads (and dead-sampler pending heads when `reclaim` is True), in arrival order."""
    heads: dict[str, Cmd] = {}
    if reclaim:
      for run, dead_list in (await self.dead_consumers(runs)).items():
        with contextlib.suppress(redis.ResponseError):
          if pend := await self.r.xpending_range(
            key("q", run), GROUP, min="-", max="+", count=1, consumername=dead_list[0]
          ):
            mid = pend[0]["message_id"]
            if entries := await self.r.xrange(key("q", run), min=mid, max=mid, count=1):
              heads[run] = parse_cmd(run, *entries[0])

    rest = [r for r in runs if r not in heads]
    p = self.r.pipeline(transaction=False)
    for run in rest:
      p.xinfo_groups(key("q", run))
    cursors = {}
    for run, groups in zip(rest, await p.execute(raise_on_error=False)):
      if isinstance(groups, Exception):
        if "no such key" not in str(groups):
          raise groups
        continue
      if group := next((g for g in groups if g["name"] == GROUP.encode()), None):
        cursors[key("q", run)] = group["last-delivered-id"]
    if not cursors and not heads:
      await asyncio.sleep(block_ms / 1000)
      return []
    if cursors:
      wait = 0 if heads else block_ms
      got = await self.r.xread(cursors, count=1, block=wait or None)
      for stream, entries in got:
        run = stream.decode().removeprefix(key("q", ""))
        heads[run] = parse_cmd(run, *entries[0])
    return sorted(heads.values(), key=lambda c: (*map(int, c.entry_id.split("-")), c.run))

  async def take_commands(
    self, runs: list[str], owner: str, count: int = 16, block_ms: int = 0, reclaim: bool = False
  ) -> list[Cmd]:
    """Claim only the selected work, reclaiming dead-sampler prompts first when `reclaim` is True."""
    cmds, need = [], dict.fromkeys(runs, count)
    if reclaim:
      for run, dead_list in (await self.dead_consumers(runs)).items():
        for dead in dead_list:
          if need[run] <= 0:
            break
          got = await self.reclaim_from_dead(run, dead, owner, need[run])
          cmds.extend(got)
          need[run] -= len(got)

    left = [r for r in runs if need[r] > 0]
    if left:
      p = self.r.pipeline(transaction=False)
      for run in left:
        p.xreadgroup(GROUP, owner, {key("q", run): ">"}, count=need[run])
      for run, got in zip(left, await p.execute(raise_on_error=False)):
        if isinstance(got, Exception):
          if "NOGROUP" not in str(got):
            raise got
          continue
        cmds += [parse_cmd(run, entry_id, fields) for _, entries in got for entry_id, fields in entries]
    if not cmds and block_ms:
      await self.peek_commands(runs, block_ms, reclaim=reclaim)
      return await self.take_commands(runs, owner, count, reclaim=reclaim)
    return cmds

  async def write_reply(
    self,
    cmd: Cmd,
    reply: Reply,
    owner: str,
    started: float | None = None,
    mutated: bool = False,
    checkpoint: dict | None = None,
    account: str = "",
  ) -> bool:
    """Write a command's reply, acknowledge the command, and count it, all at once.

    A training command commits iff `owner` still holds `tinyrl:owner:<run>`, updating `acked` and `checkpoint` together.
    A sampling command commits iff `owner`'s process lease is live and no reply has been written for `request_id` yet.
    """
    added, deleting = counts(cmd, reply, started), cmd.op == Op.DELETE_MODEL and reply.ok
    row, stream, owner_key = key("run", cmd.run), key("q", cmd.run), key("owner", cmd.run)
    lease_key, reply_key, registry = key("lease", owner), key("f", cmd.request_id), key("ckpt", cmd.run)
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          if cmd.op == Op.SAMPLE:
            await p.watch(lease_key, reply_key)
            if not await p.exists(lease_key) or await p.xlen(reply_key):
              return False
            checkpoints = {}
          else:
            await p.watch(owner_key, *([registry] if deleting else []))
            if await p.get(owner_key) != owner.encode():
              return False
            checkpoints = await p.hgetall(registry) if deleting else {}

          p.multi()
          fields = {"ok": int(reply.ok), "content_type": reply.content_type, "body": reply.body}
          p.xadd(reply_key, fields, maxlen=1, approximate=False)
          p.expire(reply_key, REPLY_TTL)
          p.xack(stream, GROUP, cmd.entry_id)
          p.xdel(stream, cmd.entry_id)
          for name, n in added.items():
            p.hincrby(row, name, n)
          if started is not None:
            vkey = key("vtime", account or cmd.run)
            p.incrby(vkey, max(1, cmd.cost))
            p.expire(vkey, SESSION_TTL)
          if mutated:
            p.hset(row, "acked", cmd.entry_id)
            p.hdel(row, "checkpoint")
          if checkpoint:
            p.hset(row, "checkpoint", json.dumps(checkpoint))
          if deleting:
            # Completion includes deletion; a client observing its reply sees no owned checkpoint or live queue.
            for cid, data in checkpoints.items():
              if json.loads(data)["owned"]:
                uri = f"tinker://{cmd.run}/{cid.decode()}"
                p.hdel(registry, cid)
                p.zrem(key("ckpts"), uri)
                p.zrem(key("expiry"), uri)
            p.srem(key("runs"), cmd.run)
            p.delete(stream, owner_key, key("reap", cmd.run))
            trace(p, Ev.RUN_DELETED, run=cmd.run)
          await p.execute()
          return True
      except redis.WatchError:
        continue

  async def fail_stranded_commands(self, runs: list[str], owner: str, min_idle_ms: int = 0) -> None:
    """Fail the commands of training runs `owner` owns that have been in flight for at least min_idle_ms, on any trainer."""
    for run_id in runs:
      stream = key("q", run_id)
      try:
        _, entries, _ = await self.r.xautoclaim(stream, GROUP, owner, min_idle_time=min_idle_ms, count=10000)
      except redis.ResponseError as e:
        if "NOGROUP" not in str(e):
          raise
        continue  # a run deleted after it was listed has nothing left to claim
      for entry_id, fields in entries:
        cmd = parse_cmd(run_id, entry_id, fields)
        reply = Reply.error(RuntimeError(f"its worker died, or lost the worker, during {cmd.op}"))
        await self.write_reply(cmd, reply, owner)

  # ***** training run ownership and process leases *****

  async def take_ownership(self, run_id: str, owner: str) -> bool:
    """Become a training run's owner if it has none, or if its previous owner's lease ran out."""
    owner_key = key("owner", run_id)
    while True:
      with contextlib.suppress(redis.WatchError):
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(owner_key)
          if (current := await p.get(owner_key)) == owner.encode():
            return True
          if current is not None:
            await p.watch(lease_key := key("lease", current.decode()))
            if await p.exists(lease_key):
              return False
          p.multi()
          p.set(owner_key, owner)
          await p.execute()
          return True

  async def hand_off(self, run_id: str, owner: str, checkpoint: dict | None = None) -> bool:
    """Give up a training run if `owner` owns it, recording `checkpoint` for the next owner to restore."""
    owner_key = key("owner", run_id)
    while True:
      with contextlib.suppress(redis.WatchError):
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(owner_key)
          if await p.get(owner_key) != owner.encode():
            return False
          p.multi()
          if checkpoint:
            p.hset(key("run", run_id), "checkpoint", json.dumps(checkpoint))
          p.delete(owner_key)
          await p.execute()
          return True

  async def run_owner(self, run_id: str) -> str | None:
    owner = await self.r.get(key("owner", run_id))
    return owner.decode() if owner else None

  async def renew_lease(self, owner: str) -> None:
    """Mark a worker process live for another LEASE_MS."""
    await self.r.set(key("lease", owner), 1, px=LEASE_MS)

  async def release_lease(self, owner: str) -> None:
    p = self.r.pipeline(transaction=True)
    p.delete(key("lease", owner))
    p.hdel(key("ready"), owner)
    await p.execute()

  async def lease_live(self, owner: str) -> bool:
    return bool(await self.r.exists(key("lease", owner)))

  # ***** shared per-account virtual time *****

  async def _raise(self, k: str, floor: int, ex: int | None = None) -> int:
    while True:
      with contextlib.suppress(redis.WatchError):
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(k)
          if (cur := int(await p.get(k) or 0)) >= floor:
            return cur
          p.multi()
          p.set(k, floor, ex=ex)
          await p.execute()
          return floor

  async def sync_vtime(self, owner: str, accounts: list[str], extra: list[str] = ()) -> dict[str, int]:
    """Publish `owner`'s ready accounts, lift any idle returners to the cluster floor, and return their virtual times."""
    rkey, p = key("ready"), self.r.pipeline(transaction=True)
    p.hset(rkey, owner, json.dumps(accounts)) if accounts else p.hdel(rkey, owner)
    p.hgetall(rkey)
    p.get(key("floor"))
    _, raw_ready, raw_floor = await p.execute()
    holders, live = sorted(k.decode() for k in raw_ready), set()
    if holders:
      lp = self.r.pipeline(transaction=False)
      for h in holders:
        lp.exists(key("lease", h))
      live = {h for h, ok in zip(holders, await lp.execute()) if ok}
      if dead := [h for h in holders if h not in live]:
        await self.r.hdel(rkey, *dead)
    if not (ordered := sorted({a for h in live for a in json.loads(raw_ready[h.encode()])} | {*accounts, *extra})):
      return {}
    vtimes = {a: int(v or 0) for a, v in zip(ordered, await self.r.mget(*(key("vtime", a) for a in ordered)))}
    floor = int(raw_floor or 0)
    for a in accounts:
      if vtimes[a] < floor:
        vtimes[a] = await self._raise(key("vtime", a), floor, SESSION_TTL)
    if (new_floor := max(floor, min(vtimes.values()))) > floor:
      await self._raise(key("floor"), new_floor)
    return {a: vtimes[a] for a in accounts}

  # ***** nodes *****

  async def node_heartbeat(
    self, node: str, gpus: int, memory: int = 0, workers: dict | None = None, lag: float = 0.0
  ) -> None:
    beat = {"gpus": gpus, "memory": memory, "until": time.time() + NODE_MS / 1000, "workers": workers or {}, "lag": lag}
    await self.r.hset(key("nodes"), node, json.dumps(beat))

  async def live_nodes(self, details: bool = False) -> dict:
    """The GPU count of every live node daemon, by id."""
    nodes = {k.decode(): json.loads(v) for k, v in (await self.r.hgetall(key("nodes"))).items()}
    if dead := [k for k, v in nodes.items() if v["until"] < time.time()]:
      await self.r.hdel(key("nodes"), *dead)
    return {k: v if details else v["gpus"] for k, v in nodes.items() if k not in dead}

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

  # ***** events *****

  async def append_event(self, _kind: Ev, **body) -> str:
    return (await trace(self.r, _kind, **body)).decode()

  async def read_events(self, after: str = "0-0", count: int = 10_000, block_ms: int = 0) -> list[Event]:
    """The entries after stream id `after`, waiting up to block_ms for one, or not at all if it is 0."""
    got = await self.r.xread({key("events"): after}, count=count, block=block_ms or None)
    out = []
    for entry_id, fields in got[0][1] if got else []:
      entry_id = entry_id.decode()
      t = int(entry_id.split("-")[0]) / 1000
      out.append(Event(entry_id, t, fields[b"kind"].decode(), json.loads(fields[b"body"])))
    return out
