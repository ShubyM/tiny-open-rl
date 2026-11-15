"""Runs, sessions, commands and replies, all on Redis.

Every training call is a command on its run's Redis stream, tinyrl:q:<run>. The node owning a worker reads its runs' streams through one consumer group, so Redis itself knows:

  what is queued       XINFO GROUPS, the group's lag
  what is in flight    XPENDING: which owner took each command, and how long ago
  what a dead trainer  XAUTOCLAIM, so the next trainer can fail it instead of leaving a client waiting
  left unfinished

A reply is a stream with a single entry, tinyrl:f:<request_id>. XREAD from id 0 blocks until it is written and never consumes it, so a client can retry retrieve_future, or many clients can wait on one, and every one gets the same answer.

A run has one owner, tinyrl:owner:<run>: the worker process whose memory holds its live state. Replies are written in a WATCH/MULTI/EXEC transaction on that key, so a process that lost the run can't answer for it. A process is live while it renews its lease, tinyrl:lease:<process>, and the scheduler frees the runs of one that stopped, say after a long pause.

A run's row carries what restores it elsewhere: `acked`, the stream id of the last command that changed its state, and `checkpoint`, where that state is saved. The reply that moves `acked` clears `checkpoint` in the same transaction, so a run has a checkpoint only while it restores the run exactly.
"""

from __future__ import annotations

import asyncio
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
LEASE_MS = getenv("LEASE_MS", 10000)  # a worker process is live this long after its last renewal
NODE_MS = 30000  # a node daemon is live this long after its last heartbeat
COUNTERS = ("steps", "failed", "cmds", "busy_ms", "wait_ms")


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
    p.xadd(key("events"), {"kind": Ev.RUN_CREATED, "body": json.dumps(body)})
    await p.execute()

  # ***** the cluster's assignment *****

  async def read_table(self) -> dict:
    """What the scheduler decides from, read together: live nodes, each job's record and worker, each worker's spec, node and GPUs, the workers holding their GPUs on each node, the accounts each worker has work ready for, and the service every account has had."""
    nodes, runs = await self.live_nodes(details=True), sorted(r.decode() for r in await self.r.smembers(key("runs")))
    p = self.r.pipeline(transaction=True)
    for run in runs:
      p.hmget(key("run", run), "record", "worker")
    for node in nodes:
      p.hgetall(key("granted", node))
    for name in ("workers", "ready", "service"):
      p.hgetall(key(name))
    p.mget(key("workers_made"), key("floor"))
    *rows, workers, ready, service, (workers_made, floor) = await p.execute()
    records, granted = rows[: len(runs)], rows[len(runs) :]
    return {
      "nodes": nodes,
      "runs": {r: (json.loads(rec), w and w.decode()) for r, (rec, w) in zip(runs, records) if rec},
      "workers": {k.decode(): json.loads(v) for k, v in workers.items()},
      "workers_made": int(workers_made or 0),
      "granted": {n: {w.decode(): json.loads(g) for w, g in held.items()} for n, held in zip(nodes, granted)},
      "ready": {k.decode(): json.loads(v) for k, v in ready.items()},
      "service": {k.decode(): int(v) for k, v in service.items()},
      "floor": int(floor or 0),
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
      p.hdel(key("ready"), *gone)
    p.set(key("workers_made"), workers_made)
    for kind, body in changes:
      p.xadd(key("events"), {"kind": kind, "body": json.dumps(body)}, maxlen=1_000_000, approximate=True)
    await p.execute()

  # ***** GPUs: the scheduler grants them, and only the node they are on gives them back *****

  async def write_grants(self, grants: dict[str, tuple[str, tuple]], lifts: dict[str, int], floor: int) -> None:
    """The workers that may use their GPUs from now on, and the service floor they were ordered by. A lift is added to an account's service, never written over it, since nodes charge accounts meanwhile."""
    p = self.r.pipeline(transaction=True)
    for worker, (node, gpus) in grants.items():
      p.hset(key("granted", node), worker, json.dumps(gpus))
    for account, lift in lifts.items():
      p.hincrby(key("service"), account, lift)
    p.set(key("floor"), floor)
    await p.execute()

  async def release_gpus(self, node: str, worker: str) -> None:
    """A node gives back a worker's GPUs, once its process has left them or is gone."""
    await self.r.hdel(key("granted", node), worker)

  async def set_ready(self, worker: str, accounts: list[str] | None) -> None:
    """The accounts a worker has work ready for, which is what asks the scheduler for its GPUs. None when it has none."""
    if accounts is None:
      await self.r.hdel(key("ready"), worker)
    else:
      await self.r.hset(key("ready"), worker, json.dumps(accounts))

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
      (f[b"worker"].decode(), run_id): {c: int(f.get(c.encode(), 0)) for c in COUNTERS}
      for run_id, f in zip(runs, await p.execute())
      if b"worker" in f
    }

  async def delete_run(self, run_id: str, worker: str) -> None:
    """Take a deleted run out of its worker and delete its stream, in one transaction."""
    p = self.r.pipeline(transaction=True)
    p.srem(key("runs"), run_id)
    p.delete(key("q", run_id), key("owner", run_id))
    await p.execute()
    await self.append_event(Ev.RUN_DELETED, run=run_id, worker=worker)

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

  async def peek_commands(self, runs: list[str], block_ms: int = 0) -> list[Cmd]:
    """Unclaimed heads, in arrival order. Looking and preparing never change stream ownership."""
    p = self.r.pipeline(transaction=False)
    for run in runs:
      p.xinfo_groups(key("q", run))
    cursors = {}
    for run, groups in zip(runs, await p.execute(raise_on_error=False)):
      if isinstance(groups, Exception):
        if "no such key" not in str(groups):
          raise groups
        continue
      if group := next((g for g in groups if g["name"] == GROUP.encode()), None):
        cursors[key("q", run)] = group["last-delivered-id"]
    if not cursors:
      await asyncio.sleep(block_ms / 1000)
      return []
    got = await self.r.xread(cursors, count=1, block=block_ms or None)
    cmds = [parse_cmd(stream.decode().removeprefix(key("q", "")), *entries[0]) for stream, entries in got]
    return sorted(cmds, key=lambda c: (*map(int, c.entry_id.split("-")), c.run))

  async def take_commands(self, runs: list[str], owner: str, count: int = 16, block_ms: int = 0) -> list[Cmd]:
    """Claim only the selected work. A stream deleted during admission cannot hide another run."""
    p = self.r.pipeline(transaction=False)
    for run in runs:
      p.xreadgroup(GROUP, owner, {key("q", run): ">"}, count=count)
    cmds = []
    for run, got in zip(runs, await p.execute(raise_on_error=False)):
      if isinstance(got, Exception):
        if "NOGROUP" not in str(got):
          raise got
        continue
      cmds += [parse_cmd(run, entry_id, fields) for _, entries in got for entry_id, fields in entries]
    if not cmds and block_ms:
      await self.peek_commands(runs, block_ms)
      return await self.take_commands(runs, owner, count)
    return cmds

  async def write_reply(
    self,
    cmd: Cmd,
    reply: Reply,
    owner: str,
    started: float | None = None,
    account: str = "",
    mutated: bool = False,
    checkpoint: dict | None = None,
  ) -> bool:
    """Write a command's reply, acknowledge the command, and count it, all at once, if `owner` owns the run. Returns False, having written nothing, if it doesn't: the run's next owner fails the command. A command run at time.time() `started` also counts its wait since it was queued and how long it ran, and charges its tokens to `account`. One that `mutated` the run's state leaves its checkpoint behind, unless `checkpoint` says where the new state is saved."""
    counts = {"failed": 1} if not reply.ok else {"steps": 1} if cmd.op == Op.OPTIM_STEP else {}
    if started is not None:
      queued_ms = int(cmd.entry_id.split("-")[0])  # a stream entry id starts with Redis' clock when it was added
      counts |= {
        "cmds": 1,
        "wait_ms": max(0, int(started * 1000) - queued_ms),
        "busy_ms": int((time.time() - started) * 1000),
      }

    owned, run = key("owner", cmd.run), key("run", cmd.run)
    deleting = cmd.op == Op.DELETE_MODEL and reply.ok
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(owned, *([key("ckpt", cmd.run)] if deleting else []))
          if await p.get(owned) != owner.encode():
            await p.unwatch()
            return False
          checkpoints = await p.hgetall(key("ckpt", cmd.run)) if deleting else {}
          p.multi()
          fields = {"ok": int(reply.ok), "content_type": reply.content_type, "body": reply.body}
          p.xadd(key("f", cmd.request_id), fields, maxlen=1, approximate=False)
          p.expire(key("f", cmd.request_id), REPLY_TTL)
          p.xack(key("q", cmd.run), GROUP, cmd.entry_id)
          p.xdel(key("q", cmd.run), cmd.entry_id)
          for name, n in counts.items():
            p.hincrby(run, name, n)
          if mutated:
            p.hset(run, "acked", cmd.entry_id)
            p.hdel(run, "checkpoint")
          if checkpoint:
            p.hset(run, "checkpoint", json.dumps(checkpoint))
          if account:
            p.hincrby(key("service"), account, cmd.cost)
          if deleting:
            # Completion includes deletion; a client observing its reply sees no owned checkpoint or live queue.
            for cid, data in checkpoints.items():
              if json.loads(data)["owned"]:
                uri = f"tinker://{cmd.run}/{cid.decode()}"
                p.hdel(key("ckpt", cmd.run), cid)
                p.zrem(key("ckpts"), uri)
                p.zrem(key("expiry"), uri)
            p.srem(key("runs"), cmd.run)
            p.delete(key("q", cmd.run), owned)
            p.xadd(
              key("events"),
              {"kind": Ev.RUN_DELETED, "body": json.dumps({"run": cmd.run})},
              maxlen=1_000_000,
              approximate=True,
            )
          await p.execute()
          return True
      except redis.WatchError:
        continue

  async def fail_stranded_commands(self, runs: list[str], owner: str, min_idle_ms: int = 0) -> None:
    """Fail the commands of runs `owner` owns that have been in flight for at least min_idle_ms, on any process. Called by the node only when it has nothing in flight; claims are failed, never replayed."""
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

  # ***** owners: the process whose memory holds a run's live state *****

  async def take_ownership(self, run_id: str, owner: str) -> bool:
    """Become a run's owner, if it has none: its last owner handed it off, or the scheduler freed it from a dead one."""
    return bool(await self.r.set(key("owner", run_id), owner, nx=True))

  async def hand_off(self, run_id: str, owner: str, checkpoint: dict | None = None) -> bool:
    """Give up a run, if `owner` owns it. `checkpoint` is where its state was just saved, which the next owner restores, so the run loses nothing."""
    owned = key("owner", run_id)
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(owned)
          if await p.get(owned) != owner.encode():
            await p.unwatch()
            return False
          p.multi()
          if checkpoint:
            p.hset(key("run", run_id), "checkpoint", json.dumps(checkpoint))
          p.delete(owned)
          await p.execute()
          return True
      except redis.WatchError:  # the scheduler freed the run
        continue

  async def free_orphaned_runs(self) -> None:
    """Take runs from owners whose lease ran out: the process died, or its node did. The next owner restores each from its checkpoint, if it has one."""
    runs = [r.decode() for r in await self.r.smembers(key("runs"))]
    owners = await self.r.mget([key("owner", r) for r in runs]) if runs else []
    held = [(run, owner) for run, owner in zip(runs, owners) if owner]
    p = self.r.pipeline()
    for _, owner in held:
      p.exists(key("lease", owner.decode()))
    dead = [(run, owner) for (run, owner), live in zip(held, await p.execute()) if not live]
    if not dead:
      return
    leases = [key("lease", owner.decode()) for _, owner in dead]
    try:
      async with self.r.pipeline(transaction=True) as p:
        await p.watch(*leases)  # a process back from a long pause renews its lease, and keeps its runs
        if not await p.exists(*leases):
          p.multi()
          for run, owner in dead:
            p.delex(key("owner", run), ifeq=owner)
          await p.execute()
    except redis.WatchError:
      pass  # the next look frees the rest

  # ***** leases: a worker process is live while it renews its own *****

  async def renew_lease(self, owner: str) -> None:
    """A process is live for another LEASE_MS. One back from a pause longer than that keeps the runs nobody freed meanwhile."""
    await self.r.set(key("lease", owner), 1, px=LEASE_MS)

  async def release_lease(self, owner: str) -> None:
    await self.r.delete(key("lease", owner))

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
      await self.r.delete(*(key("granted", n) for n in dead))  # its GPUs left with it
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
    entry_id = await self.r.xadd(
      key("events"), {"kind": Ev(_kind), "body": json.dumps(body)}, maxlen=1_000_000, approximate=True
    )
    return entry_id.decode()

  async def read_events(self, after: str = "0-0", count: int = 10_000, block_ms: int = 0) -> list[Event]:
    """The entries after stream id `after`, waiting up to block_ms for one, or not at all if it is 0."""
    got = await self.r.xread({key("events"): after}, count=count, block=block_ms or None)
    out = []
    for entry_id, fields in got[0][1] if got else []:
      entry_id = entry_id.decode()
      t = int(entry_id.split("-")[0]) / 1000
      out.append(Event(entry_id, t, fields[b"kind"].decode(), json.loads(fields[b"body"])))
    return out
