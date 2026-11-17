"""Runs, sessions, commands, replies, ownership and plans, all on Redis.

Safety rests on two Redis facts (plus each node's local `on` table):
  tinyrl:run:<r>       a training run's single-writer `(owner, epoch, pos, snap)`, fenced on every commit and detach
  tinyrl:lease:<life>  each live process incarnation's TTL; `reap_run` clears `owner` only after its lease expires
Sampling sessions are stateless readers: replicas share consumer group `q:<s>` and `commit_sample` answers & bills once.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import redis.asyncio as redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff

from tinyrl.control.accounting import Charge
from tinyrl.helpers import getenv

GROUP = "trainer"
LIVE_SECONDS, SESSION_GRACE = 30, getenv("SESSION_GRACE", 60)
REPLY_TTL, SESSION_TTL = getenv("REPLY_TTL", 3600), getenv("SESSION_TTL", 7 * 24 * 3600)
LEASE_MS, NODE_MS = getenv("LEASE_MS", 10000), 30000
RUN_META = frozenset((b"record", b"owner", b"epoch", b"pos", b"snap"))


def key(*parts: str) -> str:
  return ":".join(("tinyrl", *parts))


def wake(p, stream: str = "wake") -> None:
  p.xadd(key(stream), {"t": 1}, maxlen=1, approximate=False)


def unpack_snap(snap: str) -> tuple[str, str]:
  uri, version = snap.split("\n", 1)
  return uri, version


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
  base: str
  rank: int
  session: str = ""
  model_path: str | None = None
  lora: dict = field(default_factory=dict)
  exclusive: bool = False
  role: Role = Role.TRAINER
  train_run: str | None = None
  gpus: int = 1
  account: str = ""


@dataclass(frozen=True)
class Cmd:
  run: str
  entry_id: str
  op: Op
  body: bytes
  cost: int = 1

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
    cat = "user" if isinstance(exc, (ValueError, KeyError)) else "server"
    return cls(
      False, "application/json", json.dumps({"error": f"{type(exc).__name__}: {exc}", "category": cat}).encode()
    )


@dataclass(frozen=True)
class Event:
  id: str
  t: float
  kind: str
  body: dict


def parse_cmd(run: str, entry_id: bytes, fields: dict[bytes, bytes]) -> Cmd:
  return Cmd(run, entry_id.decode(), Op(fields[b"op"].decode()), fields[b"body"], int(fields.get(b"cost", 1)))


def finish_cmd(p, cmd: Cmd, reply: Reply, acct: str, charge: Charge | None) -> None:
  p.xack(key("q", cmd.run), GROUP, cmd.entry_id)
  p.xdel(key("q", cmd.run), cmd.entry_id)
  fk = key("f", cmd.request_id)
  fields = {"ok": int(reply.ok), "content_type": reply.content_type, "body": reply.body}
  p.xadd(fk, fields, maxlen=1, approximate=False)
  p.expire(fk, REPLY_TTL)
  counts = {"failed": 1} if not reply.ok else {"steps": 1} if cmd.op == Op.OPTIM_STEP else {}
  if charge:
    wait_ms = max(0, charge.started_ms - int(cmd.entry_id.split("-")[0]))
    counts.update(cmds=1, wait_ms=wait_ms, busy_ms=charge.busy_ms)
    if reply.ok:
      counts["tokens"] = cmd.cost
    if acct and charge.gpu_ms:
      p.hincrbyfloat(key("service"), acct, charge.gpu_ms)
  for name, n in counts.items():
    p.hincrby(key("run", cmd.run), name, n)
  wake(p)


def queue_delete(p, run_id: str) -> None:
  p.srem(key("runs"), run_id)
  p.delete(key("q", run_id))
  p.hdel(key("run", run_id), "owner", "snap")
  p.hdel(key("writing"), run_id)
  p.xadd(
    key("events"), {"kind": Ev.RUN_DELETED, "body": json.dumps({"run": run_id})}, maxlen=1_000_000, approximate=True
  )


def queue_checkpoint(p, uri: str, size: int, *, version: str, ttl=None, owned=False) -> None:
  """Queue a checkpoint's metadata and indexes in the same transaction as its publication."""
  run, cid = uri.removeprefix("tinker://").split("/", 1)
  kind, now = cid.split("/", 1)[0], time.time()
  kind = {"weights": "training", "sampler_weights": "sampler"}.get(kind, kind)
  info = {"checkpoint_id": cid, "checkpoint_type": kind, "time": now, "tinker_path": uri, "size_bytes": size}
  expires = now + ttl if ttl else None
  info.update(expires_at=expires, owned=owned, version=version)
  p.hset(key("ckpt", run), cid, json.dumps(info))
  if kind != "parked":
    p.zadd(key("ckpts"), {uri: now})
  p.zadd(key("expiry"), {uri: expires}) if expires else p.zrem(key("expiry"), uri)


def connect(url: str) -> redis.Redis:
  retry = Retry(ExponentialBackoff(cap=2.0, base=0.1), 30)
  return redis.from_url(url, socket_timeout=None, retry=retry, max_connections=10_000)


class Bus:
  def __init__(self, r: redis.Redis | None = None):
    self.r = r if r is not None else connect(getenv("REDIS_URL", "redis://localhost:6379"))

  async def transact(self, fn, *keys: str):
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(*keys)
          return await fn(p)
      except redis.WatchError:
        continue

  # ***** sessions *****

  async def heartbeat_session(self, session_id: str, metadata: dict | None = None) -> None:
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
    seen, meta = float(record[b"seen"]), json.loads(record.get(b"metadata") or "{}")
    out = {"session_id": session_id, "created": float(record[b"created"]), "seen": seen}
    return out | {"live": time.time() - seen < LIVE_SECONDS, "runs": sorted(r.decode() for r in runs), "metadata": meta}

  async def expired_sessions(self) -> list[str]:
    return [s.decode() for s in await self.r.zrangebyscore(key("seen"), 0, time.time() - SESSION_GRACE)]

  async def finish_session(self, session_id: str) -> None:
    p = self.r.pipeline()
    p.zadd(key("seen"), {session_id: 0}, xx=True)
    wake(p)
    await p.execute()

  async def end_session(self, session_id: str) -> None:
    left = False
    for run_id in (await self.get_session(session_id) or {}).get("runs", []):
      role = (await self.get_run(run_id) or {}).get("role")
      if not await self.r.sismember(key("runs"), run_id):
        continue
      if role == Role.SAMPLER:
        await self.delete_run(run_id)
      elif role == Role.TRAINER:
        left = bool(await self.submit_command(run_id, Op.DELETE_MODEL, b"{}")) or left
    await (self.r.zadd(key("seen"), {session_id: time.time()}) if left else self.r.zrem(key("seen"), session_id))

  # ***** runs and cluster state *****

  async def create_run(self, run: Run) -> None:
    zero = dict.fromkeys(("steps", "failed", "cmds", "tokens", "busy_ms", "wait_ms"), 0)
    p = self.r.pipeline(transaction=True)
    p.xgroup_create(key("q", run.id), GROUP, id="0", mkstream=True)
    p.execute_command("XCFGSET", key("q", run.id), "IDMP-DURATION", 86400, "IDMP-MAXSIZE", 10000)
    p.sadd(key("runs"), run.id)
    p.hset(key("run", run.id), mapping={"record": json.dumps({**asdict(run), "created": time.time()}), **zero})
    if run.session:
      p.sadd(key("sess", run.session, "runs"), run.id)
      p.expire(key("sess", run.session, "runs"), SESSION_TTL)
    ev = {"run": run.id, "role": run.role, "base": run.base, "rank": run.rank, "exclusive": run.exclusive}
    p.xadd(
      key("events"), {"kind": Ev.RUN_CREATED, "body": json.dumps(ev | {"train_run": run.train_run, "gpus": run.gpus})}
    )
    wake(p)
    await p.execute()

  async def read_state(self) -> dict:
    """Cluster snapshot: live nodes, runs with `(owner, pos, snap)`, stream backlog, plan, and `gpu_ms` service."""
    nodes, runs = await self.live_nodes(details=True), sorted(r.decode() for r in await self.r.smembers(key("runs")))
    p = self.r.pipeline(transaction=False)
    for r in runs:
      p.hmget(key("run", r), "record", "owner", "pos", "snap")
      p.xlen(key("q", r))  # A commit deletes its entry, so this includes both queued and pending work.
    p.get(key("plan"))
    p.hgetall(key("service"))
    *rows, plan, service = await p.execute()
    out_runs, queues = {}, {}
    for r, f, backlog in zip(runs, rows[::2], rows[1::2]):
      if not f[0]:
        continue
      info = {"record": json.loads(f[0]), "owner": f[1] and f[1].decode(), "pos": f[2] and f[2].decode()}
      out_runs[r] = info | {"snap": f[3] and f[3].decode()}
      queues[r] = backlog
    service = {k.decode(): float(v) for k, v in service.items()}
    return {"nodes": nodes, "runs": out_runs, "queues": queues, "plan": json.loads(plan or "{}"), "service": service}

  async def write_plan(self, plan: dict, changes: list[tuple[Ev, dict]] = (), notify_nodes: bool = True) -> None:
    p = self.r.pipeline(transaction=True)
    p.set(key("plan"), json.dumps(plan))
    for kind, body in changes:
      p.xadd(key("events"), {"kind": kind, "body": json.dumps(body)}, maxlen=1_000_000, approximate=True)
    if notify_nodes:
      wake(p, "plan_wake")
    await p.execute()

  async def get_run(self, run_id: str) -> dict | None:
    if not (f := await self.r.hgetall(key("run", run_id))) or b"record" not in f:
      return None
    meta = {n: f[n.encode()].decode() for n in ("owner", "pos", "snap") if f.get(n.encode())}
    return {**json.loads(f[b"record"]), **meta, **{k.decode(): int(v) for k, v in f.items() if k not in RUN_META}}

  async def join_sampling_session(self, run_id: str) -> int:
    return int(await self.r.incr(key("clients", run_id)))

  async def run_counters(self) -> dict[tuple[str, str], dict]:
    runs = sorted(r.decode() for r in await self.r.smembers(key("runs")))
    p = self.r.pipeline()
    for r in runs:
      p.hgetall(key("run", r))
    p.get(key("plan"))
    *rows, plan_raw = await p.execute()
    home = (json.loads(plan_raw) if plan_raw else {}).get("home", {})
    out = {}
    for r, f in zip(runs, rows):
      if b"record" in f and (ps := home.get(r)):
        out[(ps[0], r)] = {k.decode(): int(v) for k, v in f.items() if k not in RUN_META}
    return out

  async def delete_run(self, run_id: str) -> None:
    p = self.r.pipeline(transaction=True)
    queue_delete(p, run_id)
    wake(p)
    await p.execute()

  # ***** commands *****

  async def submit_command(
    self, run_id: str, op: Op, body: bytes, seq_id: int | None = None, cost: int = 1
  ) -> str | None:
    if seq_id is not None and (first := await self.r.get(key("seq", run_id, str(seq_id)))):
      return first.decode()
    seq = [] if seq_id is None else ["IDMP", run_id, seq_id]
    args = ["NOMKSTREAM", *seq, "*", "op", op, "body", body, "cost", cost, *(["seq", seq_id] if seq else [])]
    if (entry_id := await self.r.execute_command("XADD", key("q", run_id), *args)) is None:
      return None
    request_id = f"{run_id}/{entry_id.decode()}"
    p = self.r.pipeline()
    if seq:
      p.set(key("seq", run_id, str(seq_id)), request_id, ex=SESSION_TTL)
    wake(p)
    await p.execute()
    return request_id

  async def peek_commands(self, runs: list[str], block_ms: int = 0, idle_ms: int | None = None) -> list[Cmd]:
    """Peek one head per run. Samplers may also prepare a pending entry old enough to reclaim."""
    p = self.r.pipeline(transaction=False)
    for r in runs:
      p.xinfo_groups(key("q", r))
    cursors = {}
    pending = []
    for r, groups in zip(runs, await p.execute(raise_on_error=False)):
      if isinstance(groups, Exception):
        if "no such key" not in str(groups):
          raise groups
      elif g := next((g for g in groups if g["name"] == GROUP.encode()), None):
        cursors[key("q", r)] = g["last-delivered-id"]
        if idle_ms is not None and g["pending"]:
          pending.append(r)
    if not cursors:
      return await asyncio.sleep(block_ms / 1000) or []
    cmds = [cmd for cmd in await asyncio.gather(*(self.pending_head(r, idle_ms) for r in pending)) if cmd]
    for cmd in cmds:
      del cursors[key("q", cmd.run)]
    if cursors:
      got = await self.r.xread(cursors, count=1, block=None if cmds else block_ms or None)
      cmds.extend(parse_cmd(s.decode().removeprefix(key("q", "")), *entries[0]) for s, entries in got)
    return sorted(cmds, key=lambda c: (*map(int, c.entry_id.split("-")), c.run))

  async def pending_head(self, run: str, idle_ms: int) -> Cmd | None:
    queue, start = key("q", run), "-"
    try:
      while rows := await self.r.xpending_range(queue, GROUP, start, "+", count=1, idle=idle_ms):
        eid = rows[0]["message_id"]
        if entry := await self.r.xrange(queue, eid, eid):
          return parse_cmd(run, *entry[0])
        start = b"(" + eid  # Skip a pending ID whose entry has since been deleted.
    except redis.ResponseError as e:
      if "NOGROUP" not in str(e):
        raise

  async def take_commands(
    self, runs: list[str], consumer: str, count: int = 16, block_ms: int = 0, idle_ms: int | None = None
  ) -> list[Cmd]:
    p = self.r.pipeline(transaction=False)
    for r in runs:
      p.xreadgroup(GROUP, consumer, {key("q", r): ">"}, count=count, claim_min_idle_time=idle_ms)
    cmds = []
    for r, got in zip(runs, await p.execute(raise_on_error=False)):
      if isinstance(got, Exception):
        if "NOGROUP" not in str(got):
          raise got
      else:
        cmds += [parse_cmd(r, eid, fields) for _, entries in got for eid, fields, *_ in entries]
    if not cmds and block_ms:
      await self.peek_commands(runs, block_ms, idle_ms)
      return await self.take_commands(runs, consumer, count, idle_ms=idle_ms)
    return cmds

  async def take_trainer_command(self, run: str, token: str) -> Cmd | None:
    """Only the current epoch may move a new trainer command into the pending list."""
    k, life = key("run", run), token.rsplit(":", 1)[0]

    async def op(p):
      if await p.hget(k, "owner") != token.encode():
        raise RuntimeError(f"lost ownership of {run}")
      p.multi()
      p.xreadgroup(GROUP, life, {key("q", run): ">"}, count=1)
      [got] = await p.execute()
      return parse_cmd(run, *got[0][1][0]) if got else None

    return await self.transact(op, k)

  # ***** single-writer ownership, detachment, stateless sample commit, and process leases *****

  async def open_lease(self, life: str) -> bool:
    return bool(await self.r.set(key("lease", life), 1, nx=True, px=LEASE_MS))

  async def renew_lease(self, life: str, recover: bool = False) -> bool:
    return bool(await self.r.set(key("lease", life), 1, xx=not recover, px=LEASE_MS))

  async def drop_lease(self, life: str) -> None:
    await self.r.delete(key("lease", life))

  async def reap_run(self, run_id: str, token: str) -> bool:
    """Clear a dead process's ownership iff `token` still owns `run_id` and its lease has expired."""
    k, lk = key("run", run_id), key("lease", token.rsplit(":", 1)[0])

    async def op(p):
      if await p.hget(k, "owner") != token.encode() or await p.exists(lk):
        return None
      p.multi()
      p.hdel(k, "owner")
      p.hdel(key("writing"), run_id)
      wake(p)
      return bool(await p.execute())

    return bool(await self.transact(op, k, lk))

  async def acquire_run(self, run_id: str, life: str) -> tuple[str, str | None, str | None] | None:
    """Claim an unowned (or expired-owner) training run under a fresh epoch token if `life`'s lease is live."""
    k, lk = key("run", run_id), key("lease", life)
    if (stale := await self.r.hget(k, "owner")) and not await self.reap_run(run_id, stale.decode()):
      return None

    async def op(p):
      rec, owner, epoch, pos, snap = await p.hmget(k, "record", "owner", "epoch", "pos", "snap")
      if not rec or owner is not None or not await p.exists(lk):
        return None
      n = int(epoch or 0) + 1
      p.multi()
      p.hset(k, mapping={"owner": f"{life}:{n}", "epoch": n})
      await p.execute()
      return f"{life}:{n}", pos and pos.decode(), snap and snap.decode()

    return await self.transact(op, k, lk)

  async def begin_checkpoint(self, run: str, token: str) -> str:
    """Protect this write from collection until this owner publishes it, fails it, or is reaped."""
    k, version = key("run", run), uuid.uuid4().hex

    async def op(p):
      if await p.hget(k, "owner") != token.encode():
        raise RuntimeError(f"lost ownership of {run}")
      p.multi()
      p.hset(key("writing"), run, version)
      await p.execute()
      return version

    return await self.transact(op, k)

  async def detach_run(
    self, run_id: str, token: str, pos: str | None, snap: str | None, checkpoint: dict | None = None
  ) -> bool:
    """Release `run_id` and record its exact snapshot `snap` iff `token` still owns it at `pos`."""
    k = key("run", run_id)

    async def op(p):
      owner, cur = await p.hmget(k, "owner", "pos")
      if owner != token.encode() or (cur and cur.decode()) != pos:
        return None
      p.multi()
      if checkpoint:
        queue_checkpoint(p, **checkpoint)
      p.hdel(k, "owner")
      p.hdel(key("writing"), run_id)
      p.hset(k, "snap", snap) if snap else p.hdel(k, "snap")
      wake(p)
      return bool(await p.execute())

    return bool(await self.transact(op, k))

  async def commit_reply(
    self, cmd: Cmd, reply: Reply, token: str, acct="", mutates=False, charge=None, checkpoint=None
  ):
    """Atomically publish a trainer reply, bill `gpu_ms`, and advance `pos` (clearing `snap`) iff `token` owns it."""
    k, ck, deleting = key("run", cmd.run), key("ckpt", cmd.run), cmd.op == Op.DELETE_MODEL and reply.ok

    async def op(p):
      owner, old_snap = await p.hmget(k, "owner", "snap")
      if owner != token.encode():
        return None
      if not await p.xrange(key("q", cmd.run), cmd.entry_id, cmd.entry_id):
        return None
      ckpts = await p.hgetall(ck) if deleting else {}
      p.multi()
      if cmd.op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER):
        p.hdel(key("writing"), cmd.run)
      if checkpoint and reply.ok:
        queue_checkpoint(p, **checkpoint)
      finish_cmd(p, cmd, reply, acct, charge)
      if mutates and not deleting:
        p.hset(k, "pos", cmd.entry_id)
        p.hdel(k, "snap")
      elif deleting:
        for cid in [c for c, d in ckpts.items() if json.loads(d)["owned"]]:
          uri = f"tinker://{cmd.run}/{cid.decode()}"
          p.hdel(ck, cid)
          p.zrem(key("ckpts"), uri)
          p.zrem(key("expiry"), uri)
        queue_delete(p, cmd.run)
      await p.execute()
      return old_snap.decode() if old_snap and mutates and not deleting else ""

    res = await self.transact(op, k, key("f", cmd.request_id), *([ck] if deleting else []))
    return (False, None) if res is None else (True, res or None)

  async def commit_sample(self, cmd: Cmd, reply: Reply, account="", charge: Charge | None = None) -> bool:
    """The winning replica removes the queue entry. A late duplicate stays fenced after the reply expires."""
    fk = key("f", cmd.request_id)

    async def op(p):
      if not await p.xrange(key("q", cmd.run), cmd.entry_id, cmd.entry_id):
        return None
      p.multi()
      finish_cmd(p, cmd, reply, account, charge)
      await p.execute()
      return True

    return bool(await self.transact(op, fk))

  async def fail_stranded_commands(self, run_id: str, token: str, idle_ms: int = 0) -> None:
    while cmd := await self.pending_head(run_id, idle_ms):
      err = Reply.error(RuntimeError(f"its trainer died, or lost ownership, during {cmd.op}"))
      if not (await self.commit_reply(cmd, err, token))[0]:
        return

  # ***** nodes, replies, events *****

  async def node_heartbeat(self, node: str, gpus: int, memory: int = 0, wake_scheduler: bool = False, **info) -> None:
    p = self.r.pipeline()
    p.hset(
      key("nodes"), node, json.dumps({"gpus": gpus, "memory": memory, "until": time.time() + NODE_MS / 1000, **info})
    )
    if wake_scheduler:
      wake(p)
    await p.execute()

  async def live_nodes(self, details: bool = False) -> dict:
    nodes = {k.decode(): json.loads(v) for k, v in (await self.r.hgetall(key("nodes"))).items()}
    if dead := [k for k, v in nodes.items() if v["until"] < time.time()]:
      await self.r.hdel(key("nodes"), *dead)
    return {k: v if details else v["gpus"] for k, v in nodes.items() if k not in dead}

  async def wait_for_reply(self, request_id: str, timeout: float) -> Reply | None:
    if not (got := await self.r.xread({key("f", request_id): "0-0"}, count=1, block=int(timeout * 1000) or None)):
      return None
    f = got[0][1][0][1]
    return Reply(f[b"ok"] == b"1", f[b"content_type"].decode(), f[b"body"])

  async def queued_and_in_flight(self, run_id: str) -> dict:
    p = self.r.pipeline()
    p.xinfo_groups(key("q", run_id))
    p.xpending_range(key("q", run_id), GROUP, min="-", max="+", count=100)
    groups, pending = await p.execute(raise_on_error=False)
    if isinstance(groups, Exception):
      return {"live": False}
    in_flight = [
      {"request_id": f"{run_id}/{e['message_id'].decode()}", "trainer": e["consumer"].decode()}
      | {"seconds": e["time_since_delivered"] / 1000, "deliveries": e["times_delivered"]}
      for e in pending
    ]
    return {"live": True, "queued": groups[0]["lag"], "in_flight": in_flight}

  async def append_event(self, _kind: Ev, **body) -> str:
    p = self.r.pipeline()
    p.xadd(key("events"), {"kind": Ev(_kind), "body": json.dumps(body)}, maxlen=1_000_000, approximate=True)
    wake(p)
    return (await p.execute())[0].decode()

  async def wait_wake(self, wake_id: str, block_ms: int, stream: str = "wake") -> str:
    got = await self.r.xread({key(stream): wake_id}, count=1, block=block_ms or None)
    return got[0][1][-1][0].decode() if got else wake_id

  async def read_events(self, after: str = "0-0", count: int = 10_000, block_ms: int = 0) -> list[Event]:
    got = await self.r.xread({key("events"): after}, count=count, block=block_ms or None)
    rows = got[0][1] if got else ()
    return [
      Event(i.decode(), int(i.decode().split("-")[0]) / 1000, f[b"kind"].decode(), json.loads(f[b"body"]))
      for i, f in rows
    ]
