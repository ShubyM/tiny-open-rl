"""Runs, sessions, commands, replies, ownership generations, and cluster state on Redis.

Every training or sampling call is a command on its run's Redis stream, tinyrl:q:<run>.
A worker process has a supervision lease (tinyrl:owner:<worker>), while each job has an independent
monotonic ownership generation (gen on tinyrl:run:<run>) used to fence command claims, reply publication,
and authoritative recovery-state commits even when moving one job out of a shared worker.
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

GROUP = "trainer"
LIVE_SECONDS = 30
SESSION_GRACE = getenv("SESSION_GRACE", 60)
REPLY_TTL = getenv("REPLY_TTL", 3600)
SESSION_TTL = getenv("SESSION_TTL", 7 * 24 * 3600)
LEASE_MS = getenv("LEASE_MS", 10000)
NODE_MS = 30000
COUNTER_FIELDS = ("steps", "failed", "cmds", "busy_ms", "wait_ms")
STRING_RUN_FIELDS = (
  "worker",
  "last_acked_id",
  "last_mutating_id",
  "restore_uri",
  "restore_version",
  "restore_boundary",
  "durable_uri",
  "durable_version",
  "durable_boundary",
)
TABLE_RUN_FIELDS = ("record", "worker", "gen", *STRING_RUN_FIELDS[1:], "rollback_required")


def key(*parts: str) -> str:
  return ":".join(("tinyrl", *parts))


def id_tuple(entry_id: str | None) -> tuple[int, int]:
  if not entry_id:
    return (0, 0)
  a, _, b = entry_id.partition("-")
  return (int(a), int(b or 0))


def id_ge(a: str | None, b: str | None) -> bool:
  return id_tuple(a) >= id_tuple(b)


def id_gt(a: str | None, b: str | None) -> bool:
  return id_tuple(a) > id_tuple(b)


class Op(StrEnum):
  CREATE_MODEL = "create_model"
  FORWARD_BACKWARD = "forward_backward"
  OPTIM_STEP = "optim_step"
  SAVE_WEIGHTS = "save_weights"
  SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"
  LOAD_WEIGHTS = "load_weights"
  DELETE_MODEL = "delete_model"
  SAMPLE = "sample"


MUTATING_OPS = frozenset({Op.CREATE_MODEL, Op.FORWARD_BACKWARD, Op.OPTIM_STEP, Op.LOAD_WEIGHTS})


class Ev(StrEnum):
  """Cluster trace events."""

  RUN_CREATED = "run_created"
  ASSIGNED = "assigned"
  RUN_DELETED = "run_deleted"
  NODE_UP = "node_up"
  NODE_LOST = "node_lost"
  PLACED = "placed"
  UNPLACED = "unplaced"
  DRAINING = "draining"
  GRANTED = "granted"


class Role(StrEnum):
  TRAINER = "trainer"
  SAMPLER = "sampler"


@dataclass(frozen=True)
class Run:
  id: str
  worker: str
  base: str
  rank: int
  session: str = ""
  model_path: str | None = None
  lora: dict = field(default_factory=dict)
  exclusive: bool = False
  role: Role = Role.TRAINER
  train_run: str | None = None
  gpus: int = 1
  weight: float = 1.0
  profile: dict = field(default_factory=dict)


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
    body = {
      "error": f"{type(exc).__name__}: {exc}",
      "category": "user" if isinstance(exc, (ValueError, KeyError)) else "server",
    }
    return cls(False, "application/json", json.dumps(body).encode())


def parse_cmd(run: str, entry_id: bytes, fields: dict[bytes, bytes]) -> Cmd:
  return Cmd(run, entry_id.decode(), Op(fields[b"op"].decode()), fields[b"body"], int(fields.get(b"cost", 1)))


def _ver_id(ref: str | None) -> str:
  return ref.partition("@")[0] if ref else ""


def _dec(val: bytes | None) -> str:
  return val.decode() if val else ""


def connect(url: str) -> redis.Redis:
  return redis.from_url(
    url, socket_timeout=None, retry=Retry(ExponentialBackoff(cap=2.0, base=0.1), 30), max_connections=10_000
  )


@dataclass(frozen=True)
class Event:
  id: str
  t: float
  kind: str
  body: dict


def _queue_delete_run(
  p, run_id: str, checkpoints: dict[bytes, bytes], rv: str, dv: str, worker: str | None = None
) -> None:
  for cid, data in checkpoints.items():
    info = json.loads(data)
    if info.get("owned"):
      uri = f"tinker://{run_id}/{cid.decode()}"
      p.hdel(key("ckpt", run_id), cid)
      p.zrem(key("ckpts"), uri)
      p.zrem(key("expiry"), uri)
      if vid := _ver_id(info.get("version")):
        p.hdel(key("ver_refs", vid), f"ckpt:{uri}")
  for ver_ref in (rv, dv):
    if vid := _ver_id(ver_ref):
      p.hdel(key("ver_refs", vid), f"recovery:{run_id}")
  p.srem(key("runs"), run_id)
  p.delete(key("q", run_id), key("parked", run_id))
  if worker is not None:
    p.xadd(
      key("events"),
      {"kind": Ev.RUN_DELETED, "body": json.dumps({"run": run_id, "worker": worker})},
      maxlen=1_000_000,
      approximate=True,
    )
  p.xadd(key("wake", "scheduler"), {"t": str(time.time())}, maxlen=16, approximate=True)


class Bus:
  def __init__(self, r: redis.Redis | None = None):
    self.r = r if r is not None else connect(getenv("REDIS_URL", "redis://localhost:6379"))

  # ***** wakeups *****

  async def wake(self, *channels: str) -> None:
    if not channels:
      return
    p = self.r.pipeline(transaction=False)
    for ch in channels:
      p.xadd(key("wake", ch), {"t": str(time.time())}, maxlen=16, approximate=True)
    await p.execute()

  async def wake_cursor(self, channel: str) -> str:
    entries = await self.r.xrevrange(key("wake", channel), count=1)
    return entries[0][0].decode() if entries else "0-0"

  async def wait_wake(self, channel: str, after: str = "$", block_ms: int = 100) -> str:
    got = await self.r.xread({key("wake", channel): after}, count=1, block=block_ms or None)
    return got[0][1][-1][0].decode() if (got and got[0][1]) else after

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
    return [s.decode() for s in await self.r.zrangebyscore(key("seen"), 0, time.time() - SESSION_GRACE)]

  async def finish_session(self, session_id: str) -> None:
    await self.r.zadd(key("seen"), {session_id: 0}, xx=True)
    await self.wake("scheduler")

  async def end_session(self, session_id: str) -> None:
    left = False
    for run_id in (await self.get_session(session_id) or {}).get("runs", []):
      run = await self.get_run(run_id) or {}
      if not await self.r.sismember(key("runs"), run_id):
        continue
      if run.get("role") == Role.SAMPLER:
        await self.delete_run(run_id, run.get("worker", ""))
      elif run.get("role") == Role.TRAINER:
        left = bool(await self.submit_command(run_id, Op.DELETE_MODEL, b"{}")) or left
    await (self.r.zadd(key("seen"), {session_id: time.time()}) if left else self.r.zrem(key("seen"), session_id))

  # ***** runs *****

  async def create_run(self, run: Run) -> None:
    p = self.r.pipeline(transaction=True)
    p.xgroup_create(key("q", run.id), GROUP, id="0", mkstream=True)
    p.execute_command("XCFGSET", key("q", run.id), "IDMP-DURATION", 86400, "IDMP-MAXSIZE", 10000)
    p.sadd(key("runs"), run.id)
    mapping: dict[str, object] = {
      "record": json.dumps({**asdict(run), "created": time.time()}),
      **dict.fromkeys(COUNTER_FIELDS, 0),
      "gen": 1,
      "rollback_required": 0,
      **dict.fromkeys(STRING_RUN_FIELDS[1:], ""),
    }
    if run.worker:
      mapping["worker"] = run.worker
    p.hset(key("run", run.id), mapping=mapping)
    if run.session:
      p.sadd(key("sess", run.session, "runs"), run.id)
      p.expire(key("sess", run.session, "runs"), SESSION_TTL)
    body = {
      "run": run.id,
      "role": run.role,
      "base": run.base,
      "rank": run.rank,
      "exclusive": run.exclusive,
      "train_run": run.train_run,
      "gpus": run.gpus,
    }
    if run.worker:
      body["worker"] = run.worker
    p.xadd(key("events"), {"kind": Ev.RUN_CREATED, "body": json.dumps(body)})
    p.xadd(key("wake", "scheduler"), {"t": str(time.time())}, maxlen=16, approximate=True)
    await p.execute()

  # ***** cluster assignment table & leader fencing *****

  async def take_scheduler_lease(self, leader: str, ttl_ms: int = LEASE_MS) -> bool:
    return bool(await self.r.set(key("scheduler_owner"), leader, nx=True, px=ttl_ms))

  async def renew_scheduler_lease(self, leader: str, ttl_ms: int = LEASE_MS) -> bool:
    return bool(await self.r.set(key("scheduler_owner"), leader, ifeq=leader, px=ttl_ms))

  async def release_scheduler_lease(self, leader: str) -> None:
    await self.r.delex(key("scheduler_owner"), ifeq=leader)

  async def read_table(self) -> dict:
    """Cluster assignment, live node observations, job recovery metadata, persisted fair service, and queue readiness."""
    nodes, runs = await self.live_nodes(details=True), sorted(r.decode() for r in await self.r.smembers(key("runs")))
    p = self.r.pipeline(transaction=True)
    for run in runs:
      p.hmget(key("run", run), *TABLE_RUN_FIELDS)
    for fn, args in (
      (p.hgetall, (key("workers"),)),
      (p.get, (key("workers_made"),)),
      (p.get, (key("table_rev"),)),
      (p.hgetall, (key("service"),)),
      (p.hgetall, (key("service_obs"),)),
      (p.smembers, (key("service_ready"),)),
    ):
      fn(*args)
    *records, workers, workers_made, table_rev, svc, svc_obs, svc_ready = await p.execute()

    queues: dict[str, dict[str, int]] = {}
    if runs:
      qp = self.r.pipeline(transaction=False)
      for run in runs:
        qp.xinfo_groups(key("q", run))
      for run, groups in zip(runs, await qp.execute(raise_on_error=False)):
        g = (
          None
          if (isinstance(groups, Exception) or not groups)
          else next((x for x in groups if x["name"] == GROUP.encode()), groups[0])
        )
        queues[run] = (
          {"queued": int(g.get("lag") or 0), "in_flight": int(g.get("pending") or 0)}
          if g
          else {"queued": 0, "in_flight": 0}
        )

    runs_map, meta_map = {}, {}
    for r, vals in zip(runs, records):
      if not vals[0]:
        continue
      runs_map[r] = (json.loads(vals[0]), vals[1].decode() if vals[1] else None)
      meta_map[r] = {
        "gen": int(vals[2] or 0),
        **{f: _dec(v) for f, v in zip(STRING_RUN_FIELDS[1:], vals[3:11])},
        "rollback_required": vals[11] == b"1",
      }

    vers = [v.decode() for v in await self.r.smembers(key("versions"))]
    ver_map, ckpt_versions = {}, {}
    sampler_uris = sorted({rec["model_path"] for rec, _ in runs_map.values() if rec.get("model_path")})
    if vers or sampler_uris:
      vp = self.r.pipeline(transaction=False)
      for v in vers:
        vp.get(key("ver", v))
      for uri in sampler_uris:
        c_run, _, c_id = uri.removeprefix("tinker://").partition("/")
        vp.hget(key("ckpt", c_run), c_id)
      vres = await vp.execute()
      ver_map = {v: json.loads(raw) for v, raw in zip(vers, vres[: len(vers)]) if raw}
      ckpt_versions = {
        uri: vref
        for uri, raw in zip(sampler_uris, vres[len(vers) :])
        if raw and (vref := json.loads(raw).get("version"))
      }

    return {
      "nodes": nodes,
      "runs": runs_map,
      "meta": meta_map,
      "queues": queues,
      "workers": {k.decode(): json.loads(v) for k, v in workers.items()},
      "workers_made": int(workers_made or 0),
      "table_rev": int(table_rev or 0),
      "service": {k.decode(): float(v) for k, v in svc.items()},
      "service_obs": {k.decode(): float(v) for k, v in svc_obs.items()},
      "service_ready": {k.decode() for k in svc_ready},
      "versions": ver_map,
      "ckpt_versions": ckpt_versions,
    }

  async def write_table(
    self,
    assigned: dict[str, str],
    workers: dict[str, dict],
    gone: list[str],
    workers_made: int,
    changes: list,
    *,
    generations: dict[str, int] | None = None,
    service: dict[str, float] | None = None,
    service_obs: dict[str, float] | None = None,
    service_ready: set[str] | None = None,
    table_rev: int | None = None,
    leader: str | None = None,
    wake_nodes: list[str] | set[str] | tuple[str, ...] = (),
  ) -> bool:
    """Atomically commit the scheduler's assignment, fencing by scheduler leader and never resurrecting deleted runs."""
    touched = bool(assigned or generations)
    watch_keys = [key("runs"), *([key("scheduler_owner")] if leader is not None else [])]
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(*watch_keys)
          if leader is not None and await p.get(key("scheduler_owner")) != leader.encode():
            await p.unwatch()
            return False
          live_runs = {r.decode() for r in await p.smembers(key("runs"))} if touched else set()
          p.multi()
          for run, worker in assigned.items():
            if run in live_runs:
              p.hset(key("run", run), "worker", worker)
          for run, gen in (generations or {}).items():
            if run in live_runs:
              p.hset(key("run", run), "gen", int(gen))
          if workers:
            p.hset(key("workers"), mapping={n: json.dumps(w) for n, w in workers.items()})
          if gone:
            p.hdel(key("workers"), *gone)
          p.set(key("workers_made"), workers_made)
          if table_rev is not None:
            p.set(key("table_rev"), int(table_rev))
          for name, vals in (("service", service), ("service_obs", service_obs)):
            if vals is not None:
              p.delete(key(name))
              if vals:
                p.hset(key(name), mapping={k: repr(float(v)) for k, v in vals.items()})
          if service_ready is not None:
            p.delete(key("service_ready"))
            if service_ready:
              p.sadd(key("service_ready"), *service_ready)
          for kind, body in changes:
            p.xadd(key("events"), {"kind": kind, "body": json.dumps(body)}, maxlen=1_000_000, approximate=True)
          now_s = str(time.time())
          for node in wake_nodes:
            if node:
              p.xadd(key("wake", f"node:{node}"), {"t": now_s}, maxlen=16, approximate=True)
          await p.execute()
          return True
      except redis.WatchError:
        continue

  async def get_run(self, run_id: str) -> dict | None:
    fields = await self.r.hgetall(key("run", run_id))
    if not fields or b"record" not in fields:
      return None
    record = json.loads(fields.pop(b"record"))
    for sf in STRING_RUN_FIELDS:
      if (raw := fields.pop(sf.encode(), None)) is not None and (sf != "worker" or raw):
        record[sf] = raw.decode()
    record["gen"] = int(fields.pop(b"gen", b"0") or 0)
    record["rollback_required"] = fields.pop(b"rollback_required", b"0") == b"1"
    return {**record, **{k.decode(): int(v) for k, v in fields.items() if k.decode() in COUNTER_FIELDS}}

  async def job_generation(self, run_id: str) -> int:
    raw = await self.r.hget(key("run", run_id), "gen")
    return int(raw) if raw else 0

  async def bump_job_generation(self, run_id: str) -> int:
    return int(await self.r.hincrby(key("run", run_id), "gen", 1))

  async def join_sampling_session(self, run_id: str) -> int:
    return int(await self.r.incr(key("clients", run_id)))

  async def run_counters(self) -> dict[tuple[str, str], dict]:
    runs = sorted(r.decode() for r in await self.r.smembers(key("runs")))
    p = self.r.pipeline()
    for run_id in runs:
      p.hgetall(key("run", run_id))
    return {
      (f[b"worker"].decode(), run_id): {k.decode(): int(v) for k, v in f.items() if k.decode() in COUNTER_FIELDS}
      for run_id, f in zip(runs, await p.execute())
      if b"worker" in f
    }

  async def delete_run(self, run_id: str, worker: str) -> None:
    rv, dv = await self.r.hmget(key("run", run_id), "restore_version", "durable_version")
    ckpts = await self.r.hgetall(key("ckpt", run_id))
    p = self.r.pipeline(transaction=True)
    _queue_delete_run(p, run_id, ckpts, _dec(rv), _dec(dv))
    await p.execute()
    await self.append_event(Ev.RUN_DELETED, run=run_id, worker=worker)

  # ***** commands *****

  async def submit_command(
    self, run_id: str, op: Op, body: bytes, seq_id: int | None = None, cost: int = 1
  ) -> str | None:
    if seq_id is not None and (first := await self.r.get(key("seq", run_id, str(seq_id)))):
      return first.decode()
    seq = [] if seq_id is None else ["IDMP", run_id, seq_id]
    fields = ["op", op, "body", body, "cost", cost, *(["seq", seq_id] if seq else [])]
    if (entry_id := await self.r.execute_command("XADD", key("q", run_id), "NOMKSTREAM", *seq, "*", *fields)) is None:
      return None
    request_id = f"{run_id}/{entry_id.decode()}"
    p = self.r.pipeline(transaction=False)
    if seq:
      p.set(key("seq", run_id, str(seq_id)), request_id, ex=SESSION_TTL)
    p.xadd(key("wake", "scheduler"), {"t": str(time.time())}, maxlen=16, approximate=True)
    await p.execute()
    return request_id

  async def peek_commands(self, runs: list[str], block_ms: int = 0) -> list[Cmd]:
    if not runs:
      if block_ms:
        await asyncio.sleep(block_ms / 1000)
      return []
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
      if block_ms:
        await asyncio.sleep(block_ms / 1000)
      return []
    got = await self.r.xread(cursors, count=1, block=block_ms or None)
    cmds = [parse_cmd(stream.decode().removeprefix(key("q", "")), *entries[0]) for stream, entries in got]
    return sorted(cmds, key=lambda c: (*map(int, c.entry_id.split("-")), c.run))

  async def take_commands(
    self, runs: list[str], owner: str, count: int = 16, block_ms: int = 0, generations: dict[str, int] | None = None
  ) -> list[Cmd]:
    """Claim ready commands. When `generations` is provided, atomically fence each claim under WATCH on job gen."""
    cmds: list[Cmd] = []
    if generations is None:
      p = self.r.pipeline(transaction=False)
      for run in runs:
        p.xreadgroup(GROUP, owner, {key("q", run): ">"}, count=count)
      for run, got in zip(runs, await p.execute(raise_on_error=False)):
        if isinstance(got, Exception):
          if "NOGROUP" not in str(got):
            raise got
          continue
        cmds += [parse_cmd(run, eid, fields) for _, entries in got for eid, fields in entries]
    else:
      for run in runs:
        if (expected_gen := generations.get(run)) is None:
          continue
        rk, sk = key("run", run), key("q", run)
        while True:
          try:
            async with self.r.pipeline(transaction=True) as p:
              await p.watch(rk)
              raw_gen = await p.hget(rk, "gen")
              if raw_gen is None or int(raw_gen) != expected_gen:
                await p.unwatch()
                break
              p.multi()
              p.xreadgroup(GROUP, owner, {sk: ">"}, count=count)
              res = await p.execute()
              cmds += [parse_cmd(run, eid, fields) for _, entries in (res[0] if res else []) for eid, fields in entries]
              break
          except redis.WatchError:
            continue
          except redis.ResponseError as e:
            if "NOGROUP" in str(e):
              break
            raise
    if not cmds and block_ms:
      await self.peek_commands(runs, block_ms)
      return await self.take_commands(runs, owner, count, generations=generations)
    return cmds

  async def write_reply(
    self, cmd: Cmd, reply: Reply, worker: str, owner: str, started: float | None = None, generation: int | None = None
  ) -> bool:
    """Atomically write reply, ack command, update progress & counters, fenced by worker lease and job generation."""
    counts = {"failed": 1} if not reply.ok else {"steps": 1} if cmd.op == Op.OPTIM_STEP else {}
    if started is not None:
      counts |= {
        "cmds": 1,
        "wait_ms": max(0, int(started * 1000) - int(cmd.entry_id.split("-")[0])),
        "busy_ms": int((time.time() - started) * 1000),
      }
    ok_k, rk, deleting = key("owner", worker), key("run", cmd.run), cmd.op == Op.DELETE_MODEL and reply.ok
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(ok_k, rk, *([key("ckpt", cmd.run)] if deleting else []))
          if await p.get(ok_k) != owner.encode():
            await p.unwatch()
            return False
          rf = await p.hmget(rk, "gen", "last_acked_id", "last_mutating_id", "restore_version", "durable_version")
          if generation is not None and (rf[0] is None or int(rf[0]) != generation):
            await p.unwatch()
            return False
          checkpoints = await p.hgetall(key("ckpt", cmd.run)) if deleting else {}
          p.multi()
          p.xadd(
            key("f", cmd.request_id),
            {"ok": int(reply.ok), "content_type": reply.content_type, "body": reply.body},
            maxlen=1,
            approximate=False,
          )
          p.expire(key("f", cmd.request_id), REPLY_TTL)
          p.xack(key("q", cmd.run), GROUP, cmd.entry_id)
          p.xdel(key("q", cmd.run), cmd.entry_id)
          if rf[0] is not None:
            for name, n in counts.items():
              p.hincrby(rk, name, n)
            if reply.ok:
              if id_gt(cmd.entry_id, _dec(rf[1])):
                p.hset(rk, "last_acked_id", cmd.entry_id)
              if cmd.op in MUTATING_OPS and id_gt(cmd.entry_id, _dec(rf[2])):
                p.hset(rk, "last_mutating_id", cmd.entry_id)
              if cmd.op == Op.LOAD_WEIGHTS:
                p.hset(rk, "rollback_required", 0)
          if deleting:
            _queue_delete_run(p, cmd.run, checkpoints, _dec(rf[3]), _dec(rf[4]), worker)
          await p.execute()
          return True
      except redis.WatchError:
        continue

  async def commit_job_state(
    self,
    run_id: str,
    uri: str,
    version: str,
    boundary: str,
    durable: bool,
    worker: str = "",
    owner: str = "",
    generation: int | None = None,
    set_parked: bool = False,
    update_restore: bool = True,
  ) -> bool:
    """Atomically advance a job's restore/durable state & boundary under generation fencing, retaining previous durable recovery point."""
    rk, vid = key("run", run_id), _ver_id(version)
    vk = key("ver", vid) if vid else None
    watch_keys = [rk, *([key("owner", worker)] if worker and owner else []), *([vk] if vk else [])]
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(*watch_keys)
          if worker and owner and await p.get(key("owner", worker)) != owner.encode():
            await p.unwatch()
            return False
          vals = await p.hmget(rk, "gen", "restore_version", "restore_boundary", "durable_version", "durable_boundary")
          if (vals[0] is None and (update_restore or not await p.exists(rk))) or (
            generation is not None and int(vals[0]) != generation
          ):
            await p.unwatch()
            return False
          if vk and (vraw := await p.get(vk)) and json.loads(vraw).get("state") != "active":
            await p.unwatch()
            return False

          old_rv, old_rb, old_dv, old_db = _ver_id(_dec(vals[1])), _dec(vals[2]), _ver_id(_dec(vals[3])), _dec(vals[4])
          updates, new_rv, new_dv = {}, old_rv, old_dv
          if update_restore and (not old_rb or id_ge(boundary, old_rb)):
            updates |= {"restore_uri": uri, "restore_version": version, "restore_boundary": boundary}
            new_rv = vid
          if durable and (not old_db or id_ge(boundary, old_db)):
            updates |= {"durable_uri": uri, "durable_version": version, "durable_boundary": boundary}
            new_dv = vid
          if not update_restore and not updates:
            await p.unwatch()
            return False

          p.multi()
          if updates:
            p.hset(rk, mapping=updates)
          if set_parked:
            p.set(key("parked", run_id), uri)
          if vid and vid in (new_rv, new_dv):
            p.hset(key("ver_refs", vid), f"recovery:{run_id}", "0")
          if old_rv and old_rv not in (new_rv, new_dv):
            p.hdel(key("ver_refs", old_rv), f"recovery:{run_id}")
          if durable and old_dv and old_dv not in (new_rv, new_dv):
            p.hdel(key("ver_refs", old_dv), f"recovery:{run_id}")
          await p.execute()
          return True
      except redis.WatchError:
        continue

  async def promote_durable_state(self, run_id: str, uri: str, version: str, boundary: str) -> bool:
    """Promote an already-published version to a job's durable recovery point once bucket upload finishes."""
    return await self.commit_job_state(run_id, uri, version, boundary, durable=True, update_restore=False)

  async def _fenced_run_write(self, run_id: str, worker: str, owner: str, generation: int | None, op) -> bool:
    ok_k, rk = key("owner", worker), key("run", run_id)
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(ok_k, *([rk] if generation is not None else []))
          if await p.get(ok_k) != owner.encode():
            await p.unwatch()
            return False
          if generation is not None:
            raw_gen = await p.hget(rk, "gen")
            if raw_gen is None or int(raw_gen) != generation:
              await p.unwatch()
              return False
          p.multi()
          op(p)
          await p.execute()
          return True
      except redis.WatchError:
        continue

  async def mark_rollback_required(self, run_id: str, worker: str, owner: str, generation: int | None = None) -> bool:
    if generation is None:
      generation = await self.job_generation(run_id)
    return await self._fenced_run_write(
      run_id, worker, owner, generation, lambda p: p.hset(key("run", run_id), "rollback_required", 1)
    )

  async def fail_stranded_commands(
    self,
    runs: list[str],
    worker: str,
    owner: str,
    min_idle_ms: int = 0,
    generations: dict[str, int] | None = None,
    replay_sampler: bool = False,
  ) -> list[Cmd]:
    """Handle pending entries left by a dead owner. Fenced by job generation; optionally returns idempotent SAMPLE cmds for replay."""
    replayed: list[Cmd] = []
    for run_id in runs:
      gen = generations.get(run_id) if generations is not None else None
      if (generations is not None and gen is None) or (gen is not None and await self.job_generation(run_id) != gen):
        continue
      stream = key("q", run_id)
      try:
        deliveries = (
          {
            e["message_id"]: int(e["times_delivered"])
            for e in await self.r.xpending_range(stream, GROUP, min="-", max="+", count=10000)
          }
          if replay_sampler
          else {}
        )
        _, entries, _ = await self.r.xautoclaim(stream, GROUP, owner, min_idle_time=min_idle_ms, count=10000)
      except redis.ResponseError as e:
        if "NOGROUP" not in str(e):
          raise
        continue
      for entry_id, fields in entries:
        cmd = parse_cmd(run_id, entry_id, fields)
        if replay_sampler and cmd.op == Op.SAMPLE and deliveries.get(entry_id, 1) <= 3:
          replayed.append(cmd)
          continue
        await self.write_reply(
          cmd,
          Reply.error(RuntimeError(f"its worker died, or lost the worker, during {cmd.op}")),
          worker,
          owner,
          generation=gen,
        )
    return replayed

  # ***** parked checkpoints (compatibility bridge with committed restore_uri) *****

  async def set_parked_checkpoint(
    self, run_id: str, uri: str | None, worker: str, owner: str, generation: int | None = None
  ) -> bool:
    return await self._fenced_run_write(
      run_id,
      worker,
      owner,
      generation,
      lambda p: p.set(key("parked", run_id), uri) if uri else p.delete(key("parked", run_id)),
    )

  async def parked_checkpoint(self, run_id: str) -> str | None:
    uri = await self.r.get(key("parked", run_id))
    return uri.decode() if uri else None

  # ***** worker leases *****

  async def take_lease(self, worker: str, owner: str, recover: bool = False) -> bool:
    k, last = key("owner", worker), key("last_owner", worker)
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(k, last)
          current, previous = await p.mget(k, last)
          if current is not None or (recover and previous not in (None, owner.encode())):
            await p.unwatch()
            return False
          p.multi()
          p.set(k, owner, px=LEASE_MS)
          p.set(last, owner)
          await p.execute()
          return True
      except redis.WatchError:
        continue

  async def renew_lease(self, worker: str, owner: str) -> bool:
    return bool(await self.r.set(key("owner", worker), owner, ifeq=owner, px=LEASE_MS))

  async def release_lease(self, worker: str, owner: str) -> None:
    await self.r.delex(key("owner", worker), ifeq=owner)

  async def lease_owner(self, worker: str) -> str | None:
    owner = await self.r.get(key("owner", worker))
    return owner.decode() if owner else None

  # ***** nodes & observations *****

  async def node_heartbeat(
    self,
    node: str,
    gpus: int,
    memory: int = 0,
    workers: dict | None = None,
    lag: float = 0.0,
    wake_scheduler: bool = False,
    **extra,
  ) -> None:
    beat = {
      "gpus": gpus,
      "memory": memory,
      "until": time.time() + NODE_MS / 1000,
      "workers": workers or {},
      "lag": lag,
      **extra,
    }
    p = self.r.pipeline(transaction=False)
    p.hset(key("nodes"), node, json.dumps(beat))
    if wake_scheduler:
      p.xadd(key("wake", "scheduler"), {"t": str(time.time())}, maxlen=16, approximate=True)
    await p.execute()

  async def live_nodes(self, details: bool = False) -> dict:
    nodes = {k.decode(): json.loads(v) for k, v in (await self.r.hgetall(key("nodes"))).items()}
    if dead := [k for k, v in nodes.items() if v["until"] < time.time()]:
      await self.r.hdel(key("nodes"), *dead)
    return {k: v if details else v["gpus"] for k, v in nodes.items() if k not in dead}

  # ***** replies *****

  async def wait_for_reply(self, request_id: str, timeout: float) -> Reply | None:
    got = await self.r.xread({key("f", request_id): "0-0"}, count=1, block=int(timeout * 1000) or None)
    if not got:
      return None
    fields = got[0][1][0][1]
    return Reply(fields[b"ok"] == b"1", fields[b"content_type"].decode(), fields[b"body"])

  async def queued_and_in_flight(self, run_id: str) -> dict:
    stream = key("q", run_id)
    p = self.r.pipeline()
    p.xinfo_groups(stream)
    p.xpending_range(stream, GROUP, min="-", max="+", count=100)
    groups, pending = await p.execute(raise_on_error=False)
    if isinstance(groups, Exception):
      return {"live": False}
    in_flight = [
      {
        "request_id": f"{run_id}/{e['message_id'].decode()}",
        "trainer": e["consumer"].decode(),
        "seconds": e["time_since_delivered"] / 1000,
        "deliveries": e["times_delivered"],
      }
      for e in pending
    ]
    return {"live": True, "queued": groups[0]["lag"], "in_flight": in_flight}

  # ***** events *****

  async def append_event(self, _kind: Ev, **body) -> str:
    return (
      await self.r.xadd(
        key("events"), {"kind": Ev(_kind), "body": json.dumps(body)}, maxlen=1_000_000, approximate=True
      )
    ).decode()

  async def read_events(self, after: str = "0-0", count: int = 10_000, block_ms: int = 0) -> list[Event]:
    got = await self.r.xread({key("events"): after}, count=count, block=block_ms or None)
    return [
      Event(eid.decode(), int(eid.decode().split("-")[0]) / 1000, f[b"kind"].decode(), json.loads(f[b"body"]))
      for eid, f in (got[0][1] if got else [])
    ]
