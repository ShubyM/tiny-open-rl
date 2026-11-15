"""Ordered commands, exact checkpoint references, and immutable GPU turns on Redis."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import redis.asyncio as redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff

from tinyrl.helpers import getenv
from tinyrl.storage.store import split_version

GROUP = "trainer"
LIVE_SECONDS, SESSION_GRACE = 30, getenv("SESSION_GRACE", 60)
REPLY_TTL, SESSION_TTL = getenv("REPLY_TTL", 3600), getenv("SESSION_TTL", 7 * 24 * 3600)
LEASE_MS, NODE_MS = getenv("LEASE_MS", 10000), 30000


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
  base: str
  rank: int
  session: str = ""
  model_path: str | None = None
  lora: dict = field(default_factory=dict)
  exclusive: bool = False
  role: Role = Role.TRAINER
  train_run: str | None = None
  gpus: int = 1
  version: str | None = None


@dataclass(frozen=True)
class Cmd:
  run: str
  entry_id: str
  op: Op
  body: bytes
  cost: int = 1
  turn: str = ""

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


def connect(url: str) -> redis.Redis:
  return redis.from_url(
    url, socket_timeout=None, retry=Retry(ExponentialBackoff(cap=2.0, base=0.1), 30), max_connections=10_000
  )


def entry_key(entry: str) -> tuple[int, int]:
  return tuple(map(int, entry.split("-"))) if entry else (0, 0)


def parse_cmd(run, entry, fields, turn="") -> Cmd:
  return Cmd(run, entry.decode(), Op(fields[b"op"].decode()), fields[b"body"], int(fields.get(b"cost", 1)), turn)


def record(fields: dict) -> dict | None:
  if b"record" not in fields:
    return None
  return (
    json.loads(fields[b"record"])
    | {"state": json.loads(fields.get(b"state", b"{}"))}
    | {k.decode(): int(v) for k, v in fields.items() if k not in (b"record", b"state")}
  )


def write_run(p, run: Run) -> None:
  p.xgroup_create(key("q", run.id), GROUP, id="0", mkstream=True)
  p.sadd(key("runs"), run.id)
  p.hset(
    key("run", run.id),
    mapping={
      "record": json.dumps(asdict(run) | {"created": time.time()}),
      "state": json.dumps({"mutation": "", "created": "", "working": None, "durable": None}),
      **dict.fromkeys(("steps", "failed", "cmds", "busy_ms", "wait_ms"), 0),
    },
  )
  if run.version:
    p.hset(key("refs"), f"job:{run.id}:weights", split_version(run.version)[0])
  if run.session:
    p.sadd(key("sess", run.session, "runs"), run.id)
    p.expire(key("sess", run.session, "runs"), SESSION_TTL)


def write_future(p, request: str, reply: Reply) -> None:
  p.xadd(
    key("f", request),
    {"ok": int(reply.ok), "content_type": reply.content_type, "body": reply.body},
    maxlen=1,
    approximate=False,
  )
  p.expire(key("f", request), REPLY_TTL)


def ack(p, cmd: Cmd) -> None:
  p.xack(key("q", cmd.run), GROUP, cmd.entry_id)
  p.xdel(key("q", cmd.run), cmd.entry_id)
  p.hdel(key("refs"), f"cmd:{cmd.request_id}")


def write_checkpoint(p, saved: dict) -> None:
  uri, version, ttl = saved["uri"], saved["version"], saved.get("ttl")
  run, cid = uri.removeprefix("tinker://").split("/", 1)
  now, kind = time.time(), cid.split("/", 1)[0]
  info = {
    "checkpoint_id": cid,
    "checkpoint_type": {"weights": "training", "sampler_weights": "sampler"}.get(kind, kind),
    "time": now,
    "tinker_path": uri,
    "size_bytes": saved.get("size", 0),
    "expires_at": now + ttl if ttl else None,
    "owned": saved.get("owned", False),
    "version": version,
    "boundary": saved.get("boundary", ""),
  }
  p.hset(key("ckpt", run), cid, json.dumps(info))
  p.hset(key("refs"), f"ckpt:{uri}", split_version(version)[0])
  p.zadd(key("ckpts"), {uri: now})
  p.zadd(key("expiry"), {uri: now + ttl}) if ttl else p.zrem(key("expiry"), uri)


def write_version(p, run: str, state: dict, saved: dict) -> None:
  v = split_version(saved["version"])[0]
  p.setnx(key("ver", v), json.dumps({k: saved[k] for k in ("uri", "version")} | {"time": time.time(), "state": "live"}))
  if saved.get("meta", {}).get("optimizer", saved.get("optimizer", False)):
    for name in ("working", "durable") if saved.get("durable") else ("working",):
      if entry_key(saved["boundary"]) >= entry_key((state.get(name) or {}).get("boundary", "")):
        state[name] = saved
        p.hset(key("refs"), f"job:{run}:{name}", v)


class Bus:
  def __init__(self, r: redis.Redis | None = None):
    self.r = r if r is not None else connect(getenv("REDIS_URL", "redis://localhost:6379"))

  async def tx(self, keys, step):
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(*keys)
          write, result = await step(p)
          values = await p.execute() if write else []
          return result(values) if callable(result) else result
      except redis.WatchError:
        continue

  # Sessions and immutable job descriptions.
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
    info, runs = await p.execute()
    if not info:
      return None
    return {
      "session_id": session_id,
      "created": float(info[b"created"]),
      "seen": float(info[b"seen"]),
      "live": time.time() - float(info[b"seen"]) < LIVE_SECONDS,
      "runs": sorted(r.decode() for r in runs),
      "metadata": json.loads(info.get(b"metadata") or "{}"),
    }

  async def expired_sessions(self) -> list[str]:
    return [s.decode() for s in await self.r.zrangebyscore(key("seen"), 0, time.time() - SESSION_GRACE)]

  async def finish_session(self, session_id: str) -> None:
    await self.r.zadd(key("seen"), {session_id: 0}, xx=True)

  async def end_session(self, session_id: str) -> None:
    left = False
    for run_id in (await self.get_session(session_id) or {}).get("runs", []):
      if not await self.r.sismember(key("runs"), run_id):
        continue
      run = await self.get_run(run_id)
      if run["role"] == Role.SAMPLER:
        await self.delete_run(run_id)
      else:
        left = bool(await self.submit_command(run_id, Op.DELETE_MODEL, b"{}")) or left
    if left:
      await self.r.zadd(key("seen"), {session_id: time.time()})
    else:
      await self.r.zrem(key("seen"), session_id)

  async def create_run(self, run: Run) -> None:
    v, rkey = split_version(run.version)[0], key("run", run.id)
    if run.model_path and not v:
      raise ValueError("a sampling job needs its exact weight version")

    async def step(p):
      if await p.hexists(rkey, "record"):
        return False, None
      if v and not await self._version(p, v, existing=True):
        raise ValueError("checkpoint is missing or being deleted")
      p.multi()
      write_run(p, run)
      return True, None

    await self.tx((rkey, key("q", run.id)), step)

  async def get_run(self, run_id: str) -> dict | None:
    return record(await self.r.hgetall(key("run", run_id)))

  async def join_sampling_session(self, run_id: str) -> int:
    return int(await self.r.incr(key("clients", run_id)))

  async def run_counters(self) -> dict:
    table = await self.read_table()
    return {
      (table["claims"][r]["engine"], r): {k: run.get(k, 0) for k in ("steps", "failed", "cmds", "busy_ms", "wait_ms")}
      for r, run in table["runs"].items()
      if r in table["claims"]
    }

  async def delete_run(self, run_id: str) -> None:
    async def step(p):
      await self._remove_run(p, run_id)
      return True, None

    await self.tx((key("run", run_id),), step)

  async def _remove_run(self, p, run: str, completed: str = "") -> None:
    await p.watch(key("children", run), key("saves"))
    targets = [run, *(r.decode() for r in await p.smembers(key("children", run)))]
    rows = {}
    for r in targets:
      await p.watch(key("run", r), key("q", r), key("ckpt", r))
      info = json.loads(raw) if (raw := await p.hget(key("run", r), "record")) else {}
      owned = [c for raw in await p.hvals(key("ckpt", r)) if (c := json.loads(raw))["owned"]]
      requests = [f"{r}/{eid.decode()}" for eid, _ in await p.xrange(key("q", r))]
      rows[r] = info, owned, requests
    for request in await p.hkeys(key("saves")):
      if (r := request.decode().split("/", 1)[0]) in rows:
        rows[r][2].append(request.decode())
    p.multi()
    for table in ("claims", "moves"):
      p.hdel(key(table), *targets)
    for r, (info, owned, requests) in rows.items():
      p.hdel(key("refs"), *(f"job:{r}:{name}" for name in ("working", "durable", "weights")))
      for request in requests:
        p.hdel(key("refs"), f"cmd:{request}", f"save:{request}")
        p.hdel(key("saves"), request)
        if request != completed:
          write_future(p, request, Reply.error(RuntimeError("model deleted before command completed")))
      for c in owned:
        p.hdel(key("refs"), f"ckpt:{c['tinker_path']}")
        p.hdel(key("ckpt", r), c["checkpoint_id"])
        p.zrem(key("ckpts"), c["tinker_path"])
        p.zrem(key("expiry"), c["tinker_path"])
      if info.get("train_run"):
        p.srem(key("children", info["train_run"]), r)
      p.delete(key("q", r), key("children", r))
      p.hdel(key("run", r), "state")
    p.srem(key("runs"), *targets)

  # Placement is claims; physical GPU ownership is the existence of an immutable turn.
  async def read_table(self) -> dict:
    async def step(p):
      runs = sorted(r.decode() for r in await p.smembers(key("runs")))
      tables = {
        n: {k.decode(): json.loads(v) for k, v in (await p.hgetall(key(n))).items()} for n in ("claims", "turns", "obs")
      }
      engines = sorted(set(tables["obs"]) | {v["engine"] for n in ("claims", "turns") for v in tables[n].values()})
      p.multi()
      for run in runs:
        p.hgetall(key("run", run))
      for engine in engines:
        p.exists(key("live", engine))
      for name in ("nodes", "service", "moves", "reservations"):
        p.hgetall(key(name))
      p.get(key("fair"))
      return True, lambda values: (runs, engines, tables, values)

    runs, engines, tables, rows = await self.tx((key("runs"), key("claims"), key("turns"), key("obs")), step)
    nodes, service, moves, reservations, fair = rows[-5:]
    return {
      "runs": {r: record(v) for r, v in zip(runs, rows[: len(runs)])},
      "claims": tables["claims"],
      "turns": tables["turns"],
      "observations": tables["obs"],
      "live": [e for e, alive in zip(engines, rows[len(runs) : len(runs) + len(engines)]) if alive],
      "nodes": {n.decode(): info for n, v in nodes.items() if (info := json.loads(v))["until"] >= time.time()},
      "service": {a.decode(): float(v) for a, v in service.items()},
      "moves": {r.decode(): json.loads(v) for r, v in moves.items()},
      "reservations": {n.decode(): json.loads(v) for n, v in reservations.items()},
      "fair": json.loads(fair or "{}"),
    }

  async def start_engine(self, engine: str, node: str) -> bool:
    lkey = key("live", engine)

    async def step(p):
      if await p.hexists(key("obs"), engine) or await p.exists(lkey):
        return False, False
      info = json.loads(raw) if (raw := await p.hget(key("nodes"), node)) else {}
      assigned = [json.loads(v) for name in ("claims", "turns") for v in await p.hvals(key(name))]
      if info.get("until", 0) < time.time() or not any(v["engine"] == engine and v["node"] == node for v in assigned):
        return False, False
      p.multi()
      p.set(lkey, 1, nx=True, px=LEASE_MS)
      p.hset(key("obs"), engine, json.dumps({"node": node, "gpus": [], "ready": [], "closed": False, "pid": None}))
      return True, True

    return await self.tx((lkey, key("obs"), key("claims"), key("turns"), key("nodes")), step)

  async def write_observation(self, engine: str, obs: dict, *, turn: str | None = None) -> bool:
    async def step(p):
      previous = json.loads(raw) if (raw := await p.hget(key("obs"), engine)) else {}
      if previous.get("node") != obs["node"] or previous.get("closed"):
        return False, False
      if not obs.get("closed") and not await p.exists(key("live", engine)):
        return False, False
      if turn:
        grant = json.loads(raw) if (raw := await p.hget(key("turns"), turn)) else {}
        if (grant.get("engine"), grant.get("node"), grant.get("gpus")) != (engine, obs["node"], obs["gpus"]):
          return False, False
      if previous == obs and not turn:
        return False, True
      p.multi()
      p.hset(key("obs"), engine, json.dumps(obs))
      return True, True

    return await self.tx((key("live", engine), key("obs"), key("turns")), step)

  async def release_turn(self, turn: str, engine: str, usage: dict) -> bool:
    if any(not math.isfinite(v) or v < 0 for v in usage.values()):
      raise ValueError("turn service must be finite and nonnegative")

    async def step(p):
      grant = json.loads(raw) if (raw := await p.hget(key("turns"), turn)) else {}
      obs = json.loads(raw) if (raw := await p.hget(key("obs"), engine)) else {}
      if grant.get("engine") != engine or grant.get("node") != obs.get("node"):
        return False, False
      p.multi()
      p.hdel(key("turns"), turn)
      for account, seconds in usage.items():
        p.hincrbyfloat(key("service"), account, seconds)
      return True, True

    return await self.tx((key("turns"), key("obs")), step)

  async def _fence(self, p, turn: str, run: str, engine=None):
    grant = json.loads(raw) if (raw := await p.hget(key("turns"), turn)) else {}
    claim = json.loads(raw) if (raw := await p.hget(key("claims"), run)) else {}
    expected = grant.get("work", {}).get(run)
    if not expected or (claim.get("node"), claim.get("engine"), claim.get("token")) != (
      grant.get("node"),
      grant.get("engine"),
      expected[0],
    ):
      return None
    actual = grant["engine"]
    await p.watch(key("live", actual))
    obs = json.loads(raw) if (raw := await p.hget(key("obs"), actual)) else {}
    if (
      obs.get("closed")
      or obs.get("node") != grant["node"]
      or (engine and actual != engine)
      or not await p.exists(key("live", actual))
    ):
      return None
    run_record = record(await p.hgetall(key("run", run)))
    return (grant, run_record) if run_record and run_record["state"] else None

  async def _version(self, p, version, existing=False) -> bool:
    vkey = key("ver", split_version(version)[0])
    await p.watch(vkey)
    raw = await p.get(vkey)
    return json.loads(raw)["state"] == "live" if raw else not existing

  # Input pinning commits with enqueue and deduplication, before any scheduling.
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
      if op == Op.LOAD_WEIGHTS and (not v or not await self._version(p, v, existing=True)):
        raise ValueError("checkpoint is missing or being deleted")
      ms, n = entry_key((await p.xinfo_stream(q))["last-generated-id"].decode())
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

    return await self.tx((q, skey), step)

  async def head(self, run: str, block_ms: int = 0) -> Cmd | None:
    q = key("q", run)
    try:
      if pending := await self.r.xpending_range(q, GROUP, "-", "+", count=1):
        got = await self.r.xrange(q, min=pending[0]["message_id"], max=pending[0]["message_id"])
        if got:
          return parse_cmd(run, *got[0])
      group = next(g for g in await self.r.xinfo_groups(q) if g["name"] == GROUP.encode())
      got = await self.r.xread({q: group["last-delivered-id"]}, count=1, block=block_ms or None)
      return parse_cmd(run, *got[0][1][0]) if got else None
    except redis.ResponseError as exc:
      if "no such key" not in str(exc) and "NOGROUP" not in str(exc):
        raise
      return None

  async def take_commands(self, turn: str, run: str, count: int = 16, limit: int | None = None) -> list[Cmd]:
    q = key("q", run)

    async def step(p):
      if not (fenced := await self._fence(p, turn, run)):
        return False, []
      allowance = fenced[0]["work"][run][1]
      remaining = min(allowance, limit) if limit is not None else allowance
      if remaining <= 0:
        return False, []
      pending = [e for e in await p.xpending_range(q, GROUP, "-", "+", count=count) if e["consumer"] != turn.encode()]
      if pending:
        entries = await p.xrange(q, min=pending[0]["message_id"], max=pending[-1]["message_id"])
        ids = {e["message_id"] for e in pending}
        commands = [parse_cmd(run, eid, fs, turn) for eid, fs in entries if eid in ids]
        uncertain = [c for c in commands if c.op != Op.SAMPLE]
        if uncertain:
          p.multi()
          for cmd in uncertain:
            write_future(
              p,
              cmd.request_id,
              Reply.error(RuntimeError(f"engine lost during {cmd.op}; its unfinished command was not replayed")),
            )
            ack(p, cmd)
          p.hincrby(key("run", run), "failed", len(uncertain))
          p.hincrby(key("run", run), "cmds", len(uncertain))
          return True, []
      else:
        group = next(g for g in await p.xinfo_groups(q) if g["name"] == GROUP.encode())
        commands = [
          parse_cmd(run, eid, fs, turn)
          for eid, fs in await p.xrange(q, min=f"({group['last-delivered-id'].decode()}", count=count)
        ]
      chosen, cost = [], 0
      for cmd in commands:
        if chosen and cost + cmd.cost > remaining:
          break
        chosen.append(cmd)
        cost += cmd.cost
      if not chosen:
        return False, []
      p.multi()
      if pending:
        p.xclaim(q, GROUP, turn, 0, [c.entry_id for c in chosen])
      else:
        p.xreadgroup(GROUP, turn, {q: ">"}, count=len(chosen))
      return True, chosen

    return await self.tx((key("claims"), key("turns"), key("obs"), key("run", run), q), step)

  async def write_reply(
    self,
    cmd: Cmd,
    reply: Reply,
    engine: str,
    started: float | None = None,
    *,
    checkpoint: dict | None = None,
    mutated: bool = False,
    defer: bool = False,
  ) -> bool:
    q, rkey = key("q", cmd.run), key("run", cmd.run)
    if defer and (cmd.op != Op.SAVE_WEIGHTS or not reply.ok or not checkpoint):
      raise ValueError("only a captured SAVE can await durable storage")

    async def step(p):
      if not (fenced := await self._fence(p, cmd.turn, cmd.run, engine)):
        return False, False
      pending = await p.xpending_range(q, GROUP, cmd.entry_id, cmd.entry_id, count=1)
      if not pending or pending[0]["consumer"] != cmd.turn.encode():
        return False, False
      run, state = fenced[1], fenced[1]["state"]
      if checkpoint and not await self._version(p, checkpoint["version"], existing=cmd.op == Op.LOAD_WEIGHTS):
        return False, False
      named = bool(checkpoint and reply.ok and not defer and cmd.op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER))
      if named and cmd.op == Op.SAVE_WEIGHTS and not checkpoint.get("durable"):
        return False, False
      previous, sampler = {}, None
      if named:
        alias_run, cid = checkpoint["uri"].removeprefix("tinker://").split("/", 1)
        await p.watch(key("ckpt", alias_run))
        previous = json.loads(raw) if (raw := await p.hget(key("ckpt", alias_run), cid)) else {}
        args = json.loads(cmd.body)
        if sid := args.get("sampling_session_id"):
          await p.watch(key("run", sid), key("q", sid))
          if await p.exists(key("q", sid)) or await p.hexists(key("run", sid), "record"):
            return False, False
          sampler = Run(
            sid,
            run["base"],
            int(checkpoint.get("meta", {}).get("lora", run["rank"] > 0)),
            session=run["session"],
            model_path=checkpoint["uri"],
            exclusive=run["exclusive"],
            role=Role.SAMPLER,
            train_run=cmd.run,
            gpus=args["sampler_gpus"],
            version=checkpoint["version"],
          )
      if reply.ok and cmd.op == Op.DELETE_MODEL:
        await self._remove_run(p, cmd.run, cmd.request_id)
      else:
        p.multi()
        if checkpoint:
          write_version(p, cmd.run, state, checkpoint)
        if reply.ok and mutated:
          state["mutation"] = cmd.entry_id
        if reply.ok and cmd.op == Op.CREATE_MODEL:
          state["created"] = cmd.entry_id
        p.hset(rkey, "state", json.dumps(state))
        if named and entry_key(checkpoint["boundary"]) >= entry_key(previous.get("boundary", "")):
          write_checkpoint(p, checkpoint)
        if sampler:
          write_run(p, sampler)
          p.sadd(key("children", cmd.run), sampler.id)
      ack(p, cmd)
      if defer:
        p.hset(key("saves"), cmd.request_id, json.dumps(checkpoint))
        p.hset(key("refs"), f"save:{cmd.request_id}", split_version(checkpoint["version"])[0])
      else:
        write_future(p, cmd.request_id, reply)
        p.hincrby(rkey, "cmds", 1)
        if not reply.ok or cmd.op == Op.OPTIM_STEP:
          p.hincrby(rkey, "failed" if not reply.ok else "steps", 1)
      if started is not None:
        p.hincrby(rkey, "busy_ms", int((time.time() - started) * 1000))
        p.hincrby(rkey, "wait_ms", max(0, int(started * 1000) - int(cmd.entry_id.split("-")[0])))
      return True, True

    return await self.tx((key("claims"), key("turns"), key("obs"), rkey, q), step)

  async def export(self, turn: str, run: str, checkpoint: dict | None, engine: str) -> bool:
    async def step(p):
      if not (fenced := await self._fence(p, turn, run, engine)):
        return False, False
      record, state = fenced[1], fenced[1]["state"]
      if fenced[0]["work"][run][1] != 0:
        return False, False
      if record["role"] != Role.SAMPLER:
        if checkpoint is None:
          if state["created"] or state["mutation"]:
            return False, False
        elif not checkpoint.get("meta", {}).get("optimizer", checkpoint.get("optimizer", False)) or entry_key(
          checkpoint["boundary"]
        ) < entry_key(state["mutation"]):
          return False, False
      if checkpoint and not await self._version(p, checkpoint["version"]):
        return False, False
      p.multi()
      if checkpoint:
        write_version(p, run, state, checkpoint)
        p.hset(key("run", run), "state", json.dumps(state))
      p.hdel(key("claims"), run)
      return True, True

    return await self.tx((key("claims"), key("turns"), key("obs"), key("run", run)), step)

  # A staged SAVE owns an exact version independently of trainer/turn lifetime.
  async def pending_saves(self) -> dict:
    return {r.decode(): json.loads(v) for r, v in (await self.r.hgetall(key("saves"))).items()}

  async def finish_save(self, request: str, saved: dict) -> bool:
    run, _ = request.split("/", 1)
    checkpoint = saved | {"durable": True}
    alias_run, cid = saved["uri"].removeprefix("tinker://").split("/", 1)
    reply = Reply(
      True, "application/json", json.dumps({"type": Op.SAVE_WEIGHTS, "model_id": run, "path": saved["uri"]}).encode()
    )

    async def step(p):
      if not (raw := await p.hget(key("saves"), request)) or json.loads(raw) != saved:
        return False, False
      if not (raw := await p.hget(key("run", run), "state")) or not await self._version(
        p, saved["version"], existing=True
      ):
        return False, False
      state = json.loads(raw)
      previous = json.loads(raw) if (raw := await p.hget(key("ckpt", alias_run), cid)) else {}
      p.multi()
      write_version(p, run, state, checkpoint)
      p.hset(key("run", run), "state", json.dumps(state))
      if entry_key(saved["boundary"]) >= entry_key(previous.get("boundary", "")):
        write_checkpoint(p, checkpoint)
      write_future(p, request, reply)
      p.hincrby(key("run", run), "cmds", 1)
      p.hdel(key("saves"), request)
      p.hdel(key("refs"), f"save:{request}")
      return True, True

    return await self.tx((key("saves"), key("run", run), key("ckpt", alias_run)), step)

  async def renew_lease(self, engine: str) -> bool:
    return bool(await self.r.set(key("live", engine), 1, xx=True, px=LEASE_MS))

  async def release_lease(self, engine: str) -> None:
    await self.r.delete(key("live", engine))

  async def claim_scheduler(self, holder: str) -> bool:
    return bool(
      await self.r.set(key("owner", "scheduler"), holder, ifeq=holder, px=LEASE_MS)
      or await self.r.set(key("owner", "scheduler"), holder, nx=True, px=LEASE_MS)
    )

  async def node_heartbeat(self, node: str, gpus: int, memory: int = 0, **extra) -> None:
    await self.r.hset(
      key("nodes"), node, json.dumps({"gpus": gpus, "memory": memory, "until": time.time() + NODE_MS / 1000} | extra)
    )

  async def wait_for_reply(self, request_id: str, timeout: float) -> Reply | None:
    got = await self.r.xread({key("f", request_id): "0-0"}, count=1, block=int(timeout * 1000) or None)
    if got:
      fs = got[0][1][0][1]
      return Reply(fs[b"ok"] == b"1", fs[b"content_type"].decode(), fs[b"body"])
    return None

  async def queued_and_in_flight(self, run_id: str) -> dict:
    p = self.r.pipeline()
    p.xinfo_groups(key("q", run_id))
    p.xpending_range(key("q", run_id), GROUP, "-", "+", count=100)
    groups, pending = await p.execute(raise_on_error=False)
    if isinstance(groups, Exception):
      return {"live": False}
    return {
      "live": True,
      "queued": groups[0]["lag"],
      "in_flight": [
        {
          "request_id": f"{run_id}/{e['message_id'].decode()}",
          "trainer": e["consumer"].decode(),
          "seconds": e["time_since_delivered"] / 1000,
          "deliveries": e["times_delivered"],
        }
        for e in pending
      ],
    }
