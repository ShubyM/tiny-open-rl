"""Runs, sessions, command streams, ownership, checkpoints, and plans on Redis.

Metadata lives as UTF-8 strings (`self.s`); command and reply streams carry raw protobuf bytes (`self.r`).
Safety rests on two Redis facts (plus each node's local `on` table):
  tinyrl:run:<r>       a training run's single-writer `(owner, epoch, pos, snap)`, fenced on claim/commit/detach
  tinyrl:lease:<life>  each process incarnation's TTL; `acquire_run` replaces `owner` only after its lease expires
Each run has one FIFO stream and one consumer group. Its pending list records claims across process restarts.
Trainers claim under the current owner token; sampling replicas reclaim abandoned work and commit each entry once.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict

import redis.asyncio as redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff

from tinyrl.control.accounting import Charge
from tinyrl.helpers import getenv
from tinyrl.types import Cmd, Ev, Event, Op, Reply, Role, Run, parse_uri, unpack_snap  # noqa: F401

GROUP = "trainer"  # Persisted group name for both roles.
LIVE_SECONDS, SESSION_GRACE = 30, getenv("SESSION_GRACE", 60)
REPLY_TTL, READ_REPLY_TTL, SESSION_TTL = getenv("REPLY_TTL", 3600), 60, getenv("SESSION_TTL", 7 * 24 * 3600)
LEASE_MS, NODE_MS = getenv("LEASE_MS", 10000), 30000
COUNTERS = ("steps", "failed", "cmds", "tokens", "busy_ms", "wait_ms")
RUN_META = frozenset(("record", "owner", "epoch", "pos", "snap"))


def key(*parts: str) -> str:
  return ":".join(("tinyrl", *parts))


def life_of(token: str) -> str:
  return token.rsplit(":", 1)[0]


def wake(p, stream: str = "wake") -> None:
  p.xadd(key(stream), {"t": 1}, maxlen=1, approximate=False)


def queue_event(p, kind: Ev, body: dict) -> None:
  p.xadd(key("events"), {"kind": Ev(kind), "body": json.dumps(body)}, maxlen=1_000_000, approximate=True)


def parse_cmd(run: str, entry_id: bytes, fields: dict[bytes, bytes]) -> Cmd:
  return Cmd(run, entry_id.decode(), Op(fields[b"op"].decode()), fields[b"body"], int(fields.get(b"cost", 1)))


def parse_heads(got) -> list[Cmd]:
  pre = key("q", "")
  return [Cmd(s.decode().removeprefix(pre), e.decode(), Op(f[b"op"].decode()), b"") for s, rs in got for e, f in rs[:1]]


def is_missing_stream(err: Exception) -> bool:
  return "NOGROUP" in (msg := str(err)) or "no such key" in msg


async def read_stream(command) -> list:
  """A run can be deleted after a worker receives its assignment."""
  try:
    return await command
  except redis.ResponseError as e:
    if not is_missing_stream(e):
      raise
    return []


def finish_cmd(p, cmd: Cmd, reply: Reply, acct: str, charge: Charge | None) -> None:
  q, fk = key("q", cmd.run), key("f", cmd.request_id)
  p.xack(q, GROUP, cmd.entry_id)
  p.xdel(q, cmd.entry_id)
  p.xadd(fk, {"ok": int(reply.ok), "content_type": reply.content_type, "body": reply.body}, maxlen=1, approximate=False)
  p.expire(fk, REPLY_TTL)
  counts = {"failed": 1} if not reply.ok else {"steps": 1} if cmd.op == Op.OPTIM_STEP else {}
  if charge:
    counts.update(cmds=1, wait_ms=max(0, charge.started_ms - int(cmd.entry_id.split("-")[0])), busy_ms=charge.busy_ms)
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
  p.expire(key("run", run_id), SESSION_TTL)
  p.expire(key("clients", run_id), SESSION_TTL)
  p.hdel(key("writing"), run_id)
  queue_event(p, Ev.RUN_DELETED, {"run": run_id})


def checkpoint_field(uri: str) -> tuple[str, str]:
  run, cid = parse_uri(uri)
  return key("ckpt", run), cid


def queue_expiry(p, uri: str, expires: float | None) -> None:
  p.zadd(key("expiry"), {uri: expires}) if expires is not None else p.zrem(key("expiry"), uri)


def queue_checkpoint(p, uri: str, size: int, *, version: str, ttl=None, owned=False) -> None:
  """Publish the record, visible-page index, and expiry together with the fenced reply."""
  ck, cid = checkpoint_field(uri)
  raw_kind, now = cid.split("/", 1)[0], time.time()
  kind = {"weights": "training", "sampler_weights": "sampler"}.get(raw_kind, raw_kind)
  expires = now + ttl if ttl else None
  row = {"checkpoint_id": cid, "checkpoint_type": kind, "time": now, "tinker_path": uri, "size_bytes": size}
  p.hset(ck, cid, json.dumps(row | {"expires_at": expires, "owned": owned, "version": version}))
  if kind != "parked":
    p.zadd(key("ckpts"), {uri: now})
  queue_expiry(p, uri, expires)


def queue_unregister(p, *uris: str) -> None:
  for uri in uris:
    p.hdel(*checkpoint_field(uri))
    p.zrem(key("ckpts"), uri)
    p.zrem(key("expiry"), uri)


def connect(url: str) -> redis.Redis:
  retry = Retry(ExponentialBackoff(cap=2.0, base=0.1), 30)
  return redis.from_url(url, socket_timeout=None, retry=retry, max_connections=10_000)


class Bus:
  def __init__(self, r: redis.Redis | None = None):
    self.r = r if r is not None else connect(getenv("REDIS_URL", "redis://localhost:6379"))
    pool = self.r.connection_pool
    kw = pool.connection_kwargs | {"decode_responses": True, "max_connections": pool.max_connections}
    self.s = redis.Redis(connection_pool=redis.ConnectionPool(connection_class=pool.connection_class, **kw))

  async def aclose(self) -> None:
    await self.s.aclose(close_connection_pool=True)
    await self.r.aclose()

  async def transact(self, fn, *keys: str, raw: bool = False):
    client = self.r if raw else self.s
    while True:
      try:
        async with client.pipeline(transaction=True) as p:
          await p.watch(*keys)
          return await fn(p)
      except redis.WatchError:
        continue

  # ***** sessions *****

  async def heartbeat_session(self, session_id: str, metadata: dict | None = None) -> None:
    k, now, p = key("sess", session_id), time.time(), self.s.pipeline()
    p.hsetnx(k, "created", now)
    if metadata:
      p.hset(k, "metadata", json.dumps(metadata))
    p.hset(k, "seen", now)
    p.zadd(key("seen"), {session_id: now})
    p.expire(k, SESSION_TTL)
    p.expire(key("sess", session_id, "runs"), SESSION_TTL)
    await p.execute()

  async def get_session(self, session_id: str) -> dict | None:
    p = self.s.pipeline()
    p.hgetall(key("sess", session_id))
    p.smembers(key("sess", session_id, "runs"))
    record, runs = await p.execute()
    if not record:
      return None
    seen, meta = float(record["seen"]), json.loads(record.get("metadata") or "{}")
    out = {"session_id": session_id, "created": float(record["created"]), "seen": seen}
    return out | {"live": time.time() - seen < LIVE_SECONDS, "runs": sorted(runs), "metadata": meta}

  async def expired_sessions(self) -> list[str]:
    return await self.s.zrangebyscore(key("seen"), 0, time.time() - SESSION_GRACE)

  async def finish_session(self, session_id: str) -> None:
    p = self.s.pipeline()
    p.zadd(key("seen"), {session_id: 0}, xx=True)
    wake(p)
    await p.execute()

  async def end_session(self, session_id: str) -> None:
    left = False
    for run_id in sorted(await self.s.sinter(key("runs"), key("sess", session_id, "runs"))):
      if (await self.get_run(run_id) or {}).get("role") == Role.SAMPLER:
        await self.delete_run(run_id)
      else:
        left = bool(await self.submit_command(run_id, Op.DELETE_MODEL, b"{}")) or left
    await (self.s.zadd(key("seen"), {session_id: time.time()}) if left else self.s.zrem(key("seen"), session_id))

  # ***** runs and cluster state *****

  async def create_run(self, run: Run) -> None:
    q, p = key("q", run.id), self.s.pipeline(transaction=True)
    zero = dict.fromkeys(COUNTERS, 0)
    p.xgroup_create(q, GROUP, id="0", mkstream=True)
    p.execute_command("XCFGSET", q, "IDMP-DURATION", 86400, "IDMP-MAXSIZE", 10000)
    p.sadd(key("runs"), run.id)
    p.hset(key("run", run.id), mapping={"record": json.dumps({**asdict(run), "created": time.time()}), **zero})
    if run.session:
      p.sadd(key("sess", run.session, "runs"), run.id)
      p.expire(key("sess", run.session, "runs"), SESSION_TTL)
    fields = ("role", "base", "rank", "exclusive", "train_run", "gpus")
    queue_event(p, Ev.RUN_CREATED, {"run": run.id, **{k: getattr(run, k) for k in fields}})
    wake(p)
    await p.execute()

  async def read_plan(self) -> dict:
    return json.loads(await self.s.get(key("plan")) or "{}")

  async def account_service(self) -> dict[str, float]:
    return {a: float(v) for a, v in (await self.s.hgetall(key("service"))).items()}

  async def observations(self, *, backlogs: bool = True, counters: bool = False) -> dict:
    """Current machine capacities, registered runs, queue depths and billed GPU time."""
    nodes, ids, service = await asyncio.gather(
      self.live_nodes(details=True), self.s.smembers(key("runs")), self.account_service()
    )
    runs = sorted(ids)
    p = self.s.pipeline(transaction=False)
    fields = ("record", "owner", *COUNTERS) if counters else ("record", "owner")
    for r in runs:
      p.hmget(key("run", r), fields)
    if backlogs:
      for r in runs:
        p.xlen(key("q", r))
    rows = await p.execute()
    out_runs = {}
    for r, (rec, owner, *values) in zip(runs, rows):
      if rec:
        out_runs[r] = {"record": json.loads(rec), "owner": owner}
        if counters:
          out_runs[r]["counters"] = {k: int(v) for k, v in zip(COUNTERS, values) if v is not None}
    queues = dict(zip(runs, rows[len(runs) :]))
    return {"nodes": nodes, "runs": out_runs, "queues": queues, "service": service}

  async def read_state(self, *, backlogs: bool = True, counters: bool = False) -> dict:
    observed, plan = await asyncio.gather(self.observations(backlogs=backlogs, counters=counters), self.read_plan())
    return observed | {"plan": plan}

  async def write_plan(self, plan: dict, changes: list[tuple[Ev, dict]] = (), notify_nodes: bool = True) -> None:
    p = self.s.pipeline(transaction=True)
    p.set(key("plan"), json.dumps(plan))
    for kind, body in changes:
      queue_event(p, kind, body)
    if notify_nodes:
      wake(p, "plan_wake")
    await p.execute()

  async def get_run(self, run_id: str) -> dict | None:
    if not (f := await self.s.hgetall(key("run", run_id))) or "record" not in f:
      return None
    meta = {n: f[n] for n in ("owner", "pos", "snap") if f.get(n)}
    return {**json.loads(f["record"]), **meta, **{k: int(v) for k, v in f.items() if k not in RUN_META}}

  async def join_sampling_session(self, run_id: str) -> int:
    return int(await self.s.incr(key("clients", run_id)))

  async def delete_run(self, run_id: str) -> None:
    p = self.s.pipeline(transaction=True)
    queue_delete(p, run_id)
    wake(p)
    await p.execute()

  # ***** commands & replies *****

  async def seen_request(self, run_id: str, seq_id: int | None) -> str | None:
    return await self.s.get(key("seq", run_id, str(seq_id))) if seq_id is not None else None

  async def submit_command(
    self, run_id: str, op: Op, body: bytes, seq_id: int | None = None, cost: int = 1
  ) -> str | None:
    if first := await self.seen_request(run_id, seq_id):
      return first
    seq = [] if seq_id is None else ["IDMP", run_id, seq_id]
    args = ["NOMKSTREAM", *seq, "*", "op", op, "body", body, "cost", cost]
    if (entry_id := await self.r.execute_command("XADD", key("q", run_id), *args)) is None:
      return None
    request_id, p = f"{run_id}/{entry_id.decode()}", self.s.pipeline()
    if seq:
      p.set(key("seq", run_id, str(seq_id)), request_id, ex=SESSION_TTL)
    wake(p)
    await p.execute()
    return request_id

  async def wait_for_reply(self, request_id: str, timeout: float) -> Reply | None:
    fk = key("f", request_id)
    if not (got := await self.r.xread({fk: "0-0"}, count=1, block=int(timeout * 1000) or None)):
      return None
    f = got[0][1][0][1]
    await self.s.expire(fk, min(REPLY_TTL, READ_REPLY_TTL), lt=True)
    return Reply(f[b"ok"] == b"1", f[b"content_type"].decode(), f[b"body"])

  async def queued_and_in_flight(self, run_id: str) -> dict:
    q = key("q", run_id)
    p = self.s.pipeline()
    p.xinfo_groups(q)
    p.xpending_range(q, GROUP, "-", "+", count=100)
    if not (status := await read_stream(p.execute())):
      return {"live": False}
    groups, pending = status
    in_flight = [
      {
        "request_id": f"{run_id}/{e['message_id']}",
        "trainer": e["consumer"],
        "deliveries": e["times_delivered"],
        "seconds": e["time_since_delivered"] / 1000,
      }
      for e in pending
    ]
    return {"live": True, "queued": groups[0]["lag"], "in_flight": in_flight}

  # ***** single-writer trainer ownership & leases *****

  async def touch_leases(self, *lives: str, xx: bool = False) -> None:
    p = self.s.pipeline(transaction=False)
    for life in lives:
      p.set(key("lease", life), 1, px=LEASE_MS, xx=xx)
    await p.execute()

  async def drop_lease(self, life: str) -> None:
    await self.s.delete(key("lease", life))

  async def acquire_run(self, run_id: str, life: str) -> tuple[str, str | None, str | None] | None:
    """Claim an unowned (or expired-owner) training run under a fresh epoch token if `life`'s lease is live."""
    k, lk = key("run", run_id), key("lease", life)

    async def op(p):
      rec, owner, epoch, pos, snap = await p.hmget(k, "record", "owner", "epoch", "pos", "snap")
      # Deletion drops the queue and updates the watched run hash, which survives for inspection.
      if not rec or await p.exists(lk, key("q", run_id)) != 2:
        return None
      if owner:
        await p.watch(old_lk := key("lease", life_of(owner)))
        if await p.exists(old_lk):
          return None
      n = int(epoch or 0) + 1
      p.multi()
      p.hset(k, mapping={"owner": f"{life}:{n}", "epoch": n})
      if owner:
        p.hdel(key("writing"), run_id)
      await p.execute()
      return f"{life}:{n}", pos, snap

    return await self.transact(op, k, lk)

  async def peek_trainer(self, runs: list[str], block_ms: int = 0) -> list[Cmd]:
    """Peek one body-free head per trainer stream without claiming it."""
    if not runs:
      return await asyncio.sleep(block_ms / 1000) or []
    got = await self.r.xread({key("q", r): "0-0" for r in runs}, count=1, block=block_ms or None)
    return sorted(parse_heads(got), key=lambda c: (*map(int, c.entry_id.split("-")), c.run))

  async def claim_trainer(self, run: str, token: str) -> Cmd | None:
    """Mark the head of `run`'s FIFO stream in-flight under `token` (only the current epoch may claim)."""
    k, q = key("run", run), key("q", run)

    async def op(p):
      if await p.hget(k, "owner") != token.encode():
        raise RuntimeError(f"lost ownership of {run}")
      p.multi()
      p.xreadgroup(GROUP, life_of(token), {q: ">"}, count=1)
      [rows] = await p.execute()
      return parse_cmd(run, *rows[0][1][0]) if rows else None

    return await self.transact(op, k, raw=True)

  async def begin_checkpoint(self, run: str, token: str) -> str:
    """Protect this write from collection until this owner publishes it, fails it, or loses its lease."""
    k, version = key("run", run), uuid.uuid4().hex

    async def op(p):
      if await p.hget(k, "owner") != token:
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
      if await p.hmget(k, "owner", "pos") != [token, pos]:
        return False
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
  ) -> bool:
    """Atomically publish a trainer reply, bill `gpu_ms`, and advance `pos` (clearing `snap`) iff `token` owns it."""
    k, ck, deleting = key("run", cmd.run), key("ckpt", cmd.run), cmd.op == Op.DELETE_MODEL and reply.ok

    async def op(p):
      owner, old_snap = await p.hmget(k, "owner", "snap")
      if owner != token.encode() or not await p.xrange(key("q", cmd.run), cmd.entry_id, cmd.entry_id):
        return False
      records = map(json.loads, await p.hvals(ck)) if deleting else ()
      p.multi()
      p.hdel(key("writing"), cmd.run)
      if checkpoint and reply.ok:
        queue_checkpoint(p, **checkpoint)
      finish_cmd(p, cmd, reply, acct, charge)
      if mutates and not deleting:
        p.hset(k, "pos", cmd.entry_id)
        p.hdel(k, "snap")
        if old_snap:
          queue_unregister(p, unpack_snap(old_snap.decode())[0])
      elif deleting:
        queue_unregister(p, *(r["tinker_path"] for r in records if r["owned"]))
        queue_delete(p, cmd.run)
      return bool(await p.execute())

    return bool(await self.transact(op, k, key("f", cmd.request_id), *([ck] if deleting else []), raw=True))

  async def fail_stranded_commands(self, run_id: str, token: str) -> None:
    """Fail the command left claimed by a dead trainer incarnation before serving new ones."""
    start = "-"
    while rows := await self.pending(run_id, start, count=1):
      eid = rows[0]["message_id"]
      start = f"({eid}"
      if entry := await self.r.xrange(key("q", run_id), eid, eid):
        cmd = parse_cmd(run_id, *entry[0])
        err = Reply.error(RuntimeError(f"its trainer died, or lost ownership, during {cmd.op}"))
        if not await self.commit_reply(cmd, err, token):
          return

  async def pending(self, run: str, start: str = "-", count: int = 1000) -> list[dict]:
    return await read_stream(self.s.xpending_range(key("q", run), GROUP, start, "+", count=count))

  # ***** stateless sampler consumer groups *****

  async def abandoned(self, run: str) -> list[str]:
    """Pending sample IDs whose consumer holds no lease, oldest first (live slow replicas are never touched)."""
    rows = await self.pending(run)
    if not rows:
      return []
    holders = sorted({row["consumer"] for row in rows})
    leases = await self.s.mget([key("lease", holder) for holder in holders])
    dead = {holder for holder, live in zip(holders, leases) if live is None}
    return [row["message_id"] for row in rows if row["consumer"] in dead]

  async def peek_sampler(self, runs: list[str], block_ms: int = 0) -> list[Cmd]:
    """Peek one unclaimed or dead-replica-abandoned body-free head per sampler stream."""
    p = self.s.pipeline(transaction=False)
    for r in runs:
      p.xinfo_groups(key("q", r))
    cursors, pending = {}, []
    for r, groups in zip(runs, await p.execute(raise_on_error=False)):
      if isinstance(groups, Exception):
        if not is_missing_stream(groups):
          raise groups
      elif g := next((g for g in groups if g["name"] == GROUP), None):
        cursors[key("q", r)] = g["last-delivered-id"]
        if g["pending"]:
          pending.append(r)
    if not cursors:
      return await asyncio.sleep(block_ms / 1000) or []
    cmds = [
      Cmd(r, ids[0], Op.SAMPLE, b"")
      for r, ids in zip(pending, await asyncio.gather(*(self.abandoned(r) for r in pending)))
      if ids
    ]
    for cmd in cmds:
      del cursors[key("q", cmd.run)]
    if cursors:
      cmds.extend(parse_heads(await self.r.xread(cursors, count=1, block=None if cmds else block_ms or None)))
    return sorted(cmds, key=lambda c: (*map(int, c.entry_id.split("-")), c.run))

  async def take_commands(self, runs: list[str], consumer: str, count: int = 16) -> list[Cmd]:
    """Claim up to `count` samples per session: dead-replica abandoned entries first, then fresh entries."""
    cmds, rem = [], dict.fromkeys(runs, count)
    for r in runs:
      if ids := (await self.abandoned(r))[:count]:
        # Idle for a second so two live replicas reclaiming at once don't take the same entries from each other.
        got = await read_stream(self.r.xclaim(key("q", r), GROUP, consumer, 1000, ids))
        taken = [parse_cmd(r, eid, fields) for eid, fields in got if fields]
        cmds.extend(taken)
        rem[r] -= len(taken)
    if not (fresh := [r for r in runs if rem[r] > 0]):
      return cmds
    p = self.r.pipeline(transaction=False)
    for r in fresh:
      p.xreadgroup(GROUP, consumer, {key("q", r): ">"}, count=rem[r])
    for r, got in zip(fresh, await p.execute(raise_on_error=False)):
      if isinstance(got, Exception):
        if not is_missing_stream(got):
          raise got
      else:
        cmds.extend(parse_cmd(r, eid, fields) for _, entries in got for eid, fields in entries)
    return cmds

  async def commit_sample(self, cmd: Cmd, reply: Reply, account="", charge: Charge | None = None) -> bool:
    """The winning replica removes the queue entry. A late duplicate stays fenced after the reply expires."""
    fk = key("f", cmd.request_id)

    async def op(p):
      if not await p.xrange(key("q", cmd.run), cmd.entry_id, cmd.entry_id):
        return False
      p.multi()
      finish_cmd(p, cmd, reply, account, charge)
      return bool(await p.execute())

    return bool(await self.transact(op, fk, raw=True))

  # ***** checkpoints *****

  async def _checkpoint_rows(self, *uris: str) -> dict[str, dict]:
    p = self.s.pipeline(transaction=False)
    for uri in uris:
      p.hget(*checkpoint_field(uri))
    return {uri: json.loads(raw) for uri, raw in zip(uris, await p.execute()) if raw}

  async def get_checkpoint(self, uri: str) -> dict | None:
    return (await self._checkpoint_rows(uri)).get(uri)

  async def checkpoint_version(self, uri: str) -> str:
    if version := (await self.get_checkpoint(uri) or {}).get("version"):
      return version
    raise ValueError(f"no checkpoint at {uri}")

  async def checkpoint_versions(self, uris: list[str]) -> dict[str, str | Exception]:
    rows = await self._checkpoint_rows(*uris)
    return {u: rows.get(u, {}).get("version") or ValueError(f"no checkpoint at {u}") for u in uris}

  async def checkpoints_of_run(self, run: str) -> list[dict]:
    rows = map(json.loads, await self.s.hvals(key("ckpt", run)))
    return sorted((c for c in rows if c["checkpoint_type"] != "parked"), key=lambda c: c["time"])

  async def page_of_checkpoints(self, limit: int, offset: int) -> tuple[list[dict], int]:
    ck = key("ckpts")
    uris = await self.s.zrevrange(ck, offset, offset + limit - 1) if limit > 0 else []
    return list((await self._checkpoint_rows(*uris)).values()), await self.s.zcard(ck)

  async def set_checkpoint_ttl(self, uri: str, ttl: int | None) -> bool:
    ck, cid = checkpoint_field(uri)

    async def update(p):
      if not (raw := await p.hget(ck, cid)):
        return False
      expires = time.time() + ttl if ttl else None
      p.multi()
      p.hset(ck, cid, json.dumps(json.loads(raw) | {"expires_at": expires}))
      queue_expiry(p, uri, expires)
      return bool(await p.execute())

    return await self.transact(update, ck)

  async def unregister_checkpoints(self, *uris: str) -> int:
    p = self.s.pipeline()
    queue_unregister(p, *uris)
    return sum((await p.execute())[::3])

  async def expire_checkpoints(self) -> int:
    expiry, now = key("expiry"), time.time()
    if not await self.s.zcount(expiry, "-inf", now):
      return 0

    async def expire(p):
      due = await p.zrangebyscore(expiry, "-inf", now)
      p.multi()
      queue_unregister(p, *due)
      return sum((await p.execute())[::3])

    return await self.transact(expire, expiry)

  async def sweep_checkpoints(self, store, grace: float) -> None:
    """Read candidates, write intents, then published versions: an in-flight save spans the last two sets."""
    unused = await asyncio.to_thread(store.versions)
    unused.difference_update(await self.s.hvals(key("writing")))
    async for ck in self.s.scan_iter(match=key("ckpt", "*")):
      for row in map(json.loads, await self.s.hvals(ck)):
        unused.discard((row.get("version") or "").partition("@")[0])
    await asyncio.to_thread(store.sweep, unused, grace)

  # ***** nodes and events *****

  async def node_heartbeat(self, node: str, gpus: int, memory: int = 0, leases: tuple[str, ...] = (), **info) -> None:
    body = json.dumps({"gpus": gpus, "memory": memory, "until": time.time() + NODE_MS / 1000, **info})
    p = self.s.pipeline()
    p.hset(key("nodes"), node, body)
    for life in leases:
      p.set(key("lease", life), 1, px=LEASE_MS, xx=True)
    await p.execute()

  async def live_nodes(self, details: bool = False) -> dict:
    nodes = {k: json.loads(v) for k, v in (await self.s.hgetall(key("nodes"))).items()}
    if dead := [k for k, v in nodes.items() if v["until"] < time.time()]:
      await self.s.hdel(key("nodes"), *dead)
    return {k: v if details else v["gpus"] for k, v in nodes.items() if k not in dead}

  async def append_event(self, _kind: Ev, **body) -> str:
    p = self.s.pipeline()
    queue_event(p, _kind, body)
    wake(p)
    return (await p.execute())[0]

  async def wait_wake(self, wake_id: str, block_ms: int, stream: str = "wake") -> str:
    got = await self.s.xread({key(stream): wake_id}, count=1, block=block_ms or None)
    return got[0][1][-1][0] if got else wake_id

  async def read_events(self, after: str = "0-0", count: int = 10_000) -> list[Event]:
    rows = got[0][1] if (got := await self.s.xread({key("events"): after}, count=count)) else ()
    return [Event(eid, int(eid.split("-")[0]) / 1000, f["kind"], json.loads(f["body"])) for eid, f in rows]
