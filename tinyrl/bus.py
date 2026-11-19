"""Redis control state and command streams.

Metadata lives as plain UTF-8 strings and JSON (`self.s`); only command and reply streams carry raw protobuf bytes
(`self.r`). Safety rests on two facts (plus each node's local `on` table):
  tinyrl:run:<r>       a training run's single-writer `(owner, epoch, pos, snap, claimed)`, fenced on claim/commit/detach
  tinyrl:lease:<life>  each worker incarnation's TTL; `acquire_run` replaces `owner` only after its lease expires
Trainer runs are single-writer FIFO streams (no consumer group). Sampling sessions are stateless multi-replica readers
sharing consumer group `sampler` on `q:<s>`, with `commit_sample` answering and billing each entry once.
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

GROUP = "sampler"
LIVE_SECONDS, SESSION_GRACE = 30, getenv("SESSION_GRACE", 60)
REPLY_TTL, READ_REPLY_TTL, SESSION_TTL = getenv("REPLY_TTL", 3600), 60, getenv("SESSION_TTL", 7 * 24 * 3600)
LEASE_MS, NODE_MS = getenv("LEASE_MS", 10000), 30000
RUN_META = frozenset(("record", "owner", "epoch", "pos", "snap", "claimed"))


def key(*parts: str) -> str:
  return ":".join(("tinyrl", *parts))


def life_of(token: str) -> str:
  return token.rsplit(":", 1)[0]


def wake(p, stream: str = "wake") -> None:
  p.xadd(key(stream), {"t": 1}, maxlen=1, approximate=False)


def log_event(p, kind: Ev, body: dict) -> None:
  p.xadd(key("events"), {"kind": Ev(kind), "body": json.dumps(body)}, maxlen=1_000_000, approximate=True)


def parse_cmd(run: str, entry_id: bytes, fields: dict[bytes, bytes]) -> Cmd:
  return Cmd(run, entry_id.decode(), Op(fields[b"op"].decode()), fields[b"body"], int(fields.get(b"cost", 1)))


def is_missing_stream(err: Exception) -> bool:
  msg = str(err)
  return "NOGROUP" in msg or "no such key" in msg


def queue_checkpoint(p, uri: str, size: int, *, version: str, ttl=None, owned=False) -> None:
  """Write a checkpoint entry into the single `tinyrl:ckpts` hash inside the caller's transaction."""
  _, cid = parse_uri(uri)
  raw_kind, now = cid.split("/", 1)[0], time.time()
  kind = {"weights": "training", "sampler_weights": "sampler"}.get(raw_kind, raw_kind)
  info = {"checkpoint_id": cid, "checkpoint_type": kind, "time": now, "tinker_path": uri, "size_bytes": size}
  p.hset(
    key("ckpts"),
    uri,
    json.dumps({**info, "expires_at": now + ttl if ttl else None, "owned": owned, "version": version}),
  )


def queue_delete(p, run_id: str) -> None:
  p.srem(key("runs"), run_id)
  p.delete(key("q", run_id))
  p.hdel(key("run", run_id), "owner", "snap", "claimed")
  p.expire(key("run", run_id), SESSION_TTL)
  p.expire(key("clients", run_id), SESSION_TTL)
  p.hdel(key("writing"), run_id)
  log_event(p, Ev.RUN_DELETED, {"run": run_id})


def finish_cmd(p, cmd: Cmd, reply: Reply, acct: str, charge: Charge | None, *, grouped: bool) -> None:
  q, fk = key("q", cmd.run), key("f", cmd.request_id)
  if grouped:
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


def connect(url: str, *, decode: bool = False) -> redis.Redis:
  retry = Retry(ExponentialBackoff(cap=2.0, base=0.1), 30)
  return redis.from_url(url, socket_timeout=None, retry=retry, max_connections=10_000, decode_responses=decode)


class Bus:
  def __init__(self, r: redis.Redis | None = None):
    self.r = r if r is not None else connect(getenv("REDIS_URL", "redis://localhost:6379"))
    pool = self.r.connection_pool
    kw = pool.connection_kwargs | {"decode_responses": True, "max_connections": pool.max_connections}
    self.s = redis.Redis(connection_pool=redis.ConnectionPool(connection_class=pool.connection_class, **kw))

  async def aclose(self) -> None:
    await self.s.aclose()
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
    base = {"session_id": session_id, "created": float(record["created"]), "seen": seen}
    return {**base, "live": time.time() - seen < LIVE_SECONDS, "runs": sorted(runs), "metadata": meta}

  async def expired_sessions(self) -> list[str]:
    return await self.s.zrangebyscore(key("seen"), 0, time.time() - SESSION_GRACE)

  async def finish_session(self, session_id: str) -> None:
    p = self.s.pipeline()
    p.zadd(key("seen"), {session_id: 0}, xx=True)
    wake(p)
    await p.execute()

  async def end_session(self, session_id: str) -> None:
    deleting = False
    for run_id in (await self.get_session(session_id) or {}).get("runs", []):
      if not await self.s.sismember(key("runs"), run_id):
        continue
      if (await self.get_run(run_id))["role"] == Role.SAMPLER:
        await self.delete_run(run_id)
      else:
        deleting = bool(await self.submit_command(run_id, Op.DELETE_MODEL, b"{}")) or deleting
    await (self.s.zadd(key("seen"), {session_id: time.time()}) if deleting else self.s.zrem(key("seen"), session_id))

  # ***** runs, nodes, and cluster plan *****

  async def create_run(self, run: Run) -> None:
    q, p = key("q", run.id), self.s.pipeline(transaction=True)
    zero = dict.fromkeys(("steps", "failed", "cmds", "tokens", "busy_ms", "wait_ms"), 0)
    if run.role == Role.SAMPLER:
      p.xgroup_create(q, GROUP, id="0", mkstream=True)
    else:
      p.xadd(q, {"_": ""}, id="0-1")
      p.xdel(q, "0-1")
    p.execute_command("XCFGSET", q, "IDMP-DURATION", 86400, "IDMP-MAXSIZE", 10000)
    p.sadd(key("runs"), run.id)
    p.hset(key("run", run.id), mapping={"record": json.dumps({**asdict(run), "created": time.time()}), **zero})
    if run.session:
      p.sadd(key("sess", run.session, "runs"), run.id)
      p.expire(key("sess", run.session, "runs"), SESSION_TTL)
    fields = ("role", "base", "rank", "exclusive", "train_run", "gpus")
    log_event(p, Ev.RUN_CREATED, {"run": run.id, **{k: getattr(run, k) for k in fields}})
    wake(p)
    await p.execute()

  async def get_run(self, run_id: str) -> dict | None:
    if not (f := await self.s.hgetall(key("run", run_id))) or "record" not in f:
      return None
    meta = {n: f[n] for n in ("owner", "pos", "snap") if f.get(n)}
    counts = {k: int(v) for k, v in f.items() if k not in RUN_META}
    return {**json.loads(f["record"]), **meta, **counts}

  async def delete_run(self, run_id: str) -> None:
    p = self.s.pipeline(transaction=True)
    queue_delete(p, run_id)
    wake(p)
    await p.execute()

  async def join_sampling_session(self, run_id: str) -> int:
    return int(await self.s.incr(key("clients", run_id)))

  async def read_state(self, *, backlogs: bool = True) -> dict:
    """Cluster snapshot: live nodes, runs, plan, account `gpu_ms` service, and (for the scheduler) queue depths."""
    nodes, runs = await self.live_nodes(details=True), sorted(await self.s.smembers(key("runs")))
    p = self.s.pipeline(transaction=False)
    for r in runs:
      p.hmget(key("run", r), "record", "owner")
      if backlogs:
        p.xlen(key("q", r))  # Commits delete their entry, so xlen counts queued + in-flight commands.
    p.get(key("plan"))
    p.hgetall(key("service"))
    *rows, plan, service = await p.execute()
    step, out_runs, queues = (2 if backlogs else 1), {}, {}
    for i, r in enumerate(runs):
      rec, owner = rows[i * step]
      if rec:
        out_runs[r] = {"record": json.loads(rec), "owner": owner}
        if backlogs:
          queues[r] = rows[i * step + 1]
    svc = {k: float(v) for k, v in service.items()}
    return {"nodes": nodes, "runs": out_runs, "queues": queues, "plan": json.loads(plan or "{}"), "service": svc}

  async def read_plan(self) -> dict:
    return json.loads(await self.s.get(key("plan")) or "{}")

  async def account_service(self) -> dict[str, float]:
    return {k: float(v) for k, v in (await self.s.hgetall(key("service"))).items()}

  async def write_plan(self, plan: dict, changes: list[tuple[Ev, dict]] = (), notify_nodes: bool = True) -> None:
    p = self.s.pipeline(transaction=True)
    p.set(key("plan"), json.dumps(plan))
    for kind, body in changes:
      log_event(p, kind, body)
    if notify_nodes:
      wake(p, "plan_wake")
    await p.execute()

  async def run_counters(self) -> dict[tuple[str, str], dict]:
    runs, p = sorted(await self.s.smembers(key("runs"))), self.s.pipeline()
    for r in runs:
      p.hgetall(key("run", r))
    p.get(key("plan"))
    *rows, plan_raw = await p.execute()
    home = json.loads(plan_raw or "{}").get("home", {})
    counts = lambda f: {k: int(v) for k, v in f.items() if k not in RUN_META}  # noqa: E731
    return {(ps[0], r): counts(f) for r, f in zip(runs, rows) if "record" in f and (ps := home.get(r))}

  async def node_heartbeat(self, node: str, gpus: int, memory: int = 0, **info) -> None:
    body = json.dumps({"gpus": gpus, "memory": memory, "until": time.time() + NODE_MS / 1000, **info})
    await self.s.hset(key("nodes"), node, body)

  async def live_nodes(self, details: bool = False) -> dict:
    nodes = {k: json.loads(v) for k, v in (await self.s.hgetall(key("nodes"))).items()}
    if dead := [k for k, v in nodes.items() if v["until"] < time.time()]:
      await self.s.hdel(key("nodes"), *dead)
    return {k: v if details else v["gpus"] for k, v in nodes.items() if k not in dead}

  # ***** submitting commands and reading replies *****

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
    _, fields = got[0][1][0]
    await self.s.expire(fk, min(REPLY_TTL, READ_REPLY_TTL), lt=True)
    return Reply(fields[b"ok"] == b"1", fields[b"content_type"].decode(), fields[b"body"])

  async def queued_and_in_flight(self, run_id: str) -> dict:
    q = key("q", run_id)
    if not await self.s.exists(q):
      return {"live": False}

    def entry(eid: str, who: str, sec: float = 0.0, n: int = 1) -> dict:
      return {"request_id": f"{run_id}/{eid}", "trainer": who, "seconds": sec, "deliveries": n}

    if (await self.get_run(run_id) or {}).get("role") == Role.SAMPLER:
      groups = await self.s.xinfo_groups(q)
      pending = await self.s.xpending_range(q, GROUP, min="-", max="+", count=100)
      active = [
        entry(e["message_id"], e["consumer"], e["time_since_delivered"] / 1000, e["times_delivered"]) for e in pending
      ]
      return {"live": True, "queued": groups[0]["lag"], "in_flight": active}
    total = await self.s.xlen(q)
    claimed, owner = await self.s.hmget(key("run", run_id), "claimed", "owner")
    in_flight = [entry(claimed, life_of(owner or ""))] if claimed else []
    return {"live": True, "queued": max(0, total - len(in_flight)), "in_flight": in_flight}

  # ***** trainer ownership and single-writer commands *****

  async def open_lease(self, life: str) -> bool:
    return bool(await self.s.set(key("lease", life), 1, nx=True, px=LEASE_MS))

  async def renew_lease(self, life: str, recover: bool = False) -> bool:
    return bool(await self.s.set(key("lease", life), 1, xx=not recover, px=LEASE_MS))

  async def drop_lease(self, life: str) -> None:
    await self.s.delete(key("lease", life))

  async def acquire_run(self, run_id: str, life: str) -> tuple[str, str | None, str | None] | None:
    """Acquire `run_id` under a fresh epoch iff `life`'s lease is live and any prior owner's lease has expired."""
    k, lk = key("run", run_id), key("lease", life)

    async def op(p):
      rec, owner, epoch, pos, snap = await p.hmget(k, "record", "owner", "epoch", "pos", "snap")
      if not rec or not await p.exists(lk):
        return None
      if owner:
        await p.watch(old_lease := key("lease", life_of(owner)))
        if await p.exists(old_lease):
          return None
      n, token = int(epoch or 0) + 1, f"{life}:{int(epoch or 0) + 1}"
      p.multi()
      p.hset(k, mapping={"owner": token, "epoch": n})
      p.hdel(key("writing"), run_id)
      await p.execute()
      return token, pos, snap

    return await self.transact(op, k, lk)

  async def peek_trainer(self, runs: list[str], block_ms: int = 0, after: dict[str, str] | None = None) -> list[Cmd]:
    """Read the next unexecuted command on each training run without claiming it."""
    if not runs:
      return await asyncio.sleep(block_ms / 1000) or []
    got = await self.r.xread({key("q", r): (after or {}).get(r, "0-0") for r in runs}, count=1, block=block_ms or None)
    return [parse_cmd(s.decode().removeprefix(key("q", "")), *entries[0]) for s, entries in got]

  async def claim_trainer(self, run: str, token: str) -> Cmd | None:
    """Mark the head of `run`'s stream in-flight under `token` (only the current epoch may claim)."""
    k, q = key("run", run), key("q", run)

    async def op(p):
      if await p.hget(k, "owner") != token.encode():
        raise RuntimeError(f"lost ownership of {run}")
      if not (rows := await p.xrange(q, "-", "+", count=1)):
        return None
      cmd = parse_cmd(run, *rows[0])
      p.multi()
      p.hset(k, "claimed", cmd.entry_id)
      await p.execute()
      return cmd

    return await self.transact(op, k, q, raw=True)

  async def begin_checkpoint(self, run: str, token: str) -> str:
    """Pin a fresh checkpoint version against GC until this owner commits, detaches, or loses its lease."""
    k, version = key("run", run), uuid.uuid4().hex

    async def op(p):
      if await p.hget(k, "owner") != token:
        raise RuntimeError(f"lost ownership of {run}")
      p.multi()
      p.hset(key("writing"), run, version)
      await p.execute()
      return version

    return await self.transact(op, k)

  async def detach_run(self, run_id: str, token: str, pos: str | None, snap: str | None, checkpoint=None) -> bool:
    """Release `run_id` and record its parked snapshot `snap` iff `token` still owns it at exact `pos`."""
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
  ):
    """Atomically publish a trainer reply, bill `gpu_ms`, and advance `pos` (clearing `snap`) iff `token` owns it."""
    k, ck, deleting = key("run", cmd.run), key("ckpts"), cmd.op == Op.DELETE_MODEL and reply.ok
    prefix = f"tinker://{cmd.run}/".encode()

    async def op(p):
      owner, old_snap, claimed = await p.hmget(k, "owner", "snap", "claimed")
      if owner != token.encode() or not await p.xrange(key("q", cmd.run), cmd.entry_id, cmd.entry_id):
        return False
      all_ckpts = await p.hgetall(ck) if deleting else {}
      owned = [u for u, d in all_ckpts.items() if u.startswith(prefix) and json.loads(d)["owned"]]
      p.multi()
      p.hdel(key("writing"), cmd.run)
      if claimed == cmd.entry_id.encode():
        p.hdel(k, "claimed")
      if checkpoint and reply.ok:
        queue_checkpoint(p, **checkpoint)
      finish_cmd(p, cmd, reply, acct, charge, grouped=False)
      if mutates:
        p.hset(k, "pos", cmd.entry_id)
        p.hdel(k, "snap")
        if old_snap:
          p.hdel(ck, unpack_snap(old_snap.decode())[0])
      elif deleting:
        if owned:
          p.hdel(ck, *owned)
        queue_delete(p, cmd.run)
      return bool(await p.execute())

    return bool(await self.transact(op, k, key("f", cmd.request_id), *([ck] if deleting else []), raw=True))

  async def fail_stranded_commands(self, run_id: str, token: str) -> None:
    """Fail any command left claimed by a dead trainer incarnation before serving new ones."""
    eid = await self.s.hget(key("run", run_id), "claimed")
    if eid and (rows := await self.r.xrange(key("q", run_id), eid, eid)):
      cmd = parse_cmd(run_id, *rows[0])
      err = Reply.error(RuntimeError(f"its trainer died, or lost ownership, during {cmd.op}"))
      await self.commit_reply(cmd, err, token)

  # ***** stateless sampler consumer groups *****

  async def _abandoned_samples(self, session: str) -> list[str]:
    """Pending sample IDs whose consumer lease has expired (never touches slow live replicas)."""
    try:
      rows = await self.s.xpending_range(key("q", session), GROUP, "-", "+", count=1000)
    except redis.ResponseError as e:
      if is_missing_stream(e):
        return []
      raise
    if not rows:
      return []
    holders, p = sorted({row["consumer"] for row in rows}), self.s.pipeline(transaction=False)
    for holder in holders:
      p.exists(key("lease", holder))
    dead = {holder for holder, live in zip(holders, await p.execute()) if not live}
    return [row["message_id"] for row in rows if row["consumer"] in dead]

  async def _first_abandoned(self, session: str) -> Cmd | None:
    for eid in await self._abandoned_samples(session):
      if rows := await self.r.xrange(key("q", session), eid, eid):
        return parse_cmd(session, *rows[0])
    return None

  async def peek_samples(self, sessions: list[str], block_ms: int = 0) -> list[Cmd]:
    """Read one ready head per sampling session: dead-replica pending entries first, else next undelivered."""
    p = self.s.pipeline(transaction=False)
    for s in sessions:
      p.xinfo_groups(key("q", s))
    cursors, check_dead = {}, []
    for s, groups in zip(sessions, await p.execute(raise_on_error=False)):
      if isinstance(groups, Exception):
        if not is_missing_stream(groups):
          raise groups
      else:
        cursors[key("q", s)] = groups[0]["last-delivered-id"]
        if groups[0]["pending"]:
          check_dead.append(s)
    if not cursors:
      return await asyncio.sleep(block_ms / 1000) or []
    cmds = [c for c in await asyncio.gather(*(self._first_abandoned(s) for s in check_dead)) if c]
    for cmd in cmds:
      cursors.pop(key("q", cmd.run), None)
    if cursors:
      got = await self.r.xread(cursors, count=1, block=None if cmds else block_ms or None)
      cmds.extend(parse_cmd(s.decode().removeprefix(key("q", "")), *entries[0]) for s, entries in got)
    return cmds

  async def claim_samples(self, sessions: list[str], consumer: str, count: int = 16) -> list[Cmd]:
    """Claim up to `count` samples per session: abandoned dead-replica entries first, then fresh ones."""
    cmds, rem = [], dict.fromkeys(sessions, count)
    for s in sessions:
      if ids := (await self._abandoned_samples(s))[:count]:
        # Require 1s idle so two replicas reclaiming at once don't immediately re-steal each other's claim.
        try:
          taken = [parse_cmd(s, eid, f) for eid, f in await self.r.xclaim(key("q", s), GROUP, consumer, 1000, ids) if f]
        except redis.ResponseError as e:
          if is_missing_stream(e):
            continue
          raise
        cmds.extend(taken)
        rem[s] -= len(taken)
    if not (fresh := [s for s in sessions if rem[s] > 0]):
      return cmds
    p = self.r.pipeline(transaction=False)
    for s in fresh:
      p.xreadgroup(GROUP, consumer, {key("q", s): ">"}, count=rem[s])
    for s, got in zip(fresh, await p.execute(raise_on_error=False)):
      if isinstance(got, Exception):
        if not is_missing_stream(got):
          raise got
      else:
        cmds.extend(parse_cmd(s, eid, fields) for _, entries in got for eid, fields in entries)
    return cmds

  async def commit_sample(self, cmd: Cmd, reply: Reply, account="", charge: Charge | None = None) -> bool:
    """Publish a sample reply once: removing the stream entry fences late duplicate replicas forever."""
    fk = key("f", cmd.request_id)

    async def op(p):
      if not await p.xrange(key("q", cmd.run), cmd.entry_id, cmd.entry_id):
        return False
      p.multi()
      finish_cmd(p, cmd, reply, account, charge, grouped=True)
      return bool(await p.execute())

    return bool(await self.transact(op, fk, raw=True))

  # ***** checkpoints (single `tinyrl:ckpts` hash: uri -> json) *****

  async def get_checkpoint(self, uri: str) -> dict | None:
    parse_uri(uri)
    raw = await self.s.hget(key("ckpts"), uri)
    return json.loads(raw) if raw else None

  async def checkpoint_version(self, uri: str) -> str:
    if version := (await self.get_checkpoint(uri) or {}).get("version"):
      return version
    raise ValueError(f"no checkpoint at {uri}")

  async def _all_checkpoints(self) -> list[dict]:
    return [json.loads(v) for v in await self.s.hvals(key("ckpts"))]

  async def _visible_checkpoints(self, prefix: str = "") -> list[dict]:
    items = await self._all_checkpoints()
    return [c for c in items if c["tinker_path"].startswith(prefix) and c["checkpoint_type"] != "parked"]

  async def checkpoints_of_run(self, run: str) -> list[dict]:
    return sorted(await self._visible_checkpoints(f"tinker://{run}/"), key=lambda c: c["time"])

  async def page_of_checkpoints(self, limit: int, offset: int) -> tuple[list[dict], int]:
    visible = sorted(await self._visible_checkpoints(), key=lambda c: -c["time"])
    return visible[offset : offset + limit], len(visible)

  async def set_checkpoint_ttl(self, uri: str, ttl: int | None) -> bool:
    parse_uri(uri)
    ck = key("ckpts")

    async def update(p):
      if not (raw := await p.hget(ck, uri)):
        return False
      p.multi()
      p.hset(ck, uri, json.dumps(json.loads(raw) | {"expires_at": time.time() + ttl if ttl else None}))
      return bool(await p.execute())

    return await self.transact(update, ck)

  async def unregister_checkpoints(self, *uris: str) -> int:
    for uri in uris:
      parse_uri(uri)
    return int(await self.s.hdel(key("ckpts"), *uris)) if uris else 0

  async def expired_checkpoints(self) -> list[str]:
    now = time.time()
    return [c["tinker_path"] for c in await self._all_checkpoints() if c["expires_at"] and c["expires_at"] <= now]

  async def sweep(self, store, grace: float) -> None:
    """Reclaim store versions that are neither in-flight writes nor published checkpoints."""
    unused, writing = await asyncio.to_thread(store.versions), await self.s.hvals(key("writing"))
    published = (v.partition("@")[0] for c in await self._all_checkpoints() if (v := c.get("version")))
    await asyncio.to_thread(store.sweep, unused.difference(writing, published), grace)

  # ***** events and wakeups *****

  async def append_event(self, _kind: Ev, **body) -> str:
    p = self.s.pipeline()
    log_event(p, _kind, body)
    wake(p)
    return (await p.execute())[0]

  async def wait_wake(self, wake_id: str, block_ms: int, stream: str = "wake") -> str:
    got = await self.s.xread({key(stream): wake_id}, count=1, block=block_ms or None)
    return got[0][1][-1][0] if got else wake_id

  async def read_events(self, after: str = "0-0", count: int = 10_000) -> list[Event]:
    rows = got[0][1] if (got := await self.s.xread({key("events"): after}, count=count)) else ()
    return [Event(eid, int(eid.split("-")[0]) / 1000, f["kind"], json.loads(f["body"])) for eid, f in rows]
