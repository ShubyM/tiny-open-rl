"""Runs, sessions, commands, observations and replies, all on Redis.

Every job owns its logical state, command stream tinyrl:q:<run>, and ownership generation tinyrl:gen:<run>. Claiming commands, publishing replies, and advancing restore boundaries all check the job's generation in WATCH/MULTI transactions; worker leases supervise processes separately, so moving one job off a shared engine fences that job without stopping siblings.
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
from tinyrl.storage.store import split_version

GROUP, LIVE_SECONDS, META_FIELDS = "trainer", 30, {b"record", b"worker", b"state"}
SESSION_GRACE, REPLY_TTL, SESSION_TTL = (
  getenv("SESSION_GRACE", 60),
  getenv("REPLY_TTL", 3600),
  getenv("SESSION_TTL", 7 * 86400),
)
LEASE_MS, NODE_MS = getenv("LEASE_MS", 10000), getenv("NODE_MS", 30000)


def key(*parts: str) -> str:
  return ":".join(("tinyrl", *parts))


def entry_key(entry_id: str | None) -> tuple[int, int]:
  return (int(entry_id.split("-", 1)[0]), int(entry_id.split("-", 1)[1])) if entry_id else (0, 0)


def entry_max(a: str, b: str) -> str:
  return b if entry_key(b) >= entry_key(a) else a


def merge_locations(existing: list[str], *added: str) -> list[str]:
  return list(dict.fromkeys(p for p in [*existing, *added] if p))


def version_doc(
  cur: dict | None, uri: str, ver: str, size: int, meta: dict | None, durable: bool, now: float, *locs: str
) -> dict:
  c = cur or {}
  base = {"version": ver, "uri": c.get("uri", uri), "size_bytes": int(size or c.get("size_bytes", 0)), "state": "live"}
  tail = {"durable": bool(durable or c.get("durable")), "locations": merge_locations(c.get("locations", []), *locs)}
  return base | {"meta": c.get("meta", {}) if meta is None else meta, "time": c.get("time", now)} | tail


def ckpt_doc(
  cid: str, uri: str, size: int, ttl: int | None, owned: bool, ver: str | None, durable: bool, now: float
) -> dict:
  kind = cid.split("/", 1)[0]
  ctype = {"weights": "training", "sampler_weights": "sampler"}.get(kind, kind)
  base = {"checkpoint_id": cid, "checkpoint_type": ctype, "time": now, "tinker_path": uri, "size_bytes": size}
  return base | {"expires_at": now + ttl if ttl else None, "owned": bool(owned), "version": ver, "durable": durable}


def write_future(p, request_id: str, reply: Reply) -> None:
  p.xadd(
    key("f", request_id),
    {"ok": int(reply.ok), "content_type": reply.content_type, "body": reply.body},
    maxlen=1,
    approximate=False,
  )
  p.expire(key("f", request_id), REPLY_TTL)


def write_ckpt_index(p, ckey: str, cid: str, uri: str, info: dict, ttl: int | None, now: float) -> None:
  p.hset(ckey, cid, json.dumps(info))
  if cid.split("/", 1)[0] not in ("parked", "recovery"):
    p.zadd(key("ckpts"), {uri: now})
  p.zadd(key("expiry"), {uri: now + ttl}) if ttl else p.zrem(key("expiry"), uri)


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
  RUN_CREATED = "run_created"
  ASSIGNED = "assigned"
  RUN_DELETED = "run_deleted"
  NODE_UP = "node_up"
  NODE_LOST = "node_lost"
  PLACED = "placed"
  UNPLACED = "unplaced"
  DRAINING = "draining"
  MIGRATED = "migrated"
  RESERVED = "reserved"


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
    cat = "user" if isinstance(exc, (ValueError, KeyError, FileNotFoundError)) else "server"
    return cls(
      False, "application/json", json.dumps({"error": f"{type(exc).__name__}: {exc}", "category": cat}).encode()
    )


@dataclass(frozen=True)
class Event:
  id: str
  t: float
  kind: str
  body: dict


def default_state(run: Run) -> dict:
  ver, _ = split_version(run.lora.get("weights_version"))
  prof = {"dtype": run.lora.get("dtype", "bfloat16"), "gpus": run.gpus, "lora": run.rank > 0, "rank": run.rank}
  prof["gpu_classes"] = list(run.lora.get("gpu_classes", ()))
  acct = run.train_run if run.role == Role.SAMPLER and run.train_run else run.id
  head = {"generation": 1 if run.worker else 0, "account": acct, "profile": prof, "weights_version": ver}
  tail = {"created_entry": "", "acked_progress": "", "acked_mutation": "", "working": None, "durable": None}
  return head | tail | {"rollback": None, "pending_saves": []}


def parse_cmd(run: str, entry_id: bytes, fields: dict[bytes, bytes]) -> Cmd:
  return Cmd(run, entry_id.decode(), Op(fields[b"op"].decode()), fields[b"body"], int(fields.get(b"cost", 1)))


def connect(url: str) -> redis.Redis:
  return redis.from_url(
    url, socket_timeout=None, retry=Retry(ExponentialBackoff(cap=2.0, base=0.1), 30), max_connections=10_000
  )


class Bus:
  def __init__(self, r: redis.Redis | None = None):
    self.r = r if r is not None else connect(getenv("REDIS_URL", "redis://localhost:6379"))

  async def tx(self, watched: list[str] | tuple[str, ...], fn, *, raise_on_error: bool = True):
    """Run one Redis WATCH/MULTI transaction, retrying on concurrent modifications."""
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          if watched:
            await p.watch(*watched)
          commit, out = await fn(p)
          if not commit:
            await p.unwatch()
            return out
          res = await p.execute(raise_on_error=raise_on_error)
          return out(res) if callable(out) else out
      except redis.WatchError:
        continue

  # ***** sessions *****

  async def heartbeat_session(self, session_id: str, metadata: dict | None = None) -> None:
    k, now, p = key("sess", session_id), time.time(), self.r.pipeline()
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
    head = {
      "session_id": session_id,
      "created": float(record[b"created"]),
      "seen": seen,
      "live": time.time() - seen < LIVE_SECONDS,
    }
    return head | {"runs": sorted(r.decode() for r in runs), "metadata": meta}

  async def expired_sessions(self) -> list[str]:
    return [s.decode() for s in await self.r.zrangebyscore(key("seen"), 0, time.time() - SESSION_GRACE)]

  async def finish_session(self, session_id: str) -> None:
    await self.r.zadd(key("seen"), {session_id: 0}, xx=True)

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

  # ***** runs & generations *****

  async def create_run(self, run: Run) -> None:
    state, record = default_state(run), json.dumps({**asdict(run), "created": time.time()})
    p = self.r.pipeline(transaction=True)
    p.xgroup_create(key("q", run.id), GROUP, id="0", mkstream=True)
    p.execute_command("XCFGSET", key("q", run.id), "IDMP-DURATION", 86400, "IDMP-MAXSIZE", 10000)
    p.sadd(key("runs"), run.id)
    p.set(key("gen", run.id), str(state["generation"]))
    mapping = {
      "record": record,
      "state": json.dumps(state),
      "steps": 0,
      "failed": 0,
      "cmds": 0,
      "busy_ms": 0,
      "wait_ms": 0,
    }
    p.hset(key("run", run.id), mapping=mapping | ({"worker": run.worker} if run.worker else {}))
    if wv := state["weights_version"]:
      p.hset(key("vref", wv), f"job:{run.id}:weights", "0.0")
    if run.session:
      p.sadd(key("sess", run.session, "runs"), run.id)
      p.expire(key("sess", run.session, "runs"), SESSION_TTL)
    body = {"run": run.id, "role": run.role, "base": run.base, "rank": run.rank, "exclusive": run.exclusive}
    body |= {"train_run": run.train_run, "gpus": run.gpus} | ({"worker": run.worker} if run.worker else {})
    p.xadd(key("events"), {"kind": Ev.RUN_CREATED, "body": json.dumps(body)})
    await p.execute()

  async def get_generation(self, run_id: str) -> int:
    return int(raw) if (raw := await self.r.get(key("gen", run_id))) else 0

  async def bump_generation(self, run_id: str, worker: str | None = None) -> int:
    """Advance a job's ownership generation atomically, fencing claims, replies and state commits from prior owners."""
    gkey, rkey = key("gen", run_id), key("run", run_id)

    async def step(p):
      cur, sraw = await p.get(gkey), await p.hget(rkey, "state")
      gen = int(cur or 0) + 1
      p.multi()
      p.set(gkey, str(gen))
      if sraw:
        p.hset(rkey, "state", json.dumps(json.loads(sraw) | {"generation": gen}))
      if worker is not None:
        p.hset(rkey, "worker", worker)
      return True, gen

    return await self.tx((gkey, rkey), step)

  async def request_job_migration(self, run_id: str, target_node: str | None) -> bool:
    rkey = key("run", run_id)

    async def step(p):
      if not (sraw := await p.hget(rkey, "state")):
        return False, False
      state = json.loads(sraw)
      state["target_node"] = target_node
      if not target_node:
        state.pop("target_node", None)
      p.multi()
      p.hset(rkey, "state", json.dumps(state))
      return True, True

    return await self.tx((rkey,), step)

  async def pin_sampler_weights(self, sampler_run: str, version: str) -> bool:
    if not (ver := split_version(version)[0]):
      return False
    rkey, vkey, fkey = key("run", sampler_run), key("ver", ver), key("vref", ver)

    async def step(p):
      sraw, vraw = await p.hget(rkey, "state"), await p.get(vkey)
      if not sraw or (vraw and json.loads(vraw).get("state") == "deleting"):
        return False, False
      state = json.loads(sraw)
      prev = state.get("weights_version")
      p.multi()
      p.hset(rkey, "state", json.dumps(state | {"weights_version": ver}))
      p.hset(fkey, f"job:{sampler_run}:weights", "0.0")
      if prev and prev != ver:
        p.hdel(key("vref", prev), f"job:{sampler_run}:weights")
      return True, True

    return await self.tx((rkey, vkey, fkey), step)

  async def commit_job_state(
    self,
    run_id: str,
    generation: int,
    uri: str,
    version: str,
    boundary: str,
    size: int = 0,
    meta: dict | None = None,
    *,
    durable: bool = False,
    location: str = "",
    optimizer: bool = True,
    profile: dict | None = None,
  ) -> bool:
    """Atomically advance a job's restore version and boundary under `generation`; retain prior durable version until replaced."""
    if not optimizer or (meta or {}).get("kind") == "sampler":
      raise ValueError("sampler or weights-only export cannot be committed as a training recovery state")
    if not (ver := split_version(version)[0]):
      raise ValueError("missing checkpoint version")
    loc = location or split_version(version)[1]
    gkey, rkey, vkey, fkey = key("gen", run_id), key("run", run_id), key("ver", ver), key("vref", ver)

    async def step(p):
      graw, sraw, vraw = await p.get(gkey), await p.hget(rkey, "state"), await p.get(vkey)
      vcur = json.loads(vraw) if vraw else None
      if int(graw or 0) != generation or not sraw or (vcur and vcur.get("state") == "deleting"):
        return False, False
      state, now = json.loads(sraw), time.time()
      old_work, old_dur = (state.get("working") or {}).get("version"), (state.get("durable") or {}).get("version")
      is_dur = bool(durable or (vcur or {}).get("durable"))
      state["working"] = {"uri": uri, "version": ver, "boundary": boundary, "optimizer": True, "location": loc}
      if is_dur:
        state["durable"] = {"uri": uri, "version": ver, "boundary": boundary, "optimizer": True}
      if profile:
        state["profile"] = state.get("profile", {}) | profile
      p.multi()
      p.setnx(key("ckpt_since"), now)
      p.set(vkey, json.dumps(version_doc(vcur, uri, ver, size, meta, is_dur, now, loc)))
      p.sadd(key("vers"), ver)
      p.hset(fkey, f"job:{run_id}:work", "0.0")
      if old_work and old_work != ver:
        p.hdel(key("vref", old_work), f"job:{run_id}:work")
      if is_dur:
        p.hset(fkey, f"job:{run_id}:durable", "0.0")
        if old_dur and old_dur != ver:
          p.hdel(key("vref", old_dur), f"job:{run_id}:durable")
      p.hset(rkey, "state", json.dumps(state))
      return True, True

    return await self.tx((gkey, rkey, vkey, fkey), step)

  async def complete_durable_version(self, run_id: str, version: str, location: str = "") -> list[str]:
    """Promote an uploaded version to durable, completing waiting SAVE_WEIGHTS futures and swapping the durable ref."""
    if not (ver := split_version(version)[0]):
      return []
    rkey, ckey, vkey, fkey = key("run", run_id), key("ckpt", run_id), key("ver", ver), key("vref", ver)

    async def step(p):
      sraw, vraw = await p.hget(rkey, "state"), await p.get(vkey)
      vcur = json.loads(vraw) if vraw else None
      if vcur and vcur.get("state") == "deleting":
        return False, []
      state, now = (json.loads(sraw) if sraw else None), time.time()
      matching = [s for s in (state or {}).get("pending_saves", []) if s["version"] == ver]
      prev_ckpts = {}
      for s in matching:
        if craw := await p.hget(ckey, s["uri"].removeprefix("tinker://").split("/", 1)[1]):
          prev_ckpts[s["uri"]] = split_version(json.loads(craw).get("version"))[0]
      if not vcur and not matching:
        return False, []
      p.multi()
      if vcur:
        p.set(
          vkey, json.dumps(vcur | {"durable": True, "locations": merge_locations(vcur.get("locations", []), location)})
        )
      done: list[str] = []
      if state is not None:
        old_dur = state.get("durable") or {}
        old_ver, old_b, cand = old_dur.get("version"), old_dur.get("boundary", ""), None
        if (wk := state.get("working")) and wk.get("version") == ver and wk.get("optimizer"):
          cand = {"uri": wk["uri"], "version": ver, "boundary": wk["boundary"], "optimizer": True}
        elif matching and not state.get("rollback") and entry_key(matching[-1]["boundary"]) >= entry_key(old_b):
          cand = {"uri": matching[-1]["uri"], "version": ver, "boundary": matching[-1]["boundary"], "optimizer": True}
        if cand and entry_key(cand["boundary"]) >= entry_key(old_b):
          state["durable"] = cand
          p.hset(fkey, f"job:{run_id}:durable", "0.0")
          if old_ver and old_ver != ver:
            p.hdel(key("vref", old_ver), f"job:{run_id}:durable")
        for s in matching:
          uri, cid, ttl = s["uri"], s["uri"].removeprefix("tinker://").split("/", 1)[1], s.get("ttl")
          write_ckpt_index(
            p, ckey, cid, uri, ckpt_doc(cid, uri, s["size"], ttl, bool(s.get("owned")), ver, True, now), ttl, now
          )
          p.hset(fkey, f"ckpt:{uri}", "0.0")
          if (pv := prev_ckpts.get(uri)) and pv != ver:
            p.hdel(key("vref", pv), f"ckpt:{uri}")
          p.hdel(fkey, f"save:{s['request_id']}")
          write_future(
            p,
            s["request_id"],
            Reply(
              True, "application/json", json.dumps({"type": Op.SAVE_WEIGHTS, "model_id": run_id, "path": uri}).encode()
            ),
          )
          done.append(s["request_id"])
        if matching:
          state["pending_saves"] = [s for s in state["pending_saves"] if s["version"] != ver]
        p.hset(rkey, "state", json.dumps(state))
      return True, done

    return await self.tx((rkey, ckey, vkey, fkey), step)

  async def fail_pending_save(self, run_id: str, request_id: str, reason: str) -> bool:
    rkey = key("run", run_id)

    async def step(p):
      state = json.loads(sraw) if (sraw := await p.hget(rkey, "state")) else {}
      if not (hit := next((s for s in state.get("pending_saves", []) if s["request_id"] == request_id), None)):
        return False, False
      state["pending_saves"] = [s for s in state["pending_saves"] if s["request_id"] != request_id]
      p.multi()
      p.hset(rkey, "state", json.dumps(state))
      p.hincrby(rkey, "failed", 1)
      p.hdel(key("vref", hit["version"]), f"save:{request_id}")
      write_future(p, request_id, Reply.error(RuntimeError(reason)))
      return True, True

    return await self.tx((rkey,), step)

  async def reconcile_recovery(
    self, run_id: str, generation: int, working_ok: bool, durable_ok: bool, durable_versions: set[str] | None = None
  ) -> tuple[str, dict | None]:
    """Choose the authoritative recovery action for `generation`: restore, recreate, uncreated, or rollback."""
    gkey, rkey, dvers = key("gen", run_id), key("run", run_id), durable_versions or set()

    async def step(p):
      graw, sraw = await p.get(gkey), await p.hget(rkey, "state")
      if int(graw or 0) != generation or not sraw:
        await p.unwatch()
        raise RuntimeError(f"lost generation {generation} of {run_id}")
      state = json.loads(sraw)
      if rb := state.get("rollback"):
        return False, ("rollback", rb)
      wk, dur, acked = (
        state.get("working"),
        (state.get("durable") if durable_ok else None),
        state.get("acked_mutation", ""),
      )
      if wk and working_ok and wk.get("optimizer") and entry_key(wk["boundary"]) >= entry_key(acked):
        return False, ("restore", wk)
      old_work, dur_ver, cut = (wk or {}).get("version"), (dur or {}).get("version"), (dur or {}).get("boundary", "")
      if dur and dur.get("optimizer") and entry_key(cut) >= entry_key(acked):
        action, payload, state["working"] = "restore", dur, dur
      elif not acked and not dur:
        action, payload, state["working"] = ("recreate" if state.get("created_entry") else "uncreated"), None, None
      else:
        rb = {"acked": acked, "durable": cut, "durable_uri": (dur or {}).get("uri")}
        action, payload, state["rollback"], state["working"] = "rollback", rb, rb, dur
      kept, failed = [], []
      for s in state.get("pending_saves", []):
        landed = s["version"] in dvers or (s["version"] == dur_ver and durable_ok)
        lost = not landed and (action == "rollback" or entry_key(s["boundary"]) > entry_key(cut) or not working_ok)
        (failed if lost else kept).append(s)
      state["pending_saves"] = kept
      p.multi()
      if old_work and old_work != dur_ver:
        p.hdel(key("vref", old_work), f"job:{run_id}:work")
      if dur_ver:
        p.hset(key("vref", dur_ver), f"job:{run_id}:work", "0.0")
      for s in failed:
        p.hdel(key("vref", s["version"]), f"save:{s['request_id']}")
        p.hincrby(rkey, "failed", 1)
        write_future(
          p, s["request_id"], Reply.error(RuntimeError(f"save_weights {s['uri']} was lost before durable commit"))
        )
      p.hset(rkey, "state", json.dumps(state))
      return True, (action, payload)

    return await self.tx((gkey, rkey), step)

  # ***** the cluster's assignment, scheduler lease & observations *****

  async def claim_scheduler(self, holder: str, ttl_ms: int = LEASE_MS) -> int | None:
    """Acquire or renew the single-writer scheduler lease, returning its monotonic epoch or None if another owns it."""
    skey, ekey = key("sched_owner"), key("sched_epoch")

    async def step(p):
      cur, epoch = await p.mget(skey, ekey)
      if cur not in (None, holder.encode()):
        return False, None
      nxt = int(epoch or 0) + (0 if cur == holder.encode() else 1)
      p.multi()
      p.set(skey, holder, px=ttl_ms)
      p.set(ekey, str(nxt))
      return True, nxt

    return await self.tx((skey, ekey), step)

  async def release_scheduler(self, holder: str) -> None:
    await self.r.delex(key("sched_owner"), ifeq=holder)

  async def write_observation(self, worker: str, obs: dict) -> None:
    await self.r.hset(key("obs"), worker, json.dumps(obs | {"at": time.time()}))

  async def read_table(self) -> dict:
    """Read live nodes, jobs (records, states, queue lag), workers, observations, persisted service and versions."""
    nodes = await self.live_nodes(details=True)
    runs = sorted(r.decode() for r in await self.r.smembers(key("runs")))
    vers = sorted(v.decode() for v in await self.r.smembers(key("vers")))
    p = self.r.pipeline(transaction=False)
    for run in runs:
      p.hmget(key("run", run), "record", "worker", "state")
      p.xinfo_groups(key("q", run))
    for k in ("workers", "obs", "service", "settled"):
      p.hgetall(key(k))
    p.mget(key("workers_made"), key("reserved"), *(key("ver", v) for v in vers))
    raw = await p.execute(raise_on_error=False)
    run_rows, (workers, obs, service, settled, mvals) = raw[: 2 * len(runs)], raw[2 * len(runs) :]
    workers_made, reserved, *vraws = mvals if isinstance(mvals, list) else (0, None)
    out_runs, queues = {}, {}
    for idx, run in enumerate(runs):
      meta, groups = run_rows[2 * idx], run_rows[2 * idx + 1]
      if isinstance(meta, list) and meta[0]:
        out_runs[run] = (
          json.loads(meta[0]),
          meta[1].decode() if meta[1] else None,
          json.loads(meta[2]) if meta[2] else {},
        )
      if isinstance(groups, list) and (g := next((x for x in groups if x.get("name") == GROUP.encode()), None)):
        queues[run] = int(g.get("lag") or 0) + int(g.get("pending") or 0)
    jmap = lambda h: {k.decode(): json.loads(v) for k, v in (h or {}).items()}  # noqa: E731
    versions = {
      v: d for v, vr in zip(vers, vraws) if isinstance(vr, bytes) and (d := json.loads(vr)).get("state") == "live"
    }
    out = {
      "nodes": nodes,
      "runs": out_runs,
      "queues": queues,
      "workers": jmap(workers),
      "workers_made": int(workers_made or 0),
    }
    out |= {"observations": jmap(obs), "service": {k.decode(): float(v) for k, v in (service or {}).items()}}
    return out | {
      "settled": jmap(settled),
      "reserved": json.loads(reserved) if isinstance(reserved, bytes) else {},
      "versions": versions,
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
    settled: dict[str, dict] | None = None,
    settled_gone: list[str] | None = None,
    reserved: dict | None = None,
    obs_gone: list[str] | None = None,
    ref_adds: list[tuple[str, str]] | None = None,
    ref_dels: list[tuple[str, str]] | None = None,
    scheduler_owner: str | None = None,
  ) -> bool:
    """Publish a new fleet assignment, generation advances, persisted service and version refs in one transaction."""
    gens, adds = generations or {}, ref_adds or []
    watched = [
      *([key("sched_owner")] if scheduler_owner else []),
      *(key("run", r) for r in gens),
      *(key("ver", v) for v, _ in adds),
    ]

    async def step(p):
      if scheduler_owner and await p.get(key("sched_owner")) != scheduler_owner.encode():
        return False, False
      states = {r: await p.hget(key("run", r), "state") for r in gens}
      uniq_v = list(dict.fromkeys(v for v, _ in adds))
      live_v = {v: bool((vr := await p.get(key("ver", v))) and json.loads(vr).get("state") == "live") for v in uniq_v}
      p.multi()
      for run, worker in assigned.items():
        p.hset(key("run", run), "worker", worker)
      for run, gen in gens.items():
        p.set(key("gen", run), str(gen))
        if sraw := states.get(run):
          p.hset(key("run", run), "state", json.dumps(json.loads(sraw) | {"generation": gen}))
      if workers:
        p.hset(key("workers"), mapping={n: json.dumps(w) for n, w in workers.items()})
      for h, drop in (("workers", gone), ("obs", obs_gone), ("settled", settled_gone)):
        if drop:
          p.hdel(key(h), *drop)
      p.set(key("workers_made"), workers_made)
      if service:
        p.hset(key("service"), mapping={k: str(v) for k, v in service.items()})
      if settled:
        p.hset(key("settled"), mapping={k: json.dumps(v) for k, v in settled.items()})
      if reserved is not None:
        p.set(key("reserved"), json.dumps(reserved)) if reserved else p.delete(key("reserved"))
      for v, holder in adds:
        if live_v.get(v):
          p.hset(key("vref", v), holder, "0.0")
      for v, holder in ref_dels or ():
        p.hdel(key("vref", v), holder)
      for kind, body in changes:
        p.xadd(key("events"), {"kind": kind, "body": json.dumps(body)}, maxlen=1_000_000, approximate=True)
      return True, True

    return await self.tx(watched, step)

  async def get_run(self, run_id: str) -> dict | None:
    if not (fields := await self.r.hgetall(key("run", run_id))):
      return None
    record = json.loads(fields.pop(b"record"))
    if worker := fields.pop(b"worker", None):
      record["worker"] = worker.decode()
    if sraw := fields.pop(b"state", None):
      record["state"] = json.loads(sraw)
    return record | {name.decode(): int(value) for name, value in fields.items()}

  async def join_sampling_session(self, run_id: str) -> int:
    return int(await self.r.incr(key("clients", run_id)))

  async def run_counters(self) -> dict[tuple[str, str], dict]:
    runs, p = sorted(r.decode() for r in await self.r.smembers(key("runs"))), self.r.pipeline()
    for run_id in runs:
      p.hgetall(key("run", run_id))
    return {
      (f[b"worker"].decode(), r): {k.decode(): int(v) for k, v in f.items() if k not in META_FIELDS}
      for r, f in zip(runs, await p.execute())
      if b"worker" in f
    }

  async def delete_run(self, run_id: str, worker: str) -> None:
    rkey = key("run", run_id)

    async def step(p):
      state = json.loads(sraw) if (sraw := await p.hget(rkey, "state")) else {}
      p.multi()
      self._drop_job_refs(p, run_id, state)
      p.srem(key("runs"), run_id)
      p.delete(key("q", run_id), key("gen", run_id))
      return True, True

    await self.tx((rkey,), step)
    await self.append_event(Ev.RUN_DELETED, run=run_id, worker=worker)

  def _drop_job_refs(self, p, run_id: str, state: dict) -> None:
    for v, tag in (
      (state.get("weights_version"), "weights"),
      ((state.get("working") or {}).get("version"), "work"),
      ((state.get("durable") or {}).get("version"), "durable"),
    ):
      if v:
        p.hdel(key("vref", v), f"job:{run_id}:{tag}")
    for s in state.get("pending_saves", []):
      p.hdel(key("vref", s["version"]), f"save:{s['request_id']}")
      write_future(
        p,
        s["request_id"],
        Reply.error(RuntimeError(f"model {run_id} was deleted before save_weights {s['uri']} became durable")),
      )

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
    if seq:
      await self.r.set(key("seq", run_id, str(seq_id)), request_id, ex=SESSION_TTL)
    return request_id

  async def peek_commands(self, runs: list[str], block_ms: int = 0) -> list[Cmd]:
    if not runs:
      await asyncio.sleep(block_ms / 1000) if block_ms else None
      return []
    p = self.r.pipeline(transaction=False)
    for run in runs:
      p.xinfo_groups(key("q", run))
    cursors = {}
    for run, groups in zip(runs, await p.execute(raise_on_error=False)):
      if isinstance(groups, Exception):
        if "no such key" not in str(groups):
          raise groups
      elif group := next((g for g in groups if g["name"] == GROUP.encode()), None):
        cursors[key("q", run)] = group["last-delivered-id"]
    if not cursors:
      await asyncio.sleep(block_ms / 1000)
      return []
    got = await self.r.xread(cursors, count=1, block=block_ms or None)
    return sorted(
      (parse_cmd(s.decode().removeprefix(key("q", "")), *e[0]) for s, e in got),
      key=lambda c: (*entry_key(c.entry_id), c.run),
    )

  async def _take_fenced(self, run: str, gen: int, owner: str, count: int) -> list[Cmd]:
    gkey, qkey = key("gen", run), key("q", run)

    def unpack(res) -> list[Cmd]:
      if isinstance(got := res[0], Exception):
        if "NOGROUP" in str(got):
          return []
        raise got
      return [parse_cmd(run, eid, fields) for _, entries in got for eid, fields in entries]

    async def step(p):
      if int((await p.get(gkey)) or 0) != gen:
        return False, []
      p.multi()
      p.xreadgroup(GROUP, owner, {qkey: ">"}, count=count)
      return True, unpack

    return await self.tx((gkey,), step, raise_on_error=False)

  async def take_commands(
    self, runs: list[str] | dict[str, int], owner: str, count: int = 16, block_ms: int = 0
  ) -> list[Cmd]:
    """Claim selected work, fenced by each job's ownership generation when a {run: generation} map is given."""
    cmds: list[Cmd] = []
    if isinstance(runs, dict):
      for run, gen in runs.items():
        cmds += await self._take_fenced(run, gen, owner, count)
    else:
      p = self.r.pipeline(transaction=False)
      for run in runs:
        p.xreadgroup(GROUP, owner, {key("q", run): ">"}, count=count)
      for run, got in zip(runs, await p.execute(raise_on_error=False)):
        if isinstance(got, Exception):
          if "NOGROUP" not in str(got):
            raise got
        else:
          cmds += [parse_cmd(run, eid, fields) for _, entries in got for eid, fields in entries]
    if not cmds and block_ms and runs:
      await self.peek_commands(list(runs), block_ms)
      return await self.take_commands(runs, owner, count)
    return cmds

  async def write_reply(
    self,
    cmd: Cmd,
    reply: Reply,
    worker: str,
    owner: str,
    started: float | None = None,
    *,
    generation: int | None = None,
    mutates: bool = False,
    clear_rollback: bool = False,
    profile: dict | None = None,
    lora_config: dict | None = None,
    defer_save: dict | None = None,
  ) -> bool:
    """Publish a reply (or record a deferred durable save) iff `owner` holds `worker` AND `generation` still owns `cmd.run`."""
    counts = (
      {"failed": 1} if not reply.ok and not defer_save else {"steps": 1} if cmd.op == Op.OPTIM_STEP and reply.ok else {}
    )
    if started is not None:
      wait = max(0, int(started * 1000) - int(cmd.entry_id.split("-")[0]))
      counts |= {"cmds": 1, "wait_ms": wait, "busy_ms": int((time.time() - started) * 1000)}
    okey, gkey, rkey = key("owner", worker), key("gen", cmd.run), key("run", cmd.run)
    deleting = cmd.op == Op.DELETE_MODEL and reply.ok
    sver = split_version(defer_save["version"])[0] if defer_save else None
    watched = [
      okey,
      gkey,
      rkey,
      *([key("ckpt", cmd.run)] if deleting else []),
      *([key("ver", sver), key("vref", sver)] if sver else []),
    ]

    async def step(p):
      if await p.get(okey) != owner.encode() or (
        generation is not None and int((await p.get(gkey)) or 0) != generation
      ):
        return False, False
      sraw = await p.hget(rkey, "state")
      rraw = await p.hget(rkey, "record") if lora_config else None
      ckpts = await p.hgetall(key("ckpt", cmd.run)) if deleting else {}
      if sver and (vraw := await p.get(key("ver", sver))) and json.loads(vraw).get("state") == "deleting":
        await p.unwatch()
        raise ValueError(f"version {sver} was claimed for deletion before save committed")
      state = json.loads(sraw) if sraw else None
      if state is not None:
        if reply.ok or defer_save:
          state["acked_progress"] = entry_max(state.get("acked_progress", ""), cmd.entry_id)
        if cmd.op == Op.CREATE_MODEL and reply.ok:
          state["created_entry"] = cmd.entry_id
        if mutates and reply.ok:
          state["acked_mutation"] = entry_max(state.get("acked_mutation", ""), cmd.entry_id)
        if clear_rollback and reply.ok:
          state["rollback"] = None
        if profile and reply.ok:
          state["profile"] = state.get("profile", {}) | profile
        if defer_save and sver:
          item = {
            "request_id": cmd.request_id,
            "entry_id": cmd.entry_id,
            "uri": defer_save["uri"],
            "version": sver,
            "size": defer_save["size"],
            "ttl": defer_save.get("ttl"),
            "owned": bool(defer_save.get("owned")),
            "boundary": defer_save.get("boundary") or cmd.entry_id,
          }
          state["pending_saves"] = [*state.get("pending_saves", []), item]
      p.multi()
      p.hset(key("vref", sver), f"save:{cmd.request_id}", "0.0") if (defer_save and sver) else write_future(
        p, cmd.request_id, reply
      )
      p.xack(key("q", cmd.run), GROUP, cmd.entry_id)
      p.xdel(key("q", cmd.run), cmd.entry_id)
      for name, n in counts.items():
        p.hincrby(rkey, name, n)
      if rraw and lora_config and reply.ok:
        rec = json.loads(rraw)
        p.hset(rkey, "record", json.dumps(rec | {"lora": rec.get("lora", {}) | lora_config}))
      if state is not None and not deleting:
        p.hset(rkey, "state", json.dumps(state))
      if deleting:
        for cid, data in ckpts.items():
          if (c := json.loads(data))["owned"]:
            uri = f"tinker://{cmd.run}/{cid.decode()}"
            p.hdel(key("ckpt", cmd.run), cid)
            p.zrem(key("ckpts"), uri)
            p.zrem(key("expiry"), uri)
            if cv := split_version(c.get("version"))[0]:
              p.hdel(key("vref", cv), f"ckpt:{uri}")
        if state is not None:
          state["pending_saves"] = [s for s in state.get("pending_saves", []) if s["request_id"] != cmd.request_id]
          self._drop_job_refs(p, cmd.run, state)
          p.hset(
            rkey,
            "state",
            json.dumps(state | {"working": None, "durable": None, "weights_version": None, "pending_saves": []}),
          )
        p.srem(key("runs"), cmd.run)
        p.delete(key("q", cmd.run), key("gen", cmd.run))
        p.xadd(
          key("events"),
          {"kind": Ev.RUN_DELETED, "body": json.dumps({"run": cmd.run, "worker": worker})},
          maxlen=1_000_000,
          approximate=True,
        )
      return True, True

    return await self.tx(watched, step)

  async def reclaim_commands(
    self,
    run_id: str,
    worker: str,
    owner: str,
    generation: int | None = None,
    *,
    replay_sample: bool = False,
    min_idle_ms: int = 0,
  ) -> list[Cmd]:
    if generation is not None and await self.get_generation(run_id) != generation:
      return []
    try:
      _, entries, _ = await self.r.xautoclaim(key("q", run_id), GROUP, owner, min_idle_time=min_idle_ms, count=10000)
    except redis.ResponseError as e:
      if "NOGROUP" in str(e):
        return []
      raise
    replay = []
    for entry_id, fields in entries:
      cmd = parse_cmd(run_id, entry_id, fields)
      if replay_sample and cmd.op == Op.SAMPLE:
        replay.append(cmd)
      else:
        await self.write_reply(
          cmd,
          Reply.error(RuntimeError(f"its worker died, or lost the job, during {cmd.op}")),
          worker,
          owner,
          generation=generation,
        )
    return replay

  async def fail_stranded_commands(
    self, runs: list[str] | dict[str, int], worker: str, owner: str, min_idle_ms: int = 0
  ) -> None:
    for run_id, gen in runs.items() if isinstance(runs, dict) else ((r, None) for r in runs):
      await self.reclaim_commands(run_id, worker, owner, gen, min_idle_ms=min_idle_ms)

  # ***** worker leases, nodes, replies & events *****

  async def take_lease(self, worker: str, owner: str, recover: bool = False) -> bool:
    k, last = key("owner", worker), key("last_owner", worker)

    async def step(p):
      current, previous = await p.mget(k, last)
      if current is not None or (recover and previous not in (None, owner.encode())):
        return False, False
      p.multi()
      p.set(k, owner, px=LEASE_MS)
      p.set(last, owner)
      return True, True

    return await self.tx((k, last), step)

  async def renew_lease(self, worker: str, owner: str) -> bool:
    return bool(await self.r.set(key("owner", worker), owner, ifeq=owner, px=LEASE_MS))

  async def release_lease(self, worker: str, owner: str) -> None:
    await self.r.delex(key("owner", worker), ifeq=owner)

  async def lease_owner(self, worker: str) -> str | None:
    return owner.decode() if (owner := await self.r.get(key("owner", worker))) else None

  async def node_heartbeat(
    self, node: str, gpus: int, memory: int = 0, workers: dict | None = None, lag: float = 0.0, **extra
  ) -> None:
    beat = {
      "gpus": gpus,
      "memory": memory,
      "until": time.time() + NODE_MS / 1000,
      "workers": workers or {},
      "lag": lag,
    } | extra
    await self.r.hset(key("nodes"), node, json.dumps(beat))

  async def live_nodes(self, details: bool = False) -> dict:
    nodes = {k.decode(): json.loads(v) for k, v in (await self.r.hgetall(key("nodes"))).items()}
    if dead := [k for k, v in nodes.items() if v["until"] < time.time()]:
      await self.r.hdel(key("nodes"), *dead)
    return {k: v if details else v["gpus"] for k, v in nodes.items() if k not in dead}

  async def wait_for_reply(self, request_id: str, timeout: float) -> Reply | None:
    if not (got := await self.r.xread({key("f", request_id): "0-0"}, count=1, block=int(timeout * 1000) or None)):
      return None
    fields = got[0][1][0][1]
    return Reply(fields[b"ok"] == b"1", fields[b"content_type"].decode(), fields[b"body"])

  async def queued_and_in_flight(self, run_id: str) -> dict:
    stream, p = key("q", run_id), self.r.pipeline()
    p.xinfo_groups(stream)
    p.xpending_range(stream, GROUP, min="-", max="+", count=100)
    groups, pending = await p.execute(raise_on_error=False)
    if isinstance(groups, Exception):
      return {"live": False}
    fly = [
      {
        "request_id": f"{run_id}/{e['message_id'].decode()}",
        "trainer": e["consumer"].decode(),
        "seconds": e["time_since_delivered"] / 1000,
        "deliveries": e["times_delivered"],
      }
      for e in pending
    ]
    return {"live": True, "queued": groups[0]["lag"], "in_flight": fly}

  async def append_event(self, _kind: Ev, **body) -> str:
    return (
      await self.r.xadd(
        key("events"), {"kind": Ev(_kind), "body": json.dumps(body)}, maxlen=1_000_000, approximate=True
      )
    ).decode()

  async def read_events(self, after: str = "0-0", count: int = 10_000, block_ms: int = 0) -> list[Event]:
    got = await self.r.xread({key("events"): after}, count=count, block=block_ms or None)
    return [
      Event(eid := rid.decode(), int(eid.split("-")[0]) / 1000, f[b"kind"].decode(), json.loads(f[b"body"]))
      for rid, f in (got[0][1] if got else ())
    ]
