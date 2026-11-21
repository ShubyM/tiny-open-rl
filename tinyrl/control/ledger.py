"""Controller ledger: durable runs/checkpoints/train queues, RAM sampling, and 4 node RPCs (`heartbeat`, `take`, `finish`, `detach`)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
import weakref
from collections import deque
from dataclasses import asdict, dataclass, field, replace

from tinyrl import types as T
from tinyrl.control import scheduler
from tinyrl.control.accounting import Charge
from tinyrl.control.db import DB, Table
from tinyrl.control.fairness import admit_samples, rank_runs
from tinyrl.control.samples import SampleQueue
from tinyrl.helpers import getenv
from tinyrl.types import Assignment, Cmd, Ev, Event, FencedError, Op, Reply, Role, Run, RunStatus, Work

LIVE_SECONDS, SESSION_GRACE, GC_S = 30, getenv("SESSION_GRACE", 60), 10
REPLY_TTL, READ_REPLY_TTL, SESSION_TTL = getenv("REPLY_TTL", 3600), 60, getenv("SESSION_TTL", 7 * 24 * 3600)
LEASE_MS, SWEEP_S = getenv("LEASE_MS", 10000), getenv("SWEEP_S", 60.0)
COUNTERS = ("steps", "failed", "cmds", "tokens", "busy_ms", "wait_ms")


@dataclass
class RunRow:
  run: Run
  created: float
  status: RunStatus = RunStatus.FRESH
  owner: tuple[str, str, str] | None = None  # (worker, node, boot_id)
  snap: tuple[str, str] | None = None  # (uri, version) of detached trainer state
  counters: dict[str, int] = field(default_factory=lambda: dict.fromkeys(COUNTERS, 0))

  @property
  def live(self) -> bool:
    return self.status != RunStatus.DELETED


def make_checkpoint(uri: str, size: int, *, version: str, ttl=None, owned=False, meta: dict | None = None) -> dict:
  raw, now = (cid := T.parse_uri(uri)[1]).split("/", 1)[0], time.time()
  kind = {"weights": "training", "sampler_weights": "sampler"}.get(raw, raw)
  base = {"checkpoint_id": cid, "checkpoint_type": kind, "time": now, "tinker_path": uri, "size_bytes": size}
  return base | {"expires_at": now + ttl if ttl else None, "owned": bool(owned), "version": version, "meta": meta or {}}


class Ledger:
  def __init__(self, path: str | None = None):
    self.db = DB(getenv("TINYRL_DB", "/tmp/tinyrl.db") if path is None else path)
    self.runs = Table(self.db, "runs", RunRow)
    tabs = [Table(self.db, n) for n in ("sessions", "checkpoints", "cmds", "replies", "seen", "meta")]
    self.sessions, self.checkpoints, self.cmds, self.replies, self.seen, self.meta = tabs
    self.service = {k: float(v) for k, v in self.meta.get("service", {}).items()}
    self.plan_state = scheduler.from_plan(self.meta.get("plan", {}))
    self.samples, self.train_q, self.since = SampleQueue(), {}, {}
    for rid, r in self.runs.items():
      if r.live:
        self.samples.open(rid) if r.run.role == Role.SAMPLER else self.train_q.setdefault(rid, {})
    now, self.seq = time.monotonic(), max((c["seq"] for c in self.cmds.values()), default=0)
    for req_id, c in sorted(self.cmds.items(), key=lambda kv: kv[1]["seq"]):
      if o := c.get("owner"):
        c["owner"] = tuple(o)
      if c["run"] in self.train_q:
        self.train_q[c["run"]][req_id] = None
        if c.get("owner"):
          self.since[req_id] = now
    self.nodes, self.reaped_boots, self.events = {}, set(), deque(maxlen=100_000)
    self.waiters, self.work_event = weakref.WeakValueDictionary(), asyncio.Event()
    self.plan_due = self.gc_due = self.sweep_due = 0.0
    held = self.since or any(r.live and r.owner for r in self.runs.values())
    self.reap_after = now + LEASE_MS / 1000 if held else 0.0
    self.plan_state = self.state()

  async def aclose(self) -> None:
    if self.db is not None:
      with self.db:
        self.meta["service"] = self.service
      self.db.conn.close()
      self.db = None

  def _notify(self, req_id: str) -> None:
    if ev := self.waiters.pop(req_id, None):
      ev.set()

  def _wake(self) -> None:
    ev, self.work_event = self.work_event, asyncio.Event()
    ev.set()

  def record_event(self, kind: Ev, body: dict) -> Event:
    self.seq += 1
    self.events.append(ev := Event(f"{int((now := time.time()) * 1000)}-{self.seq}", now, str(Ev(kind)), body))
    self.plan_due = 0.0 if kind in (Ev.RUN_CREATED, Ev.RUN_DELETED, Ev.NODE_UP, Ev.NODE_LOST) else self.plan_due
    return ev

  def _is_sampler(self, rid: str, req_id: str = "") -> bool:
    return self.runs[rid].run.role == Role.SAMPLER if rid in self.runs else req_id.startswith("samp-")

  def _cmd(self, req_id: str, *, full: bool = False) -> Cmd:
    c, body = self.cmds[req_id], (self.db.get_blob(f"c:{req_id}") or b"") if full else b""
    return Cmd(c["run"], c["eid"], Op(c["op"]), body, c["cost"])

  def _bill(self, row: RunRow, cmd: Cmd, reply: Reply, charge: Charge | None) -> None:
    c = row.counters
    c["failed"], c["steps"] = c["failed"] + (not reply.ok), c["steps"] + (reply.ok and cmd.op == Op.OPTIM_STEP)
    if charge:
      c["cmds"], c["busy_ms"] = c["cmds"] + 1, c["busy_ms"] + charge.busy_ms
      c["wait_ms"] += max(0, charge.started_ms - T.entry_sort_key(cmd.entry_id)[0])
      c["tokens"] += cmd.cost if reply.ok else 0
    if charge and charge.gpu_ms and (acct := self.plan_state.account(cmd.run)):
      self.service[acct] = self.service.get(acct, 0.0) + charge.gpu_ms

  def _lose_owners(self, pred) -> None:
    for rid, row in self.runs.items():
      if row.owner and pred(row.owner):
        row.owner, row.status = None, (RunStatus.LOST if row.status == RunStatus.LIVE else row.status)
        self.runs.save(rid)

  def _reclaim(self, pred) -> None:
    self.samples.reclaim(pred)
    if any(r.owner and pred(r.owner) for r in self.runs.values()):
      with self.db:
        self._lose_owners(pred)

  def _drop_cmd(self, req_id: str, *, wake: bool = False) -> None:
    self.since.pop(req_id, None)
    self.cmds.pop(req_id, None)
    self.db.del_blob(f"c:{req_id}")
    if wake:
      self._notify(req_id)

  def _purge_run(self, rid: str, *, owned_ckpts: bool = False) -> None:
    if r := self.runs.get(rid):
      r.status, r.owner, r.snap = RunStatus.DELETED, None, None
      self.runs.save(rid)
    for k in self.train_q.pop(rid, ()):
      self._drop_cmd(k, wake=True)
    for u in [u for u, c in self.checkpoints.items() if owned_ckpts and T.parse_uri(u)[0] == rid and c["owned"]]:
      del self.checkpoints[u]

  def _fence(self, node: str, boot_id: str) -> dict | None:
    self.reap_dead_nodes()
    cur = self.nodes.get(node)
    if boot_id in self.reaped_boots or (cur is not None and cur["boot_id"] != boot_id):
      raise FencedError(f"fenced boot {node}/{boot_id}")
    if cur is not None:
      cur["until"] = time.monotonic() + LEASE_MS / 1000
    return cur

  def _head_train(self, worker: str, tok: tuple[str, str, str]) -> list[Cmd]:
    out = []
    for r in self.plan_state.runs(worker):
      row, k = self.runs.get(r), next(iter(self.train_q.get(r, ())), None)
      if row and row.live and row.owner in (None, tok) and k and not self.cmds[k].get("owner"):
        out.append(self._cmd(k))
    return out

  def _ready_samples(self, spec) -> list[tuple[Cmd, str | None]]:
    out = []
    for rid, job in sorted(self.plan_state.jobs.items()):
      if job.spec != spec or (h := next(self.samples.queued(rid, 1), None)) is None:
        continue
      mp = self.runs[rid].run.model_path
      if not mp or (ver := (self.checkpoints.get(mp) or {}).get("version")):
        out.append((h, ver if mp else None))
      else:
        self._finish_sample(h.request_id, None, Reply.error(ValueError(f"no checkpoint at {mp}")))
    return out

  # ***** sessions, runs, commands, checkpoints *****

  def heartbeat_session(self, sid: str, metadata: dict | None = None) -> None:
    s = self.sessions.get(sid) or {"created": (t := time.time()), "seen": t, "active": True, "metadata": {}}
    with self.db:
      self.sessions[sid] = s | {"seen": time.time(), "active": True, **({"metadata": metadata} if metadata else {})}

  def get_session(self, sid: str) -> dict | None:
    if (s := self.sessions.get(sid)) is None:
      return None
    runs, live = (
      sorted(r for r, row in self.runs.items() if row.run.session == sid),
      time.time() - s["seen"] < LIVE_SECONDS,
    )
    info = {"session_id": sid, "created": s["created"], "seen": s["seen"], "live": live}
    return info | {"runs": runs, "metadata": dict(s["metadata"])}

  def finish_session(self, sid: str) -> None:
    if (s := self.sessions.get(sid)) and s["active"]:
      with self.db:
        self.sessions[sid] = s | {"seen": 0.0}
    self._wake()

  def end_session(self, sid: str) -> None:
    live = sorted(r for r, row in self.runs.items() if row.live and row.run.session == sid)
    tr = [bool(self.submit_command(r, Op.DELETE_MODEL, b"{}")) for r in live if not self._is_sampler(r)]
    for r in live:
      if self._is_sampler(r):
        self.delete_run(r)
    if s := self.sessions.get(sid):
      with self.db:
        self.sessions[sid] = s | ({"seen": time.time()} if any(tr) else {"active": False})

  def create_run(self, run: Run) -> None:
    with self.db:
      self.runs[run.id] = RunRow(run, time.time())
    self.samples.open(run.id) if run.role == Role.SAMPLER else self.train_q.setdefault(run.id, {})
    info = {k: getattr(run, k) for k in ("role", "base", "rank", "exclusive", "train_run", "gpus")}
    self.record_event(Ev.RUN_CREATED, {"run": run.id} | info)
    self._wake()

  def get_run(self, rid: str) -> dict | None:
    if (r := self.runs.get(rid)) is None:
      return None
    extra = {k: list(v) for k, v in (("owner", r.owner), ("snap", r.snap)) if v}
    return {**asdict(r.run), "created": r.created, "status": str(r.status), **extra, **r.counters}

  def delete_run(self, rid: str) -> None:
    if self._is_sampler(rid):
      for k in self.samples.drop(rid):
        self._notify(k)
    else:
      for k in list(self.train_q.get(rid, ())):
        if tok := self.cmds.get(k, {}).get("owner"):
          self._finish_train(k, tok, Reply.error(RuntimeError(f"run {rid} was deleted")), dead=True)
    with self.db:
      self._purge_run(rid)
    self.record_event(Ev.RUN_DELETED, {"run": rid})
    self._wake()

  def seen_request(self, rid: str, seq_id: int | None) -> str | None:
    if seq_id is None:
      return None
    if self._is_sampler(rid):
      return self.samples.seen_id(rid, seq_id)
    return hit[0] if (hit := self.seen.get(f"{rid}:{seq_id}")) and hit[1] > time.time() else None

  def submit_command(self, rid: str, op: Op, body: bytes, seq_id: int | None = None, cost: int = 1) -> str | None:
    if existing := self.seen_request(rid, seq_id):
      return existing
    if (row := self.runs.get(rid)) is None or not row.live:
      return None
    self.seq += 1
    cmd = Cmd(rid, f"{time.time_ns() // 1_000_000}-{self.seq}", Op(op), body, cost)
    if row.run.role == Role.SAMPLER:
      self.samples.submit(cmd, seq_id, SESSION_TTL)
    else:
      with self.db:
        self.db.put_blob(f"c:{cmd.request_id}", body)
        self.cmds[cmd.request_id] = {"run": rid, "eid": cmd.entry_id, "op": str(cmd.op), "cost": cost, "seq": self.seq}
        if seq_id is not None:
          self.seen[f"{rid}:{seq_id}"] = [cmd.request_id, time.time() + SESSION_TTL]
      self.train_q[rid][cmd.request_id] = None
    self._wake()
    return cmd.request_id

  async def wait_for_reply(self, req_id: str, timeout: float) -> Reply | None:
    sampler = self._is_sampler(req_id.partition("/")[0], req_id)
    read_ttl, deadline = min(REPLY_TTL, READ_REPLY_TTL), time.monotonic() + max(0.0, timeout)
    while True:
      if sampler:
        if (rep := self.samples.read_reply(req_id, read_ttl)) is not None:
          return rep
      else:
        m, now = self.replies.get(req_id), time.time()
        if m and m[2] > now and (b := self.db.get_blob(f"r:{req_id}")) is not None:
          with self.db:
            self.replies[req_id] = [m[0], m[1], min(m[2], now + read_ttl)]
          return Reply(bool(m[0]), m[1], b)
      if (rem := deadline - time.monotonic()) <= 0:
        return None
      with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(self.waiters.setdefault(req_id, asyncio.Event()).wait(), rem)

  def queued_and_in_flight(self, rid: str) -> dict:
    now = time.monotonic()
    if self._is_sampler(rid):
      return self.samples.status(rid, now)
    if (q := self.train_q.get(rid)) is None:
      return {"live": False}
    fly = []
    for k in q:
      if o := self.cmds[k].get("owner"):
        age = max(0.0, now - self.since.get(k, now))
        fly.append({"request_id": k, "trainer": o[0], "deliveries": 1, "seconds": age})
    return {"live": True, "queued": len(q) - len(fly), "in_flight": fly}

  def get_checkpoint(self, uri: str) -> dict | None:
    T.parse_uri(uri)
    return dict(c) if (c := self.checkpoints.get(uri)) else None

  def checkpoint_version(self, uri: str) -> str:
    if ver := (self.get_checkpoint(uri) or {}).get("version"):
      return ver
    raise ValueError(f"no checkpoint at {uri}")

  def set_checkpoint_ttl(self, uri: str, ttl: int | None) -> bool:
    T.parse_uri(uri)
    if (c := self.checkpoints.get(uri)) is None:
      return False
    with self.db:
      self.checkpoints[uri] = c | {"expires_at": time.time() + ttl if ttl else None}
    return True

  def delete_checkpoint(self, uri: str) -> bool:
    T.parse_uri(uri)
    with self.db:
      return self.checkpoints.pop(uri, None) is not None

  def live_versions(self) -> list[str]:
    v = {c["version"] for c in self.checkpoints.values() if c.get("version")}
    v.update(x for c in self.cmds.values() for x in (c.get("version"), c.get("snap")) if x)
    return sorted({x.partition("@")[0] for x in v})

  # ***** placement, liveness, and maintenance *****

  def state(self) -> scheduler.State:
    nodes = {nid: (n["gpus"], n["memory"]) for nid, n in self.nodes.items() if not n["draining"]}
    jobs = {}
    for rid, r in sorted(self.runs.items()):
      if r.live:
        u = r.run
        qlen = len(self.samples.queues.get(rid, ())) if u.role == Role.SAMPLER else len(self.train_q.get(rid, ()))
        jobs[rid] = scheduler.Job(u.spec, u.train_run, u.account or ("" if u.train_run else u.session), r.owner, qlen)
    jobs = {r: j for r, j in jobs.items() if not j.spec.train_run or j.spec.train_run in jobs}
    home = {r: p for r, p in self.plan_state.home.items() if r in jobs}
    return replace(self.plan_state, nodes=nodes, home=home, jobs=jobs, service=self.service)

  def live_nodes(self, details: bool = False) -> dict:
    self.reap_dead_nodes()
    return {k: dict(v) if details else v["gpus"] for k, v in self.nodes.items()}

  def reap_node(self, node_id: str, boot_id: str) -> None:
    if (cur := self.nodes.get(node_id)) and cur["boot_id"] == boot_id:
      del self.nodes[node_id]
    self.reaped_boots.add(boot_id)
    self.record_event(Ev.NODE_LOST, {"node": node_id})
    for req_id, c in list(self.cmds.items()):
      if (o := c.get("owner")) and o[1:] == (node_id, boot_id):
        msg = f"its trainer died, or lost ownership, during {c['op']}"
        self._finish_train(req_id, o, Reply.error(RuntimeError(msg)), dead=True)
    self._reclaim(lambda o: o[1:] == (node_id, boot_id))
    self._wake()

  def reap_dead_nodes(self) -> None:
    now = time.monotonic()
    for n in [n for n in list(self.nodes.values()) if n["until"] <= now]:
      self.reap_node(n["node"], n["boot_id"])
    if self.reap_after and now >= self.reap_after:
      self.reap_after, live = 0.0, {(n["node"], n["boot_id"]) for n in self.nodes.values()}
      owners = [*(c.get("owner") for c in self.cmds.values()), *(r.owner for r in self.runs.values())]
      for nid, bid in {o[1:] for o in owners if o} - live:
        self.reap_node(nid, bid)

  def reconcile(self) -> list[tuple[Ev, dict]]:
    old, self.plan_due = scheduler.to_plan(self.plan_state), time.monotonic() + scheduler.TICK_MS / 1000
    self.plan_state, changes = scheduler.step(self.state())
    self.plan_state.service = self.service
    if (plan := scheduler.to_plan(self.plan_state)) != old:
      with self.db:
        self.meta["plan"], self.meta["service"] = plan, self.service
      for kind, body in changes:
        self.record_event(kind, body)
      if any(plan[k] != old.get(k) for k in ("procs", "home")):
        self._wake()
    return changes

  def maintain(self) -> list[tuple[Ev, dict]]:
    self.reap_dead_nodes()
    now, mono = time.time(), time.monotonic()
    for sid in [s for s, v in list(self.sessions.items()) if v["active"] and v["seen"] <= now - SESSION_GRACE]:
      self.end_session(sid)
    with self.db:
      if self.meta.get("service") != self.service:
        self.meta["service"] = self.service
      for u in [u for u, c in self.checkpoints.items() if c["expires_at"] is not None and c["expires_at"] <= now]:
        del self.checkpoints[u]
      if mono >= self.gc_due:
        self.gc_due = mono + GC_S
        self.samples.gc(now)
        for k in [k for k, m in self.replies.items() if m[2] <= now]:
          del self.replies[k]
          self.db.del_blob(f"r:{k}")
        for k in [k for k, (_, exp) in self.seen.items() if exp <= now]:
          del self.seen[k]
        for s in [s for s, v in self.sessions.items() if not v["active"] and v["seen"] <= now - SESSION_TTL]:
          del self.sessions[s]
    return self.reconcile()

  def read_events(self, after: str = "0-0", count: int = 10_000) -> list[Event]:
    return [ev for ev in self.events if T.entry_sort_key(ev.id) > T.entry_sort_key(after)][:count]

  # ***** the 4 fenced node RPCs: heartbeat, take, finish, detach *****

  def _assignments(self, node: str, boot_id: str) -> list[Assignment]:
    out, ps = [], self.plan_state
    for wid, proc in sorted(ps.procs.items()):
      if proc.node != node:
        continue
      tok = (wid, node, boot_id)
      if proc.spec.role == Role.TRAINER:
        runs, heads = tuple(ps.runs(wid)), self._head_train(wid, tok)
        held = [c["run"] for c in self.cmds.values() if c.get("owner") == tok]
        lruns = held or [c.run for c in heads] or [r for r in runs if r in ps.jobs]
        closing = tuple(c.run for c in heads if c.op == Op.OPTIM_STEP)
        lag = max(map(ps.ready_lag, lruns), default=0.0)
        out.append(Assignment(wid, proc.spec, runs, (), bool(held or heads), closing, lag))
      else:
        pairs = self._ready_samples(proc.spec)
        rids, vers = tuple(c.run for c, _ in pairs), tuple(sorted({v for _, v in pairs if v}))
        ready = any(e.owner == tok for e in self.samples.by_id.values()) or any(v is None for _, v in pairs)
        out.append(Assignment(wid, proc.spec, rids, vers, ready, (), max(map(ps.ready_lag, rids), default=0.0)))
    return out

  def _beat(self, node: str, boot_id: str, gpus: int, info: dict) -> tuple[list[Assignment], list[str] | None]:
    cur, now = self._fence(node, boot_id), time.monotonic()
    mem, draining = int(info.get("memory", 0)), bool(info.get("draining", False))
    if cur is None:
      self.record_event(Ev.NODE_UP, {"node": node, "gpus": gpus, "memory": mem})
    elif (cur["gpus"], cur["memory"], cur["draining"]) != (gpus, mem, draining):
      self.plan_due = 0.0
    on, pids = {k: list(v) for k, v in (info.get("on") or {}).items() if v}, dict(info.get("pids") or {})
    rec = {"node": node, "boot_id": boot_id, "gpus": gpus, "memory": mem, "draining": draining, "pids": pids, "on": on}
    self.nodes[node] = rec | {"lag": float(info.get("lag", 0.0)), "until": now + LEASE_MS / 1000}
    if (live_w := info.get("live_workers")) is not None:
      keep = set(live_w) | {o[0] for c in self.cmds.values() if (o := c.get("owner")) and o[1:] == (node, boot_id)}
      self._reclaim(lambda o: o[1:] == (node, boot_id) and o[0] not in keep)
    if now >= self.plan_due:
      self.reconcile()
    sweep = self.live_versions() if info.get("want_sweep") and now >= self.sweep_due else None
    self.sweep_due = now + SWEEP_S if sweep is not None else self.sweep_due
    return ([] if draining else self._assignments(node, boot_id)), sweep

  async def heartbeat(self, node: str, boot_id: str, gpus: int, wait_ms: int = 0, **info):
    plan, sweep = self._beat(node, boot_id, gpus, info)
    if wait_ms <= 0 or info.get("draining") or sweep is not None or any(a.ready or a.versions for a in plan):
      return plan, sweep
    with contextlib.suppress(TimeoutError):
      await asyncio.wait_for(self.work_event.wait(), wait_ms / 1000)
    return self._beat(node, boot_id, gpus, info)

  def take(self, node: str, boot_id: str, worker: str, **opts) -> list[Work]:
    cur, now = self._fence(node, boot_id), time.monotonic()
    if now >= self.plan_due:
      self.reconcile()
    ps = self.plan_state
    if (proc := ps.procs.get(worker)) is None or proc.node != node or (cur and cur["draining"]):
      return []
    tok, svc, hold = (worker, node, boot_id), opts.get("service") or {}, set(opts.get("holding") or ())
    if proc.spec.role == Role.TRAINER:
      if hk := next((k for k, c in self.cmds.items() if c.get("owner") == tok), None):
        if hk in hold:
          return []
        m, r = self.cmds[hk], self.cmds[hk]["run"]
        return [
          Work(self._cmd(hk, full=True), ps.account(r), asdict(self.runs[r].run), m.get("snap"), m.get("version"))
        ]
      loaded = set(opts.get("loaded") or ())
      while heads := self._head_train(worker, tok):
        rid = rank_runs(heads, ps.ready_lag, svc)[0]
        cmd, row, ver = self._cmd(next(iter(self.train_q[rid])), full=True), self.runs[rid], None
        if cmd.op == Op.LOAD_WEIGHTS:
          try:
            ver = self.checkpoint_version(json.loads(cmd.body)["uri"])
          except (KeyError, ValueError, json.JSONDecodeError) as err:
            self._finish_train(cmd.request_id, None, Reply.error(err))
            continue
        elif cmd.op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER):
          ver = uuid.uuid4().hex
        use_snap = row.status == RunStatus.SAVED and row.snap and rid not in loaded and cmd.op != Op.LOAD_WEIGHTS
        snap, self.since[cmd.request_id], row.owner = (row.snap[1] if use_snap else None), now, tok
        with self.db:
          self.cmds[cmd.request_id] = self.cmds[cmd.request_id] | {"owner": tok, "snap": snap, "version": ver}
          self.runs.save(rid)
        return [Work(cmd, ps.account(rid), asdict(row.run), snap, ver)]
      return []

    staged, slots = set(opts.get("staged") or ()), int(opts.get("slots", 1))
    out = []
    for e in self.samples.by_id.values():
      if e.owner == tok and e.cmd.request_id not in hold:
        ver = (self.checkpoints.get(self.runs[e.cmd.run].run.model_path or "") or {}).get("version")
        out.append(Work(e.cmd, ps.account(e.cmd.run), version=ver))
    vm = {c.run: v for c, v in self._ready_samples(proc.spec) if v is None or v in staged}
    cap = (opts.get("active_account", ""), opts.get("active_versions") or (), max(0, slots - len(out)))
    heads, max_l = [next(self.samples.queued(r, 1)) for r in vm], (T.MAX_LORAS if proc.spec.lora else 1)
    for c in admit_samples(heads, self.samples.queued, ps.account, ps.ready_lag, svc, vm.get, *cap, max_l):
      self.samples.claim(c, tok, now)
      out.append(Work(c, ps.account(c.run), version=vm[c.run]))
    return out

  def _finish_train(
    self, req_id: str, owner, reply: Reply, charge=None, *, mutated=False, saved=None, dead=False
  ) -> bool:
    meta = self.cmds.get(req_id)
    if meta is None or meta.get("owner") != owner or req_id not in self.train_q.get(meta["run"], ()):
      return False
    cmd = self._cmd(req_id)
    del self.train_q[cmd.run][req_id]
    row, deleting = self.runs.get(cmd.run), cmd.op == Op.DELETE_MODEL and reply.ok
    if row:
      self._bill(row, cmd, reply, charge)
    with self.db:
      self._drop_cmd(req_id)
      self.db.put_blob(f"r:{req_id}", reply.body)
      self.replies[req_id] = [reply.ok, reply.content_type, time.time() + REPLY_TTL]
      if charge and charge.gpu_ms:
        self.meta["service"] = self.service
      if row:
        if saved and reply.ok:
          self.checkpoints[saved["uri"]] = make_checkpoint(**saved)
        if deleting:
          self._purge_run(cmd.run, owned_ckpts=True)
        else:
          if mutated:
            if row.snap:
              self.checkpoints.pop(row.snap[0], None)
              row.snap = None
            row.status = RunStatus.LIVE
          if dead and owner:
            self._lose_owners(lambda o: o == owner)
          elif dead or row.status in (RunStatus.FRESH, RunStatus.LOST):
            row.owner = None
        self.runs.save(cmd.run)
    if deleting:
      self.record_event(Ev.RUN_DELETED, {"run": cmd.run})
    self._notify(req_id)
    return True

  def _finish_sample(self, req_id: str, owner, reply: Reply, charge=None) -> bool:
    if (cmd := self.samples.complete(req_id, owner, reply, REPLY_TTL)) is None:
      return False
    if row := self.runs.get(cmd.run):
      self._bill(row, cmd, reply, charge)
    self._notify(req_id)
    return True

  def finish(self, node: str, boot_id: str, worker: str, req_id: str, reply: Reply, charge=None, **eff) -> bool:
    self._fence(node, boot_id)
    tok = (worker, node, boot_id)
    if self._is_sampler(req_id.partition("/")[0], req_id):
      ok = self._finish_sample(req_id, tok, reply, charge)
    else:
      kw = {"mutated": bool(eff.get("mutated")), "saved": eff.get("saved"), "dead": bool(eff.get("dead"))}
      ok = self._finish_train(req_id, tok, reply, charge, **kw)
    if ok:
      self._wake()
    return ok

  def detach(
    self, node: str, boot_id: str, worker: str, rid: str, saved: dict | None = None, lost: bool = False
  ) -> bool:
    self._fence(node, boot_id)
    if (row := self.runs.get(rid)) is None or row.owner != (worker, node, boot_id):
      return False
    with self.db:
      row.owner = None
      if saved:
        if row.snap:
          self.checkpoints.pop(row.snap[0], None)
        self.checkpoints[saved["uri"]] = ckpt = make_checkpoint(**saved)
        row.snap, row.status = (ckpt["tinker_path"], ckpt["version"]), RunStatus.SAVED
      elif lost and row.status == RunStatus.LIVE:
        row.status = RunStatus.LOST
      self.runs.save(rid)
    self._wake()
    return True
