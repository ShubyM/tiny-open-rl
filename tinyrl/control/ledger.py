"""Controller ledger: durable runs/checkpoints/train queues, RAM sampling, and 4 typed node RPCs (`heartbeat`, `take`, `finish`, `detach`)."""

from __future__ import annotations

import asyncio
import contextlib
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
from tinyrl.control.samples import SampleQueue, flight_status
from tinyrl.helpers import LEASE_MS, SWEEP_S, getenv
from tinyrl.types import Assignment, Cmd, Effect, Ev, FencedError, NodeStatus, Op, Reply, Role, Run, RunStatus, Work

LIVE_SECONDS, SESSION_GRACE, GC_S, TICK_MS = 30, getenv("SESSION_GRACE", 60), 10, getenv("TICK_MS", 100)
REPLY_TTL, READ_REPLY_TTL, SESSION_TTL = getenv("REPLY_TTL", 3600), 60, getenv("SESSION_TTL", 7 * 24 * 3600)
COUNTERS, MAX_REAPED = ("steps", "failed", "cmds", "tokens", "busy_ms", "wait_ms"), 1024


@dataclass
class RunRow:
  run: Run
  created: float
  status: RunStatus = RunStatus.FRESH
  owner: tuple[str, str, str] | None = None  # (worker, node, boot_id)
  snap: str | None = None  # Store version of detached trainer state
  clients: int = 0
  deleted_at: float = 0.0
  counters: dict[str, int] = field(default_factory=lambda: dict.fromkeys(COUNTERS, 0))

  @property
  def live(self) -> bool:
    return self.status != RunStatus.DELETED


@dataclass
class CmdRow:
  run: str
  eid: str
  op: Op
  cost: int
  seq: int
  uri: str | None = None
  owner: tuple[str, str, str] | None = None
  snap: str | None = None
  version: str | None = None


def make_checkpoint(s: T.Saved) -> dict:
  raw, now = (cid := T.parse_uri(s.uri)[1]).split("/", 1)[0], time.time()
  kind, exp = {"weights": "training", "sampler_weights": "sampler"}.get(raw, raw), (now + s.ttl if s.ttl else None)
  row = {"checkpoint_id": cid, "checkpoint_type": kind, "time": now, "tinker_path": s.uri, "size_bytes": s.size}
  return row | {"expires_at": exp, "owned": bool(s.owned), "version": s.version, "meta": dict(s.meta)}


class Ledger:
  def __init__(self, path: str | None = None):
    self.db = DB(getenv("TINYRL_DB", "/tmp/tinyrl.db") if path is None else path)
    self.runs, self.cmds = Table(self.db, "runs", RunRow), Table(self.db, "cmds", CmdRow)
    tabs = [Table(self.db, n) for n in ("sessions", "checkpoints", "replies", "seen", "meta")]
    self.sessions, self.checkpoints, self.replies, self.seen, self.meta = tabs
    self.service = {k: float(v) for k, v in self.meta.get("service", {}).items()}
    self.service_dirty, self.reaped_boots = False, list(self.meta.get("reaped", ()))
    self.last_plan = dict(self.meta.get("plan", {}))
    self.plan_state = scheduler.from_plan(self.last_plan)
    self.samples, self.since = SampleQueue(), {}
    self.train_q = {rid: {} for rid, r in self.runs.items() if r.live and r.run.role == Role.TRAINER}
    now, self.seq = time.monotonic(), max([int(self.meta.get("seq", 0)), *(c.seq for c in self.cmds.values())])
    for req_id, c in sorted(self.cmds.items(), key=lambda kv: kv[1].seq):
      if c.run in self.train_q:
        self.train_q[c.run][req_id] = None
        if c.owner:
          self.since[req_id] = now
    self.nodes, self.events = {}, deque(maxlen=100_000)
    self.waiters, self.work_event = weakref.WeakValueDictionary(), asyncio.Event()
    self.plan_due = self.gc_due = self.sweep_due = 0.0
    held = self.since or any(r.live and r.owner for r in self.runs.values())
    self.reap_after = now + LEASE_MS / 1000 if held else 0.0
    self.plan_state = self.state()

  def flush(self) -> None:
    if self.service_dirty:
      self.meta["service"], self.service_dirty = dict(self.service), False
    if self.meta.get("seq") != self.seq:
      self.meta["seq"] = self.seq

  async def aclose(self) -> None:
    if self.db is not None:
      with self.db:
        self.flush()
      self.db.conn.close()
      self.db = None

  def notify(self, req_id: str) -> None:
    if ev := self.waiters.pop(req_id, None):
      ev.set()

  def wake(self) -> None:
    ev, self.work_event = self.work_event, asyncio.Event()
    ev.set()

  def record_event(self, kind: Ev, body: dict) -> T.Event:
    self.seq += 1
    self.events.append(ev := T.Event(f"{int((now := time.time()) * 1000)}-{self.seq}", now, str(Ev(kind)), body))
    self.plan_due = 0.0 if kind in (Ev.RUN_CREATED, Ev.RUN_DELETED, Ev.NODE_UP, Ev.NODE_LOST) else self.plan_due
    return ev

  def is_sampler(self, rid: str, req_id: str = "") -> bool:
    return self.runs[rid].run.role == Role.SAMPLER if rid in self.runs else req_id.startswith("samp-")

  def ckpt_ver(self, rid: str) -> str | None:
    mp = self.runs[rid].run.model_path if rid in self.runs else None
    return (self.checkpoints.get(mp) or {}).get("version") if mp else None

  def make_cmd(self, req_id: str, *, full: bool = False) -> Cmd:
    c, body = self.cmds[req_id], (self.db.get_blob(f"c:{req_id}") or b"") if full else b""
    return Cmd(c.run, c.eid, c.op, body, c.cost)

  def make_work(self, req_id: str) -> Work:
    c = self.cmds[req_id]
    return Work(self.make_cmd(req_id, full=True), asdict(self.runs[c.run].run), c.snap, c.version)

  def bill(self, row: RunRow, cmd: Cmd, reply: Reply, charge: Charge | None) -> None:
    c = row.counters
    c["failed"], c["steps"] = c["failed"] + (not reply.ok), c["steps"] + (reply.ok and cmd.op == Op.OPTIM_STEP)
    if charge:
      c["cmds"], c["busy_ms"] = c["cmds"] + 1, c["busy_ms"] + charge.busy_ms
      c["wait_ms"] += max(0, charge.started_ms - T.entry_sort_key(cmd.entry_id)[0])
      c["tokens"] += cmd.cost if reply.ok else 0
    if charge and charge.gpu_ms and (acct := self.plan_state.account(cmd.run)):
      self.service[acct], self.service_dirty = self.service.get(acct, 0.0) + charge.gpu_ms, True

  def lose_owners(self, pred) -> None:
    for rid, row in self.runs.items():
      if row.owner and pred(row.owner):
        row.owner, row.status = None, (RunStatus.LOST if row.status == RunStatus.LIVE else row.status)
        self.runs.save(rid)

  def reclaim(self, pred) -> None:
    self.samples.reclaim(pred)
    if any(r.owner and pred(r.owner) for r in self.runs.values()):
      with self.db:
        self.lose_owners(pred)

  def drop_cmd(self, req_id: str) -> None:
    self.since.pop(req_id, None)
    self.cmds.pop(req_id, None)
    self.db.del_blob(f"c:{req_id}")

  def purge_run(self, rid: str, *, owned_ckpts: bool = False) -> None:
    if r := self.runs.get(rid):
      r.status, r.owner, r.snap, r.deleted_at = RunStatus.DELETED, None, None, time.time()
      self.runs.save(rid)
    for k in self.train_q.pop(rid, ()):
      self.drop_cmd(k)
      self.notify(k)
    for u in [u for u, c in self.checkpoints.items() if owned_ckpts and T.parse_uri(u)[0] == rid and c["owned"]]:
      del self.checkpoints[u]

  def fence(self, node: str, boot_id: str) -> dict | None:
    self.reap_dead_nodes()
    cur = self.nodes.get(node)
    if boot_id in self.reaped_boots or (cur is not None and cur["boot_id"] != boot_id):
      raise FencedError(f"fenced boot {node}/{boot_id}")
    if cur is not None:
      cur["until"] = time.monotonic() + LEASE_MS / 1000
    return cur

  def head_train(self, worker: str, tok: tuple[str, str, str]) -> list[Cmd]:
    out = []
    for r in self.plan_state.runs(worker):
      row, k = self.runs.get(r), next(iter(self.train_q.get(r, ())), None)
      if row and row.live and row.owner in (None, tok) and k and not self.cmds[k].owner:
        out.append(self.make_cmd(k))
    return out

  def ready_samples(self, spec) -> list[tuple[Cmd, str | None]]:
    out = []
    for rid, job in sorted(self.plan_state.jobs.items()):
      if job.spec != spec or (h := next(self.samples.queued(rid, 1), None)) is None:
        continue
      if not (mp := self.runs[rid].run.model_path) or (ver := self.ckpt_ver(rid)):
        out.append((h, ver if mp else None))
      else:
        self.finish_sample(h.request_id, None, Reply.error(ValueError(f"no checkpoint at {mp}")))
    return out

  # ***** sessions, runs, commands, checkpoints *****

  def heartbeat_session(self, sid: str, metadata: dict | None = None, *, finish: bool = False) -> None:
    now = time.time()
    s = self.sessions.get(sid) or (None if finish else {"created": now, "seen": now, "active": True, "metadata": {}})
    if s and (not finish or s["active"]):
      patch = {"seen": 0.0} if finish else {"seen": now, "active": True} | ({"metadata": metadata} if metadata else {})
      with self.db:
        self.sessions[sid] = s | patch
    if finish:
      self.wake()

  def get_session(self, sid: str) -> dict | None:
    if (s := self.sessions.get(sid)) is None:
      return None
    runs, live = (
      sorted(r for r, row in self.runs.items() if row.run.session == sid),
      time.time() - s["seen"] < LIVE_SECONDS,
    )
    info = {"session_id": sid, "created": s["created"], "seen": s["seen"], "live": live}
    return info | {"runs": runs, "metadata": dict(s["metadata"])}

  def end_session(self, sid: str) -> None:
    live = sorted(r for r, row in self.runs.items() if row.live and row.run.session == sid)
    tr = [bool(self.submit_command(r, Op.DELETE_MODEL, b"{}")) for r in live if not self.is_sampler(r)]
    for r in live:
      if self.is_sampler(r):
        self.delete_run(r)
    if s := self.sessions.get(sid):
      with self.db:
        self.sessions[sid] = s | ({"seen": time.time()} if any(tr) else {"active": False})

  def create_run(self, run: Run) -> None:
    with self.db:
      self.runs[run.id] = RunRow(run, time.time())
    if run.role == Role.TRAINER:
      self.train_q.setdefault(run.id, {})
    info = {k: getattr(run, k) for k in ("role", "base", "rank", "exclusive", "train_run", "gpus")}
    self.record_event(Ev.RUN_CREATED, {"run": run.id} | info)
    self.wake()

  def join_sampling_session(self, rid: str) -> int:
    if (row := self.runs.get(rid)) is None:
      return 1
    row.clients += 1
    with self.db:
      self.runs.save(rid)
    return row.clients

  def get_run(self, rid: str) -> dict | None:
    if (r := self.runs.get(rid)) is None:
      return None
    extra = ({"owner": list(r.owner)} if r.owner else {}) | ({"snap": r.snap} if r.snap else {})
    return {**asdict(r.run), "created": r.created, "status": str(r.status), **extra, **r.counters}

  def delete_run(self, rid: str) -> None:
    if self.is_sampler(rid):
      for k in self.samples.drop(rid):
        self.notify(k)
    else:
      for k in list(self.train_q.get(rid, ())):
        if tok := self.cmds[k].owner:
          self.finish_train(k, tok, Reply.error(RuntimeError(f"run {rid} was deleted")))
    with self.db:
      self.purge_run(rid)
    self.record_event(Ev.RUN_DELETED, {"run": rid})
    self.wake()

  def seen_request(self, rid: str, seq_id: int | None) -> str | None:
    if seq_id is None:
      return None
    if self.is_sampler(rid):
      return self.samples.seen_id(rid, seq_id)
    return hit[0] if (hit := self.seen.get(f"{rid}:{seq_id}")) and hit[1] > time.time() else None

  def submit_command(self, rid: str, op: Op, body: bytes, seq_id=None, cost: int = 1, uri=None) -> str | None:
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
        self.cmds[cmd.request_id] = CmdRow(rid, cmd.entry_id, cmd.op, cost, self.seq, uri)
        if seq_id is not None:
          self.seen[f"{rid}:{seq_id}"] = [cmd.request_id, time.time() + SESSION_TTL]
      self.train_q[rid][cmd.request_id] = None
    self.wake()
    return cmd.request_id

  async def wait_for_reply(self, req_id: str, timeout: float) -> Reply | None:
    sampler = self.is_sampler(req_id.partition("/")[0], req_id)
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
    row, now = self.runs.get(rid), time.monotonic()
    if row is None or not row.live:
      return {"live": False}
    if row.run.role == Role.SAMPLER:
      return self.samples.status(rid, now)
    q = self.train_q.get(rid, {})
    return flight_status(len(q), [(k, o[0], self.since.get(k, now)) for k in q if (o := self.cmds[k].owner)], now)

  def list_checkpoints(self, run: str | None = None) -> list[dict]:
    items = (c for u, c in self.checkpoints.items() if run is None or T.parse_uri(u)[0] == run)
    return sorted(items, key=lambda c: c["time"], reverse=run is None)

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
    v.update(r.snap for r in self.runs.values() if r.live and r.snap)
    v.update(x for c in self.cmds.values() for x in (c.version, c.snap) if x)
    return sorted({x.partition("@")[0] for x in v})

  # ***** placement, liveness, and maintenance *****

  def state(self) -> scheduler.State:
    nodes = {nid: n["gpus"] for nid, n in self.nodes.items() if not n["draining"]}
    jobs = {}
    for rid, r in sorted(self.runs.items()):
      if r.live:
        u = r.run
        qlen = self.samples.backlog(rid) if u.role == Role.SAMPLER else len(self.train_q.get(rid, ()))
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
    if boot_id not in self.reaped_boots:
      self.reaped_boots = [*self.reaped_boots[-(MAX_REAPED - 1) :], boot_id]
    self.record_event(Ev.NODE_LOST, {"node": node_id})
    for req_id, c in list(self.cmds.items()):
      if c.owner and c.owner[1:] == (node_id, boot_id):
        err = RuntimeError(f"its trainer died, or lost ownership, during {c.op}")
        self.finish_train(req_id, c.owner, Reply.error(err), dead=True)
    with self.db:
      self.meta["reaped"] = list(self.reaped_boots)
    self.reclaim(lambda o: o[1:] == (node_id, boot_id))
    self.wake()

  def reap_dead_nodes(self) -> None:
    now = time.monotonic()
    for n in [n for n in list(self.nodes.values()) if n["until"] <= now]:
      self.reap_node(n["node"], n["boot_id"])
    if self.reap_after and now >= self.reap_after:
      self.reap_after, live = 0.0, {(n["node"], n["boot_id"]) for n in self.nodes.values()}
      owners = [*(c.owner for c in self.cmds.values()), *(r.owner for r in self.runs.values())]
      for nid, bid in {o[1:] for o in owners if o} - live:
        self.reap_node(nid, bid)

  def reconcile(self) -> list[tuple[Ev, dict]]:
    old, self.plan_due = self.last_plan, time.monotonic() + TICK_MS / 1000
    self.plan_state, changes = scheduler.step(self.state())
    if (plan := scheduler.to_plan(self.plan_state)) != old:
      self.last_plan = plan
      with self.db:
        self.meta["plan"] = plan
        self.flush()
      for kind, body in changes:
        self.record_event(kind, body)
      if any(plan[k] != old.get(k) for k in ("procs", "home")):
        self.wake()
    return changes

  def maintain(self) -> list[tuple[Ev, dict]]:
    self.reap_dead_nodes()
    now, mono = time.time(), time.monotonic()
    for sid in [s for s, v in list(self.sessions.items()) if v["active"] and v["seen"] <= now - SESSION_GRACE]:
      self.end_session(sid)
    run_gc = mono >= self.gc_due
    with self.db:
      self.flush()
      for u in [u for u, c in self.checkpoints.items() if c["expires_at"] is not None and c["expires_at"] <= now]:
        del self.checkpoints[u]
      if run_gc:
        self.gc_due, cut = mono + GC_S, now - SESSION_TTL
        self.samples.gc(now)
        for k in [k for k, m in self.replies.items() if m[2] <= now]:
          del self.replies[k]
          self.db.del_blob(f"r:{k}")
        for k in [k for k, m in self.seen.items() if m[1] <= now]:
          del self.seen[k]
        for s in [s for s, v in self.sessions.items() if not v["active"] and v["seen"] <= cut]:
          del self.sessions[s]
        dead = [r for r, row in self.runs.items() if 0 < row.deleted_at <= cut and row.run.session not in self.sessions]
        for r in dead:
          del self.runs[r]
    if run_gc:
      self.db.vacuum()
    return self.reconcile()

  def read_events(self, after: str = "0-0", count: int = 10_000) -> list[T.Event]:
    cut = T.entry_sort_key(after)
    return [ev for ev in self.events if T.entry_sort_key(ev.id) > cut][:count]

  # ***** the 4 fenced node RPCs: heartbeat, take, finish, detach *****

  def assignments(self, node: str, boot_id: str) -> list[Assignment]:
    out, ps = [], self.plan_state
    for wid, proc in sorted(ps.procs.items()):
      if proc.node != node:
        continue
      tok = (wid, node, boot_id)
      if proc.spec.role == Role.TRAINER:
        runs, heads = tuple(ps.runs(wid)), self.head_train(wid, tok)
        held = [c.run for c in self.cmds.values() if c.owner == tok]
        lruns = held or [c.run for c in heads] or [r for r in runs if r in ps.jobs]
        closing, lag = tuple(c.run for c in heads if c.op == Op.OPTIM_STEP), max(map(ps.ready_lag, lruns), default=0.0)
        out.append(Assignment(wid, proc.spec, runs, (), bool(held or heads), closing, lag))
      else:
        pairs = self.ready_samples(proc.spec)
        rids, vers = tuple(c.run for c, ver in pairs), tuple(sorted({ver for c, ver in pairs if ver}))
        ready = bool(self.samples.held(tok)) or any(ver is None for c, ver in pairs)
        out.append(Assignment(wid, proc.spec, rids, vers, ready, (), max(map(ps.ready_lag, rids), default=0.0)))
    return out

  async def heartbeat(self, node: str, boot_id: str, st: NodeStatus, wait_ms: int = 0):
    cur, now = self.fence(node, boot_id), time.monotonic()
    if cur is None:
      self.record_event(Ev.NODE_UP, {"node": node, "gpus": st.gpus, "memory": 0})
    elif (cur["gpus"], cur["draining"]) != (st.gpus, st.draining):
      self.plan_due = 0.0
    info = {"node": node, "boot_id": boot_id, "gpus": st.gpus, "draining": st.draining, "lag": st.lag}
    info |= {"until": now + LEASE_MS / 1000, "pids": dict(st.pids), "on": {k: list(v) for k, v in st.on.items() if v}}
    self.nodes[node] = info
    if st.live_workers is not None:
      held_w = {c.owner[0] for c in self.cmds.values() if c.owner and c.owner[1:] == (node, boot_id)}
      self.reclaim(lambda o: o[1:] == (node, boot_id) and o[0] not in (set(st.live_workers) | held_w))

    def reply():
      if (t := time.monotonic()) >= self.plan_due:
        self.reconcile()
      sweep = self.live_versions() if st.want_sweep and t >= self.sweep_due else None
      self.sweep_due = t + SWEEP_S if sweep is not None else self.sweep_due
      return ([] if st.draining else self.assignments(node, boot_id)), sweep

    plan, sweep = reply()
    if wait_ms <= 0 or st.draining or sweep is not None or any(a.ready or a.versions for a in plan):
      return plan, sweep
    with contextlib.suppress(TimeoutError):
      await asyncio.wait_for(self.work_event.wait(), wait_ms / 1000)
    self.fence(node, boot_id)
    return reply()

  def take(
    self, node: str, boot: str, wid: str, slots=1, staged=(), failed=None, service=None, holding=()
  ) -> list[Work]:
    cur, now = self.fence(node, boot), time.monotonic()
    if now >= self.plan_due:
      self.reconcile()
    ps = self.plan_state
    if (proc := ps.procs.get(wid)) is None or proc.node != node or (cur and cur["draining"]):
      return []
    tok, svc, hold = (wid, node, boot), service or {}, set(holding)
    if proc.spec.role == Role.TRAINER:
      if hk := next((k for k, c in self.cmds.items() if c.owner == tok), None):
        return [] if hk in hold else [self.make_work(hk)]
      while slots > 0 and (heads := self.head_train(wid, tok)):
        rid = rank_runs(heads, ps.ready_lag, svc)[0]
        c, row = self.cmds[k := next(iter(self.train_q[rid]))], self.runs[rid]
        ver = uuid.uuid4().hex if c.op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER) else None
        if c.op == Op.LOAD_WEIGHTS:
          try:
            ver = self.checkpoint_version(c.uri or "")
          except ValueError as err:
            self.finish_train(k, None, Reply.error(err))
            continue
        c.owner, c.version = tok, ver
        c.snap = row.snap if row.status == RunStatus.SAVED and c.op != Op.LOAD_WEIGHTS else None
        self.since[k], row.owner = now, tok
        with self.db:
          self.cmds.save(k)
          self.runs.save(rid)
        return [self.make_work(k)]
      return []

    for c, ver in self.ready_samples(proc.spec) if failed else ():
      if ver in failed:
        for q in list(self.samples.queued(c.run)):
          self.finish_sample(q.request_id, None, Reply.error(RuntimeError(failed[ver])))
    if slots <= 0:
      return []
    held_all, staged_set = self.samples.held(tok), set(staged)
    out = [Work(c, version=self.ckpt_ver(c.run)) for c in held_all if c.request_id not in hold]
    act = (ps.account(held_all[0].run) if held_all else "", tuple({v for c in held_all if (v := self.ckpt_ver(c.run))}))
    vm = {c.run: v for c, v in self.ready_samples(proc.spec) if v is None or v in staged_set}
    heads = [next(self.samples.queued(r, 1)) for r in vm]
    cap = (max(0, slots - len(out)), T.MAX_LORAS if proc.spec.lora else 1)
    for c in admit_samples(heads, self.samples.queued, ps.account, ps.ready_lag, svc, vm.get, *act, *cap):
      self.samples.claim(c, tok, now)
      out.append(Work(c, version=vm[c.run]))
    return out

  def finish_train(
    self, req_id: str, owner, reply: Reply, charge=None, *, mutated=False, saved=None, dead=False
  ) -> bool:
    if (c := self.cmds.get(req_id)) is None or c.owner != owner or req_id not in self.train_q.get(c.run, ()):
      return False
    cmd = self.make_cmd(req_id)
    del self.train_q[cmd.run][req_id]
    row, deleting = self.runs.get(cmd.run), cmd.op == Op.DELETE_MODEL and reply.ok
    if row:
      self.bill(row, cmd, reply, charge)
    with self.db:
      self.drop_cmd(req_id)
      self.db.put_blob(f"r:{req_id}", reply.body)
      self.replies[req_id] = [reply.ok, reply.content_type, time.time() + REPLY_TTL]
      self.flush()
      if row:
        if saved and reply.ok:
          self.checkpoints[saved.uri] = make_checkpoint(saved)
        if deleting:
          self.purge_run(cmd.run, owned_ckpts=True)
        else:
          if mutated:
            row.snap, row.status = None, RunStatus.LIVE
          if dead and owner:
            self.lose_owners(lambda o: o == owner)
          elif dead or row.status in (RunStatus.FRESH, RunStatus.LOST):
            row.owner = None
          self.runs.save(cmd.run)
    if deleting:
      self.record_event(Ev.RUN_DELETED, {"run": cmd.run})
    self.notify(req_id)
    return True

  def finish_sample(self, req_id: str, owner, reply: Reply, charge=None) -> bool:
    if (cmd := self.samples.complete(req_id, owner, reply, REPLY_TTL)) is None:
      return False
    if row := self.runs.get(cmd.run):
      self.bill(row, cmd, reply, charge)
    self.notify(req_id)
    return True

  def finish(
    self, node: str, boot: str, wid: str, rid: str, rep: Reply, ch: Charge | None = None, effect: Effect = Effect()
  ):
    self.fence(node, boot)
    tok = (wid, node, boot)
    if self.is_sampler(rid.partition("/")[0], rid):
      ok = self.finish_sample(rid, tok, rep, ch)
    else:
      ok = self.finish_train(rid, tok, rep, ch, mutated=effect.mutated, saved=effect.saved, dead=effect.dead)
    if ok:
      self.wake()
    return ok

  def detach(self, node: str, boot_id: str, wid: str, rid: str, snap: str | None = None) -> bool:
    self.fence(node, boot_id)
    if (row := self.runs.get(rid)) is None or row.owner != (wid, node, boot_id):
      return False
    with self.db:
      row.owner = None
      if snap:
        row.snap, row.status = snap, RunStatus.SAVED
      elif row.status == RunStatus.LIVE:
        row.status = RunStatus.LOST
      self.runs.save(rid)
    self.wake()
    return True
