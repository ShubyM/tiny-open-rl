"""Controller ledger: sessions, runs, checkpoints, placement loop, and 4 typed node RPCs (`heartbeat`, `take`, `finish`, `detach`)."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
import weakref
from collections import deque
from dataclasses import asdict, dataclass, field, replace

from tinyrl import types
from tinyrl.control import scheduler
from tinyrl.control.accounting import Charge
from tinyrl.control.db import DB, Table
from tinyrl.control.fairness import admit_samples, rank_runs
from tinyrl.control.nodes import Node, NodeTable
from tinyrl.control.samples import SampleQueue
from tinyrl.control.trains import TrainQueue
from tinyrl.helpers import SWEEP_S, getenv
from tinyrl.types import (
  Assignment,
  Checkpoint,
  Cmd,
  Ev,
  Event,
  NodeStatus,
  Op,
  Owner,
  Reply,
  Role,
  Run,
  RunStatus,
  Work,
  entry_sort_key,
  parse_uri,
)

LIVE_SECONDS, SESSION_GRACE, GC_S, TICK_MS = 30, getenv("SESSION_GRACE", 60), 10, getenv("TICK_MS", 100)
REPLY_TTL, READ_REPLY_TTL, SESSION_TTL = getenv("REPLY_TTL", 3600), 60, getenv("SESSION_TTL", 7 * 24 * 3600)
COUNTERS = ("steps", "failed", "cmds", "tokens", "busy_ms", "wait_ms")


@dataclass
class RunRow:
  run: Run
  created: float
  status: RunStatus = RunStatus.FRESH
  owner: Owner | None = None
  snapshot: str | None = None  # Store version of detached trainer state
  clients: int = 0
  deleted_at: float = 0.0
  counters: dict[str, int] = field(default_factory=lambda: dict.fromkeys(COUNTERS, 0))

  @property
  def live(self) -> bool:
    return self.status != RunStatus.DELETED


class Ledger:
  def __init__(self, path: str | None = None):
    self.db = DB(getenv("TINYRL_DB", "/tmp/tinyrl.db") if path is None else path)
    self.runs = Table(self.db, "runs", RunRow)
    self.checkpoints = Table(self.db, "checkpoints", Checkpoint)
    self.sessions, self.meta = Table(self.db, "sessions"), Table(self.db, "meta")
    self.service = {k: float(v) for k, v in self.meta.get("service", {}).items()}
    self.service_dirty = False
    self.plan_state = scheduler.from_plan(self.meta.get("plan", {}))

    now = time.monotonic()
    self.samples = SampleQueue()
    self.trains = TrainQueue(
      self.db, (r for r, row in self.runs.items() if row.live and row.run.role == Role.TRAINER), now
    )
    self.seq = max(int(self.meta.get("seq", 0)), self.trains.max_seq())
    held = bool(self.trains.since or any(r.live and r.owner for r in self.runs.values()))
    self.nodes = NodeTable(self.meta.get("expired_boots", ()), held=held, now=now)

    self.events, self.waiters, self.work_event = deque(maxlen=100_000), weakref.WeakValueDictionary(), asyncio.Event()
    self.plan_due = self.gc_due = self.sweep_due = 0.0
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

  def notify_work(self) -> None:
    ev, self.work_event = self.work_event, asyncio.Event()
    ev.set()

  def record_event(self, kind: Ev, body: dict) -> Event:
    self.seq += 1
    self.events.append(ev := Event(f"{int((now := time.time()) * 1000)}-{self.seq}", now, str(Ev(kind)), body))
    self.plan_due = 0.0 if kind in (Ev.RUN_CREATED, Ev.RUN_DELETED, Ev.NODE_UP, Ev.NODE_LOST) else self.plan_due
    return ev

  def is_sampler(self, rid: str, req_id: str = "") -> bool:
    return self.runs[rid].run.role == Role.SAMPLER if rid in self.runs else req_id.startswith("samp-")

  def ckpt_ver(self, rid: str) -> str | None:
    mp = self.runs[rid].run.model_path if rid in self.runs else None
    return ckpt.version if mp and (ckpt := self.checkpoints.get(mp)) else None

  def bill(self, row: RunRow, cmd: Cmd, reply: Reply, charge: Charge | None) -> None:
    c = row.counters
    c["failed"], c["steps"] = c["failed"] + (not reply.ok), c["steps"] + (reply.ok and cmd.op == Op.OPTIM_STEP)
    if charge:
      c["cmds"], c["busy_ms"] = c["cmds"] + 1, c["busy_ms"] + charge.busy_ms
      c["wait_ms"] += max(0, charge.started_ms - entry_sort_key(cmd.entry_id)[0])
      c["tokens"] += cmd.cost if reply.ok else 0
    if charge and charge.gpu_ms and (acct := self.plan_state.account(cmd.run)):
      self.service[acct], self.service_dirty = self.service.get(acct, 0.0) + charge.gpu_ms, True

  def clear_owners(self, pred) -> None:
    for rid, row in self.runs.items():
      if row.owner and pred(row.owner):
        row.owner, row.status = None, (RunStatus.LOST if row.status == RunStatus.LIVE else row.status)
        self.runs.save(rid)

  def reclaim(self, pred) -> None:
    self.samples.reclaim(pred)
    if any(r.owner and pred(r.owner) for r in self.runs.values()):
      with self.db:
        self.clear_owners(pred)

  def remove_run(self, rid: str, *, owned_ckpts: bool = False) -> None:
    if r := self.runs.get(rid):
      r.status, r.owner, r.snapshot, r.deleted_at = RunStatus.DELETED, None, None, time.time()
      self.runs.save(rid)
    for k in self.trains.drop(rid):
      self.notify(k)
    for u in [u for u, c in self.checkpoints.items() if owned_ckpts and parse_uri(u)[0] == rid and c.owned]:
      del self.checkpoints[u]

  def fence(self, node: str, boot_id: str) -> Node | None:
    self.expire_nodes()
    return self.nodes.fence(node, boot_id)

  def head_train(self, worker: str, tok: Owner) -> list[Cmd]:
    out = []
    for r in self.plan_state.runs(worker):
      if (row := self.runs.get(r)) and row.live and row.owner in (None, tok) and (h := self.trains.head(r)):
        out.append(self.trains.cmd(h[0]))
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
      self.notify_work()

  def get_session(self, sid: str) -> dict | None:
    if (s := self.sessions.get(sid)) is None:
      return None
    runs = sorted(r for r, row in self.runs.items() if row.run.session == sid)
    info = {
      "session_id": sid,
      "created": s["created"],
      "seen": s["seen"],
      "live": time.time() - s["seen"] < LIVE_SECONDS,
    }
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
      self.trains.open(run.id)
    info = {k: getattr(run, k) for k in ("role", "base", "rank", "exclusive", "train_run", "gpus")}
    self.record_event(Ev.RUN_CREATED, {"run": run.id} | info)
    self.notify_work()

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
    extra = ({"owner": list(r.owner)} if r.owner else {}) | ({"snapshot": r.snapshot} if r.snapshot else {})
    return {**asdict(r.run), "created": r.created, "status": str(r.status), **extra, **r.counters}

  def delete_run(self, rid: str) -> None:
    if self.is_sampler(rid):
      for k in self.samples.drop(rid):
        self.notify(k)
    else:
      for k, tok in self.trains.in_flight(rid):
        self.finish_train(k, tok, Reply.error(RuntimeError(f"run {rid} was deleted")))
    with self.db:
      self.remove_run(rid)
    self.record_event(Ev.RUN_DELETED, {"run": rid})
    self.notify_work()

  def seen_request(self, rid: str, seq_id: int | None) -> str | None:
    if seq_id is None:
      return None
    return (self.samples if self.is_sampler(rid) else self.trains).seen_id(rid, seq_id)

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
      self.trains.submit(cmd, self.seq, seq_id, SESSION_TTL, uri)
    self.notify_work()
    return cmd.request_id

  async def wait_for_reply(self, req_id: str, timeout: float) -> Reply | None:
    queue = self.samples if self.is_sampler(req_id.partition("/")[0], req_id) else self.trains
    read_ttl, deadline = min(REPLY_TTL, READ_REPLY_TTL), time.monotonic() + max(0.0, timeout)
    while True:
      if (rep := queue.read_reply(req_id, read_ttl)) is not None:
        return rep
      if (rem := deadline - time.monotonic()) <= 0:
        return None
      with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(self.waiters.setdefault(req_id, asyncio.Event()).wait(), rem)

  def queued_and_in_flight(self, rid: str) -> dict:
    if (row := self.runs.get(rid)) is None or not row.live:
      return {"live": False}
    return (self.samples if row.run.role == Role.SAMPLER else self.trains).status(rid, time.monotonic())

  def list_checkpoints(self, run: str | None = None) -> list[Checkpoint]:
    items = (c for u, c in self.checkpoints.items() if run is None or parse_uri(u)[0] == run)
    return sorted(items, key=lambda c: c.time, reverse=run is None)

  def get_checkpoint(self, uri: str) -> Checkpoint | None:
    parse_uri(uri)
    return self.checkpoints.get(uri)

  def checkpoint_version(self, uri: str) -> str:
    if ckpt := self.get_checkpoint(uri):
      return ckpt.version
    raise ValueError(f"no checkpoint at {uri}")

  def set_checkpoint_ttl(self, uri: str, ttl: int | None) -> bool:
    parse_uri(uri)
    if (c := self.checkpoints.get(uri)) is None:
      return False
    with self.db:
      self.checkpoints[uri] = replace(c, expires_at=time.time() + ttl if ttl else None)
    return True

  def delete_checkpoint(self, uri: str) -> bool:
    parse_uri(uri)
    with self.db:
      return self.checkpoints.pop(uri, None) is not None

  def live_versions(self) -> list[str]:
    v = {c.version for c in self.checkpoints.values() if c.version}
    v.update(r.snapshot for r in self.runs.values() if r.live and r.snapshot)
    v.update(self.trains.versions())
    return sorted({x.partition("@")[0] for x in v})

  # ***** placement, liveness, and maintenance *****

  def state(self) -> scheduler.State:
    jobs = {}
    for rid, r in sorted(self.runs.items()):
      if r.live:
        u = r.run
        qlen = (self.samples if u.role == Role.SAMPLER else self.trains).backlog(rid)
        jobs[rid] = scheduler.Job(u.spec, u.train_run, u.account or ("" if u.train_run else u.session), r.owner, qlen)
    jobs = {r: j for r, j in jobs.items() if not j.spec.train_run or j.spec.train_run in jobs}
    home = {r: p for r, p in self.plan_state.home.items() if r in jobs}
    return replace(self.plan_state, nodes=self.nodes.active_gpus(), home=home, jobs=jobs, service=self.service)

  def live_nodes(self) -> dict[str, int]:
    self.expire_nodes()
    return {nid: n.status.gpus for nid, n in self.nodes.items()}

  def node_status(self) -> dict[str, NodeStatus]:
    self.expire_nodes()
    return {nid: n.status for nid, n in self.nodes.items()}

  def expire_node(self, node_id: str, boot_id: str) -> None:
    expired_boots = self.nodes.remove(node_id, boot_id)
    self.record_event(Ev.NODE_LOST, {"node": node_id})
    for req_id, row in self.trains.held_on((node_id, boot_id)):
      err = RuntimeError(f"its trainer died, or lost ownership, during {row.cmd.op}")
      self.finish_train(req_id, row.owner, Reply.error(err), dead=True)
    with self.db:
      self.meta["expired_boots"] = expired_boots
    self.reclaim(lambda o: o[1:] == (node_id, boot_id))
    self.notify_work()

  def expire_nodes(self) -> None:
    for nid, bid in self.nodes.expired([*self.trains.owners(), *(r.owner for r in self.runs.values())]):
      self.expire_node(nid, bid)

  def reconcile(self) -> list[tuple[Ev, dict]]:
    self.plan_due = time.monotonic() + TICK_MS / 1000
    old = scheduler.to_plan(self.plan_state)
    self.plan_state, changes = scheduler.step(self.state())
    if (plan := scheduler.to_plan(self.plan_state)) != old:
      with self.db:
        self.meta["plan"] = plan
        self.flush()
      for kind, body in changes:
        self.record_event(kind, body)
      if any(plan[k] != old.get(k) for k in ("procs", "home")):
        self.notify_work()
    return changes

  def maintain(self) -> list[tuple[Ev, dict]]:
    self.expire_nodes()
    now, mono = time.time(), time.monotonic()
    for sid in [s for s, v in list(self.sessions.items()) if v["active"] and v["seen"] <= now - SESSION_GRACE]:
      self.end_session(sid)
    run_gc = mono >= self.gc_due
    with self.db:
      self.flush()
      for u in [u for u, c in self.checkpoints.items() if c.expires_at is not None and c.expires_at <= now]:
        del self.checkpoints[u]
      if run_gc:
        self.gc_due, cut = mono + GC_S, now - SESSION_TTL
        self.samples.gc(now)
        self.trains.gc(now)
        for s in [s for s, v in self.sessions.items() if not v["active"] and v["seen"] <= cut]:
          del self.sessions[s]
        for r in [
          r for r, row in self.runs.items() if 0 < row.deleted_at <= cut and row.run.session not in self.sessions
        ]:
          del self.runs[r]
    if run_gc:
      self.db.vacuum()
    return self.reconcile()

  def read_events(self, after: str = "0-0", count: int = 10_000) -> list[Event]:
    cut = entry_sort_key(after)
    return [ev for ev in self.events if entry_sort_key(ev.id) > cut][:count]

  # ***** the 4 fenced node RPCs: heartbeat, take, finish, detach *****

  def assignments(self, node: str, boot_id: str) -> list[Assignment]:
    out, ps = [], self.plan_state
    for wid, proc in sorted(ps.procs.items()):
      if proc.node != node:
        continue
      tok = (wid, node, boot_id)
      if proc.spec.role == Role.TRAINER:
        runs, heads = tuple(ps.runs(wid)), self.head_train(wid, tok)
        held = [h[1].cmd.run] if (h := self.trains.held_by(tok)) else []
        lruns = held or [c.run for c in heads] or [r for r in runs if r in ps.jobs]
        closing, lag = tuple(c.run for c in heads if c.op == Op.OPTIM_STEP), max(map(ps.ready_lag, lruns), default=0.0)
        out.append(Assignment(wid, proc.spec, runs, (), bool(held or heads), closing, lag))
      else:
        pairs = self.ready_samples(proc.spec)
        rids, vers = tuple(c.run for c, _ in pairs), tuple(sorted({v for _, v in pairs if v}))
        ready = bool(self.samples.held(tok)) or any(v is None for _, v in pairs)
        out.append(Assignment(wid, proc.spec, rids, vers, ready, (), max(map(ps.ready_lag, rids), default=0.0)))
    return out

  async def heartbeat(self, node: str, boot_id: str, st: NodeStatus, wait_ms: int = 0):
    is_new, changed = self.nodes.record(node, boot_id, self.fence(node, boot_id), st, time.monotonic())
    if is_new:
      self.record_event(Ev.NODE_UP, {"node": node, "gpus": st.gpus, "memory": 0})
    elif changed:
      self.plan_due = 0.0
    if st.live_workers is not None:
      keep = set(st.live_workers) | {c.owner[0] for _, c in self.trains.held_on((node, boot_id))}
      self.reclaim(lambda o: o[1:] == (node, boot_id) and o[0] not in keep)

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
    if (proc := ps.procs.get(wid)) is None or proc.node != node or (cur and cur.status.draining):
      return []
    tok, svc, hold = (wid, node, boot), service or {}, set(holding)
    if proc.spec.role == Role.TRAINER:
      if held := self.trains.held_by(tok):
        return [] if held[0] in hold else [self.trains.work(held[0], self.runs[held[1].cmd.run].run)]
      while slots > 0 and (heads := self.head_train(wid, tok)):
        rid = rank_runs(heads, ps.ready_lag, svc)[0]
        (k, item), row = self.trains.head(rid), self.runs[rid]
        op = item.cmd.op
        ver = uuid.uuid4().hex if op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER) else None
        if op == Op.LOAD_WEIGHTS:
          try:
            ver = self.checkpoint_version(item.uri or "")
          except ValueError as err:
            self.finish_train(k, None, Reply.error(err))
            continue
        snapshot = row.snapshot if row.status == RunStatus.SAVED and op != Op.LOAD_WEIGHTS else None
        row.owner = tok
        with self.db:
          self.trains.claim(k, tok, now, snapshot, ver)
          self.runs.save(rid)
        return [self.trains.work(k, row.run)]
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
    cap = (max(0, slots - len(out)), types.MAX_LORAS if proc.spec.lora else 1)
    for c in admit_samples(heads, self.samples.queued, ps.account, ps.ready_lag, svc, vm.get, *act, *cap):
      self.samples.claim(c, tok, now)
      out.append(Work(c, version=vm[c.run]))
    return out

  def finish_train(
    self, req_id: str, owner: Owner | None, reply: Reply, charge=None, *, mutated=False, checkpoint=None, dead=False
  ) -> bool:
    if (cmd := self.trains.pop_claimed(req_id, owner)) is None:
      return False
    row, deleting = self.runs.get(cmd.run), cmd.op == Op.DELETE_MODEL and reply.ok
    if row:
      self.bill(row, cmd, reply, charge)
    with self.db:
      self.trains.write_reply(req_id, reply, REPLY_TTL)
      self.flush()
      if row:
        if checkpoint and reply.ok:
          self.checkpoints[checkpoint.uri] = checkpoint
        if deleting:
          self.remove_run(cmd.run, owned_ckpts=True)
        else:
          if mutated:
            row.snapshot, row.status = None, RunStatus.LIVE
          if dead and owner:
            self.clear_owners(lambda o: o == owner)
          elif dead or row.status in (RunStatus.FRESH, RunStatus.LOST):
            row.owner = None
          self.runs.save(cmd.run)
    if deleting:
      self.record_event(Ev.RUN_DELETED, {"run": cmd.run})
    self.notify(req_id)
    return True

  def finish_sample(self, req_id: str, owner: Owner | None, reply: Reply, charge=None) -> bool:
    if (cmd := self.samples.complete(req_id, owner, reply, REPLY_TTL)) is None:
      return False
    if row := self.runs.get(cmd.run):
      self.bill(row, cmd, reply, charge)
    self.notify(req_id)
    return True

  def finish(
    self,
    node: str,
    boot: str,
    wid: str,
    rid: str,
    rep: Reply,
    ch: Charge | None = None,
    *,
    mutated: bool = False,
    checkpoint: Checkpoint | None = None,
    dead: bool = False,
  ) -> bool:
    self.fence(node, boot)
    tok = (wid, node, boot)
    if self.is_sampler(rid.partition("/")[0], rid):
      ok = self.finish_sample(rid, tok, rep, ch)
    else:
      ok = self.finish_train(rid, tok, rep, ch, mutated=mutated, checkpoint=checkpoint, dead=dead)
    if ok:
      self.notify_work()
    return ok

  def detach(self, node: str, boot_id: str, wid: str, rid: str, snapshot: str | None = None) -> bool:
    self.fence(node, boot_id)
    if (row := self.runs.get(rid)) is None or row.owner != (wid, node, boot_id):
      return False
    with self.db:
      row.owner = None
      if snapshot:
        row.snapshot, row.status = snapshot, RunStatus.SAVED
      elif row.status == RunStatus.LIVE:
        row.status = RunStatus.LOST
      self.runs.save(rid)
    self.notify_work()
    return True
