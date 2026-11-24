"""Single-process controller: relational SQLite state, unified queues, and direct fleet GPU paging."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
import uuid
import weakref
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field, replace

from tinyrl import types as T
from tinyrl.control import policy
from tinyrl.control.accounting import Charge
from tinyrl.control.sql import DB, DEC, ENC
from tinyrl.helpers import LEASE_MS, SWEEP_S, getenv

LIVE_SECONDS, SESSION_GRACE, GC_S, TICK_MS = 30, getenv("SESSION_GRACE", 60), 10, getenv("TICK_MS", 100)
REPLY_TTL, READ_REPLY_TTL, SESSION_TTL = getenv("REPLY_TTL", 3600), 60, getenv("SESSION_TTL", 7 * 24 * 3600)
SAMPLE_QUEUE_BYTES = getenv("SAMPLE_QUEUE_BYTES", 256 * 2**20)
SAMPLE_REPLY_BYTES = getenv("SAMPLE_REPLY_BYTES", 256 * 2**20)
STABILITY_TTL, STABILITY_BYTES = getenv("STABILITY_TTL", 24 * 3600), getenv("STABILITY_BYTES", 512 * 2**20)
COUNTERS = ("steps", "failed", "cmds", "tokens", "busy_ms", "wait_ms")
K, Op, Role, RS = T.TaskKind, T.Op, T.Role, T.RunStatus


def _tid() -> str:
  return uuid.uuid4().hex[:12]


@dataclass
class QueueItem:
  cmd: T.Cmd
  uri: str | None = None
  version: str | None = None
  stab_vers: tuple[str, ...] = ()
  task_id: str | None = None
  owner: T.Owner | None = None
  since: float = 0.0
  attempt: int = 0


@dataclass
class RunState:
  run: T.Run
  created: float
  status: T.RunStatus = RS.FRESH
  owner: T.Owner | None = None
  snapshot: str | None = None
  clients: int = 0
  deleted_at: float = 0.0
  fair_ms: float | None = None
  counters: dict[str, int] = field(default_factory=lambda: dict.fromkeys(COUNTERS, 0))
  queue: dict[str, QueueItem] = field(default_factory=dict)

  @property
  def live(self) -> bool:
    return self.status != RS.DELETED


@dataclass
class LiveWorker:
  id: str
  spec: T.WorkerSpec
  gpus: tuple[int, ...] = ()
  bound: tuple[int, ...] = ()
  loaded: dict[str, bool] = field(default_factory=dict)
  pid: int | None = None
  started: bool = False
  resumed_at: float = 0.0
  hold_until: float = 0.0
  idle_until: float = 0.0
  park_s: float = 0.0
  gpu_ms: float = 0.0
  stopping: bool = False
  failed_saves: set[str] = field(default_factory=set)
  retry_save_at: float = 0.0


@dataclass
class NodeState:
  id: str
  boot_id: str
  gpus: int
  gpu_mib: int = 0
  draining: bool = False
  until: float = 0.0
  lag: float = 0.0
  staged: set[str] = field(default_factory=set)
  stage: tuple[str, ...] = ()
  sent_stage: tuple[str, ...] = ()
  workers: dict[str, LiveWorker] = field(default_factory=dict)
  tasks: dict[str, T.Task] = field(default_factory=dict)
  new_tasks: list[T.Task] = field(default_factory=list)
  cancelled: list[str] = field(default_factory=list)
  push_ev: asyncio.Event = field(default_factory=asyncio.Event)

  def worker_tasks(self, wid: str) -> list[T.Task]:
    return [t for t in self.tasks.values() if t.worker == wid]

  def wake(self) -> None:
    if self.new_tasks or self.cancelled or self.stage != self.sent_stage:
      self.push_ev.set()

  def drain_push(self, ack: tuple[str, ...] = (), sweep: tuple[str, ...] = ()) -> T.ControlMsg:
    st = self.stage if self.stage != self.sent_stage else ()
    self.sent_stage = self.stage
    out = T.ControlMsg(ack=ack, tasks=tuple(self.new_tasks), cancel=tuple(self.cancelled), stage=st, sweep=sweep)
    self.new_tasks.clear()
    self.cancelled.clear()
    self.push_ev.clear()
    return out


class Controller:
  def __init__(self, path: str | None = None):
    self.db = DB(getenv("TINYRL_DB", "/tmp/tinyrl.db") if path is None else path)
    self.runs, self.nodes = {}, {}
    self.service, self.lag, self.seen, self.weights = {}, {}, {}, {}
    grace = time.monotonic() + LEASE_MS / 1000
    for rid, sp, cr, st, ow, sn, cl, dl, ct in self.db.execute("SELECT * FROM runs"):
      owner = tuple(DEC(ow)) if ow else None
      self.runs[rid] = rs = RunState(DEC(sp, type=T.Run), cr, RS(st), owner, sn, cl, dl, None, DEC(ct))
      if rs.live and owner:
        self.nodes.setdefault(owner[1], NodeState(owner[1], owner[2], 0, until=grace))
    for a, sv, lg, sn, wt in self.db.execute("SELECT * FROM accounts"):
      self.service[a], self.lag[a], self.seen[a], self.weights[a] = sv, lg, sn, wt
    self.seq = int(r[0]) if (r := self.db.one("SELECT v FROM meta WHERE k='seq'")) else 0
    q = "SELECT id, run_id, seq, op, cost, uri, stab, claim FROM commands ORDER BY seq"
    for cid, rid, seq, op, cost, uri, stab, claim in self.db.execute(q):
      self.seq = max(self.seq, seq)
      if not (rs := self.runs.get(rid)) or not rs.live:
        continue
      sv = tuple(sorted({x for x, _, _ in DEC(stab).values()})) if stab else ()
      it = QueueItem(T.Cmd(rid, cid.partition("/")[2], Op(op), b"", cost), uri, stab_vers=sv)
      if claim:
        nid, bid, cat, raw = DEC(claim)
        t = DEC(raw, type=T.Task)
        ns = self.nodes.setdefault(nid, NodeState(nid, bid, 0, until=grace))
        ns.tasks[t.id] = t
        it.task_id, it.owner, it.since = t.id, (t.worker, nid, bid), cat
        w = ns.workers.setdefault(t.worker, LiveWorker(t.worker, t.spec, t.gpus, started=True))
        w.bound = w.bound or (() if t.spec.movable else t.gpus)
      rs.queue[cid] = it
    self.made = self.sample_q_bytes = self.sample_r_bytes = 0
    self.gc_due = self.sweep_due = self.lag_due = 0.0
    self.next_wake, self.accounts_dirty = None, False
    self.sample_replies, self.sample_seen = OrderedDict(), {}
    self.events, self.waiters = deque(maxlen=100_000), weakref.WeakValueDictionary()

  def _save_run(self, rs: RunState) -> None:
    ow, ct = ENC(rs.owner) if rs.owner else None, ENC(rs.counters)
    vals = (rs.run.id, ENC(rs.run), rs.created, str(rs.status), ow, rs.snapshot, rs.clients, rs.deleted_at, ct)
    self.db.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?)", vals)

  def _save_ckpt(self, c: T.Checkpoint) -> None:
    vals = (c.uri, c.version, c.size, c.time, c.expires_at, int(c.owned), ENC(c.meta))
    self.db.execute("INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?,?,?)", vals)

  def _flush_sql(self, svc: dict[str, float] | None = None) -> None:
    s = self.service if svc is None else svc
    for a in s.keys() | self.lag.keys() | self.seen.keys() | self.weights.keys():
      vals = (a, s.get(a, 0.0), self.lag.get(a, 0.0), self.seen.get(a, 0.0), self.weights.get(a, 1.0))
      self.db.execute("INSERT OR REPLACE INTO accounts VALUES (?,?,?,?,?)", vals)
    self.db.execute("INSERT OR REPLACE INTO meta VALUES ('seq', ?)", (self.seq,))

  async def aclose(self) -> None:
    if self.db is not None:
      with self.db:
        self._flush_sql()
      self.db.close()
      self.db = None

  def _notify(self, req_id: str) -> None:
    if ev := self.waiters.pop(req_id, None):
      ev.set()

  def record_event(self, kind: T.Ev, body: dict) -> T.Event:
    self.seq += 1
    self.events.append(ev := T.Event(f"{int((now := time.time()) * 1000)}-{self.seq}", now, str(T.Ev(kind)), body))
    return ev

  def read_events(self, after: str = "0-0", count: int = 10_000) -> list[T.Event]:
    return [ev for ev in self.events if T.entry_sort_key(ev.id) > T.entry_sort_key(after)][:count]

  def is_sampler(self, rid: str, req_id: str = "") -> bool:
    return self.runs[rid].run.role == Role.SAMPLER if rid in self.runs else req_id.startswith("samp-")

  def account(self, rid: str) -> str:
    if not (rs := self.runs.get(rid)) or not rs.live:
      return rid
    if rs.run.account:
      return rs.run.account
    tr = self.runs.get(rs.run.train_run or "")
    return self.account(rs.run.train_run) if tr and tr.live else (rs.run.session or rid)

  def _live_spec(self, rs: RunState) -> bool:
    return rs.live and (not rs.run.spec.train_run or ((tr := self.runs.get(rs.run.train_run)) is not None and tr.live))

  # ***** sessions, runs, commands, checkpoints *****

  def heartbeat_session(self, sid: str, metadata: dict | None = None, *, finish: bool = False) -> None:
    now, r = time.time(), self.db.one("SELECT created, active, meta FROM sessions WHERE id=?", (sid,))
    if finish and (not r or not r[1]):
      return
    meta = metadata if (not finish and metadata) else (DEC(r[2]) if r else {})
    with self.db:
      self.db.execute(
        "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?)",
        (sid, r[0] if r else now, 0.0 if finish else now, 1, ENC(meta)),
      )

  def get_session(self, sid: str) -> dict | None:
    if not (r := self.db.one("SELECT created, seen, meta FROM sessions WHERE id=?", (sid,))):
      return None
    runs = sorted(k for k, rs in self.runs.items() if rs.run.session == sid)
    base = {"session_id": sid, "created": r[0], "seen": r[1], "live": time.time() - r[1] < LIVE_SECONDS}
    return base | {"runs": runs, "metadata": DEC(r[2])}

  def end_session(self, sid: str) -> None:
    live = sorted(r for r, rs in self.runs.items() if rs.live and rs.run.session == sid)
    tr = [bool(self.submit_command(r, Op.DELETE_MODEL, b"{}")) for r in live if not self.is_sampler(r)]
    for r in [r for r in live if self.is_sampler(r)]:
      self.delete_run(r)
    with self.db:
      sql, args = (
        ("UPDATE sessions SET seen=? WHERE id=?", (time.time(), sid))
        if any(tr)
        else ("UPDATE sessions SET active=0 WHERE id=?", (sid,))
      )
      self.db.execute(sql, args)

  def create_run(self, run: T.Run) -> None:
    with self.db:
      self._save_run(rs := RunState(run, time.time()))
    self.runs[run.id] = rs
    info = {k: getattr(run, k) for k in ("role", "base", "rank", "exclusive", "train_run", "gpus")}
    self.record_event(T.Ev.RUN_CREATED, {"run": run.id} | info)
    self.schedule()

  def join_sampling_session(self, rid: str) -> int:
    if not (rs := self.runs.get(rid)):
      return 1
    with self.db:
      self._save_run(replace(rs, clients=rs.clients + 1))
    rs.clients += 1
    return rs.clients

  def get_run(self, rid: str) -> dict | None:
    if not (rs := self.runs.get(rid)):
      return None
    extra = ({"owner": list(rs.owner)} if rs.owner else {}) | ({"snapshot": rs.snapshot} if rs.snapshot else {})
    return {**asdict(rs.run), "created": rs.created, "status": str(rs.status), **extra, **rs.counters}

  def delete_run(self, rid: str) -> None:
    if rs := self.runs.get(rid):
      for cid, it in list(rs.queue.items()):
        n = self.nodes.get(it.owner[1]) if it.owner and it.task_id else None
        n = n if n and n.boot_id == it.owner[2] else None
        if rs.run.role == Role.SAMPLER:
          self.sample_q_bytes -= len(it.cmd.body)
          if n:
            n.tasks.pop(it.task_id, None)
            n.cancelled.append(it.task_id)
            n.wake()
        elif it.owner:
          self._finish_train(cid, it.owner, T.Reply.error(RuntimeError(f"run {rid} was deleted")))
          if n:
            n.tasks.pop(it.task_id, None)
      self.sample_seen = {k: v for k, v in self.sample_seen.items() if k[0] != rid}
      now, dropped = time.time(), list(rs.queue)
      with self.db:
        self._save_run(replace(rs, status=RS.DELETED, owner=None, snapshot=None, deleted_at=now))
        self.db.execute("DELETE FROM commands WHERE run_id=?", (rid,))
      rs.status, rs.owner, rs.snapshot, rs.deleted_at = RS.DELETED, None, None, now
      rs.queue.clear()
      for cid in dropped:
        self._notify(cid)
    self.record_event(T.Ev.RUN_DELETED, {"run": rid})
    self.schedule()

  def seen_request(self, rid: str, seq_id: int | None) -> str | None:
    if seq_id is None:
      return None
    now = time.time()
    if self.is_sampler(rid):
      return h[0] if (h := self.sample_seen.get((rid, seq_id))) and h[1] > now else None
    r = self.db.one("SELECT req_id FROM dedup WHERE run_id=? AND seq_id=? AND expires>?", (rid, seq_id, now))
    return r[0] if r else None

  def submit_command(
    self, rid: str, op: T.Op, body: bytes, seq_id=None, cost=1, uri=None, stability_refs=None
  ) -> str | None:
    if existing := self.seen_request(rid, seq_id):
      return existing
    if not (rs := self.runs.get(rid)) or not rs.live:
      return None
    if not rs.queue and rs.fair_ms is None:
      acct, sp = self.account(rid), rs.run.spec
      peers = [
        o.fair_ms
        for k, o in self.runs.items()
        if o.live and o.fair_ms is not None and o.run.spec == sp and self.account(k) == acct
      ]
      rs.fair_ms = min(peers, default=0.0)
    self.seq += 1
    cmd, now = T.Cmd(rid, f"{time.time_ns() // 1_000_000}-{self.seq}", Op(op), body, cost), time.time()
    req_id = cmd.request_id
    if rs.run.role == Role.SAMPLER:
      if self.sample_q_bytes + len(body) > SAMPLE_QUEUE_BYTES:
        raise T.QueueFullError("sampling queue memory limit reached")
      ck = self.get_checkpoint(mp) if (mp := rs.run.model_path) else None
      rs.queue[req_id] = QueueItem(cmd, version=ck.version if ck else None)
      self.sample_q_bytes += len(body)
      if seq_id is not None:
        self.sample_seen[rid, seq_id] = (req_id, now + SESSION_TTL)
      if mp and ck is None:
        self._finish_sample(req_id, None, T.Reply.error(ValueError(f"no checkpoint at {mp}")), None, 0)
    else:
      with self.db:
        stab = self.db.read_stability(stability_refs, now) if stability_refs else None
        vals = (req_id, rid, self.seq, str(cmd.op), cost, uri, body, ENC(stab) if stab else None)
        self.db.execute("INSERT INTO commands VALUES (?,?,?,?,?,?,?,?,NULL)", vals)
        if seq_id is not None:
          self.db.execute("INSERT OR REPLACE INTO dedup VALUES (?,?,?,?)", (rid, seq_id, req_id, now + SESSION_TTL))
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('seq', ?)", (self.seq,))
      sv = tuple(sorted({r for r, _, _ in stab.values()})) if stab else ()
      rs.queue[req_id] = QueueItem(replace(cmd, body=b""), uri, stab_vers=sv)
    self.schedule()
    return req_id

  def _read_reply(self, req_id: str) -> T.Reply | None:
    now, rttl, rid = time.time(), min(REPLY_TTL, READ_REPLY_TTL), req_id.partition("/")[0]
    if self.is_sampler(rid, req_id):
      if (hit := self.sample_replies.get(req_id)) and hit[1] > now:
        self.sample_replies.move_to_end(req_id)
        self.sample_replies[req_id] = (hit[0], min(hit[1], now + rttl), True)
        return hit[0]
      if not ((rs := self.runs.get(rid)) and req_id in rs.queue):
        raise T.RequestGoneError(f"sampling request gone: {req_id}")
      return None
    r = self.db.one("SELECT ok, content_type, expires, body FROM replies WHERE id=?", (req_id,))
    if not r or r[2] <= now:
      return None
    with self.db:
      self.db.execute("UPDATE replies SET expires=? WHERE id=?", (min(r[2], now + rttl), req_id))
    return T.Reply(bool(r[0]), r[1], bytes(r[3]))

  async def wait_for_reply(self, req_id: str, timeout: float) -> T.Reply | None:
    end = time.monotonic() + max(0.0, timeout)
    while (rep := self._read_reply(req_id)) is None and (rem := end - time.monotonic()) > 0:
      with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(self.waiters.setdefault(req_id, asyncio.Event()).wait(), rem)
    return rep

  def queued_and_in_flight(self, rid: str) -> dict:
    if not (rs := self.runs.get(rid)) or not rs.live:
      return {"live": False}
    now = time.monotonic() if rs.run.role == Role.SAMPLER else time.time()
    fly = [
      {"request_id": k, "trainer": i.owner[0], "deliveries": 1, "seconds": max(0.0, now - (i.since or now))}
      for k, i in rs.queue.items()
      if i.owner
    ]
    return {"live": True, "queued": len(rs.queue) - len(fly), "in_flight": fly}

  def hydrate_cmd(self, req_id: str) -> T.Cmd:
    r = self.db.one("SELECT body, stab FROM commands WHERE id=?", (req_id,))
    base = self.runs[req_id.partition("/")[0]].queue[req_id].cmd
    return replace(base, body=bytes(r[0]) if r else b"", stability=DEC(r[1]) if r and r[1] else None)

  def put_checkpoint(self, ckpt: T.Checkpoint) -> None:
    with self.db:
      self._save_ckpt(ckpt)

  def list_checkpoints(self, run: str | None = None) -> list[T.Checkpoint]:
    rows = self.db.execute("SELECT * FROM checkpoints WHERE expires_at IS NULL OR expires_at>?", (time.time(),))
    out = [T.Checkpoint(*r[:5], bool(r[5]), DEC(r[6])) for r in rows if run is None or T.parse_uri(r[0])[0] == run]
    return sorted(out, key=lambda c: c.time, reverse=run is None)

  def get_checkpoint(self, uri: str) -> T.Checkpoint | None:
    T.parse_uri(uri)
    r = self.db.one(
      "SELECT * FROM checkpoints WHERE uri=? AND (expires_at IS NULL OR expires_at>?)", (uri, time.time())
    )
    return T.Checkpoint(*r[:5], bool(r[5]), DEC(r[6])) if r else None

  def checkpoint_version(self, uri: str) -> str:
    if ckpt := self.get_checkpoint(uri):
      return ckpt.version
    raise ValueError(f"no checkpoint at {uri}")

  def set_checkpoint_ttl(self, uri: str, ttl: int | None) -> bool:
    if not (c := self.get_checkpoint(uri)):
      return False
    self.put_checkpoint(replace(c, expires_at=time.time() + ttl if ttl else None))
    return True

  def delete_checkpoint(self, uri: str) -> bool:
    if not self.get_checkpoint(uri):
      return False
    with self.db:
      self.db.execute("DELETE FROM checkpoints WHERE uri=?", (uri,))
    return True

  def live_versions(self) -> list[str]:
    v = {c.version for c in self.list_checkpoints() if c.version}
    v |= {r[0] for r in self.db.execute("SELECT DISTINCT ref FROM stability WHERE expires>?", (time.time(),))}
    v |= {rs.snapshot for rs in self.runs.values() if rs.live and rs.snapshot}
    v |= {x for rs in self.runs.values() for it in rs.queue.values() for x in (it.version, *it.stab_vers) if x}
    v |= {x for n in self.nodes.values() for t in n.tasks.values() for x in (t.version, t.snapshot) if x}
    return sorted({x.partition("@")[0] for x in v})

  # ***** accounting & completion *****

  def _bill(
    self, rs: RunState, cmd: T.Cmd, reply: T.Reply, charge: Charge | None, ctrs: dict, svc: dict
  ) -> float | None:
    ctrs["failed"] += int(not reply.ok)
    ctrs["steps"] += int(reply.ok and cmd.op == Op.OPTIM_STEP)
    fair = rs.fair_ms
    if charge:
      fair = fair + charge.gpu_ms if fair is not None else None
      ctrs["cmds"], ctrs["busy_ms"] = ctrs["cmds"] + 1, ctrs["busy_ms"] + charge.busy_ms
      ctrs["wait_ms"] += max(0, charge.started_ms - T.entry_sort_key(cmd.entry_id)[0])
      ctrs["tokens"] += cmd.cost if reply.ok else 0
      if charge.gpu_ms and (acct := self.account(cmd.run)):
        svc[acct] = svc.get(acct, 0.0) + charge.gpu_ms
    return fair

  def _finish_train(
    self, req_id: str, owner, reply: T.Reply, charge=None, *, mutated=False, checkpoint=None, dead=False
  ):
    if not (rs := self.runs.get(req_id.partition("/")[0])) or not (it := rs.queue.get(req_id)) or it.owner != owner:
      return False
    cmd, now, deleting = it.cmd, time.time(), it.cmd.op == Op.DELETE_MODEL and reply.ok
    ctrs, svc = dict(rs.counters), dict(self.service)
    fair = self._bill(rs, cmd, reply, charge, ctrs, svc)
    nxt, extra = replace(rs, counters=ctrs, fair_ms=None if len(rs.queue) <= 1 else fair), []
    if deleting:
      nxt.status, nxt.owner, nxt.snapshot, nxt.deleted_at = RS.DELETED, None, None, now
    else:
      if mutated:
        nxt.snapshot, nxt.status = None, RS.LIVE
      if dead and owner:
        nxt.owner, nxt.status = None, RS.LOST if nxt.status == RS.LIVE else nxt.status
        peers = [r for r in self.runs.values() if r.run.id != cmd.run and r.owner == owner]
        extra = [replace(r, owner=None, status=RS.LOST if r.status == RS.LIVE else r.status) for r in peers]
      elif dead or nxt.status in (RS.FRESH, RS.LOST):
        nxt.owner = None
    with self.db:
      self.db.execute(
        f"DELETE FROM commands WHERE {'run_id' if deleting else 'id'}=?", (cmd.run if deleting else req_id,)
      )
      vals = (req_id, int(reply.ok), reply.content_type, now + REPLY_TTL, reply.body)
      self.db.execute("INSERT OR REPLACE INTO replies VALUES (?,?,?,?,?)", vals)
      self._flush_sql(svc)
      if checkpoint and reply.ok:
        self._save_ckpt(checkpoint)
      if deleting:
        self.db.execute("DELETE FROM checkpoints WHERE owned=1 AND uri LIKE ?", (f"tinker://{cmd.run}/%",))
      for r2 in (*extra, nxt):
        self._save_run(r2)
    self.service = svc
    for r2 in extra:
      self.runs[r2.run.id].owner, self.runs[r2.run.id].status = r2.owner, r2.status
    rs.status, rs.owner, rs.snapshot, rs.deleted_at = nxt.status, nxt.owner, nxt.snapshot, nxt.deleted_at
    rs.counters, rs.fair_ms = ctrs, nxt.fair_ms
    for k in list(rs.queue) if deleting else [req_id]:
      rs.queue.pop(k, None)
      self._notify(k)
    if deleting:
      self.record_event(T.Ev.RUN_DELETED, {"run": cmd.run})
    return True

  def _finish_sample(
    self, req_id: str, owner: T.Owner | None, reply: T.Reply, charge: Charge | None, attempt: int
  ) -> bool:
    rs = self.runs.get(req_id.partition("/")[0])
    if not rs or (it := rs.queue.get(req_id)) is None or (it.owner, it.attempt) != (owner, attempt):
      return False
    if reply.ok and reply.stability:
      with self.db:
        self.db.record_stability(reply.stability, time.time() + STABILITY_TTL, STABILITY_BYTES)
    self.sample_q_bytes -= len(rs.queue.pop(req_id).cmd.body)
    self.sample_replies[req_id] = (clean := replace(reply, stability=None), time.time() + REPLY_TTL, False)
    self.sample_r_bytes += len(clean.body)
    while self.sample_r_bytes > SAMPLE_REPLY_BYTES and len(self.sample_replies) > 1:
      vic = next((k for k, (r, _, rd) in self.sample_replies.items() if rd and r.body), None)
      vic = vic or next((k for k, (r, _, _) in self.sample_replies.items() if k != req_id and r.body), None)
      if vic is None:
        break
      self.sample_r_bytes -= len(self.sample_replies.pop(vic)[0].body)
    fair = self._bill(rs, it.cmd, clean, charge, rs.counters, self.service)
    rs.fair_ms = fair if rs.queue else None
    self.accounts_dirty |= bool(charge and charge.gpu_ms)
    self._notify(req_id)
    return True

  def _reclaim_samples(self, pred) -> None:
    for rs in self.runs.values():
      for it in rs.queue.values() if rs.run.role == Role.SAMPLER else ():
        if it.owner and pred(it):
          if it.task_id and (n := self.nodes.get(it.owner[1])) and n.boot_id == it.owner[2]:
            n.tasks.pop(it.task_id, None)
            n.cancelled.append(it.task_id)
          it.attempt, it.owner, it.task_id, it.since = it.attempt + 1, None, None, 0.0

  def _clear_owners(self, pred, *, lose: bool = True) -> None:
    ch = [
      replace(r, owner=None, status=RS.LOST if lose and r.status == RS.LIVE else r.status)
      for r in self.runs.values()
      if r.owner and pred(r)
    ]
    if ch:
      with self.db:
        for nxt in ch:
          self._save_run(nxt)
      for nxt in ch:
        self.runs[nxt.run.id].owner, self.runs[nxt.run.id].status = nxt.owner, nxt.status

  # ***** node liveness & protocol *****

  def live_nodes(self) -> dict[str, int]:
    self.expire_nodes()
    return {nid: n.gpus for nid, n in self.nodes.items() if n.gpus}

  def expire_node(self, nid: str, bid: str) -> None:
    if (cur := self.nodes.get(nid)) and cur.boot_id == bid:
      del self.nodes[nid]
    self.record_event(T.Ev.NODE_LOST, {"node": nid})
    for rs in list(self.runs.values()):
      for cid, it in list(rs.queue.items()) if rs.run.role == Role.TRAINER else ():
        if it.owner and it.owner[1:] == (nid, bid):
          err = T.Reply.error(RuntimeError(f"its trainer died, or lost ownership, during {it.cmd.op}"))
          self._finish_train(cid, it.owner, err, dead=True)
    with self.db:
      self.db.add_fenced(bid)
    self._reclaim_samples(lambda it: it.owner[1:] == (nid, bid))
    self._clear_owners(lambda rs: rs.owner[1:] == (nid, bid))

  def expire_nodes(self) -> bool:
    stale = [(nid, n.boot_id) for nid, n in list(self.nodes.items()) if n.until <= time.monotonic()]
    for nid, bid in stale:
      self.expire_node(nid, bid)
    return bool(stale)

  def _drop_worker(self, node: NodeState, wid: str) -> None:
    w, tok = node.workers.pop(wid, None), (wid, node.id, node.boot_id)
    self._reclaim_samples(lambda it: it.owner == tok)
    self._clear_owners(lambda rs: rs.owner == tok)
    if w and w.started:
      self.record_event(T.Ev.UNPLACED, {"proc": wid, "node": node.id})

  def _apply_done(self, node: NodeState, d: T.Done, now: float) -> None:
    if not (task := node.tasks.get(d.id)):
      return
    w, tok, dead = node.workers.get(task.worker), (task.worker, node.id, node.boot_id), d.dead or not d.ok
    if w and d.pid is not None:
      w.pid, w.started = d.pid, True
    if task.kind == K.SAVE and not dead:
      if (rs := self.runs.get(task.run_id)) and rs.owner == tok:
        st = RS.SAVED if d.saved else (RS.LOST if rs.status == RS.LIVE else rs.status)
        with self.db:
          self._save_run(nxt := replace(rs, owner=None, snapshot=d.saved or rs.snapshot, status=st))
        rs.owner, rs.snapshot, rs.status = nxt.owner, nxt.snapshot, nxt.status
      node.tasks.pop(d.id, None)
      if w:
        w.failed_saves.discard(task.run_id)
        if not w.stopping and w.started and task.run_id in w.loaded:
          self._issue(node, T.Task(_tid(), K.DROP, w.id, w.spec, run_id=task.run_id), now)
      return
    elif task.kind == K.SAVE:
      if d.dead:
        self._drop_worker(node, task.worker)
      elif w:
        w.failed_saves.add(task.run_id)
        w.retry_save_at = now + 0.05
    elif task.kind in (K.TRAIN, K.SAMPLE):
      eff, rep = d.effect or T.Effect(), d.reply or T.Reply.error(RuntimeError(f"{task.kind} failed"))
      if task.kind == K.TRAIN:
        self._finish_train(
          task.cmd.request_id, tok, rep, d.charge, mutated=eff.mutated, checkpoint=eff.checkpoint, dead=dead
        )
      elif dead and d.reply is None:
        self._reclaim_samples(lambda it: it.cmd.request_id == task.cmd.request_id and it.owner == tok)
      else:
        self._finish_sample(task.cmd.request_id, tok, rep, d.charge, task.attempt)
      if dead:
        self._drop_worker(node, task.worker)
      elif w:
        w.started, w.gpu_ms = True, w.gpu_ms + (d.charge.gpu_ms if d.charge else 0.0)
        if d.duration_s > 0 and w.park_s > 0:
          w.hold_until = max(w.hold_until, w.resumed_at + policy.quantum_s(w.park_s + d.duration_s))
        if task.kind == K.TRAIN:
          w.idle_until = now + policy.CYCLE_GRACE_S
          w.loaded.pop(task.cmd.run, None)
          if eff.loaded:
            w.loaded[task.cmd.run] = eff.open_grad
    elif task.kind == K.PARK:
      if dead:
        self._drop_worker(node, task.worker)
      elif w:
        w.gpus, w.park_s, w.hold_until, w.idle_until = (), d.duration_s, 0.0, 0.0
    elif task.kind == K.DROP:
      self._drop_worker(node, task.worker) if d.dead else (w and w.loaded.pop(task.run_id, None))
    elif task.kind == K.STOP:
      self._drop_worker(node, task.worker)
    node.tasks.pop(d.id, None)

  def handle(self, msg: T.NodeMsg) -> T.ControlMsg:
    self.expire_nodes()
    now = time.monotonic()
    if self.db.is_fenced(msg.boot_id):
      return T.ControlMsg(reset=True)
    node = self.nodes.get(msg.node)
    hello = msg.kind == "hello" or node is None
    lost_stage = False
    if hello:
      if node and node.boot_id != msg.boot_id:
        self.expire_node(node.id, node.boot_id)
        node = None
      was_up = bool(node and node.gpus)
      node = self.nodes.setdefault(msg.node, NodeState(msg.node, msg.boot_id, msg.gpus, msg.gpu_mib))
      if not was_up:
        self.record_event(T.Ev.NODE_UP, {"node": msg.node, "gpus": msg.gpus, "memory": msg.gpu_mib * 2**20})
        for tid, t in list(node.tasks.items()):
          if t.cmd and not t.cmd.body:
            node.tasks[tid] = replace(t, cmd=self.hydrate_cmd(t.cmd.request_id))
      node.gpu_mib, node.staged, node.sent_stage = msg.gpu_mib, set(msg.staged or ()), ()
      node.new_tasks.clear()
      node.cancelled.clear()
      for wi in msg.workers:
        if w := node.workers.get(wi.worker):
          w.pid, w.started, w.bound = wi.pid, True, wi.bound or w.bound
          if not node.worker_tasks(wi.worker):
            w.gpus, w.loaded = wi.gpus, dict(wi.loaded)
        else:
          node.workers[wi.worker] = LiveWorker(wi.worker, wi.spec, wi.gpus, wi.bound, dict(wi.loaded), wi.pid, True)
      keep_w = {wi.worker for wi in msg.workers} | {t.worker for t in node.tasks.values() if t.worker}
      for wid in [k for k, w in list(node.workers.items()) if k not in keep_w and w.started]:
        self._drop_worker(node, wid)
      self._clear_owners(lambda rs: rs.owner[1:] == (node.id, node.boot_id) and rs.owner[0] not in node.workers)
    elif node.boot_id != msg.boot_id:
      return T.ControlMsg(reset=True)
    else:
      if msg.staged is not None:
        nxt_s = set(msg.staged)
        lost_stage = bool(node.staged - nxt_s)
        node.staged, node.sent_stage = nxt_s, ()
      for wid in [k for k in msg.exited if k in node.workers and not node.worker_tasks(k)]:
        self._drop_worker(node, wid)

    node.gpus, node.until, node.lag, node.draining = msg.gpus or node.gpus, now + LEASE_MS / 1000, msg.lag, msg.draining
    for d in msg.done:
      self._apply_done(node, d, now)
    held = set(msg.active)
    if hello or msg.draining or lost_stage:
      bad = lambda it: node.draining or (it.version is not None and it.version not in node.staged)  # noqa: E731
      self._reclaim_samples(lambda it: it.owner[1:] == (node.id, node.boot_id) and it.task_id not in held and bad(it))
    self.schedule()
    sweep = ()
    if msg.want_sweep and not node.draining and now >= self.sweep_due:
      self.sweep_due, sweep = now + SWEEP_S, tuple(self.live_versions())
    pushed = node.drain_push(tuple(d.id for d in msg.done), sweep)
    return replace(pushed, tasks=tuple(t for t in node.tasks.values() if t.id not in held)) if hello else pushed

  # ***** direct fleet pager & GPU scheduler *****

  def _demand_runs(self) -> list[tuple[str, T.WorkerSpec, int]]:
    return [(self.account(r), rs.run.spec, len(rs.queue)) for r, rs in self.runs.items() if self._live_spec(rs)]

  def advance_lag(self) -> None:
    healthy = {nid: n.gpus for nid, n in self.nodes.items() if n.gpus and not n.draining}
    dem, cap = policy.account_demand(self._demand_runs(), healthy), float(sum(healthy.values()))
    nl, ns = policy.update_lag(self.lag, self.seen, self.service, cap, dem, self.weights)
    if nl != self.lag or ns != self.seen:
      self.lag, self.seen, self.accounts_dirty = nl, ns, True

  def ready_lag(self, rid: str) -> float:
    return policy.ready_lag(self.lag, self.seen, self.service, self.account(rid))

  def _issue(self, node: NodeState, task: T.Task, now: float) -> None:
    tok = (task.worker, node.id, node.boot_id)
    if task.kind == K.TRAIN:
      rs, it, cat = self.runs[task.cmd.run], self.runs[task.cmd.run].queue[task.cmd.request_id], time.time()
      claim = ENC((node.id, node.boot_id, cat, ENC(replace(task, cmd=replace(task.cmd, body=b"", stability=None)))))
      with self.db:
        self.db.execute("UPDATE commands SET claim=? WHERE id=?", (claim, task.cmd.request_id))
        self._save_run(replace(rs, owner=tok))
      rs.owner, it.task_id, it.owner, it.since = tok, task.id, tok, cat
    elif task.kind == K.SAMPLE:
      it = self.runs[task.cmd.run].queue[task.cmd.request_id]
      it.task_id, it.owner, it.since = task.id, tok, now
    if task.gpus and (w := node.workers.get(task.worker)):
      if w.gpus != task.gpus:
        w.resumed_at, w.hold_until = now, now + policy.QUANTUM_MS / 1000
      w.gpus, w.bound = task.gpus, w.bound or (() if w.spec.movable else task.gpus)
    node.tasks[task.id] = task
    node.new_tasks.append(task)

  def _detach_one(self, node: NodeState, w: LiveWorker, rid: str, now: float) -> None:
    if self.runs[rid].status == RS.LIVE and rid in w.loaded:
      self._issue(node, T.Task(_tid(), K.SAVE, w.id, w.spec, run_id=rid, version=uuid.uuid4().hex), now)
      return
    self._clear_owners(lambda rs: rs.run.id == rid, lose=False)
    if rid in w.loaded and w.started:
      self._issue(node, T.Task(_tid(), K.DROP, w.id, w.spec, run_id=rid), now)

  def _evict(self, node: NodeState, w: LiveWorker, now: float) -> None:
    w.stopping, tok = True, (w.id, node.id, node.boot_id)
    dirty = {r for r, rs in self.runs.items() if rs.live and rs.owner == tok and rs.status == RS.LIVE and r in w.loaded}
    self._clear_owners(lambda rs: rs.owner == tok and rs.run.id not in dirty, lose=False)
    if cand := sorted(dirty - w.failed_saves):
      self._issue(node, T.Task(_tid(), K.SAVE, w.id, w.spec, run_id=cand[0], version=uuid.uuid4().hex), now)
    elif not dirty:
      self._issue(node, T.Task(_tid(), K.STOP, w.id, w.spec), now) if w.started else node.workers.pop(w.id, None)

  def _train_head(self, rid: str) -> QueueItem | None:
    rs = self.runs.get(rid)
    ok = rs and rs.live and rs.run.role == Role.TRAINER and rs.queue
    return h if ok and (h := next(iter(rs.queue.values()))).owner is None else None

  def _ready_samples(self, spec: T.WorkerSpec, staged: set[str]) -> list[QueueItem]:
    sruns = (
      rs.queue.values() for rs in self.runs.values() if rs.live and rs.run.role == Role.SAMPLER and rs.run.spec == spec
    )
    return [it for q in sruns for it in q if it.owner is None and (it.version is None or it.version in staged)]

  def schedule(self) -> None:
    """Single pass: settle trivial heads, drain, page out, page fault + stage, grant GPUs, park idle."""
    now, deadlines = time.monotonic(), []
    for rid, rs in list(self.runs.items()):
      while (head := self._train_head(rid)) is not None:
        op, cid = head.cmd.op, head.cmd.request_id
        if op == Op.LOAD_WEIGHTS and not self.get_checkpoint(head.uri or ""):
          self._finish_train(cid, None, T.Reply.error(ValueError(f"no checkpoint at {head.uri}")))
        elif rs.owner is None and rs.status == RS.LOST and op not in (Op.CREATE_MODEL, Op.LOAD_WEIGHTS):
          err = KeyError(f"model {rid} lost its unsaved steps when its trainer stopped; call load_state")
          ok = T.Reply(True, "application/json", f'{{"type":"delete_model","model_id":"{rid}"}}'.encode())
          self._finish_train(cid, None, ok if op == Op.DELETE_MODEL else T.Reply.error(err))
        else:
          break
    if now >= self.lag_due:
      self.lag_due = now + TICK_MS / 1000
      self.advance_lag()

    for n in self.nodes.values():
      n.stage = ()
      for w in list(n.workers.values()):
        if w.failed_saves and now >= w.retry_save_at:
          w.failed_saves.clear()
        elif w.failed_saves:
          deadlines.append(w.retry_save_at)
      for w in [w for w in list(n.workers.values()) if n.draining and not n.worker_tasks(w.id)]:
        self._evict(n, w, now)

    hnodes = {nid: n for nid, n in sorted(self.nodes.items()) if n.gpus and not n.draining}
    healthy = {nid: n.gpus for nid, n in hnodes.items()}
    if not healthy:
      self.next_wake = min(deadlines, default=None)
      for n in self.nodes.values():
        n.wake()
      return

    sruns: dict[T.WorkerSpec, list[str]] = {}
    live_w = [(nid, w) for nid, n in hnodes.items() for w in n.workers.values() if not w.stopping]
    targets = {(nid, w.id): [] for nid, w in live_w if w.spec.role == Role.TRAINER}
    for r, rs in sorted(self.runs.items()):
      if rs.run.role == Role.SAMPLER and self._live_spec(rs):
        sruns.setdefault(rs.run.spec, []).append(r)
      elif (
        rs.live and rs.owner and (nw := (rs.owner[1], rs.owner[0])) in targets and hnodes[nw[0]].boot_id == rs.owner[2]
      ):
        targets[nw].append(r)
    sspecs = sorted(sruns, key=lambda s: (s.base, s.lora, s.exclusive, s.train_run or "", s.gpus))
    swant = {sp: policy.sampler_replicas(sp, [len(self.runs[r].queue) for r in sruns[sp]]) for sp in sspecs}

    pmap: dict[tuple[str, str], policy.Placed] = {}
    for nid, w in live_w:
      rs = sruns.get(w.spec, ()) if w.spec.role == Role.SAMPLER else targets[nid, w.id]
      rdy = [r for r in rs if self.runs[r].queue]
      pmap[nid, w.id] = policy.Placed(nid, w.id, w.spec, bool(rdy), max(map(self.ready_lag, rdy), default=0.0), len(rs))

    srank = lambda nw: (not hnodes[nw[0]].workers[nw[1]].started, not hnodes[nw[0]].worker_tasks(nw[1]), nw)  # noqa: E731
    keep_s = {
      nw for sp in sspecs for nw in sorted((k for k, p in pmap.items() if p.spec == sp), key=srank)[: swant[sp]]
    }

    saving = sum(1 for n in hnodes.values() for t in n.tasks.values() if t.kind == K.SAVE)
    for nid, n in hnodes.items():
      excl_on = {g for w in n.workers.values() if not w.stopping and w.spec.exclusive for g in w.gpus}
      for w in sorted(list(n.workers.values()), key=lambda x: x.id):
        if n.worker_tasks(w.id):
          continue
        if w.stopping:
          self._evict(n, w, now)
          continue
        is_tr = w.spec.role == Role.TRAINER
        owned = targets[nid, w.id] if is_tr else sruns.get(w.spec, ())
        ready, pl = [r for r in owned if self.runs[r].queue], list(pmap.values())
        move_ok = w.spec.gpus == 1 or any(policy.can_host(m, healthy, pl, w.spec) for m in healthy if m != nid)
        bad = w.spec.gpus > n.gpus or (not policy.can_host(nid, healthy, pl, w.spec, skip=w.id) and move_ok)
        rebind = not is_tr or (ready and any(excl_on.isdisjoint(b) for b in policy.blocks(n.gpus, w.spec.gpus)))
        blk = not w.gpus and bool(w.bound) and not excl_on.isdisjoint(w.bound) and rebind
        drop = bad or blk or ((not owned and w.started) if is_tr else (nid, w.id) not in keep_s)
        if not drop and is_tr and w.spec.shareable:
          used, _, spare, *_ = policy.node_stats(nid, n.gpus, pl)
          twin = any(p.spec == w.spec and nw != (nid, w.id) for nw, p in pmap.items())
          no_spare = spare < 0 and not any(policy.node_stats(m, g, pl)[2] >= w.spec.gpus for m, g in healthy.items())
          drop = twin and ((not ready and used > n.gpus) or no_spare)
          if not drop and len(ready) >= 2 and not saving and policy.choose_node(w.spec, healthy, pl, only_spare=True):
            self._detach_one(n, w, ready[-1], now)
            saving += 1
            continue
        if not drop and is_tr and not saving and policy.rebalance_target(pmap[nid, w.id], healthy, pl):
          drop, saving = True, saving + 1
        if drop:
          pmap.pop((nid, w.id), None)
          targets.pop((nid, w.id), None)
          self._evict(n, w, now)

    for nid, wid in [k for k, rs in targets.items() if not rs and not hnodes[k[0]].workers[k[1]].started]:
      if not hnodes[nid].worker_tasks(wid):
        del hnodes[nid].workers[wid], targets[nid, wid], pmap[nid, wid]

    sampled = {rs.run.train_run for rs in self.runs.values() if rs.live and rs.run.train_run}

    def pick_node(sp: T.WorkerSpec, **kw) -> str | None:
      owed: dict[str, int] = {}
      for nw, p in pmap.items():
        if p.spec.role == Role.TRAINER and not p.spec.lora and targets.get(nw) and not (sampled & set(targets[nw])):
          owed[p.node] = owed.get(p.node, 0) + p.spec.gpus
      return policy.choose_node(sp, healthy, list(pmap.values()), owed_gpus=owed, **kw)

    def spawn_on(spot: str, sp: T.WorkerSpec, busy: bool, lag: float = 0.0, runs: int = 0) -> str:
      used_ids = {wid for ns in self.nodes.values() for wid in ns.workers}
      self.made = next(i for i in itertools.count(self.made + 1) if f"w{i}" not in used_ids)
      wid = f"w{self.made}"
      hnodes[spot].workers[wid] = LiveWorker(wid, sp)
      pmap[spot, wid] = policy.Placed(spot, wid, sp, busy, lag, runs)
      self.record_event(T.Ev.PLACED, {"proc": wid, "node": spot})
      return wid

    for rid, rs in sorted(self.runs.items()):
      if rs.owner is not None or self._train_head(rid) is None:
        continue
      sp = rs.run.spec
      compat = [nw for nw in sorted(targets) if sp.shareable and hnodes[nw[0]].workers[nw[1]].spec == sp]
      idle_c = [nw for nw in compat if not any(self.runs[r].queue for r in targets[nw])]
      spare_spot = pick_node(sp, only_spare=True) if sp.shareable else None
      if compat and (idle_c or not spare_spot):
        targets[pk := min(idle_c or compat, key=lambda nw: (len(targets[nw]), nw))].append(rid)
        pmap[pk] = replace(pmap[pk], busy=True, lag=max(pmap[pk].lag, self.ready_lag(rid)), runs=len(targets[pk]))
      elif spot := spare_spot or pick_node(sp, prefer_cool=True):
        targets[spot, spawn_on(spot, sp, True, self.ready_lag(rid), 1)] = [rid]

    need_s: dict[str, set[str]] = {}
    for sp in sspecs:
      trs = (self.runs.get(self.runs[r].run.train_run or "") for r in sruns[sp])
      near = {tr.owner[1] for tr in trs if tr and tr.owner and tr.owner[1] in healthy}
      pool = sorted(nw for nw, p in pmap.items() if p.spec == sp)
      if not pool and (spot := pick_node(sp, near=near)):
        pool.append((spot, spawn_on(spot, sp, any(self.runs[r].queue for r in sruns[sp]))))
      while len(pool) < swant[sp] and (spot := pick_node(sp, near=near, only_spare=True)):
        pool.append((spot, spawn_on(spot, sp, True)))
      vers = {it.version for r in sruns[sp] for it in self.runs[r].queue.values() if it.owner is None and it.version}
      for nid, _ in pool:
        need_s.setdefault(nid, set()).update(vers - hnodes[nid].staged)
    for nid, vs in need_s.items():
      hnodes[nid].stage = tuple(sorted(vs))

    fair = {r: rs.fair_ms or 0.0 for r, rs in self.runs.items() if rs.live and rs.queue}
    dem_runs = self._demand_runs()
    holds: dict[tuple[str, str], tuple[str, int]] = {}
    for nid, n in hnodes.items():
      for t in n.tasks.values():
        if t.kind in (K.TRAIN, K.SAMPLE) and t.cmd:
          holds.setdefault((nid, t.worker), (self.account(t.cmd.run), len(t.gpus) or t.spec.gpus))

    keeps, wakes = {}, {}
    while True:
      lag_a = policy.projected_lag(
        self.lag, self.seen, self.service, self.weights, dem_runs, healthy, list(holds.values())
      )
      lag_r = lambda r, fn=lag_a: fn(self.account(r))  # noqa: E731
      cands, keeps, wakes = [], {}, {}
      for nid, n in hnodes.items():
        fixed, occ, plan_w, rmap = {}, set(), {}, {}
        for wid, w in n.workers.items():
          wt = n.worker_tasks(wid)
          if w.stopping or any(t.kind not in (K.TRAIN, K.SAMPLE) for t in wt):
            occ.update(w.gpus)
            continue
          if wt:
            occ.update(w.gpus)
            fixed[wid] = w.gpus
            if w.spec.role == Role.SAMPLER and len(wt) < T.MAX_SEQS:
              rmap[wid] = {it.cmd.run for it in self._ready_samples(w.spec, n.staged)}
            continue
          if w.spec.role == Role.TRAINER:
            heads = {r: h for r in targets.get((nid, wid), ()) if (h := self._train_head(r)) is not None}
            deadlines.extend((w.idle_until,) if (grace := bool(w.gpus and w.idle_until > now)) else ())
            close = any(
              og and ((r in heads and heads[r].cmd.op == Op.OPTIM_STEP) or (not heads and grace))
              for r, og in w.loaded.items()
            )
            dem = (
              policy.Demand.FINISH_STEP if close else (policy.Demand.READY if (heads or grace) else policy.Demand.IDLE)
            )
            rr, gang = set(heads), w.spec.gpus > 1
          else:
            rr, gang = {it.cmd.run for it in self._ready_samples(w.spec, n.staged)}, False
            dem = policy.Demand.READY if rr else policy.Demand.IDLE
          rmap[wid] = rr
          wlag = max(map(lag_r, rr), default=-w.gpu_ms)
          w_args = (w.spec.gpus, w.spec.exclusive, dem, w.gpus, w.bound, gang)
          plan_w[wid] = policy.Worker(*w_args, w.hold_until, wlag, w.gpu_ms)
        chosen, wakes[nid] = policy.plan_gpus(plan_w, n.gpus, now, occ, {g for b in fixed.values() for g in b})
        keeps[nid] = keep = fixed | chosen
        contended = any(pw.demand != policy.Demand.IDLE and wid not in keep for wid, pw in plan_w.items())
        for wid, blk in keep.items():
          w, rr = n.workers[wid], rmap.get(wid)
          if blk and rr and not (wid in fixed and contended) and (not w.gpus or w.gpus == blk):
            if all(set(o.gpus).isdisjoint(blk) for oid, o in n.workers.items() if oid != wid):
              prio = (w.spec.exclusive, round(max(map(lag_r, rr)), 3), round(-w.gpu_ms, 3), bool(w.gpus), nid, wid)
              cands.append((prio, n, w, blk))
      cands.sort(key=lambda x: x[0], reverse=True)
      win = next(
        ((n, w, b) for _, n, w, b in cands if self._grant(n, w, b, targets.get((n.id, w.id), ()), fair, lag_r, now)),
        None,
      )
      if not win:
        break
      holds.setdefault((win[0].id, win[1].id), (self.account(win[0].new_tasks[-1].cmd.run), len(win[2])))

    for nid, n in hnodes.items():
      deadlines.extend((wakes[nid],) if wakes.get(nid) is not None else ())
      for w in list(n.workers.values()):
        if w.gpus and not w.stopping and not n.worker_tasks(w.id) and keeps.get(nid, {}).get(w.id) != w.gpus:
          self._issue(n, T.Task(_tid(), K.PARK, w.id, w.spec), now)
    self.next_wake = min((d for d in deadlines if d > now), default=None)
    for n in self.nodes.values():
      n.wake()

  def _grant(self, node: NodeState, w: LiveWorker, blk: tuple[int, ...], runs, fair, lag_run, now: float) -> bool:
    if w.spec.role == Role.TRAINER:
      if not (heads := [h.cmd for r in runs if (h := self._train_head(r)) is not None]):
        return False
      rs = self.runs[rid := policy.rank_runs(heads, lag_run, fair)[0]]
      op = (it := self._train_head(rid)).cmd.op
      save_op = op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER)
      ver = self.checkpoint_version(it.uri or "") if op == Op.LOAD_WEIGHTS else (uuid.uuid4().hex if save_op else None)
      snap = rs.snapshot if rs.status == RS.SAVED and rid not in w.loaded and op != Op.LOAD_WEIGHTS else None
      cmd = self.hydrate_cmd(it.cmd.request_id)
      self._issue(node, T.Task(_tid(), K.TRAIN, w.id, w.spec, blk, cmd, rs.run, rid, snap, ver), now)
      return True
    active = [t for t in node.worker_tasks(w.id) if t.kind == K.SAMPLE]
    cap, act_v = (T.MAX_LORAS if w.spec.lora else 1), {t.version for t in active if t.version is not None}
    elig = node.staged if len(act_v) < cap else (node.staged & act_v)
    heads, vmap = [], {}
    for it in sorted(self._ready_samples(w.spec, elig), key=lambda i: T.entry_sort_key(i.cmd.entry_id)):
      if it.cmd.run not in vmap:
        heads.append(it.cmd)
        vmap[it.cmd.run] = it.version

    def qfn(r: str, n: int):
      return itertools.islice((i.cmd for i in self.runs[r].queue.values() if not i.owner and i.version == vmap[r]), n)

    act_a, slots = self.account(active[0].cmd.run) if active else None, T.MAX_SEQS - len(active)
    for cmd in (
      batch := policy.admit_samples(heads, qfn, self.account, lag_run, fair, vmap.get, slots, cap, act_a, act_v)
    ):
      it = self.runs[cmd.run].queue[cmd.request_id]
      self._issue(
        node,
        T.Task(_tid(), K.SAMPLE, w.id, w.spec, blk, cmd, run_id=cmd.run, version=it.version, attempt=it.attempt),
        now,
      )
    return bool(batch)

  # ***** maintenance & inspection *****

  def maintain(self) -> None:
    changed, now, mono = self.expire_nodes(), time.time(), time.monotonic()
    for (sid,) in self.db.execute(
      "SELECT id FROM sessions WHERE active=1 AND seen<=?", (now - SESSION_GRACE,)
    ).fetchall():
      self.end_session(sid)
    self.advance_lag()
    run_gc = mono >= self.gc_due
    if run_gc or self.accounts_dirty:
      with self.db:
        self._flush_sql()
        if run_gc:
          self.db.gc(now, SESSION_TTL)
      self.accounts_dirty = False
      if run_gc:
        self.gc_due, cut = mono + GC_S, now - SESSION_TTL
        for k in [k for k, (_, exp, _) in list(self.sample_replies.items()) if exp <= now]:
          self.sample_r_bytes -= len(self.sample_replies.pop(k)[0].body)
        self.sample_seen = {k: v for k, v in self.sample_seen.items() if v[1] > now}
        live_s = {r[0] for r in self.db.execute("SELECT id FROM sessions")}
        self.runs = {
          r: rs for r, rs in self.runs.items() if rs.live or rs.deleted_at > cut or rs.queue or rs.run.session in live_s
        }
        self.db.execute("PRAGMA incremental_vacuum(256)")
    if changed or (self.next_wake is not None and mono >= self.next_wake):
      self.schedule()

  def workers_of(self, rid: str) -> list[str]:
    if not (rs := self.runs.get(rid)) or not rs.live:
      return []
    if rs.run.role == Role.TRAINER:
      return [rs.owner[0]] if rs.owner else []
    ws = (w for n in self.nodes.values() if n.gpus for w in n.workers.values())
    return sorted(w.id for w in ws if not w.stopping and w.spec == rs.run.spec)

  def snapshot_scheduling(self) -> dict:
    self.expire_nodes()
    up = {nid: n for nid, n in sorted(self.nodes.items()) if n.gpus}
    procs = {}
    for nid, n in up.items():
      for wid, w in sorted(n.workers.items()):
        if not w.stopping:
          extra = ({"pid": w.pid} if w.pid else {}) | ({"gpus": list(w.gpus)} if w.gpus else {})
          procs[wid] = {"spec": asdict(w.spec), "node": nid, **extra}
    home = {r: rs.owner[0] for r, rs in sorted(self.runs.items()) if rs.live and rs.owner and rs.owner[0] in procs}
    jobs = {
      r: {"spec": asdict(rs.run.spec), "backlog": len(rs.queue)} for r, rs in sorted(self.runs.items()) if rs.live
    }
    st = {"procs": procs, "home": home, "jobs": jobs, "lag": dict(self.lag), "seen": dict(self.seen)}
    return {
      "nodes": {k: n.gpus for k, n in up.items()},
      "state": st | {"service": dict(self.service), "weights": dict(self.weights)},
    }
