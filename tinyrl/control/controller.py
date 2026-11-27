"""Controller kernel: durable runs/queues/checkpoints, live cluster nodes/workers, and swappable `scheduler`."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import time
import uuid
import weakref
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace

from tinyrl.control import pager, policy
from tinyrl.control.accounting import Charge
from tinyrl.control.db import DB, decode
from tinyrl.helpers import LEASE_MS, SWEEP_S, getenv
from tinyrl.types import (
  Checkpoint,
  Cmd,
  ControlMsg,
  Done,
  Effect,
  Ev,
  Event,
  NodeMsg,
  Op,
  Owner,
  QueueFullError,
  Reply,
  RequestGoneError,
  Role,
  Run,
  RunStatus,
  Task,
  TaskKind,
  WorkerSpec,
  entry_sort_key,
)

GC_S, LIVE_SECONDS, READ_REPLY_TTL = 10, 30, 60
SESSION_GRACE, TICK_MS = getenv("SESSION_GRACE", 60), getenv("TICK_MS", 100)
REPLY_TTL, SESSION_TTL = getenv("REPLY_TTL", 3600), getenv("SESSION_TTL", 7 * 24 * 3600)
SAMPLE_QUEUE_BYTES = getenv("SAMPLE_QUEUE_BYTES", 256 * 2**20)
SAMPLE_REPLY_BYTES = getenv("SAMPLE_REPLY_BYTES", 256 * 2**20)
STABILITY_TTL, STABILITY_BYTES = getenv("STABILITY_TTL", 24 * 3600), getenv("STABILITY_BYTES", 512 * 2**20)
COUNTERS = ("steps", "failed", "cmds", "tokens", "busy_ms", "wait_ms")


def new_task_id() -> str:
  return uuid.uuid4().hex[:12]


@dataclass
class QueueItem:
  cmd: Cmd
  uri: str | None = None
  version: str | None = None
  stab_vers: tuple[str, ...] = ()
  task_id: str | None = None
  owner: Owner | None = None
  since: float = 0.0
  attempt: int = 0


@dataclass
class RunState:
  run: Run
  created: float
  status: RunStatus = RunStatus.FRESH
  owner: Owner | None = None
  snapshot: str | None = None
  clients: int = 0
  deleted_at: float = 0.0
  fair_ms: float | None = None
  counters: dict[str, int] = field(default_factory=lambda: dict.fromkeys(COUNTERS, 0))
  queue: dict[str, QueueItem] = field(default_factory=dict)

  @property
  def live(self) -> bool:
    return self.status != RunStatus.DELETED


@dataclass
class LiveWorker:
  id: str
  spec: WorkerSpec
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
  tasks: dict[str, Task] = field(default_factory=dict)
  new_tasks: list[Task] = field(default_factory=list)
  cancelled: list[str] = field(default_factory=list)
  push_ev: asyncio.Event = field(default_factory=asyncio.Event)

  def worker_tasks(self, worker_id: str) -> list[Task]:
    return [t for t in self.tasks.values() if t.worker == worker_id]

  def record_task(self, task: Task) -> None:
    if task.gpus and (w := self.workers.get(task.worker)) is not None:
      w.gpus = task.gpus
      if not w.bound and not w.spec.movable:
        w.bound = task.gpus
    self.tasks[task.id] = task
    self.new_tasks.append(task)

  def cancel_task(self, task_id: str) -> None:
    self.tasks.pop(task_id, None)
    self.new_tasks = [t for t in self.new_tasks if t.id != task_id]
    self.cancelled.append(task_id)

  def wake(self) -> None:
    if self.new_tasks or self.cancelled or self.stage != self.sent_stage:
      self.push_ev.set()

  def drain_push(self, ack=(), sweep=None, *, hello: bool = False, active_ids: set[str] = frozenset()) -> ControlMsg:
    stage, self.sent_stage = (self.stage if self.stage != self.sent_stage else ()), self.stage
    msg = ControlMsg(ack=ack, tasks=tuple(self.new_tasks), cancel=tuple(self.cancelled), stage=stage, sweep=sweep)
    self.new_tasks.clear()
    self.cancelled.clear()
    self.push_ev.clear()
    if hello:
      stale = tuple(sorted(tid for tid in active_ids if tid not in self.tasks and tid not in ack))
      tasks = tuple(t for t in self.tasks.values() if t.id not in active_ids)
      return replace(msg, tasks=tasks, cancel=tuple(dict.fromkeys((*msg.cancel, *stale))))
    return msg


class Controller:
  """Owns durable runs/queues/checkpoints, live nodes/workers, and drives a pluggable `scheduler(ctrl, now)`."""

  def __init__(self, path: str | None = None, *, scheduler: Callable | None = None):
    self.db = DB(getenv("TINYRL_DB", "/tmp/tinyrl.db") if path is None else path)
    self.runs: dict[str, RunState] = {}
    self.nodes: dict[str, NodeState] = {}
    self.made, self.scheduler = 0, pager.schedule if scheduler is None else scheduler
    self.service, self.lag, self.seen, self.weights = self.db.load_accounts()
    seq_row = self.db.one("SELECT v FROM meta WHERE k='seq'")
    self.seq, self.accounts_dirty = (int(seq_row[0]) if seq_row else 0), False
    self.events: deque[Event] = deque(maxlen=100_000)
    self.waiters: weakref.WeakValueDictionary[str, asyncio.Event] = weakref.WeakValueDictionary()
    self.sample_q_bytes = self.sample_r_bytes = 0
    self.sample_replies: OrderedDict[str, tuple[Reply, float, bool]] = OrderedDict()
    self.sample_seen: dict[tuple[str, int], tuple[str, float]] = {}
    self.gc_due = self.sweep_due = self.lag_due = 0.0
    self.next_wake: float | None = None

    grace = time.monotonic() + LEASE_MS / 1000
    for rid, run, cr, st, ow, sn, cl, dl, ct in self.db.load_runs():
      self.runs[rid] = state = RunState(run, cr, st, ow, sn, cl, dl, counters=ct)
      if state.live and ow:
        self.nodes.setdefault(ow[1], NodeState(ow[1], ow[2], gpus=0, until=grace))
    for cid, seq, cmd, uri, s_vers, claim in self.db.load_commands():
      self.seq = max(self.seq, seq)
      if (state := self.runs.get(cmd.run)) is None or not state.live:
        continue
      item = QueueItem(cmd, uri=uri, stab_vers=s_vers)
      if claim:
        nid, bid, at, task = claim
        item.task_id, item.owner, item.since = task.id, (task.worker, nid, bid), at
        node = self.nodes.setdefault(nid, NodeState(nid, bid, gpus=0, until=grace))
        node.tasks[task.id] = task
        w = node.workers.setdefault(task.worker, LiveWorker(task.worker, task.spec, task.gpus, started=True))
        if not w.bound and not task.spec.movable:
          w.bound = task.gpus
      state.queue[cid] = item

  @property
  def tick_s(self) -> float:
    return TICK_MS / 1000

  async def aclose(self) -> None:
    if self.db is not None:
      with self.db:
        self.flush_sql()
      self.db.close()
      self.db = None

  def flush_sql(self, service: dict[str, float] | None = None) -> None:
    self.db.save_accounts(self.service if service is None else service, self.lag, self.seen, self.weights, self.seq)

  def notify(self, req_id: str) -> None:
    if waiter := self.waiters.pop(req_id, None):
      waiter.set()

  def record_event(self, kind: Ev, body: dict) -> Event:
    self.seq += 1
    now = time.time()
    ev = Event(f"{int(now * 1000)}-{self.seq}", now, str(Ev(kind)), body)
    self.events.append(ev)
    return ev

  def read_events(self, after: str = "0-0", count: int = 10_000) -> list[Event]:
    cursor = entry_sort_key(after)
    return [ev for ev in self.events if entry_sort_key(ev.id) > cursor][:count]

  def live_node(self, owner: Owner | None) -> NodeState | None:
    if owner is None:
      return None
    n = self.nodes.get(owner[1])
    return n if (n is not None and n.boot_id == owner[2]) else None

  def healthy_nodes(self) -> dict[str, NodeState]:
    return {nid: n for nid, n in sorted(self.nodes.items()) if n.gpus and not n.draining}

  def is_sampler(self, run_id: str, req_id: str = "") -> bool:
    s = self.runs.get(run_id)
    return s.run.role == Role.SAMPLER if s is not None else req_id.startswith("samp-")

  def account(self, run_id: str) -> str:
    if (s := self.runs.get(run_id)) is None or not s.live:
      return run_id
    if s.run.account:
      return s.run.account
    if (p := self.runs.get(s.run.train_run or "")) is not None and p.live:
      return self.account(s.run.train_run)
    return s.run.session or run_id

  def has_live_spec(self, s: RunState) -> bool:
    if not s.live:
      return False
    pid = s.run.spec.train_run
    return not pid or ((p := self.runs.get(pid)) is not None and p.live)

  def demand_runs(self) -> list[tuple[str, WorkerSpec, int]]:
    return [(self.account(r), s.run.spec, len(s.queue)) for r, s in self.runs.items() if self.has_live_spec(s)]

  def advance_lag(self, capacity: dict[str, int] | None = None) -> None:
    cap = {nid: n.gpus for nid, n in self.nodes.items() if n.gpus and not n.draining} if capacity is None else capacity
    demand = policy.account_demand(self.demand_runs(), cap)
    lg, sn = policy.update_lag(self.lag, self.seen, self.service, float(sum(cap.values())), demand, self.weights)
    if lg != self.lag or sn != self.seen:
      self.lag, self.seen, self.accounts_dirty = lg, sn, True

  def ready_lag(self, run_id: str) -> float:
    return policy.ready_lag(self.lag, self.seen, self.service, self.account(run_id))

  # --- sessions, runs & checkpoints ---

  def heartbeat_session(self, sid: str, metadata: dict | None = None, *, finish: bool = False) -> None:
    with self.db:
      self.db.save_session(sid, time.time(), metadata, finish=finish)

  def get_session(self, sid: str) -> dict | None:
    if (loaded := self.db.load_session(sid)) is None:
      return None
    cr, seen, meta = loaded
    runs = sorted(r for r, s in self.runs.items() if s.run.session == sid)
    return {
      "session_id": sid,
      "created": cr,
      "seen": seen,
      "live": time.time() - seen < LIVE_SECONDS,
      "runs": runs,
      "metadata": meta,
    }

  def end_session(self, sid: str) -> None:
    live = sorted(r for r, s in self.runs.items() if s.live and s.run.session == sid)
    queued = [bool(self.submit_command(r, Op.DELETE_MODEL, b"{}")) for r in live if not self.is_sampler(r)]
    for r in live:
      if self.is_sampler(r):
        self.delete_run(r)
    with self.db:
      if any(queued):
        self.db.execute("UPDATE sessions SET seen=? WHERE id=?", (time.time(), sid))
      else:
        self.db.execute("UPDATE sessions SET active=0 WHERE id=?", (sid,))

  def create_run(self, run: Run) -> None:
    state = RunState(run, time.time())
    with self.db:
      self.db.save_run(state)
    self.runs[run.id] = state
    fields = ("role", "base", "rank", "exclusive", "train_run", "gpus")
    self.record_event(Ev.RUN_CREATED, {"run": run.id, **{k: getattr(run, k) for k in fields}})
    self.schedule()

  def join_sampling_session(self, run_id: str) -> int:
    if (s := self.runs.get(run_id)) is None:
      return 1
    u = replace(s, clients=s.clients + 1)
    with self.db:
      self.db.save_run(u)
    self.runs[run_id] = u
    return u.clients

  def get_run(self, run_id: str) -> dict | None:
    if (s := self.runs.get(run_id)) is None:
      return None
    extra = ({"owner": list(s.owner)} if s.owner else {}) | ({"snapshot": s.snapshot} if s.snapshot else {})
    return {**asdict(s.run), "created": s.created, "status": str(s.status), **s.counters, **extra}

  def delete_run(self, run_id: str) -> None:
    if (s := self.runs.get(run_id)) is not None:
      deleted_reply, queued = Reply.error(RuntimeError(f"run {run_id} was deleted")), []
      for cid, it in list(s.queue.items()):
        if s.run.role == Role.SAMPLER:
          self.sample_q_bytes -= len(it.cmd.body)
          if it.owner and it.task_id and (n := self.live_node(it.owner)) is not None:
            n.cancel_task(it.task_id)
            n.wake()
        elif it.owner:
          self.finish_train(cid, it.owner, deleted_reply)
          if (
            it.task_id and (n := self.live_node(it.owner)) is not None and any(t.id == it.task_id for t in n.new_tasks)
          ):
            n.tasks.pop(it.task_id, None)
            n.new_tasks = [t for t in n.new_tasks if t.id != it.task_id]
        else:
          queued.append(cid)
      self.sample_seen = {k: v for k, v in self.sample_seen.items() if k[0] != run_id}
      now, dropped = time.time(), list(s.queue)
      u = replace(s, status=RunStatus.DELETED, owner=None, snapshot=None, deleted_at=now)
      with self.db:
        for cid in queued:
          self.db.save_reply(cid, deleted_reply, now + REPLY_TTL)
        self.db.save_run(u)
        self.db.remove_command(None, run_id, deleting=True)
      self.runs[run_id] = u
      u.queue.clear()
      for cid in dropped:
        self.notify(cid)
    self.record_event(Ev.RUN_DELETED, {"run": run_id})
    self.schedule()

  def put_checkpoint(self, ckpt: Checkpoint) -> None:
    with self.db:
      self.db.save_checkpoint(ckpt)

  def list_checkpoints(self, run: str | None = None) -> list[Checkpoint]:
    return self.db.list_checkpoints(time.time(), run)

  def get_checkpoint(self, uri: str) -> Checkpoint | None:
    return self.db.get_checkpoint(uri, time.time())

  def checkpoint_version(self, uri: str) -> str:
    if ckpt := self.get_checkpoint(uri):
      return ckpt.version
    raise ValueError(f"no checkpoint at {uri}")

  def set_checkpoint_ttl(self, uri: str, ttl: int | None) -> bool:
    if (ckpt := self.get_checkpoint(uri)) is None:
      return False
    self.put_checkpoint(replace(ckpt, expires_at=time.time() + ttl if ttl else None))
    return True

  def delete_checkpoint(self, uri: str) -> bool:
    if self.get_checkpoint(uri) is None:
      return False
    with self.db:
      self.db.execute("DELETE FROM checkpoints WHERE uri=?", (uri,))
    return True

  def live_versions(self) -> list[str]:
    stab = {r for (r,) in self.db.execute("SELECT DISTINCT ref FROM stability WHERE expires>?", (time.time(),))}
    vers = {c.version for c in self.list_checkpoints() if c.version} | stab
    vers |= {v for n in self.nodes.values() for t in n.tasks.values() for v in (t.version, t.snapshot) if v}
    for s in self.runs.values():
      if s.live and s.snapshot:
        vers.add(s.snapshot)
      for it in s.queue.values():
        vers.update((it.version,) if it.version else (), it.stab_vers)
    return sorted({v.partition("@")[0] for v in vers})

  # --- commands, replies & billing ---

  def seen_request(self, run_id: str, seq_id: int | None) -> str | None:
    if seq_id is None:
      return None
    now = time.time()
    if self.is_sampler(run_id):
      hit = self.sample_seen.get((run_id, seq_id))
      return hit[0] if hit and hit[1] > now else None
    row = self.db.one("SELECT req_id FROM dedup WHERE run_id=? AND seq_id=? AND expires>?", (run_id, seq_id, now))
    return row[0] if row else None

  def submit_command(
    self, run_id: str, op: Op, body: bytes, seq_id=None, cost=1, uri=None, stability_refs=None, *, schedule: bool = True
  ) -> str | None:
    if existing := self.seen_request(run_id, seq_id):
      return existing
    if (s := self.runs.get(run_id)) is None or not s.live:
      return None
    if not s.queue and s.fair_ms is None:
      acct, spec = self.account(run_id), s.run.spec
      peers = [
        o.fair_ms
        for k, o in self.runs.items()
        if o.live and o.fair_ms is not None and o.run.spec == spec and self.account(k) == acct
      ]
      s.fair_ms = min(peers, default=0.0)

    self.seq += 1
    now = time.time()
    cmd = Cmd(run_id, f"{time.time_ns() // 1_000_000}-{self.seq}", Op(op), body, cost)
    req_id = cmd.request_id
    if s.run.role == Role.SAMPLER:
      if self.sample_q_bytes + len(body) > SAMPLE_QUEUE_BYTES:
        raise QueueFullError("sampling queue memory limit reached")
      self.sample_q_bytes += len(body)
      ckpt = self.get_checkpoint(s.run.model_path) if s.run.model_path else None
      s.queue[req_id] = QueueItem(cmd, version=ckpt.version if ckpt else None)
      if seq_id is not None:
        self.sample_seen[run_id, seq_id] = (req_id, now + SESSION_TTL)
      if s.run.model_path and ckpt is None:
        self.finish_sample(req_id, None, Reply.error(ValueError(f"no checkpoint at {s.run.model_path}")), None, 0)
    else:
      with self.db:
        s_vers = self.db.insert_command(cmd, self.seq, uri, seq_id, now + SESSION_TTL, stability_refs, now)
      s.queue[req_id] = QueueItem(replace(cmd, body=b""), uri=uri, stab_vers=s_vers)
    if schedule:
      self.schedule()
    return req_id

  def read_reply(self, req_id: str) -> Reply | None:
    now, read_ttl, rid = time.time(), min(REPLY_TTL, READ_REPLY_TTL), req_id.partition("/")[0]
    if self.is_sampler(rid, req_id):
      if (hit := self.sample_replies.get(req_id)) and hit[1] > now:
        self.sample_replies.move_to_end(req_id)
        self.sample_replies[req_id] = (hit[0], min(hit[1], now + read_ttl), True)
        return hit[0]
      if (s := self.runs.get(rid)) is None or req_id not in s.queue:
        raise RequestGoneError(f"sampling request gone: {req_id}")
      return None
    with self.db:
      return self.db.read_reply(req_id, now, read_ttl)

  async def wait_for_reply(self, req_id: str, timeout: float) -> Reply | None:
    deadline = time.monotonic() + max(0.0, timeout)
    while (reply := self.read_reply(req_id)) is None:
      if (rem := deadline - time.monotonic()) <= 0:
        return None
      with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(self.waiters.setdefault(req_id, asyncio.Event()).wait(), rem)
    return reply

  def queued_and_in_flight(self, run_id: str) -> dict:
    if (s := self.runs.get(run_id)) is None or not s.live:
      return {"live": False}
    now = time.monotonic() if s.run.role == Role.SAMPLER else time.time()
    fly = [
      {"request_id": k, "trainer": it.owner[0], "deliveries": 1, "seconds": max(0.0, now - (it.since or now))}
      for k, it in s.queue.items()
      if it.owner
    ]
    return {"live": True, "queued": len(s.queue) - len(fly), "in_flight": fly}

  def train_head(self, run_id: str) -> QueueItem | None:
    s = self.runs.get(run_id)
    if s is None or not s.live or s.run.role != Role.TRAINER or not s.queue:
      return None
    head = next(iter(s.queue.values()))
    return head if head.owner is None else None

  def sample_heads(self, spec: WorkerSpec, staged: set[str], allow_base: bool = True) -> dict[str, QueueItem]:
    heads: dict[str, QueueItem] = {}
    for rid, s in self.runs.items():
      if s.live and s.run.role == Role.SAMPLER and s.run.spec == spec and s.queue:
        for it in s.queue.values():
          if it.owner is None and ((allow_base and it.version is None) or it.version in staged):
            heads[rid] = it
            break
    return heads

  def hydrate_cmd(self, req_id: str) -> Cmd:
    row = self.db.one("SELECT body, stab FROM commands WHERE id=?", (req_id,))
    body, stab = (bytes(row[0]), decode(row[1]) if row[1] else None) if row else (b"", None)
    return replace(self.runs[req_id.partition("/")[0]].queue[req_id].cmd, body=body, stability=stab)

  def apply_charge(
    self, state: RunState, cmd: Cmd, reply: Reply, charge: Charge | None, counters: dict, service: dict
  ) -> float | None:
    counters["failed"] += int(not reply.ok)
    counters["steps"] += int(reply.ok and cmd.op == Op.OPTIM_STEP)
    fair_ms = state.fair_ms
    if charge is not None:
      fair_ms = fair_ms + charge.gpu_ms if fair_ms is not None else None
      counters["cmds"] += 1
      counters["busy_ms"] += charge.busy_ms
      counters["wait_ms"] += max(0, charge.started_ms - entry_sort_key(cmd.entry_id)[0])
      if reply.ok:
        counters["tokens"] += cmd.cost
      if charge.gpu_ms:
        if acct := self.account(cmd.run):
          service[acct] = service.get(acct, 0.0) + charge.gpu_ms
    return fair_ms

  def finish_train(
    self,
    req_id: str,
    owner: Owner | None,
    reply: Reply,
    charge=None,
    *,
    mutated=False,
    checkpoint=None,
    dead=False,
    loaded=True,
  ) -> bool:
    s = self.runs.get(req_id.partition("/")[0])
    item = s.queue.get(req_id) if s else None
    if s is None or item is None or item.owner != owner:
      return False
    cmd, now, deleting = item.cmd, time.time(), item.cmd.op == Op.DELETE_MODEL and reply.ok
    counters, service = dict(s.counters), dict(self.service)
    fair_ms = self.apply_charge(s, cmd, reply, charge, counters, service)
    nxt = replace(s, counters=counters, fair_ms=None if len(s.queue) <= 1 else fair_ms)
    peers: list[RunState] = []

    if deleting:
      nxt = replace(nxt, status=RunStatus.DELETED, owner=None, snapshot=None, deleted_at=now)
    else:
      if mutated:
        nxt.snapshot, nxt.status = None, RunStatus.LIVE
      if dead and owner:
        nxt.owner, nxt.status = None, RunStatus.LOST if nxt.status == RunStatus.LIVE else nxt.status
        peers = [
          replace(p, owner=None, status=RunStatus.LOST if p.status == RunStatus.LIVE else p.status)
          for p in self.runs.values()
          if p.run.id != cmd.run and p.owner == owner
        ]
      elif dead or nxt.status in (RunStatus.FRESH, RunStatus.LOST) or (not loaded and nxt.status == RunStatus.SAVED):
        nxt.owner = None

    updates = [*peers, nxt]
    with self.db:
      self.db.remove_command(req_id, cmd.run, deleting=deleting)
      self.db.save_reply(req_id, reply, now + REPLY_TTL)
      if deleting:
        err = Reply.error(RuntimeError(f"run {cmd.run} was deleted"))
        for cid in s.queue:
          if cid != req_id:
            self.db.save_reply(cid, err, now + REPLY_TTL)
      self.flush_sql(service)
      if checkpoint and reply.ok:
        self.db.save_checkpoint(checkpoint)
      for u in updates:
        self.db.save_run(u)

    self.service = service
    for u in updates:
      self.runs[u.run.id] = u
    for cid in list(s.queue) if deleting else [req_id]:
      s.queue.pop(cid, None)
      self.notify(cid)
    if deleting:
      self.record_event(Ev.RUN_DELETED, {"run": cmd.run})
    return True

  def finish_sample(self, req_id: str, owner: Owner | None, reply: Reply, charge: Charge | None, attempt: int) -> bool:
    s = self.runs.get(req_id.partition("/")[0])
    item = s.queue.get(req_id) if s else None
    if s is None or item is None or item.owner != owner or item.attempt != attempt:
      return False
    if reply.ok and reply.stability:
      with self.db:
        self.db.record_stability(reply.stability, time.time() + STABILITY_TTL, STABILITY_BYTES)
    s.queue.pop(req_id)
    self.sample_q_bytes -= len(item.cmd.body)
    clean = replace(reply, stability=None)
    self.sample_replies[req_id] = (clean, time.time() + REPLY_TTL, False)
    self.sample_r_bytes += len(clean.body)
    while self.sample_r_bytes > SAMPLE_REPLY_BYTES and len(self.sample_replies) > 1:
      victim = next((k for k, (r, _, rd) in self.sample_replies.items() if rd and r.body), None) or next(
        (k for k, (r, _, _) in self.sample_replies.items() if k != req_id and r.body), None
      )
      if victim is None:
        break
      self.sample_r_bytes -= len(self.sample_replies.pop(victim)[0].body)
    fair_ms = self.apply_charge(s, item.cmd, clean, charge, s.counters, self.service)
    s.fair_ms = fair_ms if s.queue else None
    self.accounts_dirty |= bool(charge and charge.gpu_ms)
    self.notify(req_id)
    return True

  def finish_save(self, run_id: str, owner: Owner, saved: str | None) -> None:
    if (s := self.runs.get(run_id)) is None or s.owner != owner:
      return
    status = RunStatus.SAVED if saved else (RunStatus.LOST if s.status == RunStatus.LIVE else s.status)
    u = replace(s, owner=None, snapshot=saved or s.snapshot, status=status)
    with self.db:
      self.db.save_run(u)
    self.runs[run_id] = u

  def reclaim_samples(self, predicate: Callable[[QueueItem], bool]) -> None:
    for s in self.runs.values():
      if s.run.role == Role.SAMPLER:
        for it in s.queue.values():
          if it.owner and predicate(it):
            if it.task_id and (n := self.live_node(it.owner)) is not None:
              n.cancel_task(it.task_id)
            it.attempt, it.owner, it.task_id, it.since = it.attempt + 1, None, None, 0.0

  def clear_owners(self, predicate: Callable[[RunState], bool], *, lose: bool = True) -> None:
    updates = [
      replace(s, owner=None, status=RunStatus.LOST if (lose and s.status == RunStatus.LIVE) else s.status)
      for s in self.runs.values()
      if s.owner and predicate(s)
    ]
    if updates:
      with self.db:
        for u in updates:
          self.db.save_run(u)
      for u in updates:
        self.runs[u.run.id] = u

  # --- safe execution mechanisms (surface called by any swappable scheduler) ---

  def schedule(self) -> float | None:
    """Run kernel housekeeping, invoke the swappable `scheduler`, wake nodes, and record next wakeup."""
    now, deadlines = time.monotonic(), []
    for rid, s in list(self.runs.items()):
      while (head := self.train_head(rid)) is not None:
        op, req_id = head.cmd.op, head.cmd.request_id
        if op == Op.LOAD_WEIGHTS and self.get_checkpoint(head.uri or "") is None:
          self.finish_train(req_id, None, Reply.error(ValueError(f"no checkpoint at {head.uri}")))
        elif s.owner is None and s.status == RunStatus.LOST and op not in (Op.CREATE_MODEL, Op.LOAD_WEIGHTS):
          reply = (
            Reply(True, "application/json", json.dumps({"type": "delete_model", "model_id": rid}).encode())
            if op == Op.DELETE_MODEL
            else Reply.error(KeyError(f"model {rid} lost its unsaved steps when its trainer stopped; call load_state"))
          )
          self.finish_train(req_id, None, reply)
        else:
          break
    for node in self.nodes.values():
      node.stage = ()
      for w in list(node.workers.values()):
        if w.failed_saves and now >= w.retry_save_at:
          w.failed_saves.clear()
        elif w.failed_saves:
          deadlines.append(w.retry_save_at)
        if node.draining and not node.worker_tasks(w.id):
          self.evict_worker(node, w, now)
    if self.healthy_nodes():
      deadlines.extend(self.scheduler(self, now))
    for node in self.nodes.values():
      node.wake()
    self.next_wake = min((d for d in deadlines if d > now), default=None)
    return self.next_wake

  def spawn_worker(self, node_id: str, spec: WorkerSpec) -> str:
    used = {wid for n in self.nodes.values() for wid in n.workers}
    self.made = next(i for i in itertools.count(self.made + 1) if f"w{i}" not in used)
    wid = f"w{self.made}"
    self.nodes[node_id].workers[wid] = LiveWorker(wid, spec)
    self.record_event(Ev.PLACED, {"proc": wid, "node": node_id})
    return wid

  def discard_worker(self, node: NodeState, wid: str) -> None:
    if (w := node.workers.get(wid)) is not None and not w.started and not node.worker_tasks(wid):
      del node.workers[wid]

  def set_stage(self, node: NodeState, versions: Iterable[str]) -> None:
    node.stage = tuple(sorted(versions))

  def _send(self, node: NodeState, kind: TaskKind, w: LiveWorker, now: float, **kw) -> None:
    self.issue(node, Task(new_task_id(), kind, w.id, w.spec, **kw), now)

  def park_worker(self, node: NodeState, w: LiveWorker, now: float) -> None:
    self._send(node, TaskKind.PARK, w, now)

  def dispatch_train(self, node: NodeState, w: LiveWorker, block: tuple[int, ...], run_id: str, now: float) -> bool:
    if (item := self.train_head(run_id)) is None:
      return False
    s, op = self.runs[run_id], item.cmd.op
    ver = (
      self.checkpoint_version(item.uri or "")
      if op == Op.LOAD_WEIGHTS
      else (uuid.uuid4().hex if op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER) else None)
    )
    snap = s.snapshot if (s.status == RunStatus.SAVED and run_id not in w.loaded and op != Op.LOAD_WEIGHTS) else None
    self._send(
      node,
      TaskKind.TRAIN,
      w,
      now,
      gpus=block,
      cmd=self.hydrate_cmd(item.cmd.request_id),
      run=s.run,
      run_id=run_id,
      snapshot=snap,
      version=ver,
    )
    return True

  def dispatch_samples(
    self, node: NodeState, w: LiveWorker, block: tuple[int, ...], batch: list[Cmd], now: float
  ) -> bool:
    for cmd in batch:
      it = self.runs[cmd.run].queue[cmd.request_id]
      self._send(
        node, TaskKind.SAMPLE, w, now, gpus=block, cmd=cmd, run_id=cmd.run, version=it.version, attempt=it.attempt
      )
    return bool(batch)

  def issue(self, node: NodeState, task: Task, now: float) -> None:
    owner = (task.worker, node.id, node.boot_id)
    if task.kind == TaskKind.TRAIN:
      s = self.runs[task.cmd.run]
      it, at = s.queue[task.cmd.request_id], time.time()
      slim = replace(task, cmd=replace(task.cmd, body=b"", stability=None))
      with self.db:
        self.db.claim_command(task.cmd.request_id, node.id, node.boot_id, at, slim, replace(s, owner=owner))
      s.owner, it.task_id, it.owner, it.since = owner, task.id, owner, at
    elif task.kind == TaskKind.SAMPLE:
      it = self.runs[task.cmd.run].queue[task.cmd.request_id]
      it.task_id, it.owner, it.since = task.id, owner, now
    node.record_task(task)

  def detach_run(self, node: NodeState, w: LiveWorker, run_id: str, now: float) -> None:
    if self.runs[run_id].status == RunStatus.LIVE and run_id in w.loaded:
      self._send(node, TaskKind.SAVE, w, now, run_id=run_id, version=uuid.uuid4().hex)
      return
    self.clear_owners(lambda s: s.run.id == run_id, lose=False)
    if run_id in w.loaded and w.started:
      self._send(node, TaskKind.DROP, w, now, run_id=run_id)

  def evict_worker(self, node: NodeState, w: LiveWorker, now: float) -> None:
    w.stopping = True
    owner = (w.id, node.id, node.boot_id)
    dirty = {
      r for r, s in self.runs.items() if s.live and s.owner == owner and s.status == RunStatus.LIVE and r in w.loaded
    }
    self.clear_owners(lambda s: s.owner == owner and s.status != RunStatus.LIVE, lose=False)
    self.clear_owners(lambda s: s.owner == owner and s.run.id not in dirty, lose=True)
    if pending := sorted(dirty - w.failed_saves):
      self._send(node, TaskKind.SAVE, w, now, run_id=pending[0], version=uuid.uuid4().hex)
    elif not dirty:
      if w.started:
        self._send(node, TaskKind.STOP, w, now)
      else:
        node.workers.pop(w.id, None)

  # --- node liveness & protocol ---

  def live_nodes(self) -> dict[str, int]:
    self.expire_nodes()
    return {nid: n.gpus for nid, n in self.nodes.items() if n.gpus}

  def expire_node(self, nid: str, bid: str) -> None:
    if self.live_node(("", nid, bid)) is not None:
      del self.nodes[nid]
    self.record_event(Ev.NODE_LOST, {"node": nid})
    for s in list(self.runs.values()):
      if s.run.role == Role.TRAINER:
        for cid, it in list(s.queue.items()):
          if it.owner and it.owner[1:] == (nid, bid):
            err = Reply.error(RuntimeError(f"its trainer died, or lost ownership, during {it.cmd.op}"))
            self.finish_train(cid, it.owner, err, dead=True)
    with self.db:
      self.db.add_fenced(bid)
    self.reclaim_samples(lambda it: it.owner[1:] == (nid, bid))
    self.clear_owners(lambda s: s.owner[1:] == (nid, bid))

  def expire_nodes(self) -> bool:
    now = time.monotonic()
    stale = [(nid, n.boot_id) for nid, n in list(self.nodes.items()) if n.until <= now]
    for nid, bid in stale:
      self.expire_node(nid, bid)
    return bool(stale)

  def drop_worker(self, node: NodeState, wid: str) -> None:
    w, owner = node.workers.pop(wid, None), (wid, node.id, node.boot_id)
    self.reclaim_samples(lambda it: it.owner == owner)
    self.clear_owners(lambda s: s.owner == owner)
    if w and w.started:
      self.record_event(Ev.UNPLACED, {"proc": wid, "node": node.id})

  def apply_done(self, node: NodeState, done: Done, now: float) -> None:
    if (task := node.tasks.get(done.id)) is None:
      return
    w, owner, failed = node.workers.get(task.worker), (task.worker, node.id, node.boot_id), done.dead or not done.ok
    if w and done.pid is not None:
      w.pid, w.started = done.pid, True

    if task.kind == TaskKind.SAVE:
      if not failed:
        self.finish_save(task.run_id, owner, done.saved)
        if w:
          w.failed_saves.discard(task.run_id)
          if not w.stopping and w.started and task.run_id in w.loaded:
            self._send(node, TaskKind.DROP, w, now, run_id=task.run_id)
      elif done.dead:
        self.drop_worker(node, task.worker)
      elif w:
        w.failed_saves.add(task.run_id)
        w.retry_save_at = now + 0.05
    elif task.kind in (TaskKind.TRAIN, TaskKind.SAMPLE):
      eff, rep = done.effect or Effect(), done.reply or Reply.error(RuntimeError(f"{task.kind} failed"))
      if task.kind == TaskKind.TRAIN:
        self.finish_train(
          task.cmd.request_id,
          owner,
          rep,
          done.charge,
          mutated=eff.mutated,
          checkpoint=eff.checkpoint,
          dead=failed,
          loaded=eff.loaded,
        )
      elif failed and done.reply is None:
        self.reclaim_samples(lambda it: it.cmd.request_id == task.cmd.request_id and it.owner == owner)
      else:
        self.finish_sample(task.cmd.request_id, owner, rep, done.charge, task.attempt)
      if done.dead or (failed and task.kind == TaskKind.TRAIN):
        self.drop_worker(node, task.worker)
      elif w:
        w.started, w.gpu_ms = True, w.gpu_ms + (done.charge.gpu_ms if done.charge else 0.0)
        if cb := getattr(self.scheduler, "on_done", None):
          cb(w, task, done, now)
        if task.kind == TaskKind.TRAIN:
          w.loaded.pop(task.cmd.run, None)
          if eff.loaded:
            w.loaded[task.cmd.run] = eff.open_grad
            rs = self.runs.get(task.cmd.run)
            if rs and not rs.live and not w.stopping and any(s.live and s.owner == owner for s in self.runs.values()):
              self._send(node, TaskKind.DROP, w, now, run_id=task.cmd.run)
    elif task.kind == TaskKind.PARK:
      if failed:
        self.drop_worker(node, task.worker)
      elif w:
        w.gpus, w.park_s, w.hold_until, w.idle_until = (), done.duration_s, 0.0, 0.0
    elif task.kind == TaskKind.DROP:
      if done.dead:
        self.drop_worker(node, task.worker)
      elif w:
        w.loaded.pop(task.run_id, None)
    elif task.kind == TaskKind.STOP:
      self.drop_worker(node, task.worker)
    node.tasks.pop(done.id, None)

  def _sync_hello(self, node: NodeState, msg: NodeMsg) -> None:
    node.gpu_mib, node.staged, node.sent_stage = msg.gpu_mib, set(msg.staged or ()), ()
    node.new_tasks.clear()
    node.cancelled.clear()
    for info in msg.workers:
      if (w := node.workers.get(info.worker)) is not None:
        w.pid, w.started, w.bound = info.pid, True, info.bound or w.bound
        if not node.worker_tasks(info.worker):
          w.gpus, w.loaded = info.gpus, dict(info.loaded)
        elif info.loaded:
          w.loaded = {**info.loaded, **w.loaded}
      else:
        node.workers[info.worker] = LiveWorker(
          info.worker, info.spec, info.gpus, info.bound, dict(info.loaded), info.pid, started=True
        )
    alive = {i.worker for i in msg.workers} | {t.worker for t in node.tasks.values() if t.worker}
    for wid in [wid for wid, w in list(node.workers.items()) if wid not in alive and w.started]:
      self.drop_worker(node, wid)
    self.clear_owners(lambda s: s.owner[1:] == (node.id, node.boot_id) and s.owner[0] not in node.workers)

  def handle(self, msg: NodeMsg) -> ControlMsg:
    self.expire_nodes()
    now = time.monotonic()
    if self.db.is_fenced(msg.boot_id):
      return ControlMsg(reset=True)
    node = self.nodes.get(msg.node)
    is_hello, lost_stage = msg.kind == "hello" or node is None, False

    if is_hello:
      if node is not None and node.boot_id != msg.boot_id:
        self.expire_node(node.id, node.boot_id)
        node = None
      was_up = bool(node and node.gpus)
      node = self.nodes.setdefault(msg.node, NodeState(msg.node, msg.boot_id, msg.gpus, msg.gpu_mib))
      if not was_up:
        self.record_event(Ev.NODE_UP, {"node": msg.node, "gpus": msg.gpus, "memory": msg.gpu_mib * 2**20})
        for tid, task in list(node.tasks.items()):
          if task.cmd and not task.cmd.body:
            node.tasks[tid] = replace(task, cmd=self.hydrate_cmd(task.cmd.request_id))
      self._sync_hello(node, msg)
    elif node.boot_id != msg.boot_id:
      return ControlMsg(reset=True)
    else:
      if msg.staged is not None:
        rep = set(msg.staged)
        lost_stage, node.staged, node.sent_stage = bool(node.staged - rep), rep, ()
      for wid in [wid for wid in msg.exited if wid in node.workers and not node.worker_tasks(wid)]:
        self.drop_worker(node, wid)

    node.gpus, node.until, node.lag, node.draining = msg.gpus or node.gpus, now + LEASE_MS / 1000, msg.lag, msg.draining
    for done in msg.done:
      self.apply_done(node, done, now)
    active = set(msg.active)
    if is_hello or msg.draining or lost_stage:
      self.reclaim_samples(
        lambda it: (
          it.owner[1:] == (node.id, node.boot_id)
          and it.task_id not in active
          and (node.draining or (it.version is not None and it.version not in node.staged))
        )
      )
    self.schedule()
    sweep: tuple[str, ...] | None = None
    if msg.want_sweep and not node.draining and now >= self.sweep_due:
      self.sweep_due, sweep = now + SWEEP_S, tuple(self.live_versions())
    return node.drain_push(tuple(d.id for d in msg.done), sweep, hello=is_hello, active_ids=active)

  # --- maintenance & inspection ---

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
        self.flush_sql()
        if run_gc:
          self.db.gc(now, SESSION_TTL)
      self.accounts_dirty = False
      if run_gc:
        self.gc_due = mono + GC_S
        for req_id, (_, exp, _) in list(self.sample_replies.items()):
          if exp <= now:
            self.sample_r_bytes -= len(self.sample_replies.pop(req_id)[0].body)
        self.sample_seen = {k: v for k, v in self.sample_seen.items() if v[1] > now}
        live_s = {sid for (sid,) in self.db.execute("SELECT id FROM sessions")}
        cutoff = now - SESSION_TTL
        self.runs = {
          r: s for r, s in self.runs.items() if s.live or s.deleted_at > cutoff or s.queue or s.run.session in live_s
        }
        self.db.execute("PRAGMA incremental_vacuum(256)")
    if changed or (self.next_wake is not None and mono >= self.next_wake):
      self.schedule()

  def workers_of(self, run_id: str) -> list[str]:
    if (s := self.runs.get(run_id)) is None or not s.live:
      return []
    if s.run.role == Role.TRAINER:
      return [s.owner[0]] if s.owner else []
    return sorted(
      w.id for n in self.nodes.values() if n.gpus for w in n.workers.values() if not w.stopping and w.spec == s.run.spec
    )

  def snapshot_scheduling(self) -> dict:
    self.expire_nodes()
    up = {nid: n for nid, n in sorted(self.nodes.items()) if n.gpus}
    procs = {
      wid: {
        "spec": asdict(w.spec),
        "node": nid,
        **({"pid": w.pid} if w.pid else {}),
        **({"gpus": list(w.gpus)} if w.gpus else {}),
      }
      for nid, n in up.items()
      for wid, w in sorted(n.workers.items())
      if not w.stopping
    }
    home = {r: s.owner[0] for r, s in sorted(self.runs.items()) if s.live and s.owner and s.owner[0] in procs}
    jobs = {r: {"spec": asdict(s.run.spec), "backlog": len(s.queue)} for r, s in sorted(self.runs.items()) if s.live}
    state = {
      "procs": procs,
      "home": home,
      "jobs": jobs,
      "lag": dict(self.lag),
      "seen": dict(self.seen),
      "service": dict(self.service),
      "weights": dict(self.weights),
    }
    return {"nodes": {nid: n.gpus for nid, n in up.items()}, "state": state}


__all__ = [
  "COUNTERS",
  "Controller",
  "LiveWorker",
  "NodeState",
  "QueueItem",
  "RunState",
  "SAMPLE_QUEUE_BYTES",
  "SAMPLE_REPLY_BYTES",
  "TICK_MS",
  "new_task_id",
]
