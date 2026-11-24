"""Single-process controller: relational SQLite state, unified queues, and direct fleet GPU paging."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
import uuid
import weakref
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace

from tinyrl import types
from tinyrl.control import policy
from tinyrl.control.accounting import Charge
from tinyrl.control.sql import DB, decode, encode
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
  parse_uri,
)

LIVE_SECONDS = 30
GC_S = 10
READ_REPLY_TTL = 60
SESSION_GRACE = getenv("SESSION_GRACE", 60)
TICK_MS = getenv("TICK_MS", 100)
REPLY_TTL = getenv("REPLY_TTL", 3600)
SESSION_TTL = getenv("SESSION_TTL", 7 * 24 * 3600)
SAMPLE_QUEUE_BYTES = getenv("SAMPLE_QUEUE_BYTES", 256 * 2**20)
SAMPLE_REPLY_BYTES = getenv("SAMPLE_REPLY_BYTES", 256 * 2**20)
STABILITY_TTL = getenv("STABILITY_TTL", 24 * 3600)
STABILITY_BYTES = getenv("STABILITY_BYTES", 512 * 2**20)
COUNTERS = ("steps", "failed", "cmds", "tokens", "busy_ms", "wait_ms")


def new_task_id() -> str:
  return uuid.uuid4().hex[:12]


def row_to_checkpoint(row: tuple) -> Checkpoint:
  uri, version, size, created, expires_at, owned, meta_blob = row
  return Checkpoint(uri, version, size, created, expires_at, bool(owned), decode(meta_blob))


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
    return [task for task in self.tasks.values() if task.worker == worker_id]

  def wake(self) -> None:
    if self.new_tasks or self.cancelled or self.stage != self.sent_stage:
      self.push_ev.set()

  def drain_push(self, ack: tuple[str, ...] = (), sweep: tuple[str, ...] = ()) -> ControlMsg:
    stage_update = self.stage if self.stage != self.sent_stage else ()
    self.sent_stage = self.stage
    msg = ControlMsg(
      ack=ack,
      tasks=tuple(self.new_tasks),
      cancel=tuple(self.cancelled),
      stage=stage_update,
      sweep=sweep,
    )
    self.new_tasks.clear()
    self.cancelled.clear()
    self.push_ev.clear()
    return msg


class Controller:
  def __init__(self, path: str | None = None):
    db_path = getenv("TINYRL_DB", "/tmp/tinyrl.db") if path is None else path
    self.db: DB | None = DB(db_path)
    self.runs: dict[str, RunState] = {}
    self.nodes: dict[str, NodeState] = {}
    self.service: dict[str, float] = {}
    self.lag: dict[str, float] = {}
    self.seen: dict[str, float] = {}
    self.weights: dict[str, float] = {}
    self.seq = 0
    self.made = 0
    self.sample_q_bytes = 0
    self.sample_r_bytes = 0
    self.gc_due = 0.0
    self.sweep_due = 0.0
    self.lag_due = 0.0
    self.next_wake: float | None = None
    self.accounts_dirty = False
    self.sample_replies: OrderedDict[str, tuple[Reply, float, bool]] = OrderedDict()
    self.sample_seen: dict[tuple[str, int], tuple[str, float]] = {}
    self.events: deque[Event] = deque(maxlen=100_000)
    self.waiters: weakref.WeakValueDictionary[str, asyncio.Event] = weakref.WeakValueDictionary()
    self.restore_from_db()

  def restore_from_db(self) -> None:
    grace_until = time.monotonic() + LEASE_MS / 1000
    for row in self.db.execute("SELECT * FROM runs"):
      run_id, spec_blob, created, status, owner_blob, snapshot, clients, deleted_at, counters_blob = row
      owner = tuple(decode(owner_blob)) if owner_blob else None
      state = RunState(
        run=decode(spec_blob, type=Run),
        created=created,
        status=RunStatus(status),
        owner=owner,
        snapshot=snapshot,
        clients=clients,
        deleted_at=deleted_at,
        counters=decode(counters_blob),
      )
      self.runs[run_id] = state
      if state.live and owner:
        _, node_id, boot_id = owner
        self.nodes.setdefault(node_id, NodeState(node_id, boot_id, gpus=0, until=grace_until))

    for acct, svc, lag, seen, weight in self.db.execute("SELECT * FROM accounts"):
      self.service[acct] = svc
      self.lag[acct] = lag
      self.seen[acct] = seen
      self.weights[acct] = weight

    seq_row = self.db.one("SELECT v FROM meta WHERE k='seq'")
    self.seq = int(seq_row[0]) if seq_row else 0

    query = "SELECT id, run_id, seq, op, cost, uri, stab, claim FROM commands ORDER BY seq"
    for cmd_id, run_id, seq, op, cost, uri, stab_blob, claim_blob in self.db.execute(query):
      self.seq = max(self.seq, seq)
      state = self.runs.get(run_id)
      if state is None or not state.live:
        continue
      stab_vers = tuple(sorted({ref for ref, _, _ in decode(stab_blob).values()})) if stab_blob else ()
      cmd = Cmd(run_id, cmd_id.partition("/")[2], Op(op), b"", cost)
      item = QueueItem(cmd, uri=uri, stab_vers=stab_vers)
      if claim_blob:
        node_id, boot_id, claimed_at, task_blob = decode(claim_blob)
        task = decode(task_blob, type=Task)
        node = self.nodes.setdefault(node_id, NodeState(node_id, boot_id, gpus=0, until=grace_until))
        node.tasks[task.id] = task
        item.task_id = task.id
        item.owner = (task.worker, node_id, boot_id)
        item.since = claimed_at
        worker = node.workers.setdefault(task.worker, LiveWorker(task.worker, task.spec, task.gpus, started=True))
        if not worker.bound and not task.spec.movable:
          worker.bound = task.gpus
      state.queue[cmd_id] = item

  def save_run(self, state: RunState) -> None:
    values = (
      state.run.id,
      encode(state.run),
      state.created,
      str(state.status),
      encode(state.owner) if state.owner else None,
      state.snapshot,
      state.clients,
      state.deleted_at,
      encode(state.counters),
    )
    self.db.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?)", values)

  def save_checkpoint(self, ckpt: Checkpoint) -> None:
    values = (ckpt.uri, ckpt.version, ckpt.size, ckpt.time, ckpt.expires_at, int(ckpt.owned), encode(ckpt.meta))
    self.db.execute("INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?,?,?)", values)

  def flush_sql(self, service: dict[str, float] | None = None) -> None:
    svc = self.service if service is None else service
    accounts = svc.keys() | self.lag.keys() | self.seen.keys() | self.weights.keys()
    for acct in accounts:
      values = (
        acct,
        svc.get(acct, 0.0),
        self.lag.get(acct, 0.0),
        self.seen.get(acct, 0.0),
        self.weights.get(acct, 1.0),
      )
      self.db.execute("INSERT OR REPLACE INTO accounts VALUES (?,?,?,?,?)", values)
    self.db.execute("INSERT OR REPLACE INTO meta VALUES ('seq', ?)", (self.seq,))

  async def aclose(self) -> None:
    if self.db is None:
      return
    with self.db:
      self.flush_sql()
    self.db.close()
    self.db = None

  def notify(self, req_id: str) -> None:
    waiter = self.waiters.pop(req_id, None)
    if waiter is not None:
      waiter.set()

  def live_node(self, owner: Owner | None) -> NodeState | None:
    if owner is None:
      return None
    node = self.nodes.get(owner[1])
    if node is None or node.boot_id != owner[2]:
      return None
    return node

  def record_event(self, kind: Ev, body: dict) -> Event:
    self.seq += 1
    now = time.time()
    event = Event(f"{int(now * 1000)}-{self.seq}", now, str(Ev(kind)), body)
    self.events.append(event)
    return event

  def read_events(self, after: str = "0-0", count: int = 10_000) -> list[Event]:
    cursor = entry_sort_key(after)
    return [ev for ev in self.events if entry_sort_key(ev.id) > cursor][:count]

  def is_sampler(self, run_id: str, req_id: str = "") -> bool:
    state = self.runs.get(run_id)
    if state is not None:
      return state.run.role == Role.SAMPLER
    return req_id.startswith("samp-")

  def account(self, run_id: str) -> str:
    state = self.runs.get(run_id)
    if state is None or not state.live:
      return run_id
    if state.run.account:
      return state.run.account
    if state.run.train_run:
      train_state = self.runs.get(state.run.train_run)
      if train_state is not None and train_state.live:
        return self.account(state.run.train_run)
    return state.run.session or run_id

  def has_live_spec(self, state: RunState) -> bool:
    if not state.live:
      return False
    parent_id = state.run.spec.train_run
    if not parent_id:
      return True
    parent = self.runs.get(parent_id)
    return parent is not None and parent.live

  # ***** sessions, runs, commands, checkpoints *****

  def heartbeat_session(self, session_id: str, metadata: dict | None = None, *, finish: bool = False) -> None:
    now = time.time()
    row = self.db.one("SELECT created, active, meta FROM sessions WHERE id=?", (session_id,))
    if finish and (row is None or not row[1]):
      return
    created = row[0] if row else now
    seen = 0.0 if finish else now
    if not finish and metadata:
      stored_meta = metadata
    elif row:
      stored_meta = decode(row[2])
    else:
      stored_meta = {}
    with self.db:
      self.db.execute(
        "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?)",
        (session_id, created, seen, 1, encode(stored_meta)),
      )

  def get_session(self, session_id: str) -> dict | None:
    row = self.db.one("SELECT created, seen, meta FROM sessions WHERE id=?", (session_id,))
    if row is None:
      return None
    created, seen, meta_blob = row
    runs = sorted(run_id for run_id, state in self.runs.items() if state.run.session == session_id)
    return {
      "session_id": session_id,
      "created": created,
      "seen": seen,
      "live": time.time() - seen < LIVE_SECONDS,
      "runs": runs,
      "metadata": decode(meta_blob),
    }

  def end_session(self, session_id: str) -> None:
    live_runs = sorted(rid for rid, state in self.runs.items() if state.live and state.run.session == session_id)
    queued_deletes = [
      bool(self.submit_command(rid, Op.DELETE_MODEL, b"{}")) for rid in live_runs if not self.is_sampler(rid)
    ]
    for rid in live_runs:
      if self.is_sampler(rid):
        self.delete_run(rid)
    with self.db:
      if any(queued_deletes):
        self.db.execute("UPDATE sessions SET seen=? WHERE id=?", (time.time(), session_id))
      else:
        self.db.execute("UPDATE sessions SET active=0 WHERE id=?", (session_id,))

  def create_run(self, run: Run) -> None:
    state = RunState(run, time.time())
    with self.db:
      self.save_run(state)
    self.runs[run.id] = state
    self.record_event(
      Ev.RUN_CREATED,
      {
        "run": run.id,
        "role": run.role,
        "base": run.base,
        "rank": run.rank,
        "exclusive": run.exclusive,
        "train_run": run.train_run,
        "gpus": run.gpus,
      },
    )
    self.schedule()

  def join_sampling_session(self, run_id: str) -> int:
    state = self.runs.get(run_id)
    if state is None:
      return 1
    next_clients = state.clients + 1
    with self.db:
      self.save_run(replace(state, clients=next_clients))
    state.clients = next_clients
    return next_clients

  def get_run(self, run_id: str) -> dict | None:
    state = self.runs.get(run_id)
    if state is None:
      return None
    info = {**asdict(state.run), "created": state.created, "status": str(state.status), **state.counters}
    if state.owner:
      info["owner"] = list(state.owner)
    if state.snapshot:
      info["snapshot"] = state.snapshot
    return info

  def delete_run(self, run_id: str) -> None:
    state = self.runs.get(run_id)
    if state is not None:
      for cmd_id, item in list(state.queue.items()):
        node = self.live_node(item.owner) if item.task_id else None
        if state.run.role == Role.SAMPLER:
          self.sample_q_bytes -= len(item.cmd.body)
          if node is not None:
            node.tasks.pop(item.task_id, None)
            node.cancelled.append(item.task_id)
            node.wake()
        elif item.owner:
          self.finish_train(cmd_id, item.owner, Reply.error(RuntimeError(f"run {run_id} was deleted")))
          if node is not None:
            node.tasks.pop(item.task_id, None)

      self.sample_seen = {key: val for key, val in self.sample_seen.items() if key[0] != run_id}
      now = time.time()
      dropped = list(state.queue)
      with self.db:
        self.save_run(replace(state, status=RunStatus.DELETED, owner=None, snapshot=None, deleted_at=now))
        self.db.execute("DELETE FROM commands WHERE run_id=?", (run_id,))
      state.status = RunStatus.DELETED
      state.owner = None
      state.snapshot = None
      state.deleted_at = now
      state.queue.clear()
      for cmd_id in dropped:
        self.notify(cmd_id)

    self.record_event(Ev.RUN_DELETED, {"run": run_id})
    self.schedule()

  def seen_request(self, run_id: str, seq_id: int | None) -> str | None:
    if seq_id is None:
      return None
    now = time.time()
    if self.is_sampler(run_id):
      cached = self.sample_seen.get((run_id, seq_id))
      return cached[0] if cached and cached[1] > now else None
    row = self.db.one("SELECT req_id FROM dedup WHERE run_id=? AND seq_id=? AND expires>?", (run_id, seq_id, now))
    return row[0] if row else None

  def submit_command(
    self,
    run_id: str,
    op: Op,
    body: bytes,
    seq_id: int | None = None,
    cost: int = 1,
    uri: str | None = None,
    stability_refs: set[str] | None = None,
  ) -> str | None:
    existing = self.seen_request(run_id, seq_id)
    if existing:
      return existing
    state = self.runs.get(run_id)
    if state is None or not state.live:
      return None

    if not state.queue and state.fair_ms is None:
      acct = self.account(run_id)
      spec = state.run.spec
      peer_service = [
        other.fair_ms
        for other_id, other in self.runs.items()
        if other.live and other.fair_ms is not None and other.run.spec == spec and self.account(other_id) == acct
      ]
      state.fair_ms = min(peer_service, default=0.0)

    self.seq += 1
    now = time.time()
    entry_id = f"{time.time_ns() // 1_000_000}-{self.seq}"
    cmd = Cmd(run_id, entry_id, Op(op), body, cost)
    req_id = cmd.request_id

    if state.run.role == Role.SAMPLER:
      if self.sample_q_bytes + len(body) > SAMPLE_QUEUE_BYTES:
        raise QueueFullError("sampling queue memory limit reached")
      model_path = state.run.model_path
      ckpt = self.get_checkpoint(model_path) if model_path else None
      state.queue[req_id] = QueueItem(cmd, version=ckpt.version if ckpt else None)
      self.sample_q_bytes += len(body)
      if seq_id is not None:
        self.sample_seen[run_id, seq_id] = (req_id, now + SESSION_TTL)
      if model_path and ckpt is None:
        self.finish_sample(req_id, None, Reply.error(ValueError(f"no checkpoint at {model_path}")), None, attempt=0)
    else:
      with self.db:
        stab = self.db.read_stability(stability_refs, now) if stability_refs else None
        stab_blob = encode(stab) if stab else None
        values = (req_id, run_id, self.seq, str(cmd.op), cost, uri, body, stab_blob)
        self.db.execute("INSERT INTO commands VALUES (?,?,?,?,?,?,?,?,NULL)", values)
        if seq_id is not None:
          self.db.execute("INSERT OR REPLACE INTO dedup VALUES (?,?,?,?)", (run_id, seq_id, req_id, now + SESSION_TTL))
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('seq', ?)", (self.seq,))
      stab_vers = tuple(sorted({ref for ref, _, _ in stab.values()})) if stab else ()
      state.queue[req_id] = QueueItem(replace(cmd, body=b""), uri=uri, stab_vers=stab_vers)

    self.schedule()
    return req_id

  def read_reply(self, req_id: str) -> Reply | None:
    now = time.time()
    read_ttl = min(REPLY_TTL, READ_REPLY_TTL)
    run_id = req_id.partition("/")[0]
    if self.is_sampler(run_id, req_id):
      cached = self.sample_replies.get(req_id)
      if cached and cached[1] > now:
        reply, expires, _ = cached
        self.sample_replies.move_to_end(req_id)
        self.sample_replies[req_id] = (reply, min(expires, now + read_ttl), True)
        return reply
      state = self.runs.get(run_id)
      if state is None or req_id not in state.queue:
        raise RequestGoneError(f"sampling request gone: {req_id}")
      return None

    row = self.db.one("SELECT ok, content_type, expires, body FROM replies WHERE id=?", (req_id,))
    if row is None:
      return None
    ok, content_type, expires, body = row
    if expires <= now:
      return None
    with self.db:
      self.db.execute("UPDATE replies SET expires=? WHERE id=?", (min(expires, now + read_ttl), req_id))
    return Reply(bool(ok), content_type, bytes(body))

  async def wait_for_reply(self, req_id: str, timeout: float) -> Reply | None:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
      reply = self.read_reply(req_id)
      if reply is not None:
        return reply
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        return None
      waiter = self.waiters.setdefault(req_id, asyncio.Event())
      with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(waiter.wait(), remaining)

  def queued_and_in_flight(self, run_id: str) -> dict:
    state = self.runs.get(run_id)
    if state is None or not state.live:
      return {"live": False}
    now = time.monotonic() if state.run.role == Role.SAMPLER else time.time()
    in_flight = [
      {
        "request_id": req_id,
        "trainer": item.owner[0],
        "deliveries": 1,
        "seconds": max(0.0, now - (item.since or now)),
      }
      for req_id, item in state.queue.items()
      if item.owner
    ]
    return {"live": True, "queued": len(state.queue) - len(in_flight), "in_flight": in_flight}

  def hydrate_cmd(self, req_id: str) -> Cmd:
    row = self.db.one("SELECT body, stab FROM commands WHERE id=?", (req_id,))
    run_id = req_id.partition("/")[0]
    base_cmd = self.runs[run_id].queue[req_id].cmd
    body = bytes(row[0]) if row else b""
    stability = decode(row[1]) if row and row[1] else None
    return replace(base_cmd, body=body, stability=stability)

  def put_checkpoint(self, ckpt: Checkpoint) -> None:
    with self.db:
      self.save_checkpoint(ckpt)

  def list_checkpoints(self, run: str | None = None) -> list[Checkpoint]:
    rows = self.db.execute("SELECT * FROM checkpoints WHERE expires_at IS NULL OR expires_at>?", (time.time(),))
    found = [row_to_checkpoint(row) for row in rows if run is None or parse_uri(row[0])[0] == run]
    return sorted(found, key=lambda ckpt: ckpt.time, reverse=(run is None))

  def get_checkpoint(self, uri: str) -> Checkpoint | None:
    parse_uri(uri)
    row = self.db.one(
      "SELECT * FROM checkpoints WHERE uri=? AND (expires_at IS NULL OR expires_at>?)",
      (uri, time.time()),
    )
    return row_to_checkpoint(row) if row else None

  def checkpoint_version(self, uri: str) -> str:
    ckpt = self.get_checkpoint(uri)
    if ckpt is None:
      raise ValueError(f"no checkpoint at {uri}")
    return ckpt.version

  def set_checkpoint_ttl(self, uri: str, ttl: int | None) -> bool:
    ckpt = self.get_checkpoint(uri)
    if ckpt is None:
      return False
    expires_at = time.time() + ttl if ttl else None
    self.put_checkpoint(replace(ckpt, expires_at=expires_at))
    return True

  def delete_checkpoint(self, uri: str) -> bool:
    if self.get_checkpoint(uri) is None:
      return False
    with self.db:
      self.db.execute("DELETE FROM checkpoints WHERE uri=?", (uri,))
    return True

  def live_versions(self) -> list[str]:
    versions = {ckpt.version for ckpt in self.list_checkpoints() if ckpt.version}
    for (ref,) in self.db.execute("SELECT DISTINCT ref FROM stability WHERE expires>?", (time.time(),)):
      versions.add(ref)
    for state in self.runs.values():
      if state.live and state.snapshot:
        versions.add(state.snapshot)
      for item in state.queue.values():
        if item.version:
          versions.add(item.version)
        versions.update(item.stab_vers)
    for node in self.nodes.values():
      for task in node.tasks.values():
        if task.version:
          versions.add(task.version)
        if task.snapshot:
          versions.add(task.snapshot)
    return sorted({ver.partition("@")[0] for ver in versions})

  # ***** accounting & completion *****

  def apply_charge(
    self,
    state: RunState,
    cmd: Cmd,
    reply: Reply,
    charge: Charge | None,
    counters: dict[str, int],
    service: dict[str, float],
  ) -> float | None:
    if not reply.ok:
      counters["failed"] += 1
    elif cmd.op == Op.OPTIM_STEP:
      counters["steps"] += 1

    fair_ms = state.fair_ms
    if charge is not None:
      if fair_ms is not None:
        fair_ms += charge.gpu_ms
      counters["cmds"] += 1
      counters["busy_ms"] += charge.busy_ms
      submitted_ms = entry_sort_key(cmd.entry_id)[0]
      counters["wait_ms"] += max(0, charge.started_ms - submitted_ms)
      if reply.ok:
        counters["tokens"] += cmd.cost
      if charge.gpu_ms:
        acct = self.account(cmd.run)
        if acct:
          service[acct] = service.get(acct, 0.0) + charge.gpu_ms
    return fair_ms

  def finish_train(
    self,
    req_id: str,
    owner: Owner | None,
    reply: Reply,
    charge: Charge | None = None,
    *,
    mutated: bool = False,
    checkpoint: Checkpoint | None = None,
    dead: bool = False,
  ) -> bool:
    state = self.runs.get(req_id.partition("/")[0])
    item = state.queue.get(req_id) if state else None
    if state is None or item is None or item.owner != owner:
      return False

    cmd = item.cmd
    now = time.time()
    deleting = cmd.op == Op.DELETE_MODEL and reply.ok
    counters = dict(state.counters)
    service = dict(self.service)
    fair_ms = self.apply_charge(state, cmd, reply, charge, counters, service)

    next_state = replace(state, counters=counters, fair_ms=None if len(state.queue) <= 1 else fair_ms)
    peer_updates: list[RunState] = []
    if deleting:
      next_state.status = RunStatus.DELETED
      next_state.owner = None
      next_state.snapshot = None
      next_state.deleted_at = now
    else:
      if mutated:
        next_state.snapshot = None
        next_state.status = RunStatus.LIVE
      if dead and owner:
        next_state.owner = None
        if next_state.status == RunStatus.LIVE:
          next_state.status = RunStatus.LOST
        for peer in self.runs.values():
          if peer.run.id != cmd.run and peer.owner == owner:
            peer_status = RunStatus.LOST if peer.status == RunStatus.LIVE else peer.status
            peer_updates.append(replace(peer, owner=None, status=peer_status))
      elif dead or next_state.status in (RunStatus.FRESH, RunStatus.LOST):
        next_state.owner = None

    with self.db:
      if deleting:
        self.db.execute("DELETE FROM commands WHERE run_id=?", (cmd.run,))
      else:
        self.db.execute("DELETE FROM commands WHERE id=?", (req_id,))
      self.db.execute(
        "INSERT OR REPLACE INTO replies VALUES (?,?,?,?,?)",
        (req_id, int(reply.ok), reply.content_type, now + REPLY_TTL, reply.body),
      )
      self.flush_sql(service)
      if checkpoint and reply.ok:
        self.save_checkpoint(checkpoint)
      if deleting:
        self.db.execute("DELETE FROM checkpoints WHERE owned=1 AND uri LIKE ?", (f"tinker://{cmd.run}/%",))
      for updated in (*peer_updates, next_state):
        self.save_run(updated)

    self.service = service
    for updated in peer_updates:
      peer = self.runs[updated.run.id]
      peer.owner = updated.owner
      peer.status = updated.status
    state.status = next_state.status
    state.owner = next_state.owner
    state.snapshot = next_state.snapshot
    state.deleted_at = next_state.deleted_at
    state.counters = counters
    state.fair_ms = next_state.fair_ms

    finished_ids = list(state.queue) if deleting else [req_id]
    for cid in finished_ids:
      state.queue.pop(cid, None)
      self.notify(cid)
    if deleting:
      self.record_event(Ev.RUN_DELETED, {"run": cmd.run})
    return True

  def trim_sample_replies(self, keep_id: str) -> None:
    while self.sample_r_bytes > SAMPLE_REPLY_BYTES and len(self.sample_replies) > 1:
      victim = next((k for k, (rep, _, was_read) in self.sample_replies.items() if was_read and rep.body), None)
      if victim is None:
        victim = next((k for k, (rep, _, _) in self.sample_replies.items() if k != keep_id and rep.body), None)
      if victim is None:
        return
      evicted, _, _ = self.sample_replies.pop(victim)
      self.sample_r_bytes -= len(evicted.body)

  def finish_sample(
    self,
    req_id: str,
    owner: Owner | None,
    reply: Reply,
    charge: Charge | None,
    attempt: int,
  ) -> bool:
    state = self.runs.get(req_id.partition("/")[0])
    item = state.queue.get(req_id) if state else None
    if state is None or item is None or item.owner != owner or item.attempt != attempt:
      return False

    if reply.ok and reply.stability:
      with self.db:
        self.db.record_stability(reply.stability, time.time() + STABILITY_TTL, STABILITY_BYTES)

    state.queue.pop(req_id)
    self.sample_q_bytes -= len(item.cmd.body)

    stored_reply = replace(reply, stability=None)
    self.sample_replies[req_id] = (stored_reply, time.time() + REPLY_TTL, False)
    self.sample_r_bytes += len(stored_reply.body)
    self.trim_sample_replies(req_id)

    fair_ms = self.apply_charge(state, item.cmd, stored_reply, charge, state.counters, self.service)
    state.fair_ms = fair_ms if state.queue else None
    if charge and charge.gpu_ms:
      self.accounts_dirty = True
    self.notify(req_id)
    return True

  def reclaim_samples(self, predicate: Callable[[QueueItem], bool]) -> None:
    for state in self.runs.values():
      if state.run.role != Role.SAMPLER:
        continue
      for item in state.queue.values():
        if not item.owner or not predicate(item):
          continue
        node = self.live_node(item.owner) if item.task_id else None
        if node is not None:
          node.tasks.pop(item.task_id, None)
          node.cancelled.append(item.task_id)
        item.attempt += 1
        item.owner = None
        item.task_id = None
        item.since = 0.0

  def clear_owners(self, predicate: Callable[[RunState], bool], *, lose: bool = True) -> None:
    updates: list[RunState] = []
    for state in self.runs.values():
      if state.owner and predicate(state):
        next_status = RunStatus.LOST if (lose and state.status == RunStatus.LIVE) else state.status
        updates.append(replace(state, owner=None, status=next_status))
    if not updates:
      return
    with self.db:
      for updated in updates:
        self.save_run(updated)
    for updated in updates:
      state = self.runs[updated.run.id]
      state.owner = updated.owner
      state.status = updated.status

  # ***** node liveness & protocol *****

  def live_nodes(self) -> dict[str, int]:
    self.expire_nodes()
    return {node_id: node.gpus for node_id, node in self.nodes.items() if node.gpus}

  def expire_node(self, node_id: str, boot_id: str) -> None:
    current = self.nodes.get(node_id)
    if current is not None and current.boot_id == boot_id:
      del self.nodes[node_id]
    self.record_event(Ev.NODE_LOST, {"node": node_id})

    for state in list(self.runs.values()):
      if state.run.role != Role.TRAINER:
        continue
      for cmd_id, item in list(state.queue.items()):
        if item.owner and item.owner[1:] == (node_id, boot_id):
          err = Reply.error(RuntimeError(f"its trainer died, or lost ownership, during {item.cmd.op}"))
          self.finish_train(cmd_id, item.owner, err, dead=True)

    with self.db:
      self.db.add_fenced(boot_id)
    self.reclaim_samples(lambda item: item.owner[1:] == (node_id, boot_id))
    self.clear_owners(lambda state: state.owner[1:] == (node_id, boot_id))

  def expire_nodes(self) -> bool:
    now = time.monotonic()
    stale = [(nid, node.boot_id) for nid, node in list(self.nodes.items()) if node.until <= now]
    for node_id, boot_id in stale:
      self.expire_node(node_id, boot_id)
    return bool(stale)

  def drop_worker(self, node: NodeState, worker_id: str) -> None:
    worker = node.workers.pop(worker_id, None)
    owner = (worker_id, node.id, node.boot_id)
    self.reclaim_samples(lambda item: item.owner == owner)
    self.clear_owners(lambda state: state.owner == owner)
    if worker and worker.started:
      self.record_event(Ev.UNPLACED, {"proc": worker_id, "node": node.id})

  def apply_done(self, node: NodeState, done: Done, now: float) -> None:
    task = node.tasks.get(done.id)
    if task is None:
      return
    worker = node.workers.get(task.worker)
    owner = (task.worker, node.id, node.boot_id)
    failed = done.dead or not done.ok

    if worker and done.pid is not None:
      worker.pid = done.pid
      worker.started = True

    if task.kind == TaskKind.SAVE and not failed:
      state = self.runs.get(task.run_id)
      if state is not None and state.owner == owner:
        if done.saved:
          next_status = RunStatus.SAVED
        elif state.status == RunStatus.LIVE:
          next_status = RunStatus.LOST
        else:
          next_status = state.status
        updated = replace(state, owner=None, snapshot=done.saved or state.snapshot, status=next_status)
        with self.db:
          self.save_run(updated)
        state.owner = updated.owner
        state.snapshot = updated.snapshot
        state.status = updated.status
      node.tasks.pop(done.id, None)
      if worker:
        worker.failed_saves.discard(task.run_id)
        if not worker.stopping and worker.started and task.run_id in worker.loaded:
          self.issue(node, Task(new_task_id(), TaskKind.DROP, worker.id, worker.spec, run_id=task.run_id), now)
      return

    if task.kind == TaskKind.SAVE:
      if done.dead:
        self.drop_worker(node, task.worker)
      elif worker:
        worker.failed_saves.add(task.run_id)
        worker.retry_save_at = now + 0.05
    elif task.kind in (TaskKind.TRAIN, TaskKind.SAMPLE):
      effect = done.effect or Effect()
      reply = done.reply or Reply.error(RuntimeError(f"{task.kind} failed"))
      if task.kind == TaskKind.TRAIN:
        self.finish_train(
          task.cmd.request_id,
          owner,
          reply,
          done.charge,
          mutated=effect.mutated,
          checkpoint=effect.checkpoint,
          dead=failed,
        )
      elif failed and done.reply is None:
        self.reclaim_samples(lambda it: it.cmd.request_id == task.cmd.request_id and it.owner == owner)
      else:
        self.finish_sample(task.cmd.request_id, owner, reply, done.charge, task.attempt)

      if failed:
        self.drop_worker(node, task.worker)
      elif worker:
        worker.started = True
        if done.charge:
          worker.gpu_ms += done.charge.gpu_ms
        if done.duration_s > 0 and worker.park_s > 0:
          hold_end = worker.resumed_at + policy.quantum_s(worker.park_s + done.duration_s)
          worker.hold_until = max(worker.hold_until, hold_end)
        if task.kind == TaskKind.TRAIN:
          worker.idle_until = now + policy.CYCLE_GRACE_S
          worker.loaded.pop(task.cmd.run, None)
          if effect.loaded:
            worker.loaded[task.cmd.run] = effect.open_grad
    elif task.kind == TaskKind.PARK:
      if failed:
        self.drop_worker(node, task.worker)
      elif worker:
        worker.gpus = ()
        worker.park_s = done.duration_s
        worker.hold_until = 0.0
        worker.idle_until = 0.0
    elif task.kind == TaskKind.DROP:
      if done.dead:
        self.drop_worker(node, task.worker)
      elif worker:
        worker.loaded.pop(task.run_id, None)
    elif task.kind == TaskKind.STOP:
      self.drop_worker(node, task.worker)

    node.tasks.pop(done.id, None)

  def reconcile_hello(self, existing: NodeState | None, msg: NodeMsg) -> NodeState:
    if existing is not None and existing.boot_id != msg.boot_id:
      self.expire_node(existing.id, existing.boot_id)
      existing = None
    was_up = bool(existing and existing.gpus)
    node = self.nodes.setdefault(msg.node, NodeState(msg.node, msg.boot_id, msg.gpus, msg.gpu_mib))
    if not was_up:
      self.record_event(Ev.NODE_UP, {"node": msg.node, "gpus": msg.gpus, "memory": msg.gpu_mib * 2**20})
      for task_id, task in list(node.tasks.items()):
        if task.cmd and not task.cmd.body:
          node.tasks[task_id] = replace(task, cmd=self.hydrate_cmd(task.cmd.request_id))

    node.gpu_mib = msg.gpu_mib
    node.staged = set(msg.staged or ())
    node.sent_stage = ()
    node.new_tasks.clear()
    node.cancelled.clear()

    for info in msg.workers:
      worker = node.workers.get(info.worker)
      if worker is not None:
        worker.pid = info.pid
        worker.started = True
        worker.bound = info.bound or worker.bound
        if not node.worker_tasks(info.worker):
          worker.gpus = info.gpus
          worker.loaded = dict(info.loaded)
      else:
        node.workers[info.worker] = LiveWorker(
          info.worker, info.spec, info.gpus, info.bound, dict(info.loaded), info.pid, started=True
        )

    alive_ids = {info.worker for info in msg.workers} | {task.worker for task in node.tasks.values() if task.worker}
    for wid, worker in list(node.workers.items()):
      if wid not in alive_ids and worker.started:
        self.drop_worker(node, wid)
    self.clear_owners(lambda state: state.owner[1:] == (node.id, node.boot_id) and state.owner[0] not in node.workers)
    return node

  def handle(self, msg: NodeMsg) -> ControlMsg:
    self.expire_nodes()
    now = time.monotonic()
    if self.db.is_fenced(msg.boot_id):
      return ControlMsg(reset=True)

    node = self.nodes.get(msg.node)
    is_hello = msg.kind == "hello" or node is None
    lost_stage = False

    if is_hello:
      node = self.reconcile_hello(node, msg)
    elif node.boot_id != msg.boot_id:
      return ControlMsg(reset=True)
    else:
      if msg.staged is not None:
        reported_stage = set(msg.staged)
        lost_stage = bool(node.staged - reported_stage)
        node.staged = reported_stage
        node.sent_stage = ()
      for wid in msg.exited:
        if wid in node.workers and not node.worker_tasks(wid):
          self.drop_worker(node, wid)

    node.gpus = msg.gpus or node.gpus
    node.until = now + LEASE_MS / 1000
    node.lag = msg.lag
    node.draining = msg.draining

    for done in msg.done:
      self.apply_done(node, done, now)

    active_ids = set(msg.active)
    if is_hello or msg.draining or lost_stage:

      def should_reclaim(item: QueueItem) -> bool:
        if item.owner[1:] != (node.id, node.boot_id) or item.task_id in active_ids:
          return False
        return node.draining or (item.version is not None and item.version not in node.staged)

      self.reclaim_samples(should_reclaim)

    self.schedule()

    sweep: tuple[str, ...] = ()
    if msg.want_sweep and not node.draining and now >= self.sweep_due:
      self.sweep_due = now + SWEEP_S
      sweep = tuple(self.live_versions())

    ack = tuple(done.id for done in msg.done)
    pushed = node.drain_push(ack=ack, sweep=sweep)
    if is_hello:
      replay = tuple(task for task in node.tasks.values() if task.id not in active_ids)
      return replace(pushed, tasks=replay)
    return pushed

  # ***** direct fleet pager & GPU scheduler *****

  def demand_runs(self) -> list[tuple[str, WorkerSpec, int]]:
    return [
      (self.account(rid), state.run.spec, len(state.queue))
      for rid, state in self.runs.items()
      if self.has_live_spec(state)
    ]

  def advance_lag(self) -> None:
    healthy = {nid: node.gpus for nid, node in self.nodes.items() if node.gpus and not node.draining}
    demand = policy.account_demand(self.demand_runs(), healthy)
    capacity = float(sum(healthy.values()))
    next_lag, next_seen = policy.update_lag(self.lag, self.seen, self.service, capacity, demand, self.weights)
    if next_lag != self.lag or next_seen != self.seen:
      self.lag = next_lag
      self.seen = next_seen
      self.accounts_dirty = True

  def ready_lag(self, run_id: str) -> float:
    return policy.ready_lag(self.lag, self.seen, self.service, self.account(run_id))

  def issue(self, node: NodeState, task: Task, now: float) -> None:
    owner = (task.worker, node.id, node.boot_id)
    if task.kind == TaskKind.TRAIN:
      state = self.runs[task.cmd.run]
      item = state.queue[task.cmd.request_id]
      claimed_at = time.time()
      slim_task = replace(task, cmd=replace(task.cmd, body=b"", stability=None))
      claim_blob = encode((node.id, node.boot_id, claimed_at, encode(slim_task)))
      with self.db:
        self.db.execute("UPDATE commands SET claim=? WHERE id=?", (claim_blob, task.cmd.request_id))
        self.save_run(replace(state, owner=owner))
      state.owner = owner
      item.task_id = task.id
      item.owner = owner
      item.since = claimed_at
    elif task.kind == TaskKind.SAMPLE:
      item = self.runs[task.cmd.run].queue[task.cmd.request_id]
      item.task_id = task.id
      item.owner = owner
      item.since = now

    worker = node.workers.get(task.worker)
    if task.gpus and worker is not None:
      if worker.gpus != task.gpus:
        worker.resumed_at = now
        worker.hold_until = now + policy.QUANTUM_MS / 1000
      worker.gpus = task.gpus
      if not worker.bound and not worker.spec.movable:
        worker.bound = task.gpus

    node.tasks[task.id] = task
    node.new_tasks.append(task)

  def detach_run(self, node: NodeState, worker: LiveWorker, run_id: str, now: float) -> None:
    if self.runs[run_id].status == RunStatus.LIVE and run_id in worker.loaded:
      task = Task(new_task_id(), TaskKind.SAVE, worker.id, worker.spec, run_id=run_id, version=uuid.uuid4().hex)
      self.issue(node, task, now)
      return
    self.clear_owners(lambda state: state.run.id == run_id, lose=False)
    if run_id in worker.loaded and worker.started:
      self.issue(node, Task(new_task_id(), TaskKind.DROP, worker.id, worker.spec, run_id=run_id), now)

  def evict_worker(self, node: NodeState, worker: LiveWorker, now: float) -> None:
    worker.stopping = True
    owner = (worker.id, node.id, node.boot_id)
    dirty_runs = {
      rid
      for rid, state in self.runs.items()
      if state.live and state.owner == owner and state.status == RunStatus.LIVE and rid in worker.loaded
    }
    self.clear_owners(lambda state: state.owner == owner and state.run.id not in dirty_runs, lose=False)
    pending_saves = sorted(dirty_runs - worker.failed_saves)
    if pending_saves:
      task = Task(
        new_task_id(), TaskKind.SAVE, worker.id, worker.spec, run_id=pending_saves[0], version=uuid.uuid4().hex
      )
      self.issue(node, task, now)
    elif not dirty_runs:
      if worker.started:
        self.issue(node, Task(new_task_id(), TaskKind.STOP, worker.id, worker.spec), now)
      else:
        node.workers.pop(worker.id, None)

  def train_head(self, run_id: str) -> QueueItem | None:
    state = self.runs.get(run_id)
    if state is None or not state.live or state.run.role != Role.TRAINER or not state.queue:
      return None
    head = next(iter(state.queue.values()))
    return head if head.owner is None else None

  def ready_samples(self, spec: WorkerSpec, staged: set[str]) -> list[QueueItem]:
    ready: list[QueueItem] = []
    for state in self.runs.values():
      if not state.live or state.run.role != Role.SAMPLER or state.run.spec != spec:
        continue
      for item in state.queue.values():
        if item.owner is None and (item.version is None or item.version in staged):
          ready.append(item)
    return ready

  def settle_trivial_heads(self) -> None:
    for run_id, state in list(self.runs.items()):
      while (head := self.train_head(run_id)) is not None:
        op = head.cmd.op
        req_id = head.cmd.request_id
        if op == Op.LOAD_WEIGHTS and self.get_checkpoint(head.uri or "") is None:
          self.finish_train(req_id, None, Reply.error(ValueError(f"no checkpoint at {head.uri}")))
        elif state.owner is None and state.status == RunStatus.LOST and op not in (Op.CREATE_MODEL, Op.LOAD_WEIGHTS):
          if op == Op.DELETE_MODEL:
            reply = Reply(True, "application/json", f'{{"type":"delete_model","model_id":"{run_id}"}}'.encode())
          else:
            err = KeyError(f"model {run_id} lost its unsaved steps when its trainer stopped; call load_state")
            reply = Reply.error(err)
          self.finish_train(req_id, None, reply)
        else:
          break

  def evict_excess_workers(
    self,
    healthy_nodes: dict[str, NodeState],
    healthy: dict[str, int],
    sampler_runs: dict[WorkerSpec, list[str]],
    sampler_specs: list[WorkerSpec],
    sampler_want: dict[WorkerSpec, int],
    trainer_runs: dict[tuple[str, str], list[str]],
    placed: dict[tuple[str, str], policy.Placed],
    now: float,
  ) -> None:
    def sampler_rank(loc: tuple[str, str]) -> tuple:
      node = healthy_nodes[loc[0]]
      return (not node.workers[loc[1]].started, not node.worker_tasks(loc[1]), loc)

    keep_samplers = {
      loc
      for spec in sampler_specs
      for loc in sorted((k for k, p in placed.items() if p.spec == spec), key=sampler_rank)[: sampler_want[spec]]
    }

    saving = sum(1 for node in healthy_nodes.values() for task in node.tasks.values() if task.kind == TaskKind.SAVE)
    for node_id, node in healthy_nodes.items():
      exclusive_gpus = {gpu for w in node.workers.values() if not w.stopping and w.spec.exclusive for gpu in w.gpus}
      for worker in sorted(list(node.workers.values()), key=lambda w: w.id):
        if node.worker_tasks(worker.id):
          continue
        if worker.stopping:
          self.evict_worker(node, worker, now)
          continue

        loc = (node_id, worker.id)
        is_trainer = worker.spec.role == Role.TRAINER
        owned = trainer_runs[loc] if is_trainer else sampler_runs.get(worker.spec, ())
        ready = [rid for rid in owned if self.runs[rid].queue]
        all_placed = list(placed.values())

        can_move = worker.spec.gpus == 1 or any(
          policy.can_host(other, healthy, all_placed, worker.spec) for other in healthy if other != node_id
        )
        overcommitted = not policy.can_host(node_id, healthy, all_placed, worker.spec, skip=worker.id) and can_move
        invalid = worker.spec.gpus > node.gpus or overcommitted

        can_rebind = not is_trainer or (
          bool(ready) and any(exclusive_gpus.isdisjoint(b) for b in policy.blocks(node.gpus, worker.spec.gpus))
        )
        pinned_under_exclusive = (
          not worker.gpus and bool(worker.bound) and not exclusive_gpus.isdisjoint(worker.bound) and can_rebind
        )
        unneeded = (not owned and worker.started) if is_trainer else (loc not in keep_samplers)
        should_drop = invalid or pinned_under_exclusive or unneeded

        if not should_drop and is_trainer and worker.spec.shareable:
          stats = policy.node_stats(node_id, node.gpus, all_placed)
          has_peer = any(p.spec == worker.spec and other_loc != loc for other_loc, p in placed.items())
          fleet_full = stats.spare < 0 and not any(
            policy.node_stats(m, g, all_placed).spare >= worker.spec.gpus for m, g in healthy.items()
          )
          should_drop = has_peer and ((not ready and stats.used > node.gpus) or fleet_full)
          if not should_drop and len(ready) >= 2 and not saving:
            if policy.choose_node(worker.spec, healthy, all_placed, only_spare=True):
              self.detach_run(node, worker, ready[-1], now)
              saving += 1
              continue

        if not should_drop and is_trainer and not saving:
          if policy.rebalance_target(placed[loc], healthy, all_placed):
            should_drop = True
            saving += 1

        if should_drop:
          placed.pop(loc, None)
          trainer_runs.pop(loc, None)
          self.evict_worker(node, worker, now)

    for (node_id, wid), runs in list(trainer_runs.items()):
      node = healthy_nodes[node_id]
      if not runs and not node.workers[wid].started and not node.worker_tasks(wid):
        del node.workers[wid]
        del trainer_runs[node_id, wid]
        del placed[node_id, wid]

  def place_missing_workers(
    self,
    healthy_nodes: dict[str, NodeState],
    healthy: dict[str, int],
    sampler_runs: dict[WorkerSpec, list[str]],
    sampler_specs: list[WorkerSpec],
    sampler_want: dict[WorkerSpec, int],
    trainer_runs: dict[tuple[str, str], list[str]],
    placed: dict[tuple[str, str], policy.Placed],
  ) -> None:
    sampled_trainers = {state.run.train_run for state in self.runs.values() if state.live and state.run.train_run}

    def pick_node(spec: WorkerSpec, **kwargs) -> str | None:
      owed: dict[str, int] = {}
      for loc, p in placed.items():
        runs = trainer_runs.get(loc)
        if p.spec.role == Role.TRAINER and not p.spec.lora and runs and not (sampled_trainers & set(runs)):
          owed[p.node] = owed.get(p.node, 0) + p.spec.gpus
      return policy.choose_node(spec, healthy, list(placed.values()), owed_gpus=owed, **kwargs)

    def spawn_placeholder(node_id: str, spec: WorkerSpec, busy: bool, lag: float = 0.0, runs: int = 0) -> str:
      used_ids = {wid for node in self.nodes.values() for wid in node.workers}
      self.made = next(i for i in itertools.count(self.made + 1) if f"w{i}" not in used_ids)
      wid = f"w{self.made}"
      healthy_nodes[node_id].workers[wid] = LiveWorker(wid, spec)
      placed[node_id, wid] = policy.Placed(node_id, wid, spec, busy=busy, lag=lag, runs=runs)
      self.record_event(Ev.PLACED, {"proc": wid, "node": node_id})
      return wid

    for run_id, state in sorted(self.runs.items()):
      if state.owner is not None or self.train_head(run_id) is None:
        continue
      spec = state.run.spec
      compatible = [
        loc for loc in sorted(trainer_runs) if spec.shareable and healthy_nodes[loc[0]].workers[loc[1]].spec == spec
      ]
      idle_compatible = [loc for loc in compatible if not any(self.runs[r].queue for r in trainer_runs[loc])]
      spare_node = pick_node(spec, only_spare=True) if spec.shareable else None
      if compatible and (idle_compatible or not spare_node):
        chosen = min(idle_compatible or compatible, key=lambda loc: (len(trainer_runs[loc]), loc))
        trainer_runs[chosen].append(run_id)
        current = placed[chosen]
        placed[chosen] = replace(
          current,
          busy=True,
          lag=max(current.lag, self.ready_lag(run_id)),
          runs=len(trainer_runs[chosen]),
        )
      else:
        target_node = spare_node or pick_node(spec, prefer_cool=True)
        if target_node:
          wid = spawn_placeholder(target_node, spec, busy=True, lag=self.ready_lag(run_id), runs=1)
          trainer_runs[target_node, wid] = [run_id]

    needed_stage: dict[str, set[str]] = {}
    for spec in sampler_specs:
      parent_runs = (self.runs.get(self.runs[rid].run.train_run or "") for rid in sampler_runs[spec])
      near = {parent.owner[1] for parent in parent_runs if parent and parent.owner and parent.owner[1] in healthy}
      pool = sorted(loc for loc, p in placed.items() if p.spec == spec)
      if not pool:
        target_node = pick_node(spec, near=near)
        if target_node:
          has_work = any(self.runs[rid].queue for rid in sampler_runs[spec])
          pool.append((target_node, spawn_placeholder(target_node, spec, busy=has_work)))
      while len(pool) < sampler_want[spec]:
        target_node = pick_node(spec, near=near, only_spare=True)
        if not target_node:
          break
        pool.append((target_node, spawn_placeholder(target_node, spec, busy=True)))

      queued_versions = {
        item.version
        for rid in sampler_runs[spec]
        for item in self.runs[rid].queue.values()
        if item.owner is None and item.version
      }
      for node_id, _ in pool:
        missing = queued_versions - healthy_nodes[node_id].staged
        needed_stage.setdefault(node_id, set()).update(missing)

    for node_id, versions in needed_stage.items():
      healthy_nodes[node_id].stage = tuple(sorted(versions))

  def plan_node_workers(
    self,
    node_id: str,
    node: NodeState,
    trainer_runs: dict[tuple[str, str], list[str]],
    lag_of_run: Callable[[str], float],
    now: float,
    deadlines: list[float],
  ) -> tuple[dict[str, tuple[int, ...]], float | None, list[tuple]]:
    fixed: dict[str, tuple[int, ...]] = {}
    occupied: set[int] = set()
    plan_inputs: dict[str, policy.Worker] = {}
    ready_by_worker: dict[str, set[str]] = {}

    for wid, worker in node.workers.items():
      active = node.worker_tasks(wid)
      if worker.stopping or any(t.kind not in (TaskKind.TRAIN, TaskKind.SAMPLE) for t in active):
        occupied.update(worker.gpus)
        continue
      if active:
        occupied.update(worker.gpus)
        fixed[wid] = worker.gpus
        if worker.spec.role == Role.SAMPLER and len(active) < types.MAX_SEQS:
          ready_by_worker[wid] = {it.cmd.run for it in self.ready_samples(worker.spec, node.staged)}
        continue

      if worker.spec.role == Role.TRAINER:
        heads = {rid: h for rid in trainer_runs.get((node_id, wid), ()) if (h := self.train_head(rid)) is not None}
        in_grace = bool(worker.gpus and worker.idle_until > now)
        if in_grace:
          deadlines.append(worker.idle_until)
        finishing_step = any(
          open_grad and ((rid in heads and heads[rid].cmd.op == Op.OPTIM_STEP) or (not heads and in_grace))
          for rid, open_grad in worker.loaded.items()
        )
        if finishing_step:
          demand = policy.Demand.FINISH_STEP
        elif heads or in_grace:
          demand = policy.Demand.READY
        else:
          demand = policy.Demand.IDLE
        ready_runs = set(heads)
        gang = worker.spec.gpus > 1
      else:
        ready_runs = {it.cmd.run for it in self.ready_samples(worker.spec, node.staged)}
        demand = policy.Demand.READY if ready_runs else policy.Demand.IDLE
        gang = False

      ready_by_worker[wid] = ready_runs
      plan_inputs[wid] = policy.Worker(
        gpus=worker.spec.gpus,
        exclusive=worker.spec.exclusive,
        demand=demand,
        current=worker.gpus,
        bound=worker.bound,
        gang=gang,
        hold_until=worker.hold_until,
        lag=max(map(lag_of_run, ready_runs), default=-worker.gpu_ms),
        gpu_ms=worker.gpu_ms,
      )

    locked = {gpu for block in fixed.values() for gpu in block}
    chosen, wake_at = policy.plan_gpus(plan_inputs, node.gpus, now, occupied, locked)
    keep = fixed | chosen
    contended = any(pw.demand != policy.Demand.IDLE and wid not in keep for wid, pw in plan_inputs.items())

    candidates: list[tuple] = []
    for wid, block in keep.items():
      worker = node.workers[wid]
      ready_runs = ready_by_worker.get(wid)
      if not block or not ready_runs or (wid in fixed and contended):
        continue
      if worker.gpus and worker.gpus != block:
        continue
      if any(not set(other.gpus).isdisjoint(block) for other_id, other in node.workers.items() if other_id != wid):
        continue
      priority = (
        worker.spec.exclusive,
        round(max(map(lag_of_run, ready_runs)), 3),
        round(-worker.gpu_ms, 3),
        bool(worker.gpus),
        node_id,
        wid,
      )
      candidates.append((priority, node, worker, block))
    return keep, wake_at, candidates

  def schedule(self) -> None:
    """Single pass: settle trivial heads, drain, page out, page in + stage, grant GPUs, and park idle workers."""
    now = time.monotonic()
    deadlines: list[float] = []
    self.settle_trivial_heads()

    if now >= self.lag_due:
      self.lag_due = now + TICK_MS / 1000
      self.advance_lag()

    for node in self.nodes.values():
      node.stage = ()
      for worker in list(node.workers.values()):
        if worker.failed_saves and now >= worker.retry_save_at:
          worker.failed_saves.clear()
        elif worker.failed_saves:
          deadlines.append(worker.retry_save_at)
      if node.draining:
        for worker in list(node.workers.values()):
          if not node.worker_tasks(worker.id):
            self.evict_worker(node, worker, now)

    healthy_nodes = {nid: node for nid, node in sorted(self.nodes.items()) if node.gpus and not node.draining}
    healthy = {nid: node.gpus for nid, node in healthy_nodes.items()}
    if not healthy:
      self.next_wake = min(deadlines, default=None)
      for node in self.nodes.values():
        node.wake()
      return

    live_workers = [(nid, w) for nid, node in healthy_nodes.items() for w in node.workers.values() if not w.stopping]
    sampler_runs: dict[WorkerSpec, list[str]] = {}
    trainer_runs: dict[tuple[str, str], list[str]] = {
      (nid, w.id): [] for nid, w in live_workers if w.spec.role == Role.TRAINER
    }
    for run_id, state in sorted(self.runs.items()):
      if state.run.role == Role.SAMPLER and self.has_live_spec(state):
        sampler_runs.setdefault(state.run.spec, []).append(run_id)
      elif state.live and state.owner:
        wid, nid, boot_id = state.owner
        if (nid, wid) in trainer_runs and healthy_nodes[nid].boot_id == boot_id:
          trainer_runs[nid, wid].append(run_id)

    sampler_specs = sorted(sampler_runs, key=lambda s: (s.base, s.lora, s.exclusive, s.train_run or "", s.gpus))
    sampler_want = {
      spec: policy.sampler_replicas(spec, [len(self.runs[rid].queue) for rid in sampler_runs[spec]])
      for spec in sampler_specs
    }

    placed: dict[tuple[str, str], policy.Placed] = {}
    for nid, worker in live_workers:
      owned = sampler_runs.get(worker.spec, ()) if worker.spec.role == Role.SAMPLER else trainer_runs[nid, worker.id]
      ready = [rid for rid in owned if self.runs[rid].queue]
      placed[nid, worker.id] = policy.Placed(
        nid,
        worker.id,
        worker.spec,
        busy=bool(ready),
        lag=max(map(self.ready_lag, ready), default=0.0),
        runs=len(owned),
      )

    self.evict_excess_workers(
      healthy_nodes, healthy, sampler_runs, sampler_specs, sampler_want, trainer_runs, placed, now
    )
    self.place_missing_workers(healthy_nodes, healthy, sampler_runs, sampler_specs, sampler_want, trainer_runs, placed)

    fair = {rid: state.fair_ms or 0.0 for rid, state in self.runs.items() if state.live and state.queue}
    demand_runs = self.demand_runs()
    holds: dict[tuple[str, str], tuple[str, int]] = {}
    for nid, node in healthy_nodes.items():
      for task in node.tasks.values():
        if task.kind in (TaskKind.TRAIN, TaskKind.SAMPLE) and task.cmd:
          holds.setdefault((nid, task.worker), (self.account(task.cmd.run), len(task.gpus) or task.spec.gpus))

    keeps: dict[str, dict[str, tuple[int, ...]]] = {}
    wakes: dict[str, float | None] = {}
    while True:
      lag_of_acct = policy.projected_lag(
        self.lag, self.seen, self.service, self.weights, demand_runs, healthy, list(holds.values())
      )

      def lag_of_run(run_id: str, fn=lag_of_acct) -> float:
        return fn(self.account(run_id))

      candidates: list[tuple] = []
      keeps, wakes = {}, {}
      for nid, node in healthy_nodes.items():
        keeps[nid], wakes[nid], node_cands = self.plan_node_workers(nid, node, trainer_runs, lag_of_run, now, deadlines)
        candidates.extend(node_cands)

      candidates.sort(key=lambda entry: entry[0], reverse=True)
      granted = False
      for _, node, worker, block in candidates:
        runs = trainer_runs.get((node.id, worker.id), ())
        if self.grant(node, worker, block, runs, fair, lag_of_run, now):
          last_run = node.new_tasks[-1].cmd.run
          holds.setdefault((node.id, worker.id), (self.account(last_run), len(block)))
          granted = True
          break
      if not granted:
        break

    for nid, node in healthy_nodes.items():
      if wakes.get(nid) is not None:
        deadlines.append(wakes[nid])
      for worker in list(node.workers.values()):
        should_park = (
          bool(worker.gpus)
          and not worker.stopping
          and not node.worker_tasks(worker.id)
          and keeps.get(nid, {}).get(worker.id) != worker.gpus
        )
        if should_park:
          self.issue(node, Task(new_task_id(), TaskKind.PARK, worker.id, worker.spec), now)

    self.next_wake = min((d for d in deadlines if d > now), default=None)
    for node in self.nodes.values():
      node.wake()

  def grant(
    self,
    node: NodeState,
    worker: LiveWorker,
    block: tuple[int, ...],
    runs: list[str] | tuple[str, ...],
    fair: dict[str, float],
    lag_of_run: Callable[[str], float],
    now: float,
  ) -> bool:
    if worker.spec.role == Role.TRAINER:
      heads = [head.cmd for rid in runs if (head := self.train_head(rid)) is not None]
      if not heads:
        return False
      run_id = policy.rank_runs(heads, lag_of_run, fair)[0]
      state = self.runs[run_id]
      item = self.train_head(run_id)
      op = item.cmd.op
      if op == Op.LOAD_WEIGHTS:
        version = self.checkpoint_version(item.uri or "")
      elif op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER):
        version = uuid.uuid4().hex
      else:
        version = None
      needs_restore = state.status == RunStatus.SAVED and run_id not in worker.loaded and op != Op.LOAD_WEIGHTS
      snapshot = state.snapshot if needs_restore else None
      cmd = self.hydrate_cmd(item.cmd.request_id)
      task = Task(
        new_task_id(), TaskKind.TRAIN, worker.id, worker.spec, block, cmd, state.run, run_id, snapshot, version
      )
      self.issue(node, task, now)
      return True

    active = [task for task in node.worker_tasks(worker.id) if task.kind == TaskKind.SAMPLE]
    lora_cap = types.MAX_LORAS if worker.spec.lora else 1
    active_versions = {task.version for task in active if task.version is not None}
    eligible_stage = node.staged if len(active_versions) < lora_cap else (node.staged & active_versions)

    heads: list[Cmd] = []
    version_by_run: dict[str, str | None] = {}
    ready_items = sorted(
      self.ready_samples(worker.spec, eligible_stage), key=lambda it: entry_sort_key(it.cmd.entry_id)
    )
    for item in ready_items:
      if item.cmd.run not in version_by_run:
        heads.append(item.cmd)
        version_by_run[item.cmd.run] = item.version

    def queued_for_run(run_id: str, limit: int):
      target_ver = version_by_run[run_id]
      matching = (it.cmd for it in self.runs[run_id].queue.values() if it.owner is None and it.version == target_ver)
      return itertools.islice(matching, limit)

    active_account = self.account(active[0].cmd.run) if active else None
    free_slots = types.MAX_SEQS - len(active)
    batch = policy.admit_samples(
      heads,
      queued_for_run,
      self.account,
      lag_of_run,
      fair,
      version_by_run.get,
      free_slots,
      lora_cap,
      active_account,
      active_versions,
    )
    for cmd in batch:
      item = self.runs[cmd.run].queue[cmd.request_id]
      task = Task(
        new_task_id(),
        TaskKind.SAMPLE,
        worker.id,
        worker.spec,
        block,
        cmd,
        run_id=cmd.run,
        version=item.version,
        attempt=item.attempt,
      )
      self.issue(node, task, now)
    return bool(batch)

  # ***** maintenance & inspection *****

  def maintain(self) -> None:
    changed = self.expire_nodes()
    now = time.time()
    mono = time.monotonic()

    expired_sessions = self.db.execute(
      "SELECT id FROM sessions WHERE active=1 AND seen<=?",
      (now - SESSION_GRACE,),
    ).fetchall()
    for (session_id,) in expired_sessions:
      self.end_session(session_id)

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
        for req_id, (_, expires, _) in list(self.sample_replies.items()):
          if expires <= now:
            evicted, _, _ = self.sample_replies.pop(req_id)
            self.sample_r_bytes -= len(evicted.body)
        self.sample_seen = {key: val for key, val in self.sample_seen.items() if val[1] > now}
        live_sessions = {row[0] for row in self.db.execute("SELECT id FROM sessions")}
        cutoff = now - SESSION_TTL
        self.runs = {
          rid: state
          for rid, state in self.runs.items()
          if state.live or state.deleted_at > cutoff or state.queue or state.run.session in live_sessions
        }
        self.db.execute("PRAGMA incremental_vacuum(256)")

    if changed or (self.next_wake is not None and mono >= self.next_wake):
      self.schedule()

  def workers_of(self, run_id: str) -> list[str]:
    state = self.runs.get(run_id)
    if state is None or not state.live:
      return []
    if state.run.role == Role.TRAINER:
      return [state.owner[0]] if state.owner else []
    return sorted(
      worker.id
      for node in self.nodes.values()
      if node.gpus
      for worker in node.workers.values()
      if not worker.stopping and worker.spec == state.run.spec
    )

  def snapshot_scheduling(self) -> dict:
    self.expire_nodes()
    up_nodes = {nid: node for nid, node in sorted(self.nodes.items()) if node.gpus}
    procs: dict[str, dict] = {}
    for nid, node in up_nodes.items():
      for wid, worker in sorted(node.workers.items()):
        if worker.stopping:
          continue
        entry: dict = {"spec": asdict(worker.spec), "node": nid}
        if worker.pid:
          entry["pid"] = worker.pid
        if worker.gpus:
          entry["gpus"] = list(worker.gpus)
        procs[wid] = entry

    home = {
      rid: state.owner[0]
      for rid, state in sorted(self.runs.items())
      if state.live and state.owner and state.owner[0] in procs
    }
    jobs = {
      rid: {"spec": asdict(state.run.spec), "backlog": len(state.queue)}
      for rid, state in sorted(self.runs.items())
      if state.live
    }
    return {
      "nodes": {nid: node.gpus for nid, node in up_nodes.items()},
      "state": {
        "procs": procs,
        "home": home,
        "jobs": jobs,
        "lag": dict(self.lag),
        "seen": dict(self.seen),
        "service": dict(self.service),
        "weights": dict(self.weights),
      },
    }
