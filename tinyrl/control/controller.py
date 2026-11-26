"""Controller kernel: connects the durable `Ledger`, live cluster nodes/workers, and a swappable `scheduler`."""

from __future__ import annotations

import asyncio
import itertools
import json
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace

from tinyrl.control import pager
from tinyrl.control.db import DB
from tinyrl.control.ledger import COUNTERS, SESSION_TTL, Ledger, QueueItem
from tinyrl.helpers import LEASE_MS, SWEEP_S, getenv
from tinyrl.types import (
  Cmd,
  ControlMsg,
  Done,
  Effect,
  Ev,
  NodeMsg,
  Op,
  Owner,
  Reply,
  Role,
  Run,
  RunStatus,
  Task,
  TaskKind,
  WorkerSpec,
)

GC_S = 10
SESSION_GRACE = getenv("SESSION_GRACE", 60)
TICK_MS = getenv("TICK_MS", 100)


def new_task_id() -> str:
  return uuid.uuid4().hex[:12]


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
    worker = self.workers.get(task.worker)
    if task.gpus and worker is not None:
      worker.gpus = task.gpus
      if not worker.bound and not worker.spec.movable:
        worker.bound = task.gpus
    self.tasks[task.id] = task
    self.new_tasks.append(task)

  def cancel_task(self, task_id: str) -> None:
    self.tasks.pop(task_id, None)
    self.new_tasks = [t for t in self.new_tasks if t.id != task_id]
    self.cancelled.append(task_id)

  def wake(self) -> None:
    if self.new_tasks or self.cancelled or self.stage != self.sent_stage:
      self.push_ev.set()

  def drain_push(
    self,
    ack: tuple[str, ...] = (),
    sweep: tuple[str, ...] | None = None,
    *,
    hello: bool = False,
    active_ids: set[str] = frozenset(),
  ) -> ControlMsg:
    stage_update = self.stage if self.stage != self.sent_stage else ()
    self.sent_stage = self.stage
    msg = ControlMsg(
      ack=ack, tasks=tuple(self.new_tasks), cancel=tuple(self.cancelled), stage=stage_update, sweep=sweep
    )
    self.new_tasks.clear()
    self.cancelled.clear()
    self.push_ev.clear()
    if hello:
      stale_active = tuple(sorted(tid for tid in active_ids if tid not in self.tasks and tid not in ack))
      cancel = tuple(dict.fromkeys((*msg.cancel, *stale_active)))
      tasks = tuple(t for t in self.tasks.values() if t.id not in active_ids)
      return replace(msg, tasks=tasks, cancel=cancel)
    return msg


class Controller(Ledger):
  """Safety & execution kernel realizing `Ledger` state onto live nodes with a pluggable `scheduler(ctrl, now)`."""

  def __init__(self, path: str | None = None, *, scheduler: Callable | None = None):
    super().__init__(DB(getenv("TINYRL_DB", "/tmp/tinyrl.db") if path is None else path))
    self.nodes: dict[str, NodeState] = {}
    self.made = 0
    self.scheduler = pager.schedule if scheduler is None else scheduler
    self.gc_due = self.sweep_due = self.lag_due = 0.0
    self.next_wake: float | None = None

    grace_until = time.monotonic() + LEASE_MS / 1000
    for node_id, boot_id, task in self.load():
      node = self.nodes.setdefault(node_id, NodeState(node_id, boot_id, gpus=0, until=grace_until))
      if task is not None:
        node.tasks[task.id] = task
        worker = node.workers.setdefault(task.worker, LiveWorker(task.worker, task.spec, task.gpus, started=True))
        if not worker.bound and not task.spec.movable:
          worker.bound = task.gpus

  @property
  def tick_s(self) -> float:
    return TICK_MS / 1000

  async def aclose(self) -> None:
    if self.db is not None:
      with self.db:
        self.flush_sql()
      self.db.close()
      self.db = None

  def live_node(self, owner: Owner | None) -> NodeState | None:
    if owner is None:
      return None
    node = self.nodes.get(owner[1])
    return node if (node is not None and node.boot_id == owner[2]) else None

  def healthy_nodes(self) -> dict[str, NodeState]:
    return {nid: n for nid, n in sorted(self.nodes.items()) if n.gpus and not n.draining}

  def healthy_capacity(self) -> dict[str, int]:
    return {nid: n.gpus for nid, n in self.nodes.items() if n.gpus and not n.draining}

  def advance_lag(self, capacity: dict[str, int] | None = None) -> None:
    super().advance_lag(self.healthy_capacity() if capacity is None else capacity)

  def settle_trivial_heads(self) -> None:
    """Complete head commands that can be answered immediately without touching a GPU worker."""
    for run_id, state in list(self.runs.items()):
      while (head := self.train_head(run_id)) is not None:
        op, req_id = head.cmd.op, head.cmd.request_id
        if op == Op.LOAD_WEIGHTS and self.get_checkpoint(head.uri or "") is None:
          self.finish_train(req_id, None, Reply.error(ValueError(f"no checkpoint at {head.uri}")))
        elif state.owner is None and state.status == RunStatus.LOST and op not in (Op.CREATE_MODEL, Op.LOAD_WEIGHTS):
          reply = (
            Reply(True, "application/json", json.dumps({"type": "delete_model", "model_id": run_id}).encode())
            if op == Op.DELETE_MODEL
            else Reply.error(
              KeyError(f"model {run_id} lost its unsaved steps when its trainer stopped; call load_state")
            )
          )
          self.finish_train(req_id, None, reply)
        else:
          break

  def prepare_nodes(self, now: float, deadlines: list[float]) -> None:
    """Clear declarative stage sets, expire save-retry timers, and drain workers on draining nodes."""
    for node in self.nodes.values():
      node.stage = ()
      for worker in list(node.workers.values()):
        if worker.failed_saves and now >= worker.retry_save_at:
          worker.failed_saves.clear()
        elif worker.failed_saves:
          deadlines.append(worker.retry_save_at)
        if node.draining and not node.worker_tasks(worker.id):
          self.evict_worker(node, worker, now)

  def schedule(self) -> float | None:
    """Run kernel housekeeping, invoke the swappable `scheduler`, wake nodes, and record next wakeup."""
    now, deadlines = time.monotonic(), []
    self.settle_trivial_heads()
    self.prepare_nodes(now, deadlines)
    if self.healthy_nodes():
      deadlines.extend(self.scheduler(self, now))
    for node in self.nodes.values():
      node.wake()
    self.next_wake = min((d for d in deadlines if d > now), default=None)
    return self.next_wake

  def live_versions(self) -> list[str]:
    extra = {v for n in self.nodes.values() for t in n.tasks.values() for v in (t.version, t.snapshot) if v}
    return self.retained_versions(extra)

  # --- schedule-triggering client mutations ---

  def create_run(self, run: Run) -> None:
    self.add_run(run)
    self.schedule()

  def submit_command(
    self, run_id: str, op: Op, body: bytes, seq_id=None, cost=1, uri=None, stability_refs=None
  ) -> str | None:
    req_id, added = self.enqueue(run_id, op, body, seq_id, cost, uri, stability_refs)
    if added:
      self.schedule()
    return req_id

  def delete_run(self, run_id: str) -> None:
    cancelled_samples, aborted_trains = self.remove_run(run_id)
    for owner, task_id in cancelled_samples:
      if (node := self.live_node(owner)) is not None:
        node.cancel_task(task_id)
        node.wake()
    for owner, task_id in aborted_trains:
      if (node := self.live_node(owner)) is not None and any(t.id == task_id for t in node.new_tasks):
        node.tasks.pop(task_id, None)
        node.new_tasks = [t for t in node.new_tasks if t.id != task_id]
    self.schedule()

  def end_session(self, session_id: str) -> None:
    live = sorted(rid for rid, s in self.runs.items() if s.live and s.run.session == session_id)
    queued = [bool(self.submit_command(rid, Op.DELETE_MODEL, b"{}")) for rid in live if not self.is_sampler(rid)]
    for rid in live:
      if self.is_sampler(rid):
        self.delete_run(rid)
    with self.db:
      self.db.close_session(session_id, wait_for_deletes=any(queued), now=time.time())

  def reclaim_samples(self, predicate: Callable[[QueueItem], bool]) -> None:
    for owner, task_id in self.take_samples(predicate):
      if (node := self.live_node(owner)) is not None:
        node.cancel_task(task_id)

  # --- safe execution mechanisms (surface called by any swappable scheduler) ---

  def spawn_worker(self, node_id: str, spec: WorkerSpec) -> str:
    used = {wid for n in self.nodes.values() for wid in n.workers}
    self.made = next(i for i in itertools.count(self.made + 1) if f"w{i}" not in used)
    wid = f"w{self.made}"
    self.nodes[node_id].workers[wid] = LiveWorker(wid, spec)
    self.record_event(Ev.PLACED, {"proc": wid, "node": node_id})
    return wid

  def discard_worker(self, node: NodeState, worker_id: str) -> None:
    worker = node.workers.get(worker_id)
    if worker is not None and not worker.started and not node.worker_tasks(worker_id):
      del node.workers[worker_id]

  def set_stage(self, node: NodeState, versions: Iterable[str]) -> None:
    node.stage = tuple(sorted(versions))

  def park_worker(self, node: NodeState, worker: LiveWorker, now: float) -> None:
    self.issue(node, Task(new_task_id(), TaskKind.PARK, worker.id, worker.spec), now)

  def dispatch_train(
    self, node: NodeState, worker: LiveWorker, block: tuple[int, ...], run_id: str, now: float
  ) -> bool:
    if (item := self.train_head(run_id)) is None:
      return False
    state, op = self.runs[run_id], item.cmd.op
    version = (
      self.checkpoint_version(item.uri or "")
      if op == Op.LOAD_WEIGHTS
      else (uuid.uuid4().hex if op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER) else None)
    )
    restore = state.status == RunStatus.SAVED and run_id not in worker.loaded and op != Op.LOAD_WEIGHTS
    task = Task(
      new_task_id(),
      TaskKind.TRAIN,
      worker.id,
      worker.spec,
      block,
      self.hydrate_cmd(item.cmd.request_id),
      state.run,
      run_id,
      state.snapshot if restore else None,
      version,
    )
    self.issue(node, task, now)
    return True

  def dispatch_samples(
    self, node: NodeState, worker: LiveWorker, block: tuple[int, ...], batch: list[Cmd], now: float
  ) -> bool:
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

  def issue(self, node: NodeState, task: Task, now: float) -> None:
    owner = (task.worker, node.id, node.boot_id)
    if task.kind == TaskKind.TRAIN:
      state = self.runs[task.cmd.run]
      item, claimed_at = state.queue[task.cmd.request_id], time.time()
      slim = replace(task, cmd=replace(task.cmd, body=b"", stability=None))
      with self.db:
        self.db.claim_command(task.cmd.request_id, node.id, node.boot_id, claimed_at, slim, replace(state, owner=owner))
      state.owner, item.task_id, item.owner, item.since = owner, task.id, owner, claimed_at
    elif task.kind == TaskKind.SAMPLE:
      item = self.runs[task.cmd.run].queue[task.cmd.request_id]
      item.task_id, item.owner, item.since = task.id, owner, now
    node.record_task(task)

  def detach_run(self, node: NodeState, worker: LiveWorker, run_id: str, now: float) -> None:
    if self.runs[run_id].status == RunStatus.LIVE and run_id in worker.loaded:
      self.issue(
        node, Task(new_task_id(), TaskKind.SAVE, worker.id, worker.spec, run_id=run_id, version=uuid.uuid4().hex), now
      )
      return
    self.clear_owners(lambda s: s.run.id == run_id, lose=False)
    if run_id in worker.loaded and worker.started:
      self.issue(node, Task(new_task_id(), TaskKind.DROP, worker.id, worker.spec, run_id=run_id), now)

  def evict_worker(self, node: NodeState, worker: LiveWorker, now: float) -> None:
    worker.stopping = True
    owner = (worker.id, node.id, node.boot_id)
    dirty = {
      r
      for r, s in self.runs.items()
      if s.live and s.owner == owner and s.status == RunStatus.LIVE and r in worker.loaded
    }
    self.clear_owners(lambda s: s.owner == owner and s.status != RunStatus.LIVE, lose=False)
    self.clear_owners(lambda s: s.owner == owner and s.run.id not in dirty, lose=True)
    if pending := sorted(dirty - worker.failed_saves):
      self.issue(
        node,
        Task(new_task_id(), TaskKind.SAVE, worker.id, worker.spec, run_id=pending[0], version=uuid.uuid4().hex),
        now,
      )
    elif not dirty:
      if worker.started:
        self.issue(node, Task(new_task_id(), TaskKind.STOP, worker.id, worker.spec), now)
      else:
        node.workers.pop(worker.id, None)

  # --- node liveness & protocol ---

  def live_nodes(self) -> dict[str, int]:
    self.expire_nodes()
    return {nid: n.gpus for nid, n in self.nodes.items() if n.gpus}

  def expire_node(self, node_id: str, boot_id: str) -> None:
    if self.live_node(("", node_id, boot_id)) is not None:
      del self.nodes[node_id]
    self.record_event(Ev.NODE_LOST, {"node": node_id})
    for state in list(self.runs.values()):
      if state.run.role == Role.TRAINER:
        for cmd_id, item in list(state.queue.items()):
          if item.owner and item.owner[1:] == (node_id, boot_id):
            err = Reply.error(RuntimeError(f"its trainer died, or lost ownership, during {item.cmd.op}"))
            self.finish_train(cmd_id, item.owner, err, dead=True)
    with self.db:
      self.db.add_fenced(boot_id)
    self.reclaim_samples(lambda it: it.owner[1:] == (node_id, boot_id))
    self.clear_owners(lambda s: s.owner[1:] == (node_id, boot_id))

  def expire_nodes(self) -> bool:
    now = time.monotonic()
    stale = [(nid, n.boot_id) for nid, n in list(self.nodes.items()) if n.until <= now]
    for node_id, boot_id in stale:
      self.expire_node(node_id, boot_id)
    return bool(stale)

  def drop_worker(self, node: NodeState, worker_id: str) -> None:
    worker = node.workers.pop(worker_id, None)
    owner = (worker_id, node.id, node.boot_id)
    self.reclaim_samples(lambda it: it.owner == owner)
    self.clear_owners(lambda s: s.owner == owner)
    if worker and worker.started:
      self.record_event(Ev.UNPLACED, {"proc": worker_id, "node": node.id})

  def apply_done(self, node: NodeState, done: Done, now: float) -> None:
    if (task := node.tasks.get(done.id)) is None:
      return
    worker, owner, failed = (
      node.workers.get(task.worker),
      (task.worker, node.id, node.boot_id),
      done.dead or not done.ok,
    )
    if worker and done.pid is not None:
      worker.pid, worker.started = done.pid, True

    if task.kind == TaskKind.SAVE:
      if not failed:
        self.finish_save(task.run_id, owner, done.saved)
        if worker:
          worker.failed_saves.discard(task.run_id)
          if not worker.stopping and worker.started and task.run_id in worker.loaded:
            self.issue(node, Task(new_task_id(), TaskKind.DROP, worker.id, worker.spec, run_id=task.run_id), now)
      elif done.dead:
        self.drop_worker(node, task.worker)
      elif worker:
        worker.failed_saves.add(task.run_id)
        worker.retry_save_at = now + 0.05
    elif task.kind in (TaskKind.TRAIN, TaskKind.SAMPLE):
      effect, reply = done.effect or Effect(), done.reply or Reply.error(RuntimeError(f"{task.kind} failed"))
      if task.kind == TaskKind.TRAIN:
        self.finish_train(
          task.cmd.request_id,
          owner,
          reply,
          done.charge,
          mutated=effect.mutated,
          checkpoint=effect.checkpoint,
          dead=failed,
          loaded=effect.loaded,
        )
      elif failed and done.reply is None:
        self.reclaim_samples(lambda it: it.cmd.request_id == task.cmd.request_id and it.owner == owner)
      else:
        self.finish_sample(task.cmd.request_id, owner, reply, done.charge, task.attempt)
      if done.dead or (failed and task.kind == TaskKind.TRAIN):
        self.drop_worker(node, task.worker)
      elif worker:
        worker.started = True
        worker.gpu_ms += done.charge.gpu_ms if done.charge else 0.0
        if cb := getattr(self.scheduler, "on_done", None):
          cb(worker, task, done, now)
        if task.kind == TaskKind.TRAIN:
          worker.loaded.pop(task.cmd.run, None)
          if effect.loaded:
            worker.loaded[task.cmd.run] = effect.open_grad
            rs = self.runs.get(task.cmd.run)
            if (
              rs
              and not rs.live
              and not worker.stopping
              and any(s.live and s.owner == owner for s in self.runs.values())
            ):
              self.issue(node, Task(new_task_id(), TaskKind.DROP, worker.id, worker.spec, run_id=task.cmd.run), now)
    elif task.kind == TaskKind.PARK:
      if failed:
        self.drop_worker(node, task.worker)
      elif worker:
        worker.gpus, worker.park_s, worker.hold_until, worker.idle_until = (), done.duration_s, 0.0, 0.0
    elif task.kind == TaskKind.DROP:
      if done.dead:
        self.drop_worker(node, task.worker)
      elif worker:
        worker.loaded.pop(task.run_id, None)
    elif task.kind == TaskKind.STOP:
      self.drop_worker(node, task.worker)
    node.tasks.pop(done.id, None)

  def _sync_hello(self, node: NodeState, msg: NodeMsg) -> None:
    node.gpu_mib, node.staged, node.sent_stage = msg.gpu_mib, set(msg.staged or ()), ()
    node.new_tasks.clear()
    node.cancelled.clear()
    for info in msg.workers:
      if (worker := node.workers.get(info.worker)) is not None:
        worker.pid, worker.started, worker.bound = info.pid, True, info.bound or worker.bound
        if not node.worker_tasks(info.worker):
          worker.gpus, worker.loaded = info.gpus, dict(info.loaded)
        elif info.loaded:
          worker.loaded = {**info.loaded, **worker.loaded}
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
        reported = set(msg.staged)
        lost_stage, node.staged, node.sent_stage = bool(node.staged - reported), reported, ()
      for wid in [wid for wid in msg.exited if wid in node.workers and not node.worker_tasks(wid)]:
        self.drop_worker(node, wid)

    node.gpus, node.until, node.lag, node.draining = msg.gpus or node.gpus, now + LEASE_MS / 1000, msg.lag, msg.draining
    for done in msg.done:
      self.apply_done(node, done, now)
    active_ids = set(msg.active)
    if is_hello or msg.draining or lost_stage:
      self.reclaim_samples(
        lambda it: (
          it.owner[1:] == (node.id, node.boot_id)
          and it.task_id not in active_ids
          and (node.draining or (it.version is not None and it.version not in node.staged))
        )
      )
    self.schedule()
    sweep: tuple[str, ...] | None = None
    if msg.want_sweep and not node.draining and now >= self.sweep_due:
      self.sweep_due, sweep = now + SWEEP_S, tuple(self.live_versions())
    return node.drain_push(tuple(d.id for d in msg.done), sweep, hello=is_hello, active_ids=active_ids)

  # --- maintenance & inspection ---

  def maintain(self) -> None:
    changed, now, mono = self.expire_nodes(), time.time(), time.monotonic()
    for session_id in self.db.expired_sessions(now - SESSION_GRACE):
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
        self.gc(now)
        self.db.vacuum()
    if changed or (self.next_wake is not None and mono >= self.next_wake):
      self.schedule()

  def workers_of(self, run_id: str) -> list[str]:
    state = self.runs.get(run_id)
    if state is None or not state.live:
      return []
    if state.run.role == Role.TRAINER:
      return [state.owner[0]] if state.owner else []
    return sorted(
      w.id
      for n in self.nodes.values()
      if n.gpus
      for w in n.workers.values()
      if not w.stopping and w.spec == state.run.spec
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
    return {
      "nodes": {nid: n.gpus for nid, n in up.items()},
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


__all__ = ["COUNTERS", "Controller", "LiveWorker", "NodeState", "TICK_MS", "new_task_id"]
