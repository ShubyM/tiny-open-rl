"""Live physical cluster state: nodes, resident workers, leases, task outboxes, and hello sync."""

from __future__ import annotations

import asyncio
import itertools
import uuid
from dataclasses import dataclass, field, replace

from tinyrl.control import policy
from tinyrl.types import ControlMsg, NodeMsg, Owner, Task, WorkerSpec


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
    return [task for task in self.tasks.values() if task.worker == worker_id]

  def record_task(self, task: Task, now: float) -> None:
    worker = self.workers.get(task.worker)
    if task.gpus and worker is not None:
      if worker.gpus != task.gpus:
        worker.resumed_at = now
        worker.hold_until = now + policy.QUANTUM_MS / 1000
      worker.gpus = task.gpus
      if not worker.bound and not worker.spec.movable:
        worker.bound = task.gpus
    self.tasks[task.id] = task
    self.new_tasks.append(task)

  def cancel_task(self, task_id: str) -> None:
    self.tasks.pop(task_id, None)
    self.new_tasks = [task for task in self.new_tasks if task.id != task_id]
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
      ack=ack,
      tasks=tuple(self.new_tasks),
      cancel=tuple(self.cancelled),
      stage=stage_update,
      sweep=sweep,
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


class Fleet:
  """Tracks all known nodes, their live workers, and their pending/in-flight task tables."""

  def __init__(self) -> None:
    self.nodes: dict[str, NodeState] = {}
    self.made = 0

  def live_node(self, owner: Owner | None) -> NodeState | None:
    if owner is None:
      return None
    node = self.nodes.get(owner[1])
    return node if (node is not None and node.boot_id == owner[2]) else None

  def healthy_nodes(self) -> dict[str, NodeState]:
    return {nid: node for nid, node in sorted(self.nodes.items()) if node.gpus and not node.draining}

  def healthy_capacity(self) -> dict[str, int]:
    return {nid: node.gpus for nid, node in self.nodes.items() if node.gpus and not node.draining}

  def stale_nodes(self, now: float) -> list[tuple[str, str]]:
    return [(nid, node.boot_id) for nid, node in list(self.nodes.items()) if node.until <= now]

  def restore_node(self, node_id: str, boot_id: str, grace_until: float) -> NodeState:
    return self.nodes.setdefault(node_id, NodeState(node_id, boot_id, gpus=0, until=grace_until))

  def restore_task(self, node_id: str, boot_id: str, task: Task, grace_until: float) -> None:
    node = self.restore_node(node_id, boot_id, grace_until)
    node.tasks[task.id] = task
    worker = node.workers.setdefault(task.worker, LiveWorker(task.worker, task.spec, task.gpus, started=True))
    if not worker.bound and not task.spec.movable:
      worker.bound = task.gpus

  def spawn_worker(self, node_id: str, spec: WorkerSpec) -> str:
    used = {wid for node in self.nodes.values() for wid in node.workers}
    self.made = next(i for i in itertools.count(self.made + 1) if f"w{i}" not in used)
    wid = f"w{self.made}"
    self.nodes[node_id].workers[wid] = LiveWorker(wid, spec)
    return wid

  def wake_all(self) -> None:
    for node in self.nodes.values():
      node.wake()

  def sync_hello(self, node: NodeState, msg: NodeMsg) -> list[str]:
    """Reconcile a reconnecting node's reported workers; return stale started worker IDs to drop."""
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
        elif info.loaded:
          worker.loaded = {**info.loaded, **worker.loaded}
      else:
        node.workers[info.worker] = LiveWorker(
          info.worker, info.spec, info.gpus, info.bound, dict(info.loaded), info.pid, started=True
        )
    alive = {info.worker for info in msg.workers} | {task.worker for task in node.tasks.values() if task.worker}
    return [wid for wid, worker in list(node.workers.items()) if wid not in alive and worker.started]

  def sync_events(self, node: NodeState, msg: NodeMsg) -> tuple[bool, list[str]]:
    """Apply incremental stage and worker-exit reports; return `(lost_stage, exited_worker_ids)`."""
    lost_stage = False
    if msg.staged is not None:
      reported = set(msg.staged)
      lost_stage = bool(node.staged - reported)
      node.staged = reported
      node.sent_stage = ()
    exited = [wid for wid in msg.exited if wid in node.workers and not node.worker_tasks(wid)]
    return lost_stage, exited

  def task_versions(self) -> set[str]:
    out: set[str] = set()
    for node in self.nodes.values():
      for task in node.tasks.values():
        if task.version:
          out.add(task.version)
        if task.snapshot:
          out.add(task.snapshot)
    return out
