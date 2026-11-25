"""Controller engine: connects the client `Ledger`, live `Fleet`, and `pager` over `NodeMsg`."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, replace

from tinyrl.control import pager, policy
from tinyrl.control.db import DB
from tinyrl.control.fleet import Fleet, LiveWorker, NodeState, new_task_id
from tinyrl.control.ledger import COUNTERS, SESSION_TTL, Ledger, QueueItem
from tinyrl.helpers import LEASE_MS, SWEEP_S, getenv
from tinyrl.types import ControlMsg, Done, Effect, Ev, NodeMsg, Op, Reply, Role, Run, RunStatus, Task, TaskKind

GC_S = 10
SESSION_GRACE = getenv("SESSION_GRACE", 60)
TICK_MS = getenv("TICK_MS", 100)


class Controller(Ledger):
  """Realizes `Ledger` state onto a live `Fleet` via `pager.schedule` and `handle(NodeMsg)`."""

  def __init__(self, path: str | None = None):
    super().__init__(DB(getenv("TINYRL_DB", "/tmp/tinyrl.db") if path is None else path))
    self.fleet = Fleet()
    self.nodes = self.fleet.nodes
    self.gc_due = self.sweep_due = self.lag_due = 0.0
    self.next_wake: float | None = None

    grace_until = time.monotonic() + LEASE_MS / 1000
    for node_id, boot_id, task in self.load():
      if task is None:
        self.fleet.restore_node(node_id, boot_id, grace_until)
      else:
        self.fleet.restore_task(node_id, boot_id, task, grace_until)

  @property
  def tick_s(self) -> float:
    return TICK_MS / 1000

  async def aclose(self) -> None:
    if self.db is not None:
      with self.db:
        self.flush_sql()
      self.db.close()
      self.db = None

  def advance_lag(self, capacity: dict[str, int] | None = None) -> None:
    super().advance_lag(self.fleet.healthy_capacity() if capacity is None else capacity)

  def schedule(self) -> None:
    self.next_wake = pager.schedule(self)

  def live_versions(self) -> list[str]:
    return self.retained_versions(self.fleet.task_versions())

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
      if (node := self.fleet.live_node(owner)) is not None:
        node.cancel_task(task_id)
        node.wake()
    for owner, task_id in aborted_trains:
      if (node := self.fleet.live_node(owner)) is not None:
        node.tasks.pop(task_id, None)
    self.schedule()

  def end_session(self, session_id: str) -> None:
    live = sorted(rid for rid, s in self.runs.items() if s.live and s.run.session == session_id)
    queued = [bool(self.submit_command(rid, Op.DELETE_MODEL, b"{}")) for rid in live if not self.is_sampler(rid)]
    for rid in live:
      if self.is_sampler(rid):
        self.delete_run(rid)
    self.db.close_session(session_id, wait_for_deletes=any(queued), now=time.time())

  def reclaim_samples(self, predicate: Callable[[QueueItem], bool]) -> None:
    for owner, task_id in self.take_samples(predicate):
      if (node := self.fleet.live_node(owner)) is not None:
        node.cancel_task(task_id)

  # --- worker task issuance & eviction ---

  def issue(self, node: NodeState, task: Task, now: float) -> None:
    owner = (task.worker, node.id, node.boot_id)
    if task.kind == TaskKind.TRAIN:
      state = self.runs[task.cmd.run]
      item = state.queue[task.cmd.request_id]
      claimed_at = time.time()
      slim = replace(task, cmd=replace(task.cmd, body=b"", stability=None))
      self.db.claim_command(task.cmd.request_id, node.id, node.boot_id, claimed_at, slim, replace(state, owner=owner))
      state.owner, item.task_id, item.owner, item.since = owner, task.id, owner, claimed_at
    elif task.kind == TaskKind.SAMPLE:
      item = self.runs[task.cmd.run].queue[task.cmd.request_id]
      item.task_id, item.owner, item.since = task.id, owner, now
    node.record_task(task, now)

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
    self.clear_owners(lambda s: s.owner == owner and s.run.id not in dirty, lose=False)
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
    return {nid: node.gpus for nid, node in self.nodes.items() if node.gpus}

  def expire_node(self, node_id: str, boot_id: str) -> None:
    if self.fleet.live_node(("", node_id, boot_id)) is not None:
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
    self.reclaim_samples(lambda it: it.owner[1:] == (node_id, boot_id))
    self.clear_owners(lambda s: s.owner[1:] == (node_id, boot_id))

  def expire_nodes(self) -> bool:
    stale = self.fleet.stale_nodes(time.monotonic())
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
    task = node.tasks.get(done.id)
    if task is None:
      return
    worker = node.workers.get(task.worker)
    owner = (task.worker, node.id, node.boot_id)
    failed = done.dead or not done.ok
    if worker and done.pid is not None:
      worker.pid, worker.started = done.pid, True

    if task.kind == TaskKind.SAVE and not failed:
      self.finish_save(task.run_id, owner, done.saved)
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
        worker.gpu_ms += done.charge.gpu_ms if done.charge else 0.0
        if done.duration_s > 0 and worker.park_s > 0:
          worker.hold_until = max(
            worker.hold_until, worker.resumed_at + policy.quantum_s(worker.park_s + done.duration_s)
          )
        if task.kind == TaskKind.TRAIN:
          worker.idle_until = now + policy.CYCLE_GRACE_S
          worker.loaded.pop(task.cmd.run, None)
          if effect.loaded:
            worker.loaded[task.cmd.run] = effect.open_grad
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

  def handle(self, msg: NodeMsg) -> ControlMsg:
    self.expire_nodes()
    now = time.monotonic()
    if self.db.is_fenced(msg.boot_id):
      return ControlMsg(reset=True)
    node = self.nodes.get(msg.node)
    is_hello = msg.kind == "hello" or node is None
    lost_stage = False

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
      for wid in self.fleet.sync_hello(node, msg):
        self.drop_worker(node, wid)
      self.clear_owners(lambda s: s.owner[1:] == (node.id, node.boot_id) and s.owner[0] not in node.workers)
    elif node.boot_id != msg.boot_id:
      return ControlMsg(reset=True)
    else:
      lost_stage, exited = self.fleet.sync_events(node, msg)
      for wid in exited:
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
    sweep: tuple[str, ...] = ()
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
    procs: dict[str, dict] = {}
    for nid, node in up.items():
      for wid, worker in sorted(node.workers.items()):
        if not worker.stopping:
          entry: dict = {"spec": asdict(worker.spec), "node": nid}
          if worker.pid:
            entry["pid"] = worker.pid
          if worker.gpus:
            entry["gpus"] = list(worker.gpus)
          procs[wid] = entry
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


__all__ = ["COUNTERS", "Controller", "TICK_MS"]
