"""Default swappable fleet scheduler: pages workers, sets staging, grants GPU turns, and parks idle GPUs."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from tinyrl import types
from tinyrl.control import policy
from tinyrl.types import Done, Op, Role, Task, TaskKind, WorkerSpec

if TYPE_CHECKING:
  from tinyrl.control.controller import Controller, LiveWorker, NodeState


@dataclass
class FleetView:
  nodes: dict[str, NodeState]
  capacity: dict[str, int]
  sampler_runs: dict[WorkerSpec, list[str]]
  sampler_specs: list[WorkerSpec]
  sampler_want: dict[WorkerSpec, int]
  trainer_runs: dict[tuple[str, str], list[str]]
  placed: dict[tuple[str, str], policy.Placed]


def eligible_stage(worker: LiveWorker, active: list[Task], staged: set[str]) -> tuple[set[str], bool]:
  """Staged checkpoint versions (and whether base-model samples fit) alongside `active` sampler tasks."""
  if worker.spec.lora:
    active_vers = {t.version for t in active if t.version is not None}
    return (staged if len(active_vers) < types.MAX_LORAS else (staged & active_vers)), True
  if not active:
    return staged, True
  ver = active[0].version
  return ({ver} & staged if ver is not None else set()), ver is None


def build_view(ctrl: Controller, nodes: dict[str, NodeState]) -> FleetView:
  capacity = {nid: n.gpus for nid, n in nodes.items()}
  live_workers = [(nid, w) for nid, n in nodes.items() for w in n.workers.values() if not w.stopping]
  sampler_runs: dict[WorkerSpec, list[str]] = {}
  trainer_runs = {(nid, w.id): [] for nid, w in live_workers if w.spec.role == Role.TRAINER}

  for run_id, state in sorted(ctrl.runs.items()):
    if state.run.role == Role.SAMPLER and ctrl.has_live_spec(state):
      sampler_runs.setdefault(state.run.spec, []).append(run_id)
    elif state.live and state.owner:
      wid, nid, boot_id = state.owner
      if (nid, wid) in trainer_runs and nodes[nid].boot_id == boot_id:
        trainer_runs[nid, wid].append(run_id)

  sampler_specs = sorted(sampler_runs, key=lambda s: (s.base, s.lora, s.exclusive, s.train_run or "", s.gpus))
  sampler_want = {
    s: policy.sampler_replicas(s, [len(ctrl.runs[r].queue) for r in sampler_runs[s]]) for s in sampler_specs
  }
  placed: dict[tuple[str, str], policy.Placed] = {}
  for nid, w in live_workers:
    owned = sampler_runs.get(w.spec, ()) if w.spec.role == Role.SAMPLER else trainer_runs[nid, w.id]
    ready = [rid for rid in owned if ctrl.runs[rid].queue]
    placed[nid, w.id] = policy.Placed(
      nid, w.id, w.spec, busy=bool(ready), lag=max(map(ctrl.ready_lag, ready), default=0.0), runs=len(owned)
    )
  return FleetView(nodes, capacity, sampler_runs, sampler_specs, sampler_want, trainer_runs, placed)


def evict_workers(ctrl: Controller, view: FleetView, now: float) -> None:
  """Page out excess samplers and invalid, blocked, idle, or imbalanced trainers."""

  def rank(loc: tuple[str, str]) -> tuple:
    node = view.nodes[loc[0]]
    return (not node.workers[loc[1]].started, not node.worker_tasks(loc[1]), loc)

  keep_samplers = {
    loc
    for spec in view.sampler_specs
    for loc in sorted((k for k, p in view.placed.items() if p.spec == spec), key=rank)[: view.sampler_want[spec]]
  }
  saving = sum(1 for n in view.nodes.values() for t in n.tasks.values() if t.kind == TaskKind.SAVE)

  for nid, node in view.nodes.items():
    excl_gpus = {g for w in node.workers.values() if not w.stopping and w.spec.exclusive for g in w.gpus}
    for worker in sorted(list(node.workers.values()), key=lambda w: w.id):
      if node.worker_tasks(worker.id):
        continue
      if worker.stopping:
        ctrl.evict_worker(node, worker, now)
        continue

      loc, is_trainer = (nid, worker.id), worker.spec.role == Role.TRAINER
      owned = view.trainer_runs[loc] if is_trainer else view.sampler_runs.get(worker.spec, ())
      ready, all_placed = [r for r in owned if ctrl.runs[r].queue], list(view.placed.values())

      can_move = worker.spec.gpus == 1 or any(
        policy.can_host(o, view.capacity, all_placed, worker.spec) for o in view.capacity if o != nid
      )
      squeezed = not policy.can_host(nid, view.capacity, all_placed, worker.spec, skip=worker.id) and can_move
      can_rebind = not is_trainer or (
        bool(ready) and any(excl_gpus.isdisjoint(b) for b in policy.blocks(node.gpus, worker.spec.gpus))
      )
      blocked = not worker.gpus and bool(worker.bound) and not excl_gpus.isdisjoint(worker.bound) and can_rebind
      unneeded = (not owned and worker.started) if is_trainer else (loc not in keep_samplers)
      should_drop = worker.spec.gpus > node.gpus or squeezed or blocked or unneeded

      if not should_drop and is_trainer and worker.spec.shareable:
        stats = policy.node_stats(nid, node.gpus, all_placed)
        has_peer = any(p.spec == worker.spec and o != loc for o, p in view.placed.items())
        no_spare = stats.spare < 0 and not any(
          policy.node_stats(m, g, all_placed).spare >= worker.spec.gpus for m, g in view.capacity.items()
        )
        should_drop = has_peer and ((not ready and stats.used > node.gpus) or no_spare)
        if not should_drop and len(ready) >= 2 and not saving:
          detachable = [r for r in ready if r not in worker.failed_saves]
          if detachable and policy.choose_node(worker.spec, view.capacity, all_placed, only_spare=True):
            target_run = detachable[-1]
            ctrl.detach_run(node, worker, target_run, now)
            if ctrl.runs[target_run].owner is None and target_run in view.trainer_runs[loc]:
              view.trainer_runs[loc].remove(target_run)
            saving += 1
            continue

      if (
        not should_drop
        and is_trainer
        and not saving
        and policy.rebalance_target(view.placed[loc], view.capacity, all_placed)
      ):
        should_drop, saving = True, saving + 1

      if should_drop:
        view.placed.pop(loc, None)
        view.trainer_runs.pop(loc, None)
        ctrl.evict_worker(node, worker, now)


def place_workers(ctrl: Controller, view: FleetView) -> None:
  """Page in workers for unowned ready runs, scale sampler pools, and set declarative checkpoint staging."""
  sampled = {s.run.train_run for s in ctrl.runs.values() if s.live and s.run.train_run}
  spare_unstarted = {
    loc
    for loc, runs in view.trainer_runs.items()
    if not runs and not view.nodes[loc[0]].workers[loc[1]].started and not view.nodes[loc[0]].worker_tasks(loc[1])
  }

  def pick_node(spec: WorkerSpec, **kwargs) -> str | None:
    owed: dict[str, int] = {}
    for loc, p in view.placed.items():
      if (
        p.spec.role == Role.TRAINER
        and not p.spec.lora
        and (runs := view.trainer_runs.get(loc))
        and not (sampled & set(runs))
      ):
        owed[p.node] = owed.get(p.node, 0) + p.spec.gpus
    active_placed = [p for loc, p in view.placed.items() if loc not in spare_unstarted]
    return policy.choose_node(spec, view.capacity, active_placed, owed_gpus=owed, **kwargs)

  def add_worker(nid: str, spec: WorkerSpec, busy: bool, lag: float = 0.0, runs: int = 0) -> str:
    wid = ctrl.spawn_worker(nid, spec)
    view.placed[nid, wid] = policy.Placed(nid, wid, spec, busy=busy, lag=lag, runs=runs)
    return wid

  for run_id, state in sorted(ctrl.runs.items()):
    if state.owner is not None or ctrl.train_head(run_id) is None:
      continue
    spec = state.run.spec
    compat = [
      loc
      for loc in sorted(view.trainer_runs)
      if spec.shareable and loc not in spare_unstarted and view.placed[loc].spec == spec
    ]
    idle_compat = [loc for loc in compat if not any(ctrl.runs[r].queue for r in view.trainer_runs[loc])]
    spare_node = pick_node(spec, only_spare=True) if spec.shareable else None
    if compat and (idle_compat or not spare_node):
      chosen = min(idle_compat or compat, key=lambda loc: (len(view.trainer_runs[loc]), loc))
    elif target := (spare_node or pick_node(spec, prefer_cool=True)):
      if (
        reused := next(
          (loc for loc in sorted(spare_unstarted) if loc[0] == target and view.placed[loc].spec == spec), None
        )
      ) is not None:
        spare_unstarted.remove(reused)
        chosen = reused
      else:
        view.trainer_runs[target, add_worker(target, spec, busy=True, lag=ctrl.ready_lag(run_id), runs=1)] = [run_id]
        continue
    else:
      continue
    view.trainer_runs[chosen].append(run_id)
    cur = view.placed[chosen]
    view.placed[chosen] = replace(
      cur, busy=True, lag=max(cur.lag, ctrl.ready_lag(run_id)), runs=len(view.trainer_runs[chosen])
    )

  for nid, wid in spare_unstarted:
    ctrl.discard_worker(view.nodes[nid], wid)
    del view.trainer_runs[nid, wid], view.placed[nid, wid]

  needed_stage: dict[str, set[str]] = {}
  for spec in view.sampler_specs:
    parents = (ctrl.runs.get(ctrl.runs[r].run.train_run or "") for r in view.sampler_runs[spec])
    near = {p.owner[1] for p in parents if p and p.owner and p.owner[1] in view.capacity}
    pool = sorted(loc for loc, p in view.placed.items() if p.spec == spec)
    if not pool and (target := pick_node(spec, near=near)):
      pool.append((target, add_worker(target, spec, busy=any(ctrl.runs[r].queue for r in view.sampler_runs[spec]))))
    while len(pool) < view.sampler_want[spec] and (target := pick_node(spec, near=near, only_spare=True)):
      pool.append((target, add_worker(target, spec, busy=True)))
    versions = {
      it.version for r in view.sampler_runs[spec] for it in ctrl.runs[r].queue.values() if not it.owner and it.version
    }
    for nid, _ in pool:
      needed_stage.setdefault(nid, set()).update(versions - view.nodes[nid].staged)

  for nid, versions in needed_stage.items():
    ctrl.set_stage(view.nodes[nid], versions)


def plan_node(
  ctrl: Controller,
  nid: str,
  node: NodeState,
  trainer_runs: dict[tuple[str, str], list[str]],
  lag_of_run: Callable[[str], float],
  now: float,
) -> tuple[dict[str, tuple[int, ...]], list[float], list[tuple]]:
  fixed: dict[str, tuple[int, ...]] = {}
  occupied: set[int] = set()
  plan_inputs: dict[str, policy.Worker] = {}
  ready_by_worker: dict[str, set[str]] = {}
  node_deadlines: list[float] = []

  for wid, worker in node.workers.items():
    active = node.worker_tasks(wid)
    if worker.stopping or any(t.kind not in (TaskKind.TRAIN, TaskKind.SAMPLE) for t in active):
      occupied.update(worker.gpus)
      continue
    if active:
      occupied.update(worker.gpus)
      fixed[wid] = worker.gpus
      if worker.spec.role == Role.SAMPLER and len(active) < types.MAX_SEQS:
        eligible, allow_base = eligible_stage(worker, [t for t in active if t.kind == TaskKind.SAMPLE], node.staged)
        ready_by_worker[wid] = set(ctrl.sample_heads(worker.spec, eligible, allow_base))
      continue

    if worker.spec.role == Role.TRAINER:
      heads = {r: h for r in trainer_runs.get((nid, wid), ()) if (h := ctrl.train_head(r)) is not None}
      in_grace = bool(worker.gpus and worker.idle_until > now)
      if in_grace:
        node_deadlines.append(worker.idle_until)
      finishing = any(
        og and ((r in heads and heads[r].cmd.op == Op.OPTIM_STEP) or (not heads and in_grace))
        for r, og in worker.loaded.items()
      )
      demand = (
        policy.Demand.FINISH_STEP if finishing else (policy.Demand.READY if (heads or in_grace) else policy.Demand.IDLE)
      )
      ready_runs, gang = set(heads), worker.spec.gpus > 1
    else:
      ready_runs = set(ctrl.sample_heads(worker.spec, node.staged))
      demand, gang = (policy.Demand.READY if ready_runs else policy.Demand.IDLE), False

    ready_by_worker[wid] = ready_runs
    lag = max(map(lag_of_run, ready_runs), default=-worker.gpu_ms)
    plan_inputs[wid] = policy.Worker(
      worker.spec.gpus,
      worker.spec.exclusive,
      demand,
      worker.gpus,
      worker.bound,
      gang,
      worker.hold_until,
      lag,
      worker.gpu_ms,
    )

  chosen, wake_at = policy.plan_gpus(plan_inputs, node.gpus, now, occupied, {g for b in fixed.values() for g in b})
  if wake_at is not None:
    node_deadlines.append(wake_at)
  keep = fixed | chosen
  contended = any(pw.demand != policy.Demand.IDLE and wid not in keep for wid, pw in plan_inputs.items())

  candidates = []
  for wid, block in keep.items():
    worker, ready_runs = node.workers[wid], ready_by_worker.get(wid)
    if not block or not ready_runs or (wid in fixed and contended) or (worker.gpus and worker.gpus != block):
      continue
    if any(not set(o.gpus).isdisjoint(block) for oid, o in node.workers.items() if oid != wid):
      continue
    prio = (
      worker.spec.exclusive,
      round(max(map(lag_of_run, ready_runs)), 3),
      round(-worker.gpu_ms, 3),
      bool(worker.gpus),
      nid,
      wid,
    )
    candidates.append((prio, node, worker, block))
  return keep, node_deadlines, candidates


def _arm_quantum(worker: LiveWorker, block: tuple[int, ...], now: float) -> None:
  if block and worker.gpus != block:
    worker.resumed_at, worker.hold_until = now, now + policy.QUANTUM_MS / 1000


def grant_worker(
  ctrl: Controller,
  view: FleetView,
  node: NodeState,
  worker: LiveWorker,
  block: tuple[int, ...],
  fair: dict[str, float],
  lag_of_run: Callable[[str], float],
  now: float,
) -> bool:
  if worker.spec.role == Role.TRAINER:
    heads = [h.cmd for r in view.trainer_runs.get((node.id, worker.id), ()) if (h := ctrl.train_head(r)) is not None]
    if not heads:
      return False
    _arm_quantum(worker, block, now)
    return ctrl.dispatch_train(node, worker, block, policy.rank_runs(heads, lag_of_run, fair)[0], now)

  active = [t for t in node.worker_tasks(worker.id) if t.kind == TaskKind.SAMPLE]
  eligible, allow_base = eligible_stage(worker, active, node.staged)
  first_by_run = ctrl.sample_heads(worker.spec, eligible, allow_base)
  heads = [it.cmd for it in first_by_run.values()]
  ver_by_run = {rid: it.version for rid, it in first_by_run.items()}

  def queued(run_id: str, limit: int):
    target_ver = ver_by_run[run_id]
    return itertools.islice(
      (it.cmd for it in ctrl.runs[run_id].queue.values() if not it.owner and it.version == target_ver), limit
    )

  in_flight: dict[str, int] = {}
  for live_node in view.nodes.values():
    for t in live_node.tasks.values():
      if t.kind == TaskKind.SAMPLE and t.spec == worker.spec and t.cmd:
        acct = ctrl.account(t.cmd.run)
        in_flight[acct] = in_flight.get(acct, 0) + 1
  batch = policy.admit_samples(
    heads,
    queued,
    ctrl.account,
    lag_of_run,
    fair,
    ver_by_run.get,
    types.MAX_SEQS - len(active),
    types.MAX_LORAS if worker.spec.lora else 1,
    in_flight,
    {t.version for t in active if t.version is not None},
  )
  if batch:
    _arm_quantum(worker, block, now)
  return ctrl.dispatch_samples(node, worker, block, batch, now)


def worker_holds(ctrl: Controller, node: NodeState, wid: str) -> list[tuple[str, float]]:
  """Proportional GPU hold per account for one active worker."""
  tasks = [t for t in node.worker_tasks(wid) if t.kind in (TaskKind.TRAIN, TaskKind.SAMPLE) and t.cmd]
  if not tasks:
    return []
  width, counts = len(tasks[0].gpus) or tasks[0].spec.gpus, {}
  for t in tasks:
    acct = ctrl.account(t.cmd.run)
    counts[acct] = counts.get(acct, 0) + 1
  return [(acct, width * count / len(tasks)) for acct, count in counts.items()]


def dispatch_and_park(ctrl: Controller, view: FleetView, now: float, deadlines: list[float]) -> None:
  fair = {rid: s.fair_ms or 0.0 for rid, s in ctrl.runs.items() if s.live and s.queue}
  demand_runs = ctrl.demand_runs()
  holds = {(nid, wid): sh for nid, n in view.nodes.items() for wid in n.workers if (sh := worker_holds(ctrl, n, wid))}

  keeps: dict[str, dict[str, tuple[int, ...]]] = {}
  final_deadlines: list[float] = []
  while True:
    flat_holds = [share for shares in holds.values() for share in shares]
    lag_of_acct = policy.projected_lag(
      ctrl.lag, ctrl.seen, ctrl.service, ctrl.weights, demand_runs, view.capacity, flat_holds
    )

    def lag_of_run(rid: str, fn=lag_of_acct) -> float:
      return fn(ctrl.account(rid))

    candidates, keeps, final_deadlines = [], {}, []
    for nid, node in view.nodes.items():
      keeps[nid], node_dl, node_cands = plan_node(ctrl, nid, node, view.trainer_runs, lag_of_run, now)
      final_deadlines.extend(node_dl)
      candidates.extend(node_cands)

    candidates.sort(key=lambda entry: entry[0], reverse=True)
    if not any(
      grant_worker(ctrl, view, node, worker, block, fair, lag_of_run, now)
      and (holds.__setitem__((node.id, worker.id), worker_holds(ctrl, node, worker.id)) or True)
      for _, node, worker, block in candidates
    ):
      break

  deadlines.extend(final_deadlines)
  for nid, node in view.nodes.items():
    for worker in list(node.workers.values()):
      if (
        worker.gpus
        and not worker.stopping
        and not node.worker_tasks(worker.id)
        and keeps.get(nid, {}).get(worker.id) != worker.gpus
      ):
        ctrl.park_worker(node, worker, now)


def on_done(worker: LiveWorker, task: Task, done: Done, now: float) -> None:
  """Update pager-owned switch-amortization quantum and train cycle grace after a completed task."""
  if done.duration_s > 0 and worker.park_s > 0:
    worker.hold_until = max(worker.hold_until, worker.resumed_at + policy.quantum_s(worker.park_s + done.duration_s))
  if task.kind == TaskKind.TRAIN:
    worker.idle_until = now + policy.CYCLE_GRACE_S


def schedule(ctrl: Controller, now: float) -> list[float]:
  """Single policy pass over healthy nodes: advance EEVDF lag, evict, place + stage, grant GPU turns, and park idle."""
  if now >= ctrl.lag_due:
    ctrl.lag_due = now + ctrl.tick_s
    ctrl.advance_lag()
  deadlines: list[float] = []
  view = build_view(ctrl, ctrl.healthy_nodes())
  evict_workers(ctrl, view, now)
  place_workers(ctrl, view)
  dispatch_and_park(ctrl, view, now, deadlines)
  return deadlines


schedule.on_done = on_done
