"""Fleet pager: evicts excess workers, places missing workers, grants GPU blocks, and parks idle ones."""

from __future__ import annotations

import itertools
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace

from tinyrl import types
from tinyrl.control import policy
from tinyrl.control.fleet import LiveWorker, NodeState, new_task_id
from tinyrl.types import Cmd, Ev, Op, Reply, Role, RunStatus, Task, TaskKind, WorkerSpec, entry_sort_key


@dataclass
class FleetView:
  nodes: dict[str, NodeState]
  capacity: dict[str, int]
  sampler_runs: dict[WorkerSpec, list[str]]
  sampler_specs: list[WorkerSpec]
  sampler_want: dict[WorkerSpec, int]
  trainer_runs: dict[tuple[str, str], list[str]]
  placed: dict[tuple[str, str], policy.Placed]


def train_head(ctrl, run_id: str):
  state = ctrl.runs.get(run_id)
  if state is None or not state.live or state.run.role != Role.TRAINER or not state.queue:
    return None
  head = next(iter(state.queue.values()))
  return head if head.owner is None else None


def ready_samples(ctrl, spec: WorkerSpec, staged: set[str]) -> list:
  out = []
  for state in ctrl.runs.values():
    if state.live and state.run.role == Role.SAMPLER and state.run.spec == spec:
      out.extend(it for it in state.queue.values() if it.owner is None and (it.version is None or it.version in staged))
  return out


def settle_trivial_heads(ctrl) -> None:
  """Complete head commands that can be answered immediately without touching a GPU worker."""
  for run_id, state in list(ctrl.runs.items()):
    while (head := train_head(ctrl, run_id)) is not None:
      op, req_id = head.cmd.op, head.cmd.request_id
      if op == Op.LOAD_WEIGHTS and ctrl.get_checkpoint(head.uri or "") is None:
        ctrl.finish_train(req_id, None, Reply.error(ValueError(f"no checkpoint at {head.uri}")))
      elif state.owner is None and state.status == RunStatus.LOST and op not in (Op.CREATE_MODEL, Op.LOAD_WEIGHTS):
        if op == Op.DELETE_MODEL:
          reply = Reply(True, "application/json", f'{{"type":"delete_model","model_id":"{run_id}"}}'.encode())
        else:
          reply = Reply.error(
            KeyError(f"model {run_id} lost its unsaved steps when its trainer stopped; call load_state")
          )
        ctrl.finish_train(req_id, None, reply)
      else:
        break


def prepare_nodes(ctrl, now: float, deadlines: list[float]) -> None:
  for node in ctrl.nodes.values():
    node.stage = ()
    for worker in list(node.workers.values()):
      if worker.failed_saves and now >= worker.retry_save_at:
        worker.failed_saves.clear()
      elif worker.failed_saves:
        deadlines.append(worker.retry_save_at)
    if node.draining:
      for worker in list(node.workers.values()):
        if not node.worker_tasks(worker.id):
          ctrl.evict_worker(node, worker, now)


def build_view(ctrl, nodes: dict[str, NodeState]) -> FleetView:
  capacity = {nid: node.gpus for nid, node in nodes.items()}
  live_workers = [(nid, w) for nid, node in nodes.items() for w in node.workers.values() if not w.stopping]
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
    spec: policy.sampler_replicas(spec, [len(ctrl.runs[rid].queue) for rid in sampler_runs[spec]])
    for spec in sampler_specs
  }
  placed: dict[tuple[str, str], policy.Placed] = {}
  for nid, worker in live_workers:
    owned = sampler_runs.get(worker.spec, ()) if worker.spec.role == Role.SAMPLER else trainer_runs[nid, worker.id]
    ready = [rid for rid in owned if ctrl.runs[rid].queue]
    lag = max(map(ctrl.ready_lag, ready), default=0.0)
    placed[nid, worker.id] = policy.Placed(nid, worker.id, worker.spec, busy=bool(ready), lag=lag, runs=len(owned))

  return FleetView(nodes, capacity, sampler_runs, sampler_specs, sampler_want, trainer_runs, placed)


def evict_workers(ctrl, view: FleetView, now: float) -> None:
  """Page out excess samplers and invalid, blocked, idle, or imbalanced trainers."""

  def sampler_rank(loc: tuple[str, str]) -> tuple:
    node = view.nodes[loc[0]]
    return (not node.workers[loc[1]].started, not node.worker_tasks(loc[1]), loc)

  keep_samplers = {
    loc
    for spec in view.sampler_specs
    for loc in sorted((k for k, p in view.placed.items() if p.spec == spec), key=sampler_rank)[
      : view.sampler_want[spec]
    ]
  }
  saving = sum(1 for node in view.nodes.values() for task in node.tasks.values() if task.kind == TaskKind.SAVE)

  for nid, node in view.nodes.items():
    exclusive_gpus = {gpu for w in node.workers.values() if not w.stopping and w.spec.exclusive for gpu in w.gpus}
    for worker in sorted(list(node.workers.values()), key=lambda w: w.id):
      if node.worker_tasks(worker.id):
        continue
      if worker.stopping:
        ctrl.evict_worker(node, worker, now)
        continue

      loc = (nid, worker.id)
      is_trainer = worker.spec.role == Role.TRAINER
      owned = view.trainer_runs[loc] if is_trainer else view.sampler_runs.get(worker.spec, ())
      ready = [rid for rid in owned if ctrl.runs[rid].queue]
      all_placed = list(view.placed.values())

      can_move = worker.spec.gpus == 1 or any(
        policy.can_host(other, view.capacity, all_placed, worker.spec) for other in view.capacity if other != nid
      )
      squeezed = not policy.can_host(nid, view.capacity, all_placed, worker.spec, skip=worker.id) and can_move
      can_rebind = not is_trainer or (
        bool(ready) and any(exclusive_gpus.isdisjoint(b) for b in policy.blocks(node.gpus, worker.spec.gpus))
      )
      blocked_by_exclusive = (
        not worker.gpus and bool(worker.bound) and not exclusive_gpus.isdisjoint(worker.bound) and can_rebind
      )
      unneeded = (not owned and worker.started) if is_trainer else (loc not in keep_samplers)
      should_drop = worker.spec.gpus > node.gpus or squeezed or blocked_by_exclusive or unneeded

      if not should_drop and is_trainer and worker.spec.shareable:
        stats = policy.node_stats(nid, node.gpus, all_placed)
        has_peer = any(p.spec == worker.spec and other != loc for other, p in view.placed.items())
        no_fleet_spare = stats.spare < 0 and not any(
          policy.node_stats(m, g, all_placed).spare >= worker.spec.gpus for m, g in view.capacity.items()
        )
        should_drop = has_peer and ((not ready and stats.used > node.gpus) or no_fleet_spare)
        if not should_drop and len(ready) >= 2 and not saving:
          if policy.choose_node(worker.spec, view.capacity, all_placed, only_spare=True):
            ctrl.detach_run(node, worker, ready[-1], now)
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

  for (nid, wid), runs in list(view.trainer_runs.items()):
    node = view.nodes[nid]
    if not runs and not node.workers[wid].started and not node.worker_tasks(wid):
      del node.workers[wid], view.trainer_runs[nid, wid], view.placed[nid, wid]


def place_workers(ctrl, view: FleetView) -> None:
  """Page in workers for unowned ready runs, scale sampler pools, and set declarative checkpoint staging."""
  sampled = {state.run.train_run for state in ctrl.runs.values() if state.live and state.run.train_run}

  def pick_node(spec: WorkerSpec, **kwargs) -> str | None:
    owed: dict[str, int] = {}
    for loc, p in view.placed.items():
      runs = view.trainer_runs.get(loc)
      if p.spec.role == Role.TRAINER and not p.spec.lora and runs and not (sampled & set(runs)):
        owed[p.node] = owed.get(p.node, 0) + p.spec.gpus
    return policy.choose_node(spec, view.capacity, list(view.placed.values()), owed_gpus=owed, **kwargs)

  def add_worker(nid: str, spec: WorkerSpec, busy: bool, lag: float = 0.0, runs: int = 0) -> str:
    wid = ctrl.fleet.spawn_worker(nid, spec)
    view.placed[nid, wid] = policy.Placed(nid, wid, spec, busy=busy, lag=lag, runs=runs)
    ctrl.record_event(Ev.PLACED, {"proc": wid, "node": nid})
    return wid

  for run_id, state in sorted(ctrl.runs.items()):
    if state.owner is not None or train_head(ctrl, run_id) is None:
      continue
    spec = state.run.spec
    compat = [
      loc for loc in sorted(view.trainer_runs) if spec.shareable and view.nodes[loc[0]].workers[loc[1]].spec == spec
    ]
    idle_compat = [loc for loc in compat if not any(ctrl.runs[r].queue for r in view.trainer_runs[loc])]
    spare_node = pick_node(spec, only_spare=True) if spec.shareable else None
    if compat and (idle_compat or not spare_node):
      chosen = min(idle_compat or compat, key=lambda loc: (len(view.trainer_runs[loc]), loc))
      view.trainer_runs[chosen].append(run_id)
      cur = view.placed[chosen]
      view.placed[chosen] = replace(
        cur, busy=True, lag=max(cur.lag, ctrl.ready_lag(run_id)), runs=len(view.trainer_runs[chosen])
      )
    elif target := (spare_node or pick_node(spec, prefer_cool=True)):
      view.trainer_runs[target, add_worker(target, spec, busy=True, lag=ctrl.ready_lag(run_id), runs=1)] = [run_id]

  needed_stage: dict[str, set[str]] = {}
  for spec in view.sampler_specs:
    parents = (ctrl.runs.get(ctrl.runs[rid].run.train_run or "") for rid in view.sampler_runs[spec])
    near = {parent.owner[1] for parent in parents if parent and parent.owner and parent.owner[1] in view.capacity}
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
    view.nodes[nid].stage = tuple(sorted(versions))


def plan_node(
  ctrl,
  nid: str,
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
        ready_by_worker[wid] = {it.cmd.run for it in ready_samples(ctrl, worker.spec, node.staged)}
      continue

    if worker.spec.role == Role.TRAINER:
      heads = {r: h for r in trainer_runs.get((nid, wid), ()) if (h := train_head(ctrl, r)) is not None}
      in_grace = bool(worker.gpus and worker.idle_until > now)
      if in_grace:
        deadlines.append(worker.idle_until)
      finishing = any(
        open_grad and ((r in heads and heads[r].cmd.op == Op.OPTIM_STEP) or (not heads and in_grace))
        for r, open_grad in worker.loaded.items()
      )
      demand = (
        policy.Demand.FINISH_STEP if finishing else (policy.Demand.READY if (heads or in_grace) else policy.Demand.IDLE)
      )
      ready_runs, gang = set(heads), worker.spec.gpus > 1
    else:
      ready_runs = {it.cmd.run for it in ready_samples(ctrl, worker.spec, node.staged)}
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
  keep = fixed | chosen
  contended = any(pw.demand != policy.Demand.IDLE and wid not in keep for wid, pw in plan_inputs.items())

  candidates = []
  for wid, block in keep.items():
    worker, ready_runs = node.workers[wid], ready_by_worker.get(wid)
    if not block or not ready_runs or (wid in fixed and contended) or (worker.gpus and worker.gpus != block):
      continue
    if any(not set(other.gpus).isdisjoint(block) for oid, other in node.workers.items() if oid != wid):
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
  return keep, wake_at, candidates


def grant_worker(
  ctrl,
  node: NodeState,
  worker: LiveWorker,
  block: tuple[int, ...],
  runs,
  fair: dict[str, float],
  lag_of_run: Callable[[str], float],
  now: float,
) -> bool:
  if worker.spec.role == Role.TRAINER:
    heads = [h.cmd for r in runs if (h := train_head(ctrl, r)) is not None]
    if not heads:
      return False
    run_id = policy.rank_runs(heads, lag_of_run, fair)[0]
    state, item = ctrl.runs[run_id], train_head(ctrl, run_id)
    op = item.cmd.op
    if op == Op.LOAD_WEIGHTS:
      version = ctrl.checkpoint_version(item.uri or "")
    elif op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER):
      version = uuid.uuid4().hex
    else:
      version = None
    restore = state.status == RunStatus.SAVED and run_id not in worker.loaded and op != Op.LOAD_WEIGHTS
    snapshot = state.snapshot if restore else None
    task = Task(
      new_task_id(),
      TaskKind.TRAIN,
      worker.id,
      worker.spec,
      block,
      ctrl.hydrate_cmd(item.cmd.request_id),
      state.run,
      run_id,
      snapshot,
      version,
    )
    ctrl.issue(node, task, now)
    return True

  active = [task for task in node.worker_tasks(worker.id) if task.kind == TaskKind.SAMPLE]
  lora_cap = types.MAX_LORAS if worker.spec.lora else 1
  active_vers = {task.version for task in active if task.version is not None}
  eligible = node.staged if len(active_vers) < lora_cap else (node.staged & active_vers)

  heads: list[Cmd] = []
  ver_by_run: dict[str, str | None] = {}
  for item in sorted(ready_samples(ctrl, worker.spec, eligible), key=lambda it: entry_sort_key(it.cmd.entry_id)):
    if item.cmd.run not in ver_by_run:
      heads.append(item.cmd)
      ver_by_run[item.cmd.run] = item.version

  def queued(run_id: str, limit: int):
    target_ver = ver_by_run[run_id]
    return itertools.islice(
      (it.cmd for it in ctrl.runs[run_id].queue.values() if not it.owner and it.version == target_ver), limit
    )

  active_acct = ctrl.account(active[0].cmd.run) if active else None
  batch = policy.admit_samples(
    heads,
    queued,
    ctrl.account,
    lag_of_run,
    fair,
    ver_by_run.get,
    types.MAX_SEQS - len(active),
    lora_cap,
    active_acct,
    active_vers,
  )
  for cmd in batch:
    item = ctrl.runs[cmd.run].queue[cmd.request_id]
    ctrl.issue(
      node,
      Task(
        new_task_id(),
        TaskKind.SAMPLE,
        worker.id,
        worker.spec,
        block,
        cmd,
        run_id=cmd.run,
        version=item.version,
        attempt=item.attempt,
      ),
      now,
    )
  return bool(batch)


def dispatch_and_park(ctrl, view: FleetView, now: float, deadlines: list[float]) -> None:
  fair = {rid: state.fair_ms or 0.0 for rid, state in ctrl.runs.items() if state.live and state.queue}
  demand_runs = ctrl.demand_runs()
  holds: dict[tuple[str, str], tuple[str, int]] = {}
  for nid, node in view.nodes.items():
    for task in node.tasks.values():
      if task.kind in (TaskKind.TRAIN, TaskKind.SAMPLE) and task.cmd:
        holds.setdefault((nid, task.worker), (ctrl.account(task.cmd.run), len(task.gpus) or task.spec.gpus))

  keeps: dict[str, dict[str, tuple[int, ...]]] = {}
  wakes: dict[str, float | None] = {}
  while True:
    lag_of_acct = policy.projected_lag(
      ctrl.lag, ctrl.seen, ctrl.service, ctrl.weights, demand_runs, view.capacity, list(holds.values())
    )

    def lag_of_run(run_id: str, fn=lag_of_acct) -> float:
      return fn(ctrl.account(run_id))

    candidates, keeps, wakes = [], {}, {}
    for nid, node in view.nodes.items():
      keeps[nid], wakes[nid], node_cands = plan_node(ctrl, nid, node, view.trainer_runs, lag_of_run, now, deadlines)
      candidates.extend(node_cands)

    candidates.sort(key=lambda entry: entry[0], reverse=True)
    granted = False
    for _, node, worker, block in candidates:
      if grant_worker(
        ctrl, node, worker, block, view.trainer_runs.get((node.id, worker.id), ()), fair, lag_of_run, now
      ):
        holds.setdefault((node.id, worker.id), (ctrl.account(node.new_tasks[-1].cmd.run), len(block)))
        granted = True
        break
    if not granted:
      break

  for nid, node in view.nodes.items():
    if wakes.get(nid) is not None:
      deadlines.append(wakes[nid])
    for worker in list(node.workers.values()):
      if (
        worker.gpus
        and not worker.stopping
        and not node.worker_tasks(worker.id)
        and keeps.get(nid, {}).get(worker.id) != worker.gpus
      ):
        ctrl.issue(node, Task(new_task_id(), TaskKind.PARK, worker.id, worker.spec), now)


def schedule(ctrl) -> float | None:
  """Single pass: settle trivial heads, drain, page out, page in + stage, grant GPUs, and park idle."""
  now = time.monotonic()
  deadlines: list[float] = []
  settle_trivial_heads(ctrl)
  if now >= ctrl.lag_due:
    ctrl.lag_due = now + ctrl.tick_s
    ctrl.advance_lag()
  prepare_nodes(ctrl, now, deadlines)
  healthy_nodes = ctrl.fleet.healthy_nodes()
  if not healthy_nodes:
    ctrl.fleet.wake_all()
    return min(deadlines, default=None)
  view = build_view(ctrl, healthy_nodes)
  evict_workers(ctrl, view, now)
  place_workers(ctrl, view)
  dispatch_and_park(ctrl, view, now, deadlines)
  ctrl.fleet.wake_all()
  return min((d for d in deadlines if d > now), default=None)
