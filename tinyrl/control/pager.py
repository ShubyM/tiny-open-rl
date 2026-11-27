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
  cap: dict[str, int]
  s_runs: dict[WorkerSpec, list[str]]
  s_specs: list[WorkerSpec]
  s_want: dict[WorkerSpec, int]
  t_runs: dict[tuple[str, str], list[str]]
  placed: dict[tuple[str, str], policy.Placed]


def eligible_stage(w: LiveWorker, active: list[Task], staged: set[str]) -> tuple[set[str], bool]:
  if w.spec.lora:
    vers = {t.version for t in active if t.version is not None}
    return (staged if len(vers) < types.MAX_LORAS else (staged & vers)), True
  if not active:
    return staged, True
  v = active[0].version
  return ({v} & staged if v is not None else set()), v is None


def build_view(ctrl: Controller, nodes: dict[str, NodeState]) -> FleetView:
  cap = {nid: n.gpus for nid, n in nodes.items()}
  live = [(nid, w) for nid, n in nodes.items() for w in n.workers.values() if not w.stopping]
  s_runs: dict[WorkerSpec, list[str]] = {}
  t_runs = {(nid, w.id): [] for nid, w in live if w.spec.role == Role.TRAINER}
  for rid, s in sorted(ctrl.runs.items()):
    if s.run.role == Role.SAMPLER and ctrl.has_live_spec(s):
      s_runs.setdefault(s.run.spec, []).append(rid)
    elif s.live and s.owner:
      wid, nid, bid = s.owner
      if (nid, wid) in t_runs and nodes[nid].boot_id == bid:
        t_runs[nid, wid].append(rid)

  s_specs = sorted(s_runs, key=lambda s: (s.base, s.lora, s.exclusive, s.train_run or "", s.gpus))
  s_want = {s: policy.sampler_replicas(s, [len(ctrl.runs[r].queue) for r in s_runs[s]]) for s in s_specs}
  placed: dict[tuple[str, str], policy.Placed] = {}
  for nid, w in live:
    owned = s_runs.get(w.spec, ()) if w.spec.role == Role.SAMPLER else t_runs[nid, w.id]
    ready = [r for r in owned if ctrl.runs[r].queue]
    placed[nid, w.id] = policy.Placed(
      nid, w.id, w.spec, bool(ready), max(map(ctrl.ready_lag, ready), default=0.0), len(owned)
    )
  return FleetView(nodes, cap, s_runs, s_specs, s_want, t_runs, placed)


def evict_workers(ctrl: Controller, v: FleetView, now: float) -> None:
  """Page out excess samplers and invalid, blocked, idle, or imbalanced trainers."""

  def rank(loc: tuple[str, str]) -> tuple:
    n = v.nodes[loc[0]]
    return (not n.workers[loc[1]].started, not n.worker_tasks(loc[1]), loc)

  keep = {
    loc
    for spec in v.s_specs
    for loc in sorted((k for k, p in v.placed.items() if p.spec == spec), key=rank)[: v.s_want[spec]]
  }
  saving = sum(1 for n in v.nodes.values() for t in n.tasks.values() if t.kind == TaskKind.SAVE)

  for nid, node in v.nodes.items():
    excl = {g for w in node.workers.values() if not w.stopping and w.spec.exclusive for g in w.gpus}
    for w in sorted(list(node.workers.values()), key=lambda x: x.id):
      if node.worker_tasks(w.id):
        continue
      if w.stopping:
        ctrl.evict_worker(node, w, now)
        continue
      loc, is_tr = (nid, w.id), w.spec.role == Role.TRAINER
      owned = v.t_runs[loc] if is_tr else v.s_runs.get(w.spec, ())
      ready, all_p = [r for r in owned if ctrl.runs[r].queue], list(v.placed.values())
      can_move = w.spec.gpus == 1 or any(policy.can_host(o, v.cap, all_p, w.spec) for o in v.cap if o != nid)
      squeezed = not policy.can_host(nid, v.cap, all_p, w.spec, skip=w.id) and can_move
      can_rebind = not is_tr or (bool(ready) and any(excl.isdisjoint(b) for b in policy.blocks(node.gpus, w.spec.gpus)))
      blocked = not w.gpus and bool(w.bound) and not excl.isdisjoint(w.bound) and can_rebind
      drop = (
        w.spec.gpus > node.gpus or squeezed or blocked or ((not owned and w.started) if is_tr else (loc not in keep))
      )

      if not drop and is_tr and w.spec.shareable:
        st = policy.node_stats(nid, node.gpus, all_p)
        has_peer = any(p.spec == w.spec and o != loc for o, p in v.placed.items())
        no_spare = st.spare < 0 and not any(
          policy.node_stats(m, g, all_p).spare >= w.spec.gpus for m, g in v.cap.items()
        )
        drop = has_peer and ((not ready and st.used > node.gpus) or no_spare)
        if not drop and len(ready) >= 2 and not saving:
          det = [r for r in ready if r not in w.failed_saves]
          if det and policy.choose_node(w.spec, v.cap, all_p, only_spare=True):
            ctrl.detach_run(node, w, target := det[-1], now)
            if ctrl.runs[target].owner is None and target in v.t_runs[loc]:
              v.t_runs[loc].remove(target)
            saving += 1
            continue

      if not drop and is_tr and not saving and policy.rebalance_target(v.placed[loc], v.cap, all_p):
        drop, saving = True, saving + 1
      if drop:
        v.placed.pop(loc, None)
        v.t_runs.pop(loc, None)
        ctrl.evict_worker(node, w, now)


def place_workers(ctrl: Controller, v: FleetView) -> None:
  """Page in workers for unowned ready runs, scale sampler pools, and set declarative checkpoint staging."""
  sampled = {s.run.train_run for s in ctrl.runs.values() if s.live and s.run.train_run}
  spares = {
    loc
    for loc, runs in v.t_runs.items()
    if not runs and not v.nodes[loc[0]].workers[loc[1]].started and not v.nodes[loc[0]].worker_tasks(loc[1])
  }

  def pick(spec: WorkerSpec, **kw) -> str | None:
    owed: dict[str, int] = {}
    for loc, p in v.placed.items():
      if p.spec.role == Role.TRAINER and not p.spec.lora and (runs := v.t_runs.get(loc)) and not (sampled & set(runs)):
        owed[p.node] = owed.get(p.node, 0) + p.spec.gpus
    return policy.choose_node(
      spec, v.cap, [p for loc, p in v.placed.items() if loc not in spares], owed_gpus=owed, **kw
    )

  def add(nid: str, spec: WorkerSpec, busy: bool, lag: float = 0.0, runs: int = 0) -> str:
    wid = ctrl.spawn_worker(nid, spec)
    v.placed[nid, wid] = policy.Placed(nid, wid, spec, busy, lag, runs)
    return wid

  for rid, s in sorted(ctrl.runs.items()):
    if s.owner is not None or ctrl.train_head(rid) is None:
      continue
    spec = s.run.spec
    compat = [loc for loc in sorted(v.t_runs) if spec.shareable and loc not in spares and v.placed[loc].spec == spec]
    idle = [loc for loc in compat if not any(ctrl.runs[r].queue for r in v.t_runs[loc])]
    spare_n = pick(spec, only_spare=True) if spec.shareable else None
    if compat and (idle or not spare_n):
      chosen = min(idle or compat, key=lambda loc: (len(v.t_runs[loc]), loc))
    elif target := (spare_n or pick(spec, prefer_cool=True)):
      if (
        reused := next((loc for loc in sorted(spares) if loc[0] == target and v.placed[loc].spec == spec), None)
      ) is not None:
        spares.remove(reused)
        chosen = reused
      else:
        v.t_runs[target, add(target, spec, True, ctrl.ready_lag(rid), 1)] = [rid]
        continue
    else:
      continue
    v.t_runs[chosen].append(rid)
    cur = v.placed[chosen]
    v.placed[chosen] = replace(cur, busy=True, lag=max(cur.lag, ctrl.ready_lag(rid)), runs=len(v.t_runs[chosen]))

  for nid, wid in spares:
    ctrl.discard_worker(v.nodes[nid], wid)
    del v.t_runs[nid, wid], v.placed[nid, wid]

  stage: dict[str, set[str]] = {}
  for spec in v.s_specs:
    parents = (ctrl.runs.get(ctrl.runs[r].run.train_run or "") for r in v.s_runs[spec])
    near = {p.owner[1] for p in parents if p and p.owner and p.owner[1] in v.cap}
    pool = sorted(loc for loc, p in v.placed.items() if p.spec == spec)
    if not pool and (target := pick(spec, near=near)):
      pool.append((target, add(target, spec, any(ctrl.runs[r].queue for r in v.s_runs[spec]))))
    while len(pool) < v.s_want[spec] and (target := pick(spec, near=near, only_spare=True)):
      pool.append((target, add(target, spec, True)))
    vers = {it.version for r in v.s_runs[spec] for it in ctrl.runs[r].queue.values() if not it.owner and it.version}
    for nid, _ in pool:
      stage.setdefault(nid, set()).update(vers - v.nodes[nid].staged)
  for nid, vers in stage.items():
    ctrl.set_stage(v.nodes[nid], vers)


def plan_node(ctrl: Controller, nid: str, node: NodeState, t_runs: dict, lag_of: Callable, now: float):
  fixed, occupied, inputs, ready_map, dl = {}, set(), {}, {}, []
  for wid, w in node.workers.items():
    active = node.worker_tasks(wid)
    if w.stopping or any(t.kind not in (TaskKind.TRAIN, TaskKind.SAMPLE) for t in active):
      occupied.update(w.gpus)
      continue
    if active:
      occupied.update(w.gpus)
      fixed[wid] = w.gpus
      if w.spec.role == Role.SAMPLER and len(active) < types.MAX_SEQS:
        elig, allow_base = eligible_stage(w, [t for t in active if t.kind == TaskKind.SAMPLE], node.staged)
        ready_map[wid] = set(ctrl.sample_heads(w.spec, elig, allow_base))
      continue

    if w.spec.role == Role.TRAINER:
      heads = {r: h for r in t_runs.get((nid, wid), ()) if (h := ctrl.train_head(r)) is not None}
      in_grace = bool(w.gpus and w.idle_until > now)
      if in_grace:
        dl.append(w.idle_until)
      fin = any(
        og and ((r in heads and heads[r].cmd.op == Op.OPTIM_STEP) or (not heads and in_grace))
        for r, og in w.loaded.items()
      )
      demand = (
        policy.Demand.FINISH_STEP if fin else (policy.Demand.READY if (heads or in_grace) else policy.Demand.IDLE)
      )
      ready_runs, gang = set(heads), w.spec.gpus > 1
    else:
      ready_runs = set(ctrl.sample_heads(w.spec, node.staged))
      demand, gang = (policy.Demand.READY if ready_runs else policy.Demand.IDLE), False

    ready_map[wid] = ready_runs
    lag = max(map(lag_of, ready_runs), default=-w.gpu_ms)
    inputs[wid] = policy.Worker(
      w.spec.gpus, w.spec.exclusive, demand, w.gpus, w.bound, gang, w.hold_until, lag, w.gpu_ms
    )

  chosen, wake_at = policy.plan_gpus(inputs, node.gpus, now, occupied, {g for b in fixed.values() for g in b})
  if wake_at is not None:
    dl.append(wake_at)
  keep = fixed | chosen
  contended = any(pw.demand != policy.Demand.IDLE and wid not in keep for wid, pw in inputs.items())

  cands = []
  for wid, block in keep.items():
    w, rr = node.workers[wid], ready_map.get(wid)
    if not block or not rr or (wid in fixed and contended) or (w.gpus and w.gpus != block):
      continue
    if any(not set(o.gpus).isdisjoint(block) for oid, o in node.workers.items() if oid != wid):
      continue
    prio = (w.spec.exclusive, round(max(map(lag_of, rr)), 3), round(-w.gpu_ms, 3), bool(w.gpus), nid, wid)
    cands.append((prio, node, w, block))
  return keep, dl, cands


def grant_worker(
  ctrl: Controller,
  v: FleetView,
  node: NodeState,
  w: LiveWorker,
  block: tuple[int, ...],
  fair: dict,
  lag_of: Callable,
  now: float,
) -> bool:
  def arm() -> None:
    if block and w.gpus != block:
      w.resumed_at, w.hold_until = now, now + policy.QUANTUM_MS / 1000

  if w.spec.role == Role.TRAINER:
    heads = [h.cmd for r in v.t_runs.get((node.id, w.id), ()) if (h := ctrl.train_head(r)) is not None]
    if not heads:
      return False
    arm()
    return ctrl.dispatch_train(node, w, block, policy.rank_runs(heads, lag_of, fair)[0], now)

  active = [t for t in node.worker_tasks(w.id) if t.kind == TaskKind.SAMPLE]
  elig, allow_base = eligible_stage(w, active, node.staged)
  first = ctrl.sample_heads(w.spec, elig, allow_base)
  ver_of = {r: it.version for r, it in first.items()}

  def queued(rid: str, limit: int):
    tv = ver_of[rid]
    return itertools.islice(
      (it.cmd for it in ctrl.runs[rid].queue.values() if not it.owner and it.version == tv), limit
    )

  fly: dict[str, int] = {}
  for n in v.nodes.values():
    for t in n.tasks.values():
      if t.kind == TaskKind.SAMPLE and t.spec == w.spec and t.cmd:
        a = ctrl.account(t.cmd.run)
        fly[a] = fly.get(a, 0) + 1
  batch = policy.admit_samples(
    [it.cmd for it in first.values()],
    queued,
    ctrl.account,
    lag_of,
    fair,
    ver_of.get,
    types.MAX_SEQS - len(active),
    types.MAX_LORAS if w.spec.lora else 1,
    fly,
    {t.version for t in active if t.version is not None},
  )
  if batch:
    arm()
  return ctrl.dispatch_samples(node, w, block, batch, now)


def worker_holds(ctrl: Controller, node: NodeState, wid: str) -> list[tuple[str, float]]:
  tasks = [t for t in node.worker_tasks(wid) if t.kind in (TaskKind.TRAIN, TaskKind.SAMPLE) and t.cmd]
  if not tasks:
    return []
  width, counts = len(tasks[0].gpus) or tasks[0].spec.gpus, {}
  for t in tasks:
    a = ctrl.account(t.cmd.run)
    counts[a] = counts.get(a, 0) + 1
  return [(a, width * c / len(tasks)) for a, c in counts.items()]


def dispatch_and_park(ctrl: Controller, v: FleetView, now: float, deadlines: list[float]) -> None:
  fair = {r: s.fair_ms or 0.0 for r, s in ctrl.runs.items() if s.live and s.queue}
  demand = ctrl.demand_runs()
  holds = {(nid, wid): sh for nid, n in v.nodes.items() for wid in n.workers if (sh := worker_holds(ctrl, n, wid))}

  keeps, final_dl = {}, []
  while True:
    flat = [sh for shares in holds.values() for sh in shares]
    lag_acct = policy.projected_lag(ctrl.lag, ctrl.seen, ctrl.service, ctrl.weights, demand, v.cap, flat)

    def lag_of(rid: str, fn=lag_acct) -> float:
      return fn(ctrl.account(rid))

    cands, keeps, final_dl = [], {}, []
    for nid, node in v.nodes.items():
      keeps[nid], dl, nc = plan_node(ctrl, nid, node, v.t_runs, lag_of, now)
      final_dl.extend(dl)
      cands.extend(nc)

    cands.sort(key=lambda e: e[0], reverse=True)
    for _, node, w, block in cands:
      if grant_worker(ctrl, v, node, w, block, fair, lag_of, now):
        holds[node.id, w.id] = worker_holds(ctrl, node, w.id)
        break
    else:
      break

  deadlines.extend(final_dl)
  for nid, node in v.nodes.items():
    for w in list(node.workers.values()):
      if w.gpus and not w.stopping and not node.worker_tasks(w.id) and keeps.get(nid, {}).get(w.id) != w.gpus:
        ctrl.park_worker(node, w, now)


def on_done(w: LiveWorker, task: Task, done: Done, now: float) -> None:
  if done.duration_s > 0 and w.park_s > 0:
    w.hold_until = max(w.hold_until, w.resumed_at + policy.quantum_s(w.park_s + done.duration_s))
  if task.kind == TaskKind.TRAIN:
    w.idle_until = now + policy.CYCLE_GRACE_S


def schedule(ctrl: Controller, now: float) -> list[float]:
  """Single policy pass over healthy nodes: advance EEVDF lag, evict, place + stage, grant GPU turns, and park idle."""
  if now >= ctrl.lag_due:
    ctrl.lag_due = now + ctrl.tick_s
    ctrl.advance_lag()
  deadlines: list[float] = []
  v = build_view(ctrl, ctrl.healthy_nodes())
  evict_workers(ctrl, v, now)
  place_workers(ctrl, v)
  dispatch_and_park(ctrl, v, now, deadlines)
  return deadlines


schedule.on_done = on_done
