"""Choose physical GPU turns using the fleet's completed service and outstanding claims."""

from __future__ import annotations

from collections import Counter

from tinyrl.control import scheduler
from tinyrl.control.fairness import Demand, Worker, blocks, plan_gpus, update_lag
from tinyrl.helpers import getenv
from tinyrl.types import LocalWorker, Role

CYCLE_GRACE_S = getenv("CYCLE_GRACE_MS", 100) / 1000
QUANTUM_MS, MAX_QUANTUM_MS = getenv("QUANTUM_MS", 2000), getenv("MAX_QUANTUM_MS", 15000)
SWITCH_BUDGET = max(0.01, getenv("SWITCH_BUDGET", 0.1))


def projected_lag(state: scheduler.State, claims):
  """Project one quantum per occupied GPU. All requests in a sampler wave share that reservation."""
  if not claims:
    return state.ready_lag
  service = Counter(state.service)
  for run, block in claims.values():
    width = len(block) or state.jobs[run].spec.gpus
    service[state.account(run)] += width * QUANTUM_MS
  capacity = sum(state.nodes.values())
  lag, _ = update_lag(state.lag, state.seen, service, capacity, scheduler.demand(state, capacity), state.weights)
  return lambda run: lag.get(state.account(run), 0.0)


def plan_node(assignments, workers, total, claimed=()):
  """Return desired GPU blocks, time to the next decision, and workers requiring a new physical binding."""
  desired = {a.worker: a for a in assignments}
  exclusive = {g for wid, w in workers.items() if wid in desired and desired[wid].spec.exclusive for g in w.current}
  fixed, occupied, restart, candidates, deadlines = {}, set(), set(), {}, []
  for wid, w in workers.items():
    if w.busy or w.holding or wid in claimed or wid not in desired:
      occupied.update(w.current)
      if (w.holding or wid in claimed) and wid in desired and w.current:
        fixed[wid] = w.current
  for wid, a in desired.items():
    w = workers.get(wid, LocalWorker())
    if w.busy or wid in fixed:
      continue
    trainer = a.spec.role == Role.TRAINER
    if not w.current and not exclusive.isdisjoint(w.bound):
      if not trainer or (a.ready and any(exclusive.isdisjoint(b) for b in blocks(total, a.spec.gpus))):
        restart.add(wid)
        continue
    grace = CYCLE_GRACE_S - w.idle_s if trainer and w.current and w.idle_s is not None else 0.0
    if grace > 0:
      deadlines.append(grace)
    open_step = any(
      open_grad and (run in a.closing or (not a.ready and grace > 0)) for run, open_grad in w.loaded.items()
    )
    demand = Demand.FINISH_STEP if open_step else (Demand.READY if a.ready or grace > 0 else Demand.IDLE)
    quantum = min(MAX_QUANTUM_MS / 1000, max(QUANTUM_MS / 1000, w.switch_s / SWITCH_BUDGET))
    candidates[wid] = Worker(
      a.spec.gpus,
      a.spec.exclusive,
      demand,
      w.current,
      w.bound,
      trainer and a.spec.gpus > 1,
      max(0.0, quantum - w.resident_s) if w.resident_s is not None else 0.0,
      a.lag if a.ready else -w.gpu_ms,
      w.gpu_ms,
    )
  blocks_by_worker, wake = plan_gpus(candidates, total, 0.0, occupied)
  if wake is not None:
    deadlines.append(wake)
  return fixed | blocks_by_worker, min(deadlines, default=None), restart
