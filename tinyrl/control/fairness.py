"""GPU fairness from fleet service to local block selection and command order — pure data in, pure data out."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from tinyrl.types import Cmd, entry_sort_key

Demand = StrEnum("Demand", "IDLE READY FINISH_STEP")  # FINISH_STEP holds GPUs across forward_backward -> optim_step


@dataclass(frozen=True)
class Worker:
  gpus: int = 1
  exclusive: bool = False
  demand: Demand = Demand.READY
  current: tuple[int, ...] = ()
  bound: tuple[int, ...] = ()
  gang: bool = False
  hold_until: float = 0.0
  lag: float = 0.0
  gpu_ms: float = 0.0


def waterfill(capacity: float, demand: dict[str, float], weights: dict[str, float] | None = None) -> dict[str, float]:
  """Weighted progressive max-min fair rates `r*[a] <= demand[a]` summing to `min(capacity, sum(demand))`."""
  rates = {}
  pending = {a: float(d) for a, d in demand.items() if d > 0}
  weight = {a: max(1e-9, float((weights or {}).get(a, 1.0))) for a in pending}
  while pending:
    total_weight = sum(weight[a] for a in pending)
    level = min(pending[a] / weight[a] for a in pending)
    if level * total_weight >= capacity - 1e-9:
      return rates | {a: capacity * weight[a] / total_weight for a in pending}
    for a in [a for a in pending if pending[a] / weight[a] <= level + 1e-9]:
      rates[a] = pending.pop(a)
      capacity -= rates[a]
  return rates


def update_lag(lag: dict, seen: dict, service: dict, cap: float, demand: dict, weights: dict | None = None):
  """Add each account's share of newly billed GPU-ms and subtract the GPU-ms it actually received."""
  lag = dict(lag)
  accounts = lag.keys() | service.keys() | demand.keys()
  used = {a: max(0.0, service.get(a, 0.0) - seen.get(a, 0.0)) for a in accounts}
  seen = seen | {a: max(seen.get(a, 0.0), v) for a, v in service.items()}
  rates = waterfill(max(cap, 1.0), {a: max(demand.get(a, 0), float(used[a] > 0)) for a in accounts}, weights)
  work, total_rate = sum(used.values()), sum(rates.values())
  if work > 0 and total_rate > 0:
    for a, rate in rates.items():
      lag[a] = lag.get(a, 0.0) + work * (rate / total_rate) - used[a]
    # Idle accounts repay negative debt as active peers run, without banking positive credit.
    weight = {a: max(1e-9, float((weights or {}).get(a, 1.0))) for a in accounts}
    active_weight = sum(weight[a] for a in rates)
    for a in lag.keys() - rates.keys():
      lag[a] = min(0.0, lag[a] + work * weight[a] / (active_weight + weight[a]))
  return {a: v for a, v in lag.items() if a in rates or v < -1e-9}, seen


def rank_runs(heads: list[Cmd], ready_lag, service: dict[str, float]) -> list[str]:
  """Order ready runs by account EEVDF lag, least per-run service, then numeric arrival order."""
  ready = {cmd.run: cmd for cmd in reversed(heads)}
  floor = min((service[run] for run in ready if run in service), default=0.0)
  key = lambda cmd: (round(-ready_lag(cmd.run), 3), service.get(cmd.run, floor), entry_sort_key(cmd.entry_id))  # noqa: E731
  return [c.run for c in sorted(ready.values(), key=key)]


def admit_samples(heads, queued, account, ready_lag, service, ver_of, active_acct, active_vers, slots, max_loras):
  """Fill up to `slots` for the leading account (draining first if another account leads), capped by `max_loras`."""
  if slots <= 0 or not heads:
    return []
  ranked = rank_runs(heads, ready_lag, service)
  target = account(ranked[0])
  if active_acct and target != active_acct:
    return []
  vers, runs = {v for v in active_vers if v is not None}, deque()
  for r in ranked:
    if account(r) == target and ((v := ver_of(r)) is None or v in vers or len(vers) < max_loras):
      if v is not None:
        vers.add(v)
      if q := deque(queued(r, slots)):
        runs.append(q)
  out = []
  while len(out) < slots and runs:
    q = runs.popleft()
    out.append(q.popleft())
    if q:
      runs.append(q)
  return out


def blocks(total: int, width: int) -> list[tuple[int, ...]]:
  step = width if width > 0 and total % width == 0 else 1
  return [tuple(range(i, i + width)) for i in range(0, total - width + 1, step)] if 1 <= width <= total else []


def plan_gpus(workers: dict[str, Worker], total: int, now: float, occupied: set[int] = frozenset()):
  """Choose each local worker's GPU block and the next contested quantum deadline."""
  held = occupied | {g for w in workers.values() for g in w.current}
  pinned = {g for w in workers.values() if w.gang for g in w.bound}
  exclusive_waiting = any(w.exclusive and w.demand != Demand.IDLE and not w.current for w in workers.values())

  keep, deadlines = {}, []
  for name, w in workers.items():
    if not w.current:
      continue
    protected = w.demand != Demand.IDLE and not exclusive_waiting
    if protected and w.hold_until > now:
      deadlines.append(w.hold_until)
    if w.exclusive or (protected and (w.hold_until > now or w.demand == Demand.FINISH_STEP)):
      keep[name] = w.current

  taken = set().union(*keep.values())
  exclusive_gpus = {g for name, block in keep.items() if workers[name].exclusive for g in block}
  reserved, waiting = set(), False

  def priority(name: str) -> tuple:
    w = workers[name]
    return w.exclusive, round(w.lag, 3), round(-w.gpu_ms, 3), bool(w.current), name

  candidates = [name for name, w in workers.items() if name not in keep and w.demand != Demand.IDLE]
  for name in sorted(candidates, key=priority, reverse=True):
    w = workers[name]
    options = [w.bound] if w.bound else blocks(total, w.gpus)
    if free := [b for b in options if taken.isdisjoint(b) and (b == w.current or reserved.isdisjoint(b))]:
      avoid = pinned if w.exclusive else set()
      block = min(free, key=lambda b: (b != w.current, not avoid.isdisjoint(b), not held.isdisjoint(b)))
      keep[name] = block
      taken.update(block)
      if w.exclusive:
        exclusive_gpus.update(block)
    else:
      waiting = True
      reserved.update(next((b for b in options if exclusive_gpus.isdisjoint(b)), ()))

  # An uncontested resident stays warm after its quantum expires.
  keep.update((name, w.current) for name, w in workers.items() if w.current and taken.isdisjoint(w.current))
  return keep, min(deadlines, default=None) if waiting else None
