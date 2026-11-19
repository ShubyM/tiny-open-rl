"""GPU fairness from fleet service to one batch — pure data in, pure data out.

`update_lag` compares each account's fair share with its billed GPU time.
`plan_gpus` assigns GPU blocks to local workers. `select_runs` and `RunFairness` pick the next batch fairly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from tinyrl.types import Cmd


class Demand(StrEnum):
  IDLE = "idle"
  READY = "ready"
  FINISH_STEP = "finish_step"  # Hold GPUs across an open forward_backward -> optim_step boundary.


@dataclass(frozen=True)
class GpuSlot:
  gpus: int = 1
  exclusive: bool = False
  demand: Demand = Demand.READY
  current: tuple[int, ...] = ()
  bound: tuple[int, ...] = ()
  is_trainer: bool = False
  hold_until: float = 0.0
  lag: float = 0.0
  gpu_ms: float = 0.0


@dataclass
class RunFairness:
  """Intra-worker weighted-fair-queueing tracker: newly active runs enter at the active service floor."""

  by_run: dict[str, float] = field(default_factory=dict)
  floor: float = 0.0
  total_gpu_ms: float = 0.0

  def sync_active(self, runs) -> None:
    self.by_run = {r: self.by_run.get(r, self.floor) for r in runs}

  def pick(self, candidates: list[Cmd], account, ready_lag, width: int) -> list[str]:
    for c in candidates:
      self.by_run[c.run] = max(self.by_run.get(c.run, self.floor), self.floor)
    return select_runs(candidates, account, ready_lag, self.by_run, width)

  def record(self, run: str, gpu_ms: float) -> None:
    self.by_run[run] = self.by_run.get(run, self.floor) + gpu_ms
    self.total_gpu_ms += gpu_ms

  def finish_batch(self, active_runs: list[str]) -> None:
    if active_runs:
      self.floor = max(self.floor, min(self.by_run.get(r, self.floor) for r in active_runs))


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


def update_lag(lag: dict, billed: dict, service: dict, cap: float, demand: dict, weights: dict | None = None):
  """Add each account's share of newly billed GPU-ms and subtract the GPU-ms it actually received."""
  lag = dict(lag)
  accounts = lag.keys() | service.keys() | demand.keys()
  used = {a: max(0.0, service.get(a, 0.0) - billed.get(a, 0.0)) for a in accounts}
  billed = billed | {a: max(billed.get(a, 0.0), v) for a, v in service.items()}
  # The last command can finish between ticks; include any account whose service just arrived.
  rates = waterfill(max(cap, 1.0), {a: max(demand.get(a, 0), float(used[a] > 0)) for a in accounts}, weights)
  work, total_rate = sum(used.values()), sum(rates.values())
  if work > 0 and total_rate > 0:
    for a, rate in rates.items():
      lag[a] = lag.get(a, 0.0) + work * (rate / total_rate) - used[a]
    # Idle accounts repay negative debt as active peers run, but never bank positive credit while idle.
    weight = {a: max(1e-9, float((weights or {}).get(a, 1.0))) for a in accounts}
    active_weight = sum(weight[a] for a in rates)
    for a in lag.keys() - rates.keys():
      lag[a] = min(0.0, lag[a] + work * weight[a] / (active_weight + weight[a]))
  return {a: v for a, v in lag.items() if a in rates or v < -1e-9}, billed


def select_runs(heads: list[Cmd], account, ready_lag, service: dict[str, float], width: int) -> list[str]:
  """Choose one ready account, then batch up to `width` of its least-served runs."""
  ready = {cmd.run: cmd for cmd in reversed(heads)}
  floor = min((service[run] for run in ready if run in service), default=0.0)

  def rank(cmd: Cmd) -> tuple:
    return round(-ready_lag(cmd.run), 3), service.get(cmd.run, floor), tuple(map(int, cmd.entry_id.split("-")))

  if not (runs := [cmd.run for cmd in sorted(ready.values(), key=rank)]):
    return []
  chosen = account(runs[0])
  return [run for run in runs if account(run) == chosen][:width]


def blocks(total: int, width: int) -> list[tuple[int, ...]]:
  step = width if width > 0 and total % width == 0 else 1
  return [tuple(range(i, i + width)) for i in range(0, total - width + 1, step)] if 1 <= width <= total else []


def plan_gpus(
  slots: dict[str, GpuSlot], total: int, now: float, occupied: set[int] = frozenset()
) -> tuple[dict[str, tuple[int, ...]], float | None]:
  """Return desired GPU blocks and the next contested quantum deadline. The node enforces the handoff."""
  held = occupied | {g for w in slots.values() for g in w.current}
  pinned = {g for w in slots.values() if w.is_trainer for g in w.bound}
  exclusive_waiting = any(w.exclusive and w.demand != Demand.IDLE and not w.current for w in slots.values())

  keep, deadlines = {}, []
  for name, w in slots.items():
    if not w.current:
      continue
    protected = w.demand != Demand.IDLE and not exclusive_waiting
    if protected and w.hold_until > now:
      deadlines.append(w.hold_until)
    if w.exclusive or (protected and (w.hold_until > now or w.demand == Demand.FINISH_STEP)):
      keep[name] = w.current

  taken = set().union(*keep.values())
  exclusive_gpus = {g for name, block in keep.items() if slots[name].exclusive for g in block}
  reserved, waiting = set(), False

  def priority(name: str) -> tuple:
    w = slots[name]
    return w.exclusive, round(w.lag, 3), round(-w.gpu_ms, 3), bool(w.current), name

  contenders = [name for name, w in slots.items() if name not in keep and w.demand != Demand.IDLE]
  for name in sorted(contenders, key=priority, reverse=True):
    w = slots[name]
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

  # An uncontested resident stays warm on its GPUs even after its quantum expires.
  keep.update((name, w.current) for name, w in slots.items() if w.current and taken.isdisjoint(w.current))
  return keep, min(deadlines, default=None) if waiting else None
