"""GPU fairness from fleet service to one dispatch.

update_lag compares each account's fair share with its billed GPU time.
plan_gpus assigns GPU blocks to workers. select_runs picks one account for a worker's batch.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from tinyrl.bus import Cmd, Role

if TYPE_CHECKING:
  from tinyrl.control.node_daemon import Lifetime
  from tinyrl.control.scheduler import State


class Demand(Enum):
  IDLE = "idle"
  READY = "ready"
  FINISH_STEP = "finish_step"  # The trainer may keep its GPUs across the optimizer-step boundary.


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
    # Satisfy accounts capped by demand, then divide the remaining capacity among the rest.
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
  # The last command can finish between ticks. Include the account whose service just arrived.
  rates = waterfill(max(cap, 1.0), {a: max(demand.get(a, 0), float(used[a] > 0)) for a in accounts}, weights)
  work, total_rate = sum(used.values()), sum(rates.values())
  if work > 0 and total_rate > 0:
    for a, rate in rates.items():
      lag[a] = lag.get(a, 0.0) + work * (rate / total_rate) - used[a]
    # Idle accounts can repay a negative balance, but cannot bank credit for later.
    weight = {a: max(1e-9, float((weights or {}).get(a, 1.0))) for a in accounts}
    active_weight = sum(weight[a] for a in rates)
    for a in lag.keys() - rates.keys():
      lag[a] = min(0.0, lag[a] + work * weight[a] / (active_weight + weight[a]))
  return {a: v for a, v in lag.items() if a in rates or v < -1e-9}, seen


def select_runs(state: State, heads: list[Cmd], service: dict[str, float], width: int) -> list[str]:
  """Choose one ready account, then batch up to `width` of its least-served runs."""
  ready = {cmd.run: cmd for cmd in reversed(heads)}
  floor = min((service[run] for run in ready if run in service), default=0.0)

  def rank(cmd: Cmd) -> tuple:
    return -state.ready_lag(cmd.run), service.get(cmd.run, floor), tuple(map(int, cmd.entry_id.split("-")))

  runs = [cmd.run for cmd in sorted(ready.values(), key=rank)]
  if not runs:
    return []
  account = state.account(runs[0])
  return [run for run in runs if state.account(run) == account][:width]


def blocks(total: int, width: int) -> list[tuple[int, ...]]:
  step = width if width > 0 and total % width == 0 else 1
  return [tuple(range(i, i + width)) for i in range(0, total - width + 1, step)] if 1 <= width <= total else []


def plan_gpus(
  lives: dict[str, Lifetime], on: dict[str, tuple[int, ...]], total: int, now: float, draining: set[int] = frozenset()
) -> tuple[dict[str, tuple[int, ...]], float | None]:
  """Return desired blocks and the next contested quantum deadline. The node enforces the handoff."""
  current = {p: on.get(life.life, ()) for p, life in lives.items()}
  occupied = set().union(*on.values())
  pinned = {
    g for life in lives.values() if life.spec.role == Role.TRAINER and life.spec.gpus > 1 for g in life.bound_gpus
  }
  exclusive_waiting = any(
    life.spec.exclusive and life.demand != Demand.IDLE and not current[p] for p, life in lives.items()
  )
  eligible = {p: life for p, life in lives.items() if life.wanted()}

  # Keep exclusive residents. Other busy residents finish their quantum or training step unless an exclusive needs in.
  keep, deadlines = {}, []
  for p, life in eligible.items():
    block = current[p]
    if not block or not draining.isdisjoint(block):
      continue
    protected = life.demand != Demand.IDLE and not exclusive_waiting
    if protected and life.hold_until > now:
      deadlines.append(life.hold_until)
    if life.spec.exclusive or protected and (life.hold_until > now or life.demand == Demand.FINISH_STEP):
      keep[p] = block

  taken = set(draining).union(*keep.values())
  exclusive_gpus = {g for p, block in keep.items() if lives[p].spec.exclusive for g in block}
  reserved_for_waiters = set()

  def priority(p: str) -> tuple:
    life = eligible[p]
    return life.spec.exclusive, round(life.ready_lag(), 3), round(-life.gpu_ms, 3), bool(current[p]), p

  waiting = False
  contenders = [p for p, life in eligible.items() if p not in keep and life.demand != Demand.IDLE]
  for p in sorted(contenders, key=priority, reverse=True):
    life = eligible[p]
    options = [life.bound_gpus] if life.bound_gpus else blocks(total, life.spec.gpus)
    free = [b for b in options if taken.isdisjoint(b) and (b == current[p] or reserved_for_waiters.isdisjoint(b))]
    if free:
      avoid = pinned if life.spec.exclusive else set()
      block = min(free, key=lambda b: (b != current[p], not avoid.isdisjoint(b), not occupied.isdisjoint(b), b))
      keep[p] = block
      taken.update(block)
      if life.spec.exclusive:
        exclusive_gpus.update(block)
    else:
      waiting = True
      # Leave room for this gang to assemble before another waiter can take part of its block.
      reserved_for_waiters.update(next((b for b in options if exclusive_gpus.isdisjoint(b)), ()))

  # An uncontested resident can stay warm even after its turn expires.
  keep.update((p, current[p]) for p in eligible if p not in keep and current[p] and taken.isdisjoint(current[p]))
  return keep, min(deadlines, default=None) if waiting else None
