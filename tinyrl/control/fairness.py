"""GPU fairness from fleet service to one dispatch — pure data in, pure data out.

update_lag compares each account's fair share with its billed GPU time.
plan_gpus assigns GPU blocks to workers. rank_runs and admit_samples choose commands to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tinyrl.types import Cmd, entry_sort_key


class Demand(StrEnum):
  IDLE = "idle"
  READY = "ready"
  FINISH_STEP = "finish_step"  # The trainer may keep its GPUs across the optimizer-step boundary.


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


def rank_runs(heads: list[Cmd], ready_lag, service: dict[str, float]) -> list[str]:
  """Order ready runs by account EEVDF lag, least per-run service, then numeric arrival order."""
  ready = {cmd.run: cmd for cmd in reversed(heads)}
  floor = min((service[run] for run in ready if run in service), default=0.0)
  return [
    cmd.run
    for cmd in sorted(
      ready.values(),
      key=lambda cmd: (round(-ready_lag(cmd.run), 3), service.get(cmd.run, floor), entry_sort_key(cmd.entry_id)),
    )
  ]


def admit_samples(heads, queued, account, ready_lag, service, ver_of, in_flight, active_vers, max_seqs, max_loras):
  """Continuously admit ready sample commands up to engine concurrency & LoRA limits, fairly across accounts."""
  if (slots := max_seqs - sum(in_flight.values())) <= 0 or not heads:
    return []
  vers, by_acct = set(active_vers), {}
  for r in rank_runs(heads, ready_lag, service):
    if (v := ver_of(r)) in vers or len(vers) < max_loras:
      vers.add(v)
      by_acct.setdefault(account(r), []).append(r)
  qs = {r: list(queued(r, slots)) for rs in by_acct.values() for r in rs}
  demand = {a: in_flight.get(a, 0) + sum(len(qs[r]) for r in by_acct.get(a, ())) for a in in_flight | by_acct}
  caps, out = waterfill(max_seqs, demand), []
  need = {a: max(0, int(caps[a]) - in_flight.get(a, 0)) for a in by_acct}
  rem = slots - sum(need.values())
  for a, rs in by_acct.items():
    take = need[a] + (1 if rem > 0 and sum(len(qs[r]) for r in rs) > need[a] else 0)
    rem -= take - need[a]
    idx = 0
    while take > 0 and rs:
      if qs[r := rs[idx % len(rs)]]:
        out.append(qs[r].pop(0))
        take, idx = take - 1, idx + 1
      else:
        rs.remove(r)
  return out


def blocks(total: int, width: int) -> list[tuple[int, ...]]:
  step = width if width > 0 and total % width == 0 else 1
  return [tuple(range(i, i + width)) for i in range(0, total - width + 1, step)] if 1 <= width <= total else []


def plan_gpus(
  workers: dict[str, Worker], total: int, now: float, occupied: set[int] = frozenset()
) -> tuple[dict[str, tuple[int, ...]], float | None]:
  """Return desired blocks and the next contested quantum deadline. The node enforces the handoff."""
  held = occupied | {g for w in workers.values() for g in w.current}
  pinned = {g for w in workers.values() if w.gang for g in w.bound}
  exclusive_waiting = any(w.exclusive and w.demand != Demand.IDLE and not w.current for w in workers.values())

  # Keep exclusive residents. Other busy residents finish their quantum or training step unless an exclusive needs in.
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

  contenders = [name for name, w in workers.items() if name not in keep and w.demand != Demand.IDLE]
  for name in sorted(contenders, key=priority, reverse=True):
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
      # Leave room for this gang to assemble before another waiter can take part of its block.
      reserved.update(next((b for b in options if exclusive_gpus.isdisjoint(b)), ()))

  # An uncontested resident can stay warm even after its turn expires.
  keep.update((name, w.current) for name, w in workers.items() if w.current and taken.isdisjoint(w.current))
  return keep, min(deadlines, default=None) if waiting else None
