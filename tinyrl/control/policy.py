"""Pure fleet pager & EEVDF GPU scheduler: zero I/O, zero locks, zero virtual `Proc` layer."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import StrEnum

from tinyrl import types
from tinyrl.helpers import getenv
from tinyrl.types import Cmd, Role, WorkerSpec, entry_sort_key

QUANTUM_MS, MAX_QUANTUM_MS = getenv("QUANTUM_MS", 2000), getenv("MAX_QUANTUM_MS", 15000)
CYCLE_GRACE_S, REBALANCE_MS = getenv("CYCLE_GRACE_MS", 100) / 1000, getenv("REBALANCE_MS", 20_000)
SWITCH_BUDGET = max(0.01, float(getenv("SWITCH_BUDGET", 0.1)))
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


@dataclass(frozen=True)
class Placed:
  node: str
  worker: str
  spec: WorkerSpec
  busy: bool = False
  lag: float = 0.0
  runs: int = 0


def quantum_s(switch_s: float) -> float:
  return min(MAX_QUANTUM_MS / 1000, max(QUANTUM_MS / 1000, switch_s / SWITCH_BUDGET if switch_s else 0.0))


def waterfill(capacity: float, demand: dict[str, float], weights: dict[str, float] | None = None) -> dict[str, float]:
  """Weighted progressive max-min fair rates `r*[a] <= demand[a]` summing to `min(capacity, sum(demand))`."""
  rates, pending = {}, {a: float(d) for a, d in demand.items() if d > 0}
  weight = {a: max(1e-9, float((weights or {}).get(a, 1.0))) for a in pending}
  while pending:
    total_w = sum(weight[a] for a in pending)
    level = min(pending[a] / weight[a] for a in pending)
    if level * total_w >= capacity - 1e-9:
      return rates | {a: capacity * weight[a] / total_w for a in pending}
    for a in [a for a in pending if pending[a] / weight[a] <= level + 1e-9]:
      rates[a] = pending.pop(a)
      capacity -= rates[a]
  return rates


def update_lag(lag: dict, seen: dict, service: dict, cap: float, demand: dict, weights: dict | None = None):
  """Advance each account's EEVDF lag from monotonic `service - seen` GPU-ms deltas."""
  lag, accts = dict(lag), lag.keys() | service.keys() | demand.keys()
  used = {a: max(0.0, service.get(a, 0.0) - seen.get(a, 0.0)) for a in accts}
  seen = seen | {a: max(seen.get(a, 0.0), v) for a, v in service.items()}
  rates = waterfill(max(cap, 1.0), {a: max(demand.get(a, 0), float(used[a] > 0)) for a in accts}, weights)
  if (work := sum(used.values())) > 0 and (total_rate := sum(rates.values())) > 0:
    for a, rate in rates.items():
      lag[a] = lag.get(a, 0.0) + work * (rate / total_rate) - used[a]
    weight = {a: max(1e-9, float((weights or {}).get(a, 1.0))) for a in accts}
    active_w = sum(weight[a] for a in rates)
    for a in lag.keys() - rates.keys():
      lag[a] = min(0.0, lag[a] + work * weight[a] / (active_w + weight[a]))
  return {a: v for a, v in lag.items() if a in rates or v < -1e-9}, seen


def account_demand(runs: list[tuple[str, WorkerSpec, int]], nodes: dict[str, int]) -> dict[str, float]:
  biggest, cap, want = max(nodes.values(), default=0), sum(nodes.values()), {}
  for acct, spec, backlog in runs:
    if backlog > 0 and spec.gpus <= biggest:
      reps = min(cap // spec.gpus, max(1, (backlog + types.MAX_SEQS - 1) // types.MAX_SEQS)) if spec.elastic else 1
      want[acct] = want.get(acct, 0.0) + reps * spec.gpus
  return want


def ready_lag(lag: dict, seen: dict, service: dict, acct: str) -> float:
  return lag.get(acct, 0.0) - max(0.0, service.get(acct, 0.0) - seen.get(acct, 0.0))


def projected_lag(lag, seen, service, weights, runs, nodes, holds: list[tuple[str, int]]):
  if not holds:
    return lambda acct: ready_lag(lag, seen, service, acct)
  proj_svc = Counter(service)
  for acct, width in holds:
    proj_svc[acct] += width * QUANTUM_MS
  proj, _ = update_lag(lag, seen, proj_svc, float(sum(nodes.values())), account_demand(runs, nodes), weights)
  return lambda acct: proj.get(acct, 0.0)


def rank_runs(heads: list[Cmd], lag_of, service: dict[str, float]) -> list[str]:
  ready = {cmd.run: cmd for cmd in reversed(heads)}
  floor = min((service[r] for r in ready if r in service), default=0.0)
  return [
    c.run
    for c in sorted(
      ready.values(), key=lambda c: (round(-lag_of(c.run), 3), service.get(c.run, floor), entry_sort_key(c.entry_id))
    )
  ]


def admit_samples(heads, queued, account, lag_of, service, ver_of, slots, max_loras, active_acct=None, active_vers=()):
  if slots <= 0 or not heads:
    return []
  target = account((ranked := rank_runs(heads, lag_of, service))[0])
  if active_acct is not None and active_acct != target:
    return []
  vers, runs, out = {v for v in active_vers if v is not None}, deque(), []
  for r in ranked:
    if account(r) == target and ((v := ver_of(r)) is None or v in vers or len(vers) < max_loras):
      vers.update((v,) if v is not None else ())
      if q := deque(queued(r, slots)):
        runs.append(q)
  while len(out) < slots and runs:
    out.append((q := runs.popleft()).popleft())
    if q:
      runs.append(q)
  return out


def blocks(total: int, width: int) -> list[tuple[int, ...]]:
  step = width if width > 0 and total % width == 0 else 1
  return [tuple(range(i, i + width)) for i in range(0, total - width + 1, step)] if 1 <= width <= total else []


def plan_gpus(
  workers: dict[str, Worker], total: int, now: float, occ: set[int] = frozenset(), locked: set[int] = frozenset()
):
  """Select non-overlapping GPU blocks on one node and the next contested quantum deadline."""
  held = occ | locked | {g for w in workers.values() for g in w.current}
  pinned = {g for w in workers.values() if w.gang for g in w.bound}
  excl_wait = any(w.exclusive and w.demand != Demand.IDLE and not w.current for w in workers.values())
  keep, deadlines, reserved, waiting = {}, [], set(), False
  for name, w in workers.items():
    prot = bool(w.current) and w.demand != Demand.IDLE and not excl_wait
    deadlines.extend((w.hold_until,) if prot and w.hold_until > now else ())
    if w.current and (w.exclusive or (prot and (w.hold_until > now or w.demand == Demand.FINISH_STEP))):
      keep[name] = w.current
  taken = set(locked).union(*keep.values())
  excl_gpus = {g for n, b in keep.items() if workers[n].exclusive for g in b}

  def prio(n):
    return (
      workers[n].exclusive,
      round(workers[n].lag, 3),
      round(-workers[n].gpu_ms, 3),
      bool(workers[n].current),
      n,
    )  # noqa: E731

  for name in sorted(
    (n for n, w in workers.items() if n not in keep and w.demand != Demand.IDLE), key=prio, reverse=True
  ):
    w = workers[name]
    opts = [w.bound] if w.bound else blocks(total, w.gpus)
    if free := [b for b in opts if taken.isdisjoint(b) and (b == w.current or reserved.isdisjoint(b))]:
      avoid = pinned if w.exclusive else set()
      keep[name] = blk = min(free, key=lambda b: (b != w.current, not avoid.isdisjoint(b), not held.isdisjoint(b)))
      taken.update(blk)
      excl_gpus.update(blk if w.exclusive else ())
    else:
      waiting = True
      reserved.update(next((b for b in opts if excl_gpus.isdisjoint(b)), ()))
  keep.update((n, w.current) for n, w in workers.items() if w.current and taken.isdisjoint(w.current))
  return keep, min(deadlines, default=None) if waiting else None


def sampler_replicas(spec: WorkerSpec, backlogs: list[int]) -> int:
  ready, cap = [b for b in backlogs if b > 0], (types.MAX_LORAS if spec.lora else 1)
  return (
    1 if spec.exclusive else max(1, (sum(ready) + types.MAX_SEQS - 1) // types.MAX_SEQS, (len(ready) + cap - 1) // cap)
  )


def node_stats(node: str, total: int, placed: list[Placed], skip: str = "") -> tuple[int, int, int, float, float]:
  on = [p for p in placed if p.node == node and p.worker != skip]
  used, excl = sum(p.spec.gpus for p in on), sum(p.spec.gpus for p in on if p.spec.exclusive)
  busy = sum(p.spec.gpus for p in on if not p.spec.exclusive and p.busy)
  floor = min((p.lag for p in on if not p.spec.exclusive and p.busy), default=float("inf"))
  return used, excl, total - excl - busy, busy / max(1, total - excl), floor


def can_host(
  node: str, nodes: dict[str, int], placed: list[Placed], spec: WorkerSpec, skip="", only_spare=False
) -> bool:
  return (
    node in nodes and node_stats(node, nodes[node], placed, skip)[0 if only_spare else 1] + spec.gpus <= nodes[node]
  )


def choose_node(
  spec: WorkerSpec,
  nodes: dict[str, int],
  placed: list[Placed],
  *,
  near=frozenset(),
  owed_gpus=None,
  only_spare=False,
  prefer_cool=False,
):
  paired = spec.gpus * (2 if spec.role == Role.TRAINER and not spec.lora else 1)

  def rank(n: str) -> tuple:
    used, _, _, load, floor = node_stats(n, nodes[n], placed)
    cool = (round(load, 6), round(floor, 3)) if prefer_cool else ()
    return nodes[n] - used - (owed_gpus or {}).get(n, 0) < paired, n not in near, *cool, used / nodes[n], n

  return min(
    (n for n in sorted(nodes) if can_host(n, nodes, placed, spec, only_spare=only_spare)), key=rank, default=None
  )


def rebalance_target(p: Placed, nodes: dict[str, int], placed: list[Placed]) -> str | None:
  if p.spec.exclusive or not p.busy or p.node not in nodes or node_stats(p.node, nodes[p.node], placed)[2] >= 0:
    return None
  fit = {
    n: node_stats(n, nodes[n], placed) for n in sorted(nodes) if n != p.node and can_host(n, nodes, placed, p.spec)
  }
  if spare := [n for n, st in fit.items() if st[2] >= p.spec.gpus]:
    return min(spare, key=lambda n: (fit[n][0] / nodes[n], n))
  _, _, _, lw, fw = node_stats(p.node, nodes[p.node], placed, skip=p.worker)
  opts = [(round(st[3], 6), round(st[4], 3), n) for n, st in fit.items() if p.lag - st[4] >= REBALANCE_MS - 1e-9]
  return min((o for o in opts if o[:2] < (round(lw, 6), round(fw, 3))), default=(0, 0, None))[2]
