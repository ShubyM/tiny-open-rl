"""Pure fleet pager and EEVDF GPU scheduler: zero I/O, zero locks, zero virtual `Proc` layer."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from tinyrl import types
from tinyrl.helpers import getenv
from tinyrl.types import Cmd, Role, WorkerSpec, entry_sort_key

QUANTUM_MS = getenv("QUANTUM_MS", 2000)
MAX_QUANTUM_MS = getenv("MAX_QUANTUM_MS", 15000)
CYCLE_GRACE_S = getenv("CYCLE_GRACE_MS", 100) / 1000
REBALANCE_MS = getenv("REBALANCE_MS", 20_000)
SWITCH_BUDGET = max(0.01, float(getenv("SWITCH_BUDGET", 0.1)))


class Demand(StrEnum):
  IDLE = "idle"
  READY = "ready"
  FINISH_STEP = "finish_step"  # holds GPUs across forward_backward -> optim_step


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


@dataclass(frozen=True)
class NodeStats:
  used: int
  exclusive: int
  spare: int
  load: float
  min_lag: float


def quantum_s(switch_s: float) -> float:
  amortized = switch_s / SWITCH_BUDGET if switch_s else 0.0
  return min(MAX_QUANTUM_MS / 1000, max(QUANTUM_MS / 1000, amortized))


def waterfill(capacity: float, demand: dict[str, float], weights: dict[str, float] | None = None) -> dict[str, float]:
  """Weighted progressive max-min fair rates `r*[a] <= demand[a]` summing to `min(capacity, sum(demand))`."""
  rates: dict[str, float] = {}
  pending = {acct: float(d) for acct, d in demand.items() if d > 0}
  weight = {acct: max(1e-9, float((weights or {}).get(acct, 1.0))) for acct in pending}
  while pending:
    total_weight = sum(weight[acct] for acct in pending)
    level = min(pending[acct] / weight[acct] for acct in pending)
    if level * total_weight >= capacity - 1e-9:
      for acct in pending:
        rates[acct] = capacity * weight[acct] / total_weight
      return rates
    saturated = [acct for acct in pending if pending[acct] / weight[acct] <= level + 1e-9]
    for acct in saturated:
      rates[acct] = pending.pop(acct)
      capacity -= rates[acct]
  return rates


def update_lag(
  lag: dict[str, float],
  seen: dict[str, float],
  service: dict[str, float],
  capacity: float,
  demand: dict[str, float],
  weights: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
  """Advance each account's EEVDF lag from monotonic `service - seen` GPU-ms deltas."""
  next_lag = dict(lag)
  accounts = next_lag.keys() | service.keys() | demand.keys()
  used = {acct: max(0.0, service.get(acct, 0.0) - seen.get(acct, 0.0)) for acct in accounts}
  next_seen = seen | {acct: max(seen.get(acct, 0.0), total) for acct, total in service.items()}

  active_demand = {acct: max(demand.get(acct, 0.0), float(used[acct] > 0)) for acct in accounts}
  rates = waterfill(max(capacity, 1.0), active_demand, weights)
  total_work = sum(used.values())
  total_rate = sum(rates.values())
  if total_work > 0 and total_rate > 0:
    for acct, rate in rates.items():
      next_lag[acct] = next_lag.get(acct, 0.0) + total_work * (rate / total_rate) - used[acct]
    weight = {acct: max(1e-9, float((weights or {}).get(acct, 1.0))) for acct in accounts}
    active_weight = sum(weight[acct] for acct in rates)
    for acct in next_lag.keys() - rates.keys():
      repay = total_work * weight[acct] / (active_weight + weight[acct])
      next_lag[acct] = min(0.0, next_lag[acct] + repay)

  live_lag = {acct: val for acct, val in next_lag.items() if acct in rates or val < -1e-9}
  return live_lag, next_seen


def account_demand(runs: list[tuple[str, WorkerSpec, int]], nodes: dict[str, int]) -> dict[str, float]:
  biggest_node = max(nodes.values(), default=0)
  total_gpus = sum(nodes.values())
  want: dict[str, float] = {}
  for acct, spec, backlog in runs:
    if backlog <= 0 or spec.gpus > biggest_node:
      continue
    if spec.elastic:
      needed = max(1, (backlog + types.MAX_SEQS - 1) // types.MAX_SEQS)
      replicas = min(total_gpus // spec.gpus, needed)
    else:
      replicas = 1
    want[acct] = want.get(acct, 0.0) + replicas * spec.gpus
  return want


def ready_lag(lag: dict[str, float], seen: dict[str, float], service: dict[str, float], acct: str) -> float:
  unbilled = max(0.0, service.get(acct, 0.0) - seen.get(acct, 0.0))
  return lag.get(acct, 0.0) - unbilled


def projected_lag(
  lag: dict[str, float],
  seen: dict[str, float],
  service: dict[str, float],
  weights: dict[str, float],
  runs: list[tuple[str, WorkerSpec, int]],
  nodes: dict[str, int],
  holds: list[tuple[str, float]],
) -> Callable[[str], float]:
  if not holds:
    return lambda acct: ready_lag(lag, seen, service, acct)
  projected_service = Counter(service)
  for acct, width in holds:
    projected_service[acct] += width * QUANTUM_MS
  capacity = float(sum(nodes.values()))
  projected, _ = update_lag(lag, seen, projected_service, capacity, account_demand(runs, nodes), weights)
  return lambda acct: projected.get(acct, 0.0)


def rank_runs(heads: list[Cmd], lag_of: Callable[[str], float], service: dict[str, float]) -> list[str]:
  earliest = {cmd.run: cmd for cmd in reversed(heads)}
  floor = min((service[run] for run in earliest if run in service), default=0.0)
  ordered = sorted(
    earliest.values(),
    key=lambda cmd: (round(-lag_of(cmd.run), 3), service.get(cmd.run, floor), entry_sort_key(cmd.entry_id)),
  )
  return [cmd.run for cmd in ordered]


def admit_samples(
  heads: list[Cmd],
  queued: Callable[[str, int], Iterable[Cmd]],
  account: Callable[[str], str],
  lag_of: Callable[[str], float],
  service: dict[str, float],
  version_of: Callable[[str], str | None],
  slots: int,
  max_loras: int,
  in_flight: dict[str, int] | None = None,
  active_versions: Iterable[str | None] = (),
) -> list[Cmd]:
  """Continuously fill free sampler slots up to `max_loras`, equalizing in-flight slots across accounts."""
  if slots <= 0 or not heads:
    return []
  loaded_versions = {ver for ver in active_versions if ver is not None}
  runs_by_acct: dict[str, list[deque[Cmd]]] = {}
  for run in rank_runs(heads, lag_of, service):
    ver = version_of(run)
    if ver is not None and ver not in loaded_versions and len(loaded_versions) >= max_loras:
      continue
    pending = deque(queued(run, slots))
    if not pending:
      continue
    if ver is not None:
      loaded_versions.add(ver)
    runs_by_acct.setdefault(account(run), []).append(pending)

  available = {acct: sum(len(q) for q in qs) for acct, qs in runs_by_acct.items()}
  allocated = dict.fromkeys(runs_by_acct, 0)
  rank_index = {acct: i for i, acct in enumerate(runs_by_acct)}
  active_counts = in_flight or {}
  for _ in range(slots):
    eligible = [acct for acct in runs_by_acct if allocated[acct] < available[acct]]
    if not eligible:
      break
    chosen = min(eligible, key=lambda acct: (active_counts.get(acct, 0) + allocated[acct], rank_index[acct]))
    allocated[chosen] += 1

  admitted: list[Cmd] = []
  for acct, queues in runs_by_acct.items():
    ring = deque(queues)
    for _ in range(allocated[acct]):
      queue = ring.popleft()
      admitted.append(queue.popleft())
      if queue:
        ring.append(queue)
  return admitted


def blocks(total: int, width: int) -> list[tuple[int, ...]]:
  if not 1 <= width <= total:
    return []
  step = width if total % width == 0 else 1
  return [tuple(range(start, start + width)) for start in range(0, total - width + 1, step)]


def plan_gpus(
  workers: dict[str, Worker],
  total: int,
  now: float,
  occupied: set[int] = frozenset(),
  locked: set[int] = frozenset(),
) -> tuple[dict[str, tuple[int, ...]], float | None]:
  """Select non-overlapping GPU blocks on one node and the next contested quantum deadline."""
  held = occupied | locked | {gpu for w in workers.values() for gpu in w.current}
  gang_pinned = {gpu for w in workers.values() if w.gang for gpu in w.bound}
  exclusive_waiting = any(w.exclusive and w.demand != Demand.IDLE and not w.current for w in workers.values())

  keep: dict[str, tuple[int, ...]] = {}
  deadlines: list[float] = []
  for name, worker in workers.items():
    protected = bool(worker.current) and worker.demand != Demand.IDLE and not exclusive_waiting
    if protected and worker.hold_until > now:
      deadlines.append(worker.hold_until)
    must_keep = worker.exclusive or (protected and (worker.hold_until > now or worker.demand == Demand.FINISH_STEP))
    if worker.current and must_keep:
      keep[name] = worker.current

  taken = set(locked).union(*keep.values())
  exclusive_gpus = {gpu for name, block in keep.items() if workers[name].exclusive for gpu in block}
  reserved: set[int] = set()
  waiting = False

  def priority(name: str) -> tuple:
    w = workers[name]
    return (w.exclusive, round(w.lag, 3), round(-w.gpu_ms, 3), bool(w.current), name)

  contenders = [name for name, w in workers.items() if name not in keep and w.demand != Demand.IDLE]
  for name in sorted(contenders, key=priority, reverse=True):
    worker = workers[name]
    options = [worker.bound] if worker.bound else blocks(total, worker.gpus)
    free = [
      block for block in options if taken.isdisjoint(block) and (block == worker.current or reserved.isdisjoint(block))
    ]
    if free:
      avoid = gang_pinned if worker.exclusive else set()
      chosen = min(free, key=lambda b: (b != worker.current, not avoid.isdisjoint(b), not held.isdisjoint(b)))
      keep[name] = chosen
      taken.update(chosen)
      if worker.exclusive:
        exclusive_gpus.update(chosen)
    else:
      waiting = True
      for block in options:
        if exclusive_gpus.isdisjoint(block):
          reserved.update(block)
          break

  for name, worker in workers.items():
    if worker.current and taken.isdisjoint(worker.current):
      keep[name] = worker.current

  next_deadline = min(deadlines, default=None) if waiting else None
  return keep, next_deadline


def sampler_replicas(spec: WorkerSpec, backlogs: list[int]) -> int:
  if spec.exclusive:
    return 1
  ready = [count for count in backlogs if count > 0]
  lora_cap = types.MAX_LORAS if spec.lora else 1
  by_seqs = (sum(ready) + types.MAX_SEQS - 1) // types.MAX_SEQS
  by_loras = (len(ready) + lora_cap - 1) // lora_cap
  return max(1, by_seqs, by_loras)


def node_stats(node: str, total: int, placed: list[Placed], skip: str = "") -> NodeStats:
  on_node = [p for p in placed if p.node == node and p.worker != skip]
  used = sum(p.spec.gpus for p in on_node)
  exclusive = sum(p.spec.gpus for p in on_node if p.spec.exclusive)
  shared_busy = sum(p.spec.gpus for p in on_node if not p.spec.exclusive and p.busy)
  shared_capacity = total - exclusive
  load = shared_busy / max(1, shared_capacity)
  min_lag = min((p.lag for p in on_node if not p.spec.exclusive and p.busy), default=float("inf"))
  return NodeStats(used=used, exclusive=exclusive, spare=shared_capacity - shared_busy, load=load, min_lag=min_lag)


def can_host(
  node: str,
  nodes: dict[str, int],
  placed: list[Placed],
  spec: WorkerSpec,
  skip: str = "",
  only_spare: bool = False,
) -> bool:
  if node not in nodes:
    return False
  stats = node_stats(node, nodes[node], placed, skip=skip)
  committed = stats.used if only_spare else stats.exclusive
  return committed + spec.gpus <= nodes[node]


def choose_node(
  spec: WorkerSpec,
  nodes: dict[str, int],
  placed: list[Placed],
  *,
  near: set[str] | frozenset[str] = frozenset(),
  owed_gpus: dict[str, int] | None = None,
  only_spare: bool = False,
  prefer_cool: bool = False,
) -> str | None:
  paired_width = spec.gpus * (2 if spec.role == Role.TRAINER and not spec.lora else 1)

  def rank(node: str) -> tuple:
    stats = node_stats(node, nodes[node], placed)
    free_after_owed = nodes[node] - stats.used - (owed_gpus or {}).get(node, 0)
    cool_key = (round(stats.load, 6), round(stats.min_lag, 3)) if prefer_cool else ()
    return (free_after_owed < paired_width, node not in near, *cool_key, stats.used / nodes[node], node)

  eligible = [node for node in sorted(nodes) if can_host(node, nodes, placed, spec, only_spare=only_spare)]
  return min(eligible, key=rank, default=None)


def rebalance_target(target: Placed, nodes: dict[str, int], placed: list[Placed]) -> str | None:
  if target.spec.exclusive or not target.busy or target.node not in nodes:
    return None
  if node_stats(target.node, nodes[target.node], placed).spare >= 0:
    return None

  candidates = {
    node: node_stats(node, nodes[node], placed)
    for node in sorted(nodes)
    if node != target.node and can_host(node, nodes, placed, target.spec)
  }
  with_spare = [node for node, stats in candidates.items() if stats.spare >= target.spec.gpus]
  if with_spare:
    return min(with_spare, key=lambda node: (candidates[node].used / nodes[node], node))

  without = node_stats(target.node, nodes[target.node], placed, skip=target.worker)
  current_heat = (round(without.load, 6), round(without.min_lag, 3))
  cooler = [
    (round(stats.load, 6), round(stats.min_lag, 3), node)
    for node, stats in candidates.items()
    if target.lag - stats.min_lag >= REBALANCE_MS - 1e-9
    and (round(stats.load, 6), round(stats.min_lag, 3)) < current_heat
  ]
  return min(cooler)[2] if cooler else None
