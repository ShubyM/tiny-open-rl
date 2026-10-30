"""Run fairness and worker placement. The same fold/decide functions drive the cluster and simulator.

A worker is one process lifetime on fixed GPUs. A run owns its queue, entitlement and saved training state;
its worker can change only after a fenced, durable release. Redis projects assignments and carries commands.
No model names encode policy, and no GPU indices have reserved roles.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
from dataclasses import dataclass, field

from tinyrl.bus import Bus, Event, key
from tinyrl.helpers import getenv

QUANTUM = getenv("QUANTUM", 30.0)
TICK_MS = getenv("TICK_MS", 100)
BATCH = 10_000
Key = tuple[str, int]


@dataclass
class Run:
  base: str = ""
  rank: int = 0
  role: str = "trainer"
  exclusive: bool = False
  weight: float = 1.0
  memory: int = 0
  pending: int = 0
  worker: str = ""
  owed: float = 0.0
  served: float = 0.0
  vruntime: float = 0.0  # weighted attained service, aligned to the active cohort on becoming runnable
  entitled: float = 0.0
  moving: str = ""
  target: str = ""
  last_move: float = 0.0
  move_cost: float = 0.0


@dataclass
class Worker:
  node: str = ""
  gpus: tuple[int, ...] = ()
  base: str = ""
  rank: int = 0
  role: str = "trainer"
  exclusive: bool = False
  env: dict[str, str] = field(default_factory=dict)
  memory: int = 0
  reserved_for: str = ""
  started: bool = False
  stopping: bool = False
  exited: bool = False  # confirmed exit, or irrevocable fencing on a node removed from available capacity
  held: bool = False
  asked: bool = False
  since: float = 0.0
  revoked: bool = False
  idle: bool = False
  boot: float = 0.0
  turn: str = ""


@dataclass
class State:
  quantum: float = QUANTUM
  n: int = 0
  now: float = 0.0
  vtime: float = 0.0
  nodes: dict[str, tuple[int, int]] = field(default_factory=dict)
  runs: dict[str, Run] = field(default_factory=dict)
  workers: dict[str, Worker] = field(default_factory=dict)


def fair_shares(weights: dict[str, float], capacity: int) -> dict[str, float]:
  """Weighted max-min GPU occupancy for eligible single-GPU runs, capped at one GPU each."""
  left, remaining, shares = float(capacity), dict(weights), {}
  while remaining:
    unit = left / sum(remaining.values())
    capped = [r for r, weight in remaining.items() if unit * weight >= 1]
    if not capped:
      shares.update((r, unit * weight) for r, weight in remaining.items())
      break
    for r in capped:
      shares[r], left = 1.0, left - 1
      del remaining[r]
  return shares


def eligible(s: State, r: Run) -> bool:
  if not r.pending or not any(count and (not memory or r.memory <= memory) for count, memory in s.nodes.values()):
    return False
  w = s.workers.get(r.worker)
  return not w or (w.node in s.nodes and not w.exited and not w.stopping)


def advance(s: State, now: float) -> None:
  elapsed = max(0.0, now - s.now)
  shares = fair_shares({r: x.weight for r, x in s.runs.items() if eligible(s, x)}, sum(n for n, _ in s.nodes.values()))
  for r, share in shares.items():
    s.runs[r].owed += share * elapsed
    s.runs[r].entitled += share * elapsed
  s.now = max(s.now, now)


def keys(w: Worker) -> tuple[Key, ...]:
  return tuple((w.node, gpu) for gpu in w.gpus)


def members(s: State, worker: str, ready: bool = False) -> list[str]:
  return [r for r, x in s.runs.items() if x.worker == worker and (not ready or x.pending)]


def priority(s: State, worker: str) -> float:
  return -min((s.runs[r].vruntime for r in members(s, worker, True)), default=float("inf"))


def estimate(s: State, r: Run) -> int:
  return r.memory or max(
    (x.memory for x in s.runs.values() if (x.base, x.rank, x.role) == (r.base, r.rank, r.role)), default=0
  )


def compatible(s: State, r: Run, worker: str, run_id: str = "") -> bool:
  w = s.workers[worker]
  if w.stopping or w.exited or w.node not in s.nodes or (w.reserved_for and w.reserved_for != run_id):
    return False
  if (w.base, w.role, w.rank > 0) != (r.base, r.role, r.rank > 0):
    return False
  if w.reserved_for == run_id:
    return True
  if w.exclusive or r.exclusive or (r.role == "trainer" and r.rank == 0):
    return False
  needed = estimate(s, r)
  # Unknown memory is not permission to pack. Assignments reserve measured headroom immediately.
  return bool(needed and w.memory and w.memory + needed <= 0.8 * s.nodes[w.node][1])


def load(s: State, k: Key) -> int:
  return sum(
    max(len(members(s, name, True)), int(not w.started and bool(members(s, name))))
    + bool(w.reserved_for and w.reserved_for not in members(s, name))
    for name, w in s.workers.items()
    if not w.exited and k in keys(w)
  )


def location(s: State, run_id: str, exclude: set[Key] = frozenset()) -> tuple[str, Key] | None:
  r = s.runs[run_id]
  candidates = []
  for node, (count, memory) in sorted(s.nodes.items()):
    if r.memory and memory and r.memory > memory:
      continue
    for gpu in range(count):
      k = node, gpu
      if k not in exclude:
        # Prefer an idle GPU; within equal load reuse a compatible warm process.
        candidates.append(((load(s, k), 1, node, gpu, ""), ("", k)))
  for name, w in s.workers.items():
    if name != r.worker and compatible(s, r, name, run_id) and not any(k in exclude for k in keys(w)):
      k = keys(w)[0]
      candidates.append(((load(s, k), 0, *k, name), (name, k)))
  return min(candidates)[1] if candidates else None


def create_worker(s: State, run_id: str, at: Key) -> tuple[str, dict]:
  r = s.runs[run_id]
  env = {"TINYRL_ROLE": r.role, "TINYRL_LORA": str(int(r.rank > 0))}
  if r.role == "sampler":
    env["TINYRL_GPU_MEM"] = "0.8"
  return "worker_created", dict(
    worker=f"w{s.n + 1}",
    node=at[0],
    gpus=[at[1]],
    base=r.base,
    rank=r.rank,
    role=r.role,
    exclusive=r.exclusive,
    env=env,
    memory=0,
    reserved_for=run_id,
  )


def fold(state: State, e: Event) -> State:
  """Deterministic replay. Worker IDs are never reused, so late facts cannot affect a replacement."""
  s = copy.deepcopy(state)
  ready = {name for name, x in s.runs.items() if eligible(s, x)}
  advance(s, e.t)
  s.n += 1
  b = e.body
  r, w = s.runs.get(b.get("run", "")), s.workers.get(b.get("worker", ""))
  match e.kind:
    case "node_up":
      s.nodes[b["node"]] = (b["gpus"], b.get("memory", 0))
    case "node_lost":
      s.nodes.pop(b["node"], None)
      for worker in s.workers.values():
        if worker.node == b["node"]:
          worker.stopping = True
    case "run_created":
      if b["run"] not in s.runs:
        fields = {k: b[k] for k in ("base", "rank", "role", "exclusive", "weight", "memory") if k in b}
        s.runs[b["run"]] = Run(**fields, worker=b.get("pool", ""))
    case "run_deleted":
      s.runs.pop(b["run"], None)
    case "run_queued" if r:
      r.pending = max(0, b["count"])
    case "worker_created":
      if b["worker"] not in s.workers:
        fields = {k: v for k, v in b.items() if k in Worker.__dataclass_fields__}
        fields["gpus"] = tuple(b["gpus"])
        s.workers[b["worker"]] = Worker(**fields)
    case "run_assigned" if r and w and not r.worker and not w.stopping and not w.exited:
      r.worker, r.moving, r.target = b["worker"], "", ""
      r.memory = b.get("memory", r.memory)
      w.memory += r.memory
      w.reserved_for = ""
    case "run_move_requested" if r and r.worker == b["source"] and not r.moving:
      r.moving, r.target, r.last_move = b["token"], b["target"], e.t
    case "run_released" if r and r.worker == b["source"]:
      source = s.workers.get(r.worker)
      if (r.moving and r.moving == b.get("token")) or (source and source.exited and b.get("reason") == "exited"):
        if source:
          source.memory = max(0, source.memory - r.memory)
        r.worker = ""
        if r.moving:
          r.move_cost = max(r.move_cost, e.t - r.last_move)
        else:
          r.target = ""
        r.moving = ""
    case "worker_stopped" if w:
      w.stopping = True
    case "started" if w and not w.exited and not w.started and b.get("node", w.node) == w.node:
      w.started = True
    case "exited" | "worker_fenced" if w and b.get("node", w.node) == w.node:
      w.exited, w.stopping, w.held, w.asked, w.revoked = True, True, False, False, False
    case "worker_memory" if w and not w.exited:
      measured = b.get("run_memory", {})
      for name, memory in measured.items():
        if name in s.runs and s.runs[name].worker == b["worker"]:
          s.runs[name].memory = memory
      w.memory = b["memory"] + sum(s.runs[name].memory for name in members(s, b["worker"]) if name not in measured)
    case "wants_gpu" if w and w.started and not w.exited and not w.stopping:
      w.idle = False
      if not w.held:
        w.asked = True
    case "granted" if w and w.asked and not w.exited:
      w.held, w.asked, w.revoked, w.since = True, False, False, e.t
      w.turn = b["turn"]
    case "revoked" if w and w.held and b.get("turn") == w.turn:
      w.revoked = True
    case "released" if w and b.get("turn") == w.turn:
      w.held, w.revoked = False, False
    case "idle" if w and b.get("turn") == w.turn:
      w.idle = True
    case "batch_done" if w and not w.exited:
      w.boot = max(w.boot, b.get("boot", 0.0))
    case "service" if r and r.worker == b["worker"] and w and not w.exited:
      seconds = max(0.0, b["seconds"])
      r.served += seconds
      r.owed -= seconds
      r.vruntime += seconds / r.weight
      s.vtime = max(s.vtime, min((x.vruntime for x in s.runs.values() if eligible(s, x)), default=r.vruntime))
  for name, x in s.runs.items():
    if name not in ready and eligible(s, x):
      x.vruntime = max(x.vruntime, s.vtime)
  return s


def next_decision(s: State, now: float) -> tuple[str, dict] | None:
  # Finish previously requested handoffs before considering new placements.
  for name, r in sorted(s.runs.items(), key=lambda item: (item[1].vruntime, item[0])):
    if r.worker or r.moving:
      continue
    target = r.target if r.target in s.workers and compatible(s, r, r.target, name) else ""
    reserved = next(
      (wid for wid, w in s.workers.items() if w.reserved_for == name and not w.stopping and not w.exited), ""
    )
    if target or reserved:
      return "run_assigned", dict(run=name, worker=target or reserved, memory=estimate(s, r))
    if at := location(s, name):
      worker, gpu = at
      return (
        ("run_assigned", dict(run=name, worker=worker, memory=estimate(s, r)))
        if worker
        else create_worker(s, name, gpu)
      )
  for name, w in sorted(s.workers.items()):
    r = s.runs.get(w.reserved_for)
    if r and r.worker and r.worker != name and not r.moving and not w.stopping and not w.exited:
      return "run_move_requested", dict(run=w.reserved_for, source=r.worker, target=name, token=f"m{s.n + 1}")
    reserved = w.reserved_for in s.runs
    if not w.stopping and not members(s, name) and not reserved:
      return "worker_stopped", {"worker": name}
  occupied = {k for w in s.workers.values() if w.held and not w.exited for k in keys(w)}
  waiting = sorted(
    (name for name, w in s.workers.items() if w.asked and not w.stopping and not w.exited),
    key=lambda name: (-priority(s, name), name),
  )
  for name in waiting:
    if not any(k in occupied for k in keys(s.workers[name])):
      return "granted", {"worker": name, "turn": f"t{s.n + 1}"}
  for name, w in sorted(s.workers.items()):
    contending = any(set(keys(w)) & set(keys(s.workers[other])) for other in waiting)
    if w.held and not w.revoked and (w.stopping or (contending and (w.idle or now - w.since >= s.quantum))):
      return "revoked", {"worker": name, "turn": w.turn}
  # Keep moves infrequent. Fairness can force one even if throughput alone would prefer permanent residency.
  if not any(r.moving for r in s.runs.values()):
    for name, r in sorted(s.runs.items(), key=lambda item: (item[1].vruntime, item[0])):
      source = s.workers.get(r.worker)
      if not r.pending or not source or source.stopping or not source.started or now - r.last_move < s.quantum:
        continue
      source_keys = set(keys(source))
      source_load = max(load(s, k) for k in source_keys)
      if source_load < 2 or not (at := location(s, name, source_keys)):
        continue
      target, gpu = at
      target_load = load(s, gpu)
      peers = [
        x.vruntime for x in s.runs.values() if x.pending and x.worker in s.workers and gpu in keys(s.workers[x.worker])
      ]
      unfair = max(peers, default=r.vruntime) - r.vruntime > s.quantum
      gain = s.quantum * (1 / (target_load + 1) - 1 / source_load)
      cost = r.move_cost + (s.workers[target].boot if target else source.boot)
      if target_load >= source_load or not (unfair or gain > cost):
        continue
      if not target:
        # A reserved target is created once and remains available until the source acknowledges the move.
        reserved = next(
          (wid for wid, w in s.workers.items() if w.reserved_for == name and not w.stopping and not w.exited), ""
        )
        if not reserved:
          return create_worker(s, name, gpu)
        target = reserved
      return "run_move_requested", dict(run=name, source=r.worker, target=target, token=f"m{s.n + 1}")
  return None


def decide(state: State, now: float) -> list[tuple[str, dict]]:
  s, out = copy.deepcopy(state), []
  advance(s, now)
  while action := next_decision(s, now):
    out.append(action)
    s = fold(s, Event("", now, *action))
  return out


async def run(bus: Bus, quantum: float = QUANTUM) -> None:
  state, after = State(quantum=quantum), "0-0"
  while True:
    if await bus.log_size() < state.n:
      state, after = State(quantum=quantum), "0-0"
    events = await bus.events(after, count=BATCH, block_ms=TICK_MS)
    for e in events:
      state = fold(state, e)
    after = events[-1].id if events else after
    if len(events) == BATCH:
      continue
    if lost := sorted(set(state.nodes) - set(await bus.nodes())):
      for node in lost:
        await bus.emit("node_lost", node=node)
      continue
    fenced = False
    for name, w in state.workers.items():
      if w.node not in state.nodes and not w.exited:
        fenced = await bus.fence_worker(name, after)
        if fenced:
          break
      if w.exited and members(state, name):
        await bus.unassign_worker(name)
    if fenced:
      continue
    seconds, micros = await bus.r.time()
    now = max(state.now, seconds + micros / 1_000_000)
    for kind, body in decide(state, now)[:1]:
      if kind == "run_assigned":
        await bus.assign(body["run"], body["worker"], memory=body["memory"], after=after)
      elif kind == "run_move_requested":
        await bus.request_move(body["run"], body["target"], body["token"], after=after)
      else:
        await bus.emit_decision(kind, after, **body)
    await bus.r.set(key("state"), json.dumps(dataclasses.asdict(state), sort_keys=True))


if __name__ == "__main__":
  asyncio.run(run(Bus()))
