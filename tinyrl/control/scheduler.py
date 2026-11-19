"""Cluster planner: places, scales, and rebalances workers across nodes (`run -> procs -> node`).

Nodes time-slice their own GPUs locally on wall-clock quanta and enforce safety via `gpu_allocations`, `owner`, and `lease`.
`tinyrl:plan` stores only worker placement, run `assignments`, and per-account EEVDF `lag` (in `gpu_ms`).
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import itertools
from dataclasses import dataclass, field

from tinyrl.bus import Bus
from tinyrl.control.fairness import update_lag
from tinyrl.helpers import getenv
from tinyrl.types import Ev, Role

REBALANCE_MS = getenv("REBALANCE_MS", 20_000)
SCALE_BATCH, TICK_MS = getenv("SCALE_BATCH", 4), getenv("TICK_MS", 100)


@dataclass(frozen=True)
class Spec:
  role: Role = Role.TRAINER
  base: str = ""
  lora: bool = False
  exclusive: bool = False
  train_run: str | None = None
  gpus: int = 1

  @property
  def shareable(self) -> bool:
    return self.role == Role.SAMPLER or (self.lora and not self.exclusive)

  @property
  def elastic(self) -> bool:
    return self.role == Role.SAMPLER and not self.exclusive


@dataclass
class Proc:
  spec: Spec = Spec()
  node: str | None = None


@dataclass
class Job:
  spec: Spec
  train_run: str | None = None
  account: str = ""
  owner: str | None = None
  backlog: int = 0

  @property
  def want_replicas(self) -> int:
    return max(1, self.backlog // max(1, SCALE_BATCH)) if self.spec.elastic else 1


def owner_of(token: str | None) -> tuple[str, str]:
  proc, _, rest = (token or "").rsplit(":", 1)[0].partition("@")
  return proc, rest.rsplit("-", 1)[0]


@dataclass
class State:
  next_worker_id: int = 0
  nodes: dict[str, tuple[int, int]] = field(default_factory=dict)
  procs: dict[str, Proc] = field(default_factory=dict)
  assignments: dict[str, list[str]] = field(default_factory=dict)
  jobs: dict[str, Job] = field(default_factory=dict)
  lag: dict[str, float] = field(default_factory=dict)
  billed_service: dict[str, float] = field(default_factory=dict)
  service: dict[str, float] = field(default_factory=dict)
  weights: dict[str, float] = field(default_factory=dict)

  def runs(self, proc: str) -> list[str]:
    return [r for r, ps in sorted(self.assignments.items()) if proc in ps]

  def procs_on(self, node: str, skip: str = "") -> list[tuple[str, Proc]]:
    return [(n, p) for n, p in self.procs.items() if p.node == node and n != skip]

  def has_backlog(self, run: str) -> bool:
    return self.jobs[run].backlog > 0

  def is_busy(self, proc: str) -> bool:
    return any(self.has_backlog(r) for r in self.runs(proc))

  def account(self, run: str) -> str:
    if (job := self.jobs.get(run)) is None:
      return run
    return job.account or (self.account(job.train_run) if job.train_run in self.jobs else run)

  def lag_of(self, proc: str) -> float:
    return max((self.ready_lag(r) for r in self.runs(proc) if self.has_backlog(r)), default=0.0)

  def ready_lag(self, run: str) -> float:
    """Account lag minus GPU service billed since the planner's last tick."""
    account = self.account(run)
    return self.lag.get(account, 0.0) - max(0.0, self.service.get(account, 0.0) - self.billed_service.get(account, 0.0))

  def used(self, node: str) -> int:
    return sum(p.spec.gpus for _, p in self.procs_on(node))

  def excl_used(self, node: str, skip: str = "") -> int:
    return sum(p.spec.gpus for _, p in self.procs_on(node, skip) if p.spec.exclusive)

  def busy_gpus(self, node: str, skip: str = "") -> int:
    return sum(p.spec.gpus for n, p in self.procs_on(node, skip) if not p.spec.exclusive and self.is_busy(n))

  def spare(self, node: str) -> int:
    return self.nodes[node][0] - self.excl_used(node) - self.busy_gpus(node)

  def load(self, node: str, skip: str = "") -> float:
    return self.busy_gpus(node, skip) / max(1, self.nodes[node][0] - self.excl_used(node, skip))

  def can_host(self, node: str, spec: Spec, skip: str = "", spare: bool = False) -> bool:
    return (self.used(node) if spare else self.excl_used(node, skip)) + spec.gpus <= self.nodes[node][0]

  def add_proc(self, spec: Spec) -> str:
    self.next_worker_id = next(n for n in itertools.count(self.next_worker_id + 1) if f"p{n}" not in self.procs)
    name = f"p{self.next_worker_id}"
    self.procs[name] = Proc(spec)
    return name


def from_state(raw: dict) -> State:
  plan, queues = raw["plan"], raw.get("queues", {})
  nodes = {n: (int(v["gpus"]), int(v.get("memory", 0))) for n, v in raw.get("nodes", {}).items()}
  jobs = {}
  for r, info in raw.get("runs", {}).items():
    rec = info["record"]
    excl, tr, role = bool(rec.get("exclusive")), rec.get("train_run"), Role(rec.get("role", "trainer"))
    spec = Spec(role, rec.get("base", ""), rec.get("rank", 0) > 0, excl, tr if excl else None, int(rec.get("gpus", 1)))
    account = rec.get("account") or ("" if tr else rec.get("session", ""))
    jobs[r] = Job(spec, tr, account, info.get("owner"), queues.get(r, 0))
  jobs = {r: j for r, j in jobs.items() if not j.spec.train_run or j.spec.train_run in jobs}
  procs = {
    n: Proc(Spec(**(p["spec"] | {"role": Role(p["spec"]["role"])})), p.get("node"))
    for n, p in plan.get("procs", {}).items()
  }
  assignments = {r: list(ps) for r, ps in plan.get("assignments", {}).items() if r in jobs}
  floats = lambda k: {a: float(v) for a, v in plan.get(k, {}).items()}  # noqa: E731
  next_id, service = int(plan.get("next_worker_id", 0)), dict(raw.get("service", {}))
  return State(
    next_id, nodes, procs, assignments, jobs, floats("lag"), floats("billed_service"), service, floats("weights")
  )


def to_plan(s: State) -> dict:
  head = {"next_worker_id": s.next_worker_id, "procs": {n: dataclasses.asdict(p) for n, p in s.procs.items()}}
  return {**head, "assignments": s.assignments, "lag": s.lag, "billed_service": s.billed_service, "weights": s.weights}


def node_env(gpus: int, gpu_bytes: int) -> dict[str, str]:
  env = {"TINYRL_WARM_SAMPLER": "1"} if gpus >= 2 else {}
  return env | ({"TINYRL_KV_BYTES": str(24 * 2**30)} if gpu_bytes >= 64 * 2**30 else {})


# ***** placement, elastic scaling, and cross-node rebalance *****


def choose_node(s: State, proc: str, only_spare: bool = False) -> str | None:
  """Pick a node for `proc`: prefer room for its paired sampler, then its trainer's node, then least loaded."""
  spec = s.procs[proc].spec
  near = {s.procs[p].node for r in s.runs(proc) for p in s.assignments.get(s.jobs[r].train_run, ())}
  sampled = {j.train_run for j in s.jobs.values()}

  def placement_rank(node: str) -> tuple:
    unpaired = [n for n, p in s.procs_on(node) if p.spec.role == Role.TRAINER and not p.spec.lora]
    owed = sum(s.procs[n].spec.gpus for n in unpaired if not (sampled & set(s.runs(n))))
    used, capacity = s.used(node), s.nodes[node][0]
    paired = spec.gpus * (2 if spec.role == Role.TRAINER and not spec.lora else 1)
    return capacity - used - owed < paired, node not in near, used / capacity

  nodes = [n for n in sorted(s.nodes) if s.can_host(n, spec, spare=only_spare)]
  return min(nodes, key=placement_rank, default=None)


def rebalance_target(s: State, name: str) -> str | None:
  """Move busy `name` off an oversubscribed node: to a spare node, or across nodes once lag reaches `REBALANCE_MS`."""
  p = s.procs[name]
  if not s.is_busy(name) or s.spare(p.node) >= 0:
    return None
  fit = [n for n in sorted(s.nodes) if s.can_host(n, p.spec)]
  if spare := [n for n in fit if s.spare(n) >= p.spec.gpus]:
    return min(spare, key=lambda n: s.used(n) / s.nodes[n][0])

  def min_lag(node: str) -> float:
    return min(
      (s.lag_of(h) for h, q in s.procs_on(node) if not q.spec.exclusive and s.is_busy(h)), default=float("inf")
    )

  here = (round(s.load(p.node, name), 6), round(min_lag(p.node), 3))
  opts = [
    (round(s.load(n), 6), round(min_lag(n), 3), n) for n in fit if s.lag_of(name) - min_lag(n) >= REBALANCE_MS - 1e-9
  ]
  return min((o for o in opts if o[:2] < here), default=(0, 0, None))[2]


def account_demand(s: State, cap: float) -> dict[str, float]:
  """GPUs each account could use now: one run's width per backlogged run, more for an elastic sampler."""
  biggest = max((g for g, _ in s.nodes.values()), default=0)
  want = {}
  for r, j in s.jobs.items():
    if s.has_backlog(r) and j.spec.gpus <= biggest:
      reps = min(int(cap) // j.spec.gpus, j.want_replicas) if j.spec.elastic else 1
      account = s.account(r)
      want[account] = want.get(account, 0.0) + reps * j.spec.gpus
  return want


def place_new_proc(s: State, out: list, run: str, spec: Spec, node: str, rule: str, append: bool = False) -> None:
  q = s.add_proc(spec)
  s.procs[q].node = node
  s.assignments[run] = [*s.assignments[run], q] if append else [q]
  out.append((Ev.ASSIGNED, {"run": run, "proc": q}))
  out.append((Ev.PLACED, {"proc": q, "node": node, "rule": rule}))


def drop_lost(s: State, out: list) -> None:
  """Unplace workers on lost/full nodes, and trim excess sampler replicas once backlog shrinks."""
  for name, p in sorted(s.procs.items()):
    if p.node and (p.node not in s.nodes or not s.can_host(p.node, p.spec, name)):
      p.node = None
      out.append((Ev.UNPLACED, {"proc": name, "rule": "place.node_lost"}))
  for r, ps in list(s.assignments.items()):
    s.assignments[r] = list(dict.fromkeys(ps))[: s.jobs[r].want_replicas]


def assign(s: State, out: list) -> None:
  """Assign every unassigned run to the least-loaded matching shareable worker, or create a new one."""
  for r, j in sorted(s.jobs.items()):
    if not s.assignments.get(r):
      match = [n for n, p in sorted(s.procs.items()) if j.spec.shareable and p.spec == j.spec]
      target = min(match, key=lambda n: len(s.runs(n))) if match else s.add_proc(j.spec)
      s.assignments[r] = [target]
      out.append((Ev.ASSIGNED, {"run": r, "proc": target}))


def merge_and_split(s: State, out: list) -> None:
  """Coalesce crowded LoRA trainers, and split busy ones onto spare GPUs."""
  for name, p in sorted(list(s.procs.items())):
    if p.spec.role != Role.TRAINER or not p.spec.shareable or not p.node:
      continue
    idle_dup = not s.is_busy(name) and s.used(p.node) > s.nodes[p.node][0]
    busy_dup = s.spare(p.node) < 0 and not any(s.spare(n) >= p.spec.gpus for n in s.nodes)
    twins = [q for q, pr in sorted(s.procs.items()) if q != name and pr.spec == p.spec]
    if (idle_dup or busy_dup) and twins:
      for r in s.runs(name):
        s.assignments[r] = [twins[0]]
        out.append((Ev.ASSIGNED, {"run": r, "proc": twins[0]}))
      continue
    while len(busy := [r for r in s.runs(name) if s.has_backlog(r)]) >= 2 and (spot := choose_node(s, name, True)):
      place_new_proc(s, out, busy[-1], p.spec, spot, "place.split")


def scale_out(s: State, out: list) -> None:
  """Give a backlogged sampler more replicas: idle matching workers first, then new ones on spare GPUs."""
  for r, j in sorted(s.jobs.items()):
    while len(s.assignments[r]) < j.want_replicas:
      pool = [q for q, p in sorted(s.procs.items()) if p.spec == j.spec and p.node]
      if idle := [q for q in pool if not s.is_busy(q)]:
        s.assignments[r].append(idle[0])
        out.append((Ev.ASSIGNED, {"run": r, "proc": idle[0]}))
      elif spot := choose_node(s, s.assignments[r][0], True):
        place_new_proc(s, out, r, j.spec, spot, "place.scale_out", append=True)
      else:
        break


def collect_and_place(s: State, out: list) -> None:
  """Remove empty unowned workers, then place unplaced ones."""
  held = {owner_of(j.owner)[0] for j in s.jobs.values() if owner_of(j.owner)[1] in s.nodes}
  for name in [n for n in sorted(s.procs) if not s.runs(n) and n not in held]:
    if s.procs.pop(name).node:
      out.append((Ev.UNPLACED, {"proc": name, "rule": "place.emptied"}))
  for name in sorted(n for n, p in s.procs.items() if p.node is None):
    if spot := choose_node(s, name):
      s.procs[name].node = spot
      out.append((Ev.PLACED, {"proc": name, "node": spot, "rule": "place.new"}))


def rebalance(s: State, out: list) -> None:
  """Fill spare nodes, evacuate exclusive-squeezed workers, or rotate across nodes once lag reaches `REBALANCE_MS`."""
  for name, p in sorted(s.procs.items()):
    if p.node and not p.spec.exclusive and (target := rebalance_target(s, name)):
      p.node = target
      out.append((Ev.PLACED, {"proc": name, "node": target, "rule": "place.rebalance"}))


def step(state: State) -> tuple[State, list[tuple[Ev, dict]]]:
  """Pure planner transition: update account lag, then run placement phases in order."""
  s, out = copy.deepcopy(state), []
  cap = float(sum(g for g, _ in s.nodes.values()))
  s.lag, s.billed_service = update_lag(s.lag, s.billed_service, s.service, cap, account_demand(s, cap), s.weights)
  for phase in (drop_lost, assign, merge_and_split, scale_out, collect_and_place, rebalance):
    phase(s, out)
  return s, out


# ***** loop *****


async def tick(bus: Bus) -> list[tuple[Ev, dict]]:
  for session in await bus.expired_sessions():
    await bus.end_session(session)
  await bus.unregister_checkpoints(*await bus.expired_checkpoints())
  raw = await bus.read_state()
  old, (new, changes) = raw["plan"], step(from_state(raw))
  plan = to_plan(new)
  if plan != old:
    wake = plan["procs"] != old.get("procs") or plan["assignments"] != old.get("assignments")
    await bus.write_plan(plan, changes, notify_nodes=wake)
  return changes


async def run(bus: Bus) -> None:
  wake_id = "$"
  while True:
    await tick(bus)
    wake_id = await bus.wait_wake(wake_id, TICK_MS)


if __name__ == "__main__":
  asyncio.run(run(Bus()))
