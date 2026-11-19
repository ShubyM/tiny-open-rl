"""Cluster planner: places, scales, and rebalances processes across nodes (`run -> procs -> node`).

Nodes time-slice their own GPUs locally on wall-clock quanta and enforce all safety via `on`, `run.owner`, and `lease`.
`tinyrl:plan` stores only process placement, run assignment, and per-account EEVDF `lag` (in `gpu_ms`).
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import itertools
from dataclasses import dataclass, field

from tinyrl.bus import Bus, connect
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
  record: dict = field(default_factory=dict)


def owner_of(token: str | None) -> tuple[str, str]:
  proc, _, rest = (token or "").rsplit(":", 1)[0].partition("@")
  return proc, rest.rsplit("-", 1)[0]


@dataclass
class State:
  made: int = 0
  nodes: dict[str, tuple[int, int]] = field(default_factory=dict)
  procs: dict[str, Proc] = field(default_factory=dict)
  home: dict[str, str] = field(default_factory=dict)  # training run -> single trainer proc
  jobs: dict[str, Job] = field(default_factory=dict)
  lag: dict[str, float] = field(default_factory=dict)
  seen: dict[str, float] = field(default_factory=dict)
  service: dict[str, float] = field(default_factory=dict)
  weights: dict[str, float] = field(default_factory=dict)

  def runs(self, proc: str) -> list[str]:
    if not (p := self.procs.get(proc)):
      return []
    if p.spec.role == Role.SAMPLER:
      return [r for r, j in sorted(self.jobs.items()) if j.spec == p.spec]
    return [r for r, o in sorted(self.home.items()) if o == proc]

  def procs_of(self, run: str) -> list[str]:
    if not (j := self.jobs.get(run)):
      return []
    if j.spec.role == Role.SAMPLER:
      return [n for n, p in sorted(self.procs.items()) if p.spec == j.spec]
    return [p] if (p := self.home.get(run)) in self.procs else []

  def ready(self, run: str) -> bool:
    return self.jobs[run].backlog > 0

  def busy(self, proc: str) -> bool:
    return any(self.ready(r) for r in self.runs(proc))

  def account(self, run: str) -> str:
    if (job := self.jobs.get(run)) is None:
      return run
    return job.account or (self.account(job.train_run) if job.train_run in self.jobs else run)

  def lag_of(self, proc: str) -> float:
    return max((self.ready_lag(r) for r in self.runs(proc) if self.ready(r)), default=0.0)

  def ready_lag(self, run: str) -> float:
    """The account's lag, less service completed since the planner's last update."""
    account = self.account(run)
    return self.lag.get(account, 0.0) - max(0.0, self.service.get(account, 0.0) - self.seen.get(account, 0.0))

  def procs_on(self, node: str, skip: str = "") -> list[tuple[str, Proc]]:
    return [(n, p) for n, p in self.procs.items() if p.node == node and n != skip]

  def used(self, node: str) -> int:
    return sum(p.spec.gpus for _, p in self.procs_on(node))

  def excl_used(self, node: str, skip: str = "") -> int:
    return sum(p.spec.gpus for _, p in self.procs_on(node, skip) if p.spec.exclusive)

  def busy_gpus(self, node: str, skip: str = "") -> int:
    return sum(p.spec.gpus for n, p in self.procs_on(node, skip) if not p.spec.exclusive and self.busy(n))

  def spare(self, node: str) -> int:
    return self.nodes[node][0] - self.excl_used(node) - self.busy_gpus(node)

  def load(self, node: str, skip: str = "") -> float:
    return self.busy_gpus(node, skip) / max(1, self.nodes[node][0] - self.excl_used(node, skip))

  def can_host(self, node: str, spec: Spec, skip: str = "", spare: bool = False) -> bool:
    return (self.used(node) if spare else self.excl_used(node, skip)) + spec.gpus <= self.nodes[node][0]

  def add_proc(self, spec: Spec, node: str | None = None) -> str:
    self.made = next(n for n in itertools.count(self.made + 1) if f"p{n}" not in self.procs)
    self.procs[name := f"p{self.made}"] = Proc(spec, node)
    return name


def from_state(raw: dict) -> State:
  plan, queues, rnodes = raw["plan"], raw.get("queues", {}), raw.get("nodes", {})
  nodes = {n: (int(v["gpus"]), int(v.get("memory", 0))) for n, v in rnodes.items()}
  jobs, procs = {}, {}
  for r, info in raw.get("runs", {}).items():
    rec = info["record"]
    excl, tr, role = bool(rec.get("exclusive")), rec.get("train_run"), Role(rec.get("role", "trainer"))
    spec = Spec(role, rec.get("base", ""), rec.get("rank", 0) > 0, excl, tr if excl else None, int(rec.get("gpus", 1)))
    acct = rec.get("account") or ("" if tr else rec.get("session", ""))
    jobs[r] = Job(spec, tr, acct, info.get("owner"), queues.get(r, 0), rec)
  jobs = {r: j for r, j in jobs.items() if not j.spec.train_run or j.spec.train_run in jobs}
  for n, p in plan.get("procs", {}).items():
    procs[n] = Proc(Spec(**(p["spec"] | {"role": Role(p["spec"]["role"])})), p.get("node"))
  # Earlier plans stored a trainer's single home as a one-element list.
  home = {
    r: p[0] if isinstance(p, list) else p
    for r, p in plan.get("home", {}).items()
    if p and r in jobs and jobs[r].spec.role == Role.TRAINER
  }
  fv = [{a: float(v) for a, v in plan.get(k, {}).items()} for k in ("lag", "seen", "weights")]
  return State(int(plan.get("made", 0)), nodes, procs, home, jobs, fv[0], fv[1], dict(raw.get("service", {})), fv[2])


def to_plan(s: State) -> dict:
  procs = {n: dataclasses.asdict(p) for n, p in s.procs.items()}
  return {"made": s.made, "procs": procs, "home": s.home, "lag": s.lag, "seen": s.seen, "weights": s.weights}


def settings(gpus: int, gpu_bytes: int) -> dict[str, str]:
  env = {"TINYRL_WARM_SAMPLER": "1"} if gpus >= 2 else {}
  return env | ({"TINYRL_KV_BYTES": str(24 * 2**30)} if gpu_bytes >= 64 * 2**30 else {})


# ***** placement, elastic scaling, and cross-node rebalance *****


def choose_node(s: State, proc: str, only_spare: bool = False) -> str | None:
  """A node that can host `proc`: one with room for it and its sampler, then its trainer's node, then the least used."""
  spec = s.procs[proc].spec
  near = {s.procs[p].node for r in s.runs(proc) if (p := s.home.get(s.jobs[r].train_run or ""))}
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
  if not s.busy(name) or s.spare(p.node) >= 0:
    return None
  fit = [n for n in sorted(s.nodes) if s.can_host(n, p.spec)]
  if spare := [n for n in fit if s.spare(n) >= p.spec.gpus]:
    return min(spare, key=lambda n: s.used(n) / s.nodes[n][0])

  def floor(node: str) -> float:
    on = (s.lag_of(h) for h, q in s.procs_on(node) if not q.spec.exclusive and s.busy(h))
    return min(on, default=float("inf"))

  here = (round(s.load(p.node, name), 6), round(floor(p.node), 3))
  opts = [(round(s.load(n), 6), round(floor(n), 3), n) for n in fit if s.lag_of(name) - floor(n) >= REBALANCE_MS - 1e-9]
  return min((o for o in opts if o[:2] < here), default=(0, 0, None))[2]


def demand(s: State, cap: float) -> dict[str, float]:
  """The GPUs each account could use now: one run's worth per ready run, more for a backlogged elastic sampler."""
  biggest, want = max((g for g, _ in s.nodes.values()), default=0), {}
  for r, j in s.jobs.items():
    if s.ready(r) and j.spec.gpus <= biggest:
      reps = min(int(cap) // j.spec.gpus, max(1, j.backlog // max(1, SCALE_BATCH))) if j.spec.elastic else 1
      a = s.account(r)
      want[a] = want.get(a, 0.0) + reps * j.spec.gpus
  return want


def drop_lost(s: State, out: list) -> None:
  """Unplace processes whose node vanished or no longer has room for them."""
  for name, p in sorted(s.procs.items()):
    if p.node and (p.node not in s.nodes or not s.can_host(p.node, p.spec, name)):
      p.node = None
      out.append((Ev.UNPLACED, {"proc": name, "rule": "place.node_lost"}))


def assign(s: State, out: list) -> None:
  """Give every homeless training run a process: the matching shareable one with the fewest runs, or a new one."""
  for r, j in sorted(s.jobs.items()):
    if j.spec.role == Role.TRAINER and r not in s.home:
      match = [n for n, p in sorted(s.procs.items()) if j.spec.shareable and p.spec == j.spec]
      s.home[r] = target = min(match, key=lambda n: len(s.runs(n))) if match else s.add_proc(j.spec)
      out.append((Ev.ASSIGNED, {"run": r, "proc": target}))


def merge_and_split(s: State, out: list) -> None:
  """Coalesce crowded LoRA trainers, and split busy ones onto spare GPUs."""
  for name, p in sorted(list(s.procs.items())):
    if p.spec.role != Role.TRAINER or not p.spec.shareable or not p.node:
      continue
    idle_dup = not s.busy(name) and s.used(p.node) > s.nodes[p.node][0]
    busy_dup = s.spare(p.node) < 0 and not any(s.spare(n) >= p.spec.gpus for n in s.nodes)
    twins = [q for q, pr in sorted(s.procs.items()) if q != name and pr.spec == p.spec]
    if (idle_dup or busy_dup) and twins:
      for r in s.runs(name):
        s.home[r] = twins[0]
        out.append((Ev.ASSIGNED, {"run": r, "proc": twins[0]}))
      continue
    while len(busy := [r for r in s.runs(name) if s.ready(r)]) >= 2 and (spot := choose_node(s, name, True)):
      q = s.add_proc(p.spec, spot)
      s.home[busy[-1]] = q
      out.append((Ev.ASSIGNED, {"run": busy[-1], "proc": q}))
      out.append((Ev.PLACED, {"proc": q, "node": spot, "rule": "place.split"}))


def size_samplers(s: State, out: list) -> None:
  """Maintain one stateless sampler pool per active Spec: scale out on spare GPUs, scale in past backlog."""
  by_spec: dict[Spec, list[Job]] = {}
  for _, j in sorted(s.jobs.items()):
    if j.spec.role == Role.SAMPLER:
      by_spec.setdefault(j.spec, []).append(j)
  for spec, js in by_spec.items():
    pool = [n for n, p in sorted(s.procs.items()) if p.spec == spec]
    placed = [n for n in pool if s.procs[n].node]
    want = 1 if spec.exclusive else max(1, sum(j.backlog for j in js) // max(1, SCALE_BATCH))
    keep = placed[:want] if placed else (pool[:1] or [s.add_proc(spec)])
    for name in pool:
      if name not in keep and s.procs.pop(name).node:
        out.append((Ev.UNPLACED, {"proc": name, "rule": "place.scale_in"}))
    if want > 1 and s.procs[keep[0]].node is None and (spot := choose_node(s, keep[0])):
      s.procs[keep[0]].node = spot
      out.append((Ev.PLACED, {"proc": keep[0], "node": spot, "rule": "place.new"}))
    if s.procs[keep[0]].node:
      while len(keep) < want and (spot := choose_node(s, keep[0], True)):
        keep.append(q := s.add_proc(spec, spot))
        out.append((Ev.PLACED, {"proc": q, "node": spot, "rule": "place.scale_out"}))


def collect_and_place(s: State, out: list) -> None:
  """Remove empty processes, then place the unplaced ones, exclusives first."""
  held = {owner_of(j.owner)[0] for j in s.jobs.values() if owner_of(j.owner)[1] in s.nodes}
  for name in [n for n in sorted(s.procs) if not s.runs(n) and n not in held]:
    if s.procs.pop(name).node:
      out.append((Ev.UNPLACED, {"proc": name, "rule": "place.emptied"}))
  for name in sorted(n for n, p in s.procs.items() if p.node is None):
    if spot := choose_node(s, name):
      s.procs[name].node = spot
      out.append((Ev.PLACED, {"proc": name, "node": spot, "rule": "place.new"}))


def rebalance(s: State, out: list) -> None:
  """Fill spare nodes, evacuate exclusive-squeezed processes, or rotate across nodes once lag reaches `REBALANCE_MS`."""
  for name, p in sorted(s.procs.items()):
    if p.node and not p.spec.exclusive and (target := rebalance_target(s, name)):
      p.node = target
      out.append((Ev.PLACED, {"proc": name, "node": target, "rule": "place.rebalance"}))


def step(state: State) -> tuple[State, list[tuple[Ev, dict]]]:
  """Pure planner transition: update lag, then run the placement phases in order."""
  s, out = copy.deepcopy(state), []
  cap = float(sum(g for g, _ in s.nodes.values()))
  s.lag, s.seen = update_lag(s.lag, s.seen, s.service, cap, demand(s, cap), s.weights)
  for phase in (drop_lost, assign, merge_and_split, size_samplers, collect_and_place, rebalance):
    phase(s, out)
  return s, out


# ***** loop *****


async def tick(bus: Bus) -> list[tuple[Ev, dict]]:
  for session in await bus.expired_sessions():
    await bus.end_session(session)
  await bus.expire_checkpoints()
  raw = await bus.read_state()
  old, (new, changes) = raw["plan"], step(from_state(raw))
  plan = to_plan(new)
  if plan != old:
    wake = plan["procs"] != old.get("procs") or plan["home"] != old.get("home")
    await bus.write_plan(plan, changes, notify_nodes=wake)
  return changes


async def run(bus: Bus) -> None:
  wake_id = "$"
  while True:
    await tick(bus)
    wake_id = await bus.wait_wake(wake_id, TICK_MS)


if __name__ == "__main__":
  asyncio.run(run(connect()))
