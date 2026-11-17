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

from tinyrl.bus import Bus, Cmd, Ev, Role
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry

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


def owner_of(token: str | None) -> tuple[str, str]:
  proc, _, rest = (token or "").rsplit(":", 1)[0].partition("@")
  return (proc, rest.rsplit("-", 1)[0]) if rest else ("", "")


@dataclass
class State:
  made: int = 0
  nodes: dict[str, tuple[int, int]] = field(default_factory=dict)
  preparing: set[str] = field(default_factory=set)
  procs: dict[str, Proc] = field(default_factory=dict)
  home: dict[str, list[str]] = field(default_factory=dict)
  jobs: dict[str, Job] = field(default_factory=dict)
  lag: dict[str, float] = field(default_factory=dict)
  seen: dict[str, float] = field(default_factory=dict)
  service: dict[str, float] = field(default_factory=dict)
  weights: dict[str, float] = field(default_factory=dict)

  def runs(self, proc: str) -> list[str]:
    return [r for r, ps in sorted(self.home.items()) if proc in ps and r in self.jobs]

  def ready(self, run: str) -> bool:
    return self.jobs[run].backlog > 0 and run not in self.preparing

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

  def used(self, node: str, skip: str = "") -> int:
    return sum(p.spec.gpus for n, p in self.procs.items() if n != skip and p.node == node)

  def excl_used(self, node: str, skip: str = "") -> int:
    return sum(p.spec.gpus for n, p in self.procs.items() if n != skip and p.node == node and p.spec.exclusive)

  def busy_gpus(self, node: str, skip: str = "") -> int:
    shared = (n for n, p in self.procs.items() if p.node == node and not p.spec.exclusive and n != skip)
    return sum(self.procs[n].spec.gpus for n in shared if self.busy(n))

  def spare(self, node: str, skip: str = "") -> int:
    return self.nodes[node][0] - self.excl_used(node, skip) - self.busy_gpus(node, skip)

  def load(self, node: str, skip: str = "") -> float:
    return self.busy_gpus(node, skip) / max(1, self.nodes[node][0] - self.excl_used(node, skip))

  def can_host(self, node: str, spec: Spec, skip: str = "") -> bool:
    return self.excl_used(node, skip) + spec.gpus <= self.nodes[node][0]

  def add_proc(self, spec: Spec) -> str:
    self.made = next(n for n in itertools.count(self.made + 1) if f"p{n}" not in self.procs)
    self.procs[name := f"p{self.made}"] = Proc(spec)
    return name


def from_state(raw: dict) -> State:
  plan, queues, rnodes = raw.get("plan") or {}, raw.get("queues", {}), raw.get("nodes", {})
  nodes = {n: (int(v["gpus"]), int(v.get("memory", 0))) for n, v in rnodes.items()}
  prep = {r for v in rnodes.values() for r in v.get("preparing", ())}
  jobs, procs = {}, {}
  for r, info in raw.get("runs", {}).items():
    rec = info["record"]
    excl, tr, role = bool(rec.get("exclusive")), rec.get("train_run"), Role(rec.get("role", "trainer"))
    spec = Spec(role, rec.get("base", ""), rec.get("rank", 0) > 0, excl, tr if excl else None, int(rec.get("gpus", 1)))
    acct = rec.get("account") or ("" if tr else rec.get("session", ""))
    jobs[r] = Job(spec, tr, acct, info.get("owner"), queues.get(r, 0))
  jobs = {r: j for r, j in jobs.items() if not j.spec.train_run or j.spec.train_run in jobs}
  for n, p in plan.get("procs", {}).items():
    procs[n] = Proc(Spec(**(p["spec"] | {"role": Role(p["spec"]["role"])})), p.get("node"))
  home = {r: list(ps) for r, ps in plan.get("home", {}).items() if r in jobs}
  fv = [{a: float(v) for a, v in plan.get(k, {}).items()} for k in ("lag", "seen", "weights")]
  return State(
    int(plan.get("made", 0)), nodes, prep, procs, home, jobs, fv[0], fv[1], dict(raw.get("service", {})), fv[2]
  )


def to_plan(s: State) -> dict:
  procs = {n: dataclasses.asdict(p) for n, p in s.procs.items()}
  plan = {"made": s.made, "procs": procs, "home": s.home, "lag": s.lag, "seen": s.seen}
  return plan | ({"weights": s.weights} if s.weights else {})


def settings(spec: Spec, gpus: int, gpu_bytes: int) -> dict[str, str]:
  if spec.role != Role.SAMPLER:
    return {}
  env = {"TINYRL_WARM_SAMPLER": "1"} if gpus >= 2 else {}
  return env | ({"TINYRL_KV_BYTES": str(24 * 2**30)} if gpu_bytes >= 64 * 2**30 else {})


# ***** fairness (progressive water-filling + EEVDF lag in gpu_ms) *****


def waterfill(capacity: float, demand: dict[str, float], weights: dict[str, float] | None = None) -> dict[str, float]:
  """Weighted progressive max-min fair rates `r*[a] <= demand[a]` summing to `min(capacity, sum(demand))`."""
  rates, rem, left = {}, float(capacity), {a: float(d) for a, d in demand.items() if d > 0}
  w = {a: max(1e-9, float((weights or {}).get(a, 1.0))) for a in left}
  while left and rem > 1e-9:
    tw, level = sum(w[a] for a in left), min(left[a] / w[a] for a in left)
    if level * tw >= rem - 1e-9:
      return rates | {a: rem * w[a] / tw for a in left}
    for a in [a for a in left if left[a] / w[a] <= level + 1e-9]:
      rates[a] = left.pop(a)
      rem -= rates[a]
  return rates


def update_lag(lag: dict, seen: dict, service: dict, cap: float, demand: dict, weights: dict | None = None):
  """Advance EEVDF lag from monotone `service - seen` deltas (`Lag.tla`: race-free, no idle hoard, no flicker gain)."""
  lag, seen, w = dict(lag), dict(seen), weights or {}
  accts = set(lag) | set(service) | set(demand)
  deltas = {a: max(0.0, service.get(a, 0.0) - seen.get(a, 0.0)) for a in accts}
  seen |= {a: max(seen.get(a, 0.0), v) for a, v in service.items()}
  rates = waterfill(max(cap, 1.0), {a: max(demand.get(a, 0.0), float(deltas[a] > 0)) for a in accts}, w)
  if (work := sum(deltas[a] for a in rates)) > 0 and (total_r := sum(rates.values())) > 0:
    for a, r in rates.items():
      lag[a] = lag.get(a, 0.0) + work * (r / total_r) - deltas[a]
    live_w = sum(max(1e-9, float(w.get(a, 1.0))) for a in rates)
    for a in set(lag) - set(rates):
      wa = max(1e-9, float(w.get(a, 1.0)))
      lag[a] = min(0.0, lag[a] + work * wa / (live_w + wa))
  active = {a for a in accts if demand.get(a, 0.0) > 0 or deltas[a] > 0}
  return {a: (v if a in active else min(0.0, v)) for a, v in lag.items() if a in active or v < -1e-9}, seen


def select_runs(state: State, heads: list[Cmd], service: dict[str, float], width: int) -> list[str]:
  """Choose up to `width` ready runs: accounts owed the most GPU time first, then their least-served runs."""
  if width < 1:
    raise ValueError("batch width must be positive")
  ready = {cmd.run: cmd for cmd in reversed(heads)}
  floor = min((service[run] for run in ready if run in service), default=0.0)

  def rank(cmd: Cmd) -> tuple:
    arrival = tuple(map(int, cmd.entry_id.split("-")))
    return -state.ready_lag(cmd.run), service.get(cmd.run, floor), arrival, cmd.run

  return [cmd.run for cmd in sorted(ready.values(), key=rank)[:width]]


# ***** placement, elastic scaling, and cross-node rebalance *****


def choose_node(s: State, proc: str, only_spare: bool = False) -> tuple[str, bool] | None:
  spec, skip = s.procs[proc].spec, ("" if only_spare else proc)
  nodes = [n for n in sorted(s.nodes) if s.can_host(n, spec, skip) and (not only_spare or s.spare(n) >= spec.gpus)]
  if not nodes:
    return None
  near = set()
  if spec.role == Role.SAMPLER and not spec.lora:
    ts = (t for r in s.runs(proc) if (t := s.jobs[r].train_run) in s.jobs)
    near = {s.procs[p].node for t in ts for p in s.home.get(t, ()) if p in s.procs and s.procs[p].node}
  samplers = [n for n, p in s.procs.items() if (p.node or n == proc) and p.spec.role == Role.SAMPLER]
  covered = {s.jobs[r].train_run for n in samplers for r in s.runs(n)}

  def placement_rank(node: str) -> tuple:
    unpaired = [n for n, p in s.procs.items() if p.node == node and p.spec.role == Role.TRAINER and not p.spec.lora]
    owed = sum(s.procs[n].spec.gpus for n in unpaired if not (covered & set(s.runs(n))))
    used, capacity = s.used(node, skip), s.nodes[node][0]
    free = capacity - used - owed
    paired = spec.gpus * (2 if spec.role == Role.TRAINER and not spec.lora else 1)
    return free < paired, free < spec.gpus, s.spare(node, skip) < spec.gpus, node not in near, used / capacity

  node = min(nodes, key=placement_rank)
  return node, node in near


def rebalance_target(s: State, name: str) -> str | None:
  """Move `name` to a spare node, off an exclusive-squeezed node, or across nodes once lag reaches `REBALANCE_MS`."""
  p = s.procs[name]
  squeezed = s.excl_used(p.node) + p.spec.gpus > s.nodes[p.node][0]
  if not squeezed and (not s.busy(name) or s.spare(p.node) >= 0):
    return None
  fit = [n for n in sorted(s.nodes) if n != p.node and s.can_host(n, p.spec)]
  if spare := [n for n in fit if s.spare(n) >= p.spec.gpus]:
    return min(spare, key=lambda n: s.used(n) / s.nodes[n][0])
  if squeezed:
    return min(fit, key=s.load, default=None)

  def floor(node: str, skip: str = "") -> float:
    on = (
      s.lag_of(h) for h, q in s.procs.items() if h != skip and q.node == node and not q.spec.exclusive and s.busy(h)
    )
    return min(on, default=float("inf"))

  here = (round(s.load(p.node, name), 6), round(floor(p.node, name), 3))
  opts = [(round(s.load(n), 6), round(floor(n), 3), n) for n in fit if s.lag_of(name) - floor(n) >= REBALANCE_MS - 1e-9]
  return min((o for o in opts if o[:2] < here), default=(0, 0, None))[2]


def step(state: State) -> tuple[State, list[tuple[Ev, dict]]]:
  """Pure planner transition: update lag, assign/merge/split/scale runs, place processes, and rebalance across nodes."""
  s, out = copy.deepcopy(state), []
  cap, biggest = float(sum(g for g, _ in s.nodes.values())), max((g for g, _ in s.nodes.values()), default=0)
  demand: dict[str, float] = {}
  for r, j in s.jobs.items():
    if s.ready(r) and j.spec.gpus <= biggest:
      elastic = j.spec.role == Role.SAMPLER and not j.spec.exclusive and cap > j.spec.gpus
      reps = min(int(cap) // j.spec.gpus, max(1, j.backlog // max(1, SCALE_BATCH))) if elastic else 1
      a = s.account(r)
      demand[a] = demand.get(a, 0.0) + reps * j.spec.gpus
  s.lag, s.seen = update_lag(s.lag, s.seen, s.service, cap, demand, s.weights)

  def spawn(run: str, spec: Spec, node: str, rule: str, append: bool = False) -> None:
    q = s.add_proc(spec)
    s.procs[q].node = node
    s.home[run] = [*s.home[run], q] if append else [q]
    out.append((Ev.ASSIGNED, {"run": run, "proc": q}))
    out.append((Ev.PLACED, {"proc": q, "node": node, "rule": rule}))

  # 1. Unplace vanished/shrunk nodes and drop extra sampler replicas from oversubscribed nodes.
  for name, p in sorted(s.procs.items()):
    over = p.node in s.nodes and (
      p.spec.gpus > s.nodes[p.node][0] or (p.spec.exclusive and not s.can_host(p.node, p.spec, name))
    )
    if p.node and (p.node not in s.nodes or over):
      p.node = None
      out.append((Ev.UNPLACED, {"proc": name, "rule": "place.node_lost"}))
  for r, ps in list(s.home.items()):
    ps = [p for p in dict.fromkeys(ps) if p in s.procs and s.procs[p].spec == s.jobs[r].spec]
    want = max(1, s.jobs[r].backlog // max(1, SCALE_BATCH)) if s.jobs[r].spec.role == Role.SAMPLER and s.ready(r) else 1
    s.home[r] = [ps[0], *(p for p in ps[1:] if (nd := s.procs[p].node) and s.spare(nd) >= 0)][:want] if ps else []

  # 2. Assign homeless runs; coalesce crowded LoRA trainers, split busy ones, and scale out samplers onto spare GPUs.
  for r, j in sorted(s.jobs.items()):
    if not s.home.get(r):
      match = [n for n, p in sorted(s.procs.items()) if j.spec.shareable and p.spec == j.spec]
      if match and s.ready(r) and any(s.can_host(n, j.spec) and s.spare(n) >= j.spec.gpus for n in s.nodes):
        match = [n for n in match if not s.busy(n) and (not s.procs[n].node or s.can_host(s.procs[n].node, j.spec))]
      s.home[r] = [target := min(match, key=lambda n: (len(s.runs(n)), n)) if match else s.add_proc(j.spec)]
      out.append((Ev.ASSIGNED, {"run": r, "proc": target}))

  for name, p in sorted(list(s.procs.items())):
    if p.spec.role != Role.TRAINER or not p.spec.shareable or not p.node:
      continue
    squeezed = not s.can_host(p.node, p.spec, name)
    idle_dup = not s.busy(name) and s.used(p.node) > s.nodes[p.node][0]
    busy_dup = (squeezed or s.spare(p.node) < 0) and not any(
      s.can_host(n, p.spec) and s.spare(n) >= p.spec.gpus for n in s.nodes
    )
    if idle_dup or busy_dup:
      twins = [q for q, pr in sorted(s.procs.items()) if q != name and pr.spec == p.spec and pr.node and s.runs(q)]
      if twin := next((q for q in twins if idle_dup or squeezed or s.procs[q].node == p.node), None):
        for r in s.runs(name):
          s.home[r] = [twin]
          out.append((Ev.ASSIGNED, {"run": r, "proc": twin}))
        continue
    while len(busy := [r for r in s.runs(name) if s.ready(r)]) >= 2 and (spot := choose_node(s, name, True)):
      move = max(busy[1:], key=lambda r: (s.lag.get(s.account(r), 0.0), r))
      if owner_of(s.jobs[move].owner)[0] not in ("", name):
        break
      spawn(move, p.spec, spot[0], "place.split")

  for r, j in sorted(s.jobs.items()):
    elastic = j.spec.role == Role.SAMPLER and not j.spec.exclusive and s.ready(r)
    want = j.backlog // max(1, SCALE_BATCH) if elastic else 1
    while len(s.home[r]) < want:
      pool = [q for q, p in sorted(s.procs.items()) if q not in s.home[r] and p.spec == j.spec and p.node]
      if idle := [q for q in pool if not s.busy(q) and s.spare(s.procs[q].node) >= j.spec.gpus]:
        s.home[r].append(idle[0])
        out.append((Ev.ASSIGNED, {"run": r, "proc": idle[0]}))
      elif spot := choose_node(s, s.home[r][0], True):
        spawn(r, j.spec, spot[0], "place.scale_out", append=True)
      else:
        break

  # 3. Garbage-collect empty processes and place unplaced processes (exclusives first).
  held = {owner_of(j.owner)[0] for j in s.jobs.values() if owner_of(j.owner)[1] in s.nodes}
  for name in [n for n in sorted(s.procs) if not s.runs(n) and n not in held]:
    if s.procs.pop(name).node:
      out.append((Ev.UNPLACED, {"proc": name, "rule": "place.emptied"}))
  unplaced = sorted((n for n, p in s.procs.items() if p.node is None), key=lambda n: (not s.procs[n].spec.exclusive, n))
  for name in unplaced:
    if spot := choose_node(s, name):
      s.procs[name].node = spot[0]
      rule = "place.near_its_trainer" if spot[1] else "place.least_loaded_node"
      out.append((Ev.PLACED, {"proc": name, "node": spot[0], "rule": rule}))

  # 4. Cross-node rebalance: fill spare nodes, evacuate exclusive-squeezed procs, or rotate on REBALANCE_MS.
  for name in sorted(s.procs, key=lambda n: (-s.lag_of(n), n)):
    p = s.procs[name]
    if p.node and not p.spec.exclusive and s.runs(name) and (target := rebalance_target(s, name)):
      p.node = target
      out.append((Ev.PLACED, {"proc": name, "node": target, "rule": "place.rebalance"}))
  return s, out


# ***** loop *****


async def tick(bus: Bus, registry: Registry | None = None) -> list[tuple[Ev, dict]]:
  for session in await bus.expired_sessions():
    await bus.end_session(session)
  if registry:
    await registry.unregister_checkpoints(*await registry.expired_checkpoints())
  raw = await bus.read_state()
  old, (new, changes) = raw.get("plan") or {}, step(from_state(raw))
  plan = to_plan(new)
  if changes or plan != old:
    wake = bool(changes) or plan["procs"] != old.get("procs") or plan["home"] != old.get("home")
    await bus.write_plan(plan, changes, notify_nodes=wake)
  for r, j in sorted(new.jobs.items()):
    op, on_node = owner_of(j.owner)
    if j.owner and j.spec.role == Role.TRAINER and ([op] != new.home.get(r) or on_node not in new.nodes):
      await bus.reap_run(r, j.owner)
  return changes


async def run(bus: Bus) -> None:
  registry, wake_id = Registry(bus), "$"
  while True:
    await tick(bus, registry)
    wake_id = await bus.wait_wake(wake_id, TICK_MS)


if __name__ == "__main__":
  asyncio.run(run(Bus()))
