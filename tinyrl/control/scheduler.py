"""Cluster planner: EEVDF lag + switching-cost hysteresis over a paged buffer pool of GPUs.

`tinyrl:plan` holds `procs` (each process's node and active GPU slot), `home` (each run's host process(es)),
and per-account `(lag, seen)`. Nodes enforce all safety locally via `on`, `run.owner`, and `lease`; the plan is a hint.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import itertools
from dataclasses import dataclass, field

from tinyrl.bus import Bus, Cmd, Ev, Op, Role
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry

TURN, SCALE_BATCH, TICK_MS = getenv("TURN", 200_000), getenv("SCALE_BATCH", 4), getenv("TICK_MS", 100)


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
  gpus: tuple[int, ...] = ()
  cost: float = 0.0
  at: float = 0.0


@dataclass
class Job:
  spec: Spec
  train_run: str | None = None
  account: str = ""
  owner: str | None = None
  tokens: int = 0
  backlog: int = 0


def blocks(total: int, width: int) -> list[tuple[int, ...]]:
  step = width if width > 0 and total % width == 0 else 1
  return [tuple(range(i, i + width)) for i in range(0, total - width + 1, step)] if 1 <= width <= total else []


def switch_cost(proc: Proc, node: str, block: tuple[int, ...]) -> float:
  """Amortized token threshold `c / eps` to activate `proc` on `(node, block)`."""
  s = proc.spec
  if proc.node == node and proc.gpus == block and block:
    return 0.0
  hop = s.role == Role.TRAINER and s.lora and s.gpus == 1
  if proc.node is None or (proc.node == node and (not proc.gpus or hop)):
    return float(4 * TURN if s.role == Role.SAMPLER else TURN)
  if proc.node == node:
    return float(6 * TURN if s.role == Role.SAMPLER else 2 * TURN)
  return float(6 * TURN if s.role == Role.SAMPLER else (4 * TURN if s.lora else 8 * TURN))


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

  def tokens(self, proc: str) -> float:
    return float(sum(self.jobs[r].tokens for r in self.runs(proc)))

  def account(self, run: str) -> str:
    j = self.jobs[run]
    return j.account or (self.account(j.train_run) if j.train_run in self.jobs else run)

  def lag_of(self, proc: str) -> float:
    return max((self.lag.get(self.account(r), 0.0) for r in self.runs(proc) if self.ready(r)), default=0.0)

  def free(self, node: str, width: int) -> list[tuple[int, ...]]:
    taken = {g for p in self.procs.values() if p.node == node for g in p.gpus}
    return [b for b in blocks(self.nodes[node][0], width) if not (taken & set(b))]

  def used(self, node: str, skip: str = "") -> int:
    return sum(p.spec.gpus for n, p in self.procs.items() if n != skip and p.node == node)

  def add_proc(self, spec: Spec) -> str:
    self.made = next(n for n in itertools.count(self.made + 1) if f"p{n}" not in self.procs)
    self.procs[name := f"p{self.made}"] = Proc(spec)
    return name

  def activate(self, proc: str, node: str, block: tuple[int, ...], cost: float | None = None) -> None:
    p = self.procs[proc]
    p.cost = switch_cost(p, node, block) if cost is None else cost
    p.node, p.gpus, p.at = node, block, self.tokens(proc)


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
    jobs[r] = Job(spec, tr, acct, info.get("owner"), info.get("tokens", 0), queues.get(r, 0))
  jobs = {r: j for r, j in jobs.items() if not j.spec.train_run or j.spec.train_run in jobs}
  for n, p in plan.get("procs", {}).items():
    spec = Spec(**(p["spec"] | {"role": Role(p["spec"]["role"])}))
    procs[n] = Proc(spec, p.get("node"), tuple(p.get("gpus", ())), float(p.get("cost", 0)), float(p.get("at", 0)))
  home = {r: list(ps) for r, ps in plan.get("home", {}).items() if r in jobs}
  fv = [{a: float(v) for a, v in plan.get(k, {}).items()} for k in ("lag", "seen", "weights")]
  return State(
    int(plan.get("made", 0)), nodes, prep, procs, home, jobs, fv[0], fv[1], dict(raw.get("service", {})), fv[2]
  )


def to_plan(s: State) -> dict:
  procs = {
    n: {"spec": dataclasses.asdict(p.spec), "node": p.node, "gpus": list(p.gpus), "cost": p.cost, "at": p.at}
    for n, p in s.procs.items()
  }
  plan = {"made": s.made, "procs": procs, "home": s.home, "lag": s.lag, "seen": s.seen}
  return plan | ({"weights": s.weights} if s.weights else {})


def settings(spec: Spec, gpus: int, gpu_bytes: int) -> dict[str, str]:
  if spec.role != Role.SAMPLER:
    return {}
  env = {"TINYRL_WARM_SAMPLER": "1"} if gpus >= 2 else {}
  return env | ({"TINYRL_KV_BYTES": str(24 * 2**30)} if gpu_bytes >= 64 * 2**30 else {})


# ***** progressive water-filling, EEVDF lag, and intra-process selection *****


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


def update_lag(
  lag: dict, seen: dict, service: dict, capacity: float, demand: dict, weights: dict | None = None
) -> tuple[dict, dict]:
  """Advance EEVDF lag from monotone `service - seen` deltas (`Lag.tla`: race-free, no idle hoard, no flicker gain)."""
  lag, seen, w = dict(lag), dict(seen), weights or {}
  accts = set(lag) | set(service) | set(demand)
  deltas = {a: max(0.0, service.get(a, 0.0) - seen.get(a, 0.0)) for a in accts}
  seen |= {a: max(seen.get(a, 0.0), v) for a, v in service.items()}
  rates = waterfill(max(capacity, 1.0), {a: max(demand.get(a, 0.0), float(deltas[a] > 0)) for a in accts}, w)
  if (work := sum(deltas[a] for a in rates)) > 0 and (total_r := sum(rates.values())) > 0:
    for a, r in rates.items():
      lag[a] = lag.get(a, 0.0) + work * (r / total_r) - deltas[a]
    live_w = sum(max(1e-9, float(w.get(a, 1.0))) for a in rates)
    for a in set(lag) - set(rates):
      wa = max(1e-9, float(w.get(a, 1.0)))
      lag[a] = min(0.0, lag[a] + work * wa / (live_w + wa))
  active = {a for a, d in demand.items() if d > 0}
  return {a: (v if a in active else min(0.0, v)) for a, v in lag.items() if a in active or v < -1e-9}, seen


def select_runs(heads: list[Cmd], service: dict[str, float], width: int, prefer: str | None = None) -> list[str]:
  if width < 1:
    raise ValueError("batch width must be positive")
  ready = dict(reversed([(c.run, c) for c in heads]))
  floor = min((service[r] for r in ready if r in service), default=0.0)

  def rank(c: Cmd) -> tuple:
    prio = (c.run != prefer, c.op != Op.SAVE_WEIGHTS_FOR_SAMPLER, service.get(c.run, floor))
    return (*prio, *map(int, c.entry_id.split("-")), c.run)

  return [c.run for c in sorted(ready.values(), key=rank)[:width]]


# ***** placement, scaling, and EEVDF GPU arbitration *****


def choose_node(s: State, proc: str, only_free: bool = False) -> tuple[str, tuple[int, ...], bool] | None:
  spec = s.procs[proc].spec
  nodes = sorted(n for n, (g, _) in s.nodes.items() if g >= spec.gpus and (not only_free or s.free(n, spec.gpus)))
  if not nodes:
    return None
  near = set()
  if spec.role == Role.SAMPLER and not spec.lora:
    ts = (t for r in s.runs(proc) if (t := s.jobs[r].train_run) in s.jobs)
    near = {s.procs[p].node for t in ts for p in s.home.get(t, ()) if p in s.procs and s.procs[p].node}
  covered = {
    s.jobs[r].train_run
    for n, p in s.procs.items()
    if (p.node or n == proc) and p.spec.role == Role.SAMPLER
    for r in s.runs(n)
  }

  def room(node: str, need: int) -> bool:
    unpaired = [p for p in s.procs.values() if p.node == node and p.spec.role == Role.TRAINER and not p.spec.lora]
    owed = sum(p.spec.gpus for n, p in s.procs.items() if p in unpaired and not (covered & set(s.runs(n))))
    return bool(s.free(node, spec.gpus)) and s.used(node, proc) + owed + need <= s.nodes[node][0]

  pair = 2 * spec.gpus if spec.role == Role.TRAINER and not spec.lora else spec.gpus
  for pred in (lambda n: room(n, pair), lambda n: room(n, spec.gpus), lambda n: n in near):
    nodes = [n for n in nodes if pred(n)] or nodes
  node = min(nodes, key=lambda n: s.used(n, proc) / s.nodes[n][0])
  fb = s.free(node, spec.gpus)
  return node, (fb[0] if fb else ()), node in near


def slot_choice(s: State, name: str, node: str, b: tuple[int, ...], g_now: dict, rates: dict) -> tuple | None:
  """Score placing parked `name` onto `(node, b)`, or `None` if safety/hysteresis rejects the switch."""
  p, my_lag = s.procs[name], s.lag_of(name)
  holders = [h for h, pr in s.procs.items() if pr.node == node and set(pr.gpus) & set(b)]
  if any(s.procs[h].spec.exclusive for h in holders):
    return None
  cost, busy = switch_cost(p, node, b), [h for h in holders if s.busy(h)]
  if not p.spec.exclusive and holders and not busy and node != p.node and my_lag < TURN - 1e-9:
    if (s.used(p.node) - p.spec.gpus) / s.nodes[p.node][0] <= s.used(node) / s.nodes[node][0] + 1e-9:
      return None
  if not p.spec.exclusive and busy:
    mine = {s.account(r) for r in s.runs(name) if s.ready(r)}
    under = any(g_now.get(a, 0.0) < rates.get(a, 0.0) - 1e-9 for a in mine if a in rates)
    for h in busy:
      ran, ha = s.tokens(h) - s.procs[h].at, {s.account(r) for r in s.runs(h) if s.ready(r)}
      if ha == mine and node == p.node:
        ok = ran >= max(s.procs[h].cost, cost) - 1e-9
      else:
        over = any(g_now.get(a, 0.0) > rates.get(a, 0.0) + 1e-9 for a in ha)
        ok = under and over and ran >= s.procs[h].cost - 1e-9 and my_lag - s.lag_of(h) >= cost - 1e-9
      if not ok:
        return None
  hlag = max((s.lag_of(h) for h in busy), default=-1e30)
  load = s.used(node, name) / s.nodes[node][0]
  return (bool(busy), bool(holders), cost, len(holders), hlag, node != p.node, load, node, b, holders)


def step(state: State) -> tuple[State, list[tuple[Ev, dict]]]:
  """Pure planner transition: assign/merge/split/scale runs, place processes, and arbitrate GPUs."""
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

  def spawn(run: str, spec: Spec, spot: tuple[str, tuple[int, ...], bool], rule: str, append: bool = False) -> None:
    q = s.add_proc(spec)
    s.home[run] = [*s.home[run], q] if append else [q]
    s.activate(q, spot[0], spot[1])
    out.append((Ev.ASSIGNED, {"run": run, "proc": q}))
    out.append((Ev.PLACED, {"proc": q, "node": spot[0], "gpus": list(spot[1]), "rule": rule}))

  # 1. Unplace vanished or undersized nodes and scale in idle or preempted extra sampler replicas.
  for name, p in sorted(s.procs.items()):
    if p.node and (p.node not in s.nodes or p.spec.gpus > s.nodes[p.node][0]):
      p.node, p.gpus, p.cost, p.at = None, (), 0.0, 0.0
      out.append((Ev.UNPLACED, {"proc": name, "rule": "place.node_lost"}))
    elif p.node and any(g >= s.nodes[p.node][0] for g in p.gpus):
      p.gpus, p.cost, p.at = (), 0.0, 0.0
  for r, ps in list(s.home.items()):
    ps = [p for p in dict.fromkeys(ps) if p in s.procs and s.procs[p].spec == s.jobs[r].spec]
    want = max(1, s.jobs[r].backlog // max(1, SCALE_BATCH)) if s.jobs[r].spec.role == Role.SAMPLER and s.ready(r) else 1
    s.home[r] = [ps[0], *(p for p in ps[1:] if s.procs[p].gpus)][:want] if ps else []

  # 2. Assign homeless runs; dynamically merge, split, and scale out across free GPUs.
  for r, j in sorted(s.jobs.items()):
    if not s.home.get(r):
      match = [n for n, p in sorted(s.procs.items()) if j.spec.shareable and p.spec == j.spec]
      if match and s.ready(r) and any(s.free(n, j.spec.gpus) for n in s.nodes):
        match = [n for n in match if not s.busy(n)]
      s.home[r] = [target := min(match, key=lambda n: (len(s.runs(n)), n)) if match else s.add_proc(j.spec)]
      out.append((Ev.ASSIGNED, {"run": r, "proc": target}))

  for name, p in sorted(list(s.procs.items())):
    if p.spec.role != Role.TRAINER or not p.spec.shareable or not p.node:
      continue
    if not p.gpus and not any(s.free(n, p.spec.gpus) for n in s.nodes):
      if twin := next((q for q, pr in sorted(s.procs.items()) if q != name and pr.spec == p.spec and pr.gpus), None):
        for r in s.runs(name):
          s.home[r] = [twin]
          out.append((Ev.ASSIGNED, {"run": r, "proc": twin}))
      continue
    while len(busy := [r for r in s.runs(name) if s.ready(r)]) >= 2 and (spot := choose_node(s, name, True)):
      move = max(busy[1:], key=lambda r: (s.lag.get(s.account(r), 0.0), r))
      if owner_of(s.jobs[move].owner)[0] not in ("", name):
        break
      spawn(move, p.spec, spot, "place.split")

  for r, j in sorted(s.jobs.items()):
    cap_reps = (
      j.backlog // max(1, SCALE_BATCH) if j.spec.role == Role.SAMPLER and not j.spec.exclusive and s.ready(r) else 1
    )
    while len(s.home[r]) < cap_reps:
      if idle := [
        q for q, p in sorted(s.procs.items()) if q not in s.home[r] and p.spec == j.spec and p.gpus and not s.busy(q)
      ]:
        s.home[r].append(idle[0])
        out.append((Ev.ASSIGNED, {"run": r, "proc": idle[0]}))
      elif spot := choose_node(s, s.home[r][0], True):
        spawn(r, j.spec, spot, "place.scale_out", append=True)
      else:
        break

  # 3. Garbage-collect empty processes and place unplaced processes.
  held = {owner_of(j.owner)[0] for j in s.jobs.values() if owner_of(j.owner)[1] in s.nodes}
  for name in [n for n in sorted(s.procs) if not s.runs(n) and n not in held]:
    if s.procs.pop(name).node:
      out.append((Ev.UNPLACED, {"proc": name, "rule": "place.emptied"}))
  unplaced = sorted((n for n, p in s.procs.items() if p.node is None), key=lambda n: (not s.procs[n].spec.exclusive, n))
  for name in unplaced:
    if chosen := choose_node(s, name):
      node, block, near = chosen
      s.activate(name, node, block) if block else setattr(s.procs[name], "node", node)
      rule = "place.near_its_trainer" if near else "place.least_loaded_node"
      out.append((Ev.PLACED, {"proc": name, "node": node, "gpus": list(block), "rule": rule}))

  # 4. Spread, rebalance, and EEVDF-preempt across local (L1) and remote (L2) GPU blocks.
  rates, defer = waterfill(cap, demand, s.weights), set()
  order = sorted(s.procs, key=lambda n: (not s.procs[n].spec.exclusive, not s.busy(n), -s.lag_of(n), s.tokens(n), n))
  for name in order:
    p = s.procs[name]
    if not p.node or p.gpus or not s.runs(name):
      continue
    if not s.busy(name) and not p.spec.exclusive:
      if local := s.free(p.node, p.spec.gpus):
        s.activate(name, p.node, local[0])
      continue
    g_now: dict[str, float] = {}
    for q, pr in s.procs.items():
      if pr.node and pr.gpus and (accts := {s.account(r) for r in s.runs(q) if s.ready(r)}):
        for a in accts:
          g_now[a] = g_now.get(a, 0.0) + len(pr.gpus) / len(accts)
    mine = {s.account(r) for r in s.runs(name) if s.ready(r)}
    slots = [
      (n, b)
      for n, (g, _) in sorted(s.nodes.items())
      if g >= p.spec.gpus and (n == p.node or s.busy(name) or p.spec.exclusive)
      for b in blocks(g, p.spec.gpus)
    ]
    choices = [
      c for n, b in slots if (c := slot_choice(s, name, n, b, g_now, rates)) and not (c[0] and mine and mine <= defer)
    ]
    if not choices:
      defer |= mine
      continue
    _, _, cost, _, _, moved, _, node, b, holders = min(choices)
    for h in holders:
      s.procs[h].gpus, s.procs[h].cost, s.procs[h].at = (), 0.0, 0.0
    s.activate(name, node, b, cost)
    if moved:
      out.append((Ev.PLACED, {"proc": name, "node": node, "gpus": list(b), "rule": "place.rebalance"}))
    wait = [n for n, pr in s.procs.items() if pr.node == node and not pr.gpus and not pr.spec.exclusive and s.busy(n)]
    for h in [*sorted(wait, key=lambda n: (-s.lag_of(n), n)), *holders]:
      if not s.procs[h].gpus and (spare := s.free(node, s.procs[h].spec.gpus)):
        s.activate(h, node, spare[0])
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
    old_slots = {n: (p.get("node"), p.get("gpus")) for n, p in old.get("procs", {}).items()}
    new_slots = {n: (p.get("node"), p.get("gpus")) for n, p in plan["procs"].items()}
    await bus.write_plan(
      plan, changes, notify_nodes=bool(changes) or new_slots != old_slots or plan["home"] != old.get("home")
    )
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
