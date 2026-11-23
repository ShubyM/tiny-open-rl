"""Pure cluster placement and EEVDF lag update (`run -> proc -> node`). Zero I/O, zero locks."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

import msgspec

from tinyrl.control.fairness import update_lag
from tinyrl.helpers import getenv
from tinyrl.types import MAX_LORAS, MAX_SEQS, Ev, Owner, Role
from tinyrl.types import WorkerSpec as Spec

REBALANCE_MS = getenv("REBALANCE_MS", 20_000)
PLAN_FIELDS = ("made", "procs", "home", "lag", "seen", "weights")


@dataclass
class Proc:
  spec: Spec = Spec()
  node: str | None = None


@dataclass
class Job:
  spec: Spec
  train_run: str | None = None
  account: str = ""
  owner: Owner | None = None  # process holding this run's live weights
  backlog: int = 0


@dataclass
class State:
  made: int = 0
  nodes: dict[str, int] = field(default_factory=dict)
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

  def busy(self, proc: str) -> bool:
    return any(self.jobs[r].backlog > 0 for r in self.runs(proc))

  def account(self, run: str) -> str:
    if (job := self.jobs.get(run)) is None:
      return run
    return job.account or (self.account(job.train_run) if job.train_run in self.jobs else run)

  def lag_of(self, proc: str) -> float:
    return max((self.ready_lag(r) for r in self.runs(proc) if self.jobs[r].backlog > 0), default=0.0)

  def ready_lag(self, run: str) -> float:
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
    return self.nodes[node] - self.excl_used(node) - self.busy_gpus(node)

  def load(self, node: str, skip: str = "") -> float:
    return self.busy_gpus(node, skip) / max(1, self.nodes[node] - self.excl_used(node, skip))

  def can_host(self, node: str, spec: Spec, skip: str = "", spare: bool = False) -> bool:
    return (self.used(node) if spare else self.excl_used(node, skip)) + spec.gpus <= self.nodes[node]

  def add_proc(self, spec: Spec, node: str | None = None) -> str:
    self.made = next(n for n in itertools.count(self.made + 1) if f"p{n}" not in self.procs)
    self.procs[name := f"p{self.made}"] = Proc(spec, node)
    return name


def from_plan(plan: dict) -> State:
  return msgspec.convert({k: plan[k] for k in PLAN_FIELDS if k in plan}, State)


def to_plan(s: State) -> dict:
  return msgspec.to_builtins({k: getattr(s, k) for k in PLAN_FIELDS})


def choose_node(s: State, proc: str, only_spare: bool = False) -> str | None:
  """Pick a node for `proc`: room for a full-FT sampler pair first, then its trainer's node, then least used."""
  spec = s.procs[proc].spec
  preferred = {s.procs[p].node for r in s.runs(proc) if (p := s.home.get(s.jobs[r].train_run or ""))}
  sampled = {j.train_run for j in s.jobs.values()}

  def rank(node: str) -> tuple:
    unpaired = [n for n, p in s.procs_on(node) if p.spec.role == Role.TRAINER and not p.spec.lora]
    reserved = sum(s.procs[n].spec.gpus for n in unpaired if not (sampled & set(s.runs(n))))
    used, capacity = s.used(node), s.nodes[node]
    paired = spec.gpus * (2 if spec.role == Role.TRAINER and not spec.lora else 1)
    return capacity - used - reserved < paired, node not in preferred, used / capacity

  return min((n for n in sorted(s.nodes) if s.can_host(n, spec, spare=only_spare)), key=rank, default=None)


def rebalance_target(s: State, name: str) -> str | None:
  """Move busy `name` off an oversubscribed node: to a spare node, or across nodes once lag hits `REBALANCE_MS`."""
  p = s.procs[name]
  if not s.busy(name) or s.spare(p.node) >= 0:
    return None
  fit = [n for n in sorted(s.nodes) if s.can_host(n, p.spec)]
  if spare := [n for n in fit if s.spare(n) >= p.spec.gpus]:
    return min(spare, key=lambda n: s.used(n) / s.nodes[n])

  def floor(node: str) -> float:
    return min((s.lag_of(h) for h, q in s.procs_on(node) if not q.spec.exclusive and s.busy(h)), default=float("inf"))

  here = (round(s.load(p.node, name), 6), round(floor(p.node), 3))
  opts = [(round(s.load(n), 6), round(floor(n), 3), n) for n in fit if s.lag_of(name) - floor(n) >= REBALANCE_MS - 1e-9]
  return min((o for o in opts if o[:2] < here), default=(0, 0, None))[2]


def demand(s: State, cap: float) -> dict[str, float]:
  biggest, want = max(s.nodes.values(), default=0), {}
  for r, j in s.jobs.items():
    if j.backlog > 0 and j.spec.gpus <= biggest:
      reps = min(int(cap) // j.spec.gpus, max(1, (j.backlog + MAX_SEQS - 1) // MAX_SEQS)) if j.spec.elastic else 1
      a = s.account(r)
      want[a] = want.get(a, 0.0) + reps * j.spec.gpus
  return want


def drop_lost(s: State, out: list) -> None:
  s.home = {r: p for r, p in s.home.items() if r in s.jobs}
  for name, p in sorted(s.procs.items()):
    if p.node and (p.node not in s.nodes or not s.can_host(p.node, p.spec, name)):
      p.node = None
      out.append((Ev.UNPLACED, {"proc": name, "rule": "place.node_lost"}))


def assign(s: State, out: list) -> None:
  for r, j in sorted(s.jobs.items()):
    if j.spec.role == Role.TRAINER and r not in s.home:
      match = [n for n, p in sorted(s.procs.items()) if j.spec.shareable and p.spec == j.spec]
      s.home[r] = target = min(match, key=lambda n: len(s.runs(n))) if match else s.add_proc(j.spec)
      out.append((Ev.ASSIGNED, {"run": r, "proc": target}))


def merge_and_split(s: State, out: list) -> None:
  for name, p in sorted(list(s.procs.items())):
    if p.spec.role != Role.TRAINER or not p.spec.shareable or not p.node:
      continue
    idle_dup = not s.busy(name) and s.used(p.node) > s.nodes[p.node]
    busy_dup = s.spare(p.node) < 0 and not any(s.spare(n) >= p.spec.gpus for n in s.nodes)
    others = [q for q, pr in sorted(s.procs.items()) if q != name and pr.spec == p.spec]
    if (idle_dup or busy_dup) and others:
      for r in s.runs(name):
        s.home[r] = others[0]
        out.append((Ev.ASSIGNED, {"run": r, "proc": others[0]}))
      continue
    while len(busy := [r for r in s.runs(name) if s.jobs[r].backlog > 0]) >= 2 and (node := choose_node(s, name, True)):
      s.home[busy[-1]] = q = s.add_proc(p.spec, node)
      out.append((Ev.ASSIGNED, {"run": busy[-1], "proc": q}))
      out.append((Ev.PLACED, {"proc": q, "node": node, "rule": "place.split"}))


def size_samplers(s: State, out: list) -> None:
  by_spec: dict[Spec, list[Job]] = {}
  for _, j in sorted(s.jobs.items()):
    if j.spec.role == Role.SAMPLER:
      by_spec.setdefault(j.spec, []).append(j)
  for spec, js in by_spec.items():
    pool = [n for n, p in sorted(s.procs.items()) if p.spec == spec]
    placed, ready = [n for n in pool if s.procs[n].node], [j for j in js if j.backlog > 0]
    by_seqs = (sum(j.backlog for j in ready) + MAX_SEQS - 1) // MAX_SEQS
    by_slots = (len(ready) + (cap := MAX_LORAS if spec.lora else 1) - 1) // cap
    want = 1 if spec.exclusive else max(1, by_seqs, by_slots)
    keep = placed[:want] if placed else (pool[:1] or [s.add_proc(spec)])
    for name in pool:
      if name not in keep and s.procs.pop(name).node:
        out.append((Ev.UNPLACED, {"proc": name, "rule": "place.scale_in"}))
    if want > 1 and s.procs[keep[0]].node is None and (node := choose_node(s, keep[0])):
      s.procs[keep[0]].node = node
      out.append((Ev.PLACED, {"proc": keep[0], "node": node, "rule": "place.new"}))
    if s.procs[keep[0]].node:
      while len(keep) < want and (node := choose_node(s, keep[0], True)):
        keep.append(q := s.add_proc(spec, node))
        out.append((Ev.PLACED, {"proc": q, "node": node, "rule": "place.scale_out"}))


def collect_and_place(s: State, out: list) -> None:
  held = {j.owner[0] for j in s.jobs.values() if j.owner and j.owner[1] in s.nodes}
  for name in [n for n in sorted(s.procs) if not s.runs(n) and n not in held]:
    if s.procs.pop(name).node:
      out.append((Ev.UNPLACED, {"proc": name, "rule": "place.emptied"}))
  for name in sorted(n for n, p in s.procs.items() if p.node is None):
    if node := choose_node(s, name):
      s.procs[name].node = node
      out.append((Ev.PLACED, {"proc": name, "node": node, "rule": "place.new"}))


def rebalance(s: State, out: list) -> None:
  for name, p in sorted(s.procs.items()):
    if p.node and not p.spec.exclusive and (target := rebalance_target(s, name)):
      p.node = target
      out.append((Ev.PLACED, {"proc": name, "node": target, "rule": "place.rebalance"}))


def step(st: State) -> tuple[State, list[tuple[Ev, dict]]]:
  copied = {k: dict(getattr(st, k)) for k in ("nodes", "home", "jobs", "lag", "seen", "weights")}
  s = replace(st, procs={k: Proc(p.spec, p.node) for k, p in st.procs.items()}, **copied)
  cap, out = float(sum(s.nodes.values())), []
  s.lag, s.seen = update_lag(s.lag, s.seen, s.service, cap, demand(s, cap), s.weights)
  for phase in (drop_lost, assign, merge_and_split, size_samplers, collect_and_place, rebalance):
    phase(s, out)
  return s, out
