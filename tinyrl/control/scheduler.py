"""Cluster placement, rebalancing, and elastic scaling from a single Redis table."""

import asyncio
import copy
import itertools
import math
from dataclasses import asdict, dataclass, field

from tinyrl.bus import Bus, Ev, Role
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry

PROMPTS_PER_SAMPLER, TICK_MS = getenv("PROMPTS_PER_SAMPLER", 32), getenv("TICK_MS", 100)


@dataclass(frozen=True)
class WorkerSpec:
  role: Role = Role.TRAINER
  base: str = ""
  lora: bool = False
  exclusive: bool = False
  train_run: str | None = None
  gpus: int = 1


@dataclass
class Job:
  spec: WorkerSpec
  worker: str = ""
  train_run: str | None = None
  backlog: int = 0
  pinned: bool = False


@dataclass
class Worker:
  spec: WorkerSpec = WorkerSpec()
  node: str | None = None
  gpus: tuple[int, ...] = ()
  vtime: int | None = None
  served: int | None = None
  pid: int | None = None


@dataclass
class Assignment:
  workers_made: int = 0
  nodes: dict[str, tuple[int, int]] = field(default_factory=dict)
  occupied: dict[str, dict[str, tuple[int, ...]]] = field(default_factory=dict)
  workers: dict[str, Worker] = field(default_factory=dict)
  jobs: dict[str, Job] = field(default_factory=dict)


def from_table(table: dict) -> Assignment:
  s = Assignment(table["workers_made"], {n: (v["gpus"], v.get("memory", 0)) for n, v in table["nodes"].items()})
  for n, v in table["nodes"].items():
    if occ := {w: tuple(i["gpus"]) for w, i in v.get("workers", {}).items() if i.get("gpus")}:
      s.occupied[n] = occ
  for job, (rec, worker) in table["runs"].items():
    role, excl, tr = Role(rec.get("role", "trainer")), rec.get("exclusive", False), rec.get("train_run") or None
    spec = WorkerSpec(role, rec.get("base", ""), rec.get("rank", 0) > 0, excl, tr if excl else None, rec.get("gpus", 1))
    s.jobs[job] = Job(spec, worker or "", tr, int(table.get("backlog", {}).get(job, 0)), bool(rec.get("worker")))
  s.jobs = {r: x for r, x in s.jobs.items() if not x.spec.train_run or x.spec.train_run in s.jobs}
  for name, w in table["workers"].items():
    spec = WorkerSpec(**w["spec"] | {"role": Role(w["spec"]["role"])})
    s.workers[name] = Worker(spec, w["node"], tuple(w.get("gpus") or ()))
  for x in [j for j in s.jobs.values() if j.worker]:
    s.workers.setdefault(x.worker, Worker(x.spec))
  return s


def jobs_of(s: Assignment, worker: str) -> list[str]:
  w = s.workers.get(worker)
  shared = w is not None and w.spec.role == Role.SAMPLER
  return [r for r, x in s.jobs.items() if x.worker == worker or (shared and not x.worker and x.spec == w.spec)]


def settings(spec: WorkerSpec, gpus: int, gpu_bytes: int) -> dict[str, str]:
  kv = {"TINYRL_KV_BYTES": str(24 * 1024**3)} if gpu_bytes >= 64 * 2**30 else {}
  return ({"TINYRL_WARM_SAMPLER": "1"} if gpus >= 2 else {}) | kv if spec.role == Role.SAMPLER else {}


def at(s: Assignment, node: str, g: int) -> list[Worker]:
  return [w for w in s.workers.values() if w.node == node and g in w.gpus]


def move_cost(spec: WorkerSpec) -> int:
  return 5 if spec.role == Role.SAMPLER else (2 if spec.lora else 4)


def pool_of(s: Assignment, spec: WorkerSpec) -> list[str]:
  if spec.exclusive or (spec.role == Role.TRAINER and not spec.lora):
    return []
  pinned = {x.worker for x in s.jobs.values() if x.pinned and x.worker}
  return sorted(n for n, w in s.workers.items() if w.spec == spec and n not in pinned)


def extras(s: Assignment) -> list[str]:
  return [n for n, w in s.workers.items() if len(p := pool_of(s, w.spec)) > 1 and n in p]


def load_key(s: Assignment, n: str) -> tuple:
  return (sum(s.jobs[r].backlog > 0 for r in jobs_of(s, n)), len(jobs_of(s, n)), n)


def spec_demand(s: Assignment, spec: WorkerSpec) -> int:
  jobs = [x for x in s.jobs.values() if x.spec == spec and not (x.worker if spec.role == Role.SAMPLER else x.pinned)]
  if spec.role == Role.SAMPLER:
    return max(1, math.ceil(sum(x.backlog for x in jobs) / PROMPTS_PER_SAMPLER)) if jobs else 0
  return max(1, sum(x.backlog > 0 for x in jobs)) if jobs else 0


def trainer_nodes(s: Assignment, n: str) -> set[str]:
  w = s.workers[n].spec
  ts = [s.jobs.get(s.jobs[r].train_run or "") for r in jobs_of(s, n)] if w.role == Role.SAMPLER and not w.lora else []
  return {s.workers[t.worker].node for t in ts if t and t.worker in s.workers and s.workers[t.worker].node}


def draining_map(s: Assignment) -> dict[str, list[int]]:
  ok = {(n, w.node, w.gpus) for n, w in s.workers.items()}
  return {d: [g for n, p in o.items() if (n, d, p) not in ok for g in p] for d, o in s.occupied.items() if d in s.nodes}


def gpu_phi(desired: list[Worker], nd: int = 0) -> float:
  if any(w.spec.exclusive for w in desired):
    return math.inf if len(desired) > 1 else 0.0
  occ = [6 if w.spec.role == Role.SAMPLER else 1 for w in desired] + [1] * nd
  return 0.0 if len(occ) <= 1 else float(2 * len(occ) * (len(occ) - 1) + sum(occ))


def phi(s: Assignment, draining: dict[str, list[int]] | None = None) -> float:
  dmap = draining_map(s) if draining is None else draining
  gpus = [gpu_phi(at(s, nd, g), dmap.get(nd, ()).count(g)) for nd, (ng, _) in s.nodes.items() for g in range(ng)]
  miss = sum(2.0 for n, w in s.workers.items() if w.node and (near := trainer_nodes(s, n)) and w.node not in near)
  return sum(gpus) + miss


def gscore(there: list[Worker], w: Worker, nd: int, g: int) -> tuple:
  mob = max(((x.spec.gpus, move_cost(x.spec)) for x in there), default=(0, 0))
  return (gpu_phi([*there, w], nd) - gpu_phi(there, nd), mob, g)


def best_placement(s: Assignment, worker: str, draining: dict[str, list[int]], empty_only: bool = False):
  w, width, cands = s.workers[worker], s.workers[worker].spec.gpus, []
  for node, (ngpus, _) in sorted(s.nodes.items()):
    if ngpus < width:
      continue
    dlist, there = draining.get(node, ()), {g: [x for x in at(s, node, g) if x is not w] for g in range(ngpus)}
    if empty_only or w.spec.exclusive:
      if len(empty := [g for g in range(ngpus) if not there[g] and g not in dlist]) < width:
        continue
      gpus = tuple(empty[:width])
    else:
      gpus = tuple(sorted(g for *_, g in sorted(gscore(there[g], w, dlist.count(g), g) for g in range(ngpus))[:width]))
    used = sum(x.spec.gpus for x in s.workers.values() if x is not w and x.node == node)
    old, w.node, w.gpus = (w.node, w.gpus), node, gpus
    cand_phi = phi(s, draining)
    w.node, w.gpus = old
    if not math.isinf(cand_phi):
      tight = not w.spec.lora and w.spec.role == Role.TRAINER and ngpus < 2 * width
      cands.append(((cand_phi, tight, used / ngpus, -(ngpus - used) if used else 0, node), (cand_phi, node, gpus)))
  return min(cands)[1] if cands else None


def place_exclusive(s: Assignment, worker: str, draining: dict[str, list[int]]) -> None:
  w, cands = s.workers[worker].spec.gpus, []
  if any(ng >= w and sum(not at(s, n, g) for g in range(ng)) >= w for n, (ng, _) in s.nodes.items()):
    return
  for nd, (ng, _) in sorted(s.nodes.items()):
    clear = [g for g in range(ng) if g not in draining.get(nd, ()) and all(not x.spec.exclusive for x in at(s, nd, g))]
    if len(clear) >= w:
      gpus = tuple(sorted(sorted(clear, key=lambda g: (sum(move_cost(x.spec) for x in at(s, nd, g)), g))[:w]))
      t = copy.deepcopy(s)
      moved = sorted(n for n, x in t.workers.items() if x.node == nd and {*x.gpus} & {*gpus})
      for n in moved:
        t.workers[n].node, t.workers[n].gpus = None, ()
      t.workers[worker].node, t.workers[worker].gpus = nd, gpus
      for n in moved:
        if fit := best_placement(t, n, draining):
          _, t.workers[n].node, t.workers[n].gpus = fit
      if all(t.workers[n].node for n in moved) and (sc := phi(t, draining)) < math.inf:
        cands.append((sc, nd, t.workers))
  if cands:
    s.workers = min(cands, key=lambda x: x[:2])[2]


def new_worker(s: Assignment, spec: WorkerSpec) -> str:
  s.workers_made = next(n for n in itertools.count(s.workers_made + 1) if f"w{n}" not in s.workers)
  s.workers[name := f"w{s.workers_made}"] = Worker(spec)
  return name


def unplaced(s: Assignment, biggest: int) -> list[str]:
  cands = [n for n, w in s.workers.items() if not w.node and w.spec.gpus <= biggest]
  return sorted(cands, key=lambda n: (not s.workers[n].spec.exclusive, n))


def shrink_targets(s: Assignment, biggest: int) -> set[str]:
  free, ex = {n: sum(not at(s, n, g) for g in range(ng)) for n, (ng, _) in s.nodes.items()}, set(extras(s))
  for u in unplaced(s, biggest):
    w = s.workers[u].spec.gpus
    near = {n for n in trainer_nodes(s, u) if n in s.nodes and s.nodes[n][0] >= w}
    if near and (fit := sorted(n for n in near if free[n] >= w)):
      free[fit[0]] -= w
    elif near and any(s.workers[n].node in near for n in ex):
      return near
    elif fit := sorted(n for n, f in free.items() if s.nodes[n][0] >= w and f >= w):
      free[fit[0]] -= w
    else:
      excl = {d: {g for x in s.workers.values() if x.node == d and x.spec.exclusive for g in x.gpus} for d in free}
      return {d for d, (ng, _) in s.nodes.items() if ng - len(excl[d]) >= w}
  return set()


def shrink_key(s: Assignment, name: str) -> tuple:
  w, num = s.workers[name], int(name[1:]) if name.startswith("w") and name[1:].isdigit() else 0
  ratio = spec_demand(s, w.spec) / len(pool_of(s, w.spec))
  return (bool(w.node), ratio, len(jobs_of(s, name)), move_cost(w.spec), -num, name)


def drop_replica(s: Assignment, victim: str, was: dict, out: list) -> None:
  if (spec := s.workers.pop(victim).spec).role == Role.TRAINER:
    sibs = pool_of(s, spec)
    for r, job in sorted(s.jobs.items()):
      if job.worker == victim:
        job.worker = target = min(sibs, key=lambda n: load_key(s, n))
        out.append((Ev.ASSIGNED, {"run": r, "worker": target}))
  if was.pop(victim, None) is not None:
    out.append((Ev.UNPLACED, {"worker": victim, "rule": "scale.shrink"}))


def drop_crowded(s: Assignment, was: dict, out: list) -> None:
  while bad := [n for n in extras(s) if not (w := s.workers[n]).node or any(len(at(s, w.node, g)) > 1 for g in w.gpus)]:
    drop_replica(s, min(bad, key=lambda n: shrink_key(s, n)), was, out)


def grow_cand(s: Assignment, spec: WorkerSpec, draining: dict) -> tuple | None:
  if not (count := len(pool_of(s, spec))) or (demand := spec_demand(s, spec)) <= count:
    return None
  s.workers[p := f"_p{s.workers_made}"] = Worker(spec)
  fit, _ = best_placement(s, p, draining, empty_only=True), s.workers.pop(p)
  return ((demand / count, demand - count, str(spec)), spec, fit[1], fit[2]) if fit else None


def decide(state: Assignment) -> tuple[Assignment, list[tuple[Ev, dict]]]:
  s, out = copy.deepcopy(state), []
  s.occupied = {n: v for n, v in s.occupied.items() if n in s.nodes}
  was = {n: (w.node, w.gpus) for n, w in s.workers.items() if w.node}

  for name, w in sorted(s.workers.items()):
    if w.node and (w.node not in s.nodes or len(w.gpus) != w.spec.gpus or any(g >= s.nodes[w.node][0] for g in w.gpus)):
      rule = "place.node_lost" if w.node not in s.nodes else "place.resized"
      w.node, w.gpus = None, ()
      was.pop(name, None)
      out.append((Ev.UNPLACED, {"worker": name, "rule": rule}))

  for name, job in sorted(s.jobs.items()):
    if job.spec.role == Role.SAMPLER:
      if not job.worker and not any(w.spec == job.spec for w in s.workers.values()):
        new_worker(s, job.spec)
    elif not job.worker or job.worker not in s.workers:
      sibs = [] if job.pinned else pool_of(s, job.spec)
      wname = min(sibs, key=lambda n: load_key(s, n)) if sibs else (job.worker or new_worker(s, job.spec))
      s.jobs[name].worker = wname
      s.workers.setdefault(wname, Worker(job.spec))
      out.append((Ev.ASSIGNED, {"run": name, "worker": wname}))

  for name in sorted(list(s.workers)):
    if not jobs_of(s, name):
      if was.pop(name, None) is not None:
        out.append((Ev.UNPLACED, {"worker": name, "rule": "place.emptied"}))
      del s.workers[name]

  biggest = max((g for g, _ in s.nodes.values()), default=0)
  drop_crowded(s, was, out)
  while (targets := shrink_targets(s, biggest)) and (ex := [n for n in extras(s) if s.workers[n].node in targets]):
    drop_replica(s, min(ex, key=lambda n: shrink_key(s, n)), was, out)

  draining = draining_map(s)
  for name in unplaced(s, biggest):
    if fit := best_placement(s, name, draining):
      _, s.workers[name].node, s.workers[name].gpus = fit
    elif s.workers[name].spec.exclusive:
      place_exclusive(s, name, draining)
  drop_crowded(s, was, out)
  draining = draining_map(s)

  while 0 < (cur_phi := phi(s, draining)) < math.inf:
    best_gain, best_move = 0.0, None
    for name, w in sorted(s.workers.items()):
      if w.node and (cand := best_placement(s, name, draining)):
        if (cand[1], cand[2]) != (w.node, w.gpus) and (gain := (cur_phi - cand[0]) - move_cost(w.spec)) > best_gain:
          best_gain, best_move = gain, (name, cand[1], cand[2])
    if best_move is None:
      break
    s.workers[best_move[0]].node, s.workers[best_move[0]].gpus = best_move[1], best_move[2]
  drop_crowded(s, was, out)

  for name, w in sorted(s.workers.items()):
    if w.node and was.get(name) != (w.node, w.gpus):
      near = w.node in trainer_nodes(s, name)
      rule = "place.rebalanced" if name in was else ("place.near_its_trainer" if near else "place.phi")
      out.append((Ev.PLACED, {"worker": name, "node": w.node, "gpus": list(w.gpus), "rule": rule}))

  for spec in sorted({w.spec for w in s.workers.values() if w.spec.role == Role.TRAINER}, key=str):
    while sibs := [n for n in pool_of(s, spec) if s.workers[n].node]:
      lo, hi = min(sibs, key=lambda n: load_key(s, n)), max(sibs, key=lambda n: load_key(s, n))
      if load_key(s, lo)[0] > 0 or load_key(s, hi)[0] < 2:
        break
      s.jobs[run := next(r for r in sorted(jobs_of(s, hi)) if s.jobs[r].backlog > 0)].worker = lo
      out.append((Ev.ASSIGNED, {"run": run, "worker": lo}))

  if not unplaced(s, biggest):
    while cands := [c for spec in {w.spec for w in s.workers.values()} if (c := grow_cand(s, spec, draining))]:
      _, spec, node, gpus = max(cands, key=lambda x: x[0])
      sibs, name = pool_of(s, spec), new_worker(s, spec)
      s.workers[name].node, s.workers[name].gpus = node, gpus
      if spec.role == Role.TRAINER:
        src = max(sibs, key=lambda n: load_key(s, n))
        runs = [r for r in sorted(jobs_of(s, src)) if s.jobs[r].backlog > 0] or sorted(jobs_of(s, src))
        s.jobs[runs[-1]].worker = name
        out.append((Ev.ASSIGNED, {"run": runs[-1], "worker": name}))
      out.append((Ev.PLACED, {"worker": name, "node": node, "gpus": list(gpus), "rule": "scale.grow"}))

  return s, out


async def run(bus: Bus) -> None:
  registry = Registry(bus)
  while True:
    await bus.read_events("$", count=1, block_ms=TICK_MS)
    for session in await bus.expired_sessions():
      await bus.end_session(session)
    await registry.unregister_checkpoints(*await registry.expired_checkpoints())
    s = from_table(await bus.read_table())
    new, changes = decide(s)
    if changes:
      assigned = {r: x.worker for r, x in new.jobs.items() if x.worker != s.jobs[r].worker}
      workers = {n: {"spec": asdict(w.spec), "node": w.node, "gpus": list(w.gpus)} for n, w in new.workers.items()}
      await bus.write_table(assigned, workers, sorted(s.workers.keys() - new.workers.keys()), new.workers_made, changes)


if __name__ == "__main__":
  asyncio.run(run(Bus()))
