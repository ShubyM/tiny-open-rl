"""The cluster assigns jobs to workers and workers to nodes. Its current assignment is a table in Redis: each job's worker, and each worker's spec and node. The scheduler reads it, fixes what's wrong and writes it back; events only trace what changed. Each node chooses its workers' GPUs and their turns on them itself: see turns.py. Run one cluster scheduler."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import itertools
from dataclasses import dataclass, field

from tinyrl.bus import Bus, Ev, Role
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry

# ***** the assignment *****


@dataclass(frozen=True)
class WorkerSpec:
  """What a worker process loads and needs: a trainer or a sampler, of which base model, for LoRA adapters or full weights, alone or not, and on how many GPUs. Jobs with the same spec share a worker, except full fine-tune and exclusive training jobs, which each get their own."""

  role: Role = Role.TRAINER
  base: str = ""
  lora: bool = False
  exclusive: bool = False
  train_run: str | None = None  # an exclusive job's sampling sessions name it, so they share only with each other
  gpus: int = 1


@dataclass
class Job:
  """A client's training run or sampling session: the worker spec it needs, and the worker serving it, empty until assigned."""

  spec: WorkerSpec
  worker: str = ""
  train_run: str | None = None  # the training run a sampling session samples


@dataclass
class Worker:
  """One trainer or sampler process, serving one or more jobs with its spec. The cluster assigns it a node; the node picks its GPUs."""

  spec: WorkerSpec = WorkerSpec()
  node: str | None = None  # None until it is placed
  gpus: tuple[int, ...] = ()  # desired GPUs, chosen by the node; a running process may still be on its old ones
  # The node's local copy records actual GPUs and reservations; cluster assignments leave these fields empty.
  ticket: int = 0
  served: int | None = None  # tokens it has served on its GPUs this turn, None while it is parked
  pid: int | None = None


@dataclass
class Assignment:
  """The cluster's current assignment: the live nodes, each job's worker, and each worker's node. A node's own copy also holds its workers' GPUs and turns."""

  workers_made: int = 0  # so the next new worker is w<workers_made + 1>
  nodes: dict[str, tuple[int, int]] = field(default_factory=dict)  # live node -> (GPUs, bytes per GPU)
  workers: dict[str, Worker] = field(default_factory=dict)
  jobs: dict[str, Job] = field(default_factory=dict)


def from_table(table: dict) -> Assignment:
  """The assignment as Redis holds it: live nodes, each job's record and worker, and each worker's spec and node."""
  nodes = {n: (v["gpus"], v.get("memory", 0)) for n, v in table["nodes"].items()}
  s = Assignment(workers_made=table["workers_made"], nodes=nodes)
  for job, (rec, worker) in table["runs"].items():
    role, base, lora = Role(rec.get("role", "trainer")), rec.get("base", ""), rec.get("rank", 0) > 0
    exclusive, train_run = rec.get("exclusive", False), rec.get("train_run")
    spec = WorkerSpec(role, base, lora, exclusive, exclusive and train_run or None, rec.get("gpus", 1))
    s.jobs[job] = Job(spec, worker or "", train_run)
  # an exclusive job's sampling sessions end with it
  s.jobs = {r: x for r, x in s.jobs.items() if not x.spec.train_run or x.spec.train_run in s.jobs}
  for name, w in table["workers"].items():
    s.workers[name] = Worker(WorkerSpec(**w["spec"] | {"role": Role(w["spec"]["role"])}), w["node"])
  for x in s.jobs.values():
    if x.worker:  # created in a worker already, as tests do
      s.workers.setdefault(x.worker, Worker(x.spec))
  return s


def assign(s: Assignment, job: str, worker: str) -> None:
  """A job joins a worker. Its first job makes the worker, with its spec."""
  s.jobs[job].worker = worker
  s.workers.setdefault(worker, Worker(s.jobs[job].spec))


def jobs_of(s: Assignment, worker: str) -> list[str]:
  w = s.workers.get(worker)
  shared = w is not None and w.spec.role == Role.SAMPLER
  return [r for r, x in s.jobs.items() if x.worker == worker or (shared and not x.worker and x.spec == w.spec)]


# ***** workers *****


def existing_worker(s: Assignment, job: Job) -> str | None:
  """The worker a new job joins: the one with its spec. A full fine-tune or exclusive training job is alone, and an exclusive job's sampling sessions share only with each other."""
  spec = job.spec
  if spec.role == Role.TRAINER and (not spec.lora or spec.exclusive):
    return None
  return next((name for name, w in s.workers.items() if w.spec == spec), None)


def settings(spec: WorkerSpec, gpus: int, gpu_bytes: int) -> dict[str, str]:
  """The environment a worker's process starts with on a node with this many GPUs of this size."""
  env = {}
  if spec.role != Role.SAMPLER:
    return env
  if gpus >= 2:  # samplers have GPUs of their own, so they start their engine before the first command
    env["TINYRL_WARM_SAMPLER"] = "1"
  if gpu_bytes >= 64 * 2**30:
    env["TINYRL_KV_BYTES"] = str(24 * 1024**3)
  return env


# ***** decide *****


def decide(state: Assignment) -> tuple[Assignment, list[tuple[Ev, dict]]]:
  """The cluster after fixing what's wrong, and what changed, in order. Each change lands in the copy before the next is chosen, so placements see one another."""
  s, out = copy.deepcopy(state), []
  for name, w in sorted(s.workers.items()):
    if w.node and w.node not in s.nodes:  # its node's heartbeat ran out
      w.node, w.gpus = None, ()
      out.append((Ev.UNPLACED, {"worker": name, "rule": "place.node_lost"}))
  for name, job in sorted(s.jobs.items()):
    if not job.worker:
      if not (worker := existing_worker(s, job)):
        s.workers_made = next(n for n in itertools.count(s.workers_made + 1) if f"w{n}" not in s.workers)
        worker = f"w{s.workers_made}"
      assign(s, name, worker)
      out.append((Ev.ASSIGNED, {"run": name, "worker": worker}))
  for name, w in sorted(s.workers.items()):
    if not jobs_of(s, name):
      if w.node:
        out.append((Ev.UNPLACED, {"worker": name, "rule": "place.emptied"}))
      del s.workers[name]
  biggest = max((g for g, _ in s.nodes.values()), default=0)
  unplaced = [n for n, w in s.workers.items() if w.node is None and w.spec.gpus <= biggest]
  for name in sorted(unplaced, key=lambda name: (not s.workers[name].spec.exclusive, name)):
    out.append((Ev.PLACED, placed := place(s, name)))
    s.workers[name].node = placed["node"]
  return s, out


def future_sampler(spec: WorkerSpec) -> WorkerSpec | None:
  """The sampler a full fine-tune's trainer will have. Its weights are big, so it should start beside the trainer."""
  return dataclasses.replace(spec, role=Role.SAMPLER) if spec.role == Role.TRAINER and not spec.lora else None


def has_room(s: Assignment, node: str, *new: WorkerSpec, owed: set[str] = frozenset()) -> bool:
  """Whether workers of these specs could start on the node now, each on empty GPUs. A trainer owed a sampler keeps a GPU for it."""
  there = [(name, w.spec) for name, w in s.workers.items() if w.node == node]
  taken = sum(spec.gpus + (future_sampler(spec).gpus if name in owed else 0) for name, spec in there)
  return taken + sum(spec.gpus for spec in new) <= s.nodes[node][0]


def prefer(options: list, test) -> list:
  """The options that pass, or all of them if none do."""
  return [o for o in options if test(o)] or options


def trainer_nodes(s: Assignment, worker: str) -> set[str]:
  """The nodes of a full fine-tune sampler's trainers, where its weights are. None for others: a LoRA adapter is small to fetch."""
  if s.workers[worker].spec.role != Role.SAMPLER or s.workers[worker].spec.lora:
    return set()
  trainers = [s.jobs.get(s.jobs[r].train_run or "") for r in jobs_of(s, worker)]
  return {s.workers[t.worker].node for t in trainers if t and t.worker in s.workers}


def owed_a_sampler(s: Assignment, but_for: str) -> set[str]:
  """Full fine-tunes' trainers whose sampler isn't placed yet, so they keep a GPU for it. But not the trainers of `but_for`, since it is that sampler."""
  samplers = [x for x, w in s.workers.items() if w.node and w.spec.role == Role.SAMPLER]
  placed = {s.jobs[r].train_run for x in samplers for r in jobs_of(s, x)}  # the trainers that have their sampler
  mine = {s.jobs[t].worker for r in jobs_of(s, but_for) if (t := s.jobs[r].train_run) in s.jobs}
  return {x for x, w in s.workers.items() if future_sampler(w.spec) and not placed & set(jobs_of(s, x))} - mine


def place(s: Assignment, worker: str) -> dict:
  """The node a worker goes to: each preference narrows the nodes, in order, and the first left wins."""
  spec, near, owed = s.workers[worker].spec, trainer_nodes(s, worker), owed_a_sampler(s, but_for=worker)
  with_sampler = [k for k in (spec, future_sampler(spec)) if k]
  nodes = sorted(n for n, (gpus, _) in s.nodes.items() if gpus >= spec.gpus)
  nodes = prefer(nodes, lambda n: has_room(s, n, *with_sampler, owed=owed))
  nodes = prefer(nodes, lambda n: has_room(s, n, spec, owed=owed))
  nodes = prefer(nodes, lambda n: n in near)  # waiting near for a sampler costs minutes, fetching weights seconds
  node = min(nodes, key=lambda n: sum(w.node == n for w in s.workers.values()) / s.nodes[n][0])  # fewest per GPU
  rule = "place.near_its_trainer" if node in near else "place.least_loaded_node"
  return {"worker": worker, "node": node, "rule": rule}


# ***** the loop *****

TICK_MS = getenv("TICK_MS", 100)  # how long it waits for an entry before it looks at the table again


async def run(bus: Bus) -> None:
  registry = Registry(bus)
  while True:
    await bus.read_events("$", count=1, block_ms=TICK_MS)  # wake on anything new, or each tick
    for session in await bus.expired_sessions():  # their clients stopped heartbeating
      await bus.end_session(session)
    # checkpoints whose TTL ran out. The node daemons' sweeps delete their files
    await registry.unregister_checkpoints(*await registry.expired_checkpoints())
    s = from_table(await bus.read_table())
    new, changes = decide(s)
    if not changes:
      continue
    assigned = {r: x.worker for r, x in new.jobs.items() if x.worker != s.jobs[r].worker}
    workers = {n: {"spec": dataclasses.asdict(w.spec), "node": w.node} for n, w in new.workers.items()}
    await bus.write_table(assigned, workers, sorted(s.workers.keys() - new.workers.keys()), new.workers_made, changes)


if __name__ == "__main__":
  asyncio.run(run(Bus()))
