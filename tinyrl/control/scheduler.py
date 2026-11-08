"""The cluster assigns runs to workers and workers to nodes. Its current assignment is a table in Redis: each run's worker, and each worker's kind and node. The scheduler reads it, fixes what's wrong and writes it back; events only trace what changed. A node chooses GPUs directly from its own running processes. Run one cluster scheduler."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
from dataclasses import dataclass, field

from tinyrl.bus import Bus, Cmd, Ev, Op, Role, key
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry


def select_runs(heads: list[Cmd], service: dict[str, float], width: int, prefer: str | None = None) -> list[str]:
  """Choose ready heads by measured node execution seconds per run, not a global tenant share.
  The caller bounds preferred continuations. New runs join the active service floor.
  """
  if width < 1:
    raise ValueError("batch width must be positive")
  ready = {}
  for cmd in heads:
    ready.setdefault(cmd.run, cmd)
  floor = min((service[run] for run in ready if run in service), default=0.0)
  ordered = sorted(
    ready.values(),
    key=lambda cmd: (
      cmd.run != prefer,
      service.get(cmd.run, floor),
      cmd.op != Op.SAVE_WEIGHTS_FOR_SAMPLER,
      tuple(map(int, cmd.entry_id.split("-"))),
      cmd.run,
    ),
  )
  return [cmd.run for cmd in ordered[:width]]


# ***** workers *****


def choose_worker(s: State, run: Run) -> str | None:
  """The worker a new run joins: the one serving its kind of run. A full fine-tune or exclusive training run is alone, and an exclusive run's sampling sessions share only with each other."""
  k = run.kind
  if k.role == Role.TRAINER and (not k.lora or k.exclusive):
    return None
  return next((name for name, w in s.workers.items() if w.kind == k), None)


def settings(k: Kind, gpus: int, gpu_bytes: int) -> dict[str, str]:
  """The environment a worker's process starts with on a node with this many GPUs of this size."""
  env = {}
  if k.role != Role.SAMPLER:
    return env
  if gpus >= 2:  # samplers have GPUs of their own, so they start their engine before the first command
    env["TINYRL_WARM_SAMPLER"] = "1"
  if gpu_bytes >= 64 * 2**30 and not k.lora and not k.exclusive and k.gpus == 1:
    env.update(TINYRL_GPU_MEM="0.42", TINYRL_KV_BYTES=str(16 * 1024**3))
  elif gpu_bytes >= 64 * 2**30:
    env["TINYRL_KV_BYTES"] = str(24 * 1024**3)
  return env


def budget(s: State, k: Kind, node: str | None) -> int:
  """Bytes a worker's process may use on each of its GPUs on a node, 0 if it isn't limited and so takes a GPU alone."""
  gpus, gpu_bytes = s.nodes.get(node, (0, 0))
  return int(float(settings(k, gpus, gpu_bytes).get("TINYRL_GPU_MEM", 0)) * gpu_bytes)


def fit(memory: list[int], gpu_bytes: int) -> bool:
  """Whether workers with these memory budgets can share a GPU. Every budget has to be known, and 0 means unknown."""
  return len(memory) == 1 or (gpu_bytes > 0 and all(memory) and sum(memory) <= gpu_bytes)


# ***** state *****

QUANTUM = getenv("QUANTUM", 30.0)  # seconds a busy holder keeps its GPUs while others wait, engine boot aside


@dataclass(frozen=True)
class Kind:
  """What runs a worker serves. Runs of one kind share a worker."""

  role: Role = Role.TRAINER
  base: str = ""
  lora: bool = False
  exclusive: bool = False
  train_run: str | None = None  # an exclusive run's sampling sessions name it, so they share only with each other
  gpus: int = 1


@dataclass
class Run:
  kind: Kind
  worker: str = ""
  train_run: str | None = None  # the training run a sampling session samples


@dataclass
class Worker:
  kind: Kind = Kind()
  node: str | None = None  # None until it is placed
  gpus: tuple[int, ...] = ()  # desired GPUs, chosen by the node; a running process may still be on its old ones
  # The node's local copy records actual GPUs and reservations; cluster assignments leave these fields empty.
  ticket: int = 0
  since: float | None = None
  busy: float = 0.0
  pid: int | None = None


@dataclass
class State:
  quantum: float = QUANTUM
  made: int = 0  # workers made so far, so the next is w<made + 1>
  nodes: dict[str, tuple[int, int]] = field(default_factory=dict)  # live node -> (GPUs, bytes per GPU)
  workers: dict[str, Worker] = field(default_factory=dict)
  runs: dict[str, Run] = field(default_factory=dict)


def state_from(table: dict) -> State:
  """The cluster as Redis holds it: live nodes, each run's kind and worker, and each worker's kind and node."""
  s = State(made=table["made"], nodes={n: (v["gpus"], v.get("memory", 0)) for n, v in table["nodes"].items()})
  for run, (rec, worker) in table["runs"].items():
    ex, tr = rec.get("exclusive", False), rec.get("train_run")
    k = Kind(
      Role(rec.get("role", "trainer")),
      rec.get("base", ""),
      rec.get("rank", 0) > 0,
      ex,
      ex and tr or None,
      rec.get("gpus", 1),
    )
    s.runs[run] = Run(k, worker or "", tr)
  # an exclusive run's sampling sessions end with it
  s.runs = {r: x for r, x in s.runs.items() if not x.kind.train_run or x.kind.train_run in s.runs}
  for name, w in table["workers"].items():
    s.workers[name] = Worker(Kind(**w["kind"] | {"role": Role(w["kind"]["role"])}), w["node"])
  for x in s.runs.values():
    if x.worker:  # created in a worker already, as tests do
      s.workers.setdefault(x.worker, Worker(x.kind))
  return s


def keep_gpus(new: State, old: State) -> State:
  """A node keeps the GPUs it chose for each worker that stays where it was."""
  for name, w in new.workers.items():
    if (was := old.workers.get(name)) and was.node == w.node:
      w.gpus = was.gpus
  return new


def join(s: State, run: str, worker: str) -> None:
  """A run joins a worker. Its first run makes the worker, of its kind."""
  s.runs[run].worker = worker
  s.workers.setdefault(worker, Worker(s.runs[run].kind))


def members(s: State, worker: str) -> list[str]:
  return [r for r, x in s.runs.items() if x.worker == worker]


# ***** fold *****


# ***** decide *****


def decide(state: State) -> tuple[State, list[tuple[Ev, dict]]]:
  """The cluster after fixing what's wrong, and what changed, in order. Each change lands in the copy before the next is chosen, so placements see one another."""
  s, out = copy.deepcopy(state), []
  for name, w in sorted(s.workers.items()):
    if w.node and w.node not in s.nodes:  # its node's heartbeat ran out
      w.node, w.gpus = None, ()
      out.append((Ev.UNPLACED, {"worker": name, "rule": "place.node_lost"}))
  for run, r in sorted(s.runs.items()):
    if not r.worker:
      if not (worker := choose_worker(s, r)):
        s.made += 1
        worker = f"w{s.made}"
      join(s, run, worker)
      out.append((Ev.ASSIGNED, {"run": run, "worker": worker}))
  for name, w in sorted(s.workers.items()):
    if not members(s, name):
      if w.node:
        out.append((Ev.UNPLACED, {"worker": name, "rule": "place.emptied"}))
      del s.workers[name]
  cap = max((g for g, _ in s.nodes.values()), default=0)
  waiting = [n for n, w in s.workers.items() if w.node is None and w.kind.gpus <= cap]
  for name in sorted(waiting, key=lambda name: (not s.workers[name].kind.exclusive, name)):
    out.append((Ev.PLACED, placed := place(s, name)))
    s.workers[name].node = placed["node"]
  return s, out


def sampler_of(k: Kind) -> Kind | None:
  """The sampler a full fine-tune's trainer will have. Its weights are big, so it should start beside the trainer."""
  return dataclasses.replace(k, role=Role.SAMPLER) if k.role == Role.TRAINER and not k.lora else None


def room(s: State, node: str, *new: Kind, waiting: set[str] = frozenset()) -> bool:
  """Whether workers of these kinds could start on the node now: on empty GPUs, or packed by memory next to others. A waiting trainer keeps a GPU for its sampler."""
  gpus, gpu_bytes = s.nodes[node]
  held: list[list[Kind]] = []  # what each GPU holds, packed first-fit the way choose() packs

  def put(k: Kind) -> bool:
    for kinds in held:
      if (
        not k.exclusive
        and not any(x.exclusive for x in kinds)
        and fit([budget(s, x, node) for x in [*kinds, k]], gpu_bytes)
      ):
        kinds.append(k)
        return True
    if len(held) < gpus:
      held.append([k])
      return True
    return False

  for name, w in s.workers.items():
    for k in [w.kind, sampler_of(w.kind)] if name in waiting else [w.kind]:
      for _ in range(k.gpus if w.node == node else 0):
        put(k)  # past the GPUs, trainers take turns
  return all(put(k) for k in new for _ in range(k.gpus))


def place(s: State, name: str) -> dict:
  """Choose a node with room, keeping a full fine-tune's sampler near its checkpoint files."""
  me = s.workers[name]
  load = {n: sum(w.node == n for w in s.workers.values()) / g for n, (g, _) in s.nodes.items() if g >= me.kind.gpus}
  trainers = [s.runs.get(s.runs[r].train_run or "") for r in members(s, name)]
  near = {s.workers[t.worker].node for t in trainers if t and t.worker in s.workers}
  near = near if me.kind.role == Role.SAMPLER and not me.kind.lora else set()  # a LoRA adapter is small to fetch
  # a full fine-tune's trainer goes where its sampler fits beside it, and keeps that GPU until the sampler comes
  pair = [k for k in (me.kind, sampler_of(me.kind)) if k]
  sampled = {
    s.runs[r].train_run for x, w in s.workers.items() if w.node and w.kind.role == Role.SAMPLER for r in members(s, x)
  }
  waiting = {x for x, w in s.workers.items() if sampler_of(w.kind) and not sampled & set(members(s, x))}
  waiting -= {t.worker for t in trainers if t}  # the GPU kept for this sampler is its own

  # near only with room: waiting there for a sampler costs minutes, fetching weights seconds
  def score(n: str) -> tuple:
    return not room(s, n, *pair, waiting=waiting), not room(s, n, me.kind, waiting=waiting), n not in near, load[n]

  node = min(sorted(load), key=score)
  rule = "place.near_its_trainer" if node in near else "place.least_loaded_node"
  return {"worker": name, "node": node, "rule": rule}


def place_gpus(s: State, node: str, running) -> None:
  """Choose desired GPUs in place. Actual process placements and reservations stay in the node's running table."""
  if node not in s.nodes:
    return
  for name, w in sorted(s.workers.items(), key=lambda item: (not item[1].kind.exclusive, item[0])):
    if w.node != node or not members(s, name):
      continue
    while not w.gpus:
      gpus, aside = choose(s, node, running, name)
      if aside:
        s.workers[aside[0]].gpus = choose(s, node, running, aside[0], avoid=gpus)[0]
      else:
        w.gpus = gpus
  # Spread trainers after departures. A GPU is empty only after its old process has left it.
  gpus, gpu_bytes = s.nodes[node]
  occupied = {g for r in running.values() for g in r.gpus}
  while True:
    there = {g: [n for n, w in s.workers.items() if w.node == node and g in w.gpus] for g in range(gpus)}
    empty = [g for g in range(gpus) if not there[g] and g not in occupied]
    movable = {
      g: [n for n in names if takes_turns(s.workers[n]) and len(s.workers[n].gpus) == 1]
      for g, names in there.items()
      if len(names) > 1 and not fit([budget(s, s.workers[n].kind, node) for n in names], gpu_bytes)
    }
    if not empty or not (crowded := [g for g, names in movable.items() if names]):
      return
    g = max(crowded, key=lambda g: (len(there[g]), -g))
    name = min(movable[g], key=lambda n: (not s.workers[n].kind.lora, n in running and running[n].since is not None, n))
    s.workers[name].gpus = (empty[0],)


def has_waiter(running, gpus: tuple[int, ...]) -> bool:
  return any(r.ticket > 0 and any(g in r.gpus for g in gpus) for r in running.values())


def can_run(s: State, node: str, running, name: str) -> bool:
  """Reserve all actual GPUs together, in ticket order on each GPU, including processes still draining after unplacement."""
  r = running[name]
  if r.ticket <= 0 or r.since is not None or not r.gpus or node not in s.nodes:
    return False
  for gpu in r.gpus:
    if any(x.ticket > 0 and (x.ticket, n) < (r.ticket, name) and gpu in x.gpus for n, x in running.items()):
      return False
    memory = [budget(s, x.kind, node) for x in running.values() if x.since is not None and gpu in x.gpus]
    if not fit([*memory, budget(s, r.kind, node)], s.nodes[node][1]):
      return False
  return True


def takes_turns(w: Worker) -> bool:
  """Whether a worker may share a GPU it doesn't fit on. A trainer parks and resumes in about a second. A sampler wakes its engine slowly, and an exclusive worker is alone by definition."""
  return w.kind.role == Role.TRAINER and not w.kind.exclusive


def choose(s: State, node: str, running, name: str, avoid: tuple[int, ...] = ()) -> tuple[tuple[int, ...], list[str]]:
  """Choose GPUs and trainers to move aside. Pack known memory budgets, keep exclusive workers alone, and otherwise take turns with the least busy trainers. A sampler moves trainers rather than sharing with them when there is room."""
  gpus, gpu_bytes = s.nodes[node]
  me, width, memory = s.workers[name], s.workers[name].kind.gpus, budget(s, s.workers[name].kind, node)
  there = {
    g: [(n, w) for n, w in s.workers.items() if n != name and w.node == node and g in w.gpus] for g in range(gpus)
  }

  def fits_on(g: int) -> bool:
    others = [w for _, w in there[g]]
    ok = not me.kind.exclusive and not any(w.kind.exclusive for w in others)
    return not others or (ok and fit([budget(s, w.kind, node) for w in others] + [memory], gpu_bytes))

  def trainers_only(g: int) -> bool:
    return bool(there[g]) and all(takes_turns(w) and w.kind.gpus <= gpus - width for _, w in there[g])

  def clear(g: int) -> bool:
    return fits_on(g) or trainers_only(g)

  options = [g for g in range(gpus) if g not in avoid]
  if len(fitting := [g for g in options if fits_on(g)]) >= width:
    return tuple(sorted(fitting, key=lambda g: (bool(there[g]) != bool(memory), g))[:width]), []
  busy = {g: sum(running[n].busy for n, _ in there[g] if n in running) for g in options}
  if takes_turns(me) or len(options) <= width:  # with no spare GPUs there is nowhere to move to
    return tuple(sorted(options, key=lambda g: (not clear(g), busy[g], g))[:width]), []
  # as few moves as it can, LoRA trainers first, since a full fine-tune saves its weights and AdamW state far slower
  lora = {g: all(w.kind.lora for _, w in there[g]) for g in options}
  moves = {g: 0 if fits_on(g) else sum(w.kind.gpus for _, w in there[g]) for g in options}
  chosen = tuple(sorted(options, key=lambda g: (not clear(g), moves[g], not lora[g], busy[g], g))[:width])
  aside = sorted({n for g in chosen if not fits_on(g) for n, _ in there[g]})
  return chosen, aside if all(map(clear, chosen)) else []


# ***** the loop *****

TICK_MS = getenv("TICK_MS", 100)  # how long it waits for an entry before it looks at the table again


async def run(bus: Bus, quantum: float = QUANTUM) -> None:
  registry = Registry(bus)
  while True:
    await bus.read_events("$", count=1, block_ms=TICK_MS)  # wake on anything new, or each tick
    for session in await bus.expired_sessions():  # their clients stopped heartbeating
      await bus.end_session(session)
    # checkpoints whose TTL ran out. The node daemons' sweeps delete their files
    await registry.unregister_checkpoints(*await registry.expired_checkpoints())
    s = state_from(await bus.read_table())
    s.quantum = quantum
    new, changes = decide(s)
    if not changes:
      continue
    assigned = {r: x.worker for r, x in new.runs.items() if x.worker != s.runs[r].worker}
    workers = {n: {"kind": dataclasses.asdict(w.kind), "node": w.node} for n, w in new.workers.items()}
    await bus.write_table(assigned, workers, sorted(s.workers.keys() - new.workers.keys()), new.made, changes)
    await bus.r.set(key("state"), json.dumps(dataclasses.asdict(new), sort_keys=True))


if __name__ == "__main__":
  asyncio.run(run(Bus()))
