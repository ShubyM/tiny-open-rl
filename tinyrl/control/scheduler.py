"""The cluster assigns runs to workers and workers to nodes. Each node chooses GPUs and gives turns locally.

Both decisions use the pure rules here. Only cluster assignments go through Redis: local_decide is folded directly by the node that owns the processes, so a grant and its execution cannot cross in the event log. Run exactly one cluster scheduler; restarting it replays assignments without restarting models.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import time
from dataclasses import dataclass, field

from tinyrl.bus import Bus, Ev, Event, Role, key
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry

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

Key = tuple[str, int]  # (node, GPU index)

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
  gpus: tuple[int, ...] = ()
  on: tuple[Key, ...] = ()  # the GPUs its process is on, until it exits, wherever the worker is placed now
  asked: int = 0  # its place in line, the entry count when it asked. 0 while not waiting
  since: float | None = None  # when its turn started, pushed back by engine boot. None while it holds no GPU
  revoked: bool = False  # told to give its GPUs back
  idle: bool = False  # its worker found no commands since its last batch
  busy: float = 0.0  # seconds its batches took, so far
  booting: bool = False


@dataclass
class State:
  quantum: float = QUANTUM
  n: int = 0  # entries folded
  nodes: dict[str, tuple[int, int]] = field(default_factory=dict)  # node -> (GPUs, bytes per GPU)
  workers: dict[str, Worker] = field(default_factory=dict)
  runs: dict[str, Run] = field(default_factory=dict)


def holders(s: State, k: Key) -> list[str]:
  return sorted(name for name, w in s.workers.items() if w.since is not None and k in w.on)


def first_in_line(s: State, k: Key) -> str | None:
  waiting = [(w.asked, name) for name, w in s.workers.items() if w.asked and k in w.on]
  return min(waiting)[1] if waiting else None


def uses(s: State, w: Worker) -> int:
  """The budget of the process running now, from the node it runs on, which a moved worker's new node may not be."""
  return budget(s, w.kind, w.on[0][0]) if w.on else 0


def fits(s: State, k: Key, name: str) -> bool:
  """A worker joins a GPU's holders if nobody holds it, or if its budget and every holder's are known and fit."""
  return fit([uses(s, s.workers[n]) for n in (*holders(s, k), name)], s.nodes[k[0]][1])


def join(s: State, run: str, worker: str) -> None:
  """A run joins a worker. Its first run makes the worker, of its kind."""
  s.runs[run].worker = worker
  s.workers.setdefault(worker, Worker(s.runs[run].kind))


def members(s: State, worker: str) -> list[str]:
  return [r for r, x in s.runs.items() if x.worker == worker]


def stop(w: Worker) -> None:
  """Its process is gone, and its turn with it."""
  w.on, w.asked, w.since, w.revoked, w.idle, w.booting = (), 0, None, False, False, False


# ***** fold *****


def fold(state: State, e: Event) -> State:
  """The state after entry e. Changes and returns the state it is given, and ignores entries it doesn't know."""
  s = state
  s.n += 1
  b = e.body
  w = s.workers.get(b.get("worker", ""))
  match e.kind:
    case Ev.RUN_CREATED:
      ex, tr = b.get("exclusive", False), b.get("train_run")
      k = Kind(
        Role(b.get("role", "trainer")), b.get("base", ""), b.get("rank", 0) > 0, ex, ex and tr or None, b.get("gpus", 1)
      )
      s.runs[b["run"]] = Run(k, train_run=b.get("train_run"))
      if b.get("worker"):  # created in a worker already, as tests do
        join(s, b["run"], b["worker"])
    case Ev.ASSIGNED if b["run"] in s.runs:
      join(s, b["run"], b["worker"])
    case Ev.RUN_DELETED:  # an exclusive run's sampling sessions end with it
      s.runs = {r: x for r, x in s.runs.items() if r != b["run"] and x.kind.train_run != b["run"]}
    case Ev.NODE_UP:  # a node daemon (re)started, and its processes died with the old one
      s.nodes[b["node"]] = (b["gpus"], b.get("memory", 0))
      for x in s.workers.values():
        if any(node == b["node"] for node, _ in x.on):
          stop(x)
    case Ev.NODE_LOST:
      s.nodes.pop(b["node"], None)
      for x in s.workers.values():
        if any(node == b["node"] for node, _ in x.on):
          stop(x)
        if x.node == b["node"]:
          x.node, x.gpus = None, ()
    case Ev.PLACED if w:  # cluster placement names a node; local placement also names its GPUs
      if w.node != b["node"]:
        w.gpus = ()
      w.node = b["node"]
      if "gpus" in b:
        w.gpus = tuple(b["gpus"])
    case Ev.UNPLACED if w:
      w.node, w.gpus = None, ()
    case Ev.STARTED if w and w.node == b["node"]:
      stop(w)
      w.on = tuple((w.node, i) for i in w.gpus)
    case Ev.EXITED if w:
      stop(w)
    case Ev.WANTS_GPU if w and w.on:
      w.idle, w.booting = False, b.get("booting", False)
      if w.since is None and not w.asked:
        w.asked = s.n
    case Ev.BATCH_DONE if w:
      w.idle, w.busy, w.booting = False, w.busy + b.get("seconds", 0.0), False
      if w.since is not None:
        w.since += b.get("boot", 0.0)
    case Ev.IDLE if w:
      w.idle = True
    case Ev.RELEASED if w:
      w.since, w.revoked = None, False
    case Ev.GRANTED if w:
      w.asked, w.since = 0, e.t
    case Ev.REVOKED if w:
      w.revoked = True
  s.workers = {name: x for name, x in s.workers.items() if x.node or x.on or members(s, name)}
  return s


# ***** decide *****


def decide(state: State, now: float, steps=None) -> list[tuple[Ev, dict]]:
  """Fold each decision into a copy before choosing the next, so placements and grants see one another."""
  s, out = copy.deepcopy(state), []
  while d := next((d for step in steps or (assign, unplace, place_next) if (d := step(s, now))), None):
    s = fold(s, Event("", now, *d))
    out.append(d)
  return out


def local_decide(state: State, node: str, now: float) -> list[tuple[Ev, dict]]:
  s = copy.copy(state)
  s.workers = {n: w for n, w in state.workers.items() if w.node == node or any(k[0] == node for k in w.on)}
  s.nodes = {node: state.nodes[node]} if node in state.nodes else {}
  return decide(s, now, (place_gpus, spread, grant, revoke)) if s.nodes else []


def place_gpus(s: State, now: float) -> tuple[Ev, dict] | None:
  for name, w in sorted(s.workers.items(), key=lambda item: (not item[1].kind.exclusive, item[0])):
    if w.node in s.nodes and not w.gpus and members(s, name):
      gpus, why, aside = choose(s, w.node, name)
      if aside:
        to, why, _ = choose(s, w.node, aside[0], avoid=tuple(gpus))
        return Ev.PLACED, {"worker": aside[0], "node": w.node, "gpus": to, "rule": "place.moved_aside", "gpu_rule": why}
      return Ev.PLACED, {"worker": name, "node": w.node, "gpus": gpus, "rule": "place.local", "gpu_rule": why}


def assign(s: State, now: float) -> tuple[Ev, dict] | None:
  for run_id, r in sorted(s.runs.items()):
    if not r.worker:
      return Ev.ASSIGNED, {"run": run_id, "worker": choose_worker(s, r) or f"w{s.n + 1}"}


def unplace(s: State, now: float) -> tuple[Ev, dict] | None:
  for name, w in sorted(s.workers.items()):
    if w.node and not members(s, name):
      return Ev.UNPLACED, {"worker": name, "rule": "place.emptied"}


def place_next(s: State, now: float) -> tuple[Ev, dict] | None:
  """Place one worker that has runs and no GPU, exclusive ones first."""
  cap = max((g for g, _ in s.nodes.values()), default=0)
  unplaced = [name for name, w in s.workers.items() if w.node is None and members(s, name) and w.kind.gpus <= cap]
  if unplaced:
    return Ev.PLACED, place(s, min(unplaced, key=lambda name: (not s.workers[name].kind.exclusive, name)))


def placed(w: Worker) -> bool:
  return w.on == tuple((w.node, g) for g in w.gpus)


def parks_in_place(w: Worker) -> bool:
  """A multi-GPU trainer saves its runs with collectives on the GPUs its process is on, so it takes one last turn there before it leaves them."""
  return w.kind.role == Role.TRAINER and len(w.on) > 1


def settled(w: Worker) -> bool:
  return placed(w) or parks_in_place(w)


def grant(s: State, now: float) -> tuple[Ev, dict] | None:
  for _, name in sorted((w.asked, name) for name, w in s.workers.items() if w.asked):
    w = s.workers[name]
    if w.on and settled(w) and all(first_in_line(s, k) == name and fits(s, k, name) for k in w.on):
      return Ev.GRANTED, {"worker": name, "rule": "turn.granted"}


def revoke(s: State, now: float) -> tuple[Ev, dict] | None:
  for name, w in sorted(s.workers.items()):
    if w.since is None or w.revoked:
      continue
    if not settled(w):
      return Ev.REVOKED, {"worker": name, "rule": "turn.moved"}
    if not any(first_in_line(s, k) for k in w.on):
      continue
    if w.idle:
      return Ev.REVOKED, {"worker": name, "rule": "turn.idle_holder_yields"}
    if now >= w.since + s.quantum and not w.booting:
      return Ev.REVOKED, {"worker": name, "rule": "turn.quantum_expired"}


def place(s: State, name: str) -> dict:
  """Choose a node with room, keeping a full fine-tune's sampler near its checkpoint files."""
  me = s.workers[name]
  load = {n: sum(w.node == n for w in s.workers.values()) / g for n, (g, _) in s.nodes.items() if g >= me.kind.gpus}
  trainers = [s.runs.get(s.runs[r].train_run or "") for r in members(s, name)]
  near = {s.workers[t.worker].node for t in trainers if t and t.worker in s.workers}
  near = near if me.kind.role == Role.SAMPLER and not me.kind.lora else set()  # a LoRA adapter is small to fetch
  node = min(sorted(load), key=lambda n: (n not in near, load[n]))
  rule = "place.near_its_trainer" if node in near else "place.least_loaded_node"
  return {"worker": name, "node": node, "rule": rule}


def spread(s: State, now: float) -> tuple[Ev, dict] | None:
  """Move one trainer off a shared GPU onto an empty one on the same node. Empty means no worker is placed on it and no old process still runs on it. Only trainers that take turns move: workers that share because their memory fits stay put. LoRA trainers move first, since a full fine-tune is slow to save and restore. Then a trainer waiting for its turn moves before the one using the GPU."""
  for node, (gpus, gpu_bytes) in sorted(s.nodes.items()):
    there = {g: [n for n, w in s.workers.items() if w.node == node and g in w.gpus] for g in range(gpus)}
    running = {g for w in s.workers.values() for n, g in w.on if n == node}
    empty = [g for g in range(gpus) if not there[g] and g not in running]
    movable = {
      g: [n for n in names if takes_turns(s.workers[n]) and len(s.workers[n].gpus) == 1]
      for g, names in there.items()
      if len(names) > 1 and not fit([budget(s, s.workers[n].kind, node) for n in names], gpu_bytes)
    }
    if empty and (crowded := [g for g, names in movable.items() if names]):
      g = max(crowded, key=lambda g: (len(there[g]), -g))
      name = min(movable[g], key=lambda n: (not s.workers[n].kind.lora, s.workers[n].since is not None, n))
      return Ev.PLACED, {"worker": name, "node": node, "gpus": [empty[0]], "rule": "place.spread"}


def takes_turns(w: Worker) -> bool:
  """Whether a worker may share a GPU it doesn't fit on. A trainer parks and resumes in about a second. A sampler wakes its engine slowly, and an exclusive worker is alone by definition."""
  return w.kind.role == Role.TRAINER and not w.kind.exclusive


def choose(s: State, node: str, name: str, avoid: tuple[int, ...] = ()) -> tuple[list[int], str, list[str]]:
  """The GPUs on a node a worker goes to, why, and the trainers it moves out of the way. An exclusive worker is alone, and others stay off its GPU. A worker takes an empty GPU, or one its memory budget fits on next to the workers there, and a worker with a budget fills such a GPU before it takes an empty one. With neither, a worker that takes turns takes them with the least busy trainers. One that doesn't takes those trainers' GPU, and they move elsewhere."""
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
    chosen = sorted(fitting, key=lambda g: (bool(there[g]) != bool(memory), g))[:width]
    return chosen, "fits_next_to_others" if any(there[g] for g in chosen) else "empty_gpu", []
  busy = {g: sum(w.busy for _, w in there[g]) for g in options}
  if takes_turns(me) or len(options) <= width:  # with no spare GPUs there is nowhere to move to
    return sorted(options, key=lambda g: (not clear(g), busy[g], g))[:width], "takes_turns", []
  # as few moves as it can, LoRA trainers first, since a full fine-tune saves its weights and AdamW state far slower
  lora = {g: all(w.kind.lora for _, w in there[g]) for g in options}
  moves = {g: 0 if fits_on(g) else sum(w.kind.gpus for _, w in there[g]) for g in options}
  chosen = sorted(options, key=lambda g: (not clear(g), moves[g], not lora[g], busy[g], g))[:width]
  aside = sorted({n for g in chosen if not fits_on(g) for n, _ in there[g]})
  return (chosen, "moves_trainers_aside", aside) if all(map(clear, chosen)) else (chosen, "takes_turns", [])


# ***** the loop *****

TICK_MS = getenv("TICK_MS", 100)  # how long it waits for an entry before it checks quanta and heartbeats again
BATCH = 10_000  # entries read at once. It decides only once a read comes back short, so never from half the log


async def run(bus: Bus, quantum: float = QUANTUM) -> None:
  state, after, registry = State(quantum=quantum), "0-0", Registry(bus)
  while True:
    if await bus.event_log_size() < state.n:  # the log shrank, so Redis was flushed
      state, after = State(quantum=quantum), "0-0"
    events = await bus.read_events(after, count=BATCH, block_ms=TICK_MS)
    for e in events:
      state = fold(state, e)
    after = events[-1].id if events else after
    if len(events) == BATCH:
      continue
    if lost := sorted(set(state.nodes) - set(await bus.live_nodes())):  # their heartbeats ran out
      for node in lost:
        await bus.append_event(Ev.NODE_LOST, node=node)
      continue  # decide once they are folded, so nothing is placed on them
    for session in await bus.expired_sessions():  # their clients stopped heartbeating
      await bus.end_session(session)
    # checkpoints whose TTL ran out. The node daemons' sweeps delete their files
    await registry.unregister_checkpoints(*await registry.expired_checkpoints())
    for decision, body in decide(state, time.time()):  # folded when it is read back on the next pass
      if decision == Ev.ASSIGNED:
        await bus.assign_run_to_worker(body["run"], body["worker"])
      else:
        await bus.append_event(decision, **body)
    if events:
      await bus.r.set(key("state"), json.dumps(dataclasses.asdict(state), sort_keys=True))


if __name__ == "__main__":
  asyncio.run(run(Bus()))
