"""The scheduler: which worker a run joins, where each worker runs, and whose turn it is on each GPU.

The gateway, node daemons and workers append facts to one log, tinyrl:events: what happened. python -m tinyrl.control.scheduler
folds every entry into a State and appends what decide says back to the same log: placed and unplaced, granted and
revoked. A decision changes the state only once it is folded from the log, so the state is always fold(log), and
replaying the log rebuilds every state the scheduler saw. Each decision names the rule in RULES that made it. Run
exactly one. A restarted scheduler folds the whole log again and carries on, since its earlier decisions are in it.

Everything here but run is a pure function. The gateway records what a run needs, choose_worker picks its worker, and
node daemons fold the log to learn what to run.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import time
from dataclasses import dataclass, field

from tinyrl.bus import Bus, Event, key
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry

# ***** workers *****


def choose_worker(s: State, run: Run) -> str | None:
  """The worker a new run joins: the one serving its kind of run. A full fine-tune or exclusive training run is alone,
  and an exclusive run's sampling sessions share only with each other."""
  k = run.kind
  if k.role == "trainer" and (not k.lora or k.exclusive):
    return None
  return next((name for name, w in s.workers.items() if w.kind == k), None)


def settings(k: Kind, gpus: int, gpu_bytes: int) -> dict[str, str]:
  """The environment a worker's process starts with on a node with this many GPUs of this size."""
  env = {}
  if k.role != "sampler":
    return env
  if gpus >= 2:  # samplers have GPUs of their own, so they start their engine before the first command
    env["TINYRL_WARM_SAMPLER"] = "1"
  if gpu_bytes >= 64 * 2**30 and not k.lora and not k.exclusive:
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

RULES = {
  "place.least_loaded_node": "a worker with runs goes to the node with the fewest workers per GPU, on choose's GPU",
  "place.near_its_trainer": "a full fine-tune's sampler goes to its trainer's node, where its weights are saved",
  "place.moved_aside": "a trainer moves to take turns with other trainers, so a sampler or exclusive worker gets a GPU",
  "place.emptied": "a worker with no runs is unplaced, which stops its process",
  "turn.granted": "a worker that asked gets its GPUs once it is first in line on each and fits next to their holders",
  "turn.idle_holder_yields": "a holder that found no commands gives its GPUs back as soon as someone waits for one",
  "turn.quantum_expired": "a busy holder gives its GPUs back QUANTUM seconds into its turn if someone waits for one",
}


@dataclass(frozen=True)
class Kind:
  """What runs a worker serves. Runs of one kind share a worker."""

  role: str = "trainer"
  base: str = ""
  lora: bool = False
  exclusive: bool = False
  train_run: str | None = None  # an exclusive run's sampling sessions name it, so they share only with each other


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


@dataclass
class State:
  quantum: float = QUANTUM
  n: int = 0  # entries folded
  nodes: dict[str, tuple[int, int]] = field(default_factory=dict)  # node -> (GPUs, bytes per GPU)
  workers: dict[str, Worker] = field(default_factory=dict)
  runs: dict[str, Run] = field(default_factory=dict)


def keys(w: Worker) -> tuple[Key, ...]:
  return w.on


def holders(s: State, k: Key) -> list[str]:
  return sorted(name for name, w in s.workers.items() if w.since is not None and k in keys(w))


def first_in_line(s: State, k: Key) -> str | None:
  waiting = [(w.asked, name) for name, w in s.workers.items() if w.asked and k in keys(w)]
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
  w.on, w.asked, w.since, w.revoked, w.idle = (), 0, None, False, False


# ***** fold *****


def fold(state: State, e: Event) -> State:
  """The state after entry e. Never changes the state it is given, and ignores entries it doesn't know."""
  s = copy.deepcopy(state)
  s.n += 1
  b = e.body
  w = s.workers.get(b.get("worker", ""))
  match e.kind:
    case "run_created":
      exclusive = b.get("exclusive", False)
      k = Kind(
        b.get("role", "trainer"),
        b.get("base", ""),
        b.get("rank", 0) > 0,
        exclusive,
        exclusive and b.get("train_run") or None,
      )
      s.runs[b["run"]] = Run(k, train_run=b.get("train_run"))
      if b.get("worker"):  # created in a worker already, as tests do
        join(s, b["run"], b["worker"])
    case "assigned" if b["run"] in s.runs:
      join(s, b["run"], b["worker"])
    case "run_deleted":  # an exclusive run's sampling sessions end with it
      s.runs = {r: x for r, x in s.runs.items() if r != b["run"] and x.kind.train_run != b["run"]}
    case "node_up":  # a node daemon (re)started, and its processes died with the old one
      s.nodes[b["node"]] = (b["gpus"], b.get("memory", 0))
      for x in s.workers.values():
        if any(node == b["node"] for node, _ in x.on):
          stop(x)
    case "node_lost":
      s.nodes.pop(b["node"], None)
      for x in s.workers.values():
        if any(node == b["node"] for node, _ in x.on):
          stop(x)
        if x.node == b["node"]:
          x.node, x.gpus = None, ()
    case "placed" if w:  # a process still running elsewhere keeps its GPUs and its turn until it exits
      w.node, w.gpus = b["node"], tuple(b["gpus"])
    case "unplaced" if w:
      w.node, w.gpus = None, ()
    case "started" if w and w.node == b["node"]:
      stop(w)
      w.on = tuple((w.node, i) for i in w.gpus)
    case "exited" if w:
      stop(w)
    case "wants_gpu" if w and w.on:
      w.idle = False
      if w.since is None and not w.asked:
        w.asked = s.n
    case "batch_done" if w:
      w.idle, w.busy = False, w.busy + b.get("seconds", 0.0)
      if w.since is not None:
        w.since += b.get("boot", 0.0)
    case "idle" if w:
      w.idle = True
    case "released" if w:
      w.since, w.revoked = None, False
    case "granted" if w:
      w.asked, w.since = 0, e.t
    case "revoked" if w:
      w.revoked = True
  s.workers = {name: x for name, x in s.workers.items() if x.node or x.on or members(s, name)}
  return s


# ***** decide *****


def decide(state: State, now: float) -> list[tuple[str, dict]]:
  """The decisions to append now, in order. Each is folded into a copy before the next is chosen, so one call never
  grants two workers that don't fit together, or places two workers blind to each other. Once they are folded from the
  log, deciding again returns none."""
  s, out = state, []
  while d := next_decision(s, now):
    s = fold(s, Event("", now, *d))
    out.append(d)
  return out


def next_decision(s: State, now: float) -> tuple[str, dict] | None:
  for run_id, r in sorted(s.runs.items()):
    if not r.worker:
      return "assigned", {"run": run_id, "worker": choose_worker(s, r) or f"w{s.n + 1}"}
  for name, w in sorted(s.workers.items()):
    if w.node and not members(s, name):
      return "unplaced", {"worker": name, "rule": "place.emptied"}
  unplaced = [name for name, w in s.workers.items() if w.node is None and members(s, name)]
  if unplaced and s.nodes:
    return "placed", place(s, min(unplaced, key=lambda name: (not s.workers[name].kind.exclusive, name)))
  for _, name in sorted((w.asked, name) for name, w in s.workers.items() if w.asked):
    if all(first_in_line(s, k) == name and fits(s, k, name) for k in keys(s.workers[name])):
      return "granted", {"worker": name, "rule": "turn.granted"}
  for name, w in sorted(s.workers.items()):
    if w.since is None or w.revoked or not any(first_in_line(s, k) for k in keys(w)):
      continue
    if w.idle:
      return "revoked", {"worker": name, "rule": "turn.idle_holder_yields"}
    if now >= w.since + s.quantum:
      return "revoked", {"worker": name, "rule": "turn.quantum_expired"}
  return None


def place(s: State, name: str) -> dict:
  """Where a worker goes: the node with the fewest workers per GPU, and the GPU choose picks there. When trainers have
  to move out of the way first, the decision is the move of one of them, and the worker goes once its GPU is clear."""
  load = {n: sum(w.node == n for w in s.workers.values()) / gpus for n, (gpus, _) in s.nodes.items()}
  me, trainers = s.workers[name], [s.runs.get(s.runs[r].train_run or "") for r in members(s, name)]
  near = {s.workers[t.worker].node for t in trainers if t and t.worker in s.workers}
  near = near if me.kind.role == "sampler" and not me.kind.lora else set()  # a LoRA adapter is small to fetch
  node = min(sorted(load), key=lambda n: (n not in near, load[n]))
  gpu, why, aside = choose(s, node, name)
  if aside:
    to, to_why, _ = choose(s, node, aside[0], avoid=gpu)
    return {
      "worker": aside[0],
      "node": node,
      "gpus": [to],
      "for": name,
      "rule": "place.moved_aside",
      "gpu_rule": to_why,
    }
  rule = "place.near_its_trainer" if node in near else "place.least_loaded_node"
  return {"worker": name, "node": node, "gpus": [gpu], "rule": rule, "gpu_rule": why}


def takes_turns(w: Worker) -> bool:
  """Whether a worker may share a GPU it doesn't fit on. A trainer parks and resumes in about a second. A sampler wakes
  its engine slowly, and an exclusive worker is alone by definition."""
  return w.kind.role == "trainer" and not w.kind.exclusive


def choose(s: State, node: str, name: str, avoid: int = -1) -> tuple[int, str, list[str]]:
  """The GPU on a node a worker goes to, why, and the trainers it moves out of the way. An exclusive worker is alone,
  and others stay off its GPU. A worker takes an empty GPU, or one its memory budget fits on next to the workers there,
  and a worker with a budget fills such a GPU before it takes an empty one. With neither, a worker that takes turns
  takes them with the least busy trainers. One that doesn't takes those trainers' GPU, and they move elsewhere."""
  gpus, gpu_bytes = s.nodes[node]
  me, memory = s.workers[name], budget(s, s.workers[name].kind, node)
  there = {
    g: [(n, w) for n, w in s.workers.items() if n != name and w.node == node and g in w.gpus] for g in range(gpus)
  }

  def fits_on(g: int) -> bool:
    others = [w for _, w in there[g]]
    if not others:
      return True
    return (
      not me.kind.exclusive
      and not any(w.kind.exclusive for w in others)
      and fit([budget(s, w.kind, node) for w in others] + [memory], gpu_bytes)
    )

  def trainers_only(g: int) -> bool:
    return bool(there[g]) and all(takes_turns(w) for _, w in there[g])

  options = [g for g in range(gpus) if g != avoid]
  if fitting := [g for g in options if fits_on(g)]:
    g = min(fitting, key=lambda g: (bool(there[g]) != bool(memory), g))
    return g, "fits_next_to_others" if there[g] else "empty_gpu", []
  busy = {g: sum(w.busy for _, w in there[g]) for g in options}
  if takes_turns(me) or len(options) == 1:  # with one GPU there is nowhere to move to
    g = min(options, key=lambda g: (not trainers_only(g), busy[g], g))
    return g, "takes_turns", []
  # as few moves as it can, LoRA trainers first, since a full fine-tune saves its weights and AdamW state far slower
  lora = {g: all(w.kind.lora for _, w in there[g]) for g in options}
  g = min(options, key=lambda g: (not trainers_only(g), len(there[g]), not lora[g], busy[g], g))
  return (g, "moves_trainers_aside", sorted(n for n, _ in there[g])) if trainers_only(g) else (g, "takes_turns", [])


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
        await bus.append_event("node_lost", node=node)
      continue  # decide once they are folded, so nothing is placed on them
    for session in await bus.expired_sessions():  # their clients stopped heartbeating
      await bus.end_session(session)
    # checkpoints whose TTL ran out. The node daemons' sweeps delete their files
    await registry.unregister_checkpoints(*await registry.expired_checkpoints())
    for decision, body in decide(state, time.time()):  # folded when it is read back on the next pass
      if decision == "assigned":
        await bus.assign_run_to_worker(body["run"], body["worker"])
      else:
        await bus.append_event(decision, **body)
    if events:
      await bus.r.set(key("state"), json.dumps(dataclasses.asdict(state), sort_keys=True))


if __name__ == "__main__":
  asyncio.run(run(Bus()))
