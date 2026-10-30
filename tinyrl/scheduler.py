"""Runs to workers, workers to GPUs, and GPU turns. The same fold drives the scheduler and the node daemons.

A run is a training model or a sampling session, with its own command stream. A worker is one process on fixed GPUs
serving one or more runs. Every fact and decision is an entry in one log, so the state is a fold over it.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
from dataclasses import dataclass, field

from tinyrl.bus import Bus, Event, key
from tinyrl.helpers import getenv

QUANTUM = getenv("QUANTUM", 30.0)  # seconds a busy holder keeps its GPU while a worker that doesn't fit waits
SAMPLER_MEM = getenv("SAMPLER_MEM", 0.4)  # share of a GPU a vLLM worker holds while it has a turn
TICK_MS = getenv("TICK_MS", 100)
BATCH = 10_000
EPS = 1e-6
Key = tuple[str, int]


@dataclass
class Run:
  role: str = "trainer"
  base: str = ""
  lora: bool = True
  exclusive: bool = False
  train_run: str = ""  # the training run a sampling session samples
  worker: str = ""


@dataclass
class Worker:
  role: str = "trainer"
  base: str = ""
  lora: bool = True
  exclusive: bool = False
  train_run: str = ""
  node: str = ""
  gpus: tuple[int, ...] = ()
  mem: float = 1.0
  env: dict[str, str] = field(default_factory=dict)
  started: bool = False
  stopping: bool = False
  exited: bool = False
  asked: int = 0  # place in line, 0 if not waiting
  since: float | None = None  # when its turn started, None if it holds none
  turn: str = ""
  revoked: bool = False
  idle: bool = False


@dataclass
class State:
  quantum: float = QUANTUM
  n: int = 0
  now: float = 0.0
  nodes: dict[str, tuple[int, int]] = field(default_factory=dict)  # node -> (GPUs, bytes per GPU)
  runs: dict[str, Run] = field(default_factory=dict)
  workers: dict[str, Worker] = field(default_factory=dict)


def keys(w: Worker) -> set[Key]:
  return {(w.node, g) for g in w.gpus}


def live(s: State, w: Worker) -> bool:
  return not w.stopping and not w.exited and w.node in s.nodes


def members(s: State, worker: str) -> list[str]:
  return [r for r, x in s.runs.items() if x.worker == worker]


def used(s: State, k: Key, holding: bool = False) -> float:
  return sum(
    w.mem for w in s.workers.values() if not w.exited and k in keys(w) and (not holding or w.since is not None)
  )


# ***** policy *****


def kind(x: Run | Worker) -> tuple:
  return x.role, x.base, x.lora, x.exclusive and x.train_run


def choose_worker(s: State, r: Run) -> str | None:
  """The worker a new run joins: the one serving its kind of run. A full fine-tune or exclusive training run is
  alone, and an exclusive run's sampling sessions share only with each other."""
  if r.role == "trainer" and (r.exclusive or not r.lora):
    return None
  return next((name for name, w in s.workers.items() if live(s, w) and kind(w) == kind(r)), None)


def need(role: str) -> float:
  return SAMPLER_MEM if role == "sampler" else 1.0


def place(s: State, mem: float) -> Key | None:
  """The least used GPU a new worker fits on, else the least used one, where it takes turns."""
  options = [
    (used(s, (node, g)) + mem > 1 + EPS, used(s, (node, g)), node, g)
    for node, (n, _) in sorted(s.nodes.items())
    for g in range(n)
  ]
  return min(options)[2:] if options else None


def fits(s: State, w: Worker) -> bool:
  return all(used(s, k, holding=True) + w.mem <= 1 + EPS for k in keys(w))


# ***** fold *****


def fold(state: State, e: Event) -> State:
  """The state after entry e. Unknown entries change nothing, and worker IDs are never reused."""
  s = copy.deepcopy(state)
  s.n += 1
  s.now = max(s.now, e.t)
  b = e.body
  r, w = s.runs.get(b.get("run", "")), s.workers.get(b.get("worker", ""))
  match e.kind:
    case "node_up":
      s.nodes[b["node"]] = (b["gpus"], b.get("memory", 0))
    case "node_lost":
      s.nodes.pop(b["node"], None)
      for x in s.workers.values():
        if x.node == b["node"]:
          x.stopping = True
    case "run_created" if b["run"] not in s.runs:
      s.runs[b["run"]] = Run(
        b.get("role", "trainer"),
        b.get("base", ""),
        b.get("rank", 1) > 0,
        b.get("exclusive", False),
        b.get("train_run") or "",
        b.get("pool") or "",
      )
    case "run_deleted":
      s.runs.pop(b["run"], None)
    case "worker_created" if b["worker"] not in s.workers:
      fields = {k: v for k, v in b.items() if k in Worker.__dataclass_fields__}
      s.workers[b["worker"]] = Worker(**fields | {"gpus": tuple(b["gpus"])})
      if r:
        r.worker = b["worker"]
    case "run_assigned" if r and not r.worker:
      r.worker = b["worker"]
    case "run_released" if r and r.worker == b.get("source"):
      r.worker = ""
    case "worker_stopped" if w:
      w.stopping = True
    case "started" if w and not w.exited and b.get("node", w.node) == w.node:
      w.started = True
    case "exited" | "worker_fenced" if w and b.get("node", w.node) == w.node:
      w.exited, w.stopping, w.since, w.asked, w.revoked = True, True, None, 0, False
    case "wants_gpu" if w and w.started and not w.stopping and not w.exited:
      w.idle = False
      if w.since is None and not w.asked:
        w.asked = s.n
    case "granted" if w and w.asked and not w.exited:
      w.since, w.turn, w.asked, w.revoked, w.idle = e.t, b["turn"], 0, False, False
    case "revoked" if w and w.since is not None and b.get("turn") == w.turn:
      w.revoked = True
    case "released" if w and b.get("turn") == w.turn:
      w.since, w.revoked = None, False
    case "idle" if w and b.get("turn") == w.turn:
      w.idle = True
  return s


# ***** decide *****


def next_decision(s: State, now: float) -> tuple[str, dict] | None:
  for run_id, r in s.runs.items():
    if r.worker:
      continue
    if worker := choose_worker(s, r):
      return "run_assigned", {"run": run_id, "worker": worker}
    mem = need(r.role)
    if at := place(s, mem):
      env = {"TINYRL_GPU_MEM": str(mem)} if r.role == "sampler" else {}
      body = dict(worker=f"w{s.n + 1}", run=run_id, role=r.role, base=r.base, lora=r.lora, exclusive=r.exclusive)
      body["train_run"] = r.train_run
      return "worker_created", body | dict(node=at[0], gpus=[at[1]], mem=mem, env=env)
  for name, w in s.workers.items():
    if not w.stopping and not w.exited and not members(s, name):
      return "worker_stopped", {"worker": name}
  waiting = sorted(
    (x for x in s.workers.items() if x[1].asked and not x[1].stopping and not x[1].exited), key=lambda x: x[1].asked
  )
  blocked: set[Key] = set()  # first come first served on each GPU, so a big waiter isn't passed forever
  for name, w in waiting:
    if keys(w) & blocked:
      continue
    if fits(s, w):
      return "granted", {"worker": name, "turn": f"t{s.n + 1}"}
    blocked |= keys(w)
  for name, w in s.workers.items():
    if w.since is None or w.revoked:
      continue
    crowded = any(keys(w) & keys(x) and not fits(s, x) for _, x in waiting)
    if w.stopping or (crowded and (w.idle or now - w.since >= s.quantum)):
      return "revoked", {"worker": name, "turn": w.turn}
  return None


def decide(state: State, now: float) -> list[tuple[str, dict]]:
  """Every decision due now, each folded before the next is chosen. Tests and replays use it; the loop takes one."""
  s, out = state, []
  while d := next_decision(s, now):
    out.append(d)
    s = fold(s, Event("", now, *d))
  return out


async def run(bus: Bus, quantum: float = QUANTUM) -> None:
  state, after, saved = State(quantum=quantum), "0-0", -1
  while True:
    if await bus.log_size() < state.n:  # the log shrank, so Redis was flushed
      state, after = State(quantum=quantum), "0-0"
    events = await bus.events(after, count=BATCH, block_ms=TICK_MS)
    for e in events:
      state = fold(state, e)
    after = events[-1].id if events else after
    if len(events) == BATCH:
      continue
    if lost := sorted(set(state.nodes) - set(await bus.nodes())):
      for node in lost:
        await bus.emit("node_lost", node=node)
      continue
    fenced = False
    for name, w in state.workers.items():
      if w.node not in state.nodes and not w.exited:
        if fenced := await bus.fence_worker(name, after):
          break
      if w.exited and members(state, name):
        await bus.unassign_worker(name)
    if fenced:
      continue
    seconds, micros = await bus.r.time()
    if d := next_decision(state, max(state.now, seconds + micros / 1_000_000)):
      kind, body = d
      if kind in ("run_assigned", "worker_created"):
        await bus.assign(body.pop("run"), body.pop("worker"), after=after, kind=kind, **body)
      else:
        await bus.emit_decision(kind, after, **body)
    if state.n != saved:
      await bus.r.set(key("state"), json.dumps(dataclasses.asdict(state), sort_keys=True))
      saved = state.n


if __name__ == "__main__":
  asyncio.run(run(Bus()))
