"""Where pools run and whose turn it is on each GPU, decided from one log: tinyrl:events.

The gateway, node daemons and workers append facts to the log, what happened. The scheduler folds every entry into a
State and appends what decide says back to the same log: placed and unplaced, granted and revoked. A decision changes
the state only once it is folded from the log, so the state is always fold(log), and replaying the log rebuilds every
state the scheduler saw. Each decision names the rule in RULES that made it.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field

from tinyrl import schedule

Key = tuple[str, int]  # (node, GPU index)

RULES = {
  "place.least_loaded_node": "a pool with runs, or a warm sampler, goes to the node with the fewest pools per GPU, on "
  "the GPUs schedule.choose_gpus picks and with schedule.settings",
  "place.emptied": "a pool with no runs and no warm sampler is unplaced, which stops its process",
  "turn.granted": "a pool that asked gets its GPUs once it is first in line on each and fits next to their holders",
  "turn.idle_holder_yields": "a holder that found no commands gives its GPUs back as soon as someone waits for one",
  "turn.quantum_expired": "a busy holder gives its GPUs back QUANTUM seconds into its turn if someone waits for one",
}


@dataclass(frozen=True)
class Event:
  id: str  # the stream id, <ms>-<n>
  t: float  # log time: the ms part of the id, in seconds
  kind: str
  body: dict


@dataclass(frozen=True)
class Policy:
  quantum: float = 30.0  # seconds of its turn a busy holder keeps while others wait. Engine boot does not count


@dataclass
class Pool:
  runs: tuple[str, ...] = ()
  warm: bool = False  # a sampler kept running without runs, so a run's first sample doesn't wait for its engine
  exclusive: bool = False
  node: str | None = None  # where it is placed, None until it is
  gpus: tuple[int, ...] = ()
  memory: int = 0  # bytes it uses on each of its GPUs while it holds them, 0 if unknown
  env: dict[str, str] = field(default_factory=dict)  # settings its process starts with
  running: bool = False  # its process started and has not exited
  asked: int = 0  # its place in line, the entry count when it asked. 0 while not waiting
  since: float | None = None  # when its turn started, pushed back by engine boot. None while it holds no GPU
  revoked: bool = False  # told to give its GPUs back
  idle: bool = False  # its worker found no commands since its last batch


@dataclass
class State:
  policy: Policy = Policy()
  n: int = 0  # entries folded
  nodes: dict[str, tuple[int, int]] = field(default_factory=dict)  # node -> (GPUs, bytes per GPU)
  pools: dict[str, Pool] = field(default_factory=dict)


def keys(p: Pool) -> list[Key]:
  return [(p.node, i) for i in p.gpus] if p.running else []


def holders(s: State, k: Key) -> list[str]:
  return sorted(name for name, p in s.pools.items() if p.since is not None and k in keys(p))


def first_in_line(s: State, k: Key) -> str | None:
  waiting = [(p.asked, name) for name, p in s.pools.items() if p.asked and k in keys(p)]
  return min(waiting)[1] if waiting else None


def fits(s: State, k: Key, name: str) -> bool:
  """A pool joins a GPU's holders if nobody holds it, or if its memory and every holder's are known and fit together."""
  memory, gpu = [s.pools[n].memory for n in (*holders(s, k), name)], s.nodes[k[0]][1]
  return len(memory) == 1 or (gpu > 0 and all(memory) and sum(memory) <= gpu)


def stop(p: Pool) -> None:
  """Its process is gone, and its turn with it."""
  p.running, p.asked, p.since, p.revoked, p.idle = False, 0, None, False, False


# ***** fold: what an entry changes *****


def fold(state: State, e: Event) -> State:
  """The state after entry e. Never changes the state it is given, and ignores entries it doesn't know."""
  s = copy.deepcopy(state)
  s.n += 1
  b = e.body
  p = s.pools.get(b.get("pool", ""))
  match e.kind:
    case "run_created":
      q = s.pools.setdefault(b["pool"], Pool())
      q.runs, q.exclusive = tuple(sorted({*q.runs, b["run"]})), q.exclusive or b.get("exclusive", False)
    case "run_deleted":
      for name, q in s.pools.items():
        q.runs = tuple(r for r in q.runs if r != b["run"])
        q.warm = q.warm and name not in (f"sampler:fft:{b['run']}", f"sampler:lora:{b['run']}")
    case "sampler_warmed":
      q = s.pools.setdefault(b["pool"], Pool())
      q.warm, q.exclusive = True, q.exclusive or b.get("exclusive", False)
    case "node_up":  # a node daemon (re)started, and its processes died with the old one
      s.nodes[b["node"]] = (b["gpus"], b.get("memory", 0))
      for q in s.pools.values():
        if q.node == b["node"]:
          stop(q)
    case "node_lost":
      s.nodes.pop(b["node"], None)
      for q in s.pools.values():
        if q.node == b["node"]:
          stop(q)
          q.node, q.gpus = None, ()
    case "placed" if p:  # a new place means a new process, so whatever ran before has lost its turn
      stop(p)
      p.node, p.gpus, p.memory, p.env = b["node"], tuple(b["gpus"]), b["memory"], b["env"]
    case "unplaced" if p:
      del s.pools[b["pool"]]
    case "started" if p and p.node == b["node"]:
      stop(p)
      p.running = True
    case "exited" if p:
      stop(p)
    case "wants_gpu" if p and p.running:
      p.idle = False
      if p.since is None and not p.asked:
        p.asked = s.n
    case "batch_done" if p:
      p.idle = False
      if p.since is not None:
        p.since += b.get("boot", 0.0)
    case "idle" if p:
      p.idle = True
    case "released" if p:
      p.since, p.revoked = None, False
    case "granted" if p:
      p.asked, p.since = 0, e.t
    case "revoked" if p:
      p.revoked = True
  return s


# ***** decide: what the scheduler appends next *****


def decide(state: State, now: float) -> list[tuple[str, dict]]:
  """The decisions to append now, in order. Each is folded into a copy before the next is chosen, so one call never
  grants two pools that don't fit together, or places two pools blind to each other. Once they are folded from the
  log, deciding again returns none."""
  s, out = state, []
  while d := next_decision(s, now):
    s = fold(s, Event("", now, *d))
    out.append(d)
  return out


def next_decision(s: State, now: float) -> tuple[str, dict] | None:
  for name, p in sorted(s.pools.items()):
    if not (p.runs or p.warm):
      return "unplaced", {"pool": name, "rule": "place.emptied"}
  unplaced = [name for name, p in s.pools.items() if p.node is None]
  if unplaced and s.nodes:
    exclusive = {name for name, p in s.pools.items() if p.exclusive}
    return "placed", place(s, schedule.start_order(unplaced, exclusive)[0], exclusive)
  for _, name in sorted((p.asked, name) for name, p in s.pools.items() if p.asked):
    if all(first_in_line(s, k) == name and fits(s, k, name) for k in keys(s.pools[name])):
      return "granted", {"pool": name, "rule": "turn.granted"}
  for name, p in sorted(s.pools.items()):
    if p.since is None or p.revoked or not any(first_in_line(s, k) for k in keys(p)):
      continue
    if p.idle:
      return "revoked", {"pool": name, "rule": "turn.idle_holder_yields"}
    if now >= p.since + s.policy.quantum:
      return "revoked", {"pool": name, "rule": "turn.quantum_expired"}
  return None


def place(s: State, name: str, exclusive: set[str]) -> dict:
  load = {n: sum(p.node == n for p in s.pools.values()) / gpus for n, (gpus, _) in s.nodes.items()}
  node = min(sorted(load), key=load.get)
  gpus, gpu_bytes = s.nodes[node]
  running = [schedule.RunningPool(q, list(p.gpus)) for q, p in sorted(s.pools.items()) if p.node == node]
  chosen = schedule.choose_gpus(name, 1, gpus, running, exclusive)
  env = schedule.settings(name, gpus, gpu_bytes // 2**20, exclusive)
  memory = int(float(env.get("TINYRL_GPU_MEM", 0)) * gpu_bytes)
  body = {"pool": name, "node": node, "gpus": chosen.choice, "memory": memory, "env": env}
  return {**body, "rule": "place.least_loaded_node", "gpu_rule": chosen.rule}


def check(s: State) -> list[str]:
  """Every broken invariant, as one message each. Empty means the state is sound."""
  bad = []
  for node, (gpus, gpu) in sorted(s.nodes.items()):
    for k in [(node, i) for i in range(gpus)]:
      held = holders(s, k)
      memory = [s.pools[n].memory for n in held]
      if len(held) > 1 and not (gpu and all(memory) and sum(memory) <= gpu):
        bad.append(f"{k} has holders that do not fit together")
  for name, p in sorted(s.pools.items()):
    if p.node is not None and p.node not in s.nodes:
      bad.append(f"{name} is placed on {p.node}, which is not up")
    if p.running and p.node is None:
      bad.append(f"{name} runs but is not placed")
    if p.asked and p.since is not None:
      bad.append(f"{name} holds its GPUs and waits for them")
    if p.revoked and p.since is None:
      bad.append(f"{name} is revoked but holds nothing")
  return bad


def to_json(s: State) -> str:
  return json.dumps(dataclasses.asdict(s), sort_keys=True)
