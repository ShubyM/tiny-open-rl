"""GPU turns, decided from one log: tinyrl:events.

Node daemons and workers append facts to the log, what happened. The scheduler folds every entry into a State and
appends what decide says back to the same log: granted and revoked. A decision changes the state only once it is folded
from the log, so the state is always fold(log), and replaying the log rebuilds every state the scheduler saw. Each
decision names the rule in RULES that made it.

Pools are placed by schedule.py. The node daemon reports each process it starts in a started entry.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field

Key = tuple[str, int]  # (node, GPU index)

RULES = {
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
  node: str
  gpus: tuple[int, ...]
  memory: int = 0  # bytes it uses on each of its GPUs while it holds them, 0 if unknown
  asked: int = 0  # its place in line, the entry count when it asked. 0 while not waiting
  since: float | None = None  # when its turn started, pushed back by engine boot. None while it holds no GPU
  revoked: bool = False  # told to give its GPUs back
  idle: bool = False  # its worker found no commands since its last batch


@dataclass
class State:
  policy: Policy = Policy()
  n: int = 0  # entries folded
  gpus: dict[Key, int] = field(default_factory=dict)  # GPU -> bytes, 0 if unknown
  pools: dict[str, Pool] = field(default_factory=dict)  # pools with a running process


def keys(p: Pool) -> list[Key]:
  return [(p.node, i) for i in p.gpus]


def holders(s: State, k: Key) -> list[str]:
  return sorted(name for name, p in s.pools.items() if p.since is not None and k in keys(p))


def first_in_line(s: State, k: Key) -> str | None:
  waiting = [(p.asked, name) for name, p in s.pools.items() if p.asked and k in keys(p)]
  return min(waiting)[1] if waiting else None


def fits(s: State, k: Key, name: str) -> bool:
  """A pool joins a GPU's holders if nobody holds it, or if its memory and every holder's are known and fit together."""
  memory = [s.pools[n].memory for n in (*holders(s, k), name)]
  return len(memory) == 1 or (s.gpus.get(k, 0) > 0 and all(memory) and sum(memory) <= s.gpus[k])


# ***** fold: what an entry changes *****


def fold(state: State, e: Event) -> State:
  """The state after entry e. Never changes the state it is given, and ignores entries it doesn't know."""
  s = copy.deepcopy(state)
  s.n += 1
  b = e.body
  p = s.pools.get(b.get("pool", ""))
  match e.kind:
    case "node_up":  # a node daemon (re)started, and its processes died with the old one
      s.pools = {name: q for name, q in s.pools.items() if q.node != b["node"]}
      s.gpus.update({(b["node"], i): b.get("memory", 0) for i in range(b["gpus"])})
    case "started":
      s.pools[b["pool"]] = Pool(b["node"], tuple(b["gpus"]), b.get("memory", 0))
    case "exited":
      s.pools.pop(b["pool"], None)
    case "wants_gpu" if p:
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
  grants two pools that don't fit together. Once they are folded from the log, deciding again returns none."""
  s, out = state, []
  while d := next_decision(s, now):
    s = fold(s, Event("", now, *d))
    out.append(d)
  return out


def next_decision(s: State, now: float) -> tuple[str, dict] | None:
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


def check(s: State) -> list[str]:
  """Every broken invariant, as one message each. Empty means the state is sound."""
  bad = []
  for k in sorted(s.gpus):
    held = holders(s, k)
    memory = [s.pools[n].memory for n in held]
    if len(held) > 1 and not (s.gpus[k] and all(memory) and sum(memory) <= s.gpus[k]):
      bad.append(f"{k} has holders that do not fit together")
  for name, p in sorted(s.pools.items()):
    if p.asked and p.since is not None:
      bad.append(f"{name} holds its GPUs and waits for them")
    if p.revoked and p.since is None:
      bad.append(f"{name} is revoked but holds nothing")
  return bad


def to_json(s: State) -> str:
  d = dataclasses.asdict(s)
  d["gpus"] = [[*k, memory] for k, memory in sorted(s.gpus.items())]
  return json.dumps(d, sort_keys=True)
