"""The scheduler: which pool a run joins, where each pool runs, and whose turn it is on each GPU.

The gateway, node daemons and workers append facts to one log, tinyrl:events: what happened. python -m tinyrl.scheduler
folds every entry into a State and appends what decide says back to the same log: placed and unplaced, granted and
revoked. A decision changes the state only once it is folded from the log, so the state is always fold(log), and
replaying the log rebuilds every state the scheduler saw. Each decision names the rule in RULES that made it. Run
exactly one. A restarted scheduler folds the whole log again and carries on, since its earlier decisions are in it.

Everything here but run is a pure function. The gateway calls choose_pool and choose_sampler_pool to pick a run's pool,
and node daemons fold the log to learn what to run.
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

# ***** pools *****


def parse_pool(pool: str) -> tuple[str, bool, str]:
  """A pool's role (trainer or sampler), whether it serves many runs, and the model it serves, read from its name."""
  if pool.startswith("lora:"):
    body = pool.removeprefix("lora:")
    return "trainer", True, body.rsplit(":", 1)[0] if ":" in body and body.rsplit(":", 1)[1].isdigit() else body
  if pool.startswith(("sampler:fft:", "sampler:lora:")):
    return "sampler", False, pool.split(":", 2)[2]
  if pool.startswith("sampler:"):
    return "sampler", True, pool.removeprefix("sampler:")
  return "trainer", False, pool


def choose_pool(
  run: str,
  base_model: str,
  rank: int,
  exclusive: bool,
  shards: list[tuple[str, float | None, int, int]],
  gpus: int,
  target_util: float = 0.7,
  new_run_util: float = 0.04,
) -> dict:
  """The pool a new training run joins, with the rule and numbers behind it. A shard is (pool, its last reported busy
  fraction or None, runs placed on it since, runs)."""
  if rank == 0:
    return {"choice": run, "rule": "dedicated_fft", "base": base_model, "rank": rank}
  if exclusive:
    return {"choice": run, "rule": "dedicated_exclusive", "base": base_model, "rank": rank}
  ordered = sorted(shards, key=lambda shard: int(shard[0].rsplit(":", 1)[1]))
  measured = [(util, runs - placed) for _, util, placed, runs in ordered if util is not None and runs > placed]
  per_run = sum(u for u, _ in measured) / sum(n for _, n in measured) if measured else new_run_util
  scores = [(util or 0.0) + placed * per_run for _, util, placed, _ in ordered]
  detail = {"base": base_model, "rank": rank, "per_run": per_run, "target_util": target_util, "gpus": gpus}
  detail["scores"] = {shard[0]: round(score, 4) for shard, score in zip(ordered, scores)}
  if scores and min(scores) < target_util:
    return {"choice": ordered[scores.index(min(scores))][0], "rule": "reuse_shard_under_target", **detail}
  if len(ordered) >= max(1, gpus):
    return {"choice": ordered[scores.index(min(scores))][0], "rule": "reuse_shard_at_gpu_cap", **detail}
  return {"choice": f"lora:{base_model}:{len(ordered)}", "rule": "open_new_shard", **detail}


def choose_sampler_pool(base_model: str, lora: bool, train_run: str | None = None, exclusive: bool = False) -> dict:
  detail = {"base": base_model, "train_run": train_run, "exclusive": exclusive}
  if not lora and (exclusive or not base_model):
    return {"choice": f"sampler:fft:{train_run}", "rule": "dedicated_fft_sampler", **detail}
  if not lora:
    return {"choice": f"sampler:fft:{base_model}", "rule": "shared_fft_sampler", **detail}
  if exclusive:
    return {"choice": f"sampler:lora:{train_run}", "rule": "dedicated_exclusive_sampler", **detail}
  return {"choice": f"sampler:{base_model}", "rule": "shared_lora_sampler", **detail}


def settings(pool: str, gpus: int, gpu_mib: int, exclusive: set[str] = frozenset()) -> dict[str, str]:
  env = {}
  if parse_pool(pool)[0] != "sampler":
    return env
  if gpus >= 2:  # samplers have GPUs of their own, so they start their engine before the first command
    env["TINYRL_WARM_SAMPLER"] = "1"
  if gpu_mib >= 65536 and pool.startswith("sampler:fft:") and pool not in exclusive:
    env.update(TINYRL_GPU_MEM="0.42", TINYRL_KV_BYTES=str(16 * 1024**3))
  elif gpu_mib >= 65536:
    env["TINYRL_KV_BYTES"] = str(24 * 1024**3)
  return env


def fit(memory: list[int], gpu_bytes: int) -> bool:
  """Whether pools with these memory budgets can share a GPU. Every budget has to be known, and 0 means unknown."""
  return len(memory) == 1 or (gpu_bytes > 0 and all(memory) and sum(memory) <= gpu_bytes)


# ***** state *****

Key = tuple[str, int]  # (node, GPU index)

QUANTUM = getenv("QUANTUM", 30.0)  # seconds a busy holder keeps its GPUs while others wait, engine boot aside

RULES = {
  "place.least_loaded_node": "a pool with runs, or a warm sampler, goes to the node with the fewest pools per GPU, on "
  "the GPU choose picks and with settings",
  "place.moved_aside": "a trainer moves to take turns with other trainers, so a sampler or exclusive pool gets its GPU",
  "place.emptied": "a pool with no runs and no warm sampler is unplaced, which stops its process",
  "turn.granted": "a pool that asked gets its GPUs once it is first in line on each and fits next to their holders",
  "turn.idle_holder_yields": "a holder that found no commands gives its GPUs back as soon as someone waits for one",
  "turn.quantum_expired": "a busy holder gives its GPUs back QUANTUM seconds into its turn if someone waits for one",
}


@dataclass
class Pool:
  runs: tuple[str, ...] = ()
  warm: bool = False  # a sampler kept running without runs, so a run's first sample doesn't wait for its engine
  exclusive: bool = False
  node: str | None = None  # None until it is placed
  gpus: tuple[int, ...] = ()
  memory: int = 0  # bytes a process placed now may use on each of its GPUs, 0 if unknown
  env: dict[str, str] = field(default_factory=dict)
  on: tuple[Key, ...] = ()  # the GPUs its process is on, until it exits, wherever the pool is placed now
  uses: int = 0  # bytes that process may use on each of them
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
  pools: dict[str, Pool] = field(default_factory=dict)


def keys(p: Pool) -> tuple[Key, ...]:
  return p.on


def holders(s: State, k: Key) -> list[str]:
  return sorted(name for name, p in s.pools.items() if p.since is not None and k in keys(p))


def first_in_line(s: State, k: Key) -> str | None:
  waiting = [(p.asked, name) for name, p in s.pools.items() if p.asked and k in keys(p)]
  return min(waiting)[1] if waiting else None


def fits(s: State, k: Key, name: str) -> bool:
  """A pool joins a GPU's holders if nobody holds it, or if its memory and every holder's are known and fit together."""
  return fit([s.pools[n].uses for n in (*holders(s, k), name)], s.nodes[k[0]][1])


def stop(p: Pool) -> None:
  """Its process is gone, and its turn with it."""
  p.on, p.uses, p.asked, p.since, p.revoked, p.idle = (), 0, 0, None, False, False


# ***** fold *****


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
        if any(node == b["node"] for node, _ in q.on):
          stop(q)
    case "node_lost":
      s.nodes.pop(b["node"], None)
      for q in s.pools.values():
        if any(node == b["node"] for node, _ in q.on):
          stop(q)
        if q.node == b["node"]:
          q.node, q.gpus = None, ()
    case "placed" if p:  # a process still running elsewhere keeps its GPUs and its turn until it exits
      p.node, p.gpus, p.memory, p.env = b["node"], tuple(b["gpus"]), b["memory"], b["env"]
    case "unplaced" if p:
      p.node, p.gpus = None, ()
    case "started" if p and p.node == b["node"]:
      stop(p)
      p.on, p.uses = tuple((p.node, i) for i in p.gpus), p.memory
    case "exited" if p:
      stop(p)
    case "wants_gpu" if p and p.on:
      p.idle = False
      if p.since is None and not p.asked:
        p.asked = s.n
    case "batch_done" if p:
      p.idle, p.busy = False, p.busy + b.get("seconds", 0.0)
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
  s.pools = {name: q for name, q in s.pools.items() if q.runs or q.warm or q.node or q.on}
  return s


# ***** decide *****


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
    if p.node and not (p.runs or p.warm):
      return "unplaced", {"pool": name, "rule": "place.emptied"}
  unplaced = [name for name, p in s.pools.items() if p.node is None and (p.runs or p.warm)]
  if unplaced and s.nodes:
    return "placed", place(s, min(unplaced, key=lambda name: (not s.pools[name].exclusive, name)))
  for _, name in sorted((p.asked, name) for name, p in s.pools.items() if p.asked):
    if all(first_in_line(s, k) == name and fits(s, k, name) for k in keys(s.pools[name])):
      return "granted", {"pool": name, "rule": "turn.granted"}
  for name, p in sorted(s.pools.items()):
    if p.since is None or p.revoked or not any(first_in_line(s, k) for k in keys(p)):
      continue
    if p.idle:
      return "revoked", {"pool": name, "rule": "turn.idle_holder_yields"}
    if now >= p.since + s.quantum:
      return "revoked", {"pool": name, "rule": "turn.quantum_expired"}
  return None


def place(s: State, name: str) -> dict:
  """Where a pool goes: the node with the fewest pools per GPU, and the GPU choose picks there. When trainers have to
  move out of the way first, the decision is the move of one of them, and the pool is placed once its GPU is clear."""
  load = {n: sum(p.node == n for p in s.pools.values()) / gpus for n, (gpus, _) in s.nodes.items()}
  node = min(sorted(load), key=load.get)
  gpus, gpu_bytes = s.nodes[node]
  env = settings(name, gpus, gpu_bytes // 2**20, {q for q, p in s.pools.items() if p.exclusive})
  memory = int(float(env.get("TINYRL_GPU_MEM", 0)) * gpu_bytes)
  gpu, why, aside = choose(s, node, name, memory)
  if aside:
    trainer = s.pools[aside[0]]
    to, to_why, _ = choose(s, node, aside[0], trainer.memory, avoid=gpu)
    body = {"pool": aside[0], "node": node, "gpus": [to], "memory": trainer.memory, "env": trainer.env, "for": name}
    return {**body, "rule": "place.moved_aside", "gpu_rule": to_why}
  body = {"pool": name, "node": node, "gpus": [gpu], "memory": memory, "env": env}
  return {**body, "rule": "place.least_loaded_node", "gpu_rule": why}


def takes_turns(name: str, p: Pool) -> bool:
  """Whether a pool may share a GPU it doesn't fit on. A trainer parks and resumes in about a second. A sampler wakes
  its engine slowly, and an exclusive pool is alone by definition."""
  return parse_pool(name)[0] == "trainer" and not p.exclusive


def choose(s: State, node: str, name: str, memory: int, avoid: int = -1) -> tuple[int, str, list[str]]:
  """The GPU on a node a pool goes to, why, and the trainers it moves out of the way. An exclusive pool is alone, and
  others stay off its GPU. A pool takes an empty GPU, or one its memory budget fits on next to the pools there, and a
  pool with a budget fills such a GPU before it takes an empty one. With neither, a pool that takes turns takes them
  with the least busy trainers. One that doesn't takes those trainers' GPU, and they move to take turns elsewhere."""
  gpus, gpu_bytes = s.nodes[node]
  me = s.pools[name]
  there = {g: [(q, p) for q, p in s.pools.items() if q != name and p.node == node and g in p.gpus] for g in range(gpus)}

  def fits_on(g: int) -> bool:
    others = [p for _, p in there[g]]
    if not others:
      return True
    return (
      not me.exclusive
      and not any(p.exclusive for p in others)
      and fit([p.memory for p in others] + [memory], gpu_bytes)
    )

  def trainers_only(g: int) -> bool:
    return bool(there[g]) and all(takes_turns(q, p) for q, p in there[g])

  options = [g for g in range(gpus) if g != avoid]
  if fitting := [g for g in options if fits_on(g)]:
    g = min(fitting, key=lambda g: (bool(there[g]) != bool(memory), g))
    return g, "fits_next_to_others" if there[g] else "empty_gpu", []
  busy = {g: sum(p.busy for _, p in there[g]) for g in options}
  if takes_turns(name, me) or len(options) == 1:  # with one GPU there is nowhere to move to
    g = min(options, key=lambda g: (not trainers_only(g), busy[g], g))
    return g, "takes_turns", []
  g = min(options, key=lambda g: (not trainers_only(g), len(there[g]), busy[g], g))  # as few moves as it can
  return (g, "moves_trainers_aside", sorted(q for q, _ in there[g])) if trainers_only(g) else (g, "takes_turns", [])


# ***** the loop *****

TICK_MS = getenv("TICK_MS", 100)  # how long it waits for an entry before it checks quanta and heartbeats again
BATCH = 10_000  # entries read at once. It decides only once a read comes back short, so never from half the log


async def run(bus: Bus, quantum: float = QUANTUM) -> None:
  state, after = State(quantum=quantum), "0-0"
  while True:
    if await bus.log_size() < state.n:  # the log shrank, so Redis was flushed
      state, after = State(quantum=quantum), "0-0"
    events = await bus.events(after, count=BATCH, block_ms=TICK_MS)
    for e in events:
      state = fold(state, e)
    after = events[-1].id if events else after
    if len(events) == BATCH:
      continue
    if lost := sorted(set(state.nodes) - set(await bus.nodes())):  # their heartbeats ran out
      for node in lost:
        await bus.emit("node_lost", node=node)
      continue  # decide once they are folded, so nothing is placed on them
    for kind, body in decide(state, time.time()):
      await bus.emit(kind, **body)  # folded when it is read back on the next pass
    if events:
      await bus.r.set(key("state"), json.dumps(dataclasses.asdict(state), sort_keys=True))


if __name__ == "__main__":
  asyncio.run(run(Bus()))
