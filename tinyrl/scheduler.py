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
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

from tinyrl.bus import Bus, Event, key
from tinyrl.helpers import getenv

# ***** pools: which pool a run joins, and which GPUs and settings a pool gets *****

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class PoolName:
  name: str
  role: Literal["trainer", "sampler"]
  shared: bool
  target: str

  @classmethod
  def parse(cls, name: str) -> PoolName:
    if name.startswith("lora:"):
      body = name.removeprefix("lora:")
      target = body.rsplit(":", 1)[0] if ":" in body and body.rsplit(":", 1)[1].isdigit() else body
      return cls(name, "trainer", True, target)
    if name.startswith(("sampler:fft:", "sampler:lora:")):
      return cls(name, "sampler", False, name.split(":", 2)[2])
    if name.startswith("sampler:"):
      return cls(name, "sampler", True, name.removeprefix("sampler:"))
    return cls(name, "trainer", False, name)

  def pairs_with(self, other: PoolName) -> bool:
    return self.role != other.role and self.shared == other.shared and self.target == other.target


@dataclasses.dataclass(frozen=True)
class Decision(Generic[T]):
  choice: T
  rule: str
  detail: dict[str, Any]

  def as_dict(self) -> dict[str, Any]:
    return {"choice": self.choice, "rule": self.rule, **self.detail}


@dataclasses.dataclass(frozen=True)
class ShardLoad:
  pool: str
  util: float | None
  placed: int
  runs: int


@dataclasses.dataclass(frozen=True)
class RunningPool:
  pool: str
  gpus: list[int]


@dataclasses.dataclass(frozen=True)
class Zones:
  trainer: tuple[int, int]
  sampler: tuple[int, int]
  partitioned: bool


@dataclasses.dataclass(frozen=True, order=True)
class BlockRank:
  paired_overlap: int
  out_of_zone: bool
  exclusive_overlap: int
  in_reserved: bool
  out_of_subzone: bool
  cross_role_overlap: int
  shared_overlap: int
  gpu_load: int
  tie: int


def choose_pool(
  run: str,
  base_model: str,
  rank: int,
  exclusive: bool,
  shards: list[ShardLoad],
  gpus: int,
  target_util: float = 0.7,
  new_run_util: float = 0.04,
) -> Decision[str]:
  if rank == 0:
    return Decision(run, "dedicated_fft", {"base": base_model, "rank": rank})
  if exclusive:
    return Decision(run, "dedicated_exclusive", {"base": base_model, "rank": rank})

  ordered = sorted(shards, key=lambda s: int(s.pool.rsplit(":", 1)[1]))
  measured = [(s.util, s.runs - s.placed) for s in ordered if s.util is not None and s.runs > s.placed]
  per_run = sum(u for u, _ in measured) / sum(n for _, n in measured) if measured else new_run_util

  scores = [(s.util or 0.0) + s.placed * per_run for s in ordered]
  detail = {
    "base": base_model,
    "rank": rank,
    "per_run": per_run,
    "target_util": target_util,
    "gpus": gpus,
    "scores": {s.pool: round(sc, 4) for s, sc in zip(ordered, scores)},
  }

  if scores and min(scores) < target_util:
    return Decision(ordered[scores.index(min(scores))].pool, "reuse_shard_under_target", detail)
  elif len(ordered) >= max(1, gpus):
    return Decision(ordered[scores.index(min(scores))].pool, "reuse_shard_at_gpu_cap", detail)
  else:
    return Decision(f"lora:{base_model}:{len(ordered)}", "open_new_shard", detail)


def choose_sampler_pool(
  base_model: str, lora: bool, train_run: str | None = None, exclusive: bool = False
) -> Decision[str]:
  detail = {"base": base_model, "train_run": train_run, "exclusive": exclusive}
  if not lora:
    if exclusive or not base_model:
      return Decision(f"sampler:fft:{train_run}", "dedicated_fft_sampler", detail)
    return Decision(f"sampler:fft:{base_model}", "shared_fft_sampler", detail)
  if exclusive:
    return Decision(f"sampler:lora:{train_run}", "dedicated_exclusive_sampler", detail)
  return Decision(f"sampler:{base_model}", "shared_lora_sampler", detail)


def start_order(pools: Iterable[str], exclusive: set[str] = frozenset()) -> list[str]:
  return sorted(
    pools,
    key=lambda name: (
      name not in exclusive,
      not PoolName.parse(name).shared,
      PoolName.parse(name).role == "sampler",
      name,
    ),
  )


def gpu_zones(gpus: int, width: int) -> Zones:
  half = (gpus // 2 // width) * width
  total = (gpus // width) * width
  if half >= width:
    return Zones((0, half), (half, total), True)
  else:
    return Zones((0, gpus), (0, gpus), False)


def settings(pool: str, gpus: int, gpu_mib: int, exclusive: set[str] = frozenset()) -> dict[str, str]:
  """The environment a pool's process starts with, on a node with `gpus` GPUs of `gpu_mib` MiB each."""
  env = {}
  if PoolName.parse(pool).role != "sampler":
    return env
  if gpu_zones(gpus, 1).partitioned:
    env["TINYRL_WARM_SAMPLER"] = "1"
  if gpu_mib >= 65536 and pool.startswith("sampler:fft:") and pool not in exclusive:
    env.update(TINYRL_GPU_MEM="0.42", TINYRL_KV_BYTES=str(16 * 1024**3))
  elif gpu_mib >= 65536:
    env["TINYRL_KV_BYTES"] = str(24 * 1024**3)
  return env


def choose_gpus(
  pool: str, width: int, gpus: int, running: list[RunningPool], exclusive: set[str] = frozenset()
) -> Decision[list[int]]:
  if not (1 <= width <= gpus):
    raise ValueError(f"invalid width {width} for {gpus} gpus")

  target = PoolName.parse(pool)
  zones = gpu_zones(gpus, width)
  is_excl = pool in exclusive
  z0, z1 = zones.trainer if target.role == "trainer" else zones.sampler
  blocks = (z1 - z0) // width

  excl_in_role = sum(1 for r in running if r.pool in exclusive and PoolName.parse(r.pool).role == target.role)
  if is_excl and all(r.pool != pool for r in running):
    excl_in_role += 1

  if not zones.partitioned or blocks <= 1:
    excl_range = (z1, z1)
    sz = (z0, z1)
  elif blocks == 2:
    excl_range = (z1 - width, z1) if excl_in_role > 0 else (z1, z1)
    sz = (z0, z0 + width) if (target.shared and not is_excl) else (z0 + width, z1)
  elif blocks == 3:
    excl_range = (z1 - width, z1) if excl_in_role > 0 else (z1, z1)
    if is_excl:
      sz = (z1 - width, z1)
    elif target.shared:
      sz = (z0, z0 + 2 * width)
    else:
      sz = (z0 + 2 * width, z1)
  else:
    eb = min(blocks, max(1, excl_in_role))
    excl_range = (z1 - eb * width, z1)
    rem = (z1 - width) - z0
    shared_end = z0 + max(width, ((2 * rem) // 3 // width) * width)
    if is_excl:
      sz = excl_range
    elif target.shared:
      sz = (z0, shared_end)
    else:
      sz = (shared_end, z1 - width)

  parsed_running = [(r, PoolName.parse(r.pool), set(r.gpus)) for r in running]
  ranks = []

  for first in range(0, gpus - width + 1, width):
    block = set(range(first, first + width))

    def over(pred, b=block):
      return sum(len(b & rg) for r, rp, rg in parsed_running if pred(r, rp))

    in_reserved = (not is_excl) and (first < excl_range[1] and first + width > excl_range[0])
    in_zone = z0 <= first and first + width <= z1
    out_of_zone = not in_zone
    out_of_subzone = not (sz[0] <= first and first + width <= sz[1])
    tie = -first if is_excl else first

    rank = BlockRank(
      paired_overlap=over(lambda _, rp: target.pairs_with(rp)),
      out_of_zone=out_of_zone,
      exclusive_overlap=over(lambda r, _: r.pool in exclusive),
      in_reserved=in_reserved,
      out_of_subzone=out_of_subzone,
      cross_role_overlap=over(lambda _, rp: rp.role != target.role),
      shared_overlap=over(lambda _, rp: rp.shared),
      gpu_load=over(lambda _, __: True),
      tie=tie,
    )
    ranks.append((rank, first))

  winner_rank, winner_first = min(ranks, key=lambda x: x[0])

  if (
    is_excl and excl_range[1] > excl_range[0] and not winner_rank.out_of_subzone and winner_rank.exclusive_overlap == 0
  ):
    rule = "reserved_exclusive"
  elif (
    not winner_rank.in_reserved
    and not winner_rank.out_of_zone
    and not winner_rank.out_of_subzone
    and winner_rank.gpu_load == 0
  ):
    rule = "solo_in_zone"
  elif not winner_rank.in_reserved and not winner_rank.out_of_zone and not winner_rank.out_of_subzone:
    rule = "timesliced_in_subzone" if not target.shared else "least_loaded_in_zone"
  else:
    rule = "overflow_least_penalty"

  detail = {
    "role": target.role,
    "shared": target.shared,
    "exclusive": is_excl,
    "width": width,
    "zones": dataclasses.asdict(zones),
    "subzone": list(sz),
    "reserved": list(excl_range),
    "candidates": {
      f"{f}..{f + width - 1}" if width > 1 else str(f): {**dataclasses.asdict(r), "first": f} for r, f in ranks
    },
  }
  return Decision(list(range(winner_first, winner_first + width)), rule, detail)


# ***** state: what the log says *****

Key = tuple[str, int]  # (node, GPU index)

RULES = {
  "place.least_loaded_node": "a pool with runs, or a warm sampler, goes to the node with the fewest pools per GPU, on "
  "the GPUs choose_gpus picks and with settings",
  "place.emptied": "a pool with no runs and no warm sampler is unplaced, which stops its process",
  "turn.granted": "a pool that asked gets its GPUs once it is first in line on each and fits next to their holders",
  "turn.idle_holder_yields": "a holder that found no commands gives its GPUs back as soon as someone waits for one",
  "turn.quantum_expired": "a busy holder gives its GPUs back QUANTUM seconds into its turn if someone waits for one",
}


@dataclass(frozen=True)
class Policy:
  quantum: float = getenv("QUANTUM", 30.0)  # seconds of its turn a busy holder keeps while others wait, boot aside


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
    return "placed", place(s, start_order(unplaced, exclusive)[0], exclusive)
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
  running = [RunningPool(q, list(p.gpus)) for q, p in sorted(s.pools.items()) if p.node == node]
  chosen = choose_gpus(name, 1, gpus, running, exclusive)
  env = settings(name, gpus, gpu_bytes // 2**20, exclusive)
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


# ***** the loop *****

TICK_MS = getenv("TICK_MS", 100)  # how long it waits for an entry before it checks quanta and heartbeats again
BATCH = 10_000  # entries read at once. It decides only once a read comes back short, so never from half the log


async def run(bus: Bus, policy: Policy = Policy()) -> None:
  state, after = State(policy=policy), "0-0"
  while True:
    if await bus.log_size() < state.n:  # Redis was flushed: start over
      state, after = State(policy=policy), "0-0"
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
      await bus.r.set(key("state"), to_json(state))


if __name__ == "__main__":
  asyncio.run(run(Bus()))
