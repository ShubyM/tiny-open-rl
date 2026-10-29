import collections
import dataclasses
from collections.abc import Iterable
from typing import Any, Generic, Literal, TypeVar

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class Pool:
  name: str
  role: Literal["trainer", "sampler"]
  shared: bool
  target: str

  @classmethod
  def parse(cls, name: str) -> "Pool":
    if name.startswith("lora:"):
      body = name.removeprefix("lora:")
      target = body.rsplit(":", 1)[0] if ":" in body and body.rsplit(":", 1)[1].isdigit() else body
      return cls(name, "trainer", True, target)
    if name.startswith(("sampler:fft:", "sampler:lora:")):
      return cls(name, "sampler", False, name.split(":", 2)[2])
    if name.startswith("sampler:"):
      return cls(name, "sampler", True, name.removeprefix("sampler:"))
    return cls(name, "trainer", False, name)

  def pairs_with(self, other: "Pool") -> bool:
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
  exclusive_overlap: int
  in_reserved: bool
  out_of_zone: bool
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


def place_pools(
  active_pools: set[str], nodes: dict[str, int], placed: dict[str, str]
) -> tuple[dict[str, str], dict[str, Decision[str]]]:
  next_placed = {p: n for p, n in placed.items() if p in active_pools}
  decisions = {}

  for pool in sorted(active_pools):
    if next_placed.get(pool) not in nodes and nodes:
      load = collections.Counter(v for v in next_placed.values() if v in nodes)
      ratios = {n: round(load[n] / g, 4) for n, g in nodes.items()}
      winner = min(nodes, key=lambda n: load[n] / nodes[n])
      next_placed[pool] = winner
      rule = "reassigned_from_dead_node" if placed.get(pool) else "least_loaded_node"
      decisions[pool] = Decision(
        winner, rule, {"previous": placed.get(pool), "pools_per_gpu": ratios, "node_gpus": dict(nodes)}
      )

  return next_placed, decisions


def start_order(pools: Iterable[str], exclusive: set[str] = frozenset()) -> list[str]:
  return sorted(
    pools,
    key=lambda name: (name not in exclusive, not Pool.parse(name).shared, Pool.parse(name).role == "sampler", name),
  )


def gpu_zones(gpus: int, width: int) -> Zones:
  half = (gpus // 2 // width) * width
  total = (gpus // width) * width
  if half >= width:
    return Zones((0, half), (half, total), True)
  else:
    return Zones((0, gpus), (0, gpus), False)


def choose_gpus(
  pool: str, width: int, gpus: int, running: list[RunningPool], exclusive: set[str] = frozenset()
) -> Decision[list[int]]:
  if not (1 <= width <= gpus):
    raise ValueError(f"invalid width {width} for {gpus} gpus")

  target = Pool.parse(pool)
  zones = gpu_zones(gpus, width)
  is_excl = pool in exclusive
  z0, z1 = zones.trainer if target.role == "trainer" else zones.sampler
  blocks = (z1 - z0) // width

  excl_in_role = sum(1 for r in running if r.pool in exclusive and Pool.parse(r.pool).role == target.role)
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

  parsed_running = [(r, Pool.parse(r.pool), set(r.gpus)) for r in running]
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
      exclusive_overlap=over(lambda r, _: r.pool in exclusive),
      in_reserved=in_reserved,
      out_of_zone=out_of_zone,
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
