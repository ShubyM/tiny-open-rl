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
  reserved: tuple[int, int]
  partitioned: bool


@dataclasses.dataclass(frozen=True, order=True)
class BlockRank:
  in_reserved: bool
  paired_overlap: int
  out_of_zone: bool
  out_of_subzone: bool
  exclusive_overlap: int
  cross_role_overlap: int
  gpu_load: int
  shared_overlap: int
  first: int


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


def gpu_zones(gpus: int, width: int, running: list[RunningPool], pool: str, exclusive: set[str] = frozenset()) -> Zones:
  et = sum(len(r.gpus) for r in running if r.pool in exclusive and Pool.parse(r.pool).role == "trainer")
  es = sum(len(r.gpus) for r in running if r.pool in exclusive and Pool.parse(r.pool).role == "sampler")

  if pool in exclusive and all(r.pool != pool for r in running):
    target = Pool.parse(pool)
    if target.role == "trainer":
      et += width
    else:
      es += width

  has_non_excl = any(r.pool not in exclusive for r in running) or (pool not in exclusive)
  cap_limit = max(0, gpus - width) if has_non_excl else gpus
  excl_cap = (min(cap_limit, et + max(es, et)) // width) * width
  res0 = gpus - excl_cap
  reserved = (res0, gpus)
  free = res0

  if free > 4 * width:
    t_cap = max(width, (free // 2 // width) * width)
    return Zones((0, t_cap), (t_cap, res0), reserved, True)
  else:
    return Zones((0, res0), (0, res0), reserved, False)


def subzone(zones: Zones, width: int, target: Pool, is_exclusive: bool) -> tuple[int, int]:
  if is_exclusive:
    return zones.reserved if zones.reserved[1] > zones.reserved[0] else (0, zones.reserved[1])
  z0, z1 = zones.trainer if target.role == "trainer" else zones.sampler
  if zones.partitioned and (z1 - z0) >= 3 * width:
    split = z0 + max(width, ((2 * (z1 - z0)) // 3 // width) * width)
    if not target.shared:
      return (split, z1)
  return (z0, z1)


def choose_gpus(
  pool: str, width: int, gpus: int, running: list[RunningPool], exclusive: set[str] = frozenset()
) -> Decision[list[int]]:
  if not (1 <= width <= gpus):
    raise ValueError(f"invalid width {width} for {gpus} gpus")

  target = Pool.parse(pool)
  zones = gpu_zones(gpus, width, running, pool, exclusive)
  is_excl = pool in exclusive
  sz = subzone(zones, width, target, is_excl)
  res0 = zones.reserved[0]
  excl_cap = zones.reserved[1] - res0

  parsed_running = [(r, Pool.parse(r.pool), set(r.gpus)) for r in running]

  ranks = []
  for first in range(0, gpus - width + 1, width):
    block = set(range(first, first + width))

    def over(pred, b=block):
      return sum(len(b & rg) for r, rp, rg in parsed_running if pred(r, rp))

    if is_excl:
      in_zone = (first >= res0) if excl_cap > 0 else True
    elif target.role == "trainer":
      in_zone = zones.trainer[0] <= first and first + width <= zones.trainer[1]
    else:
      in_zone = zones.sampler[0] <= first and first + width <= zones.sampler[1]

    rank = BlockRank(
      in_reserved=(first + width > res0) if not is_excl else False,
      paired_overlap=over(lambda _, rp: target.pairs_with(rp)),
      out_of_zone=not in_zone,
      out_of_subzone=not (sz[0] <= first and first + width <= sz[1]),
      exclusive_overlap=over(lambda r, _: r.pool in exclusive),
      cross_role_overlap=over(lambda _, rp: rp.role != target.role),
      gpu_load=over(lambda _, __: True),
      shared_overlap=over(lambda _, rp: rp.shared),
      first=first,
    )
    ranks.append(rank)

  winner = min(ranks)

  if is_excl and excl_cap > 0 and winner.first >= res0:
    rule = "reserved_exclusive"
  elif not winner.in_reserved and not winner.out_of_zone and not winner.out_of_subzone and winner.gpu_load == 0:
    rule = "solo_in_zone"
  elif not winner.in_reserved and not winner.out_of_zone and not winner.out_of_subzone:
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
    "candidates": {
      f"{r.first}..{r.first + width - 1}" if width > 1 else str(r.first): dataclasses.asdict(r) for r in ranks
    },
  }
  return Decision(list(range(winner.first, winner.first + width)), rule, detail)
