"""Fleet scheduler: assigns jobs to workers, workers to physical GPU bundles on nodes, bounded budgets, and storage preparation.

The scheduler is a pure-state loop (`from_table -> decide -> write_table`) fenced by a single-writer Redis lease.
Nodes report physical GPU release, worker phases, per-job cumulative usage, and local version/transfer state; the
scheduler never reuses a physical GPU until release is observed, settles fair service across logical job accounts
(trainer + sampler sharing one account), serializes wide-job reservations, caps exclusive reservations, demotes
coldest host-parked trainers under host-RAM pressure, and populates restore/prefetch inputs for nodes to execute.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import itertools
import os
import socket
from dataclasses import dataclass, field

from tinyrl.bus import Bus, Ev, Role
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry
from tinyrl.storage.store import parse_version

TICK_MS, BUDGET_S = getenv("TICK_MS", 100), float(getenv("BUDGET_S", 10.0))
EXCLUSIVE_CAP, HOST_RAM_HIGH = float(getenv("EXCLUSIVE_CAP", 0.5)), float(getenv("HOST_RAM_HIGH", 0.8))
PREFETCH_HYSTERESIS, PEER_BPS, REMOTE_BPS = 0.15, 1e9, 3e8


# ***** worker specs & memory budgets *****


@dataclass(frozen=True)
class WorkerSpec:
  """What a worker process loads and needs: role, base model, LoRA vs full FT, exclusive flag, and GPU width."""

  role: Role = Role.TRAINER
  base: str = ""
  lora: bool = False
  exclusive: bool = False
  train_run: str | None = None
  gpus: int = 1


def settings(spec: WorkerSpec, gpus: int, gpu_bytes: int) -> dict[str, str]:
  """The environment a worker's process starts with on a node with this many GPUs of this size."""
  if spec.role != Role.SAMPLER:
    return {}
  env = {"TINYRL_WARM_SAMPLER": "1"} if gpus >= 2 else {}
  if gpu_bytes >= 64 * 2**30 and not spec.lora and not spec.exclusive and spec.gpus == 1:
    env.update(TINYRL_GPU_MEM="0.42", TINYRL_KV_BYTES=str(16 * 1024**3))
  elif gpu_bytes >= 64 * 2**30:
    env["TINYRL_KV_BYTES"] = str(24 * 1024**3)
  return env


def is_movable(spec: WorkerSpec, node_gpus: int) -> bool:
  """Single-GPU LoRA trainers bind CUDA_VISIBLE_DEVICES to all GPUs and can resume on any GPU without restart."""
  return spec.role == Role.TRAINER and spec.lora and spec.gpus == 1 and node_gpus > 1


def memory_budget(s: Assignment, spec: WorkerSpec, node: str | None) -> int:
  """Bytes a worker's process may use on each GPU on a node, 0 if unlimited (takes a GPU alone)."""
  ni = s.nodes.get(node or "")
  return int(float(settings(spec, ni.gpus, ni.memory).get("TINYRL_GPU_MEM", 0)) * ni.memory) if ni else 0


def fits(memory: list[int], gpu_bytes: int) -> bool:
  """Whether workers with these memory budgets can share a GPU. Every budget must be known (nonzero)."""
  return len(memory) == 1 or (gpu_bytes > 0 and all(memory) and sum(memory) <= gpu_bytes)


# ***** observations, jobs, workers, and cluster assignment *****


@dataclass
class JobObs:
  gen: int = 0
  usage: float = 0.0
  tokens: int = 0
  completed_id: str = ""
  state: str = "active"


@dataclass
class WorkerObs:
  rev: int = 0
  pid: int | None = None
  owner: str = ""
  gpus: tuple[int, ...] = ()
  gpus_held: tuple[int, ...] = ()
  gpu_released: bool = True
  phase: str = "idle"
  host_ram_bytes: int = 0
  jobs: dict[str, JobObs] = field(default_factory=dict)


@dataclass
class NodeInfo:
  gpus: int
  memory: int = 0
  host_ram: int = 0
  host_ram_used: int = 0
  disk_free: int = 0
  gpu_class: str = ""
  peer: str = ""
  workers: dict[str, WorkerObs] = field(default_factory=dict)
  versions: dict[str, dict] = field(default_factory=dict)


@dataclass
class Job:
  """A training run or sampling session with authoritative generation, queue readiness, and recovery pointers."""

  spec: WorkerSpec
  worker: str = ""
  train_run: str | None = None
  gen: int = 0
  weight: float = 1.0
  queued: int = 0
  in_flight: int = 0
  restore_uri: str = ""
  restore_version: str = ""
  restore_boundary: str = ""
  durable_uri: str = ""
  durable_version: str = ""
  durable_boundary: str = ""
  last_mutating_id: str = ""
  rollback_required: bool = False
  model_path: str | None = None
  model_version: str = ""
  gpu_class: str = ""

  @property
  def ready(self) -> bool:
    return (self.queued + self.in_flight) > 0


@dataclass
class Worker:
  """One fleet-assigned worker process: node, physical GPU bundle, intent, budgets, job generations, and data inputs."""

  spec: WorkerSpec = WorkerSpec()
  node: str | None = None
  gpus: tuple[int, ...] = ()
  rev: int = 0
  grant_rev: int = 0
  intent: str = "park"
  budgets: dict[str, float] = field(default_factory=dict)
  grant_settled: dict[str, float] = field(default_factory=dict)
  generations: dict[str, int] = field(default_factory=dict)
  drain_jobs: dict[str, str] = field(default_factory=dict)
  restore_refs: dict[str, dict] = field(default_factory=dict)
  prefetch: list[dict] = field(default_factory=list)
  last_active_rev: int = 0
  reserve_node: str | None = None


@dataclass
class Assignment:
  """The fleet assignment table plus observed node state, persisted fair service, and version locations."""

  workers_made: int = 0
  table_rev: int = 0
  nodes: dict[str, NodeInfo] = field(default_factory=dict)
  workers: dict[str, Worker] = field(default_factory=dict)
  jobs: dict[str, Job] = field(default_factory=dict)
  service: dict[str, float] = field(default_factory=dict)
  service_obs: dict[str, float] = field(default_factory=dict)
  service_ready: set[str] = field(default_factory=set)
  versions: dict[str, dict] = field(default_factory=dict)
  exclusive_cap: float = EXCLUSIVE_CAP


def worker_to_dict(w: Worker) -> dict:
  d = dataclasses.asdict(w)
  d["gpus"] = list(w.gpus)
  return d


def worker_from_dict(d: dict) -> Worker:
  spec = WorkerSpec(**(d["spec"] | {"role": Role(d["spec"]["role"])}))
  dr = d.get("drain_jobs") or {}
  return Worker(
    spec=spec,
    node=d.get("node"),
    gpus=tuple(d.get("gpus") or ()),
    rev=int(d.get("rev", 0)),
    grant_rev=int(d.get("grant_rev", 0)),
    intent=str(d.get("intent", "park")),
    budgets={k: float(v) for k, v in (d.get("budgets") or {}).items()},
    grant_settled={k: float(v) for k, v in (d.get("grant_settled") or {}).items()},
    generations={k: int(v) for k, v in (d.get("generations") or {}).items()},
    drain_jobs=dict(dr) if isinstance(dr, dict) else {j: "" for j in dr},
    restore_refs=dict(d.get("restore_refs") or {}),
    prefetch=list(d.get("prefetch") or []),
    last_active_rev=int(d.get("last_active_rev", 0)),
    reserve_node=d.get("reserve_node"),
  )


def _obs_from_dict(d: dict) -> WorkerObs:
  jobs = {
    jid: JobObs(
      int(jd.get("gen", 0)),
      float(jd.get("usage", 0.0)),
      int(jd.get("tokens", 0)),
      str(jd.get("completed_id", "")),
      str(jd.get("state", "active")),
    )
    for jid, jd in (d.get("jobs") or {}).items()
  }
  gpus, held = tuple(d.get("gpus") or ()), tuple(d.get("gpus_held") or ())
  return WorkerObs(
    int(d.get("rev", 0)),
    d.get("pid"),
    str(d.get("owner", "")),
    gpus,
    held,
    bool(d.get("gpu_released", not held)),
    str(d.get("phase", "running" if held else "idle")),
    int(d.get("host_ram_bytes", 0)),
    jobs,
  )


def from_table(table: dict) -> Assignment:
  """Build an Assignment snapshot from Redis table contents."""
  nodes: dict[str, NodeInfo] = {}
  for n, v in table.get("nodes", {}).items():
    if isinstance(v, NodeInfo):
      nodes[n] = v
    elif isinstance(v, tuple):
      nodes[n] = NodeInfo(gpus=v[0], memory=v[1])
    else:
      nodes[n] = NodeInfo(
        int(v["gpus"]),
        int(v.get("memory", 0)),
        int(v.get("host_ram", 0)),
        int(v.get("host_ram_used", 0)),
        int(v.get("disk_free", 0)),
        str(v.get("gpu_class", "")),
        str(v.get("peer", "")),
        {wn: _obs_from_dict(wd) for wn, wd in (v.get("workers") or {}).items()},
        dict(v.get("versions") or {}),
      )

  s = Assignment(
    workers_made=int(table.get("workers_made", 0)),
    table_rev=int(table.get("table_rev", 0)),
    nodes=nodes,
    service={k: float(v) for k, v in (table.get("service") or {}).items()},
    service_obs={k: float(v) for k, v in (table.get("service_obs") or {}).items()},
    service_ready=set(table.get("service_ready") or ()),
    versions=dict(table.get("versions") or {}),
  )
  meta_map, queues, ckpt_vers = table.get("meta") or {}, table.get("queues") or {}, table.get("ckpt_versions") or {}
  for job_id, (rec, worker) in table.get("runs", {}).items():
    role, base, lora = Role(rec.get("role", "trainer")), rec.get("base", ""), rec.get("rank", 0) > 0
    exclusive, train_run = bool(rec.get("exclusive", False)), rec.get("train_run")
    spec = WorkerSpec(role, base, lora, exclusive, exclusive and train_run or None, int(rec.get("gpus", 1)))
    m, q, mp = meta_map.get(job_id) or {}, queues.get(job_id) or {}, rec.get("model_path")
    s.jobs[job_id] = Job(
      spec=spec,
      worker=worker or "",
      train_run=train_run,
      gen=int(m.get("gen", 0)),
      weight=max(1e-6, float(rec.get("weight", 1.0) or 1.0)),
      queued=int(q.get("queued", 0)),
      in_flight=int(q.get("in_flight", 0)),
      restore_uri=str(m.get("restore_uri", "")),
      restore_version=str(m.get("restore_version", "")),
      restore_boundary=str(m.get("restore_boundary", "")),
      durable_uri=str(m.get("durable_uri", "")),
      durable_version=str(m.get("durable_version", "")),
      durable_boundary=str(m.get("durable_boundary", "")),
      last_mutating_id=str(m.get("last_mutating_id", "")),
      rollback_required=bool(m.get("rollback_required", False)),
      model_path=mp,
      model_version=parse_version(ckpt_vers.get(mp, ""))[0] if mp else "",
      gpu_class=str((rec.get("profile") or {}).get("gpu_class", "")),
    )

  s.jobs = {r: x for r, x in s.jobs.items() if not x.spec.train_run or x.spec.train_run in s.jobs}
  for name, w in table.get("workers", {}).items():
    s.workers[name] = worker_from_dict(w) if isinstance(w, dict) else copy.deepcopy(w)
  for x in s.jobs.values():
    if x.worker:
      s.workers.setdefault(x.worker, Worker(x.spec))
  return s


def jobs_of(s: Assignment, worker: str) -> list[str]:
  return [r for r, x in s.jobs.items() if x.worker == worker]


def active_jobs_of(s: Assignment, worker: str) -> list[str]:
  draining = w.drain_jobs if (w := s.workers.get(worker)) else {}
  return [r for r, x in s.jobs.items() if x.worker == worker and r not in draining]


def job_account(s: Assignment, job_id: str) -> str:
  return (j.train_run or job_id) if (j := s.jobs.get(job_id)) else job_id


def account_weight(s: Assignment, acct: str) -> float:
  if acct in s.jobs:
    return max(1e-6, s.jobs[acct].weight)
  return max((j.weight for jid, j in s.jobs.items() if job_account(s, jid) == acct), default=1.0)


def worker_obs(s: Assignment, name: str) -> WorkerObs | None:
  w = s.workers.get(name)
  return s.nodes[w.node].workers.get(name) if (w and w.node and w.node in s.nodes) else None


def existing_worker(s: Assignment, job: Job, exclude: str = "") -> str | None:
  """The non-draining worker a new job joins: same spec, except full-FT and exclusive trainers stay alone."""
  if job.spec.role == Role.TRAINER and (not job.spec.lora or job.spec.exclusive):
    return None
  return next(
    (n for n, w in sorted(s.workers.items()) if n != exclude and w.intent != "drain" and w.spec == job.spec), None
  )


def _new_worker(s: Assignment, spec: WorkerSpec) -> str:
  s.workers_made = next(n for n in itertools.count(s.workers_made + 1) if f"w{n}" not in s.workers)
  name = f"w{s.workers_made}"
  s.workers[name] = Worker(spec)
  return name


def _assign_job(s: Assignment, job_id: str, worker: str) -> None:
  job = s.jobs[job_id]
  job.worker, job.gen = worker, job.gen or 1
  s.workers.setdefault(worker, Worker(job.spec)).generations[job_id] = job.gen


def _reset_worker(s: Assignment, name: str, rule: str, out: list[tuple[Ev, dict]]) -> None:
  w = s.workers[name]
  for jid in jobs_of(s, name):
    s.jobs[jid].gen += 1
  w.node, w.gpus, w.intent, w.grant_rev = None, (), "park", 0
  w.budgets, w.grant_settled, w.drain_jobs, w.reserve_node = {}, {}, {}, None
  w.generations = {j: s.jobs[j].gen for j in jobs_of(s, name)}
  w.rev += 1
  out.append((Ev.UNPLACED, {"worker": name, "rule": rule}))


def gpus_still_held(s: Assignment, name: str) -> tuple[int, ...]:
  """Physical GPUs currently occupied or not yet confirmed released by worker `name`."""
  w = s.workers.get(name)
  if w is None or not w.node or w.node not in s.nodes:
    return ()
  if (obs := worker_obs(s, name)) is not None:
    if obs.gpus_held:
      return obs.gpus_held
    if not obs.gpu_released:
      return obs.gpus or w.gpus
    if w.intent == "run" and w.gpus:
      return w.gpus
    if w.intent in ("park", "drain") and w.grant_rev > 0 and obs.rev <= w.grant_rev:
      return obs.gpus or w.gpus
    return ()
  return w.gpus if (w.intent == "run" or (w.intent in ("park", "drain") and w.grant_rev > 0)) else ()


def needed_versions(s: Assignment, worker: str) -> list[tuple[str, str, str, list[str]]]:
  """Exact (job_id, uri, ver_id, peers) inputs needed to prepare/restore jobs assigned to `worker`."""
  out: list[tuple[str, str, str, list[str]]] = []
  for jid in active_jobs_of(s, worker):
    j = s.jobs[jid]
    if j.spec.role == Role.SAMPLER:
      uri, ver_ref = (j.model_path or "", j.model_version) if (j.model_path and j.model_version) else ("", "")
    elif j.restore_version and not j.rollback_required:
      uri, ver_ref = j.restore_uri, j.restore_version
    else:
      uri, ver_ref = (j.durable_uri, j.durable_version) if j.durable_version else ("", "")
    ver_id, inline_peer = parse_version(ver_ref)
    if not ver_id or not uri:
      continue
    locs = (s.versions.get(ver_id) or {}).get("locations") or {}
    peers = [inline_peer] if inline_peer else []
    for loc_node, peer_addr in locs.items():
      if peer_addr and peer_addr not in peers and (loc_node in s.nodes or ":" in loc_node):
        peers.append(peer_addr)
    out.append((jid, uri, ver_id, peers))
  return out


def _job_exclusive_account(s: Assignment, name: str) -> str | None:
  w = s.workers[name]
  if not w.spec.exclusive:
    return None
  return w.spec.train_run or (jobs_of(s, name)[0] if jobs_of(s, name) else name)


def _exclusive_gpu_limit(s: Assignment) -> int:
  total = sum(ni.gpus for ni in s.nodes.values())
  return total if (total <= 1 or s.exclusive_cap >= 1.0) else int(total * s.exclusive_cap)


def _placed_exclusive_gpus(s: Assignment) -> tuple[int, dict[str, str]]:
  """Total GPUs reserved by placed exclusive workers, plus exclusive trainer account -> placed node."""
  used, trainer_node = 0, {}
  for name, w in s.workers.items():
    if w.node is None or w.node not in s.nodes or w.intent == "drain" or not w.spec.exclusive:
      continue
    used += w.spec.gpus
    if w.spec.role == Role.TRAINER:
      for jid in jobs_of(s, name):
        trainer_node[jid] = w.node
  return used, trainer_node


def _node_free_gpus_for(s: Assignment, node: str, spec: WorkerSpec, exclude: str = "") -> int:
  """Count GPUs on `node` where `spec` could run immediately without displacing current occupants/exclusive slots."""
  ni, mem, free = s.nodes[node], memory_budget(s, spec, node), 0
  for g in range(ni.gpus):
    holders = [
      w
      for wn, w in s.workers.items()
      if wn != exclude
      and w.node == node
      and (g in gpus_still_held(s, wn) or (w.spec.exclusive and w.intent != "drain" and g in w.gpus))
    ]
    if not holders or (
      not spec.exclusive
      and not any(h.spec.exclusive for h in holders)
      and all(h.intent == "run" for h in holders)
      and fits([memory_budget(s, h.spec, node) for h in holders] + [mem], ni.memory)
    ):
      free += 1
  return free


def _estimate_state_ready(s: Assignment, node: str, needs: list[tuple[str, str, str, list[str]]], lora: bool) -> float:
  ni, total_s = s.nodes[node], 0.0
  for _, _, ver_id, peers in needs:
    vobs, vrec = ni.versions.get(ver_id), s.versions.get(ver_id) or {}
    locs = vrec.get("locations") or {}
    if (vobs and vobs.get("ready")) or node in locs:
      continue
    size = int((vobs or {}).get("bytes") or vrec.get("size_bytes") or (5_000_000 if lora else 1_000_000_000))
    bps = PEER_BPS if peers else REMOTE_BPS
    if vobs and float(vobs.get("progress", 0.0)) > 0.0:
      total_s += (size * max(0.0, 1.0 - float(vobs["progress"])) / bps) * (1.0 - PREFETCH_HYSTERESIS)
    else:
      total_s += 1e6 if (not vrec.get("durable", True) and not peers and locs and node not in locs) else (size / bps)
  return total_s


def _eligible_nodes(s: Assignment, name: str) -> list[str]:
  w = s.workers[name]
  req_classes = {s.jobs[j].gpu_class for j in jobs_of(s, name) if s.jobs[j].gpu_class}
  my_acct = _job_exclusive_account(s, name)
  nodes = []
  for n, ni in sorted(s.nodes.items()):
    if ni.gpus < w.spec.gpus or (req_classes and ni.gpu_class and ni.gpu_class not in req_classes):
      continue
    if w.spec.exclusive:
      other_excl = sum(
        ow.spec.gpus
        for on, ow in s.workers.items()
        if on != name
        and ow.node == n
        and ow.intent != "drain"
        and ow.spec.exclusive
        and _job_exclusive_account(s, on) != my_acct
      )
      if ni.gpus - other_excl < w.spec.gpus:
        continue
    nodes.append(n)
  return nodes


def place_worker(s: Assignment, name: str) -> dict | None:
  """Choose the best eligible node for `name` using storage readiness, GPU availability, and exclusive caps."""
  w, eligible = s.workers[name], _eligible_nodes(s, name)
  if not eligible:
    return None
  if w.spec.exclusive:
    used_excl, trainer_nodes = _placed_exclusive_gpus(s)
    if used_excl + w.spec.gpus > _exclusive_gpu_limit(s):
      if w.spec.role == Role.SAMPLER and (tn := trainer_nodes.get(w.spec.train_run or "")) in eligible:
        return {"worker": name, "node": tn, "rule": "place.exclusive_paired"}
      return None

  needs = needed_versions(s, name)

  def score(n: str) -> tuple[float, float, str]:
    ni = s.nodes[n]
    free = _node_free_gpus_for(s, n, w.spec, exclude=name)
    busy = sum(x.spec.gpus for x in s.workers.values() if x.node == n and x.intent != "drain")
    gpu_avail = 0.0 if free >= w.spec.gpus else max(0.5, (busy / ni.gpus) * BUDGET_S)
    state_ready = _estimate_state_ready(s, n, needs, w.spec.lora)
    load = sum(x.node == n and x.intent != "drain" for x in s.workers.values()) / ni.gpus
    return (max(gpu_avail, state_ready) + (0.0 if free >= w.spec.gpus else 0.01), load, n)

  best = min(eligible, key=score)
  rule = (
    "place.storage_local" if needs and _estimate_state_ready(s, best, needs, w.spec.lora) == 0.0 else "place.least_cost"
  )
  return {"worker": name, "node": best, "rule": rule}


def _sync_restore_and_prefetch(s: Assignment) -> bool:
  """Update restore_refs and prefetch lists on every placed non-draining worker; return True if anything changed."""
  changed = False
  for name, w in sorted(s.workers.items()):
    if w.node is None or w.node not in s.nodes or w.intent == "drain":
      if w.restore_refs or w.prefetch:
        w.restore_refs, w.prefetch, changed = {}, [], True
      continue
    ni, refs, prefetch, seen = s.nodes[w.node], {}, [], set()
    for jid, uri, ver_id, peers in needed_versions(s, name):
      refs[jid] = {"uri": uri, "version": ver_id, "peers": peers}
      vobs, locs = ni.versions.get(ver_id) or {}, (s.versions.get(ver_id) or {}).get("locations") or {}
      if not vobs.get("ready") and w.node not in locs and ver_id not in seen:
        seen.add(ver_id)
        prefetch.append({"uri": uri, "version": ver_id, "peers": peers, "job": jid})
    if refs != w.restore_refs or prefetch != w.prefetch:
      w.restore_refs, w.prefetch, changed = refs, prefetch, True
  return changed


def _allocate_node_gpus(s: Assignment, node_name: str, out: list[tuple[Ev, dict]]) -> None:
  """Allocate physical GPUs and bounded budgets on one node with fair service and wide-job anti-starvation."""
  ni = s.nodes[node_name]
  local = [n for n, w in sorted(s.workers.items()) if w.node == node_name and w.intent != "drain"]
  if not local:
    return

  def has_work(n: str) -> bool:
    return any(s.jobs[j].ready for j in active_jobs_of(s, n))

  def is_preparing(n: str) -> bool:
    return (obs := worker_obs(s, n)) is not None and obs.phase == "preparing"

  def warm_only(n: str) -> bool:
    w = s.workers[n]
    return (
      not has_work(n)
      and w.spec.role == Role.SAMPLER
      and ni.gpus >= 2
      and w.grant_rev == 0
      and bool(active_jobs_of(s, n))
    )

  def over_budget(n: str) -> bool:
    w = s.workers[n]
    return w.intent == "run" and sum(w.grant_settled.values()) >= (sum(w.budgets.values()) or BUDGET_S * w.spec.gpus)

  reserved_budget: dict[str, float] = {}
  for w in s.workers.values():
    if w.intent == "run":
      for jid, bud in w.budgets.items():
        acct = job_account(s, jid)
        reserved_budget[acct] = reserved_budget.get(acct, 0.0) + max(0.0, bud - w.grant_settled.get(jid, 0.0))

  def worker_priority(n: str) -> tuple[bool, bool, float, str]:
    jobs = [j for j in active_jobs_of(s, n) if s.jobs[j].ready] or active_jobs_of(s, n)
    scores = [
      (s.service.get(a, 0.0) + reserved_budget.get(a, 0.0)) / account_weight(s, a)
      for j in jobs
      for a in (job_account(s, j),)
    ]
    return (not has_work(n), not s.workers[n].spec.exclusive, min(scores, default=0.0), n)

  occupied: dict[int, list[str]] = {g: [] for g in range(ni.gpus)}
  for n, w in s.workers.items():
    if w.node == node_name:
      for g in gpus_still_held(s, n):
        if g in occupied:
          occupied[g].append(n)

  excl_reserved_by: dict[int, str] = {}
  for n in local:
    w = s.workers[n]
    if w.spec.exclusive and w.gpus:
      acct = _job_exclusive_account(s, n) or n
      for g in w.gpus:
        if g in occupied:
          excl_reserved_by.setdefault(g, acct)
  for n in local:
    w = s.workers[n]
    if w.spec.exclusive and not w.gpus:
      acct = _job_exclusive_account(s, n) or n
      owned = sorted(g for g, a in excl_reserved_by.items() if a == acct)
      if len(owned) >= w.spec.gpus:
        w.gpus = tuple(owned[: w.spec.gpus])
      else:
        avail_excl = sorted(
          (g for g in range(ni.gpus) if g not in excl_reserved_by or excl_reserved_by[g] == acct),
          key=lambda g: (excl_reserved_by.get(g) != acct, bool(occupied[g]), g),
        )
        if len(avail_excl) >= w.spec.gpus:
          w.gpus = tuple(avail_excl[: w.spec.gpus])
          for g in w.gpus:
            excl_reserved_by.setdefault(g, acct)

  def can_occupy(n: str, g: int) -> bool:
    w, my_excl = s.workers[n], _job_exclusive_account(s, n)
    if (w.spec.exclusive and w.gpus and g not in w.gpus) or (g in excl_reserved_by and excl_reserved_by[g] != my_excl):
      return False
    holders = [h for h in occupied[g] if h != n]
    if not holders:
      return True
    if w.spec.exclusive or any(s.workers[h].spec.exclusive or s.workers[h].intent != "run" for h in holders):
      return False
    return fits(
      [memory_budget(s, s.workers[h].spec, node_name) for h in holders] + [memory_budget(s, w.spec, node_name)],
      ni.memory,
    )

  def pick_bundle(n: str, blocked: set[int]) -> tuple[int, ...] | None:
    w, obs = s.workers[n], worker_obs(s, n)
    if (
      obs
      and obs.pid is not None
      and obs.phase not in ("idle", "drained", "failed")
      and obs.gpus
      and not is_movable(w.spec, ni.gpus)
    ):
      return obs.gpus if all(g not in blocked and g in occupied and can_occupy(n, g) for g in obs.gpus) else None
    avail = [g for g in range(ni.gpus) if g not in blocked and can_occupy(n, g)]
    if len(avail) < w.spec.gpus:
      return None
    mem = memory_budget(s, w.spec, node_name)
    avail.sort(key=lambda g: (bool(occupied[g]) != bool(mem), g not in w.gpus, g))
    return tuple(avail[: w.spec.gpus])

  yielded: set[str] = set()
  for _ in range(2):
    reserver = next((n for n in local if s.workers[n].reserve_node == node_name), None)
    if reserver and (not has_work(reserver) or s.workers[reserver].intent == "run"):
      s.workers[reserver].reserve_node, reserver = None, None

    waiting_ready = sorted(
      (n for n in local if s.workers[n].intent == "park" and has_work(n) and not is_preparing(n)), key=worker_priority
    )
    newly_reserved = False
    if (
      reserver is None
      and waiting_ready
      and s.workers[waiting_ready[0]].spec.gpus > 1
      and pick_bundle(waiting_ready[0], set()) is None
    ):
      reserver = waiting_ready[0]
      s.workers[reserver].reserve_node, newly_reserved = node_name, True
      out.append((Ev.DRAINING, {"worker": reserver, "node": node_name, "rule": "reserve.wide"}))

    reserved_gpus: set[int] = set()
    if reserver is not None:
      rw, robs, my_excl = s.workers[reserver], worker_obs(s, reserver), _job_exclusive_account(s, reserver)
      eligible_g = [g for g in range(ni.gpus) if g not in excl_reserved_by or excl_reserved_by[g] == my_excl]
      if (
        robs
        and robs.pid is not None
        and robs.phase not in ("idle", "drained", "failed")
        and robs.gpus
        and all(g in eligible_g for g in robs.gpus)
      ):
        target = tuple(robs.gpus)
      elif len(rw.gpus) == rw.spec.gpus and all(g in eligible_g for g in rw.gpus):
        target = rw.gpus
      else:
        target = tuple(sorted(eligible_g, key=lambda g: (len(occupied[g]), g))[: rw.spec.gpus])
      if len(target) == rw.spec.gpus:
        reserved_gpus = set(target)

    for n in local:
      w = s.workers[n]
      if w.intent != "run":
        continue
      my_excl = _job_exclusive_account(s, n)
      contenders = (
        [c for c in waiting_ready if _job_exclusive_account(s, c) == my_excl]
        if w.spec.exclusive
        else [c for c in waiting_ready if pick_bundle(c, reserved_gpus) is None]
      )
      pressure = (
        bool(contenders)
        or (bool(reserved_gpus & set(w.gpus)) and n != reserver)
        or any(excl_reserved_by.get(g) not in (None, my_excl) for g in w.gpus)
      )
      if pressure and (not has_work(n) or is_preparing(n) or over_budget(n)):
        w.intent, w.rev, w.budgets, w.grant_settled = "park", w.rev + 1, {}, {}
        yielded.add(n)
        out.append(
          (
            Ev.DRAINING,
            {"worker": n, "node": node_name, "rule": "turn.idle_yield" if not has_work(n) else "turn.budget_yield"},
          )
        )
      elif over_budget(n) and has_work(n):
        jobs = active_jobs_of(s, n)
        share = (BUDGET_S * w.spec.gpus) / max(1, len(jobs))
        w.budgets, w.rev = {j: w.grant_settled.get(j, 0.0) + share for j in jobs}, w.rev + 1
        out.append((Ev.GRANTED, {"worker": n, "node": node_name, "gpus": list(w.gpus), "rule": "turn.extend"}))

    granted_any = False

    def grant(n: str, bundle: tuple[int, ...], rule: str) -> None:
      nonlocal granted_any
      granted_any, w, jobs = True, s.workers[n], active_jobs_of(s, n)
      share = (BUDGET_S * w.spec.gpus) / max(1, len(jobs))
      w.intent, w.gpus, w.rev = "run", bundle, w.rev + 1
      w.grant_rev, w.grant_settled, w.budgets = w.rev, {}, {j: share for j in jobs}
      w.generations = {j: s.jobs[j].gen for j in jobs_of(s, n)}
      w.last_active_rev, w.reserve_node = s.table_rev + 1, None
      for g in bundle:
        if n not in occupied[g]:
          occupied[g].append(n)
        if w.spec.exclusive:
          excl_reserved_by[g] = _job_exclusive_account(s, n) or n
      for j, bud in w.budgets.items():
        acct = job_account(s, j)
        reserved_budget[acct] = reserved_budget.get(acct, 0.0) + bud
      out.append((Ev.GRANTED, {"worker": n, "node": node_name, "gpus": list(bundle), "rule": rule}))

    if reserver is not None and reserved_gpus and all(can_occupy(reserver, g) for g in sorted(reserved_gpus)):
      bundle = tuple(sorted(reserved_gpus))
      reserved_gpus = set()
      grant(reserver, bundle, "turn.wide_grant")

    candidates = sorted(
      (
        n
        for n in local
        if n not in yielded
        and s.workers[n].intent == "park"
        and not gpus_still_held(s, n)
        and ((has_work(n) and not is_preparing(n)) or warm_only(n))
      ),
      key=worker_priority,
    )
    for n in candidates:
      if n != reserver and (bundle := pick_bundle(n, reserved_gpus)):
        grant(n, bundle, "turn.grant")
    if not granted_any and not newly_reserved:
      break


def _rebalance_blocked_workers(s: Assignment, out: list[tuple[Ev, dict]]) -> None:
  """Move idle unstarted workers that could not get GPUs on their current node to any eligible node with free GPUs."""
  moved_workers: set[str] = set()
  while True:
    moved_any = False
    for n, w in sorted(s.workers.items()):
      if n in moved_workers or w.node is None or w.node not in s.nodes or w.intent != "park" or w.spec.exclusive:
        continue
      if gpus_still_held(s, n) or not any(s.jobs[j].ready for j in active_jobs_of(s, n)):
        continue
      if (obs := worker_obs(s, n)) is not None and (obs.phase != "idle" or not obs.gpu_released or obs.pid is not None):
        continue
      other = next(
        (
          on
          for on in _eligible_nodes(s, n)
          if on != w.node
          and not any(ow.reserve_node == on for ow in s.workers.values())
          and _node_free_gpus_for(s, on, w.spec, exclude=n) >= w.spec.gpus
        ),
        None,
      )
      if other is None:
        continue
      old_node, was_reserver = w.node, w.reserve_node == w.node
      w.node, w.gpus, w.reserve_node, w.rev = other, (), None, w.rev + 1
      moved_workers.add(n)
      moved_any = True
      out.append((Ev.PLACED, {"worker": n, "node": other, "rule": "place.rebalance"}))
      _allocate_node_gpus(s, other, out)
      if was_reserver:
        _allocate_node_gpus(s, old_node, out)
    if not moved_any:
      break


# ***** decide *****


def decide(state: Assignment) -> tuple[Assignment, list[tuple[Ev, dict]]]:
  """Compute the next cluster assignment and events deterministically from observed state."""
  s, out = copy.deepcopy(state), []

  # 1. Reconcile lost nodes: conservatively charge remaining budget, fence jobs by bumping gen, unplace workers.
  for name, w in sorted(s.workers.items()):
    if w.node and w.node not in s.nodes:
      if w.intent == "run":
        for jid, bud in w.budgets.items():
          if (rem := max(0.0, bud - w.grant_settled.get(jid, 0.0))) > 0.0:
            acct = job_account(s, jid)
            s.service[acct] = s.service.get(acct, 0.0) + rem
      _reset_worker(s, name, "place.node_lost", out)

  # 2. Settle cumulative per-job usage from live node observations into persisted fair service.
  live_obs_keys: set[str] = set()
  for ni in s.nodes.values():
    for wname, obs in sorted(ni.workers.items()):
      if not obs.owner:
        continue
      for jid, jobs_o in sorted(obs.jobs.items()):
        okey = f"{obs.owner}:{jid}"
        live_obs_keys.add(okey)
        if (delta := max(0.0, jobs_o.usage - s.service_obs.get(okey, 0.0))) > 0.0:
          s.service_obs[okey], acct = jobs_o.usage, job_account(s, jid)
          s.service[acct] = s.service.get(acct, 0.0) + delta
          if wname in s.workers:
            s.workers[wname].grant_settled[jid] = s.workers[wname].grant_settled.get(jid, 0.0) + delta
  for old_key in list(s.service_obs):
    if old_key not in live_obs_keys:
      del s.service_obs[old_key]

  ready_accounts = {job_account(s, jid) for jid, j in s.jobs.items() if j.ready}
  floor_pool = (ready_accounts & s.service_ready) or {a for a in ready_accounts if a in s.service}
  floor_norm = min((s.service.get(a, 0.0) / account_weight(s, a) for a in floor_pool), default=0.0)
  for acct in sorted(ready_accounts - s.service_ready):
    s.service[acct] = max(s.service.get(acct, 0.0), floor_norm * account_weight(s, acct))
  s.service_ready = ready_accounts
  live_accounts = {job_account(s, jid) for jid in s.jobs}
  for acct in list(s.service):
    if acct not in live_accounts:
      del s.service[acct]

  # 3. Reconcile failed workers and completed full/single-job drains (always confirming GPU release & bumping gen).
  for name, w in sorted(list(s.workers.items())):
    obs = worker_obs(s, name)
    if w.node and obs and obs.phase == "failed" and obs.gpu_released and obs.rev >= w.rev:
      _reset_worker(s, name, "worker.failed", out)
      continue
    if (
      w.node
      and w.intent == "drain"
      and obs
      and obs.rev >= w.rev
      and obs.gpu_released
      and obs.phase in ("drained", "failed")
    ):
      if not jobs_of(s, name):
        del s.workers[name]
        out.append((Ev.UNPLACED, {"worker": name, "rule": "place.emptied"}))
      else:
        _reset_worker(s, name, "place.drained", out)
      continue
    for jid, target in sorted(list(w.drain_jobs.items())):
      if jid not in s.jobs or s.jobs[jid].worker != name:
        w.drain_jobs.pop(jid, None)
        continue
      jobs_o = obs.jobs.get(jid) if obs else None
      if (jobs_o and jobs_o.state == "drained") or (obs and obs.gpu_released and obs.phase in ("drained", "failed")):
        for d in (w.drain_jobs, w.generations, w.budgets, w.grant_settled):
          d.pop(jid, None)
        w.rev += 1
        s.jobs[jid].gen += 1
        dest = (
          target
          if target and target != name and target in s.workers and s.workers[target].intent != "drain"
          else _new_worker(s, s.jobs[jid].spec)
        )
        _assign_job(s, jid, dest)
        s.workers[dest].rev += 1
        out.append((Ev.ASSIGNED, {"run": jid, "worker": dest}))

  # 4. Assign unassigned jobs to workers and sync generations; drain or delete empty workers.
  for jid, job in sorted(s.jobs.items()):
    if not job.worker or job.worker not in s.workers:
      worker = existing_worker(s, job) or _new_worker(s, job.spec)
      _assign_job(s, jid, worker)
      out.append((Ev.ASSIGNED, {"run": jid, "worker": worker}))
    elif jid not in s.workers[job.worker].drain_jobs:
      job.gen = job.gen or 1
      s.workers[job.worker].generations[jid] = job.gen

  for name, w in sorted(list(s.workers.items())):
    if jobs_of(s, name):
      continue
    obs = worker_obs(s, name)
    if (
      w.node is None
      or w.node not in s.nodes
      or (obs is None and w.grant_rev == 0)
      or (obs is not None and obs.gpu_released and obs.phase in ("idle", "drained", "failed"))
    ):
      del s.workers[name]
      out.append((Ev.UNPLACED, {"worker": name, "rule": "place.emptied"}))
    elif w.intent != "drain":
      w.intent, w.rev, w.budgets, w.reserve_node = "drain", w.rev + 1, {}, None
      out.append((Ev.DRAINING, {"worker": name, "node": w.node, "rule": "drain.emptied"}))

  # 5. Host-RAM pressure: demote coldest locally parked trainers to durable storage.
  for node_name, ni in sorted(s.nodes.items()):
    used = ni.host_ram_used or sum(o.host_ram_bytes for o in ni.workers.values())
    if ni.host_ram <= 0 or used <= HOST_RAM_HIGH * ni.host_ram:
      continue
    parked = [
      (s.workers[n].last_active_rev, n, obs.host_ram_bytes)
      for n in sorted(s.workers)
      if s.workers[n].node == node_name
      and s.workers[n].spec.role == Role.TRAINER
      and s.workers[n].intent == "park"
      and (obs := ni.workers.get(n)) is not None
      and obs.phase == "parked"
      and obs.gpu_released
      and obs.host_ram_bytes > 0
    ]
    reclaimed = 0
    for _, n, nbytes in sorted(parked):
      if used - reclaimed <= HOST_RAM_HIGH * ni.host_ram:
        break
      s.workers[n].intent, s.workers[n].rev, s.workers[n].reserve_node = "drain", s.workers[n].rev + 1, None
      reclaimed += nbytes
      out.append((Ev.DRAINING, {"worker": n, "node": node_name, "rule": "drain.host_ram"}))

  # 6. Place unplaced non-draining workers (exclusive first so dedicated bundles are respected).
  unplaced = [n for n, w in s.workers.items() if w.node is None and w.intent != "drain" and jobs_of(s, n)]
  for name in sorted(unplaced, key=lambda n: (not s.workers[n].spec.exclusive, n)):
    if placed := place_worker(s, name):
      s.workers[name].node, s.workers[name].rev = placed["node"], s.workers[name].rev + 1
      out.append((Ev.PLACED, placed))

  # 7. Per-node physical GPU & budget allocation, then cross-node rebalance of idle blocked workers.
  for node_name in sorted(s.nodes):
    _allocate_node_gpus(s, node_name, out)
  _rebalance_blocked_workers(s, out)

  # 8. Populate restore_refs & prefetch instructions for every placed worker.
  if _sync_restore_and_prefetch(s) or out:
    s.table_rev += 1
  return s, out


# ***** scheduler loop *****


def _worker_ref_versions(w: Worker) -> set[str]:
  vers = {parse_version(r.get("version"))[0] for r in w.restore_refs.values()} | {
    parse_version(p.get("version"))[0] for p in w.prefetch
  }
  return {v for v in vers if v}


async def _sync_assignment_refs(registry: Registry, old: Assignment, new: Assignment) -> None:
  """Protect every version referenced by an active assignment via assign:<worker>:<rev> refs."""
  for name, nw in new.workers.items():
    ow = old.workers.get(name)
    new_vers, old_vers = _worker_ref_versions(nw), _worker_ref_versions(ow) if ow else set()
    if ow is None or nw.rev != ow.rev or new_vers != old_vers:
      for ver in new_vers:
        await registry.acquire_ref(ver, f"assign:{name}:{nw.rev}")
      if ow is not None:
        for ver in old_vers:
          if nw.rev != ow.rev or ver not in new_vers:
            await registry.release_ref(ver, f"assign:{name}:{ow.rev}")
  for gone in old.workers.keys() - new.workers.keys():
    for ver in _worker_ref_versions(old.workers[gone]):
      await registry.release_ref(ver, f"assign:{gone}:{old.workers[gone].rev}")


def _state_changed(old: Assignment, new: Assignment, changes: list) -> bool:
  return bool(
    changes
    or old.workers != new.workers
    or old.service != new.service
    or old.service_obs != new.service_obs
    or old.service_ready != new.service_ready
    or {j: (x.worker, x.gen) for j, x in old.jobs.items()} != {j: (x.worker, x.gen) for j, x in new.jobs.items()}
  )


async def run(bus: Bus) -> None:
  registry, leader = Registry(bus), f"sched-{socket.gethostname()}-{os.getpid()}-{os.urandom(4).hex()}"
  cursor, held = await bus.wake_cursor("scheduler"), False
  try:
    while True:
      held = (await bus.renew_scheduler_lease(leader)) if held else (await bus.take_scheduler_lease(leader))
      if not held:
        await asyncio.sleep(TICK_MS / 1000)
        continue
      for session in await bus.expired_sessions():
        await bus.end_session(session)
      await registry.unregister_checkpoints(*await registry.expired_checkpoints())

      s = from_table(await bus.read_table())
      new, changes = decide(s)
      if _state_changed(s, new, changes):
        await _sync_assignment_refs(registry, s, new)
        assigned = {r: x.worker for r, x in new.jobs.items() if r not in s.jobs or x.worker != s.jobs[r].worker}
        generations = {r: x.gen for r, x in new.jobs.items() if r not in s.jobs or x.gen != s.jobs[r].gen}
        gone = sorted(s.workers.keys() - new.workers.keys())
        wake_nodes = {
          w.node for n, w in new.workers.items() if w.node and (n not in s.workers or s.workers[n] != w)
        } | {s.workers[n].node for n in gone if s.workers[n].node}
        ok = await bus.write_table(
          assigned,
          {n: worker_to_dict(w) for n, w in new.workers.items()},
          gone,
          new.workers_made,
          changes,
          generations=generations,
          service=new.service,
          service_obs=new.service_obs,
          service_ready=new.service_ready,
          table_rev=new.table_rev,
          leader=leader,
          wake_nodes=wake_nodes,
        )
        if not ok:
          held = False
          continue
      cursor = await bus.wait_wake("scheduler", cursor, block_ms=TICK_MS)
  finally:
    if held:
      await bus.release_scheduler_lease(leader)


if __name__ == "__main__":
  asyncio.run(run(Bus()))
