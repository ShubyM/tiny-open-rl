"""Fleet scheduler: assigns physical intra-node GPU bundles, ownership generations, bounded work budgets, drain intents and storage preparation from one inspectable Redis table + node observations, under a single-writer lease."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import itertools
import os
import socket
import time
from dataclasses import dataclass, field

from tinyrl.bus import LEASE_MS, Bus, Cmd, Ev, Op, Role, entry_key
from tinyrl.control import estimator
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry

TURN, TICK_MS, MAX_SHARED = getenv("TURN", 200_000), getenv("TICK_MS", 100), getenv("MAX_SHARED_JOBS", 16)
STARVE_S, RESTORE_S = getenv("STARVE_S", 2.0), getenv("RESTORE_S", 0.5)
EXCLUSIVE_FRACTION, EXCLUSIVE_MAX_GPUS = float(os.getenv("EXCLUSIVE_FRACTION", "1.0")), getenv("EXCLUSIVE_MAX_GPUS", 0)


def select_runs(heads: list[Cmd], service: dict[str, float], width: int, prefer: str | None = None) -> list[str]:
  """Bounded in-assignment dispatch ordering among ready heads on a shared worker."""
  if width < 1:
    raise ValueError("batch width must be positive")
  ready: dict[str, Cmd] = {}
  for cmd in heads:
    ready.setdefault(cmd.run, cmd)
  floor = min((service[run] for run in ready if run in service), default=0.0)
  ordered = sorted(
    ready.values(),
    key=lambda c: (
      c.run != prefer,
      c.op != Op.SAVE_WEIGHTS_FOR_SAMPLER,
      service.get(c.run, floor),
      entry_key(c.entry_id),
      c.run,
    ),
  )
  return [cmd.run for cmd in ordered[:width]]


def settings(spec: WorkerSpec, gpus: int, gpu_bytes: int, gpu_class: str = "") -> dict[str, str]:
  env = {"TINYRL_GPU_CLASS": gpu_class} if gpu_class else {}
  if spec.role != Role.SAMPLER:
    return env
  if gpus >= 2:
    env["TINYRL_WARM_SAMPLER"] = "1"
  if gpu_bytes >= 64 * 2**30 and not spec.lora and not spec.exclusive and spec.gpus == 1:
    env.update(TINYRL_GPU_MEM="0.42", TINYRL_KV_BYTES=str(16 * 1024**3))
  elif gpu_bytes >= 64 * 2**30:
    env["TINYRL_KV_BYTES"] = str(24 * 1024**3)
  return env


def memory_budget(node: NodeInfo | None, spec: WorkerSpec) -> int:
  return (
    int(float(settings(spec, node.gpus, node.gpu_bytes, node.gpu_class).get("TINYRL_GPU_MEM", 0)) * node.gpu_bytes)
    if node
    else 0
  )


def fits(memory: list[int], gpu_bytes: int, overhead: int = 0) -> bool:
  return (len(memory) == 1 and (gpu_bytes <= 0 or not memory[0] or memory[0] + overhead <= gpu_bytes)) or (
    gpu_bytes > 0 and all(memory) and sum(memory) + overhead <= gpu_bytes
  )


# ***** records *****


@dataclass
class NodeInfo:
  gpus: int = 0
  gpu_bytes: int = 0
  host_bytes: int = 0
  gpu_ids: tuple[int, ...] = ()
  gpu_class: str = ""
  rate: float = 0.0
  bundles: list[list[int]] = field(default_factory=list)
  peer: str = ""
  cached: set[str] = field(default_factory=set)
  models: set[str] = field(default_factory=set)
  peer_bps: float = 512.0 * 2**20
  bucket_bps: float = 192.0 * 2**20

  def __post_init__(self):
    self.gpu_ids = self.gpu_ids or (tuple(range(self.gpus)) if self.gpus > 0 else ())
    self.gpus = len(self.gpu_ids)


@dataclass(frozen=True)
class WorkerSpec:
  role: Role = Role.TRAINER
  base: str = ""
  lora: bool = False
  exclusive: bool = False
  train_run: str | None = None
  gpus: int = 1


@dataclass
class Job:
  spec: WorkerSpec
  worker: str = ""
  train_run: str | None = None
  account: str = ""
  generation: int = 0
  queued: int = 0
  restore_ver: str | None = None
  weights_ver: str | None = None
  gpu_classes: tuple[str, ...] = ()
  created: float = 0.0
  target_node: str | None = None


@dataclass
class Worker:
  spec: WorkerSpec = WorkerSpec()
  node: str | None = None
  gpus: tuple[int, ...] = ()
  revision: int = 0
  intent: str = "idle"
  drain_mode: str = ""
  budgets: dict[str, int] = field(default_factory=dict)
  generations: dict[str, int] = field(default_factory=dict)
  restores: dict[str, str] = field(default_factory=dict)
  draining_jobs: list[str] = field(default_factory=list)
  prepare: list[str] = field(default_factory=list)
  predicted: dict[str, float] = field(default_factory=dict)
  issued_at: float = 0.0
  pid: int | None = None


@dataclass
class Observation:
  worker: str = ""
  node: str = ""
  revision: int = 0
  generations: dict[str, int] = field(default_factory=dict)
  gpus: tuple[int, ...] = ()
  phase: str = "released"
  gpu_state: str = "released"
  pids_clear: bool = True
  gpu_bytes: int = 0
  host_bytes: int = 0
  usage: dict[str, float] = field(default_factory=dict)
  served: dict[str, int] = field(default_factory=dict)
  boundaries: dict[str, str] = field(default_factory=dict)
  exported_jobs: list[str] = field(default_factory=list)
  local_state: list[str] = field(default_factory=list)
  pid: int | None = None
  at: float = 0.0


@dataclass
class Assignment:
  workers_made: int = 0
  nodes: dict[str, NodeInfo] = field(default_factory=dict)
  workers: dict[str, Worker] = field(default_factory=dict)
  jobs: dict[str, Job] = field(default_factory=dict)
  observations: dict[str, Observation] = field(default_factory=dict)
  service: dict[str, float] = field(default_factory=dict)
  settled: dict[str, dict] = field(default_factory=dict)
  reserved: dict[str, dict] = field(default_factory=dict)
  versions: dict[str, dict] = field(default_factory=dict)


def node_from_dict(v: dict | tuple) -> NodeInfo:
  if isinstance(v, tuple):
    return NodeInfo(gpus=v[0], gpu_bytes=v[1])
  c, g = v.get("cache") or {}, int(v.get("gpus", 0))
  return NodeInfo(
    g,
    int(v.get("memory", 0)),
    int(v.get("host_memory", 0)),
    tuple(v.get("gpu_ids") or range(g)),
    v.get("gpu_class", ""),
    float(v.get("rate", 0.0)),
    list(v.get("bundles") or []),
    c.get("peer") or v.get("peer", ""),
    set(c.get("versions") or v.get("cached") or ()),
    set(v.get("models") or ()),
    float(c.get("peer_bps") or 512.0 * 2**20),
    float(c.get("bucket_bps") or 192.0 * 2**20),
  )


def worker_to_dict(w: Worker) -> dict:
  return dataclasses.asdict(w) | {"gpus": list(w.gpus)}


def worker_from_dict(w: dict) -> Worker:
  spec = WorkerSpec(**(w["spec"] | {"role": Role(w["spec"]["role"])}))
  keep = {f.name: w[f.name] for f in dataclasses.fields(Worker) if f.name in w and f.name not in ("spec", "gpus")}
  return Worker(spec=spec, gpus=tuple(w.get("gpus") or ()), **keep)


def obs_from_dict(name: str, o: dict) -> Observation:
  keep = {
    f.name: o[f.name] for f in dataclasses.fields(Observation) if f.name in o and f.name not in ("worker", "gpus")
  }
  return Observation(worker=name, gpus=tuple(o.get("gpus") or ()), **keep)


def from_table(table: dict) -> Assignment:
  queues = table.get("queues") or {}
  s = Assignment(
    int(table.get("workers_made", 0)),
    {n: node_from_dict(v) for n, v in table["nodes"].items()},
    {},
    {},
    {k: obs_from_dict(k, v) for k, v in (table.get("observations") or {}).items()},
    dict(table.get("service") or {}),
    copy.deepcopy(table.get("settled") or {}),
    copy.deepcopy(table.get("reserved") or {}),
    dict(table.get("versions") or {}),
  )
  for job_id, row in table["runs"].items():
    rec, worker, st = row[0], row[1], (row[2] if len(row) > 2 else {})
    role, lora, excl, tr = (
      Role(rec.get("role", "trainer")),
      rec.get("rank", 0) > 0,
      bool(rec.get("exclusive", False)),
      rec.get("train_run"),
    )
    spec = WorkerSpec(role, rec.get("base", ""), lora, excl, tr if excl else None, int(rec.get("gpus", 1)))
    rver = (st.get("working") or {}).get("version") or (st.get("durable") or {}).get("version")
    cls = tuple(
      (st.get("profile") or {}).get("gpu_classes")
      or st.get("gpu_classes")
      or (rec.get("lora") or {}).get("gpu_classes")
      or ()
    )
    acct = st.get("account") or (tr if role == Role.SAMPLER and tr else job_id)
    s.jobs[job_id] = Job(
      spec,
      worker or "",
      tr,
      acct,
      int(st.get("generation") or (1 if worker else 0)),
      int(queues.get(job_id, 0)),
      rver,
      st.get("weights_version"),
      cls,
      float(rec.get("created", 0.0)),
      st.get("target_node"),
    )
  s.jobs = {r: x for r, x in s.jobs.items() if not x.spec.train_run or x.spec.train_run in s.jobs}
  for r, x in s.jobs.items():
    if x.train_run and x.account == x.train_run and x.train_run not in s.jobs and x.account not in s.service:
      x.account = r
  s.workers = {name: worker_from_dict(w) for name, w in table["workers"].items()}
  for r, x in s.jobs.items():
    if x.worker:
      s.workers.setdefault(x.worker, Worker(x.spec)).generations.setdefault(r, max(1, x.generation))
  return s


def jobs_of(s: Assignment, worker: str) -> list[str]:
  return [r for r, x in s.jobs.items() if x.worker == worker]


def bump_rev(w: Worker, now: float, intent: str, drain_mode: str = "", gpus: tuple[int, ...] | None = None) -> None:
  new_gpus = w.gpus if gpus is None else gpus
  if (w.intent, w.drain_mode, w.gpus) != (intent, drain_mode, new_gpus) or w.revision == 0:
    w.revision += 1
    w.intent, w.drain_mode, w.gpus, w.issued_at = intent, drain_mode, new_gpus, now


def node_eligible(s: Assignment, node: str, spec: WorkerSpec, classes: tuple[str, ...] = ()) -> bool:
  info = s.nodes.get(node)
  prof = {"gpus": spec.gpus, "gpu_classes": classes, "lora": spec.lora, "role": str(spec.role)}
  return bool(info and estimator.compatible(prof, info.gpu_class, info.gpu_ids, info.bundles))


def worker_classes(s: Assignment, worker: str) -> tuple[str, ...]:
  if not (sets := [set(s.jobs[r].gpu_classes) for r in jobs_of(s, worker) if s.jobs[r].gpu_classes]):
    return ()
  return tuple(sorted(common)) if (common := set.intersection(*sets)) else ("__none__",)


def is_released(obs: Observation | None) -> bool:
  return obs is None or (obs.gpu_state == "released" and obs.pids_clear)


def is_parked_or_released(w: Worker, obs: Observation | None) -> bool:
  return is_released(obs) or bool(
    obs and obs.gpu_state == "offloaded" and (w.intent != "drain" or obs.revision >= w.revision)
  )


def held_gpus(s: Assignment, node: str, exclude: str = "") -> tuple[dict[int, list[str]], dict[int, int]]:
  """Map each GPU on `node` to active/unreleased holder workers and residual offloaded CUDA context bytes."""
  info = s.nodes[node]
  active, offload = {g: [] for g in info.gpu_ids}, {g: 0 for g in info.gpu_ids}
  for name in sorted(set(s.workers) | set(s.observations)):
    if name == exclude:
      continue
    w, obs = s.workers.get(name), s.observations.get(name)
    if obs and obs.node == node:
      if obs.gpu_state == "offloaded":
        per = max(obs.gpu_bytes, estimator.OVERHEAD if info.gpu_bytes else 0) // max(1, len(obs.gpus))
        for g in (g for g in obs.gpus if g in offload):
          offload[g] += per
      elif not is_released(obs):
        for g in (g for g in obs.gpus if g in active and name not in active[g]):
          active[g].append(name)
    if (
      w
      and w.node == node
      and w.gpus
      and (w.intent in ("run", "idle") or (w.intent == "drain" and not is_parked_or_released(w, obs)))
    ):
      for g in (g for g in w.gpus if g in active and name not in active[g]):
        active[g].append(name)
  return active, offload


def bundle_free(s: Assignment, node: str, bundle: tuple[int, ...], worker: str, ignore_res: bool = False) -> bool:
  info, me, res = s.nodes[node], s.workers[worker], s.reserved.get(node)
  if res and not ignore_res and res.get("worker") != worker and (set(bundle) & set(res.get("gpus") or ())):
    return False
  active, offload = held_gpus(s, node, exclude=worker)
  my_mem = memory_budget(info, me.spec)
  for g in bundle:
    if not (holders := active.get(g, [])):
      continue
    specs = [s.workers[h].spec for h in holders if h in s.workers]
    if me.spec.exclusive or len(specs) != len(holders) or any(x.exclusive for x in specs):
      return False
    if not fits([*(memory_budget(info, x) for x in specs), my_mem], info.gpu_bytes, offload.get(g, 0)):
      return False
  return True


def shareable_worker(s: Assignment, job: Job) -> str | None:
  spec = job.spec
  if spec.role == Role.TRAINER and (not spec.lora or spec.exclusive):
    return None
  candidates = []
  for name, w in sorted(s.workers.items()):
    if (
      w.spec != spec
      or (w.intent == "drain" and w.drain_mode in ("export", "close"))
      or len(jobs_of(s, name)) >= MAX_SHARED
    ):
      continue
    if (job.target_node and w.node and w.node != job.target_node) or (
      w.node and job.gpu_classes and not node_eligible(s, w.node, spec, job.gpu_classes)
    ):
      continue
    if job.gpu_classes and (cls := worker_classes(s, name)) and not (set(cls) & set(job.gpu_classes)):
      continue
    candidates.append(name)
  if not candidates:
    return None
  return min(
    candidates, key=lambda n: (bool(job.target_node and s.workers[n].node != job.target_node), len(jobs_of(s, n)), n)
  )


def assign_job(s: Assignment, job_id: str, worker: str, *, new_gen: bool = True) -> dict[str, int]:
  job, w, bumped = s.jobs[job_id], s.workers.setdefault(worker, Worker(s.jobs[job_id].spec)), {}
  job.worker = worker
  if new_gen or job.generation == 0:
    job.generation += 1
    bumped[job_id] = job.generation
  w.generations[job_id] = job.generation
  if job.spec.role == Role.TRAINER and job.restore_ver:
    w.restores[job_id] = job.restore_ver
  return bumped


def worker_ready(s: Assignment, worker: str) -> int:
  w = s.workers[worker]
  total = sum(s.jobs[r].queued for r in jobs_of(s, worker) if r not in w.draining_jobs)
  if (
    total == 0
    and w.spec.role == Role.SAMPLER
    and w.node
    and s.nodes[w.node].gpus >= 2
    and worker not in s.observations
    and jobs_of(s, worker)
  ):
    return 1  # warm sampler startup when uncontended on multi-GPU nodes
  return total


def settle_service(s: Assignment, now: float) -> float:
  """Fold cumulative observed usage deltas into persisted job-account service and lift returning idle accounts to floor."""
  was_active = {a for w in s.workers.values() if w.intent == "run" for a in w.predicted}
  prev_ready = set((s.settled.get("__ready__") or {}).get("accounts") or ())
  for name, obs in sorted(s.observations.items()):
    prev = (s.settled.get(name) or {}).get("usage") or {}
    if not obs.usage and not prev:
      continue
    merged = dict(prev)
    for acct, total in obs.usage.items():
      if (delta := float(total) - float(prev.get(acct, 0.0))) > 1e-9:
        s.service[acct] = round(s.service.get(acct, 0.0) + delta, 6)
        merged[acct] = float(total)
      else:
        merged.setdefault(acct, float(total))
    s.settled[name] = {"usage": merged, "at": obs.at or now}
  ready_accts = {j.account for j in s.jobs.values() if j.queued > 0}
  continuing = was_active | (ready_accts & prev_ready)
  floor = min((s.service.get(a, 0.0) for a in continuing if a in s.service), default=0.0)
  for acct in sorted(ready_accts):
    s.service[acct] = floor if acct not in continuing and s.service.get(acct, 0.0) < floor else s.service.get(acct, 0.0)
  s.settled["__ready__"] = {"accounts": sorted(ready_accts)}
  return floor


def worker_priority(s: Assignment, worker: str, floor: float) -> float:
  members = jobs_of(s, worker)
  accts = list(dict.fromkeys(s.jobs[r].account for r in members if s.jobs[r].queued > 0)) or [
    s.jobs[r].account for r in members[:1]
  ]
  return min(
    (
      s.service.get(a, floor) + sum(w.predicted.get(a, 0.0) for w in s.workers.values() if w.intent == "run")
      for a in accts
    ),
    default=floor,
  )


def state_transfer_s(s: Assignment, worker: str, node: str) -> float:
  info, obs = s.nodes[node], s.observations.get(worker)
  if obs and obs.node == node and obs.gpu_state in ("active", "paused", "offloaded"):
    return 0.0
  vers = {
    v
    for r in jobs_of(s, worker)
    if (v := (s.jobs[r].weights_ver if s.jobs[r].spec.role == Role.SAMPLER else s.jobs[r].restore_ver))
  }
  return sum(
    estimator.transfer_seconds(s.versions.get(v), info.peer, info.cached, info.peer_bps, info.bucket_bps) for v in vers
  )


def candidate_makespan(
  s: Assignment, worker: str, node: str, bundle: tuple[int, ...], tokens: int
) -> tuple[float, float]:
  info, me = s.nodes[node], s.workers[worker]
  xfer, exec_s = (
    state_transfer_s(s, worker, node),
    estimator.exec_seconds(tokens or TURN, me.spec.gpus, info.gpu_class, info.rate),
  )
  if bundle_free(s, node, bundle, worker):
    wait = 0.0
  else:
    holders = {h for g in bundle for h in held_gpus(s, node, exclude=worker)[0].get(g, ())}
    busy = sum(1 for h in holders if h in s.workers and worker_ready(s, h) > 0)
    wait = (
      1e6
      if any(h not in s.workers or s.workers[h].spec.exclusive for h in holders)
      else busy * estimator.exec_seconds(TURN, 1, info.gpu_class, info.rate) + (0.2 if holders else 0.0)
    )
  trainers = (
    {s.jobs[t].worker for r in jobs_of(s, worker) if (t := s.jobs[r].train_run) in s.jobs}
    if me.spec.role == Role.SAMPLER and not me.spec.lora
    else set()
  )
  bonus = -0.02 if any(t in s.workers and s.workers[t].node == node for t in trainers) else 0.0
  return max(wait, xfer) + exec_s + bonus, xfer


def best_placement(s: Assignment, worker: str) -> tuple[str | None, tuple[int, ...], float, float]:
  w, cls = s.workers[worker], worker_classes(s, worker)
  tokens = sum(min(TURN, max(1, s.jobs[r].queued) * 1024) for r in jobs_of(s, worker))
  forced = next((s.jobs[r].target_node for r in jobs_of(s, worker) if s.jobs[r].target_node), None)
  best: tuple[str | None, tuple[int, ...], float, float] = (None, (), 1e18, 0.0)
  best_score: tuple = ()
  for node in sorted(s.nodes):
    if (forced and forced in s.nodes and node != forced) or not node_eligible(s, node, w.spec, cls):
      continue
    info = s.nodes[node]
    load = sum(1 for n, x in s.workers.items() if n != worker and x.node == node) / max(1, info.gpus)
    for bundle in estimator.eligible_bundles(info.gpu_ids, w.spec.gpus, info.bundles):
      ms, xfer = candidate_makespan(s, worker, node, bundle, tokens)
      stay = 0 if (w.node == node and w.gpus == bundle) else (1 if w.node == node else 2)
      score = (round(ms, 4), not bundle_free(s, node, bundle, worker), round(load, 4), stay, node, bundle)
      if best[0] is None or score < best_score:
        best, best_score = (node, bundle, ms, xfer), score
  return best


def exclusive_admitted(s: Assignment) -> set[str]:
  """Admit exclusive workers up to the fleet exclusive cap in arrival order; excess exclusive requests stay queued."""
  cap = (
    EXCLUSIVE_MAX_GPUS if EXCLUSIVE_MAX_GPUS > 0 else int(sum(n.gpus for n in s.nodes.values()) * EXCLUSIVE_FRACTION)
  )
  excl = [n for n, w in s.workers.items() if w.spec.exclusive and jobs_of(s, n)]

  def order(n: str) -> tuple:
    created = min((s.jobs[r].created for r in jobs_of(s, n)), default=0.0)
    return s.workers[n].node is None, created, s.workers[n].spec.role == Role.SAMPLER, n

  admitted, used = set(), 0
  for name in sorted(excl, key=order):
    if used + s.workers[name].spec.gpus <= cap:
      admitted.add(name)
      used += s.workers[name].spec.gpus
  return admitted


def waiting_on_io(s: Assignment, worker: str) -> bool:
  o = s.observations.get(worker)
  return bool(o and o.phase == "preparing" and o.gpu_state in ("active", "offloaded"))


def turn_done(s: Assignment, worker: str, obs: Observation | None, ready_n: int) -> bool:
  w = s.workers[worker]
  if w.intent == "idle" or ready_n == 0:
    return True
  if not obs or obs.revision < w.revision or not w.budgets:
    return False
  if waiting_on_io(s, worker):
    return True
  ready_jobs = [r for r in w.budgets if r in s.jobs and s.jobs[r].queued > 0 and r not in w.draining_jobs]
  return bool(
    sum(obs.served.values()) >= TURN or (ready_jobs and all(obs.served.get(r, 0) >= w.budgets[r] for r in ready_jobs))
  )


def assign_budgets(s: Assignment, worker: str, node: str, now: float) -> None:
  w, info = s.workers[worker], s.nodes[node]
  members = [r for r in jobs_of(s, worker) if r not in w.draining_jobs]
  ready = [r for r in members if s.jobs[r].queued > 0]
  w.budgets = {r: TURN for r in (ready or members)}
  w.restores = {
    r: s.jobs[r].restore_ver for r in members if s.jobs[r].spec.role == Role.TRAINER and s.jobs[r].restore_ver
  }
  pred: dict[str, float] = {}
  if ready:
    share = (
      estimator.exec_seconds(TURN, w.spec.gpus, info.gpu_class, info.rate)
      * estimator.service_rate(w.spec.gpus, info.gpu_class, info.rate)
      / len(ready)
    )
    for r in ready:
      pred[s.jobs[r].account] = round(pred.get(s.jobs[r].account, 0.0) + share, 6)
  w.predicted, w.issued_at = pred, now


def update_prepare(s: Assignment) -> None:
  for name, w in s.workers.items():
    if not w.node or w.node not in s.nodes or (w.intent == "drain" and w.drain_mode in ("export", "close")):
      w.prepare = []
      continue
    cached, obs = s.nodes[w.node].cached, s.observations.get(name)
    live = bool(obs and obs.node == w.node and obs.gpu_state in ("active", "paused", "offloaded"))
    needed: list[str] = []
    for r in jobs_of(s, name):
      j = s.jobs[r]
      ver = (
        j.weights_ver
        if j.spec.role == Role.SAMPLER
        else (None if live and r in (obs.generations or {}) else j.restore_ver)
      )
      if ver and ver not in cached and ver not in needed and ver in s.versions:
        needed.append(ver)
    w.prepare = needed


# ***** decide *****


def decide(state: Assignment, now: float | None = None) -> tuple[Assignment, list[tuple[Ev, dict]], dict[str, int]]:
  """Compute the next fleet assignment, trace events, and per-job generation advances."""
  now = time.time() if now is None else now
  s = copy.deepcopy(state)
  out: list[tuple[Ev, dict]] = []
  gen_bumps: dict[str, int] = {}
  floor = settle_service(s, now)

  def next_worker() -> str:
    s.workers_made = next(n for n in itertools.count(s.workers_made + 1) if f"w{n}" not in s.workers)
    return f"w{s.workers_made}"

  def bump_jobs(name: str) -> None:
    for r in jobs_of(s, name):
      s.jobs[r].generation += 1
      gen_bumps[r] = s.workers[name].generations[r] = s.jobs[r].generation

  def unplace(name: str, *, keep_node: bool = False) -> None:
    w = s.workers[name]
    w.node, w.gpus, w.intent, w.drain_mode = (w.node if keep_node else None), (), "idle", ""
    w.budgets, w.predicted, w.prepare, w.draining_jobs = (
      {},
      {},
      ([] if not keep_node else w.prepare),
      ([] if not keep_node else w.draining_jobs),
    )
    w.revision += 1

  def drain(name: str, mode: str, **extra) -> None:
    w = s.workers[name]
    bump_rev(w, now, "drain", mode)
    w.budgets, w.predicted = {}, {}
    out.append((Ev.DRAINING, {"worker": name, "mode": mode, **extra}))

  # 1. Reconcile lost/ineligible/shrunk nodes and drop dead reservations & observations.
  for node in [
    n
    for n, r in s.reserved.items()
    if n not in s.nodes or r.get("worker") not in s.workers or not jobs_of(s, r["worker"])
  ]:
    del s.reserved[node]
  for name in [n for n, o in s.observations.items() if o.node not in s.nodes]:
    del s.observations[name]
  for name, w in sorted(s.workers.items()):
    if w.node and w.node not in s.nodes:
      if w.intent == "run" and w.predicted and (cap := sum(w.predicted.values())) > 0:
        scale = min(1.0, max(0.0, now - (w.issued_at or now)) * max(1, w.spec.gpus) / cap)
        for acct, p in w.predicted.items():
          s.service[acct] = round(s.service.get(acct, 0.0) + p * scale, 6)
      unplace(name)
      bump_jobs(name)
      out.append((Ev.UNPLACED, {"worker": name, "rule": "place.node_lost"}))
    elif w.node and not node_eligible(s, w.node, w.spec, worker_classes(s, name)):
      unplace(name)
      bump_jobs(name)
    elif w.node and w.gpus and any(g not in s.nodes[w.node].gpu_ids for g in w.gpus):
      unplace(name, keep_node=True)

  # 2. Complete finished per-job drains, crash recoveries, and worker export/close drains.
  for name, w in sorted(s.workers.items()):
    obs = s.observations.get(name)
    if w.draining_jobs and obs and obs.exported_jobs:
      for r in [r for r in list(w.draining_jobs) if r in obs.exported_jobs and r in s.jobs]:
        w.draining_jobs.remove(r)
        for d in (w.generations, w.budgets, w.restores):
          d.pop(r, None)
        gen_bumps |= assign_job(s, r, nw := next_worker(), new_gen=True)
        w.revision += 1
        out.append((Ev.MIGRATED, {"run": r, "from": name, "worker": nw}))
    if w.node and obs and obs.phase == "failed" and is_released(obs):
      del s.observations[name]
      unplace(name, keep_node=True)
      bump_jobs(name)
    elif (
      w.intent == "drain"
      and w.drain_mode in ("export", "close")
      and obs
      and obs.revision >= w.revision
      and is_released(obs)
    ):
      del s.observations[name]
      old_node = w.node
      unplace(name)
      bump_jobs(name)
      out.append((Ev.UNPLACED, {"worker": name, "from": old_node, "rule": "drain.exported"}))

  # 3. Single-job migration requests out of shared workers (fencing only the moved job).
  for r, job in sorted(s.jobs.items()):
    w = s.workers.get(job.worker) if job.worker and job.target_node else None
    if (
      not w
      or not w.node
      or w.node == job.target_node
      or not node_eligible(s, job.target_node, job.spec, job.gpu_classes)
    ):
      continue
    if not [x for x in jobs_of(s, job.worker) if x != r]:
      if w.intent != "drain":
        drain(job.worker, "export", to=job.target_node)
    elif job.spec.role == Role.SAMPLER:
      old_w = job.worker
      w.generations.pop(r, None)
      w.budgets.pop(r, None)
      w.revision += 1
      gen_bumps |= assign_job(s, r, nw := (shareable_worker(s, job) or next_worker()), new_gen=True)
      out.append((Ev.MIGRATED, {"run": r, "from": old_w, "worker": nw}))
    elif r not in w.draining_jobs:
      w.draining_jobs.append(r)
      w.budgets.pop(r, None)
      w.revision += 1
      out.append((Ev.DRAINING, {"worker": job.worker, "run": r, "mode": "job_export", "to": job.target_node}))

  # 4. Assign unassigned jobs to shared or dedicated workers and retire empty workers.
  for r, job in sorted(s.jobs.items()):
    if not job.worker:
      gen_bumps |= assign_job(s, r, worker := (shareable_worker(s, job) or next_worker()), new_gen=True)
      if s.workers[worker].node:
        s.workers[worker].revision += 1
      out.append((Ev.ASSIGNED, {"run": r, "worker": worker, "generation": job.generation}))
    elif (w := s.workers[job.worker]).generations.get(r) != job.generation and r not in w.draining_jobs:
      w.generations[r] = job.generation
      w.revision += 1
  for name, w in sorted(s.workers.items()):
    live_members = set(jobs_of(s, name))
    if stale := [r for r in w.generations if r not in live_members]:
      for r in stale:
        for d in (w.generations, w.budgets, w.restores):
          d.pop(r, None)
      if live_members:
        w.revision += 1
    if not live_members:
      if not is_released(s.observations.get(name)) and w.node in s.nodes:
        bump_rev(w, now, "drain", "close")
        w.budgets, w.predicted = {}, {}
      else:
        if w.node:
          out.append((Ev.UNPLACED, {"worker": name, "rule": "place.emptied"}))
        del s.workers[name]
        s.observations.pop(name, None)

  # 5. Enforce exclusive admission cap and place unplaced workers onto eligible (node, bundle).
  admitted_excl = exclusive_admitted(s)
  unplaced = [
    n for n, w in s.workers.items() if w.node is None and jobs_of(s, n) and (not w.spec.exclusive or n in admitted_excl)
  ]
  for name in sorted(
    unplaced, key=lambda n: (not s.workers[n].spec.exclusive, worker_priority(s, n, floor), -s.workers[n].spec.gpus, n)
  ):
    node, bundle, _, _ = best_placement(s, name)
    if node is not None:
      w = s.workers[name]
      w.node, w.gpus, w.intent, w.drain_mode, w.revision = (
        node,
        (bundle if bundle_free(s, node, bundle, name) else ()),
        "idle",
        "",
        w.revision + 1,
      )
      out.append((Ev.PLACED, {"worker": name, "node": node, "gpus": list(bundle), "rule": "place.makespan"}))

  # 6. Makespan-guided migration when a free destination beats transfer + restore hysteresis.
  for name, w in sorted(s.workers.items(), key=lambda kv: (worker_priority(s, kv[0], floor), kv[0])):
    if not w.node or w.spec.exclusive or w.intent == "drain" or w.draining_jobs or not worker_ready(s, name):
      continue
    members = jobs_of(s, name)
    if w.gpus and bundle_free(s, w.node, w.gpus, name) and not (len(members) == 1 and s.jobs[members[0]].target_node):
      continue
    if not (
      cur_b := w.gpus
      or (estimator.eligible_bundles(s.nodes[w.node].gpu_ids, w.spec.gpus, s.nodes[w.node].bundles) or [()])[0]
    ):
      continue
    cur_ms, _ = candidate_makespan(s, name, w.node, cur_b, TURN)
    cand_node, cand_b, cand_ms, xfer = best_placement(s, name)
    if (
      cand_node
      and cand_node != w.node
      and bundle_free(s, cand_node, cand_b, name)
      and cur_ms - cand_ms > xfer + RESTORE_S
    ):
      if is_released(s.observations.get(name)):
        old, w.node, w.gpus, w.intent, w.drain_mode, w.revision = w.node, cand_node, cand_b, "idle", "", w.revision + 1
        bump_jobs(name)
        out.append((Ev.MIGRATED, {"worker": name, "from": old, "node": cand_node, "gpus": list(cand_b)}))
      else:
        drain(name, "export", to=cand_node)

  # 7. Host-memory pressure: export parked workers when offloaded RAM exceeds node.host_bytes.
  for node, info in sorted(s.nodes.items()):
    if info.host_bytes <= 0:
      continue
    parked = [
      (n, o)
      for n, o in s.observations.items()
      if o.node == node and o.gpu_state == "offloaded" and n in s.workers and s.workers[n].intent != "drain"
    ]
    total_host = sum(o.host_bytes for _, o in parked)
    for name, obs in sorted(parked, key=lambda item: (-worker_priority(s, item[0], floor), item[1].at)):
      if total_host <= info.host_bytes:
        break
      drain(name, "export", reason="host_memory")
      total_host -= obs.host_bytes

  # 8. Anti-starvation bundle reservation for wide and admitted exclusive workers.
  for node, info in sorted(s.nodes.items()):
    if node in s.reserved:
      rw, rb = s.reserved[node]["worker"], tuple(s.reserved[node]["gpus"])
      if bundle_free(s, node, rb, rw, ignore_res=True):
        del s.reserved[node]
        assign_budgets(s, rw, node, now)
        bump_rev(s.workers[rw], now, "run" if worker_ready(s, rw) else "idle", "", rb)
      else:
        active, _ = held_gpus(s, node, exclude=rw)
        for h in sorted(
          {
            x
            for g in rb
            for x in active.get(g, ())
            if x in s.workers and s.workers[x].intent != "drain" and not s.workers[x].spec.exclusive
          }
        ):
          drain(h, "park", **{"for": rw})
      continue
    local_ready = [n for n, w in s.workers.items() if w.node == node and worker_ready(s, n) > 0 and w.intent != "drain"]
    min_prio = min((worker_priority(s, n, floor) for n in local_ready), default=0.0)
    wide = [
      n
      for n in local_ready
      if (s.workers[n].spec.gpus > 1 or s.workers[n].spec.exclusive)
      and not (s.workers[n].gpus and bundle_free(s, node, s.workers[n].gpus, n))
    ]
    for wn in sorted(
      wide, key=lambda n: (not s.workers[n].spec.exclusive, worker_priority(s, n, floor), -s.workers[n].spec.gpus, n)
    ):
      bundles = estimator.eligible_bundles(info.gpu_ids, s.workers[wn].spec.gpus, info.bundles)
      if free := next((b for b in bundles if bundle_free(s, node, b, wn)), None):
        assign_budgets(s, wn, node, now)
        bump_rev(s.workers[wn], now, "run", "", free)
        continue
      waited = now - min((s.jobs[r].created for r in jobs_of(s, wn)), default=now)
      if bundles and (
        s.workers[wn].spec.exclusive or worker_priority(s, wn, floor) <= min_prio + 1e-6 or waited >= STARVE_S
      ):
        active, _ = held_gpus(s, node, exclude=wn)
        if cands := [
          b
          for b in bundles
          if not any(h in s.workers and s.workers[h].spec.exclusive for g in b for h in active.get(g, ()))
        ]:
          chosen = min(cands, key=lambda b: (len({h for g in b for h in active.get(g, ())}), b))
          s.reserved[node] = {"worker": wn, "gpus": list(chosen), "since": now}
          out.append((Ev.RESERVED, {"node": node, "worker": wn, "gpus": list(chosen)}))
          for h in sorted(
            {x for g in chosen for x in active.get(g, ()) if x in s.workers and s.workers[x].intent != "drain"}
          ):
            drain(h, "park", **{"for": wn})
          break

  # 9. Per-node GPU turn arbitration & bounded work grants in fair-service order.
  for node, info in sorted(s.nodes.items()):
    here = [n for n, w in s.workers.items() if w.node == node and jobs_of(s, n)]
    for name in here:
      w = s.workers[name]
      if w.intent == "drain" and w.drain_mode == "park" and is_parked_or_released(w, s.observations.get(name)):
        w.intent, w.drain_mode, w.budgets, w.predicted, w.gpus = "idle", "", {}, {}, ()
    ordered = sorted(
      here, key=lambda n: (not s.workers[n].spec.exclusive, waiting_on_io(s, n), worker_priority(s, n, floor), n)
    )
    wanting = [n for n in ordered if worker_ready(s, n) > 0 and s.workers[n].intent != "drain"]
    for name in here:
      w, obs = s.workers[name], s.observations.get(name)
      if w.intent == "drain" or not w.gpus:
        continue
      ready_n, my_prio, stalled = worker_ready(s, name), worker_priority(s, name, floor), waiting_on_io(s, name)
      contenders = [
        c
        for c in wanting
        if c != name
        and not (s.workers[c].gpus and s.workers[c].intent == "run")
        and not waiting_on_io(s, c)
        and (s.workers[c].spec.exclusive or stalled or worker_priority(s, c, floor) <= my_prio or ready_n == 0)
        and not any(
          bundle_free(s, node, b, c)
          for b in estimator.eligible_bundles(info.gpu_ids, s.workers[c].spec.gpus, info.bundles)
        )
      ]
      if contenders and not w.spec.exclusive and turn_done(s, name, obs, ready_n):
        drain(name, "park", **{"for": contenders[0]})
      elif ready_n == 0:
        w.budgets, w.predicted = {}, {}
        bump_rev(w, now, "idle", "")
      elif not stalled and (
        w.intent == "idle"
        or turn_done(s, name, obs, ready_n)
        or any(r not in w.budgets for r in jobs_of(s, name) if s.jobs[r].queued > 0 and r not in w.draining_jobs)
      ):
        assign_budgets(s, name, node, now)
        w.revision, w.intent, w.drain_mode = w.revision + 1, "run", ""
    for name in ordered:
      w, ready_n = s.workers[name], worker_ready(s, name)
      if w.intent == "drain" or w.gpus or (ready_n == 0 and not w.spec.exclusive) or waiting_on_io(s, name):
        continue
      bundles = estimator.eligible_bundles(info.gpu_ids, w.spec.gpus, info.bundles)
      if (obs := s.observations.get(name)) and obs.gpus in bundles:
        bundles = [obs.gpus, *(b for b in bundles if b != obs.gpus)]
      if free := next((b for b in bundles if bundle_free(s, node, b, name)), None):
        assign_budgets(s, name, node, now) if ready_n > 0 else (w.budgets.clear(), w.predicted.clear())
        bump_rev(w, now, "run" if ready_n > 0 else "idle", "", free)
      elif ready_n > 0:
        active, _ = held_gpus(s, node, exclude=name)
        for b in bundles:
          if (hs := [h for g in b for h in active.get(g, ()) if h in s.workers]) and all(
            not s.workers[h].spec.exclusive and s.workers[h].intent == "idle" for h in hs
          ):
            for h in dict.fromkeys(hs):
              drain(h, "park", **{"for": name})
            break

  update_prepare(s)
  return s, out, gen_bumps


# ***** persistence & loop *****


def version_refs(s: Assignment) -> set[tuple[str, str]]:
  return {(v, f"assign:{n}") for n, w in s.workers.items() for v in [*w.prepare, *w.restores.values()] if v}


async def commit(
  bus: Bus, old: Assignment, new: Assignment, changes: list, gen_bumps: dict[str, int], holder: str | None = None
) -> bool:
  old_w, new_w = (
    {n: worker_to_dict(w) for n, w in old.workers.items()},
    {n: worker_to_dict(w) for n, w in new.workers.items()},
  )
  assigned = {r: x.worker for r, x in new.jobs.items() if r not in old.jobs or x.worker != old.jobs[r].worker}
  gone, obs_gone = sorted(old_w.keys() - new_w.keys()), sorted(old.observations.keys() - new.observations.keys())
  settled_gone = sorted(set(old.settled) - set(new.workers) - set(new.observations) - {"__ready__"})
  old_refs, new_refs = version_refs(old), version_refs(new)
  dirty_w = {n: d for n, d in new_w.items() if old_w.get(n) != d}
  svc_dirty = {k: v for k, v in new.service.items() if old.service.get(k) != v}
  set_dirty = {k: v for k, v in new.settled.items() if old.settled.get(k) != v}
  res_dirty = new.reserved if new.reserved != old.reserved else None
  dirty = (changes, gen_bumps, assigned, dirty_w, gone, obs_gone, svc_dirty, set_dirty, settled_gone)
  if not any((*dirty, res_dirty is not None, old_refs != new_refs)):
    return False
  kw = {
    "generations": gen_bumps,
    "service": svc_dirty,
    "settled": set_dirty,
    "settled_gone": settled_gone,
    "reserved": res_dirty,
  }
  kw |= {
    "obs_gone": obs_gone,
    "ref_adds": sorted(new_refs - old_refs),
    "ref_dels": sorted(old_refs - new_refs),
    "scheduler_owner": holder,
  }
  return await bus.write_table(assigned, dirty_w, gone, new.workers_made, changes, **kw)


async def step(bus: Bus, holder: str | None = None, *, expire: bool = False) -> bool:
  """Run one reconciled scheduler pass; if `holder` is set, only the active lease owner may publish."""
  if holder and await bus.claim_scheduler(holder, LEASE_MS) is None:
    return False
  if expire:
    registry = Registry(bus)
    for session in await bus.expired_sessions():
      await bus.end_session(session)
    if expired := await registry.expired_checkpoints():
      await registry.unregister_checkpoints(*expired)
  s = from_table(await bus.read_table())
  new, changes, gen_bumps = decide(s)
  return await commit(bus, s, new, changes, gen_bumps, holder)


async def run(bus: Bus, holder: str = "") -> None:
  holder = holder or f"sched-{socket.gethostname()}-{os.getpid()}-{os.urandom(4).hex()}"
  try:
    while True:
      await step(bus, holder, expire=True)
      await bus.read_events("$", count=1, block_ms=TICK_MS)
  finally:
    await bus.release_scheduler(holder)


if __name__ == "__main__":
  asyncio.run(run(Bus()))
