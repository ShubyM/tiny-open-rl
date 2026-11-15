"""The fleet owns GPU bundles and bounded job grants. Nodes execute and acknowledge them."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import itertools
import json
import secrets
from dataclasses import dataclass, field

from tinyrl.bus import Bus, Role, key
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry

TURN, TICK_MS = getenv("TURN", 200_000), getenv("TICK_MS", 100)


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
  generation: int = 0
  account: str = ""
  target_node: str | None = None


@dataclass
class Worker:
  spec: WorkerSpec = WorkerSpec()
  node: str | None = None
  gpus: tuple[int, ...] = ()
  revision: int = 0
  budgets: dict[str, int] = field(default_factory=dict)
  members: dict[str, int] = field(default_factory=dict)
  owner: str = ""


@dataclass
class Assignment:
  nodes: dict[str, dict] = field(default_factory=dict)
  workers: dict[str, Worker] = field(default_factory=dict)
  jobs: dict[str, Job] = field(default_factory=dict)
  observations: dict[str, dict] = field(default_factory=dict)
  owners: dict[str, str] = field(default_factory=dict)
  service: dict[str, float] = field(default_factory=dict)
  settled: dict[str, dict] = field(default_factory=dict)


def jobs_of(s: Assignment, worker: str) -> list[str]:
  return [r for r, j in s.jobs.items() if j.worker == worker]


def from_table(table: dict) -> Assignment:
  s = Assignment(**{k: table.get(k, {}) for k in ("nodes", "observations", "owners", "service", "settled")})
  for r, (rec, worker, st) in table["runs"].items():
    role = Role(rec.get("role", "trainer"))
    exclusive, train_run = rec.get("exclusive", False), rec.get("train_run")
    spec = WorkerSpec(
      role, rec.get("base", ""), rec.get("rank", 0) > 0, exclusive, train_run if exclusive else None, rec.get("gpus", 1)
    )
    s.jobs[r] = Job(
      spec,
      worker or "",
      st.get("generation", 0),
      train_run or r,
      st.get("target_node"),
    )
  s.jobs = {r: j for r, j in s.jobs.items() if not j.spec.train_run or j.spec.train_run in s.jobs}
  for n, d in table["workers"].items():
    spec = WorkerSpec(**d["spec"] | {"role": Role(d["spec"]["role"])})
    s.workers[n] = Worker(**d | {"spec": spec, "gpus": tuple(d.get("gpus", ()))})
  for j in s.jobs.values():
    if j.worker:
      s.workers.setdefault(j.worker, Worker(j.spec))
  return s


def observed(s: Assignment, name: str) -> dict:
  w, o = s.workers[name], s.observations.get(name, {})
  owner = s.owners.get(name)
  return o if o.get("node") == w.node and (o.get("owner") == owner or (not owner and o.get("closed"))) else {}


def bundles(info: dict, width: int) -> list[tuple[int, ...]]:
  ids = tuple(info.get("gpu_ids", range(info["gpus"])))
  return sorted(
    {
      tuple(b)
      for group in info.get("bundles") or [ids]
      for b in itertools.combinations(group, width)
      if set(b) <= set(ids)
    }
  )


def decide(state: Assignment) -> Assignment:
  s = copy.deepcopy(state)
  obs = {n: observed(s, n) for n in s.workers}
  for n, o in obs.items():
    if not o:
      continue
    prev = s.settled.get(n, {})
    used = prev.get("usage", {}) if prev.get("owner") == o.get("owner") else {}
    for account, total in o.get("usage", {}).items():
      s.service[account] = s.service.get(account, 0.0) + max(0.0, total - used.get(account, 0.0))
      used[account] = max(total, used.get(account, 0.0))
    s.settled[n] = {"owner": o.get("owner"), "usage": used}

  def change(w: Worker, **values) -> None:
    if any(getattr(w, k) != v for k, v in values.items()):
      for k, v in values.items():
        setattr(w, k, v)
      w.revision += 1

  def released(n: str) -> bool:
    o, w = obs.get(n, {}), s.workers[n]
    return bool(o.get("released") and o.get("revision") == w.revision and o.get("owner") == w.owner)

  def advance(n: str) -> None:
    s.workers[n].revision += 1
    for r in jobs_of(s, n):
      s.jobs[r].generation += 1
      if r in s.workers[n].members:
        s.workers[n].members[r] = s.jobs[r].generation

  for n, w in s.workers.items():
    o = obs[n]
    if w.node and w.node not in s.nodes:
      change(w, node=None, gpus=(), budgets={}, owner="")
      advance(n)
      obs[n] = {}
    elif o and s.owners.get(n) and o["owner"] != w.owner:
      if w.owner:
        advance(n)
      change(w, owner=o["owner"], budgets={})
    elif o.get("closed") and o.get("revision") == w.revision and w.members and (released(n) or not w.owner):
      advance(n)
      change(w, owner="", gpus=(), budgets={})
  for r, j in s.jobs.items():
    if not j.worker:
      continue
    w, o = s.workers[j.worker], obs.get(j.worker, {})
    moving = (
      w.node and j.target_node in s.nodes and w.node != j.target_node and bundles(s.nodes[j.target_node], j.spec.gpus)
    )
    if not moving:
      change(w, members=w.members | {r: j.generation})
    else:
      if r in w.members:
        change(
          w,
          members={k: v for k, v in w.members.items() if k != r},
          budgets={k: v for k, v in w.budgets.items() if k != r},
        )
      elif o.get("revision") == w.revision and o.get("owner") == w.owner and r not in o.get("resident", {}):
        j.worker = ""
  for r, j in s.jobs.items():
    if j.worker:
      continue
    sharing = j.spec.role == Role.SAMPLER or (j.spec.lora and not j.spec.exclusive)
    name = next(
      (
        n
        for n, w in s.workers.items()
        if sharing and w.spec == j.spec and (not j.target_node or w.node in (None, j.target_node))
      ),
      None,
    )
    if name is None:
      s.workers[name := f"w:{r}:{j.generation + 1}"] = Worker(j.spec)
    j.worker, j.generation = name, j.generation + 1
    change(s.workers[name], members=s.workers[name].members | {r: j.generation})
  exports = {}
  for n, w in list(s.workers.items()):
    members = {r: g for r, g in w.members.items() if r in s.jobs and s.jobs[r].worker == n}
    resident = set(obs.get(n, {}).get("resident", {}))
    leaving = (resident - members.keys()) | (resident & set(obs.get(n, {}).get("export", ())))
    exports[n] = set(jobs_of(s, n)) & leaving
    change(w, members=members, budgets={r: b for r, b in w.budgets.items() if r in members or r in exports[n]})
    if not jobs_of(s, n) and (
      not w.node or (not w.owner and not s.owners.get(n)) or (released(n) and obs[n].get("closed"))
    ):
      del s.workers[n]
      s.settled.pop(n, None)
    elif not w.node and members:
      target = next((s.jobs[r].target_node for r in members if s.jobs[r].target_node in s.nodes), None)
      eligible = [k for k, v in s.nodes.items() if bundles(v, w.spec.gpus) and (not target or k == target)]
      if eligible:
        node = min(eligible, key=lambda k: (sum(x.node == k for x in s.workers.values()) / s.nodes[k]["gpus"], k))
        change(w, node=node)

  ready_jobs = {
    n: sorted((w.members.keys() & set(obs.get(n, {}).get("ready", ()))) | exports[n]) for n, w in s.workers.items()
  }
  ready = {s.jobs[r].account for runs in ready_jobs.values() for r in runs}
  previous = s.settled.get("ready", {})
  continuing = ready & set(previous.get("accounts", ()))
  floor = max(
    previous.get("floor", 0.0),
    min((s.service.get(a, 0.0) for a in continuing or previous.get("accounts", ())), default=0.0),
  )
  for a in ready:
    s.service[a] = max(s.service.get(a, 0.0), floor) if a not in continuing else s.service.get(a, 0.0)
  s.settled["ready"] = {"accounts": sorted(ready), "floor": floor}
  s.service = {a: s.service[a] for a in {j.account for j in s.jobs.values()} & s.service.keys()}

  def priority(n):
    return min((s.service.get(s.jobs[r].account, floor) for r in ready_jobs[n]), default=float("inf"))

  unfinished = {
    n
    for n, w in s.workers.items()
    if w.budgets
    and ready_jobs[n]
    and (
      obs.get(n, {}).get("revision") != w.revision
      or any(obs[n].get("served", {}).get(r, 0) < b for r, b in w.budgets.items() if r in ready_jobs[n])
    )
  }
  actual = {n: (w.node, w.gpus) for n, w in s.workers.items() if w.gpus and (w.budgets or not released(n))}
  actual.update({n: (o["node"], tuple(o["gpus"])) for n, o in obs.items() if o.get("gpus") and not o.get("released")})
  cached = {n: (o["node"], tuple(o["gpus"])) for n, o in obs.items() if o.get("gpus") and not o.get("closed")}
  accounts = {n: next((s.jobs[r].account for r in jobs_of(s, n)), n) for n in s.workers}
  reserved = {(w.node, g): accounts[n] for n, w in s.workers.items() if w.spec.exclusive for g in w.gpus}
  planned, granted = {}, set()

  def available(node: str, bundle: tuple[int, ...], name: str, holders: dict) -> bool:
    return all(n == name or at != node or set(bundle).isdisjoint(gs) for n, (at, gs) in holders.items())

  # Fair order reserves a blocked gang's whole bundle until old grants finish. There is no separate reservation policy.
  for n in sorted(
    s.workers,
    key=lambda n: (not s.workers[n].spec.exclusive, priority(n), sum(obs.get(n, {}).get("usage", {}).values()), n),
  ):
    w, o = s.workers[n], obs.get(n, {})
    if not w.node or (not ready_jobs[n] and not w.spec.exclusive):
      continue
    choices = bundles(s.nodes[w.node], w.spec.gpus)
    if o.get("gpus") and not o.get("closed"):
      choices = [tuple(o["gpus"])]
    if w.spec.exclusive:
      foreign = {
        k: v for k, v in (actual | cached).items() if not s.workers[k].spec.exclusive or accounts[k] != accounts[n]
      }
      choices = [b for b in choices if available(w.node, b, n, foreign)]
    choices.sort(
      key=lambda b: (
        sum(reserved.get((w.node, g)) != accounts[n] for g in b),
        not available(w.node, b, n, actual),
        b != w.gpus,
        b,
      )
    )
    bundle = next(
      (
        b
        for b in choices
        if available(w.node, b, n, planned)
        and all((w.node, g) not in reserved or (w.spec.exclusive and reserved[w.node, g] == accounts[n]) for g in b)
      ),
      None,
    )
    if bundle is None:
      continue
    if w.spec.exclusive:
      reserved.update({(w.node, g): accounts[n] for g in bundle})
    if not ready_jobs[n]:
      change(w, gpus=bundle, budgets={})
      continue
    planned[n] = (w.node, bundle)
    if n in unfinished:
      granted.add(n)
    elif available(w.node, bundle, n, actual):
      selected = [r for r in ready_jobs[n] if s.service.get(s.jobs[r].account, floor) == priority(n)]
      served = {} if o.get("closed") else o.get("served", {})
      budgets = {r: served.get(r, 0) + max(1, TURN // len(selected)) for r in selected}
      change(w, gpus=bundle, budgets=budgets)
      granted.add(n)
      actual[n] = (w.node, bundle)
  for n, w in s.workers.items():
    if n not in granted and n not in unfinished:
      if w.budgets:
        change(w, budgets={})
      elif released(n) and not w.spec.exclusive:
        change(w, gpus=())
  return s


async def commit(bus: Bus, old: Assignment, new: Assignment, holder: str | None = None) -> bool:
  jobs = {
    r: j
    for r, j in new.jobs.items()
    if r not in old.jobs or (j.worker, j.generation) != (old.jobs[r].worker, old.jobs[r].generation)
  }
  workers = {n: dataclasses.asdict(w) for n, w in new.workers.items() if n not in old.workers or w != old.workers[n]}
  gone = sorted(old.workers.keys() - new.workers.keys())
  if not any((jobs, workers, gone, old.service != new.service, old.settled != new.settled)):
    return False
  owner_keys = [key("owner", n) for n in old.owners]
  rkeys = {r: key("run", r) for r in jobs}

  async def step(p):
    if holder and await p.get(key("owner", "scheduler")) != holder.encode():
      return False, False
    owners = await p.mget(owner_keys) if owner_keys else []
    if any((current or b"").decode() != owner for current, owner in zip(owners, old.owners.values())):
      return False, False
    observed = await p.hgetall(key("obs"))
    for n in old.owners:
      current, previous = json.loads(observed.get(n.encode(), b"{}")), old.observations.get(n, {})
      if any(
        current.get(k) != previous.get(k)
        for k in ("node", "owner", "revision", "gpus", "released", "closed", "resident")
      ):
        return False, False
    states = {r: json.loads(raw) if (raw := await p.hget(k, "state")) else None for r, k in rkeys.items()}
    if any(st is None for st in states.values()):
      return False, False
    p.multi()
    for r, j in jobs.items():
      p.hset(rkeys[r], mapping={"worker": j.worker, "state": json.dumps(states[r] | {"generation": j.generation})})
    p.delete(key("service"), key("settled"))
    for table, values in (("workers", workers), ("service", new.service), ("settled", new.settled)):
      if values:
        p.hset(key(table), mapping={k: json.dumps(v) for k, v in values.items()})
    if gone:
      for table in ("workers", "obs", "settled"):
        p.hdel(key(table), *gone)
    return True, True

  return await bus.tx([key("owner", "scheduler"), key("obs"), *rkeys.values(), *owner_keys], step)


async def run(bus: Bus) -> None:
  registry, holder = Registry(bus), secrets.token_hex(16)
  while True:
    await asyncio.sleep(TICK_MS / 1000)
    if not await bus.claim_scheduler(holder):
      continue
    for session in await bus.expired_sessions():
      await bus.end_session(session)
    await registry.unregister_checkpoints(*await registry.expired_checkpoints())
    old = from_table(await bus.read_table())
    await commit(bus, old, decide(old), holder)


if __name__ == "__main__":
  asyncio.run(run(Bus()))
