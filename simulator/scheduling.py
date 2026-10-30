"""Small, independent GPU-turn benchmark; no model kernels or scheduler implementation here.

One GPU serves integer units of compute, with fixed quanta, arrivals and finite demand.
Changing workers costs time; the first worker is already warm. A quantum may finish
early when its work runs out. Work is divisible at the observation horizon. This is
a service-allocation model, not a prediction of training convergence or request JCT.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Job:
  name: str
  arrival: int
  work: int
  weight: int = 1


@dataclass(frozen=True)
class Segment:
  kind: str
  start: int
  end: int
  job: str | None = None


@dataclass(frozen=True)
class Metrics:
  service: dict[str, int]
  reference: dict[str, float]
  normalized_service: dict[str, float]
  minimum_service: float
  jain: float
  useful_work: int
  switch_time: int
  idle_time: int
  maximum_wait: int
  unfinished: int


def validate(jobs: tuple[Job, ...], horizon: int, quantum: int, switch: int) -> None:
  if any(
    type(value) is not int
    for value in (horizon, quantum, switch, *(v for j in jobs for v in (j.arrival, j.work, j.weight)))
  ):
    raise ValueError("the reference model uses integer times, work and weights")
  if horizon <= 0 or quantum <= 0 or switch < 0:
    raise ValueError("positive horizon/quantum and nonnegative switch time required")
  if len({j.name for j in jobs}) != len(jobs):
    raise ValueError("job names must be unique")
  if any(j.arrival < 0 or j.work <= 0 or j.weight <= 0 for j in jobs):
    raise ValueError("nonnegative arrivals and positive work/weights required")


def fair_reference(jobs: tuple[Job, ...], horizon: int) -> dict[str, float]:
  """Ideal weighted processor sharing, with neither switching nor quantum delay.

  Arrivals and completions change the eligible population. Unlike raw cumulative
  service, this denominator does not penalize a job simply for arriving late.
  """
  remaining = [float(j.work) for j in jobs]
  now = 0.0
  while now < horizon:
    active = [i for i, j in enumerate(jobs) if j.arrival <= now and remaining[i] > 1e-10]
    next_arrival = min((j.arrival for j in jobs if j.arrival > now), default=horizon)
    if not active:
      now = min(float(next_arrival), horizon)
      continue
    weight = sum(jobs[i].weight for i in active)
    duration = min(horizon - now, next_arrival - now, *(remaining[i] * weight / jobs[i].weight for i in active))
    for i in active:
      remaining[i] = max(0.0, remaining[i] - duration * jobs[i].weight / weight)
    now += duration
  return {j.name: j.work - remaining[i] for i, j in enumerate(jobs)}


def measure(jobs: tuple[Job, ...], segments: tuple[Segment, ...], horizon: int) -> Metrics:
  """Include unserved eligible jobs and unfinished waiting intervals in every report."""
  service = {j.name: 0 for j in jobs}
  last = {j.name: min(j.arrival, horizon) for j in jobs}
  maximum_wait = 0
  for part in segments:
    if part.kind == "compute":
      maximum_wait = max(maximum_wait, part.start - last[part.job])
      service[part.job] += part.end - part.start
      last[part.job] = part.end
  for j in jobs:
    if service[j.name] < j.work:
      maximum_wait = max(maximum_wait, horizon - last[j.name])
  reference = fair_reference(jobs, horizon)
  normalized = {name: service[name] / amount for name, amount in reference.items() if amount > 1e-10}
  values = list(normalized.values())
  squares = sum(v * v for v in values)
  jain = sum(values) ** 2 / (len(values) * squares) if squares else 0.0
  return Metrics(
    service,
    reference,
    normalized,
    min(values, default=0.0),
    jain,
    sum(service.values()),
    sum(p.end - p.start for p in segments if p.kind == "switch"),
    sum(p.end - p.start for p in segments if p.kind == "idle"),
    maximum_wait,
    sum(j.arrival < horizon and service[j.name] < j.work for j in jobs),
  )


# A policy chooses a ready job index, or None to idle for one integer time unit.
Policy = Callable[[int, tuple[int, ...], int | None], int | None]


def simulate(jobs: tuple[Job, ...], horizon: int, quantum: int, switch: int, policy: Policy) -> tuple[Segment, ...]:
  validate(jobs, horizon, quantum, switch)
  now, last, remaining, trace = 0, None, [j.work for j in jobs], []
  while now < horizon:
    chosen = policy(now, tuple(remaining), last)
    if chosen is None:
      trace.append(Segment("idle", now, now + 1))
      now += 1
      continue
    if chosen not in range(len(jobs)) or jobs[chosen].arrival > now or not remaining[chosen]:
      raise ValueError("policy selected an unavailable job")
    if last is not None and last != chosen and switch:
      end = min(now + switch, horizon)
      trace.append(Segment("switch", now, end, jobs[chosen].name))
      now = end
    end = min(now + quantum, now + remaining[chosen], horizon)
    if end > now:
      trace.append(Segment("compute", now, end, jobs[chosen].name))
      remaining[chosen] -= end - now
    now, last = end, chosen
  return tuple(trace)


def fifo_policy(jobs: tuple[Job, ...]) -> Policy:
  """FIFO requests, requeued after each quantum: the former serial GPU-turn baseline."""
  queue, seen = [], set()
  previous = None

  def choose(now: int, remaining: tuple[int, ...], last: int | None) -> int | None:
    nonlocal previous
    arriving = sorted((j.arrival, i) for i, j in enumerate(jobs) if j.arrival <= now and i not in seen)
    queue.extend(i for _, i in arriving)
    seen.update(i for _, i in arriving)
    if previous is not None and remaining[previous]:
      queue.append(previous)
    previous = queue.pop(0) if queue else None
    return previous

  return choose


def exact_plan(jobs: tuple[Job, ...], horizon: int, quantum: int, switch: int) -> tuple[int | None, ...]:
  """Exact finite-horizon optimum: max minimum normalized service, then total work.

  Enumerates all legal choices, including idling. This oracle knows future arrivals;
  it is a benchmark bound, not an online policy or an optimality claim for production.
  """
  validate(jobs, horizon, quantum, switch)
  if len(jobs) > 4 or horizon > 12:
    raise ValueError("exact oracle is limited to four jobs and twelve time units")
  reference = fair_reference(jobs, horizon)

  @lru_cache(None)
  def solve(now: int, remaining: tuple[int, ...], last: int | None) -> tuple[tuple[float, int], tuple[int | None, ...]]:
    if now >= horizon:
      work = [j.work - remaining[i] for i, j in enumerate(jobs)]
      ratios = [work[i] / reference[j.name] for i, j in enumerate(jobs) if reference[j.name] > 1e-10]
      return (min(ratios, default=0.0), sum(work)), ()
    score, tail = solve(now + 1, remaining, last)
    plan = (None, *tail)
    for i, job in enumerate(jobs):
      if job.arrival > now or not remaining[i]:
        continue
      start = min(horizon, now + (switch if last is not None and last != i else 0))
      amount = min(quantum, remaining[i], horizon - start)
      rest = list(remaining)
      rest[i] -= amount
      candidate, suffix = solve(start + amount, tuple(rest), i)
      if candidate > score:
        score, plan = candidate, (i, *suffix)
    return score, plan

  return solve(0, tuple(j.work for j in jobs), None)[1]


def replay(plan: tuple[int | None, ...]) -> Policy:
  choices = iter(plan)
  return lambda now, remaining, last: next(choices)


class SchedulerPolicy:
  """Adapt the real fold/decide to the same reference workload, without copying policy.

  Each run has an already-warm worker pinned to the one GPU. Releases are acknowledged
  immediately. Its service events charge compute plus command-associated switching,
  as the trainer does; the report still separates useful compute from overhead.
  Decisions happen at quantum boundaries, while arrivals retain their timestamps.
  """

  def __init__(self, jobs: tuple[Job, ...], quantum: int):
    from tinyrl import scheduler

    self.scheduler, self.jobs = scheduler, jobs
    self.state = scheduler.State(quantum=quantum)
    self.previous = tuple(j.work for j in jobs)
    self.previous_time = 0
    self.seen: set[int] = set()
    self.emit(0, "node_up", node="node", gpus=1, memory=80 * 1024**3)

  def emit(self, now: int, kind: str, **body) -> None:
    from tinyrl.bus import Event

    self.state = self.scheduler.fold(self.state, Event(str(self.state.n), now, kind, body))

  def __call__(self, now: int, remaining: tuple[int, ...], last: int | None) -> int | None:
    for arrival, i in sorted(
      (j.arrival, i) for i, j in enumerate(self.jobs) if j.arrival <= now and i not in self.seen
    ):
      job, worker = self.jobs[i], f"worker:{i}"
      self.emit(arrival, "run_created", run=job.name, base=job.name, rank=1, weight=job.weight)
      self.emit(arrival, "worker_created", worker=worker, node="node", gpus=[0], base=job.name, rank=1)
      self.emit(arrival, "run_assigned", run=job.name, worker=worker)
      self.emit(arrival, "started", worker=worker, node="node")
      self.emit(arrival, "run_queued", run=job.name, count=1)
      self.seen.add(i)
    for i in sorted(self.seen):
      job, worker = self.jobs[i], f"worker:{i}"
      if served := self.previous[i] - remaining[i]:
        # Between decisions there is either one selected job's switch + compute,
        # or an idle tick. Only the selected job's remaining work decreases.
        self.emit(now, "service", run=job.name, worker=worker, seconds=max(served, now - self.previous_time))
      self.emit(now, "run_queued", run=job.name, count=int(remaining[i] > 0))
      self.emit(now, "wants_gpu" if remaining[i] else "idle", worker=worker, turn=self.state.workers[worker].turn)
    self.previous = remaining
    self.previous_time = now
    # Runtime workers acknowledge revocation asynchronously. Here each acknowledgement
    # is immediate so this benchmark isolates allocation policy from release latency.
    for _ in range(2 * len(self.jobs) + 2):
      decisions = self.scheduler.decide(self.state, now)
      for kind, body in decisions:
        self.emit(now, kind, **body)
      revoked = [name for name, worker in self.state.workers.items() if worker.revoked]
      if not revoked:
        held = [i for i in sorted(self.seen) if self.state.workers[f"worker:{i}"].held and remaining[i]]
        if len(held) > 1:
          raise AssertionError("real scheduler granted overlapping GPU turns")
        return held[0] if held else None
      for worker in revoked:
        self.emit(now, "released", worker=worker, turn=self.state.workers[worker].turn)
        if any(remaining[i] and worker == f"worker:{i}" for i in self.seen):
          self.emit(now, "wants_gpu", worker=worker)
    raise AssertionError("scheduler decisions failed to settle")


def frozen_workloads() -> tuple[tuple[Job, ...], ...]:
  """Fixed evaluation seeds; changing these changes the benchmark, not just its noise."""
  return tuple(
    tuple(Job(f"j{i}", rng.randrange(4), rng.randrange(2, 9), rng.randrange(1, 3)) for i in range(3))
    for rng in (random.Random(seed) for seed in (17, 43, 109, 271, 811, 2027))
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--horizon", type=int, default=10)
  parser.add_argument(
    "--actual", action="store_true", help="also execute tinyrl.scheduler.fold/decide (needs tinyrl dependencies)"
  )
  args = parser.parse_args()
  rows = []
  for index, jobs in enumerate(frozen_workloads()):
    policies = [("fifo", fifo_policy(jobs)), ("oracle", replay(exact_plan(jobs, args.horizon, 2, 1)))]
    if args.actual:
      policies.append(("actual", SchedulerPolicy(jobs, 2)))
    for name, policy in policies:
      metrics = measure(jobs, simulate(jobs, args.horizon, 2, 1, policy), args.horizon)
      rows.append({"case": index, "policy": name, **asdict(metrics)})
  print(json.dumps(rows, indent=2))


if __name__ == "__main__":
  main()
