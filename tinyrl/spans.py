"""Spans: what each process spends its time on, and the node daemon's trace of them.

Wrap anything that takes time in `with span(name):`. Spans may nest and may end on any thread. A span records the worker whose task it ran in (`worker`, set by the node daemon when it starts serving a worker), so one process's spans sort by worker. A trainer child reports its spans with each reply's effects, and the node absorbs them under the worker it serves.

With TINYRL_TRACE set, the node daemon appends to the tinyrl:trace stream, once a tick: every event it folds (GPU turns, batches, idle, placements, which it never writes to tinyrl:events), the spans that ended, and samples of its event loop's lag. tools/trace.py draws them with the event log on one timeline.
"""

import asyncio
import collections
import contextlib
import contextvars
import functools
import inspect
import json
import os
import time

from tinyrl.helpers import getenv

# a folder: each trainer writes a trace of its CUDA kernels and copies there
PROFILE = getenv("TINYRL_PROFILE", "")
TRACE = getenv("TINYRL_TRACE", "")
worker: contextvars.ContextVar[str] = contextvars.ContextVar("worker", default="")
# spans that ended since the last report. A deque, so threads can append while report() drains
ended: collections.deque = collections.deque()


@contextlib.contextmanager
def span(name: str):
  start = time.time()
  try:
    yield
  finally:
    ended.append([name, round(start, 4), round(time.time() - start, 4), worker.get()])


def timed(fn):
  """Make every call of fn, a function or a coroutine function, a span named after it."""
  if inspect.iscoroutinefunction(fn):

    @functools.wraps(fn)
    async def awaited(*args, **kwargs):
      with span(fn.__name__):
        return await fn(*args, **kwargs)

    return awaited

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    with span(fn.__name__):
      return fn(*args, **kwargs)

  return wrapper


def report() -> list[list]:
  """The spans that ended since the last report."""
  return [ended.popleft() for _ in range(len(ended))]


def absorb(spans: list[list]) -> None:
  """Another process's spans, under this task's worker."""
  ended.extend([name, start, seconds, w or worker.get()] for name, start, seconds, w in spans)


class Trace:
  """What a node daemon exports to tinyrl:trace: the events it folds, the spans that ended, and its loop lag."""

  def __init__(self, node: str):
    self.node, self.events, self.lag = node, [], []

  def event(self, kind: str, t: float, body: dict) -> None:
    if TRACE:
      self.events.append((kind, {**body, "t": round(t, 4), "node": self.node}))

  async def watch_lag(self, every: float = 0.02) -> None:
    """How late the loop runs a timer, sampled every `every` seconds: while it is busy, every other task waits this long."""
    while True:
      before = time.monotonic()
      await asyncio.sleep(every)
      self.lag.append([round(time.time(), 3), round((time.monotonic() - before - every) * 1000, 1)])

  async def flush(self, r) -> None:
    entries, spans = self.events, report()  # drained even with tracing off, so the deque never grows
    self.events = []
    if not TRACE:
      return
    if spans:
      entries.append(("spans", {"node": self.node, "spans": spans}))
    if self.lag:
      entries.append(("lag", {"node": self.node, "samples": self.lag}))
      self.lag = []
    if entries:
      p = r.pipeline(transaction=False)
      for kind, body in entries:
        p.xadd("tinyrl:trace", {"kind": kind, "body": json.dumps(body)}, maxlen=2_000_000, approximate=True)
      await p.execute()


@contextlib.contextmanager
def profiled(name: str):
  """Record this process's CUDA kernels and copies into PROFILE/name-pid.json.gz when PROFILE is set. Spans say which command each kernel ran in, by time."""
  if not PROFILE:
    yield
    return
  import torch

  prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
  prof.start()
  try:
    yield
  finally:
    prof.stop()
    os.makedirs(PROFILE, exist_ok=True)
    prof.export_chrome_trace(f"{PROFILE}/{name}-{os.getpid()}.json.gz")
