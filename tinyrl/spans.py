"""Spans: what each process spends its time on. A span is a range in a TINYRL_PROFILE trace of ops and CUDA kernels, and, once reported with the process's next event, a [name, start, seconds] entry in the log, which tools/trace.py draws.

Wrap anything that takes time in `with span(name):`. Spans may nest and may end on any thread.
"""

import collections
import contextlib
import functools
import inspect
import os
import time

from tinyrl.helpers import getenv

PROFILE = getenv(
  "TINYRL_PROFILE", ""
)  # a folder: each trainer and vLLM server writes a trace of its ops and CUDA kernels there
ended: collections.deque = (
  collections.deque()
)  # spans that ended since the last report. A deque, so threads can append while report() drains


@contextlib.contextmanager
def span(name: str):
  start = time.time()
  try:
    if PROFILE:
      from torch.profiler import record_function

      with record_function(name):
        yield
    else:
      yield
  finally:
    ended.append([name, round(start, 4), round(time.time() - start, 4)])


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
  """The spans that ended since the last report, for the event the process appends next."""
  return [ended.popleft() for _ in range(len(ended))]


@contextlib.contextmanager
def profiled(name: str):
  """Record this process's ops and CUDA kernels, from every thread, into PROFILE/name-pid.json.gz when PROFILE is set."""
  if not PROFILE:
    yield
    return
  import torch
  from torch._C._profiler import _ExperimentalConfig

  activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
  prof = torch.profiler.profile(
    activities=activities, experimental_config=_ExperimentalConfig(profile_all_threads=True)
  )
  prof.start()
  try:
    yield
  finally:
    prof.stop()
    os.makedirs(PROFILE, exist_ok=True)
    prof.export_chrome_trace(f"{PROFILE}/{name}-{os.getpid()}.json.gz")
