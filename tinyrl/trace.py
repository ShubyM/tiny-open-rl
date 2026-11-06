"""NVTX ranges, for a timeline of tinyrl's steps next to its CUDA kernels in Nsight Systems. Without a profiler attached, a range costs about a microsecond."""

from contextlib import contextmanager

import nvtx


@contextmanager
def span(name: str):
  token = nvtx.start_range(
    name, domain="tinyrl"
  )  # start and end, not push and pop: asyncio tasks interleave on a thread
  try:
    yield
  finally:
    nvtx.end_range(token)
