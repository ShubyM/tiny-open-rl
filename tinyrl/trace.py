"""NVTX ranges, for a timeline of tinyrl's steps next to its CUDA kernels in Nsight Systems. Without a profiler attached, a range costs about a microsecond."""

from contextlib import contextmanager

try:
  import nvtx
except ImportError:  # macOS, which has no nvtx wheels
  nvtx = None


@contextmanager
def span(name: str):
  if nvtx is None:
    yield
    return
  # start and end, not push and pop: asyncio tasks interleave on a thread
  token = nvtx.start_range(name, domain="tinyrl")
  try:
    yield
  finally:
    nvtx.end_range(token)
