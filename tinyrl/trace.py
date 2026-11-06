"""Optional Nsight ranges. Process ranges may overlap across asyncio tasks."""

from contextlib import contextmanager

from tinyrl.helpers import getenv


@contextmanager
def trace(name: str, request_id: str = ""):
  if not getenv("TINYRL_NVTX"):
    yield
    return
  import nvtx

  # Keep registered messages bounded; the same request bytes correlate parent and child.
  token = nvtx.start_range(name, domain="tinyrl", payload=request_id.encode() if request_id else None)
  try:
    yield
  finally:
    nvtx.end_range(token)
