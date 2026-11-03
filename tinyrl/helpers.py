import contextlib
import functools
import os
from typing import Any, TypeVar, overload

T = TypeVar("T")


@overload
def getenv(key: str) -> int: ...
@overload
def getenv(key: str, default: T) -> T: ...
@functools.cache
def getenv(key: str, default: Any = 0):
  """Read an environment variable, converted to the type of its default."""
  return type(default)(os.getenv(key, default))


DEBUG = getenv("DEBUG")
PROFILE = getenv(
  "TINYRL_PROFILE", ""
)  # a folder: each trainer and vLLM server writes a trace of its ops and CUDA kernels there


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
