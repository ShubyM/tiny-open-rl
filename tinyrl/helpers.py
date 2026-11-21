import functools
import os
import shutil
import tempfile
from typing import Any


@functools.cache
def getenv(key: str, default: Any = 0):
  """Read an environment variable, converted to the type of its default."""
  return type(default)(os.getenv(key, default))


def atomic_dir(folder: str, fill) -> str:
  """Populate `folder` atomically via a sibling tempdir if it does not already exist."""
  if not os.path.isdir(folder):
    os.makedirs(parent := os.path.dirname(folder), exist_ok=True)
    tmp = tempfile.mkdtemp(dir=parent)
    try:
      fill(tmp)
      try:
        os.rename(tmp, folder)
      except OSError:
        if not os.path.isdir(folder):
          raise
    finally:
      shutil.rmtree(tmp, ignore_errors=True)
  return folder


DEBUG = getenv("DEBUG")
