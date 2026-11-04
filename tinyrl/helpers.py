import functools
import os
from typing import Any


@functools.cache
def getenv(key: str, default: Any = 0):
  """Read an environment variable, converted to the type of its default."""
  return type(default)(os.getenv(key, default))


DEBUG = getenv("DEBUG")
