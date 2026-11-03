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
