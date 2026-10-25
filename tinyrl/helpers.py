import functools
import os
from pathlib import Path
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


def sync_directory(path: str | os.PathLike) -> None:
  fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
  try:
    os.fsync(fd)
  finally:
    os.close(fd)


def make_directory(path: str | os.PathLike) -> None:
  """Persist each new parent entry before acknowledging files beneath it."""
  missing, current = [], Path(path).absolute()
  while not current.exists():
    missing.append(current)
    current = current.parent
  for directory in reversed(missing):
    directory.mkdir(exist_ok=True)
    sync_directory(directory.parent)
