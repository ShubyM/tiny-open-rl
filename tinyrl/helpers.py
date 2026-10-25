import functools
import os
import tempfile
from collections.abc import Callable
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


def is_hex(value: str, length: int) -> bool:
  return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def sync_directory(path: str | os.PathLike) -> None:
  fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
  try: os.fsync(fd)
  finally: os.close(fd)


def make_directory(path: str | os.PathLike) -> Path:
  """Persist each new parent entry before acknowledging files beneath it."""
  target = Path(path).absolute()
  missing, cur = [], target
  while not cur.exists():
    missing.append(cur)
    cur = cur.parent
  for d in reversed(missing):
    d.mkdir(exist_ok=True)
    sync_directory(d.parent)
  return target


def atomic_link(target: str | os.PathLike, write: Callable[[str], None], prefix: str = ".tmp-") -> None:
  """Write, fsync, and hard-link a new immutable file without ever overwriting `target`."""
  dest = Path(target)
  fd, tmp = tempfile.mkstemp(prefix=prefix, dir=dest.parent)
  os.close(fd)
  try:
    write(tmp)
    with open(tmp, "rb") as f: os.fsync(f.fileno())
    os.link(tmp, dest)
    os.unlink(tmp)
    sync_directory(dest.parent)
  finally: Path(tmp).unlink(missing_ok=True)
