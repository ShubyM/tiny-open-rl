"""Immutable request/result bytes outside Redis and the worker control protocol.

The coordinator and workers share this directory. Content hashes make retries
reuse the same durable blob. Referenced blobs must be retained; there is no
age-based garbage collection that could delete an input still waiting in a queue.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from tinyrl.helpers import make_directory, sync_directory

INLINE_BYTES = 64 * 1024


class PayloadStore:
  def __init__(self, root: str | os.PathLike | None = None):
    self.root = Path(root or os.getenv("PAYLOAD_DIR", f"{os.getenv('CKPT', '/tmp/tinyrl')}/payloads"))
    make_directory(self.root)

  def path(self, ref: str) -> Path:
    if len(ref) != 64 or any(c not in "0123456789abcdef" for c in ref):
      raise ValueError("invalid payload content hash")
    return self.root / f"{ref}.blob"

  def put(self, body: bytes) -> str:
    ref = hashlib.sha256(body).hexdigest()
    target = self.path(ref)
    fd, temporary = tempfile.mkstemp(prefix=".payload-", dir=self.root)
    try:
      with os.fdopen(fd, "wb") as out:
        out.write(body)
        out.flush()
        os.fsync(out.fileno())
      try:
        os.link(temporary, target)
      except FileExistsError:
        self.get(ref, len(body))  # never acknowledge an existing corrupt blob
      os.unlink(temporary)
      sync_directory(self.root)
    finally:
      Path(temporary).unlink(missing_ok=True)
    return ref

  def get(self, ref: str, size: int) -> bytes:
    path = self.path(ref)
    if type(size) is not int or size < 0:
      raise ValueError("payload length does not match its reference")
    with path.open("rb") as source:
      if os.fstat(source.fileno()).st_size != size:
        raise ValueError("payload length does not match its reference")
      body = source.read(size + 1)  # bounded even if corruption grows the file after fstat
    if len(body) != size or hashlib.sha256(body).hexdigest() != ref:
      raise ValueError("payload content does not match its reference")
    return body
