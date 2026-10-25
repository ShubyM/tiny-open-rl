"""Content-addressed immutable request and reply blobs shared by coordinator and workers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from tinyrl.helpers import atomic_link, is_hex, make_directory, sync_directory

INLINE_BYTES = 64 * 1024


class PayloadStore:
  def __init__(self, root: str | os.PathLike | None = None):
    self.root = make_directory(root or os.getenv("PAYLOAD_DIR", f"{os.getenv('CKPT', '/tmp/tinyrl')}/payloads"))

  def path(self, ref: str) -> Path:
    if not is_hex(ref, 64): raise ValueError("invalid payload content hash")
    return self.root / f"{ref}.blob"

  def put(self, body: bytes) -> str:
    ref = hashlib.sha256(body).hexdigest()
    try: atomic_link(self.path(ref), lambda tmp: Path(tmp).write_bytes(body), ".payload-")
    except FileExistsError:
      self.get(ref, len(body))  # never acknowledge an existing corrupt blob
      sync_directory(self.root)
    return ref

  def get(self, ref: str, size: int) -> bytes:
    path = self.path(ref)
    if type(size) is not int or size < 0: raise ValueError("payload length does not match its reference")
    with path.open("rb") as f:
      if os.fstat(f.fileno()).st_size != size: raise ValueError("payload length does not match its reference")
      body = f.read(size + 1)  # bounded even if corruption grows the file after fstat
    if len(body) != size or hashlib.sha256(body).hexdigest() != ref:
      raise ValueError("payload content does not match its reference")
    return body
