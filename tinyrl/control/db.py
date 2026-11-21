"""Transactional SQLite WAL store backing write-through `Table` dicts and raw request/reply blobs."""

from __future__ import annotations

import os
import sqlite3

import msgspec

_INIT = "CREATE TABLE IF NOT EXISTS kv (ns TEXT, k TEXT, v BLOB, PRIMARY KEY (ns, k))"
_PRAGMAS = (
  "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000; PRAGMA auto_vacuum=INCREMENTAL"
)


class DB:
  def __init__(self, path: str):
    if path != ":memory:":
      os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    self.conn = sqlite3.connect(path, isolation_level="IMMEDIATE")
    self.conn.executescript(f"{_PRAGMAS}; {_INIT}")

  def __enter__(self):
    return self.conn.__enter__()

  def __exit__(self, *exc):
    return self.conn.__exit__(*exc)

  def get_blob(self, key: str) -> bytes | None:
    row = self.conn.execute("SELECT v FROM kv WHERE ns='blobs' AND k=?", (key,)).fetchone()
    return bytes(row[0]) if row else None

  def put_blob(self, key: str, body: bytes) -> None:
    self.conn.execute("INSERT OR REPLACE INTO kv VALUES ('blobs', ?, ?)", (key, body))

  def del_blob(self, key: str) -> None:
    self.conn.execute("DELETE FROM kv WHERE ns='blobs' AND k=?", (key,))


class Table(dict):
  def __init__(self, db: DB, ns: str, cls=None):
    self.db, self.ns = db, ns
    dec = (lambda v: msgspec.msgpack.decode(v, type=cls)) if cls else msgspec.msgpack.decode
    super().__init__((k, dec(v)) for k, v in db.conn.execute("SELECT k, v FROM kv WHERE ns=?", (ns,)))

  def __setitem__(self, key: str, value) -> None:
    super().__setitem__(key, value)
    self.db.conn.execute("INSERT OR REPLACE INTO kv VALUES (?, ?, ?)", (self.ns, key, msgspec.msgpack.encode(value)))

  def __delitem__(self, key: str) -> None:
    super().__delitem__(key)
    self.db.conn.execute("DELETE FROM kv WHERE ns=? AND k=?", (self.ns, key))

  def pop(self, key: str, *default):
    if key in self:
      self.db.conn.execute("DELETE FROM kv WHERE ns=? AND k=?", (self.ns, key))
    return super().pop(key, *default)

  def save(self, key: str) -> None:
    self[key] = self[key]
