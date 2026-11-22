"""Transactional SQLite WAL store backing rollback-safe write-through `Table` dicts and raw blobs."""

from __future__ import annotations

import os
import sqlite3

import msgspec

PRAGMAS = "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000; PRAGMA auto_vacuum=INCREMENTAL"
DDL = "CREATE TABLE IF NOT EXISTS kv (ns TEXT, k TEXT, v BLOB, PRIMARY KEY (ns, k))"


class DB:
  def __init__(self, path: str):
    if path != ":memory:":
      os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    self.conn, self.tables = sqlite3.connect(path, isolation_level="IMMEDIATE"), []
    self.conn.executescript(f"{PRAGMAS}; {DDL}")

  def __enter__(self):
    return self.conn.__enter__()

  def __exit__(self, exc_type, exc, tb):
    try:
      return self.conn.__exit__(exc_type, exc, tb)
    finally:
      if exc_type is not None:
        for tab in self.tables:
          tab.reload()

  def get_blob(self, key: str) -> bytes | None:
    row = self.conn.execute("SELECT v FROM kv WHERE ns='blobs' AND k=?", (key,)).fetchone()
    return bytes(row[0]) if row else None

  def put_blob(self, key: str, body: bytes) -> None:
    self.conn.execute("INSERT OR REPLACE INTO kv VALUES ('blobs', ?, ?)", (key, body))

  def del_blob(self, key: str) -> None:
    self.conn.execute("DELETE FROM kv WHERE ns='blobs' AND k=?", (key,))

  def vacuum(self) -> None:
    self.conn.execute("PRAGMA incremental_vacuum(256)")


class Table(dict):
  def __init__(self, db: DB, ns: str, cls=None):
    self.db, self.ns = db, ns
    self.dec = (lambda v: msgspec.msgpack.decode(v, type=cls)) if cls else msgspec.msgpack.decode
    db.tables.append(self)
    self.reload()

  def reload(self) -> None:
    self.clear()
    self.update((k, self.dec(v)) for k, v in self.db.conn.execute("SELECT k, v FROM kv WHERE ns=?", (self.ns,)))

  def __setitem__(self, key: str, value) -> None:
    self.db.conn.execute("INSERT OR REPLACE INTO kv VALUES (?, ?, ?)", (self.ns, key, msgspec.msgpack.encode(value)))
    super().__setitem__(key, value)

  def __delitem__(self, key: str) -> None:
    self.db.conn.execute("DELETE FROM kv WHERE ns=? AND k=?", (self.ns, key))
    super().__delitem__(key)

  def pop(self, key: str, *default):
    if key in self:
      self.db.conn.execute("DELETE FROM kv WHERE ns=? AND k=?", (self.ns, key))
    return super().pop(key, *default)

  def save(self, key: str) -> None:
    self[key] = self[key]
