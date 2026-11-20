"""The controller's key-value store: rows in SQLite, or in RAM, and dicts that write through to them."""

from __future__ import annotations

import os
import sqlite3

import msgspec


class KV:
  """Transactional SQLite key-value store backing `Table` dicts and on-disk binary blobs."""

  def __init__(self, path: str):
    if path != ":memory:":
      os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    self.conn = sqlite3.connect(path, isolation_level="IMMEDIATE")
    for p in ("journal_mode=WAL", "synchronous=NORMAL", "busy_timeout=5000", "auto_vacuum=INCREMENTAL"):
      self.conn.execute(f"PRAGMA {p}")
    self.conn.execute("CREATE TABLE IF NOT EXISTS kv (ns TEXT, k TEXT, v BLOB, PRIMARY KEY (ns, k))")
    self.conn.commit()
    self.close = self.conn.close

  def __enter__(self):
    return self.conn.__enter__()

  def __exit__(self, *exc):
    return self.conn.__exit__(*exc)

  def get(self, ns: str, key: str) -> bytes | None:
    row = self.conn.execute("SELECT v FROM kv WHERE ns=? AND k=?", (ns, key)).fetchone()
    return bytes(row[0]) if row else None

  def put(self, ns: str, key: str, value: str | bytes) -> None:
    self.conn.execute("INSERT OR REPLACE INTO kv VALUES (?, ?, ?)", (ns, key, value))

  def delete(self, ns: str, key: str) -> None:
    self.conn.execute("DELETE FROM kv WHERE ns=? AND k=?", (ns, key))

  def load(self, ns: str) -> list[tuple[str, str]]:
    return self.conn.execute("SELECT k, v FROM kv WHERE ns=?", (ns,)).fetchall()


class RamKV:
  """`KV` without the disk, for sampling, which a restart may forget."""

  def __init__(self):
    self.rows = {}

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    return False

  def get(self, ns: str, key: str) -> bytes | None:
    return self.rows.get((ns, key))

  def put(self, ns: str, key: str, value: str | bytes) -> None:
    self.rows[ns, key] = value

  def delete(self, ns: str, key: str) -> None:
    self.rows.pop((ns, key), None)


class Table(dict):
  """In-memory dict persisted to a namespace in `KV`."""

  def __init__(self, kv: KV, ns: str):
    self.kv, self.ns = kv, ns
    super().__init__((k, msgspec.msgpack.decode(v)) for k, v in kv.load(ns))

  def __setitem__(self, key: str, value) -> None:
    super().__setitem__(key, value)
    self.kv.put(self.ns, key, msgspec.msgpack.encode(value))

  def __delitem__(self, key: str) -> None:
    super().__delitem__(key)
    self.kv.delete(self.ns, key)

  def pop(self, key: str, *default):
    if key not in self:
      return default[0] if default else super().pop(key)
    val = self[key]
    del self[key]
    return val

  def save(self, key: str) -> None:
    self[key] = self[key]
