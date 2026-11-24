"""Relational SQLite WAL store for durable controller state (zero ORM; single source of truth)."""

from __future__ import annotations

import os
import sqlite3

import msgspec

from tinyrl.types import StabSlice

SCHEMA_VERSION = 1
MAX_FENCED = 1024

encode = msgspec.msgpack.encode
decode = msgspec.msgpack.decode

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, created REAL, seen REAL, active INT, meta BLOB
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, spec BLOB, created REAL, status TEXT, owner BLOB,
  snapshot TEXT, clients INT, deleted_at REAL, counters BLOB
);
CREATE TABLE IF NOT EXISTS checkpoints (
  uri TEXT PRIMARY KEY, version TEXT, size INT, time REAL, expires_at REAL, owned INT, meta BLOB
);
CREATE TABLE IF NOT EXISTS commands (
  id TEXT PRIMARY KEY, run_id TEXT, seq INT, op TEXT, cost INT,
  uri TEXT, body BLOB, stab BLOB, claim BLOB
);
CREATE TABLE IF NOT EXISTS replies (
  id TEXT PRIMARY KEY, ok INT, content_type TEXT, expires REAL, body BLOB
);
CREATE TABLE IF NOT EXISTS dedup (
  run_id TEXT, seq_id INT, req_id TEXT, expires REAL, PRIMARY KEY (run_id, seq_id)
);
CREATE TABLE IF NOT EXISTS stability (
  seq_id TEXT PRIMARY KEY, ref TEXT NOT NULL, offset INT NOT NULL, length INT NOT NULL,
  expires REAL NOT NULL, start_byte INT UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS accounts (
  id TEXT PRIMARY KEY, service REAL, lag REAL, seen REAL, weight REAL
);
CREATE TABLE IF NOT EXISTS fenced (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, boot_id TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS meta (
  k TEXT PRIMARY KEY, v INT
);
CREATE INDEX IF NOT EXISTS idx_commands_run_seq ON commands(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_replies_expires ON replies(expires);
CREATE INDEX IF NOT EXISTS idx_dedup_expires ON dedup(expires);
CREATE INDEX IF NOT EXISTS idx_stability_expires ON stability(expires);
"""


class DB:
  def __init__(self, path: str):
    if path != ":memory:":
      os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    self.conn = sqlite3.connect(path, isolation_level="IMMEDIATE")
    try:
      version = self.one("PRAGMA user_version")[0]
      if version > SCHEMA_VERSION:
        raise RuntimeError(f"database schema version {version} is newer than supported {SCHEMA_VERSION}")
      for pragma in ("journal_mode=WAL", "synchronous=NORMAL", "busy_timeout=5000", "auto_vacuum=INCREMENTAL"):
        self.conn.execute(f"PRAGMA {pragma}")
      self.conn.executescript(f"BEGIN IMMEDIATE; {SCHEMA_SQL} PRAGMA user_version={SCHEMA_VERSION}; COMMIT;")
    except BaseException:
      self.conn.close()
      raise
    self.execute = self.conn.execute

  def close(self) -> None:
    self.conn.close()

  def __enter__(self):
    return self.conn.__enter__()

  def __exit__(self, *exc):
    return self.conn.__exit__(*exc)

  def one(self, sql: str, params: tuple = ()):
    return self.conn.execute(sql, params).fetchone()

  def add_fenced(self, boot_id: str) -> None:
    self.execute("INSERT OR IGNORE INTO fenced (boot_id) VALUES (?)", (boot_id,))
    self.execute("DELETE FROM fenced WHERE seq <= (SELECT COALESCE(MAX(seq), 0) - ? FROM fenced)", (MAX_FENCED,))

  def is_fenced(self, boot_id: str) -> bool:
    return self.one("SELECT 1 FROM fenced WHERE boot_id=?", (boot_id,)) is not None

  def record_stability(self, slices: dict[str, StabSlice], expires: float, budget: int) -> None:
    if not slices:
      return
    row = self.one("SELECT start_byte + MAX(1, length) FROM stability ORDER BY start_byte DESC LIMIT 1")
    start = end = row[0] if row else 0
    for seq_id, (ref, offset, length) in slices.items():
      start = end
      end += max(1, length)
      self.execute(
        "INSERT OR REPLACE INTO stability VALUES (?,?,?,?,?,?)", (seq_id, ref, offset, length, expires, start)
      )
    cutoff = min(start, end - budget)
    self.execute("DELETE FROM stability WHERE start_byte < ?", (cutoff,))

  def read_stability(self, seq_ids: set[str], now: float) -> dict[str, StabSlice]:
    found: dict[str, StabSlice] = {}
    for seq_id in seq_ids:
      row = self.one("SELECT ref, offset, length FROM stability WHERE seq_id=? AND expires>?", (seq_id, now))
      if row is not None:
        found[seq_id] = row
    return found

  def gc(self, now: float, session_ttl: float) -> None:
    for table in ("replies", "dedup", "stability"):
      self.execute(f"DELETE FROM {table} WHERE expires<=?", (now,))
    self.execute("DELETE FROM checkpoints WHERE expires_at IS NOT NULL AND expires_at<=?", (now,))
    stale_before = now - session_ttl
    self.execute("DELETE FROM sessions WHERE active=0 AND seen<=?", (stale_before,))
    self.execute(
      "DELETE FROM runs WHERE deleted_at>0 AND deleted_at<=? AND id NOT IN (SELECT run_id FROM commands)",
      (stale_before,),
    )
