"""Relational SQLite WAL store for durable controller state (zero ORM; single source of truth)."""

from __future__ import annotations

import os
import sqlite3

import msgspec

from tinyrl.types import StabSlice

SCHEMA_VERSION, MAX_FENCED = 1, 1024
ENC, DEC = msgspec.msgpack.encode, msgspec.msgpack.decode
TABLES = (
  "sessions (id TEXT PRIMARY KEY, created REAL, seen REAL, active INT, meta BLOB)",
  "runs (id TEXT PRIMARY KEY, spec BLOB, created REAL, status TEXT, owner BLOB, snapshot TEXT, clients INT, deleted_at REAL, counters BLOB)",
  "checkpoints (uri TEXT PRIMARY KEY, version TEXT, size INT, time REAL, expires_at REAL, owned INT, meta BLOB)",
  "commands (id TEXT PRIMARY KEY, run_id TEXT, seq INT, op TEXT, cost INT, uri TEXT, body BLOB, stab BLOB, claim BLOB)",
  "replies (id TEXT PRIMARY KEY, ok INT, content_type TEXT, expires REAL, body BLOB)",
  "dedup (run_id TEXT, seq_id INT, req_id TEXT, expires REAL, PRIMARY KEY (run_id, seq_id))",
  "stability (seq_id TEXT PRIMARY KEY, ref TEXT NOT NULL, offset INT NOT NULL, length INT NOT NULL, expires REAL NOT NULL, start_byte INT UNIQUE NOT NULL)",
  "accounts (id TEXT PRIMARY KEY, service REAL, lag REAL, seen REAL, weight REAL)",
  "fenced (seq INTEGER PRIMARY KEY AUTOINCREMENT, boot_id TEXT UNIQUE)",
  "meta (k TEXT PRIMARY KEY, v INT)",
)
INDEXES = ("commands(run_id,seq)", "replies(expires)", "dedup(expires)", "stability(expires)")


class DB:
  def __init__(self, path: str):
    if path != ":memory:":
      os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    self.conn = sqlite3.connect(path, isolation_level="IMMEDIATE")
    try:
      if (ver := self.one("PRAGMA user_version")[0]) > SCHEMA_VERSION:
        raise RuntimeError(f"database schema version {ver} is newer than supported {SCHEMA_VERSION}")
      for p in ("journal_mode=WAL", "synchronous=NORMAL", "busy_timeout=5000", "auto_vacuum=INCREMENTAL"):
        self.conn.execute(f"PRAGMA {p}")
      ddl = "".join(f"CREATE TABLE IF NOT EXISTS {t};" for t in TABLES)
      ddl += "".join(f"CREATE INDEX IF NOT EXISTS idx_{i} ON {x};" for i, x in enumerate(INDEXES))
      self.conn.executescript(f"BEGIN IMMEDIATE; {ddl} PRAGMA user_version={SCHEMA_VERSION}; COMMIT;")
    except BaseException:
      self.conn.close()
      raise
    self.execute, self.close = self.conn.execute, self.conn.close

  def __enter__(self):
    return self.conn.__enter__()

  def __exit__(self, *exc):
    return self.conn.__exit__(*exc)

  def one(self, sql: str, params: tuple = ()):
    return self.conn.execute(sql, params).fetchone()

  def add_fenced(self, boot_id: str) -> None:
    self.execute("INSERT OR IGNORE INTO fenced (boot_id) VALUES (?)", (boot_id,))
    self.execute(f"DELETE FROM fenced WHERE seq <= (SELECT COALESCE(MAX(seq), 0) - {MAX_FENCED} FROM fenced)")

  def is_fenced(self, boot_id: str) -> bool:
    return self.one("SELECT 1 FROM fenced WHERE boot_id=?", (boot_id,)) is not None

  def record_stability(self, slices: dict[str, StabSlice], expires: float, budget: int) -> None:
    row = self.one("SELECT start_byte + MAX(1, length) FROM stability ORDER BY start_byte DESC LIMIT 1")
    start = end = row[0] if row else 0
    for sid, (ref, off, ln) in slices.items():
      start, end = end, end + max(1, ln)
      self.execute("INSERT OR REPLACE INTO stability VALUES (?,?,?,?,?,?)", (sid, ref, off, ln, expires, start))
    if slices:
      self.execute("DELETE FROM stability WHERE start_byte < ?", (min(start, end - budget),))

  def read_stability(self, seqs: set[str], now: float) -> dict[str, StabSlice]:
    q = "SELECT ref, offset, length FROM stability WHERE seq_id=? AND expires>?"
    return {s: r for s in seqs if (r := self.one(q, (s, now)))}

  def gc(self, now: float, session_ttl: float) -> None:
    for tbl in ("replies", "dedup", "stability"):
      self.execute(f"DELETE FROM {tbl} WHERE expires<=?", (now,))
    self.execute("DELETE FROM checkpoints WHERE expires_at IS NOT NULL AND expires_at<=?", (now,))
    self.execute("DELETE FROM sessions WHERE active=0 AND seen<=?", (now - session_ttl,))
    q = "DELETE FROM runs WHERE deleted_at>0 AND deleted_at<=? AND id NOT IN (SELECT run_id FROM commands)"
    self.execute(q, (now - session_ttl,))
