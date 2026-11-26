"""Durable SQLite WAL store: the only module in tinyrl that speaks SQL. Callers own `with db:` transactions."""

from __future__ import annotations

import os
import sqlite3

import msgspec

from tinyrl.types import Checkpoint, Cmd, Op, Reply, Run, RunStatus, StabSlice, Task, parse_uri

SCHEMA_VERSION = 1
MAX_FENCED = 1024
encode, decode = msgspec.msgpack.encode, msgspec.msgpack.decode

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created REAL, seen REAL, active INT, meta BLOB);
CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, spec BLOB, created REAL, status TEXT, owner BLOB, snapshot TEXT, clients INT, deleted_at REAL, counters BLOB);
CREATE TABLE IF NOT EXISTS checkpoints (uri TEXT PRIMARY KEY, version TEXT, size INT, time REAL, expires_at REAL, owned INT, meta BLOB);
CREATE TABLE IF NOT EXISTS commands (id TEXT PRIMARY KEY, run_id TEXT, seq INT, op TEXT, cost INT, uri TEXT, body BLOB, stab BLOB, claim BLOB);
CREATE TABLE IF NOT EXISTS replies (id TEXT PRIMARY KEY, ok INT, content_type TEXT, expires REAL, body BLOB);
CREATE TABLE IF NOT EXISTS dedup (run_id TEXT, seq_id INT, req_id TEXT, expires REAL, PRIMARY KEY (run_id, seq_id));
CREATE TABLE IF NOT EXISTS stability (seq_id TEXT PRIMARY KEY, ref TEXT NOT NULL, offset INT NOT NULL, length INT NOT NULL, expires REAL NOT NULL, start_byte INT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS accounts (id TEXT PRIMARY KEY, service REAL, lag REAL, seen REAL, weight REAL);
CREATE TABLE IF NOT EXISTS fenced (seq INTEGER PRIMARY KEY AUTOINCREMENT, boot_id TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v INT);
CREATE INDEX IF NOT EXISTS idx_commands_run_seq ON commands(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_replies_expires ON replies(expires);
CREATE INDEX IF NOT EXISTS idx_dedup_expires ON dedup(expires);
CREATE INDEX IF NOT EXISTS idx_stability_expires ON stability(expires);
"""


def checkpoint_from_row(row: tuple) -> Checkpoint:
  return Checkpoint(row[0], row[1], row[2], row[3], row[4], bool(row[5]), decode(row[6]))


class DB:
  def __init__(self, path: str):
    if path != ":memory:":
      os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    self.conn = sqlite3.connect(path, isolation_level="IMMEDIATE")
    try:
      version = self.one("PRAGMA user_version")[0]
      if version > SCHEMA_VERSION:
        raise RuntimeError(f"database schema version {version} is newer than supported {SCHEMA_VERSION}")
      if version == 0 and self.one("SELECT 1 FROM sqlite_master WHERE type='table' AND name='kv'"):
        if self.one("SELECT 1 FROM kv LIMIT 1"):
          raise RuntimeError(
            "legacy KV database detected; back up or migrate TINYRL_DB before starting the relational controller"
          )
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

  def save_session(self, session_id: str, now: float, metadata: dict | None = None, *, finish: bool = False) -> None:
    row = self.one("SELECT created, active, meta FROM sessions WHERE id=?", (session_id,))
    if finish and (row is None or not row[1]):
      return
    created, seen = (row[0] if row else now), (0.0 if finish else now)
    stored = metadata if (not finish and metadata) else (decode(row[2]) if row else {})
    self.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?)", (session_id, created, seen, 1, encode(stored)))

  def load_session(self, session_id: str) -> tuple[float, float, dict] | None:
    row = self.one("SELECT created, seen, meta FROM sessions WHERE id=?", (session_id,))
    return (row[0], row[1], decode(row[2])) if row else None

  def close_session(self, session_id: str, *, wait_for_deletes: bool, now: float) -> None:
    if wait_for_deletes:
      self.execute("UPDATE sessions SET seen=? WHERE id=?", (now, session_id))
    else:
      self.execute("UPDATE sessions SET active=0 WHERE id=?", (session_id,))

  def expired_sessions(self, seen_before: float) -> list[str]:
    return [sid for (sid,) in self.execute("SELECT id FROM sessions WHERE active=1 AND seen<=?", (seen_before,))]

  def session_ids(self) -> set[str]:
    return {sid for (sid,) in self.execute("SELECT id FROM sessions")}

  def load_runs(self) -> list[tuple]:
    return [
      (rid, decode(sp, type=Run), cr, RunStatus(st), tuple(decode(ow)) if ow else None, sn, cl, dl, decode(ct))
      for rid, sp, cr, st, ow, sn, cl, dl, ct in self.execute("SELECT * FROM runs")
    ]

  def save_run(self, s) -> None:
    vals = (
      s.run.id,
      encode(s.run),
      s.created,
      str(s.status),
      encode(s.owner) if s.owner else None,
      s.snapshot,
      s.clients,
      s.deleted_at,
      encode(s.counters),
    )
    self.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?)", vals)

  def load_accounts(self) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    service, lag, seen, weights = {}, {}, {}, {}
    for acct, svc, lg, sn, wt in self.execute("SELECT * FROM accounts"):
      service[acct], lag[acct], seen[acct], weights[acct] = svc, lg, sn, wt
    return service, lag, seen, weights

  def load_seq(self) -> int:
    row = self.one("SELECT v FROM meta WHERE k='seq'")
    return int(row[0]) if row else 0

  def save_accounts(self, service: dict, lag: dict, seen: dict, weights: dict, seq: int) -> None:
    for acct in service.keys() | lag.keys() | seen.keys() | weights.keys():
      vals = (acct, service.get(acct, 0.0), lag.get(acct, 0.0), seen.get(acct, 0.0), weights.get(acct, 1.0))
      self.execute("INSERT OR REPLACE INTO accounts VALUES (?,?,?,?,?)", vals)
    self.execute("INSERT OR REPLACE INTO meta VALUES ('seq', ?)", (seq,))

  def load_commands(self) -> list[tuple]:
    out = []
    for cid, rid, seq, op, cost, uri, stab, claim in self.execute(
      "SELECT id, run_id, seq, op, cost, uri, stab, claim FROM commands ORDER BY seq"
    ):
      stab_vers = tuple(sorted({ref for ref, _, _ in decode(stab).values()})) if stab else ()
      cmd = Cmd(rid, cid.partition("/")[2], Op(op), b"", cost)
      decoded_claim = None
      if claim:
        nid, bid, at, task_blob = decode(claim)
        decoded_claim = (nid, bid, at, decode(task_blob, type=Task))
      out.append((cid, seq, cmd, uri, stab_vers, decoded_claim))
    return out

  def insert_command(
    self,
    cmd: Cmd,
    seq: int,
    uri: str | None,
    seq_id: int | None,
    dedup_exp: float,
    stab_refs: set[str] | None,
    now: float,
  ) -> tuple[str, ...]:
    stab = self.read_stability(stab_refs, now) if stab_refs else None
    vals = (cmd.request_id, cmd.run, seq, str(cmd.op), cmd.cost, uri, cmd.body, encode(stab) if stab else None)
    self.execute("INSERT INTO commands VALUES (?,?,?,?,?,?,?,?,NULL)", vals)
    if seq_id is not None:
      self.execute("INSERT OR REPLACE INTO dedup VALUES (?,?,?,?)", (cmd.run, seq_id, cmd.request_id, dedup_exp))
    self.execute("INSERT OR REPLACE INTO meta VALUES ('seq', ?)", (seq,))
    return tuple(sorted({ref for ref, _, _ in stab.values()})) if stab else ()

  def claim_command(self, req_id: str, node_id: str, boot_id: str, claimed_at: float, slim_task: Task, state) -> None:
    self.execute(
      "UPDATE commands SET claim=? WHERE id=?", (encode((node_id, boot_id, claimed_at, encode(slim_task))), req_id)
    )
    self.save_run(state)

  def load_command_payload(self, req_id: str) -> tuple[bytes, dict[str, StabSlice] | None]:
    row = self.one("SELECT body, stab FROM commands WHERE id=?", (req_id,))
    return (bytes(row[0]), decode(row[1]) if row[1] else None) if row else (b"", None)

  def remove_command(self, req_id: str | None, run_id: str, *, deleting: bool) -> None:
    if deleting:
      self.execute("DELETE FROM commands WHERE run_id=?", (run_id,))
      prefix = f"tinker://{run_id}/"
      self.execute("DELETE FROM checkpoints WHERE owned=1 AND substr(uri, 1, ?) = ?", (len(prefix), prefix))
    elif req_id is not None:
      self.execute("DELETE FROM commands WHERE id=?", (req_id,))

  def seen_dedup(self, run_id: str, seq_id: int, now: float) -> str | None:
    row = self.one("SELECT req_id FROM dedup WHERE run_id=? AND seq_id=? AND expires>?", (run_id, seq_id, now))
    return row[0] if row else None

  def save_reply(self, req_id: str, reply: Reply, expires: float) -> None:
    self.execute(
      "INSERT OR REPLACE INTO replies VALUES (?,?,?,?,?)",
      (req_id, int(reply.ok), reply.content_type, expires, reply.body),
    )

  def read_reply(self, req_id: str, now: float, read_ttl: float) -> Reply | None:
    row = self.one("SELECT ok, content_type, expires, body FROM replies WHERE id=?", (req_id,))
    if row is None or row[2] <= now:
      return None
    self.execute("UPDATE replies SET expires=? WHERE id=?", (min(row[2], now + read_ttl), req_id))
    return Reply(bool(row[0]), row[1], bytes(row[3]))

  def save_checkpoint(self, c: Checkpoint) -> None:
    self.execute(
      "INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?,?,?)",
      (c.uri, c.version, c.size, c.time, c.expires_at, int(c.owned), encode(c.meta)),
    )

  def get_checkpoint(self, uri: str, now: float) -> Checkpoint | None:
    parse_uri(uri)
    row = self.one("SELECT * FROM checkpoints WHERE uri=? AND (expires_at IS NULL OR expires_at>?)", (uri, now))
    return checkpoint_from_row(row) if row else None

  def list_checkpoints(self, now: float, run: str | None = None) -> list[Checkpoint]:
    rows = self.execute("SELECT * FROM checkpoints WHERE expires_at IS NULL OR expires_at>?", (now,))
    found = [checkpoint_from_row(r) for r in rows if run is None or parse_uri(r[0])[0] == run]
    return sorted(found, key=lambda c: c.time, reverse=(run is None))

  def delete_checkpoint(self, uri: str) -> None:
    self.execute("DELETE FROM checkpoints WHERE uri=?", (uri,))

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
      start, end = end, end + max(1, length)
      self.execute(
        "INSERT OR REPLACE INTO stability VALUES (?,?,?,?,?,?)", (seq_id, ref, offset, length, expires, start)
      )
    self.execute("DELETE FROM stability WHERE start_byte < ?", (min(start, end - budget),))

  def read_stability(self, seq_ids: set[str], now: float) -> dict[str, StabSlice]:
    rows = {
      s: self.one("SELECT ref, offset, length FROM stability WHERE seq_id=? AND expires>?", (s, now)) for s in seq_ids
    }
    return {s: r for s, r in rows.items() if r is not None}

  def stability_refs(self, now: float) -> set[str]:
    return {ref for (ref,) in self.execute("SELECT DISTINCT ref FROM stability WHERE expires>?", (now,))}

  def gc(self, now: float, session_ttl: float) -> None:
    for table in ("replies", "dedup", "stability"):
      self.execute(f"DELETE FROM {table} WHERE expires<=?", (now,))
    self.execute("DELETE FROM checkpoints WHERE expires_at IS NOT NULL AND expires_at<=?", (now,))
    stale = now - session_ttl
    self.execute("DELETE FROM sessions WHERE active=0 AND seen<=?", (stale,))
    self.execute(
      "DELETE FROM runs WHERE deleted_at>0 AND deleted_at<=? AND id NOT IN (SELECT run_id FROM commands)", (stale,)
    )

  def vacuum(self, pages: int = 256) -> None:
    self.execute(f"PRAGMA incremental_vacuum({pages})")
