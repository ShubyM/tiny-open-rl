"""Durable SQLite WAL store: the only module in tinyrl that speaks SQL. Callers own `with db:` transactions."""

from __future__ import annotations

import os
import sqlite3

import msgspec

from tinyrl.types import Checkpoint, Cmd, Op, Reply, Run, RunStatus, StabSlice, Task, parse_uri

SCHEMA_VERSION, MAX_FENCED = 1, 1024
encode, decode = msgspec.msgpack.encode, msgspec.msgpack.decode

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, active INT, seen REAL, data BLOB);
CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, deleted_at REAL, data BLOB);
CREATE TABLE IF NOT EXISTS checkpoints (uri TEXT PRIMARY KEY, expires_at REAL, owned INT, data BLOB);
CREATE TABLE IF NOT EXISTS commands (id TEXT PRIMARY KEY, run_id TEXT, seq INT, op TEXT, cost INT, uri TEXT, body BLOB, stab BLOB, claim BLOB);
CREATE TABLE IF NOT EXISTS replies (id TEXT PRIMARY KEY, expires REAL, data BLOB);
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


class DB:
  def __init__(self, path: str):
    if path != ":memory:":
      os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    self.conn = sqlite3.connect(path, isolation_level="IMMEDIATE")
    try:
      ver = self.one("PRAGMA user_version")[0]
      if ver > SCHEMA_VERSION:
        raise RuntimeError(f"database schema version {ver} is newer than supported {SCHEMA_VERSION}")
      if ver == 0 and self.one("SELECT 1 FROM sqlite_master WHERE type='table' AND name='kv'"):
        if self.one("SELECT 1 FROM kv LIMIT 1"):
          raise RuntimeError("legacy KV database detected; back up or migrate TINYRL_DB before starting")
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

  def save_session(self, sid: str, now: float, meta: dict | None = None, *, finish: bool = False) -> None:
    row = self.one("SELECT active, data FROM sessions WHERE id=?", (sid,))
    if finish and (row is None or not row[0]):
      return
    created, prev = decode(row[1]) if row else (now, {})
    data = encode((created, meta if (not finish and meta) else prev))
    self.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?)", (sid, 1, 0.0 if finish else now, data))

  def load_session(self, sid: str) -> tuple[float, float, dict] | None:
    if (row := self.one("SELECT seen, data FROM sessions WHERE id=?", (sid,))) is None:
      return None
    created, meta = decode(row[1])
    return created, row[0], meta

  def load_runs(self) -> list[tuple]:
    out = []
    for rid, deleted_at, blob in self.execute("SELECT id, deleted_at, data FROM runs"):
      rb, cr, st, ow, sn, cl, ct = decode(blob)
      out.append((rid, decode(rb, type=Run), cr, RunStatus(st), tuple(ow) if ow else None, sn, cl, deleted_at, ct))
    return out

  def save_run(self, s) -> None:
    blob = encode((encode(s.run), s.created, str(s.status), s.owner, s.snapshot, s.clients, s.counters))
    self.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?)", (s.run.id, s.deleted_at, blob))

  def load_accounts(self) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    service, lag, seen, weights = {}, {}, {}, {}
    for a, svc, lg, sn, wt in self.execute("SELECT * FROM accounts"):
      service[a], lag[a], seen[a], weights[a] = svc, lg, sn, wt
    return service, lag, seen, weights

  def save_accounts(self, service: dict, lag: dict, seen: dict, weights: dict, seq: int) -> None:
    for a in service.keys() | lag.keys() | seen.keys() | weights.keys():
      vals = (a, service.get(a, 0.0), lag.get(a, 0.0), seen.get(a, 0.0), weights.get(a, 1.0))
      self.execute("INSERT OR REPLACE INTO accounts VALUES (?,?,?,?,?)", vals)
    self.execute("INSERT OR REPLACE INTO meta VALUES ('seq', ?)", (seq,))

  def load_commands(self) -> list[tuple]:
    out = []
    sql = "SELECT id, run_id, seq, op, cost, uri, stab, claim FROM commands ORDER BY seq"
    for cid, rid, seq, op, cost, uri, stab, claim in self.execute(sql):
      s_vers = tuple(sorted({ref for ref, _, _ in decode(stab).values()})) if stab else ()
      cmd = Cmd(rid, cid.partition("/")[2], Op(op), b"", cost)
      if claim:
        nid, bid, at, tb = decode(claim)
        claim = (nid, bid, at, decode(tb, type=Task))
      out.append((cid, seq, cmd, uri, s_vers, claim))
    return out

  def insert_command(self, cmd: Cmd, seq: int, uri, seq_id, exp: float, refs: set[str] | None, now: float) -> tuple:
    stab = self.read_stability(refs, now) if refs else None
    vals = (cmd.request_id, cmd.run, seq, str(cmd.op), cmd.cost, uri, cmd.body, encode(stab) if stab else None)
    self.execute("INSERT INTO commands VALUES (?,?,?,?,?,?,?,?,NULL)", vals)
    if seq_id is not None:
      self.execute("INSERT OR REPLACE INTO dedup VALUES (?,?,?,?)", (cmd.run, seq_id, cmd.request_id, exp))
    self.execute("INSERT OR REPLACE INTO meta VALUES ('seq', ?)", (seq,))
    return tuple(sorted({ref for ref, _, _ in stab.values()})) if stab else ()

  def claim_command(self, req_id: str, nid: str, bid: str, at: float, slim: Task, state) -> None:
    self.execute("UPDATE commands SET claim=? WHERE id=?", (encode((nid, bid, at, encode(slim))), req_id))
    self.save_run(state)

  def remove_command(self, req_id: str | None, run_id: str, *, deleting: bool) -> None:
    if deleting:
      self.execute("DELETE FROM commands WHERE run_id=?", (run_id,))
      prefix = f"tinker://{run_id}/"
      self.execute("DELETE FROM checkpoints WHERE owned=1 AND substr(uri, 1, ?) = ?", (len(prefix), prefix))
    elif req_id is not None:
      self.execute("DELETE FROM commands WHERE id=?", (req_id,))

  def save_reply(self, req_id: str, reply: Reply, expires: float) -> None:
    self.execute("INSERT OR REPLACE INTO replies VALUES (?,?,?)", (req_id, expires, encode(reply)))

  def read_reply(self, req_id: str, now: float, read_ttl: float) -> Reply | None:
    row = self.one("SELECT expires, data FROM replies WHERE id=?", (req_id,))
    if row is None or row[0] <= now:
      return None
    self.execute("UPDATE replies SET expires=? WHERE id=?", (min(row[0], now + read_ttl), req_id))
    return decode(row[1], type=Reply)

  def save_checkpoint(self, c: Checkpoint) -> None:
    self.execute("INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?)", (c.uri, c.expires_at, int(c.owned), encode(c)))

  def get_checkpoint(self, uri: str, now: float) -> Checkpoint | None:
    parse_uri(uri)
    row = self.one("SELECT data FROM checkpoints WHERE uri=? AND (expires_at IS NULL OR expires_at>?)", (uri, now))
    return decode(row[0], type=Checkpoint) if row else None

  def list_checkpoints(self, now: float, run: str | None = None) -> list[Checkpoint]:
    rows = self.execute("SELECT uri, data FROM checkpoints WHERE expires_at IS NULL OR expires_at>?", (now,))
    found = [decode(blob, type=Checkpoint) for uri, blob in rows if run is None or parse_uri(uri)[0] == run]
    return sorted(found, key=lambda c: c.time, reverse=(run is None))

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
    for sid, (ref, off, ln) in slices.items():
      start, end = end, end + max(1, ln)
      self.execute("INSERT OR REPLACE INTO stability VALUES (?,?,?,?,?,?)", (sid, ref, off, ln, expires, start))
    self.execute("DELETE FROM stability WHERE start_byte < ?", (min(start, end - budget),))

  def read_stability(self, seq_ids: set[str], now: float) -> dict[str, StabSlice]:
    sql = "SELECT ref, offset, length FROM stability WHERE seq_id=? AND expires>?"
    return {s: r for s in seq_ids if (r := self.one(sql, (s, now))) is not None}

  def gc(self, now: float, session_ttl: float) -> None:
    for table in ("replies", "dedup", "stability"):
      self.execute(f"DELETE FROM {table} WHERE expires<=?", (now,))
    self.execute("DELETE FROM checkpoints WHERE expires_at IS NOT NULL AND expires_at<=?", (now,))
    stale = now - session_ttl
    self.execute("DELETE FROM sessions WHERE active=0 AND seen<=?", (stale,))
    sql = "DELETE FROM runs WHERE deleted_at>0 AND deleted_at<=? AND id NOT IN (SELECT run_id FROM commands)"
    self.execute(sql, (stale,))
