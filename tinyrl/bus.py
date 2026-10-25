"""The coordinator's private, durable command journal.

Only the coordinator opens this file. Workers use its HTTP protocol, never SQLite.
A command is admitted before its future is returned and claimed before dispatch.
GPU execution and snapshot transfers happen outside journal transactions. Lost
workers fail their uncertain work; mutating commands are never automatically replayed.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from tinyrl.helpers import getenv, make_directory, sync_directory

LIVE_SECONDS = 120
REPLY_TTL = getenv("REPLY_TTL", 3600)
SESSION_TTL = getenv("SESSION_TTL", 7 * 24 * 3600)
WORKER_TTL = getenv("WORKER_TTL", 30.0)
MAX_PENDING_COMMANDS = getenv("MAX_PENDING_COMMANDS", 1024)
MAX_COMMAND_BYTES = getenv("MAX_COMMAND_BYTES", 64 << 20)
MAX_PENDING_BYTES = getenv("MAX_PENDING_BYTES", 256 << 20)


class Op(StrEnum):
  CREATE_MODEL = "create_model"
  FORWARD_BACKWARD = "forward_backward"
  OPTIM_STEP = "optim_step"
  SAVE_WEIGHTS = "save_weights"
  SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"
  LOAD_WEIGHTS = "load_weights"
  DELETE_MODEL = "delete_model"
  SNAPSHOT = "snapshot"
  RESTORE = "restore"
  ACTIVATE = "activate"
  DETACH = "detach"


CONTROL = {Op.SNAPSHOT, Op.RESTORE, Op.ACTIVATE, Op.DETACH}


@dataclass(frozen=True)
class Run:
  id: str
  pool: str
  base: str
  rank: int
  session: str = ""


@dataclass(frozen=True)
class Cmd:
  run: str
  entry_id: str
  op: Op
  body: bytes
  worker: str = ""
  generation: int = 0
  token: str = ""

  @property
  def request_id(self) -> str:
    return f"{self.run}/{self.entry_id}"


@dataclass(frozen=True)
class Reply:
  ok: bool
  content_type: str
  body: bytes
  artifact: str = ""
  recover: bool = False


def error(message: str, category: str = "server") -> Reply:
  return Reply(False, "application/json", json.dumps({"error": message, "category": category}).encode())


def encode(value) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class Bus:
  def __init__(self, path: str | os.PathLike | None = None):
    configured = str(path or getenv("JOURNAL", "./tinyrl-journal.sqlite"))
    if configured == ":memory:":
      raise ValueError("the coordinator journal must be persistent")
    self.path = str(Path(configured).resolve())
    make_directory(Path(self.path).parent)
    new_file = not Path(self.path).exists()
    self._lock = open(f"{self.path}.lock", "a+b")
    try:
      fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
      self._lock.close()
      raise RuntimeError(f"another coordinator owns {self.path}") from None
    try:
      self.db = sqlite3.connect(self.path, isolation_level=None)
      self.db.row_factory = sqlite3.Row
      self.db.execute("PRAGMA journal_mode=WAL")
      self.db.execute("PRAGMA synchronous=FULL")
      self.db.execute("PRAGMA foreign_keys=ON")
      self.db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, created REAL NOT NULL, seen REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS workers(
          id TEXT PRIMARY KEY, pool TEXT NOT NULL, seen REAL NOT NULL, live INTEGER NOT NULL);
        CREATE UNIQUE INDEX IF NOT EXISTS live_pool ON workers(pool) WHERE live=1;
        CREATE TABLE IF NOT EXISTS runs(
          id TEXT PRIMARY KEY, spec TEXT NOT NULL, creation_key TEXT UNIQUE, creation_digest TEXT NOT NULL,
          created REAL NOT NULL, pool TEXT NOT NULL, owner TEXT NOT NULL DEFAULT '',
          generation INTEGER NOT NULL DEFAULT 1,
          phase TEXT NOT NULL DEFAULT 'active', initialized INTEGER NOT NULL DEFAULT 0,
          cursor INTEGER NOT NULL DEFAULT 0, next_seq INTEGER NOT NULL DEFAULT 0,
          closing INTEGER NOT NULL DEFAULT 0, steps INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0,
          snapshot TEXT NOT NULL DEFAULT '', move_id TEXT NOT NULL DEFAULT '', target_pool TEXT NOT NULL DEFAULT '',
          source_owner TEXT NOT NULL DEFAULT '', source_pool TEXT NOT NULL DEFAULT '',
          source_generation INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS commands(
          id TEXT PRIMARY KEY, run TEXT NOT NULL REFERENCES runs(id), lane TEXT NOT NULL, seq INTEGER,
          op TEXT NOT NULL, body BLOB NOT NULL, dispatched_body BLOB, digest TEXT NOT NULL, idem TEXT,
          status TEXT NOT NULL DEFAULT 'queued', worker TEXT NOT NULL DEFAULT '', pool TEXT NOT NULL DEFAULT '',
          generation INTEGER NOT NULL DEFAULT 0, token TEXT NOT NULL DEFAULT '', move_id TEXT NOT NULL DEFAULT '',
          created REAL NOT NULL, claimed REAL, completed REAL, ok INTEGER, content_type TEXT, reply BLOB,
          artifact TEXT NOT NULL DEFAULT '', recover INTEGER NOT NULL DEFAULT 0,
          outcome_digest TEXT NOT NULL DEFAULT '',
          UNIQUE(run,lane,seq), UNIQUE(run,idem));
        CREATE INDEX IF NOT EXISTS command_queue ON commands(run,status,lane,seq);
        CREATE INDEX IF NOT EXISTS worker_commands ON commands(worker,status);
        CREATE TABLE IF NOT EXISTS artifacts(alias TEXT PRIMARY KEY, ref TEXT NOT NULL, request_id TEXT NOT NULL);
      """)
      if new_file:
        sync_directory(Path(self.path).parent)
    except BaseException:
      if hasattr(self, "db"):
        self.db.close()
      self._lock.close()
      raise
    self.changed = asyncio.Event()
    self._started = time.time()  # reconnect grace after a coordinator process restart
    self.closed = False

  @contextlib.contextmanager
  def _transaction(self):
    self.db.execute("BEGIN IMMEDIATE")
    try:
      yield
      self.db.execute("COMMIT")
    except BaseException:
      self.db.execute("ROLLBACK")
      raise
    self.changed.set()

  def _row(self, query, args=()):
    row = self.db.execute(query, args).fetchone()
    return dict(row) if row is not None else None

  def _run(self, run):
    return self._row("SELECT * FROM runs WHERE id=?", (run,))

  def _cmd(self, row) -> Cmd:
    return Cmd(
      row["run"],
      row["id"],
      Op(row["op"]),
      row["dispatched_body"] or row["body"],
      row["worker"],
      row["generation"],
      row["token"],
    )

  def _insert(self, run, op, body, lane="user", seq=None, idem=None, **fields):
    identifier = uuid.uuid4().hex
    digest = hashlib.sha256(op.encode() + b"\0" + body).hexdigest()
    values = {
      "id": identifier,
      "run": run,
      "op": str(op),
      "body": body,
      "lane": lane,
      "seq": seq,
      "idem": idem,
      "digest": digest,
      "created": time.time(),
      **fields,
    }
    self.db.execute(
      f"INSERT INTO commands ({','.join(values)}) VALUES ({','.join('?' for _ in values)})", tuple(values.values())
    )
    return f"{run}/{identifier}"

  def _control(self, run, op, body, *, worker="", pool="", generation=0, move_id=""):
    if op == Op.DETACH:
      existing = self._row(
        "SELECT id FROM commands WHERE run=? AND op=? AND worker=? AND generation=? "
        "AND status IN ('queued','executing')",
        (run, op, worker, generation),
      )
      if existing:
        return f"{run}/{existing['id']}"
    return self._insert(
      run, op, encode(body), "control", worker=worker, pool=pool, generation=generation, move_id=move_id
    )

  async def close(self):
    if not self.closed:
      self.closed = True
      self.db.close()
      self._lock.close()
      self.changed.set()

  async def session(self, session_id: str):
    now = time.time()
    with self._transaction():
      self.db.execute(
        "INSERT INTO sessions VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET seen=excluded.seen", (session_id, now, now)
      )

  async def session_info(self, session_id: str):
    row = self._row("SELECT * FROM sessions WHERE id=?", (session_id,))
    if not row or time.time() - row["seen"] > SESSION_TTL:
      return None
    runs = [
      r["id"] for r in self.db.execute("SELECT id,spec FROM runs") if json.loads(r["spec"]).get("session") == session_id
    ]
    return {
      "session_id": session_id,
      "created": row["created"],
      "seen": row["seen"],
      "live": time.time() - row["seen"] < LIVE_SECONDS,
      "runs": sorted(runs),
    }

  async def create(self, run: Run, *, body: bytes | dict | None = None, idempotency_key: str | None = None):
    if isinstance(body, dict):
      body = encode(body)
    if body is not None and len(body) > MAX_COMMAND_BYTES:
      raise ValueError("command exceeds MAX_COMMAND_BYTES")
    spec = {**asdict(run), "lora": json.loads(body).get("lora", {}) if body else {}}
    digest = hashlib.sha256(encode(spec) + (body or b"")).hexdigest()
    with self._transaction():
      prior = (
        self._row("SELECT * FROM runs WHERE creation_key=?", (idempotency_key,))
        if idempotency_key
        else self._run(run.id)
      )
      if prior:
        if prior["creation_digest"] != digest:
          raise ValueError("creation identity was already used with different model configuration")
        run_id = prior["id"]
      else:
        self.db.execute(
          "INSERT INTO runs(id,spec,creation_key,creation_digest,created,pool) VALUES (?,?,?,?,?,?)",
          (run.id, encode(spec).decode(), idempotency_key, digest, time.time(), run.pool),
        )
        run_id = run.id
        if body is not None:
          self._insert(run.id, Op.CREATE_MODEL, body, "bootstrap", -1)
    result = await self.run(run_id)
    initial = self._row("SELECT id FROM commands WHERE run=? AND lane='bootstrap'", (run_id,))
    if initial:
      result["request_id"] = f"{run_id}/{initial['id']}"
    return result

  async def run(self, run_id: str):
    row = self._run(run_id)
    return self._info(row)

  def _info(self, row):
    if not row:
      return None
    return {
      **json.loads(row["spec"]),
      **{
        k: row[k]
        for k in (
          "created",
          "pool",
          "owner",
          "generation",
          "phase",
          "steps",
          "failed",
          "cursor",
          "snapshot",
          "target_pool",
        )
      },
      "closing": bool(row["closing"]),
    }

  async def runs(self, pool: str):
    return [
      r[0]
      for r in self.db.execute(
        "SELECT id FROM runs WHERE pool=? AND phase NOT IN ('deleted','parked') ORDER BY created", (pool,)
      )
    ]

  async def submit(
    self, run_id: str, op: Op, body: bytes, *, seq_id: int | None = None, idempotency_key: str | None = None
  ):
    op = Op(op)
    if op in CONTROL:
      raise ValueError("internal control operations cannot be submitted by clients")
    if len(body) > MAX_COMMAND_BYTES:
      raise ValueError("command exceeds MAX_COMMAND_BYTES")
    if seq_id is not None and (
      isinstance(seq_id, bool) or not isinstance(seq_id, int) or not 0 <= seq_id < (1 << 63) - 1
    ):
      raise ValueError("seq_id must be a nonnegative 63-bit integer")
    lane, seq = ("bootstrap", -1) if op == Op.CREATE_MODEL else ("user", seq_id)
    digest = hashlib.sha256(op.encode() + b"\0" + body).hexdigest()
    with self._transaction():
      r = self._run(run_id)
      if not r:
        return None
      prior = (
        self._row("SELECT * FROM commands WHERE run=? AND idem=?", (run_id, idempotency_key))
        if idempotency_key
        else None
      )
      if not prior and seq is not None:
        prior = self._row("SELECT * FROM commands WHERE run=? AND lane=? AND seq=?", (run_id, lane, seq))
      if prior:
        if prior["digest"] != digest or (seq is not None and prior["seq"] != seq):
          raise ValueError("same command identity has different content")
        if idempotency_key and prior["idem"] != idempotency_key:
          if prior["idem"] is not None:
            raise ValueError("sequence already has a different idempotency key")
          self.db.execute("UPDATE commands SET idem=? WHERE id=?", (idempotency_key, prior["id"]))
        return f"{run_id}/{prior['id']}"
      if r["phase"] == "deleted":
        return None
      if op == Op.DELETE_MODEL and r["closing"]:
        prior = self._row("SELECT id FROM commands WHERE run=? AND op=? ORDER BY created LIMIT 1", (run_id, op))
        return f"{run_id}/{prior['id']}"
      if r["closing"]:
        deletion = self._row(
          "SELECT seq FROM commands WHERE run=? AND op=? ORDER BY created LIMIT 1", (run_id, Op.DELETE_MODEL)
        )
        if seq is None or deletion is None or seq >= deletion["seq"]:
          raise ValueError("run is closing")
      if r["phase"] == "recovery" and op != Op.DELETE_MODEL:
        raise ValueError("run requires explicit checkpoint recovery")
      if op == Op.DELETE_MODEL and r["phase"] in ("parked", "recovery"):
        # There is no active execution boundary to wait for. Deleting a failed
        # or parked run must not require restoring its model onto a GPU first.
        request = self._insert(run_id, op, body, lane, seq, idempotency_key)
        self._delete(r)
        self.db.execute(
          "UPDATE commands SET status='done',ok=1,content_type='application/json',reply=?,completed=? WHERE id=?",
          (encode({"type": Op.DELETE_MODEL, "model_id": run_id}), time.time(), request.rsplit("/", 1)[1]),
        )
        return request
      if lane == "user":
        seq = r["next_seq"] if seq is None else seq
        if seq < r["cursor"]:
          raise ValueError("sequence precedes the completed or failed command boundary")
      pending = self.db.execute(
        "SELECT count(*),coalesce(sum(length(body)),0) FROM commands WHERE run=? AND status IN ('queued','executing')",
        (run_id,),
      ).fetchone()
      if pending[0] >= MAX_PENDING_COMMANDS or pending[1] + len(body) > MAX_PENDING_BYTES:
        raise ValueError("run command queue capacity exceeded")
      if (
        lane == "user"
        and seq > r["cursor"]
        and (pending[0] >= MAX_PENDING_COMMANDS - 1 or pending[1] + len(body) > MAX_PENDING_BYTES - MAX_COMMAND_BYTES)
      ):
        # Out-of-order admission reserves capacity for the missing predecessor;
        # otherwise a full queue could prevent its own sequence gap being filled.
        raise ValueError("run command queue capacity reserved for its next sequence")
      request = self._insert(run_id, op, body, lane, seq, idempotency_key)
      if lane == "user":
        self.db.execute("UPDATE runs SET next_seq=max(next_seq,?) WHERE id=?", (seq + 1, run_id))
      if op == Op.DELETE_MODEL:
        self.db.execute("UPDATE runs SET closing=1 WHERE id=?", (run_id,))
      return request

  async def register(self, worker: str, pool: str):
    if not worker or not pool:
      raise ValueError("worker incarnation and pool are required")
    with self._transaction():
      prior = self._row("SELECT * FROM workers WHERE id=?", (worker,))
      if prior:
        if prior["pool"] != pool:
          raise ValueError("worker incarnation cannot change pools")
        if not prior["live"]:
          return {"registered": False, "reason": "worker incarnation was lost"}
        self.db.execute("UPDATE workers SET seen=? WHERE id=?", (time.time(), worker))
        return {"registered": True, "worker": worker, "pool": pool}
      existing = self._row("SELECT id FROM workers WHERE pool=? AND live=1", (pool,))
      if existing:
        return {"registered": False, "reason": "pool already has a worker", "owner": existing["id"]}
      self.db.execute("INSERT INTO workers VALUES (?,?,?,1)", (worker, pool, time.time()))
    return {"registered": True, "worker": worker, "pool": pool}

  async def heartbeat(self, worker: str):
    with self._transaction():
      return bool(self.db.execute("UPDATE workers SET seen=? WHERE id=? AND live=1", (time.time(), worker)).rowcount)

  async def take(self, pool: str, worker: str, count: int = 16, block_ms: int = 1000):
    deadline = time.monotonic() + max(0, block_ms) / 1000
    count = max(1, min(int(count), 64))
    while True:
      self.changed.clear()
      with self._transaction():
        w = self._row("SELECT * FROM workers WHERE id=? AND live=1 AND pool=?", (worker, pool))
        if not w:
          return []
        # Lost HTTP responses redeliver the same attempts, never create another execution.
        pending = list(
          self.db.execute(
            "SELECT * FROM commands WHERE worker=? AND status='executing' ORDER BY created LIMIT ?", (worker, count)
          )
        )
        if pending:
          return [self._cmd(c) for c in pending]
        taken = []
        controls = list(
          self.db.execute(
            "SELECT * FROM commands WHERE lane='control' AND status='queued' "
            "AND (worker=? OR (worker='' AND pool=?)) ORDER BY created",
            (worker, pool),
          )
        )
        busy_runs = set()
        for c in controls:
          r = self._run(c["run"])
          if c["op"] != Op.DETACH:
            if c["move_id"] != r["move_id"]:
              self._cancel_command(c["id"], "obsolete move")
              continue
            if self._row("SELECT id FROM commands WHERE run=? AND status='executing' AND op<>?", (c["run"], Op.DETACH)):
              continue
          if c["run"] in busy_runs:
            continue
          taken.append(self._claim(c, worker, c["generation"]))
          busy_runs.add(c["run"])
          if len(taken) == count:
            break
        if len(taken) < count:
          for r in list(
            self.db.execute(
              "SELECT * FROM runs WHERE pool=? AND phase='active' AND (owner='' OR owner=?) ORDER BY created",
              (pool, worker),
            )
          ):
            if r["id"] in busy_runs:
              continue
            if self._row("SELECT id FROM commands WHERE run=? AND status='executing' AND op<>?", (r["id"], Op.DETACH)):
              continue
            c = self._row("SELECT * FROM commands WHERE run=? AND lane='bootstrap' AND status='queued'", (r["id"],))
            if not c:
              c = self._row(
                "SELECT * FROM commands WHERE run=? AND lane='user' AND seq=? AND status='queued'",
                (r["id"], r["cursor"]),
              )
            if c:
              self.db.execute("UPDATE runs SET owner=? WHERE id=?", (worker, r["id"]))
              taken.append(self._claim(c, worker, r["generation"]))
            if len(taken) == count:
              break
        if taken:
          return taken
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        return []
      # A bounded timer also handles a missed wakeup or a disconnected long poll.
      await asyncio.sleep(min(0.05, remaining))

  def _claim(self, c, worker, generation):
    body = c["body"]
    if c["op"] == Op.SNAPSHOT:
      # Movement pauses new claims, but the previously dispatched command may
      # finish after movement was requested. Name the actual quiescent prefix.
      body = encode({"boundary": self._run(c["run"])["cursor"]})
    elif c["op"] == Op.LOAD_WEIGHTS:
      args = json.loads(body)
      saved = self._row("SELECT ref FROM artifacts WHERE alias=?", (args["uri"],))
      body = encode({**args, "ref": saved["ref"] if saved else ""})
    self.db.execute(
      "UPDATE commands SET status='executing',worker=?,generation=?,token=?,claimed=?,dispatched_body=? WHERE id=?",
      (worker, generation, uuid.uuid4().hex, time.time(), body, c["id"]),
    )
    return self._cmd(self._row("SELECT * FROM commands WHERE id=?", (c["id"],)))

  async def finish(self, cmd: Cmd, reply: Reply):
    outcome = hashlib.sha256(
      encode([reply.ok, reply.content_type, reply.artifact, reply.recover]) + b"\0" + reply.body
    ).hexdigest()
    with self._transaction():
      c = self._row("SELECT * FROM commands WHERE id=? AND run=?", (cmd.entry_id, cmd.run))
      if not c or (c["worker"], c["generation"], c["token"]) != (cmd.worker, cmd.generation, cmd.token):
        return "stale"
      if c["status"] == "done":
        if c["outcome_digest"] != outcome:
          raise ValueError("conflicting duplicate completion")
        return "already_committed"
      if c["status"] != "executing":
        return "stale"
      r = self._run(cmd.run)
      if c["lane"] == "control":
        if c["op"] != Op.DETACH and (c["move_id"] != r["move_id"] or r["phase"] not in ("moving", "activating")):
          return "stale"
      elif (r["owner"], r["generation"]) != (cmd.worker, cmd.generation) or r["phase"] in ("recovery", "deleted"):
        return "stale"
      if (
        reply.ok
        and c["op"] in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER, Op.SNAPSHOT)
        and not re.fullmatch(r"[0-9a-f]{32}", reply.artifact)
      ):
        raise ValueError("successful snapshot completion requires an immutable artifact ref")
      self.db.execute(
        "UPDATE commands SET status='done',ok=?,content_type=?,reply=?,artifact=?,recover=?,"
        "outcome_digest=?,completed=? WHERE id=?",
        (
          int(reply.ok),
          reply.content_type,
          reply.body,
          reply.artifact,
          int(reply.recover),
          outcome,
          time.time(),
          c["id"],
        ),
      )
      if c["lane"] == "control":
        self._finish_control(r, c, reply)
      else:
        if reply.ok and c["op"] in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER):
          uri = json.loads(c["body"])["uri"]
          self.db.execute(
            "INSERT INTO artifacts VALUES (?,?,?) ON CONFLICT(alias) "
            "DO UPDATE SET ref=excluded.ref,request_id=excluded.request_id",
            (uri, reply.artifact, cmd.request_id),
          )
        if c["lane"] == "user":
          self.db.execute("UPDATE runs SET cursor=? WHERE id=?", (c["seq"] + 1, cmd.run))
        if not reply.ok:
          self.db.execute("UPDATE runs SET failed=failed+1 WHERE id=?", (cmd.run,))
          if c["op"] == Op.DELETE_MODEL:
            self.db.execute("UPDATE runs SET closing=0 WHERE id=?", (cmd.run,))
        elif c["op"] == Op.OPTIM_STEP:
          self.db.execute("UPDATE runs SET steps=steps+1 WHERE id=?", (cmd.run,))
        elif c["op"] == Op.CREATE_MODEL:
          self.db.execute("UPDATE runs SET initialized=1 WHERE id=?", (cmd.run,))
        if reply.recover or (c["op"] == Op.CREATE_MODEL and not reply.ok):
          self._fail_run(cmd.run, "run requires checkpoint recovery after a failed mutation")
        elif c["op"] == Op.DELETE_MODEL and reply.ok:
          self._terminal(cmd.run, "run was deleted")
          self.db.execute("UPDATE runs SET phase='deleted',owner='',generation=generation+1 WHERE id=?", (cmd.run,))
    return "committed"

  def _cancel_command(self, identifier, message, status="cancelled"):
    reply = error(message)
    self.db.execute(
      "UPDATE commands SET status=?,ok=0,content_type=?,reply=?,completed=? "
      "WHERE id=? AND status IN ('queued','executing')",
      (status, reply.content_type, reply.body, time.time(), identifier),
    )

  def _terminal(self, run, message):
    for c in list(
      self.db.execute(
        "SELECT id,status FROM commands WHERE run=? AND lane<>'control' AND status IN ('queued','executing')", (run,)
      )
    ):
      self._cancel_command(c["id"], message, "unknown" if c["status"] == "executing" else "cancelled")

  def _fail_run(self, run, message):
    self._terminal(run, message)
    for c in list(
      self.db.execute(
        "SELECT id FROM commands WHERE run=? AND lane='control' AND status IN ('queued','executing') AND op<>?",
        (run, Op.DETACH),
      )
    ):
      self._cancel_command(c["id"], message)
    self.db.execute(
      "UPDATE runs SET phase='recovery',generation=generation+1,cursor=next_seq,move_id='',closing=0 WHERE id=?", (run,)
    )

  async def request_move(self, run: str, target_pool: str):
    with self._transaction():
      r = self._run(run)
      if not r:
        raise KeyError(run)
      if r["phase"] in ("moving", "activating") and r["target_pool"] == target_pool:
        return self._info(r)
      if r["closing"] or r["phase"] not in ("active", "parked"):
        raise ValueError("run is not available for movement")
      if r["pool"] == target_pool:
        return self._info(r)
      if r["phase"] == "active" and not r["initialized"]:
        raise ValueError("run must finish model creation before movement")
      self._begin_move(r, target_pool)
      if r["phase"] == "parked":
        if not target_pool:
          raise ValueError("parked run requires a target pool")
        self._restore_control(self._run(run), r["snapshot"])
      else:
        self._control(
          run,
          Op.SNAPSHOT,
          {"boundary": r["cursor"]},
          worker=r["owner"],
          pool=r["pool"],
          generation=r["generation"],
          move_id=self._run(run)["move_id"],
        )
    return await self.run(run)

  def _begin_move(self, r, target):
    self.db.execute(
      "UPDATE runs SET phase='moving',move_id=?,target_pool=?,source_owner=?,source_pool=?,"
      "source_generation=? WHERE id=?",
      (uuid.uuid4().hex, target, r["owner"], r["pool"], r["generation"], r["id"]),
    )

  def _restore_control(self, r, ref, *, recovery=False):
    self._control(
      r["id"],
      Op.RESTORE,
      {"ref": ref, "boundary": r["cursor"], "spec": json.loads(r["spec"]), "recovery": recovery},
      pool=r["target_pool"],
      generation=r["generation"] + 1,
      move_id=r["move_id"],
    )

  def _detach_source(self, r):
    if r["source_owner"] and self._row("SELECT id FROM workers WHERE id=? AND live=1", (r["source_owner"],)):
      self._control(
        r["id"], Op.DETACH, {}, worker=r["source_owner"], pool=r["source_pool"], generation=r["source_generation"]
      )

  def _finish_control(self, r, c, reply):
    op = Op(c["op"])
    if op == Op.DETACH:
      if reply.ok and (r["source_owner"], r["source_generation"]) == (c["worker"], c["generation"]):
        self.db.execute("UPDATE runs SET source_owner='',source_pool='' WHERE id=?", (r["id"],))
      return
    if not reply.ok:
      if op == Op.ACTIVATE or reply.recover or not r["source_owner"]:
        self._fail_run(r["id"], "checkpoint restore or activation failed; explicit recovery required")
      else:
        if op == Op.RESTORE:
          self._control(r["id"], Op.DETACH, {}, worker=c["worker"], pool=r["target_pool"], generation=c["generation"])
        self.db.execute("UPDATE runs SET phase='active',move_id='',target_pool='' WHERE id=?", (r["id"],))
      return
    if op == Op.SNAPSHOT:
      self.db.execute("UPDATE runs SET snapshot=? WHERE id=?", (reply.artifact, r["id"]))
      if not r["target_pool"]:
        self.db.execute(
          "UPDATE runs SET phase='parked',pool='',owner='',generation=generation+1,move_id='' WHERE id=?", (r["id"],)
        )
        self._detach_source(r)
      else:
        self._restore_control(r, reply.artifact)
    elif op == Op.RESTORE:
      self.db.execute(
        "UPDATE runs SET owner=?,pool=?,generation=?,phase='activating' WHERE id=?",
        (c["worker"], r["target_pool"], c["generation"], r["id"]),
      )
      self._control(
        r["id"],
        Op.ACTIVATE,
        {"boundary": r["cursor"]},
        worker=c["worker"],
        pool=r["target_pool"],
        generation=c["generation"],
        move_id=r["move_id"],
      )
    elif op == Op.ACTIVATE:
      self.db.execute("UPDATE runs SET phase='active',initialized=1,move_id='' WHERE id=?", (r["id"],))
      self._detach_source(r)

  async def recover(self, run: str, checkpoint_ref: str, target_pool: str | None = None):
    if not re.fullmatch(r"[0-9a-f]{32}", checkpoint_ref):
      raise ValueError("recovery requires a durable checkpoint ref")
    with self._transaction():
      r = self._run(run)
      if not r:
        raise KeyError(run)
      if r["phase"] != "recovery":
        raise ValueError("run is not awaiting recovery")
      target = target_pool or r["pool"] or json.loads(r["spec"])["pool"]
      if not target:
        raise ValueError("recovery requires a target pool")
      # A failed mutation may leave an unusable model on a still-live worker.
      # Dispose of that generation, but never use it as a rollback source.
      self._detach_owner(r)
      self._detach_source(r)
      self._begin_move({**r, "owner": ""}, target)
      self.db.execute("UPDATE runs SET snapshot=?,closing=0 WHERE id=?", (checkpoint_ref, run))
      self._restore_control(self._run(run), checkpoint_ref, recovery=True)
    return await self.run(run)

  async def ckpt(self, uri: str):
    saved = self._row("SELECT ref FROM artifacts WHERE alias=?", (uri,))
    return saved["ref"] if saved else None

  def _lose_worker(self, worker):
    if not self.db.execute("UPDATE workers SET live=0 WHERE id=? AND live=1", (worker,)).rowcount:
      return 0
    affected = 0
    for r in list(
      self.db.execute("SELECT * FROM runs WHERE owner=? AND phase NOT IN ('deleted','recovery','parked')", (worker,))
    ):
      self._fail_run(r["id"], "worker was lost; execution outcome is uncertain, recover from a checkpoint")
      affected += 1
    for c in list(
      self.db.execute(
        "SELECT * FROM commands WHERE worker=? AND lane='control' AND status IN ('queued','executing')", (worker,)
      )
    ):
      r = self._run(c["run"])
      self._cancel_command(c["id"], "worker was lost")
      if c["op"] == Op.RESTORE and r["phase"] == "moving" and r["move_id"] == c["move_id"]:
        if r["source_owner"]:
          self.db.execute("UPDATE runs SET phase='active',move_id='',target_pool='' WHERE id=?", (r["id"],))
        else:
          self._fail_run(r["id"], "restore worker was lost")
    return affected

  async def worker_lost(self, worker: str):
    with self._transaction():
      return self._lose_worker(worker)

  async def lose_worker(self, run: str, owner: str, generation: int):
    with self._transaction():
      r = self._run(run)
      if not r or (r["owner"], r["generation"]) != (owner, generation):
        return False
      self._lose_worker(owner)
      return True

  async def expire_workers(self):
    with self._transaction():
      lost = 0
      if time.time() - self._started >= WORKER_TTL:
        for w in list(self.db.execute("SELECT id FROM workers WHERE live=1 AND seen<?", (time.time() - WORKER_TTL,))):
          lost += self._lose_worker(w["id"])
      self.db.execute(
        "UPDATE commands SET reply=NULL,body=x'',dispatched_body=NULL "
        "WHERE reply IS NOT NULL AND completed IS NOT NULL AND completed<?",
        (time.time() - REPLY_TTL,),
      )
      return lost

  async def result(self, request_id: str, timeout: float):
    try:
      run, identifier = request_id.rsplit("/", 1)
    except ValueError:
      return error("unknown future", "user")
    deadline = time.monotonic() + max(0, timeout)
    while True:
      self.changed.clear()
      c = self._row("SELECT * FROM commands WHERE id=? AND run=?", (identifier, run))
      if not c:
        return error("unknown future", "user")
      if c["completed"] is not None:
        if time.time() - c["completed"] >= REPLY_TTL or c["reply"] is None:
          return error("future result expired", "user")
        return Reply(bool(c["ok"]), c["content_type"], c["reply"], c["artifact"], bool(c["recover"]))
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        return None
      try:
        await asyncio.wait_for(self.changed.wait(), min(remaining, 0.25))
      except TimeoutError:
        pass

  async def queue(self, run_id: str):
    r = self._run(run_id)
    if not r or r["phase"] == "deleted":
      return {"live": False}
    queued = self.db.execute("SELECT count(*) FROM commands WHERE run=? AND status='queued'", (run_id,)).fetchone()[0]
    inflight = [
      {
        "request_id": f"{run_id}/{c['id']}",
        "trainer": c["worker"],
        "seconds": max(0, time.time() - c["claimed"]),
        "deliveries": 1,
        "generation": c["generation"],
      }
      for c in self.db.execute("SELECT * FROM commands WHERE run=? AND status='executing'", (run_id,))
    ]
    return {"live": True, "queued": queued, "in_flight": inflight, "phase": r["phase"]}

  async def drop(self, run_id: str, pool: str = ""):
    """Administrative terminal deletion; trainer deletion is committed by finish."""
    with self._transaction():
      r = self._run(run_id)
      if r:
        self._delete(r)

  def _delete(self, r):
    self._terminal(r["id"], "run was deleted")
    for c in list(
      self.db.execute(
        "SELECT id FROM commands WHERE run=? AND lane='control' AND status IN ('queued','executing') AND op<>?",
        (r["id"], Op.DETACH),
      )
    ):
      self._cancel_command(c["id"], "run was deleted")
    self._detach_owner(r)
    self._detach_source(r)
    self.db.execute(
      "UPDATE runs SET phase='deleted',owner='',closing=1,generation=generation+1,move_id='' WHERE id=?", (r["id"],)
    )

  def _detach_owner(self, r):
    if r["owner"] and self._row("SELECT id FROM workers WHERE id=? AND live=1", (r["owner"],)):
      latest = self._row(
        "SELECT generation FROM commands WHERE run=? AND worker=? AND claimed IS NOT NULL "
        "ORDER BY claimed DESC LIMIT 1",
        (r["id"], r["owner"]),
      )
      if latest:
        self._control(r["id"], Op.DETACH, {}, worker=r["owner"], pool=r["pool"], generation=latest["generation"])
