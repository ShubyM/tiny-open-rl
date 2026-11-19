"""In-memory control-plane Bus backed by SQLite WAL for durable metadata.

Durable tables (`sessions`, `runs`, `checkpoints`, `meta`) are plain Python dicts (`Table`) that write through to a
single SQLite WAL file. Command queues, replies, leases, nodes, and events live in RAM; lost sampling promises raise
PromiseGoneError (HTTP 410) so the Tinker SDK resubmits them.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sqlite3
import time
import uuid
import weakref
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass

from tinyrl.control.accounting import Charge
from tinyrl.helpers import getenv
from tinyrl.rpc import RemoteBus, serve_bus  # noqa: F401
from tinyrl.types import Cmd, Ev, Event, Op, Reply, Role, Run, parse_uri, unpack_snap

LIVE_SECONDS, SESSION_GRACE = 30, getenv("SESSION_GRACE", 60)
REPLY_TTL, READ_REPLY_TTL, SESSION_TTL = getenv("REPLY_TTL", 3600), 60, getenv("SESSION_TTL", 7 * 24 * 3600)
LEASE_MS, NODE_MS = getenv("LEASE_MS", 10000), 30000
SAMPLE_QUEUE_BYTES = getenv("SAMPLE_QUEUE_BYTES", 256 * 2**20)
SAMPLE_REPLY_BYTES = getenv("SAMPLE_REPLY_BYTES", 256 * 2**20)
COUNTERS = ("steps", "failed", "cmds", "tokens", "busy_ms", "wait_ms")


class PromiseGoneError(LookupError):
  """Raised when a stateless sampling promise no longer exists in RAM (maps to HTTP 410)."""


class BackpressureError(RuntimeError):
  """Raised when in-memory sampling queues hit SAMPLE_QUEUE_BYTES (maps to HTTP 429)."""


@dataclass
class Claim:
  cmd: Cmd
  consumer: str
  claimed_at: float
  deliveries: int = 1


def life_of(token: str) -> str:
  return token.rsplit(":", 1)[0]


def entry_sort_key(entry_id: str) -> tuple[int, int]:
  ms, seq = entry_id.split("-", 1)
  return int(ms), int(seq)


def charge_counts(cmd: Cmd, reply: Reply, charge: Charge | None) -> dict[str, int]:
  counts = {"failed": 1} if not reply.ok else {"steps": 1} if cmd.op == Op.OPTIM_STEP else {}
  if charge:
    wait_ms = max(0, charge.started_ms - int(cmd.entry_id.split("-")[0]))
    counts.update(cmds=1, wait_ms=wait_ms, busy_ms=charge.busy_ms)
    if reply.ok:
      counts["tokens"] = cmd.cost
  return counts


def make_checkpoint(uri: str, size: int, *, version: str, ttl=None, owned=False) -> dict:
  _, cid = parse_uri(uri)
  raw_kind, now = cid.split("/", 1)[0], time.time()
  kind = {"weights": "training", "sampler_weights": "sampler"}.get(raw_kind, raw_kind)
  return {
    "checkpoint_id": cid,
    "checkpoint_type": kind,
    "time": now,
    "tinker_path": uri,
    "size_bytes": size,
    "expires_at": now + ttl if ttl else None,
    "owned": bool(owned),
    "version": version,
  }


class Table(dict):
  """In-memory dict backed by a namespace in SQLite (`kv(ns, k, v)`)."""

  def __init__(self, db: sqlite3.Connection, ns: str):
    self.db, self.ns = db, ns
    super().__init__((k, json.loads(v)) for k, v in db.execute("SELECT k, v FROM kv WHERE ns = ?", (ns,)))

  def __setitem__(self, key: str, value) -> None:
    super().__setitem__(key, value)
    self.db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?, ?)", (self.ns, key, json.dumps(value)))

  def __delitem__(self, key: str) -> None:
    super().__delitem__(key)
    self.db.execute("DELETE FROM kv WHERE ns = ? AND k = ?", (self.ns, key))

  def pop(self, key: str, *default):
    if key in self:
      value = self[key]
      del self[key]
      return value
    return default[0] if default else super().pop(key)

  def save(self, key: str) -> None:
    self[key] = self[key]


class Bus:
  """Single-control-plane bus: write-through SQLite Tables for durable metadata, RAM for queues & liveness."""

  def __init__(self, path: str | None = None):
    self.path = getenv("TINYRL_DB", "/tmp/tinyrl.db") if path is None else path
    if self.path != ":memory:":
      os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
    self.db = sqlite3.connect(self.path, isolation_level="IMMEDIATE")
    for pragma in ("journal_mode=WAL", "synchronous=NORMAL", "busy_timeout=5000", "auto_vacuum=INCREMENTAL"):
      self.db.execute(f"PRAGMA {pragma}")
    self.db.execute("CREATE TABLE IF NOT EXISTS kv (ns TEXT, k TEXT, v TEXT, PRIMARY KEY (ns, k))")
    self.db.commit()

    self.sessions = Table(self.db, "sessions")
    self.runs = Table(self.db, "runs")
    self.checkpoints = Table(self.db, "checkpoints")
    self.meta = Table(self.db, "meta")
    self.service: dict[str, float] = {k: float(v) for k, v in self.meta.get("service", {}).items()}

    self.queues: dict[str, dict[str, Cmd]] = {rid: {} for rid, r in self.runs.items() if r["live"]}
    self.in_flight: dict[str, dict[str, Claim]] = {rid: {} for rid in self.queues}
    self.replies: OrderedDict[str, tuple[Reply, float, int]] = OrderedDict()
    self.seen: dict[tuple[str, int], tuple[str, float]] = {}
    self.nodes: dict[str, dict] = {}
    self.leases: dict[str, float] = {}
    self.dropped: set[str] = set()
    self.events: deque[Event] = deque(maxlen=100_000)
    self.waiters: weakref.WeakValueDictionary[str, asyncio.Event] = weakref.WeakValueDictionary()
    self.wake_seq = {"wake": 0, "plan_wake": 0}
    self.seq = self.sample_bytes = self.reply_bytes = 0
    self.dirty_runs: set[str] = set()
    self.service_dirty = False

    has_owner = any(r["live"] and r.get("owner") for r in self.runs.values())
    self.reap_after = time.monotonic() + LEASE_MS / 1000 if has_owner else 0.0

  def is_sampler(self, run_id: str) -> bool:
    run = self.runs.get(run_id)
    return bool(run and run["record"]["role"] == Role.SAMPLER)

  def flush_dirty(self) -> None:
    if self.service_dirty:
      self.meta["service"] = self.service
      self.service_dirty = False
    for run_id in self.dirty_runs:
      if run_id in self.runs:
        self.runs.save(run_id)
    self.dirty_runs.clear()

  async def aclose(self) -> None:
    if self.db is not None:
      with self.db:
        self.flush_dirty()
      self.db.close()
      self.db = None

  def notify(self, key: str) -> None:
    if ev := self.waiters.pop(key, None):
      ev.set()

  def wake(self, stream: str = "wake") -> None:
    self.wake_seq[stream] += 1
    self.notify(f"wake:{stream}")

  def record_event(self, kind: Ev, body: dict) -> Event:
    self.seq += 1
    now = time.time()
    ev = Event(f"{int(now * 1000)}-{self.seq}", now, str(Ev(kind)), body)
    self.events.append(ev)
    return ev

  async def peek(self, runs: list[str], heads_fn, block_ms: int) -> list[Cmd]:
    if not runs:
      if block_ms > 0:
        await asyncio.sleep(block_ms / 1000)
      return []
    if (cmds := heads_fn(runs)) or block_ms <= 0:
      return cmds
    tasks = [asyncio.create_task(self.waiters.setdefault(f"run:{r}", asyncio.Event()).wait()) for r in runs]
    try:
      await asyncio.wait(tasks, timeout=block_ms / 1000, return_when=asyncio.FIRST_COMPLETED)
    finally:
      for t in tasks:
        t.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)
    return heads_fn(runs)

  # ***** sessions *****

  async def heartbeat_session(self, session_id: str, metadata: dict | None = None) -> None:
    now = time.time()
    session = self.sessions.get(session_id) or {"created": now, "seen": now, "active": True, "metadata": {}}
    session["seen"], session["active"] = now, True
    if metadata:
      session["metadata"] = metadata
    with self.db:
      self.sessions[session_id] = session

  async def get_session(self, session_id: str) -> dict | None:
    if (session := self.sessions.get(session_id)) is None:
      return None
    runs = sorted(rid for rid, r in self.runs.items() if r["record"].get("session") == session_id)
    return {
      "session_id": session_id,
      "created": session["created"],
      "seen": session["seen"],
      "live": time.time() - session["seen"] < LIVE_SECONDS,
      "runs": runs,
      "metadata": dict(session["metadata"]),
    }

  async def expired_sessions(self) -> list[str]:
    cutoff = time.time() - SESSION_GRACE
    return [sid for sid, s in self.sessions.items() if s["active"] and s["seen"] <= cutoff]

  async def finish_session(self, session_id: str) -> None:
    if (session := self.sessions.get(session_id)) and session["active"]:
      session["seen"] = 0.0
      with self.db:
        self.sessions.save(session_id)
    self.wake()

  async def end_session(self, session_id: str) -> None:
    live = sorted(rid for rid, r in self.runs.items() if r["live"] and r["record"].get("session") == session_id)
    has_trainers = False
    for rid in live:
      if self.is_sampler(rid):
        await self.delete_run(rid)
      else:
        has_trainers = bool(await self.submit_command(rid, Op.DELETE_MODEL, b"{}")) or has_trainers
    if session := self.sessions.get(session_id):
      if has_trainers:
        session["seen"] = time.time()
      else:
        session["active"] = False
      with self.db:
        self.sessions.save(session_id)

  # ***** runs and cluster state *****

  async def create_run(self, run: Run) -> None:
    row = {
      "live": True,
      "record": {**asdict(run), "created": time.time()},
      "owner": None,
      "epoch": 0,
      "pos": None,
      "snap": None,
      "writing": None,
      "clients": 0,
      "counters": dict.fromkeys(COUNTERS, 0),
    }
    self.queues[run.id] = {}
    self.in_flight[run.id] = {}
    with self.db:
      self.runs[run.id] = row
    if run.role == Role.SAMPLER:
      self.wake("plan_wake")
    fields = ("role", "base", "rank", "exclusive", "train_run", "gpus")
    self.record_event(Ev.RUN_CREATED, {"run": run.id, **{k: getattr(run, k) for k in fields}})
    self.wake()

  async def read_state(self, *, backlogs: bool = True, counters: bool = False) -> dict:
    nodes = await self.live_nodes(details=True)
    runs, queues = {}, {}
    for rid in sorted(self.runs):
      row = self.runs[rid]
      if not row["live"]:
        continue
      info = {"record": dict(row["record"]), "owner": row["owner"]}
      if counters:
        info["counters"] = dict(row["counters"])
      runs[rid] = info
      if backlogs:
        queues[rid] = len(self.queues.get(rid, ())) + len(self.in_flight.get(rid, ()))
    return {
      "nodes": nodes,
      "runs": runs,
      "queues": queues,
      "plan": dict(self.meta.get("plan", {})),
      "service": dict(self.service),
    }

  async def write_plan(self, plan: dict, changes: list[tuple[Ev, dict]] = (), notify_nodes: bool = True) -> None:
    with self.db:
      self.meta["plan"] = plan
      self.flush_dirty()
    for kind, body in changes:
      self.record_event(kind, body)
    if notify_nodes:
      self.wake("plan_wake")

  async def get_run(self, run_id: str) -> dict | None:
    if (row := self.runs.get(run_id)) is None:
      return None
    meta = {k: row[k] for k in ("owner", "pos", "snap") if row[k]}
    return {**row["record"], **meta, **row["counters"]}

  async def join_sampling_session(self, run_id: str) -> int:
    if (row := self.runs.get(run_id)) is None:
      return 1
    row["clients"] += 1
    with self.db:
      self.runs.save(run_id)
    return row["clients"]

  async def delete_run(self, run_id: str) -> None:
    is_sampler = self.is_sampler(run_id)
    if row := self.runs.get(run_id):
      row["live"] = False
      row["owner"] = row["snap"] = row["writing"] = None
      with self.db:
        self.runs.save(run_id)
        self.flush_dirty()
    queued = list(self.queues.pop(run_id, {}).values())
    claimed = [c.cmd for c in self.in_flight.pop(run_id, {}).values()]
    for cmd in (*queued, *claimed):
      if is_sampler:
        self.sample_bytes -= len(cmd.body)
      self.notify(f"reply:{cmd.request_id}")
    if is_sampler:
      for k in [k for k in self.seen if k[0] == run_id]:
        del self.seen[k]
    self.record_event(Ev.RUN_DELETED, {"run": run_id})
    self.wake()

  # ***** commands & replies (RAM) *****

  async def seen_request(self, run_id: str, seq_id: int | None) -> str | None:
    entry = None if seq_id is None else self.seen.get((run_id, seq_id))
    return entry[0] if entry and entry[1] > time.time() else None

  async def submit_command(
    self, run_id: str, op: Op, body: bytes, seq_id: int | None = None, cost: int = 1
  ) -> str | None:
    if existing := await self.seen_request(run_id, seq_id):
      return existing
    if run_id not in self.queues:
      return None
    if self.is_sampler(run_id):
      if self.sample_bytes + len(body) > SAMPLE_QUEUE_BYTES:
        raise BackpressureError("sampling queue memory limit reached")
      self.sample_bytes += len(body)
    self.seq += 1
    cmd = Cmd(run_id, f"{time.time_ns() // 1_000_000}-{self.seq}", Op(op), body, cost)
    self.queues[run_id][cmd.entry_id] = cmd
    if seq_id is not None:
      self.seen[(run_id, seq_id)] = (cmd.request_id, time.time() + SESSION_TTL)
    self.notify(f"run:{run_id}")
    self.wake()
    return cmd.request_id

  async def wait_for_reply(self, request_id: str, timeout: float) -> Reply | None:
    run_id, _, entry_id = request_id.partition("/")
    is_sampler = self.is_sampler(run_id) or request_id.startswith("samp-")
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
      now = time.time()
      if (entry := self.replies.get(request_id)) is not None:
        reply, expires_at, size = entry
        if expires_at > now:
          self.replies[request_id] = (reply, min(expires_at, now + min(REPLY_TTL, READ_REPLY_TTL)), size)
          self.replies.move_to_end(request_id)
          return reply
        self.reply_bytes -= self.replies.pop(request_id)[2]
      active = entry_id in self.queues.get(run_id, ()) or entry_id in self.in_flight.get(run_id, ())
      if is_sampler and not active:
        raise PromiseGoneError(f"sampling promise gone: {request_id}")
      if (remaining := deadline - time.monotonic()) <= 0:
        return None
      with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(self.waiters.setdefault(f"reply:{request_id}", asyncio.Event()).wait(), remaining)

  async def queued_and_in_flight(self, run_id: str) -> dict:
    if run_id not in self.queues:
      return {"live": False}
    now = time.monotonic()
    in_flight = [
      {
        "request_id": c.cmd.request_id,
        "trainer": c.consumer,
        "deliveries": c.deliveries,
        "seconds": max(0.0, now - c.claimed_at),
      }
      for c in self.in_flight[run_id].values()
    ]
    return {"live": True, "queued": len(self.queues[run_id]), "in_flight": in_flight}

  # ***** trainer ownership & leases *****

  async def touch_leases(self, *lives: str, xx: bool = False, ttl_ms: int | None = None) -> None:
    now, ttl = time.monotonic(), (LEASE_MS if ttl_ms is None else ttl_ms) / 1000
    for life in lives:
      if not xx or self.leases.get(life, 0.0) > now:
        self.leases[life] = now + ttl
        self.dropped.discard(life)

  async def drop_lease(self, life: str) -> None:
    self.leases.pop(life, None)
    self.dropped.add(life)
    for rid in self.queues:
      if self.is_sampler(rid):
        self.notify(f"run:{rid}")

  async def acquire_run(self, run_id: str, life: str) -> tuple[str, str | None, str | None] | None:
    now = time.monotonic()
    if self.leases.get(life, 0.0) <= now:
      return None
    row = self.runs.get(run_id)
    if not row or not row["live"]:
      return None
    if row["owner"]:
      prev = life_of(row["owner"])
      if self.leases.get(prev, 0.0) > now or (now < self.reap_after and prev not in self.dropped):
        return None
    row["epoch"] += 1
    row["owner"] = token = f"{life}:{row['epoch']}"
    row["writing"] = None
    with self.db:
      self.runs.save(run_id)
    return token, row["pos"], row["snap"]

  def trainer_heads(self, runs: list[str]) -> list[Cmd]:
    heads = [
      Cmd(r, (c := next(iter(q.values()))).entry_id, c.op, b"", c.cost) for r in runs if (q := self.queues.get(r))
    ]
    return sorted(heads, key=lambda c: (*entry_sort_key(c.entry_id), c.run))

  async def peek_trainer(self, runs: list[str], block_ms: int = 0) -> list[Cmd]:
    return await self.peek(runs, self.trainer_heads, block_ms)

  async def claim_trainer(self, run: str, token: str) -> Cmd | None:
    row = self.runs.get(run)
    if not row or row["owner"] != token:
      raise RuntimeError(f"lost ownership of {run}")
    if not (queue := self.queues.get(run)):
      return None
    eid = next(iter(queue))
    cmd = queue.pop(eid)
    self.in_flight[run][eid] = Claim(cmd, life_of(token), time.monotonic(), 1)
    return cmd

  async def begin_checkpoint(self, run: str, token: str) -> str:
    row = self.runs.get(run)
    if not row or row["owner"] != token:
      raise RuntimeError(f"lost ownership of {run}")
    row["writing"] = version = uuid.uuid4().hex
    with self.db:
      self.runs.save(run)
    return version

  async def detach_run(
    self, run_id: str, token: str, pos: str | None, snap: str | None, checkpoint: dict | None = None
  ) -> bool:
    row = self.runs.get(run_id)
    if not row or row["owner"] != token or row["pos"] != pos:
      return False
    row["owner"] = row["writing"] = None
    row["snap"] = snap
    with self.db:
      if checkpoint:
        ckpt = make_checkpoint(**checkpoint)
        self.checkpoints[ckpt["tinker_path"]] = ckpt
      self.runs.save(run_id)
    self.wake()
    return True

  async def commit_reply(
    self, cmd: Cmd, reply: Reply, token: str, acct="", mutates=False, charge=None, checkpoint=None
  ) -> bool:
    row = self.runs.get(cmd.run)
    if not row or row["owner"] != token:
      return False
    if (
      self.in_flight.get(cmd.run, {}).pop(cmd.entry_id, None) is None
      and self.queues.get(cmd.run, {}).pop(cmd.entry_id, None) is None
    ):
      return False
    deleting = cmd.op == Op.DELETE_MODEL and reply.ok
    row["writing"] = None
    for name, delta in charge_counts(cmd, reply, charge).items():
      row["counters"][name] += delta
    gpu_ms = charge.gpu_ms if acct and charge and charge.gpu_ms else 0.0
    if gpu_ms:
      self.service[acct] = self.service.get(acct, 0.0) + gpu_ms
    with self.db:
      if checkpoint and reply.ok:
        ckpt = make_checkpoint(**checkpoint)
        self.checkpoints[ckpt["tinker_path"]] = ckpt
      if mutates and not deleting:
        row["pos"] = cmd.entry_id
        if row["snap"]:
          self.checkpoints.pop(unpack_snap(row["snap"])[0], None)
          row["snap"] = None
      elif deleting:
        row["live"] = False
        row["owner"] = row["snap"] = None
        for uri in [u for u, c in self.checkpoints.items() if parse_uri(u)[0] == cmd.run and c["owned"]]:
          del self.checkpoints[uri]
      self.runs.save(cmd.run)
      if gpu_ms:
        self.meta["service"] = self.service
    self.replies[cmd.request_id] = (reply, time.time() + REPLY_TTL, 0)
    if deleting:
      self.queues.pop(cmd.run, None)
      self.in_flight.pop(cmd.run, None)
      self.record_event(Ev.RUN_DELETED, {"run": cmd.run})
    self.notify(f"reply:{cmd.request_id}")
    self.wake()
    return True

  async def fail_stranded_commands(self, run_id: str, token: str) -> None:
    for eid in sorted(self.in_flight.get(run_id, {}), key=entry_sort_key):
      cmd = self.in_flight[run_id][eid].cmd
      err = Reply.error(RuntimeError(f"its trainer died, or lost ownership, during {cmd.op}"))
      if not await self.commit_reply(cmd, err, token):
        return

  # ***** stateless sampler queues *****

  def abandoned(self, run_id: str, now: float) -> list[str]:
    claims = self.in_flight.get(run_id) or {}
    return sorted((eid for eid, c in claims.items() if self.leases.get(c.consumer, 0.0) <= now), key=entry_sort_key)

  def sample_heads(self, runs: list[str]) -> list[Cmd]:
    now, out = time.monotonic(), []
    for r in runs:
      if dead := self.abandoned(r, now):
        out.append(Cmd(r, dead[0], Op.SAMPLE, b""))
      elif q := self.queues.get(r):
        c = next(iter(q.values()))
        out.append(Cmd(r, c.entry_id, c.op, b"", c.cost))
    return sorted(out, key=lambda c: (*entry_sort_key(c.entry_id), c.run))

  async def peek_sampler(self, runs: list[str], block_ms: int = 0) -> list[Cmd]:
    return await self.peek([r for r in runs if r in self.queues and self.is_sampler(r)], self.sample_heads, block_ms)

  async def take_commands(self, runs: list[str], consumer: str, count: int = 16) -> list[Cmd]:
    now, taken = time.monotonic(), []
    for r in runs:
      if r not in self.queues or not self.is_sampler(r):
        continue
      rem, claims, q = count, self.in_flight[r], self.queues[r]
      for eid in self.abandoned(r, now)[:rem]:
        c = claims[eid]
        if now - c.claimed_at < 1.0:
          continue
        c.consumer, c.claimed_at, c.deliveries = consumer, now, c.deliveries + 1
        taken.append(c.cmd)
        rem -= 1
      while rem > 0 and q:
        eid = next(iter(q))
        cmd = q.pop(eid)
        claims[eid] = Claim(cmd, consumer, now, 1)
        taken.append(cmd)
        rem -= 1
    return taken

  async def commit_sample(self, cmd: Cmd, reply: Reply, account="", charge: Charge | None = None) -> bool:
    claims = self.in_flight.get(cmd.run)
    if not claims or claims.pop(cmd.entry_id, None) is None:
      return False
    self.sample_bytes -= len(cmd.body)
    size = len(reply.body)
    self.replies[cmd.request_id] = (reply, time.time() + REPLY_TTL, size)
    self.replies.move_to_end(cmd.request_id)
    self.reply_bytes += size
    while self.reply_bytes > SAMPLE_REPLY_BYTES and len(self.replies) > 1:
      for req_id, (_, _, sz) in list(self.replies.items()):
        if sz:
          self.reply_bytes -= self.replies.pop(req_id)[2]
          break
      else:
        break
    if row := self.runs.get(cmd.run):
      for name, delta in charge_counts(cmd, reply, charge).items():
        row["counters"][name] += delta
      self.dirty_runs.add(cmd.run)
    if account and charge and charge.gpu_ms:
      self.service[account] = self.service.get(account, 0.0) + charge.gpu_ms
      self.service_dirty = True
    self.notify(f"reply:{cmd.request_id}")
    self.wake()
    return True

  # ***** checkpoints *****

  async def get_checkpoint(self, uri: str) -> dict | None:
    parse_uri(uri)
    return dict(c) if (c := self.checkpoints.get(uri)) else None

  async def checkpoint_version(self, uri: str) -> str:
    if version := (await self.get_checkpoint(uri) or {}).get("version"):
      return version
    raise ValueError(f"no checkpoint at {uri}")

  async def checkpoint_versions(self, uris: list[str]) -> dict[str, str | Exception]:
    result: dict[str, str | Exception] = {}
    for uri in uris:
      try:
        parse_uri(uri)
        ckpt = self.checkpoints.get(uri)
        result[uri] = ckpt["version"] if ckpt else ValueError(f"no checkpoint at {uri}")
      except ValueError as err:
        result[uri] = err
    return result

  async def checkpoints_of_run(self, run: str) -> list[dict]:
    items = [
      dict(c) for u, c in self.checkpoints.items() if parse_uri(u)[0] == run and c["checkpoint_type"] != "parked"
    ]
    return sorted(items, key=lambda c: c["time"])

  async def page_of_checkpoints(self, limit: int, offset: int) -> tuple[list[dict], int]:
    visible = sorted(
      (dict(c) for c in self.checkpoints.values() if c["checkpoint_type"] != "parked"),
      key=lambda c: c["time"],
      reverse=True,
    )
    return (visible[offset : offset + limit] if limit > 0 else []), len(visible)

  async def set_checkpoint_ttl(self, uri: str, ttl: int | None) -> bool:
    parse_uri(uri)
    if (ckpt := self.checkpoints.get(uri)) is None:
      return False
    ckpt["expires_at"] = time.time() + ttl if ttl else None
    with self.db:
      self.checkpoints.save(uri)
    return True

  async def unregister_checkpoints(self, *uris: str) -> int:
    for uri in uris:
      parse_uri(uri)
    with self.db:
      return sum(self.checkpoints.pop(uri, None) is not None for uri in uris)

  async def expire_checkpoints(self) -> int:
    now = time.time()
    expired = [u for u, c in self.checkpoints.items() if c["expires_at"] is not None and c["expires_at"] <= now]
    for k in [k for k, v in self.replies.items() if v[1] <= now]:
      self.reply_bytes -= self.replies.pop(k)[2]
    for k in [k for k, (_, exp) in self.seen.items() if exp <= now]:
      del self.seen[k]
    with self.db:
      self.flush_dirty()
      for u in expired:
        del self.checkpoints[u]
      for sid in [s for s, v in self.sessions.items() if not v["active"] and v["seen"] <= now - SESSION_TTL]:
        del self.sessions[sid]
    return len(expired)

  async def live_versions(self) -> list[str]:
    versions = {c["version"] for c in self.checkpoints.values() if c.get("version")}
    versions.update(r["writing"] for r in self.runs.values() if r.get("writing"))
    return sorted({v.partition("@")[0] for v in versions})

  async def sweep_checkpoints(self, store, grace: float) -> None:
    unused = await asyncio.to_thread(store.versions)
    unused.difference_update(await self.live_versions())
    await asyncio.to_thread(store.sweep, unused, grace)

  # ***** nodes and events (RAM) *****

  async def node_heartbeat(
    self, node: str, gpus: int, memory: int = 0, leases: tuple[str, ...] = (), ttl_ms: int | None = None, **info
  ) -> list[str]:
    self.nodes[node] = {"gpus": gpus, "memory": memory, "until": time.time() + NODE_MS / 1000, **info}
    now, ttl, lost = time.monotonic(), (LEASE_MS if ttl_ms is None else ttl_ms) / 1000, []
    for life in leases:
      if self.leases.get(life, 0.0) > now:
        self.leases[life] = now + ttl
      else:
        self.leases.pop(life, None)
        lost.append(life)
    return lost

  async def live_nodes(self, details: bool = False) -> dict:
    now = time.time()
    self.nodes = {k: v for k, v in self.nodes.items() if v["until"] >= now}
    return {k: dict(v) if details else v["gpus"] for k, v in self.nodes.items()}

  async def append_event(self, _kind: Ev, **body) -> str:
    ev = self.record_event(_kind, body)
    self.wake()
    return ev.id

  async def wait_wake(self, wake_id: str, block_ms: int, stream: str = "wake") -> str:
    if wake_id != "$" and self.wake_seq[stream] > int(wake_id):
      return str(self.wake_seq[stream])
    if block_ms > 0:
      with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(self.waiters.setdefault(f"wake:{stream}", asyncio.Event()).wait(), block_ms / 1000)
    return str(self.wake_seq[stream])

  async def read_events(self, after: str = "0-0", count: int = 10_000) -> list[Event]:
    cutoff = entry_sort_key(after)
    return [ev for ev in self.events if entry_sort_key(ev.id) > cutoff][:count]


def connect(target: str | None = None) -> Bus | RemoteBus:
  url = target if target is not None else os.getenv("CONTROL_URL", "")
  return RemoteBus(url) if url.startswith(("http://", "https://")) else Bus(target)
