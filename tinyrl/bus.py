"""Single-process control plane: transactional KV trainer state, RAM sampling, and one `exchange` RPC."""

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
from dataclasses import asdict

import msgspec

from tinyrl import types
from tinyrl.control import scheduler
from tinyrl.control.accounting import Charge
from tinyrl.control.fairness import Demand, Worker, blocks, plan_gpus, select_runs
from tinyrl.helpers import getenv
from tinyrl.types import Cmd, Ev, NodeReply, NodeReport, Op, Reply, Role, Run, Turn, TurnResult, parse_uri, unpack_snap

LIVE_SECONDS, SESSION_GRACE = 30, getenv("SESSION_GRACE", 60)
REPLY_TTL, READ_REPLY_TTL, SESSION_TTL = getenv("REPLY_TTL", 3600), 60, getenv("SESSION_TTL", 7 * 24 * 3600)
LEASE_MS, QUANTUM_MS = getenv("LEASE_MS", 10000), getenv("QUANTUM_MS", 2000)
MAX_QUANTUM_MS, CYCLE_GRACE_MS = getenv("MAX_QUANTUM_MS", 15000), getenv("CYCLE_GRACE_MS", 100)
SWITCH_BUDGET = float(getenv("SWITCH_BUDGET", 0.1))
SAMPLE_RUNS, SAMPLE_COMMANDS = getenv("SAMPLE_RUNS", 3), getenv("SAMPLE_COMMANDS", 128)
SAMPLE_QUEUE_BYTES = getenv("SAMPLE_QUEUE_BYTES", 256 * 2**20)
SAMPLE_REPLY_BYTES = getenv("SAMPLE_REPLY_BYTES", 256 * 2**20)
COUNTERS = ("steps", "failed", "cmds", "tokens", "busy_ms", "wait_ms")
GC_S = 10  # how often expired replies, dedupe keys and sessions are swept: their TTLs are a minute or more


class PromiseGoneError(LookupError):
  """Raised when a stateless sampling promise no longer exists in RAM (maps to HTTP 410)."""


class BackpressureError(RuntimeError):
  """Raised when in-memory sampling queues hit SAMPLE_QUEUE_BYTES (maps to HTTP 429)."""


class WorkerRuntime:
  def __init__(self, wid: str, spec: types.WorkerSpec, resident=(), bound=(), pid=None, loaded=(), park_s=0.0):
    self.id, self.spec, self.resident, self.bound, self.pid = wid, spec, resident, bound, pid
    self.loaded, self.open_grad, self.run_service = set(loaded), set(), {}
    self.hold_until = self.idle_since = self.gpu_ms = self.service_floor = 0.0
    self.park_s, self.active_turn = park_s, None


class NodeRuntime:
  def __init__(self, node: str, boot_id: str, gpus: int, memory: int = 0):
    self.node, self.boot_id, self.gpus, self.memory = node, boot_id, gpus, memory
    self.draining, self.lag, self.until, self.next_wake = False, 0.0, 0.0, None
    self.workers, self.staged, self.stage_errors = {}, set(), {}


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


def entry_sort_key(entry_id: str) -> tuple[int, int]:
  ms, seq = entry_id.split("-", 1)
  return int(ms), int(seq)


def owner_token(node: str, boot_id: str, worker: str) -> str:
  return f"{worker}@{node}:{boot_id}"


def make_checkpoint(uri: str, size: int, *, version: str, ttl=None, owned=False) -> dict:
  _, cid = parse_uri(uri)
  raw, now = cid.split("/", 1)[0], time.time()
  kind = {"weights": "training", "sampler_weights": "sampler"}.get(raw, raw)
  meta = {"checkpoint_id": cid, "checkpoint_type": kind, "time": now, "tinker_path": uri, "size_bytes": size}
  return meta | {"expires_at": now + ttl if ttl else None, "owned": bool(owned), "version": version}


class Bus:
  """Central control plane: transactional KV state + single-call `exchange` scheduler."""

  def __init__(self, path: str | None = None):
    self.path = getenv("TINYRL_DB", "/tmp/tinyrl.db") if path is None else path
    self.db = KV(self.path)
    tables = [Table(self.db, n) for n in ("sessions", "runs", "checkpoints", "meta", "cmds", "replies", "seen")]
    self.sessions, self.runs, self.checkpoints, self.meta, self.cmds, self.replies, self.seen = tables
    self.service = {k: float(v) for k, v in self.meta.get("service", {}).items()}
    self.queues = {rid: {} for rid, r in self.runs.items() if r["live"]}
    self.in_flight, self.sample_replies, self.sample_seen = {rid: {} for rid in self.queues}, OrderedDict(), {}
    self.nodes, self.reaped_boots, self.active_turns = {}, set(), {}
    self.events, self.waiters = deque(maxlen=100_000), weakref.WeakValueDictionary()
    self.work_event, self.seq = asyncio.Event(), max((c["seq"] for c in self.cmds.values()), default=0)
    self.sample_bytes, self.reply_bytes, self.dirty_runs, self.service_dirty = 0, 0, set(), False
    self.plan_due = self.gc_due = 0.0  # monotonic deadlines: 0 runs the planner, or the sweep, at the next chance
    self.plan_state, now = scheduler.from_state(self.snapshot_state()), time.monotonic()
    self.plan_state.service = self.service
    for req_id, c in sorted(self.cmds.items(), key=lambda kv: kv[1]["seq"]):
      if (rid := c["run"]) not in self.queues:
        continue
      cmd = Cmd(rid, c["eid"], Op(c["op"]), self.db.get("blobs", f"c:{req_id}") or b"", c["cost"])
      if tid := c.get("turn"):
        self.in_flight[rid][c["eid"]] = (cmd, tid, now)
        wid = (self.runs[rid].get("owner") or "").split("@", 1)[0]
        self.active_turns[tid] = (c.get("node", ""), c.get("boot", ""), Turn(tid, "train", wid, cmd=cmd))
      else:
        self.queues[rid][c["eid"]] = cmd
    pending = bool(self.active_turns) or any(r["live"] and r.get("owner") for r in self.runs.values())
    self.reap_after = now + LEASE_MS / 1000 if pending else 0.0

  def is_sampler(self, run_id: str) -> bool:
    return bool((run := self.runs.get(run_id)) and run["record"]["role"] == Role.SAMPLER)

  def flush_dirty(self) -> None:
    if self.service_dirty:
      self.meta["service"], self.service_dirty = self.service, False
    for rid in self.dirty_runs:
      if rid in self.runs:
        self.runs.save(rid)
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

  def wake_all(self) -> None:
    ev, self.work_event = self.work_event, asyncio.Event()
    ev.set()

  def record_event(self, kind: Ev, body: dict) -> types.Event:
    self.seq += 1
    now = time.time()
    ev = types.Event(f"{int(now * 1000)}-{self.seq}", now, str(Ev(kind)), body)
    self.events.append(ev)
    if kind in (Ev.RUN_CREATED, Ev.RUN_DELETED, Ev.NODE_UP, Ev.NODE_LOST):
      self.plan_due = 0.0  # the planner's inputs changed
    return ev

  def clear_owners(self, pred) -> None:
    with self.db:
      for rid, row in self.runs.items():
        if (owner := row.get("owner")) and pred(rid, owner):
          row["owner"] = row["writing"] = None
          self.runs.save(rid)

  def drop_run_cmds(self, run_id: str) -> None:
    for req_id in [k for k, c in self.cmds.items() if c["run"] == run_id]:
      del self.cmds[req_id]
      self.db.delete("blobs", f"c:{req_id}")

  def save_checkpoint(self, saved: dict) -> dict:
    ckpt = make_checkpoint(**saved)
    self.checkpoints[ckpt["tinker_path"]] = ckpt
    return ckpt

  def record_usage(self, cmd: Cmd, reply: Reply, charge: Charge | None) -> None:
    if not (row := self.runs.get(cmd.run)):
      return
    c = row["counters"]
    if not reply.ok:
      c["failed"] += 1
    elif cmd.op == Op.OPTIM_STEP:
      c["steps"] += 1
    if charge:
      c["cmds"], c["busy_ms"] = c["cmds"] + 1, c["busy_ms"] + charge.busy_ms
      c["wait_ms"] += max(0, charge.started_ms - int(cmd.entry_id.split("-")[0]))
      c["tokens"] += cmd.cost if reply.ok else 0

  # ***** sessions & runs *****

  async def heartbeat_session(self, session_id: str, metadata: dict | None = None) -> None:
    now = time.time()
    s = self.sessions.get(session_id) or {"created": now, "seen": now, "active": True, "metadata": {}}
    s.update(seen=now, active=True, **({"metadata": metadata} if metadata else {}))
    with self.db:
      self.sessions[session_id] = s

  async def get_session(self, session_id: str) -> dict | None:
    if (s := self.sessions.get(session_id)) is None:
      return None
    runs = sorted(rid for rid, r in self.runs.items() if r["record"].get("session") == session_id)
    live = time.time() - s["seen"] < LIVE_SECONDS
    out = {"session_id": session_id, "created": s["created"], "seen": s["seen"], "live": live}
    return out | {"runs": runs, "metadata": dict(s["metadata"])}

  async def finish_session(self, session_id: str) -> None:
    if (s := self.sessions.get(session_id)) and s["active"]:
      s["seen"] = 0.0
      with self.db:
        self.sessions.save(session_id)
    self.wake_all()

  async def end_session(self, session_id: str) -> None:
    live = sorted(rid for rid, r in self.runs.items() if r["live"] and r["record"].get("session") == session_id)
    has_tr = False
    for rid in live:
      if self.is_sampler(rid):
        await self.delete_run(rid)
      else:
        has_tr = bool(await self.submit_command(rid, Op.DELETE_MODEL, b"{}")) or has_tr
    if s := self.sessions.get(session_id):
      s["seen" if has_tr else "active"] = time.time() if has_tr else False
      with self.db:
        self.sessions.save(session_id)

  async def create_run(self, run: Run) -> None:
    row = {"live": True, "record": {**asdict(run), "created": time.time()}, "clients": 0}
    row |= {"owner": None, "pos": None, "snap": None, "writing": None, "counters": dict.fromkeys(COUNTERS, 0)}
    self.queues[run.id], self.in_flight[run.id] = {}, {}
    with self.db:
      self.runs[run.id] = row
    ev = {"run": run.id, **{k: getattr(run, k) for k in ("role", "base", "rank", "exclusive", "train_run", "gpus")}}
    self.record_event(Ev.RUN_CREATED, ev)
    self.wake_all()

  def snapshot_state(self, *, backlogs: bool = True, counters: bool = False) -> dict:
    nodes, runs, queues = {}, {}, {}
    for nid, n in self.nodes.items():
      on = {wid: list(w.resident) for wid, w in n.workers.items() if w.resident}
      pids = {wid: w.pid for wid, w in n.workers.items() if w.pid is not None}
      nodes[nid] = {"gpus": n.gpus, "memory": n.memory, "draining": n.draining, "lag": n.lag, "on": on, "pids": pids}
    for rid in sorted(self.runs):
      if (row := self.runs[rid])["live"]:
        extra = {"counters": dict(row["counters"])} if counters else {}
        runs[rid] = {"record": dict(row["record"]), "owner": row["owner"], **extra}
        if backlogs:
          queues[rid] = len(self.queues.get(rid, ())) + len(self.in_flight.get(rid, ()))
    plan = dict(self.meta.get("plan", {}))
    return {"nodes": nodes, "runs": runs, "queues": queues, "plan": plan, "service": dict(self.service)}

  async def read_state(self, *, backlogs: bool = True, counters: bool = False) -> dict:
    self.reap_dead_nodes()
    return self.snapshot_state(backlogs=backlogs, counters=counters)

  async def live_nodes(self, details: bool = False) -> dict:
    self.reap_dead_nodes()
    snap = self.snapshot_state(backlogs=False)["nodes"]
    return snap if details else {nid: info["gpus"] for nid, info in snap.items()}

  async def get_run(self, run_id: str) -> dict | None:
    if (row := self.runs.get(run_id)) is None:
      return None
    return {**row["record"], **{k: row[k] for k in ("owner", "pos", "snap") if row[k]}, **row["counters"]}

  async def join_sampling_session(self, run_id: str) -> int:
    if (row := self.runs.get(run_id)) is None:
      return 1
    row["clients"] += 1
    with self.db:
      self.runs.save(run_id)
    return row["clients"]

  async def delete_run(self, run_id: str) -> None:
    is_samp = self.is_sampler(run_id)
    if row := self.runs.get(run_id):
      row.update(live=False, owner=None, snap=None, writing=None)
      with self.db:
        self.runs.save(run_id)
        if not is_samp:
          self.drop_run_cmds(run_id)
        self.flush_dirty()
    for cmd in (*self.queues.pop(run_id, {}).values(), *(c for c, _, _ in self.in_flight.pop(run_id, {}).values())):
      self.sample_bytes -= len(cmd.body) if is_samp else 0
      self.notify(f"reply:{cmd.request_id}")
    if is_samp:
      for k in [k for k in self.sample_seen if k.startswith(f"{run_id}:")]:
        del self.sample_seen[k]
    self.record_event(Ev.RUN_DELETED, {"run": run_id})
    self.wake_all()

  # ***** commands & replies (durable KV for trainers, RAM for samplers) *****

  async def seen_request(self, run_id: str, seq_id: int | None) -> str | None:
    if seq_id is None:
      return None
    store = self.sample_seen if self.is_sampler(run_id) else self.seen
    return e[0] if (e := store.get(f"{run_id}:{seq_id}")) and e[1] > time.time() else None

  async def submit_command(
    self, run_id: str, op: Op, body: bytes, seq_id: int | None = None, cost: int = 1
  ) -> str | None:
    if existing := await self.seen_request(run_id, seq_id):
      return existing
    if run_id not in self.queues:
      return None
    is_samp = self.is_sampler(run_id)
    if is_samp and self.sample_bytes + len(body) > SAMPLE_QUEUE_BYTES:
      raise BackpressureError("sampling queue memory limit reached")
    self.seq += 1
    cmd, exp = Cmd(run_id, f"{time.time_ns() // 1_000_000}-{self.seq}", Op(op), body, cost), time.time() + SESSION_TTL
    if is_samp:
      self.sample_bytes += len(body)
      if seq_id is not None:
        self.sample_seen[f"{run_id}:{seq_id}"] = [cmd.request_id, exp]
    else:
      meta = {"run": run_id, "eid": cmd.entry_id, "op": str(cmd.op), "cost": cost, "seq": self.seq}
      with self.db:
        self.db.put("blobs", f"c:{cmd.request_id}", body)
        self.cmds[cmd.request_id] = meta
        if seq_id is not None:
          self.seen[f"{run_id}:{seq_id}"] = [cmd.request_id, exp]
    self.queues[run_id][cmd.entry_id] = cmd
    self.wake_all()
    return cmd.request_id

  async def wait_for_reply(self, request_id: str, timeout: float) -> Reply | None:
    run_id, _, entry_id = request_id.partition("/")
    is_samp = self.is_sampler(run_id) or request_id.startswith("samp-")
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
      now, read_ttl = time.time(), min(REPLY_TTL, READ_REPLY_TTL)
      if is_samp:
        if (entry := self.sample_replies.get(request_id)) is not None:
          reply, exp, size = entry
          if exp > now:
            self.sample_replies[request_id] = (reply, min(exp, now + read_ttl), size)
            self.sample_replies.move_to_end(request_id)
            return reply
          self.reply_bytes -= self.sample_replies.pop(request_id)[2]
        if entry_id not in self.queues.get(run_id, ()) and entry_id not in self.in_flight.get(run_id, ()):
          raise PromiseGoneError(f"sampling promise gone: {request_id}")
      elif (m := self.replies.get(request_id)) and m[2] > now:
        if (body := self.db.get("blobs", f"r:{request_id}")) is not None:
          m[2] = min(m[2], now + read_ttl)
          with self.db:
            self.replies.save(request_id)
          return Reply(bool(m[0]), m[1], body)
      if (rem := deadline - time.monotonic()) <= 0:
        return None
      with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(self.waiters.setdefault(f"reply:{request_id}", asyncio.Event()).wait(), rem)

  async def queued_and_in_flight(self, run_id: str) -> dict:
    if run_id not in self.queues:
      return {"live": False}
    now, items = time.monotonic(), []
    for cmd, tid, t0 in self.in_flight[run_id].values():
      tr = self.active_turns[tid][2].worker if tid in self.active_turns else ""
      items.append({"request_id": cmd.request_id, "trainer": tr, "deliveries": 1, "seconds": max(0.0, now - t0)})
    return {"live": True, "queued": len(self.queues[run_id]), "in_flight": items}

  # ***** single-RPC exchange protocol & central scheduling *****

  def commit_trainer_cmd(
    self, cmd: Cmd, reply: Reply, *, charge: Charge | None = None, mutated=False, saved=None, clear_owner=False
  ) -> None:
    self.in_flight.get(cmd.run, {}).pop(cmd.entry_id, None)
    self.queues.get(cmd.run, {}).pop(cmd.entry_id, None)
    row, deleting = self.runs.get(cmd.run), cmd.op == Op.DELETE_MODEL and reply.ok
    acct = self.plan_state.account(cmd.run)
    gpu_ms = charge.gpu_ms if acct and charge and charge.gpu_ms else 0.0
    if gpu_ms:
      self.service[acct] = self.service.get(acct, 0.0) + gpu_ms
    with self.db:
      self.cmds.pop(cmd.request_id, None)
      self.db.delete("blobs", f"c:{cmd.request_id}")
      self.db.put("blobs", f"r:{cmd.request_id}", reply.body)
      self.replies[cmd.request_id] = [reply.ok, reply.content_type, time.time() + REPLY_TTL]
      if row:
        row["writing"] = None
        self.record_usage(cmd, reply, charge)
        if saved and reply.ok:
          self.save_checkpoint(saved)
        if mutated and not deleting:
          row["pos"] = cmd.entry_id
          if row["snap"]:
            self.checkpoints.pop(unpack_snap(row["snap"])[0], None)
            row["snap"] = None
        elif deleting:
          row.update(live=False, owner=None, snap=None)
          self.drop_run_cmds(cmd.run)
          for uri in [u for u, c in self.checkpoints.items() if parse_uri(u)[0] == cmd.run and c["owned"]]:
            del self.checkpoints[uri]
        if clear_owner and not deleting:
          row["owner"] = None
        self.runs.save(cmd.run)
      if gpu_ms:
        self.meta["service"] = self.service
    if deleting:
      self.queues.pop(cmd.run, None)
      self.in_flight.pop(cmd.run, None)
      self.record_event(Ev.RUN_DELETED, {"run": cmd.run})
    self.notify(f"reply:{cmd.request_id}")

  def commit_sample_cmd(self, cmd: Cmd, reply: Reply, turn_id: str | None = None, charge: Charge | None = None) -> bool:
    claims = self.in_flight.get(cmd.run)
    if turn_id is not None:
      if not claims or (entry := claims.get(cmd.entry_id)) is None or entry[1] != turn_id:
        return False
      claims.pop(cmd.entry_id, None)
    elif self.queues.get(cmd.run, {}).pop(cmd.entry_id, None) is None:
      return False
    size = len(reply.body)
    self.sample_bytes, self.reply_bytes = self.sample_bytes - len(cmd.body), self.reply_bytes + size
    self.sample_replies[cmd.request_id] = (reply, time.time() + REPLY_TTL, size)
    self.sample_replies.move_to_end(cmd.request_id)
    while self.reply_bytes > SAMPLE_REPLY_BYTES and len(self.sample_replies) > 1:
      if (victim := next((k for k, (_, _, sz) in self.sample_replies.items() if sz), None)) is None:
        break
      self.reply_bytes -= self.sample_replies.pop(victim)[2]
    self.record_usage(cmd, reply, charge)
    if cmd.run in self.runs:
      self.dirty_runs.add(cmd.run)
    if (acct := self.plan_state.account(cmd.run)) and charge and charge.gpu_ms:
      self.service[acct], self.service_dirty = self.service.get(acct, 0.0) + charge.gpu_ms, True
    self.notify(f"reply:{cmd.request_id}")
    return True

  def reap_node(self, node_id: str, boot_id: str) -> None:
    if (cur := self.nodes.get(node_id)) and cur.boot_id == boot_id:
      del self.nodes[node_id]
    self.reaped_boots.add(boot_id)
    self.record_event(Ev.NODE_LOST, {"node": node_id})
    for tid, (tn, tb, turn) in list(self.active_turns.items()):
      if (tn, tb) != (node_id, boot_id):
        continue
      del self.active_turns[tid]
      if turn.kind == "train" and turn.cmd:
        err = Reply.error(RuntimeError(f"its trainer died, or lost ownership, during {turn.cmd.op}"))
        self.commit_trainer_cmd(turn.cmd, err, clear_owner=True)
      elif turn.kind == "sample":
        for cmd in reversed(turn.cmds):
          if (claims := self.in_flight.get(cmd.run)) and (e := claims.get(cmd.entry_id)) and e[1] == tid:
            claims.pop(cmd.entry_id)
            if (q := self.queues.get(cmd.run)) is not None:
              self.queues[cmd.run] = {cmd.entry_id: cmd, **q}
    self.clear_owners(lambda _, o: o.endswith(f"@{node_id}:{boot_id}"))
    self.wake_all()

  def reap_dead_nodes(self) -> None:
    now = time.monotonic()
    for n in [n for n in self.nodes.values() if n.until <= now]:
      self.reap_node(n.node, n.boot_id)
    if self.reap_after and now >= self.reap_after:
      self.reap_after, live = 0.0, {(n.node, n.boot_id) for n in self.nodes.values()}
      for tn, tb in {t[:2] for t in self.active_turns.values()} - live:
        self.reap_node(tn, tb)
      self.clear_owners(lambda _, o: "@" in o and ":" in o and tuple(o.split("@", 1)[1].rsplit(":", 1)) not in live)

  def observe_node(self, report: NodeReport) -> NodeRuntime:
    now, node = time.monotonic(), self.nodes.get(report.node)
    if node and node.boot_id != report.boot_id:
      self.reap_node(node.node, node.boot_id)
      node = None
    if node is None:
      self.nodes[report.node] = node = NodeRuntime(report.node, report.boot_id, report.gpus, report.memory)
      self.record_event(Ev.NODE_UP, {"node": report.node, "gpus": report.gpus, "memory": report.memory})
    if (node.gpus, node.memory, node.draining) != (report.gpus, report.memory, report.draining):
      self.plan_due = 0.0
    node.gpus, node.memory, node.draining, node.lag = report.gpus, report.memory, report.draining, report.lag
    node.until, node.staged = now + LEASE_MS / 1000, set(report.staged)
    node.stage_errors.update(report.stage_errors)
    reported = {ws.worker: ws for ws in report.workers}
    for wid, ws in reported.items():
      if w := node.workers.get(wid):
        w.pid, w.park_s, w.bound = ws.pid, ws.park_s, ws.bound or w.bound
        if w.active_turn is None:
          w.resident, w.loaded = ws.gpus, set(ws.loaded)
      elif proc := self.plan_state.procs.get(wid):
        node.workers[wid] = WorkerRuntime(wid, proc.spec, ws.gpus, ws.bound, ws.pid, ws.loaded, ws.park_s)
    for wid in [k for k, w in node.workers.items() if w.active_turn is None and k not in reported]:
      del node.workers[wid]
      self.clear_owners(lambda _, o, t=owner_token(node.node, node.boot_id, wid): o == t)
    return node

  def finish_once(self, node_id: str, boot_id: str, res: TurnResult) -> None:
    if (entry := self.active_turns.get(res.turn_id)) is None or entry[:2] != (node_id, boot_id):
      return
    _, _, turn = self.active_turns.pop(res.turn_id)
    node, now = self.nodes.get(node_id), time.monotonic()
    w = node.workers.get(turn.worker) if node else None
    if w and w.active_turn == res.turn_id:
      w.active_turn = None
    if w and res.ok and res.resume_s > 0:
      sw = (res.park_s + res.resume_s) / max(0.01, SWITCH_BUDGET) if res.park_s else 0.0
      w.hold_until = now + min(MAX_QUANTUM_MS / 1000, max(QUANTUM_MS / 1000, sw))

    def bill(rid: str, ch: Charge | None) -> None:
      if w and ch and ch.gpu_ms:
        w.gpu_ms += ch.gpu_ms
        w.run_service[rid] = w.run_service.get(rid, w.service_floor) + ch.gpu_ms

    if turn.kind == "train" and turn.cmd:
      cmd, rep = turn.cmd, res.reply or Reply.error(RuntimeError("trainer failed"))
      if w and res.ok:
        w.idle_since = now
        (w.loaded.add if res.loaded else w.loaded.discard)(cmd.run)
        (w.open_grad.add if cmd.op == Op.FORWARD_BACKWARD and rep.ok else w.open_grad.discard)(cmd.run)
        bill(cmd.run, res.charge)
      self.commit_trainer_cmd(cmd, rep, charge=res.charge, mutated=res.mutated, saved=res.saved, clear_owner=not res.ok)
    elif turn.kind == "sample":
      by_req, fb = {c.request_id: c for c in turn.cmds}, res.reply or Reply.error(RuntimeError("sampler failed"))
      for req_id, rep, ch in res.samples:
        if (cmd := by_req.pop(req_id, None)) and self.commit_sample_cmd(cmd, rep, turn.id, ch):
          bill(cmd.run, ch)
      for cmd in by_req.values():
        self.commit_sample_cmd(cmd, fb, turn.id)
    elif turn.kind == "snapshot":
      rid, row = turn.args["run"], self.runs.get(turn.args["run"])
      if row and row.get("owner") == owner_token(node_id, boot_id, turn.worker):
        row["owner"] = row["writing"] = None
        with self.db:
          if res.ok and res.saved:
            ckpt = self.save_checkpoint(res.saved)
            row["snap"] = f"{ckpt['tinker_path']}\n{ckpt['version']}"
          self.runs.save(rid)
      if w:
        w.loaded.discard(rid)
        w.open_grad.discard(rid)
    elif turn.kind == "close":
      self.clear_owners(lambda _, o, t=owner_token(node_id, boot_id, turn.worker): o == t)
    if w and res.ok and w.run_service:
      w.service_floor = max(w.service_floor, min(w.run_service.values()))
    drop_w = turn.kind == "close" or turn.args.get("close") or (not res.ok and turn.kind in ("train", "sample"))
    if drop_w and node:
      node.workers.pop(turn.worker, None)
    self.wake_all()

  def reconcile(self) -> list[tuple[Ev, dict]]:
    old, self.plan_due = self.meta.get("plan", {}), time.monotonic() + scheduler.TICK_MS / 1000
    self.plan_state, changes = scheduler.step(scheduler.from_state(self.snapshot_state()))
    self.plan_state.service = self.service  # ready_lag reads service billed since this step
    if (plan := scheduler.to_plan(self.plan_state)) != old:
      with self.db:
        self.meta["plan"] = plan
        self.flush_dirty()
      for kind, body in changes:
        self.record_event(kind, body)
      if any(plan[k] != old.get(k) for k in ("procs", "home")):  # lag alone moves no work
        self.wake_all()
    return changes

  def issue(self, node: NodeRuntime, turn: Turn) -> None:
    self.active_turns[turn.id] = (node.node, node.boot_id, turn)
    w = node.workers.setdefault(turn.worker, WorkerRuntime(turn.worker, turn.spec))
    w.active_turn, now = turn.id, time.monotonic()
    if turn.gpus:
      w.hold_until = now + QUANTUM_MS / 1000 if w.resident != turn.gpus else w.hold_until
      w.resident = turn.gpus
      if not w.bound and (turn.spec.role == Role.SAMPLER or turn.spec.gpus > 1):
        w.bound = turn.gpus
      for o in node.workers.values():
        if o.id != w.id and o.resident and not set(o.resident).isdisjoint(turn.gpus):
          o.resident = ()
    if turn.kind == "train" and turn.cmd:
      cmd, row = turn.cmd, self.runs[turn.cmd.run]
      self.queues[cmd.run].pop(cmd.entry_id, None)
      self.in_flight[cmd.run][cmd.entry_id] = (cmd, turn.id, now)
      row["owner"] = owner_token(node.node, node.boot_id, turn.worker)
      if cmd.op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER):
        row["writing"] = turn.args.get("version")
      with self.db:
        if c := self.cmds.get(cmd.request_id):
          c.update(turn=turn.id, node=node.node, boot=node.boot_id)
          self.cmds.save(cmd.request_id)
        self.runs.save(cmd.run)
    elif turn.kind == "sample":
      for cmd in turn.cmds:
        self.queues[cmd.run].pop(cmd.entry_id, None)
        self.in_flight[cmd.run][cmd.entry_id] = (cmd, turn.id, now)
    elif turn.kind == "snapshot" and (ver := turn.args.get("version")):
      rid = turn.args["run"]
      self.runs[rid]["writing"] = ver
      with self.db:
        self.runs.save(rid)

  def ready_trainer_cmds(self, node: NodeRuntime, wid: str) -> list[Cmd]:
    tok, out = owner_token(node.node, node.boot_id, wid), []
    for r in self.plan_state.runs(wid):
      row, q = self.runs.get(r), self.queues.get(r)
      if row and row["live"] and row.get("owner") in (None, tok) and not self.in_flight.get(r) and q:
        out.append(next(iter(q.values())))
    return out

  def ready_sampler_cmds(self, node: NodeRuntime, spec: types.WorkerSpec, prefetch: set[str]) -> list[Cmd]:
    out = []
    for rid, job in sorted(self.plan_state.jobs.items()):
      if job.spec != spec or not (q := self.queues.get(rid)):
        continue
      head, mp = next(iter(q.values())), job.record.get("model_path")
      if not mp:
        out.append(head)
        continue
      try:
        ver = self.checkpoint_version_sync(mp)
      except ValueError as err:
        self.commit_sample_cmd(head, Reply.error(err))
        continue
      if err_msg := node.stage_errors.pop(ver, None):
        self.commit_sample_cmd(head, Reply.error(RuntimeError(err_msg)))
      elif ver in node.staged:
        out.append(head)
      else:
        prefetch.add(ver)
    return out

  def schedule_node(self, node: NodeRuntime) -> list[str]:
    now, evicting = time.monotonic(), set()
    placed = {} if node.draining else {p: pr for p, pr in self.plan_state.procs.items() if pr.node == node.node}
    excl = {g for w in node.workers.values() if w.spec.exclusive for g in w.resident}

    # Phase 1: Drain, evict moved/blocked workers, and detach runs reassigned off a worker (off-GPU).
    for wid, w in list(node.workers.items()):
      if w.active_turn is not None:
        continue
      tok = owner_token(node.node, node.boot_id, wid)
      owned = [rid for rid, r in self.runs.items() if r["live"] and r.get("owner") == tok]
      blocked = not w.resident and bool(w.bound) and bool(excl & set(w.bound))
      free_blk = blocked and w.spec.gpus > 1 and any(excl.isdisjoint(b) for b in blocks(node.gpus, w.spec.gpus))
      rebind = free_blk and w.spec.role == Role.TRAINER and bool(self.ready_trainer_cmds(node, wid))
      if wid not in placed or (blocked and w.spec.role == Role.SAMPLER) or rebind:
        evicting.add(wid)
        dirty = [r for r in owned if self.runs[r]["pos"] is not None and self.runs[r]["snap"] is None and r in w.loaded]
        self.clear_owners(lambda r, o, t=tok, d=set(dirty): o == t and r not in d)
        args = {"run": dirty[0], "version": uuid.uuid4().hex, "close": len(dirty) == 1} if dirty else {}
        self.issue(node, Turn(uuid.uuid4().hex[:12], "snapshot" if dirty else "close", wid, w.spec, args=args))
      elif w.spec.role == Role.TRAINER:
        want = set(self.plan_state.runs(wid))
        for r in [r for r in owned if r not in want]:
          dirty = self.runs[r]["pos"] is not None and self.runs[r]["snap"] is None and r in w.loaded
          if dirty or r in w.loaded:
            args = {"run": r, "version": uuid.uuid4().hex if dirty else None, "drop": True}
            self.issue(node, Turn(uuid.uuid4().hex[:12], "snapshot", wid, w.spec, args=args))
            break
          self.clear_owners(lambda rid, _, target=r: rid == target)

    if node.draining:
      node.next_wake = None
      return []

    # Phase 2: Gather ready commands + off-GPU checkpoint prefetch, then assign GPUs via plan_gpus.
    prefetch, ready_by, workers_plan, occupied, linger_at = set(), {}, {}, set(), []
    for wid, proc in sorted(placed.items()):
      if wid in evicting:
        continue
      spec, is_new = proc.spec, wid not in node.workers
      w = node.workers.get(wid) or WorkerRuntime(wid, spec)
      active = w.active_turn is not None
      if active and w.resident:
        occupied.update(w.resident)
      if spec.role == Role.SAMPLER:
        ready = [] if active else self.ready_sampler_cmds(node, spec, prefetch)
        demand = Demand.READY if (ready or active or (is_new and node.gpus >= 2)) else Demand.IDLE
      else:
        ready = [] if active else self.ready_trainer_cmds(node, wid)
        grace = (w.idle_since + CYCLE_GRACE_MS / 1000) if w.resident else 0.0
        linger = bool(w.resident and grace > now)
        if linger:
          linger_at.append(grace)
        closing = any(c.op == Op.OPTIM_STEP and c.run in w.open_grad for c in ready)
        hold_grad = active or closing or (linger and not ready and bool(w.open_grad))
        demand = Demand.FINISH_STEP if hold_grad else (Demand.READY if (ready or linger) else Demand.IDLE)
      ready_by[wid] = ready
      runs = [c.run for c in ready] or [r for r in self.plan_state.runs(wid) if r in self.plan_state.jobs]
      lag = max(map(self.plan_state.ready_lag, runs), default=-w.gpu_ms)
      gang, hold = spec.role == Role.TRAINER and spec.gpus > 1, float("inf") if active else w.hold_until
      workers_plan[wid] = Worker(spec.gpus, spec.exclusive, demand, w.resident, w.bound, gang, hold, lag, w.gpu_ms)

    keep, wake_at = plan_gpus(workers_plan, node.gpus, now, occupied)
    node.next_wake = min((d for d in (wake_at, *linger_at) if d is not None and d > now), default=None)

    # Phase 3: Issue concrete turns for granted GPU blocks.
    for wid, gpus in keep.items():
      w, spec, ready = node.workers.get(wid), placed[wid].spec, ready_by.get(wid, [])
      if (w and w.active_turn is not None) or (occupied & set(gpus)):
        continue
      tid = uuid.uuid4().hex[:12]
      if not ready:
        if spec.role == Role.SAMPLER and w is None and node.gpus >= 2:
          self.issue(node, Turn(tid, "sample", wid, spec, gpus))
        continue
      svc, floor = (w.run_service, w.service_floor) if w else ({}, 0.0)
      for c in ready:
        svc[c.run] = max(svc.get(c.run, floor), floor)
      if spec.role == Role.TRAINER:
        [rid] = select_runs(ready, self.plan_state.account, self.plan_state.ready_lag, svc, 1)
        cmd, row, ver = next(iter(self.queues[rid].values())), self.runs[rid], None
        if cmd.op == Op.LOAD_WEIGHTS:
          try:
            ver = self.checkpoint_version_sync(json.loads(cmd.body)["uri"])
          except (KeyError, ValueError) as err:
            self.commit_trainer_cmd(cmd, Reply.error(err))
            continue
        elif cmd.op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER):
          ver = uuid.uuid4().hex
        need_snap = row["snap"] and not (w and rid in w.loaded) and cmd.op != Op.LOAD_WEIGHTS
        saved = unpack_snap(row["snap"])[1] if need_snap else None
        self.issue(
          node,
          Turn(
            tid, "train", wid, spec, gpus, cmd, args={"record": dict(row["record"]), "saved": saved, "version": ver}
          ),
        )
      else:
        rids = select_runs(
          ready, self.plan_state.account, self.plan_state.ready_lag, svc, SAMPLE_RUNS if spec.lora else 1
        )
        cmds = [cmd for r in rids for cmd in list(self.queues[r].values())[:SAMPLE_COMMANDS]]
        vers = {
          r: (self.checkpoints.get(self.runs[r]["record"].get("model_path") or "") or {}).get("version") for r in rids
        }
        if cmds:
          self.issue(node, Turn(tid, "sample", wid, spec, gpus, cmds=cmds, args={"versions": vers}))
    return sorted(prefetch)

  def exchange_sync(self, report: NodeReport) -> NodeReply:
    self.reap_dead_nodes()
    if report.boot_id in self.reaped_boots:
      return NodeReply(reset=True)
    node, ack = self.observe_node(report), []
    for result in report.finished:
      self.finish_once(report.node, report.boot_id, result)
      ack.append(result.turn_id)
    if time.monotonic() >= self.plan_due:
      self.reconcile()
    prefetch, live = self.schedule_node(node), self.live_versions_sync() if report.want_sweep else None
    mine = [t for tn, tb, t in self.active_turns.values() if (tn, tb) == (report.node, report.boot_id)]
    return NodeReply(ack, [t for t in mine if t.id not in report.active], prefetch, live)

  async def exchange(self, report: NodeReport, wait_ms: int = 0) -> NodeReply:
    r = self.exchange_sync(report)
    busy = report.draining or r.reset or r.acknowledged or r.turns or r.prefetch or r.live_versions is not None
    if wait_ms <= 0 or busy:
      return r
    node, timeout = self.nodes.get(report.node), wait_ms / 1000
    if node and node.next_wake is not None:
      timeout = min(timeout, max(0.0, node.next_wake - time.monotonic()))
    if timeout > 0:
      with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(self.work_event.wait(), timeout)
    return self.exchange_sync(report)

  # ***** checkpoints, maintenance, and events *****

  def checkpoint_version_sync(self, uri: str) -> str:
    parse_uri(uri)
    if version := (self.checkpoints.get(uri) or {}).get("version"):
      return version
    raise ValueError(f"no checkpoint at {uri}")

  async def get_checkpoint(self, uri: str) -> dict | None:
    parse_uri(uri)
    return dict(c) if (c := self.checkpoints.get(uri)) else None

  async def checkpoint_version(self, uri: str) -> str:
    return self.checkpoint_version_sync(uri)

  async def checkpoints_of_run(self, run: str) -> list[dict]:
    items = [
      dict(c) for u, c in self.checkpoints.items() if parse_uri(u)[0] == run and c["checkpoint_type"] != "parked"
    ]
    return sorted(items, key=lambda c: c["time"])

  async def page_of_checkpoints(self, limit: int, offset: int) -> tuple[list[dict], int]:
    vis = sorted(
      (dict(c) for c in self.checkpoints.values() if c["checkpoint_type"] != "parked"), key=lambda c: -c["time"]
    )
    return (vis[offset : offset + limit] if limit > 0 else []), len(vis)

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

  def live_versions_sync(self) -> list[str]:
    v = {c["version"] for c in self.checkpoints.values() if c.get("version")}
    v.update(r["writing"] for r in self.runs.values() if r.get("writing"))
    return sorted({x.partition("@")[0] for x in v})

  async def expire_checkpoints(self) -> int:
    now = time.time()
    expired = [u for u, c in self.checkpoints.items() if c["expires_at"] is not None and c["expires_at"] <= now]
    with self.db:
      for u in expired:
        del self.checkpoints[u]
      if time.monotonic() >= self.gc_due:  # these tables grow with every command, so scan them rarely
        self.gc_due = time.monotonic() + GC_S
        self.flush_dirty()
        for k in [k for k, v in self.sample_replies.items() if v[1] <= now]:
          self.reply_bytes -= self.sample_replies.pop(k)[2]
        for k in [k for k, (_, exp) in self.sample_seen.items() if exp <= now]:
          del self.sample_seen[k]
        for k in [k for k, v in self.replies.items() if v[2] <= now]:
          del self.replies[k]
          self.db.delete("blobs", f"r:{k}")
        for k in [k for k, (_, exp) in self.seen.items() if exp <= now]:
          del self.seen[k]
        for sid in [s for s, v in self.sessions.items() if not v["active"] and v["seen"] <= now - SESSION_TTL]:
          del self.sessions[sid]
    return len(expired)

  async def maintain(self) -> list[tuple[Ev, dict]]:
    self.reap_dead_nodes()
    cutoff = time.time() - SESSION_GRACE
    for sid in [s for s, v in list(self.sessions.items()) if v["active"] and v["seen"] <= cutoff]:
      await self.end_session(sid)
    await self.expire_checkpoints()
    return self.reconcile()

  async def read_events(self, after: str = "0-0", count: int = 10_000) -> list[types.Event]:
    cutoff = entry_sort_key(after)
    return [ev for ev in self.events if entry_sort_key(ev.id) > cutoff][:count]


class ControlClient:
  """Async HTTP client for the controller's single `POST /internal/exchange` RPC."""

  def __init__(self, url: str | None = None):
    self.url = (getenv("CONTROL_URL", "http://127.0.0.1:8000") if url is None else url).rstrip("/")
    self.session = None

  async def aclose(self) -> None:
    if self.session and not self.session.closed:
      await self.session.close()
      self.session = None

  async def exchange(self, report: NodeReport, wait_ms: int = 0) -> NodeReply:
    import aiohttp

    if self.session is None or self.session.closed:
      self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
    t = aiohttp.ClientTimeout(total=max(5.0, wait_ms / 1000 + 5.0))
    url, params = f"{self.url}/internal/exchange", {"wait_ms": str(wait_ms)}
    async with self.session.post(url, params=params, data=msgspec.msgpack.encode(report), timeout=t) as r:
      if r.status >= 400:
        raise aiohttp.ClientError(f"control server returned {r.status}")
      return msgspec.msgpack.decode(await r.read(), type=NodeReply)


def connect(target: str | None = None) -> Bus | ControlClient:
  url = target if target is not None else os.getenv("CONTROL_URL", "")
  return ControlClient(url) if url.startswith(("http://", "https://")) else Bus(target)
