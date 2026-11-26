"""Client state ledger: sessions, runs, queues, bounded sample cache, checkpoints, and billing."""

from __future__ import annotations

import asyncio
import contextlib
import time
import weakref
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace

from tinyrl.control import policy
from tinyrl.control.accounting import Charge
from tinyrl.control.db import DB
from tinyrl.helpers import getenv
from tinyrl.types import (
  Checkpoint,
  Cmd,
  Ev,
  Event,
  Op,
  Owner,
  QueueFullError,
  Reply,
  RequestGoneError,
  Role,
  Run,
  RunStatus,
  WorkerSpec,
  entry_sort_key,
)

LIVE_SECONDS, READ_REPLY_TTL = 30, 60
REPLY_TTL, SESSION_TTL = getenv("REPLY_TTL", 3600), getenv("SESSION_TTL", 7 * 24 * 3600)
SAMPLE_QUEUE_BYTES, SAMPLE_REPLY_BYTES = (
  getenv("SAMPLE_QUEUE_BYTES", 256 * 2**20),
  getenv("SAMPLE_REPLY_BYTES", 256 * 2**20),
)
STABILITY_TTL, STABILITY_BYTES = getenv("STABILITY_TTL", 24 * 3600), getenv("STABILITY_BYTES", 512 * 2**20)
COUNTERS = ("steps", "failed", "cmds", "tokens", "busy_ms", "wait_ms")


@dataclass
class QueueItem:
  cmd: Cmd
  uri: str | None = None
  version: str | None = None
  stab_vers: tuple[str, ...] = ()
  task_id: str | None = None
  owner: Owner | None = None
  since: float = 0.0
  attempt: int = 0


@dataclass
class RunState:
  run: Run
  created: float
  status: RunStatus = RunStatus.FRESH
  owner: Owner | None = None
  snapshot: str | None = None
  clients: int = 0
  deleted_at: float = 0.0
  fair_ms: float | None = None
  counters: dict[str, int] = field(default_factory=lambda: dict.fromkeys(COUNTERS, 0))
  queue: dict[str, QueueItem] = field(default_factory=dict)

  @property
  def live(self) -> bool:
    return self.status != RunStatus.DELETED


class SampleCache:
  """Bounded volatile memory for queued sample bodies (`429`), reply LRU (`410`), and seq dedup."""

  def __init__(self) -> None:
    self.queue_bytes = self.reply_bytes = 0
    self.replies: OrderedDict[str, tuple[Reply, float, bool]] = OrderedDict()
    self.seen: dict[tuple[str, int], tuple[str, float]] = {}

  def reserve(self, size: int, limit: int) -> None:
    if self.queue_bytes + size > limit:
      raise QueueFullError("sampling queue memory limit reached")
    self.queue_bytes += size

  def release(self, size: int) -> None:
    self.queue_bytes -= size

  def remember(self, run_id: str, seq_id: int, req_id: str, expires: float) -> None:
    self.seen[run_id, seq_id] = (req_id, expires)

  def lookup_seq(self, run_id: str, seq_id: int, now: float) -> str | None:
    hit = self.seen.get((run_id, seq_id))
    return hit[0] if hit and hit[1] > now else None

  def drop_run(self, run_id: str) -> None:
    self.seen = {k: v for k, v in self.seen.items() if k[0] != run_id}

  def put_reply(self, req_id: str, reply: Reply, expires: float, limit: int) -> None:
    self.replies[req_id] = (reply, expires, False)
    self.reply_bytes += len(reply.body)
    while self.reply_bytes > limit and len(self.replies) > 1:
      victim = next((k for k, (r, _, was_read) in self.replies.items() if was_read and r.body), None) or next(
        (k for k, (r, _, _) in self.replies.items() if k != req_id and r.body), None
      )
      if victim is None:
        return
      self.reply_bytes -= len(self.replies.pop(victim)[0].body)

  def read_reply(self, req_id: str, now: float, read_ttl: float) -> Reply | None:
    hit = self.replies.get(req_id)
    if not hit or hit[1] <= now:
      return None
    reply, expires, _ = hit
    self.replies.move_to_end(req_id)
    self.replies[req_id] = (reply, min(expires, now + read_ttl), True)
    return reply

  def gc(self, now: float) -> None:
    for req_id, (_, expires, _) in list(self.replies.items()):
      if expires <= now:
        self.reply_bytes -= len(self.replies.pop(req_id)[0].body)
    self.seen = {k: v for k, v in self.seen.items() if v[1] > now}


class Ledger:
  """Owns all logical run, session, checkpoint, command, reply, and EEVDF account state."""

  def __init__(self, db: DB):
    self.db = db
    self.runs: dict[str, RunState] = {}
    self.samples = SampleCache()
    self.service, self.lag, self.seen, self.weights = db.load_accounts()
    self.seq = db.load_seq()
    self.accounts_dirty = False
    self.events: deque[Event] = deque(maxlen=100_000)
    self.waiters: weakref.WeakValueDictionary[str, asyncio.Event] = weakref.WeakValueDictionary()

  def _apply_runs(self, updates: list[RunState]) -> None:
    for u in updates:
      s = self.runs[u.run.id]
      s.status, s.owner, s.snapshot, s.clients, s.deleted_at, s.fair_ms, s.counters = (
        u.status,
        u.owner,
        u.snapshot,
        u.clients,
        u.deleted_at,
        u.fair_ms,
        u.counters,
      )

  def load(self) -> list[tuple[str, str, object]]:
    """Populate `runs` and durable queues from SQLite; return claimed `(node_id, boot_id, task|None)`."""
    claims: list[tuple[str, str, object]] = []
    for rid, run, created, status, owner, snapshot, clients, deleted_at, counters in self.db.load_runs():
      self.runs[rid] = state = RunState(run, created, status, owner, snapshot, clients, deleted_at, counters=counters)
      if state.live and owner:
        claims.append((owner[1], owner[2], None))
    for cmd_id, seq, cmd, uri, stab_vers, claim in self.db.load_commands():
      self.seq = max(self.seq, seq)
      state = self.runs.get(cmd.run)
      if state is None or not state.live:
        continue
      item = QueueItem(cmd, uri=uri, stab_vers=stab_vers)
      if claim:
        node_id, boot_id, claimed_at, task = claim
        item.task_id, item.owner, item.since = task.id, (task.worker, node_id, boot_id), claimed_at
        claims.append((node_id, boot_id, task))
      state.queue[cmd_id] = item
    return claims

  def flush_sql(self, service: dict[str, float] | None = None) -> None:
    self.db.save_accounts(self.service if service is None else service, self.lag, self.seen, self.weights, self.seq)

  def notify(self, req_id: str) -> None:
    if waiter := self.waiters.pop(req_id, None):
      waiter.set()

  def record_event(self, kind: Ev, body: dict) -> Event:
    self.seq += 1
    now = time.time()
    event = Event(f"{int(now * 1000)}-{self.seq}", now, str(Ev(kind)), body)
    self.events.append(event)
    return event

  def read_events(self, after: str = "0-0", count: int = 10_000) -> list[Event]:
    cursor = entry_sort_key(after)
    return [ev for ev in self.events if entry_sort_key(ev.id) > cursor][:count]

  def is_sampler(self, run_id: str, req_id: str = "") -> bool:
    state = self.runs.get(run_id)
    return state.run.role == Role.SAMPLER if state is not None else req_id.startswith("samp-")

  def account(self, run_id: str) -> str:
    state = self.runs.get(run_id)
    if state is None or not state.live:
      return run_id
    if state.run.account:
      return state.run.account
    parent = self.runs.get(state.run.train_run or "")
    if parent is not None and parent.live:
      return self.account(state.run.train_run)
    return state.run.session or run_id

  def has_live_spec(self, state: RunState) -> bool:
    if not state.live:
      return False
    parent_id = state.run.spec.train_run
    return not parent_id or ((parent := self.runs.get(parent_id)) is not None and parent.live)

  def demand_runs(self) -> list[tuple[str, WorkerSpec, int]]:
    return [(self.account(rid), s.run.spec, len(s.queue)) for rid, s in self.runs.items() if self.has_live_spec(s)]

  def advance_lag(self, capacity: dict[str, int]) -> None:
    demand = policy.account_demand(self.demand_runs(), capacity)
    next_lag, next_seen = policy.update_lag(
      self.lag, self.seen, self.service, float(sum(capacity.values())), demand, self.weights
    )
    if next_lag != self.lag or next_seen != self.seen:
      self.lag, self.seen, self.accounts_dirty = next_lag, next_seen, True

  def ready_lag(self, run_id: str) -> float:
    return policy.ready_lag(self.lag, self.seen, self.service, self.account(run_id))

  # --- sessions, runs & checkpoints ---

  def heartbeat_session(self, session_id: str, metadata: dict | None = None, *, finish: bool = False) -> None:
    with self.db:
      self.db.save_session(session_id, time.time(), metadata, finish=finish)

  def get_session(self, session_id: str) -> dict | None:
    if (loaded := self.db.load_session(session_id)) is None:
      return None
    created, seen, metadata = loaded
    runs = sorted(rid for rid, s in self.runs.items() if s.run.session == session_id)
    return {
      "session_id": session_id,
      "created": created,
      "seen": seen,
      "live": time.time() - seen < LIVE_SECONDS,
      "runs": runs,
      "metadata": metadata,
    }

  def add_run(self, run: Run) -> None:
    state = RunState(run, time.time())
    with self.db:
      self.db.save_run(state)
    self.runs[run.id] = state
    info = {k: getattr(run, k) for k in ("role", "base", "rank", "exclusive", "train_run", "gpus")}
    self.record_event(Ev.RUN_CREATED, {"run": run.id, **info})

  def join_sampling_session(self, run_id: str) -> int:
    if (state := self.runs.get(run_id)) is None:
      return 1
    updated = replace(state, clients=state.clients + 1)
    with self.db:
      self.db.save_run(updated)
    self._apply_runs([updated])
    return state.clients

  def get_run(self, run_id: str) -> dict | None:
    if (state := self.runs.get(run_id)) is None:
      return None
    info = {**asdict(state.run), "created": state.created, "status": str(state.status), **state.counters}
    if state.owner:
      info["owner"] = list(state.owner)
    if state.snapshot:
      info["snapshot"] = state.snapshot
    return info

  def remove_run(self, run_id: str) -> tuple[list[tuple[Owner, str]], list[tuple[Owner, str]]]:
    """Mark `run_id` deleted; return `(cancelled_samples, aborted_trains)` as `(owner, task_id)` pairs."""
    cancelled_samples: list[tuple[Owner, str]] = []
    aborted_trains: list[tuple[Owner, str]] = []
    if (state := self.runs.get(run_id)) is not None:
      deleted_reply, queued_trains = Reply.error(RuntimeError(f"run {run_id} was deleted")), []
      for cmd_id, item in list(state.queue.items()):
        if state.run.role == Role.SAMPLER:
          self.samples.release(len(item.cmd.body))
          if item.owner and item.task_id:
            cancelled_samples.append((item.owner, item.task_id))
        elif item.owner:
          self.finish_train(cmd_id, item.owner, deleted_reply)
          if item.task_id:
            aborted_trains.append((item.owner, item.task_id))
        else:
          queued_trains.append(cmd_id)
      self.samples.drop_run(run_id)
      now, dropped = time.time(), list(state.queue)
      updated = replace(state, status=RunStatus.DELETED, owner=None, snapshot=None, deleted_at=now)
      with self.db:
        for cmd_id in queued_trains:
          self.db.save_reply(cmd_id, deleted_reply, now + REPLY_TTL)
        self.db.save_run(updated)
        self.db.remove_command(None, run_id, deleting=True)
      self._apply_runs([updated])
      state.queue.clear()
      for cmd_id in dropped:
        self.notify(cmd_id)
    self.record_event(Ev.RUN_DELETED, {"run": run_id})
    return cancelled_samples, aborted_trains

  def put_checkpoint(self, ckpt: Checkpoint) -> None:
    with self.db:
      self.db.save_checkpoint(ckpt)

  def list_checkpoints(self, run: str | None = None) -> list[Checkpoint]:
    return self.db.list_checkpoints(time.time(), run)

  def get_checkpoint(self, uri: str) -> Checkpoint | None:
    return self.db.get_checkpoint(uri, time.time())

  def checkpoint_version(self, uri: str) -> str:
    if ckpt := self.get_checkpoint(uri):
      return ckpt.version
    raise ValueError(f"no checkpoint at {uri}")

  def set_checkpoint_ttl(self, uri: str, ttl: int | None) -> bool:
    if (ckpt := self.get_checkpoint(uri)) is None:
      return False
    self.put_checkpoint(replace(ckpt, expires_at=time.time() + ttl if ttl else None))
    return True

  def delete_checkpoint(self, uri: str) -> bool:
    if self.get_checkpoint(uri) is None:
      return False
    with self.db:
      self.db.delete_checkpoint(uri)
    return True

  def retained_versions(self, extra: set[str] = frozenset()) -> list[str]:
    versions = {c.version for c in self.list_checkpoints() if c.version} | self.db.stability_refs(time.time()) | extra
    for state in self.runs.values():
      if state.live and state.snapshot:
        versions.add(state.snapshot)
      for item in state.queue.values():
        if item.version:
          versions.add(item.version)
        versions.update(item.stab_vers)
    return sorted({ver.partition("@")[0] for ver in versions})

  # --- commands, replies & billing ---

  def seen_request(self, run_id: str, seq_id: int | None) -> str | None:
    if seq_id is None:
      return None
    now = time.time()
    return (
      self.samples.lookup_seq(run_id, seq_id, now)
      if self.is_sampler(run_id)
      else self.db.seen_dedup(run_id, seq_id, now)
    )

  def enqueue(
    self, run_id: str, op: Op, body: bytes, seq_id=None, cost=1, uri=None, stability_refs=None
  ) -> tuple[str | None, bool]:
    """Append a command; return `(request_id, newly_enqueued)`."""
    if existing := self.seen_request(run_id, seq_id):
      return existing, False
    state = self.runs.get(run_id)
    if state is None or not state.live:
      return None, False
    if not state.queue and state.fair_ms is None:
      acct, spec = self.account(run_id), state.run.spec
      peers = [
        o.fair_ms
        for k, o in self.runs.items()
        if o.live and o.fair_ms is not None and o.run.spec == spec and self.account(k) == acct
      ]
      state.fair_ms = min(peers, default=0.0)

    self.seq += 1
    now = time.time()
    cmd = Cmd(run_id, f"{time.time_ns() // 1_000_000}-{self.seq}", Op(op), body, cost)
    req_id = cmd.request_id
    if state.run.role == Role.SAMPLER:
      self.samples.reserve(len(body), SAMPLE_QUEUE_BYTES)
      ckpt = self.get_checkpoint(state.run.model_path) if state.run.model_path else None
      state.queue[req_id] = QueueItem(cmd, version=ckpt.version if ckpt else None)
      if seq_id is not None:
        self.samples.remember(run_id, seq_id, req_id, now + SESSION_TTL)
      if state.run.model_path and ckpt is None:
        self.finish_sample(req_id, None, Reply.error(ValueError(f"no checkpoint at {state.run.model_path}")), None, 0)
    else:
      with self.db:
        stab_vers = self.db.insert_command(cmd, self.seq, uri, seq_id, now + SESSION_TTL, stability_refs, now)
      state.queue[req_id] = QueueItem(replace(cmd, body=b""), uri=uri, stab_vers=stab_vers)
    return req_id, True

  def read_reply(self, req_id: str) -> Reply | None:
    now, read_ttl, run_id = time.time(), min(REPLY_TTL, READ_REPLY_TTL), req_id.partition("/")[0]
    if self.is_sampler(run_id, req_id):
      if reply := self.samples.read_reply(req_id, now, read_ttl):
        return reply
      state = self.runs.get(run_id)
      if state is None or req_id not in state.queue:
        raise RequestGoneError(f"sampling request gone: {req_id}")
      return None
    with self.db:
      return self.db.read_reply(req_id, now, read_ttl)

  async def wait_for_reply(self, req_id: str, timeout: float) -> Reply | None:
    deadline = time.monotonic() + max(0.0, timeout)
    while (reply := self.read_reply(req_id)) is None:
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        return None
      with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(self.waiters.setdefault(req_id, asyncio.Event()).wait(), remaining)
    return reply

  def queued_and_in_flight(self, run_id: str) -> dict:
    state = self.runs.get(run_id)
    if state is None or not state.live:
      return {"live": False}
    now = time.monotonic() if state.run.role == Role.SAMPLER else time.time()
    in_flight = [
      {"request_id": k, "trainer": it.owner[0], "deliveries": 1, "seconds": max(0.0, now - (it.since or now))}
      for k, it in state.queue.items()
      if it.owner
    ]
    return {"live": True, "queued": len(state.queue) - len(in_flight), "in_flight": in_flight}

  def train_head(self, run_id: str) -> QueueItem | None:
    state = self.runs.get(run_id)
    if state is None or not state.live or state.run.role != Role.TRAINER or not state.queue:
      return None
    head = next(iter(state.queue.values()))
    return head if head.owner is None else None

  def sample_heads(self, spec: WorkerSpec, staged: set[str], allow_base: bool = True) -> dict[str, QueueItem]:
    heads: dict[str, QueueItem] = {}
    for run_id, state in self.runs.items():
      if state.live and state.run.role == Role.SAMPLER and state.run.spec == spec and state.queue:
        for item in state.queue.values():
          if item.owner is None and ((allow_base and item.version is None) or item.version in staged):
            heads[run_id] = item
            break
    return heads

  def hydrate_cmd(self, req_id: str) -> Cmd:
    body, stability = self.db.load_command_payload(req_id)
    return replace(self.runs[req_id.partition("/")[0]].queue[req_id].cmd, body=body, stability=stability)

  def apply_charge(
    self, state: RunState, cmd: Cmd, reply: Reply, charge: Charge | None, counters: dict, service: dict
  ) -> float | None:
    counters["failed"] += int(not reply.ok)
    counters["steps"] += int(reply.ok and cmd.op == Op.OPTIM_STEP)
    fair_ms = state.fair_ms
    if charge is not None:
      fair_ms = fair_ms + charge.gpu_ms if fair_ms is not None else None
      counters["cmds"] += 1
      counters["busy_ms"] += charge.busy_ms
      counters["wait_ms"] += max(0, charge.started_ms - entry_sort_key(cmd.entry_id)[0])
      if reply.ok:
        counters["tokens"] += cmd.cost
      if charge.gpu_ms:
        if acct := self.account(cmd.run):
          service[acct] = service.get(acct, 0.0) + charge.gpu_ms
    return fair_ms

  def finish_train(
    self,
    req_id: str,
    owner: Owner | None,
    reply: Reply,
    charge=None,
    *,
    mutated=False,
    checkpoint=None,
    dead=False,
    loaded=True,
  ) -> bool:
    state = self.runs.get(req_id.partition("/")[0])
    item = state.queue.get(req_id) if state else None
    if state is None or item is None or item.owner != owner:
      return False
    cmd, now, deleting = item.cmd, time.time(), item.cmd.op == Op.DELETE_MODEL and reply.ok
    counters, service = dict(state.counters), dict(self.service)
    fair_ms = self.apply_charge(state, cmd, reply, charge, counters, service)
    next_state = replace(state, counters=counters, fair_ms=None if len(state.queue) <= 1 else fair_ms)
    peer_updates: list[RunState] = []

    if deleting:
      next_state = replace(next_state, status=RunStatus.DELETED, owner=None, snapshot=None, deleted_at=now)
    else:
      if mutated:
        next_state.snapshot, next_state.status = None, RunStatus.LIVE
      if dead and owner:
        next_state.owner = None
        if next_state.status == RunStatus.LIVE:
          next_state.status = RunStatus.LOST
        peer_updates = [
          replace(p, owner=None, status=RunStatus.LOST if p.status == RunStatus.LIVE else p.status)
          for p in self.runs.values()
          if p.run.id != cmd.run and p.owner == owner
        ]
      elif (
        dead
        or next_state.status in (RunStatus.FRESH, RunStatus.LOST)
        or (not loaded and next_state.status == RunStatus.SAVED)
      ):
        next_state.owner = None

    updates = [*peer_updates, next_state]
    with self.db:
      self.db.remove_command(req_id, cmd.run, deleting=deleting)
      self.db.save_reply(req_id, reply, now + REPLY_TTL)
      if deleting:
        deleted_reply = Reply.error(RuntimeError(f"run {cmd.run} was deleted"))
        for cid in state.queue:
          if cid != req_id:
            self.db.save_reply(cid, deleted_reply, now + REPLY_TTL)
      self.flush_sql(service)
      if checkpoint and reply.ok:
        self.db.save_checkpoint(checkpoint)
      for updated in updates:
        self.db.save_run(updated)

    self.service = service
    self._apply_runs(updates)
    for cid in list(state.queue) if deleting else [req_id]:
      state.queue.pop(cid, None)
      self.notify(cid)
    if deleting:
      self.record_event(Ev.RUN_DELETED, {"run": cmd.run})
    return True

  def finish_sample(self, req_id: str, owner: Owner | None, reply: Reply, charge: Charge | None, attempt: int) -> bool:
    state = self.runs.get(req_id.partition("/")[0])
    item = state.queue.get(req_id) if state else None
    if state is None or item is None or item.owner != owner or item.attempt != attempt:
      return False
    if reply.ok and reply.stability:
      with self.db:
        self.db.record_stability(reply.stability, time.time() + STABILITY_TTL, STABILITY_BYTES)
    state.queue.pop(req_id)
    self.samples.release(len(item.cmd.body))
    clean = replace(reply, stability=None)
    self.samples.put_reply(req_id, clean, time.time() + REPLY_TTL, SAMPLE_REPLY_BYTES)
    fair_ms = self.apply_charge(state, item.cmd, clean, charge, state.counters, self.service)
    state.fair_ms = fair_ms if state.queue else None
    self.accounts_dirty |= bool(charge and charge.gpu_ms)
    self.notify(req_id)
    return True

  def finish_save(self, run_id: str, owner: Owner, saved: str | None) -> None:
    state = self.runs.get(run_id)
    if state is None or state.owner != owner:
      return
    status = RunStatus.SAVED if saved else (RunStatus.LOST if state.status == RunStatus.LIVE else state.status)
    updated = replace(state, owner=None, snapshot=saved or state.snapshot, status=status)
    with self.db:
      self.db.save_run(updated)
    self._apply_runs([updated])

  def take_samples(self, predicate: Callable[[QueueItem], bool]) -> list[tuple[Owner, str]]:
    """Release matching claimed sample items back to queued; return cancelled `(owner, task_id)` pairs."""
    cancelled: list[tuple[Owner, str]] = []
    for state in self.runs.values():
      if state.run.role != Role.SAMPLER:
        continue
      for item in state.queue.values():
        if item.owner and predicate(item):
          if item.task_id:
            cancelled.append((item.owner, item.task_id))
          item.attempt, item.owner, item.task_id, item.since = item.attempt + 1, None, None, 0.0
    return cancelled

  def clear_owners(self, predicate: Callable[[RunState], bool], *, lose: bool = True) -> None:
    updates = [
      replace(s, owner=None, status=RunStatus.LOST if (lose and s.status == RunStatus.LIVE) else s.status)
      for s in self.runs.values()
      if s.owner and predicate(s)
    ]
    if updates:
      with self.db:
        for updated in updates:
          self.db.save_run(updated)
      self._apply_runs(updates)

  def gc(self, now: float) -> None:
    self.samples.gc(now)
    live_sessions, cutoff = self.db.session_ids(), now - SESSION_TTL
    self.runs = {
      r: s for r, s in self.runs.items() if s.live or s.deleted_at > cutoff or s.queue or s.run.session in live_sessions
    }
