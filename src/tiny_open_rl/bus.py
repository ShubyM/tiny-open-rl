"""RuntimeBus: Single Redis data plane for sessions, runs, trainer streams, and blocking futures.

Replaces `store.py` (561 LOC), `session_registry.py` (104 LOC), `model_metadata.py` (131 LOC),
`sampling_store.py` (72 LOC), and the `popped`/`answered`/`ledger` tracking in `telemetry.py`.

Every mutation uses native Redis commands (`HSET`, `HSETNX`, `HINCRBY`, `SADD`, `XADD`, `XREADGROUP`,
`XACK`, `XAUTOCLAIM`, `BLMOVE`) or atomic pipelines — zero in-process `asyncio.Lock`s.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis
from redis.exceptions import ResponseError


def _b(val: str | bytes) -> bytes:
  return val if isinstance(val, bytes) else str(val).encode("utf-8")


def _s(val: str | bytes | None) -> str:
  if val is None:
    return ""
  return val.decode("utf-8") if isinstance(val, bytes) else str(val)


@dataclass(frozen=True)
class StreamEntry:
  """One command claimed from `open_rl:stream:<run_id>` into the `trainer` consumer group PEL."""

  stream_id: str
  run_id: str
  req_id: str
  op: str
  wire: str  # "proto" or "json"
  body: bytes  # Raw Protobuf bytes (when wire=="proto") or UTF-8 JSON bytes


@dataclass(frozen=True)
class InFlightOp:
  """An unacknowledged entry in `open_rl:stream:<run_id>`'s Pending Entries List (`XPENDING`)."""

  stream_id: str
  req_id: str
  op: str
  worker_id: str
  idle_seconds: float


@dataclass(frozen=True)
class StreamStatus:
  """Live queue depth (`lag`) and in-flight operations read directly from the Redis Stream."""

  queued: int
  in_flight: list[InFlightOp]


class RuntimeBus:
  """Unified Redis control & data bus for Tiny-Open-RL."""

  def __init__(self, client: Any | None = None, ttl_seconds: int = 7 * 86400) -> None:
    self.redis = client if client is not None else default_redis_client()
    self.ttl = ttl_seconds
    self._rr_cursor: int = 0

  # -- 1. Sessions (Native Hashes + Sets, zero asyncio.Lock) ------------------

  async def upsert_session(
    self,
    session_id: str,
    *,
    metadata: dict[str, Any] | None = None,
    run_id: str | None = None,
  ) -> None:
    """Create or heartbeat a session and atomically attach a `run_id` if provided."""
    now = str(time.time())
    sess_key = f"open_rl:session:{session_id}"
    runs_key = f"open_rl:session:{session_id}:runs"
    async with self.redis.pipeline(transaction=True) as pipe:
      pipe.hsetnx(sess_key, "session_id", session_id)
      pipe.hsetnx(sess_key, "created_at", now)
      pipe.hset(sess_key, "last_seen_at", now)
      if metadata is not None:
        redacted = {k: v for k, v in metadata.items() if not any(s in k.lower() for s in ("key", "token", "secret", "password"))}
        pipe.hset(sess_key, "metadata", json.dumps(redacted))
      pipe.expire(sess_key, self.ttl)
      if run_id:
        pipe.sadd(runs_key, run_id)
        pipe.expire(runs_key, self.ttl)
        pipe.sadd(f"open_rl:run:{run_id}:sessions", session_id)
        pipe.expire(f"open_rl:run:{run_id}:sessions", self.ttl)
      await pipe.execute()

  async def get_session(self, session_id: str, live_ttl: float = 120.0) -> dict[str, Any] | None:
    """Read a session's Hash and attached runs Set. Liveness is `now - last_seen_at < live_ttl`."""
    sess_key = f"open_rl:session:{session_id}"
    raw_hash, raw_runs = await asyncio.gather(
      self.redis.hgetall(sess_key),
      self.redis.smembers(f"open_rl:session:{session_id}:runs"),
    )
    if not raw_hash:
      return None
    fields = {_s(k): _s(v) for k, v in raw_hash.items()}
    last_seen = float(fields.get("last_seen_at") or 0.0)
    return {
      "session_id": session_id,
      "created_at": float(fields.get("created_at") or last_seen),
      "last_seen_at": last_seen,
      "live": (time.time() - last_seen) < live_ttl,
      "metadata": json.loads(fields.get("metadata") or "{}"),
      "model_ids": sorted(_s(r) for r in raw_runs),
    }

  # -- 2. Runs & Pools (Native Hashes + Sets) ---------------------------------

  async def create_run(
    self,
    run_id: str,
    pool_id: str,
    base_model: str,
    fine_tuning_type: str = "lora",
    *,
    session_id: str | None = None,
    config: dict[str, Any] | None = None,
  ) -> None:
    """Persist run hash, add `run_id` to `open_rl:pool:<pool_id>:runs`, and initialize its stream consumer group."""
    now = str(time.time())
    run_key = f"open_rl:run:{run_id}"
    stream_key = f"open_rl:stream:{run_id}"
    try:
      await self.redis.xgroup_create(stream_key, "trainer", id="0", mkstream=True)
    except ResponseError as exc:
      if "BUSYGROUP" not in str(exc):
        raise
    async with self.redis.pipeline(transaction=True) as pipe:
      pipe.hset(
        run_key,
        mapping={
          "run_id": run_id,
          "pool_id": pool_id,
          "base_model": base_model,
          "fine_tuning_type": fine_tuning_type,
          "status": "active",
          "created_at": now,
          "steps": "0",
          "failed": "0",
          "config": json.dumps(config or {}),
        },
      )
      pipe.expire(run_key, self.ttl)
      pipe.sadd("open_rl:runs", run_id)
      pipe.sadd(f"open_rl:pool:{pool_id}:runs", run_id)
      pipe.expire(stream_key, self.ttl)
      await pipe.execute()
    if session_id:
      await self.upsert_session(session_id, run_id=run_id)

  async def get_run(self, run_id: str) -> dict[str, Any] | None:
    raw_hash, raw_sessions = await asyncio.gather(
      self.redis.hgetall(f"open_rl:run:{run_id}"),
      self.redis.smembers(f"open_rl:run:{run_id}:sessions"),
    )
    if not raw_hash:
      return None
    fields = {_s(k): _s(v) for k, v in raw_hash.items()}
    return {
      "run_id": fields["run_id"],
      "pool_id": fields["pool_id"],
      "base_model": fields["base_model"],
      "fine_tuning_type": fields.get("fine_tuning_type", "lora"),
      "status": fields.get("status", "active"),
      "created_at": float(fields.get("created_at") or 0.0),
      "steps": int(fields.get("steps") or 0),
      "failed": int(fields.get("failed") or 0),
      "config": json.loads(fields.get("config") or "{}"),
      "session_ids": sorted(_s(s) for s in raw_sessions),
    }

  async def resolve_run(self, identifier: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Resolve `identifier` as an exact `run_id`, or as a `session_id` (returning `(None, run_ids)` if >1)."""
    run = await self.get_run(identifier)
    if run is not None:
      return run, [identifier]
    sess = await self.get_session(identifier)
    if sess is None or not sess["model_ids"]:
      return None, []
    if len(sess["model_ids"]) == 1:
      only = await self.get_run(sess["model_ids"][0])
      return only, sess["model_ids"]
    return None, sess["model_ids"]

  async def delete_run(self, run_id: str) -> None:
    run = await self.get_run(run_id)
    if run is None:
      return
    pool_id = run["pool_id"]
    async with self.redis.pipeline(transaction=True) as pipe:
      pipe.srem(f"open_rl:pool:{pool_id}:runs", run_id)
      pipe.hset(f"open_rl:run:{run_id}", "status", "deleted")
      pipe.delete(f"open_rl:stream:{run_id}")
      await pipe.execute()

  # -- 3. Trainer Stream Queue (XADD / XREADGROUP / XACK / XPENDING) ----------

  async def enqueue(
    self,
    run_id: str,
    req_id: str,
    op: str,
    body: bytes | dict[str, Any],
    *,
    wire: str = "json",
  ) -> str:
    """Append a command (raw Protobuf bytes or JSON dict) to `open_rl:stream:<run_id>`."""
    raw_body = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    stream_key = f"open_rl:stream:{run_id}"
    entry_id = await self.redis.xadd(
      stream_key,
      {"run_id": _b(run_id), "req_id": _b(req_id), "op": _b(op), "wire": _b(wire), "body": raw_body},
      maxlen=10000,
    )
    return _s(entry_id)

  async def read_batch(
    self,
    pool_id: str,
    worker_id: str,
    *,
    max_count: int = 64,
    block_ms: int = 200,
  ) -> list[StreamEntry]:
    """Read the next batch for one run in `pool_id` (round-robin across active runs in the pool).

    Claimed entries stay in the stream's Pending Entries List (PEL) under `worker_id` until `complete()` runs.
    """
    raw_runs = await self.redis.smembers(f"open_rl:pool:{pool_id}:runs")
    runs = sorted(_s(r) for r in raw_runs)
    if not runs:
      if block_ms > 0:
        await asyncio.sleep(block_ms / 1000.0)
      return []

    shift = self._rr_cursor % len(runs)
    self._rr_cursor += 1
    ordered_runs = runs[shift:] + runs[:shift]

    # Try each run in round-robin order without blocking first so one stream gets drained per turn.
    for idx, run_id in enumerate(ordered_runs):
      stream_key = f"open_rl:stream:{run_id}"
      wait = block_ms if idx == len(ordered_runs) - 1 else 0
      resp = await self.redis.xreadgroup(
        groupname="trainer",
        consumername=worker_id,
        streams={stream_key: ">"},
        count=max_count,
        block=wait if wait > 0 else None,
      )
      if resp:
        return self._parse_xread_response(resp)
    return []

  def _parse_xread_response(self, resp: Any) -> list[StreamEntry]:
    entries: list[StreamEntry] = []
    for _stream_name, messages in resp:
      for msg_id, fields in messages:
        entries.append(self._parse_entry(msg_id, fields))
    return entries

  def _parse_entry(self, msg_id: Any, fields: dict[Any, Any]) -> StreamEntry:
    norm = {_s(k): v for k, v in fields.items()}
    body = norm.get("body", b"")
    return StreamEntry(
      stream_id=_s(msg_id),
      run_id=_s(norm.get("run_id")),
      req_id=_s(norm.get("req_id")),
      op=_s(norm.get("op")),
      wire=_s(norm.get("wire") or "json"),
      body=body if isinstance(body, bytes) else str(body).encode("utf-8"),
    )

  async def complete(
    self,
    entry: StreamEntry,
    result: bytes | dict[str, Any],
    *,
    wire: str = "json",
    is_error: bool = False,
    is_optim_step: bool = False,
    elapsed_seconds: float = 0.0,
    metrics: dict[str, float] | None = None,
    future_ttl: int = 300,
  ) -> None:
    """Atomically publish the future reply, XACK + XDEL the stream entry, and update run counters/ops."""
    future_key = f"open_rl:future:{entry.req_id}"
    stream_key = f"open_rl:stream:{entry.run_id}"
    run_key = f"open_rl:run:{entry.run_id}"
    envelope = self._pack_future_envelope(result, wire=wire)

    op_record = {
      "at": time.time(),
      "request_id": entry.req_id,
      "run_id": entry.run_id,
      "operation": entry.op,
      "status": "failed" if is_error else "succeeded",
      "elapsed_seconds": round(elapsed_seconds, 6),
      "metrics": metrics or {},
    }

    async with self.redis.pipeline(transaction=True) as pipe:
      pipe.rpush(future_key, envelope)
      pipe.ltrim(future_key, -1, -1)
      pipe.expire(future_key, future_ttl)
      pipe.xack(stream_key, "trainer", entry.stream_id)
      pipe.xdel(stream_key, entry.stream_id)
      if is_error:
        pipe.hincrby(run_key, "failed", 1)
      elif is_optim_step:
        pipe.hincrby(run_key, "steps", 1)
      pipe.rpush(f"open_rl:ops:{entry.run_id}", json.dumps(op_record))
      pipe.ltrim(f"open_rl:ops:{entry.run_id}", -2000, -1)
      await pipe.execute()

  async def fail_orphans(self, pool_id: str, worker_id: str, *, min_idle_ms: int = 0) -> int:
    """Claim unacknowledged PEL entries left by a crashed trainer container and fail their futures fast."""
    raw_runs = await self.redis.smembers(f"open_rl:pool:{pool_id}:runs")
    failed_count = 0
    for raw_run in raw_runs:
      run_id = _s(raw_run)
      stream_key = f"open_rl:stream:{run_id}"
      claimed = await self.redis.xautoclaim(stream_key, "trainer", worker_id, min_idle_ms, start_id="0-0", count=100)
      messages = claimed[1] if isinstance(claimed, (list, tuple)) and len(claimed) >= 2 else []
      for msg_id, fields in messages:
        if not fields:
          continue
        entry = self._parse_entry(msg_id, fields)
        err = {"type": "RequestFailedResponse", "error_message": f"Trainer worker {worker_id} restarted while {entry.op} was in flight"}
        await self.complete(entry, err, wire="json", is_error=True)
        failed_count += 1
    return failed_count

  async def stream_status(self, run_id: str) -> StreamStatus:
    """Read `queued` (`lag`) and `in_flight` (`XPENDING`) directly from `open_rl:stream:<run_id>`."""
    stream_key = f"open_rl:stream:{run_id}"
    try:
      groups = await self.redis.xinfo_groups(stream_key)
    except ResponseError:
      return StreamStatus(queued=0, in_flight=[])

    grp = next((g for g in groups if _s(g.get("name") or g.get(b"name")) == "trainer"), None)
    if grp is None:
      return StreamStatus(queued=0, in_flight=[])

    lag = int(grp.get("lag") if grp.get("lag") is not None else (grp.get(b"lag") or 0))
    pending_count = int(grp.get("pending") if grp.get("pending") is not None else (grp.get(b"pending") or 0))
    if pending_count == 0:
      return StreamStatus(queued=lag, in_flight=[])

    raw_pending = await self.redis.xpending_range(stream_key, "trainer", min="-", max="+", count=100)
    raw_entries = dict(await self.redis.xrange(stream_key, min="-", max="+"))
    in_flight: list[InFlightOp] = []
    for item in raw_pending:
      msg_id_raw = item.get("message_id") or item.get(b"message_id")
      msg_id = _s(msg_id_raw)
      consumer = _s(item.get("consumer") or item.get(b"consumer"))
      idle_ms = int(item.get("time_since_delivered") or item.get(b"time_since_delivered") or 0)
      fields = raw_entries.get(msg_id_raw) or raw_entries.get(msg_id) or {}
      norm = {_s(k): _s(v) for k, v in fields.items() if k != "body" and k != b"body"}
      in_flight.append(
        InFlightOp(
          stream_id=msg_id,
          req_id=norm.get("req_id", ""),
          op=norm.get("op", ""),
          worker_id=consumer,
          idle_seconds=round(idle_ms / 1000.0, 3),
        )
      )
    return StreamStatus(queued=lag, in_flight=in_flight)

  async def recent_ops(self, run_id: str, last: int = 50) -> list[dict[str, Any]]:
    raw = await self.redis.lrange(f"open_rl:ops:{run_id}", -last, -1)
    return [json.loads(_s(item)) for item in raw]

  # -- 4. Blocking, Idempotent Futures (BLMOVE) -------------------------------

  def _pack_future_envelope(self, payload: bytes | dict[str, Any], *, wire: str) -> bytes:
    if wire == "proto":
      raw = payload if isinstance(payload, bytes) else b""
      return b"P:" + raw
    raw_json = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    return b"J:" + raw_json

  async def set_result(self, req_id: str, result: bytes | dict[str, Any], *, wire: str = "json", ttl: int = 300) -> None:
    future_key = f"open_rl:future:{req_id}"
    envelope = self._pack_future_envelope(result, wire=wire)
    async with self.redis.pipeline(transaction=True) as pipe:
      pipe.rpush(future_key, envelope)
      pipe.ltrim(future_key, -1, -1)
      pipe.expire(future_key, ttl)
      await pipe.execute()

  async def wait_result(self, req_id: str, timeout: float = 5.0) -> tuple[str, bytes | dict[str, Any]]:
    """Wait on `open_rl:future:<req_id>` via non-destructive `BLMOVE` (`LEFT` -> `RIGHT`).

    Returns `("proto", raw_bytes)` for Protobuf results or `("json", dict)` for JSON results.
    Calling `wait_result` multiple times for the same `req_id` returns the identical result without blocking.
    """
    future_key = f"open_rl:future:{req_id}"
    raw = await self.redis.blmove(future_key, future_key, timeout, "LEFT", "RIGHT")
    if raw is None:
      return "json", {"type": "try_again", "request_id": req_id, "queue_state": "active"}
    buf = raw if isinstance(raw, bytes) else str(raw).encode("utf-8")
    if buf.startswith(b"P:"):
      return "proto", buf[2:]
    if buf.startswith(b"J:"):
      return "json", json.loads(buf[2:].decode("utf-8"))
    return "json", json.loads(buf.decode("utf-8"))


# ---------------------------------------------------------------------------
# Self-contained in-memory async Redis Stream/Hash/Set/BLMOVE implementation
# so `tiny-open-rl` runs out-of-the-box without an external daemon, while
# using the exact same `RuntimeBus` Redis commands as production.
# ---------------------------------------------------------------------------


class _MemoryPipeline:
  def __init__(self, store: MemoryStreamRedis) -> None:
    self._store = store
    self._ops: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

  async def __aenter__(self) -> _MemoryPipeline:
    return self

  async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    pass

  def __getattr__(self, name: str) -> Any:
    fn = getattr(self._store, name)

    def _record(*args: Any, **kwargs: Any) -> _MemoryPipeline:
      self._ops.append((fn, args, kwargs))
      return self

    return _record

  async def execute(self) -> list[Any]:
    return [await fn(*args, **kwargs) for fn, args, kwargs in self._ops]


class MemoryStreamRedis:
  """In-memory implementation of the Redis Hash, Set, List (`BLMOVE`), and Stream (`XREADGROUP`/`XPENDING`/`XAUTOCLAIM`) API."""

  def __init__(self) -> None:
    self._hashes: dict[str, dict[bytes, bytes]] = {}
    self._sets: dict[str, set[bytes]] = {}
    self._lists: dict[str, list[bytes]] = {}
    self._list_events: dict[str, asyncio.Event] = {}
    # stream_key -> list of (entry_id_str, dict[bytes, bytes])
    self._streams: dict[str, list[tuple[str, dict[bytes, bytes]]]] = {}
    # (stream_key, group) -> {"cursor_idx": int, "pel": dict[entry_id, (consumer, delivered_ms, fields)]}
    self._groups: dict[tuple[str, str], dict[str, Any]] = {}
    self._stream_events: dict[str, asyncio.Event] = {}
    self._seq: int = 0

  def pipeline(self, transaction: bool = True) -> _MemoryPipeline:
    return _MemoryPipeline(self)

  async def expire(self, key: str, seconds: int) -> bool:
    return True

  async def delete(self, *keys: str) -> int:
    for k in keys:
      self._hashes.pop(k, None)
      self._sets.pop(k, None)
      self._lists.pop(k, None)
      self._streams.pop(k, None)
    return len(keys)

  # -- Hashes -----------------------------------------------------------------

  async def hset(self, key: str, key_field: str | None = None, value: Any = None, mapping: dict[str, Any] | None = None) -> int:
    h = self._hashes.setdefault(key, {})
    if key_field is not None:
      h[_b(key_field)] = _b(value)
    for k, v in (mapping or {}).items():
      h[_b(k)] = _b(v)
    return 1

  async def hsetnx(self, key: str, field: str, value: Any) -> int:
    h = self._hashes.setdefault(key, {})
    fb = _b(field)
    if fb in h:
      return 0
    h[fb] = _b(value)
    return 1

  async def hincrby(self, key: str, field: str, amount: int = 1) -> int:
    h = self._hashes.setdefault(key, {})
    fb = _b(field)
    cur = int(_s(h.get(fb, b"0"))) + amount
    h[fb] = _b(str(cur))
    return cur

  async def hgetall(self, key: str) -> dict[bytes, bytes]:
    return dict(self._hashes.get(key, {}))

  # -- Sets -------------------------------------------------------------------

  async def sadd(self, key: str, *values: Any) -> int:
    s = self._sets.setdefault(key, set())
    before = len(s)
    for v in values:
      s.add(_b(v))
    return len(s) - before

  async def srem(self, key: str, *values: Any) -> int:
    s = self._sets.get(key)
    if not s:
      return 0
    before = len(s)
    for v in values:
      s.discard(_b(v))
    return before - len(s)

  async def smembers(self, key: str) -> set[bytes]:
    return set(self._sets.get(key, set()))

  # -- Lists & BLMOVE ---------------------------------------------------------

  async def rpush(self, key: str, *values: Any) -> int:
    lst = self._lists.setdefault(key, [])
    for v in values:
      lst.append(_b(v))
    if key in self._list_events:
      self._list_events[key].set()
    return len(lst)

  async def ltrim(self, key: str, start: int, end: int) -> bool:
    lst = self._lists.get(key)
    if lst is not None:
      n = len(lst)
      s = max(0, n + start if start < 0 else start)
      e = (n + end if end < 0 else end) + 1
      self._lists[key] = lst[s:e]
    return True

  async def lrange(self, key: str, start: int, end: int) -> list[bytes]:
    lst = self._lists.get(key, [])
    n = len(lst)
    s = max(0, n + start if start < 0 else start)
    e = (n + end if end < 0 else end) + 1
    return list(lst[s:e])

  async def blmove(self, first_list: str, second_list: str, timeout: float, src: str = "LEFT", dest: str = "RIGHT") -> bytes | None:
    deadline = time.monotonic() + max(0.0, float(timeout or 0.0))
    while True:
      lst = self._lists.get(first_list)
      if lst:
        val = lst.pop(0 if src == "LEFT" else -1)
        dst = self._lists.setdefault(second_list, [])
        if dest == "RIGHT":
          dst.append(val)
        else:
          dst.insert(0, val)
        return val
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        return None
      ev = self._list_events.setdefault(first_list, asyncio.Event())
      ev.clear()
      try:
        await asyncio.wait_for(ev.wait(), timeout=remaining)
      except TimeoutError:
        return None

  # -- Streams (XGROUP, XADD, XREADGROUP, XACK, XDEL, XAUTOCLAIM, XPENDING) ---

  async def xgroup_create(self, name: str, groupname: str, id: str = "$", mkstream: bool = False) -> bool:
    if mkstream:
      self._streams.setdefault(name, [])
    key = (name, groupname)
    if key in self._groups:
      raise ResponseError("BUSYGROUP Consumer Group name already exists")
    stream = self._streams.get(name, [])
    self._groups[key] = {"last_seq": 0 if id == "0" else (int(stream[-1][0].split("-")[1]) if stream else 0), "pel": {}}
    return True

  async def xadd(self, name: str, fields: dict[str, Any], id: str = "*", maxlen: int | None = None) -> bytes:
    self._seq += 1
    entry_id = f"{int(time.time() * 1000)}-{self._seq}" if id == "*" else id
    encoded = {_b(k): _b(v) for k, v in fields.items()}
    stream = self._streams.setdefault(name, [])
    stream.append((entry_id, encoded))
    if maxlen and len(stream) > maxlen:
      del stream[: len(stream) - maxlen]
    if name in self._stream_events:
      self._stream_events[name].set()
    return _b(entry_id)

  async def xreadgroup(
    self,
    groupname: str,
    consumername: str,
    streams: dict[str, str],
    count: int | None = None,
    block: int | None = None,
  ) -> list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]]:
    limit = count or 64
    deadline = time.monotonic() + ((block or 0) / 1000.0)
    while True:
      out: list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]] = []
      now_ms = int(time.time() * 1000)
      for stream_key, pos in streams.items():
        grp = self._groups.get((stream_key, groupname))
        if grp is None:
          continue
        stream = self._streams.get(stream_key, [])
        if pos == ">":
          unread = [(eid, flds) for eid, flds in stream if int(eid.split("-")[1]) > grp["last_seq"]]
          picked = unread[:limit]
          if picked:
            grp["last_seq"] = int(picked[-1][0].split("-")[1])
            for eid, flds in picked:
              grp["pel"][eid] = {"consumer": consumername, "delivered_ms": now_ms, "fields": dict(flds)}
            out.append((_b(stream_key), [(_b(eid), dict(flds)) for eid, flds in picked]))
      if out or not block or time.monotonic() >= deadline:
        return out
      first_stream = next(iter(streams))
      ev = self._stream_events.setdefault(first_stream, asyncio.Event())
      ev.clear()
      try:
        await asyncio.wait_for(ev.wait(), timeout=max(0.001, deadline - time.monotonic()))
      except TimeoutError:
        return []

  async def xack(self, name: str, groupname: str, *ids: str) -> int:
    grp = self._groups.get((name, groupname))
    if not grp:
      return 0
    acked = 0
    for eid in ids:
      if grp["pel"].pop(_s(eid), None) is not None:
        acked += 1
    return acked

  async def xdel(self, name: str, *ids: str) -> int:
    stream = self._streams.get(name)
    if not stream:
      return 0
    drop = {_s(i) for i in ids}
    before = len(stream)
    self._streams[name] = [(eid, flds) for eid, flds in stream if eid not in drop]
    return before - len(self._streams[name])

  async def xautoclaim(
    self,
    name: str,
    groupname: str,
    consumername: str,
    min_idle_time: int,
    start_id: str = "0-0",
    count: int = 100,
  ) -> tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]], list[bytes]]:
    grp = self._groups.get((name, groupname))
    if not grp:
      return (b"0-0", [], [])
    now_ms = int(time.time() * 1000)
    claimed: list[tuple[bytes, dict[bytes, bytes]]] = []
    for eid, info in list(grp["pel"].items()):
      if now_ms - info["delivered_ms"] >= min_idle_time and len(claimed) < count:
        info["consumer"] = consumername
        info["delivered_ms"] = now_ms
        claimed.append((_b(eid), dict(info["fields"])))
    return (b"0-0", claimed, [])

  async def xinfo_groups(self, name: str) -> list[dict[str, Any]]:
    if name not in self._streams:
      raise ResponseError("ERR no such key")
    stream = self._streams[name]
    out: list[dict[str, Any]] = []
    for (s_name, g_name), grp in self._groups.items():
      if s_name == name:
        lag = sum(1 for eid, _ in stream if int(eid.split("-")[1]) > grp["last_seq"])
        out.append({"name": g_name, "pending": len(grp["pel"]), "lag": lag})
    return out

  async def xpending_range(self, name: str, groupname: str, min: str = "-", max: str = "+", count: int = 100) -> list[dict[str, Any]]:
    grp = self._groups.get((name, groupname))
    if not grp:
      return []
    now_ms = int(time.time() * 1000)
    return [
      {
        "message_id": _b(eid),
        "consumer": _b(info["consumer"]),
        "time_since_delivered": max_val if (max_val := now_ms - info["delivered_ms"]) > 0 else 0,
        "times_delivered": 1,
      }
      for eid, info in list(grp["pel"].items())[:count]
    ]

  async def xrange(self, name: str, min: str = "-", max: str = "+") -> list[tuple[bytes, dict[bytes, bytes]]]:
    return [(_b(eid), dict(flds)) for eid, flds in self._streams.get(name, [])]


def default_redis_client() -> Any:
  redis_url = os.environ.get("REDIS_URL")
  if redis_url:
    return redis.from_url(redis_url, decode_responses=False, socket_timeout=30)
  return MemoryStreamRedis()
