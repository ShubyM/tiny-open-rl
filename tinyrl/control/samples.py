"""Ephemeral in-memory sampling queues and bounded LRU reply cache (429 backpressure, 410 on lost promises)."""

from __future__ import annotations

import itertools
import time
from collections import OrderedDict
from dataclasses import dataclass, replace

from tinyrl.control.db import DB
from tinyrl.helpers import getenv
from tinyrl.types import Cmd, Owner, Reply, Work

SAMPLE_QUEUE_BYTES = getenv("SAMPLE_QUEUE_BYTES", 256 * 2**20)
SAMPLE_REPLY_BYTES = getenv("SAMPLE_REPLY_BYTES", 256 * 2**20)
STABILITY_TTL, STABILITY_BYTES = getenv("STABILITY_TTL", 24 * 3600), getenv("STABILITY_BYTES", 512 * 2**20)


class RequestGoneError(LookupError):
  """Raised when a sampling request is no longer in memory (maps to HTTP 410)."""


class QueueFullError(RuntimeError):
  """Raised when in-memory sampling queues hit SAMPLE_QUEUE_BYTES (maps to HTTP 429)."""


@dataclass
class Entry:
  cmd: Cmd
  version: str | None = None
  owner: Owner | None = None
  block: tuple[int, ...] = ()
  since: float = 0.0
  attempt: int = 0

  def work(self) -> Work:
    return Work(self.cmd, version=self.version, attempt=self.attempt)


def flight_status(total: int, active: list[tuple[str, str, float]], now: float) -> dict:
  fly = [{"request_id": k, "trainer": w, "deliveries": 1, "seconds": max(0.0, now - t)} for k, w, t in active]
  return {"live": True, "queued": total - len(fly), "in_flight": fly}


class SampleQueue:
  def __init__(self, db: DB):
    self.db = db
    self.queues, self.replies, self.seen = {}, {}, {}
    self.by_id: OrderedDict[str, Entry] = OrderedDict()
    self.queued_bytes = self.reply_bytes = 0

  def backlog(self, run_id: str) -> int:
    return sum(len(group) for group in self.queues.get(run_id, {}).values())

  def held(self, owner: Owner) -> list[Work]:
    return [e.work() for e in self.by_id.values() if e.owner == owner]

  def claimed_blocks(self):
    return ((e.owner, e.cmd.run, e.block) for e in self.by_id.values() if e.owner)

  def versions(self):
    return {e.version for e in self.by_id.values() if e.version}

  def drop(self, run_id: str) -> list[str]:
    entries = [e for group in self.queues.pop(run_id, {}).values() for e in group.values()]
    self.queued_bytes -= sum(len(e.cmd.body) for e in entries)
    for e in entries:
      self.by_id.pop(e.cmd.request_id, None)
    self.seen = {k: v for k, v in self.seen.items() if k[0] != run_id}
    return [e.cmd.request_id for e in entries]

  def seen_id(self, run_id: str, seq_id: int) -> str | None:
    return hit[0] if (hit := self.seen.get((run_id, seq_id))) and hit[1] > time.time() else None

  def submit(self, cmd: Cmd, seq_id: int | None, session_ttl: float, version: str | None = None) -> None:
    if self.queued_bytes + len(cmd.body) > SAMPLE_QUEUE_BYTES:
      raise QueueFullError("sampling queue memory limit reached")
    group = self.queues.setdefault(cmd.run, {}).setdefault(version, {})
    group[cmd.entry_id] = self.by_id[cmd.request_id] = Entry(cmd, version)
    self.queued_bytes += len(cmd.body)
    if seq_id is not None:
      self.seen[cmd.run, seq_id] = (cmd.request_id, time.time() + session_ttl)

  def read_reply(self, req_id: str, read_ttl: float) -> Reply | None:
    if hit := self.replies.get(req_id):
      if hit[1] > (now := time.time()):
        self.replies[req_id] = (hit[0], min(hit[1], now + read_ttl), True)
        return hit[0]
      self.reply_bytes -= len(self.replies.pop(req_id)[0].body)
    if req_id not in self.by_id:
      raise RequestGoneError(f"sampling request not found: {req_id}")
    return None

  def queued(self, run: str, limit: int, version: str | None = None):
    group = self.queues.get(run, {}).get(version, {})
    return itertools.islice((e.cmd for e in group.values() if e.owner is None), limit)

  def heads(self, run: str):
    for version in self.queues.get(run, ()):
      if cmd := next(self.queued(run, 1, version), None):
        yield cmd, version

  def claim(self, cmd: Cmd, owner: Owner, now: float, block: tuple[int, ...] = ()) -> Work:
    # Iterating claimed entries must replay the same order as admission, including interleaved runs.
    entry = self.by_id[cmd.request_id]
    entry.owner, entry.since, entry.block = owner, now, block
    self.by_id.move_to_end(cmd.request_id)
    return entry.work()

  def complete(self, req_id: str, owner: Owner | None, reply: Reply, ttl: float, attempt: int = 0) -> Cmd | None:
    if (entry := self.by_id.get(req_id)) is None or (entry.owner, entry.attempt) != (owner, attempt):
      return None
    if reply.ok and reply.stability:
      with self.db:
        self.db.record_stability(reply.stability, time.time() + STABILITY_TTL, STABILITY_BYTES)
      reply = replace(reply, stability=None)
    del self.by_id[req_id]
    groups = self.queues[entry.cmd.run]
    del groups[entry.version][entry.cmd.entry_id]
    if not groups[entry.version]:
      del groups[entry.version]
    self.queued_bytes, self.reply_bytes = self.queued_bytes - len(entry.cmd.body), self.reply_bytes + len(reply.body)
    self.replies[req_id] = (reply, time.time() + ttl, False)
    while self.reply_bytes > SAMPLE_REPLY_BYTES and len(self.replies) > 1:
      oldest = next((k for k, (r, _, read) in self.replies.items() if read and r.body), None)
      oldest = oldest or next((k for k, (r, _, _) in self.replies.items() if k != req_id and r.body), None)
      if oldest is None:
        break
      self.reply_bytes -= len(self.replies.pop(oldest)[0].body)
    return entry.cmd

  def reclaim(self, pred) -> None:
    for e in self.by_id.values():
      if e.owner and pred(e):
        e.attempt += 1
        e.owner, e.since, e.block = None, 0.0, ()

  def status(self, run_id: str) -> dict:
    entries = (e for group in self.queues.get(run_id, {}).values() for e in group.values())
    active = [(e.cmd.request_id, e.owner[0], e.since) for e in entries if e.owner]
    return flight_status(self.backlog(run_id), active, time.monotonic())

  def gc(self, now: float) -> None:
    self.db.conn.execute("DELETE FROM stability WHERE expires<=?", (now,))
    for k in [k for k, (_, exp, _) in self.replies.items() if exp <= now]:
      self.reply_bytes -= len(self.replies.pop(k)[0].body)
    self.seen = {k: v for k, v in self.seen.items() if v[1] > now}
