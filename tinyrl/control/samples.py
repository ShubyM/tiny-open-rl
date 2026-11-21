"""Ephemeral in-memory sampling queues and bounded LRU reply cache (429 backpressure, 410 on lost promises)."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

from tinyrl.helpers import getenv
from tinyrl.types import Cmd, Reply

SAMPLE_QUEUE_BYTES = getenv("SAMPLE_QUEUE_BYTES", 256 * 2**20)
SAMPLE_REPLY_BYTES = getenv("SAMPLE_REPLY_BYTES", 256 * 2**20)


class PromiseGoneError(LookupError):
  """Raised when a stateless sampling promise no longer exists in RAM (maps to HTTP 410)."""


class BackpressureError(RuntimeError):
  """Raised when in-memory sampling queues hit SAMPLE_QUEUE_BYTES (maps to HTTP 429)."""


@dataclass
class Entry:
  cmd: Cmd
  owner: tuple[str, str, str] | None = None
  since: float = 0.0


class SampleQueue:
  def __init__(self):
    self.queues, self.by_id, self.replies, self.seen = {}, {}, {}, {}
    self.queued_bytes = self.reply_bytes = 0

  def open(self, run_id: str) -> None:
    self.queues.setdefault(run_id, {})

  def drop(self, run_id: str) -> list[str]:
    entries = list(self.queues.pop(run_id, {}).values())
    for e in entries:
      self.queued_bytes -= len(e.cmd.body)
      self.by_id.pop(e.cmd.request_id, None)
    self.seen = {k: v for k, v in self.seen.items() if k[0] != run_id}
    return [e.cmd.request_id for e in entries]

  def seen_id(self, run_id: str, seq_id: int | None) -> str | None:
    hit = None if seq_id is None else self.seen.get((run_id, seq_id))
    return hit[0] if hit and hit[1] > time.time() else None

  def submit(self, cmd: Cmd, seq_id: int | None, session_ttl: float) -> None:
    if self.queued_bytes + len(cmd.body) > SAMPLE_QUEUE_BYTES:
      raise BackpressureError("sampling queue memory limit reached")
    self.queues[cmd.run][cmd.entry_id] = self.by_id[cmd.request_id] = Entry(cmd)
    self.queued_bytes += len(cmd.body)
    if seq_id is not None:
      self.seen[cmd.run, seq_id] = (cmd.request_id, time.time() + session_ttl)

  def read_reply(self, request_id: str, read_ttl: float) -> Reply | None:
    if (hit := self.replies.get(request_id)) and hit[1] > (now := time.time()):
      reply, exp = self.replies.pop(request_id)
      self.replies[request_id] = (reply, min(exp, now + read_ttl))
      return reply
    if request_id not in self.by_id:
      raise PromiseGoneError(f"sampling promise gone: {request_id}")
    return None

  def queued(self, run_id: str, limit: int | None = None):
    gen = (e.cmd for e in self.queues.get(run_id, {}).values() if e.owner is None)
    return gen if limit is None else itertools.islice(gen, limit)

  def claim(self, cmd: Cmd, owner: tuple[str, str, str], now: float) -> None:
    self.by_id[cmd.request_id].owner, self.by_id[cmd.request_id].since = owner, now

  def complete(self, request_id: str, owner: tuple[str, str, str] | None, reply: Reply, ttl: float) -> Cmd | None:
    if (entry := self.by_id.get(request_id)) is None or entry.owner != owner:
      return None
    del self.by_id[request_id]
    self.queues.get(entry.cmd.run, {}).pop(entry.cmd.entry_id, None)
    self.queued_bytes, self.reply_bytes = self.queued_bytes - len(entry.cmd.body), self.reply_bytes + len(reply.body)
    self.replies[request_id] = (reply, time.time() + ttl)
    while self.reply_bytes > SAMPLE_REPLY_BYTES and len(self.replies) > 1:
      if (victim := next((k for k, (r, _) in self.replies.items() if r.body), None)) is None:
        break
      self.reply_bytes -= len(self.replies.pop(victim)[0].body)
    return entry.cmd

  def reclaim(self, pred) -> None:
    for e in self.by_id.values():
      if e.owner and pred(e.owner):
        e.owner, e.since = None, 0.0

  def status(self, run_id: str, now: float) -> dict:
    if (q := self.queues.get(run_id)) is None:
      return {"live": False}
    fly = []
    for e in q.values():
      if e.owner:
        fly.append(
          {"request_id": e.cmd.request_id, "trainer": e.owner[0], "deliveries": 1, "seconds": max(0.0, now - e.since)}
        )
    return {"live": True, "queued": len(q) - len(fly), "in_flight": fly}

  def gc(self, now: float) -> None:
    for k in [k for k, (_, exp) in self.replies.items() if exp <= now]:
      self.reply_bytes -= len(self.replies.pop(k)[0].body)
    self.seen = {k: v for k, v in self.seen.items() if v[1] > now}
