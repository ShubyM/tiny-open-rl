"""Durable SQLite-backed FIFO training command queues, replies, and seq_id deduplication."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from tinyrl.control.db import DB, Table
from tinyrl.control.samples import flight_status
from tinyrl.types import Cmd, Owner, Reply, Run, Work


@dataclass
class CmdRow:
  cmd: Cmd
  seq: int
  uri: str | None = None
  owner: Owner | None = None
  snapshot: str | None = None
  version: str | None = None


class TrainQueue:
  def __init__(self, db: DB, live_runs, now: float):
    self.db = db
    self.cmds = Table(db, "cmds", CmdRow)
    self.replies, self.seen = Table(db, "replies"), Table(db, "seen")
    self.queues: dict[str, dict[str, None]] = {rid: {} for rid in live_runs}
    self.since: dict[str, float] = {}
    for req_id, row in sorted(self.cmds.items(), key=lambda kv: kv[1].seq):
      if row.cmd.run in self.queues:
        self.queues[row.cmd.run][req_id] = None
        if row.owner:
          self.since[req_id] = now

  def open(self, run_id: str) -> None:
    self.queues.setdefault(run_id, {})

  def backlog(self, run_id: str) -> int:
    return len(self.queues.get(run_id, ()))

  def max_seq(self) -> int:
    return max((row.seq for row in self.cmds.values()), default=0)

  def owners(self) -> list[Owner]:
    return [row.owner for row in self.cmds.values() if row.owner]

  def versions(self) -> set[str]:
    return {x for row in self.cmds.values() for x in (row.version, row.snapshot) if x}

  def seen_id(self, run_id: str, seq_id: int) -> str | None:
    return hit[0] if (hit := self.seen.get(f"{run_id}:{seq_id}")) and hit[1] > time.time() else None

  def submit(self, cmd: Cmd, seq: int, seq_id: int | None, session_ttl: float, uri: str | None = None) -> None:
    with self.db:
      self.db.put_blob(f"c:{cmd.request_id}", cmd.body)
      self.cmds[cmd.request_id] = CmdRow(replace(cmd, body=b""), seq, uri)
      if seq_id is not None:
        self.seen[f"{cmd.run}:{seq_id}"] = [cmd.request_id, time.time() + session_ttl]
    self.queues.setdefault(cmd.run, {})[cmd.request_id] = None

  def read_reply(self, req_id: str, read_ttl: float) -> Reply | None:
    meta, now = self.replies.get(req_id), time.time()
    if meta and meta[2] > now and (body := self.db.get_blob(f"r:{req_id}")) is not None:
      with self.db:
        self.replies[req_id] = [meta[0], meta[1], min(meta[2], now + read_ttl)]
      return Reply(bool(meta[0]), meta[1], body)
    return None

  def cmd(self, req_id: str, *, full: bool = False) -> Cmd:
    base = self.cmds[req_id].cmd
    return replace(base, body=self.db.get_blob(f"c:{req_id}") or b"") if full else base

  def work(self, req_id: str, run: Run) -> Work:
    row = self.cmds[req_id]
    return Work(self.cmd(req_id, full=True), run, row.snapshot, row.version)

  def head(self, run_id: str) -> tuple[str, CmdRow] | None:
    if (k := next(iter(self.queues.get(run_id, ())), None)) and not (row := self.cmds[k]).owner:
      return k, row
    return None

  def held_by(self, owner: Owner) -> tuple[str, CmdRow] | None:
    return next(((k, row) for k, row in self.cmds.items() if row.owner == owner), None)

  def held_on(self, node_boot: tuple[str, str]) -> list[tuple[str, CmdRow]]:
    return [(k, row) for k, row in self.cmds.items() if row.owner and row.owner[1:] == node_boot]

  def in_flight(self, run_id: str) -> list[tuple[str, Owner]]:
    return [(k, o) for k in self.queues.get(run_id, ()) if (o := self.cmds[k].owner)]

  def claim(self, req_id: str, owner: Owner, now: float, snapshot: str | None, ver: str | None) -> None:
    row = self.cmds[req_id]
    row.owner, row.snapshot, row.version, self.since[req_id] = owner, snapshot, ver, now
    self.cmds.save(req_id)

  def pop_claimed(self, req_id: str, owner: Owner | None) -> Cmd | None:
    if (row := self.cmds.get(req_id)) is None or row.owner != owner or req_id not in self.queues.get(row.cmd.run, ()):
      return None
    del self.queues[row.cmd.run][req_id]
    return row.cmd

  def remove(self, req_id: str) -> None:
    self.since.pop(req_id, None)
    self.cmds.pop(req_id, None)
    self.db.del_blob(f"c:{req_id}")

  def write_reply(self, req_id: str, reply: Reply, ttl: float) -> None:
    self.remove(req_id)
    self.db.put_blob(f"r:{req_id}", reply.body)
    self.replies[req_id] = [reply.ok, reply.content_type, time.time() + ttl]

  def drop(self, run_id: str) -> list[str]:
    dropped = list(self.queues.pop(run_id, ()))
    for k in dropped:
      self.remove(k)
    return dropped

  def status(self, run_id: str, now: float) -> dict:
    q = self.queues.get(run_id, {})
    return flight_status(len(q), [(k, o[0], self.since.get(k, now)) for k in q if (o := self.cmds[k].owner)], now)

  def gc(self, now: float) -> None:
    for k in [k for k, m in self.replies.items() if m[2] <= now]:
      del self.replies[k]
      self.db.del_blob(f"r:{k}")
    for k in [k for k, m in self.seen.items() if m[1] <= now]:
      del self.seen[k]
