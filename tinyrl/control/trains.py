"""Durable SQLite-backed FIFO training command queues, replies, and seq_id deduplication."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import msgspec

from tinyrl.control.db import DB, Table
from tinyrl.control.samples import flight_status
from tinyrl.types import Cmd, Op, Owner, Reply, Run, Work


@dataclass
class CmdRow:
  cmd: Cmd
  seq: int
  uri: str | None = None
  owner: Owner | None = None
  snapshot: str | None = None
  version: str | None = None
  block: tuple[int, ...] = ()
  claimed_at: float = 0.0


class TrainQueue:
  def __init__(self, db: DB):
    self.db = db
    self.cmds = Table(db, "cmds", CmdRow)
    self.replies, self.seen = Table(db, "replies"), Table(db, "seen")

  def rows(self, run_id: str):
    return ((key, row) for key, row in self.cmds.items() if row.cmd.run == run_id)

  def max_seq(self) -> int:
    return max((row.seq for row in self.cmds.values()), default=0)

  def owners(self) -> list[Owner]:
    return [row.owner for row in self.cmds.values() if row.owner]

  def versions(self) -> set[str]:
    return {x for row in self.cmds.values() for x in (row.version, row.snapshot) if x}

  def seen_id(self, run_id: str, seq_id: int) -> str | None:
    return hit[0] if (hit := self.seen.get(f"{run_id}:{seq_id}")) and hit[1] > time.time() else None

  def submit(
    self, cmd: Cmd, seq: int, seq_id: int | None, session_ttl: float, uri=None, stability_refs: set[str] | None = None
  ) -> None:
    with self.db:
      self.db.put_blob(f"c:{cmd.request_id}", cmd.body)
      if stability_refs and (blobs := self.db.read_stability(stability_refs, time.time())):
        self.db.put_blob(f"s:{cmd.request_id}", msgspec.msgpack.encode(blobs))
      self.cmds[cmd.request_id] = CmdRow(replace(cmd, body=b""), seq, uri)
      if seq_id is not None:
        self.seen[f"{cmd.run}:{seq_id}"] = [cmd.request_id, time.time() + session_ttl]

  def read_reply(self, req_id: str, read_ttl: float) -> Reply | None:
    meta, now = self.replies.get(req_id), time.time()
    if meta and meta[2] > now and (body := self.db.get_blob(f"r:{req_id}")) is not None:
      with self.db:
        self.replies[req_id] = [meta[0], meta[1], min(meta[2], now + read_ttl)]
      return Reply(bool(meta[0]), meta[1], body)
    return None

  def cmd(self, req_id: str) -> Cmd:
    base = self.cmds[req_id].cmd
    raw = self.db.get_blob(f"s:{req_id}") if base.op == Op.FORWARD_BACKWARD else None
    return replace(
      base, body=self.db.get_blob(f"c:{req_id}") or b"", stability=msgspec.msgpack.decode(raw) if raw else None
    )

  def work(self, req_id: str, run: Run) -> Work:
    row = self.cmds[req_id]
    return Work(self.cmd(req_id), run, row.snapshot, row.version)

  def heads(self, eligible: set[str]) -> dict[str, CmdRow]:
    first = {}
    for row in self.cmds.values():
      run = row.cmd.run
      if run in eligible and (run not in first or row.seq < first[run].seq):
        first[run] = row
    # A claimed FIFO head blocks the whole run until its receipt commits.
    return {run: row for run, row in first.items() if row.owner is None}

  def claimed_blocks(self):
    return ((r.owner, r.cmd.run, r.block) for r in self.cmds.values() if r.owner)

  def held_by(self, owner: Owner) -> tuple[str, CmdRow] | None:
    return next(((k, row) for k, row in self.cmds.items() if row.owner == owner), None)

  def held_on(self, node_boot: tuple[str, str]) -> list[tuple[str, CmdRow]]:
    return [(k, row) for k, row in self.cmds.items() if row.owner and row.owner[1:] == node_boot]

  def in_flight(self, run_id: str) -> list[tuple[str, Owner]]:
    return [(k, row.owner) for k, row in self.rows(run_id) if row.owner]

  def claim(self, req_id: str, owner: Owner, snapshot: str | None, ver: str | None, block: tuple[int, ...]) -> None:
    row = self.cmds[req_id]
    row.owner, row.snapshot, row.version, row.block, row.claimed_at = owner, snapshot, ver, block, time.time()
    self.cmds.save(req_id)

  def remove(self, req_id: str) -> None:
    self.cmds.pop(req_id, None)
    self.db.del_blob(f"c:{req_id}")
    self.db.del_blob(f"s:{req_id}")

  def write_reply(self, req_id: str, reply: Reply, ttl: float) -> None:
    self.remove(req_id)
    self.db.put_blob(f"r:{req_id}", reply.body)
    self.replies[req_id] = [reply.ok, reply.content_type, time.time() + ttl]

  def drop(self, run_id: str) -> list[str]:
    dropped = [key for key, _ in self.rows(run_id)]
    for k in dropped:
      self.remove(k)
    return dropped

  def status(self, run_id: str) -> dict:
    rows, now = list(self.rows(run_id)), time.time()
    active = [(k, r.owner[0], r.claimed_at or now) for k, r in rows if r.owner]
    return flight_status(len(rows), active, now)

  def gc(self, now: float) -> None:
    for k in [k for k, m in self.replies.items() if m[2] <= now]:
      del self.replies[k]
      self.db.del_blob(f"r:{k}")
    for k in [k for k, m in self.seen.items() if m[1] <= now]:
      del self.seen[k]
