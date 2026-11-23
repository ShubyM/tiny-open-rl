"""One-time conversion of the SQLite Bus rows to the Ledger format."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time

import msgspec

from tinyrl.types import Checkpoint


async def checkpoint_meta(checkpoint: Checkpoint | None) -> dict:
  if checkpoint is None:
    return {}
  if not checkpoint.meta.get("legacy_header"):
    return checkpoint.meta
  # The old checkpoint rows kept the model description only in the immutable safetensors header.
  from tinyrl.storage.store import Store

  try:
    meta = await asyncio.to_thread(lambda: Store().read_meta(checkpoint.version))
  except (OSError, IndexError, ValueError):
    return {}
  return Checkpoint.model_info(meta)


def run_row(row: dict) -> dict:
  run = row["record"]
  snapshot = row["snap"][1] if row["snap"] else None
  if not row["live"]:
    status = "deleted"
  elif snapshot:
    status = "saved"
  elif row["pos"]:
    status = "live" if row["owner"] else "lost"
  else:
    status = "fresh"
  return {
    "created": run.pop("created"),
    "run": run,
    "owner": row["owner"],
    "status": status,
    "snapshot": snapshot,
    "clients": row["clients"],
    "counters": row["counters"],
    "deleted_at": time.time() if status == "deleted" else 0.0,
  }


def command_row(conn: sqlite3.Connection, key: str, row: dict) -> dict:
  cmd = dict(run=row["run"], entry_id=row["eid"], op=row["op"], cost=row["cost"], body=b"")
  item = dict(cmd=cmd, seq=row["seq"])
  if cmd["op"] == "load_weights":
    body = conn.execute("SELECT v FROM kv WHERE ns='blobs' AND k=?", (f"c:{key}",)).fetchone()
    try:
      uri = json.loads(body[0])["uri"]
      if isinstance(uri, str):
        item["uri"] = uri
    except (ValueError, TypeError, KeyError):
      pass  # Invalid input will fail the request when it reaches the queue head.
  if assignment := row.get("assignment"):
    turn = msgspec.msgpack.decode(assignment)
    item["owner"] = [turn["worker"], row["node"], row["boot"]]
    item.update(snapshot=turn["args"].get("saved"), version=turn["args"].get("version"), block=turn["gpus"])
  return item


def schema_version(conn: sqlite3.Connection) -> int:
  version = conn.execute("PRAGMA user_version").fetchone()[0]
  if version not in (0, 1):
    raise ValueError(f"unsupported ledger database version {version}")
  return version


def upgrade(conn: sqlite3.Connection) -> None:
  with conn:
    if schema_version(conn) == 1:
      return
    for ns in ("runs", "cmds", "checkpoints"):
      for key, raw in conn.execute("SELECT k,v FROM kv WHERE ns=?", (ns,)).fetchall():
        row = msgspec.msgpack.decode(raw)
        if ns == "runs" and "record" in row:
          row = run_row(row)
        elif ns == "cmds" and "eid" in row:
          row = command_row(conn, key, row)
        elif ns == "checkpoints" and "tinker_path" in row:
          row = Checkpoint(
            key,
            row["version"],
            row["size_bytes"],
            row["time"],
            row["expires_at"],
            row["owned"],
            {"legacy_header": True},
          )
        else:
          continue
        conn.execute("UPDATE kv SET v=? WHERE ns=? AND k=?", (msgspec.msgpack.encode(row), ns, key))
    conn.execute("PRAGMA user_version=1")
