"""Worker control transport. Only the coordinator accesses Redis.

Retries resend the same execution identity and outcome. The trainer retains an
unacknowledged result; a transport retry never repeats a numerical operation.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import asdict, replace

import httpx

from tinyrl.bus import MAX_REPLY_BYTES, Cmd, Op, Reply
from tinyrl.payloads import INLINE_BYTES, PayloadStore


def pack(value: Cmd | Reply) -> dict:
  fields = asdict(value)
  fields["body"] = "" if value.body_ref else base64.b64encode(value.body).decode()
  return fields


def command(fields: dict) -> Cmd:
  return Cmd(**{**fields, "op": Op(fields["op"]), "body": base64.b64decode(fields["body"], validate=True)})


def reply(fields: dict) -> Reply:
  return Reply(**{**fields, "body": base64.b64decode(fields["body"], validate=True)})


class WorkerClient:
  def __init__(
    self,
    url: str | None = None,
    token: str | None = None,
    http: httpx.AsyncClient | None = None,
    payloads: PayloadStore | None = None,
  ):
    token = os.getenv("COORDINATOR_TOKEN", "") if token is None else token
    self.http = http or httpx.AsyncClient(
      base_url=url or os.getenv("COORDINATOR_URL", "http://127.0.0.1:8000"),
      headers={"Authorization": f"Bearer {token}"},
      timeout=60,
    )
    self.payloads = payloads if payloads is not None else PayloadStore()

  async def _post(self, method: str, body: dict):
    delay = 0.1
    while True:
      try:
        response = await self.http.post(f"/internal/{method}", json=body)
        if response.status_code < 500:
          response.raise_for_status()
          return response.json()
      except httpx.TransportError:
        pass
      await asyncio.sleep(delay)
      delay = min(delay * 2, 2)

  async def register(self, worker: str, pool: str) -> dict:
    return await self._post("register", {"worker": worker, "pool": pool})

  async def heartbeat(self, worker: str) -> bool:
    return (await self._post("heartbeat", {"worker": worker}))["live"]

  async def take(self, pool: str, worker: str, count: int = 16, block_ms: int = 1000) -> list[Cmd]:
    # Numerical execution is serial. Prefetching a batch of blob-backed commands
    # would hold count * MAX_COMMAND_BYTES before any one of them can execute.
    fields = await self._post("take", {"pool": pool, "worker": worker, "count": min(count, 1), "block_ms": block_ms})
    commands = []
    for item in fields["commands"]:
      cmd = command(item)
      if cmd.body_ref:
        try:
          body = await self._storage(self.payloads.get, cmd.body_ref, cmd.body_size)
          cmd = replace(cmd, body=body)
        except (FileNotFoundError, ValueError) as error:
          outcome = Reply(
            False,
            "application/json",
            json.dumps(
              {
                "error": f"input payload unavailable: {error}",
                "category": "server",
              }
            ).encode(),
          )
          await self.finish(cmd, outcome)  # no numerical operation has run
          continue
      commands.append(cmd)
    return commands

  async def _storage(self, operation, *args):
    delay = 0.1
    while True:
      try:
        return await asyncio.to_thread(operation, *args)
      except FileNotFoundError:
        raise
      except OSError:
        await asyncio.sleep(delay)
        delay = min(delay * 2, 2)

  async def finish(self, cmd: Cmd, result: Reply) -> str:
    if (result.body_size if result.body_ref else len(result.body)) > MAX_REPLY_BYTES:
      result = Reply(
        False,
        "application/json",
        b'{"error":"result exceeds MAX_REPLY_BYTES; recover from checkpoint","category":"server"}',
        recover=True,
      )
    if not result.body_ref and len(result.body) > INLINE_BYTES:
      ref = await self._storage(self.payloads.put, result.body)
      result = replace(result, body=b"", body_ref=ref, body_size=len(result.body))
    return (await self._post("finish", {"command": pack(cmd), "reply": pack(result)}))["status"]

  async def close(self) -> None:
    await self.http.aclose()
