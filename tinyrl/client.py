"""Worker transport. Only the coordinator opens the journal.

Retries resend the same execution identity and outcome. The trainer retains an
unacknowledged result; a transport retry never repeats a numerical operation.
"""

from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import asdict

import httpx

from tinyrl.bus import Cmd, Op, Reply


def pack(value: Cmd | Reply) -> dict:
  fields = asdict(value)
  fields["body"] = base64.b64encode(value.body).decode()
  return fields


def command(fields: dict) -> Cmd:
  return Cmd(**{**fields, "op": Op(fields["op"]), "body": base64.b64decode(fields["body"], validate=True)})


def reply(fields: dict) -> Reply:
  return Reply(**{**fields, "body": base64.b64decode(fields["body"], validate=True)})


class WorkerClient:
  def __init__(self, url: str | None = None, token: str | None = None, http: httpx.AsyncClient | None = None):
    token = os.getenv("COORDINATOR_TOKEN", "") if token is None else token
    self.http = http or httpx.AsyncClient(
      base_url=url or os.getenv("COORDINATOR_URL", "http://127.0.0.1:8000"),
      headers={"Authorization": f"Bearer {token}"},
      timeout=60,
    )

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
    fields = await self._post("take", {"pool": pool, "worker": worker, "count": count, "block_ms": block_ms})
    return [command(c) for c in fields["commands"]]

  async def finish(self, cmd: Cmd, result: Reply) -> str:
    return (await self._post("finish", {"command": pack(cmd), "reply": pack(result)}))["status"]

  async def close(self) -> None:
    await self.http.aclose()
