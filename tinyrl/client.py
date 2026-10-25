"""Worker HTTP transport. Only the coordinator accesses Redis."""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace

import httpx

from tinyrl.bus import MAX_REPLY_BYTES, Cmd, Reply, error, pack, unpack_cmd
from tinyrl.payloads import INLINE_BYTES, PayloadStore


async def _retry(call, retryable):
  delay = 0.1
  while True:
    try: return await call()
    except retryable:
      await asyncio.sleep(delay)
      delay = min(delay * 2, 2)


class WorkerClient:
  def __init__(self, url: str | None = None, token: str | None = None,
               http: httpx.AsyncClient | None = None, payloads: PayloadStore | None = None):
    token = os.getenv("COORDINATOR_TOKEN", "") if token is None else token
    self.http = http or httpx.AsyncClient(
      base_url=url or os.getenv("COORDINATOR_URL", "http://127.0.0.1:8000"),
      headers={"Authorization": f"Bearer {token}"}, timeout=60,
    )
    self.payloads = payloads if payloads is not None else PayloadStore()

  async def _post(self, method: str, body: dict):
    async def send():
      resp = await self.http.post(f"/internal/{method}", json=body)
      if resp.status_code >= 500: raise httpx.TransportError(f"coordinator {resp.status_code}")
      resp.raise_for_status()
      return resp.json()
    return await _retry(send, httpx.TransportError)

  async def _storage(self, fn, *args):
    async def run():
      try: return await asyncio.to_thread(fn, *args)
      except FileNotFoundError: raise
      except OSError as e: raise TimeoutError from e
    return await _retry(run, TimeoutError)

  async def register(self, worker: str, pool: str) -> dict:
    return await self._post("register", {"worker": worker, "pool": pool})

  async def heartbeat(self, worker: str) -> bool:
    return (await self._post("heartbeat", {"worker": worker}))["live"]

  async def take(self, pool: str, worker: str, count: int = 1, block_ms: int = 1000) -> list[Cmd]:
    # Fetch one command at a time so queued blob-backed batches do not pile up in worker RAM.
    fields = await self._post("take", {"pool": pool, "worker": worker, "count": 1, "block_ms": block_ms})
    out = []
    for raw in fields["commands"]:
      cmd = unpack_cmd(raw)
      if cmd.body_ref:
        try: cmd = replace(cmd, body=await self._storage(self.payloads.get, cmd.body_ref, cmd.body_size))
        except (FileNotFoundError, ValueError) as e:
          await self.finish(cmd, error(f"input payload unavailable: {e}"))
          continue
      out.append(cmd)
    return out

  async def finish(self, cmd: Cmd, result: Reply) -> str:
    size = result.body_size if result.body_ref else len(result.body)
    if size > MAX_REPLY_BYTES:
      result = replace(error("result exceeds MAX_REPLY_BYTES; recover from checkpoint"), recover=True)
    elif not result.body_ref and size > INLINE_BYTES:
      ref = await self._storage(self.payloads.put, result.body)
      result = replace(result, body=b"", body_ref=ref, body_size=size)
    return (await self._post("finish", {"command": pack(cmd), "reply": pack(result)}))["status"]

  async def close(self) -> None: await self.http.aclose()
