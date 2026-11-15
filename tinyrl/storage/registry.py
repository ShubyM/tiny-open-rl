"""Checkpoint names and actual version owners. Only an unreferenced version can be deleted."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time

from tinyrl.bus import Bus, key
from tinyrl.storage.store import Store, manifest

SWEEP_S, SWEEP_GRACE_S = 60, 600


class Registry:
  def __init__(self, bus: Bus):
    self.bus, self.r = bus, bus.r

  async def get_checkpoint(self, uri: str) -> dict | None:
    run, cid = uri.removeprefix("tinker://").split("/", 1)
    return json.loads(info) if (info := await self.r.hget(key("ckpt", run), cid)) else None

  async def version(self, uri: str) -> str | None:
    return (await self.get_checkpoint(uri) or {}).get("version")

  async def checkpoints_of_run(self, run: str) -> list[dict]:
    found = [json.loads(v) for v in (await self.r.hgetall(key("ckpt", run))).values()]
    return sorted((c for c in found if c["checkpoint_type"] != "parked"), key=lambda c: c["time"])

  async def page_of_checkpoints(self, limit: int, offset: int) -> tuple[list[dict], int]:
    uris = [u.decode() for u in await self.r.zrevrange(key("ckpts"), offset, offset + limit - 1)]
    found = [await self.get_checkpoint(u) for u in uris]
    return [c for c in found if c], await self.r.zcard(key("ckpts"))

  async def set_checkpoint_ttl(self, uri: str, ttl: int | None) -> bool:
    run, cid = uri.removeprefix("tinker://").split("/", 1)
    ckey = key("ckpt", run)

    async def step(p):
      if not (raw := await p.hget(ckey, cid)):
        return False, False
      info = json.loads(raw) | {"expires_at": time.time() + ttl if ttl else None}
      p.multi()
      p.hset(ckey, cid, json.dumps(info))
      p.zadd(key("expiry"), {uri: info["expires_at"]}) if ttl else p.zrem(key("expiry"), uri)
      return True, True

    return await self.bus.tx((ckey,), step)

  async def unregister_checkpoints(self, *uris: str) -> int:
    removed = 0
    for uri in uris:
      run, cid = uri.removeprefix("tinker://").split("/", 1)
      ckey = key("ckpt", run)

      async def step(p, uri=uri, cid=cid, ckey=ckey):
        p.multi()
        p.hdel(ckey, cid)
        p.zrem(key("ckpts"), uri)
        p.zrem(key("expiry"), uri)
        p.hdel(key("refs"), f"ckpt:{uri}")
        return True, lambda replies: replies[0]

      removed += await self.bus.tx((ckey,), step)
    return removed

  async def expired_checkpoints(self) -> list[str]:
    return [u.decode() for u in await self.r.zrangebyscore(key("expiry"), 0, time.time())]

  async def maintain(self, store: Store, interval: float) -> None:
    """Finish captured SAVEs independently of trainer lifetime, and collect unreferenced versions."""
    swept = 0.0
    while True:
      for request, cp in (await self.bus.pending_saves()).items():
        with contextlib.suppress(OSError):
          if manifest(store.folder(cp["version"])):
            await asyncio.to_thread(store.upload_version, cp["uri"], cp["version"])
          if await asyncio.to_thread(store.is_durable, cp["uri"], cp["version"]):
            await self.bus.finish_save(request, cp)
      if time.time() - swept > SWEEP_S:
        await self.sweep(store, SWEEP_GRACE_S)
        swept = time.time()
      await asyncio.sleep(interval)

  async def sweep(self, store: Store, grace: float) -> None:
    versions = [k async for k in self.r.scan_iter(match=key("ver", "*"))]
    if versions:

      async def claim(p):
        held = set(await p.hvals(key("refs")))
        records = [(k, json.loads(raw)) for k, raw in zip(versions, await p.mget(*versions)) if raw]
        dead = [
          (k, info) for k, info in records if k.rsplit(b":", 1)[-1] not in held and time.time() - info["time"] >= grace
        ]
        p.multi()
        for k, info in dead:
          p.set(k, json.dumps(info | {"state": "deleting"}))
        return True, dead

      for vkey, info in await self.bus.tx((key("refs"), *versions), claim):
        if await asyncio.to_thread(store.delete_version, info["uri"], info["version"]):
          await self.r.delete(vkey)
    await asyncio.to_thread(store.trim_cache)
