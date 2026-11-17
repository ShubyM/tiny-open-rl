"""The checkpoint registry, on Redis: every checkpoint the store holds, who owns it, when it expires, and the version a node reads it by. The node daemons' sweeps delete what it lets go."""

from __future__ import annotations

import asyncio
import json
import time

from tinyrl.bus import Bus, key, queue_checkpoint
from tinyrl.storage.store import Store


class Registry:
  def __init__(self, bus: Bus):
    self.bus, self.r = bus, bus.r

  async def register_checkpoint(
    self, uri: str, size: int, ttl: int | None = None, owned: bool = False, version: str | None = None
  ) -> None:
    p = self.r.pipeline(transaction=True)
    queue_checkpoint(p, uri, size, ttl, owned, version)
    await p.execute()

  async def get_checkpoint(self, uri: str) -> dict | None:
    run, cid = uri.removeprefix("tinker://").split("/", 1)
    info = await self.r.hget(key("ckpt", run), cid)
    return json.loads(info) if info else None

  async def version(self, uri: str) -> str | None:
    """The version to read a checkpoint by."""
    return (await self.get_checkpoint(uri) or {}).get("version")

  async def checkpoints_of_run(self, run: str) -> list[dict]:
    found = [json.loads(v) for v in (await self.r.hgetall(key("ckpt", run))).values()]
    return sorted((c for c in found if c["checkpoint_type"] != "parked"), key=lambda c: c["time"])

  async def page_of_checkpoints(self, limit: int, offset: int) -> tuple[list[dict], int]:
    """A page of every run's checkpoints, newest first, and how many there are."""
    uris = [u.decode() for u in await self.r.zrevrange(key("ckpts"), offset, offset + limit - 1)]
    found = [await self.get_checkpoint(u) for u in uris]
    return [c for c in found if c], await self.r.zcard(key("ckpts"))

  async def set_checkpoint_ttl(self, uri: str, ttl: int | None) -> bool:
    """Make a checkpoint expire ttl seconds from now, or never with None."""
    run, cid = uri.removeprefix("tinker://").split("/", 1)
    k = key("ckpt", run)

    async def update(p):
      if not (data := await p.hget(k, cid)):
        return False
      expires = time.time() + ttl if ttl else None
      p.multi()
      p.hset(k, cid, json.dumps(json.loads(data) | {"expires_at": expires}))
      p.zadd(key("expiry"), {uri: expires}) if ttl else p.zrem(key("expiry"), uri)
      return bool(await p.execute())

    return await self.bus.transact(update, k)

  async def unregister_checkpoints(self, *uris: str) -> int:
    """Take checkpoints out of the registry. The node daemons' sweeps then delete their files."""
    p = self.r.pipeline(transaction=True)
    for uri in uris:
      run, cid = uri.removeprefix("tinker://").split("/", 1)
      p.hdel(key("ckpt", run), cid)
      p.zrem(key("ckpts"), uri)
      p.zrem(key("expiry"), uri)
    return sum((await p.execute())[::3]) if uris else 0

  async def checkpoints_owned_by_run(self, run: str) -> list[str]:
    found = [json.loads(v) for v in (await self.r.hgetall(key("ckpt", run))).values()]
    return [c["tinker_path"] for c in found if c["owned"]]

  async def expired_checkpoints(self) -> list[str]:
    """The checkpoints whose TTL has run out."""
    return [u.decode() for u in await self.r.zrangebyscore(key("expiry"), 0, time.time())]

  async def registered(self) -> set[str]:
    """Every checkpoint in the registry, parked ones too."""
    uris = set()
    async for k in self.r.scan_iter(match=key("ckpt", "*")):
      for info in map(json.loads, await self.r.hvals(k)):
        uris.add(info["tinker_path"])
    return uris

  async def sweep(self, store: Store, grace: float) -> None:
    """Let the node's store delete what the registry doesn't have. Checkpoints older than the registry itself are left alone, so turning it on never deletes what came before."""
    since = await self.r.get(key("ckpt_since"))  # when the first checkpoint was registered
    await asyncio.to_thread(store.sweep, await self.registered(), float(since) if since else None, grace)
