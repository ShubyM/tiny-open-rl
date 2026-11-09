"""The checkpoint registry, on Redis: every checkpoint the store holds, who owns it, when it expires, and the version another node waits for. The node daemons' sweeps delete what it lets go."""

from __future__ import annotations

import asyncio
import json
import time

from redis.exceptions import WatchError

from tinyrl.bus import Bus, key
from tinyrl.storage.store import Store


class Registry:
  def __init__(self, bus: Bus):
    self.r = bus.r

  async def register_checkpoint(
    self,
    uri: str,
    size: int,
    ttl: int | None = None,
    owned: bool = False,
    version: str | None = None,
    node: str | None = None,
  ) -> None:
    """Record a checkpoint the store now holds. One the run owns, like a sampling session's weights or the state a stopping trainer parked, goes when the run is deleted. The rest stay until deleted, or their TTL runs out."""
    run, cid = uri.removeprefix("tinker://").split("/", 1)
    now, kind = time.time(), cid.split("/", 1)[0]
    info = {
      "checkpoint_id": cid,
      "checkpoint_type": {"weights": "training", "sampler_weights": "sampler"}.get(kind, kind),
      "time": now,
    }
    info |= {"tinker_path": uri, "size_bytes": size, "expires_at": now + ttl if ttl else None, "owned": owned}
    info["version"], info["node"] = version, node  # the version to read, and the node uploading it
    p = self.r.pipeline(transaction=True)
    p.setnx(key("ckpt_since"), now)
    p.hset(key("ckpt", run), cid, json.dumps(info))
    if kind != "parked":
      p.zadd(key("ckpts"), {uri: now})
    p.zadd(key("expiry"), {uri: now + ttl}) if ttl else p.zrem(key("expiry"), uri)
    await p.execute()

  async def get_checkpoint(self, uri: str) -> dict | None:
    run, cid = uri.removeprefix("tinker://").split("/", 1)
    info = await self.r.hget(key("ckpt", run), cid)
    return json.loads(info) if info else None

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
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(key("ckpt", run))
          if (data := await p.hget(key("ckpt", run), cid)) is None:
            return False
          info = json.loads(data) | {"expires_at": time.time() + ttl if ttl else None}
          p.multi()
          p.hset(key("ckpt", run), cid, json.dumps(info))
          p.zadd(key("expiry"), {uri: info["expires_at"]}) if ttl else p.zrem(key("expiry"), uri)
          await p.execute()
          return True
      except WatchError:
        continue

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

  async def registered(self) -> tuple[set[str], set[str]]:
    """Every checkpoint in the registry, parked ones too, and their versions."""
    uris, versions = set(), set()
    async for k in self.r.scan_iter(match=key("ckpt", "*")):
      run = k.decode().removeprefix(key("ckpt", ""))
      for cid, info in (await self.r.hgetall(k)).items():
        uris.add(f"tinker://{run}/{cid.decode()}")
        versions.add(json.loads(info).get("version"))
    return uris, versions - {None}

  async def sweep(self, store: Store, grace: float) -> None:
    """Let the node's store delete what the registry doesn't have. Checkpoints older than the registry itself are left alone, so turning it on never deletes what came before."""
    since = await self.r.get(key("ckpt_since"))  # when the first checkpoint was registered
    await asyncio.to_thread(store.sweep, *await self.registered(), float(since) if since else None, grace)
