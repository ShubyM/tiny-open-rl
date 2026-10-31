"""The checkpoint registry, on Redis: every checkpoint the store holds, who owns it, when it expires, and the version another node waits for. The node daemons' sweeps delete what it lets go."""

from __future__ import annotations

import json
import time

from tinyrl.bus import Bus, key


class Registry:
  def __init__(self, bus: Bus):
    self.bus = bus

  @property
  def r(self):
    return self.bus.r  # read when used, so a worker made without a bus, as tests do, still starts

  async def register_checkpoint(
    self, uri: str, size: int, ttl: int | None = None, owned: bool = False, version: str | None = None
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
    info["version"] = version  # which version to wait for, when another node is still uploading it
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
    if (info := await self.get_checkpoint(uri)) is None:
      return False
    run, cid = uri.removeprefix("tinker://").split("/", 1)
    info["expires_at"] = time.time() + ttl if ttl else None
    p = self.r.pipeline(transaction=True)
    p.hset(key("ckpt", run), cid, json.dumps(info))
    p.zadd(key("expiry"), {uri: info["expires_at"]}) if ttl else p.zrem(key("expiry"), uri)
    await p.execute()
    return True

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

  async def registered_versions(self) -> set[str]:
    """The version of every checkpoint in the registry, parked ones too."""
    found = set()
    async for k in self.r.scan_iter(match=key("ckpt", "*")):
      found |= {json.loads(v).get("version") for v in await self.r.hvals(k)}
    return found - {None}

  async def first_registered_at(self) -> float | None:
    """When the first checkpoint was registered. A sweep leaves files older than that alone."""
    since = await self.r.get(key("ckpt_since"))
    return float(since) if since else None
