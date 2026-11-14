"""The authoritative checkpoint & version registry on Redis.

Logical checkpoints (tinker://<run>/<kind>/<name>) map to immutable version records (tinyrl:ver:<ver>).
Version identity is separate from copy locations and durability status. Active readers, assignments,
logical checkpoints, and recovery points hold references (tinyrl:ver_refs:<ver>) that protect retention
under WATCH/MULTI; deletion is claimed only while a version is unreferenced.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable

from redis.exceptions import WatchError

from tinyrl.bus import Bus, key
from tinyrl.storage.store import Store, parse_version


def _live_ref(val: bytes | str | float, now: float) -> bool:
  t = float(val)
  return t == 0.0 or t > now


class Registry:
  def __init__(self, bus: Bus):
    self.r = bus.r

  async def _edit_version(self, version: str, fn: Callable[[dict], bool | None]) -> bool:
    """Apply `fn(rec)` to an existing version record under WATCH/MULTI. Return True if present and accepted."""
    ver_id, _ = parse_version(version)
    if not ver_id:
      return False
    vk = key("ver", ver_id)
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(vk)
          if not (raw := await p.get(vk)):
            await p.unwatch()
            return False
          rec = json.loads(raw)
          changed = fn(rec)
          if changed is False:
            await p.unwatch()
            return False
          if changed is None:
            await p.unwatch()
            return True
          p.multi()
          p.set(vk, json.dumps(rec))
          await p.execute()
          return True
      except WatchError:
        continue

  async def register_version(
    self,
    version: str,
    uri: str,
    size: int,
    kind: str = "training",
    durable: bool = False,
    node: str = "",
    peer: str = "",
    meta: dict | None = None,
    refs: tuple[str, ...] = (),
  ) -> dict:
    """Create or update an authoritative version record and attach initial permanent references."""
    ver_id, inline_peer = parse_version(version)
    if not ver_id:
      raise ValueError("version id required")
    peer_addr, loc_key, now = peer or inline_peer, node or peer or inline_peer, time.time()
    vk, rk = key("ver", ver_id), key("ver_refs", ver_id)
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(vk)
          if raw := await p.get(vk):
            rec = json.loads(raw)
            if rec.get("state") == "deleting":
              await p.unwatch()
              raise RuntimeError(f"version {ver_id} is deleting")
            rec["durable"] = bool(rec.get("durable") or durable)
            if size:
              rec["size_bytes"] = size
            if meta:
              rec["meta"] = meta
          else:
            rec = {
              "version": ver_id,
              "uri": uri,
              "size_bytes": size,
              "durable": bool(durable),
              "locations": {},
              "state": "active",
              "kind": kind,
              "created": now,
              "meta": meta or {},
            }
          if peer_addr and loc_key:
            rec.setdefault("locations", {})[loc_key] = peer_addr
          p.multi()
          p.setnx(key("ckpt_since"), now)
          p.set(vk, json.dumps(rec))
          p.sadd(key("versions"), ver_id)
          for ref in refs:
            p.hset(rk, ref, "0")
          await p.execute()
          return rec
      except WatchError:
        continue

  async def get_version(self, version: str) -> dict | None:
    ver_id, _ = parse_version(version)
    raw = await self.r.get(key("ver", ver_id)) if ver_id else None
    return json.loads(raw) if raw else None

  async def mark_durable(self, version: str) -> bool:
    """Record confirmed durable storage commit for a version."""

    def _set(rec: dict) -> bool | None:
      if rec.get("durable"):
        return None
      rec["durable"] = True
      return True

    return await self._edit_version(version, _set)

  async def add_location(self, version: str, node: str, peer: str) -> bool:
    """Record a node holding a complete local copy of a version."""
    if not node:
      return False

    def _add(rec: dict) -> bool | None:
      if rec.get("state") != "active":
        return False
      rec.setdefault("locations", {})[node] = peer
      return True

    return await self._edit_version(version, _add)

  async def remove_location(self, version: str, node: str) -> None:
    if not node:
      return

    def _rm(rec: dict) -> bool | None:
      return (rec["locations"].pop(node, None) is not None) or None if "locations" in rec else None

    await self._edit_version(version, _rm)

  async def acquire_ref(self, version: str, ref_id: str, ttl: float | None = None) -> bool:
    """Protect a version from deletion. Fails if deletion was already claimed or version is gone."""
    ver_id, _ = parse_version(version)
    if not ver_id:
      return False
    vk, rk, val = key("ver", ver_id), key("ver_refs", ver_id), str(time.time() + ttl) if ttl else "0"
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(vk, rk)
          raw = await p.get(vk)
          if not raw or json.loads(raw).get("state") != "active":
            await p.unwatch()
            return False
          p.multi()
          p.hset(rk, ref_id, val)
          await p.execute()
          return True
      except WatchError:
        continue

  async def release_ref(self, version: str, ref_id: str) -> None:
    if ver_id := parse_version(version)[0]:
      await self.r.hdel(key("ver_refs", ver_id), ref_id)

  async def version_refs(self, version: str) -> dict[str, float]:
    if not (ver_id := parse_version(version)[0]):
      return {}
    now = time.time()
    return {
      k.decode(): float(v) for k, v in (await self.r.hgetall(key("ver_refs", ver_id))).items() if _live_ref(v, now)
    }

  async def claim_delete(self, version: str) -> bool:
    """Atomically claim a version for deletion only while zero live references protect it."""
    if not (ver_id := parse_version(version)[0]):
      return False
    vk, rk = key("ver", ver_id), key("ver_refs", ver_id)
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(vk, rk)
          if not (raw := await p.get(vk)):
            await p.unwatch()
            return False
          rec = json.loads(raw)
          if rec.get("state") == "deleting":
            await p.unwatch()
            return True
          refs, now = await p.hgetall(rk), time.time()
          if any(_live_ref(v, now) for v in refs.values()):
            await p.unwatch()
            return False
          expired = [k for k, v in refs.items() if not _live_ref(v, now)]
          rec["state"] = "deleting"
          p.multi()
          p.set(vk, json.dumps(rec))
          if expired:
            p.hdel(rk, *expired)
          await p.execute()
          return True
      except WatchError:
        continue

  async def resolve_version(self, uri: str, version: str | None = None) -> tuple[str, list[str], dict | None]:
    """Resolve an explicit version or a logical URI to (ver_id, peer_addrs, version_record)."""
    ver_id, inline_peer = parse_version(version)
    if not ver_id:
      ver_id, ckpt_peer = parse_version(((await self.get_checkpoint(uri)) or {}).get("version"))
      inline_peer = inline_peer or ckpt_peer
    rec = await self.get_version(ver_id) if ver_id else None
    peers = [inline_peer] if inline_peer else []
    for peer in (rec or {}).get("locations", {}).values():
      if peer and peer not in peers:
        peers.append(peer)
    return ver_id, peers, rec

  # ***** logical checkpoints *****

  async def register_checkpoint(
    self,
    uri: str,
    size: int,
    ttl: int | None = None,
    owned: bool = False,
    version: str | None = None,
    durable: bool = False,
    node: str = "",
    meta: dict | None = None,
  ) -> None:
    """Record a logical checkpoint pointing to an immutable version, protecting that version via a ckpt ref."""
    run, cid = uri.removeprefix("tinker://").split("/", 1)
    now, kind = time.time(), cid.split("/", 1)[0]
    ctype = {"weights": "training", "sampler_weights": "sampler"}.get(kind, kind)
    ver_id, inline_peer = parse_version(version)
    if ver_id:
      await self.register_version(
        ver_id, uri, size, kind=ctype, durable=durable, node=node, peer=inline_peer, meta=meta, refs=(f"ckpt:{uri}",)
      )
    info = {
      "checkpoint_id": cid,
      "checkpoint_type": ctype,
      "time": now,
      "tinker_path": uri,
      "size_bytes": size,
      "expires_at": now + ttl if ttl else None,
      "owned": owned,
      "version": version,
      "version_id": ver_id or None,
      "durable": bool(durable),
    }
    ck = key("ckpt", run)
    while True:
      try:
        async with self.r.pipeline(transaction=True) as p:
          await p.watch(ck)
          prev_raw = await p.hget(ck, cid)
          prev_ver = parse_version(json.loads(prev_raw).get("version"))[0] if prev_raw else ""
          p.multi()
          p.setnx(key("ckpt_since"), now)
          p.hset(ck, cid, json.dumps(info))
          if kind != "parked":
            p.zadd(key("ckpts"), {uri: now})
          p.zadd(key("expiry"), {uri: now + ttl}) if ttl else p.zrem(key("expiry"), uri)
          if prev_ver and prev_ver != ver_id:
            p.hdel(key("ver_refs", prev_ver), f"ckpt:{uri}")
          await p.execute()
          return
      except WatchError:
        continue

  async def get_checkpoint(self, uri: str) -> dict | None:
    run, cid = uri.removeprefix("tinker://").split("/", 1)
    info = await self.r.hget(key("ckpt", run), cid)
    return json.loads(info) if info else None

  async def version(self, uri: str) -> str | None:
    """The version reference to read a checkpoint by."""
    return (await self.get_checkpoint(uri) or {}).get("version")

  async def checkpoints_of_run(self, run: str) -> list[dict]:
    found = [json.loads(v) for v in (await self.r.hgetall(key("ckpt", run))).values()]
    return sorted((c for c in found if c["checkpoint_type"] != "parked"), key=lambda c: c["time"])

  async def page_of_checkpoints(self, limit: int, offset: int) -> tuple[list[dict], int]:
    """A page of every run's checkpoints, newest first, and how many there are."""
    uris = [u.decode() for u in await self.r.zrevrange(key("ckpts"), offset, offset + limit - 1)]
    return [c for c in [await self.get_checkpoint(u) for u in uris] if c], await self.r.zcard(key("ckpts"))

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
    """Unregister logical checkpoints and release their ckpt references on underlying versions."""
    if not uris:
      return 0
    existing = await asyncio.gather(*(self.get_checkpoint(u) for u in uris))
    p = self.r.pipeline(transaction=True)
    for uri, info in zip(uris, existing):
      run, cid = uri.removeprefix("tinker://").split("/", 1)
      p.hdel(key("ckpt", run), cid)
      p.zrem(key("ckpts"), uri)
      p.zrem(key("expiry"), uri)
      if info and (ver := parse_version(info.get("version"))[0]):
        p.hdel(key("ver_refs", ver), f"ckpt:{uri}")
    res = await p.execute()
    idx, removed = 0, 0
    for info in existing:
      removed += res[idx]
      idx += 4 if (info and parse_version(info.get("version"))[0]) else 3
    return removed

  async def checkpoints_owned_by_run(self, run: str) -> list[str]:
    return [c["tinker_path"] for c in map(json.loads, (await self.r.hgetall(key("ckpt", run))).values()) if c["owned"]]

  async def expired_checkpoints(self) -> list[str]:
    """The checkpoints whose TTL has run out."""
    return [u.decode() for u in await self.r.zrangebyscore(key("expiry"), 0, time.time())]

  async def _all_checkpoints(self) -> list[dict]:
    out = []
    async for k in self.r.scan_iter(match=key("ckpt", "*")):
      out.extend(map(json.loads, await self.r.hvals(k)))
    return out

  async def registered(self) -> set[str]:
    """Every logical checkpoint URI in the registry, parked ones too."""
    return {c["tinker_path"] for c in await self._all_checkpoints()}

  async def protected_versions(self, ckpts: list[dict] | None = None) -> set[str]:
    """Every version ID currently protected by at least one live reference or logical checkpoint."""
    protected, now = set(), time.time()
    vers = [v.decode() for v in await self.r.smembers(key("versions"))]
    if vers:
      p = self.r.pipeline(transaction=False)
      for v in vers:
        p.get(key("ver", v))
        p.hgetall(key("ver_refs", v))
      res = await p.execute()
      for i, v in enumerate(vers):
        raw, refs = res[2 * i], res[2 * i + 1]
        if raw and json.loads(raw).get("state") == "active" and any(_live_ref(val, now) for val in refs.values()):
          protected.add(v)
    for info in (await self._all_checkpoints()) if ckpts is None else ckpts:
      if ver := parse_version(info.get("version"))[0]:
        protected.add(ver)
    return protected

  async def sweep(self, store: Store, grace: float) -> None:
    """Reclaim unreferenced versions older than grace (via claim_delete first), then sweep store & trim cache."""
    since_raw = await self.r.get(key("ckpt_since"))
    since = float(since_raw) if since_raw else None
    if since is not None:
      before = time.time() - grace
      for ver_id in [v.decode() for v in await self.r.smembers(key("versions"))]:
        rec = await self.get_version(ver_id)
        if not rec:
          await self.r.srem(key("versions"), ver_id)
          continue
        if rec.get("state") == "deleting" or (
          since < float(rec.get("created", 0.0)) < before and await self.claim_delete(ver_id)
        ):
          if await asyncio.to_thread(store.delete_version, rec["uri"], ver_id):
            p = self.r.pipeline(transaction=True)
            p.delete(key("ver", ver_id), key("ver_refs", ver_id))
            p.srem(key("versions"), ver_id)
            await p.execute()
    ckpts = await self._all_checkpoints()
    registered = {c["tinker_path"] for c in ckpts}
    versionless = {c["tinker_path"] for c in ckpts if not parse_version(c.get("version"))[0]}
    protected = await self.protected_versions(ckpts)
    await asyncio.to_thread(store.sweep, registered, since, grace, protected, versionless)
