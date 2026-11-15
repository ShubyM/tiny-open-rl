"""The checkpoint registry, on Redis: immutable versions, their locations, durability and retention references, plus the logical checkpoint names that point at them. Sweep claims deletion only while a version has no live references."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid

from tinyrl.bus import Bus, ckpt_doc, key, merge_locations, version_doc, write_ckpt_index
from tinyrl.storage.store import Store, split_version

READ_REF_TTL = 120.0


def live_refs(raw: dict[bytes, bytes], now: float) -> dict[str, float]:
  return {h.decode(): exp for h, b in raw.items() if (exp := float(b)) == 0.0 or exp > now}


class Registry:
  def __init__(self, bus: Bus):
    self.bus, self.r = bus, bus.r

  # ***** immutable versions & references *****

  async def publish_version(
    self,
    uri: str,
    version: str,
    size: int,
    *,
    meta: dict | None = None,
    durable: bool = False,
    location: str = "",
    refs: dict[str, float] | None = None,
  ) -> str:
    """Publish or update an immutable version record and attach initial protective references atomically."""
    ver, hint = split_version(version)
    if not ver:
      raise ValueError(f"missing version for {uri}")
    vkey, rkey = key("ver", ver), key("vref", ver)

    async def step(p):
      cur = json.loads(raw) if (raw := await p.get(vkey)) else None
      if cur and cur.get("state") == "deleting":
        await p.unwatch()
        raise ValueError(f"version {ver} of {uri} is being deleted")
      now = time.time()
      p.multi()
      p.setnx(key("ckpt_since"), now)
      p.set(vkey, json.dumps(version_doc(cur, uri, ver, size, meta, durable, now, hint, location)))
      p.sadd(key("vers"), ver)
      if refs:
        p.hset(rkey, mapping={h: str(now + ttl if ttl else 0.0) for h, ttl in refs.items()})
      return True, ver

    return await self.bus.tx((vkey, rkey), step)

  async def get_version(self, version: str | None) -> dict | None:
    ver, _ = split_version(version)
    return json.loads(raw) if ver and (raw := await self.r.get(key("ver", ver))) else None

  async def _mutate_version(self, version: str, fn) -> dict | None:
    ver, hint = split_version(version)
    if not ver:
      return None
    vkey = key("ver", ver)

    async def step(p):
      info = json.loads(raw) if (raw := await p.get(vkey)) else None
      updated = fn(dict(info), hint) if info else None
      if not updated or updated == info:
        return False, (info if updated == info else None)
      p.multi()
      p.set(vkey, json.dumps(updated))
      return True, updated

    return await self.bus.tx((vkey,), step)

  async def mark_durable(self, version: str, location: str = "") -> dict | None:
    def upd(info: dict, hint: str) -> dict | None:
      if info.get("state") != "live":
        return None
      return info | {"durable": True, "locations": merge_locations(info.get("locations", []), hint, location)}

    return await self._mutate_version(version, upd)

  async def set_location(self, version: str, peer: str, present: bool) -> None:
    def upd(i: dict, _: str) -> dict:
      locs = i.get("locations", [])
      return i | {"locations": merge_locations(locs, peer) if present else [x for x in locs if x != peer]}

    if peer:
      await self._mutate_version(version, upd)

  async def acquire_ref(self, version: str, holder: str, ttl: float = 0.0) -> dict | None:
    """Protect `version` under `holder`. Returns None if the version is missing or already claimed for deletion."""
    if not (ver := split_version(version)[0]):
      return None
    vkey, rkey = key("ver", ver), key("vref", ver)

    async def step(p):
      info = json.loads(raw) if (raw := await p.get(vkey)) else None
      if not info or info.get("state") != "live":
        return False, None
      p.multi()
      p.hset(rkey, holder, str(time.time() + ttl if ttl > 0 else 0.0))
      return True, info

    return await self.bus.tx((vkey, rkey), step)

  async def release_ref(self, version: str | None, holder: str) -> None:
    if ver := split_version(version)[0]:
      await self.r.hdel(key("vref", ver), holder)

  async def version_refs(self, version: str) -> dict[str, float]:
    ver, _ = split_version(version)
    return live_refs(await self.r.hgetall(key("vref", ver)), time.time()) if ver else {}

  async def claim_delete(self, version: str, grace: float = 0.0) -> dict | None:
    """Atomically claim `version` for deletion iff no live reference holds it; later acquire_ref calls fail."""
    if not (ver := split_version(version)[0]):
      return None
    vkey, rkey = key("ver", ver), key("vref", ver)

    async def step(p):
      raw, refs = await p.get(vkey), await p.hgetall(rkey)
      if not raw:
        p.multi()
        p.delete(rkey)
        p.srem(key("vers"), ver)
        return True, None
      info = json.loads(raw)
      if info.get("state") == "deleting":
        return False, info
      now = time.time()
      active = live_refs(refs, now)
      if active or (grace > 0 and now - float(info.get("time", 0.0)) < grace):
        if expired := [k for k in refs if k.decode() not in active]:
          p.multi()
          p.hdel(rkey, *expired)
          return True, None
        return False, None
      claimed = info | {"state": "deleting"}
      p.multi()
      p.set(vkey, json.dumps(claimed))
      p.delete(rkey)
      return True, claimed

    return await self.bus.tx((vkey, rkey), step)

  async def finish_delete(self, version: str) -> None:
    if ver := split_version(version)[0]:
      p = self.r.pipeline(transaction=True)
      p.delete(key("ver", ver), key("vref", ver))
      p.srem(key("vers"), ver)
      await p.execute()

  async def resolve(
    self, uri: str, version: str | None = None
  ) -> tuple[str | None, tuple[str, ...], bool, dict | None]:
    """Resolve an explicit version or a registered URI into (version, peer_locations, durable, version_info)."""
    ver, hint = split_version(version)
    if ver is None and (ckpt := await self.get_checkpoint(uri)) and ckpt.get("version"):
      ver, hint = split_version(ckpt["version"])
    if ver is None:
      return None, (), False, None
    if (info := await self.get_version(ver)) and info.get("state") == "deleting":
      raise ValueError(f"version {ver} of {uri} has been deleted")
    return ver, tuple(merge_locations((info or {}).get("locations", []), hint)), bool((info or {}).get("durable")), info

  @contextlib.asynccontextmanager
  async def read_ref(self, uri: str, version: str | None = None, ttl: float = READ_REF_TTL):
    """Hold a cluster-visible reader reference on the resolved version for the duration of a read or prepare."""
    explicit, hint = split_version(version)
    holder = f"read:{uuid.uuid4().hex[:16]}"
    while True:
      ver, peers, durable, info = await self.resolve(uri, version)
      if ver is None or info is None:
        yield ver, peers, durable
        return
      if claimed := await self.acquire_ref(ver, holder, ttl=ttl):
        try:
          yield ver, tuple(merge_locations(claimed.get("locations", []), hint)), bool(claimed.get("durable"))
        finally:
          await self.release_ref(ver, holder)
        return
      if explicit:
        raise ValueError(f"version {explicit} of {uri} has been deleted")

  # ***** named checkpoints *****

  async def register_checkpoint(
    self,
    uri: str,
    size: int,
    ttl: int | None = None,
    owned: bool = False,
    version: str | None = None,
    *,
    meta: dict | None = None,
    durable: bool = False,
    location: str = "",
  ) -> str | None:
    """Record a checkpoint pointing at an immutable version, swapping the name's reference cleanly on overwrite."""
    run, cid = uri.removeprefix("tinker://").split("/", 1)
    ver, hint = split_version(version)
    ckey, holder = key("ckpt", run), f"ckpt:{uri}"

    async def step(p):
      prev_ver = split_version(json.loads(raw).get("version"))[0] if (raw := await p.hget(ckey, cid)) else None
      cur_ver = json.loads(vraw) if ver and (vraw := await p.get(key("ver", ver))) else None
      if cur_ver and cur_ver.get("state") == "deleting":
        await p.unwatch()
        raise ValueError(f"version {ver} of {uri} is being deleted")
      now, is_dur = time.time(), bool(durable or (cur_ver or {}).get("durable"))
      p.multi()
      p.setnx(key("ckpt_since"), now)
      write_ckpt_index(p, ckey, cid, uri, ckpt_doc(cid, uri, size, ttl, owned, ver, is_dur, now), ttl, now)
      if ver:
        p.set(key("ver", ver), json.dumps(version_doc(cur_ver, uri, ver, size, meta, is_dur, now, hint, location)))
        p.sadd(key("vers"), ver)
        p.hset(key("vref", ver), holder, "0.0")
      if prev_ver and prev_ver != ver:
        p.hdel(key("vref", prev_ver), holder)
      return True, ver

    return await self.bus.tx([ckey, *([key("ver", ver), key("vref", ver)] if ver else [])], step)

  async def get_checkpoint(self, uri: str) -> dict | None:
    run, cid = uri.removeprefix("tinker://").split("/", 1)
    return json.loads(info) if (info := await self.r.hget(key("ckpt", run), cid)) else None

  async def version(self, uri: str) -> str | None:
    return (await self.get_checkpoint(uri) or {}).get("version")

  async def checkpoints_of_run(self, run: str) -> list[dict]:
    found = [json.loads(v) for v in (await self.r.hgetall(key("ckpt", run))).values()]
    return sorted((c for c in found if c["checkpoint_type"] not in ("parked", "recovery")), key=lambda c: c["time"])

  async def page_of_checkpoints(self, limit: int, offset: int) -> tuple[list[dict], int]:
    uris = [u.decode() for u in await self.r.zrevrange(key("ckpts"), offset, offset + limit - 1)]
    return [c for u in uris if (c := await self.get_checkpoint(u))], await self.r.zcard(key("ckpts"))

  async def set_checkpoint_ttl(self, uri: str, ttl: int | None) -> bool:
    run, cid = uri.removeprefix("tinker://").split("/", 1)
    ckey = key("ckpt", run)

    async def step(p):
      if (data := await p.hget(ckey, cid)) is None:
        return False, False
      info = json.loads(data) | {"expires_at": time.time() + ttl if ttl else None}
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

      async def step(p, ckey=ckey, cid=cid, uri=uri):
        ver = split_version(json.loads(raw).get("version"))[0] if (raw := await p.hget(ckey, cid)) else None
        p.multi()
        p.hdel(ckey, cid)
        p.zrem(key("ckpts"), uri)
        p.zrem(key("expiry"), uri)
        if ver:
          p.hdel(key("vref", ver), f"ckpt:{uri}")
        return True, lambda res: int(res[0])

      removed += await self.bus.tx((ckey,), step)
    return removed

  async def expired_checkpoints(self) -> list[str]:
    return [u.decode() for u in await self.r.zrangebyscore(key("expiry"), 0, time.time())]

  async def all_versions(self) -> dict[str, dict]:
    if not (vers := sorted(v.decode() for v in await self.r.smembers(key("vers")))):
      return {}
    return {v: json.loads(raw) for v, raw in zip(vers, await self.r.mget(*(key("ver", v) for v in vers))) if raw}

  async def registered(self) -> set[str]:
    uris = {i["uri"] for i in (await self.all_versions()).values() if i.get("state") == "live"}
    async for k in self.r.scan_iter(match=key("ckpt", "*")):
      uris.update(json.loads(v)["tinker_path"] for v in await self.r.hvals(k))
    return uris

  async def sync_store(self, store: Store) -> None:
    if not store.remote:
      return
    cached = set(await asyncio.to_thread(store.cached_versions))
    for ver, info in (await self.all_versions()).items():
      if info.get("state") != "live":
        continue
      if not info.get("durable") and ver in cached and await asyncio.to_thread(store.is_durable, info["uri"], ver):
        await self.mark_durable(ver, store.peer)
      elif store.peer and (store.peer in info.get("locations", [])) != (ver in cached):
        await self.set_location(ver, store.peer, ver in cached)

  async def sweep(self, store: Store, grace: float, live_peers: set[str] | None = None) -> None:
    await self.sync_store(store)
    now, pinned, candidates = time.time(), set(), []
    for ver, info in (await self.all_versions()).items():
      if info.get("state") == "deleting":
        candidates.append(ver)
      elif refs := live_refs(await self.r.hgetall(key("vref", ver)), now):
        if any(not h.startswith("ckpt:") for h in refs):
          pinned.add(ver)
      else:
        candidates.append(ver)
    for ver in candidates:
      if not (claimed := await self.claim_delete(ver, grace=grace)) or not await asyncio.to_thread(
        store.delete_version, claimed["uri"], ver
      ):
        continue
      await self.set_location(ver, store.peer, False)
      locs = ((await self.get_version(ver)) or claimed).get("locations", [])
      if not store.remote or not [p for p in locs if p and p != store.peer and (live_peers is None or p in live_peers)]:
        await self.finish_delete(ver)
    since = await self.r.get(key("ckpt_since"))
    live = {v for v, i in (await self.all_versions()).items() if i.get("state") == "live"}
    evicted = await asyncio.to_thread(
      store.sweep, await self.registered(), float(since) if since else None, grace, pinned, live
    )
    for ver in evicted if store.peer else ():
      await self.set_location(ver, store.peer, False)
