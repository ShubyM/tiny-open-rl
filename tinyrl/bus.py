# every training call is an entry on its run's redis stream. a pool's trainer reads its runs' streams through one consumer group, so
# redis itself knows what is queued (lag), what is in flight and on which trainer (the PEL), and what to fail when a trainer dies
# (XAUTOCLAIM). a reply is a one entry stream: XREAD from 0 blocks until it is written and never consumes it, so retries are free.
from __future__ import annotations
import asyncio, json, random, time
from dataclasses import dataclass, asdict
import redis.asyncio as redis
from tinyrl.helpers import getenv

GROUP, LIVE, REPLY_TTL, SESSION_TTL = "trainer", 120, getenv("REPLY_TTL", 3600), getenv("SESSION_TTL", 7*86400)
def key(*parts:str) -> str: return ":".join(("tinyrl", *parts))

@dataclass(frozen=True)
class Run: id:str; pool:str; base:str; rank:int; session:str = "" # noqa: E702

@dataclass(frozen=True)
class Cmd:
  run:str; id:str; op:str; body:bytes # noqa: E702
  @property
  def rid(self) -> str: return f"{self.run}/{self.id}"

@dataclass(frozen=True)
class Reply: ok:bool; ct:str; body:bytes # noqa: E702

def cmd(run:str, eid:bytes, f:dict[bytes, bytes]) -> Cmd: return Cmd(run, eid.decode(), f[b"op"].decode(), f[b"b"])

class Bus:
  # a pool has one trainer, and every pending entry in the pool's streams is that trainer's
  def __init__(self, r:redis.Redis|None=None): self.r = redis.from_url(getenv("REDIS_URL", "redis://localhost:6379")) if r is None else r

  async def session(self, sid:str) -> None:
    k, now = key("sess", sid), time.time()
    await self.r.pipeline().hsetnx(k, "created", now).hset(k, "seen", now).expire(k, SESSION_TTL).expire(f"{k}:runs", SESSION_TTL).execute()

  async def session_info(self, sid:str) -> dict|None:
    h, runs = await self.r.pipeline().hgetall(key("sess", sid)).smembers(key("sess", sid, "runs")).execute()
    if not h: return None
    seen = float(h[b"seen"])
    return {"session_id": sid, "created": float(h[b"created"]), "seen": seen, "live": time.time()-seen < LIVE,
            "runs": sorted(r.decode() for r in runs)}

  async def create(self, run:Run) -> None:
    # one transaction: a run is in its pool exactly when its stream and consumer group exist
    p = self.r.pipeline().xgroup_create(key("q", run.id), GROUP, id="0", mkstream=True).sadd(key("pool", run.pool), run.id)
    p.hset(key("run", run.id), mapping={"rec": json.dumps({**asdict(run), "created": time.time()}), "steps": 0, "failed": 0})
    if run.session: p.sadd(key("sess", run.session, "runs"), run.id).expire(key("sess", run.session, "runs"), SESSION_TTL)
    await p.execute()

  async def run(self, rid:str) -> dict|None:
    if not (h := await self.r.hgetall(key("run", rid))): return None
    return {**json.loads(h.pop(b"rec")), **{k.decode():int(v) for k,v in h.items()}}

  async def runs(self, pool:str) -> list[str]: return [r.decode() for r in await self.r.smembers(key("pool", pool))]

  async def drop(self, run:str, pool:str) -> None: await self.r.pipeline().srem(key("pool", pool), run).delete(key("q", run)).execute()

  async def submit(self, run:str, op:str, body:bytes) -> str|None:
    # NOMKSTREAM: an unknown or deleted run has no stream, so the write fails instead of queueing work nobody will read
    eid = await self.r.xadd(key("q", run), {"op": op, "b": body}, nomkstream=True)
    return None if eid is None else f"{run}/{eid.decode()}"

  async def take(self, pool:str, worker:str, count:int=16, block:int=1000) -> list[Cmd]:
    if not (runs := await self.runs(pool)): return await asyncio.sleep(block/1000, [])
    # every run with work gets up to count entries per take, in a fresh order each time
    got = await self.r.xreadgroup(GROUP, worker, {key("q", r): ">" for r in random.sample(runs, len(runs))}, count=count, block=block)
    return [cmd(s.decode().removeprefix(key("q", "")), i, f) for s,es in got for i,f in es]

  async def finish(self, c:Cmd, rep:Reply) -> None:
    # the reply, the ack and the run's counters land together or not at all
    p = self.r.pipeline().xadd(key("f", c.rid), {"ok": int(rep.ok), "ct": rep.ct, "b": rep.body}, maxlen=1).expire(key("f", c.rid), REPLY_TTL)
    p.xack(key("q", c.run), GROUP, c.id).xdel(key("q", c.run), c.id)
    if not rep.ok or c.op == "optim_step": p.hincrby(key("run", c.run), "steps" if rep.ok else "failed", 1)
    await p.execute()

  async def reclaim(self, pool:str, worker:str) -> list[Cmd]:
    # at startup every pending entry in the pool is from a trainer that died. a trainer has one take in flight, so one claim gets them all
    return [cmd(r, i, f) for r in await self.runs(pool) for i,f in (await self.r.xautoclaim(key("q", r), GROUP, worker, 0, count=10000))[1]]

  async def result(self, rid:str, timeout:float) -> Reply|None:
    if not (got := await self.r.xread({key("f", rid): "0-0"}, count=1, block=int(timeout*1000) or None)): return None
    f = got[0][1][0][1]
    return Reply(f[b"ok"] == b"1", f[b"ct"].decode(), f[b"b"])

  async def queue(self, run:str) -> dict:
    groups, pending = await self.r.pipeline().xinfo_groups(k:=key("q", run)).xpending_range(k, GROUP, "-", "+", 100).execute(raise_on_error=False)
    if isinstance(groups, Exception): return {"live": False}
    inflight = [{"request_id": f"{run}/{p['message_id'].decode()}", "trainer": p["consumer"].decode(), "seconds": p["time_since_delivered"]/1000,
                 "deliveries": p["times_delivered"]} for p in pending]
    return {"live": True, "queued": groups[0]["lag"], "in_flight": inflight}
