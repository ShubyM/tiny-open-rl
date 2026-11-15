"""Claims own live job state. Immutable turns own GPU execution until physical release."""

from __future__ import annotations

import asyncio
import copy
import itertools
import json
import secrets

from tinyrl.bus import Bus, Role, key
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry

TURN, TICK_MS = getenv("TURN", 200_000), getenv("TICK_MS", 100)


def account(run: str, info: dict) -> str:
  return info.get("train_run") or run


def compatible(run: str, info: dict) -> tuple:
  role, lora = info.get("role", Role.TRAINER), bool(info.get("rank", 0))
  private = account(run, info) if info.get("exclusive") else run if role == Role.TRAINER and not lora else None
  return role, info.get("base", ""), lora, info.get("gpus", 1), private


def bundles(info: dict, width: int) -> list[tuple[int, ...]]:
  ids = tuple(info.get("gpu_ids", range(info["gpus"])))
  return sorted(
    {
      tuple(b)
      for group in info.get("bundles") or [ids]
      for b in itertools.combinations(group, width)
      if set(b) <= set(ids)
    }
  )


def decide(table: dict) -> dict:
  s = copy.deepcopy(table)
  for name in ("claims", "turns", "observations", "service", "moves", "reservations", "fair"):
    s.setdefault(name, {})
  live = set(s.get("live", ()))
  runs, nodes, claims, turns, obs = (s[n] for n in ("runs", "nodes", "claims", "turns", "observations"))
  busy = {t["engine"] for t in turns.values()}
  eligible = {n for n, v in nodes.items() if v.get("eligible", True) and not v.get("draining")}
  for r, c in list(claims.items()):
    o = obs.get(c["engine"])
    cold = not (o or {}).get("gpus") and c["engine"] not in busy
    if (
      r not in runs
      or c["node"] not in nodes
      or (cold and (c["node"] not in eligible or r in s["moves"]))
      or (o is not None and (o.get("closed") or o.get("node") != c["node"] or c["engine"] not in live))
    ):
      del claims[r]
  s["moves"] = {r: n for r, n in s["moves"].items() if r in runs}
  moves = s["moves"]
  for r, c in claims.items():
    if moves.get(r) == c["node"] and c["node"] in eligible and not obs.get(c["engine"], {}).get("draining"):
      del moves[r]

  owned, load = {}, dict.fromkeys(nodes, 0)
  for r, c in claims.items():
    owned.setdefault(c["engine"], []).append(r)
    load[c["node"]] += runs[r].get("gpus", 1)

  def jobs(engine):
    return owned.get(engine, ())

  def occupied(node, bundle, holders):
    return any(v["node"] == node and set(bundle) & set(v["gpus"]) for v in holders)

  def capacity(node, bundle, r, admit=False):
    owned = s["reservations"].get(node, {})
    a = account(r, runs[r])
    if runs[r].get("exclusive"):
      return all(owned.get(str(g)) == a or (admit and str(g) not in owned) for g in bundle)
    return all(str(g) not in owned for g in bundle)

  # Dedicated capacity follows accounts with claims on this node, not an engine's lifetime.
  s["reservations"] = {
    n: {g: a for g, a in held.items() if any(c["node"] == n and account(r, runs[r]) == a for r, c in claims.items())}
    for n, held in s["reservations"].items()
    if n in nodes
  }
  for r, info in sorted(runs.items()):
    if r in claims:
      continue
    target = moves.get(r)
    choices = []
    for n in sorted(eligible):
      if target and n != target:
        continue
      options = [b for b in bundles(nodes[n], info.get("gpus", 1)) if capacity(n, b, r, admit=True)]
      if info.get("exclusive"):
        foreign = [
          o
          for e, o in obs.items()
          if e in live
          and not o.get("closed")
          and o.get("gpus")
          and (
            not jobs(e) or any(not runs[j].get("exclusive") or account(j, runs[j]) != account(r, info) for j in jobs(e))
          )
        ]
        foreign += [
          t for t in turns.values() if any(j not in runs or account(j, runs[j]) != account(r, info) for j in t["work"])
        ]
        options = [b for b in options if not occupied(n, b, foreign)]
      if not options:
        continue
      own = s["reservations"].get(n, {})
      bundle = min(options, key=lambda b: (sum(own.get(str(g)) != account(r, info) for g in b), b))
      choices.append(
        (not any(own.get(str(g)) == account(r, info) for g in bundle), load[n] / nodes[n]["gpus"], n, bundle)
      )
    if not choices:
      continue
    _, _, node, bundle = min(choices)
    engines = [e for e, rs in owned.items() if claims[rs[0]]["node"] == node]
    sharing = [
      e
      for e in engines
      if not obs.get(e, {}).get("draining") and compatible(jobs(e)[0], runs[jobs(e)[0]]) == compatible(r, info)
    ]
    used = sum(runs[jobs(e)[0]].get("gpus", 1) for e in engines)
    engine = (
      min(sharing, key=lambda e: (len(jobs(e)), e))
      if sharing and used >= nodes[node]["gpus"]
      else "e-" + secrets.token_hex(12)
    )
    claims[r] = {"node": node, "engine": engine, "token": secrets.token_hex(16)}
    owned.setdefault(engine, []).append(r)
    load[node] += info.get("gpus", 1)
    moves.pop(r, None)
    if info.get("exclusive"):
      s["reservations"].setdefault(node, {}).update({str(g): account(r, info) for g in bundle})
  s["reservations"] = {n: v for n, v in s["reservations"].items() if v}

  exporting = {
    r for r, c in claims.items() if r in moves or c["node"] not in eligible or obs.get(c["engine"], {}).get("draining")
  }
  ready = {
    r for r, c in claims.items() if c["engine"] in obs and (r in exporting or r in obs[c["engine"]].get("ready", ()))
  }
  active = {account(r, runs[r]) for r in ready}
  previous = s["fair"]
  continuing = active & set(previous.get("active", ()))
  floor = max(
    previous.get("floor", 0.0),
    min((s["service"].get(a, 0.0) for a in continuing or previous.get("active", ())), default=0.0),
  )
  for a in active - continuing:
    s["service"][a] = max(s["service"].get(a, 0.0), floor)
  accounts = {account(r, i) for r, i in runs.items()}
  last = {a: e for a, e in previous.get("last", {}).items() if a in accounts}
  s["fair"] = {"floor": floor, "active": sorted(active), "last": last}
  s["service"] = {a: s["service"][a] for a in accounts & s["service"].keys()}
  debt = {}

  def charge(turn):
    for r, (token, _) in turn["work"].items():
      c = claims.get(r, {})
      if turn["engine"] in live and (c.get("engine"), c.get("token")) == (turn["engine"], token):
        a = account(r, runs[r])
        debt[a] = debt.get(a, 0.0) + 1 / len(turn["work"])

  for t in turns.values():
    charge(t)

  def priority(r):
    a = account(r, runs[r])
    return debt.get(a, 0.0), s["service"].get(a, floor), claims[r]["engine"] == last.get(a)

  pending = {c["engine"] for r, c in claims.items() if r in ready} - busy
  planned = []
  # Issued turns count before receipts arrive, so one account cannot take every free GPU first.
  while pending:
    engine = min(pending, key=lambda e: (min(priority(r) for r in jobs(e) if r in ready), e))
    pending.remove(engine)
    work = [r for r in jobs(engine) if r in ready]
    first, node = work[0], claims[work[0]]["node"]
    if node not in nodes:
      continue
    bound = obs.get(engine, {}).get("gpus")
    options = [b for b in bundles(nodes[node], runs[first].get("gpus", 1)) if not bound or b == tuple(bound)]
    options = [b for b in options if capacity(node, b, first)]
    options.sort(key=lambda b: (occupied(node, b, turns.values()), b))
    bundle = next((b for b in options if not occupied(node, b, planned)), None)
    if bundle is None:
      continue
    # A waiting gang keeps its fair position while existing turns finish; later turns cannot refill its bundle.
    planned.append({"node": node, "gpus": bundle})
    if occupied(node, bundle, turns.values()):
      continue
    best = min(map(priority, work))
    selected = [r for r in work if priority(r) == best]
    turn = {
      "node": node,
      "engine": engine,
      "gpus": list(bundle),
      "work": {r: [claims[r]["token"], 0 if r in exporting else max(1, TURN // len(selected))] for r in selected},
    }
    turns["t-" + secrets.token_hex(12)] = turn
    charge(turn)
    last.update({account(r, runs[r]): engine for r in selected})
  referenced = {c["engine"] for c in claims.values()} | {t["engine"] for t in turns.values()}
  s["observations"] = {e: o for e, o in obs.items() if not o.get("closed") or e in referenced}
  return s


async def commit(bus: Bus, old: dict, new: dict, holder: str | None = None) -> bool:
  if any(new["turns"].get(t) != v for t, v in old["turns"].items()):
    raise ValueError("only release_turn may retire a turn")
  writable = ("claims", "turns", "moves", "service", "reservations", "fair")
  retired = old["observations"].keys() - new["observations"].keys()
  if not retired and all(old.get(n, {}) == new.get(n, {}) for n in writable):
    return False
  watched = [key(n) for n in ("runs", "nodes", "claims", "turns", "moves", "obs", "service", "reservations", "fair")]
  watched += [key("run", r) for r in old["runs"]] + [key("live", e) for e in old["live"]] + [key("owner", "scheduler")]

  async def step(p):
    if holder and await p.get(key("owner", "scheduler")) != holder.encode():
      return False, False
    current = await bus.read_table()
    if current != old:
      return False, False
    p.multi()
    if retired:
      p.hdel(key("obs"), *retired)
    for name in writable[:-1]:
      changed = {k: json.dumps(v) for k, v in new[name].items() if old[name].get(k) != v}
      if changed:
        p.hset(key(name), mapping=changed)
      if gone := old[name].keys() - new[name].keys():
        p.hdel(key(name), *gone)
    p.set(key("fair"), json.dumps(new["fair"]))
    return True, True

  return await bus.tx(watched, step)


async def run(bus: Bus) -> None:
  registry, holder = Registry(bus), secrets.token_hex(16)
  while True:
    await asyncio.sleep(TICK_MS / 1000)
    if not await bus.claim_scheduler(holder):
      continue
    for session in await bus.expired_sessions():
      await bus.end_session(session)
    await registry.unregister_checkpoints(*await registry.expired_checkpoints())
    old = await bus.read_table()
    await commit(bus, old, decide(old), holder)


if __name__ == "__main__":
  asyncio.run(run(Bus()))
