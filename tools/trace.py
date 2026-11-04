"""One timeline of a tinyrl cluster, for ui.perfetto.dev: the event log's system view (each worker's process, GPU turns, waits, batches and the commands in them, placements, revokes) and, with --profile, every TINYRL_PROFILE trace of trainer and vLLM ops and CUDA kernels on the same clock.

usage: REDIS_URL=redis://... python tools/trace.py out.json.gz [--since unix_s] [--until unix_s] [--profile dir]
"""

import argparse
import asyncio
import collections
import glob
import gzip
import json
import os

from tinyrl.bus import Bus, connect


async def read_events(since: float, until: float) -> list:
  bus, out, after = Bus(connect(os.environ.get("REDIS_URL", "redis://localhost:6379"))), [], f"{int(since * 1000)}-0"
  while got := await bus.read_events(after, count=10000):
    out += [e for e in got if e.t <= until]
    if got[-1].t > until:
      break
    after = got[-1].id
  return out


def system_view(events: list, t0: float) -> list:
  """Chrome trace events for the log: pid 1, a thread per worker (named by its role, base model and runs) and per node daemon."""
  us = lambda t: (t - t0) * 1e6  # noqa: E731
  runs, worker_runs, tids, out = {}, collections.defaultdict(set), {}, []
  opened = collections.defaultdict(dict)  # worker -> name of a span the log opens -> its start
  reported = collections.defaultdict(list)  # worker, or "node <id>" -> the spans its process reported

  def tid(w: str, lane: int = 0) -> int:
    return tids.setdefault((w, lane), len(tids) + 1)

  def span(w: str, name: str, start: float, end: float, args: dict | None = None, lane: int = 0) -> None:
    out.append(
      {
        "ph": "X",
        "pid": 1,
        "tid": tid(w, lane),
        "name": name,
        "ts": us(start),
        "dur": max(0.0, (end - start) * 1e6),
        "args": args or {},
      }
    )

  for e in events:
    b, w = e.body, e.body.get("worker")
    reported[w or f"node {b.get('node')}"] += b.get("spans", [])
    if e.kind == "run_created":
      runs[b["run"]] = b
    if e.kind == "assigned" and b.get("run") in runs:
      worker_runs[w].add(b["run"])
    if not w:
      continue
    o = opened[w]
    match e.kind:
      case "started":
        o["process"] = e.t
      case "exited" if "process" in o:
        span(w, "process", o.pop("process"), e.t)
        if b.get("stopping"):
          span(w, "stopping", e.t - b["stopping"], e.t)
      case "wants_gpu" if "wait" not in o and "turn" not in o:
        o["wait"] = e.t
      case "granted":
        if "wait" in o:
          span(w, "wait for turn", o.pop("wait"), e.t)
        o["turn"] = e.t
      case "released" if "turn" in o:
        span(w, "turn", o.pop("turn"), e.t)
      case "batch_done":
        start = e.t - b.get("seconds", 0.0)
        span(w, "batch", start, e.t, {"queued": b.get("queued")})
        ops = b.get("ops", [])  # logs from before spans: a trainer's commands, in the order they ran, back to back
        if (
          sum(ms for _, ms in ops) <= b.get("seconds", 0.0) * 1000 * 1.01
        ):  # a sampler's ran at once, so they don't fit and aren't drawn
          for op, ms in ops:
            span(w, op, start, start + ms / 1000)
            start += ms / 1000
      case "placed" | "unplaced" | "revoked" | "idle":
        out.append(
          {
            "ph": "i",
            "s": "t",
            "pid": 1,
            "tid": tid(w),
            "name": e.kind,
            "ts": us(e.t),
            "args": {k: v for k, v in b.items() if k != "spans"},
          }
        )
  for (
    w,
    got,
  ) in (
    reported.items()
  ):  # spans nest by time; ones that overlap without nesting, like concurrent samples, get lanes of their own
    lanes: list[list[float]] = []  # per lane, the ends of the spans still open on it
    for name, start, seconds in sorted(got, key=lambda x: (x[1], -x[2])):
      end = start + seconds
      for ends in lanes:
        while ends and ends[-1] <= start:
          ends.pop()
      lane = next((i for i, ends in enumerate(lanes) if not ends or end <= ends[-1]), len(lanes))
      if lane == len(lanes):
        lanes.append([])
      lanes[lane].append(end)
      span(w, name, start, end, lane=lane)
  for (w, lane), n in tids.items():
    rs = [runs[r] for r in worker_runs[w] if r in runs]
    r = rs[0] if rs else {}
    label = (
      w
      if w.startswith("node ")
      else f"{w} {r.get('role', '?')} {r.get('base', '').split('/')[-1]} {'lora' if r.get('rank') else 'full'} x{len(rs)}"
    )
    out.append(
      {
        "ph": "M",
        "pid": 1,
        "tid": n,
        "name": "thread_name",
        "args": {"name": label + (f" (overlap {lane})" if lane else "")},
      }
    )
  out.append({"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "tinyrl event log"}})
  return out


def profiles(folder: str, t0: float, since: float, until: float) -> list:
  """Every torch profiler trace in folder, moved onto the log's clock, each file under pids of its own."""
  out = []
  files = sorted(glob.glob(f"{folder}/**/*.json*", recursive=True))
  for i, path in enumerate(files):
    with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
      trace = json.load(f)
    shift = trace.get("baseTimeNanoseconds", 0) / 1e3 - t0 * 1e6  # an event's wall clock is base + ts
    pids = {}
    name = os.path.basename(path).split(".")[0]
    for e in trace.get("traceEvents", []):
      if "pid" not in e:
        continue
      if "ts" in e:
        e["ts"] = float(e["ts"]) + shift
        if not since * 1e6 - t0 * 1e6 <= e["ts"] <= until * 1e6 - t0 * 1e6:
          continue
      e["pid"] = pids.setdefault(e["pid"], 1000 * (i + 1) + len(pids))
      if e.get("ph") == "M" and e.get("name") == "process_name":
        e["args"] = {"name": f"{name} {e.get('args', {}).get('name', '')}"}
      out.append(e)
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--since", type=float, default=0.0)
  ap.add_argument("--until", type=float, default=float("inf"))
  ap.add_argument("--profile", default="")
  args = ap.parse_args()
  events = asyncio.run(read_events(args.since, args.until))
  t0 = args.since or (events[0].t if events else 0.0)
  trace = system_view(events, t0) + (profiles(args.profile, t0, args.since or t0, args.until) if args.profile else [])
  with (gzip.open if args.out.endswith(".gz") else open)(args.out, "wt") as f:
    json.dump({"traceEvents": trace, "displayTimeUnit": "ms"}, f)
  print(f"{len(trace)} trace events from {len(events)} log entries -> {args.out}")


if __name__ == "__main__":
  main()
