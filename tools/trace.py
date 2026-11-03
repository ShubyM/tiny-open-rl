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
  """Chrome trace events for the log: pid 1, a thread per worker, named by its role, base model and runs."""
  us = lambda t: (t - t0) * 1e6  # noqa: E731
  runs, worker_runs, tids, out = {}, collections.defaultdict(set), {}, []
  open_spans = collections.defaultdict(dict)  # worker -> span name -> start

  def tid(w: str) -> int:
    return tids.setdefault(w, len(tids) + 1)

  def span(w: str, name: str, start: float, end: float, args: dict | None = None) -> None:
    out.append(
      {
        "ph": "X",
        "pid": 1,
        "tid": tid(w),
        "name": name,
        "ts": us(start),
        "dur": max(0.0, (end - start) * 1e6),
        "args": args or {},
      }
    )

  for e in events:
    b, w = e.body, e.body.get("worker")
    if e.kind == "run_created":
      runs[b["run"]] = b
    if e.kind == "assigned" and b.get("run") in runs:
      worker_runs[w].add(b["run"])
    if not w:
      continue
    spans = open_spans[w]
    match e.kind:
      case "started":
        spans["process"] = e.t
      case "exited" if "process" in spans:
        span(w, "process", spans.pop("process"), e.t)
      case "wants_gpu" if "wait" not in spans and "turn" not in spans:
        spans["wait"] = e.t
      case "granted":
        if "wait" in spans:
          span(w, "wait for turn", spans.pop("wait"), e.t)
        spans["turn"] = e.t
      case "released" if "turn" in spans:
        span(w, "turn", spans.pop("turn"), e.t)
      case "batch_done":
        start = e.t - b.get("seconds", 0.0)
        span(w, "batch", start, e.t, {"queued": b.get("queued")})
        at = start
        for op, ms in b.get("ops", []):  # in the order they ran, back to back
          span(w, op, at, at + ms / 1000)
          at += ms / 1000
      case "placed" | "unplaced" | "revoked" | "idle":
        out.append({"ph": "i", "s": "t", "pid": 1, "tid": tid(w), "name": e.kind, "ts": us(e.t), "args": b})
  for w, n in tids.items():
    rs = [runs[r] for r in worker_runs[w] if r in runs]
    r = rs[0] if rs else {}
    label = (
      f"{w} {r.get('role', '?')} {r.get('base', '').split('/')[-1]} {'lora' if r.get('rank') else 'full'} x{len(rs)}"
    )
    out.append({"ph": "M", "pid": 1, "tid": n, "name": "thread_name", "args": {"name": label}})
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
