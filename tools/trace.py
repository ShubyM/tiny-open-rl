"""One timeline of a tinyrl cluster, for the dashboard's Trace tab or ui.perfetto.dev: the event log's system view (each worker's process, GPU turns, waits, batches and the commands in them, placements, revokes) and, with --profile, every TINYRL_PROFILE trace of trainer and vLLM ops and CUDA kernels on the same clock.

usage: REDIS_URL=redis://... python tools/trace.py out.json.gz [--since unix_s] [--until unix_s] [--profile dir [--raw]]
"""

import argparse
import asyncio
import bisect
import collections
import concurrent.futures
import glob
import gzip
import json
import os
import sys

from tinyrl.bus import Bus, connect

try:
  from orjson import loads  # 3-5x faster on 100 MB vLLM traces
except ImportError:
  from json import loads


async def read_events(since: float, until: float) -> list:
  bus, out, after = Bus(connect(os.environ.get("REDIS_URL", "redis://localhost:6379"))), [], f"{int(since * 1000)}-0"
  while got := await bus.read_events(after, count=10000):
    out += [e for e in got if e.t <= until]
    if got[-1].t > until:
      break
    after = got[-1].id
  return out


def system_view(events: list, t0: float) -> list:
  """Chrome trace events for the log: pid 1, a thread per worker (named by its role, base model and runs) and per node daemon. Spans that overlap without nesting, like a sampler's concurrent requests, are async slices on their worker's thread."""
  us = lambda t: (t - t0) * 1e6  # noqa: E731
  runs, worker_runs, tids, out = {}, collections.defaultdict(set), {}, []
  opened = collections.defaultdict(dict)  # worker -> name of a span the log opens -> its start
  reported = collections.defaultdict(list)  # worker, or "node <id>" -> [name, start, seconds] its process reported

  def tid(w: str) -> int:
    return tids.setdefault(w, len(tids) + 1)

  def span(w: str, name: str, start: float, end: float, args: dict | None = None) -> None:
    dur = max(0.0, (end - start) * 1e6)
    out.append({"ph": "X", "pid": 1, "tid": tid(w), "name": name, "ts": us(start), "dur": dur, "args": args or {}})

  def overlapping(w: str, name: str, start: float, end: float, args: dict | None = None) -> None:
    e = {"cat": "overlap", "id": len(out), "pid": 1, "tid": tid(w), "name": name, "args": args or {}}
    out.extend([{**e, "ph": "b", "ts": us(start)}, {**e, "ph": "e", "ts": us(end)}])

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
        start, ops = e.t - b.get("seconds", 0.0), b.get("ops", [])  # ops: logs from before spans
        span(w, "batch", start, e.t, {"queued": b.get("queued")})
        concurrent = (
          sum(ms for _, ms in ops) > b.get("seconds", 0.0) * 1010
        )  # a sampler's ran at once, a trainer's in turn
        for op, ms in ops:
          (overlapping if concurrent else span)(
            w, op, start, start + ms / 1000, {"approx": True} if concurrent else None
          )
          start += 0 if concurrent else ms / 1000
      case "placed" | "unplaced" | "revoked" | "idle":
        args = {k: v for k, v in b.items() if k != "spans"}
        out.append({"ph": "i", "s": "t", "pid": 1, "tid": tid(w), "name": e.kind, "ts": us(e.t), "args": args})
  for w, got in reported.items():
    stack: list[float] = []  # the ends of the spans open around the next one
    for name, start, seconds in sorted(got, key=lambda x: (x[1], -x[2])):
      end = start + seconds
      while stack and stack[-1] <= start:
        stack.pop()
      if stack and end > stack[-1]:
        overlapping(w, name, start, end)
      else:
        stack.append(end)
        span(w, name, start, end)
  for w, n in tids.items():
    rs = [runs[r] for r in worker_runs[w] if r in runs]
    r = rs[0] if rs else {}
    kind = "lora" if r.get("rank") else "full"
    label = (
      w if w.startswith("node ") else f"{w} {r.get('role', '?')} {r.get('base', '').split('/')[-1]} {kind} x{len(rs)}"
    )
    out.append({"ph": "M", "pid": 1, "tid": n, "name": "thread_name", "args": {"name": label}})
  out.append({"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "tinyrl event log"}})
  return out


def worker_of(path: str) -> str:
  """The worker a TINYRL_PROFILE trace is from: trainer-<worker>-<pid>.json.gz, or a file under sampler-<worker>/."""
  for part in reversed(path.split("/")):
    if part.startswith(("trainer-", "sampler-")):
      return part.split("-")[1].split(".")[0]
  return ""


def union(intervals: list) -> float:
  total, end = 0.0, -1e30
  for s, e in sorted(intervals):
    if e > end:
      total, end = total + e - max(s, end), e
  return total


def summarize(path: str, t0: float, commands: list, raw: bool, bin_s: float) -> tuple[list, list, list]:
  """One trace's GPU-busy counter bins, what the GPU did during each of commands, and (raw) all its events, on the log's clock."""
  try:
    with (gzip.open if path.endswith(".gz") else open)(path, "rb") as f:
      trace = loads(f.read())
  except (EOFError, OSError, ValueError) as e:  # cut off: its process was killed while writing it
    print(f"skipping {path}: {e}", file=sys.stderr)
    return [], [], []
  shift = trace.get("baseTimeNanoseconds", 0) / 1e3 - t0 * 1e6  # an event's wall clock is base + ts
  events = trace.get("traceEvents", [])
  kernels = sorted(
    (e["ts"] + shift, e["ts"] + shift + e.get("dur", 0), e["name"]) for e in events if e.get("cat") == "kernel"
  )
  copies = [
    (e["ts"] + shift, e.get("dur", 0), e["name"]) for e in events if e.get("cat") in ("gpu_memcpy", "gpu_memset")
  ]
  counter, notes, width = [], [], bin_s * 1e6
  if kernels:
    bins = collections.defaultdict(list)
    for s, e, _ in kernels:  # a kernel goes in every bin it touches, clipped to it
      for b in range(int(s // width), int(e // width) + 1):
        bins[b].append((max(s, b * width), min(e, (b + 1) * width)))
    counter = [(b * width, round(union(bins.get(b, [])) / width, 3)) for b in range(min(bins), max(bins) + 2)]
    starts = [k[0] for k in kernels]
    for lo, hi in commands:
      window = kernels[bisect.bisect_left(starts, lo - 1e6) : bisect.bisect_left(starts, hi)]  # no kernel runs a second
      inside = [(max(s, lo), min(e, hi), n) for s, e, n in window if e > lo and s < hi]
      top, moved = collections.Counter(), collections.Counter()
      for s, e, n in inside:
        top[n[:60]] += e - s
      for s, d, n in copies:
        if lo <= s < hi:
          moved[n.split(" (")[0]] += d
      busy = f"{100 * union([(s, e) for s, e, _ in inside]) / max(1.0, hi - lo):.0f}%"
      notes.append(
        {
          "gpu_busy": busy,
          "copies": {k: f"{v / 1e3:.0f} ms" for k, v in moved.items()},
          "top_kernels": {k: f"{v / 1e3:.0f} ms" for k, v in top.most_common(3)},
        }
      )
  rest = []
  if raw:
    for e in events:
      if "ts" in e:
        e["ts"] = float(e["ts"]) + shift
      rest.append(e)
  return counter, notes, rest


def profiles(folder: str, t0: float, view: list, raw: bool, bin_s: float = 0.25) -> list:
  """What every TINYRL_PROFILE trace in folder adds, on the log's clock: per worker a "gpu busy" counter (the share of each bin its kernels kept the GPU busy), and on each of its commands in view, its GPU-busy share, copies and top kernels. raw adds every event of every trace, for ui.perfetto.dev. Traces are read in parallel."""
  tid_of = {e["args"]["name"].split(" ")[0]: e["tid"] for e in view if e.get("name") == "thread_name"}
  commands = collections.defaultdict(list)  # tid -> its command spans in view
  for e in view:
    if e.get("ph") == "X" and e["name"] not in ("process", "turn", "wait for turn", "batch", "stopping"):
      commands[e["tid"]].append(e)
  paths = [p for p in sorted(glob.glob(f"{folder}/**/*.json*", recursive=True)) if raw or tid_of.get(worker_of(p))]
  out = []
  with concurrent.futures.ProcessPoolExecutor(min(16, len(paths) or 1)) as pool:
    jobs = [(p, tid_of.get(worker_of(p))) for p in paths]
    spans = [[(c["ts"], c["ts"] + c["dur"]) for c in commands[tid]] if tid else [] for _, tid in jobs]
    for i, ((path, tid), (counter, notes, rest)) in enumerate(
      zip(jobs, pool.map(summarize, paths, [t0] * len(paths), spans, [raw] * len(paths), [bin_s] * len(paths)))
    ):
      for ts, busy in counter:
        out.append({"ph": "C", "pid": 1, "tid": tid, "name": "gpu busy", "ts": ts, "args": {"busy": busy}})
      for c, note in zip(commands[tid] if tid else [], notes):
        c["args"].update(note)
      pids = {}
      for e in rest:
        if "pid" in e:
          e["pid"] = pids.setdefault(e["pid"], 1000 * (i + 1) + len(pids))
          if e.get("ph") == "M" and e.get("name") == "process_name":
            e["args"] = {"name": f"{os.path.basename(path).split('.')[0]} {e.get('args', {}).get('name', '')}"}
          out.append(e)
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--since", type=float, default=0.0)
  ap.add_argument("--until", type=float, default=float("inf"))
  ap.add_argument(
    "--profile", default="", help="a TINYRL_PROFILE folder: adds GPU-busy counters and per-command kernel summaries"
  )
  ap.add_argument("--raw", action="store_true", help="with --profile, also every kernel and op, for ui.perfetto.dev")
  args = ap.parse_args()
  events = asyncio.run(read_events(args.since, args.until))
  t0 = args.since or (events[0].t if events else 0.0)
  trace = system_view(events, t0)
  trace += profiles(args.profile, t0, trace, args.raw) if args.profile else []
  with (gzip.open if args.out.endswith(".gz") else open)(args.out, "wt") as f:
    json.dump({"traceEvents": trace, "displayTimeUnit": "ms"}, f)
  print(f"{len(trace)} trace events from {len(events)} log entries -> {args.out}")


if __name__ == "__main__":
  main()
