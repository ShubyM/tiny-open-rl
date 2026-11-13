"""Trajectories as files a person or an agent can read.

  <root>/<run>/run.json                     name, model, facts, status
  <root>/<run>/steps.jsonl                  one line per step: {"step": 0, "metrics": {any numbers}}
  <root>/<run>/telemetry.jsonl              numbers over time: {"t": seconds, "name": ..., "labels": {...}, "value": ...}
  <root>/<run>/<step>/<group>/<n>.jsonl     an episode, one event per line
  <root>/<run>/<step>/<group>/<n>/files/    what it left behind

An episode's first line is {"type": "episode", "metrics": {any numbers}, "tags": {any strings}}. Then its messages in order,
{"type": "message", "role": "system" | "user" | "assistant" | "tool", "text": ...}; an assistant's may carry thinking, calls,
tokens and seconds, a tool's a name and error. A {"type": "list", "name": ..., "rows": [...]} is a table the client logged.

  traj.py ls <root>/<run> [metric]                                    a run as text: a row per task, a number per episode
  traj.py show <episode.jsonl> [turn]                                 an episode in about a page, or one turn in full
"""

import json
import os
import re
import sys
from pathlib import Path


def events(path) -> list[dict]:
  return [json.loads(line) for line in open(path)]


def short(text: str, n: int) -> str:
  text = " ".join(text.split())
  return text if len(text) <= n else text[: n - 1] + "…"


# ***** export from lab_rl.py's saved step *****


# ***** read a root *****


def telemetry(path: Path, buckets: int = 240) -> list[dict]:
  """A run's numbers over time as charts of lines, a point per bucket of time. A gauge is its mean over the bucket. A counter,
  which is a name ending in _total, is its rise per second, summed over the runs that came and went under one worker."""
  rows = events(path)
  if not rows:
    return []
  start = min(r["t"] for r in rows)
  width = max(30.0, (max(r["t"] for r in rows) - start) / buckets)
  counter = re.compile(r"_total$|^lab_(sampled_tokens|turns|episodes_done)$")
  gauges, rises, last = {}, {}, {}
  for r in sorted(rows, key=lambda r: r["t"]):
    labels, bucket = tuple(sorted(r["labels"].items())), int((r["t"] - start) // width)
    if not counter.search(r["name"]):
      gauges.setdefault((r["name"], labels), {}).setdefault(bucket, []).append(r["value"])
      continue
    name = counter.sub(lambda m: "" if m[0] == "_total" else m[0], r["name"]) + " per second"
    line = rises.setdefault((name, tuple(kv for kv in labels if kv[0] != "run")), {})
    line[bucket] = line.get(bucket, 0.0) + max(0.0, r["value"] - last.get((r["name"], labels), r["value"]))
    last[r["name"], labels] = r["value"]
  lines = {key: {b: sum(v) / len(v) for b, v in by.items()} for key, by in gauges.items()}
  lines |= {key: {b: v / width for b, v in by.items()} for key, by in rises.items()}
  charts: dict[tuple, list] = {}
  for (name, labels), by in lines.items():
    same = [k for n, k in lines if n == name]
    # many lines under one name are split by the label with the fewest values, as sixteen GPUs are by their two hosts
    counts = {k: len({dict(ls).get(k) for ls in same}) for k, _ in labels}
    by_label = min((k for k in counts if counts[k] > 1), key=counts.get, default=None) if len(same) > 8 else None
    # a host's name is long and mostly shared, so only where it differs is shown
    values = sorted({dict(ls).get(by_label, "") for ls in same}) if by_label else [""]
    title = name + (f" · {dict(labels)[by_label][len(os.path.commonprefix(values)) :]}" if by_label else "")
    label = " ".join(f"{k} {v}" for k, v in labels if k != by_label and counts[k] > 1)
    points = [[round(b * width / 60, 1), float(f"{v:.4g}")] for b, v in sorted(by.items())]
    charts.setdefault(title, []).append({"label": label, "points": points})
  return [{"name": name, "lines": sorted(ls, key=lambda line: line["label"])} for name, ls in sorted(charts.items())]


def load(root: str) -> dict:
  """Every run under a root, as the page takes it."""
  runs = []
  for run in sorted(p.parent for p in Path(root).glob("*/run.json")):
    steps = {s["step"]: s["metrics"] for s in events(run / "steps.jsonl")} if (run / "steps.jsonl").exists() else {}
    episodes = []
    for path in sorted(run.glob("*/*/*.jsonl"), key=lambda p: (int(p.parts[-3]), p.parts[-2], int(p.stem))):
      lines = events(path)
      files = sorted(f for f in (path.parent / path.stem / "files").rglob("*") if f.is_file())
      episode = {
        **lines[0],
        "step": int(path.parts[-3]),
        "path": path.relative_to(root).as_posix(),
        "events": lines[1:],
      }
      episode["files"] = [
        {"path": f.relative_to(path.parent / path.stem / "files").as_posix(), "size": f.stat().st_size} for f in files
      ]
      for f, entry in zip(files, episode["files"]):
        if f.suffix in {".md", ".txt", ".json", ".csv", ".py", ".jsonl"} and f.stat().st_size < 200_000:
          entry["text"] = f.read_text(errors="replace")
      episodes.append(episode)
    seen = telemetry(run / "telemetry.jsonl") if (run / "telemetry.jsonl").exists() else []
    about = json.loads((run / "run.json").read_text())
    runs.append({"id": run.name, **about, "steps": steps, "telemetry": seen, "episodes": episodes})
  return {"runs": runs}


def ls(run: str, metric: str = "reward") -> str:
  root, name = Path(run).parent, Path(run).name
  data = next(r for r in load(str(root))["runs"] if r["id"] == name)
  out = [f"# {data['name']}", " · ".join([data["model"], *data["facts"], data["status"]]), ""]
  for step, numbers in data["steps"].items():
    out.append(f"step {step}: " + ", ".join(f"{k} {v:.4g}" for k, v in numbers.items()))
  names = sorted({k for e in data["episodes"] for k in e["metrics"]})
  out += [
    "",
    f"{metric} of each episode, a row per task (other metrics: {', '.join(n for n in names if n != metric)})",
    "",
  ]
  groups: dict[tuple, list[dict]] = {}
  for e in data["episodes"]:
    groups.setdefault((e["step"], e["group"]), []).append(e)
  for (step, group), episodes in groups.items():
    values = "  ".join(f"{e['metrics'].get(metric, float('nan')):.4g}" for e in episodes)
    out.append(f"step {step}  {short(group, 70):70}  {values}")
  out += ["", f"an episode: traj.py show {root}/<run>/<step>/<task with / as __>/<n>.jsonl"]
  return "\n".join(out)


def call_text(call: dict) -> str:
  return f"{call['name']}({', '.join(f'{k}={short(str(v), 60)!r}' for k, v in call['arguments'].items())})"


def show(path: str, turn: int | None = None) -> str:
  lines = events(path)
  head, messages = lines[0], [e for e in lines if e["type"] == "message"]
  if turn:
    out = []
    for m in (m for m in messages if m.get("turn") == turn):
      label = (
        f"tool {m['name']}" + (" (error)" if m.get("error") else "")
        if m["role"] == "tool"
        else f"assistant, turn {turn}"
      )
      body = [f"thinking: {m['thinking']}"] if m.get("thinking") else []
      body += [m["text"]] if m.get("text") else []
      body += [json.dumps(c, indent=1) for c in m.get("calls", [])]
      out.append(f"## {label}\n" + "\n\n".join(body))
    return "\n\n".join(out)
  out = [f"# {head['group']} · episode {head['n']}", ""]
  out.append(
    ", ".join(f"{k} {v:.4g}" for k, v in head["metrics"].items())
    + "".join(f", {k}: {v}" for k, v in head["tags"].items())
  )
  out += ["", "user: " + short(next(m["text"] for m in messages if m["role"] == "user"), 400), ""]
  for m in (m for m in messages if m["role"] == "assistant"):
    replies = [r for r in messages if r["role"] == "tool" and r["turn"] == m["turn"]]
    did = "; ".join(map(call_text, m.get("calls", []))) or "says " + repr(short(m.get("text", ""), 110))
    back = "; ".join(("ERROR " if r.get("error") else "") + repr(short(r["text"], 70)) for r in replies)
    out.append(
      f"{m['turn']:>2}. {did}" + (f"  ->  {back}" if back else "") + (f"  [{m['error']}]" if m.get("error") else "")
    )
  files = sorted(
    f.relative_to(Path(path).with_suffix("") / "files")
    for f in (Path(path).with_suffix("") / "files").rglob("*")
    if f.is_file()
  )
  out += ["", "files: " + (", ".join(map(str, files)) or "none")]
  out += [f"{e['name']}: {len(e['rows'])} rows, in the file" for e in lines if e["type"] == "list"]
  out += ["", f"a turn in full: traj.py show {path} <turn>"]
  return "\n".join(out)


if __name__ == "__main__":
  match sys.argv[1:]:
    case ["ls", run, *metric]:
      print(ls(run, *metric))
    case ["show", path]:
      print(show(path))
    case ["show", path, turn]:
      print(show(path, int(turn)))
    case _:
      raise SystemExit(__doc__)
