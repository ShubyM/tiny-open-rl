"""Trajectories as files a person or an agent can read.

  <root>/<run>/run.json                     name, model, facts, status
  <root>/<run>/steps.jsonl                  one line per step: {"step": 0, "metrics": {any numbers}}
  <root>/<run>/telemetry.jsonl              numbers over time: {"t": seconds, "name": ..., "labels": {...}, "value": ...}
  <root>/<run>/<step>/<group>/<n>.jsonl     an episode, one event per line
  <root>/<run>/<step>/<group>/<n>/files/    what it left behind

An episode's first line is {"type": "episode", "metrics": {any numbers}, "tags": {any strings}}. Then its messages in order,
{"type": "message", "role": "system" | "user" | "assistant" | "tool", "text": ...}; an assistant's may carry thinking, calls,
tokens and seconds, a tool's a name and error. A {"type": "list", "name": ..., "rows": [...]} is a table the client logged.

  traj.py export out/full4/step-0.pkl <root> <run> [name] [fact]...   a step lab_rl.py saved, into a run
  traj.py ls <root>/<run> [metric]                                    a run as text: a row per task, a number per episode
  traj.py show <episode.jsonl> [turn]                                 an episode in about a page, or one turn in full
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path


def events(path) -> list[dict]:
  return [json.loads(line) for line in open(path)]


def short(text: str, n: int) -> str:
  text = " ".join(text.split())
  return text if len(text) <= n else text[: n - 1] + "…"


# ***** export from lab_rl.py's saved step *****


def export(pkl: str, root: str, run: str, name: str = "", *facts: str) -> None:
  import pickle

  import lab_rl as h

  reply = re.compile(r'response:(\w+)\{value:<\|"\|>(.*?)<\|"\|>\}<tool_response\|>', re.S)
  qwen_reply = re.compile(r"<tool_response>\n(.*?)\n</tool_response>", re.S)  # which does not name its tool
  folder, log = Path(root, run), Path(pkl).parent / "log.jsonl"
  kind, number = Path(pkl).stem.split("-")  # step-3.pkl, or eval-5.pkl: the held-out tasks as they went before step 5
  logged = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
  step = next((line for line in reversed(logged) if line.get(kind) == int(number)), {}) | {"step": int(number)}
  folder.mkdir(parents=True, exist_ok=True)
  status = os.environ.get("TRAJ_STATUS", "Completed")
  (folder / "run.json").write_text(
    json.dumps({"name": name or run, "model": h.MODEL, "facts": list(facts), "status": status})
  )
  numbers = {k: v for k, v in step.items() if isinstance(v, (int, float)) and k not in ("step", "eval")}
  steps = {s["step"]: s for s in events(folder / "steps.jsonl")} if (folder / "steps.jsonl").exists() else {}
  steps[step["step"]] = {"step": step["step"], "metrics": numbers}
  (folder / "steps.jsonl").write_text("".join(json.dumps(steps[k]) + "\n" for k in sorted(steps)))
  groups: dict[str, list[dict]] = {}
  for ep in pickle.load(open(pkl, "rb")):
    groups.setdefault(ep["task"], []).append(ep)
  for task, episodes in groups.items():
    for n, ep in enumerate(sorted(episodes, key=lambda ep: -ep["reward"]), 1):
      tokens, sampled, results = ep["tokens"], ep["sampled"], h.LAB / "results" / ep["run_id"]
      scores = json.loads(scores.read_text()) if (scores := results / "scores.json").exists() else {}
      metrics = {"reward": ep["reward"], "turns": ep["turns"], "tokens": len(tokens), "seconds": ep["rollout_s"]}
      metrics |= {"sampled_tokens": sum(len(t) for _, t, _ in sampled), "grade_seconds": ep["grade_s"]}
      metrics |= {"sample_seconds": ep["sample_s"]} if "sample_s" in ep else {}
      metrics |= {"criteria_passed": scores["n_passed"]} if scores.get("n_criteria") else {}
      lines = [{"type": "episode", "group": task, "n": n, "metrics": metrics, "tags": {"end": ep["end"]}}]
      prompt = h.tok.decode(tokens[: sampled[0][0]] if sampled else tokens, skip_special_tokens=False)
      if h.QWEN:
        system = prompt.split("<|im_start|>user\n")[0].removeprefix("<|im_start|>system\n").removesuffix("<|im_end|>\n")
      else:
        system = prompt.split("<|turn>user\n")[0].removeprefix("<bos><|turn>system\n").removesuffix("<turn|>\n")
      lines.append({"type": "message", "role": "system", "text": system, "tokens": len(h.tok.encode(system))})
      lines.append({"type": "message", "role": "user", "text": h.load_task(task)["instructions"]})
      for i, (at, turn, _) in enumerate(sampled):
        text = h.tok.decode(turn, skip_special_tokens=False)
        try:
          said = h.parse(text)
          message = {
            "thinking": said.get("thinking") or said.get("reasoning_content", ""),
            "text": said.get("content", ""),
          }
          message["calls"] = [c["function"] for c in said.get("tool_calls") or []]
        except ValueError:
          message = {"text": text, "error": "this turn's tool call could not be parsed, which ended the episode"}
        lines.append({"type": "message", "role": "assistant", "turn": i + 1, "tokens": len(turn), **message})
        after = tokens[at + len(turn) : sampled[i + 1][0] if i + 1 < len(sampled) else len(tokens)]
        after = h.tok.decode(after, skip_special_tokens=False)
        named = (
          zip([c["name"] for c in message.get("calls", [])], qwen_reply.findall(after))
          if h.QWEN
          else reply.findall(after)
        )
        for tool, result in named:
          failed = result.startswith("Error")  # how LAB's tools report one
          lines.append(
            {"type": "message", "role": "tool", "turn": i + 1, "name": tool, "text": result, "error": failed}
          )
      if scores.get("criteria_results"):
        lines.append({"type": "list", "name": "criteria", "rows": scores["criteria_results"]})
      path = folder / str(step["step"]) / task.replace("/", "__")
      path.mkdir(parents=True, exist_ok=True)
      (path / f"{n}.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
      if (results / "output").is_dir() and any((results / "output").iterdir()):
        shutil.copytree(results / "output", path / str(n) / "files" / "output", dirs_exist_ok=True)
  print(f"{sum(map(len, groups.values()))} episodes of step {step['step']} in {folder}")


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
    case ["export", pkl, root, run, *rest]:
      export(pkl, root, run, *rest)
    case ["ls", run, *metric]:
      print(ls(run, *metric))
    case ["show", path]:
      print(show(path))
    case ["show", path, turn]:
      print(show(path, int(turn)))
    case _:
      raise SystemExit(__doc__)
