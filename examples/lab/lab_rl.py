"""LAB legal-agent RL on tinyrl. LAB supplies the tasks, the tools and the grader; Gemma 4 or a Qwen 3.5 family model runs through
the Tinker SDK. An episode's tools run in an Agent Substrate environment (SUBSTRATE_ENDPOINT), which is suspended while the model
generates, in a sandbox from an Agent Sandbox warm pool (SANDBOX_WARMPOOL), or in LAB's own podman container.

One step samples GROUP episodes for each of BATCH tasks, grades them, and trains on each episode's sampled tokens with
advantage = reward - its task's mean reward. Every step's episodes are saved, so replay.py can train on them again.
Every EVAL_EVERY steps, and after the last, the held-out tasks are run once each without training. The run saves its state
after each step and starts from it again, so a run that died is rerun with the same command.

  LAB_ROOT=/path/to/harvey-labs OPENAI_BASE_URL=http://judge/v1 OPENAI_API_KEY=unused python lab_rl.py

The task split, the system prompt and the Gemma 4 turn handling follow open-rl's examples/harvey_labs, whose sandbox clients
this uses. See README.md.
"""

import asyncio
import functools
import json
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tinker
from huggingface_hub import hf_hub_download
from tinker import types
from transformers import AutoTokenizer
from transformers.utils.chat_parsing import parse_response

LAB = Path(os.environ["LAB_ROOT"]).resolve()
sys.path.insert(0, str(LAB))
from harness.tools import ToolExecutor, get_all_tool_definitions  # noqa: E402


def env(name: str, default):
  return type(default)(os.environ.get(name, default))


MODEL = env("MODEL", "google/gemma-4-E4B-it")
BASE_URL = env("BASE_URL", "http://localhost:18000")
OUT = Path(env("OUT", "out")).resolve()
GPUS = env("GPUS", 1)  # the trainer's width
RANK = env("RANK", 32)
LR = env("LR", 2e-4)
STEPS = env("STEPS", 1)
EVAL_EVERY = env("EVAL_EVERY", 0)  # run the held-out tasks before every so many steps and after the last; 0 for never
EVAL_TASKS = env("EVAL_TASKS", 50)
SAMPLERS = env("SAMPLERS", 1)  # sampler workers, each on a GPU of its own, that a step's episodes are dealt over
AT_ONCE = env(
  "AT_ONCE", 0
)  # episodes rolled out at once, 0 for all. Past what the sampler's cache holds, they evict each other's context
BATCH = env("BATCH", 8)  # tasks per step
GROUP = env("GROUP", 6)  # episodes per task
TASK = env("TASK", "")  # train on this one task instead of the split
MAX_TURNS = env("MAX_TURNS", 40)
TURN_TOKENS = env("TURN_TOKENS", 8192)
TRAJ_TOKENS = env("TRAJ_TOKENS", 32768)
TOOL_TOKENS = env("TOOL_TOKENS", 4096)
THINK = env("THINK", 1)
JUDGE = env("JUDGE", "gpt-gemma-4-31b-it")
JUDGE_PARALLEL = env("JUDGE_PARALLEL", 16)
GRADING = env("GRADING", 6)  # episodes graded at once
WARMPOOL = env("SANDBOX_WARMPOOL", "")  # the Agent Sandbox warm pool to claim sandboxes from; local podman without one
SANDBOX_NAMESPACE = env("SANDBOX_NAMESPACE", "lab-sandboxes")
SUBSTRATE = env(
  "SUBSTRATE_ENDPOINT", ""
)  # ate-env-api of an Agent Substrate cluster, whose sandboxes suspend while the model generates
SUBSTRATE_IMAGE = env("SUBSTRATE_IMAGE", "")  # LAB's sandbox image, pinned by digest
SUBSTRATE_IDLE = env("SUBSTRATE_IDLE", 30.0)  # seconds without a tool call before a sandbox is suspended
SCRAPE = [u for u in env("SCRAPE", "").split(",") if u]  # Prometheus endpoints to record beside the gateway's own
SCRAPE_EVERY = env("SCRAPE_EVERY", 30.0)
SCRAPE_KEEP = re.compile(env("SCRAPE_KEEP", r"tinyrl_|DCGM_FI_DEV_(GPU_UTIL|FB_USED|POWER_USAGE|GPU_TEMP)$"))
EXPORT = env("EXPORT", "")  # "<run> <name> <fact>;<fact>": write each step as trajectory files under OUT/trajectories
LIVE = {"episodes_running": 0, "episodes_grading": 0, "episodes_done": 0, "sampled_tokens": 0, "turns": 0}

tok = AutoTokenizer.from_pretrained(MODEL)
RESPONSE = json.loads(Path(hf_hub_download(MODEL, "tokenizer_config.json")).read_text())["response_template"]
QWEN = "<|im_end|>" in tok.get_vocab()  # Qwen's chat format; Gemma 4's otherwise
STOP = [tok.convert_tokens_to_ids(t) for t in (("<|im_end|>",) if QWEN else ("<turn|>", "<|tool_response>"))]
TOOLS = [{"type": "function", "function": t} for t in get_all_tool_definitions()]
SKILLS = sorted(p.parent.name for p in (LAB / "harness" / "skills").glob("*/SKILL.md"))


# ***** tasks *****


def load_task(name: str) -> dict | None:
  """A task that can run and be graded: instructions, criteria and documents."""
  folder = LAB / "tasks" / name
  config = json.loads((folder / "task.json").read_text())
  documents = folder / "documents"
  if not (config.get("instructions") and config.get("criteria") and documents.is_dir() and any(documents.rglob("*"))):
    return None
  return {"name": name, "instructions": config["instructions"], "documents": documents}


def split() -> tuple[list[dict], list[dict]]:
  """open-rl's split: 50 held-out tasks from a seeded shuffle, and for training the rest that share no family with them."""
  names = sorted(p.parent.relative_to(LAB / "tasks").as_posix() for p in (LAB / "tasks").rglob("task.json"))
  tasks = {n: t for n in names if (t := load_task(n))}
  shuffled = list(tasks)
  random.Random(0).shuffle(shuffled)
  held_out = shuffled[300:350]
  family = lambda n: re.sub(r"/scenario-\d+$", "", n)  # noqa: E731
  train = [n for n in tasks if family(n) not in {family(h) for h in held_out}]
  random.Random(242).shuffle(train)
  return [tasks[n] for n in train], [tasks[n] for n in held_out]


# ***** prompt *****


def render(messages: list[dict], **kw) -> list[int]:
  return tok.apply_chat_template(
    messages, tokenize=True, return_dict=False, enable_thinking=bool(THINK), preserve_thinking=True, **kw
  )


# what the template writes to open a model turn, which a sampled turn's text follows
OPENING = tok.decode(
  render([{"role": "user", "content": "-"}], add_generation_prompt=True)[
    len(render([{"role": "user", "content": "-"}])) :
  ]
)


def parse(text: str) -> dict:
  """A sampled turn as a message with its thinking, text and tool calls."""
  # after tool replies Gemma's template has already opened the thinking channel
  opened = "<channel|>" in text and not text.startswith("<|channel>")
  return parse_response(text, RESPONSE, prefix=OPENING + ("<|channel>thought\n" if opened else ""))


def system_prompt(task: dict) -> str:
  prompt = (LAB / "harness" / "system_prompt.md").read_text()
  for skill in SKILLS:
    prompt += f"\n\n## Skill: {skill}\n\n{(LAB / 'harness' / 'skills' / skill / 'SKILL.md').read_text()}"
  outputs = list(dict.fromkeys(re.findall(r"`([^`]+\.(?:docx|xlsx|pptx|pdf|md|txt))`", task["instructions"], re.I)))
  if not outputs:
    return prompt
  # the write tool and bash see the output folder under different paths, which the model otherwise gets wrong
  prompt += (
    "\n\n## Required output-path contract\n\n"
    f"The required deliverables are {', '.join(f'`/workspace/output/{o}`' for o in outputs)}. "
    "For the `write` tool, use a bare relative `file_path` without an `output/` prefix; the tool already stores it under "
    "`/workspace/output`. For `bash` and skill scripts, use absolute `/workspace/output/...` paths. Create every command "
    "input before using it, and confirm and validate all requested deliverables before stopping."
  )
  if docx := [o for o in outputs if o.lower().endswith(".docx")]:
    prompt += (
      "\nFor a new DOCX authored from Markdown, call `write` with a bare name such as `draft.md`, then run "
      f"`python skills/docx/scripts/generate_from_md.py /workspace/output/draft.md /workspace/output/{docx[0]}` and "
      f"validate it with `python skills/docx/scripts/validate.py /workspace/output/{docx[0]}`."
    )
  return prompt


def first_prompt(task: dict) -> list[int]:
  messages = [{"role": "system", "content": system_prompt(task)}, {"role": "user", "content": task["instructions"]}]
  return render(messages, tools=TOOLS, add_generation_prompt=True)


def tool_turn(calls: list[dict], results: list[str], last: int) -> list[int]:
  """The tokens that follow a sampled turn of tool calls: the tools' replies and the next turn's opening, as the template writes them. The sampled turn itself stays as sampled."""
  # a user turn first, without which Qwen's template renders nothing
  user, assistant = {"role": "user", "content": "-"}, {"role": "assistant", "content": "", "tool_calls": calls}
  replies = [{"role": "tool", "name": c["function"]["name"], "content": r} for c, r in zip(calls, results)]
  turn, after = (
    render([user, assistant], add_generation_prompt=False),
    render([user, assistant, *replies], add_generation_prompt=True),
  )
  if after[: len(turn)] != turn:
    raise ValueError("the template rewrote a tool-calling turn when its replies were added")
  # sampling stops at the stop token that ends the turn, with or without returning it
  stop = max(i for i, t in enumerate(turn) if t in STOP)
  return ([] if last == turn[stop] else [turn[stop]]) + after[stop + 1 :]


def bounded(result: str) -> str:
  tokens = tok.encode(result, add_special_tokens=False)
  if len(tokens) <= TOOL_TOKENS:
    return result
  kept = tok.decode(tokens[:TOOL_TOKENS], skip_special_tokens=False)
  kept = kept[: kept.rfind("\n") + 1] or kept
  return f"{kept.rstrip()}\n\n[Observation truncated at {TOOL_TOKENS:,} tokens. Run a narrower command, or read with an offset and a limit, to get the rest.]"


# ***** sandboxes *****


@functools.cache
def remote_sandboxes():
  if SUBSTRATE:
    from harvey_labs.substrate_sandbox import SubstrateSandboxFactory

    return SubstrateSandboxFactory(SUBSTRATE, SUBSTRATE_IMAGE, "ate-env", SUBSTRATE_IDLE)
  from harvey_labs.agent_sandbox import AgentSandboxFactory

  return AgentSandboxFactory(WARMPOOL, SANDBOX_NAMESPACE)


async def open_sandbox(task: dict, run_id: str):
  """Where an episode's tools run: a sandbox claimed from an Agent Sandbox warm pool or an Agent Substrate environment, through open-rl's clients, or LAB's own podman container. Returns how to run a tool, the tools' metrics, and how to bring the outputs home and let the sandbox go."""
  run = LAB / "results" / run_id
  if SUBSTRATE or WARMPOOL:
    from harvey_labs.sandbox import SandboxRequest

    folders = {"documents_dir": task["documents"], "workspace_dir": run / "workspace", "output_dir": run / "output"}
    box = await remote_sandboxes()(SandboxRequest(lab_root=LAB, run_id=run_id, command_timeout=60, **folders))

    async def close():
      try:
        await box.collect_outputs(run / "output")
      finally:
        await box.cleanup()

    return box.execute_tool, box.tool_metrics, close

  from sandbox.sandbox import Sandbox

  box = Sandbox(task["documents"], run / "output", run / "workspace", default_timeout=60)
  await asyncio.to_thread(box.start)
  tools = ToolExecutor(sandbox=box, shell_timeout=60)
  return lambda *call: asyncio.to_thread(tools.execute, *call), tools.get_metrics, lambda: asyncio.to_thread(box.stop)


# ***** one episode *****


async def episode(sampler: tinker.SamplingClient, task: dict, prompt: list[int], grading: asyncio.Semaphore) -> dict:
  run_id = f"tinyrl/{task['name'].replace('/', '__')}/{uuid.uuid4().hex[:12]}"
  run = LAB / "results" / run_id
  (run / "output").mkdir(parents=True)
  (run / "workspace").mkdir()
  for skill in SKILLS:
    if (scripts := LAB / "harness" / "skills" / skill / "scripts").exists():
      shutil.copytree(scripts, run / "workspace" / "skills" / skill / "scripts")
  tokens, sampled, end, started, sampling = list(prompt), [], "turns", time.time(), 0.0
  LIVE["episodes_running"] += 1
  try:
    execute, metrics, close = await open_sandbox(task, run_id)
  except BaseException:
    LIVE["episodes_running"] -= 1
    raise
  try:
    for _ in range(MAX_TURNS):
      room = TRAJ_TOKENS - len(tokens)
      if room < 2048:
        end = "context"
        break
      params = types.SamplingParams(max_tokens=min(TURN_TOKENS, room), stop=STOP)
      asked = time.time()
      seq = (await sampler.sample_async(types.ModelInput.from_ints(tokens), 1, params)).sequences[0]
      sampling += time.time() - asked
      sampled.append((len(tokens), list(seq.tokens), list(seq.logprobs)))
      LIVE["sampled_tokens"] += len(seq.tokens)
      LIVE["turns"] += 1
      tokens += seq.tokens
      text = tok.decode(seq.tokens, skip_special_tokens=False)
      try:
        message = parse(text)
      except ValueError:
        end = "malformed"
        break
      calls = message.get("tool_calls")
      if seq.stop_reason == "length":
        end = "length"
        break
      if not calls:
        end = "malformed" if "tool_call>" in text else "done"  # <|tool_call> for Gemma, <tool_call> for Qwen
        break
      results = [await execute(c["function"]["name"], c["function"]["arguments"]) for c in calls]
      tokens += tool_turn(calls, [bounded(r) for r in results], tokens[-1])
  finally:
    LIVE["episodes_running"] -= 1
    (run / "metrics.json").write_text(json.dumps(metrics()))
    await close()
  rolled = time.time()
  LIVE["episodes_grading"] += 1
  try:
    async with grading:
      reward, graded = await asyncio.to_thread(grade, run_id, task["name"])
  finally:
    LIVE["episodes_grading"] -= 1
  LIVE["episodes_done"] += 1
  return {
    "task": task["name"],
    "run_id": run_id,
    "tokens": tokens,
    "sampled": sampled,
    "reward": reward,
    "end": end,
    "turns": len(sampled),
    "rollout_s": rolled - started,
    "sample_s": sampling,
    "grade_s": time.time() - rolled,
    **graded,
  }


def grade(run_id: str, task: str) -> tuple[float, dict]:
  """The fraction of the task's rubric that LAB's judge passes, 0 when the episode wrote nothing."""
  run = LAB / "results" / run_id
  if not any(p.is_file() and p.stat().st_size for p in (run / "output").rglob("*")):
    return 0.0, {"graded": "no output"}
  python = LAB / ".venv" / "bin" / "python"
  flags = ["--run-id", run_id, "--task", task, "--judge-model", JUDGE, "--parallel", str(JUDGE_PARALLEL)]
  done = subprocess.run([python, "-m", "evaluation.run_eval", *flags], cwd=LAB, capture_output=True, text=True)
  (run / "grading.log").write_text(done.stdout + done.stderr)
  scores = json.loads(scores_file.read_text()) if (scores_file := run / "scores.json").exists() else {}
  if done.returncode or not scores.get("n_criteria"):
    return 0.0, {"graded": "judge failed"}
  passed = {"criteria": scores["n_criteria"], "criteria_passed": scores["n_passed"]}
  return scores["n_passed"] / scores["n_criteria"], {"graded": "yes", **passed}


# ***** training *****


def datum(ep: dict, advantage: float) -> types.Datum:
  """The whole episode as one sequence. Only sampled tokens carry an advantage, so the prompt and the tools' replies are context."""
  tokens = ep["tokens"]
  logprobs, advantages = [0.0] * (len(tokens) - 1), [0.0] * (len(tokens) - 1)
  for at, _, turn_logprobs in ep["sampled"]:
    for k, lp in enumerate(turn_logprobs):
      logprobs[at + k - 1], advantages[at + k - 1] = lp, advantage
  inputs = {"target_tokens": tokens[1:], "logprobs": logprobs, "advantages": advantages}
  return types.Datum(model_input=types.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs=inputs)


def datums(episodes: list[dict]) -> list[types.Datum]:
  by_task: dict[str, list[dict]] = {}
  for ep in episodes:
    by_task.setdefault(ep["task"], []).append(ep)
  out = []
  for group in by_task.values():
    mean = sum(ep["reward"] for ep in group) / len(group)
    out += [datum(ep, ep["reward"] - mean) for ep in group if ep["reward"] != mean and ep["sampled"]]
  return out


async def train(trainer: tinker.TrainingClient, data: list[types.Datum]) -> dict:
  start = time.time()
  forward = await (await trainer.forward_backward_async(data, "importance_sampling")).result_async()
  mid = time.time()
  optim = await (await trainer.optim_step_async(types.AdamParams(learning_rate=LR))).result_async()
  return {
    "forward_backward_s": mid - start,
    "optim_s": time.time() - mid,
    "loss": forward.metrics.get("loss:sum"),
    "grad_norm": (optim.metrics or {}).get("grad_norm:mean"),
    "datums": len(data),
    "train_tokens": sum(d.model_input.length for d in data),
  }


# ***** a run *****


async def done(call):
  return await (await call).result_async()


async def rollouts(
  samplers: list[tinker.SamplingClient], tasks: list[dict], group: int, grading: asyncio.Semaphore
) -> list[dict]:
  room = asyncio.Semaphore(AT_ONCE or len(tasks) * group)

  async def one(i: int, task: dict) -> dict:
    async with room:
      return await episode(samplers[i % len(samplers)], task, first_prompt(task), grading)

  runs = [one(i, task) for i, task in enumerate(task for task in tasks for _ in range(group))]
  results = await asyncio.gather(*runs, return_exceptions=True)
  episodes = [r for r in results if isinstance(r, dict)]
  for r in results:
    if not isinstance(r, dict):
      print(f"episode failed: {r!r}", flush=True)
  if 2 * len(episodes) < len(results):  # the sampler or the sandboxes are down, and the step is to be run again
    raise RuntimeError(f"{len(results) - len(episodes)} of {len(results)} episodes failed")
  return episodes


def summary(episodes: list[dict]) -> dict:
  criteria = sum(ep.get("criteria", 0) for ep in episodes)
  return {
    "episodes": len(episodes),
    "reward": sum(ep["reward"] for ep in episodes) / max(1, len(episodes)),
    # the criteria passed over the criteria of the episodes the judge graded, which is how LAB pools a score
    "pass_rate": sum(ep.get("criteria_passed", 0) for ep in episodes) / criteria if criteria else 0.0,
    "ends": {e: sum(ep["end"] == e for ep in episodes) for e in sorted({ep["end"] for ep in episodes})},
    "graded": {g: sum(ep["graded"] == g for ep in episodes) for g in sorted({ep["graded"] for ep in episodes})},
    "tokens": sum(len(ep["tokens"]) for ep in episodes),
    "longest": max((len(ep["tokens"]) for ep in episodes), default=0),
    "sampled_tokens": sum(len(turn) for ep in episodes for _, turn, _ in ep["sampled"]),
    "turns": sum(ep["turns"] for ep in episodes),
    "sample_s": sum(ep["sample_s"] for ep in episodes),
    "rollout_s": sum(ep["rollout_s"] for ep in episodes),
  }


def record(kind: str, number: int, line: dict) -> None:
  """A step's line in the log, and its episodes as trajectory files, written in the background."""
  with open(OUT / "log.jsonl", "a") as f:
    f.write(json.dumps(line) + "\n")
  print(json.dumps(line), flush=True)
  if EXPORT:
    run, name, facts = (EXPORT.split(" ", 2) + ["", ""])[:3]
    run, name = (f"{run}-eval", f"{name} · held-out tasks") if kind == "eval" else (run, name)
    command = [sys.executable, "traj.py", "export", OUT / f"{kind}-{number}.pkl", OUT / "trajectories", run, name]
    with open(OUT / "export.log", "a") as log:
      subprocess.Popen([*command, *filter(None, facts.split(";"))], stdout=log, stderr=log, cwd=Path(__file__).parent)


def scrape(url: str) -> list[dict]:
  rows = []
  for line in urllib.request.urlopen(url, timeout=5).read().decode().splitlines():
    m = re.match(r"(\w+)(?:\{(.*)\})?\s+(\S+)$", line)
    if m and SCRAPE_KEEP.match(m[1]):
      labels = dict(
        kv for kv in re.findall(r'(\w+)="([^"]*)"', m[2] or "") if kv[0] in ("gpu", "Hostname", "worker", "run", "node")
      )
      rows.append({"name": m[1], "labels": labels, "value": float(m[3])})
  return rows


async def telemetry():
  """Every SCRAPE_EVERY seconds: this client's counters, the gateway's and the scraped endpoints', a line each in telemetry.jsonl."""
  while True:
    now, rows = time.time(), [{"name": f"lab_{k}", "labels": {}, "value": v} for k, v in LIVE.items()]
    for url in [f"{BASE_URL}/metrics", *SCRAPE]:
      try:
        rows += await asyncio.to_thread(scrape, url)
      except Exception as e:
        rows.append({"name": "lab_scrape_failed", "labels": {"url": url, "error": type(e).__name__}, "value": 1})
    with open(OUT / "telemetry.jsonl", "a") as f:
      f.writelines(json.dumps({"t": round(now, 1), **row}) + "\n" for row in rows)
    await asyncio.sleep(SCRAPE_EVERY)


async def main():
  OUT.mkdir(parents=True, exist_ok=True)
  asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(4 * BATCH * GROUP))
  train_tasks, held_out = ([load_task(TASK)] * (BATCH * STEPS), []) if TASK else split()
  state = json.loads(p.read_text()) if (p := OUT / "state.json").exists() else {"step": 0, "path": None}
  # a session for each sampler: tinyrl gives sessions that number themselves differently a sampler worker each
  key = os.getenv("TINKER_API_KEY", "tml-tinyrl")
  services = [
    tinker.ServiceClient(base_url=BASE_URL, api_key=key, user_metadata={"openrl.sampler": str(i)})
    for i in range(SAMPLERS)
  ]
  trainer = await services[0].create_lora_training_client_async(
    MODEL, rank=RANK, user_metadata={"openrl.gpus": str(GPUS)}
  )
  if state["path"]:
    await done(trainer.load_state_with_optimizer_async(state["path"]))
  print(
    f"model {trainer.model_id} on {GPUS} GPUs, judge {JUDGE}, from step {state['step']} {state['path'] or ''}",
    flush=True,
  )
  grading, polling = asyncio.Semaphore(GRADING), asyncio.create_task(telemetry())
  for step in range(state["step"], STEPS + 1):
    start = time.time()
    evaluate = (
      EVAL_EVERY and held_out and (step % EVAL_EVERY == 0 or step == STEPS) and not (OUT / f"eval-{step}.pkl").exists()
    )
    if step == STEPS and not evaluate:
      break
    if SAMPLERS == 1:
      samplers = [await trainer.save_weights_and_get_sampling_client_async()]
    else:
      path = (await done(trainer.save_weights_for_sampler_async(f"step-{step}-{int(start)}"))).path
      samplers = await asyncio.gather(*(s.create_sampling_client_async(model_path=path) for s in services))
    saved = time.time()
    if evaluate:
      episodes = await rollouts(samplers, held_out[:EVAL_TASKS], 1, grading)
      with open(OUT / f"eval-{step}.pkl", "wb") as f:
        pickle.dump(episodes, f)
      record("eval", step, {"eval": step, **summary(episodes), "eval_s": time.time() - saved})
    if step == STEPS:
      break
    start, saved = start + (time.time() - saved if evaluate else 0), time.time() if evaluate else saved
    episodes = await rollouts(samplers, train_tasks[step * BATCH : (step + 1) * BATCH], GROUP, grading)
    rolled = time.time()
    with open(OUT / f"step-{step}.pkl", "wb") as f:
      pickle.dump(episodes, f)
    data = datums(episodes)
    line = {
      "step": step,
      "save_and_sampler_s": saved - start,
      "rollouts_and_grading_s": rolled - saved,
      "failed": BATCH * GROUP - len(episodes),
      **summary(episodes),
      **(await train(trainer, data) if data else {"datums": 0}),
    }
    kept = await done(trainer.save_state_async(f"step-{step + 1}-{uuid.uuid4().hex[:6]}"))
    (OUT / "state.json").write_text(json.dumps({"step": step + 1, "path": kept.path}))
    line["step_s"] = time.time() - start
    record("step", step, line)
  polling.cancel()


if __name__ == "__main__":
  asyncio.run(main())
