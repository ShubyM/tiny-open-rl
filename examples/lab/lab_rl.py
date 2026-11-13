"""LAB legal-agent RL on tinyrl, as a tinker-cookbook recipe. LAB supplies the tasks, the tools and the grader; the cookbook supplies
the loop: it samples GROUP episodes for each of BATCH tasks, trains on each episode's sampled tokens with advantage = reward - its
task's mean reward, runs the held-out tasks every EVAL_EVERY steps, and keeps checkpoints to start again from.

What is here is the environment. An episode opens its own sandbox with its first prompt, reads each sampled turn's tool calls
itself, runs them in that sandbox, and when it ends, closes the sandbox and has LAB's judge grade what was written. The sandbox
is an Agent Substrate environment (substrate.py), which is suspended while the model generates, or LAB's own podman container.

  LAB_ROOT=/path/to/harvey-labs PYTHONPATH=$LAB_ROOT OPENAI_BASE_URL=http://judge/v1 OPENAI_API_KEY=unused \\
    python lab_rl.py base_url=http://localhost:18000 steps=35 eval_every=5

The held-out tasks, the order of the training tasks and the system prompt are those of open-rl's examples/harvey_labs, so the two
train and evaluate on the same tasks. Nothing is imported from it. Prompts are written by the model's chat template. See README.md.
"""

import asyncio
import itertools
import json
import logging
import os
import random
import re
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chz
import tinker
from harness.tools import ToolExecutor, get_all_tool_definitions
from huggingface_hub import hf_hub_download
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl import train
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.rl.rollout_limits import RolloutLimits
from tinker_cookbook.rl.types import (
  Env,
  EnvGroupBuilder,
  InitialObservationOverflow,
  RLDataset,
  RLDatasetBuilder,
  StepResult,
)
from transformers import AutoTokenizer
from transformers.utils.chat_parsing import parse_response

LAB = Path(os.environ["LAB_ROOT"]).resolve()  # LAB's checkout, which is also on PYTHONPATH for its harness and sandbox
SKILLS = sorted(p.parent.name for p in (LAB / "harness" / "skills").glob("*/SKILL.md"))
# how an episode ended, as the cookbook names it in a rollout's metrics
STOPS = {
  "done": "completed",
  "turns": "max_turns",
  "length": "max_tokens",
  "context": "context_overflow",
  "malformed": "parse_error",
}


@chz.chz
class Config:
  model: str = "google/gemma-4-E4B-it"  # Gemma 4, or a Qwen 3.5 family model
  base_url: str = "http://localhost:18000"
  out: str = "out"  # the cookbook's logs and checkpoints, and the episodes as files under trajectories/
  run: str = "lab"  # the run's name among the trajectories
  rank: int = 32
  lr: float = 2e-4
  steps: int = 1
  batch: int = 8  # tasks per step
  group: int = 6  # episodes per task
  task: str = ""  # train on this one task instead of the split
  eval_every: int = 0  # run the held-out tasks before every so many steps and after the last; 0 for never
  eval_tasks: int = 50
  save_every: int = 5
  max_turns: int = 40
  turn_tokens: int = 8192
  traj_tokens: int = 32768
  tool_tokens: int = 4096
  think: bool = True
  judge: str = "gpt-gemma-4-31b-it"
  judge_parallel: int = 16
  grading: int = 6  # episodes graded at once
  substrate_endpoint: str = ""  # ate-env-api of an Agent Substrate cluster; LAB's podman container without one
  substrate_image: str = ""  # LAB's sandbox image, pinned by digest
  substrate_idle: float = 30.0  # seconds without a tool call before a sandbox is suspended


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
  # sorted as paths, folder by folder, which is the order open-rl shuffles: as strings, "a-b/c" would come before "a/c"
  names = [p.parent.relative_to(LAB / "tasks").as_posix() for p in sorted((LAB / "tasks").rglob("task.json"))]
  tasks = {n: t for n in names if (t := load_task(n))}
  shuffled = list(tasks)
  random.Random(0).shuffle(shuffled)
  held_out = shuffled[300:350]
  family = lambda n: re.sub(r"/scenario-\d+$", "", n)  # noqa: E731
  train = [n for n in tasks if family(n) not in {family(h) for h in held_out}]
  random.Random(242).shuffle(train)
  return [tasks[n] for n in train], [tasks[n] for n in held_out]


# ***** prompt *****


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


class Chat:
  """A model's tokenizer and chat template: the tokens an episode is given, and its sampled turns read back as tool calls."""

  def __init__(self, model: str, think: bool, tool_tokens: int):
    self.tok, self.think, self.tool_tokens = AutoTokenizer.from_pretrained(model), think, tool_tokens
    self.response = json.loads(Path(hf_hub_download(model, "tokenizer_config.json")).read_text())["response_template"]
    ends = (
      ("<|im_end|>",) if "<|im_end|>" in self.tok.get_vocab() else ("<turn|>", "<|tool_response>")
    )  # Qwen's, or Gemma 4's
    self.stop = [self.tok.convert_tokens_to_ids(t) for t in ends]
    self.tools = [{"type": "function", "function": t} for t in get_all_tool_definitions()]
    # what the template writes to open a model turn, which a sampled turn's text follows
    user = [{"role": "user", "content": "-"}]
    self.opening = self.tok.decode(self.render(user, add_generation_prompt=True)[len(self.render(user)) :])

  def render(self, messages: list[dict], **kw) -> list[int]:
    return self.tok.apply_chat_template(
      messages, tokenize=True, return_dict=False, enable_thinking=self.think, preserve_thinking=True, **kw
    )

  def parse(self, text: str) -> dict:
    """A sampled turn as a message with its thinking, text and tool calls."""
    # after tool replies Gemma's template has already opened the thinking channel
    opened = "<channel|>" in text and not text.startswith("<|channel>")
    return parse_response(text, self.response, prefix=self.opening + ("<|channel>thought\n" if opened else ""))

  def first_prompt(self, task: dict) -> list[int]:
    messages = [{"role": "system", "content": system_prompt(task)}, {"role": "user", "content": task["instructions"]}]
    return self.render(messages, tools=self.tools, add_generation_prompt=True)

  def tool_turn(self, calls: list[dict], results: list[str], last: int) -> list[int]:
    """The tokens that follow a sampled turn of tool calls: the tools' replies and the next turn's opening, as the template writes them. The sampled turn itself stays as sampled."""
    # a user turn first, without which Qwen's template renders nothing
    user, assistant = {"role": "user", "content": "-"}, {"role": "assistant", "content": "", "tool_calls": calls}
    replies = [{"role": "tool", "name": c["function"]["name"], "content": r} for c, r in zip(calls, results)]
    turn, after = (
      self.render([user, assistant], add_generation_prompt=False),
      self.render([user, assistant, *replies], add_generation_prompt=True),
    )
    if after[: len(turn)] != turn:
      raise ValueError("the template rewrote a tool-calling turn when its replies were added")
    # sampling stops at the stop token that ends the turn, with or without returning it
    stop = max(i for i, t in enumerate(turn) if t in self.stop)
    return ([] if last == turn[stop] else [turn[stop]]) + after[stop + 1 :]

  def bounded(self, result: str) -> str:
    tokens = self.tok.encode(result, add_special_tokens=False)
    if len(tokens) <= self.tool_tokens:
      return result
    kept = self.tok.decode(tokens[: self.tool_tokens], skip_special_tokens=False)
    kept = kept[: kept.rfind("\n") + 1] or kept
    return f"{kept.rstrip()}\n\n[Observation truncated at {self.tool_tokens:,} tokens. Run a narrower command, or read with an offset and a limit, to get the rest.]"


# ***** an episode *****


async def open_sandbox(cfg: Config, task: dict, run: Path):
  """Where an episode's tools run. Returns how to run a tool, the tools' metrics, and how to bring the outputs home and let the sandbox go."""
  if cfg.substrate_endpoint:
    import substrate  # which needs the Substrate client, and a local run does not

    box = await substrate.open_sandbox(
      cfg.substrate_endpoint,
      cfg.substrate_image,
      cfg.substrate_idle,
      task["documents"],
      run / "workspace",
      run / "output",
    )
    return box.execute, box.metrics, box.close

  from sandbox.sandbox import Sandbox

  box = Sandbox(task["documents"], run / "output", run / "workspace", default_timeout=60)
  await asyncio.to_thread(box.start)
  tools = ToolExecutor(sandbox=box, shell_timeout=60)
  return lambda *call: asyncio.to_thread(tools.execute, *call), tools.get_metrics, lambda: asyncio.to_thread(box.stop)


def grade(cfg: Config, run_id: str, task: str) -> tuple[float, dict]:
  """The fraction of the task's rubric that LAB's judge passes, 0 when the episode wrote nothing."""
  run = LAB / "results" / run_id
  if not any(p.is_file() and p.stat().st_size for p in (run / "output").rglob("*")):
    return 0.0, {"graded": "no output"}
  python = LAB / ".venv" / "bin" / "python"
  flags = ["--run-id", run_id, "--task", task, "--judge-model", cfg.judge, "--parallel", str(cfg.judge_parallel)]
  done = subprocess.run([python, "-m", "evaluation.run_eval", *flags], cwd=LAB, capture_output=True, text=True)
  (run / "grading.log").write_text(done.stdout + done.stderr)
  scores = json.loads(scores_file.read_text()) if (scores_file := run / "scores.json").exists() else {}
  if done.returncode or not scores.get("n_criteria"):
    return 0.0, {"graded": "judge failed"}
  passed = {
    "criteria": scores["n_criteria"],
    "criteria_passed": scores["n_passed"],
    "rows": scores.get("criteria_results"),
  }
  return scores["n_passed"] / scores["n_criteria"], {"graded": "yes", **passed}


class Episode(Env):
  """One rollout of a task, with a sandbox of its own from its first prompt to its end. It ends itself, by its turns and its room
  for tokens, so that it is the one to close the sandbox and grade."""

  def __init__(self, group: "Group"):
    self.group, self.cfg, self.chat, self.task = group, group.cfg, group.chat, group.task
    # the cookbook's runner reads this: each call may sample a turn's worth of tokens, or what room is left
    self.rollout_limits = RolloutLimits(
      max_turn_tokens=self.cfg.turn_tokens, max_trajectory_tokens=self.cfg.traj_tokens
    )
    self.run_id = f"tinyrl/{self.task['name'].replace('/', '__')}/{uuid.uuid4().hex[:12]}"
    self.run = LAB / "results" / self.run_id
    self.tokens: list[int] = []
    self.turns = self.sampled = 0
    self.sampling = 0.0
    self.close = None  # how to let the sandbox go, while there is one
    # the episode as the trajectory files tell it: its messages in order
    system = {"type": "message", "role": "system", "text": system_prompt(self.task)}
    self.lines = [system, {"type": "message", "role": "user", "text": self.task["instructions"]}]

  def observation(self) -> tinker.ModelInput:
    self.asked = time.time()
    return tinker.ModelInput.from_ints(self.tokens)

  async def initial_observation(self):
    self.tokens = self.chat.first_prompt(self.task)
    if self.cfg.traj_tokens - len(self.tokens) < 2048:
      return InitialObservationOverflow(reward=0.0, metrics={"stop/context_overflow": 1.0})
    (self.run / "output").mkdir(parents=True)
    (self.run / "workspace").mkdir()
    for skill in SKILLS:
      if (scripts := LAB / "harness" / "skills" / skill / "scripts").exists():
        shutil.copytree(scripts, self.run / "workspace" / "skills" / skill / "scripts")
    self.started = time.time()
    self.execute, self.metrics, self.close = await open_sandbox(self.cfg, self.task, self.run)
    return self.observation(), self.chat.stop

  async def step(self, action: list[int], *, extra=None) -> StepResult:
    self.sampling += time.time() - self.asked
    self.tokens += action
    self.turns, self.sampled = self.turns + 1, self.sampled + len(action)
    text, end, calls = self.chat.tok.decode(action, skip_special_tokens=False), None, None
    line = {"type": "message", "role": "assistant", "turn": self.turns, "tokens": len(action)}
    try:
      said = self.chat.parse(text)
      calls = said.get("tool_calls")
      line |= {"thinking": said.get("thinking") or said.get("reasoning_content", ""), "text": said.get("content", "")}
      line["calls"] = [c["function"] for c in calls or []]
    except ValueError:
      end = "malformed"
      line |= {"text": text, "error": "this turn's tool call could not be parsed, which ended the episode"}
    self.lines.append(line)
    if end:
      pass
    elif (extra or {}).get("stop_reason") == "length":
      end = "length"
    elif not calls:
      end = "malformed" if "tool_call>" in text else "done"  # <|tool_call> for Gemma, <tool_call> for Qwen
    else:
      results = [
        self.chat.bounded(await self.execute(c["function"]["name"], c["function"]["arguments"])) for c in calls
      ]
      for call, result in zip(calls, results):
        reply = {
          "type": "message",
          "role": "tool",
          "turn": self.turns,
          "name": call["function"]["name"],
          "text": result,
        }
        self.lines.append(reply | {"error": result.startswith("Error")})  # how LAB's tools report one
      self.tokens += self.chat.tool_turn(calls, results, self.tokens[-1])
      if self.turns >= self.cfg.max_turns:
        end = "turns"
      elif self.cfg.traj_tokens - len(self.tokens) < 2048:
        end = "context"
    if not end:
      return StepResult(0.0, False, self.observation(), self.chat.stop)
    reward, metrics = await self.finish(end)
    return StepResult(reward, True, self.observation(), self.chat.stop, metrics={f"stop/{STOPS[end]}": 1.0} | metrics)

  async def shut(self) -> None:
    """Let the sandbox go, once: at the episode's end, or by its group when the episode failed."""
    if close := self.close:
      self.close = None
      await close()

  async def finish(self, end: str) -> tuple[float, dict]:
    (self.run / "metrics.json").write_text(json.dumps(self.metrics()))
    await self.shut()
    rolled = time.time()
    async with self.group.grading:
      reward, graded = await asyncio.to_thread(grade, self.cfg, self.run_id, self.task["name"])
    metrics = {"turns": self.turns, "tokens": len(self.tokens), "sampled_tokens": self.sampled}
    metrics |= {"graded": float(graded["graded"] == "yes"), "wrote_nothing": float(graded["graded"] == "no output")}
    metrics |= {k: graded[k] for k in ("criteria", "criteria_passed") if k in graded}
    seconds = {"seconds": rolled - self.started, "sample_seconds": self.sampling, "grade_seconds": time.time() - rolled}
    self.group.write(self, {"reward": reward} | metrics | seconds, {"end": end}, graded.get("rows"))
    return reward, metrics


class Group(EnvGroupBuilder):
  """A task's episodes in one step. The episodes own their sandboxes; the group only lets go of those an error left open."""

  def __init__(
    self, cfg: Config, chat: Chat, grading: asyncio.Semaphore, task: dict, size: int, step: int, held_out: bool
  ):
    self.cfg, self.chat, self.grading, self.task, self.size = cfg, chat, grading, task, size
    self.step, self.held_out = (
      step,
      held_out,
    )  # a held-out task's group is rolled out again at each eval, a step further on
    self.episodes: list[Episode] = []
    self.number = itertools.count(1)

  async def make_envs(self) -> list[Episode]:
    self.episodes += (made := [Episode(self) for _ in range(self.size)])
    return made

  async def cleanup(self) -> None:
    await asyncio.gather(*(episode.shut() for episode in self.episodes), return_exceptions=True)
    self.episodes, self.number = [], itertools.count(1)
    self.step += self.cfg.eval_every if self.held_out else 0

  def logging_tags(self) -> list[str]:
    return ["lab"]

  def write(self, episode: Episode, metrics: dict, tags: dict, criteria: list | None) -> None:
    """The episode as files a person or an agent can read (traj.py), beside the run's others: its messages, and what it wrote."""
    run = Path(self.cfg.out, "trajectories", self.cfg.run + ("-eval" if self.held_out else ""))
    folder, n = run / str(self.step) / self.task["name"].replace("/", "__"), next(self.number)
    folder.mkdir(parents=True, exist_ok=True)
    lines = [{"type": "episode", "group": self.task["name"], "n": n, "metrics": metrics, "tags": tags}, *episode.lines]
    lines += [{"type": "list", "name": "criteria", "rows": criteria}] if criteria else []
    (folder / f"{n}.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    if any((episode.run / "output").iterdir()):
      shutil.copytree(episode.run / "output", folder / str(n) / "files" / "output", dirs_exist_ok=True)
    name = f"{self.cfg.run} · held-out tasks" if self.held_out else self.cfg.run
    (run / "run.json").write_text(json.dumps({"name": name, "model": self.cfg.model, "facts": [], "status": "Running"}))


class Batches(RLDataset):
  def __init__(self, groups: list[list[Group]]):
    self.groups = groups

  def get_batch(self, index: int) -> list[Group]:
    return self.groups[index]

  def __len__(self) -> int:
    return len(self.groups)


@chz.chz
class Lab(RLDatasetBuilder):
  cfg: Config

  async def __call__(self) -> tuple[Batches, Batches | None]:
    cfg = self.cfg
    # the tools and the grader run in threads, a few for each episode of a step
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(4 * cfg.batch * cfg.group))
    chat, grading = Chat(cfg.model, cfg.think, cfg.tool_tokens), asyncio.Semaphore(cfg.grading)
    tasks, held_out = ([load_task(cfg.task)] * (cfg.batch * cfg.steps), []) if cfg.task else split()
    steps = [tasks[i * cfg.batch : (i + 1) * cfg.batch] for i in range(len(tasks) // cfg.batch)]
    batches = [
      [Group(cfg, chat, grading, t, cfg.group, step, False) for t in batch] for step, batch in enumerate(steps)
    ]
    evals = [Group(cfg, chat, grading, t, 1, 0, True) for t in held_out[: cfg.eval_tasks]]
    return Batches(batches), Batches([evals]) if evals and cfg.eval_every else None


# ***** a run *****


async def last_eval(run: train.Config) -> None:
  """The held-out tasks on the last weights. The cookbook's loop runs them before a step, and so never after the last one."""
  _, held_out = await run.dataset_builder()
  saved = checkpoint_utils.get_last_checkpoint(run.log_path, required_key="sampler_path")
  if held_out is None or saved is None:
    return
  for group in held_out.get_batch(0):
    group.step = run.max_steps
  sampler = await tinker.ServiceClient(base_url=run.base_url).create_sampling_client(model_path=saved.sampler_path)
  evaluator = RLTestSetEvaluator(held_out, max_tokens=run.max_tokens, strategy=run.effective_rollout_strategy())
  metrics = await train.run_single_evaluation(evaluator, run, run.max_steps, sampler, "test")
  print(
    json.dumps({"eval": run.max_steps} | {k: v for k, v in metrics.items() if isinstance(v, (int, float))}), flush=True
  )


async def main(cfg: Config) -> None:
  run = train.Config(
    model_name=cfg.model,
    base_url=cfg.base_url,
    log_path=cfg.out,
    recipe_name="tinyrl_lab",
    dataset_builder=Lab(cfg=cfg),
    learning_rate=cfg.lr,
    lora_rank=cfg.rank,
    max_tokens=cfg.turn_tokens,
    max_steps=cfg.steps,
    eval_every=cfg.eval_every,
    save_every=cfg.save_every,
    remove_constant_reward_groups=True,  # a task whose episodes all scored the same has nothing to teach
    rollout_error_tolerance=True,  # an episode that fails, on its sandbox or the sampler, is tried again
    num_groups_to_log=0,
  )
  # the cookbook logs two groups of every step token by token, which for long episodes is megabytes a step
  logging.getLogger(train.__name__).addFilter(lambda record: record.funcName != "print_group")
  await train.main(run)
  if cfg.eval_every:
    await last_eval(run)
  for told in Path(cfg.out, "trajectories").glob("*/run.json"):
    told.write_text(json.dumps(json.loads(told.read_text()) | {"status": "Completed"}))


if __name__ == "__main__":
  asyncio.run(main(chz.entrypoint(Config)))
