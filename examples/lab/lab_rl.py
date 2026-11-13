"""LAB legal-agent RL on tinyrl, as a tinker-cookbook recipe. LAB supplies the tasks, the tools and the grader; the cookbook supplies
the rest: its tool-using environment (build_agent_tool_env) runs an episode, and its loop samples GROUP episodes for each of BATCH
tasks, trains on each episode's sampled tokens with advantage = reward - its task's mean reward, runs the held-out tasks every
EVAL_EVERY steps, and keeps checkpoints to start again from.

What is here is what joins them: LAB's tools as cookbook tools, each episode's sandbox, which is an Agent Substrate environment
(substrate.py) opened by the episode's first tool call, and the episode's end, where the sandbox is closed and LAB's judge
grades what was written.

  LAB_ROOT=/path/to/harvey-labs PYTHONPATH=$LAB_ROOT OPENAI_BASE_URL=http://judge/v1 OPENAI_API_KEY=unused \\
    python lab_rl.py base_url=http://localhost:18000 substrate_endpoint=ate-env-api.ate-env:7777 substrate_image=... steps=35

The held-out tasks, the order of the training tasks and the system prompt are those of open-rl's examples/harvey_labs, so the two
train and evaluate on the same tasks. Nothing is imported from it. Prompts are written by the cookbook's renderer for the model,
or for Gemma 4, which it has none for, by gemma4.py. See README.md.
"""

import asyncio
import contextlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chz
import substrate
import tinker
from gemma4 import SAMPLED, Gemma4
from harness.tools import get_all_tool_definitions
from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers.base import Message, RenderedMessage, Renderer
from tinker_cookbook.rl import train
from tinker_cookbook.rl.rollout_limits import RolloutLimits
from tinker_cookbook.rl.rollout_presets import RolloutConfig
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, Trajectory
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.tool_use import build_agent_tool_env, simple_tool_result
from tinker_cookbook.tool_use.types import ToolInput, ToolResult

LAB = Path(os.environ["LAB_ROOT"]).resolve()  # LAB's checkout, which is also on PYTHONPATH for its harness and sandbox
SKILLS = sorted(p.parent.name for p in (LAB / "harness" / "skills").glob("*/SKILL.md"))


@chz.chz
class Config:
  substrate_endpoint: str  # ate-env-api of an Agent Substrate cluster
  substrate_image: str  # LAB's sandbox image, pinned by digest
  substrate_idle: float = 30.0  # seconds without a call before a sandbox is suspended
  model: str = "google/gemma-4-E4B-it"
  renderer: str = ""  # a cookbook renderer by name; without one, the cookbook's choice for the model, or gemma4.py
  think: bool = True  # Gemma 4's thinking; for other models the renderer's name says
  base_url: str = "http://localhost:18000"
  out: str = "out"  # the cookbook's logs and checkpoints
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
  judge: str = "gpt-gemma-4-31b-it"
  judge_parallel: int = 16
  grading: int = 6  # episodes graded at once
  capture: bool = False  # keep every call's tokens in the cookbook's capture store, under out/capture


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


# ***** the model's side: prompts and sampled turns *****


def as_sampled(renderer: Renderer) -> Renderer:
  """A cookbook renderer made to keep a sampled turn in the conversation as the tokens that were sampled. The cookbook's write a
  past turn again from its parsed message, which for Qwen differs in whitespace from what was sampled: the next prompt would then
  not extend this one, and an episode would train as one sequence per turn instead of one. gemma4.py does the same for Gemma 4."""

  class AsSampled(type(renderer)):
    has_extension_property = True

    def parse_response(self, response: list[int]):
      message, end = super().parse_response(response)
      message[SAMPLED] = list(response)
      return message, end

    def render_message(self, message: Message, ctx) -> RenderedMessage:
      if SAMPLED not in message:
        return super().render_message(message, ctx)
      # the opening the turn was sampled after, and the turn, ended by the stop token whether or not sampling returned it
      opening, tokens, stop = self._get_generation_suffix("assistant", ctx), message[SAMPLED], self._end_message_token
      turn = tinker.EncodedTextChunk(tokens=tokens if tokens[-1] == stop else [*tokens, stop])
      return RenderedMessage(header=tinker.EncodedTextChunk(tokens=opening), output=[turn])

  renderer.__class__ = AsSampled  # the cookbook builds its renderers by name, so the built one is given the class
  return renderer


def renderer_for(cfg: Config) -> Renderer:
  tokenizer = get_tokenizer(cfg.model)
  if "gemma-4" in cfg.model and not cfg.renderer:
    return Gemma4(tokenizer, cfg.model, cfg.think)
  name = cfg.renderer or model_info.get_recommended_renderer_name(cfg.model)
  return as_sampled(renderers.get_renderer(name, tokenizer, model_name=cfg.model))


def bounded(tokenizer, result: str, limit: int) -> str:
  tokens = tokenizer.encode(result, add_special_tokens=False)
  if len(tokens) <= limit:
    return result
  kept = tokenizer.decode(tokens[:limit], skip_special_tokens=False)
  kept = kept[: kept.rfind("\n") + 1] or kept
  return f"{kept.rstrip()}\n\n[Observation truncated at {limit:,} tokens. Run a narrower command, or read with an offset and a limit, to get the rest.]"


# ***** an episode *****


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
  passed = {"criteria": scores["n_criteria"], "criteria_passed": scores["n_passed"]}
  return scores["n_passed"] / scores["n_criteria"], {"graded": "yes", **passed}


class LabTool:
  """One of LAB's tools as a cookbook tool, run in its rollout's sandbox."""

  def __init__(self, rollout: "Rollout", spec: dict):
    self.rollout, self.name = rollout, spec["name"]
    self.description, self.parameters_schema = spec["description"], spec["parameters"]

  async def run(self, input: ToolInput) -> ToolResult:
    result = await self.rollout.execute(self.name, input.arguments)
    return simple_tool_result(result, call_id=input.call_id or "", name=self.name)


class Rollout:
  """One episode: the cookbook's environment for it, a sandbox of its own, opened by its first tool call, and its end, where the
  sandbox is closed and what the episode wrote is graded, once."""

  def __init__(self, group: "Group"):
    self.group, self.cfg, self.task = group, group.cfg, group.task
    cfg, renderer, specs = self.cfg, group.renderer, get_all_tool_definitions()
    self.run_id = f"tinyrl/{self.task['name'].replace('/', '__')}/{uuid.uuid4().hex[:12]}"
    self.run = LAB / "results" / self.run_id  # where LAB's grader looks for what an episode wrote
    self.box = self.tools = self.graded = None
    tools, user = [LabTool(self, spec) for spec in specs], {"role": "user", "content": self.task["instructions"]}
    first = [*renderer.create_conversation_prefix_with_tools(specs, system_prompt(self.task)), user]
    # the cookbook's runner gives each call a turn's worth of tokens, or what room is left
    limits = RolloutLimits(max_turn_tokens=cfg.turn_tokens, max_trajectory_tokens=cfg.traj_tokens)
    rollout = RolloutConfig(limits=limits, tool_execution="sequential")  # a turn's calls in the order it made them
    # an episode ends on its turns, and with less than 2048 tokens of room for its next turn
    ends = dict(max_turns=cfg.max_turns, max_trajectory_tokens=cfg.traj_tokens, max_generation_tokens=2048)
    # and on a cut-off turn. The cookbook grades neither that ending nor a turn it cannot read, so their rewards
    # stand for no grade: Group.compute_group_rewards grades those
    ungraded = dict(terminate_on_length=True, failed_parse_reward=0.0, context_overflow_reward=0.0)
    self.env = build_agent_tool_env(renderer, tools, first, self.finish, rollout_config=rollout, **ends, **ungraded)

  async def execute(self, name: str, arguments: dict) -> str:
    if self.box is None:
      (self.run / "output").mkdir(parents=True)
      (self.run / "workspace").mkdir()
      for skill in SKILLS:
        if (scripts := LAB / "harness" / "skills" / skill / "scripts").exists():
          shutil.copytree(scripts, self.run / "workspace" / "skills" / skill / "scripts")
      cfg, documents, workspace = self.cfg, self.task["documents"], self.run / "workspace"
      where = cfg.substrate_endpoint, cfg.substrate_image, cfg.substrate_idle
      box = await substrate.open_sandbox(*where, documents, workspace)
      files = substrate.Files(box, asyncio.get_running_loop(), documents, workspace, self.run / "output", timeout=60)
      self.box, self.tools = box, substrate.Tools(sandbox=files, shell_timeout=60)
    result = await asyncio.to_thread(self.tools.execute, name, arguments)
    if self.box.gone:  # which LAB's tools would have told the model as one more error
      raise self.box.gone
    return bounded(self.group.renderer.tokenizer, result, self.cfg.tool_tokens)

  async def shut(self) -> None:
    """Let the sandbox go, once: at the episode's end, or by its group when the episode failed."""
    if box := self.box:
      self.box = None
      await box.cleanup()

  async def finish(self, messages: list[Message] | None = None) -> tuple[float, dict]:
    """The episode's grade, which the cookbook's environment asks for at an episode's end, and the group for the rest."""
    if self.graded is None:
      self.graded = await self.end()
    return self.graded

  async def end(self) -> tuple[float, dict]:
    if self.box:
      (self.run / "metrics.json").write_text(json.dumps(self.tools.get_metrics() | {"suspends": self.box.suspends}))
      try:
        await substrate.bring_home(self.box, self.run / "workspace", self.run / "output")
      finally:
        await self.shut()
    async with self.group.grading:
      reward, graded = await asyncio.to_thread(grade, self.cfg, self.run_id, self.task["name"])
    metrics = {"graded": float(graded["graded"] == "yes"), "wrote_nothing": float(graded["graded"] == "no output")}
    return reward, metrics | {k: graded[k] for k in ("criteria", "criteria_passed") if k in graded}


class Group(EnvGroupBuilder):
  """A task's episodes in one step."""

  def __init__(self, cfg: Config, renderer: Renderer, grading: asyncio.Semaphore, task: dict, size: int):
    self.cfg, self.renderer, self.grading, self.task, self.size = cfg, renderer, grading, task, size
    self.rollouts: dict[int, Rollout] = {}

  async def make_envs(self) -> list[Env]:
    made = [Rollout(self) for _ in range(self.size)]
    self.rollouts |= {id(rollout.env): rollout for rollout in made}
    return [rollout.env for rollout in made]

  async def compute_group_rewards(self, trajectories: list[Trajectory], envs: list[Env]) -> list[tuple[float, dict]]:
    """The cookbook's environment has an episode graded when it ends by itself, or by its turns. One that ended on a turn that could
    not be read, on a turn's token limit or out of room was not, and still has its sandbox: it is graded here, as its group reward."""

    async def late(env: Env) -> tuple[float, dict]:
      rollout = self.rollouts[id(env)]
      return (0.0, {}) if rollout.graded else await rollout.finish()

    return await asyncio.gather(*map(late, envs))

  async def cleanup(self) -> None:
    await asyncio.gather(*(rollout.shut() for rollout in self.rollouts.values()), return_exceptions=True)
    self.rollouts = {}

  def logging_tags(self) -> list[str]:
    return ["lab"]


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
    renderer, grading = renderer_for(cfg), asyncio.Semaphore(cfg.grading)
    tasks, held_out = ([load_task(cfg.task)] * (cfg.batch * cfg.steps), []) if cfg.task else split()
    steps = [tasks[i * cfg.batch : (i + 1) * cfg.batch] for i in range(len(tasks) // cfg.batch)]
    batches = [[Group(cfg, renderer, grading, t, cfg.group) for t in batch] for batch in steps]
    evals = [Group(cfg, renderer, grading, t, 1) for t in held_out[: cfg.eval_tasks]]
    return Batches(batches), Batches([evals]) if evals and cfg.eval_every else None


# ***** a run *****


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
  captured = contextlib.nullcontext()
  if cfg.capture:
    from tinker_cookbook.capture.store import capture_to_store  # which starts a store daemon, so only when asked for

    captured = capture_to_store("lab", data_dir=str(Path(cfg.out, "capture")))
  with captured:
    await train.main(run)


if __name__ == "__main__":
  asyncio.run(main(chz.entrypoint(Config)))
