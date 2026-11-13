# LAB on tinyrl

Reinforcement learning on [Harvey LAB](https://github.com/ShubyM/harvey-labs), a benchmark of legal tasks: the model reads a
matter's documents with tools, writes the deliverable, and a judge model grades it against the task's criteria. One step samples
`group` episodes for each of `batch` tasks and trains on every sampled token with advantage = reward − its task's mean.

| File | What it is |
|---|---|
| `lab_rl.py` | The recipe: the config, LAB's tasks and tools joined to tinker-cookbook's environment and training loop. |
| `gemma4.py` | A cookbook renderer for Gemma 4, which the cookbook has none for. |
| `substrate.py` | An episode's sandbox as an Agent Substrate environment, behind the cookbook's `SandboxInterface`. |
| `Dockerfile`, `requirements.txt` | The run as an image for the cluster. |

Almost all of it is [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)'s. Its `rl.train` is the loop:
sampling, advantages, the training step, the evals on held-out tasks, checkpoints and logs. Its `build_agent_tool_env` is the
environment: it renders the conversation, reads the tool calls out of a sampled turn, runs the tools, and ends an episode on its
turns or its room for tokens. What is here joins LAB to them:

- **Tools.** Each of LAB's tools is a cookbook `Tool` that runs LAB's own code in the rollout's sandbox. A turn's calls run in the
  order the model made them.
- **Sandboxes.** A rollout has a sandbox of its own, opened by its first tool call and closed when the episode ends.
- **Grading.** When an episode ends, what it wrote comes home and LAB's judge grades it. The cookbook's environment asks for the
  grade when an episode stops by itself or runs out of turns. For the other endings (a turn that cannot be read, a turn cut off
  by its token limit, no room left) it does not, so the group grades those as their group reward: every episode that wrote
  something is graded, however it ended.
- **Sampled turns stay as sampled.** The cookbook renders a conversation again from its parsed messages at every turn, and a
  turn written again differs from the sampled one in whitespace. The next prompt would then not extend the last, and an episode
  of 50 turns would train as 50 sequences instead of one. Both renderers here keep a sampled turn's tokens on its message and
  write those: `gemma4.py`, and `as_sampled` in `lab_rl.py`, which does it to any cookbook renderer.

The tasks, tools and grader are LAB's. The held-out tasks and the order of the training tasks are the same as in open-rl's
`examples/harvey_labs` recipe, so a run here and a run there can be compared; no code is shared with it. First prompts are the
same as open-rl's too: for Qwen 3.8 the cookbook's `qwen3_8_xhigh_reasoning` renderer writes them, for Gemma 4 the model's own
chat template does, through `gemma4.py`.

## Sandboxes

An episode's tools run in an [Agent Substrate](https://github.com/agent-substrate/substrate) environment, reached through
`ate-env-api`: set `substrate_endpoint` and `substrate_image`, and `SUBSTRATE_ENV_AUTH_TOKEN` in the environment.

- `substrate_image` is LAB's sandbox image pinned by digest; Substrate runs it unmodified. The token is the one in the
  `ate-env-api-token` secret of the `ate-env` namespace.
- An environment is suspended after `substrate_idle` seconds (30) without a call, so it holds no worker while the model
  generates, and the next call resumes it.
- Substrate keeps an environment until it is deleted, so the run deletes its environments when an episode ends and when it is
  stopped with SIGTERM. A run that is killed outright leaves them behind.
- Substrate kills an environment that goes over its memory limit and does not start it again. A tool call that fails on such an
  environment fails the episode (the cookbook's `SandboxTerminatedError`), and the cookbook's loop rolls it out again.

## Run it

Against a tinyrl gateway, with a judge that speaks the OpenAI API:

```bash
LAB_ROOT=/path/to/harvey-labs PYTHONPATH=$LAB_ROOT OPENAI_BASE_URL=http://judge/v1 OPENAI_API_KEY=unused \
  python lab_rl.py base_url=http://localhost:18000 substrate_endpoint=ate-env-api.ate-env:7777 \
    substrate_image=<registry>/lab-sandbox@sha256:... steps=35 eval_every=5
```

In a cluster, build the image from this folder with the Substrate client beside the files:

```bash
cp -r examples/lab build && cp -r <sandboxenv-rl>/clients/python build/ate-env-client && docker build build
```

The client must be the commit `ate-env-api` was built from.

Settings are `name=value` arguments, the fields of `Config` at the top of `lab_rl.py`. The ones that shape a run:

| Setting | Default | |
|---|---|---|
| `model` | `google/gemma-4-E4B-it` | Gemma 4, or a model the cookbook has a renderer for. |
| `renderer` | | A cookbook renderer by name, e.g. `qwen3_8_medium_reasoning`; without one, the cookbook's choice for the model. |
| `batch`, `group`, `steps` | 8, 6, 1 | Tasks per step, episodes per task, steps. |
| `eval_every`, `eval_tasks` | 0, 50 | Held-out tasks run before every so many steps and after the last. |
| `traj_tokens`, `turn_tokens`, `max_turns` | 32768, 8192, 40 | An episode's context, a turn's length, and its turns. |
| `substrate_endpoint`, `substrate_image`, `substrate_idle` | | The sandboxes, as above. |
| `out` | `out` | The cookbook's logs and checkpoints. |
| `capture` | off | Keep every sampling and training call's tokens in the cookbook's capture store, under `out/capture`. |

Two things tinyrl takes from the session rather than from the recipe: the trainer's width, as `TINKER_TAGS=openrl.gpus=4` in
the environment, and the API key, as `TINKER_API_KEY`.

An episode leaves a folder under `$LAB_ROOT/results/tinyrl/<task>/<id>/`, which is where LAB's grader reads it: `output/` (what
it wrote), `messages.jsonl` (the conversation, as cookbook messages), `metrics.json` (LAB's tool counts), and the grader's
`scores.json` and `grading.log`. With `capture=true` the cookbook also keeps each call's prompt and sampled tokens; a prompt is
the whole conversation so far, so that store grows with the square of an episode's turns.

What the cookbook's loop does not give: it samples through one sampling client, so a step's episodes share one sampler, and a
run that is restarted resumes from the cookbook's checkpoint, which needs a gateway that answers `weights_info`.

## What has been run

- **Two steps on Agent Substrate, with the held-out tasks before each and after the last.** Gemma 4 E4B, 2 tasks × 2, 2 held-out
  tasks, 65,536 token episodes, a 5 second idle limit, `capture=true`, tinker-cookbook 0.6.1 and the Tinker SDK 0.39.0. Nothing
  failed. Held-out reward was 0.138, then 0.104, then 0.333 after the last step; with two tasks that is noise, not learning.
  Both training steps ran, on 198,423 and 56,006 tokens (the second on one task: the other's two episodes scored the same and
  were dropped), and the cookbook's check that the trainer sees the tokens the sampler produced (the KL between their
  log-probabilities) read 0.0005 and 0.0006. Checkpoints were saved after each step, and the capture store held a row for each
  sampling call.
- **A scripted rollout, through the cookbook's runner, on real sandboxes.** A policy that says fixed turns in place of a model,
  with Gemma 4's renderer. An episode that looks, then writes and checks in one turn of two calls, then stops: the second call
  sees the first one's file, the episode becomes one training sequence over all its tokens with the advantage on exactly the
  sampled ones, each call is given the smaller of a turn's tokens and the room left, and no sandbox is left. Episodes that end
  on the turn limit, on a cut-off turn and on a tool call that cannot be read are all graded, the last two as a group reward.
  One whose sampler fails leaves no sandbox behind. One whose sandbox is killed for its memory (a command that takes 6 GiB)
  fails with `SandboxTerminatedError` on the next call instead of going on.
- **Renderers, token for token.** Gemma 4 E4B, thinking on and off, three tasks: first prompts (6,460, 6,413 and 6,420 tokens
  with thinking) and every prompt after five constructed sampled turns (thinking then a call, two calls in a turn, a call
  without its stop token, text before a call, a last answer) are the tokens the model's chat template writes around the
  sampled turns. Qwen 3.8 27B, two tasks: the first prompt of the first training task is open-rl's (6,517 tokens), and after
  each constructed turn, some with whitespace the template would not write, the next prompt is the last prompt, the turn as
  sampled, and what the chat template writes after it. The cookbook reads a Qwen turn without its stop token as broken, so the
  sampler has to return the stop token; tinyrl's does.
- **Tasks against open-rl's recipe.** The 50 held-out tasks and the first 300 training tasks are the same, in the same order.
- **`substrate.py` against open-rl's Substrate sandbox, before it was put behind the cookbook's interface.** 23 tool calls on
  the same task: 22 gave the same result, with the same tool metrics and the same files brought home; the other was a grep
  count whose lines came in a different order, since LAB's grep runs on the host here and in the sandbox there. An output
  folder holding a link to `/etc/passwd` leaked nothing. Not rerun on this version.
- **Before the cookbook's environment, the same episode logic as a plain script: 35 steps of 8 tasks × 6 on 4 B200s.**
  Gemma 4 E4B, with sandboxes from an Agent Sandbox warm pool. Reward on the 50 held-out tasks went from 0.087 to 0.380, and
  the pass rate from 0.168 to 0.403. Its training tasks were in a slightly different order from position 127 on, since fixed.

Not done: a full run of this version, a Qwen run of this version, this folder's image, resuming a run from the cookbook's
checkpoint, and several samplers under the cookbook's loop.
