# LAB on tinyrl

Reinforcement learning on [Harvey LAB](https://github.com/ShubyM/harvey-labs), a benchmark of legal tasks: the model reads a
matter's documents with tools, writes the deliverable, and a judge model grades it against the task's criteria. One step samples
`group` episodes for each of `batch` tasks and trains on every sampled token with advantage = reward − its task's mean.

| File | What it is |
|---|---|
| `lab_rl.py` | The recipe: the config, the environment, and the call into tinker-cookbook's training loop. |
| `substrate.py` | An episode's sandbox as an Agent Substrate environment. |
| `traj.py` | Reads the episodes the run writes as files: list a run, show an episode. |
| `Dockerfile`, `requirements.txt` | The run as an image for the cluster. |

The loop is [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)'s `rl.train`: sampling, advantages, the
training step, the evals on held-out tasks, checkpoints and logs. The environment is ours. An episode opens a sandbox of its own
with its first prompt, reads the tool calls out of each sampled turn itself, runs them in that sandbox, and when it ends, closes
the sandbox and has LAB's judge grade what was written. Its group does nothing with sandboxes but let go of one that an error
left open. Sampled tokens are trained on as they were sampled: an episode's next prompt is its last one, the turn, and the
tools' replies appended.

The tasks, tools and grader are LAB's. The held-out tasks and the order of the training tasks are the same as in open-rl's
`examples/harvey_labs` recipe, so a run here and a run there can be compared; no code is shared with it. Prompts are written
by the model's own chat template. For Gemma 4 that gives open-rl's first prompt token for token. For Qwen 3.8 it does not:
open-rl and the cookbook's renderer declare each tool as `{"name": ...}`, the chat template given OpenAI-style tools as
`{"type": "function", "function": {...}}`, which makes a first prompt of 6,561 tokens where theirs is 6,517.

## Sandboxes

An episode's tools run in one of two places:

| Set | Sandbox |
|---|---|
| `substrate_endpoint`, `substrate_image`, and `SUBSTRATE_ENV_AUTH_TOKEN` in the environment | An [Agent Substrate](https://github.com/agent-substrate/substrate) environment through `ate-env-api`. It is suspended after `substrate_idle` seconds (30) without a tool call, so it holds no worker while the model generates, and the next tool call resumes it. |
| neither | LAB's own podman container, on the machine the script runs on. |

`substrate_image` is LAB's sandbox image pinned by digest; Substrate runs it unmodified. The token is the one in the
`ate-env-api-token` secret of the `ate-env` namespace. Substrate keeps an environment until it is deleted, so the run deletes
its environments when an episode ends and when it is stopped with SIGTERM. A run that is killed outright leaves them behind.

## Run it

Locally, against a tinyrl gateway, with podman and a judge that speaks the OpenAI API:

```bash
LAB_ROOT=/path/to/harvey-labs PYTHONPATH=$LAB_ROOT OPENAI_BASE_URL=http://judge/v1 OPENAI_API_KEY=unused \
  python lab_rl.py base_url=http://localhost:18000 steps=35 eval_every=5
```

In a cluster, build the image from this folder with the Substrate client beside the files:

```bash
cp -r examples/lab build && cp -r <sandboxenv-rl>/clients/python build/ate-env-client && docker build build
```

The client must be the commit `ate-env-api` was built from.

Settings are `name=value` arguments, the fields of `Config` at the top of `lab_rl.py`. The ones that shape a run:

| Setting | Default | |
|---|---|---|
| `model` | `google/gemma-4-E4B-it` | Gemma 4, or a Qwen 3.5 family model. |
| `batch`, `group`, `steps` | 8, 6, 1 | Tasks per step, episodes per task, steps. |
| `eval_every`, `eval_tasks` | 0, 50 | Held-out tasks run before every so many steps and after the last. |
| `traj_tokens`, `turn_tokens`, `max_turns` | 32768, 8192, 40 | An episode's context, a turn's length, and its turns. |
| `substrate_endpoint`, `substrate_image`, `substrate_idle` | | The sandboxes, as above. |
| `out`, `run` | `out`, `lab` | The cookbook's logs and checkpoints, and the episodes as files under `out/trajectories/<run>`. |

Two things tinyrl takes from the session rather than from the recipe: the trainer's width, as `TINKER_TAGS=openrl.gpus=4` in
the environment, and the API key, as `TINKER_API_KEY`.

What the cookbook's loop does not give: it samples through one sampling client, so a step's episodes share one sampler, and a
run that is restarted resumes from the cookbook's checkpoint, which needs a gateway that answers `weights_info`.

## What has been run

- **Two steps on Agent Substrate, with the held-out tasks before each and after the last.** Gemma 4 E4B, 2 tasks × 2, 2 held-out
  tasks, 65,536 token episodes, a 5 second idle limit, tinker-cookbook 0.6.1 and the Tinker SDK 0.39.0. All 14 episodes ran,
  none failed, and no suspend failed. Held-out reward was 0.096, then 0.325, then 0.329 after the last step;
  with two tasks that is noise, not learning. Both training steps ran, on 94,286 and 110,858 tokens, and the cookbook's check
  that the trainer sees the tokens the sampler produced (the KL between their log-probabilities) read 0.0007 both times.
  Checkpoints were saved after each step.
- **A scripted rollout, through the cookbook's runner, on real sandboxes.** A policy that says fixed turns in place of a model:
  an episode that looks, writes, checks and stops becomes one training sequence over all its tokens with the advantage on
  exactly the sampled ones; each call is given the smaller of a turn's tokens and the room left; an episode ends itself on its
  turn limit, a cut-off turn and a tool call that cannot be read; and one whose sampler fails leaves no sandbox behind.
- **This folder's image, one step with the held-out tasks before and after.** Same settings: eval reward 0.28, the training
  step on 172,298 tokens with a sampler-trainer KL of 0.0004, the last eval, exit 0. (The image of the commit before the
  search script was removed; the scripted rollout above was rerun after.)
- **`substrate.py` against open-rl's Substrate sandbox, on the same task.** 23 tool calls: glob, grep in three modes, read of a
  docx, a missing file, write, edit, a write to the read-only documents, bash with an exit code, a bash timeout, and searches
  of files the episode made with bash, of a file it deleted, and of an output folder holding a link to `/etc/passwd`. 22 gave
  the same result, with the same tool metrics and the same files brought home. The other was a grep count whose lines came in
  a different order: LAB's grep runs on the host here and in the sandbox there, and neither sorts. The link leaked nothing,
  and was left behind when the outputs came home; open-rl's sandbox fails to bring its outputs home when one is there.
- **Tasks and prompt against open-rl's recipe.** The 50 held-out tasks and the first 300 training tasks are the same, in the
  same order. For Gemma 4 E4B the first prompts of three tasks are the same tokens (6,460, 6,270 and 6,420); for Qwen 3.8
  they differ in how the tools are declared, as said above.
- **Before the move onto the cookbook's loop, the same episode logic as a plain script: 35 steps of 8 tasks × 6 on 4 B200s.**
  Gemma 4 E4B, with sandboxes from an Agent Sandbox warm pool. Reward on the 50 held-out tasks went from 0.087 to 0.380, and
  the pass rate from 0.168 to 0.403. Its training tasks were in a slightly different order from position 127 on, since fixed.

Not done: a full run of this version, resuming a run from the cookbook's checkpoint, and Qwen at 262,144 tokens on main, which
needs a trainer that splits a sequence over GPUs.
