# LAB on tinyrl

Reinforcement learning on [Harvey LAB](https://github.com/ShubyM/harvey-labs), a benchmark of legal tasks: the model reads a
matter's documents with tools, writes the deliverable, and a judge model grades it against the task's criteria. One step samples
`GROUP` episodes for each of `BATCH` tasks and trains on every sampled token with advantage = reward − its task's mean.

| File | What it is |
|---|---|
| `lab_rl.py` | The run: episodes, grading, training, evals on held-out tasks, state to start again from. |
| `substrate.py` | An episode's sandbox as an Agent Substrate environment. |
| `replay.py` | Trains on saved episodes again, to time the trainer alone. |
| `traj.py` | Episodes as files a person or an agent can read: export, list, show. |
| `Dockerfile`, `requirements.txt` | The run as an image for the cluster. |

It is a plain script on the Tinker SDK, with nothing but LAB, the SDK, a tokenizer and the Substrate client under it. The tasks,
tools and grader are LAB's. The held-out tasks, the order of the training tasks and the first prompt are the same as in
open-rl's `examples/harvey_labs` recipe, so a run here and a run there can be compared; no code is shared with it.

## Sandboxes

An episode's tools run in one of two places:

| Set | Sandbox |
|---|---|
| `SUBSTRATE_ENDPOINT`, `SUBSTRATE_IMAGE`, `SUBSTRATE_ENV_AUTH_TOKEN` | An [Agent Substrate](https://github.com/agent-substrate/substrate) environment through `ate-env-api`. It is suspended after `SUBSTRATE_IDLE` seconds (30) without a tool call, so it holds no worker while the model generates, and the next tool call resumes it. |
| neither | LAB's own podman container, on the machine the script runs on. |

`SUBSTRATE_IMAGE` is LAB's sandbox image pinned by digest; Substrate runs it unmodified. The token is the one in the
`ate-env-api-token` secret of the `ate-env` namespace. Substrate keeps an environment until it is deleted, so the run deletes
its environments when an episode ends and when it is stopped with SIGTERM. A run that is killed outright leaves them behind.

## Run it

Locally, against a tinyrl gateway, with podman and a judge that speaks the OpenAI API:

```bash
LAB_ROOT=/path/to/harvey-labs PYTHONPATH=$LAB_ROOT BASE_URL=http://localhost:18000 \
OPENAI_BASE_URL=http://judge/v1 OPENAI_API_KEY=unused python lab_rl.py
```

In a cluster, build the image from this folder with the Substrate client beside the files:

```bash
cp -r examples/lab build && cp -r <sandboxenv-rl>/clients/python build/ate-env-client && docker build build
```

The client must be the commit `ate-env-api` was built from.

The settings are the names at the top of `lab_rl.py`, read from the environment. The ones that shape a run:

| Setting | Default | |
|---|---|---|
| `MODEL` | `google/gemma-4-E4B-it` | Gemma 4, or a Qwen 3.5 family model. |
| `GPUS` | 1 | The trainer's width. |
| `SAMPLERS` | 1 | Sampler workers that a step's episodes are dealt over. More than one needs a tinyrl that places sampler replicas. |
| `BATCH`, `GROUP`, `STEPS` | 8, 6, 1 | Tasks per step, episodes per task, steps. |
| `EVAL_EVERY`, `EVAL_TASKS` | 0, 50 | Held-out tasks run before every so many steps and after the last. |
| `TRAJ_TOKENS`, `TURN_TOKENS`, `MAX_TURNS` | 32768, 8192, 40 | An episode's context, a turn's length, and its turns. |
| `EXPORT` | | `"<run> <name> <fact>;<fact>"` writes each step as trajectory files under `OUT/trajectories`. |

## What has been run

- **This folder's image, one step on Agent Substrate.** Gemma 4 E4B, 2 tasks × 2, 65,536 token episodes, a 5 second idle limit:
  4 episodes, none failed, each suspended 1 to 7 times, one deliverable written and graded, and the training step ran on its
  task's two episodes. Three episodes wrote nothing, which an untrained E4B often does.
- **`substrate.py` against open-rl's Substrate sandbox, on the same task.** 17 tool calls (glob, grep in three modes, read of a
  docx, a missing file, write, edit, a write to the read-only documents, bash with an exit code, a bash timeout) gave the same
  results, the same tool metrics and the same files brought home.
- **Tasks and prompt against open-rl's recipe.** The 50 held-out tasks and the first 300 training tasks are the same, in the
  same order, and the first prompt of a task is the same 6,460 tokens for Gemma 4 E4B.
- **An earlier version of `lab_rl.py`, 35 steps of 8 tasks × 6 on 4 B200s.** Gemma 4 E4B, with sandboxes from an Agent Sandbox
  warm pool. Reward on the 50 held-out tasks went from 0.087 to 0.380, and the pass rate from 0.168 to 0.403. Its training
  tasks were in a slightly different order from position 127 on, which has since been fixed.

Not done: a full run with `substrate.py`, and Qwen at 262,144 tokens on main, which needs a trainer that splits a sequence
over GPUs.
