# LAB on tinyrl

Reinforcement learning on [Harvey LAB](https://github.com/ShubyM/harvey-labs), a benchmark of legal tasks: the model reads a
matter's documents with tools, writes the deliverable, and a judge model grades it against the task's criteria. One step samples
`GROUP` episodes for each of `BATCH` tasks and trains on every sampled token with advantage = reward − its task's mean.

| File | What it is |
|---|---|
| `lab_rl.py` | The run: episodes, grading, training, evals on held-out tasks, state to start again from. |
| `replay.py` | Trains on saved episodes again, to time the trainer alone. |
| `traj.py` | Episodes as files a person or an agent can read: export, list, show. |
| `Dockerfile` | The run as an image for the cluster. |

It is a plain script on the Tinker SDK. The tasks, tools and grader are LAB's. The task split, the prompt and the sandbox
clients are those of open-rl's `examples/harvey_labs` recipe. That recipe also runs against tinyrl as it is, except that
resuming a run needs `weights_info` (#108).

## Sandboxes

An episode's tools run in one of three places, chosen by the environment:

| Set | Sandbox |
|---|---|
| `SUBSTRATE_ENDPOINT`, `SUBSTRATE_IMAGE`, `SUBSTRATE_ENV_AUTH_TOKEN` | An [Agent Substrate](https://github.com/agent-substrate/substrate) environment through `ate-env-api`. It is suspended after `SUBSTRATE_IDLE` seconds (30) without a tool call, so it holds no worker while the model generates, and the next tool call resumes it. |
| `SANDBOX_WARMPOOL`, `KUBECONFIG` | A sandbox claimed from an Agent Sandbox warm pool. |
| neither | LAB's own podman container, on the machine the script runs on. |

`SUBSTRATE_IMAGE` is LAB's sandbox image pinned by digest; Substrate runs it unmodified. The token is the one in the
`ate-env-api-token` secret of the `ate-env` namespace.

## Run it

Locally, against a tinyrl gateway, with podman and a judge that speaks the OpenAI API:

```bash
LAB_ROOT=/path/to/harvey-labs PYTHONPATH=$LAB_ROOT BASE_URL=http://localhost:18000 \
OPENAI_BASE_URL=http://judge/v1 OPENAI_API_KEY=unused python lab_rl.py
```

In a cluster, build the image from a folder with the three things it copies:

```bash
mkdir build && cp -r <open-rl>/examples build/examples && cp -r <sandboxenv-rl>/clients/python build/ate-env-client \
  && cp -r examples/lab build/tinyrl_lab
docker build -f examples/lab/Dockerfile build
```

The Substrate client must be the commit `ate-env-api` was built from. open-rl's lock pins it.

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

- **Gemma 4 E4B, 35 steps of 8 tasks × 6, on 4 B200s, Agent Sandbox pool.** Reward on the 50 held-out tasks went from 0.087
  to 0.380, and the pass rate from 0.168 to 0.403.
- **Gemma 4 E4B, one step of 2 tasks × 2, on Agent Substrate.** 4 episodes, none failed, each suspended 7 to 11 times with a
  5 second idle limit, 2 deliverables graded, and the training step ran.
- **24 Substrate sandboxes at once, without a model.** A sandbox with a task's documents in it took 2.1 s (median), a tool
  call 0.35 s, and a tool call that resumes a suspended sandbox 0.93 s.
- **Qwen3.8-27B with 262,144 token episodes.** A run is in progress. It needs a trainer that splits a sequence over GPUs,
  which is not in tinyrl yet.

Not done: a full run on Agent Substrate, and this recipe on the abstractions of `tinker-cookbook`.
