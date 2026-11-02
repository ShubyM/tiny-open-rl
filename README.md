# tiny-open-rl

The [Tinker](https://thinkingmachines.ai/tinker/) training API, built on Redis Streams and PyTorch.

The Tinker SDK trains through it unchanged, for LoRA adapters and full fine-tunes alike:

```python
client = tinker.ServiceClient(base_url="http://localhost:8000")
trainer = client.create_lora_training_client(base_model="Qwen/Qwen3-0.6B", rank=8)

for batch in data:
  trainer.forward_backward(batch, "importance_sampling")
  trainer.optim_step(types.AdamParams(learning_rate=4e-5))

sampler_path = trainer.save_weights_for_sampler("step-100").result().path  # a PEFT adapter vLLM can load
```

`base_model` is a hub id, or a folder of `config.json` and `*.safetensors` anywhere fsspec reaches (`gs://...`, `s3://...`,
`/mnt/models/...`). A node fetches a bucket model once into `MODEL_CACHE`. See [models.py](tinyrl/storage/models.py).

## Run it

```sh
pip install -e .
redis-server &                                          # Redis 8.4 or newer
python -m tinyrl.api.gateway &                          # the Tinker HTTP API, on :8000
python -m tinyrl.control.scheduler &                    # assigns runs, places workers and grants GPU turns
python -m tinyrl.control.node_daemon                    # runs a trainer or sampler for each worker placed on it
```

A **worker** is one trainer or sampler process and the runs it serves. The gateway records what a run needs, and
the scheduler picks its worker: LoRA runs on the same base model share a trainer, sampling sessions on the same base model and
kind share a sampler, and an exclusive run's sessions share a sampler of their own. A full fine-tune
(`user_metadata={"openrl.fine_tuning_type": "full"}`) or an exclusive run (`"openrl.exclusive": "true"`) gets a worker of
its own. Worker names encode nothing.

Run one node daemon per GPU machine (`GPUS`, default 1) and one scheduler. The scheduler places each worker that has runs
on the node daemon with the fewest workers per GPU, on GPUs it picks, and moves a dead node daemon's workers to live ones.
Each node daemon runs `python -m tinyrl.workers.trainer <worker>` or `python -m tinyrl.workers.sampler <worker>` for the workers placed on it,
and starts them again if they die. On multi-GPU nodes, a sampler starts its engine at launch. On a single GPU, it starts on its first command.
The engine is vLLM, `vllm serve` on a local port, unless `ENGINE` names another.

Trainers on one GPU take turns, granted by the scheduler. Between commands, a trainer with nothing to do, or
one that has had the GPU for 30 s, yields it to a worker that waits: it copies its weights, gradients and AdamW state to pinned host memory,
frees the GPU, and copies them back on its next turn. A command already running finishes before a revoke takes effect;
the quantum bounds turns at command boundaries, not the duration of an individual backward pass.

### Checkpoints

`CKPT` is where checkpoints live. On one machine it is a folder (`/tmp/tinyrl` unless set), and nothing leaves the
machine. Set it to a bucket (`gs://...`, `s3://...`) to keep checkpoints past a machine, or to share them across
machines. A save still lands in the node's `CKPT_CACHE` first and replies, and the node daemon uploads it behind, so a
bucket never holds up a step. A sampler on the node reads the cached copy; one on another node waits for the upload.
The reply means the local save is complete. Losing that node before upload finishes can lose the checkpoint.

Each save has an immutable version. The worker publishes that version in Redis with its reply under the same lease
check, and readers request that exact version. Uploads write separate version manifests, so a late upload cannot
replace a newer save. Historical versions remain readable until the logical checkpoint is deleted. A sampler resolves
and pins its checkpoint files before asking for a GPU turn; cache trimming cannot remove files while they are loading.

A full fine-tune's sampler weights change in few bytes a step, so they upload as the bytes that changed since the save
before, in chains of at most `KEYFRAME_EVERY` (8) saves that start with a whole one. Another node rebuilds a save from
the newest one it has and checks each chunk against the CRC the saving node took. The node daemons delete the links no registered checkpoint needs.

### On Kubernetes

[k8s/](k8s) runs Redis with an append-only file, the gateway, the scheduler, and a node daemon on every node labeled `tinyrl-gpu=true`.
Build the image from [Dockerfile](Dockerfile), point `CKPT` in [k8s/kustomization.yaml](k8s/kustomization.yaml) at a
bucket the nodes' service account can write, and run `kubectl apply -k k8s/`. A drained node's trainers save their runs,
and a dead node's workers move to live nodes. `python k8s/e2e.py` breaks a running cluster on purpose and checks it
survives.

## How it works

```
sdk --http--> gateway --XADD--> tinyrl:q:<run> --XREADGROUP--> trainer --XADD--> tinyrl:f:<run>/<id> --XREAD--> gateway --> sdk
```

Every training or sampling call is a command on its run's Redis stream, and its reply is a stream with one entry, named by the request. That's the
whole design. Redis does the bookkeeping that would otherwise take a database, a ledger and a heartbeat.

* **The queue reads itself.** A worker's trainer reads its runs' streams through one consumer group so `XINFO GROUPS` says what's queued,
  and `XPENDING` says what's in flight, on which trainer, and for how long. `GET /api/v1/runs/<id>` just asks.
* **A dead trainer fails fast.** The commands it took stay in the pending list. The next trainer to start claims them with `XAUTOCLAIM`
  and fails them, so no client waits on work that died with its process.
* **A stopping trainer saves its runs.** On SIGTERM a trainer finishes its current command, saves every run's weights,
  accumulated gradients and AdamW state, and returns its unstarted commands without changing their IDs. The next trainer
  restores the saved state before executing them. Explicit client loads clear accumulated gradients.
* **Retries are free.** `XREAD` from id `0` blocks until the reply is written and never consumes it, so
  `retrieve_future` can be retried, or waited on by many clients, and every one of them gets the same answer.
  Admission atomically records each request identity with its command. A repeated `seq_id` returns the same future,
  including after that future's reply expires; it never enqueues the mutation again. An expired or unknown future returns
  HTTP 404, which terminates SDK polling. Request identities live for the run's lifetime, then `SESSION_TTL` after deletion.
* **Deleted runs take no work.** A run is live exactly while its stream exists, so `XADD NOMKSTREAM` turns a call on a deleted run into a 404.
* **Replies land atomically.** The reply, acknowledgement, counters and any saved checkpoint registration are written in
  one transaction that checks the worker's current owner. A stale worker can leave unpublished files, but cannot publish
  them or acknowledge commands.
* **One wire format.** `forward_backward` stays protobuf from the SDK to the trainer, which reads the tensors straight out of the bytes.
  Older SDKs' JSON is turned into the same protobuf at the gateway, and JSON clients get the output decoded back to JSON.
* **One event log.** The gateway, node daemons and workers append what happened to `tinyrl:events`. Each reader folds events
  into its own state; only speculative decisions copy it. The scheduler saves a state and stream cursor before trimming
  the covered prefix. Restarted or lagging readers resume from that snapshot and its tail. GPU events name the process
  instance, so an old process cannot release a replacement's grant.
* **One trainer.** A base model, with a PEFT adapter per run or the whole model for a single run, behind `create_run`, `forward_backward`,
  `optim_step`, `training_state`, `sampler_weights`, `restore_run` and `delete_run`. Logprobs are computed `CHUNK` tokens at a time under checkpointing, so the
  `[tokens, vocab]` logits never exist all at once. The optimizer updates fp32 master weights and keeps fp32 AdamW state, while the
  forward pass runs in bf16 under autocast. A LoRA run's frozen base model stays bf16.
* **Portable checkpoints.** Training checkpoints (`save_weights`) and sampler weights (`save_weights_for_sampler`) go through
  [store.py](tinyrl/storage/store.py) behind `put(uri)` and `get(uri)` as pure `safetensors` (no `torch.save` pickle), so any trainer can
  resume a run with its exact fp32 weights and AdamW state. `CKPT` can be a bucket (`gs://...`, with `pip install .[gcs]`), so
  every node sees every checkpoint.
* **One owner per worker.** A trainer or sampler serves a worker only while it holds the worker's lease. A WATCH/MULTI/EXEC
  transaction checks the lease before any reply or checkpoint registration lands. Recovery also checks the last owner:
  an expired lease can be retaken only if no replacement acquired it in between. That small ownership record is retained
  after a worker ends; expiring it would let an arbitrarily delayed predecessor resume stale state.
* **Failures cannot silently become training state.** Inputs and complete checkpoint shapes are checked before mutation.
  If backward, an optimizer step, or checkpoint installation fails after mutation begins, the affected in-memory run is
  removed. Its next training operation requires an explicit checkpoint load; other LoRA runs remain usable.

### Upgrading from the earlier queue format

Drain the old command queues and start fresh SDK sessions when upgrading this version. Admission deduplication and
the reserved consumer for returned, unstarted commands change the Redis queue format; mixed old and new workers are
not supported. Previously saved checkpoints remain readable.

## Reading the code

Start with the docstring at the top of [bus.py](tinyrl/bus.py), which explains the streams. Then read [gateway.py](tinyrl/api/gateway.py)
to see how each Tinker call becomes a command, [scheduler.py](tinyrl/control/scheduler.py) to see how the event log folds into placements and turns,
[store.py](tinyrl/storage/store.py) to see how checkpoints are stored, and [trainer.py](tinyrl/workers/trainer.py) and
[model.py](tinyrl/workers/model.py) to see how a command runs on the model.

| file | lines | what's in it |
|---|---:|---|
| [tinyrl/bus.py](tinyrl/bus.py) | 379 | runs, sessions, commands, replies, leases, node heartbeats and `tinyrl:events` on Redis |
| [tinyrl/helpers.py](tinyrl/helpers.py) | 12 | typed `getenv` and the `DEBUG` flag |
| [tinyrl/__init__.py](tinyrl/__init__.py) | 1 | top-level `Bus`, `Cmd`, `Reply` and `Run` exports |
| [tinyrl/api/gateway.py](tinyrl/api/gateway.py) | 333 | the Tinker HTTP routes for sessions, runs, checkpoints and Prometheus metrics |
| [tinyrl/api/wire.py](tinyrl/api/wire.py) | 162 | Tinker requests and replies between protobuf, numpy and JSON |
| [tinyrl/control/scheduler.py](tinyrl/control/scheduler.py) | 315 | folds the event log to assign runs, place workers and grant GPU turns |
| [tinyrl/control/node_daemon.py](tinyrl/control/node_daemon.py) | 143 | heartbeats, runs the workers placed on its node, uploads saves and sweeps checkpoints |
| [tinyrl/control/turns.py](tinyrl/control/turns.py) | 58 | `GpuTurn`, a worker's side of requesting, holding and yielding GPUs |
| [tinyrl/workers/trainer.py](tinyrl/workers/trainer.py) | 243 | the trainer loop for leases, queued commands, GPU turns and SIGTERM saves |
| [tinyrl/workers/model.py](tinyrl/workers/model.py) | 383 | the PyTorch and PEFT model, RL losses, token-budget batching and host parking |
| [tinyrl/workers/sampler.py](tinyrl/workers/sampler.py) | 209 | batches sampling calls, loads checkpoints and takes turns through a Redis-free `Engine` |
| [tinyrl/workers/vllm.py](tinyrl/workers/vllm.py) | 212 | drives `vllm serve` over HTTP for completions, adapters, weight reloads and sleep |
| [tinyrl/workers/snapshot.py](tinyrl/workers/snapshot.py) | 120 | CRIU snapshots of an asleep vLLM, restored in seconds instead of a boot |
| [tinyrl/storage/store.py](tinyrl/storage/store.py) | 454 | `safetensors` `put` and `get` with local `CKPT_CACHE`, bucket uploads and delta chains |
| [tinyrl/storage/registry.py](tinyrl/storage/registry.py) | 72 | saved checkpoints on Redis with their owners, sizes, versions and TTLs |
| [tinyrl/storage/models.py](tinyrl/storage/models.py) | 55 | base models from the hub, a bucket or a folder, cached in `MODEL_CACHE` |
| **total** | **3151** | counted by [sz.py](sz.py), excluding generated `tinyrl/api/proto` |

`python sz.py` counts token-bearing lines the way tinygrad does, and CI caps `tinyrl/` at 3,200 lines. The increase from
3,000 covers the acceptance, ownership, restore and retention checks; the unused estimator and duplicate worker/cache
bookkeeping have been removed. Outside `tinyrl/`,
[examples/tinker_sdk.py](examples/tinker_sdk.py) trains and samples through the unmodified SDK. The rest of the repo holds
[k8s/](k8s), the browser [simulator/](simulator/README.md), [docs/kubernetes.md](docs/kubernetes.md), and the integration suites in [test/](test).

Settings are environment variables: `REDIS_URL`, `PORT`, `MODEL_CACHE` (where a node keeps base models it fetched), `MODEL_MIRROR` (a folder or bucket nodes read hub models from first), `CKPT` (the
checkpoint directory or bucket the trainers and samplers share), `CKPT_CACHE` (where a node keeps the checkpoints it saved or downloaded, in front of a bucket `CKPT`), `CKPT_CACHE_BYTES` (the most it keeps, least recently read first, 0 for half its disk), `KEYFRAME_EVERY`, `TOKEN_BUDGET` (padded tokens per pass; unset, each call sizes its passes to the free
GPU memory), `CHUNK`, `GRAD_CKPT`, `WAIT`, `REPLY_TTL`, `SESSION_TTL`, `SESSION_GRACE`, `LEASE_MS`, `GPUS`, `QUANTUM`, `IDLE_MS`,
`ENGINE`, `VLLM_ARGS`, `CKPT_TTL` (the TTL, at least 3600 s, of a checkpoint saved without one; 0 keeps it until deleted), `TINYRL_SNAPSHOTS` (where a node keeps vLLM snapshots; unset, samplers boot vLLM every time) and `DEBUG=1`.

## Telemetry

Every run's hash counts the commands it ran (`cmds`), the time they spent queued (`wait_ms`, from the stream entry id to the
trainer starting it) and running (`busy_ms`), plus `steps` and `failed`. The trainer writes them in the same transaction as
the reply. `GET /api/v1/runs/<run>` shows one run's counters, and `GET /metrics` shows every run's in Prometheus' text
format, as `tinyrl_<counter>_total{worker, run}`. Time counters are exported in seconds. `busy` is command wall time,
including loading and saving; concurrent sampler calls can overlap, so it is not GPU utilization or allocated service.
These counters support operational queries:

| measure | PromQL |
|---|---|
| command wall seconds per second | `sum by (worker) (rate(tinyrl_busy_seconds_total[5m]))` |
| mean queueing delay per call | `sum by (worker) (rate(tinyrl_wait_seconds_total[5m])) / sum by (worker) (rate(tinyrl_cmds_total[5m]))` |
| slowdown, (wait + run) / run | `1 + sum by (worker) (rate(tinyrl_wait_seconds_total[5m])) / sum by (worker) (rate(tinyrl_busy_seconds_total[5m]))` |
| throughput, optimizer steps per second | `sum by (worker) (rate(tinyrl_steps_total[5m]))` |
| failure rate | `sum by (worker) (rate(tinyrl_failed_total[5m])) / sum by (worker) (rate(tinyrl_cmds_total[5m]))` |

A run's wait includes waiting on its own earlier calls, since a run's calls run in order.
Per-user fairness needs service accounting by owner and the full set of runnable runs, including those that received
zero service. A Jain index over only runs with positive `busy` time hides starvation, so it is not reported here.

## Throughput simulator

The [interactive simulator](simulator/README.md) explores GPU switching, arrival rates, queue growth, and data-transfer bottlenecks.
Open `simulator/index.html` in a browser; it runs locally without installing dependencies.

## Test

```sh
python -m unittest discover -s test -t .                             # starts its own redis-server, or set REDIS_URL
TINKER_PYTHON=$(which python) python -m unittest test.test_tinker   # runs examples/tinker_sdk.py against redis, gateway, scheduler and node_daemon
```

Integration tests use a real `redis-server`, real tiny random Llama models, and the public Tinker SDK. Engine and process
unit tests use fakes; CUDA parking tests require a GPU. CPU tests do not establish vLLM throughput or cluster fairness.

## Not here yet

* `create_training_client_from_state`.
* Multi-GPU trainers (FSDP, context parallelism).
* Capacity-aware worker replication, host-memory admission, and per-owner fair service accounting. Compatible LoRA
  runs currently join the same worker; worker turns alone do not guarantee equal service per user.
* Moving runs between live nodes to balance load, parking idle runs, and the `/dev/shm` and peer checkpoint tiers. The
  design is in [docs/kubernetes.md](docs/kubernetes.md).
