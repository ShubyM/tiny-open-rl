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
redis-server &                                  # Redis 8.4 or newer
python -m tinyrl.api.gateway &                      # the Tinker HTTP API, on :8000
python -m tinyrl.control.scheduler &                    # decides whose turn it is on each shared GPU
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
and starts them again if they die. A sampler starts its engine on its first command.
The engine is vLLM, `vllm serve` on a local port, unless `ENGINE` names another.

Trainers on one GPU take turns, granted by the scheduler. Between batches, a trainer with nothing to do, or
one that has had the GPU for 30 s, yields it to a trainer that waits: it copies its weights, gradients and AdamW state to pinned host memory,
frees the GPU, and copies them back on its next turn.

### Checkpoints

`CKPT` is where checkpoints live. On one machine it is a folder (`/tmp/tinyrl` unless set), and nothing leaves the
machine. Set it to a bucket (`gs://...`, `s3://...`) to keep checkpoints past a machine, or to share them across
machines. A save still lands in the node's `CKPT_CACHE` first and replies, and the node daemon uploads it behind, so a
bucket never holds up a step. A sampler on the node reads the cached copy; one on another node waits for the upload.

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

Every training call is a command on its run's Redis stream, and its reply is a stream with one entry, named by the request. That's the
whole design. Redis does the bookkeeping that would otherwise take a database, a ledger and a heartbeat.

* **The queue reads itself.** A worker's trainer reads its runs' streams through one consumer group so `XINFO GROUPS` says what's queued,
  and `XPENDING` says what's in flight, on which trainer, and for how long. `GET /api/v1/runs/<id>` just asks.
* **A dead trainer fails fast.** The commands it took stay in the pending list. The next trainer to start claims them with `XAUTOCLAIM`
  and fails them, so no client waits on work that died with its process.
* **A stopping trainer saves its runs.** On SIGTERM a trainer finishes its batch, saves every run's weights and AdamW state
  to the store, and the worker's next trainer, on any node, restores them before running their next command.
* **Retries are free.** `XREAD` from id `0` blocks until the reply is written and never consumes it, so
  `retrieve_future` can be retried, or waited on by many clients, and every one of them gets the same answer.
  A call the SDK retries with the same `seq_id` runs once, and every attempt gets the first attempt's answer.
* **Deleted runs take no work.** A run is live exactly while its stream exists, so `XADD NOMKSTREAM` turns a call on a deleted run into a 404.
* **Replies land atomically.** The reply, the acknowledgement and the run's step counter are written in one transaction.
* **One wire format.** `forward_backward` stays protobuf from the SDK to the trainer, which reads the tensors straight out of the bytes.
  Older SDKs' JSON is turned into the same protobuf at the gateway, and JSON clients get the output decoded back to JSON.
* **One trainer.** A base model, with a PEFT adapter per run or the whole model for a single run, behind `create`, `forward_backward`,
  `optim_step`, `state`, `weights`, `restore` and `delete`. Logprobs are computed `CHUNK` tokens at a time under checkpointing, so the
  `[tokens, vocab]` logits never exist all at once. The optimizer updates fp32 master weights and keeps fp32 AdamW state, while the
  forward pass runs in bf16 under autocast. A LoRA run's frozen base model stays bf16.
* **Portable checkpoints.** Training checkpoints (`save_weights`) and sampler weights (`save_weights_for_sampler`) go through
  [store.py](tinyrl/storage/store.py) behind `put(uri)` and `get(uri)` as pure `safetensors` (no `torch.save` pickle), so any trainer can
  resume a run with its exact fp32 weights and AdamW state. `CKPT` can be a bucket (`gs://...`, with `pip install .[gcs]`), so
  every node sees every checkpoint.
* **One owner per worker.** A trainer serves a worker only while it holds the worker's lease. A WATCH/MULTI/EXEC transaction on the lease key checks the lease before any reply lands.

## Reading the code

Start with the docstring at the top of [bus.py](tinyrl/bus.py), which explains the streams. Then read [gateway.py](tinyrl/api/gateway.py)
to see how each Tinker call becomes a command, [store.py](tinyrl/storage/store.py) to see how checkpoints are stored, and
[trainer.py](tinyrl/workers/trainer.py) to see how a command runs on the model.

| file | lines | what's in it |
|---|---:|---|
| [tinyrl/bus.py](tinyrl/bus.py) | 321 | runs, sessions, commands, replies, leases and the event log on Redis |
| [tinyrl/helpers.py](tinyrl/helpers.py) | 12 | `getenv` |
| [tinyrl/api/gateway.py](tinyrl/api/gateway.py) | 303 | the Tinker HTTP API |
| [tinyrl/api/wire.py](tinyrl/api/wire.py) | 155 | the Tinker protobuf, to numpy and back |
| [tinyrl/control/scheduler.py](tinyrl/control/scheduler.py) | 258 | folds the event log, places workers on nodes and grants GPU turns |
| [tinyrl/control/node_daemon.py](tinyrl/control/node_daemon.py) | 127 | runs the workers placed on its node, uploads its saves and sweeps its checkpoints |
| [tinyrl/control/turns.py](tinyrl/control/turns.py) | 43 | a worker's side of taking turns on a GPU |
| [tinyrl/control/estimator.py](tinyrl/control/estimator.py) | 35 | how much GPU memory a worker needs |
| [tinyrl/workers/trainer.py](tinyrl/workers/trainer.py) | 221 | the trainer loop: commands in, replies out |
| [tinyrl/workers/model.py](tinyrl/workers/model.py) | 307 | the losses, token budget batching, and the model a trainer serves |
| [tinyrl/workers/sampler.py](tinyrl/workers/sampler.py) | 215 | sampling workers, through an engine that knows nothing of Redis |
| [tinyrl/workers/vllm.py](tinyrl/workers/vllm.py) | 151 | the vLLM engine: `vllm serve` over HTTP |
| [tinyrl/storage/store.py](tinyrl/storage/store.py) | 395 | `safetensors` checkpoints behind `put(uri)` and `get(uri)`, on a folder or a bucket |
| [tinyrl/storage/registry.py](tinyrl/storage/registry.py) | 69 | the checkpoints the store holds, their owners, TTLs and versions |
| [tinyrl/storage/models.py](tinyrl/storage/models.py) | 55 | base models from the hub, a bucket or a folder |

`python sz.py` counts lines of code the way tinygrad does. Settings are environment variables: `REDIS_URL`, `PORT`, `MODEL_CACHE` (where a node keeps base models it fetched), `MODEL_MIRROR` (a folder or bucket nodes read hub models from first), `CKPT` (the
checkpoint directory or bucket the trainers and samplers share), `CKPT_CACHE` (where a node keeps the checkpoints it saved or downloaded, in front of a bucket `CKPT`), `CKPT_CACHE_BYTES` (the most it keeps, least recently read first, 0 for half its disk), `KEYFRAME_EVERY`, `TOKEN_BUDGET` (padded tokens per pass; unset, each call sizes its passes to the free
GPU memory), `CHUNK`, `GRAD_CKPT`, `WAIT`, `REPLY_TTL`, `SESSION_TTL`, `LEASE_MS`, `GPUS`, `QUANTUM`, `IDLE_MS`,
`ENGINE`, `VLLM_ARGS` and `DEBUG=1`.

## Telemetry

Every run's hash counts the commands it ran (`cmds`), the time they spent queued (`wait_ms`, from the stream entry id to the
trainer starting it) and running (`busy_ms`), plus `steps` and `failed`. The trainer writes them in the same transaction as
the reply. `GET /api/v1/runs/<run>` shows one run's counters, and `GET /metrics` shows every run's in Prometheus' text
format, as `tinyrl_<counter>_total{worker, run}` in seconds. The numbers scheduling papers report are queries over those
counters:

| measure | PromQL |
|---|---|
| utilization of a worker's trainer | `sum by (worker) (rate(tinyrl_busy_seconds_total[5m]))` |
| mean queueing delay per call | `sum by (worker) (rate(tinyrl_wait_seconds_total[5m])) / sum by (worker) (rate(tinyrl_cmds_total[5m]))` |
| slowdown, (wait + run) / run | `1 + sum by (worker) (rate(tinyrl_wait_seconds_total[5m])) / sum by (worker) (rate(tinyrl_busy_seconds_total[5m]))` |
| throughput, optimizer steps per second | `sum by (worker) (rate(tinyrl_steps_total[5m]))` |
| fairness, Jain's index of the GPU time active runs get (1 is equal shares) | `sum(rate(tinyrl_busy_seconds_total[5m]) > 0) ^ 2 / (count(rate(tinyrl_busy_seconds_total[5m]) > 0) * sum((rate(tinyrl_busy_seconds_total[5m]) > 0) ^ 2))` |
| failure rate | `sum by (worker) (rate(tinyrl_failed_total[5m])) / sum by (worker) (rate(tinyrl_cmds_total[5m]))` |

A run's wait includes waiting on its own earlier calls, since a run's calls run in order.

## Throughput simulator

The [interactive simulator](simulator/README.md) explores GPU switching, arrival rates, queue growth, and data-transfer bottlenecks.
Open `simulator/index.html` in a browser; it runs locally without installing dependencies.

## Test

```sh
python -m unittest discover -s test -t .                             # starts its own redis-server, or set REDIS_URL
TINKER_PYTHON=$(which python) python -m unittest test.test_tinker   # the real SDK, against redis, the gateway and a trainer as processes
```

Every test runs against a real `redis-server` and a real (tiny, random) Llama. Nothing is faked.

## Not here yet

* `create_training_client_from_state`.
* Multi-GPU trainers (FSDP, context parallelism).
* Moving runs between live nodes to balance load, parking idle runs, and the `/dev/shm` and peer checkpoint tiers. The
  design is in [docs/kubernetes.md](docs/kubernetes.md).
