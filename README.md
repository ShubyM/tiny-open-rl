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

## Run it

```sh
pip install -e .
redis-server &                                  # Redis 8.4 or newer
python -m tinyrl.gateway &                      # the Tinker HTTP API, on :8000
python -m tinyrl.scheduler &                    # decides whose turn it is on each shared GPU
python -m tinyrl.node_daemon                    # executes worker placements and confirms process exits
```

A **run** is a training model or a sampling session, with its own command queue. A **worker** is one trainer or sampler
process on fixed GPUs, serving one or more runs; every replacement gets a fresh ID. The gateway records what a run
needs (role, base model, LoRA or full, exclusive), and the scheduler decides the rest with three rules:

* **Which worker.** A LoRA run joins the trainer for its base model, and a sampling session joins the vLLM worker for
  its base model and kind. A full fine-tune (`user_metadata={"openrl.fine_tuning_type": "full"}`) or an exclusive run
  (`"openrl.exclusive": "true"`) gets a worker of its own.
* **Which GPU.** A trainer needs a whole GPU while it computes, a sampler `SAMPLER_MEM` of one (0.4). A new worker goes
  on the least used GPU it fits on, else the least used one.
* **Whose turn.** A worker computes when it fits next to the workers already computing on its GPU, so two samplers
  run at once and trainers take turns. Waiters are served first come first served. A holder gives way when it goes
  idle or after `QUANTUM` (30 seconds) while a waiter doesn't fit; trainers park their tensors in host memory and
  samplers put vLLM to sleep.

Run one node daemon per GPU machine (`GPUS`, default 1) and one scheduler. Samplers use vLLM unless `ENGINE` names
another engine. Worker names and GPU indices encode no policy.

### On Kubernetes

[k8s/](k8s) runs Redis with an append-only file, the gateway, the scheduler, and a node daemon on every node labeled `tinyrl-gpu=true`.
Build the image from [Dockerfile](Dockerfile), point `CKPT` in [k8s/kustomization.yaml](k8s/kustomization.yaml) at a
bucket the nodes' service account can write, and run `kubectl apply -k k8s/`. A drained node's workers save their runs.
After a lost node's owner leases expire, the scheduler retires those worker IDs before assigning replacements elsewhere. `python k8s/e2e.py` breaks a running cluster on purpose and checks it
survives.

## How it works

```
sdk --http--> gateway --XADD--> tinyrl:q:<run> --XREADGROUP--> trainer --XADD--> tinyrl:f:<run>/<id> --XREAD--> gateway --> sdk
```

Every training call is a command on its run's Redis stream, and its reply is a stream with one entry, named by the request. That's the
whole design. Redis does the bookkeeping that would otherwise take a database, a ledger and a heartbeat.

* **The queue reads itself.** A worker reads its assigned runs' streams through one consumer group so `XINFO GROUPS` says what's queued,
  and `XPENDING` says what's in flight, on which trainer, and for how long. `GET /api/v1/runs/<id>` just asks.
* **A dead trainer fails fast.** The commands it took stay in the pending list. The next trainer to start claims them with `XAUTOCLAIM`
  and fails them, so no client waits on work that died with its process.
* **A stopping worker saves its runs.** On SIGTERM a trainer finishes its batch, saves every run's weights and AdamW state
  to the store, and the run's next worker, on any node, restores them before running their next command.
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
  [store.py](tinyrl/store.py) behind `put(uri)` and `get(uri)` as pure `safetensors` (no `torch.save` pickle), so any trainer can
  resume a run with its exact fp32 weights and AdamW state. `CKPT` can be a bucket (`gs://...`, with `pip install .[gcs]`), so
  every node sees every checkpoint.
* **Fenced execution.** Claims, replies and saved-state pointers validate both the process lease and the run assignment.
  Retired worker IDs cannot acquire ownership again. Scheduler decisions compare the event-log tail before committing.
  A late release also has to match its GPU turn token.
* **Recovery assumptions.** A node daemon stops its children when it loses Redis connectivity. Its replacement must have
  exclusive access to the physical GPUs: deploy one daemon per machine/container allocation and terminate the previous
  process group before replacing it. Lease fencing prevents stale commits; it does not prove that remote CUDA work stopped.
  Unsaved state after an abrupt crash is reported as lost, never silently reconstructed.
* **Replay.** Lifecycle facts and Redis projections commit together. The event log is not blindly trimmed; durable snapshot
  compaction is still needed before using this as an indefinitely running control plane. Upgrade this changed log schema
  with drained workers and a fresh control log.

## Reading the code

Start with the docstring at the top of [bus.py](tinyrl/bus.py), which explains the streams. Then read [gateway.py](tinyrl/gateway.py)
to see how each Tinker call becomes a command, [store.py](tinyrl/store.py) to see how checkpoints are stored, and
[trainer.py](tinyrl/trainer.py) to see how a command runs on the model.

The scheduling code is [scheduler.py](tinyrl/scheduler.py): plain run/worker records, event folding, the three rules
and one decision loop. [turns.py](tinyrl/turns.py) implements the worker side of a GPU turn, and
[node_daemon.py](tinyrl/node_daemon.py) executes placements. The Redis key/record field `pool` remains a dispatch-format
compatibility name for a worker ID; there is no separate pool object or pool placement policy.

`python sz.py` counts lines of code the way tinygrad does. Settings are environment variables: `REDIS_URL`, `PORT`, `CKPT` (the
checkpoint directory or bucket the trainers and samplers share), `CKPT_CACHE` (where a node keeps checkpoints it downloaded), `TOKEN_BUDGET` (padded tokens per pass; unset, each call sizes its passes to the free
GPU memory), `CHUNK`, `GRAD_CKPT`, `WAIT`, `REPLY_TTL`, `SESSION_TTL`, `LEASE_MS`, `GPUS`, `QUANTUM`, `SAMPLER_MEM`,
`IDLE_MS`, `ENGINE`, `VLLM_ARGS` and `DEBUG=1`.

## Telemetry

Every run's hash counts the commands it ran (`cmds`), the time they spent queued (`wait_ms`, from the stream entry id to the
trainer starting it) and running (`busy_ms`), plus `steps` and `failed`. The trainer writes them in the same transaction as
the reply. `GET /api/v1/runs/<run>` shows one run's counters, and `GET /metrics` shows every run's in Prometheus' text
format, as `tinyrl_<counter>_total{pool, run}` in seconds. These command counters describe API work, not hardware utilization or training convergence:

| measure | PromQL |
|---|---|
| utilization of a pool's trainer | `sum by (pool) (rate(tinyrl_busy_seconds_total[5m]))` |
| mean queueing delay per call | `sum by (pool) (rate(tinyrl_wait_seconds_total[5m])) / sum by (pool) (rate(tinyrl_cmds_total[5m]))` |
| slowdown, (wait + run) / run | `1 + sum by (pool) (rate(tinyrl_wait_seconds_total[5m])) / sum by (pool) (rate(tinyrl_busy_seconds_total[5m]))` |
| throughput, optimizer steps per second | `sum by (pool) (rate(tinyrl_steps_total[5m]))` |
| failure rate | `sum by (pool) (rate(tinyrl_failed_total[5m])) / sum by (pool) (rate(tinyrl_cmds_total[5m]))` |

A run's wait includes waiting on its own earlier calls, since a run's calls run in order.

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
