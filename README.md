# tiny-open-rl

The [Tinker](https://thinkingmachines.ai/tinker/) training API, with a durable coordinator and PyTorch workers.

The Tinker SDK trains LoRA adapters and full fine-tunes through the same API:

```python
client = tinker.ServiceClient(base_url="http://localhost:8000")
trainer = client.create_lora_training_client(base_model="Qwen/Qwen3-0.6B", rank=8)

for batch in data:
  trainer.forward_backward(batch, "importance_sampling")
  trainer.optim_step(types.AdamParams(learning_rate=4e-5))

sampler_path = trainer.save_weights_for_sampler("step-100").result().path
```

## Run it

Install Redis 7 or newer, then start Redis, the coordinator, and a worker:

```sh
pip install -e .
mkdir -p data/redis data/checkpoints
redis-server --dir "$PWD/data/redis" --appendonly yes --appendfsync always --maxmemory-policy noeviction &
export COORDINATOR_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export REDIS_URL="redis://127.0.0.1:6379/0"
export REDIS_NAMESPACE="tinyrl"
export CKPT="$PWD/data/checkpoints"
python -m tinyrl.gateway &
python -m tinyrl.trainer lora:Qwen/Qwen3-0.6B
```

The gateway is the coordinator. It owns command processing in Redis; trainers connect over HTTP to `COORDINATOR_URL`
(default `http://127.0.0.1:8000`). Give coordinator and workers the same `COORDINATOR_TOKEN`. The token authenticates the
internal worker endpoints. Public API authentication and deployment TLS belong at the service boundary.

Redis stores command metadata, ownership, and results with AOF persistence. `CKPT` is a checkpoint directory shared
by coordinator and workers. Large request/reply bodies use immutable files in `PAYLOAD_DIR` (default `CKPT/payloads`);
all processes must read the same shared directory. These filesystem stores must survive process/node replacement.

A **pool** groups runs served by one trainer process. LoRA runs on the same base share `lora:<base_model>`. A full
fine-tune (`user_metadata={"openrl.fine_tuning_type": "full"}`) or exclusive LoRA (`"openrl.exclusive": "true"`) uses its
model id as its pool. Start its trainer with `python -m tinyrl.trainer <model_id>`. Automatic worker provisioning is
future scheduler work.

## How it works

```text
SDK → coordinator HTTP API → Redis command metadata and results
              ↕ authenticated worker HTTP API
           trainers ↔ shared payload files and immutable tensor snapshots
```

The coordinator orders each run's commands, assigns them to registered worker sessions, and commits replies,
counters, and checkpoint references using native Redis transactions. Scheduling and recovery policy stays in Python;
there are no Lua scripts. Client sequence ids and idempotency keys prevent retries from applying the same training
command twice. Workers never decide ownership or publish checkpoint aliases directly. Bodies above 64 KiB are stored
once by SHA-256 reference; Redis and the worker protocol carry the reference, and workers read the shared files.
HTTP workers fetch one command at a time so queued batches do not accumulate as prefetched payloads in worker RAM.
Raw HTTP clients must preserve a sequence id or `Idempotency-Key` across admission retries; requests with neither
are treated as new commands. The Tinker SDK supplies sequence ids.

Each assignment carries a generation. A result from a replaced worker cannot complete a newer assignment. A completed
reply can be retrieved repeatedly. Coordinator restarts preserve Redis state; they do not replay uncertain optimizer
steps automatically. Lost worker state is reported explicitly so the client can restore a checkpoint.

`Model.state` describes training state with named tensors, including accumulated gradients and optimizer state.
`Model.weights` produces sampler weights. Both use immutable `safetensors` bundles and metadata through `Store`.
The coordinator binds a logical `tinker://` checkpoint name to its immutable reference only when the save completes.
To export sampler weights for an external reader, resolve the returned URI with
`POST /api/v1/resolve_checkpoint` and `{"path": "tinker://..."}`, then call `Store(root=CKPT).export(ref)` on a machine
that can read the checkpoint directory. Sampling itself is not implemented.

Authenticated `POST /internal/move` with `{"run_id": "...", "target_pool": "..."}` requests a persisted snapshot/restore
handoff. Start a trainer for the destination pool first; an empty target parks the run durably. Inspect its phase with
`GET /api/v1/runs/<id>`. A run requiring recovery can use `POST /internal/recover` with its `run_id`, checkpoint `path`,
and optional `target_pool`; supply an immutable snapshot `ref` instead of `path` to recover a migration snapshot.
Automatic placement decisions remain the scheduler's responsibility.

The model uses fp32 trainable weights and AdamW state, with bf16 GPU forward passes. LoRA runs share a frozen base.
Logprobs are computed in chunks to avoid materializing a full tokens-by-vocabulary tensor.

## Reading the code

- [bus.py](tinyrl/bus.py): coordinator state transitions and Redis command processing.
- [gateway.py](tinyrl/gateway.py): Tinker API and authenticated worker endpoints.
- [client.py](tinyrl/client.py): workers' HTTP connection to the coordinator.
- [payloads.py](tinyrl/payloads.py): immutable request/result files shared by coordinator and workers.
- [trainer.py](tinyrl/trainer.py): training, snapshots, and the worker loop.
- [store.py](tinyrl/store.py): immutable tensor bundles and HF/PEFT export.
- [wire.py](tinyrl/wire.py): protobuf and JSON tensor conversion.

`python sz.py` reports the maintained line-count budget. Runtime settings include `REDIS_URL`, `REDIS_NAMESPACE`,
`COORDINATOR_TTL`, `CKPT`, `PAYLOAD_DIR`, `COORDINATOR_URL`, `COORDINATOR_TOKEN`, `PORT`, `TOKEN_BUDGET`, `CHUNK`,
`GRAD_CKPT`, `WAIT`, and `DEBUG`.

## Test

```sh
pip install -e '.[testing]' tinker
python -m unittest discover -s test -t .
TINKER_PYTHON=$(which python) python -m unittest test.test_tinker
```

Tests launch isolated Redis servers with AOF persistence and use a separate namespace per test. Set `REDIS_SERVER` if
the executable is not on `PATH`. The SDK test launches real coordinator and trainer processes, kills the coordinator,
and continues training after replacement against the same Redis.

## Deployment and remaining work

The coordinator is one active process per Redis namespace, protected by a lease checked during metadata transitions.
Restart can pause command dispatch until the old lease expires. Redis persistence and failover settings determine
whether acknowledged commands survive a Redis failure; asynchronous replica promotion can lose them. See
[the Kubernetes design](docs/kubernetes.md) for restart, fencing, migration, and durability contracts.
An instance that loses its coordinator lease must restart; a readiness probe alone cannot make it regain ownership.

Kubernetes provisioning, automatic pool resizing, agent-driven GPU time-slicing, vLLM sampling, and multi-GPU trainers
remain future integrations. The design keeps those capabilities and assigns their state transitions to the coordinator.
