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

```sh
pip install -e .
export COORDINATOR_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export JOURNAL="$PWD/data/journal.sqlite3"
export CKPT="$PWD/data/checkpoints"
python -m tinyrl.gateway &
python -m tinyrl.trainer lora:Qwen/Qwen3-0.6B
```

The gateway is the coordinator. It alone opens the SQLite journal; trainers connect over HTTP to `COORDINATOR_URL`
(default `http://127.0.0.1:8000`). Give coordinator and workers the same `COORDINATOR_TOKEN`. The token authenticates the
internal worker endpoints. Public API authentication and deployment TLS belong at the service boundary.

`JOURNAL` must live on persistent storage. `CKPT` is a checkpoint directory shared by the trainers that exchange state.
The journal stores command and ownership metadata; checkpoint tensors live in `CKPT`.

A **pool** groups runs served by one trainer process. LoRA runs on the same base share `lora:<base_model>`. A full
fine-tune (`user_metadata={"openrl.fine_tuning_type": "full"}`) or exclusive LoRA (`"openrl.exclusive": "true"`) uses its
model id as its pool. Start its trainer with `python -m tinyrl.trainer <model_id>`. Automatic worker provisioning is
future scheduler work.

## How it works

```text
SDK → coordinator HTTP API → SQLite command journal
                              ↕ authenticated worker HTTP API
                           trainer → immutable tensor store
```

The coordinator accepts commands durably, orders each run's commands, assigns them to registered worker sessions,
and commits replies, counters, and checkpoint references in transactions. Client sequence ids and idempotency keys
prevent retries from applying the same training command twice. Workers never open the journal or publish checkpoint
aliases directly.

Each assignment carries a generation. A result from a replaced worker cannot complete a newer assignment. A completed
reply can be retrieved repeatedly. Coordinator restarts preserve the journal; they do not replay uncertain optimizer
steps automatically. Lost worker state is reported explicitly so the client can restore a checkpoint.

`Model.state` describes training state with named tensors, including accumulated gradients and optimizer state.
`Model.weights` produces sampler weights. Both use immutable `safetensors` bundles and metadata through `Store`.
The coordinator binds a logical `tinker://` checkpoint name to its immutable reference only when the save completes.
To export sampler weights for an external reader, resolve the returned URI with
`POST /api/v1/resolve_checkpoint` and `{"path": "tinker://..."}`, then call `Store(root=CKPT).export(ref)` on a machine
that can read the checkpoint directory. Sampling itself is not implemented.

Authenticated `POST /internal/move` with `{"run_id": "...", "target_pool": "..."}` requests a journaled snapshot/restore
handoff. Start a trainer for the destination pool first; an empty target parks the run durably. Inspect its phase with
`GET /api/v1/runs/<id>`. A run requiring recovery can use `POST /internal/recover` with its `run_id`, checkpoint `path`,
and optional `target_pool`; supply an immutable snapshot `ref` instead of `path` to recover a migration snapshot.
Automatic placement decisions remain the scheduler's responsibility.

The model uses fp32 trainable weights and AdamW state, with bf16 GPU forward passes. LoRA runs share a frozen base.
Logprobs are computed in chunks to avoid materializing a full tokens-by-vocabulary tensor.

## Reading the code

- [bus.py](tinyrl/bus.py): coordinator state transitions and the private SQLite journal.
- [gateway.py](tinyrl/gateway.py): Tinker API and authenticated worker endpoints.
- [client.py](tinyrl/client.py): workers' HTTP connection to the coordinator.
- [trainer.py](tinyrl/trainer.py): training, snapshots, and the worker loop.
- [store.py](tinyrl/store.py): immutable tensor bundles and HF/PEFT export.
- [wire.py](tinyrl/wire.py): protobuf and JSON tensor conversion.

`python sz.py` reports the maintained line-count budget. Runtime settings include `JOURNAL`, `CKPT`,
`COORDINATOR_URL`, `COORDINATOR_TOKEN`, `PORT`, `TOKEN_BUDGET`, `CHUNK`, `GRAD_CKPT`, `WAIT`, and `DEBUG`.

## Test

```sh
pip install -e '.[testing]' tinker
python -m unittest discover -s test -t .
TINKER_PYTHON=$(which python) python -m unittest test.test_tinker
```

Tests use temporary SQLite journals and small random Llama models. The SDK test launches real coordinator and trainer
processes and communicates over HTTP. No Redis service is required.

## Deployment and remaining work

The coordinator is one active process with a persistent journal. Restart and node failover can pause command dispatch;
a second replica with an independent journal is not a safe failover mechanism. See [the Kubernetes design](docs/kubernetes.md)
for restart, fencing, migration, and durability contracts.

Kubernetes provisioning, automatic pool resizing, agent-driven GPU time-slicing, vLLM sampling, and multi-GPU trainers
remain future integrations. The design keeps those capabilities and assigns their state transitions to the coordinator.
