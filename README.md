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
python -m tinyrl.trainer lora:Qwen/Qwen3-0.6B   # trains every LoRA on Qwen3-0.6B
```

A **pool** is the set of runs one trainer serves. LoRA runs on the same base model share the pool `lora:<base_model>`.

A full fine-tune (`user_metadata={"openrl.fine_tuning_type": "full"}`) or an exclusive LoRA (`"openrl.exclusive": "true"`) gets a pool
of its own, named by its model id. Start its trainer with `python -m tinyrl.trainer <model_id>`; it exits when the run is deleted.
Starting trainers for new pools is the scheduler's job.

## How it works

```
sdk --http--> gateway --XADD--> tinyrl:q:<run> --XREADGROUP--> trainer --XADD--> tinyrl:f:<run>/<id> --XREAD--> gateway --> sdk
```

Every training call is a command on its run's Redis stream, and its reply is a stream with one entry, named by the request. That's the
whole design. Redis does the bookkeeping that would otherwise take a database, a ledger and a heartbeat:

* **The queue reads itself.** A pool's trainer reads its runs' streams through one consumer group, so `XINFO GROUPS` says what's queued,
  and `XPENDING` says what's in flight, on which trainer, and for how long. `GET /api/v1/runs/<id>` just asks.
* **A dead trainer fails fast.** The commands it took stay in the pending list. The next trainer to start claims them with `XAUTOCLAIM`
  and fails them, so no client waits on work that died with its process.
* **Retries are free.** `XREAD` from id `0` blocks until the reply is written and never consumes it, so `retrieve_future` can be retried,
  or waited on by many clients, and every one of them gets the same answer.
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
  resume a run with its exact fp32 weights and AdamW state.
* **One owner per pool.** A trainer serves a pool only while it holds the pool's lease. A WATCH/MULTI/EXEC transaction on the lease key checks the lease before any reply lands.

## Reading the code

Start with the docstring at the top of [bus.py](tinyrl/bus.py), which explains the streams. Then read [gateway.py](tinyrl/gateway.py)
to see how each Tinker call becomes a command, [store.py](tinyrl/store.py) to see how checkpoints are stored, and
[trainer.py](tinyrl/trainer.py) to see how a command runs on the model.

| file | lines | what's in it |
|---|---:|---|
| [tinyrl/trainer.py](tinyrl/trainer.py) | 318 | the losses, token budget batching, the model, and the trainer loop |
| [tinyrl/bus.py](tinyrl/bus.py) | 181 | runs, sessions, commands and replies on Redis |
| [tinyrl/gateway.py](tinyrl/gateway.py) | 152 | the Tinker HTTP API |
| [tinyrl/wire.py](tinyrl/wire.py) | 94 | the Tinker protobuf, to numpy and back |
| [tinyrl/store.py](tinyrl/store.py) | 38 | `safetensors` checkpoint store behind `put(uri)` and `get(uri)` |
| [tinyrl/helpers.py](tinyrl/helpers.py) | 12 | `getenv` |

`python sz.py` counts lines of code the way tinygrad does. Settings are environment variables: `REDIS_URL`, `PORT`, `CKPT` (the
checkpoint disk the trainers and samplers share), `TOKEN_BUDGET` (padded tokens per pass; unset, each call sizes its passes to the free
GPU memory), `CHUNK`, `GRAD_CKPT`, `WAIT`, `REPLY_TTL`, `SESSION_TTL`, `LEASE_MS` and `DEBUG=1`.

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

* Sampling (`create_sampling_session`, `asample`). vLLM can already serve the adapters `save_weights_for_sampler` writes.
* `create_training_client_from_state`.
* Multi-GPU trainers (FSDP, context parallelism).
* Running on Kubernetes: a scheduler that starts trainers for new pools, GPU time-slicing, and parking idle runs. The design is
  in [docs/kubernetes.md](docs/kubernetes.md).
