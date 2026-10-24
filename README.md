# tiny-open-rl

The [Tinker](https://thinkingmachines.ai/tinker/) training API on Redis Streams and PyTorch, in 370 lines.

The Tinker SDK trains through it unchanged, LoRA adapters and full fine-tunes alike:

```python
client = tinker.ServiceClient(base_url="http://localhost:8000")
trainer = client.create_lora_training_client(base_model="Qwen/Qwen3-0.6B", rank=8)
for batch in data:
  trainer.forward_backward(batch, "importance_sampling")
  trainer.optim_step(types.AdamParams(learning_rate=4e-5))
sampler_path = trainer.save_weights_for_sampler("step-100").result().path  # a peft adapter vllm loads
```

## Run it

```sh
pip install -e .
redis-server &
python -m tinyrl.gateway &                      # the tinker http api, on :8000
python -m tinyrl.trainer lora:Qwen/Qwen3-0.6B   # trains every lora on Qwen3-0.6B
```

A pool is the set of runs one trainer serves. LoRA runs on the same base model share the pool `lora:<base_model>`. A full fine-tune
(`user_metadata={"openrl.fine_tuning_type": "full"}`) or an exclusive LoRA (`"openrl.exclusive": "true"`) is a pool of its own, named by its
model id: `python -m tinyrl.trainer <model_id>`, which exits when the run is deleted. Starting trainers for new pools is the scheduler's job.

## How it works

```
sdk --http--> gateway --XADD--> tinyrl:q:<run> --XREADGROUP--> trainer --XADD--> tinyrl:f:<run>/<id> --XREAD--> gateway --> sdk
```

Every training call is an entry on its run's Redis Stream, and its reply is a one-entry stream named by the request. That's the whole design,
and Redis does the bookkeeping that usually takes a database, a ledger and a heartbeat:

* **The queue reads itself.** A pool's trainer reads its runs' streams through one consumer group, so `XINFO GROUPS` is what's queued and
  `XPENDING` is what's in flight, on which trainer, for how long. `GET /api/v1/runs/<id>` just asks.
* **A dead trainer fails fast.** What it took sits in the pending list until a new trainer starts, claims it with `XAUTOCLAIM`, and fails
  it, so no client waits on work that died with its process.
* **Retries are free.** `XREAD` from `0` blocks until the reply is written and never consumes it, so `retrieve_future` can be retried, or
  polled by many waiters, and every one of them gets the same answer.
* **Deleted runs take no work.** A run is live exactly while its stream exists. `XADD NOMKSTREAM` turns a submit to a deleted run into a 404.
* **Replies land atomically.** The reply, the ack and the run's step counter go in one `MULTI`.
* **One wire format.** `forward_backward` stays protobuf from the SDK to the trainer, which reads tensors straight out of the bytes. Old SDKs'
  JSON becomes the same protobuf at the gateway, and JSON clients get the output decoded back.
* **One trainer.** A base model, and a PEFT adapter per run or the whole model for one, behind `create`, `forward_backward`, `optim_step`,
  `save`, `load_weights` and `delete`. Logits are computed `CHUNK` tokens at a time under checkpointing, so `[tokens, vocab]` never exists.

## Files

| file | lines | |
|---|---:|---|
| [tinyrl/trainer.py](tinyrl/trainer.py) | 145 | losses, token budget batching, the model, the loop |
| [tinyrl/gateway.py](tinyrl/gateway.py) | 95 | the Tinker HTTP API |
| [tinyrl/bus.py](tinyrl/bus.py) | 62 | runs, sessions, commands and replies on Redis |
| [tinyrl/wire.py](tinyrl/wire.py) | 57 | Tinker protobuf to numpy and back |
| [tinyrl/helpers.py](tinyrl/helpers.py) | 10 | `getenv` |

`python sz.py` counts them the way tinygrad does. Knobs are environment variables: `REDIS_URL`, `PORT`, `CKPT` (the checkpoint disk the
trainers and samplers share), `TOKEN_BUDGET`, `CHUNK`, `GRAD_CKPT`, `WAIT`, `REPLY_TTL`, `SESSION_TTL` and `DEBUG=1`.

## Test

```sh
python -m unittest discover -s test -t .    # starts its own redis-server, or set REDIS_URL
TINKER_PYTHON=$(which python) python -m unittest test.test_tinker   # the real sdk against redis, the gateway and a trainer as processes
```

Every test runs against a real `redis-server` and a real (tiny, random) Llama: no fakes.

## Not here yet

Sampling (`create_sampling_session`, `asample`): vLLM serves the adapters `save_weights_for_sampler` writes. Also
`create_training_client_from_state`, multi-GPU FSDP and context parallel trainers, GPU time-slicing, and the scheduler that starts trainers.
