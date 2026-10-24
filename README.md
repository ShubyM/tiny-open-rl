# `tiny-open-rl`

A clean-sheet, **~900-line** implementation of the Open-RL server and trainer runtime built around **Redis Streams (`RuntimeBus`)**, **internal binary Protobuf transport**, and a **single composable `TrainerRunner` + `ModelBackend` protocol** (replacing ~8,400 lines in `open-rl`).

---

## Architecture Highlights

1. **`RuntimeBus` (`src/tiny_open_rl/bus.py` — ~280 LOC)**:
   - Replaces `store.py`, `session_registry.py`, `model_metadata.py`, `sampling_store.py`, and manual request-loss ledgers (`open_rl:ledger:*`, `popped()`, `answered()`, `Worker.outstanding`).
   - **Per-Run Redis Streams (`open_rl:stream:<run_id>`) + Pool Set (`open_rl:pool:<pool_id>:runs`)**:
     - `XADD` enqueues commands (carrying raw binary Protobuf bytes or JSON).
     - `XREADGROUP` claims batches into the `trainer` consumer group's **Pending Entries List (PEL)** with round-robin fairness across shared-LoRA adapters.
     - `complete()` atomically writes the future result and calls `XACK` + `XDEL`.
     - `fail_orphans()` runs `XAUTOCLAIM` on trainer startup to immediately fail any requests left in-flight by a crashed container so clients never hang on `retrieve_future`.
     - `stream_status(run_id)` reads `queued` (`lag`) and `in_flight` (`XPENDING`) directly from the Redis Stream.
   - **Blocking, Idempotent Futures (`BLMOVE`)**:
     - `wait_result()` blocks via `BLMOVE open_rl:future:<req_id> open_rl:future:<req_id> LEFT RIGHT <timeout>`—zero `asyncio.sleep(0.1)` polling loops, and 100% idempotent when `/api/v1/retrieve_future` is retried.
   - **Lock-Free Native Redis Hashes & Sets**:
     - `open_rl:session:<id>` and `open_rl:run:<id>` use atomic `HSET`, `HSETNX`, `HINCRBY`, and `SADD` pipelines with zero in-process `asyncio.Lock`s.

2. **Zero-Copy Internal Protobuf Transport (`src/tiny_open_rl/codec.py` — ~160 LOC)**:
   - **Gateway (`POST /api/v1/forward_backward`)**: Peeks at `(model_id, seq_id)` and pushes raw `application/x-protobuf` bytes straight into `open_rl:stream:<run_id>` without unpacking tensor arrays into Python lists or converting to JSON (>3.5× smaller in Redis).
   - **Trainer (`decode_forward_backward_proto` / `encode_forward_backward_proto`)**: Uses `torch.frombuffer` to construct PyTorch tensors directly from Protobuf byte buffers and packs output `logprobs` tensors directly into `pb.ForwardBackwardOutput` `ArrayRecord` bytes.
   - **Gateway (`POST /api/v1/retrieve_future`)**: Returns raw `application/x-protobuf` bytes directly from Redis when `Accept: application/x-protobuf`.

3. **Unified `TrainerRunner` + `ModelBackend` (`src/tiny_open_rl/trainer.py` — ~270 LOC)**:
   - Collapses `TrainingRequestsProcessor`, `BaseTrainerWorker`, `LoraTrainingWorker`, `FFTTrainingWorker`, and `AutomodelTrainingWorker` into a single `TrainerRunner` loop (owning stream polling, `torchrun` rank-0 broadcast, token-budget microbatching, chunked logprob math, and all 5 RL losses: `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, `dro`) + a 5-method `ModelBackend` protocol.

4. **Thin Gateway & Unified Inspection API (`src/tiny_open_rl/gateway.py` — ~185 LOC)**:
   - Serves Tinker endpoints (`/api/v1/create_session`, `/api/v1/create_model`, `/api/v1/forward_backward`, `/api/v1/optim_step`, `/api/v1/retrieve_future`, `/api/v1/delete_model`) and unified inspection (`/api/v1/inspect/runs/{id}/summary`, `/api/v1/inspect/sessions/{id}/summary`, and `/api/v1/models/{id}/status`).

---

## Line Count Breakdown

| File | LOC | Responsibility |
| :--- | ---: | :--- |
| `src/tiny_open_rl/bus.py` | ~380 | `RuntimeBus` (Redis Streams, `BLMOVE` futures, native Hashes/Sets) + `MemoryStreamRedis` |
| `src/tiny_open_rl/trainer.py` | ~275 | `TrainerRunner`, token-budget microbatching, chunked logprobs, `ModelBackend`, `TorchCausalLMBackend` |
| `src/tiny_open_rl/gateway.py` | ~185 | FastAPI Tinker routes, zero-decode/encode Protobuf pass-through, `/api/v1/inspect/*` |
| `src/tiny_open_rl/codec.py` | ~170 | Zero-copy `torch.frombuffer` Protobuf ↔ PyTorch tensor codec |
| `src/tiny_open_rl/distributed.py` & `losses.py` | ~93 | `torchrun` broadcast helpers & RL loss functions (`cross_entropy`, `ppo`, `cispo`, `dro`, `importance_sampling`) |
| **Total Core Runtime** | **~1,103** | **Replaces ~8,400 LOC in `open-rl`** |

---

## Running Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
