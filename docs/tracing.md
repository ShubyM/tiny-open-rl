# Tracing

Every tinyrl process says what it spends time on, and one tool lays it all on one clock.

## Spans

Wrap anything that takes time in a span:

```python
from tinyrl import spans

with spans.span("forward_backward"):
  ...

@spans.timed  # every call is a span named after the function; works on coroutines too
def copy_to_host(...):
  ...
```

- Spans nest, may end on any thread, and cost a `time.time()` and a deque append.
- A worker's spans ride on the next event it appends anyway (`batch_done`, `idle`, `released`). What's left when it exits goes in a `spans` event. A node daemon appends its own (uploads, for example) in a `spans` event each tick.
- New code that takes time should be a span. The scheduler ignores spans, so adding one never changes behavior.

What's spanned today:

| Process | Spans |
|---|---|
| trainer | every command; `load`, `restore_run`, `training_state`, `sampler_weights`, `move_tensors`, `park_on_host`, `resume_on_gpu` |
| store (any process) | `copy_to_host`, `write_checkpoint_files`, `get`, `local_folder`, `download_files`, `rebuild_chain_link`, `upload_version` |
| sampler | `run_sample` per request; vLLM `start`, `restore`, `load_checkpoint`, `sleep`, `wake` |
| node daemon | `stopping` on every `exited`: how long a process took to exit after it was told to |

## Kernel traces

Set `TINYRL_PROFILE=/ckpt-cache/traces/<name>` on the node daemon (its workers inherit it):

- Each trainer records its CPU ops, CUDA kernels and copies from every thread with `torch.profiler`. Every span is a labelled range in it. It writes `trainer-<worker>-<pid>.json.gz` when it exits.
- Each sampler starts vLLM with its torch profiler on and writes `rank0.*.pt.trace.json.gz` when it closes. Profiled samplers boot instead of restoring a snapshot, since a profiler can't be carried through one.
- Profiling slows a run (about 30 s on the 23-job benchmark), so use it to find where time goes, not to time things.
- Traces are written at exit: wait for the workers to exit before restarting the node daemon, or the last ones are cut off.

## One timeline

```bash
REDIS_URL=redis://... python tools/trace.py out.json.gz --since <unix s> --until <unix s> [--profile <folder of traces>]
```

Open `out.json.gz` in https://ui.perfetto.dev. Without `--profile` it draws the event log: per worker, its process (and how long it took to stop), waits, GPU turns, batches, every span, placements and revokes. With `--profile` it adds every kernel trace, shifted onto the log's clock (a trace event's wall time is `baseTimeNanoseconds + ts`). Trainer spans and the matching log entries line up within a few ms.
