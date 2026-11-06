# Tracing

Every tinyrl process says what it spends time on, and one tool lays it all on one clock.

## Spans

Wrap anything that takes time in a span:

```python
from tinyrl import spans

with spans.span("forward_backward"):
  ...

@spans.timed  # every call is a span named after the function; works on coroutines too
def write_checkpoint_files(...):
  ...
```

- Spans nest, may end on any thread, and cost a `time.time()` and a deque append.
- A span records the worker whose task it ran in. The node daemon sets `spans.worker` when it starts serving a worker, and every task and thread that worker's service starts inherits it.
- A trainer child returns its spans with each reply's effects, and the node absorbs them under the worker it serves. The vLLM engine runs inside the node daemon, so its spans need no transport.
- New code that takes time should be a span. Nothing reads spans but the trace, so adding one never changes behavior.

What's spanned today:

| Process | Spans |
|---|---|
| node daemon, per worker | each step's phases: `peek`, `stage`, `acquire`, `take`, `post`, `sweep`; each command by its op, with the child `call` and the `reply` write inside it |
| trainer child | `child` around every command; `load`, `forward_backward`, `optim_step`, `training_state`, `sampler_weights`, `restore_run`, `move_tensors`, `park_on_host`, `resume_on_gpu` |
| store (any process) | `put`, `write_checkpoint_files`, `upload_version`, `get`, `local_folder`, `download_files`, `rebuild_chain_link` |
| vLLM engine | `start`, `restore`, `ready`, `load_checkpoint`, `sleep`, `wake`; snapshot `save` and `restore` |

## The node daemon's trace

Since the node owns GPU turns, it folds `wants_gpu`, `granted`, `revoked`, `released`, `batch_done`, `idle` and its local placements without writing them to `tinyrl:events`. Set `TINYRL_TRACE=1` on the node daemon and it appends to the `tinyrl:trace` stream once a tick:

- every event it folded, with the time it folded it (`t`) and its node;
- the spans that ended since the last tick, as `[name, start, seconds, worker]`;
- `lag`: 50 Hz samples of how late its event loop ran a 20 ms timer. While the loop is busy, every other task waits this long, so a lag spike is time no worker's command could advance.

The stream keeps about the last two million entries. With the variable unset the node writes nothing and drains its spans each tick.

## Kernel traces

Set `TINYRL_PROFILE=/ckpt-cache/traces/<name>` on the node daemon (its trainer children inherit it): each trainer records its CUDA kernels and copies with `torch.profiler` and writes `trainer-<pid>.json.gz` when it exits. Profiling slows a run, so use it to find where time goes, not to time things. Traces are written at exit, so wait for the trainers to exit before restarting the node daemon.

## One timeline

```bash
REDIS_URL=redis://... python tools/trace.py out.json.gz --since <unix s> --until <unix s> [--profile <folder of traces>]
```

Open `out.json.gz` in the dashboard's Trace tab or https://ui.perfetto.dev. It draws a thread per worker (its process, waits, GPU turns, batches, every span, placements and revokes) and one per node daemon (its own spans and a `loop lag ms` counter). With `--profile` it adds every kernel trace, shifted onto the log's clock, and a `gpu busy` counter per trainer.
