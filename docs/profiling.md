# Profiling with Nsight Systems

Set `TINYRL_NVTX=1` to annotate scheduling and execution in the `tinyrl` NVTX
domain. Install the optional dependency in the node and model process environment:

```sh
python -m pip install -e '.[profiling]'
```

The default path does not import NVTX. `trace(name, request_id)` keeps range names
stable and puts the UTF-8 identifier in the byte payload. Command ranges carry
`run/stream-entry` IDs; GPU and process ranges carry worker IDs. Parent and child
ranges can be correlated without registering an ever-growing set of messages.
It uses [start/end ranges](https://nvidia.github.io/NVTX/python/annotation_types.html)
because asyncio tasks can overlap across `await`. Exceptions and cancellation
still close the corresponding range.

## Capture

Install [Nsight Systems](https://developer.nvidia.com/nsight-systems) on the perf
node and check `nsys --version` and `nsys profile --help`. Wrap the usual node
daemon launch, keeping its normal Redis, storage, GPU and snapshot settings:

```sh
TINYRL_NVTX=1 nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --delay=120 --duration=30 --kill=none --wait=all \
  --output=tinyrl \
  python -m tinyrl.control.node_daemon
```

The delay and duration are examples; choose a window containing the workload
phase being measured. `--kill=none` keeps a timed capture from terminating the
daemon. Stop the daemon normally when the workload is finished. Check the report
contains every intended trainer rank and sampling process. In particular,
capture of CRIU-restored vLLM processes needs verification on the target node;
ordinary descendant capture does not establish restore compatibility.

For framework detail, recent versions support `--pytorch=functions-trace`,
including vLLM annotations and request IDs. Add Python sampling, GIL tracing or
NCCL tracing only when investigating a gap that needs them; check installed
options and permissions. The [user guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
documents capture scope and version-dependent features.

## Read the report

Start with a particular command or run and compare `node.prepare` completion,
`node.execute`, CUDA activity and `node.publish`. GPU admission is nonblocking:
the short `node.request_gpu` and `node.grant_gpu` ranges mark its endpoints for a
worker. A request may end in cleanup instead of a grant. `node.wait_gpu` measures
the separate blocking admission needed by a final multi-GPU checkpoint. For run fairness,
compare ready runs sharing an engine: switching runs can happen without parking
the model. GPU handoff is separate; inspect `node.park` when another worker needs
the reservation. The [scheduling controls](../README.md#scheduling-controls) bound
run continuations and sampling waves.
An asynchronous range measures elapsed time, including waits. A node-side range
does not automatically attribute kernels launched by another process; correlate
its request payload with the trainer-side range. Do not add overlapping ranges
as if they were exclusive time.

`trainer.execute` is recorded by rank 0, which owns the command RPC. Other
distributed ranks are visible through their CUDA, NCCL and framework traces.

Nsight Systems provides a preview `nsight-systems` AI skill for CLI and report
analysis. Follow the installed version's [installation guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/)
and use the actual `.nsys-rep` data, installed help, and supported analysis recipes.
The [analysis guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/) describes
the workflow. Nsight Compute's AI assistance is a follow-up for an identified
kernel; the CUDA documentation MCP alone does not collect system traces.

After a change, run correctness tests and repeat the same workload with
`TINYRL_NVTX` unset and without the profiler. Compare per-run waiting and completion
latency alongside total throughput. Keep ordinary queue and ownership metrics
available for runs that are not being profiled.
