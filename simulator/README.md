# Training throughput simulator

An interactive design diagram and discrete-event simulator for exploring arrivals,
GPU time, job switching, control latency, and shared data bandwidth. This models
a proposed scheduling policy; it does not execute or benchmark the trainer.

## Run

Open `index.html` in a modern browser, or serve this directory from the repo root:

```sh
python3 -m http.server 8001 --directory simulator
```

Then visit <http://localhost:8001>. Everything runs locally in the browser. D3
is bundled, so there is no CDN, build step, package install, Redis, or GPU requirement.

## Explore

The seven core inputs are GPU slots, jobs/minute, steps/job, tokens/step,
GPU seconds per 4,096 tokens, seconds per switch, and steps before yielding.
Start by measuring timings for one model and sequence-length range, then compare
1, 4, and 16 steps per turn with the same seed.

Advanced controls add control service time, bytes/token, shared bandwidth,
state MiB/switch, client pauses, and variability. State transfers share the data
link with step payloads. When using state MiB/switch, exclude that transfer from
switch seconds to avoid counting it twice.

**Explore uncertainty** runs 24 scenarios. Choose either independent changes of
up to ±25% in selected mean inputs, or fixed means with different random seeds.
The resulting spread is a sensitivity experiment, not a confidence interval.

Watch unfinished work alongside throughput and latency. Completion latency only
includes completed jobs; an overloaded system can leave its slowest jobs waiting
beyond the simulation horizon. GPU allocation separates training, local switching,
control/data service and waits, and idle time. The control and link bars measure
those shared resources' service utilization separately.

## Model and limits

- FIFO runnable jobs, homogeneous GPU slots, and up to the configured number of
  steps per turn. Continuing the same job avoids another switch.
- A worker remains occupied during control and transfer waits. Transfers do not
  overlap compute on that worker. Every job pays an initial installation.
- Renewal arrivals: CV 0 is periodic, CV 1 is Poisson, and larger CV clusters
  arrivals. Job lengths and service noise use lognormal draws. This does not
  simulate sustained traffic surges. CV means standard deviation divided by mean.
- Compute time scales linearly with tokens as a local approximation. Tiny,
  variable job lengths round to at least one step, which can raise their mean.
- Runs stop at the time horizon without draining queued work. Warm-up is excluded
  from measurements; event and trace limits are reported explicitly.
- VRAM eviction, mixed models, multi-GPU jobs, cross-job batching, failures,
  topology, and overlapping transfers are not modeled.

The page includes rate equations and more detail on these assumptions. Calibrate
with real measurements before using the results for capacity decisions.

## Test and edit

```sh
node simulator/engine-tests.cjs
```

The tests use Node's built-in assertions (Node 18+). They check resource ceilings,
overload, switching, transfer contention, reproducibility, work conservation,
latency censoring, and validation. No dependencies need to be installed.

Edit `index.html` for the interface and `throughput-engine.js` for the model.
Both the browser and tests load the same engine; there is no generated bundle
to rebuild. `vendor/d3.v7.9.0.min.js` is D3 7.9.0 from
<https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js>; its license is in
`vendor/LICENSE.d3`.
