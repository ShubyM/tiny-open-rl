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

## GPU-turn reference benchmark

`scheduling.py` is a separate, small reference model for evaluating turn policies:

```sh
python3 -m simulator.scheduling
python3 -m unittest test.test_fairness
```

With the project's Python dependencies installed, add `--actual` to compare the
production scheduler with both references. `SchedulerPolicy` invokes the real
`tinyrl.scheduler.fold` and `decide` functions; it does not copy their ranking logic.
Each run gets one already-warm worker on the same GPU. Arrivals retain their
timestamps, decisions occur at quantum boundaries, and release acknowledgements
are immediate. Weighted attained-service accounting charges compute plus
command-associated switching, matching the trainer's occupancy accounting;
reported useful work excludes switching. New runnable runs join the current
virtual-service cohort rather than competing against historical unspent capacity.
Separate property tests cover delayed acknowledgements and
shuffled facts from an old worker after replacement. A late-arrival regression
checks that recurring switch costs do not accumulate unbounded unpaid debt for
older jobs and prevent a new job from receiving service.

It models one already-warm GPU, integer compute demand and arrivals, a fixed
quantum, and a cost when changing workers. Compute is divisible at the observation
horizon. It does not model kernels, memory placement, batching, worker startup,
training convergence, or overlapping transfers. Its compute service is a resource
metric, not a claim about model quality or training goodput.

Every eligible job remains in the report, including jobs receiving zero service.
Service is normalized against ideal weighted processor sharing with the same
arrivals and finite demand, without switching cost. The report includes minimum
normalized service, Jain's index, useful compute, switching and idle time,
unfinished jobs, and the longest wait including unfinished waiting intervals.
Jain alone is insufficient: serving 99 jobs while starving one can score 0.99,
and equalizing tiny allocations can score 1 while most GPU time is wasted.

The FIFO baseline requeues a worker after each turn, matching the former serial
one-GPU turn policy; it does not reproduce the former placement zones. The exact
oracle searches every legal turn choice and idle tick for at most four jobs and
twelve time units. Its objective is lexicographic: maximize minimum normalized
service, then total compute. It knows future arrivals and can wait longer than an
online policy. It is an optimum only for this stated model and objective; it is
neither a production policy nor a universal lower bound on latency.

The committed [scheduling-results.json](scheduling-results.json) is the default `--actual` output.
Six fixed workload seeds are evaluation cases, not an optimization dataset. The
tests independently enumerate a tiny oracle case and check the homogeneous bound:
with equal continuously runnable jobs, no switching cost, and complete rounds,
round-robin gives each job the ideal equal share. Arrivals, weights, switching,
placement, and delayed release require separate evidence; that bound says nothing
about their global optimum.

The default six cases give the following minimum normalized service. These cases
were fixed before connecting the actual scheduler; they expose its remaining gaps.

| Case | FIFO turns | Actual scheduler | Exact oracle |
| --- | ---: | ---: | ---: |
| 0 | 0.545 | 0.545 | 0.545 |
| 1 | 0.500 | 0.500 | 0.500 |
| 2 | 0.500 | 0.500 | 0.667 |
| 3 | 0.577 | 0.577 | 0.577 |
| 4 | 0.429 | 0.429 | 0.667 |
| 5 | 0.364 | 0.545 | 0.727 |

In cases 2, 4 and 5, the actual scheduler performs seven units of compute while the
oracle performs eight. In case 5, the actual scheduler improves minimum useful
service over FIFO, but increases the longest wait from six to eight units.
This is a tradeoff, not evidence of a latency guarantee. The tests also keep the
former placement regression separate: eight distinct shared LoRA workers on an
eight-GPU node previously used only lanes 0 and 1; the actual scheduler uses all
eight idle GPUs. A separate four-GPU/five-run scenario executes source-release,
target-assignment and GPU-turn events to check that the extra run can relocate
instead of remaining permanently limited to half a GPU. That lifecycle scenario
uses instantaneous checkpoint, startup and release acknowledgements, and checks
physical turn exclusion after every event. It is not a migration performance
benchmark. None of these scenarios establishes cluster-wide optimality.

This separation follows the evaluation distinction in
[Gavel, sections 7.1 and 7.5](https://people.eecs.berkeley.edu/~matei/papers/2020/osdi_gavel.pdf):
objective quality and realization of an ideal allocation are different questions.
[Themis, sections 3.2 and 6](https://www.usenix.org/system/files/nsdi20-paper-mahajan.pdf)
measures the distribution and worst finish-time fairness as well as GPU time for
equal work. [Pollux, section 5.3.1](https://www.usenix.org/system/files/osdi21-qiao.pdf)
also shows that pushing instantaneous equality harder can worsen eventual fairness
through extra reallocations. The small oracle here establishes none of those
papers' broader performance claims for this implementation.
