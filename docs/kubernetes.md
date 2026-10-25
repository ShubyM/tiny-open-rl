# tinyrl on Kubernetes

**Status: design.** The first piece, one owner per pool, is built and tested on the
[`k8s-prototype`](https://github.com/ShubyM/tiny-open-rl/tree/k8s-prototype) branch. Nothing else here exists yet.

Today you start a trainer for each pool by hand: `python -m tinyrl.trainer <pool>`. Nothing starts trainers for new pools,
sizes them, puts them on GPUs, shares a GPU between them, grows them when they're busy, or brings them back after a crash.
This is the design for the piece that does, on GKE.

## The idea

**Kubernetes supplies GPUs. Redis decides everything else.**

```mermaid
flowchart LR
  SDK["Tinker SDK"] --> GW["gateway"]
  GW -- "XADD commands" --> R[("Redis")]
  S["scheduler"] -- "runs into pools, pools onto agents" --> R
  S -- "creates and deletes agent pods" --> K["Kubernetes + cluster autoscaler"]
  K --> A1["agent pod, 1 GPU"]
  K --> A2["agent pod, 1 GPU"]
  K --> A8["agent pod, 8 GPUs"]
  A1 -- "starts, parks, resumes" --> T1["trainer for Qwen3-8B LoRA pool #1"]
  A2 --> T2["trainer for Qwen3-8B LoRA pool #2"]
  A1 --> T3["trainer for a 1.5B full fine-tune"]
  A8 --> T4["trainer for an 8B full fine-tune"]
  T1 & T2 & T3 & T4 -- "lease, take, finish" --> R
```

- **A pool is the set of runs one trainer serves**, as the README says today: some LoRA runs on one base model, or one full
  fine-tune. What changes is who makes pools. Today [`pool_for`](../tinyrl/gateway.py) puts every LoRA run on a base into
  one pool, so one GPU serves them all. Here the scheduler makes pools, and a base model gets as many as its load needs.
- **What gets placed is a pool's trainer process**, onto an agent pod that holds GPUs. The scheduler creates agent pods
  with plain `nvidia.com/gpu` requests, sized to the pools that need them (1, 2, 4 or 8 GPUs). An agent starts a trainer
  for each pool placed on it and decides which of them are on its GPUs.
- **Every workload scales with demand**, each the way its state allows. LoRA pools grow and shrink by moving runs between
  them: a run moves by streaming its state directly to the new trainer via `peer://` in under 100 ms (or parking to disk if
  idle). A full fine-tune holds a GPU only while it trains. Samplers, when tinyrl has them, scale by replicas.
- **State transport** moves pure tensors where they need to go. A fast `Store` abstraction handles time-slicing
  (`mem://` at 50 GB/s), pool resizing and sampler sync (`peer://` at 12–50 GB/s), and cold parking (`file://` at 1 GB/s).
  Because state moves in milliseconds rather than minutes, dynamic resizing is cheap and weight sync never stalls training.
- **A pool has one owner.** A trainer serves a pool only while it holds the pool's lease in Redis, and every reply is
  written by a script that checks the lease first.

## What runs where

| Piece | Runs as | Code | Does |
|---|---|---|---|
| gateway | Deployment | [gateway.py](../tinyrl/gateway.py) | the Tinker API, as today. It stops picking a run's pool: it records the run's base, rank and LoRA config, and the scheduler does the rest. While a run has no trainer, `retrieve_future` answers `queue_state: "paused_capacity"`, which the SDK logs, instead of `"active"`. |
| scheduler | Deployment, 1 replica | new: `tinyrl/scheduler.py` | puts runs into pools and pools onto agents; grows and shrinks each base model's LoRA pools with their load; creates agent pods when nothing has room, and deletes empty ones |
| agent | one per agent pod | new: `tinyrl/agent.py` | starts and stops a trainer per pool placed on it; decides which pools are resident on its GPUs |
| trainer | a process in an agent pod | [trainer.py](../tinyrl/trainer.py), extended | serves its pool while it holds the lease; parks, releases and restores runs |
| Redis | StatefulSet, AOF on | | streams, leases, placements, heartbeats, load |

Kubernetes sees a gateway, a scheduler, Redis and some agent pods. The scheduler needs one namespaced Role (`create`,
`delete`, `list` on Pods). Nothing is privileged, nothing uses the host network, and there are no CRDs. A pre-pull
DaemonSet on GPU node pools, or GKE image streaming, keeps a new agent from waiting on the image.

## State

Everything [bus.py](../tinyrl/bus.py) has today stays. What's new, and who writes it:

| Key | Type | Written by | Read by |
|---|---|---|---|
| `pool` in `tinyrl:run:<run>`, and `tinyrl:pool:<pool>` | the run's pool, and each pool's runs | **whoever holds the run**: the scheduler while the run is in no pool, its pool's trainer while it's in one | trainers, scheduler |
| `release_to` in `tinyrl:run:<run>` | a request: move the run to this pool, or to none | scheduler; the trainer clears it when it's done | the run's trainer |
| `parked` in `tinyrl:run:<run>` | URI of the run's state (`mem://`, `peer://`, or `file://`), if not in GPU memory | the trainer that parked it, and the one that loads it | trainers, scheduler |
| `tinyrl:owner:<pool>` | string with a 10 s TTL: the lease | the trainer that holds it | trainers, scheduler, gateway |
| `tinyrl:place:<pool>` | string: an agent id | scheduler | agents |
| `tinyrl:agent:<id>` | hash {gpus, gpu_bytes, host_bytes, gpu_model} with a 30 s TTL: a heartbeat | that agent | scheduler |
| `tinyrl:load:<pool>` | hash {gpu_bytes, host_bytes, busy, wait}: measured | the pool's trainer | scheduler, agent |

A run's membership has one writer at a time, and the writer changes hands with the run. The scheduler can only put a run
into a pool while it's in none (checked in the same script that writes it), and only the trainer holding a run can take
it out, under that pool's lease. So two parties never both think they own a run.

## A run's life

```mermaid
sequenceDiagram
  participant C as client
  participant G as gateway
  participant R as Redis
  participant S as scheduler
  participant A as agent
  participant T as trainer
  C->>G: create_model
  G->>R: create the run with no pool, XADD create_model
  S->>R: queued work, no pool: put it in a pool on its base with room, or make a pool
  S->>A: place a new pool on an agent with room, or create an agent pod
  A->>T: start a trainer for the pool
  T->>R: SET owner:pool NX PX 10000, fail what a dead owner left
  T->>R: XREADGROUP, run create_model, finish
  C->>G: retrieve_future, "paused_capacity" until a trainer owns the run's pool
  Note over A,T: time-slicing: the agent says park or resume, trainer moves pool via mem:// (host RAM, 50 GB/s)
  S->>R: the pool is overloaded: make a second pool, release_to = it, for half its runs
  T->>R: between batches: stream runs to new pool via peer:// (<100 ms), move them to the new pool
  S->>R: a run idle for 30 min: release_to = none
  T->>R: park it to file:// on disk, take it out of the pool
  S->>R: a pool with no runs for 10 min: DEL place:pool, and its trainer stops
```

## Scaling every workload

Every workload scales up and down with demand, but not in the same way, because what each one holds is different:

| Workload | What it holds | How it scales | Signal | What a change costs |
|---|---|---|---|---|
| LoRA trainer pool | a base model, plus a small adapter and its AdamW state per run | more or fewer pools per base; runs move between them | `busy`, `wait` | about a second per run moved |
| full fine-tune trainer | a whole model at 12 to 16 B/param | in time, not width: it holds a GPU only while it trains, and its GPU count is set by what fits | the agent's resident set | a host RAM park, 0.4 to 2 s |
| sampler (coming) | weights it can reload, and a KV cache it can drop | replicas | its sampling queue | loading the weights; nothing to save when it scales down |

Both kinds of trainer report the same two numbers once a minute: how busy they were (`busy`, the fraction of the minute
spent running commands) and how long commands waited (`wait`, the mean time from a command's `XADD` to its start, read from
its stream entry ID, which is a timestamp). The scheduler uses them for every decision below.

### LoRA trainer pools: more pools per base

One GPU serving every LoRA run on a base is a queue that grows with the number of runs. Say a run's RL step spends 10 s
training and 50 s sampling and scoring: one GPU keeps up with about six such runs, and from the seventh on, every run
waits for every other's `forward_backward`. Ten GPUs serving one run each is the opposite mistake. So:

- **Assigning.** A run with queued work and no pool (a new run, or one coming back from idle) goes into the pool on its
  base with the most room: memory for its adapter, and `busy` below the grow threshold. If none has room, the scheduler
  makes a pool. A full fine-tune, or a run marked `openrl.exclusive`, always gets a pool of its own.
- **Growing.** When a LoRA pool's commands wait more than 5 s on average for a minute while it's more than 85% busy, and it
  has at least two active runs, the scheduler makes a second pool on the same base, waits until a trainer owns it, then asks
  the busy pool to release half its active runs to it.
- **Shrinking.** When two pools on a base have both been under 30% busy for 10 minutes and their runs fit on one GPU, the
  scheduler asks the smaller to release all of its runs to the other. A pool with no runs is retired.
- **The hard cap is memory**: the base model, plus each run's adapter state, plus a floor for activations, must fit the GPU.
  Load sets the soft cap below it. `MAX_RUNS_PER_POOL` is there for operators who want a fixed one.
- **Hysteresis.** The grow and shrink thresholds are far apart, and a base is resized at most once every two minutes, so
  pools don't flap.

**Releasing a run** is one operation, used for growing, shrinking, idling and draining. Between batches, so nothing of the
run is in flight, the trainer packages the run's adapter weights and fp32 AdamW state and transfers it through the `Store`:
streams directly to the target trainer over `peer://` (or `mem://` if on the same agent), or writes to `file://` if idling
to disk. In one script under its pool's lease, it moves the run from its pool to the target (or to none), records the state
URI in `parked`, and clears `release_to`. Commands that arrive meanwhile wait in the run's stream. The target pool's trainer
sees the run in its pool, recreates the adapter from the run's recorded config, loads the state via `Store.get`, and deletes
`parked`.

| Moving a run | State | Over `peer://` (100 Gbps) | Over shared disk (Filestore) |
|---|---|---|---|
| 0.5B LoRA, rank 32 on every linear layer | 17.6M adapter parameters × 12 B = 0.21 GB | **17 ms** | 0.2 s (2 GB/s) |
| 8B LoRA (Qwen3-8B), rank 32, output head included | 92M × 12 B = 1.1 GB | **88 ms** | 1.1 s (2 GB/s) |

A new pool on a base costs one base model load from the shared Hugging Face cache, about 8 s for an 8B model on Filestore,
plus an agent pod if no agent has room.

### Full fine-tunes: elastic in time, not in width

A full fine-tune is one run, so it can't be split across pools, and making its trainer wider (more GPUs, data parallel)
buys little:

- **Training isn't where an RL step goes.** In open-rl's own two-job full fine-tune benchmark (`docs/fft/fft.md` §9), a
  steady-state step was about 25 s sampling, 25 s training and 74 to 88 s writing weights for the sampler to the shared
  disk, 124 to 138 s in all. Doubling the trainer's GPUs would save at most about 12 s of that.
- **Changing width means moving everything.** Going from one GPU to two reshards the weights, gradients and AdamW moments,
  12 to 16 bytes per parameter, and changes the order of the gradient reductions, so the run's numbers shift.

So a full fine-tune scales in time instead. It gets the fewest GPUs its state fits on, chosen at placement. Between its
training steps, while its client samples and scores, it holds no GPU memory if another pool wants it: the agent parks it
to host RAM, 0.4 s for a 1.5B model, and that GPU serves other runs. Packing several full fine-tunes onto the GPUs they
take turns on is how they use accelerators well. Its width changes only when it's next placed, after a release or an idle
park, and only when it has to (an 8B full fine-tune with fp32 master weights needs more than one 80 GB GPU), once tinyrl
has FSDP.

The bigger win for full fine-tunes is the 74 to 88 s of weight sync, not the trainer: put the run's sampler on the trainer's
agent, so new weights go through host memory instead of the shared disk.

### Samplers: replicas

Generating rollouts splits cleanly across prompts and samples, and a sampler holds nothing it can't rebuild: weights it
can reload and a KV cache it can drop. So samplers scale the simple way, with replicas: more when the sampling queue
grows, fewer when it drains, and scaling down saves nothing because there's nothing to save. A LoRA sampler serves every
adapter on its base, so its replicas serve every run on that base. A full fine-tune's sampler serves one run, and each of
its replicas reloads the weights after every training step, so its replicas are worth adding only while sampling takes
much longer than that reload.

## State transport: getting weights somewhere fast

A checkpoint is not a container, a VM snapshot or an opaque CUDA driver dump. It is a piece of state that lets any trainer
process on any compatible GPU resume a run:

- **The tensors**: model weights (bf16 for a LoRA base, bf16/fp32 for adapters or a full fine-tune), fp32 master weights, and
  AdamW momentum and variance moments ($m$ and $v$ in `optimizer.state`).
- **The metadata**: step count, learning rate schedule, and RNG seeds.

Nothing else matters. If you can get those tensors into a trainer's memory quickly, a run can pause, resume, change GPUs,
split across pools, move between nodes, or push updated weights to a fleet of samplers, anywhere in the cluster.

### The `Store` abstraction

Instead of coupling checkpoints to file paths on a slow shared disk, state moves through a uniform `Store`:

```python
class Store:
  """Transfers model and optimizer state between devices, processes and nodes."""
  async def put(self, uri: str, state: dict[str, Tensor]) -> None: ...
  async def get(self, uri: str) -> dict[str, Tensor]: ...
```

The URI scheme picks the fastest physical path available:

| Tier | URI scheme | Physical link | Speed | What it's for |
|---|---|---|---|---|
| **Host memory** | `mem://<run>` | PCIe Gen5 / NVLink intra-node | 50–64 GB/s | Taking turns on a GPU (time-slicing); zero-network trainer-to-sampler weight handover on the same node |
| **Peer stream** | `peer://<agent>:<port>/<run>` | 100G–800G cluster fabric (TCP / RoCE RDMA) | 12–50 GB/s | Moving runs between pools (dynamic resizing); migrating off a preempted spot node; cross-node sampler updates |
| **Durable storage** | `file://<path>` or `gcs://<bucket>/<path>` | Shared Filestore / Lustre / GCS | 0.5–2 GB/s | Cold idle parking (>30 min) when nodes scale down; explicit user `save_weights("step-100")`; disaster backup |

### What state transfer costs

Transfer times across physical tiers on an H100 node with a 100 Gbps network fabric (or 400 Gbps on A3 Ultra / H200):

| Workload | State size | `mem://` (PCIe, 50 GB/s) | `peer://` (100 Gbps, 12.5 GB/s) | `peer://` (400 Gbps, 50 GB/s) | Shared disk (1 GB/s Filestore) |
|---|---|---|---|---|---|
| **LoRA run** (rank 32 on 8B base) | 1.13 GB (adapter + fp32 AdamW) | **23 ms** | **90 ms** | **23 ms** | 1.1 s |
| **1.5B full fine-tune** | 18.5 GB (1.54B × 12 B) | **0.37 s** | **1.48 s** | **0.37 s** | 18.5 s |
| **8B sampler weights** (inference only) | 16.4 GB (8.19B × 2 B bf16) | **0.33 s** | **1.31 s** | **0.33 s** | 16.4 s (74–88 s on NFS) |
| **8B full fine-tune** (8.19B × 12 B) | 98.3 GB (fp32 master + moments) | **2.0 s** (0.25 s/GPU across 8) | **7.8 s** | **2.0 s** | 49–98 s |

### Why fast transport enables dynamic resizing

When state transfer is fast, operations that are normally painful become cheap:

1. **Splitting and merging LoRA pools.** When the scheduler detects that pool `lora:Qwen3-8B:1` is overloaded, it starts
   `lora:Qwen3-8B:2` on another agent. Moving three runs over `peer://` takes under 300 ms total. The old trainer writes
   state into the peer stream between batches; the new trainer reads it into its base model and immediately begins taking
   commands. Neither touches disk.
2. **Eliminating the RL weight sync bottleneck.** In open-rl's full fine-tune benchmark (`docs/fft/fft.md` §9), saving
   weights to shared NFS took 74 to 88 seconds out of every 128 s step—over 60% of the entire run time. With `Store`, a
   trainer pushes 16.4 GB of bf16 weights to a local sampler over `mem://` in **0.33 s**, or to remote sampler pods over
   `peer://` in **1.31 s**. The steady-state RL step drops from 138 s to ~51 s, more than doubling cluster throughput.
3. **Resuming anywhere.** If a spot node receives a 30 s termination notice, the agent streams its runs' state to a
   replacement agent pod in 2 seconds. The scheduler updates `tinyrl:place:<pool>`, and training resumes on the new node
   without lost steps. If an agent crashes without warning, the run resumes from its last periodic cold checkpoint on
   `file://`.

## Pools onto agents

The scheduler's other half places pools, and it's the same loop:

```python
async def tick(self) -> None:
  agents = await self.bus.agents()  # live heartbeats, each with its GPU and host memory
  pools = await self.bus.pools()  # each pool's runs, load, placement and size

  await self.resize(pools)  # assign, grow, shrink and idle-release runs, as above

  for pool in pools.values():
    if pool.placed_on and pool.placed_on not in agents:  # its agent is gone: the lease runs out, place it again
      await self.bus.unplace(pool.name)
    elif pool.placed_on and pool.empty_for_s > EMPTY_POOL_S:  # no runs left: retire it
      await self.bus.unplace(pool.name)

  waiting = [p for p in pools.values() if p.runs and not p.placed_on]
  for pool in sorted(waiting, key=lambda p: p.size.gpu_bytes, reverse=True):  # best-fit decreasing
    agent = best_fit(agents.values(), pool.size, pool.gpus)
    if agent:
      await self.bus.place(pool.name, agent.id)
      agent.reserve(pool.size)
    else:
      await self.kube.ensure_pending_agent(pool.gpus, pool.size)  # at most one pending agent pod per shape

  for agent in agents.values():
    if agent.empty_for_s > EMPTY_AGENT_S:
      await self.kube.delete_agent(agent.id)  # the autoscaler then removes the empty node
```

- **A pool fits an agent** when its GPU footprint fits one of the agent's GPU groups on its own, and the footprints placed on
  the agent fit its host RAM, since a pool that isn't resident is parked there. So an agent can hold more pools than its
  GPUs fit at once, and it takes turns between them. `OVERSUBSCRIBE` caps that at a multiple of GPU memory.
- **Waiting isn't always load.** A pool whose commands wait while it's rarely busy is losing turns on a crowded agent, not
  short of compute; the scheduler moves the pool to an agent with room instead of splitting it.
- **Scaling up** is one pending agent pod per GPU shape. kube-scheduler packs agent pods onto nodes, and when none fits, the
  cluster autoscaler adds a node; a nodeSelector picks L4, H100, H200 or B200 by the GPU memory the pool needs.
- **Scaling down** deletes agents that have had no pools for 10 minutes. Runs leave pools only by parking to disk, and pools
  are retired only when empty, so deleting an empty agent can't lose anything. Queue lag is deliberately not the signal: an
  RL run's queue is empty while it samples, but its state still needs a home.
- **Multi-GPU pools** get an agent pod that requests N GPUs, so they land on one node as a gang.

## Estimator

No tables of models. Before a run's first placement, build its model on the meta device from its config
(`AutoConfig.from_pretrained`) and count parameters. [trainer.py](../tinyrl/trainer.py) keeps fp32 master weights, so a full
fine-tune holds 16 bytes per parameter at peak (weights, gradients and two AdamW moments) and 12 between steps, because
`optimizer.zero_grad()` frees the gradients. A LoRA pool holds 2 bytes per parameter of bf16 base, plus 16 per adapter
parameter of each run. Activations don't need estimating: [`token_budget`](../tinyrl/trainer.py) sizes every pass to the
memory that's free.

The trainer writes what it measured to `tinyrl:load:<pool>` once a minute, and the scheduler and agent use that from then on.
A wrong guess costs one placement, not a run.

## Parking: time-slicing and checkpointing are one mechanism

`Model.park` moves tensors off the GPU: the weights, the gradients, and the AdamW moments, which live in `optimizer.state`,
not in the module, so moving the module isn't enough.

| Park to | Store URI | For | Decided by | Survives |
|---|---|---|---|---|
| host RAM, a whole pool | `mem://` | taking turns on a GPU | the agent | nothing; it's the same process |
| peer stream, one run | `peer://` | moving runs between pools, spot migration | the scheduler, by asking for a release | a single node migration |
| disk, one run, with optimizer | `file://` | idle runs, draining nodes, user checkpoints | the scheduler or client | a process crash, a lost node |

**The agent decides what's on its GPUs, and nothing else does.** Every second it picks the resident set: pools with queued
work, least recently served first, as many as fit; then idle pools, if there's room left. It tells the rest to park over
their stdin and waits for each to say it has before it tells another to resume, so two pools never race for the same
memory. A busy pool keeps its place for at least 30 seconds, or until its queue is empty, so pools don't thrash. When
everything fits, nothing parks, and CUDA interleaves the processes' kernels by itself.

**LoRA pools park like any other.** A LoRA pool with active runs stays resident because it has queued work. An idle one
parks cheaply: an 8B base is 16.4 GB of bf16, about 0.33 s to host RAM. Idle runs inside a busy LoRA pool don't hold memory
either: after 30 minutes without a command they're released to disk, one at a time.

**A park is a checkpoint you could load**, and it's restored only if nothing has happened to the run since: `parked`
is written as the run leaves its pool and deleted by the trainer that loads it via `Store.get`. So a crash after a restore can't bring back
stale state; it surfaces as an error, the way a crash does today.

What a pool's trip to host RAM costs on an H100, at about 50 GB/s pinned (20 GB/s pageable):

| Pool | State between steps | Each way |
|---|---|---|
| 0.5B LoRA base with one rank 32 run | 0.99 GB bf16 base + 0.28 GB of adapter state | 25 ms |
| 1.5B full fine-tune | 1.54B × 12 B = 18.5 GB | 0.37 s pinned, 0.9 s pageable |
| 8B full fine-tune (Qwen3-8B) | 8.19B × 12 B = 98 GB, 131 GB at peak | 2.0 s on one GPU, 0.25 s per GPU across 8 |

With fp32 master weights an 8B full fine-tune doesn't fit an 80 GB H100 at all; it needs an H200 or B200, or FSDP. Three
1.5B full fine-tunes share an 80 GB GPU between steps (55 GB) without parking, or two at peak (49 GB).

## One owner per pool

Two trainers serving one pool would split its runs' commands between two copies of the model. A trainer that pauses (a
long GC, a frozen VM, a network partition) and comes back must not answer for a pool someone else has taken over. Three
things together make that hold. All three are on the `k8s-prototype` branch: 57 more lines of code in `bus.py` and
`trainer.py`.

1. **The lease.** A trainer serves a pool only after `SET tinyrl:owner:<pool> <worker> NX PX 10000` succeeds. It renews the
   lease every 2.5 s from the event loop while commands run in a thread, so a long `forward_backward` doesn't let it lapse.
   If a renewal fails, the trainer stops. It also stops reading and running commands once half a lease has passed since its
   last renewal, so a trainer coming back from a pause stops before it touches anything.
2. **The fenced finish.** `Bus.finish` writes the reply, the acknowledgement and the count in one Lua script that first
   checks the lease is still the caller's. A trainer that lost its pool can't answer, so a client only ever sees the owner's
   reply. Releasing a run goes through the same check.

   ```lua
   if redis.call('GET', KEYS[1]) ~= ARGV[1] then
     return 0
   end
   redis.call('XADD', KEYS[2], 'MAXLEN', '=', 1, '*', 'ok', ARGV[3], 'content_type', ARGV[4], 'body', ARGV[5])
   redis.call('EXPIRE', KEYS[2], ARGV[6])
   redis.call('XACK', KEYS[3], ARGV[8], ARGV[2])
   redis.call('XDEL', KEYS[3], ARGV[2])
   if ARGV[7] ~= '' then
     redis.call('HINCRBY', KEYS[4], ARGV[7], 1)
   end
   return 1
   ```

3. **The stranded sweep.** A trainer that paused while blocked in `XREADGROUP` can be handed a command after its lease ran
   out, and it will never answer it. So between batches, when it has nothing in flight itself, the owner claims anything
   that has been in flight anywhere for 5 seconds and fails it.

`test/test_lease.py` runs two real trainer processes on one pool against a real `redis-server`. The second waits, and never
reads from the pool. The first is paused with `SIGSTOP` while it's blocked reading, so Redis hands it the next command. Its
lease runs out; the second takes the pool over and fails the stranded command ("its trainer died, or lost the pool, during
optim_step"). After `SIGCONT` the first finds its lease gone and exits within about a second. The client's reply is still
the second's, nothing is left in flight, and the second goes on serving the pool. Another test restores a run from
`save` with its optimizer into a fresh `Model` and checks that the next three steps give the same losses and bitwise-equal
weights as the run that never stopped, which is what moving a run between pools relies on.

## When things fail

| Failure | What happens | What's lost |
|---|---|---|
| a trainer dies in the middle of `optim_step` | the lease expires within 10 s; the agent starts a new trainer, which takes the lease, claims the command and fails it | the in-memory state of the pool's runs, except runs parked to disk. Later calls get the error they get today ("not on this trainer, ... its trainer restarted"), and the client loads its last checkpoint. |
| a trainer dies while releasing a run | if it died before the release script ran, the run is still in its pool with no state, as above; after, the run is in its new pool with `parked` set, and loads there | as above, or nothing |
| a node dies | its agent's heartbeat stops within 30 s; the scheduler places its pools elsewhere, creating an agent if it has to | the same as a trainer dying |
| the scheduler dies | nothing; trainers keep serving, and the Deployment restarts it | nothing |
| Redis restarts (AOF, fsync every second) | streams and leases come back; up to a second of writes may not, so a lease can expire early and the takeover path runs | possibly one in-flight command, failed |
| Redis fails over to a replica | a lease written just before can be missing, so a second trainer can take the pool. The first one's next renewal fails within 2.5 s and it stops; its replies were fenced, and what it held is swept and failed | a few commands, failed, never answered twice |
| a paused trainer comes back | the deadline, the fenced finish and the sweep, above | one in-flight command, failed |
| a paused trainer writes `save_weights` files | the command is failed, but files may exist at its path | nothing, though a failed save can leave files behind |

## Samplers

When tinyrl has a sampler (vLLM serving what `save_weights_for_sampler` writes), it's a second kind of process on an agent.
It parks with vLLM's `sleep(level=1)` and resumes with `wake_up()`, and the agent treats it like a trainer, with its
sampling queue as its demand. Its replicas scale with that queue (see Scaling every workload), and the scheduler places
them on the agents that have their runs' trainers, so a run's sampler and trainer take turns on one GPU and weight sync can
go through the node's `/dev/shm` (`mem://`) instead of the shared disk. When samplers run on dedicated GPU nodes to handle
large rollout batches, the trainer pushes updated weights to them over `peer://` (1.31 s for 8B) instead of stalling on
shared NFS. For small models both stay resident and only their kernels interleave.

A full fine-tune's sampler checkpoint is fp32, because the master weights are. vLLM's `dtype="auto"` would serve an fp32
checkpoint in fp16, so the sampler should run with `--dtype bfloat16`, or the save should write a bf16 copy.

## Code

| File | Lines | Change |
|---|---:|---|
| [bus.py](../tinyrl/bus.py) | 160 → ~210 | `own`, `renew`, `release`; the fenced `finish`; `reclaim` with a minimum idle time (prototyped); assigning and releasing runs; placement, heartbeat and load helpers |
| [trainer.py](../tinyrl/trainer.py) | 271 → ~335 | the lease loop and the sweep in `Trainer.serve` (prototyped); `Model.park`; releasing runs to disk and restoring them; reporting load; obeying park and resume |
| `store.py` | new, ~80 | tiered state transport: `mem://` (host RAM / `/dev/shm`), `peer://` (TCP / RDMA streaming socket), `file://` (safetensors on disk / GCS) |
| `agent.py` | new, ~110 | the heartbeat, a trainer per placed pool, the residency loop |
| `scheduler.py` | new, ~170 | assigning runs, growing and shrinking pools, placing pools, agent pods through the Kubernetes API |
| `estimator.py` | new, ~25 | parameter count from the config on the meta device, and bytes per parameter |
| [gateway.py](../tinyrl/gateway.py) | 154 → ~152 | `pool_for` moves to the scheduler; the run records its LoRA config; `paused_capacity` |
| `k8s/` | ~120 lines of YAML | gateway, Redis with AOF, the scheduler and its Role, a pre-pull DaemonSet |

About 1,180 lines of Python, up from 692.

## Testing

Like everything else in tinyrl, against a real `redis-server`, real processes and CPU models, with no cluster: the lease
takeover and the sweep (done), a run restored into a fresh trainer (done), moving runs between two real trainers while
commands keep arriving, the assignment, resizing, placement and residency decisions, and an agent running real trainers
through park and resume. A real cluster is needed for: a GPU park actually freeing memory, autoscaling in both directions,
losing a node, and vLLM's sleep and wake.

## How this differs from open-rl

open-rl keeps the truth in Kubernetes. The API server writes `Workload` objects, a placement controller turns each into a
`ClaimLedger`, a DRA `ResourceClaim` and a pod, and a node daemon takes turns between the pods that share a claim. tinyrl
keeps the truth in Redis, which already holds every command, and asks Kubernetes only for GPUs.

| | open-rl | tinyrl |
|---|---|---|
| What gets placed | a pod per worker process | a pool's process, onto an agent pod that already has GPUs |
| LoRA capacity | one worker per base model; its Workload name's instance index "stays 0 until adapter capacity accounting exists" | as many pools per base as load needs; runs move between them over `peer://` in ~90 ms |
| Placement | a Go controller (~2,700 lines, with a 1,274-line design doc): binpack or spread, kube-scheduler as the capacity probe, ledger seats booked by compare-and-swap, assignment IDs, a 2-minute wedge grace | a Python loop (~170 lines): runs into pools by load, pools onto agents by measured size |
| Sharing a GPU | several pods share one DRA claim, seated on a `ClaimLedger` | the processes in one agent pod share its GPUs, requested as plain `nvidia.com/gpu` |
| Who takes turns | a time-slicer DaemonSet on the host network; workers `ACQUIRE` and `RELEASE` it around each batch; one resident per claim | the agent, for its own GPUs; pools stay resident together when they fit, least recently served first when they don't |
| Getting state off the GPU | two ways: cuda-checkpoint through llm-d's snapshot agent (privileged, host PID namespace, finds processes from env tags), and the full fine-tune worker's own sleep and wake | one: the trainer parks pure tensors to host RAM (`mem://`, 50 GB/s), or streams via `peer://` / `file://` |
| State transport | separate mechanisms: POSIX file writes for weights, cuda-checkpoint UVM memory dumps for time-slicing | unified `Store`: pure tensors over `mem://` (PCIe, 50 GB/s), `peer://` (network, 12–50 GB/s), or `file://` (durable disk) |
| Weight sync to samplers | written to shared NFS (`/mnt/shared/open-rl/...`), stalling the step for 74 to 88 s | streamed via `Store`: `mem://` on the same node (0.33 s) or `peer://` across nodes (1.31 s) |
| LoRA workers | exclusive: always resident, never share a GPU | park like any other pool, and their idle runs go to disk |
| Checkpoints | separate from parking | a run parked to disk is a `save_weights` checkpoint |
| Moving work | recreate the worker's pod, since a pod's claims can't change | stream the run's tensors to the target pool's trainer over `peer://` (<100 ms) |
| One owner | agreement between the Workload's status, a ledger seat, an assignment ID and the pod's env; workers are trusted to obey the daemon | a lease per pool, checked by the same script that writes each reply or moves a run |
| Queues | Redis lists drained with `LPOP`, so work in flight is tracked separately | Redis streams: `XPENDING` knows what's in flight, `XAUTOCLAIM` hands it to the next owner |
| Sizing | tables of parameter counts and KV-cache bytes per model name; unknown models are sized as Qwen3-8B | the parameter count from the model's config, then what the trainer measures |
| Scaling up | a standing pending claim and its unschedulable pod | one pending agent pod per GPU shape |
| Scaling down | a claim retires when the last workload seated on it is deleted | idle runs go to disk, empty pools retire, and agents empty for 10 minutes are deleted |
| Privileges | two CRDs; a ClusterRole on nodes, pods and ResourceSlices; a DRA driver; a host-network DaemonSet; a privileged host-PID DaemonSet | one namespaced Role over pods |
| Size | ~4,000 lines: the controller, the time-slicer (677), the worker managers (432), the estimator (128), plus generated CRDs | ~400 lines of Python |

What open-rl has that this doesn't, yet: samplers, with weight sync to vLLM and time-sliced vLLM processes; multi-GPU
training; DRA's device selection, for picking GPUs by attributes rather than node pools; `kubectl get workloads` showing
every worker and where it is, where tinyrl makes you ask Redis or the gateway; and runs on real clusters. Everything here
past the lease is still a design.

## Alternatives considered

- **A pod per pool, scaled to zero between turns.** Parking to disk on every RL turn costs 9 s for a 1.5B full fine-tune
  and 49 s for an 8B one on Filestore, each way.
- **Trainers claiming pools themselves, with no scheduler.** A worker that parks a pool in its host RAM and lets go of it
  lets another node take the pool and quietly restart it from an old checkpoint.
- **Replicas of one LoRA pool, each owning some of its runs by a lease per run.** It resizes too, but every trainer renews
  a lease per run, and fencing a reply means checking the run's lease rather than the pool's. Pools that runs move between
  keep one lease per process.
- **A fixed number of runs per LoRA pool.** It's a stand-in for load, and a bad one: ten runs that are sampling cost the
  trainer nothing, and three that train back to back can saturate it. Memory is the hard cap; load sets the rest.
- **Spreading one LoRA pool across GPUs** (one process, data parallel). It speeds up one big batch, not many runs, and adds
  collectives between GPUs to every step.
- **Widening a full fine-tune's trainer while it runs.** Training was about a fifth of a step in open-rl's benchmark, and
  every change of width reshards 12 to 16 bytes per parameter and shifts the run's numbers.
- **Autoscaling on queue lag** (HPA or KEDA). An RL run's queue is empty while it samples, and a Deployment scale-down kills
  whichever pod it picks, parked state and all.
- **One process per GPU that hosts every pool.** Residency needs no messages, but one crash takes down every pool on the
  GPU, and samplers are separate processes anyway.
- **An agent DaemonSet.** It never makes a pod pending, so the cluster never scales up.
- **cuda-checkpoint.** It needs a privileged agent in the host PID namespace to find processes, and it moves a whole CUDA
  context instead of the tensors that matter.
- **Staging state migrations and weight sync through shared disk (Filestore / NFS).** In open-rl's benchmark, writing
  weights to shared disk took 74 to 88 s out of a 138 s step, dominating the entire RL cycle. Staging moving runs through
  NFS also incurs filesystem lock contention and metadata latency. Direct peer streaming (`peer://`) moves state at wire
  speed (12 to 50 GB/s) directly into the receiving trainer's memory.

## Open questions

1. Are the resizing thresholds right: grow at a 5 s mean wait while 85% busy, shrink below 30% busy for 10 minutes?
2. Should agents oversubscribe their GPUs by default, or only when asked?
3. Keep Tinker's semantics, where a crash is an error and you resume from your own checkpoint, or have trainers park runs to
   disk every N steps so a lost node costs at most N steps?
4. One agent pod per pool shape, or one per node holding all of its GPUs?
5. A single Redis with AOF, or replicas with failover, which the lease makes safe but which can fail a few commands each
   time?
