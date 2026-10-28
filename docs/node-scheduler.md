# Fair scheduling inside a node daemon

Status: proposed design. This document changes no runtime behavior and makes no measured fairness or latency claim.
It revises the local GPU arbitration policy in [the Kubernetes plan](kubernetes.md#taking-turns-on-a-node-daemons-gpus).
The initial implementation targets one node. Multiple GPUs on that node are scheduled as groups with fixed widths;
distributed trainers and multi-node execution remain separate implementation work.

## Problem and scope

Today trainers and samplers independently acquire a Redis lease for a GPU. They yield after a command batch when
another process is waiting and the holder has exceeded `HOLD_S`, or has become idle. The next process to retry wins.
That provides cooperative sharing, but no explicit service entitlement, bounded wait, or atomic multi-GPU reservation.
`HOLD_S` is checked after a fetched batch; it is not a bound on that batch's execution time.

Keep the node daemon, shared LoRA pools, separate engine processes, and Redis command streams. Give the daemon sole
authority over local execution grants and memory admission. An engine executes commands and acknowledges transitions;
it does not compete with other engines for GPU leases.

The initial policy aims for approximate weighted GPU-time fairness among admitted, backlogged accounts, subject to
feasible GPU groups and memory constraints. It is work-conserving where a feasible allocation is available without
breaking a pending reservation. Equal GPU time does not imply equal training progress, end-to-end latency, or slowdown.
Indivisible GPU groups and switching overhead prevent exact instantaneous shares; simulation must establish the useful
operating range before making stronger claims.

## Ownership and units

| Unit | Meaning | Owner |
|---|---|---|
| Run account | A run's service entitlement, shared by its training and sampling work | Node scheduler initially; tenant hierarchy can be added later |
| Pool | Compatible LoRA runs sharing one base model | Trainer implementation |
| Execution group | Processes that must start, park, stop, and recover together; includes all ranks of one engine | Node daemon |
| Execution grant | Permission for a group to execute on a particular GPU set | Node daemon |
| Assignment generation | Identity of the currently authoritative run/worker placement | Cluster control state in Redis |

A run's sampler sessions carry its account identity. A standalone sampling client gets its own account. Multiple
sampler replicas do not create additional entitlement. A LoRA pool can contain several accounts; it does not receive
a separate share simply because its engine is resident. If tenants can freely create runs, introduce tenant-level
weights before claiming fairness between tenants.

The cluster scheduler chooses a node; the node daemon chooses the local GPU set and execution turns. Engine results
and checkpoint aliases remain fenced by assignment generation and process incarnation. Local execution grants are
not a replacement for those durable ownership checks. Existing Redis Streams can remain the command transport;
a coordinator HTTP service is not required by this proposal.

The daemon starts a fresh incarnation on restart. Initially, restart the whole daemon pod and its children together,
then recover runs under fresh assignment generations. Adoption of surviving processes requires a separate, explicit
reconciliation protocol. Process supervision must cover descendants such as vLLM's server and worker processes.

## Whole-node deployment

One daemon pod owns all GPUs assigned to tinyrl on its node. It starts trainers and samplers as child execution groups
from its installed runtime, retaining node-local model caches, compile caches, and bounded warm engines. There is no
new workload pod or image pull on each local scheduling turn.

An image still has to become available on each new node. Image caching and GKE image streaming are startup mechanisms,
not scheduler correctness mechanisms. Model loading, CUDA initialization, and compilation are separate warm-up costs.

For heterogeneous node shapes, use mutually exclusive DaemonSets with explicit GPU requests matching each shape.
The current one-GPU manifest is not an all-GPU request for an eight-GPU node. The node must reserve sufficient CPU
and host RAM for its daemon and system services as well as GPU execution.

Keep the existing plan's DaemonSet for reserved nodes and whole-node agent pods for burst capacity. Both use the same
daemon implementation. If burst nodes also use DaemonSets, provision node capacity explicitly; Redis backlog alone
does not drive the Kubernetes autoscaler. Placeholder pods requesting GPUs already reserved by the daemon double-count
demand. Node provisioning is outside this single-node scheduling loop.

## Byte budgets and pipeline admission

Bound data movement separately from GPU service. A request-count limit treats a short prompt and a large training
batch alike; byte budgets provide backpressure proportional to the estimated payload. Use a sliding window whose
credits return when a stage completes, rather than a bytes-per-second token bucket. Slower stages then retain their
credits longer and naturally reduce new submissions.

| Stage | Account for | Release or transfer the reservation when |
|---|---|---|
| Client dispatch | Estimated input bytes being serialized and submitted | Server accepts the command and returns its request ID |
| Server input queue | Accepted payloads waiting for execution, plus bounded reorder buffers | Payload leaves the queue; any execution/staging copy has its own accounted reservation |
| GPU execution | GPU-seconds and admitted GPU/host memory | Engine acknowledges release or all relevant processes exit |
| Server result retention | Completed outputs awaiting retrieval or expiration | The acknowledged retrieval/retention contract permits deletion, or retention expires |
| Client result fetch | Advertised result bytes being downloaded and decoded | Retrieval and deserialization finish |

All physical copies must be accounted for during a handoff; releasing one reservation must not create an unbudgeted
intermediate buffer. Retained results and accepted commands need both byte and count limits, since tiny objects still
consume metadata and connection resources. Client windows smooth traffic, server admission bounds aggregate backlog,
and GPU-time accounting selects service among runnable accounts. Each has a different resource and lifetime.

### Existing SDK controls

The inspected Tinker SDK at commit `3ba0810` already fetches `/api/v1/client/config` when its internal holder is
constructed. The server can return the following supported fields instead of requiring a new client implementation:

| Field | SDK default | Meaning |
|---|---|---|
| `fwdbwd_max_chunk_len` | 1,024 | Target maximum examples per training chunk |
| `fwdbwd_max_chunk_bytes_count` | 5,000,000 bytes | Target estimated uncompressed protobuf bytes per training chunk |
| `fwdbwd_dispatch_bytes_semaphore_size` | 50,000,000 bytes | Training payloads concurrently being submitted |
| `sample_dispatch_bytes_semaphore_size` | 10 MiB | Estimated prompt bytes concurrently being submitted |
| `inflight_response_bytes_semaphore_size` | 50 MiB | Concurrent result fetches using advertised payload sizes |

These are built-in SDK defaults, not measured settings for tinyrl or evidence of Tinker's deployed production quotas.
Training estimates include token storage and loss-input arrays before optional compression. An oversized individual
example is not split by the chunker, so the chunk target is not an enforced maximum request size. The server must
validate sizes itself and bound decompression, parsing, and staging memory, not just compressed HTTP bytes.

The SDK releases training and sampling dispatch credits after submission returns an accepted future, before waiting
for GPU execution. For example, a 500 MB batch split into roughly 100 chunks targeting 5 MB can upload through a
nominal 50 MB window; accepted chunks free credits for subsequent uploads. Without a separate server input budget,
this merely transfers the backlog into Redis or another server-side store.

Its byte semaphore is intentionally soft: it admits while its remaining balance is nonnegative, then subtracts the
new request's estimate. A 10 MB budget can admit two 7 MB requests before blocking the next. The limits belong to a
holder and can be multiplied by additional holders/processes. They do not cap the caller's original dataset, cached
results, or total process memory. The server needs independently enforced aggregate and per-account limits.

Sampling also uses request-count gates and a shared backoff signal. Following recent backpressure it reduces dispatch
concurrency and charges new prompt submissions at 20 times their normal byte estimate. This is feedback for upload
pressure, not an estimate of decode work or a KV-cache allocation. Result-fetch credits are acquired when completion
metadata supplies a payload size and released after retrieval; application-held results can outlive those credits.

The byte-budget fields are in startup client config, not the separately refreshed dynamic config. Changes affect newly
constructed holders; do not promise immediate retuning of existing clients. Enable compression, metadata-only result
retrieval, or other protocol features only after implementing their matching server behavior. Values should follow
measurements of gateway, Redis, and worker pressure rather than copying another deployment's defaults.

Reference code: [config fields](https://github.com/thinking-machines-lab/tinker/blob/3ba0810907775e9938e912ed457af5dd08313708/src/tinker/types/client_config_response.py),
[holder budgets](https://github.com/thinking-machines-lab/tinker/blob/3ba0810907775e9938e912ed457af5dd08313708/src/tinker/lib/internal_client_holder.py),
[training dispatch](https://github.com/thinking-machines-lab/tinker/blob/3ba0810907775e9938e912ed457af5dd08313708/src/tinker/lib/public_interfaces/training_client.py),
[sampling dispatch](https://github.com/thinking-machines-lab/tinker/blob/3ba0810907775e9938e912ed457af5dd08313708/src/tinker/lib/public_interfaces/sampling_client.py),
and [result fetch](https://github.com/thinking-machines-lab/tinker/blob/3ba0810907775e9938e912ed457af5dd08313708/src/tinker/lib/api_future_impl.py).

### Server admission and progress

Reserve queue bytes atomically with durable command acceptance. Duplicate admissions reuse the original command and
reservation. Cancellation, failure, and payload handoff release each reservation once, at the correct lifecycle point;
a disconnected client or ambiguous admission response does not prove the command was rejected. Bound unretrieved
outputs without re-executing a training mutation when publication or download fails. Keep independent capacity for
heartbeats, cancellation, ownership checks, and recovery so payload backpressure cannot block the control operations
needed to drain the queue. Admission across accounts also needs a starvation policy; GPU fairness cannot repair an
account that never gets into the server queue.

Strict server queue limits must be designed together with sequence ordering. The inspected SDK's parallel training
path submits later chunks first and waits for their acceptance before submitting the first chunk. If those later
chunks fill a bounded server queue that cannot drain before the missing first sequence arrives, retrying admission
can deadlock before the client even sends that first chunk. Reserving headroom for a gap-closing request alone is not
sufficient when the client is still waiting for acceptance of the other chunks.

Initially return `parallel_fwdbwd_chunks: false` while introducing bounded admission and a proper per-run sequencer,
so the SDK submits the leading chunk first. Supporting the parallel mode later requires an explicit bounded logical
batch reservation/admission protocol or another proven progress rule. Oversized batches need a terminal rejection or
supported streaming path, rather than infinite admission retries. Preserve stable request identities through retries.

## Fairness accounting

Track each admitted account's weight, runnable state, cumulative GPU-seconds, virtual service, and oldest queued work.
Weights must be positive and controlled by the service. Initially, weights are fixed while work is admitted.

While a group holds `g` GPUs for `dt` seconds, charge `g * dt` GPU-seconds. For account `i` with weight `w_i`:

```text
virtual_service[i] += charged_gpu_seconds[i] / weight[i]
```

Use a monotonic clock for intervals. Charge execution-grant occupancy, including wake, restore, and park operations
that prevent another group from using the GPU set. Release is complete only after acknowledgement or confirmed process
exit. GPU utilization and command counts are efficiency signals; they are not the service entitlement.

For example, four GPUs held for five seconds and one GPU held for twenty seconds each cost twenty GPU-seconds.
An account that can use only one GPU is allowed to saturate at that demand; other accounts can use otherwise idle
capacity. The policy must not idle usable capacity solely to force raw cumulative service counters to match.

Training and sampling charge the same run account. For a shared engine, initially execute bounded chunks from one
account at a time. A later shared sampling batch divides its measured charge using an explicit work estimate, such as
prefill and decode token work; the component charges must sum to the measured total. A deterministic attribution rule
is required before enabling cross-account batching. Warm-up costs are charged by a documented rule too: initially,
charge the triggering account, record the cost separately, and evaluate this cold-start disadvantage in simulation.

An account is runnable when it has queued work that an admitted engine can execute. External reward computation,
an empty queue, and unresolved dependencies do not accrue a service claim. Waiting for its admitted parked engine to
get a turn does count as runnable. Memory-infeasible requests wait for admission with a visible reason; they must not
silently enter an execution queue they can never leave.

Maintain a monotonically advancing service floor from runnable accounts. New accounts start at the floor. On wake-up,
use `max(previous_virtual_service, floor)` so inactivity does not accumulate an unlimited priority advantage. Preserve
account identity through engine replacement, pool changes, and sampler creation. Periodically snapshot accounting
with its daemon incarnation. Recover all accounts from the last successful snapshot and report the unsaved interval;
an extended Redis outage can extend that interval. Never reset just a replaced account to zero. Do not persist local
GPU grants as if they prove that a restarted process still owns physical memory.

## Selection and GPU reservations

At an arrival, completion, execution-budget expiry, yield, admission change, or failure:

1. Account elapsed grant time and update runnable accounts.
2. Respect active grants until their supported yield boundary, unless the group has failed.
3. Order runnable accounts by virtual service, with oldest queued work as a deterministic tie-breaker. Enumerate
   their compatible engine groups and GPU masks, including memory and topology constraints.
4. Select the next owed account before checking whether its GPU group fits immediately. A currently resident engine
   gets a bounded preference among nearby service scores; it cannot repeatedly bypass an account that is owed service.
   The exact preference bound is a simulation parameter, not an unbounded locality bonus.
5. If the next owed group cannot start because GPUs are occupied, reserve a feasible GPU set and request the conflicting
   groups to yield. Do not repeatedly skip it in favor of narrower jobs.
6. Backfill unused resources only when doing so cannot delay the reservation. Grant a group its entire GPU set together.

On eight GPUs, a four-GPU trainer can starve behind a continuous stream of one-GPU samplers if selection considers
only jobs that fit immediately. Once the trainer is owed a turn, reserve four compatible GPUs. As their occupants
reach safe boundaries, keep those GPUs for the trainer. Backfill only with work whose supported yield bound and
switching margin fit before the reserved start; with no reliable bound, leave that capacity reserved.

Reservations belong to one daemon incarnation and have explicit preparation deadlines. A failed launch or a missing
park acknowledgement triggers cancellation or group recovery, rather than leaving a permanent capacity reservation.
Cancelling a reservation cancels its intent; occupied GPU masks stay unavailable until release or exit is confirmed.
Fairness is approximate under these constraints; a CFS-like ordering alone is not a proof of weighted max-min fairness
for arbitrary gang shapes. Evaluate attained shares against small feasible scheduling baselines in the simulator.

Start with one active execution grant per physical GPU. Concurrent kernels from independent groups are a later packing
optimization, requiring admission and attribution rules. Fixed GPU widths simplify the initial implementation;
changing a trainer's width requires a separate distributed checkpoint/restore contract.

## Quanta and safe switching

The engine adapter exposes operations with these semantics:

```text
run_chunk(account, budget) -> completed work, measured duration, safe-boundary status
request_yield(grant_id)   -> request a supported boundary; never permission to reuse memory yet
park(grant_id)           -> acknowledgement of synchronized memory release, plus measured residual memory
resume(grant_id, gpu_set) -> acknowledgement that the group can execute
stop(group_id)           -> confirmation that all descendants exited
```

These are interface requirements, not claims that the current PyTorch or vLLM adapters support bounded mid-command
preemption. Preserve per-run command order. A partially executed command needs an explicit continuation before it
can be split into chunks; never replay a mutating command to imitate preemption. Prefetch must not mark a large batch
as in-flight before its execution grant is available.

Measure total park/resume cost `c` for each relevant transition. Choose useful execution quantum `q` to amortize it.
For switching overhead budget `epsilon`, `c / (q + c) <= epsilon` requires:

```text
q >= c * (1 - epsilon) / epsilon
```

At a 5% target this is about `19 * c`; a 0.2-second switch suggests about four seconds of useful execution. Include
contention for host bandwidth when measuring simultaneous switches. Use measured latency, not just tensor bytes
divided by peak bandwidth.

Latency is also bounded by the longest unsupported interruption interval. If switching cost and command duration
make a requested latency target infeasible, report that constraint or limit admission; scheduler ranking cannot fix it.
Measure cold engine loading separately from warm resume, and bound the number of warm processes and cached models.

## Memory admission and physical release

Reserve GPU weights, gradients, optimizer tensors, activation headroom, residual contexts, and transfer overhead.
Reserve host RAM for parked tensors, model/cache copies, checkpoint staging, and concurrent transfers. `/dev/shm`
counts against host RAM; it is not an additional memory tier with an independent budget. Bound and evict caches.

Engine estimates are refined with measurements. Independent engines must not each treat the same free GPU bytes as
their available activation budget. An engine that cannot fit individually is rejected or routed to an appropriate
shape, rather than endlessly restarted.

Keep readiness and residency separate: an engine can be idle or backlogged independently of whether it is parked.
Use an explicit residency transition:

```text
PARKED -> WAKING -> RESIDENT -> PARKING -> PARKED
                     |            |
                     +-> FAILED <-+
```

Every transition is attached to daemon incarnation, group identity, and grant sequence. An old acknowledgement cannot
complete a newer transition. The daemon never reuses an overlapping GPU set while a group may still be executing or
releasing memory. If a transition times out, terminate the relevant process group, confirm exit, and assess GPU health;
a timeout or lease expiry alone does not prove memory release.

## Recovery, isolation, and future multi-node groups

Local parking preserves a live process's numerical state. Durable recovery needs a complete snapshot: trainable
weights, accumulated gradients, optimizer state, model/adapter configuration, RNG state, and included command boundary.
Write immutable objects before publishing aliases under the current assignment generation. Uncertain optimizer
execution requires explicit recovery; do not silently retry it. These are prerequisites, not properties added by the
fairness queue. PR #2 explores several of these contracts but also replaces the runtime; it is not an incremental
dependency of the Kubernetes PR stack.

An execution group is the failure unit. Several LoRA runs in one process share that failure boundary. The whole daemon
container is a broader lifecycle/resource boundary. Controlled model engines and arbitrary user code have different
isolation requirements; subprocesses and `CUDA_VISIBLE_DEVICES` are not tenant security boundaries. Dedicated GPUs or
nodes can provide stronger separation when required. GPU resets and node/driver failures may disrupt multiple groups.

Initially support local fixed-width groups. For multi-node execution, extend reservation, launch, and park
acknowledgements across participating daemons under one group generation. All ranks must reach a collective-safe
boundary before any daemon gives the group's GPUs to another engine. A failed rank normally requires stopping and
recovering that distributed group, not unrelated groups. Multi-node parking and elastic width changes require their
own protocol and tests; the single-node queue does not implement them automatically.

## Implementation sequence and validation

1. Define accounts, groups, grants, engine acknowledgements, and the supported safe boundaries. Replace local Redis
   GPU arbitration with daemon-controlled grants for existing one-GPU engines. Retain distributed ownership fencing.
2. Add service accounting and a pure selection function. Build deterministic event-driven trace simulation before
   tuning locality preference or switching quanta. Keep policy separate from process supervision.
3. Add host/GPU memory admission, server input/result byte budgets, compatible SDK dispatch settings, cache bounds,
   and measured switching costs. Establish sequence ordering and admission progress before enabling parallel training
   chunks. Add GPU-mask reservations when real multi-GPU engines are available; test wider groups in simulation earlier.
4. Add tenant hierarchy or shared-batch attribution when the corresponding product requirement is introduced.
   Keep burst provisioning, live cross-node balancing, and multi-node suspension outside the first implementation.

The simulation suite should include:

- Equal and weighted continuously backlogged accounts; report attained GPU-time shares and feasible allocation limits.
- One run alternating training and sampling, plus multiple sampler replicas; account identity must not multiply shares.
- Shared LoRA pools, parked engines, and adversarial locality; resident engines cannot capture all service.
- Bursty arrivals, sleepers, and replacements; no unlimited idle credit or account reset advantage.
- A wide trainer behind endless narrow jobs; verify a reservation eventually starts it under bounded yield assumptions.
- Variable command lengths and switching costs; measure queue latency and actual overhead, including quantum overruns.
- Infeasible memory requests, cache pressure, and simultaneous transfers; no double-admission of the same memory.
- Mixed tiny and oversized payloads, compression, slow uploads, and bursts of completed results; verify every byte
  reservation's lifetime and client soft-window overshoot without treating SDK budgets as hard server caps.
- Many client holders and one flooding account; enforce aggregate/per-account queue limits and preserve control traffic.
- Out-of-order training chunks with a full queue, retrying admissions, cancellation, and expired/unretrieved results;
  prove admission progress and single-release accounting, including a batch larger than the entire queue budget.
- Failed park/resume, late acknowledgements, and daemon restart; never grant overlapping execution by assuming release.

Report normalized service while accounts are runnable, maximum and percentile waits, switching overhead, admission
failures, queued/result bytes, stage-specific budget waits, and useful throughput. Jain's index of raw busy time alone
cannot establish weighted, demand-aware fairness.
Then test real PyTorch/vLLM processes and GPU memory release; CPU simulation cannot establish CUDA safety or engine
preemption bounds. No such scheduler implementation or validation result is included in this proposal.

## Related work and reading guide

These are systems research papers describing different mechanisms and objectives, not interchangeable scheduler
libraries. The policy above is a proposed synthesis and needs its own evaluation.

- [Gandiva: Introspective Cluster Scheduling for Deep Learning, OSDI 2018](https://www.usenix.org/conference/osdi18/presentation/xiao).
  Uses repetitive training behavior for GPU time-sharing, migration, and placement. Start here for the execution
  mechanisms and feedback from measurements. It does not establish the correctness of our Redis ownership protocol.
- [Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications, MLSys 2020](https://proceedings.mlsys.org/paper_files/paper/2020/hash/d9cd83bc91b8c36a0c7c0fcca59228f2-Abstract.html).
  Studies fast switching, memory sharing, and iteration scheduling in a consolidated GPU execution service. Especially
  relevant to local memory management. Its TensorFlow-integrated mechanisms are not drop-in PyTorch/vLLM adapters.
- [CFS Scheduler design, Linux kernel documentation](https://docs.kernel.org/scheduler/sched-design-CFS.html).
  Explains virtual runtime, runnable entities, weights, and scheduling granularity. This is implementation background,
  not a GPU research paper. GPU gangs, memory, and expensive switching require additional rules and evaluation.
- [Tiresias: A GPU Cluster Manager for Distributed Deep Learning, NSDI 2019](https://www.usenix.org/conference/nsdi19/presentation/gu).
  Schedules jobs without knowing their duration, using attained service that accounts for GPU count and elapsed time.
  Its objective is reducing job completion time. Its lifetime least-attained-service policy is different from our
  weighted sharing of currently backlogged accounts and wake-up floor.
- [Themis: Fair and Efficient GPU Cluster Scheduling, NSDI 2020](https://www.usenix.org/conference/nsdi20/presentation/mahajan).
  Introduces finish-time fairness and a two-level auction-based scheduler for placement-sensitive distributed jobs.
  Read it when deciding whether equal resource shares are an adequate user-facing fairness contract. This proposal
  initially uses GPU-time shares and does not adopt its auction mechanism or claim finish-time fairness.
- [Gavel: Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads, OSDI 2020](https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak).
  Uses effective throughput to adapt allocation policies to heterogeneous accelerators, then realizes allocations in
  scheduling rounds. Relevant when the fleet mixes GPU types; raw GPU-seconds are not comparable progress across types.

For the initial single-node implementation, read Gandiva and Salus for mechanisms, then the short CFS documentation
for accounting. Themis helps select the fairness objective; Tiresias and Gavel inform later policy extensions.

## Relationship to the current plan and PRs

At main commit `819081f`, node supervision, LoRA pool sharding, pinned-host-memory park/resume, sampler sleep/wake,
per-run counters, and Redis GPU leases exist. Local scheduling is still a retry race and the launcher selects one GPU.
The Kubernetes document describes memory admission, GPU groups, and load-based moves that are not all implemented.

The reviewed open PR stack adds [bucket checkpoints (#13)](https://github.com/ShubyM/tiny-open-rl/pull/13),
[node placement (#14)](https://github.com/ShubyM/tiny-open-rl/pull/14),
[drain recovery (#15)](https://github.com/ShubyM/tiny-open-rl/pull/15),
[GKE deployment and Redis restart handling (#16)](https://github.com/ShubyM/tiny-open-rl/pull/16), and
[failure testing (#17)](https://github.com/ShubyM/tiny-open-rl/pull/17). Those changes support the runtime but do not
implement weighted local scheduling. [PR #2](https://github.com/ShubyM/tiny-open-rl/pull/2) is an alternative coordinator
and immutable-state design whose contracts need deliberate reconciliation with the current trainer/sampler runtime.

The fleet-level utilization, relief, and consolidation loop can remain a placement policy. The proposed local fair
queue determines service within a node. Utilization targets alone do not specify which backlogged account is owed a turn.
