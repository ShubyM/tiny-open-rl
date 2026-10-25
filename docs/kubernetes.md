# tinyrl on Kubernetes

**Status: architecture and deployment design.** The code contains a singleton coordinator with an embedded SQLite
journal, authenticated HTTP workers, immutable tensor snapshots, and explicit run handoff/recovery commands.
The Kubernetes scheduler/agent, automatic placement/resizing, vLLM sampler, and multi-GPU trainer described below are
not implemented. Tests exercise CPU models and real
processes; they do not establish GPU memory-release or cluster-failover performance.

## Authority and execution

Kubernetes supplies GPU pods. The coordinator owns durable commands, worker assignments, checkpoint aliases, and
results. Agents decide which of their local processes can use GPU memory. The tensor store moves immutable state.

```text
clients → coordinator API → private SQLite journal on a persistent block volume
                  ↕
           worker HTTP connections
                  ↕
          GPU agents and trainers ↔ immutable tensor store
```

One logical authority does not mean one GPU or one trainer. Many trainers, agents, gateway connections, and runs can
operate concurrently. The coordinator serializes short metadata transitions; tensors and training computation stay
outside it. Workers never connect to SQLite directly. There is no Redis script layer or second authoritative queue.

This ownership model does not depend on excluding Redis. Redis can supply disposable caches or notifications if
independently scaled consumers need them; notification loss must be recoverable by reading the journal. It could also
replace the metadata backend behind the coordinator with an explicit persistence and failover contract. The current
singleton already has a durable command journal and local wakeups, so adding a second command queue would add
coordination without removing any responsibility. Choose a backend for its operational requirements, while keeping
one authority for command acceptance, ownership, and completion.

A run is the stable unit of command order and training state. A worker is a process incarnation that can host multiple
compatible LoRA runs sharing one frozen base. Pools describe this sharing. An agent is the local GPU-capacity manager.
Assignment generations prevent old worker results from becoming current state after reassignment. Worker sessions
provide liveness without requiring an independently renewed lease for every run.

## Durable commands

Before returning an accepted request id, the coordinator commits its command and deduplication identity. The run's
sequence establishes execution order even when HTTP requests arrive out of order. Repeating an accepted command with
the same identity returns its original request; reusing that identity for a different command is an error.

Dispatch records the worker session, assignment generation, and command identity. Completion checks that assignment
and commits the reply, counters, and any checkpoint alias together. Duplicate completion is harmless. A stale worker
can upload an immutable blob, but cannot publish it as the result or checkpoint of a current assignment.

Workers retain an in-flight command's result until completion is acknowledged. Following a coordinator restart they
reconnect and reconcile that result; they must not blindly execute the command again. If a worker also loses its state,
the coordinator reports the uncertain command as failed. Training updates cannot be made transactional merely by
journaling their replies.

Deleting a run closes admission and terminates queued requests explicitly. Futures that no longer exist or have expired
return terminal errors rather than leaving clients polling forever.

## Tensor state and residency

A training snapshot includes fp32 trainable weights, optimizer state, accumulated gradients, model/adapter configuration,
and random-generator state. A snapshot boundary identifies the last included command. Gradient accumulation makes a
command boundary different from an optimizer-step boundary: moving a run after `forward_backward` must preserve the
gradients needed by the following `optim_step`.

Exact snapshot continuation currently requires the same CUDA device count. Cross-device optimizer restoration is
rejected when its random-generator state cannot be restored; portable CPU/GPU or distributed resharding is separate
work. Parking a live model in CPU RAM preserves its state without rebuilding it from a checkpoint.

Sampler weights are a separate projection of this state. Immutable references identify physical blobs; logical
`tinker://` names are coordinator-owned aliases. The store does not choose placements or grant execution authority.
A new blob becomes a checkpoint only after it is completely written and its reference is committed. Retention must
keep referenced blobs and snapshots needed by in-progress transfers; eviction and garbage collection need explicit
reference/pinning rules.

RAM parking and cross-node migration share a state representation, but have different physical work. A resident LoRA
worker shares one base model; parking that worker should preserve or offload the base once. Reconstructing the whole
model from Hugging Face on every resume is not equivalent to a fast residency switch. Durable migration snapshots may
reference immutable base weights rather than duplicate them for each adapter.

An agent grants both persistent tensor memory and activation headroom. Several trainers independently measuring the
same free GPU memory cannot safely reserve it. The agent waits for a synchronized park acknowledgement, including
release of allocator-held GPU memory, before admitting another resident set. Multi-GPU workers park and resume as a
group at collective-safe boundaries. CPU-only tests cannot validate these CUDA guarantees.

## Live migration and elastic pools

All moves use one coordinator-owned transfer record. Splitting and merging LoRA pools, draining an agent, moving away
from a crowded GPU, and idling a run differ in their placement objective, not in who owns the cutover.
The current authenticated `/internal/move` endpoint initiates this handoff; automatic placement policy remains future work.

1. Reserve destination capacity and record the transfer intent. A warming destination must be placeable before it has
   runs; otherwise waiting for its readiness before moving any run creates a deadlock.
2. Stop dispatch at a recorded command boundary and ask the source to quiesce.
3. Write and pin a complete snapshot. Record its immutable reference and durability status.
4. Restore at the destination and receive a readiness acknowledgement. The destination cannot run new commands yet.
5. Commit the new owner/generation and next command position together, then resume dispatch at the destination.
6. Release the source copy and transfer pins only when the committed state is recoverable under the chosen policy.

These are persisted phases so a restarted coordinator can continue or abort a transfer. A source that dies before the
snapshot exists cannot promise lossless migration. A target that dies before cutover leaves the source authoritative.
Worker loss can put a run into explicit recovery even when a snapshot exists; the coordinator must not silently resume
older state. Snapshot-based recovery after a committed cutover requires retaining that snapshot. Commands arriving
during migration stay ordered in the run's journal.

The current `/internal/recover` endpoint accepts a `run_id`, exactly one checkpoint `path` or immutable snapshot `ref`,
and an optional `target_pool`. Recovery is an explicit operation; automatic retry and checkpoint-selection policy are
not implemented.

The future scheduler can grow busy LoRA pools, merge underused ones, and provision agent pods for pending capacity.
Memory sets the hard admission limit; measured demand and queue delay determine placement and resizing. Migration
should preserve these capabilities rather than introducing special paths for each scaling operation.

Full fine-tunes use their own execution group. Changing GPU width requires a portable checkpoint/resharding contract
and coordinated rank restart. A flat parameter-name map alone does not implement distributed checkpointing.

Samplers are independently placed replicas using immutable weight references. A sampling request must pin the exact
weight version it requests; active batches and prefix/KV caches need an explicit version-switch protocol. Agent-driven
vLLM sleep/wake is a future integration with the same local capacity manager.

## Deploying the coordinator

Use one coordinator StatefulSet replica on CPU nodes and a CSI `ReadWriteOncePod` persistent volume. Mount a block
volume with a local filesystem and keep the SQLite database and its WAL together. Use WAL mode and `synchronous=FULL`
for acknowledged metadata durability. Do not share a SQLite WAL database between machines over NFS.
[SQLite documents this application-server usage](https://www.sqlite.org/whentouse.html),
[WAL's host requirement](https://www.sqlite.org/wal.html), and
[the durability setting](https://sqlite.org/pragma.html#pragma_synchronous).

Only the coordinator opens `JOURNAL`. Workers use `COORDINATOR_URL` and authenticate with `COORDINATOR_TOKEN`.
The tensor store uses `CKPT`, separate from the metadata volume. Keep journal backups and tensor retention consistent;
use the [SQLite backup API](https://sqlite.org/backup.html) or another WAL-aware backup method.

During an ordinary update, stop admission/dispatch, flush metadata, stop the old process, and restart against the same
journal. GPU workers can retain their model state during the control-plane pause. New commands and result publication
wait until the coordinator has recovered. This deployment has restart availability, not uninterrupted HA.

`ReadWriteOncePod` restricts the PVC to one pod; ordinary `ReadWriteOnce` can allow several pods on one node.
[Kubernetes access modes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#access-modes).
The pod-count setting is not proof that a partitioned old process is dead. Force-deleting a StatefulSet pod can create
another while the original is still running, and a Kubernetes leader-election lease does not provide fencing.
[StatefulSet warning](https://kubernetes.io/docs/tasks/run-application/force-delete-stateful-set-pod/),
[client-go leader-election contract](https://pkg.go.dev/k8s.io/client-go/tools/leaderelection).

For node failover, confirm the old process/node is stopped or use a proven infrastructure fence before promotion.
Google regional Persistent Disk force-attach prevents the original VM from writing to the disk, but does not stop its
network RPCs. Assignment generations, coordinator-only publication, and conditional resource operations still matter.
[Google's disk-failover guarantee](https://docs.cloud.google.com/compute/docs/disks/repd-failover).
Kubernetes also requires care around forced detach and out-of-service nodes:
[node-shutdown procedures](https://kubernetes.io/docs/concepts/cluster-administration/node-shutdown/).

## Failure contract

| Failure | Recovery | Limit |
|---|---|---|
| Coordinator process exits | Recover journal and reconnect workers | Dispatch/publication pause; do not replay uncertain optimizer work |
| Coordinator node partitions | Wait for confirmed termination or effective fencing before replacement | Safety may require waiting |
| Metadata disk is lost | Restore a consistent backup and invalidate old sessions | Backup age bounds lost metadata |
| Worker pauses or partitions | Advance affected assignments; reject late results | Old GPU computation may continue without publication authority |
| Worker/node loses training state | Restore a known snapshot and report ambiguous/lost work | A durable reply is not a durable tensor update |
| Migration source dies | Explicit recovery from a complete retained snapshot | Otherwise recover from the previous checkpoint and report lost work |
| Migration target dies | Before cutover, retain the live source; after cutover, require explicit snapshot recovery | Automatic retry is future policy; snapshot retention must outlive loading |
| A distributed rank fails | Fence and restart the affected execution group | Requires distributed snapshot and collective recovery support |

Lossless recovery of acknowledged training progress requires durable state or a defined replay protocol in addition
to the command journal. That policy must be explicit; a metadata transaction does not manufacture missing gradients
or weights after GPU memory disappears.

Zero-downtime HA is a separate availability requirement. Independent SQLite replicas plus leader election do not meet
it. Automatic HA needs synchronously replicated authoritative state and enforced fencing using an established system;
failover and network partitions still impose availability limits. No application-written consensus is needed for the
single active coordinator deployment described here.

## Validation

The implementation should be validated against temporary real SQLite journals, real HTTP coordinator/worker processes,
and tiny CPU models: command ordering and retry identity, terminal future states, stale-worker rejection, coordinator
restart, complete snapshot continuation, checkpoint publication, and migration phase interruptions.

A GPU cluster is required to validate memory release, concurrent residency admission, multi-GPU failure handling,
vLLM version changes, provisioning/scale-down, and node fencing. Transfer bandwidth and migration latency should be
measured there rather than inferred from peak device specifications.
