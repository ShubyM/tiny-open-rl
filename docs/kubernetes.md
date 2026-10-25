# tinyrl on Kubernetes

**Status: architecture and deployment design.** The code contains one active coordinator per Redis namespace,
authenticated HTTP workers, shared immutable payload/tensor files, and explicit run handoff/recovery commands.
The Kubernetes scheduler/agent, automatic placement/resizing, vLLM sampler, and multi-GPU trainer described below are
not implemented. Tests exercise CPU models and real
processes; they do not establish GPU memory-release or cluster-failover performance.

## Authority and execution

Kubernetes supplies GPU pods. The coordinator owns durable commands, worker assignments, checkpoint aliases, and
results. Agents decide which of their local processes can use GPU memory. The tensor store moves immutable state.

```text
clients → coordinator API → Redis command metadata, assignments, and results
                  ↕
           worker HTTP connections
                  ↕
          GPU agents and trainers ↔ shared payload files and tensor snapshots
```

One logical authority does not mean one GPU or one trainer. Many trainers, agents, gateway connections, and runs can
operate concurrently. The coordinator serializes short metadata transitions; tensors and training computation stay
outside it. Redis stores request-processing state using native primitives and transactions. Scheduling, assignment,
and recovery decisions stay in coordinator code; there is no Lua policy layer. Workers use the authenticated
coordinator API rather than making independent ownership decisions against Redis.

A run is the stable unit of command order and training state. A worker is a process incarnation that can host multiple
compatible LoRA runs sharing one frozen base. Pools describe this sharing. An agent is the local GPU-capacity manager.
Assignment generations prevent old worker results from becoming current state after reassignment. Worker sessions
provide liveness without requiring an independently renewed lease for every run.

## Durable commands

Before returning an accepted request id, the coordinator commits its command and deduplication identity to Redis. The run's
sequence establishes execution order even when HTTP requests arrive out of order. Repeating an accepted command with
the same identity returns its original request; reusing that identity for a different command is an error.

Dispatch records the worker session, assignment generation, and command identity. Completion checks that assignment
and commits the reply, counters, and any checkpoint alias together. Duplicate completion is harmless. A stale worker
can upload an immutable blob, but cannot publish it as the result or checkpoint of a current assignment.

Workers retain an in-flight command's result until completion is acknowledged. Following a coordinator restart they
reconnect and reconcile that result; they must not blindly execute the command again. If a worker also loses its state,
the coordinator reports the uncertain command as failed. Training updates cannot be made transactional merely by
persisting their replies.

Deleting a run closes admission and terminates queued requests explicitly. Futures that no longer exist or have expired
return terminal errors rather than leaving clients polling forever.

Training payloads and results above 64 KiB are stored as immutable files addressed by SHA-256. Redis metadata and
the worker HTTP protocol carry their references, and workers read payloads directly from the shared mount.
`PAYLOAD_DIR` defaults to `CKPT/payloads`; coordinator and workers must see the same files. Small bodies stay inline. This keeps large batches
and numerical outputs out of Redis transactions without introducing a second command queue. The current storage
implementation uses shared filesystems; an object-storage/S3 backend is not implemented.

Request and result bodies are capped at 64 MiB by default. If numerical execution produces an oversized result, the
worker reports an explicit failure requiring recovery; it does not retry the training mutation to produce a smaller
response. Workers fetch one command at a time to bound prefetched payload memory.

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
during migration stay ordered in the run's request records.

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

## Deploying Redis and the coordinator

Use Redis 7 or newer on persistent storage, with `appendonly yes`, `appendfsync always`, and
`maxmemory-policy noeviction` for the documented durability contract. Snapshot-only persistence or AOF `everysec`
can lose accepted commands after a crash. Configure backups and capacity monitoring; a full metadata store must
stop acceptance rather than evict command or deduplication state. These settings are deployment requirements,
not something the coordinator silently changes on a shared Redis service.
[Redis persistence](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/).

The coordinator uses `REDIS_URL` and `REDIS_NAMESPACE`. Workers use `COORDINATOR_URL` and authenticate with
`COORDINATOR_TOKEN`. Mount `CKPT` and `PAYLOAD_DIR` on shared persistent storage with the same view from every
process that needs them. Keep Redis backups and blob retention consistent: a restored reference is useful only if
its payload or checkpoint still exists. Garbage collection must preserve queued payloads, retained results,
checkpoint aliases, and migration snapshots; automatic blob garbage collection is not implemented.

One coordinator holds a renewable namespace lease. Metadata transitions verify its ownership with
`WATCH`/`MULTI`/`EXEC`, so a process whose lease expired cannot publish a new assignment or completion against the
same Redis history. Losing the lease is terminal for that coordinator instance. Assignment generations and worker
sessions separately reject late worker results. The lease does not stop already-running GPU computation.
[Redis transaction semantics](https://redis.io/docs/latest/develop/using-commands/transactions/).

The namespace is private to the coordinator; external writes can violate its key types and invariants. Redis does not
roll back runtime errors inside `EXEC`. Such errors stop that coordinator and log an instruction to inspect and repair
the namespace before restarting; they are not treated as ordinary connection retries.

For a normal update, stop the old coordinator and release its lease, then start its replacement against the same
Redis namespace. After a crash, wait for lease expiry (`COORDINATOR_TTL`, default 10 seconds). Workers retain their
model and unacknowledged result while reconnecting. Dispatch and result publication pause during replacement;
this is restart availability, not uninterrupted HA. The health endpoint checks and renews coordinator authority.

A coordinator that has lost its lease returns HTTP 503 and never reacquires it in the same process. Readiness removes
traffic but cannot repair this state: configure a liveness/restart policy that replaces a fenced instance. Allow startup
and restart backoff enough time for the previous lease to expire, plus Redis connection and process-start time; a
replacement started while another lease is still live can fail startup and needs retrying. Avoid a short probe budget
that continually kills replacements during the expected failover pause. Probe and lease timings belong in the deployment
configuration; the two-second lease in the SDK test is only for that controlled test.

Deploy one active coordinator process, with Redis access restricted to the coordinator service. Kubernetes pod
counts and leader-election leases alone do not fence a partitioned old process. Redis lease checks protect metadata
only while all participants use the same authoritative Redis history. A replacement Redis primary must not coexist
with an old writable primary or silently roll back accepted work.
[StatefulSet warning](https://kubernetes.io/docs/tasks/run-application/force-delete-stateful-set-pod/),
[client-go leader-election contract](https://pkg.go.dev/k8s.io/client-go/tools/leaderelection).

Redis replication is asynchronous. Sentinel/Cluster promotion can lose acknowledged writes, and `WAIT` does not
turn it into a strongly consistent store. Do not assume AOF on one primary makes replica promotion lossless.
Automatic failover needs an explicit data-loss and fencing contract; after a rollback, stop admission and reconcile
affected sessions and model state before resuming. The application does not implement that disaster-recovery policy.
[Redis replication guarantees](https://redis.io/docs/latest/operate/oss_and_stack/management/replication/).

## Failure contract

| Failure | Recovery | Limit |
|---|---|---|
| Coordinator process exits | Reacquire expired/released lease and reconnect against the same Redis | Dispatch/publication pause; do not replay uncertain optimizer work |
| Coordinator node partitions | Redis lease checks reject its later transitions; replacement waits for expiry | Requires a single authoritative Redis history; old computation may continue |
| Redis process restarts on its intact AOF | Recover metadata, then reconcile coordinator and workers | Durability depends on the configured fsync policy and storage |
| Redis disk loss or replica rollback | Stop admission; restore/reconcile metadata and invalidate affected sessions | Backup/replication lag bounds lost metadata; not automatic |
| Shared payload/checkpoint storage is lost | Restore the referenced files before execution or recovery | Redis references cannot reconstruct missing tensors or request bodies |
| Worker pauses or partitions | Advance affected assignments; reject late results | Old GPU computation may continue without publication authority |
| Worker/node loses training state | Restore a known snapshot and report ambiguous/lost work | A durable reply is not a durable tensor update |
| Migration source dies | Explicit recovery from a complete retained snapshot | Otherwise recover from the previous checkpoint and report lost work |
| Migration target dies | Before cutover, retain the live source; after cutover, require explicit snapshot recovery | Automatic retry is future policy; snapshot retention must outlive loading |
| A distributed rank fails | Fence and restart the affected execution group | Requires distributed snapshot and collective recovery support |

Lossless recovery of acknowledged training progress requires durable state or a defined replay protocol in addition
to durable request records. That policy must be explicit; a metadata transaction does not manufacture missing gradients
or weights after GPU memory disappears.

Zero-downtime HA is a separate availability requirement. A second coordinator and asynchronous Redis failover do not
by themselves provide it. Lossless failover needs a storage contract that preserves acknowledged state and enforces
fencing; network partitions still impose availability limits. The current deployment relies on native Redis primitives
and one coordinator authority rather than implementing application-level consensus.

## Validation

The implementation should be validated against isolated real Redis servers with AOF enabled, real HTTP coordinator/worker processes,
and tiny CPU models: command ordering and retry identity, terminal future states, stale-worker rejection, coordinator
restart, complete snapshot continuation, checkpoint publication, and migration phase interruptions.

A GPU cluster is required to validate memory release, concurrent residency admission, multi-GPU failure handling,
vLLM version changes, provisioning/scale-down, and node fencing. Transfer bandwidth and migration latency should be
measured there rather than inferred from peak device specifications.
