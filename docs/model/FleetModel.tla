----------------------------- MODULE FleetModel -----------------------------
\* The minimal state model for independently restorable jobs.
\*
\* One row per fact, one writer per row:
\*   job      assigned (scheduler); owner, acknowledged, checkpoint (the owning worker)
\*   worker   one process launch with a unique id, a lease, and private memory
\*   gpu lock gpu -> worker id; the scheduler grants, only the node side frees
\*
\* A position in a job's command log names one exact training state. Every mutating
\* command and every explicit load gets a fresh position, so equal positions mean equal state.
\* Position 0 is the state a job is created with, which is always reproducible.
\*
\* Anything slow is its own step. Restoring a checkpoint, running a command and writing
\* checkpoint files are private to a worker; each is followed by one transaction on the
\* job row. A process can stall, lose its lease or be fenced between any two steps, and
\* everything it read from the row may be stale by the time it acts.
\*
\* The five rule switches are TRUE in the design. Setting one to FALSE removes
\* that rule, to show which invariant it is there to protect.

EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS
  Jobs, Workers, Nodes, Gpus,
  GpuNode,                    \* [Gpus -> Nodes]
  MaxCommands,                \* bound on commands per job
  None, Bucket,               \* "nothing", and the durable store as a checkpoint location
  OnlyNodeFreesGpu,           \* R1: a lapsed lease never frees a GPU
  ParkBeforeRelease,          \* R2: a worker leaves the GPU before deleting its lock
  WritesCheckOwner,           \* R3: a worker writes a job's row only while job.owner is its id
  RestoreChecksAcknowledged,  \* R4: a worker restores a job only from a checkpoint at acknowledged
  OneAtATime                  \* R5: with a command or checkpoint in flight for a job, a worker starts nothing else for it

ASSUME MaxCommands \in Nat
ASSUME GpuNode \in [Gpus -> Nodes]
ASSUME {OnlyNodeFreesGpu, ParkBeforeRelease, WritesCheckOwner, RestoreChecksAcknowledged, OneAtATime} \subseteq BOOLEAN

VARIABLES
  assigned,                   \* job -> worker it should run on (scheduler)
  owner,                      \* job -> worker whose memory holds its live state (compare-and-set)
  acknowledged,               \* job -> position of the last acknowledged mutation
  issued,                     \* job -> last position handed out
  checkpoint,                 \* job -> latest committed checkpoint [position, location]; the location is a node or Bucket
  history,                    \* ghost: job -> the positions that are ancestors of its current state
  handedOff,                  \* ghost: job -> it last left a worker by a planned handoff
  process,                    \* worker -> "new" | "run" | "dead" | "gone"
  workerNode,                 \* worker -> node
  lease,                      \* worker -> its lease key exists
  memory,                     \* worker -> job -> position of the state in its memory, or None
  inFlight,                   \* worker -> job -> position of a command run but not yet committed, or None
  loading,                    \* worker -> job -> the checkpoint an uncommitted explicit load read, or None
  written,                    \* worker -> job -> position of checkpoint files written but not yet committed, or None
  gpuLock,                    \* gpu -> worker holding it in the lock table, or None
  onGpu,                      \* worker -> gpus its process is physically using
  nodeUp                      \* node -> the machine is running

jobVars == <<assigned, owner, acknowledged, issued, checkpoint, history, handedOff>>
workerVars == <<process, workerNode, lease, memory, inFlight, loading, written>>
gpuVars == <<gpuLock, onGpu>>
vars == <<jobVars, workerVars, gpuVars, nodeUp>>

CheckpointRef == [position : 0..MaxCommands, location : Nodes \cup {Bucket}]
NoJobs == [j \in Jobs |-> None]

TypeOK ==
  /\ assigned \in [Jobs -> Workers \cup {None}]
  /\ owner \in [Jobs -> Workers \cup {None}]
  /\ acknowledged \in [Jobs -> 0..MaxCommands]
  /\ issued \in [Jobs -> 0..MaxCommands]
  /\ checkpoint \in [Jobs -> CheckpointRef]
  /\ \A j \in Jobs : history[j] \in Seq(0..MaxCommands)
  /\ handedOff \in [Jobs -> BOOLEAN]
  /\ process \in [Workers -> {"new", "run", "dead", "gone"}]
  /\ workerNode \in [Workers -> Nodes \cup {None}]
  /\ lease \in [Workers -> BOOLEAN]
  /\ memory \in [Workers -> [Jobs -> (0..MaxCommands) \cup {None}]]
  /\ inFlight \in [Workers -> [Jobs -> (0..MaxCommands) \cup {None}]]
  /\ loading \in [Workers -> [Jobs -> CheckpointRef \cup {None}]]
  /\ written \in [Workers -> [Jobs -> (0..MaxCommands) \cup {None}]]
  /\ gpuLock \in [Gpus -> Workers \cup {None}]
  /\ onGpu \in [Workers -> SUBSET Gpus]
  /\ nodeUp \in [Nodes -> BOOLEAN]

Init ==
  /\ assigned = [j \in Jobs |-> None]
  /\ owner = [j \in Jobs |-> None]
  /\ acknowledged = [j \in Jobs |-> 0]
  /\ issued = [j \in Jobs |-> 0]
  /\ checkpoint = [j \in Jobs |-> [position |-> 0, location |-> Bucket]]
  /\ history = [j \in Jobs |-> <<0>>]
  /\ handedOff = [j \in Jobs |-> TRUE]
  /\ process = [w \in Workers |-> "new"]
  /\ workerNode = [w \in Workers |-> None]
  /\ lease = [w \in Workers |-> FALSE]
  /\ memory = [w \in Workers |-> NoJobs]
  /\ inFlight = [w \in Workers |-> NoJobs]
  /\ loading = [w \in Workers |-> NoJobs]
  /\ written = [w \in Workers |-> NoJobs]
  /\ gpuLock = [g \in Gpus |-> None]
  /\ onGpu = [w \in Workers |-> {}]
  /\ nodeUp = [n \in Nodes |-> TRUE]

LocksOf(w) == {g \in Gpus : gpuLock[g] = w}
\* A checkpoint can be read from the bucket, or from the node that wrote it while that node is up.
Readable(j) == IF checkpoint[j].location = Bucket THEN TRUE ELSE nodeUp[checkpoint[j].location]
\* The worker has something in flight for the job.
Busy(w, j) == inFlight[w][j] # None \/ written[w][j] # None
\* What every write to a job's row checks in its transaction.
MayWrite(w, j) == WritesCheckOwner => owner[j] = w
UpTo(s, x) == SelectSeq(s, LAMBDA y : y <= x)
Last(s) == s[Len(s)]

\* ***** the scheduler *****

StartWorker(w, n) ==
  /\ process[w] = "new" /\ nodeUp[n]
  /\ process' = [process EXCEPT ![w] = "run"]
  /\ workerNode' = [workerNode EXCEPT ![w] = n]
  /\ lease' = [lease EXCEPT ![w] = TRUE]
  /\ UNCHANGED <<jobVars, memory, inFlight, loading, written, gpuVars, nodeUp>>

\* The scheduler sees only the lease, so it may assign a job to a process that has already died.
Assign(j, w) ==
  /\ process[w] # "new" /\ lease[w] /\ assigned[j] # w
  /\ assigned' = [assigned EXCEPT ![j] = w]
  /\ UNCHANGED <<owner, acknowledged, issued, checkpoint, history, handedOff, workerVars, gpuVars, nodeUp>>

\* Take a job away from its worker. The design does this when the owner's lease is gone;
\* here it may happen at any moment, so safety cannot depend on when.
Fence(j) ==
  /\ owner[j] # None
  /\ owner' = [owner EXCEPT ![j] = None]
  /\ handedOff' = [handedOff EXCEPT ![j] = FALSE]
  /\ UNCHANGED <<assigned, acknowledged, issued, checkpoint, history, workerVars, gpuVars, nodeUp>>

GrantGpus(w, G) ==
  /\ process[w] # "new" /\ lease[w]
  /\ G # {} /\ LocksOf(w) = {}
  /\ \A g \in G : GpuNode[g] = workerNode[w] /\ gpuLock[g] = None
  /\ gpuLock' = [g \in Gpus |-> IF g \in G THEN w ELSE gpuLock[g]]
  /\ UNCHANGED <<jobVars, workerVars, onGpu, nodeUp>>

\* Only without R1: the scheduler frees the GPUs of a worker whose lease lapsed.
SchedulerFreesGpus(w) ==
  /\ ~OnlyNodeFreesGpu
  /\ process[w] # "new" /\ ~lease[w] /\ LocksOf(w) # {}
  /\ gpuLock' = [g \in Gpus |-> IF gpuLock[g] = w THEN None ELSE gpuLock[g]]
  /\ UNCHANGED <<jobVars, workerVars, onGpu, nodeUp>>

\* ***** a worker and its GPUs *****

UseGpus(w) ==
  /\ process[w] = "run" /\ onGpu[w] = {} /\ LocksOf(w) # {}
  /\ onGpu' = [onGpu EXCEPT ![w] = LocksOf(w)]
  /\ UNCHANGED <<jobVars, workerVars, gpuLock, nodeUp>>

Park(w) ==
  /\ process[w] = "run" /\ onGpu[w] # {}
  /\ \A j \in Jobs : inFlight[w][j] = None
  /\ onGpu' = [onGpu EXCEPT ![w] = {}]
  /\ UNCHANGED <<jobVars, workerVars, gpuLock, nodeUp>>

ReleaseGpus(w) ==
  /\ process[w] = "run" /\ LocksOf(w) # {}
  /\ ParkBeforeRelease => onGpu[w] = {}
  /\ gpuLock' = [g \in Gpus |-> IF gpuLock[g] = w THEN None ELSE gpuLock[g]]
  /\ UNCHANGED <<jobVars, workerVars, onGpu, nodeUp>>

\* ***** a worker and a job: private steps *****

\* Read the job's row and load its checkpoint into memory. Nothing checks that this worker still owns the job.
Restore(w, j) ==
  /\ process[w] = "run"
  /\ memory[w][j] = None /\ ~Busy(w, j)
  /\ Readable(j)
  /\ RestoreChecksAcknowledged => checkpoint[j].position = acknowledged[j]
  /\ memory' = [memory EXCEPT ![w][j] = checkpoint[j].position]
  /\ UNCHANGED <<jobVars, process, workerNode, lease, inFlight, loading, written, gpuVars, nodeUp>>

\* Claim the next command and run it in memory. The claim checks nothing.
RunCommand(w, j) ==
  /\ process[w] = "run"
  /\ memory[w][j] # None /\ inFlight[w][j] = None
  /\ OneAtATime => written[w][j] = None
  /\ onGpu[w] # {}
  /\ issued[j] < MaxCommands
  /\ issued' = [issued EXCEPT ![j] = @ + 1]
  /\ memory' = [memory EXCEPT ![w][j] = issued[j] + 1]
  /\ inFlight' = [inFlight EXCEPT ![w][j] = issued[j] + 1]
  /\ UNCHANGED <<assigned, owner, acknowledged, checkpoint, history, handedOff, process, workerNode, lease, loading, written, gpuVars, nodeUp>>

\* The client's explicit restore is a command too: it gets a fresh position, which then names the loaded state.
\* It needs nothing in memory, which is how a job leaves a rollback.
StartLoad(w, j) ==
  /\ process[w] = "run"
  /\ ~Busy(w, j)
  /\ Readable(j)
  /\ issued[j] < MaxCommands
  /\ loading' = [loading EXCEPT ![w][j] = checkpoint[j]]
  /\ issued' = [issued EXCEPT ![j] = @ + 1]
  /\ memory' = [memory EXCEPT ![w][j] = issued[j] + 1]
  /\ inFlight' = [inFlight EXCEPT ![w][j] = issued[j] + 1]
  /\ UNCHANGED <<assigned, owner, acknowledged, checkpoint, history, handedOff, process, workerNode, lease, written, gpuVars, nodeUp>>

\* Write the state in memory to checkpoint files. Nothing shared changes yet.
WriteCheckpoint(w, j) ==
  /\ process[w] = "run"
  /\ memory[w][j] # None /\ ~Busy(w, j)
  /\ written' = [written EXCEPT ![w][j] = memory[w][j]]
  /\ UNCHANGED <<jobVars, process, workerNode, lease, memory, inFlight, loading, gpuVars, nodeUp>>

\* ***** a worker and a job: one transaction on the job row *****

\* Become the job's owner. The worker first drops whatever it held for the job, so it starts from the row.
TakeOwnership(w, j) ==
  /\ process[w] = "run"
  /\ assigned[j] = w /\ owner[j] = None
  /\ OneAtATime => ~Busy(w, j)
  /\ owner' = [owner EXCEPT ![j] = w]
  /\ memory' = [memory EXCEPT ![w][j] = None]
  /\ UNCHANGED <<assigned, acknowledged, issued, checkpoint, history, handedOff, process, workerNode, lease, inFlight, loading, written, gpuVars, nodeUp>>

\* Publish a command's reply. A worker that no longer owns the job publishes nothing and forgets it.
CommitReply(w, j) ==
  LET p == inFlight[w][j]
      src == loading[w][j]
  IN
  /\ process[w] = "run" /\ p # None
  /\ IF MayWrite(w, j)
       THEN /\ acknowledged' = [acknowledged EXCEPT ![j] = p]
            /\ memory' = memory
            /\ IF src # None
                 THEN /\ history' = [history EXCEPT ![j] = Append(UpTo(@, src.position), p)]
                      /\ checkpoint' = [checkpoint EXCEPT ![j] = [position |-> p, location |-> src.location]]
                 ELSE /\ history' = [history EXCEPT ![j] = Append(@, p)]
                      /\ UNCHANGED checkpoint
       ELSE /\ memory' = [memory EXCEPT ![w][j] = None]
            /\ UNCHANGED <<acknowledged, history, checkpoint>>
  /\ inFlight' = [inFlight EXCEPT ![w][j] = None]
  /\ loading' = [loading EXCEPT ![w][j] = None]
  /\ UNCHANGED <<assigned, owner, issued, handedOff, process, workerNode, lease, written, gpuVars, nodeUp>>

\* Commit a written checkpoint and keep serving the job.
CommitCheckpoint(w, j) ==
  /\ process[w] = "run" /\ written[w][j] # None
  /\ MayWrite(w, j) /\ assigned[j] = w
  /\ checkpoint' = [checkpoint EXCEPT ![j] = [position |-> written[w][j], location |-> workerNode[w]]]
  /\ written' = [written EXCEPT ![w][j] = None]
  /\ UNCHANGED <<assigned, owner, acknowledged, issued, history, handedOff, process, workerNode, lease, memory, inFlight, loading, gpuVars, nodeUp>>

\* Planned handoff: committing the checkpoint and giving up ownership are one write.
HandOff(w, j) ==
  /\ process[w] = "run" /\ written[w][j] # None
  /\ MayWrite(w, j) /\ assigned[j] # w
  /\ checkpoint' = [checkpoint EXCEPT ![j] = [position |-> written[w][j], location |-> workerNode[w]]]
  /\ owner' = [owner EXCEPT ![j] = None]
  /\ handedOff' = [handedOff EXCEPT ![j] = TRUE]
  /\ memory' = [memory EXCEPT ![w][j] = None]
  /\ written' = [written EXCEPT ![w][j] = None]
  /\ UNCHANGED <<assigned, acknowledged, issued, history, process, workerNode, lease, inFlight, loading, gpuVars, nodeUp>>

\* Give up ownership without writing a checkpoint: the worker holds nothing, or the checkpoint is already at acknowledged.
ReleaseOwnership(w, j) ==
  /\ process[w] = "run" /\ ~Busy(w, j)
  /\ MayWrite(w, j)
  /\ memory[w][j] = None \/ checkpoint[j].position = acknowledged[j]
  /\ owner' = [owner EXCEPT ![j] = None]
  /\ handedOff' = [handedOff EXCEPT ![j] = (handedOff[j] \/ checkpoint[j].position = acknowledged[j])]
  /\ memory' = [memory EXCEPT ![w][j] = None]
  /\ UNCHANGED <<assigned, acknowledged, issued, checkpoint, history, process, workerNode, lease, inFlight, loading, written, gpuVars, nodeUp>>

\* The transaction for a written checkpoint failed: the files are orphans and the worker forgets the job.
DiscardCheckpoint(w, j) ==
  /\ process[w] = "run" /\ written[w][j] # None
  /\ ~MayWrite(w, j)
  /\ written' = [written EXCEPT ![w][j] = None]
  /\ memory' = [memory EXCEPT ![w][j] = None]
  /\ UNCHANGED <<jobVars, process, workerNode, lease, inFlight, loading, gpuVars, nodeUp>>

\* ***** storage *****

\* The node uploads the latest checkpoint behind; once in the bucket it outlives the node.
Upload(j) ==
  /\ checkpoint[j].location # Bucket /\ nodeUp[checkpoint[j].location]
  /\ checkpoint' = [checkpoint EXCEPT ![j].location = Bucket]
  /\ UNCHANGED <<assigned, owner, acknowledged, issued, history, handedOff, workerVars, gpuVars, nodeUp>>

\* ***** failures, and the node cleaning up after them *****

LeaseExpire(w) ==
  /\ process[w] \in {"run", "dead"} /\ lease[w]
  /\ lease' = [lease EXCEPT ![w] = FALSE]
  /\ UNCHANGED <<jobVars, process, workerNode, memory, inFlight, loading, written, gpuVars, nodeUp>>

\* A process that stalled past its lease, or lost Redis for a while, holds it again and carries on.
\* Whatever was fenced meanwhile it no longer owns; the rest it never lost.
RenewLease(w) ==
  /\ process[w] = "run" /\ ~lease[w]
  /\ lease' = [lease EXCEPT ![w] = TRUE]
  /\ UNCHANGED <<jobVars, process, workerNode, memory, inFlight, loading, written, gpuVars, nodeUp>>

ProcessesDie(S) ==
  /\ process' = [w \in Workers |-> IF w \in S THEN "dead" ELSE process[w]]
  /\ onGpu' = [w \in Workers |-> IF w \in S THEN {} ELSE onGpu[w]]
  /\ memory' = [w \in Workers |-> IF w \in S THEN NoJobs ELSE memory[w]]
  /\ inFlight' = [w \in Workers |-> IF w \in S THEN NoJobs ELSE inFlight[w]]
  /\ loading' = [w \in Workers |-> IF w \in S THEN NoJobs ELSE loading[w]]
  /\ written' = [w \in Workers |-> IF w \in S THEN NoJobs ELSE written[w]]

\* The process is gone: a crash, or its node daemon killing it.
Crash(w) ==
  /\ process[w] = "run"
  /\ ProcessesDie({w})
  /\ UNCHANGED <<jobVars, workerNode, lease, gpuLock, nodeUp>>

\* The node daemon has confirmed the process group is gone, and frees its GPUs.
ConfirmDead(w) ==
  /\ process[w] = "dead" /\ nodeUp[workerNode[w]]
  /\ process' = [process EXCEPT ![w] = "gone"]
  /\ gpuLock' = [g \in Gpus |-> IF gpuLock[g] = w THEN None ELSE gpuLock[g]]
  /\ UNCHANGED <<jobVars, workerNode, lease, memory, inFlight, loading, written, onGpu, nodeUp>>

NodeDown(n) ==
  /\ nodeUp[n]
  /\ nodeUp' = [nodeUp EXCEPT ![n] = FALSE]
  /\ ProcessesDie({w \in Workers : process[w] = "run" /\ workerNode[w] = n})
  /\ UNCHANGED <<jobVars, workerNode, lease, gpuLock>>

Next ==
  \/ \E w \in Workers, n \in Nodes : StartWorker(w, n)
  \/ \E j \in Jobs, w \in Workers :
       \/ Assign(j, w)
       \/ Restore(w, j) \/ RunCommand(w, j) \/ StartLoad(w, j) \/ WriteCheckpoint(w, j)
       \/ TakeOwnership(w, j) \/ CommitReply(w, j) \/ CommitCheckpoint(w, j)
       \/ HandOff(w, j) \/ ReleaseOwnership(w, j) \/ DiscardCheckpoint(w, j)
  \/ \E j \in Jobs : Fence(j) \/ Upload(j)
  \/ \E w \in Workers, G \in SUBSET Gpus : GrantGpus(w, G)
  \/ \E w \in Workers :
       UseGpus(w) \/ Park(w) \/ ReleaseGpus(w) \/ SchedulerFreesGpus(w) \/ LeaseExpire(w) \/ RenewLease(w) \/ Crash(w) \/ ConfirmDead(w)
  \/ \E n \in Nodes : NodeDown(n)

Spec == Init /\ [][Next]_vars

\* ***** what must always hold *****

\* No two processes are physically on one GPU, and a process on a GPU holds its lock.
GpuExclusive == \A g \in Gpus : Cardinality({w \in Workers : g \in onGpu[w]}) <= 1
GpuUserHoldsLock == \A w \in Workers : \A g \in onGpu[w] : gpuLock[g] = w

\* The state an owner serves a job from is exactly the acknowledged one: never old tensors under a newer cursor.
MemoryIsAcknowledged ==
  \A j \in Jobs : \A w \in Workers :
    (owner[j] = w /\ memory[w][j] # None /\ inFlight[w][j] = None) => memory[w][j] = acknowledged[j]

\* Acknowledged progress only moves forward, along one line of history.
HistoryMovesForward ==
  \A j \in Jobs :
    /\ acknowledged[j] = Last(history[j])
    /\ \A i \in 1..(Len(history[j]) - 1) : history[j][i] < history[j][i + 1]

\* The checkpoint is a state the job really passed through, never ahead of what was acknowledged.
InHistory(j, x) == \E i \in 1..Len(history[j]) : history[j][i] = x
CheckpointInHistory == \A j \in Jobs : InHistory(j, checkpoint[j].position)
CheckpointOrder == \A j \in Jobs : checkpoint[j].position <= acknowledged[j] /\ acknowledged[j] <= issued[j]

\* A planned handoff loses nothing: the checkpoint it leaves is at acknowledged.
HandoffLossless == \A j \in Jobs : (handedOff[j] /\ owner[j] = None) => checkpoint[j].position = acknowledged[j]

=============================================================================
