----------------------------- MODULE PagedFleet -----------------------------
\* Minimal control-plane model for GPU sharing, gang scheduling, and rebalancing.
\*
\* Truth (safety state):
\*   run[r]   = (owner, pos, snap)   single-writer CAS record for stateful runs
\*   lease[p] = BOOLEAN              process liveness TTL
\*   on[p]    = SUBSET Gpus          physical GPUs held on p's node (node-local)
\*
\* Plan (pure Lampson hint — arbitrary / split-brain planner has zero safety authority):
\*   slot[g]  = p | None             target L0 process for GPU g
\*   home[r]  = p | None             target process for run r (None = evict to store)
\*
\* Data-plane transitions:
\*   Promote / Demote                L1 (host RAM) <-> L0 (GPU VRAM); keeps dirty owner
\*   Attach  / Flush + Detach        L2/L3 (store) <-> process; transfers owner via snap
\*   SaveState / DetachClean         clean run detaches with zero extra write
\*   LoadState                       recovers a run even after a dirty crash
\*   SampleExec / SampleCommit       stateless readers execute concurrently, commit once

EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS
  Runs,                 \* stateful (writer) runs
  SampleCmds,           \* stateless (reader) commands shared across all processes
  Procs,                \* process launches (unique ids)
  Nodes,                \* cluster machines
  Gpus,                 \* physical GPUs across nodes
  GpuNode,              \* [Gpus -> Nodes]
  ProcWidth,            \* [Procs -> Nat \ {0}] (1 = single-GPU, >=2 = multi-GPU gang)
  MaxPos,               \* bound on mutations per run
  None, Bucket,         \* sentinel values
  ClearAfterExit,       \* R1: node clears on[p] only after p physically left GPUs / died
  OwnerFencesWrite,     \* R2: CommitReply / Detach CAS-checks owner[r] = p
  ExactSnapOnly,        \* R3: dirty commit clears snap (only exact pos is restorable)
  ReapChecksLease,      \* R4: Reap clears owner[r] = p only while ~lease[p]
  SerializeRunOps,      \* R5: at most one in-flight cmd or flush per (p, r)
  SampleSetNX           \* R6: reader commit uses SET NX on reply[c]

ASSUME GpuNode \in [Gpus -> Nodes]
ASSUME ProcWidth \in [Procs -> Nat \ {0}]
ASSUME MaxPos \in Nat
ASSUME {ClearAfterExit, OwnerFencesWrite, ExactSnapOnly,
        ReapChecksLease, SerializeRunOps, SampleSetNX} \subseteq BOOLEAN

VARIABLES
  \* --- Plan (arbitrary hints; written by any number of racing planners) ---
  slot,         \* [Gpus -> Procs \cup {None}]
  home,         \* [Runs -> Procs \cup {None}]
  \* --- Truth: run row in store ---
  owner,        \* [Runs -> Procs \cup {None}]
  pos,          \* [Runs -> 0..MaxPos]
  issued,       \* [Runs -> 0..MaxPos]
  snap,         \* [Runs -> [pos : 0..MaxPos, loc : Nodes \cup {Bucket, None}]]
  \* --- Truth: stateless reader replies ---
  sampleReplied,\* SUBSET SampleCmds
  sampleCharges,\* [SampleCmds -> 0..Cardinality(Procs)] (ghost: times billed)
  \* --- Truth: node & process physical state ---
  nodeUp,       \* [Nodes -> BOOLEAN]
  procState,    \* [Procs -> {"unused", "live", "dead", "reaped"}]
  procNode,     \* [Procs -> Nodes \cup {None}]
  lease,        \* [Procs -> BOOLEAN]
  on,           \* [Procs -> SUBSET Gpus]   node daemon's local occupancy record
  vram,         \* [Procs -> SUBSET Gpus]   physical VRAM occupancy (may lag release if R1 off)
  mem,          \* [Procs -> [Runs -> (0..MaxPos) \cup {None}]]  L0/L1 run tensors
  running,      \* [Procs -> [Runs -> (1..MaxPos) \cup {None}]]  in-flight mutation/load
  attaching,    \* [Procs -> [Runs -> (0..MaxPos) \cup {None}]]  in-flight snap load
  flushed,      \* [Procs -> [Runs -> (0..MaxPos) \cup {None}]]  finished local flush awaiting Detach/Save
  \* --- Ghost variables for invariants ---
  history,      \* [Runs -> Seq(0..MaxPos)]
  detached      \* [Runs -> BOOLEAN] true iff last release of owner[r] was a planned Detach

planVars   == <<slot, home>>
runVars    == <<owner, pos, issued, snap, history, detached>>
sampleVars == <<sampleReplied, sampleCharges>>
procVars   == <<procState, procNode, lease, on, vram, mem, running, attaching, flushed>>
vars       == <<planVars, runVars, sampleVars, nodeUp, procVars>>

EmptyRuns == [r \in Runs |-> None]
NoSnap    == [pos |-> 0, loc |-> None]
Last(s)   == s[Len(s)]

NodeGpus(n) == {g \in Gpus : GpuNode[g] = n}
WantGpus(p) == IF procNode[p] = None THEN {} ELSE {g \in NodeGpus(procNode[p]) : slot[g] = p}
SnapReachable(s) == s.loc = Bucket \/ (s.loc \in Nodes /\ nodeUp[s.loc])
SnapExact(r) == snap[r].loc # None /\ snap[r].pos = pos[r]
RunIdle(p, r) == ~SerializeRunOps \/ (running[p][r] = None /\ flushed[p][r] = None /\ attaching[p][r] = None)
CanCommit(p, r) == ~OwnerFencesWrite \/ owner[r] = p

SnapSet == [pos : 0..MaxPos, loc : Nodes \cup {Bucket, None}]

TypeOK ==
  /\ slot \in [Gpus -> Procs \cup {None}]
  /\ home \in [Runs -> Procs \cup {None}]
  /\ owner \in [Runs -> Procs \cup {None}]
  /\ pos \in [Runs -> 0..MaxPos]
  /\ issued \in [Runs -> 0..MaxPos]
  /\ snap \in [Runs -> SnapSet]
  /\ sampleReplied \subseteq SampleCmds
  /\ sampleCharges \in [SampleCmds -> 0..Cardinality(Procs)]
  /\ nodeUp \in [Nodes -> BOOLEAN]
  /\ procState \in [Procs -> {"unused", "live", "dead", "reaped"}]
  /\ procNode \in [Procs -> Nodes \cup {None}]
  /\ lease \in [Procs -> BOOLEAN]
  /\ on \in [Procs -> SUBSET Gpus]
  /\ vram \in [Procs -> SUBSET Gpus]
  /\ mem \in [Procs -> [Runs -> (0..MaxPos) \cup {None}]]
  /\ running \in [Procs -> [Runs -> (1..MaxPos) \cup {None}]]
  /\ attaching \in [Procs -> [Runs -> (0..MaxPos) \cup {None}]]
  /\ flushed \in [Procs -> [Runs -> (0..MaxPos) \cup {None}]]
  /\ detached \in [Runs -> BOOLEAN]

Init ==
  /\ slot = [g \in Gpus |-> None]
  /\ home = [r \in Runs |-> None]
  /\ owner = [r \in Runs |-> None]
  /\ pos = [r \in Runs |-> 0]
  /\ issued = [r \in Runs |-> 0]
  /\ snap = [r \in Runs |-> NoSnap]
  /\ sampleReplied = {}
  /\ sampleCharges = [c \in SampleCmds |-> 0]
  /\ nodeUp = [n \in Nodes |-> TRUE]
  /\ procState = [p \in Procs |-> "unused"]
  /\ procNode = [p \in Procs |-> None]
  /\ lease = [p \in Procs |-> FALSE]
  /\ on = [p \in Procs |-> {}]
  /\ vram = [p \in Procs |-> {}]
  /\ mem = [p \in Procs |-> EmptyRuns]
  /\ running = [p \in Procs |-> EmptyRuns]
  /\ attaching = [p \in Procs |-> EmptyRuns]
  /\ flushed = [p \in Procs |-> EmptyRuns]
  /\ history = [r \in Runs |-> <<0>>]
  /\ detached = [r \in Runs |-> FALSE]

\* ***** Arbitrary / split-brain planner actions (pure hints) *****

CanSetSlot(g, target) == slot[g] # target
SetSlot(g, target) ==
  /\ CanSetSlot(g, target)
  /\ slot' = [slot EXCEPT ![g] = target]
  /\ UNCHANGED <<home, runVars, sampleVars, nodeUp, procVars>>

CanSetHome(r, target) == home[r] # target
SetHome(r, target) ==
  /\ CanSetHome(r, target)
  /\ home' = [home EXCEPT ![r] = target]
  /\ UNCHANGED <<slot, runVars, sampleVars, nodeUp, procVars>>

\* ***** Process lifecycle & L1 <-> L0 (Promote / Demote) *****

\* Spawn starts an empty process in L1 (host RAM) on node n — occupies zero GPUs.
CanSpawn(p, n) == procState[p] = "unused" /\ nodeUp[n]
Spawn(p, n) ==
  /\ CanSpawn(p, n)
  /\ procState' = [procState EXCEPT ![p] = "live"]
  /\ procNode' = [procNode EXCEPT ![p] = n]
  /\ lease' = [lease EXCEPT ![p] = TRUE]
  /\ UNCHANGED <<planVars, runVars, sampleVars, nodeUp, on, vram, mem, running, attaching, flushed>>

\* Promote (L1 -> L0): claim all WantGpus(p) atomically on p's node when free and full-width.
\* Handles fresh entry, time-slice resume, intra-node GPU hop, and wide gang admission.
CanPromote(p) ==
  /\ procState[p] = "live" /\ on[p] = {} /\ vram[p] = {}
  /\ LET G == WantGpus(p) IN
       /\ Cardinality(G) = ProcWidth[p]
       /\ \A q \in Procs : on[q] \cap G = {}
Promote(p) ==
  /\ CanPromote(p)
  /\ LET G == WantGpus(p) IN
       /\ on' = [on EXCEPT ![p] = G]
       /\ vram' = [vram EXCEPT ![p] = G]
  /\ UNCHANGED <<planVars, runVars, sampleVars, nodeUp, procState, procNode, lease, mem, running, attaching, flushed>>

\* Demote step 1 (L0 -> L1): after any in-flight command finishes, copy tensors to pinned host RAM.
\* Dirty mem[p][r] and owner[r] stay intact; no disk or network I/O needed.
CanDemoteLeave(p) ==
  /\ procState[p] = "live" /\ vram[p] # {} /\ on[p] # WantGpus(p)
  /\ \A r \in Runs : running[p][r] = None
DemoteLeave(p) ==
  /\ CanDemoteLeave(p)
  /\ vram' = [vram EXCEPT ![p] = {}]
  /\ UNCHANGED <<planVars, runVars, sampleVars, nodeUp, procState, procNode, lease, on, mem, running, attaching, flushed>>

\* Demote step 2: node clears on[p] so the next process can promote (zero scheduler round-trips).
\* With ClearAfterExit=FALSE, node clears on[p] before p finished leaving VRAM (or on lease expiry).
CanDemoteClear(p) ==
  /\ on[p] # {} /\ nodeUp[procNode[p]]
  /\ IF ClearAfterExit
       THEN procState[p] = "live" /\ vram[p] = {}
       ELSE (procState[p] = "live" /\ on[p] # WantGpus(p)) \/ ~lease[p]
DemoteClear(p) ==
  /\ CanDemoteClear(p)
  /\ on' = [on EXCEPT ![p] = {}]
  /\ UNCHANGED <<planVars, runVars, sampleVars, nodeUp, procState, procNode, lease, vram, mem, running, attaching, flushed>>

\* ***** Run residency: store (L2/L3) <-> process (Acquire / Attach / Flush / Detach) *****

\* CAS owner[r]: None -> p when plan places r on p.
CanAcquire(p, r) ==
  /\ procState[p] = "live" /\ home[r] = p /\ owner[r] = None
  /\ mem[p][r] = None /\ attaching[p][r] = None /\ running[p][r] = None /\ flushed[p][r] = None
Acquire(p, r) ==
  /\ CanAcquire(p, r)
  /\ owner' = [owner EXCEPT ![r] = p]
  /\ detached' = [detached EXCEPT ![r] = FALSE]
  /\ UNCHANGED <<planVars, pos, issued, snap, history, sampleVars, nodeUp, procVars>>

\* Start loading r into p (fresh if pos=0, or from snap[r] in L1 or L0 — off-GPU staging allowed!).
CanStartAttach(p, r) ==
  /\ procState[p] = "live" /\ owner[r] = p /\ mem[p][r] = None /\ RunIdle(p, r)
  /\ \/ pos[r] = 0
     \/ (snap[r].loc # None /\ SnapReachable(snap[r]) /\ (~ExactSnapOnly \/ snap[r].pos = pos[r]))
StartAttach(p, r) ==
  /\ CanStartAttach(p, r)
  /\ LET loadedPos == IF pos[r] = 0 THEN 0 ELSE snap[r].pos IN
       attaching' = [attaching EXCEPT ![p][r] = loadedPos]
  /\ UNCHANGED <<planVars, runVars, sampleVars, nodeUp, procState, procNode, lease, on, vram, mem, running, flushed>>

CanFinishAttach(p, r) == procState[p] = "live" /\ attaching[p][r] # None
FinishAttach(p, r) ==
  /\ CanFinishAttach(p, r)
  /\ mem' = [mem EXCEPT ![p][r] = attaching[p][r]]
  /\ attaching' = [attaching EXCEPT ![p][r] = None]
  /\ UNCHANGED <<planVars, runVars, sampleVars, nodeUp, procState, procNode, lease, on, vram, running, flushed>>

\* Flush writes r's exact in-memory state to node-local store (L2).
\* Works in L0 (split one LoRA while siblings keep running) AND in L1 (after Demote freed the GPU).
CanFlush(p, r) ==
  /\ procState[p] = "live" /\ mem[p][r] # None /\ RunIdle(p, r)
Flush(p, r) ==
  /\ CanFlush(p, r)
  /\ flushed' = [flushed EXCEPT ![p][r] = mem[p][r]]
  /\ UNCHANGED <<planVars, runVars, sampleVars, nodeUp, procState, procNode, lease, on, vram, mem, running, attaching>>

\* Client save_state: records exact snap[r] while keeping r owned & live in p.
CanCommitSave(p, r) ==
  /\ procState[p] = "live" /\ flushed[p][r] # None /\ CanCommit(p, r)
  /\ (~ExactSnapOnly \/ flushed[p][r] = pos[r])
CommitSave(p, r) ==
  /\ CanCommitSave(p, r)
  /\ snap' = [snap EXCEPT ![r] = [pos |-> flushed[p][r], loc |-> procNode[p]]]
  /\ flushed' = [flushed EXCEPT ![p][r] = None]
  /\ UNCHANGED <<planVars, owner, pos, issued, history, detached, sampleVars, nodeUp,
                 procState, procNode, lease, on, vram, mem, running, attaching>>

\* Planned Detach (rebalance / split / merge / drain / L1 evict): commits snap and clears owner[r].
CanDetach(p, r) ==
  /\ procState[p] = "live" /\ home[r] # p /\ flushed[p][r] # None /\ CanCommit(p, r)
  /\ (~ExactSnapOnly \/ flushed[p][r] = pos[r])
Detach(p, r) ==
  /\ CanDetach(p, r)
  /\ snap' = [snap EXCEPT ![r] = [pos |-> flushed[p][r], loc |-> procNode[p]]]
  /\ owner' = [owner EXCEPT ![r] = None]
  /\ detached' = [detached EXCEPT ![r] = TRUE]
  /\ mem' = [mem EXCEPT ![p][r] = None]
  /\ flushed' = [flushed EXCEPT ![p][r] = None]
  /\ UNCHANGED <<planVars, pos, issued, history, sampleVars, nodeUp,
                 procState, procNode, lease, on, vram, running, attaching>>

\* Zero-copy DetachClean: when snap[r] already matches pos[r] (e.g. right after save_state or fresh pos=0),
\* detaching writes zero bytes and just releases owner[r].
CanDetachClean(p, r) ==
  /\ procState[p] = "live" /\ home[r] # p /\ RunIdle(p, r) /\ CanCommit(p, r)
  /\ (owner[r] = p \/ mem[p][r] # None)
  /\ (pos[r] = 0 \/ (SnapExact(r) /\ SnapReachable(snap[r])))
DetachClean(p, r) ==
  /\ CanDetachClean(p, r)
  /\ owner' = [owner EXCEPT ![r] = None]
  /\ detached' = [detached EXCEPT ![r] = (pos[r] > 0)]
  /\ mem' = [mem EXCEPT ![p][r] = None]
  /\ UNCHANGED <<planVars, pos, issued, snap, history, sampleVars, nodeUp,
                 procState, procNode, lease, on, vram, running, attaching, flushed>>

\* ***** Executing commands in L0 *****

\* Mutating command (forward_backward / optim_step): requires p in L0 (vram[p] # {}) and r attached.
CanRunCmd(p, r) ==
  /\ procState[p] = "live" /\ vram[p] # {} /\ mem[p][r] # None
  /\ RunIdle(p, r) /\ issued[r] < MaxPos
RunCmd(p, r) ==
  /\ CanRunCmd(p, r)
  /\ LET next == issued[r] + 1 IN
       /\ issued' = [issued EXCEPT ![r] = next]
       /\ mem' = [mem EXCEPT ![p][r] = next]
       /\ running' = [running EXCEPT ![p][r] = next]
  /\ UNCHANGED <<planVars, owner, pos, snap, history, detached, sampleVars, nodeUp,
                 procState, procNode, lease, on, vram, attaching, flushed>>

\* Explicit client load_state: works even when r has no exact snap (dirty crash recovery).
CanRunLoad(p, r) ==
  /\ procState[p] = "live" /\ vram[p] # {} /\ owner[r] = p
  /\ RunIdle(p, r) /\ issued[r] < MaxPos
RunLoad(p, r) ==
  /\ CanRunLoad(p, r)
  /\ LET next == issued[r] + 1 IN
       /\ issued' = [issued EXCEPT ![r] = next]
       /\ mem' = [mem EXCEPT ![p][r] = next]
       /\ running' = [running EXCEPT ![p][r] = next]
  /\ UNCHANGED <<planVars, owner, pos, snap, history, detached, sampleVars, nodeUp,
                 procState, procNode, lease, on, vram, attaching, flushed>>

\* Commit mutating reply: CAS checks owner[r] = p, advances pos[r], clears stale snap[r] (now dirty).
CanCommitReply(p, r) ==
  /\ procState[p] = "live" /\ running[p][r] # None /\ CanCommit(p, r)
CommitReply(p, r) ==
  /\ CanCommitReply(p, r)
  /\ LET next == running[p][r] IN
       /\ pos' = [pos EXCEPT ![r] = next]
       /\ snap' = [snap EXCEPT ![r] = IF ExactSnapOnly THEN NoSnap ELSE snap[r]]
       /\ history' = [history EXCEPT ![r] = Append(@, next)]
       /\ running' = [running EXCEPT ![p][r] = None]
  /\ UNCHANGED <<planVars, owner, issued, detached, sampleVars, nodeUp,
                 procState, procNode, lease, on, vram, mem, attaching, flushed>>

\* Fenced process notices it lost owner[r] and discards its private copy of r.
CanDropFenced(p, r) ==
  /\ procState[p] = "live" /\ owner[r] # p
  /\ (mem[p][r] # None \/ running[p][r] # None \/ flushed[p][r] # None \/ attaching[p][r] # None)
DropFenced(p, r) ==
  /\ CanDropFenced(p, r)
  /\ mem' = [mem EXCEPT ![p][r] = None]
  /\ running' = [running EXCEPT ![p][r] = None]
  /\ flushed' = [flushed EXCEPT ![p][r] = None]
  /\ attaching' = [attaching EXCEPT ![p][r] = None]
  /\ UNCHANGED <<planVars, runVars, sampleVars, nodeUp, procState, procNode, lease, on, vram>>

\* Stateless sampler command: any live L0 process may execute & try to commit (competing consumers).
CanCommitSample(p, c) ==
  /\ procState[p] = "live" /\ vram[p] # {}
  /\ (~SampleSetNX \/ c \notin sampleReplied)
  /\ sampleCharges[c] < Cardinality(Procs)
CommitSample(p, c) ==
  /\ CanCommitSample(p, c)
  /\ sampleReplied' = sampleReplied \cup {c}
  /\ sampleCharges' = [sampleCharges EXCEPT ![c] = @ + 1]
  /\ UNCHANGED <<planVars, runVars, nodeUp, procVars>>

\* ***** Background L2 -> L3 upload *****

CanUpload(r) == snap[r].loc \in Nodes /\ nodeUp[snap[r].loc]
Upload(r) ==
  /\ CanUpload(r)
  /\ snap' = [snap EXCEPT ![r] = [@ EXCEPT !.loc = Bucket]]
  /\ UNCHANGED <<planVars, owner, pos, issued, history, detached, sampleVars, nodeUp, procVars>>

\* ***** Failures, leases, and crash recovery *****

CanLeaseExpire(p) == procState[p] \in {"live", "dead"} /\ lease[p]
LeaseExpire(p) ==
  /\ CanLeaseExpire(p)
  /\ lease' = [lease EXCEPT ![p] = FALSE]
  /\ UNCHANGED <<planVars, runVars, sampleVars, nodeUp,
                 procState, procNode, on, vram, mem, running, attaching, flushed>>

\* A paused process renews its lease on recovery; keeps any run nobody reaped, fenced on any reaped run.
CanRenewLease(p) == procState[p] = "live" /\ ~lease[p]
RenewLease(p) ==
  /\ CanRenewLease(p)
  /\ lease' = [lease EXCEPT ![p] = TRUE]
  /\ UNCHANGED <<planVars, runVars, sampleVars, nodeUp,
                 procState, procNode, on, vram, mem, running, attaching, flushed>>

\* Atomic WATCH(run[r], lease[p]): free owner[r] only while owner[r] = p and lease[p] is absent.
CanReap(r, p) == owner[r] = p /\ (~ReapChecksLease \/ ~lease[p])
Reap(r, p) ==
  /\ CanReap(r, p)
  /\ owner' = [owner EXCEPT ![r] = None]
  /\ detached' = [detached EXCEPT ![r] = FALSE]
  /\ UNCHANGED <<planVars, pos, issued, snap, history, sampleVars, nodeUp, procVars>>

KillSet(S) ==
  /\ procState' = [p \in Procs |-> IF p \in S THEN "dead" ELSE procState[p]]
  /\ vram'      = [p \in Procs |-> IF p \in S THEN {} ELSE vram[p]]
  /\ mem'       = [p \in Procs |-> IF p \in S THEN EmptyRuns ELSE mem[p]]
  /\ running'   = [p \in Procs |-> IF p \in S THEN EmptyRuns ELSE running[p]]
  /\ attaching' = [p \in Procs |-> IF p \in S THEN EmptyRuns ELSE attaching[p]]
  /\ flushed'   = [p \in Procs |-> IF p \in S THEN EmptyRuns ELSE flushed[p]]

CanCrash(p) == procState[p] = "live"
Crash(p) ==
  /\ CanCrash(p)
  /\ KillSet({p})
  /\ UNCHANGED <<planVars, runVars, sampleVars, nodeUp, procNode, lease, on>>

\* OS waitpid succeeded -> kernel reclaimed VRAM -> node safely clears on[p].
CanWaitPid(p) == procState[p] = "dead" /\ nodeUp[procNode[p]]
WaitPid(p) ==
  /\ CanWaitPid(p)
  /\ procState' = [procState EXCEPT ![p] = "reaped"]
  /\ on' = [on EXCEPT ![p] = {}]
  /\ UNCHANGED <<planVars, runVars, sampleVars, nodeUp, procNode, lease, vram, mem, running, attaching, flushed>>

CanNodeDown(n) == nodeUp[n]
NodeDown(n) ==
  /\ CanNodeDown(n)
  /\ nodeUp' = [nodeUp EXCEPT ![n] = FALSE]
  /\ on' = [p \in Procs |-> IF procNode[p] = n THEN {} ELSE on[p]]
  /\ KillSet({p \in Procs : procState[p] = "live" /\ procNode[p] = n})
  /\ UNCHANGED <<planVars, runVars, sampleVars, procNode, lease>>

Next ==
  \/ \E g \in Gpus, t \in Procs \cup {None} : SetSlot(g, t)
  \/ \E r \in Runs, t \in Procs \cup {None} : SetHome(r, t)
  \/ \E p \in Procs, n \in Nodes : Spawn(p, n)
  \/ \E p \in Procs :
       \/ Promote(p) \/ DemoteLeave(p) \/ DemoteClear(p)
       \/ LeaseExpire(p) \/ RenewLease(p) \/ Crash(p) \/ WaitPid(p)
  \/ \E p \in Procs, r \in Runs :
       \/ Acquire(p, r) \/ StartAttach(p, r) \/ FinishAttach(p, r)
       \/ Flush(p, r) \/ CommitSave(p, r) \/ Detach(p, r) \/ DetachClean(p, r)
       \/ RunCmd(p, r) \/ RunLoad(p, r) \/ CommitReply(p, r) \/ DropFenced(p, r)
       \/ Reap(r, p)
  \/ \E p \in Procs, c \in SampleCmds : CommitSample(p, c)
  \/ \E r \in Runs : Upload(r)
  \/ \E n \in Nodes : NodeDown(n)

Spec == Init /\ [][Next]_vars

\* ***** Safety invariants (must hold even under arbitrary split-brain planners) *****

\* I1: At most one process physically occupies any GPU, and every occupant is full-width & recorded in on.
GpuExclusive == \A g \in Gpus : Cardinality({p \in Procs : g \in vram[p]}) <= 1
GangAtomic   == \A p \in Procs : vram[p] = {} \/ (Cardinality(vram[p]) = ProcWidth[p] /\ vram[p] \subseteq on[p])

\* I2: A run's owner (when idle between commands) holds tensors matching exact committed pos[r].
MemoryMatchesPos ==
  \A r \in Runs, p \in Procs :
    (owner[r] = p /\ mem[p][r] # None /\ running[p][r] = None) => mem[p][r] = pos[r]

\* I3: Committed history moves strictly forward in one linear chain.
LinearHistory ==
  \A r \in Runs :
    /\ pos[r] = Last(history[r])
    /\ \A i \in 1..(Len(history[r]) - 1) : history[r][i] < history[r][i + 1]

\* I4: Any stored snapshot is a real ancestor <= pos[r], and while exact it equals pos[r].
SnapValid ==
  \A r \in Runs :
    snap[r].loc # None =>
      /\ snap[r].pos <= pos[r]
      /\ \E i \in 1..Len(history[r]) : history[r][i] = snap[r].pos

\* I5: Planned Detach (rebalance / split / merge / drain / evict) never loses a single committed step.
DetachLossless ==
  \A r \in Runs : (detached[r] /\ owner[r] = None) => (snap[r].loc # None /\ snap[r].pos = pos[r])

\* I6: Stateless sampler commands are billed at most once across K concurrent replicas.
SampleChargedOnce == \A c \in SampleCmds : sampleCharges[c] <= 1

\* I7: An owner with a live lease never loses uncheckpointed steps (only Detach or lease expiry clears owner).
LiveOwnerKept ==
  [][\A r \in Runs, p \in Procs :
       (owner[r] = p /\ lease[p] /\ owner'[r] = None /\ pos'[r] > 0) =>
         (snap'[r].loc # None /\ snap'[r].pos = pos'[r])]_vars

=============================================================================
