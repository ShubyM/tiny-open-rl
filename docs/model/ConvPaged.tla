----------------------------- MODULE ConvPaged -----------------------------
\* Bounded-termination & deadlock-freedom check for PagedFleet node reconciliation
\* (including multi-GPU gang preemption, intra-node GPU swaps, cross-process run migration,
\* and clean eviction to store).
\*
\* Arbitrary churn runs first; then Freeze locks any valid (slot, home) plan.
\* Every transition after Freeze must strictly decrease natural-number Rank, and
\* deadlock checking is ENABLED so any non-converged stuck state fails immediately.

EXTENDS MCPaged

VARIABLES frozen

convVars == <<vars, frozen>>

ValidPlan ==
  /\ \A p \in Procs :
       LET G == WantGpus(p) IN G = {} \/ Cardinality(G) = ProcWidth[p]
  /\ \A r \in Runs : home[r] \in Procs \cup {None}

Converged ==
  /\ frozen
  /\ \A p \in Procs : on[p] = WantGpus(p) /\ vram[p] = WantGpus(p)
  /\ \A r \in Runs :
       /\ (home[r] = None => owner[r] = None /\ (pos[r] = 0 \/ SnapExact(r)))
       /\ (home[r] # None => owner[r] = home[r] /\ mem[home[r]][r] = pos[r])
  /\ \A p \in Procs, r \in Runs :
       running[p][r] = None /\ attaching[p][r] = None /\ flushed[p][r] = None

ConvInit ==
  /\ slot = [g \in Gpus |-> None]
  /\ home = [r \in Runs |-> None]
  /\ owner = [r \in Runs |-> None]
  /\ pos = [r \in Runs |-> 0]
  /\ issued = [r \in Runs |-> 0]
  /\ snap = [r \in Runs |-> NoSnap]
  /\ sampleReplied = {}
  /\ sampleCharges = [c \in SampleCmds |-> 0]
  /\ nodeUp = [n \in Nodes |-> TRUE]
  /\ procState = [p \in Procs |-> "live"]
  /\ procNode = [p \in Procs |-> IF g3 \in Gpus /\ p = p2 THEN n2 ELSE n1]
  /\ lease = [p \in Procs |-> TRUE]
  /\ on = [p \in Procs |-> {}]
  /\ vram = [p \in Procs |-> {}]
  /\ mem = [p \in Procs |-> EmptyRuns]
  /\ running = [p \in Procs |-> EmptyRuns]
  /\ attaching = [p \in Procs |-> EmptyRuns]
  /\ flushed = [p \in Procs |-> EmptyRuns]
  /\ history = [r \in Runs |-> <<0>>]
  /\ detached = [r \in Runs |-> FALSE]
  /\ frozen = FALSE

Freeze == ~frozen /\ ValidPlan /\ frozen' = TRUE /\ UNCHANGED vars

ReconcileStep ==
  /\ frozen /\ frozen' = TRUE
  /\ \/ \E p \in Procs : DemoteLeave(p) \/ DemoteClear(p) \/ Promote(p)
     \/ \E p \in Procs, r \in Runs :
          \/ CommitReply(p, r)
          \/ (home[r] # p /\ owner[r] = p /\ flushed[p][r] = None
              /\ ~(pos[r] = 0 \/ SnapExact(r)) /\ Flush(p, r))
          \/ Detach(p, r) \/ DetachClean(p, r)
          \/ (home[r] = p /\ flushed[p][r] # None /\ CommitSave(p, r))
          \/ Acquire(p, r) \/ (home[r] = p /\ StartAttach(p, r)) \/ FinishAttach(p, r)

ChurnStep ==
  /\ ~frozen /\ frozen' = FALSE
  /\ \/ \E g \in Gpus, t \in Procs \cup {None} : SetSlot(g, t)
     \/ \E r \in Runs, t \in Procs \cup {None} : SetHome(r, t)
     \/ \E p \in Procs : Promote(p) \/ DemoteLeave(p) \/ DemoteClear(p)
     \/ \E p \in Procs, r \in Runs :
          \/ Acquire(p, r) \/ (home[r] = p /\ StartAttach(p, r)) \/ FinishAttach(p, r)
          \/ (home[r] # p /\ owner[r] = p /\ Flush(p, r))
          \/ Detach(p, r) \/ DetachClean(p, r)
          \/ RunCmd(p, r) \/ CommitReply(p, r)

ConvTerminated == Converged /\ UNCHANGED convVars
ConvNext == ChurnStep \/ Freeze \/ ReconcileStep \/ ConvTerminated
ConvSpec == ConvInit /\ [][ConvNext]_convVars

GpuRank(p) ==
  IF on[p] = WantGpus(p) /\ vram[p] = WantGpus(p) THEN 0
  ELSE IF on[p] = {} THEN 1
  ELSE IF vram[p] = {} THEN 2
  ELSE 3

RunRank(r) ==
  LET h == home[r]
      o == owner[r]
  IN IF o # None /\ o # h THEN
       (IF running[o][r] # None \/ attaching[o][r] # None THEN 6
        ELSE IF flushed[o][r] # None \/ pos[r] = 0 \/ SnapExact(r) THEN 4
        ELSE 5)
     ELSE IF o = None THEN
       (IF h = None THEN 0 ELSE 3)
     ELSE
       (IF running[h][r] # None \/ flushed[h][r] # None THEN 1
        ELSE IF mem[h][r] # None THEN 0
        ELSE IF attaching[h][r] # None THEN 1
        ELSE 2)

RECURSIVE SumP(_), SumR(_)
SumP(S) == IF S = {} THEN 0 ELSE LET x == CHOOSE y \in S : TRUE IN GpuRank(x) + SumP(S \ {x})
SumR(S) == IF S = {} THEN 0 ELSE LET x == CHOOSE y \in S : TRUE IN RunRank(x) + SumR(S \ {x})

Rank == SumP(Procs) + SumR(Runs)
ConvTypeOK == TypeOK /\ frozen \in BOOLEAN
RankDecreases == [][(frozen /\ frozen' /\ convVars' # convVars) => Rank' < Rank]_convVars
ZeroRankIsConverged == frozen => (Rank = 0 <=> Converged)

=============================================================================
