-------------------------------- MODULE Fair --------------------------------
\* Fair service with one stored number. Each account has a service total that only grows.
\* The scheduler keeps a floor, and on every pass lifts the runnable accounts up to it.
\* This is the rule a CPU scheduler uses when a sleeping task wakes.
\*
\* A pass is two steps, a read and a write, and nodes charge accounts in between.
\* AddGap is how the write lifts an account: TRUE adds the gap it computed from the read,
\* FALSE overwrites the total with the lifted value.

EXTENDS Naturals, FiniteSets

CONSTANTS Accounts, MaxService, None, AddGap

ASSUME AddGap \in BOOLEAN

VARIABLES
  service,   \* account -> service received, charged by the node as commands commit
  runnable,  \* accounts with work ready
  floor,     \* the scheduler's one stored number
  holder,    \* the account holding the GPU, or None
  lastRead,      \* what the scheduler read at the start of its pass, or None between passes
  lifted,      \* ghost: runnable accounts the scheduler has already lifted to the floor
  charged,      \* ghost: account -> everything nodes charged it
  added      \* ghost: account -> everything lifts added to it

vars == <<service, runnable, floor, holder, lastRead, lifted, charged, added>>

Min(S) == CHOOSE x \in S : \A y \in S : x <= y
Max(a, b) == IF a >= b THEN a ELSE b
\* Lifts never raise the highest total, so charges bound it.
Top == Cardinality(Accounts) * MaxService

TypeOK ==
  /\ service \in [Accounts -> 0..Top]
  /\ runnable \subseteq Accounts
  /\ floor \in 0..Top
  /\ holder \in Accounts \cup {None}
  /\ lastRead \in [svc : [Accounts -> 0..Top], run : SUBSET Accounts] \cup {None}
  /\ lifted \subseteq runnable
  /\ charged \in [Accounts -> 0..MaxService]
  /\ added \in [Accounts -> 0..Top]

Init ==
  /\ service = [a \in Accounts |-> 0]
  /\ runnable = {} /\ lifted = {}
  /\ floor = 0
  /\ holder = None /\ lastRead = None
  /\ charged = [a \in Accounts |-> 0]
  /\ added = [a \in Accounts |-> 0]

Wake(a) ==
  /\ a \notin runnable
  /\ runnable' = runnable \cup {a}
  /\ UNCHANGED <<service, floor, holder, lastRead, lifted, charged, added>>

Sleep(a) ==
  /\ a \in runnable /\ a # holder
  /\ runnable' = runnable \ {a}
  /\ lifted' = lifted \ {a}
  /\ UNCHANGED <<service, floor, holder, lastRead, charged, added>>

\* A node charges the account it is serving, at any time.
Charge ==
  /\ holder # None /\ charged[holder] < MaxService
  /\ service' = [service EXCEPT ![holder] = @ + 1]
  /\ charged' = [charged EXCEPT ![holder] = @ + 1]
  /\ UNCHANGED <<runnable, floor, holder, lastRead, lifted, added>>

Yield ==
  /\ holder # None
  /\ holder' = None
  /\ UNCHANGED <<service, runnable, floor, lastRead, lifted, charged, added>>

Read ==
  /\ lastRead = None /\ runnable # {}
  /\ lastRead' = [svc |-> service, run |-> runnable]
  /\ UNCHANGED <<service, runnable, floor, holder, lifted, charged, added>>

\* Raise the floor, lift what was read up to it, and give a free GPU to a least-served account.
Write ==
  /\ lastRead # None
  /\ LET f == Max(floor, Min({lastRead.svc[a] : a \in lastRead.run}))
         target == [a \in Accounts |-> IF a \in lastRead.run THEN Max(lastRead.svc[a], f) ELSE lastRead.svc[a]]
         gap == [a \in Accounts |-> target[a] - lastRead.svc[a]]
         live == lastRead.run \cap runnable
     IN /\ floor' = f
        /\ service' = [a \in Accounts |-> IF a \notin lastRead.run THEN service[a]
                                           ELSE IF AddGap THEN service[a] + gap[a] ELSE target[a]]
        /\ added' = [a \in Accounts |-> added[a] + gap[a]]
        /\ lifted' = live
        /\ IF holder = None /\ live # {}
             THEN \E a \in live : (\A b \in live : target[a] <= target[b]) /\ holder' = a
             ELSE holder' = holder
  /\ lastRead' = None
  /\ UNCHANGED <<runnable, charged>>

Next == Read \/ Write \/ Charge \/ Yield \/ \E a \in Accounts : Wake(a) \/ Sleep(a)

Spec == Init /\ [][Next]_vars

\* No charge is ever lost: a total is exactly what nodes charged plus what lifts added.
NoLostCharge == \A a \in Accounts : service[a] = charged[a] + added[a]

\* An account back from idle holds no credit: once lifted, nobody runnable is below the floor.
NoIdleCredit == \A a \in lifted : service[a] >= floor

FloorOnlyRises == [][floor' >= floor]_vars

\* An account that was already waiting keeps its place and its debt: a pass adds nothing to it.
WaitersKeepPlace == [][\A a \in lifted : a \in runnable' => added'[a] = added[a]]_vars

\* A free GPU goes to an account that was least-served among those the pass read.
GrantIsLeastServed ==
  [][(holder = None /\ holder' # None) =>
       \A b \in lastRead.run \cap runnable : Max(lastRead.svc[holder'], floor') <= Max(lastRead.svc[b], floor')]_vars

=============================================================================
