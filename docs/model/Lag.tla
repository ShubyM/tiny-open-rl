-------------------------------- MODULE Lag --------------------------------
\* Model of EEVDF lag fairness + cost-hysteresis switching with async node charges.
\*
\* Nodes ONLY increment service[a] (monotone counter of real work served).
\* Planner ONLY owns lag[a] and seen[a] (zero write-write race with nodes):
\*   delta[a]        = snapSvc[a] - seen[a]
\*   participants    = snapRun \cup {a : delta[a] > 0}
\*   lag'[a]         = lag[a] + (tot / |participants|) - delta[a]   (in fixed-point 1/Unit)
\* On sleep: positive lag drops to 0 (no hoarding); negative lag persists (even if the account
\* sleeps before PlanRead observes its delta!) and decays only when active peers complete work.
\* Non-idle preemption requires: ran >= Cost * Scale AND lag'[waiter] - lag'[holder] >= Cost * Scale.

EXTENDS Naturals, Integers, FiniteSets

CONSTANTS
  Accounts,
  MaxWork,      \* bound on work executed per account
  Cost,         \* switch cost (>= 1)
  Scale,        \* 1/eps (>= 1): minimum work per non-idle switch is Cost * Scale
  None,
  UseDelta,     \* TRUE: planner subtracts (snapSvc - seen); FALSE: planner resets service := 0 (loses async charge)
  KeepNegLag,   \* TRUE: sleeping account keeps negative debt; FALSE: sleeping wipes negative debt (flicker exploit)
  AmortizeCost  \* TRUE: preempt only after ran >= Cost * Scale; FALSE: preempt anytime (breaks goodput bound)

Unit   == 12    \* fixed common denominator (divisible by 1, 2, 3, 4) so lag[a] has exact uniform units
MinRun == Cost * Scale
Thresh == Unit * MinRun
Bound  == Unit * Cardinality(Accounts) * MaxWork

ASSUME Cost \in Nat \ {0} /\ Scale \in Nat \ {0} /\ MaxWork \in Nat \ {0}
ASSUME {UseDelta, KeepNegLag, AmortizeCost} \subseteq BOOLEAN
ASSUME \A k \in 1..(Cardinality(Accounts) + 1) : Unit % k = 0

VARIABLES
  service,       \* [Accounts -> 0..MaxWork]       node-incremented service counter in Redis
  execTotal,     \* [Accounts -> 0..MaxWork]       ghost: true total work executed by nodes
  accounted,     \* [Accounts -> 0..MaxWork]       ghost: total work absorbed by planner lag updates
  seen,          \* [Accounts -> 0..MaxWork]       planner's last observed service[a]
  lag,           \* [Accounts -> -Bound..Bound]    planner's debt to account a (in exact 1/Unit work units)
  runnable,      \* SUBSET Accounts                accounts with ready queued/in-flight work
  holder,        \* Accounts \cup {None}           account currently holding the GPU
  ran,           \* 0..MaxWork                     work executed by holder since entering GPU
  switchCost,    \* Nat                            cumulative non-idle preemption cost paid
  amortizedWork, \* Nat                            cumulative work completed prior to non-idle preemptions
  snapSvc,       \* [Accounts -> 0..MaxWork] \cup {None}  async planner read snapshot of service
  snapRun        \* SUBSET Accounts                       async planner read snapshot of runnable

vars == <<service, execTotal, accounted, seen, lag, runnable, holder, ran,
          switchCost, amortizedWork, snapSvc, snapRun>>

Min(a, b) == IF a <= b THEN a ELSE b

TypeOK ==
  /\ service   \in [Accounts -> 0..MaxWork]
  /\ execTotal \in [Accounts -> 0..MaxWork]
  /\ accounted \in [Accounts -> 0..MaxWork]
  /\ seen      \in [Accounts -> 0..MaxWork]
  /\ lag       \in [Accounts -> -Bound..Bound]
  /\ runnable  \subseteq Accounts
  /\ holder    \in Accounts \cup {None}
  /\ ran       \in 0..MaxWork
  /\ snapSvc   \in [Accounts -> 0..MaxWork] \cup {None}
  /\ snapRun   \subseteq Accounts

Init ==
  /\ service       = [a \in Accounts |-> 0]
  /\ execTotal     = [a \in Accounts |-> 0]
  /\ accounted     = [a \in Accounts |-> 0]
  /\ seen          = [a \in Accounts |-> 0]
  /\ lag           = [a \in Accounts |-> 0]
  /\ runnable      = {}
  /\ holder        = None
  /\ ran           = 0
  /\ switchCost    = 0
  /\ amortizedWork = 0
  /\ snapSvc       = None
  /\ snapRun       = {}

CanWake(a) == a \notin runnable
Wake(a) ==
  /\ CanWake(a)
  /\ runnable' = runnable \cup {a}
  /\ UNCHANGED <<service, execTotal, accounted, seen, lag, holder, ran,
                 switchCost, amortizedWork, snapSvc, snapRun>>

\* Going idle forfeits positive credit (<= 0), and keeps negative debt when KeepNegLag is TRUE.
CanSleep(a) == a \in runnable
Sleep(a) ==
  /\ CanSleep(a)
  /\ runnable' = runnable \ {a}
  /\ holder'   = IF holder = a THEN None ELSE holder
  /\ ran'      = IF holder = a THEN 0 ELSE ran
  /\ lag'      = [lag EXCEPT ![a] = IF KeepNegLag THEN Min(0, @) ELSE 0]
  /\ UNCHANGED <<service, execTotal, accounted, seen, switchCost, amortizedWork, snapSvc, snapRun>>

\* Node executes 1 unit of work for holder and increments service[holder] (interleaves with PlanRead/PlanWrite).
CanExecute == holder # None /\ execTotal[holder] < MaxWork
Execute ==
  /\ CanExecute
  /\ service'   = [service   EXCEPT ![holder] = @ + 1]
  /\ execTotal' = [execTotal EXCEPT ![holder] = @ + 1]
  /\ ran'       = ran + 1
  /\ UNCHANGED <<accounted, seen, lag, runnable, holder, switchCost, amortizedWork, snapSvc, snapRun>>

CanPlanRead ==
  /\ snapSvc = None
  /\ (runnable # {} \/ \E a \in Accounts : service[a] > seen[a])
PlanRead ==
  /\ CanPlanRead
  /\ snapSvc' = service
  /\ snapRun' = runnable
  /\ UNCHANGED <<service, execTotal, accounted, seen, lag, runnable, holder, ran, switchCost, amortizedWork>>

\* Planner updates lag in fixed-point 1/Unit units over participants = snapRun \cup {a : delta[a] > 0}
\* (so sleeping before PlanRead NEVER hides executed work), keeps holder if it still wins,
\* fills an idle GPU immediately, and preempts a busy holder iff ran >= MinRun and lag diff >= Thresh.
CanPlanWrite == snapSvc # None
PlanWrite ==
  /\ CanPlanWrite
  /\ LET delta == [a \in Accounts |-> snapSvc[a] - seen[a]]
         part  == snapRun \cup {a \in Accounts : delta[a] > 0}
         m     == Cardinality(part)
         RECURSIVE SumD(_)
         SumD(S) == IF S = {} THEN 0 ELSE LET x == CHOOSE y \in S : TRUE IN delta[x] + SumD(S \ {x})
         tot   == SumD(part)
         share == IF m > 0 THEN (Unit \div m) * tot ELSE 0
         decay == IF m > 0 THEN (Unit \div (m + 1)) * tot ELSE 0
         rawLag == [a \in Accounts |->
                      IF a \in part THEN lag[a] + share - Unit * delta[a]
                      ELSE IF KeepNegLag THEN Min(0, lag[a] + decay)
                      ELSE 0]
         nextLag == [a \in Accounts |->
                      IF a \in runnable THEN rawLag[a]
                      ELSE IF KeepNegLag THEN Min(0, rawLag[a]) ELSE 0]
         live == runnable
     IN /\ lag'       = nextLag
        /\ accounted' = [a \in Accounts |-> accounted[a] + delta[a]]
        /\ seen'      = [a \in Accounts |-> IF UseDelta THEN snapSvc[a] ELSE 0]
        /\ service'   = [a \in Accounts |-> IF UseDelta THEN service[a] ELSE 0]
        /\ IF live = {} THEN
             /\ holder' = holder /\ ran' = ran
             /\ UNCHANGED <<switchCost, amortizedWork>>
           ELSE IF holder = None THEN
             /\ \E w \in live : (\A b \in live : nextLag[w] >= nextLag[b]) /\ holder' = w
             /\ ran' = 0
             /\ UNCHANGED <<switchCost, amortizedWork>>
           ELSE IF (\E w \in live : w # holder /\ nextLag[w] - nextLag[holder] >= Thresh)
                   /\ (~AmortizeCost \/ ran >= MinRun) THEN
             /\ \E w \in live : w # holder /\ (\A b \in live : nextLag[w] >= nextLag[b])
                                /\ nextLag[w] - nextLag[holder] >= Thresh /\ holder' = w
             /\ switchCost'    = switchCost + Cost
             /\ amortizedWork' = amortizedWork + ran
             /\ ran'           = 0
           ELSE
             /\ holder' = holder /\ ran' = ran
             /\ UNCHANGED <<switchCost, amortizedWork>>
  /\ snapSvc' = None
  /\ snapRun' = {}
  /\ UNCHANGED <<execTotal, runnable>>

Next == Execute \/ PlanRead \/ PlanWrite \/ \E a \in Accounts : Wake(a) \/ Sleep(a)
Spec == Init /\ [][Next]_vars

\* I1: Zero lost service charges under concurrent Execute between PlanRead and PlanWrite.
ExactAccounting ==
  \A a \in Accounts :
    /\ seen[a] <= service[a]
    /\ accounted[a] + (service[a] - seen[a]) = execTotal[a]

\* I2: Non-idle switching overhead never exceeds eps = 1/Scale of completed GPU work (goodput >= Scale/(Scale+1)).
GoodputBounded == switchCost * Scale <= amortizedWork

\* I3: No idle credit hoarding, no free lag gain on sleep/wake, and sleeping before PlanRead never dodges debt.
NoIdleHoard == \A a \in Accounts : a \notin runnable => lag[a] <= 0

NoFlickerGain ==
  [][\A a \in Accounts :
       \* Lag can increase ONLY when some OTHER account completed newly observed work.
       /\ (lag'[a] > lag[a] => (snapSvc # None /\ \E b \in Accounts \ {a} : snapSvc[b] > seen[b]))
       \* Executing sole work while a peer is active ALWAYS reduces lag[a], even if a slept before PlanRead.
       /\ ((snapSvc # None /\ snapSvc' = None /\ snapSvc[a] > seen[a]
            /\ (\E b \in Accounts \ {a} : b \in snapRun)
            /\ ~(\E b \in Accounts \ {a} : snapSvc[b] > seen[b]))
             => lag'[a] < lag[a])]_vars

\* I4: Zero wasted handoffs and exact cost hysteresis on every non-idle preemption.
NoWastedSwap ==
  [][(holder # None /\ holder' # None /\ holder' # holder) =>
       lag'[holder'] - lag'[holder] >= Thresh]_vars

=============================================================================
