------------------------------ MODULE MCPaged ------------------------------
\* Finite model instantiations of PagedFleet: multi-node, wide gangs, split-brain planner.

EXTENDS PagedFleet, TLC

CONSTANTS n1, n2, g1, g2, g3, p1, p2, p3

\* Topology A: 2 nodes, 1 GPU each (cross-node rebalance, split-brain, crashes).
TwoNodesOneGpu == g1 :> n1 @@ g3 :> n2
AllUnitWidth   == [p \in Procs |-> 1]

\* Topology B: node n1 has 2 GPUs {g1, g2}, node n2 has 1 GPU {g3};
\* p3 is a 2-GPU gang sharing n1 with 1-GPU processes p1, p2 (gang admission + intra-node hop + rebalance).
GangNode == g1 :> n1 @@ g2 :> n1 @@ g3 :> n2
OneGang  == [p \in Procs |-> IF p = p3 THEN 2 ELSE 1]

\* Topology C: single 2-GPU node {g1, g2} with 2-GPU gang p3 + two 1-GPU processes p1, p2.
SingleGangNode == g1 :> n1 @@ g2 :> n1

\* Topology D: single 1-GPU node {g1} for multi-run sharing, split, merge, and L0<->L1 turns.
SingleGpu == g1 :> n1

UnitSymmetry    == Permutations(Procs)
GangSymmetry    == Permutations({p1, p2}) \cup Permutations({g1, g2})
TwoGpuSymmetry  == Permutations({g1, g2})
RunProcSymmetry == Permutations(Procs) \cup Permutations(Runs)

=============================================================================
