--------------------------------- MODULE MC ---------------------------------
\* Finite instances of FleetModel for TLC.
EXTENDS FleetModel, TLC

CONSTANTS n1, n2, g1, g2, g3

\* Two GPUs on one node, so two workers contend for a GPU and a wide worker needs both; one GPU on another node.
TwoNodes == (g1 :> n1) @@ (g2 :> n1) @@ (g3 :> n2)
\* One GPU on each of two nodes.
OneEach == (g1 :> n1) @@ (g3 :> n2)
\* Both GPUs on one node; the other node has none.
OneNode == (g1 :> n1) @@ (g2 :> n1)
\* A single GPU.
OneGpu == (g1 :> n1)

Symmetry == Permutations(Workers) \cup Permutations(Jobs)
=============================================================================
