#!/bin/bash
# Verify the paged GPU buffer-pool (`PagedFleet.tla`, `ConvPaged.tla`) and EEVDF lag (`Lag.tla`) models with TLC.
# Usage:
#   TLA2TOOLS_JAR=/path/to/tla2tools.jar ./docs/model/check.sh          # fast smoke suite (~20s)
#   TLA2TOOLS_JAR=/path/to/tla2tools.jar ./docs/model/check.sh --full   # exhaustive 7 positive + 9 negative (~30m)
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
JAR="${TLA2TOOLS_JAR:-$DIR/tla2tools.jar}"
[[ -f "$JAR" ]] || { echo "error: set TLA2TOOLS_JAR to tla2tools.jar" >&2; exit 1; }
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp "$DIR"/*.tla "$WORK"/
cd "$WORK"

expect_rc() {
  local name=$1 want=$2 got=$3
  local summary
  summary=$(grep -E "Invariant .* is violated|Property .* is violated|Deadlock reached|distinct states found|Finished in|Error:" "$name.log" | tr '\n' ' ')
  echo "[$name rc=$got] $summary"
  if [[ "$want" == "0" && "$got" -ne 0 ]] || [[ "$want" == "fail" && "$got" -eq 0 ]]; then
    echo "FAILED: $name (expected $want, got rc=$got)" >&2
    tail -n 40 "$name.log" >&2
    exit 1
  fi
}

check_paged() {
  local name=$1 want=$2 nodes=$3 runs=$4 scmds=$5 procs=$6 gpus=$7 topo=$8 widths=$9 maxpos=${10} sym=${11}
  shift 11
  declare -A rule=(
    [ClearAfterExit]=TRUE [OwnerFencesWrite]=TRUE [ExactSnapOnly]=TRUE
    [ReapChecksLease]=TRUE [SerializeRunOps]=TRUE [SampleSetNX]=TRUE
  )
  for kv in "$@"; do rule[${kv%%=*}]=${kv#*=}; done
  {
    echo "SPECIFICATION Spec"
    echo "CONSTANTS"
    echo "  Nodes = {$nodes}  Runs = {$runs}  SampleCmds = {$scmds}  Procs = {$procs}  Gpus = {$gpus}"
    echo "  n1 = n1  n2 = n2  g1 = g1  g2 = g2  g3 = g3  p1 = p1  p2 = p2  p3 = p3"
    echo "  GpuNode <- $topo  ProcWidth <- $widths  MaxPos = $maxpos  None = None  Bucket = Bucket"
    for k in "${!rule[@]}"; do echo "  $k = ${rule[$k]}"; done
    [[ "$sym" != "NONE" ]] && echo "SYMMETRY $sym"
    echo "INVARIANTS TypeOK GpuExclusive GangAtomic MemoryMatchesPos LinearHistory SnapValid DetachLossless SampleChargedOnce"
    echo "PROPERTIES LiveOwnerKept"
  } > "$name.cfg"
  set +e
  java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -deadlock -workers auto -metadir "$WORK/m_$name" -config "$name.cfg" MCPaged.tla > "$name.log" 2>&1
  local rc=$?
  set -e
  expect_rc "$name" "$want" "$rc"
}

check_conv() {
  local name=$1 nodes=$2 runs=$3 procs=$4 gpus=$5 topo=$6 widths=$7 maxpos=$8
  {
    echo "SPECIFICATION ConvSpec"
    echo "CONSTANTS"
    echo "  Nodes = {$nodes}  Runs = {$runs}  SampleCmds = {}  Procs = {$procs}  Gpus = {$gpus}"
    echo "  n1 = n1  n2 = n2  g1 = g1  g2 = g2  g3 = g3  p1 = p1  p2 = p2  p3 = p3"
    echo "  GpuNode <- $topo  ProcWidth <- $widths  MaxPos = $maxpos  None = None  Bucket = Bucket"
    echo "  ClearAfterExit = TRUE  OwnerFencesWrite = TRUE  ExactSnapOnly = TRUE"
    echo "  ReapChecksLease = TRUE  SerializeRunOps = TRUE  SampleSetNX = TRUE"
    echo "INVARIANTS ConvTypeOK GpuExclusive GangAtomic MemoryMatchesPos DetachLossless ZeroRankIsConverged"
    echo "PROPERTIES RankDecreases"
  } > "$name.cfg"
  set +e
  java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -workers auto -metadir "$WORK/m_$name" -config "$name.cfg" ConvPaged.tla > "$name.log" 2>&1
  local rc=$?
  set -e
  expect_rc "$name" "0" "$rc"
}

check_lag() {
  local name=$1 want=$2 accts=$3 maxwork=$4; shift 4
  declare -A rule=([UseDelta]=TRUE [KeepNegLag]=TRUE [AmortizeCost]=TRUE)
  for kv in "$@"; do rule[${kv%%=*}]=${kv#*=}; done
  {
    echo "SPECIFICATION Spec"
    echo "CONSTANTS"
    echo "  Accounts = {$accts}  MaxWork = $maxwork  Cost = 1  Scale = 2  None = None"
    for k in "${!rule[@]}"; do echo "  $k = ${rule[$k]}"; done
    echo "SYMMETRY AccountSymmetry"
    echo "INVARIANTS TypeOK ExactAccounting GoodputBounded NoIdleHoard"
    echo "PROPERTIES NoFlickerGain NoWastedSwap"
  } > "$name.cfg"
  set +e
  java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -deadlock -workers auto -metadir "$WORK/m_$name" -config "$name.cfg" MCLag.tla > "$name.log" 2>&1
  local rc=$?
  set -e
  expect_rc "$name" "$want" "$rc"
}

if [[ "${1:-}" == "--full" ]]; then
  # 7 exhaustive positive configurations (122,744,619 distinct states total)
  check_paged ok_2node  0 "n1, n2" "r1"     ""       "p1, p2"     "g1, g3"     TwoNodesOneGpu AllUnitWidth 2 UnitSymmetry
  check_paged ok_sample 0 "n1, n2" "r1"     "s1, s2" "p1, p2"     "g1, g3"     TwoNodesOneGpu AllUnitWidth 1 UnitSymmetry
  check_paged ok_gang   0 "n1"     "r1"     ""       "p1, p2, p3" "g1, g2"     SingleGangNode OneGang      1 GangSymmetry
  check_paged ok_2runs  0 "n1"     "r1, r2" ""       "p1, p2"     "g1"         SingleGpu      AllUnitWidth 1 RunProcSymmetry
  check_conv  conv_gang   "n1"     "r1"              "p1, p2, p3" "g1, g2"     SingleGangNode OneGang      1
  check_conv  conv_2node  "n1, n2" "r1"              "p1, p2"     "g1, g2, g3" GangNode       AllUnitWidth 1
  check_lag   ok_lag    0 "a1, a2, a3" 3
else
  # Fast smoke configurations (~20s)
  check_paged smoke_paged 0 "n1" "r1" "s1" "p1, p2"     "g1"     SingleGpu      AllUnitWidth 1 UnitSymmetry
  check_conv  conv_gang   "n1"   "r1"      "p1, p2, p3" "g1, g2" SingleGangNode OneGang      1
  check_lag   smoke_lag   0 "a1, a2" 2
fi

# All 9 negative rule mutations: dropping any safety/accounting rule breaks its exact property.
check_paged neg_r1    fail "n1, n2" "r1" ""       "p1, p2" "g1, g3" TwoNodesOneGpu AllUnitWidth 1 UnitSymmetry ClearAfterExit=FALSE
check_paged neg_r2    fail "n1, n2" "r1" ""       "p1, p2" "g1, g3" TwoNodesOneGpu AllUnitWidth 2 UnitSymmetry OwnerFencesWrite=FALSE
check_paged neg_r3    fail "n1, n2" "r1" ""       "p1, p2" "g1, g3" TwoNodesOneGpu AllUnitWidth 2 UnitSymmetry ExactSnapOnly=FALSE
check_paged neg_r4    fail "n1, n2" "r1" ""       "p1, p2" "g1, g3" TwoNodesOneGpu AllUnitWidth 1 UnitSymmetry ReapChecksLease=FALSE
check_paged neg_r5    fail "n1, n2" "r1" ""       "p1, p2" "g1, g3" TwoNodesOneGpu AllUnitWidth 2 UnitSymmetry SerializeRunOps=FALSE
check_paged neg_r6    fail "n1, n2" "r1" "s1, s2" "p1, p2" "g1, g3" TwoNodesOneGpu AllUnitWidth 1 UnitSymmetry SampleSetNX=FALSE
check_lag   neg_delta fail "a1, a2" 2 UseDelta=FALSE
check_lag   neg_sleep fail "a1, a2" 2 KeepNegLag=FALSE
check_lag   neg_amort fail "a1, a2" 2 AmortizeCost=FALSE
echo "ALL CHECKS PASSED"
