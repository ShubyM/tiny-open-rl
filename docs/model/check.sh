#!/bin/bash
# Check one instance of FleetModel with TLC. Needs java and tla2tools.jar (github.com/tlaplus/tlaplus/releases) beside this file.
# Usage: check.sh <name> <jobs> <workers> <gpus> <topology> <max commands> [RULE=FALSE ...]
#   check.sh ok "j1" "w1, w2" "g1, g3" OneEach 2                          every rule on: no invariant is violated
#   check.sh no_r3 "j1" "w1, w2" "g1, g3" OneEach 2 WritesCheckOwner=FALSE  one rule off: TLC prints the trace that breaks
# Fair.tla has its own config: java -cp tla2tools.jar tlc2.TLC -deadlock -config Fair.cfg Fair.tla
set -u
cd "$(dirname "$0")"
name=$1 jobs=$2 workers=$3 gpus=$4 topology=$5 commands=$6
shift 6
declare -A rule=([OnlyNodeFreesGpu]=TRUE [ParkBeforeRelease]=TRUE [WritesCheckOwner]=TRUE [RestoreChecksAcknowledged]=TRUE [OneAtATime]=TRUE)
for kv in "$@"; do rule[${kv%%=*}]=${kv#*=}; done
{
  echo "SPECIFICATION Spec"
  echo "CONSTANTS"
  echo "  Jobs = {$jobs}"
  echo "  Workers = {$workers}"
  echo "  Nodes = {n1, n2}"
  echo "  Gpus = {$gpus}"
  echo "  n1 = n1  n2 = n2  g1 = g1  g2 = g2  g3 = g3"
  echo "  GpuNode <- $topology"
  echo "  MaxCommands = $commands"
  echo "  None = None  Bucket = Bucket"
  for k in "${!rule[@]}"; do echo "  $k = ${rule[$k]}"; done
  echo "SYMMETRY Symmetry"
  echo "INVARIANTS ${INVS:-TypeOK GpuExclusive GpuUserHoldsLock MemoryIsAcknowledged HistoryMovesForward CheckpointInHistory CheckpointOrder HandoffLossless}"
} > "$name.cfg"
# -deadlock turns deadlock checking off: a behaviour legitimately ends when every worker id is used up.
java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC -deadlock -workers auto -metadir "/tmp/tlc_$name" -config "$name.cfg" MC.tla | tee "$name.log" |
  grep -E "Invariant .* is violated|distinct states found|Finished in|Error:"
