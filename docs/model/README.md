# Paged Control-Plane Model (TLA+)

These specifications model and exhaustively verify the `tinyrl` control plane across arbitrary split-brain plans, process crashes, stale-fencing races, multi-GPU gangs, stateless sampler replicas, and concurrent worker service updates.

## Separation of Mechanism and Policy

| Module | Truth / Hint | Code mapping |
| --- | --- | --- |
| `PagedFleet.tla` | Ground truth: `gpu_allocations` (node GPU table), `run` (`owner`, `epoch`, `last_cmd_id`, `snapshot`, `claimed`), `lease`. Hints: `procs`, `assignments`. | `tinyrl/bus.py`, `tinyrl/control/node_daemon.py` |
| `ConvPaged.tla` | Finite termination rank proving deadlock-free convergence to any well-formed `(procs, assignments)` once crashes/plan edits stop. | `NodeDaemon.tick`, `Worker.run` |
| `Lag.tla` | EEVDF `lag` updated from monotone `service - billed_service` deltas (`gpu_ms`) with minimum-quantum switch amortization. | `fairness.py` (`waterfill`, `update_lag`, `plan_gpus`), `scheduler.py` (`rebalance_target`), `node_daemon.py` (`GpuScheduler.turn`) |

## Verified Properties and Rule Necessity

Every safety and accounting rule has a boolean flag in the model; flipping any single flag to `FALSE` fails TLC on its target property:

| Flag | Rule | Property violated when `FALSE` |
| --- | --- | --- |
| `ClearAfterExit` | Release `node.on[life]` only after `park()` or child exit | `GpuExclusive` |
| `OwnerFencesWrite` | `commit_reply` and `detach_run` require exact `owner == token` | `MemoryMatchesPos` |
| `ExactSnapOnly` | `detach_run` records `snap` iff `run.pos == pos`; mutation clears superseded `snap` | `MemoryMatchesPos` |
| `ReapChecksLease` | `acquire_run` replaces `owner` only inside `WATCH run, lease` when the prior owner's lease key is gone | `LiveOwnerKept` |
| `SerializeRunOps` | Ownership checks and state mutations run in one Redis `WATCH`/`MULTI` transaction | `MemoryMatchesPos` |
| `SampleSetNX` | Stateless sampler replicas publish `f:<req>` and bill `service` once (`xlen == 0`) | `SampleChargedOnce` |
| `UseDelta` | Planner advances `lag` from `service[a] - seen[a]` (never overwriting `service`) | `ExactAccounting` |
| `KeepNegLag` | Sleeping preserves negative debt (decaying only as active peers receive work) and zeros positive idle credit | `NoFlickerGain` |
| `AmortizeCost` | Preemption requires `ran >= holder.cost` and `lag[waiter] - lag[holder] >= switch_cost` | `GoodputBounded` |

## Running TLC

```bash
chmod +x docs/model/check.sh
TLA2TOOLS_JAR=/path/to/tla2tools.jar ./docs/model/check.sh          # fast smoke + all 9 negative checks (~20s)
TLA2TOOLS_JAR=/path/to/tla2tools.jar ./docs/model/check.sh --full   # all 7 exhaustive models (122.7M states) + 9 negative checks
```
