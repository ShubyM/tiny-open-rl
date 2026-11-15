# The state the control plane keeps, and the rules that keep it right

The scheduler, the node daemons and the worker processes share a few Redis keys. Each key has one meaning and one
kind of writer, and five rules say when a write is allowed. [FleetModel.tla](FleetModel.tla) is that design as a TLA+
specification, and [Fair.tla](Fair.tla) is the fair-service floor. TLC checked both exhaustively at small sizes.

## State

| Key | Holds | Written by |
|---|---|---|
| `tinyrl:run:<run>` `worker` | the worker a run is assigned to | the scheduler |
| `tinyrl:owner:<run>` | the worker process whose memory holds the run's live state | a process takes it with `SET NX` and hands it off; the scheduler frees it once the owner's lease is gone |
| `tinyrl:run:<run>` `acked` | the stream id of the last command that changed the run's state | the owner, with the reply |
| `tinyrl:run:<run>` `checkpoint` | where exactly that state is saved; absent when it is saved nowhere | the owner, with the reply or the handoff |
| `tinyrl:lease:<process>` | a key with a TTL: the process is live | the process |
| `tinyrl:granted:<node>` | worker → the GPUs it may be on | the scheduler adds; only that node removes |
| `tinyrl:ready` | worker → the accounts it has work ready for | the worker's node |
| `tinyrl:service`, `tinyrl:floor` | tokens served to each account, and the service floor | nodes charge with the reply; the scheduler lifts |

A process id names one launch and is never used again. A stream id names one exact training state: every command
that changes a run gets a new one.

## Rules

| | Rule | In the code | Without it |
|---|---|---|---|
| R1 | Only the node a GPU is on gives it back. A lease that ran out frees no GPU. | `NodeDaemon.release` is the only caller of `Bus.release_gpus`; a node whose heartbeat ran out takes its grants with it | a stalled process is still on the GPU when the next one starts |
| R2 | A process leaves its GPUs before the node gives them back. | `turn()` parks, then releases; `cleanup()` closes, then releases | the same |
| R3 | Only a run's owner writes its row. | `Bus.write_reply` and `Bus.hand_off` are transactions on `tinyrl:owner:<run>` | a process that lost the run moves `acked` or the checkpoint behind its new owner |
| R4 | A run is restored only from a checkpoint of its acknowledged state. | the reply that moves `acked` deletes `checkpoint` in the same transaction, so a checkpoint that is there is exact | old tensors under a newer cursor |
| R5 | With a command or a save in flight for a run, its process starts nothing else for it. | a trainer runs one command at a time; a run just taken is dropped from memory and its in-flight commands failed first (`own()`) | a command from before the run was freed lands on reloaded state |

A new owner restores a run from its checkpoint. A run with `acked` and no checkpoint lost state, and fails until its
client loads a save. A run without `acked` never changed, and starts again from its record.

## What TLC checked

In the model every slow step (restoring, running a command, writing checkpoint files) is separate from the transaction
that publishes it, and between any two steps a process may stall, lose its lease and renew it, crash, lose its node,
or have its run freed by the scheduler.

- No two processes are on one GPU, and a process on a GPU holds its grant.
- The state an owner serves a run from is exactly the acknowledged one.
- Acknowledged progress moves forward along one history, and the checkpoint is a state the run passed through.
- A planned handoff loses nothing.
- Fair service: no charge is lost, an account back from idle holds no credit, a waiting account keeps its place, the
  floor only rises, and a free GPU goes to a least-served account.

| Instance | Distinct states | Result |
|---|---:|---|
| 1 job, 2 workers, 2 nodes with one GPU each, 2 commands | 1,276,680 | no violation |
| 1 job, 3 workers, same nodes, 2 commands | 45,067,632 | no violation |
| 1 job, 2 workers, same nodes, 3 commands | 4,350,148 | no violation |
| 2 jobs, 2 workers, one GPU, 1 command each | 60,862,452 | no violation |
| Fair service, 3 accounts | 2,853,948 | no violation |

Each rule switched off alone (1 job, 3 workers) breaks an invariant: R1 and R2 break GPU exclusivity, R3 and R4 break
"memory is the acknowledged state", R5 breaks "a handoff loses nothing". Lifting an account by overwriting its service
instead of adding the gap loses a charge.

## Where the code differs from the model

- The model keeps a checkpoint with its position and compares it with `acknowledged`. The code keeps the checkpoint
  only while the two are equal.
- In the model the scheduler may free a run at any moment. In the code it does so only while the owner's lease is gone.
- The model locks single GPUs. The code grants a worker all its GPUs at once, lets workers with known memory budgets
  share one (`scheduler.can_share`), and each node also checks its own processes before starting one (`clear()`).
- A wide trainer that is stopping takes its GPUs back to save through all its ranks. It waits for the node's own
  check, not for a grant, since a stopping node no longer reads the table.
- Commands are claimed without a check; only replies are fenced.
- A local store has no versions, so there a checkpoint is named by its URI alone.

## Limits

These are exhaustive checks of small instances, not a proof for every size. Nothing here shows progress (that a run
is eventually served, or that no account starves). Wide workers, token budgets, exclusive workers, several nodes
coming and going, and checkpoint deletion are not in the model. Nothing checks that the code follows the model
except the tests.

## Running it

`./check.sh ok "j1" "w1, w2" "g1, g3" OneEach 2` checks the smallest instance in seconds, and
`./check.sh no_r3 "j1" "w1, w2" "g1, g3" OneEach 2 WritesCheckOwner=FALSE` shows the trace that breaks without R3.
See the top of [check.sh](check.sh).
