import asyncio
import json
import random
import tempfile
import time
import types
import unittest
from dataclasses import replace
from unittest import mock

import torch

from test.execution import InlineProcess, TestNode
from test.helpers import BusTest, FakeEngine, new_id, tiny
from test.test_api import proto_body
from test.test_sampler import JOB
from test.test_trainer import LORA, counting
from tinyrl.api import wire
from tinyrl.bus import QUANTUM_MS, Cmd, Ev, Op, Role, Run
from tinyrl.control import scheduler
from tinyrl.control.fairness import Demand, Worker, admit_samples, plan_gpus, rank_runs, update_lag, waterfill
from tinyrl.control.node_daemon import NodeDaemon, WorkerHandle
from tinyrl.control.scheduler import (
  REBALANCE_MS,
  Job,
  Proc,
  Spec,
  State,
  from_plan,
  step,
  to_plan,
)
from tinyrl.storage.store import Store
from tinyrl.types import MAX_SEQS
from tinyrl.workers.model import Model

GIB = 2**30


class Cluster:
  """Nodes, runs and queues that step the pure L2 cluster planner the way `Bus.reconcile` does."""

  def __init__(self):
    self.nodes, self.runs, self.queues, self.service, self.s = {}, {}, {}, {}, State()

  def settle(self) -> list[tuple[Ev, dict]]:
    self.s.nodes = {n: (v["gpus"], v["memory"]) for n, v in self.nodes.items()}
    self.s.jobs, self.s.service = scheduler.jobs(self.runs, self.queues), dict(self.service)
    self.s, changes = step(self.s)
    return changes

  def node(self, name: str, gpus: int, memory: int = 80 * GIB) -> None:
    self.nodes[name] = {"gpus": gpus, "memory": memory}
    self.settle()

  def lose(self, node: str) -> None:
    self.nodes.pop(node, None)

  def run(self, name: str, *, backlog: int = 0, owner: list[str] | None = None, **body) -> str:
    self.runs[name] = {"record": {"role": "trainer", "base": "a", "rank": 0} | body, "owner": owner}
    self.queues[name] = backlog
    self.settle()
    return self.s.procs_of(name)[0]

  def queue(self, run: str, n: int) -> None:
    self.queues[run] = n

  def serve(self, run: str, gpu_ms: float) -> None:
    acct = self.s.account(run)
    self.service[acct] = self.service.get(acct, 0.0) + gpu_ms

  def delete(self, run: str) -> None:
    self.runs.pop(run, None)
    self.queues.pop(run, None)


def apply_gpus(
  workers: dict[str, Worker],
  quanta: dict[str, float],
  state: State,
  total: int,
  now: float,
) -> tuple[dict[str, tuple[int, ...]], float | None]:
  """Drive pure `plan_gpus` on `Worker` snapshots, resetting `hold_until` only when a worker's block changes."""
  ranked = {
    p: replace(w, lag=max((state.ready_lag(r) for r in state.runs(p) if state.ready(r)), default=-w.gpu_ms))
    for p, w in workers.items()
  }
  keep, wake_at = plan_gpus(ranked, total, now)
  for p, w in list(workers.items()):
    want = keep.get(p, ())
    until = now + quanta[p] if want and want != w.current else w.hold_until
    workers[p] = replace(w, current=want, hold_until=until)
  return keep, wake_at


def check_plan_invariants(test: unittest.TestCase, s: State) -> None:
  for node, (total, _) in s.nodes.items():
    test.assertLessEqual(s.excl_used(node), total)
  for p in s.procs.values():
    test.assertTrue(p.node is None or (p.node in s.nodes and p.spec.gpus <= s.nodes[p.node][0]))
  for r, proc in s.home.items():
    test.assertIn(r, s.jobs)
    test.assertEqual(s.jobs[r].spec.role, Role.TRAINER)
    test.assertIn(proc, s.procs)


def check_gpu_invariants(test: unittest.TestCase, keep: dict, workers: dict[str, Worker], total: int) -> None:
  seen = []
  for p, gs in keep.items():
    test.assertEqual(len(gs), workers[p].gpus)
    test.assertEqual(len(set(gs)), workers[p].gpus)
    test.assertTrue(all(0 <= g < total for g in gs))
    seen.extend(gs)
  test.assertEqual(len(seen), len(set(seen)))


class TestCluster(unittest.TestCase):
  def test_runs_are_assigned_and_empty_procs_wait_for_live_owner_detach(self):
    c = Cluster()
    c.node("n", 8)
    a = c.run("a", rank=8)
    self.assertEqual(c.run("b", rank=8), a)
    self.assertEqual(c.s.procs[a].node, "n")
    c.delete("a")
    self.assertEqual(c.settle(), [])
    c.run("c")
    c.runs["c"]["owner"] = [a, "n", "boot1"]
    c.delete("b")
    self.assertEqual(c.settle(), [])
    self.assertIn(a, c.s.procs)
    c.runs["c"]["owner"] = None
    self.assertEqual(c.settle(), [(Ev.UNPLACED, {"proc": a, "rule": "place.emptied"})])
    self.assertNotIn(a, c.s.procs)

  def test_a_new_proc_never_takes_a_name_in_use(self):
    c = Cluster()
    c.node("n", 1)
    c.s.procs["p1"] = Proc(Spec(lora=True), "n")
    self.assertNotEqual(c.run("f"), "p1")

  def test_procs_go_to_the_node_with_the_fewest_procs_per_gpu(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 2)
    procs = [c.run(name, rank=4, base=name) for name in ("r1", "r2", "r3")]
    self.assertEqual([c.s.procs[p].node for p in procs], ["a", "b", "b"])

  def test_full_fine_tune_sampler_stays_near_its_training_run(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 1)
    a, b = c.run("a"), c.run("b")
    self.assertEqual([c.s.procs[p].node for p in (a, b)], ["a", "b"])
    sampler = c.run("sample", role="sampler", train_run="b")
    self.assertEqual(c.s.procs[sampler].node, "b")

  def test_full_fine_tune_sampler_goes_elsewhere_when_its_trainers_node_is_full(self):
    c = Cluster()
    c.node("a", 2)
    c.node("b", 8)
    x, other = c.run("x", exclusive=True), c.run("r", rank=4)
    self.assertEqual([c.s.procs[p].node for p in (x, other)], ["a", "b"])
    c.s.procs[other].node = "a"
    sampler = c.run("sample", role="sampler", train_run="x", exclusive=True)
    self.assertEqual(c.s.procs[sampler].node, "b")

  def test_a_full_fine_tunes_trainer_keeps_room_beside_it_for_its_sampler(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 2)
    c.node("c", 8)
    t = c.run("t")
    others = [c.run(f"r{i}", rank=4, base=f"m{i}") for i in range(5)]
    self.assertEqual([c.s.procs[p].node for p in [t, *others]], ["b", "a", "c", "c", "c", "c"])
    lora = c.run("q", role="sampler", rank=4, base="m0")
    sampler = c.run("sample", role="sampler", train_run="t")
    self.assertEqual([c.s.procs[p].node for p in (lora, sampler)], ["c", "b"])

  def test_a_second_sampler_stays_near_its_trainer_when_other_free_gpus_are_owed(self):
    c = Cluster()
    c.node("a", 2)
    c.node("b", 8)
    t = c.run("t")
    others = [c.run(f"r{i}") for i in range(4)]
    self.assertEqual([c.s.procs[p].node for p in [t, *others]], ["a", "b", "b", "b", "b"])
    q = c.run("q", role="sampler", train_run="t")
    g = c.run("g", role="sampler", train_run="t", base="b")
    self.assertEqual([c.s.procs[p].node for p in (q, g)], ["a", "a"])

  def test_lost_or_shrunk_node_procs_are_reassigned(self):
    c = Cluster()
    c.node("a", 2)
    p = c.run("r", gpus=2)
    c.node("b", 2)
    c.lose("a")
    changes = c.settle()
    self.assertEqual(changes[0], (Ev.UNPLACED, {"proc": p, "rule": "place.node_lost"}))
    self.assertEqual(c.s.procs[p].node, "b")
    c.nodes["b"]["gpus"] = 1
    c.settle()
    self.assertIsNone(c.s.procs[p].node)

  def test_multi_gpu_proc_waits_for_a_node_that_has_enough_gpus(self):
    c = Cluster()
    c.node("a", 1)
    p = c.run("wide", gpus=2)
    self.assertIsNone(c.s.procs[p].node)
    c.node("b", 4)
    self.assertEqual(c.s.procs[p].node, "b")

  def test_lora_runs_share_when_idle_split_when_busy_and_merge_when_squeezed(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 1)
    p1 = c.run("r1", rank=8)
    self.assertEqual(c.run("r2", rank=8), p1)
    others = [c.run("r3", rank=8, base="b"), c.run("f"), c.run("x", rank=8, exclusive=True)]
    self.assertEqual(len({p1, *others}), 4)

    c = Cluster()
    c.node("a", 1)
    p1 = c.run("r1", rank=8, backlog=1)
    self.assertEqual(c.run("r2", rank=8, backlog=1), p1)
    c.node("b", 1)
    p2 = c.s.home["r2"]
    self.assertNotEqual(p1, p2)
    self.assertEqual({c.s.procs[p1].node, c.s.procs[p2].node}, {"a", "b"})
    c.run("excl", exclusive=True, backlog=1)
    c.settle()  # the trainer on the exclusive's node is unplaced and joins the other node
    c.settle()  # where the two merge
    self.assertEqual(c.s.home["r1"], c.s.home["r2"])
    check_plan_invariants(self, c.s)

  def test_subscribed_node_with_idle_peers_does_not_split_or_scale_out_thrash(self):
    c = Cluster()
    c.node("a", 2)
    s1 = c.run("s1", role="sampler", base="m1")
    c.run("s2", role="sampler", base="m2")
    p1 = c.run("r1", rank=8, backlog=1)
    self.assertEqual(c.run("r2", rank=8, backlog=1), p1)
    self.assertEqual(c.s.home["r1"], c.s.home["r2"])
    c.queue("r1", 0)
    c.queue("r2", 0)
    c.queue("s1", 12)
    c.settle()
    self.assertEqual((c.s.procs_of("s1"), c.s.made), ([s1], 3))
    check_plan_invariants(self, c.s)

  def test_sampling_sessions_share_only_their_kind_and_exclusive_training_run(self):
    c = Cluster()
    c.run("X", rank=8, exclusive=True)
    c.run("Z", rank=8, exclusive=True)
    a = c.run("a", role="sampler", rank=8, train_run="A")
    b = c.run("b", role="sampler", rank=8, train_run="B")
    fft = c.run("f", role="sampler", train_run="A")
    x = c.run("x", role="sampler", rank=8, train_run="X", exclusive=True)
    y = c.run("y", role="sampler", rank=8, train_run="X", exclusive=True)
    z = c.run("z", role="sampler", rank=8, train_run="Z", exclusive=True)
    self.assertEqual(a, b)
    self.assertEqual(x, y)
    self.assertEqual(len({a, b, fft, x, y, z}), 4)
    self.assertNotEqual(c.s.home["X"], c.s.home["Z"])  # exclusive LoRA runs on one base still get a trainer each

  def test_a_sampling_session_without_an_account_bills_its_training_run(self):
    c = Cluster()
    c.run("t", session="s1")
    c.run("q", role="sampler", train_run="t", session="s2")
    c.run("base", role="sampler", session="s3")
    self.assertEqual([c.s.account(r) for r in ("t", "q", "base")], ["s1", "s1", "s3"])

  def test_a_proc_is_owed_what_its_ready_runs_are_owed(self):
    jobs = {"a": Job(Spec(lora=True), account="A", backlog=1), "b": Job(Spec(lora=True), account="B")}
    s = State(procs={"p": Proc(Spec(lora=True), "n")}, home={"a": "p", "b": "p"}, jobs=jobs, lag={"A": 5.0, "B": 9.0})
    self.assertEqual(s.lag_of("p"), 5.0)

  def test_demand_is_what_each_account_could_run_now(self):
    c = Cluster()
    c.node("a", 2)
    c.node("b", 2)
    c.run("idle", rank=4, base="z", account="idle")  # nothing queued: nothing
    c.run("wide", gpus=4, backlog=1, account="wide")  # wider than any node: nothing
    c.run("t", rank=4, backlog=9, account="t")  # a trainer is one process however long its queue
    c.run("x", role="sampler", train_run="t", exclusive=True, backlog=9, account="x")  # so is an exclusive sampler
    c.run("s", role="sampler", base="m", backlog=3 * MAX_SEQS, account="s")  # a shared sampler: a replica per MAX_SEQS
    c.run("big", role="sampler", base="n", backlog=99 * MAX_SEQS, account="big")  # up to every GPU there is
    self.assertEqual(scheduler.demand(c.s, 4.0), {"t": 1, "x": 1, "s": 3, "big": 4})

  def test_a_backlogged_shared_sampler_gets_a_replica_per_batch_and_an_exclusive_one_does_not(self):
    c = Cluster()
    for node in "abcde":
      c.node(node, 1)
    c.run("t")
    c.run("x", role="sampler", train_run="t", exclusive=True, backlog=3 * MAX_SEQS)
    c.run("s", role="sampler", backlog=3 * MAX_SEQS)
    self.assertEqual((len(c.s.procs_of("x")), len(set(c.s.procs_of("s")))), (1, 3))

  def test_unplaced_sampler_pool_does_not_scale_out_onto_busy_single_gpu_node(self):
    spec = Spec(Role.SAMPLER, "tiny", lora=True, gpus=1)
    state = State(
      nodes={"n1": (1, 0)},
      jobs={"s1": Job(spec, backlog=4 * MAX_SEQS, account="alice")},
    )
    new, _ = step(state)
    self.assertEqual(len(new.procs), 1)
    self.assertEqual(next(iter(new.procs.values())).node, "n1")

  def test_scale_out_reuses_existing_placed_pool_before_spawning_more(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 2)
    first = c.run("s", role="sampler")
    c.s.procs["p9"] = Proc(c.s.procs[first].spec, "b")
    c.runs["o"], c.queues["o"] = dict(c.runs["s"]), 0
    c.queue("s", 2 * MAX_SEQS)
    c.settle()
    self.assertEqual(c.s.procs_of("s"), [first, "p9"])  # the existing replica on b, not a third process
    c.queue("s", 3 * MAX_SEQS)
    c.queue("o", 1)
    c.settle()
    self.assertEqual(len(c.s.procs_of("s")), 3)  # a third replica spawns on b's other GPU
    new_proc = next(p for p in c.s.procs_of("s") if p not in (first, "p9"))
    self.assertEqual(c.s.procs[new_proc].node, "b")

  def test_two_busy_sessions_do_not_split_their_sampler(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 1)
    self.assertEqual(c.run("s1", role="sampler", rank=4), c.run("s2", role="sampler", rank=4))
    c.queue("s1", 1)
    c.queue("s2", 1)
    c.settle()
    self.assertEqual(c.s.procs_of("s1"), c.s.procs_of("s2"))
    self.assertEqual(len(c.s.procs_of("s1")), 1)

  def test_two_busy_fft_samplers_scale_out_onto_spare_gpu(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 1)
    c.run("f1", role="sampler", rank=0)
    c.run("f2", role="sampler", rank=0)
    c.queue("f1", 1)
    c.queue("f2", 1)
    c.settle()
    self.assertEqual(len(c.s.procs_of("f1")), 2)

  def test_idle_lora_trainers_stay_split_while_their_node_has_room(self):
    c = Cluster()
    c.node("a", 2)
    self.assertNotEqual(c.run("r1", rank=4, backlog=1), c.run("r2", rank=4, backlog=1))
    c.queue("r1", 0)
    c.queue("r2", 0)
    c.settle()
    self.assertNotEqual(c.s.home["r1"], c.s.home["r2"])

  def test_a_busy_lora_trainer_moves_to_a_spare_node_rather_than_merging(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 1)
    p1, p2 = c.run("r1", rank=4, backlog=1), c.run("r2", rank=4, backlog=1)
    c.lose("b")
    c.settle()
    self.assertEqual((c.s.procs[p1].node, c.s.procs[p2].node), ("a", "a"))
    c.node("c", 1)
    self.assertEqual((c.s.home["r1"], c.s.home["r2"]), (p1, p2))
    self.assertEqual({c.s.procs[p1].node, c.s.procs[p2].node}, {"a", "c"})

  def test_an_idle_proc_stays_on_a_crowded_node(self):
    c = Cluster()
    c.node("a", 1)
    idle = c.run("idle", rank=4, base="x")
    for i in range(2):
      c.run(f"busy{i}", rank=4, base=f"y{i}", backlog=1)
    c.node("b", 1)
    self.assertEqual(c.s.procs[idle].node, "a")

  def test_a_busy_proc_beside_an_idle_one_is_not_crowded(self):
    c = Cluster()
    c.node("a", 1)
    c.run("idle", rank=4, base="x")
    busy = c.run("busy", rank=4, base="y", backlog=1)
    c.node("b", 1)
    self.assertEqual(c.s.procs[busy].node, "a")

  def test_an_exclusive_proc_holds_its_gpus_whether_or_not_it_is_busy(self):
    jobs = {"r": Job(Spec(exclusive=True), backlog=1)}
    s = State(nodes={"n": (2, 0)}, procs={"x": Proc(Spec(exclusive=True), "n")}, home={"r": "x"}, jobs=jobs)
    self.assertEqual(s.spare("n"), 1)

  def test_crowding_moves_a_shared_proc_and_never_an_exclusive_one(self):
    c = Cluster()
    c.node("a", 2)
    x = c.run("x", exclusive=True, backlog=1)
    shared = [c.run(f"r{i}", rank=4, base=f"m{i}", backlog=1) for i in range(2)]
    c.node("b", 1)
    self.assertEqual(c.s.procs[x].node, "a")
    self.assertEqual(sorted(c.s.procs[p].node for p in shared), ["a", "b"])

  def test_scale_out_does_not_reuse_a_sampler_that_has_no_node(self):
    c = Cluster()
    c.node("a", 1)
    first = c.run("s", role="sampler")
    c.s.procs["p9"] = Proc(c.s.procs[first].spec, None)
    c.runs["o"], c.queues["o"] = dict(c.runs["s"]), 0
    c.queue("s", 2 * MAX_SEQS)
    c.settle()
    self.assertEqual(c.s.procs_of("s"), [first])

  def test_rotation_weighs_a_mover_against_the_busy_shared_procs_on_the_other_node(self):
    def target(c_lag: float, n2: int = 1, **extra) -> str | None:
      specs = {"me": (Spec(lora=True, base="me"), "n1", 1), "b": (Spec(lora=True, base="b"), "n1", 1)}
      specs |= {"c": (Spec(lora=True, base="c"), "n2", 1)} | extra
      s = State(nodes={"n1": (1, 0), "n2": (n2, 0)}, lag={"me": 30000.0, "b": 25000.0, "c": c_lag, "x": -50000.0})
      for name, (spec, node, backlog) in specs.items():
        s.procs[name], s.home[name], s.jobs[name] = Proc(spec, node), name, Job(spec, account=name, backlog=backlog)
      return scheduler.rebalance_target(s, "me")

    self.assertEqual(target(5000.0), "n2")  # owed REBALANCE_MS more than `c`, who is owed less than `b`
    self.assertIsNone(target(15000.0))  # not owed enough more than `c`
    self.assertIsNone(target(15000.0, d=(Spec(lora=True, base="d"), "n2", 0)))  # an idle proc there changes nothing
    self.assertIsNone(target(15000.0, 2, x=(Spec(exclusive=True), "n2", 1)))  # nor does an exclusive one

  def test_a_new_run_joins_the_matching_trainer_with_the_fewest_runs(self):
    c = Cluster()
    c.node("a", 2)
    first = c.run("r1", rank=4)
    c.s.procs["p9"] = Proc(c.s.procs[first].spec, "a")
    c.runs["r2"], c.queues["r2"], c.s.home["r2"] = dict(c.runs["r1"]), 0, first
    c.runs["r3"], c.queues["r3"], c.s.home["r3"] = dict(c.runs["r1"]), 0, "p9"
    self.assertEqual(c.run("r4", rank=4), "p9")

  def test_only_a_trainer_is_owed_room_for_a_sampler(self):
    c = Cluster()
    c.node("a", 2)
    q = c.run("q", role="sampler")
    c.node("b", 2)
    c.run("r1", rank=4, base="m1")
    self.assertEqual([c.s.procs[p].node for p in (q, c.s.home["r1"])], ["a", "b"])
    new = c.run("r2", rank=4, base="m2")
    self.assertEqual(c.s.procs[new].node, "a")

  def test_an_empty_proc_owned_from_a_lost_node_is_collected(self):
    c = Cluster()
    c.node("n", 1)
    c.run("r")
    c.s.procs["p9"] = Proc(Spec(), None)
    c.runs["r"]["owner"] = ["p9", "gone-1", "1"]
    c.settle()
    self.assertNotIn("p9", c.s.procs)

  def test_sampler_settings_follow_the_node(self):
    self.assertEqual(scheduler.settings(1, 24 * GIB), {})
    self.assertEqual(scheduler.settings(2, 80 * GIB), {"TINYRL_KV_BYTES": str(24 * GIB)})

  def test_exclusive_evacuates_squeezed_shareable_procs_and_never_overcommits(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 1)
    pa = c.run("ra", backlog=1)
    pb = c.run("rb", backlog=1)
    self.assertEqual({c.s.procs[pa].node, c.s.procs[pb].node}, {"a", "b"})
    x = c.run("x", exclusive=True, backlog=1)
    c.settle()  # the proc that no longer fits beside `x` is unplaced, and placed again on the other node
    x_node = c.s.procs[x].node
    other = "b" if x_node == "a" else "a"
    self.assertEqual((c.s.procs[pa].node, c.s.procs[pb].node), (other, other))
    # Even huge lag never lets a shareable proc move onto `x`'s fully exclusive node.
    c.s.lag["ra"] = c.s.lag["rb"] = 100 * REBALANCE_MS
    c.settle()
    self.assertEqual((c.s.procs[pa].node, c.s.procs[pb].node), (other, other))
    # A third exclusive waits unplaced when both 1-GPU nodes already hold exclusives.
    y = c.run("y", exclusive=True)
    z = c.run("z", exclusive=True)
    self.assertEqual({c.s.procs[x].node, c.s.procs[y].node}, {"a", "b"})
    self.assertIsNone(c.s.procs[z].node)
    check_plan_invariants(self, c.s)

  def test_random_clusters_settle_round_trip_and_preserve_invariants(self):
    for seed in range(100):
      rng, c = random.Random(seed), Cluster()
      for _ in range(80):
        roll, name = rng.random(), f"x{rng.randrange(8)}"
        if roll < 0.15:
          c.node(name, rng.choice([1, 2, 4]))
        elif roll < 0.23:
          c.lose(name)
        elif roll < 0.65 and name not in c.runs:
          role = rng.choice(["trainer", "trainer", "sampler"])
          excl = rng.random() < 0.15 and role == "trainer"
          c.run(
            name,
            role=role,
            rank=rng.choice([0, 8]),
            base=str(rng.randrange(3)),
            gpus=rng.choice([1, 2]),
            exclusive=excl,
            backlog=rng.choice([0, 1, 2, 8]),
          )
        elif roll < 0.82 and c.runs:
          r = rng.choice(list(c.runs))
          c.queue(r, rng.choice([0, 1, 8]))
          if rng.random() < 0.5 and r in c.s.jobs:
            c.serve(r, rng.choice([500.0, 2000.0, 25_000.0]))
        else:
          c.delete(name)
        c.settle()
        check_plan_invariants(self, c.s)
        saved = from_plan(to_plan(c.s))  # the plan comes back from the database as it went in
        saved.nodes, saved.jobs, saved.service = c.s.nodes, c.s.jobs, c.s.service
        self.assertEqual(saved, c.s)
        c.settle()
        check_plan_invariants(self, c.s)
        again, changes = step(c.s)
        self.assertEqual(changes, [])
        self.assertEqual(again, c.s)


class TestLocalQuantum(unittest.TestCase):
  def make_daemon(self, gpus: int) -> NodeDaemon:
    store = Store(tempfile.mkdtemp())
    node = NodeDaemon(types.SimpleNamespace(r=None), gpus=gpus, node="n", gpu_mib=0, store=store)  # type: ignore[arg-type]
    self.addCleanup(store.close)
    return node

  def test_ready_runs_share_service_even_when_commands_have_different_costs(self):
    heads = [Cmd(run, "1-0", Op.FORWARD_BACKWARD, b"") for run in ("slow", "fast")]
    service = {"slow": 0.0, "fast": 0.0}
    state = State()
    for _ in range(11):
      run = rank_runs(heads, state.ready_lag, service)[0]
      service[run] += 10 if run == "slow" else 1
    self.assertEqual(service, {"slow": 10.0, "fast": 10.0})

  def test_equal_service_uses_numeric_stream_arrival_order(self):
    heads = [
      Cmd("later", "10-0", Op.OPTIM_STEP, b""),
      Cmd("same_ms_later", "9-10", Op.OPTIM_STEP, b""),
      Cmd("first", "9-2", Op.OPTIM_STEP, b""),
    ]
    state = State(jobs={cmd.run: Job(Spec(), account="alice") for cmd in heads})
    self.assertEqual(rank_runs(heads, state.ready_lag, {}), ["first", "same_ms_later", "later"])
    # Sub-microsecond float drift from split waves must not outrank run service or arrival order.
    tiny_drift = {"later": 1e-12, "same_ms_later": 0.0, "first": 0.0}
    self.assertEqual(rank_runs(heads, tiny_drift.__getitem__, {})[0], "first")

  def test_account_lag_precedes_run_service_and_arrival(self):
    cluster = Cluster()
    cluster.run("current", account="alice")
    cluster.run("save", account="alice")
    cluster.run("waiting", account="bob")
    state = cluster.s
    state.lag = {"alice": 10, "bob": 5}
    state.seen = {"alice": 100}
    state.service = {"alice": 110}
    heads = [
      Cmd("current", "1-0", Op.OPTIM_STEP, b""),
      Cmd("save", "2-0", Op.SAVE_WEIGHTS_FOR_SAMPLER, b""),
      Cmd("waiting", "3-0", Op.OPTIM_STEP, b""),
    ]
    self.assertEqual(rank_runs(heads, state.ready_lag, {"waiting": 1000})[0], "waiting")

  def test_continuous_sampler_saves_share_a_trainer_with_training_work(self):
    cluster = Cluster()
    for run in ("saving", "training"):
      cluster.run(run, account="alice")
    heads = [Cmd("saving", "1-0", Op.SAVE_WEIGHTS_FOR_SAMPLER, b""), Cmd("training", "2-0", Op.OPTIM_STEP, b"")]
    service = {"saving": 0, "training": 0}
    for _ in range(12):
      run = rank_runs(heads, cluster.s.ready_lag, service)[0]
      service[run] += 3 if run == "saving" else 1
    self.assertEqual(service, {"saving": 9, "training": 9})

  def test_admit_samples_never_exceeds_free_slots(self):
    queues = {
      "b1": [Cmd("b1", f"1-{i}", Op.SAMPLE, b"{}") for i in range(2)],
      "b2": [Cmd("b2", f"2-{i}", Op.SAMPLE, b"{}") for i in range(2)],
    }
    heads = [q[0] for q in queues.values()]
    admitted = admit_samples(
      heads,
      lambda r, n: queues[r][:n],
      lambda r: "B",
      lambda r: 0.0,
      {},
      lambda r: None,
      {"A": 6},
      set(),
      8,
      16,
    )
    self.assertEqual(len(admitted), 2)
    self.assertEqual({c.run for c in admitted}, {"b1", "b2"})

  def test_local_guard_enforces_gpu_exclusivity_under_concurrent_handoffs(self):
    async def run():
      node = self.make_daemon(4)
      active_gpus: dict[str, tuple[int, ...]] = {}
      violations: list[str] = []

      class FakeModel:
        proc = None

        def __init__(self, wid):
          self.wid = wid

        async def resume(self, gpus):
          await asyncio.sleep(0)
          if overlap := {k: v for k, v in active_gpus.items() if k != self.wid and set(v) & set(gpus)}:
            violations.append(f"{self.wid} resumed on {gpus} while {overlap}")
          active_gpus[self.wid] = gpus

        async def park(self):
          await asyncio.sleep(0)
          active_gpus.pop(self.wid, None)

        async def close(self):
          active_gpus.pop(self.wid, None)

      node.open_trainer = lambda wid, width, env: FakeModel(wid)
      rng = random.Random(7)
      specs = [
        (f"p{i}", Spec(Role.TRAINER, "b", gpus=w), tuple(range(g, g + w)))
        for i, (g, w) in enumerate([(0, 2), (2, 2), (0, 1), (1, 1), (2, 1), (3, 1)])
      ]

      async def use(wid, spec, gpus):
        h = node.workers.setdefault(wid, WorkerHandle(wid, spec, FakeModel(wid)))
        async with node.gpu_cond:
          h.active += 1
        try:
          await node.claim_gpus(h, gpus)
          await h.ensure_gpu(gpus)
          await asyncio.sleep(0)
        finally:
          async with node.gpu_cond:
            h.active -= 1
            if h.active == 0:
              h.claimed = ()
            node.gpu_cond.notify_all()

      for _ in range(40):
        batch = rng.sample(specs, 3)
        await asyncio.gather(*(use(*item) for item in batch))
        flat = [g for h in node.workers.values() for g in h.resident]
        self.assertEqual(len(flat), len(set(flat)))
      self.assertEqual(violations, [])

    asyncio.run(run())

  def test_active_quantum_prevents_thrash_and_overtime_hands_off_with_zero_wait(self):
    c = Cluster()
    c.node("n", 1)
    p1 = c.run("a", backlog=1)
    workers, quanta = {p1: Worker(hold_until=2.0)}, {p1: 2.0}

    # Lone client enters OVERTIME at t=2s: no switch, timer stays disarmed, the deadline never resets.
    for t in (0.0, 0.5, 1.9, 2.5, 10.0):
      keep, wake = apply_gpus(workers, quanta, c.s, 1, t)
      self.assertEqual((keep, wake, workers[p1].hold_until), ({p1: (0,)}, None, 2.0))
      workers[p1] = replace(workers[p1], gpu_ms=workers[p1].gpu_ms + 500.0)
      c.serve("a", 500.0)
      c.settle()

    # Late joiner `p2` arrives at t=10s: `p1` is already in overtime -> `p2` gets GPU immediately, held for its quantum.
    p2 = c.run("b", backlog=1)
    workers[p2], quanta[p2] = Worker(hold_until=2.0), 2.0
    keep, wake = apply_gpus(workers, quanta, c.s, 1, 10.0)
    self.assertEqual((keep, wake), ({p2: (0,)}, None))
    # Five rapid command boundaries inside [10.0, 12.0) all keep `p2` and arm the quantum timer for t=12.0.
    for t in (10.1, 10.4, 10.9, 11.5, 11.9):
      keep, wake = apply_gpus(workers, quanta, c.s, 1, t)
      self.assertEqual((keep, wake, workers[p2].hold_until), ({p2: (0,)}, 12.0, 12.0))
    # If `p2` goes idle early (RL ping-pong), `p1` takes GPU immediately; if both idle, last owner stays warm.
    workers[p2] = replace(workers[p2], demand=Demand.IDLE)
    keep, wake = apply_gpus(workers, quanta, c.s, 1, 11.95)
    self.assertEqual((keep, wake), ({p1: (0,)}, None))
    workers[p1] = replace(workers[p1], demand=Demand.IDLE)
    keep, wake = apply_gpus(workers, quanta, c.s, 1, 12.1)
    self.assertEqual((keep, wake), ({p1: (0,)}, None))

  def test_asymmetric_switch_cost_and_account_grouping_share_gpu_ms_fairly(self):
    # Trainer (quantum=1s) vs slow-waking sampler (quantum=6s): trainer stays across overtime to catch up with few swaps.
    c = Cluster()
    c.node("n", 1)
    pt = c.run("train", backlog=1)
    ps = c.run("samp", role="sampler", backlog=1)
    quanta = {pt: 1.0, ps: 6.0}
    workers = {p: Worker(hold_until=q) for p, q in quanta.items()}
    switches, last, dt = 0, None, 0.5
    for i in range(240):
      keep, _ = apply_gpus(workers, quanta, c.s, 1, i * dt)
      [h] = keep
      switches += h != last
      last = h
      workers[h] = replace(workers[h], gpu_ms=workers[h].gpu_ms + dt * 1000)
      c.serve(c.s.runs(h)[0], dt * 1000)
      c.settle()
    self.assertAlmostEqual(c.service["train"] / sum(c.service.values()), 0.5, delta=0.05)
    self.assertLessEqual(switches, 24)  # switches once per ~6s sampler quantum, not every 1s trainer quantum

    # Account grouping: Alice's trainer + sampler pair splits 50/50 against Bob's solo trainer (25/25/50).
    c = Cluster()
    c.node("n", 1)
    t1 = c.run("t1", session="alice", backlog=1)
    s1 = c.run("s1", role="sampler", train_run="t1", backlog=1)
    t2 = c.run("t2", session="bob", backlog=1)
    quanta = {p: 1.0 if p != s1 else 4.0 for p in (t1, s1, t2)}
    workers = {p: Worker(hold_until=q) for p, q in quanta.items()}
    for i in range(240):
      keep, _ = apply_gpus(workers, quanta, c.s, 1, i * dt)
      [h] = keep
      workers[h] = replace(workers[h], gpu_ms=workers[h].gpu_ms + dt * 1000)
      c.serve(c.s.runs(h)[0], dt * 1000)
      c.settle()
    self.assertAlmostEqual(c.service["alice"] / (c.service["alice"] + c.service["bob"]), 0.5, delta=0.06)
    self.assertAlmostEqual(workers[t1].gpu_ms / (workers[t1].gpu_ms + workers[s1].gpu_ms), 0.5, delta=0.08)

  def test_gang_hold_prevents_leapfrog_starvation_and_achieves_waterfill_shares(self):
    c = Cluster()
    c.node("n", 2)
    p1 = c.run("a", backlog=1)
    p2 = c.run("b", backlog=1)
    p3 = c.run("d", rank=4, backlog=1)
    gang = c.run("wide", gpus=2, backlog=1)
    quanta = dict.fromkeys((p1, p2, p3, gang), 2.0)
    # Stagger `p1` (overtime at t=2) and `p2` (active until t=3) while `gang` has highest lag and `p3` waits.
    workers = {
      p1: Worker(current=(0,), hold_until=2.0),
      p2: Worker(current=(1,), hold_until=3.0),
      p3: Worker(hold_until=2.0),
      gang: Worker(gpus=2, hold_until=2.0),
    }
    c.s.lag = {"wide": 5000.0, "d": 1000.0, "a": 0.0, "b": 0.0}
    keep, wake = apply_gpus(workers, quanta, c.s, 2, 2.2)
    # `p3` is blocked from leapfrogging onto GPU 0; overtime `p1` keeps computing without clock reset until `p2` ends at 3.0.
    self.assertEqual((keep, wake, workers[p1].hold_until), ({p1: (0,), p2: (1,)}, 3.0, 2.0))
    keep, _ = apply_gpus(workers, quanta, c.s, 2, 3.0)
    self.assertEqual(keep, {gang: (0, 1)})

    # Steady-state 2 singles + 1 gang on 2 GPUs -> exact 1/3 `gpu_ms` each (waterfill C=2, d=(1,1,2)).
    c = Cluster()
    c.node("n", 2)
    p1, p2, gang = c.run("a", backlog=1), c.run("b", backlog=1), c.run("wide", gpus=2, backlog=1)
    quanta = dict.fromkeys((p1, p2, gang), 1.0)
    workers = {p: Worker(gpus=c.s.procs[p].spec.gpus, hold_until=1.0) for p in quanta}
    both_ticks = gang_ticks = 0
    for i in range(180):
      keep, _ = apply_gpus(workers, quanta, c.s, 2, i * 0.5)
      check_gpu_invariants(self, keep, workers, 2)
      gang_ticks += keep == {gang: (0, 1)}
      both_ticks += p1 in keep and p2 in keep
      for h, gs in keep.items():
        ms = 500.0 * len(gs)
        workers[h] = replace(workers[h], gpu_ms=workers[h].gpu_ms + ms)
        c.serve(c.s.runs(h)[0], ms)
      c.settle()
    self.assertGreater(gang_ticks, 25)
    self.assertGreater(both_ticks, 25)
    total = sum(c.service.values())
    for acct in ("a", "b", "wide"):
      self.assertAlmostEqual(c.service[acct] / total, 1 / 3, delta=0.05)

  def test_random_local_quantum_schedules_never_overlap_or_violate_exclusives(self):
    for seed in range(100):
      rng, s, now = random.Random(seed), State(), 0.0
      total = rng.choice([1, 2, 4])
      workers, quanta = {}, {}
      for i in range(6):
        w = rng.choice([1, 1, 2]) if total >= 2 else 1
        quanta[f"p{i}"] = q = rng.choice([0.5, 2.0])
        workers[f"p{i}"] = Worker(gpus=w, exclusive=rng.random() < 0.15, hold_until=q)
      for _ in range(60):
        now += rng.uniform(0.0, 1.5)
        for p, w in list(workers.items()):
          demand = Demand.READY if rng.random() < 0.7 else Demand.IDLE
          ms = w.gpu_ms + (rng.uniform(10.0, 500.0) if w.current else 0.0)
          workers[p] = replace(w, demand=demand, gpu_ms=ms)
        keep, wake = apply_gpus(workers, quanta, s, total, now)
        check_gpu_invariants(self, keep, workers, total)
        if wake is not None:
          self.assertGreater(wake, now)


class TestMultiNodeFairness(unittest.TestCase):
  def test_waterfill_equal_capped_and_weighted(self):
    self.assertEqual(waterfill(2, {"a": 1, "b": 1, "c": 1}), {"a": 2 / 3, "b": 2 / 3, "c": 2 / 3})
    self.assertEqual(waterfill(4, {"a": 1, "b": 4, "c": 4}), {"a": 1.0, "b": 1.5, "c": 1.5})
    self.assertEqual(waterfill(4, {"a": 4, "b": 4}, {"a": 3.0, "b": 1.0}), {"a": 3.0, "b": 1.0})
    self.assertEqual(waterfill(8, {"a": 1, "b": 2}), {"a": 1.0, "b": 2.0})

  def test_update_lag_conserves_service_blocks_flicker_and_handles_async_deltas(self):
    lag, seen = update_lag({}, {}, {"a": 300.0, "b": 0.0}, 1.0, {"a": 1.0, "b": 1.0})
    self.assertAlmostEqual(lag["a"], -150.0)
    self.assertAlmostEqual(lag["b"], 150.0)

    lag, seen = update_lag({}, {}, {"a": 200.0, "b": 0.0}, 1.0, {"b": 1.0})
    self.assertAlmostEqual(lag["a"], -100.0)
    self.assertAlmostEqual(lag["b"], 100.0)

    for _ in range(5):
      lag, seen = update_lag(lag, seen, {"a": 200.0, "b": 0.0}, 1.0, {"a": 1.0, "b": 1.0})
      lag, seen = update_lag(lag, seen, {"a": 200.0, "b": 0.0}, 1.0, {"b": 1.0})
    self.assertAlmostEqual(lag["a"], -100.0)

    lag2, _ = update_lag(lag, seen, {"a": 200.0, "b": 0.0}, 1.0, {})
    self.assertNotIn("b", lag2)
    lag, seen = update_lag(lag, seen, {"a": 200.0, "b": 200.0}, 1.0, {"b": 1.0})
    self.assertNotIn("a", lag)

    lag, seen = update_lag({}, {}, {"a": 100.0, "b": 0.0}, 1.0, {"a": 1.0, "b": 1.0})
    lag, seen = update_lag(lag, seen, {"a": 250.0, "b": 50.0}, 1.0, {"a": 1.0, "b": 1.0})
    self.assertAlmostEqual(lag["a"], -100.0)
    self.assertAlmostEqual(lag["b"], 100.0)
    # `finish_cmd` deletes `b`'s last queued entry (`demand["b"] == 0`) in the same tx that increments `service["b"]`:
    # positive catch-up lag is conserved while `deltas["b"] > 0`, and clears only once `b` is truly idle (`deltas == 0`).
    lag, seen = update_lag(lag, seen, {"a": 250.0, "b": 80.0}, 1.0, {"a": 1.0})
    self.assertAlmostEqual(lag["a"], -85.0)
    self.assertAlmostEqual(lag["b"], 85.0)

  def simulate(self, c: Cluster, steps: int, dt: float = 0.5, quantum: float = 1.0) -> int:
    """Co-simulate per-node `plan_gpus` (L1) and cluster `scheduler.step` (L2)."""
    workers: dict[str, tuple[str, Worker]] = {}
    moves = 0
    for i in range(steps):
      now = i * dt
      for n in c.nodes:
        local = {}
        for p, pr in c.s.procs_on(n):
          if c.s.runs(p):
            w = (
              workers[p][1]
              if p in workers and workers[p][0] == n
              else Worker(gpus=pr.spec.gpus, exclusive=pr.spec.exclusive, hold_until=quantum)
            )
            local[p] = replace(w, demand=Demand.READY if c.s.busy(p) else Demand.IDLE)
        keep, _ = apply_gpus(local, dict.fromkeys(local, quantum), c.s, c.nodes[n]["gpus"], now)
        for p, w in local.items():
          if keep.get(p) and w.demand != Demand.IDLE:
            ms = dt * 1000.0 * len(keep[p])
            w = replace(w, gpu_ms=w.gpu_ms + ms)
            c.serve(c.s.runs(p)[0], ms)
          workers[p] = (n, w)
      moves += sum(1 for ev, body in c.settle() if ev == Ev.PLACED and body.get("rule") == "place.rebalance")
    return moves

  def test_four_on_two_weighted_and_sybil_settle_without_cross_node_thrash(self):
    # Skewed 3-vs-1 equal accounts: migrates once to 2-vs-2, then zero further migrations forever.
    c = Cluster()
    c.node("a", 1)
    c.node("b", 1)
    procs = [c.run(f"r{i}", rank=4, base=f"m{i}", backlog=1) for i in range(4)]
    for i, n in enumerate(("a", "a", "a", "b")):
      c.s.procs[procs[i]].node = n
    moves = self.simulate(c, 240, dt=0.5, quantum=1.0)
    self.assertEqual(moves, 1)
    self.assertEqual(sorted(sum(p.node == n for p in c.s.procs.values()) for n in ("a", "b")), [2, 2])
    total = sum(c.service.values())
    for i in range(4):
      self.assertAlmostEqual(c.service[f"r{i}"] / total, 0.25, delta=0.05)

    # Weighted 3-on-2 where `{a(w=1), b(w=1)}` on node `a` and `{c(w=2)}` on node `b` already match waterfill: 0 moves.
    c = Cluster()
    c.node("a", 1)
    c.node("b", 1)
    c.s.weights = {"r0": 1.0, "r1": 1.0, "r2": 2.0}
    p0, p1, p2 = (c.run(f"r{i}", rank=4, base=f"m{i}", backlog=1) for i in range(3))
    c.s.procs[p0].node, c.s.procs[p1].node, c.s.procs[p2].node = "a", "a", "b"
    self.assertEqual(self.simulate(c, 200, dt=0.5, quantum=1.0), 0)

    # Sybil: Alice puts 3 busy trainers on node `a` while Bob has 1 on node `b` -> 0 migrations, exact 50/50 split.
    c = Cluster()
    c.node("a", 1)
    c.node("b", 1)
    sybils = [c.run(f"s{i}", rank=4, base=f"m{i}", account="alice", backlog=1) for i in range(3)]
    solo = c.run("b0", rank=4, base="mb", account="bob", backlog=1)
    for p in sybils:
      c.s.procs[p].node = "a"
    c.s.procs[solo].node = "b"
    self.assertEqual(self.simulate(c, 200, dt=0.5, quantum=1.0), 0)
    self.assertAlmostEqual(c.service["alice"] / (c.service["alice"] + c.service["bob"]), 0.5, delta=0.02)

  def test_three_busy_trainers_on_two_nodes_rotate_to_one_third_each(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 1)
    for i in range(3):
      c.run(f"r{i}", rank=4, base=f"m{i}", backlog=1)
    with mock.patch.object(scheduler, "REBALANCE_MS", 5_000):
      moves = self.simulate(c, 360, dt=0.5, quantum=1.0)
    total = sum(c.service.values())
    for i in range(3):
      self.assertAlmostEqual(c.service[f"r{i}"] / total, 1 / 3, delta=0.04)
    self.assertTrue(4 <= moves <= 30, moves)


class TestMultiNodeExecution(BusTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.bucket = f"file://{tempfile.mkdtemp()}"
    self.nodes: list[TestNode] = []
    self.tasks: list[asyncio.Task] = []

  async def asyncTearDown(self):
    for task in self.tasks:
      task.cancel()
    await asyncio.gather(*self.tasks, return_exceptions=True)
    for node in self.nodes:
      await node.aclose()
    await super().asyncTearDown()

  async def add_node(self, name: str, gpus: int = 1, engine=FakeEngine) -> TestNode:
    store = Store(self.bucket, cache=tempfile.mkdtemp())
    node = TestNode(self.bus, store=store, gpus=gpus, node=name, engine=engine)
    self.nodes.append(node)
    await node.tick()

    async def loop():
      while True:
        await node.tick()
        await asyncio.sleep(0.01)

    self.tasks.append(asyncio.create_task(loop()))
    return node

  def start_scheduler(self) -> None:
    self.start_maintenance(0.01)

  async def submit(self, run: str, op: Op, body: bytes = b"", cost: int = 100) -> str:
    req = await self.bus.submit_command(run, op, body, cost=cost)
    assert req is not None
    return req

  async def run_cmd(self, run: str, op: Op, body: bytes = b"", cost: int = 100) -> bytes:
    reply = await self.bus.wait_for_reply(await self.submit(run, op, body, cost), 10)
    self.assertTrue(reply.ok, reply.body)
    return reply.body

  async def create(self, run: str, rank: int = 0, lora: dict | None = None, **kw) -> None:
    cfg = {} if lora is None else lora
    await self.bus.create_run(Run(run, "tiny", rank, lora=cfg, **kw))
    await self.submit(run, Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": rank, "lora": cfg}).encode())

  async def until(self, pred, timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
      while not await pred():
        await asyncio.sleep(0.01)

  def live_params(self, run: str) -> dict[str, torch.Tensor]:
    for node in self.nodes:
      for proc, h in node.workers.items():
        if run in h.loaded and proc in node.processes:
          return node.processes[proc].trainer.model.params[run]
    raise AssertionError(f"no live owner found for {run}")

  async def test_the_plan_is_written_only_when_it_changes(self):
    from tinyrl.kv import Table
    from tinyrl.types import NodeReport

    await self.bus.exchange(NodeReport("n", "bn", 1))
    await self.bus.create_run(Run("r", "tiny", 4, lora=LORA, account="acct"))
    writes, orig_set = 0, Table.__setitem__

    def count_writes(tbl, k, v):
      nonlocal writes
      if tbl is self.bus.meta and k == "plan":
        writes += 1
      orig_set(tbl, k, v)

    with mock.patch.object(Table, "__setitem__", count_writes):
      await self.bus.maintain()  # the run gets a process, and the process a node
      await self.bus.maintain()  # nothing changed
      self.bus.service["acct"] = self.bus.service.get("acct", 0.0) + 5.0
      await self.bus.maintain()  # account service moved
      await self.bus.create_run(Run("r2", "tiny", 4, lora=LORA, account="acct"))
      await self.bus.maintain()  # a run joined the process
      self.bus.nodes.pop("n", None)
      await self.bus.exchange(NodeReport("m", "bm", 1))  # the process moved to m
    self.assertEqual(writes, 4)

  async def test_new_node_rebalances_mid_accumulation_with_exact_model_parameters(self):
    opt = json.dumps({"adam_params": {"learning_rate": 1e-2}}).encode()
    await self.add_node("node-a", gpus=1)
    self.start_scheduler()
    r1, r2, ctrl = new_id(), new_id(), new_id()
    for r in (r1, r2):
      await self.create(r)
      await self.run_cmd(r, Op.FORWARD_BACKWARD, proto_body(r, 10))
      await self.run_cmd(r, Op.OPTIM_STEP, opt)
      await self.run_cmd(r, Op.FORWARD_BACKWARD, proto_body(r, 10))

    await self.add_node("node-b", gpus=1)
    reqs = [
      (await self.submit(r, Op.FORWARD_BACKWARD, proto_body(r, 8)), await self.submit(r, Op.OPTIM_STEP, opt))
      for r in (r1, r2)
    ]
    for a, b in reqs:
      self.assertTrue((await self.bus.wait_for_reply(a, 10)).ok)
      self.assertTrue((await self.bus.wait_for_reply(b, 10)).ok)

    state = self.bus.state()
    self.assertEqual({state.procs[state.home[r]].node for r in (r1, r2)}, {"node-a", "node-b"})

    ref = Model(tiny)
    ref.create_run(ctrl, "tiny", 0, {})
    for n, step_after in ((10, True), (10, False), (8, True)):
      ref.forward_backward(ctrl, [counting(n, 0), counting(n, 40)], "cross_entropy", {}, False)
      if step_after:
        ref.optim_step(ctrl, {"learning_rate": 1e-2})
    expected, _ = ref.training_state(ctrl)
    for r in (r1, r2):
      uri = f"tinker://{r}/weights/after-rebalance"
      await self.run_cmd(r, Op.SAVE_WEIGHTS, json.dumps({"uri": uri}).encode())
      version = await self.bus.checkpoint_version(uri)
      actual, _ = await asyncio.to_thread(self.nodes[0].store.get, version)
      torch.testing.assert_close(actual, expected)

  async def test_ready_trainer_runs_while_sampler_stays_backlogged_on_one_gpu(self):
    sampling, release = asyncio.Event(), asyncio.Event()

    class BusySampler(FakeEngine):
      async def sample(self, job, name):
        sampling.set()
        await release.wait()
        await asyncio.sleep(0.01)
        return await super().sample(job, name)

    with (
      mock.patch("tinyrl.bus.QUANTUM_MS", 15),
      mock.patch("tinyrl.bus.MAX_SEQS", 2),
    ):
      await self.add_node("shared", gpus=1, engine=BusySampler)
      self.start_scheduler()
      await self.create("train", account="team")
      await self.run_cmd("train", Op.FORWARD_BACKWARD, proto_body("train", 6))
      await self.bus.create_run(Run("sample", "tiny", 4, role=Role.SAMPLER, account="team"))
      for _ in range(64):
        await self.submit("sample", Op.SAMPLE, json.dumps(JOB).encode())
      try:
        await asyncio.wait_for(sampling.wait(), 5)
        requests = [await self.submit("train", Op.OPTIM_STEP, b'{"adam_params":{}}') for _ in range(3)]
      finally:
        release.set()
      for request in requests:
        self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
      self.assertGreater((await self.bus.queued_and_in_flight("sample"))["queued"], 0)

  async def test_shared_trainer_splits_gpu_time_by_account_not_run_count(self):
    node = await self.add_node("single-gpu")
    accounts = {new_id(): account for account in ("alice", "alice", "alice", "bob")}
    observed = {account: 0.0 for account in accounts.values()}
    completed = []
    original = InlineProcess.call

    async def timed_call(process, method, *args):
      if method != "execute" or args[0].op != Op.OPTIM_STEP:
        return await original(process, method, *args)
      run = args[0].run
      start = time.monotonic()
      await asyncio.sleep(0.012 if accounts[run] == "alice" else 0.004)
      result = await original(process, method, *args)
      observed[accounts[run]] += time.monotonic() - start
      completed.append(run)
      return result

    self.start_scheduler()
    for run, account in accounts.items():
      await self.bus.create_run(Run(run, "tiny", 4, lora=LORA, account=account))
      await self.run_cmd(run, Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode())
    with mock.patch.object(InlineProcess, "call", timed_call):
      for run in accounts:
        for _ in range(100):
          await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
      async with asyncio.timeout(30):
        while len(completed) < 80:
          await asyncio.sleep(0.005)

    self.assertEqual(len(node.processes), 1)
    self.assertEqual(set(completed), set(accounts))
    self.assertAlmostEqual(observed["alice"] / sum(observed.values()), 0.5, delta=0.2)

  async def test_lora_runs_split_onto_new_node_and_merge_back_with_exact_weights(self):
    opt = json.dumps({"adam_params": {"learning_rate": 1e-2}}).encode()
    await self.add_node("node-a", gpus=1)
    self.start_scheduler()
    r1, r2 = new_id(), new_id()
    for r in (r1, r2):
      await self.create(r, rank=4, lora=LORA)
      await self.run_cmd(r, Op.FORWARD_BACKWARD, proto_body(r, 10))
      await self.run_cmd(r, Op.OPTIM_STEP, opt)
    s0 = self.bus.state()
    self.assertEqual(s0.home[r1], s0.home[r2])

    await self.add_node("node-b", gpus=1)
    futs = [
      (await self.submit(r, Op.FORWARD_BACKWARD, proto_body(r, 9)), await self.submit(r, Op.OPTIM_STEP, opt))
      for r in (r1, r2)
    ]
    await self.bus.maintain()
    s1 = self.bus.state()
    self.assertNotEqual(s1.home[r1], s1.home[r2])
    self.assertEqual({s1.procs[s1.home[r]].node for r in (r1, r2)}, {"node-a", "node-b"})
    for a, b in futs:
      self.assertTrue((await self.bus.wait_for_reply(a, 10)).ok)
      self.assertTrue((await self.bus.wait_for_reply(b, 10)).ok)

    excl = new_id()
    await self.create(excl, exclusive=True)
    await self.run_cmd(excl, Op.FORWARD_BACKWARD, proto_body(excl, 6))

    async def merged() -> bool:
      st = self.bus.state()
      return st.home.get(r1) == st.home.get(r2)

    await self.until(merged)
    for r in (r1, r2):
      await self.run_cmd(r, Op.FORWARD_BACKWARD, proto_body(r, 11))
      await self.run_cmd(r, Op.OPTIM_STEP, opt)

    ref = Model(tiny)
    for r in (r1, r2):
      ref.create_run(r, "tiny", 4, LORA)
      for n in (10, 9, 11):
        ref.forward_backward(r, [counting(n, 0), counting(n, 40)], "cross_entropy", {}, False)
        ref.optim_step(r, {"learning_rate": 1e-2})
      torch.testing.assert_close(self.live_params(r), ref.params[r])

  async def test_stateless_sampler_scales_out_across_nodes_recovers_crash_and_scales_in(self):
    served_by: list[int] = []

    class SlowEngine(FakeEngine):
      async def sample(self, job: dict, name: str | None):
        served_by.append(id(self))
        await asyncio.sleep(0.05)
        return await super().sample(job, name)

    await self.add_node("node-a", gpus=1, engine=SlowEngine)
    await self.add_node("node-b", gpus=1, engine=SlowEngine)
    self.start_scheduler()

    sid = new_id()
    await self.bus.create_run(Run(sid, "tiny", 4, role=Role.SAMPLER, account="team-s"))
    total_cmds = 12
    payload = json.dumps(JOB).encode()
    with mock.patch("tinyrl.bus.MAX_SEQS", 2), mock.patch("tinyrl.control.scheduler.MAX_SEQS", 4):
      reqs = [await self.submit(sid, Op.SAMPLE, payload, cost=50) for _ in range(total_cmds)]

      async def scaled_out() -> bool:
        st = self.bus.state()
        return len(st.procs_of(sid)) >= 2 and len(set(served_by)) >= 2

      await self.until(scaled_out)
      for req in reqs:
        reply = await self.bus.wait_for_reply(req, 10)
        self.assertTrue(reply.ok)
        self.assertEqual(len(wire.sample_json(reply.body)["sequences"]), 1)

    self.assertEqual((await self.bus.get_run(sid))["tokens"], total_cmds * 50)
    svc_before = self.bus.service["team-s"]
    self.assertGreaterEqual(svc_before, total_cmds * 20.0)

    async def scaled_in() -> bool:
      return len(self.bus.state().procs_of(sid)) == 1

    await self.until(scaled_in)

    # Simulate the active sampler's node dying mid-sample -> surviving node reclaims and answers.
    st = self.bus.state()
    active_node = st.procs[st.procs_of(sid)[0]].node
    req = await self.bus.submit_command(sid, Op.SAMPLE, payload, cost=50)
    self.bus.nodes[active_node].until = 0.0
    self.assertTrue((await self.bus.wait_for_reply(req, 5)).ok)
    self.assertEqual((await self.bus.get_run(sid))["tokens"], (total_cmds + 1) * 50)
    self.assertGreater(self.bus.service["team-s"], svc_before)

  async def test_controller_enforces_gpu_exclusivity_oversized_rejection_and_single_writer(self):
    node_a = await self.add_node("node-a", gpus=1)
    node_b = await self.add_node("node-b", gpus=1)
    self.start_scheduler()
    r1, r2, gang = new_id(), new_id(), new_id()
    await self.create(r1)
    await self.create(r2)
    await self.create(gang, gpus=2)

    fb, opt = proto_body(r1, 10), json.dumps({"adam_params": {"learning_rate": 1e-2}}).encode()
    futs = []
    for _ in range(3):
      futs.append(await self.submit(r1, Op.FORWARD_BACKWARD, fb))
      futs.append(await self.submit(r1, Op.OPTIM_STEP, opt))

    for req in futs:
      async with asyncio.timeout(10):
        while (await self.bus.wait_for_reply(req, 0)) is None:
          for node in (node_a, node_b):
            flat = [g for h in node.workers.values() for g in h.resident]
            self.assertEqual(len(flat), len(set(flat)))  # GpuExclusive
            self.assertTrue(all(h.spec.gpus <= node.gpus for h in node.workers.values()))  # oversized gang rejected
          holders = [f"{n.id}:{p}" for n in (node_a, node_b) for p, h in n.workers.items() if r1 in h.loaded]
          self.assertLessEqual(len(holders), 1)  # SingleWriter
          await asyncio.sleep(0.005)
      self.assertTrue((await self.bus.wait_for_reply(req, 5)).ok)

    ref, ctrl = Model(tiny), new_id()
    ref.create_run(ctrl, "tiny", 0, {})
    for _ in range(3):
      ref.forward_backward(ctrl, [counting(10, 0), counting(10, 40)], "cross_entropy", {}, False)
      ref.optim_step(ctrl, {"learning_rate": 1e-2})
    torch.testing.assert_close(self.live_params(r1), ref.params[ctrl])

  async def test_local_quantum_shares_one_gpu_and_overtime_hands_off_immediately(self):
    await self.add_node("node-a", gpus=1)
    self.start_scheduler()
    a, b = new_id(), new_id()
    before = time.monotonic()
    await self.create(a, account="alice")
    await self.run_cmd(a, Op.FORWARD_BACKWARD, proto_body(a, 6))
    pa = self.bus.state().home[a]
    w_a = self.bus.nodes["node-a"].workers[pa]
    # Push `a` past its minimum quantum (into OVERTIME) without touching `b` yet.
    self.assertGreaterEqual(w_a.hold_until, before + QUANTUM_MS / 1000)
    w_a.hold_until = expired = time.monotonic() - 1.0
    # Running alone after the deadline keeps the original turn.
    await self.run_cmd(a, Op.FORWARD_BACKWARD, proto_body(a, 6))
    self.assertEqual(w_a.hold_until, expired)
    # Late-joiner `b` arrives while `a` is busy in OVERTIME -> preempts without waiting a full 2s quantum.
    await self.create(b, account="bob")
    t0 = time.monotonic()
    fa = [await self.submit(a, Op.FORWARD_BACKWARD, proto_body(a, 6)) for _ in range(2)]
    fb = await self.submit(b, Op.FORWARD_BACKWARD, proto_body(b, 6))
    self.assertTrue((await self.bus.wait_for_reply(fb, 5)).ok)
    self.assertLess(time.monotonic() - t0, 1.5)
    for req in fa:
      self.assertTrue((await self.bus.wait_for_reply(req, 5)).ok)
    svc = self.bus.service
    self.assertGreater(svc["alice"], 0.0)
    self.assertGreater(svc["bob"], 0.0)


if __name__ == "__main__":
  unittest.main()
