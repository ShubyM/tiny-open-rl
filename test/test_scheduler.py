import asyncio
import json
import random
import tempfile
import types
import unittest
from unittest import mock

import torch

from test.execution import TestNode
from test.helpers import FakeEngine, RedisTest, new_id, tiny
from test.test_api import proto_body
from test.test_sampler import JOB
from test.test_trainer import LORA, counting
from tinyrl.api import wire
from tinyrl.bus import Bus, Cmd, Ev, Op, Role, Run, connect
from tinyrl.control import scheduler
from tinyrl.control.node_daemon import NodeDaemon
from tinyrl.control.scheduler import (
  SCALE_BATCH,
  TURN,
  Proc,
  Spec,
  State,
  from_state,
  select_runs,
  step,
  to_plan,
  update_lag,
  waterfill,
)
from tinyrl.storage.store import Store
from tinyrl.workers.model import Model

GIB = 2**30


class Cluster:
  """In-memory mirror of `bus.read_state()` for stepping the pure scheduler."""

  def __init__(self):
    self.nodes, self.runs, self.queues, self.service, self.s = {}, {}, {}, {}, State()

  def raw(self) -> dict:
    return {
      "nodes": self.nodes,
      "runs": self.runs,
      "queues": {r: q for r, q in self.queues.items() if q > 0},
      "service": dict(self.service),
      "plan": to_plan(self.s),
    }

  def settle(self) -> list[tuple[Ev, dict]]:
    self.s, changes = step(from_state(self.raw()))
    return changes

  def node(self, name: str, gpus: int, memory: int = 80 * GIB, preparing: tuple[str, ...] = ()) -> None:
    self.nodes[name] = {"gpus": gpus, "memory": memory, "preparing": list(preparing)}
    self.settle()

  def lose(self, node: str) -> None:
    self.nodes.pop(node, None)

  def run(self, name: str, *, backlog: int = 0, owner: str | None = None, tokens: int = 0, **body) -> str:
    self.runs[name] = {"record": {"role": "trainer", "base": "a", "rank": 0} | body, "owner": owner, "tokens": tokens}
    self.queues[name] = backlog
    self.settle()
    return self.s.home[name][0]

  def queue(self, run: str, n: int) -> None:
    self.queues[run] = n

  def serve(self, run: str, amount: int) -> None:
    self.runs[run]["tokens"] += amount
    acct = self.s.account(run)
    self.service[acct] = self.service.get(acct, 0.0) + amount

  def delete(self, run: str) -> None:
    self.runs.pop(run, None)
    self.queues.pop(run, None)

  def active(self, node: str | None = None) -> dict[str, tuple[int, ...]]:
    return {n: p.gpus for n, p in self.s.procs.items() if p.gpus and (node is None or p.node == node)}


def check_state_invariants(test: unittest.TestCase, s: State) -> None:
  for node, (total, _) in s.nodes.items():
    seen: list[int] = []
    for p in s.procs.values():
      if p.node == node and p.gpus:
        test.assertEqual(len(p.gpus), p.spec.gpus)
        test.assertEqual(len(set(p.gpus)), p.spec.gpus)
        test.assertTrue(all(0 <= g < total for g in p.gpus))
        seen.extend(p.gpus)
    test.assertEqual(len(seen), len(set(seen)))
  for p in s.procs.values():
    test.assertTrue(p.node is None or p.node in s.nodes)
    if p.node is None:
      test.assertEqual(p.gpus, ())
  for r, ps in s.home.items():
    test.assertIn(r, s.jobs)
    if s.jobs[r].spec.role == Role.TRAINER:
      test.assertLessEqual(len(ps), 1)
    for proc in ps:
      test.assertIn(proc, s.procs)


class TestCluster(unittest.TestCase):
  def test_runs_are_assigned_and_empty_procs_wait_for_live_owner_detach(self):
    c = Cluster()
    c.node("n", 8)
    a = c.run("a", rank=8)
    self.assertEqual(c.run("b", rank=8), a)
    self.assertEqual((c.s.procs[a].node, c.s.procs[a].gpus), ("n", (0,)))
    c.delete("a")
    self.assertEqual(c.settle(), [])
    # While a live owner still holds another run on `a`, GC waits until `owner` clears.
    c.run("c")
    c.runs["c"]["owner"] = f"{a}@n-1:1"
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
    self.assertEqual([c.s.procs[p].gpus for p in procs], [(0,), (0,), (1,)])

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
    x = c.run("x", exclusive=True)
    other = c.run("r", rank=4)
    self.assertEqual([c.s.procs[p].node for p in (x, other)], ["a", "b"])
    c.s.procs[other].node, c.s.procs[other].gpus = "a", (1,)
    sampler = c.run("sample", role="sampler", train_run="x", exclusive=True)
    self.assertEqual((c.s.procs[sampler].node, c.s.procs[sampler].gpus), ("b", (0,)))

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

  def test_lost_node_procs_are_reassigned_and_claim_free_gpus(self):
    c = Cluster()
    c.node("a", 1)
    p = c.run("r")
    c.node("b", 1)
    c.lose("a")
    changes = c.settle()
    self.assertEqual(changes[0], (Ev.UNPLACED, {"proc": p, "rule": "place.node_lost"}))
    self.assertEqual((c.s.procs[p].node, c.s.procs[p].gpus), ("b", (0,)))

  def test_multi_gpu_proc_waits_for_a_node_that_has_enough_gpus(self):
    c = Cluster()
    c.node("a", 1)
    p = c.run("wide", gpus=2)
    self.assertEqual((c.s.procs[p].node, c.s.procs[p].gpus), (None, ()))
    c.node("b", 4)
    self.assertEqual((c.s.procs[p].node, c.s.procs[p].gpus), ("b", (0, 1)))

  def test_lora_runs_share_when_idle_split_when_busy_and_merge_when_squeezed(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 1)
    p1 = c.run("r1", rank=8)
    self.assertEqual(c.run("r2", rank=8), p1)
    others = [c.run("r3", rank=8, base="b"), c.run("f"), c.run("x", rank=8, exclusive=True)]
    self.assertEqual(len({p1, *others}), 4)

    # Fresh 2-node cluster: two busy LoRA runs split across a and b, then merge back when b goes exclusive.
    c = Cluster()
    c.node("a", 1)
    p1 = c.run("r1", rank=8, backlog=1)
    self.assertEqual(c.run("r2", rank=8, backlog=1), p1)
    c.node("b", 1)
    p2 = c.s.home["r2"][0]
    self.assertNotEqual(p1, p2)
    self.assertEqual({c.s.procs[p1].node, c.s.procs[p2].node}, {"a", "b"})
    c.run("excl", exclusive=True, backlog=1)
    c.settle()
    self.assertEqual(c.s.home["r1"], c.s.home["r2"])
    check_state_invariants(self, c.s)

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

  def test_exclusive_procs_stay_alone_and_displace_non_exclusive_holders(self):
    c = Cluster()
    c.node("n", 2)
    a, b = c.run("a", backlog=1), c.run("b", backlog=1)
    self.assertEqual([c.s.procs[a].gpus, c.s.procs[b].gpus], [(0,), (1,)])
    x = c.run("x", exclusive=True)
    self.assertEqual(len(c.s.procs[x].gpus), 1)
    check_state_invariants(self, c.s)
    # Neither busy non-exclusive proc can ever take x's GPU, even with huge lag.
    c.s.lag["a"] = c.s.lag["b"] = 100 * TURN
    c.settle()
    self.assertEqual(len(c.s.procs[x].gpus), 1)
    check_state_invariants(self, c.s)

  def test_multi_gpu_sampler_takes_free_pair_and_preserves_exclusivity(self):
    c = Cluster()
    c.node("n", 4)
    wide = c.run("wide", gpus=2, backlog=1)
    single = c.run("single", backlog=1)
    sample = c.run("sample", role="sampler", rank=4, gpus=2, backlog=1)
    self.assertEqual(c.s.procs[wide].gpus, (0, 1))
    self.assertEqual(c.s.procs[single].gpus, (2,))
    self.assertEqual(c.s.procs[sample].gpus, ())
    # As soon as `single` finishes its backlog, `sample` immediately claims `(2, 3)` without waiting for a turn.
    c.queue("single", 0)
    c.settle()
    self.assertEqual([c.s.procs[n].gpus for n in (wide, single, sample)], [(0, 1), (), (2, 3)])
    check_state_invariants(self, c.s)

  def test_random_clusters_settle_round_trip_and_never_overlap_gpus(self):
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
            c.serve(r, rng.choice([TURN // 2, TURN, 4 * TURN]))
        else:
          c.delete(name)
        c.settle()
        check_state_invariants(self, c.s)
        self.assertEqual(from_state(c.raw()), c.s)
        # At most one follow-up step if phase-4 preemption enabled a LoRA merge, then strictly idempotent.
        c.settle()
        check_state_invariants(self, c.s)
        again, changes = step(c.s)
        self.assertEqual(changes, [])
        self.assertEqual(again, c.s)


class TestNodeExclusivity(unittest.TestCase):
  def make_daemon(self, gpus: int) -> NodeDaemon:
    store = Store(tempfile.mkdtemp())
    node = NodeDaemon(types.SimpleNamespace(r=None), gpus=gpus, node="n", gpu_mib=0, store=store)  # type: ignore[arg-type]
    self.addCleanup(store.close)
    return node

  def test_ready_runs_share_service_even_when_commands_have_different_costs(self):
    heads = [Cmd(run, "1-0", Op.FORWARD_BACKWARD, b"") for run in ("slow", "fast")]
    service = {"slow": 0.0, "fast": 0.0}
    for _ in range(11):
      [run] = select_runs(heads, service, 1)
      service[run] += 10 if run == "slow" else 1
    self.assertEqual(service, {"slow": 10.0, "fast": 10.0})

  def test_claim_gpus_enforces_exclusivity_and_exact_gang_width(self):
    node = self.make_daemon(2)
    self.assertFalse(node.claim_gpus("wide@n-1", (0,), 2))  # partial width rejected
    self.assertFalse(node.claim_gpus("dup@n-1", (0, 0), 2))  # duplicate GPU rejected
    self.assertFalse(node.claim_gpus("oob@n-1", (2,), 1))  # out of range rejected
    self.assertTrue(node.claim_gpus("a@n-1", (0,), 1))
    self.assertFalse(node.claim_gpus("b@n-1", (0,), 1))
    self.assertFalse(node.claim_gpus("wide@n-1", (0, 1), 2))
    self.assertTrue(node.claim_gpus("c@n-1", (1,), 1))
    node.release_gpus("a@n-1")
    self.assertFalse(node.claim_gpus("wide@n-1", (0, 1), 2))
    node.release_gpus("c@n-1")
    self.assertTrue(node.claim_gpus("wide@n-1", (0, 1), 2))

  def test_random_claims_and_releases_never_overcommit_a_gpu(self):
    for seed in range(300):
      rng, node = random.Random(seed), self.make_daemon(4)
      specs = {}
      for i in range(10):
        w = rng.choice([1, 1, 2, 4])
        first = rng.randrange(5 - w)
        specs[f"p{i}@n-1"] = (tuple(range(first, first + w)), w)
      for _ in range(100):
        life = rng.choice(list(specs))
        if life in node.on and rng.random() < 0.5:
          node.release_gpus(life)
        else:
          gpus, w = specs[life]
          node.claim_gpus(life, gpus, w)
        flat = [g for gs in node.on.values() for g in gs]
        self.assertEqual(len(flat), len(set(flat)))


class TestFairnessAndHysteresis(unittest.TestCase):
  def test_waterfill_equal_capped_and_weighted(self):
    self.assertEqual(waterfill(2, {"a": 1, "b": 1, "c": 1}), {"a": 2 / 3, "b": 2 / 3, "c": 2 / 3})
    self.assertEqual(waterfill(4, {"a": 1, "b": 4, "c": 4}), {"a": 1.0, "b": 1.5, "c": 1.5})
    self.assertEqual(waterfill(4, {"a": 4, "b": 4}, {"a": 3.0, "b": 1.0}), {"a": 3.0, "b": 1.0})
    self.assertEqual(waterfill(8, {"a": 1, "b": 2}), {"a": 1.0, "b": 2.0})

  def test_update_lag_conserves_service_blocks_flicker_and_handles_async_deltas(self):
    # Active conservation: sum of lags across active accounts stays zero.
    lag, seen = update_lag({}, {}, {"a": 300.0, "b": 0.0}, 1.0, {"a": 1.0, "b": 1.0})
    self.assertAlmostEqual(lag["a"], -150.0)
    self.assertAlmostEqual(lag["b"], 150.0)

    # Sleep-before-read: `a` runs 200 then clears queue before tick; `a` still accrues debt (`Lag.tla` delta > 0).
    lag, seen = update_lag({}, {}, {"a": 200.0, "b": 0.0}, 1.0, {"b": 1.0})
    self.assertAlmostEqual(lag["a"], -100.0)
    self.assertAlmostEqual(lag["b"], 100.0)

    # Sleep/wake flicker: `a` cannot erase negative debt without `b` actually receiving service.
    for _ in range(5):
      lag, seen = update_lag(lag, seen, {"a": 200.0, "b": 0.0}, 1.0, {"a": 1.0, "b": 1.0})
      lag, seen = update_lag(lag, seen, {"a": 200.0, "b": 0.0}, 1.0, {"b": 1.0})
    self.assertAlmostEqual(lag["a"], -100.0)

    # Positive idle hoarding is zeroed when `b` goes idle; `a`'s debt decays to 0 once `b` runs 200.
    lag2, _ = update_lag(lag, seen, {"a": 200.0, "b": 0.0}, 1.0, {})
    self.assertNotIn("b", lag2)
    lag, seen = update_lag(lag, seen, {"a": 200.0, "b": 200.0}, 1.0, {"b": 1.0})
    self.assertNotIn("a", lag)

    # Concurrent worker execution between read and write is captured on the next tick via `seen`.
    lag, seen = update_lag({}, {}, {"a": 100.0, "b": 0.0}, 1.0, {"a": 1.0, "b": 1.0})
    lag, seen = update_lag(lag, seen, {"a": 250.0, "b": 50.0}, 1.0, {"a": 1.0, "b": 1.0})
    self.assertAlmostEqual(lag["a"], -100.0)
    self.assertAlmostEqual(lag["b"], 100.0)

  def test_idle_yield_is_instant_and_highest_lag_holder_never_swaps(self):
    c = Cluster()
    c.node("n", 1)
    p1 = c.run("a", backlog=1)
    p2 = c.run("b", backlog=1)
    self.assertEqual((c.s.procs[p1].gpus, c.s.procs[p2].gpus), ((0,), ()))
    # `a` runs less than a turn then goes idle: `b` gets the GPU immediately with 0 wait.
    c.serve("a", 10)
    c.queue("a", 0)
    c.settle()
    self.assertEqual((c.s.procs[p1].gpus, c.s.procs[p2].gpus), ((), (0,)))
    # If `b` already has higher lag than `a` (e.g. `b` is weighted 10x), `b` keeps holding without pointless swap.
    c.s.weights = {"a": 1.0, "b": 10.0}
    c.queue("a", 1)
    c.serve("b", TURN)
    c.settle()
    self.assertEqual(c.s.procs[p2].gpus, (0,))

  def test_account_grouping_shares_equally_between_rl_pair_and_solo_trainer(self):
    c = Cluster()
    c.node("n", 1)
    c.run("t1", session="alice", backlog=1)
    c.run("s1", role="sampler", train_run="t1", backlog=1)
    c.run("t2", session="bob", backlog=1)
    chunk = TURN // 2
    for _ in range(160):
      [holder] = c.active("n")
      r = c.s.runs(holder)[0]
      c.serve(r, chunk)
      c.settle()
    alice, bob = c.service["alice"], c.service["bob"]
    self.assertAlmostEqual(alice / (alice + bob), 0.5, delta=0.06)
    self.assertGreater(c.runs["t1"]["tokens"], 0)
    self.assertGreater(c.runs["s1"]["tokens"], 0)

  def test_switching_cost_amortization_gives_sampler_coarser_slices_and_equal_tokens(self):
    c = Cluster()
    c.node("n", 1)
    pt = c.run("train", backlog=1)
    ps = c.run("samp", role="sampler", backlog=1)
    turns: dict[str, int] = {pt: 0, ps: 0}
    last, chunk = None, TURN // 2
    for _ in range(160):
      [holder] = c.active("n")
      if holder != last:
        turns[holder] += 1
        last = holder
      c.serve(c.s.runs(holder)[0], chunk)
      c.settle()
    ratio = c.service["train"] / (c.service["train"] + c.service["samp"])
    self.assertAlmostEqual(ratio, 0.5, delta=0.05)
    # Sampler holds for >= 4*TURN once running, so total switches stay small and goodput stays high.
    self.assertGreaterEqual(c.runs["samp"]["tokens"] / turns[ps], 4 * TURN - chunk)
    self.assertLessEqual(turns[pt] + turns[ps], 36)

  def test_gang_preemption_and_spare_gpu_backfill_on_two_gpus(self):
    c = Cluster()
    c.node("n", 2)
    p1 = c.run("a", backlog=1)
    p2 = c.run("b", backlog=1)
    gang = c.run("wide", gpus=2, backlog=1)
    chunk = TURN // 2
    both_single_ticks = gang_ticks = 0
    for _ in range(120):
      check_state_invariants(self, c.s)
      act = c.active("n")
      if gang in act:
        self.assertEqual(act, {gang: (0, 1)})
        gang_ticks += 1
      elif p1 in act and p2 in act:
        both_single_ticks += 1
      for h, gs in act.items():
        c.serve(c.s.runs(h)[0], chunk * len(gs))
      c.settle()
    self.assertGreater(gang_ticks, 20)
    self.assertGreater(both_single_ticks, 20)
    total = sum(c.service.values())
    # Waterfill with capacity 2 and demands (1, 1, 2) gives equal rates (2/3, 2/3, 2/3) -> 1/3 share per account.
    for acct in ("a", "b", "wide"):
      self.assertAlmostEqual(c.service[acct] / total, 1 / 3, delta=0.05)

  def test_four_busy_trainers_on_two_nodes_settle_in_one_migration(self):
    c = Cluster()
    c.node("a", 1)
    procs = [c.run(f"r{i}", rank=4, base=f"m{i}", backlog=1) for i in range(3)]
    c.node("b", 1)
    procs.append(c.run("r3", rank=4, base="m3", backlog=1))
    # Force an initial 3-vs-1 imbalance on `a` vs `b`.
    for i, node, gpus in ((0, "a", (0,)), (1, "a", ()), (2, "a", ()), (3, "b", (0,))):
      p = c.s.procs[procs[i]]
      p.node, p.gpus, p.cost, p.at = node, gpus, float(TURN if gpus else 0), 0.0
    moves, chunk = 0, TURN // 2
    for _ in range(160):
      for h in c.active():
        c.serve(c.s.runs(h)[0], chunk)
      moves += sum(1 for ev, body in c.settle() if ev == Ev.PLACED and body.get("rule") == "place.rebalance")
    self.assertEqual(moves, 1)
    self.assertEqual(sorted(sum(1 for p in c.s.procs.values() if p.node == n) for n in ("a", "b")), [2, 2])
    total = sum(c.service.values())
    for i in range(4):
      self.assertAlmostEqual(c.service[f"r{i}"] / total, 0.25, delta=0.04)

  def test_three_busy_trainers_on_two_nodes_rotate_to_one_third_each(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 1)
    for i in range(3):
      c.run(f"r{i}", rank=4, base=f"m{i}", backlog=1)
    moves, chunk = 0, TURN // 2
    for _ in range(300):
      for h in c.active():
        c.serve(c.s.runs(h)[0], chunk)
      moves += sum(1 for ev, body in c.settle() if ev == Ev.PLACED and body.get("rule") == "place.rebalance")
    total = sum(c.service.values())
    for i in range(3):
      self.assertAlmostEqual(c.service[f"r{i}"] / total, 1 / 3, delta=0.04)
    # Cross-node migrations are damped by L2 switch_cost (4*TURN), never thrashing every tick.
    self.assertGreater(moves, 2)
    self.assertLess(moves, 50)


class TestMultiNodeExecution(RedisTest):
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
      await node.bus.r.aclose()
    await super().asyncTearDown()

  async def add_node(self, name: str, gpus: int = 1, engine=FakeEngine) -> TestNode:
    store = Store(self.bucket, cache=tempfile.mkdtemp())
    node = TestNode(Bus(connect(self.url)), store=store, gpus=gpus, node=name, engine=engine)
    self.nodes.append(node)
    await node.announce_node()

    async def loop():
      while True:
        await node.tick()
        await asyncio.sleep(0.01)

    self.tasks.append(asyncio.create_task(loop()))
    return node

  def start_scheduler(self) -> None:
    bus = Bus(connect(self.url))
    self.addAsyncCleanup(bus.r.aclose)

    async def loop():
      while True:
        await scheduler.tick(bus)
        await asyncio.sleep(0.01)

    self.tasks.append(asyncio.create_task(loop()))

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
      for proc, lf in node.lives.items():
        if run in lf.owned and proc in node.processes:
          return node.processes[proc].trainer.model.params[run]
    raise AssertionError(f"no live owner found for {run}")

  async def test_new_node_rebalances_mid_accumulation_with_exact_model_parameters(self):
    opt = json.dumps({"adam_params": {"learning_rate": 1e-2}}).encode()
    await self.add_node("node-a", gpus=1)
    self.start_scheduler()
    r1, r2, ctrl = new_id(), new_id(), new_id()
    for r in (r1, r2):
      await self.create(r)
      await self.run_cmd(r, Op.FORWARD_BACKWARD, proto_body(r, 10))
      await self.run_cmd(r, Op.OPTIM_STEP, opt)
      await self.run_cmd(r, Op.FORWARD_BACKWARD, proto_body(r, 10))  # pending grads in memory before rebalance

    # Queue second micro-batch + optim_step on both runs right as `node-b` joins -> one migrates mid-accumulation.
    await self.add_node("node-b", gpus=1)
    reqs = [
      (await self.submit(r, Op.FORWARD_BACKWARD, proto_body(r, 8)), await self.submit(r, Op.OPTIM_STEP, opt))
      for r in (r1, r2)
    ]
    for a, b in reqs:
      self.assertTrue((await self.bus.wait_for_reply(a, 10)).ok)
      self.assertTrue((await self.bus.wait_for_reply(b, 10)).ok)

    state = from_state(await self.bus.read_state())
    self.assertEqual({state.procs[state.home[r][0]].node for r in (r1, r2)}, {"node-a", "node-b"})

    # Unmigrated control model receives the identical command sequence and must match both runs bit-for-bit.
    ref = Model(tiny)
    ref.create_run(ctrl, "tiny", 0, {})
    for n, step_after in ((10, True), (10, False), (8, True)):
      ref.forward_backward(ctrl, [counting(n, 0), counting(n, 40)], "cross_entropy", {}, False)
      if step_after:
        ref.optim_step(ctrl, {"learning_rate": 1e-2})
    for r in (r1, r2):
      torch.testing.assert_close(self.live_params(r), ref.params[ctrl])

  async def test_lora_runs_split_onto_new_node_and_merge_back_with_exact_weights(self):
    opt = json.dumps({"adam_params": {"learning_rate": 1e-2}}).encode()
    await self.add_node("node-a", gpus=1)
    self.start_scheduler()
    r1, r2 = new_id(), new_id()
    for r in (r1, r2):
      await self.create(r, rank=4, lora=LORA)
      await self.run_cmd(r, Op.FORWARD_BACKWARD, proto_body(r, 10))
      await self.run_cmd(r, Op.OPTIM_STEP, opt)
    s0 = from_state(await self.bus.read_state())
    self.assertEqual(s0.home[r1], s0.home[r2])  # initially sharing one LoRA trainer process

    # Add `node-b` and fire simultaneous work on both runs -> one splits onto `node-b`.
    await self.add_node("node-b", gpus=1)
    futs = [
      (await self.submit(r, Op.FORWARD_BACKWARD, proto_body(r, 9)), await self.submit(r, Op.OPTIM_STEP, opt))
      for r in (r1, r2)
    ]
    await scheduler.tick(self.bus)
    s1 = from_state(await self.bus.read_state())
    self.assertNotEqual(s1.home[r1], s1.home[r2])
    self.assertEqual({s1.procs[s1.home[r][0]].node for r in (r1, r2)}, {"node-a", "node-b"})
    for a, b in futs:
      self.assertTrue((await self.bus.wait_for_reply(a, 10)).ok)
      self.assertTrue((await self.bus.wait_for_reply(b, 10)).ok)

    # Launch an exclusive run claiming one of the two nodes -> the displaced LoRA run merges back into its twin.
    excl = new_id()
    await self.create(excl, exclusive=True)
    await self.run_cmd(excl, Op.FORWARD_BACKWARD, proto_body(excl, 6))

    async def merged() -> bool:
      st = from_state(await self.bus.read_state())
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
    total_cmds = 2 * SCALE_BATCH + 4
    payload = json.dumps(JOB).encode()
    with mock.patch("tinyrl.control.node_daemon.SAMPLE_COMMANDS", 2):
      reqs = [await self.submit(sid, Op.SAMPLE, payload, cost=50) for _ in range(total_cmds)]

      async def scaled_out() -> bool:
        st = from_state(await self.bus.read_state())
        return len(st.home.get(sid, ())) >= 2 and len(set(served_by)) >= 2

      await self.until(scaled_out)
      for req in reqs:
        reply = await self.bus.wait_for_reply(req, 10)
        self.assertTrue(reply.ok)
        self.assertEqual(len(wire.sample_json(reply.body)["sequences"]), 1)

    # Every command is billed exactly once despite concurrent multi-node readers.
    self.assertAlmostEqual((await self.bus.read_state())["service"]["team-s"], total_cmds * 50.0)

    # Once the queue drains, `home[sid]` scales back down to 1 replica.
    async def scaled_in() -> bool:
      return len(from_state(await self.bus.read_state()).home.get(sid, ())) == 1

    await self.until(scaled_in)

    # Crash failover: strand a claimed sample under a dead replica consumer and verify a surviving replica reclaims it.
    orig_reclaim = Bus.reclaim_commands
    with mock.patch.object(
      Bus, "reclaim_commands", lambda b, runs, c, idle_ms=10, count=64: orig_reclaim(b, runs, c, 10, count)
    ):
      stranded = await self.submit(sid, Op.SAMPLE, payload, cost=50)
      claimed = await self.bus.take_commands([sid], "dead@gone-1", 1)
      if claimed:
        await asyncio.sleep(0.03)
        kick = await self.submit(sid, Op.SAMPLE, payload, cost=50)
        self.assertTrue((await self.bus.wait_for_reply(kick, 5)).ok)
        self.assertTrue((await self.bus.wait_for_reply(stranded, 5)).ok)
        self.assertAlmostEqual((await self.bus.read_state())["service"]["team-s"], (total_cmds + 2) * 50.0)
      else:
        self.assertTrue((await self.bus.wait_for_reply(stranded, 5)).ok)
        self.assertAlmostEqual((await self.bus.read_state())["service"]["team-s"], (total_cmds + 1) * 50.0)

  async def test_split_brain_plan_never_breaks_gpu_exclusivity_or_single_writer_safety(self):
    node_a = await self.add_node("node-a", gpus=2)
    node_b = await self.add_node("node-b", gpus=1)
    r1, r2, gang = new_id(), new_id(), new_id()
    await self.create(r1)
    await self.create(r2)
    await self.create(gang, gpus=2)

    # Deliberately malformed split-brain plan:
    # - `p1` and `p2` both claim GPU `[0]` on `node-a`
    # - `pg` (`width=2`) is given partial GPU `[1]` on `node-a`
    # - writer run `r1` is placed in `home` of BOTH `p1` (`node-a`) and `p3` (`node-b`) simultaneously
    spec1 = {"role": "trainer", "base": "tiny", "lora": False, "exclusive": False, "train_run": None, "gpus": 1}
    bad_plan = {
      "made": 4,
      "procs": {
        "p1": {"spec": spec1, "node": "node-a", "gpus": [0], "cost": 0.0, "at": 0.0},
        "p2": {"spec": spec1, "node": "node-a", "gpus": [0], "cost": 0.0, "at": 0.0},
        "pg": {"spec": spec1 | {"gpus": 2}, "node": "node-a", "gpus": [1], "cost": 0.0, "at": 0.0},
        "p3": {"spec": spec1, "node": "node-b", "gpus": [0], "cost": 0.0, "at": 0.0},
      },
      "home": {r1: ["p1", "p3"], r2: ["p2"], gang: ["pg"]},
      "lag": {},
      "seen": {},
    }
    await self.bus.write_plan(bad_plan)

    fb, opt = proto_body(r1, 10), json.dumps({"adam_params": {"learning_rate": 1e-2}}).encode()
    futs = []
    for _ in range(3):
      futs.append(await self.submit(r1, Op.FORWARD_BACKWARD, fb))
      futs.append(await self.submit(r1, Op.OPTIM_STEP, opt))

    for req in futs:
      async with asyncio.timeout(10):
        while (await self.bus.r.xlen(f"tinyrl:f:{req}")) == 0:
          for node in (node_a, node_b):
            flat = [g for gs in node.on.values() for g in gs]
            self.assertEqual(len(flat), len(set(flat)))  # GpuExclusive
            self.assertTrue(all(not k.startswith("pg@") for k in node.on))  # partial gang rejected
          holders = [f"{n.id}:{p}" for n in (node_a, node_b) for p, lf in n.lives.items() if r1 in lf.owned]
          self.assertLessEqual(len(holders), 1)  # SingleWriter
          await asyncio.sleep(0.005)
      self.assertTrue((await self.bus.wait_for_reply(req, 5)).ok)

    ref, ctrl = Model(tiny), new_id()
    ref.create_run(ctrl, "tiny", 0, {})
    for _ in range(3):
      ref.forward_backward(ctrl, [counting(10, 0), counting(10, 40)], "cross_entropy", {}, False)
      ref.optim_step(ctrl, {"learning_rate": 1e-2})
    torch.testing.assert_close(self.live_params(r1), ref.params[ctrl])


if __name__ == "__main__":
  unittest.main()
