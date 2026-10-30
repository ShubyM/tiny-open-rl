"""Policy properties and asynchronous protocol traces, rather than pins to GPU lane numbers."""

import copy
import random
import unittest

from tinyrl.bus import Event
from tinyrl.scheduler import State, decide, fair_shares, fold, keys

GIB = 2**30


class Log:
  def __init__(self, quantum=1):
    self.s, self.t, self.events = State(quantum=quantum), 0.0, []

  def put(self, kind, **body):
    if kind in ("released", "idle", "revoked") and "turn" not in body and body.get("worker") in self.s.workers:
      body["turn"] = self.s.workers[body["worker"]].turn
    e = Event(str(len(self.events)), self.t, kind, body)
    self.events.append(e)
    self.s = fold(self.s, e)

  def tick(self, t=None):
    if t is not None:
      self.t = t
    actions = decide(self.s, self.t)
    for kind, body in actions:
      self.put(kind, **body)
    assert not decide(self.s, self.t), "decide must converge after its decisions are folded"
    return actions

  def node(self, count=1):
    self.put("node_up", node="n", gpus=count, memory=80 * GIB)

  def run(self, name, worker="", **kw):
    self.put("run_created", run=name, pool=worker, base="model", rank=8, **kw)
    self.put("run_queued", run=name, count=1)

  def worker(self, name, run, gpu=0, **kw):
    self.run(run)
    self.put("worker_created", worker=name, node="n", gpus=[gpu], base="model", rank=8, **kw)
    self.put("run_assigned", worker=name, run=run, memory=GIB)
    self.put("started", worker=name, node="n")


def check(s):
  held = [k for w in s.workers.values() if w.held for k in keys(w)]
  assert len(held) == len(set(held)), "two processes have an exclusive GPU turn"
  for w in s.workers.values():
    assert not (w.held and w.asked)
    assert not w.revoked or w.held
    assert not w.exited or not w.held
  for r in s.runs.values():
    assert r.pending >= 0
    assert abs(r.owed - (r.entitled - r.served)) < 1e-6


class TestPolicy(unittest.TestCase):
  def test_capped_weighted_shares(self):
    self.assertEqual(fair_shares(dict.fromkeys("ABCDE", 1), 4), dict.fromkeys("ABCDE", 0.8))
    self.assertEqual(fair_shares({"A": 1, "B": 1}, 4), {"A": 1, "B": 1})
    self.assertEqual(fair_shares({"A": 3, "B": 1, "C": 1, "D": 1}, 2), {"A": 1, "B": 1 / 3, "C": 1 / 3, "D": 1 / 3})
    self.assertEqual(fair_shares({"A": 1}, 0), {"A": 0})

  def test_starting_workers_reserve_all_eight_gpus(self):
    log = Log()
    log.node(8)
    for i in range(8):
      log.run(str(i))
      log.tick()
    self.assertEqual({w.gpus for w in log.s.workers.values()}, {(i,) for i in range(8)})
    self.assertTrue(all(not w.started for w in log.s.workers.values()))

  def test_free_gpu_before_packing_warm_worker(self):
    log = Log()
    log.node(2)
    log.worker("warm", "A")
    log.put("worker_memory", worker="warm", memory=10 * GIB, run_memory={"A": GIB})
    log.run("B")
    log.tick()
    self.assertNotEqual(log.s.runs["B"].worker, "warm")
    self.assertEqual(log.s.workers[log.s.runs["B"].worker].gpus, (1,))

  def test_compatible_reuse_when_cluster_full(self):
    log = Log()
    log.node()
    log.worker("warm", "A")
    log.put("worker_memory", worker="warm", memory=10 * GIB, run_memory={"A": GIB})
    for name in ("B", "C"):
      log.run(name)
      log.tick()
    self.assertEqual(log.s.runs["C"].worker, "warm")
    self.assertEqual(log.s.workers["warm"].memory, 12 * GIB)

  def test_unknown_or_full_memory_means_separate_process(self):
    for memory in (0, 64 * GIB):
      log = Log()
      log.node()
      log.worker("warm", "A")
      log.put("worker_memory", worker="warm", memory=memory, run_memory={"A": GIB})
      log.run("B")
      log.tick()
      self.assertNotEqual(log.s.runs["B"].worker, "warm")

  def test_known_oversized_run_stays_unassigned(self):
    log = Log()
    log.node()
    log.run("A", memory=90 * GIB)
    self.assertEqual(log.tick(), [])

  def test_idle_run_does_not_bank_entitlement(self):
    log = Log()
    log.node()
    log.run("A")
    log.put("run_queued", run="A", count=0)
    log.t = 100
    log.put("run_queued", run="A", count=1)
    self.assertEqual(log.s.runs["A"].entitled, 0)
    log.t = 101
    log.put("unrelated")
    self.assertEqual(log.s.runs["A"].entitled, 1)

  def test_new_arrival_has_no_historical_debt(self):
    log = Log()
    log.node()
    log.run("A")
    log.t = 10
    log.run("B")
    self.assertEqual(log.s.runs["A"].entitled, 10)
    self.assertEqual(log.s.runs["B"].entitled, 0)
    log.t = 12
    log.put("unrelated")
    self.assertEqual(log.s.runs["A"].entitled, 11)
    self.assertEqual(log.s.runs["B"].entitled, 1)

  def test_stopping_keeps_turn_until_release_or_exit(self):
    for ack in ("released", "exited"):
      log = Log()
      log.node()
      log.worker("old", "A")
      log.worker("next", "B")
      log.put("wants_gpu", worker="old")
      log.tick()
      log.put("worker_stopped", worker="old")
      log.put("wants_gpu", worker="next")
      log.tick(10)
      self.assertTrue(log.s.workers["old"].held)
      self.assertFalse(log.s.workers["next"].held)
      log.put(ack, worker="old")
      log.tick()
      self.assertTrue(log.s.workers["next"].held)

  def test_late_old_events_cannot_release_replacement(self):
    log = Log()
    log.node()
    log.worker("old", "A")
    log.put("exited", worker="old")
    log.worker("new", "B")
    log.put("wants_gpu", worker="new")
    log.tick()
    for kind in ("exited", "released", "idle", "wants_gpu", "started"):
      log.put(kind, worker="old", node="n")
    self.assertTrue(log.s.workers["new"].held)
    self.assertFalse(log.s.workers["old"].held)
    check(log.s)

  def test_node_loss_does_not_claim_a_process_has_exited(self):
    log = Log()
    log.node()
    log.worker("worker", "A")
    log.put("wants_gpu", worker="worker")
    log.tick()
    log.put("node_lost", node="n")
    log.tick(2)
    self.assertTrue(log.s.workers["worker"].held)
    self.assertFalse(log.s.workers["worker"].exited)
    self.assertEqual(log.s.runs["A"].worker, "worker")

  def test_handoff_waits_for_exact_release_then_keeps_debt(self):
    log = Log()
    log.node(2)
    log.worker("source", "A")
    log.put("worker_created", worker="target", node="n", gpus=[1], base="model", rank=8, reserved_for="A")
    log.tick(3)
    token = log.s.runs["A"].moving
    self.assertTrue(token)
    log.put("run_released", run="A", source="source", token="old", uri="x")
    self.assertEqual(log.s.runs["A"].worker, "source")
    log.put("run_released", run="A", source="source", token=token, uri="x")
    before = log.s.runs["A"].owed
    log.tick()
    self.assertEqual(log.s.runs["A"].worker, "target")
    self.assertEqual(log.s.runs["A"].owed, before)
    log.put("service", worker="source", run="A", seconds=100)
    self.assertEqual(log.s.runs["A"].owed, before)

  def test_spread_crowded_runs_when_gpu_becomes_free(self):
    log = Log()
    log.node()
    log.worker("source", "A")
    log.put("worker_memory", worker="source", memory=10 * GIB, run_memory={"A": GIB})
    log.run("B")
    log.tick()
    log.put("node_up", node="n", gpus=2, memory=80 * GIB)
    actions = log.tick(2)
    self.assertIn("run_move_requested", [kind for kind, _ in actions])
    moved = next(r for r in log.s.runs.values() if r.moving)
    self.assertEqual(log.s.workers[moved.target].gpus, (1,))

  def test_cost_prevents_unprofitable_optional_move(self):
    log = Log()
    log.node()
    log.worker("source", "A")
    log.put("worker_memory", worker="source", memory=10 * GIB, run_memory={"A": GIB})
    log.run("B")
    log.tick()
    log.put("batch_done", worker="source", boot=100)
    log.put("node_up", node="n", gpus=2, memory=80 * GIB)
    self.assertEqual(log.tick(2), [])

  def test_fairness_can_force_a_costly_move(self):
    log = Log()
    log.node(2)
    log.worker("source", "A")
    log.worker("target", "C", gpu=1)
    for worker, name in (("source", "A"), ("target", "C")):
      log.put("worker_memory", worker=worker, memory=10 * GIB, run_memory={name: GIB})
    log.run("B")
    log.put("run_assigned", worker="source", run="B", memory=GIB)
    log.put("batch_done", worker="source", boot=100)
    log.t = 10
    log.put("service", worker="target", run="C", seconds=9)
    actions = log.tick()
    self.assertIn("run_move_requested", [kind for kind, _ in actions])

  def test_newcomer_aligns_with_active_service_despite_uncharged_overhead(self):
    log = Log()
    log.node()
    log.worker("old", "A")
    log.t = 1000
    log.put("service", worker="old", run="A", seconds=100)
    log.worker("new", "B")
    self.assertEqual(log.s.runs["B"].vruntime, 100)
    self.assertEqual(log.s.runs["B"].entitled, 0)
    log.put("service", worker="old", run="A", seconds=1)
    log.put("wants_gpu", worker="old")
    log.put("wants_gpu", worker="new")
    log.tick()
    self.assertTrue(log.s.workers["new"].held)

  def test_replay_and_decide_do_not_mutate_input(self):
    log = Log()
    log.node(2)
    log.run("A")
    before = copy.deepcopy(log.s)
    first = decide(log.s, 1)
    self.assertEqual(log.s, before)
    self.assertEqual(first, decide(log.s, 1))
    log.tick(1)
    replay = State(quantum=1)
    for event in log.events:
      replay = fold(replay, event)
    self.assertEqual(replay, log.s)

  def test_random_delayed_worker_facts_preserve_safety(self):
    for seed in range(100):
      rng, log = random.Random(seed), Log()
      log.node(2)
      for i in range(5):
        log.worker(f"w{i}", f"r{i}", gpu=i % 2)
      for _ in range(100):
        name = rng.choice(list(log.s.workers))
        w = log.s.workers[name]
        log.t += rng.random()
        if w.revoked and rng.random() < 0.5:
          kind = "released"
        else:
          kind = rng.choice(("wants_gpu", "idle", "started", "exited", "batch_done"))
        log.put(kind, worker=name, node=w.node)
        log.tick()
        check(log.s)
