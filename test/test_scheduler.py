"""Policy properties and asynchronous protocol traces, rather than pins to GPU numbers."""

import copy
import random
import unittest

from tinyrl.bus import Event
from tinyrl.scheduler import SAMPLER_MEM, State, decide, fold, keys

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

  def run(self, name, base="model", rank=8, role="trainer", **kw):
    self.put("run_created", run=name, pool="", base=base, rank=rank, role=role, **kw)

  def start(self):
    for name, w in self.s.workers.items():
      if not w.started:
        self.put("started", worker=name, node=w.node)

  def worker(self, name, run, gpu=0, mem=1.0):
    self.run(run, exclusive=True)
    self.put("worker_created", worker=name, run=run, node="n", gpus=[gpu], base="model", lora=True, mem=mem)
    self.put("started", worker=name, node="n")

  def holding(self):
    return {name for name, w in self.s.workers.items() if w.since is not None}


def check(s):
  for k in {k for w in s.workers.values() for k in keys(w)}:
    held = sum(w.mem for w in s.workers.values() if w.since is not None and k in keys(w))
    assert held <= 1 + 1e-6, f"workers holding {k} need more than the GPU"
  for w in s.workers.values():
    assert not (w.since is not None and w.asked)
    assert not w.revoked or w.since is not None
    assert not w.exited or w.since is None


class TestPolicy(unittest.TestCase):
  def test_lora_runs_of_one_base_share_a_worker_full_and_exclusive_do_not(self):
    log = Log()
    log.node(8)
    for name in ("A", "B"):
      log.run(name)
    log.run("C", base="other")
    log.run("F1", rank=0)
    log.run("F2", rank=0)
    log.run("X", exclusive=True)
    log.tick()
    worker = {r: x.worker for r, x in log.s.runs.items()}
    self.assertEqual(worker["A"], worker["B"])
    self.assertEqual(len(set(worker.values())), 5)

  def test_sampling_sessions_of_one_base_share_a_vllm_worker(self):
    log = Log()
    log.node(2)
    for name in ("s1", "s2", "s3"):
      log.run(name, role="sampler", rank=1)
    log.run("f1", role="sampler", rank=0)
    log.tick()
    worker = {r: x.worker for r, x in log.s.runs.items()}
    self.assertEqual(len({worker["s1"], worker["s2"], worker["s3"]}), 1)
    self.assertNotEqual(worker["f1"], worker["s1"])
    w = log.s.workers[worker["s1"]]
    self.assertEqual((w.role, w.mem, w.env["TINYRL_GPU_MEM"]), ("sampler", SAMPLER_MEM, str(SAMPLER_MEM)))

  def test_an_exclusive_runs_sampling_sessions_share_one_worker_of_their_own(self):
    log = Log()
    log.node(2)
    for name, train in (("x1", "X"), ("x2", "X"), ("y1", "Y")):
      log.run(name, role="sampler", rank=0, exclusive=True, train_run=train)
    log.run("shared", role="sampler", rank=0, train_run="Z")
    log.tick()
    worker = {r: x.worker for r, x in log.s.runs.items()}
    self.assertEqual(worker["x1"], worker["x2"])
    self.assertEqual(len({worker["x1"], worker["y1"], worker["shared"]}), 3)

  def test_new_workers_take_the_least_used_gpu_they_fit(self):
    log = Log()
    log.node(3)
    log.run("T", rank=0)
    log.run("s", role="sampler", base="a")
    log.run("u", role="sampler", base="b")
    log.run("v", role="sampler", base="c")
    log.tick()
    gpu = {r: log.s.workers[x.worker].gpus for r, x in log.s.runs.items()}
    self.assertEqual(gpu["T"], (0,))
    self.assertEqual({gpu["s"], gpu["u"]}, {(1,), (2,)})
    self.assertIn(gpu["v"], {(1,), (2,)}, "a third sampler shares a sampler's GPU, not the trainer's")

  def test_samplers_compute_together_and_a_trainer_waits_for_a_revoke(self):
    log = Log(quantum=10)
    log.node()
    log.worker("s1", "a", mem=0.4)
    log.worker("s2", "b", mem=0.4)
    log.worker("t", "c", mem=1.0)
    for name in ("s1", "s2", "t"):
      log.put("wants_gpu", worker=name)
    log.tick(0)
    self.assertEqual(log.holding(), {"s1", "s2"})
    log.tick(5)
    self.assertEqual(log.holding(), {"s1", "s2"})
    log.tick(10)
    self.assertTrue(log.s.workers["s1"].revoked and log.s.workers["s2"].revoked)
    log.put("released", worker="s1")
    log.put("wants_gpu", worker="s1")
    log.tick()
    self.assertNotIn("s1", log.holding(), "an earlier waiter that doesn't fit yet is not passed")
    log.put("released", worker="s2")
    log.tick()
    self.assertEqual(log.holding(), {"t"})
    check(log.s)

  def test_waiters_are_served_first_come_first_served(self):
    log = Log()
    log.node()
    for name in ("a", "b", "c"):
      log.worker(name, name.upper())
    for name in ("b", "c", "a"):
      log.put("wants_gpu", worker=name)
    order = []
    for _ in range(3):
      log.tick()
      (holder,) = log.holding()
      order.append(holder)
      log.put("released", worker=holder)
    self.assertEqual(order, ["b", "c", "a"])

  def test_idle_holder_gives_way_before_the_quantum(self):
    log = Log(quantum=100)
    log.node()
    log.worker("a", "A")
    log.worker("b", "B")
    log.put("wants_gpu", worker="a")
    log.tick()
    log.put("wants_gpu", worker="b")
    self.assertEqual(log.tick(1), [])
    log.put("idle", worker="a")
    self.assertEqual([kind for kind, _ in log.tick(2)], ["revoked"])

  def test_worker_without_runs_stops_and_an_exited_workers_runs_are_reassigned(self):
    log = Log()
    log.node(2)
    log.run("A")
    log.tick()
    first = log.s.runs["A"].worker
    log.put("exited", worker=first, node="n")
    log.put("run_released", run="A", source=first, reason="exited")
    log.tick()
    self.assertNotIn(log.s.runs["A"].worker, ("", first))
    log.put("run_deleted", run="A")
    log.tick()
    self.assertTrue(log.s.workers[log.events[-1].body["worker"]].stopping)

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
      self.assertEqual(log.holding(), {"old"})
      log.put(ack, worker="old", node="n")
      log.tick()
      self.assertEqual(log.holding(), {"next"})

  def test_late_old_events_cannot_release_replacement(self):
    log = Log()
    log.node()
    log.worker("old", "A")
    log.put("exited", worker="old", node="n")
    log.worker("new", "B")
    log.put("wants_gpu", worker="new")
    log.tick()
    for kind in ("exited", "released", "idle", "wants_gpu", "started"):
      log.put(kind, worker="old", node="n")
    self.assertEqual(log.holding(), {"new"})
    check(log.s)

  def test_node_loss_does_not_claim_a_process_has_exited(self):
    log = Log()
    log.node()
    log.worker("worker", "A")
    log.put("wants_gpu", worker="worker")
    log.tick()
    log.put("node_lost", node="n")
    log.tick(2)
    self.assertEqual(log.holding(), {"worker"})
    self.assertFalse(log.s.workers["worker"].exited)
    self.assertEqual(log.s.runs["A"].worker, "worker")

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
        log.worker(f"w{i}", f"r{i}", gpu=i % 2, mem=rng.choice((0.4, 1.0)))
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


if __name__ == "__main__":
  unittest.main()
