import copy
import dataclasses
import math
import random
import unittest
from unittest import mock

from tinyrl.bus import Cmd, Ev, Op, Role
from tinyrl.control import scheduler
from tinyrl.control.scheduler import Assignment, Job, Worker, WorkerSpec, decide, from_table, jobs_of, phi
from tinyrl.control.turns import can_run, select_runs, someone_waits

GIB = 2**30


class Cluster:
  """The table the scheduler reads, kept here as it would be in Redis: node heartbeats, run records, backlogs, and workers."""

  def __init__(self):
    self.nodes, self.records, self.backlogs, self.s = {}, {}, {}, Assignment()

  def table(self):
    return {
      "nodes": self.nodes,
      "runs": {r: (rec, self.s.jobs[r].worker if r in self.s.jobs else "") for r, rec in self.records.items()},
      "backlog": {r: b for r, b in self.backlogs.items() if r in self.records and b},
      "workers": {
        n: {"spec": dataclasses.asdict(w.spec), "node": w.node, "gpus": list(w.gpus)} for n, w in self.s.workers.items()
      },
      "workers_made": self.s.workers_made,
    }

  def settle(self):
    self.s, decisions = decide(from_table(self.table()))
    return decisions

  def node(self, name, gpus, memory=80 * GIB, workers=None):
    self.nodes[name] = {"gpus": gpus, "memory": memory, **({"workers": workers} if workers else {})}
    self.settle()

  def lose(self, node):
    self.nodes.pop(node, None)

  def run(self, name, backlog=0, **body):
    self.records[name] = {"role": "trainer", "base": "a", "rank": 0} | body
    self.backlogs[name] = backlog
    self.settle()
    return self.s.jobs[name].worker or next((n for n in self.s.workers if name in jobs_of(self.s, n)), "")

  def set_backlog(self, name, backlog):
    self.backlogs[name] = backlog
    return self.settle()

  def delete(self, run):
    self.records.pop(run, None)
    self.backlogs.pop(run, None)


class Local:
  """Single-node harness exercising cluster Phi placement alongside node-local virtual-time GPU turns."""

  def __init__(self, test, gpus=1, memory=80 * GIB):
    self.test, self.s, self.running, self.clock = test, Assignment(nodes={"n": (gpus, memory)}), {}, 0

  def add(self, name, *, at=None, sampler=False, lora=False, exclusive=False, width=1):
    spec = WorkerSpec(Role.SAMPLER if sampler else Role.TRAINER, name, lora, exclusive, gpus=width)
    self.s.workers[name] = Worker(spec, "n" if at is not None else None, () if at is None else tuple(at))
    self.s.jobs[name] = Job(spec, name)
    if at is None:
      self.place()
    self.running[name] = Worker(spec, "n", self.s.workers[name].gpus)
    return self.running[name]

  def place(self):
    self.s.occupied = {"n": {n: r.gpus for n, r in self.running.items() if r.gpus}} if self.running else {}
    self.s, _ = decide(self.s)

  def drop(self, name):
    self.s.jobs.pop(name)
    self.s.workers.pop(name, None)
    self.place()

  def request(self, name, vtime=None):
    if vtime is None:
      self.clock += 1
      vtime = self.clock
    self.running[name].vtime = vtime

  def grants(self):
    granted = []
    ready = [n for n, r in self.running.items() if r.vtime is not None and r.served is None]
    for name in sorted(ready, key=lambda n: (self.running[n].vtime, n)):
      if can_run(self.s, "n", self.running, name):
        self.running[name].served = 0
        granted.append(name)
    self.check()
    return granted

  def park(self, name):
    self.running[name].served, self.running[name].vtime = None, None

  def check(self):
    for gpu in range(self.s.nodes["n"][0]):
      held = [r for r in self.running.values() if r.served is not None and gpu in r.gpus]
      self.test.assertLessEqual(len(held), 1)  # one process is on a GPU at a time


class TestCluster(unittest.TestCase):
  def test_runs_are_assigned_and_empty_workers_are_removed(self):
    c = Cluster()
    c.node("n", 8)
    a = c.run("a", rank=8)
    self.assertEqual(c.run("b", rank=8), a)
    self.assertEqual((c.s.workers[a].node, c.s.workers[a].gpus), ("n", (0,)))
    c.delete("a")
    self.assertEqual(c.settle(), [])
    c.delete("b")
    self.assertEqual(c.settle(), [(Ev.UNPLACED, {"worker": a, "rule": "place.emptied"})])
    self.assertNotIn(a, c.s.workers)

  def test_a_new_worker_never_takes_a_name_in_use(self):
    c = Cluster()
    c.node("n", 1)
    c.s.workers["w1"] = Worker(WorkerSpec(lora=True), "n", (0,))
    self.assertNotEqual(c.run("f"), "w1")

  def test_workers_go_to_the_node_with_the_fewest_workers_per_gpu(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 2)
    workers = [c.run(name, rank=4, base=name) for name in ("r1", "r2", "r3")]
    self.assertEqual([c.s.workers[w].node for w in workers], ["a", "b", "b"])
    self.assertEqual([c.s.workers[w].gpus for w in workers], [(0,), (0,), (1,)])

  def test_full_fine_tune_sampler_stays_near_its_training_run(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 1)
    a, b = c.run("a"), c.run("b")
    self.assertEqual([c.s.workers[w].node for w in (a, b)], ["a", "b"])
    sampler = c.run("sample", role="sampler", train_run="b")
    self.assertEqual(c.s.workers[sampler].node, "b")

  def test_full_fine_tune_sampler_goes_elsewhere_when_its_trainers_node_is_full(self):
    c = Cluster()
    c.node("a", 2)
    c.node("b", 8)
    x = c.run("x", exclusive=True)
    other = c.run("r", rank=4)
    self.assertEqual([c.s.workers[w].node for w in (x, other)], ["a", "b"])
    c.s.workers[other].node, c.s.workers[other].gpus = "a", (1,)
    sampler = c.run("sample", role="sampler", train_run="x", exclusive=True)
    self.assertEqual(c.s.workers[sampler].node, "b")

  def test_a_full_fine_tunes_trainer_prefers_room_beside_it_for_its_sampler(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 2)
    c.node("c", 8)
    t = c.run("t")
    others = [c.run(f"r{i}", rank=4, base=f"m{i}") for i in range(5)]
    self.assertEqual([c.s.workers[w].node for w in [t, *others]], ["b", "a", "c", "c", "c", "c"])
    lora = c.run("q", role="sampler", rank=4, base="m0")
    sampler = c.run("sample", role="sampler", train_run="t")
    self.assertEqual([c.s.workers[w].node for w in (lora, sampler)], ["c", "b"])

  def test_samplers_get_dedicated_gpus_by_consolidating_trainers_and_prefer_their_trainers_node(self):
    c = Cluster()
    c.node("a", 2)
    c.node("b", 1)
    t = c.run("t")
    r0 = c.run("r0")
    r1 = c.run("r1")
    self.assertEqual([c.s.workers[w].node for w in (t, r0, r1)], ["a", "a", "b"])
    q = c.run("q", role="sampler", train_run="t")
    self.assertEqual(c.s.workers[q].node, "a")
    self.assertEqual(c.s.workers[t].gpus, c.s.workers[r0].gpus)
    self.assertNotEqual(c.s.workers[q].gpus, c.s.workers[t].gpus)

  def test_live_sampler_does_not_restart_just_to_chase_locality(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 2)
    t = c.run("t")  # goes to b:(0,)
    other = c.run("other")  # goes to a:(0,)
    c.s.workers[other].node, c.s.workers[other].gpus = "b", (1,)  # fill b temporarily
    sampler = c.run("s", role="sampler", train_run="t")
    self.assertEqual((c.s.workers[sampler].node, c.s.workers[sampler].gpus), ("a", (0,)))
    c.delete("other")  # b:(1,) opens up, but H(sampler)=5 > +2 locality gain, so sampler stays warm on a:(0,)!
    c.settle()
    self.assertEqual((c.s.workers[t].node, c.s.workers[sampler].node), ("b", "a"))

  def test_lost_node_workers_are_reassigned_onto_live_gpus(self):
    c = Cluster()
    c.node("a", 1)
    w = c.run("r")
    c.node("b", 1)
    c.lose("a")
    self.assertEqual(c.settle()[0], (Ev.UNPLACED, {"worker": w, "rule": "place.node_lost"}))
    self.assertEqual((c.s.workers[w].node, c.s.workers[w].gpus), ("b", (0,)))

  def test_multi_gpu_worker_waits_for_a_node_that_has_enough_gpus(self):
    c = Cluster()
    c.node("a", 1)
    w = c.run("wide", gpus=2)
    self.assertEqual((c.s.workers[w].node, c.s.workers[w].gpus), (None, ()))
    c.node("b", 4)
    self.assertEqual((c.s.workers[w].node, c.s.workers[w].gpus), ("b", (0, 1)))

  def test_lora_runs_share_when_idle_and_full_or_exclusive_runs_get_their_own_workers(self):
    c = Cluster()
    a, b = c.run("a", rank=8), c.run("b", rank=8)
    others = [c.run("c", rank=8, base="b"), c.run("f"), c.run("x", rank=8, exclusive=True)]
    self.assertEqual(a, b)
    self.assertEqual(len({a, b, *others}), 4)

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

  def test_elastic_samplers_scale_out_into_empty_gpus_stay_warm_and_shrink_without_overshrinking_on_drain(self):
    with mock.patch.object(scheduler, "PROMPTS_PER_SAMPLER", 10):
      c = Cluster()
      c.node("n", 4)
      c.run("s1", role="sampler", rank=8, backlog=35)  # ceil(35/10) = 4 replicas across all 4 empty GPUs
      self.assertEqual(len(c.s.workers), 4)
      self.assertEqual(sorted(w.gpus for w in c.s.workers.values()), [(0,), (1,), (2,), (3,)])

      # Backlog drops to 0: all 4 replicas stay warm on idle GPUs!
      c.set_backlog("s1", 0)
      self.assertEqual(len(c.s.workers), 4)

      # Exclusive trainer arrives while w4 is still draining in node heartbeat: shrinks ONLY 1 replica, waits for drain, then places!
      old_workers = {n: {"gpus": list(w.gpus), "pid": 1} for n, w in c.s.workers.items()}
      c.nodes["n"]["workers"] = old_workers
      excl = c.run("excl", exclusive=True)
      samplers = [n for n, w in c.s.workers.items() if w.spec.role == Role.SAMPLER]
      self.assertEqual(len(samplers), 3)  # shrunk exactly 1, NOT all 3!
      self.assertIsNone(c.s.workers[excl].node)  # waiting for shrunk sampler's process to exit its GPU
      self.assertEqual(c.settle(), [])  # subsequent ticks while draining do NOT over-shrink remaining 3 replicas!

      # Shrunk process exits heartbeat -> exclusive trainer immediately takes the freed GPU!
      c.nodes["n"]["workers"] = {n: info for n, info in old_workers.items() if n in samplers}
      c.settle()
      self.assertIsNotNone(c.s.workers[excl].node)
      self.assertEqual(len({w.gpus for w in c.s.workers.values()}), 4)

  def test_concurrently_backlogged_lora_runs_scale_out_and_merge_back_when_gpu_needed(self):
    c = Cluster()
    c.node("n", 2)
    c.run("a", rank=8, backlog=4)
    c.run("b", rank=8, backlog=4)
    self.assertNotEqual(c.s.jobs["a"].worker, c.s.jobs["b"].worker)
    self.assertEqual(sorted(w.gpus for w in c.s.workers.values()), [(0,), (1,)])

    # A full fine-tune arrives needing its 1st GPU: the 2nd LoRA trainer shrinks and merges its run back onto its sibling!
    fft = c.run("fft", rank=0)
    self.assertEqual(c.s.jobs["a"].worker, c.s.jobs["b"].worker)
    self.assertNotEqual(c.s.jobs["a"].worker, fft)
    self.assertEqual(phi(c.s), 0.0)

  def test_random_clusters_settle_in_one_tick_and_preserve_invariants(self):
    with mock.patch.object(scheduler, "PROMPTS_PER_SAMPLER", 10):
      for seed in range(300):
        rng, c = random.Random(seed), Cluster()
        for _ in range(60):
          roll, name = rng.random(), str(rng.randrange(8))
          if roll < 0.15:
            c.node(name, rng.choice([1, 2, 4]))
          elif roll < 0.25:
            c.lose(name)
          elif roll < 0.8 and name not in c.s.jobs:
            role = rng.choice(["trainer", "sampler"])
            excl = rng.random() < 0.2
            train_run = None
            if role == "sampler" and excl:
              excl_trainers = [r for r, rec in c.records.items() if rec["role"] == "trainer" and rec.get("exclusive")]
              if not excl_trainers:
                continue
              train_run = rng.choice(excl_trainers)
            c.run(
              name,
              role=role,
              rank=rng.choice([0, 8]),
              base=str(rng.randrange(3)),
              gpus=rng.choice([1, 2]),
              exclusive=excl,
              train_run=train_run,
              backlog=rng.choice([0, 0, 5, 25]),
            )
          else:
            c.delete(name)
          c.settle()
          self.assertEqual(decide(c.s)[1], [], f"not 1-tick idempotent on seed {seed}")
          self.assertEqual(from_table(c.table()), c.s)
          self.assertFalse(math.isinf(phi(c.s)), f"exclusivity violated on seed {seed}")
          for w in c.s.workers.values():
            if w.node is None:
              self.assertEqual(w.gpus, ())
            else:
              self.assertIn(w.node, c.s.nodes)
              self.assertEqual(len(w.gpus), w.spec.gpus)
              self.assertTrue(all(0 <= g < c.s.nodes[w.node][0] for g in w.gpus))


class TestPlacement(unittest.TestCase):
  def test_workers_take_empty_gpus_while_available(self):
    p = Local(self, 8)
    for n in ("a", "b", "c"):
      p.add(n, lora=True)
    self.assertEqual([p.s.workers[n].gpus for n in ("a", "b", "c")], [(0,), (1,), (2,)])

  def test_samplers_take_empty_gpus_while_available(self):
    p = Local(self, 8)
    for name in ("q", "g"):
      p.add(name, sampler=True)
    p.add("lora", sampler=True, lora=True)
    p.add("exclusive", sampler=True, exclusive=True)
    self.assertEqual(
      {n: w.gpus for n, w in p.s.workers.items()},
      {"q": (0,), "g": (1,), "lora": (2,), "exclusive": (3,)},
    )

  def test_exclusive_workers_are_alone_and_clear_movable_workers_when_needed(self):
    p = Local(self, 2)
    p.add("exclusive", exclusive=True)
    p.add("a")
    p.add("b")
    self.assertEqual([p.s.workers[n].gpus for n in ("exclusive", "a", "b")], [(0,), (1,), (1,)])

    # Even if non-exclusive workers already occupy all GPUs, an arriving exclusive worker displaces one onto the other!
    p = Local(self, 2)
    p.add("a", lora=True)
    p.add("b", lora=True)
    p.add("exclusive", exclusive=True)
    excl_gpu = p.s.workers["exclusive"].gpus
    self.assertEqual(len(excl_gpu), 1)
    self.assertNotEqual(p.s.workers["a"].gpus, excl_gpu)
    self.assertNotEqual(p.s.workers["b"].gpus, excl_gpu)

  def test_multi_gpu_sampler_moves_single_gpu_trainer_aside_via_phi_descent(self):
    p = Local(self, 4)
    p.add("wide", width=2)
    p.add("single")
    p.add("sample", sampler=True, lora=True, width=2)
    self.assertEqual([p.s.workers[n].gpus for n in ("wide", "single", "sample")], [(0, 1), (0,), (2, 3)])

  def test_spreading_waits_until_old_process_leaves_its_actual_gpu(self):
    p = Local(self, 2)
    p.add("old", at=(1,))
    for name in ("a", "b"):
      p.add(name, at=(0,), lora=True)
    p.drop("old")  # "old" is still in p.running on GPU 1!
    self.assertEqual([p.s.workers[n].gpus for n in ("a", "b")], [(0,), (0,)])
    p.running.pop("old")  # now "old" exited GPU 1
    p.place()
    self.assertEqual(sorted(p.s.workers[n].gpus for n in ("a", "b")), [(0,), (1,)])
    before = copy.deepcopy(p.s)
    p.place()
    self.assertEqual(p.s, before)


class TestReservations(unittest.TestCase):
  def test_ready_runs_share_service_even_when_commands_have_different_costs(self):
    heads = [Cmd(run, "1-0", Op.FORWARD_BACKWARD, b"") for run in ("slow", "fast")]
    service = {"slow": 0.0, "fast": 0.0}
    for _ in range(11):
      [run] = select_runs(heads, service, 1)
      service[run] += 10 if run == "slow" else 1
    self.assertEqual(service, {"slow": 10.0, "fast": 10.0})

  def test_least_served_vtime_wins_gpu_turn_regardless_of_arrival_or_worker_name(self):
    p = Local(self)
    for name in ("a", "b", "c"):
      p.add(name)
    p.request("a", vtime=300)
    p.request("b", vtime=50)
    p.request("c", vtime=150)
    order = []
    for _ in range(3):
      [name] = p.grants()
      order.append(name)
      p.park(name)
    self.assertEqual(order, ["b", "c", "a"])

  def test_someone_waits_skips_self_handoff_until_a_waiter_has_smaller_vtime_or_holder_goes_idle(self):
    p = Local(self)
    holder, waiter = p.add("holder"), p.add("waiter")
    holder.vtime, holder.served = 100, 250_000
    waiter.vtime, waiter.served = 500, None
    self.assertFalse(someone_waits(p.running, "holder"))
    holder.vtime = 600
    self.assertTrue(someone_waits(p.running, "holder"))
    holder.vtime, waiter.vtime = None, 999_999
    self.assertTrue(someone_waits(p.running, "holder"))

  def test_one_process_is_on_a_gpu_at_a_time(self):
    p = Local(self)
    for name in ("a", "b"):
      p.add(name, sampler=True)
      p.request(name)
    self.assertEqual(p.grants(), ["a"])
    self.assertEqual(p.grants(), [])
    p.park("a")
    self.assertEqual(p.grants(), ["b"])

  def test_multi_gpu_reservation_is_atomic_and_cannot_be_bypassed_on_either_gpu(self):
    p = Local(self, 2)
    p.add("holder", at=(0,))
    p.add("wide", at=(0, 1), width=2)
    p.add("later", at=(1,))
    for name in ("holder", "wide", "later"):
      p.request(name)
    self.assertEqual(p.grants(), ["holder"])
    self.assertIsNone(p.running["wide"].served)
    self.assertIsNone(p.running["later"].served)
    p.park("holder")
    self.assertEqual(p.grants(), ["wide"])
    p.park("wide")
    self.assertEqual(p.grants(), ["later"])

  def test_different_gpus_can_run_concurrently(self):
    p = Local(self, 2)
    for name, gpu in (("a", 0), ("b", 1)):
      p.add(name, at=(gpu,))
      p.request(name)
    self.assertEqual(p.grants(), ["a", "b"])

  def test_random_requests_and_exits_never_overcommit_a_gpu(self):
    for seed in range(300):
      rng, p = random.Random(seed), Local(self, 4)
      for i in range(10):
        width = rng.choice([1, 1, 2, 4])
        first = rng.randrange(5 - width)
        p.add(str(i), at=tuple(range(first, first + width)), width=width, sampler=rng.random() < 0.5)
      for _ in range(100):
        name = rng.choice(list(p.running))
        r = p.running[name]
        if r.served is not None:
          p.park(name)
        elif r.vtime is None:
          p.request(name, vtime=rng.randrange(20))
        p.grants()


if __name__ == "__main__":
  unittest.main()
