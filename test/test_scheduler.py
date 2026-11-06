import copy
import dataclasses
import random
import unittest

from tinyrl.bus import Ev, Event, Role
from tinyrl.control.scheduler import (
  Kind,
  Run,
  State,
  Worker,
  budget,
  can_run,
  decide,
  fit,
  fold,
  has_waiter,
  place_gpus,
  settings,
)

GIB, MIB = 2**30, 2**20


class Cluster:
  """Only cluster assignments are logged. Local GPU placement and reservations never enter this log."""

  def __init__(self):
    self.s, self.entries = State(), []

  def append(self, kind, **body):
    self.entries.append(Event(str(len(self.entries)), 0.0, kind, body))
    fold(self.s, self.entries[-1])

  def settle(self):
    decisions = decide(self.s, 0.0)
    for kind, body in decisions:
      self.append(kind, **body)
    return decisions

  def node(self, name, gpus):
    self.append(Ev.NODE_UP, node=name, gpus=gpus, memory=80 * GIB)
    self.settle()

  def run(self, name, **body):
    self.append(Ev.RUN_CREATED, **({"run": name, "role": "trainer", "base": "a", "rank": 0} | body))
    self.settle()
    return self.s.runs[name].worker


class Local:
  """Direct placement and reservations, with the same small runtime fields the node owns."""

  def __init__(self, test, gpus=1, memory=80 * GIB):
    self.test, self.s, self.running, self.ticket = test, State(nodes={"n": (gpus, memory)}), {}, 0

  def add(self, name, *, at=None, sampler=False, lora=False, exclusive=False, width=1, busy=0.0):
    kind = Kind(Role.SAMPLER if sampler else Role.TRAINER, name, lora, exclusive, gpus=width)
    self.s.workers[name] = Worker(kind, "n", () if at is None else tuple(at))
    self.s.runs[name] = Run(kind, name)
    if at is None:
      self.place()
    self.running[name] = Worker(kind, "n", self.s.workers[name].gpus, busy=busy)
    return self.running[name]

  def place(self):
    place_gpus(self.s, "n", self.running)

  def drop(self, name):
    self.s.runs.pop(name)
    self.s.workers.pop(name)
    self.place()

  def request(self, name):
    self.ticket += 1
    self.running[name].ticket = self.ticket

  def grants(self, now=0.0):
    granted = []
    for name in sorted(self.running, key=lambda n: (self.running[n].ticket, n)):
      if can_run(self.s, "n", self.running, name):
        self.running[name].since, self.running[name].ticket = now, 0
        granted.append(name)
    self.check()
    return granted

  def check(self):
    for gpu in range(self.s.nodes["n"][0]):
      held = [r for r in self.running.values() if r.since is not None and gpu in r.gpus]
      if held:
        self.test.assertTrue(fit([budget(self.s, r.kind, "n") for r in held], self.s.nodes["n"][1]))
    for r in self.running.values():
      self.test.assertFalse(r.ticket and r.since is not None)


class TestCluster(unittest.TestCase):
  def test_runs_are_assigned_and_empty_workers_are_removed(self):
    c = Cluster()
    c.node("n", 8)
    a = c.run("a", rank=8)
    self.assertEqual(c.run("b", rank=8), a)
    self.assertEqual(c.s.workers[a].node, "n")
    self.assertEqual(c.s.workers[a].gpus, ())  # the cluster never chooses local GPUs
    c.append(Ev.RUN_DELETED, run="a")
    self.assertEqual(c.settle(), [])
    c.append(Ev.RUN_DELETED, run="b")
    self.assertEqual(c.settle(), [(Ev.UNPLACED, {"worker": a, "rule": "place.emptied"})])
    self.assertNotIn(a, c.s.workers)

  def test_dedicated_sampling_sessions_end_with_their_training_run(self):
    c = Cluster()
    c.node("n", 8)
    trainer = c.run("train")
    sampler = c.run("sample", role="sampler", train_run="train", exclusive=True)
    c.append(Ev.RUN_DELETED, run="train")
    self.assertEqual({body["worker"] for _, body in c.settle()}, {trainer, sampler})
    self.assertEqual(c.s.workers, {})
    self.assertEqual(c.s.runs, {})

  def test_workers_go_to_the_node_with_the_fewest_workers_per_gpu(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 2)
    workers = [c.run(name) for name in ("r1", "r2", "r3")]
    self.assertEqual([c.s.workers[w].node for w in workers], ["a", "b", "b"])

  def test_full_fine_tune_sampler_stays_near_its_training_run(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 1)
    a, b = c.run("a"), c.run("b")
    self.assertEqual([c.s.workers[w].node for w in (a, b)], ["a", "b"])
    sampler = c.run("sample", role="sampler", train_run="b")
    self.assertEqual(c.s.workers[sampler].node, "b")

  def test_lost_node_workers_are_reassigned_and_forget_their_desired_gpus(self):
    c = Cluster()
    c.node("a", 1)
    w = c.run("r")
    c.s.workers[w].gpus = (0,)
    c.node("b", 1)
    c.append(Ev.NODE_LOST, node="a")
    self.assertEqual(c.s.workers[w].gpus, ())
    c.settle()
    self.assertEqual((c.s.workers[w].node, c.s.workers[w].gpus), ("b", ()))

  def test_multi_gpu_worker_waits_for_a_node_that_has_enough_gpus(self):
    c = Cluster()
    c.node("a", 1)
    w = c.run("wide", gpus=2)
    self.assertIsNone(c.s.workers[w].node)
    c.node("b", 4)
    self.assertEqual(c.s.workers[w].node, "b")

  def test_lora_runs_share_but_full_or_exclusive_runs_get_their_own_workers(self):
    c = Cluster()
    a, b = c.run("a", rank=8), c.run("b", rank=8)
    others = [c.run("c", rank=8, base="b"), c.run("f"), c.run("x", rank=8, exclusive=True)]
    self.assertEqual(a, b)
    self.assertEqual(len({a, b, *others}), 4)

  def test_sampling_sessions_share_only_their_kind_and_exclusive_training_run(self):
    c = Cluster()
    a = c.run("a", role="sampler", rank=8, train_run="A")
    b = c.run("b", role="sampler", rank=8, train_run="B")
    fft = c.run("f", role="sampler", train_run="A")
    x = c.run("x", role="sampler", rank=8, train_run="X", exclusive=True)
    y = c.run("y", role="sampler", rank=8, train_run="X", exclusive=True)
    z = c.run("z", role="sampler", rank=8, train_run="Z", exclusive=True)
    self.assertEqual(a, b)
    self.assertEqual(x, y)
    self.assertEqual(len({a, b, fft, x, y, z}), 4)

  def test_decide_does_not_mutate_its_input(self):
    c = Cluster()
    c.node("n", 8)
    c.append(Ev.RUN_CREATED, run="r", role="trainer", base="a", rank=8)
    before = dataclasses.asdict(c.s)
    self.assertTrue(decide(c.s, 0.0))
    self.assertEqual(dataclasses.asdict(c.s), before)

  def test_random_cluster_logs_replay_without_local_execution_state(self):
    for seed in range(100):
      rng, c = random.Random(seed), Cluster()
      for _ in range(100):
        roll, name = rng.random(), str(rng.randrange(8))
        if roll < 0.15:
          c.node(name, rng.choice([1, 2, 4]))
        elif roll < 0.25:
          c.append(Ev.NODE_LOST, node=name)
        elif roll < 0.8 and name not in c.s.runs:
          c.run(name, rank=rng.choice([0, 8]), base=str(rng.randrange(3)), gpus=rng.choice([1, 2]))
        else:
          c.append(Ev.RUN_DELETED, run=name)
        c.settle()
        self.assertEqual(decide(c.s, 0.0), [])
        self.assertTrue(all(w.node is None or w.node in c.s.nodes for w in c.s.workers.values()))
      replay = State()
      for event in c.entries:
        fold(replay, event)
      self.assertEqual(replay, c.s)
      self.assertTrue(all(w.gpus == () for w in replay.workers.values()))


class TestPlacement(unittest.TestCase):
  def test_workers_take_empty_gpus_while_available(self):
    p = Local(self, 8)
    for n in ("a", "b", "c"):
      p.add(n, lora=True)
    self.assertEqual([p.s.workers[n].gpus for n in ("a", "b", "c")], [(0,), (1,), (2,)])

  def test_shared_fft_samplers_pack_their_memory_budgets(self):
    p = Local(self, 8, memory=81559 * MIB)
    for name in ("q", "g"):
      p.add(name, sampler=True)
    p.add("lora", sampler=True, lora=True)
    p.add("exclusive", sampler=True, exclusive=True)
    shares = {"TINYRL_WARM_SAMPLER": "1", "TINYRL_GPU_MEM": "0.42", "TINYRL_KV_BYTES": str(16 * GIB)}
    alone = {"TINYRL_WARM_SAMPLER": "1", "TINYRL_KV_BYTES": str(24 * GIB)}
    self.assertEqual(
      {n: (w.gpus, settings(w.kind, 8, 81559 * MIB)) for n, w in p.s.workers.items()},
      {"q": ((0,), shares), "g": ((0,), shares), "lora": ((1,), alone), "exclusive": ((2,), alone)},
    )
    self.assertEqual(budget(p.s, p.running["q"].kind, "n"), int(0.42 * 81559 * MIB))
    self.assertEqual(settings(Kind(Role.SAMPLER), 8, 23034 * MIB), {"TINYRL_WARM_SAMPLER": "1"})
    self.assertEqual(settings(Kind(Role.SAMPLER, lora=True), 1, 81559 * MIB), {"TINYRL_KV_BYTES": str(24 * GIB)})

  def test_trainer_takes_turns_with_the_least_busy_trainers(self):
    p = Local(self, 2)
    p.add("busy", busy=50)
    p.add("light", busy=5)
    p.add("new")
    self.assertEqual(p.s.workers["new"].gpus, p.s.workers["light"].gpus)

  def test_sampler_moves_the_least_busy_trainers_aside(self):
    p = Local(self, 2)
    p.add("busy", busy=50)
    light = p.add("light", busy=5)
    p.add("sampler", sampler=True, lora=True)
    self.assertEqual(p.s.workers["light"].gpus, p.s.workers["busy"].gpus)
    self.assertNotEqual(p.s.workers["sampler"].gpus, p.s.workers["busy"].gpus)
    self.assertEqual(light.gpus, p.s.workers["sampler"].gpus)  # a placement decision does not move the actual process

  def test_budgeted_worker_fills_a_gpu_before_using_an_empty_one(self):
    p = Local(self, 2)
    p.add("train")
    p.add("a", sampler=True)
    p.drop("train")
    p.running.pop("train")
    p.add("b", sampler=True)
    self.assertEqual(p.s.workers["a"].gpus, p.s.workers["b"].gpus)

  def test_exclusive_workers_are_alone_and_others_stay_off_their_gpu(self):
    p = Local(self, 2)
    p.add("exclusive", exclusive=True)
    p.add("a")
    p.add("b")
    self.assertEqual([p.s.workers[n].gpus for n in ("exclusive", "a", "b")], [(0,), (1,), (1,)])
    p = Local(self, 2)
    p.add("exclusive", at=(1,), sampler=True, exclusive=True)
    p.add("a", sampler=True)
    self.assertEqual(p.s.workers["a"].gpus, (0,))

  def test_sampler_moves_as_few_trainers_as_possible(self):
    p = Local(self, 2)
    for name, gpu, busy in (("c", 1, 5), ("a", 0, 1), ("b", 0, 1)):
      p.add(name, at=(gpu,), busy=busy)
    p.add("sample", sampler=True, lora=True)
    self.assertEqual((p.s.workers["sample"].gpus, p.s.workers["c"].gpus), ((1,), (0,)))

  def test_sampler_moves_lora_before_full_fine_tune(self):
    p = Local(self, 2)
    p.add("full", at=(0,), busy=1)
    p.add("lora", at=(1,), lora=True, busy=9)
    p.add("sample", sampler=True, lora=True)
    self.assertEqual((p.s.workers["sample"].gpus, p.s.workers["lora"].gpus), ((1,), (0,)))

  def test_multi_gpu_sampler_moves_single_gpu_trainer_aside(self):
    p = Local(self, 4)
    p.add("wide", width=2)
    p.add("single")
    p.add("sample", sampler=True, lora=True, width=2)
    self.assertEqual([p.s.workers[n].gpus for n in ("wide", "single", "sample")], [(0, 1), (0,), (2, 3)])

  def test_benchmark_arrivals_move_lora_trainers_and_pack_full_samplers(self):
    p = Local(self, 8, memory=81559 * MIB)
    for name, busy in {"x": 4, "lora_g": 7, "lora_q": 7.7, "q1": 1.7, "q2": 2.4, "g1": 4.8}.items():
      p.add(name, lora=name.startswith("lora"), exclusive=name == "x", busy=busy)
    for name in ("fft_q", "sample_q", "fft_g", "fft_x", "sample_g"):
      p.add(name, sampler=True, lora=name.startswith("sample"), exclusive=name == "fft_x")
    gpu = {n: w.gpus[0] for n, w in p.s.workers.items()}
    self.assertEqual({gpu["lora_g"], gpu["lora_q"]}, {gpu["q1"], gpu["q2"]})
    self.assertEqual(gpu["fft_q"], gpu["fft_g"])
    self.assertEqual(
      sorted(gpu[n] for n in ("x", "g1", "sample_q", "fft_x", "sample_g", "q1", "q2", "fft_q")), list(range(8))
    )

  def test_spreading_waits_until_old_process_leaves_its_actual_gpu(self):
    p = Local(self, 2)
    p.add("old", at=(1,))
    for name in ("a", "b"):
      p.add(name, at=(0,))
    p.drop("old")
    self.assertEqual([p.s.workers[n].gpus for n in ("a", "b")], [(0,), (0,)])
    p.running.pop("old")
    p.place()
    self.assertEqual([p.s.workers[n].gpus for n in ("a", "b")], [(1,), (0,)])
    before = copy.deepcopy(p.s)
    p.place()
    self.assertEqual(p.s, before)

  def test_spreading_prefers_lora_and_then_a_waiter_to_the_holder(self):
    p = Local(self, 2)
    p.add("old", at=(1,))
    p.add("full", at=(0,))
    p.add("a", at=(0,), lora=True)
    p.add("b", at=(0,), lora=True)
    p.request("a")
    self.assertEqual(p.grants(), ["a"])
    p.drop("old")
    p.running.pop("old")
    p.place()
    self.assertEqual(p.s.workers["b"].gpus, (1,))
    self.assertEqual(p.s.workers["a"].gpus, (0,))

  def test_benchmark_trainers_spread_back_after_the_previous_wave_leaves(self):
    p = Local(self, 8)
    old = [f"old{g}" for g in (0, 1, 3, 4, 5, 6, 7)]
    for name in old:
      p.add(name, at=(int(name[3:]),))
    trainers = ("lora_g", "lora_q", "q1", "q2", "g1")
    for name in trainers:
      p.add(name, at=(2,), lora=name.startswith("lora"))
    for name in old:
      p.drop(name)
      p.running.pop(name)
      p.place()
    self.assertEqual(len({p.s.workers[n].gpus for n in trainers}), len(trainers))

  def test_random_arrivals_do_not_make_samplers_share_when_trainers_can_move(self):
    rng = random.Random(42)
    for _ in range(200):
      p = Local(self, rng.choice([2, 4, 8]))
      names = [f"lora{i}" for i in range(3)] + [f"train{i}" for i in range(4)] + ["sample_a", "sample_b"]
      for name in rng.sample(names, rng.randint(1, len(names))):
        there = [[w for w in p.s.workers.values() if g in w.gpus] for g in range(p.s.nodes["n"][0])]
        can_move = any(ws and all(w.kind.role == Role.TRAINER for w in ws) for ws in there)
        p.add(name, sampler=name.startswith("sample"), lora=not name.startswith("train"), busy=rng.random() * 10)
        if name.startswith("sample") and can_move:
          mine = p.s.workers[name].gpus
          self.assertFalse(any(set(w.gpus) & set(mine) for n, w in p.s.workers.items() if n != name))

  def test_only_the_owning_node_changes_desired_gpu_placement(self):
    p = Local(self, 2)
    kind = Kind()
    p.s.nodes["other"] = (2, 80 * GIB)
    p.s.workers["remote"] = Worker(kind, "other")
    p.s.runs["remote"] = Run(kind, "remote")
    p.add("local")
    self.assertEqual(p.s.workers["remote"].gpus, ())
    self.assertEqual(p.s.workers["local"].gpus, (0,))
    before = copy.deepcopy(p.s)
    place_gpus(p.s, "missing", p.running)
    self.assertEqual(p.s, before)


class TestReservations(unittest.TestCase):
  def test_fifo_order_is_the_order_of_requests_not_worker_names(self):
    p = Local(self)
    for name in ("a", "b", "c"):
      p.add(name)
    for name in ("a", "c", "b"):
      p.request(name)
    order = []
    for _ in range(3):
      [name] = p.grants()
      order.append(name)
      p.running[name].since = None
    self.assertEqual(order, ["a", "c", "b"])

  def test_only_eligible_waiters_count_as_demand(self):
    p = Local(self, 2)
    p.add("a", at=(0,))
    p.add("b", at=(1,))
    self.assertFalse(has_waiter(p.running, (0, 1)))
    p.request("b")
    self.assertFalse(has_waiter(p.running, (0,)))
    self.assertTrue(has_waiter(p.running, (1,)))
    self.assertEqual(p.grants(), ["b"])
    self.assertFalse(has_waiter(p.running, (0, 1)))
    self.assertFalse(can_run(p.s, "n", p.running, "b"))

  def test_a_single_pass_cannot_reserve_two_incompatible_workers(self):
    p = Local(self)
    for name in ("a", "b"):
      p.add(name)
      p.request(name)
    self.assertEqual(p.grants(), ["a"])
    self.assertEqual(p.grants(), [])

  def test_workers_with_known_budgets_share_but_unknown_budgets_wait(self):
    p = Local(self)
    for name in ("a", "b", "c"):
      p.add(name, sampler=True)
      p.request(name)
    p.add("trainer")
    p.request("trainer")
    self.assertEqual(p.grants(), ["a", "b"])
    p.running["a"].since = None
    self.assertEqual(p.grants(), ["c"])
    p.running["b"].since = None
    self.assertEqual(p.grants(), [])
    p.running["c"].since = None
    self.assertEqual(p.grants(), ["trainer"])

  def test_unknown_gpu_memory_never_allows_concurrent_reservations(self):
    p = Local(self, memory=0)
    for name in ("a", "b"):
      p.add(name, sampler=True)
      p.request(name)
    self.assertEqual(p.grants(), ["a"])

  def test_multi_gpu_reservation_is_atomic_and_cannot_be_bypassed_on_either_gpu(self):
    p = Local(self, 2)
    p.add("holder", at=(0,))
    p.add("wide", at=(0, 1), width=2)
    p.add("later", at=(1,))
    for name in ("holder", "wide", "later"):
      p.request(name)
    self.assertEqual(p.grants(), ["holder"])
    self.assertIsNone(p.running["wide"].since)
    self.assertIsNone(p.running["later"].since)
    p.running["holder"].since = None
    self.assertEqual(p.grants(), ["wide"])
    p.running["wide"].since = None
    self.assertEqual(p.grants(), ["later"])

  def test_different_gpus_can_run_concurrently(self):
    p = Local(self, 2)
    for name, gpu in (("a", 0), ("b", 1)):
      p.add(name, at=(gpu,))
      p.request(name)
    self.assertEqual(p.grants(), ["a", "b"])

  def test_unplacement_does_not_free_an_actual_reservation(self):
    p = Local(self)
    for name in ("a", "b"):
      p.add(name)
      p.request(name)
    self.assertEqual(p.grants(), ["a"])
    p.drop("a")
    self.assertNotIn("a", p.s.workers)
    self.assertEqual(p.grants(), [])
    p.running.pop("a")
    self.assertEqual(p.grants(), ["b"])

  def test_desired_move_does_not_free_the_actual_gpu(self):
    p = Local(self, 2)
    for name in ("a", "b"):
      p.add(name, at=(0,))
      p.request(name)
    self.assertEqual(p.grants(), ["a"])
    p.s.workers["a"].gpus = (1,)
    self.assertEqual(p.grants(), [])
    p.running["a"].since = None
    p.running["a"].gpus = (1,)
    self.assertEqual(p.grants(), ["b"])

  def test_unplaced_multi_gpu_trainer_can_reacquire_actual_gpus_for_its_checkpoint(self):
    p = Local(self, 2)
    p.add("wide", at=(0, 1), width=2)
    p.drop("wide")
    p.request("wide")
    self.assertEqual(p.grants(), ["wide"])

  def test_local_reservations_use_actual_node_memory_even_after_a_cluster_move(self):
    p = Local(self)
    p.s.nodes["other"] = (1, 20 * GIB)
    for name in ("a", "b"):
      p.add(name, sampler=True)
      p.request(name)
      p.s.workers[name].node = "other"
    self.assertEqual(p.grants(), ["a", "b"])

  def test_missing_node_or_empty_actual_placement_cannot_be_granted(self):
    p = Local(self)
    p.add("a", at=())
    p.request("a")
    self.assertFalse(can_run(p.s, "n", p.running, "a"))
    p.running["a"].gpus = (0,)
    self.assertFalse(can_run(p.s, "missing", p.running, "a"))

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
        if r.since is not None:
          r.since = None  # parking or process exit has completed
        elif not r.ticket:
          p.request(name)
        p.grants()


if __name__ == "__main__":
  unittest.main()
