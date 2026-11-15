import copy
import random
import unittest

from tinyrl.bus import Cmd, Ev, Op, Role
from tinyrl.control.scheduler import (
  Assignment,
  Job,
  Worker,
  WorkerSpec,
  decide,
  from_table,
  grant,
  memory_budget,
  place_gpus,
  select_runs,
  to_table,
)

GIB, MIB = 2**30, 2**20


class Cluster:
  """The table the scheduler reads, kept here as it would be in Redis: node heartbeats, run records and the workers it wrote."""

  def __init__(self):
    self.nodes, self.records, self.s = {}, {}, Assignment()

  def table(self):
    return {
      "nodes": self.nodes,
      "runs": {r: (rec, self.s.jobs[r].worker if r in self.s.jobs else "") for r, rec in self.records.items()},
      "workers": to_table(self.s),
      "workers_made": self.s.workers_made,
    }

  def settle(self):
    self.s, decisions = decide(from_table(self.table()))
    return decisions

  def node(self, name, gpus, memory=80 * GIB):
    self.nodes[name] = {"gpus": gpus, "memory": memory}
    self.settle()

  def lose(self, node):
    self.nodes.pop(node, None)

  def run(self, name, **body):
    self.records[name] = {"role": "trainer", "base": "a", "rank": 0} | body
    self.settle()
    return self.s.jobs[name].worker

  def delete(self, run):
    self.records.pop(run, None)


class Local:
  """One node as the scheduler sees it: its workers, their GPUs, who asked for theirs, and who holds theirs."""

  def __init__(self, test, gpus=1, memory=80 * GIB):
    self.test, self.s, self.ready, self.service = test, Assignment(nodes={"n": (gpus, memory)}), {}, {}

  def add(self, name, *, at=None, sampler=False, lora=False, exclusive=False, width=1):
    spec = WorkerSpec(Role.SAMPLER if sampler else Role.TRAINER, name, lora, exclusive, gpus=width)
    self.s.workers[name] = Worker(spec, "n", () if at is None else tuple(at))
    self.s.jobs[name] = Job(spec, name)
    if at is None:
      self.place()

  def place(self):
    place_gpus(self.s, "n")

  def drop(self, name):
    self.s.jobs.pop(name)
    self.s.workers.pop(name)
    self.place()

  def request(self, name, served=0):
    self.ready[name], self.service[name] = [name], served

  def release(self, name):
    """Its work is done and its node gave its GPUs back."""
    self.s.granted.pop(name)
    self.ready.pop(name, None)

  def grants(self):
    made, _, _ = grant(self.s, self.ready, self.service, 0)
    self.s.granted |= made
    self.check()
    return list(made)

  def check(self):
    for gpu in range(self.s.nodes["n"][0]):
      held = [n for n, (_, gpus) in self.s.granted.items() if gpu in gpus]
      if len(held) > 1:
        sizes = [memory_budget(self.s, self.s.workers[n].spec, "n") for n in held]
        self.test.assertTrue(all(sizes))
        self.test.assertLessEqual(sum(sizes), self.s.nodes["n"][1])


class TestCluster(unittest.TestCase):
  def test_runs_are_assigned_and_empty_workers_are_removed(self):
    c = Cluster()
    c.node("n", 8)
    a = c.run("a", rank=8)
    self.assertEqual(c.run("b", rank=8), a)
    self.assertEqual(c.s.workers[a].node, "n")
    self.assertEqual(c.s.workers[a].gpus, (0,))  # the cluster chooses its GPUs too
    c.delete("a")
    self.assertEqual(c.settle(), [])
    c.delete("b")
    self.assertEqual(c.settle(), [(Ev.UNPLACED, {"worker": a, "rule": "place.emptied"})])
    self.assertNotIn(a, c.s.workers)

  def test_a_new_worker_never_takes_a_name_in_use(self):
    c = Cluster()
    c.node("n", 1)
    c.s.workers["w1"] = Worker(
      WorkerSpec(lora=True), "n"
    )  # a worker the counter doesn't know about, as after a lost counter
    self.assertNotEqual(c.run("f"), "w1")

  def test_workers_go_to_the_node_with_the_fewest_workers_per_gpu(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 2)
    workers = [c.run(name, rank=4, base=name) for name in ("r1", "r2", "r3")]
    self.assertEqual([c.s.workers[w].node for w in workers], ["a", "b", "b"])

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
    c.s.workers[other].node = "a"  # it took the GPU x keeps for its sampler
    sampler = c.run("sample", role="sampler", train_run="x", exclusive=True)
    self.assertEqual(c.s.workers[sampler].node, "b")

  def test_a_full_fine_tunes_trainer_keeps_room_beside_it_for_its_sampler(self):
    c = Cluster()
    c.node("a", 1)
    c.node("b", 2)
    c.node("c", 8)
    t = c.run("t")
    others = [c.run(f"r{i}", rank=4, base=f"m{i}") for i in range(5)]
    self.assertEqual([c.s.workers[w].node for w in [t, *others]], ["b", "a", "c", "c", "c", "c"])
    lora = c.run("q", role="sampler", rank=4, base="m0")
    sampler = c.run("sample", role="sampler", train_run="t")
    self.assertEqual([c.s.workers[w].node for w in (lora, sampler)], ["c", "b"])  # the GPU t kept went to its sampler

  def test_samplers_packed_by_memory_leave_room_near_their_trainer(self):
    c = Cluster()
    c.node("a", 2, memory=81559 * MIB)
    c.node("b", 8)
    t = c.run("t")
    others = [c.run(f"r{i}") for i in range(4)]
    self.assertEqual([c.s.workers[w].node for w in [t, *others]], ["a", "b", "b", "b", "b"])
    q = c.run("q", role="sampler", train_run="t")
    g = c.run("g", role="sampler", train_run="t", base="b")
    self.assertEqual([c.s.workers[w].node for w in (q, g)], ["a", "a"])  # two full fine-tune samplers pack one GPU

  def test_lost_node_workers_are_reassigned_and_choose_gpus_again(self):
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
    c.run("X", rank=8, exclusive=True)  # an exclusive run's sessions end with it, so it has to exist
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

  def test_random_clusters_settle_and_survive_the_table(self):
    for seed in range(100):
      rng, c = random.Random(seed), Cluster()
      for _ in range(100):
        roll, name = rng.random(), str(rng.randrange(8))
        if roll < 0.15:
          c.node(name, rng.choice([1, 2, 4]))
        elif roll < 0.25:
          c.lose(name)
        elif roll < 0.8 and name not in c.s.jobs:
          c.run(name, rank=rng.choice([0, 8]), base=str(rng.randrange(3)), gpus=rng.choice([1, 2]))
        else:
          c.delete(name)
        c.settle()
        self.assertEqual(decide(c.s)[1], [])
        self.assertTrue(all(w.node is None or w.node in c.s.nodes for w in c.s.workers.values()))
        self.assertEqual(from_table(c.table()), c.s)
        self.assertTrue(all(len(w.gpus) == (w.spec.gpus if w.node else 0) for w in c.s.workers.values()))


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
    self.assertEqual(
      {n: w.gpus for n, w in p.s.workers.items()},
      {"q": (0,), "g": (0,), "lora": (1,), "exclusive": (2,)},
    )

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

  def test_multi_gpu_sampler_moves_single_gpu_trainer_aside(self):
    p = Local(self, 4)
    p.add("wide", width=2)
    p.add("single")
    p.add("sample", sampler=True, lora=True, width=2)
    self.assertEqual([p.s.workers[n].gpus for n in ("wide", "single", "sample")], [(0, 1), (0,), (2, 3)])

  def test_spreading_waits_until_old_process_leaves_its_actual_gpu(self):
    p = Local(self, 2)
    p.add("old", at=(1,))
    for name in ("a", "b"):
      p.add(name, at=(0,))
    p.s.granted["old"] = ("n", (1,))  # its process is still there
    p.drop("old")
    self.assertEqual([p.s.workers[n].gpus for n in ("a", "b")], [(0,), (0,)])
    p.s.granted.pop("old")
    p.place()
    self.assertEqual([p.s.workers[n].gpus for n in ("a", "b")], [(1,), (0,)])
    before = copy.deepcopy(p.s)
    p.place()
    self.assertEqual(p.s, before)


class TestGrants(unittest.TestCase):
  def test_ready_runs_share_service_even_when_commands_have_different_costs(self):
    heads = [Cmd(run, "1-0", Op.FORWARD_BACKWARD, b"") for run in ("slow", "fast")]
    service = {"slow": 0.0, "fast": 0.0}
    for _ in range(11):
      [run] = select_runs(heads, service, 1)
      service[run] += 10 if run == "slow" else 1
    self.assertEqual(service, {"slow": 10.0, "fast": 10.0})

  def test_the_least_served_account_goes_first_whatever_the_worker_names(self):
    p = Local(self)
    for name, served in (("a", 2), ("b", 1), ("c", 0)):
      p.add(name)
      p.request(name, served)
    order = []
    for _ in range(3):
      [name] = p.grants()
      order.append(name)
      p.release(name)
    self.assertEqual(order, ["c", "b", "a"])

  def test_an_account_back_from_idle_holds_no_credit_and_a_waiting_one_keeps_its_place(self):
    p = Local(self)
    for name in ("busy", "idle", "behind"):
      p.add(name)
    ready, service = {"busy": ["busy"], "behind": ["behind"]}, {"busy": 100, "behind": 130}
    _, lifts, floor = grant(p.s, ready, service, 0)
    self.assertEqual((lifts, floor), ({}, 100))  # nobody waiting is changed, and the floor follows the least served
    ready["idle"] = ["idle"]  # it has had no service at all
    made, lifts, floor = grant(p.s, ready, service | {"idle": 0}, floor)
    self.assertEqual((lifts, floor), ({"idle": 100}, 100))  # so it joins beside the least served, not ahead of everyone
    self.assertEqual(list(made), ["busy"])
    self.assertEqual(grant(p.s, {"idle": ["idle"]}, {"idle": 0}, 100)[1], {"idle": 100})  # even with no one else left

  def test_workers_with_known_budgets_share_but_unknown_budgets_wait(self):
    p = Local(self)
    for name in ("a", "b", "c"):
      p.add(name, sampler=True)
      p.request(name)
    p.add("trainer")
    p.request("trainer")
    self.assertEqual(p.grants(), ["a", "b"])
    p.release("a")
    self.assertEqual(p.grants(), ["c"])
    p.release("b")
    self.assertEqual(p.grants(), [])
    p.release("c")
    self.assertEqual(p.grants(), ["trainer"])

  def test_unknown_gpu_memory_never_allows_concurrent_grants(self):
    p = Local(self, memory=0)
    for name in ("a", "b"):
      p.add(name, sampler=True)
      p.request(name)
    self.assertEqual(p.grants(), ["a"])

  def test_a_wide_grant_is_atomic_and_cannot_be_bypassed_on_either_gpu(self):
    p = Local(self, 2)
    p.add("holder", at=(0,))
    p.add("wide", at=(0, 1), width=2)
    p.add("later", at=(1,))
    for served, name in enumerate(("holder", "wide", "later")):
      p.request(name, served)
    self.assertEqual(p.grants(), ["holder"])
    self.assertEqual(p.grants(), [])  # GPU 1 is free, but it is the wide worker's next
    p.release("holder")
    self.assertEqual(p.grants(), ["wide"])
    p.release("wide")
    self.assertEqual(p.grants(), ["later"])

  def test_different_gpus_are_granted_together(self):
    p = Local(self, 2)
    for name, gpu in (("a", 0), ("b", 1)):
      p.add(name, at=(gpu,))
      p.request(name)
    self.assertEqual(p.grants(), ["a", "b"])

  def test_a_grant_outlives_its_worker_until_the_node_gives_it_back(self):
    p = Local(self)
    for name in ("gone", "next"):
      p.add(name)
    p.request("gone")
    self.assertEqual(p.grants(), ["gone"])
    p.drop("gone")  # deleted, but its process is still on the GPU
    p.request("next")
    self.assertEqual(p.grants(), [])
    p.release("gone")
    self.assertEqual(p.grants(), ["next"])

  def test_random_requests_and_releases_never_overcommit_a_gpu(self):
    for seed in range(300):
      rng, p = random.Random(seed), Local(self, 4)
      for i in range(10):
        width = rng.choice([1, 1, 2, 4])
        first = rng.randrange(5 - width)
        p.add(str(i), at=tuple(range(first, first + width)), width=width, sampler=rng.random() < 0.5)
      for _ in range(100):
        name = rng.choice(list(p.s.workers))
        if name in p.s.granted:
          p.release(name)
        else:
          p.request(name, rng.randrange(5))
        p.grants()


if __name__ == "__main__":
  unittest.main()
