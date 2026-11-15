import dataclasses
import unittest

from tinyrl.bus import Role
from tinyrl.control.scheduler import TURN, Assignment, Job, WorkerSpec, bundles, decide, from_table


class Fleet:
  def __init__(self, **nodes):
    self.s = Assignment(nodes={n: {"gpus": g, "memory": 80 * 2**30} for n, g in nodes.items()})

  def step(self):
    self.s = decide(self.s)
    return self.s

  def add(self, run, *, account=None, **spec):
    self.s.jobs[run] = Job(WorkerSpec(**spec), account=account or run)
    self.step()
    return self.s.jobs[run].worker

  def report(self, name, **fields):
    w = self.s.workers[name]
    owner = fields.pop("owner", self.s.owners.get(name, name + "-owner"))
    self.s.owners[name] = owner
    self.s.observations[name] = {
      "node": w.node,
      "owner": owner,
      "revision": w.revision,
      "gpus": list(w.gpus),
      "released": False,
      "closed": False,
      "resident": dict(w.members),
      "usage": {},
      "served": {},
      "ready": list(w.members),
      **fields,
    }

  def boot(self, name):
    self.report(name, released=True, gpus=[], resident={})
    self.step()
    self.report(name)
    self.step()


class TestFleet(unittest.TestCase):
  def test_reuse_waits_for_current_owner_and_release_revision(self):
    f = Fleet(n=1)
    a = f.add("a", role=Role.SAMPLER, base="first")
    f.boot(a)
    b = f.add("b", role=Role.SAMPLER, base="second")
    f.boot(b)
    self.assertFalse(f.s.workers[b].budgets)  # separate engines never share an active GPU
    f.report(a, usage={"a": 3}, ready=[])  # staging releases its turn even before the token ceiling
    previous_revision = f.s.workers[a].revision
    f.step()
    self.assertFalse(f.s.workers[a].budgets)
    for stale in ({"revision": previous_revision}, {"owner": "old-owner"}):
      f.report(a, released=True, **stale)
      f.s.owners[a] = a + "-owner"
      f.step()
      self.assertFalse(f.s.workers[b].budgets)
    f.report(a, released=True)
    f.step()
    self.assertEqual(f.s.workers[b].gpus, (0,))
    self.assertTrue(f.s.workers[b].budgets)

  def test_owner_restart_resets_counter_baseline_and_grant_ceiling(self):
    f = Fleet(n=1)
    w = f.add("r")
    revision = f.s.workers[w].revision
    f.report(w, released=True, closed=True, resident={})
    del f.s.owners[w]
    f.step()
    self.assertGreater(f.s.workers[w].revision, revision)  # a crash before the first owner binding must retry
    f.boot(w)
    f.report(w, usage={"r": 100}, served={"r": 5 * TURN})
    f.step()
    gen = f.s.jobs["r"].generation
    f.report(w, owner="replacement", released=True, served={}, usage={"r": 7})
    f.step()
    self.assertEqual(f.s.service["r"], 107)
    self.assertEqual(f.s.jobs["r"].generation, gen + 1)
    self.assertEqual(f.s.workers[w].owner, "replacement")
    self.assertEqual(f.s.workers[w].budgets, {"r": TURN})

  def test_trainer_and_sampler_share_service_and_idle_accounts_return_at_floor(self):
    f = Fleet(n=2)
    train = f.add("train")
    sample = f.add("sample", role=Role.SAMPLER, account="train", lora=True)
    other = f.add("other")
    for w in (train, sample, other):
      f.boot(w)
    f.report(train, usage={"train": 3})
    f.report(sample, usage={"train": 7})
    f.step()
    self.assertEqual(f.s.service["train"], 10)
    for w in (train, sample):
      f.report(w, ready=[])
    f.s.service["other"] = 20
    f.step()
    f.report(train)
    f.step()
    self.assertEqual(f.s.service["train"], 20)
    for w in (train, other):
      f.report(w, ready=[])
    f.step()
    f.boot(f.add("late"))
    self.assertEqual(f.s.service["late"], 20)  # a wholly idle fleet retains its fairness floor

  def test_waiting_gang_protects_its_bundle_without_preempting_unfinished_turns(self):
    f = Fleet(n=2)
    a, b = f.add("a"), f.add("b")
    for w in (a, b):
      f.boot(w)
    f.report(a, usage={"a": 5})
    f.report(b, usage={"b": 5})
    wide = f.add("wide", gpus=2)
    f.boot(wide)
    self.assertTrue(f.s.workers[a].budgets and f.s.workers[b].budgets)
    self.assertFalse(f.s.workers[wide].budgets)
    for w, r in ((a, "a"), (b, "b")):
      f.report(w, served={r: TURN}, usage={r: 10})
    f.step()
    f.report(a, released=True, usage={"a": 10})
    f.step()
    self.assertFalse(f.s.workers[a].budgets or f.s.workers[wide].budgets)
    f.report(b, released=True, usage={"b": 10})
    f.step()
    self.assertEqual(f.s.workers[wide].gpus, (0, 1))
    f.boot(wide)
    f.report(wide, served={"wide": TURN}, usage={"wide": 30})
    f.step()
    f.report(wide, released=True, usage={"wide": 30})
    f.step()
    for w, r in ((a, "a"), (b, "b")):
      f.report(w, served={r: TURN}, usage={r: 10})
    f.step()
    self.assertTrue(f.s.workers[a].budgets and f.s.workers[b].budgets)
    self.assertFalse(f.s.workers[wide].budgets)

  def test_shared_job_move_waits_for_residency_ack_then_fences_only_that_job(self):
    f = Fleet(a=1, b=1)
    w = f.add("x", lora=True)
    self.assertEqual(f.add("y", lora=True), w)
    f.boot(w)
    old_generation = f.s.jobs["x"].generation
    old_revision = f.s.workers[w].revision
    f.s.jobs["x"].target_node = "b"
    f.step()
    self.assertNotIn("x", f.s.workers[w].members)
    f.report(w, revision=old_revision, resident={})
    f.step()
    self.assertEqual(f.s.jobs["x"].worker, w)
    f.report(w, resident={"y": f.s.jobs["y"].generation})
    f.step()
    moved = f.s.jobs["x"].worker
    self.assertNotEqual(moved, w)
    self.assertEqual(f.s.workers[moved].node, "b")
    self.assertEqual(f.s.jobs["x"].generation, old_generation + 1)
    self.assertEqual(f.s.jobs["y"].generation, 1)
    del f.s.nodes["a"]
    f.step()
    self.assertEqual(f.s.jobs["y"].generation, 2)
    f.report(w, usage={"x": 9, "y": 2})
    del f.s.jobs["x"]
    f.step()
    self.assertNotIn("x", f.s.service)  # an old owner report cannot retain a finished account

  def test_exclusive_idle_capacity_and_physical_bundles(self):
    f = Fleet(n=2)
    ex = f.add("exclusive", exclusive=True)
    a, b = f.add("a"), f.add("b")
    f.boot(a)
    f.boot(b)
    self.assertEqual(f.s.workers[ex].gpus, (0,))
    self.assertEqual(f.s.workers[a].gpus, (1,))
    self.assertFalse(f.s.workers[b].gpus)
    self.assertEqual(bundles({"gpus": 4, "bundles": [[0, 1], [2, 3]]}, 1), [(0,), (1,), (2,), (3,)])
    self.assertEqual(bundles({"gpus": 4, "bundles": [[0, 1], [2, 3]]}, 2), [(0, 1), (2, 3)])
    table = dataclasses.asdict(f.s)
    table["runs"] = {
      r: (
        {
          "role": j.spec.role,
          "base": j.spec.base,
          "rank": 8 if j.spec.lora else 0,
          "exclusive": j.spec.exclusive,
          "gpus": j.spec.gpus,
        },
        j.worker,
        {"generation": j.generation, "target_node": j.target_node},
      )
      for r, j in f.s.jobs.items()
    }
    self.assertEqual(from_table(table), f.s)
    f = Fleet(n=3)
    ordinary = f.add("ordinary")
    f.boot(ordinary)
    f.report(ordinary, ready=[])
    f.step()
    f.report(ordinary, ready=[], released=True)
    f.step()
    owner = f.add("owner", exclusive=True)
    sampler = f.add("owned", role=Role.SAMPLER, exclusive=True, train_run="owner", account="owner")
    self.assertEqual(f.s.workers[owner].gpus, (1,))
    self.assertEqual(f.s.workers[sampler].gpus, (1,))
    f.report(ordinary, released=True, gpus=[0])
    f.step()
    self.assertEqual(f.s.workers[ordinary].gpus, (0,))
    self.assertTrue(f.s.workers[ordinary].budgets)  # exclusive admission cannot strand a foreign parked cache

  def test_exclusive_trainer_and_its_sampler_alternate_on_the_reserved_gpu(self):
    f = Fleet(n=1)
    train = f.add("train", exclusive=True)
    f.boot(train)
    sample = f.add("sample", role=Role.SAMPLER, exclusive=True, train_run="train", account="train")
    other = f.add("other")
    f.boot(sample)
    f.boot(other)
    for previous, following, run, cost in ((train, sample, "train", 10), (sample, train, "sample", 12)):
      f.report(previous, served={run: TURN}, usage={"train": cost})
      f.step()
      self.assertFalse(f.s.workers[previous].budgets)
      self.assertFalse(f.s.workers[following].budgets)
      f.report(previous, released=True, served={run: TURN}, usage={"train": cost})
      f.step()
      self.assertEqual(f.s.workers[following].gpus, (0,))
      self.assertTrue(f.s.workers[following].budgets)
      self.assertFalse(f.s.workers[other].budgets)
      if following == sample:
        f.boot(sample)

  def test_parked_gang_export_gets_a_grant_before_ownership_moves(self):
    f = Fleet(a=2, b=2)
    w = f.add("move", gpus=2)
    f.boot(w)
    f.report(w, ready=[])
    f.step()
    f.report(w, released=True, ready=[])
    f.step()
    f.report(w, released=True, gpus=[0, 1], ready=[])
    f.s.jobs["move"].target_node = "b"
    f.step()
    self.assertNotIn("move", f.s.workers[w].members)
    self.assertIn("move", f.s.workers[w].budgets)
    self.assertEqual(f.s.workers[w].gpus, (0, 1))
    self.assertEqual(f.s.jobs["move"].worker, w)
    f.report(w, resident={})
    f.step()
    self.assertEqual(f.s.workers[f.s.jobs["move"].worker].node, "b")


if __name__ == "__main__":
  unittest.main()
