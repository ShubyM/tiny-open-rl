import copy
import unittest

from tinyrl.control.scheduler import TURN, account, bundles, decide


class Fleet:
  def __init__(self, **nodes):
    self.s = {k: {} for k in ("runs", "claims", "turns", "observations", "service", "moves", "reservations", "fair")}
    self.s["live"] = []
    self.s["nodes"] = {n: {"gpus": g} for n, g in nodes.items()}

  def step(self):
    before = copy.deepcopy(self.s)
    self.s = decide(self.s)
    for t, grant in before["turns"].items():
      assert self.s["turns"][t] == grant
    return self.s

  def add(self, run, **info):
    self.s["runs"][run] = {
      "role": "trainer",
      "base": "model",
      "rank": 0,
      "gpus": 1,
      "state": {"created": "1-0", "mutation": "1-0", "working": None, "durable": None},
      **info,
    }
    self.step()
    return self.s["claims"].get(run, {}).get("engine")

  def report(self, engine, **fields):
    previous = self.s["observations"].get(engine, {})
    owned = {r: c for r, c in self.s["claims"].items() if c["engine"] == engine}
    node = next(iter(owned.values()))["node"] if owned else previous["node"]
    self.s["live"] = sorted(set(self.s["live"]) | {engine})
    self.s["observations"][engine] = {
      "node": node,
      "gpus": previous.get("gpus", []),
      "ready": list(owned),
      "closed": False,
      **fields,
    }

  def boot(self, engine):
    self.report(engine)
    self.step()
    if turn := self.turn(engine):
      self.s["observations"][engine]["gpus"] = list(turn[1]["gpus"])

  def turn(self, engine):
    return next(((t, v) for t, v in self.s["turns"].items() if v["engine"] == engine), None)

  def release(self, engine, **usage):
    t, _ = self.turn(engine)
    del self.s["turns"][t]  # an explicit release receipt; expiry alone never does this
    for a, value in usage.items():
      self.s["service"][a] = self.s["service"].get(a, 0) + value


class TestFleet(unittest.TestCase):
  def test_preparation_precedes_grant_and_pending_launch_is_not_dead(self):
    f = Fleet(n=1)
    e = f.add("r")
    claim = dict(f.s["claims"]["r"])
    self.assertFalse(f.s["turns"])
    f.step()
    self.assertEqual(f.s["claims"]["r"], claim)
    f.report(e, ready=[])
    f.step()
    self.assertFalse(f.s["turns"])
    f.report(e)
    f.step()
    _, turn = f.turn(e)
    self.assertEqual(turn["work"], {"r": [claim["token"], TURN]})
    self.assertEqual(turn["gpus"], [0])
    for observed in (False, True):
      cold = Fleet(n=1)
      e = cold.add("pending")
      if observed:
        cold.report(e, ready=[])
      cold.s["nodes"]["n"]["draining"] = True
      cold.step()
      self.assertFalse(cold.s["claims"] or cold.s["turns"])

  def test_engine_loss_changes_claim_but_never_recycles_a_turn(self):
    f = Fleet(n=1)
    e = f.add("r")
    f.boot(e)
    old_claim, old_turn = dict(f.s["claims"]["r"]), f.turn(e)
    f.s["live"].remove(e)
    f.step()
    new = f.s["claims"]["r"]
    self.assertNotEqual(new["engine"], e)
    self.assertNotEqual(new["token"], old_claim["token"])
    f.boot(new["engine"])
    self.assertEqual(f.s["turns"], dict([old_turn]))
    f.s["nodes"].clear()
    f.step()
    self.assertFalse(f.s["claims"])
    self.assertEqual(f.s["turns"], dict([old_turn]))
    f.s["nodes"]["n"] = {"gpus": 1}
    f.step()
    replacement = f.s["claims"]["r"]["engine"]
    f.boot(replacement)
    self.assertIsNone(f.turn(replacement))
    f.s["nodes"]["fresh"] = {"gpus": 1}
    f.s["moves"]["r"] = "fresh"
    f.step()
    replacement = f.s["claims"]["r"]["engine"]
    f.add("other")
    f.s["moves"]["other"] = "fresh"
    f.step()
    f.report(replacement)
    f.report(f.s["claims"]["other"]["engine"])
    f.s["service"]["other"] = 100
    f.step()
    self.assertIsNotNone(f.turn(replacement))  # a quarantined old turn is not useful service debt
    self.assertEqual(f.s["turns"][old_turn[0]], old_turn[1])
    f.release(e)
    f.s["observations"][e]["closed"] = True
    f.step()
    self.assertIsNotNone(f.turn(replacement))
    self.assertNotIn(e, f.s["observations"])

  def test_use_capacity_then_share_compatible_lora_engine(self):
    f = Fleet(n=2)
    a, b = f.add("a", rank=8), f.add("b", rank=16)
    c = f.add("c", rank=8)
    self.assertNotEqual(a, b)
    self.assertIn(c, (a, b))
    for e in (a, b):
      f.boot(e)
    self.assertEqual({tuple(t["gpus"]) for t in f.s["turns"].values()}, {(0,), (1,)})
    self.assertTrue(any(len(t["work"]) == 2 for t in f.s["turns"].values()))
    self.assertEqual(bundles({"gpus": 4, "bundles": [[0, 1], [2, 3]]}, 1), [(0,), (1,), (2,), (3,)])
    self.assertEqual(bundles({"gpus": 4, "bundles": [[0, 1], [2, 3]]}, 2), [(0, 1), (2, 3)])

  def test_outstanding_turns_prevent_one_account_from_taking_the_fleet(self):
    f = Fleet(n=2)
    train = f.add("train")
    sample = f.add("sample", role="sampler", train_run="train")
    other = f.add("other")
    for e in (train, sample, other):
      f.report(e)
    f.step()
    served = {account(r, f.s["runs"][r]) for t in f.s["turns"].values() for r in t["work"]}
    self.assertEqual(served, {"train", "other"})
    self.assertEqual(len(f.s["turns"]), 2)
    for e in (train, sample, other):
      if f.turn(e):
        f.release(e, **{"train" if e != other else "other": 1})
    f.report(other, ready=[])
    f.step()
    self.assertEqual({t["engine"] for t in f.s["turns"].values()}, {train, sample})

  def test_idle_fairness_floor_and_finished_account_cleanup(self):
    f = Fleet(n=1)
    a, b = f.add("a"), f.add("b")
    f.report(a)
    f.report(b)
    f.s["service"] = {"a": 10, "b": 20}
    f.step()
    f.report(a, ready=[])
    f.step()
    f.report(a)
    f.step()
    self.assertEqual(f.s["service"]["a"], 20)
    for e in (a, b):
      f.report(e, ready=[])
    f.step()
    late = f.add("late")
    f.report(late)
    f.step()
    self.assertEqual(f.s["service"]["late"], 20)
    del f.s["runs"]["b"]
    f.step()
    self.assertNotIn("b", f.s["service"])

  def test_waiting_gang_keeps_fair_position_until_old_turns_finish(self):
    f = Fleet(n=2)
    a, b = f.add("a"), f.add("b")
    f.boot(a)
    f.boot(b)
    wide = f.add("wide", gpus=2)
    f.boot(wide)
    self.assertIsNone(f.turn(wide))
    f.release(a, a=2)
    f.step()
    self.assertIsNone(f.turn(a))
    self.assertIsNone(f.turn(wide))
    f.release(b, b=2)
    f.step()
    self.assertEqual(f.turn(wide)[1]["gpus"], [0, 1])
    f.release(wide, wide=8)
    f.step()
    self.assertTrue(f.turn(a) and f.turn(b))

  def test_export_moves_one_claim_without_residency_absence_handshake(self):
    f = Fleet(a=1)
    e = f.add("x", rank=8)
    self.assertEqual(f.add("y", rank=8), e)
    f.boot(e)
    other_claim = dict(f.s["claims"]["y"])
    f.s["nodes"]["b"] = {"gpus": 1}
    f.s["moves"]["x"] = "b"
    f.step()
    self.assertGreater(f.turn(e)[1]["work"]["x"][1], 0)
    f.release(e)
    f.step()
    self.assertEqual(f.turn(e)[1]["work"]["x"][1], 0)
    del f.s["claims"]["x"]  # Bus.export publishes its checkpoint and clears only this claim
    f.s["runs"]["x"]["state"]["working"] = {"version": "v2", "boundary": "5-0"}
    f.step()
    self.assertEqual(f.s["claims"]["x"]["node"], "b")
    self.assertEqual(f.s["claims"]["y"], other_claim)
    self.assertNotIn("x", f.s["moves"])
    self.assertIn("x", f.s["observations"][e]["ready"])
    f.s["nodes"]["a"]["draining"] = True
    f.release(e)
    f.step()
    self.assertEqual(f.turn(e)[1]["work"]["y"][1], 0)

  def test_exclusive_capacity_is_for_account_and_does_not_steal_fixed_cache(self):
    f = Fleet(n=3)
    ordinary = f.add("ordinary")
    f.boot(ordinary)
    f.release(ordinary)
    f.report(ordinary, ready=[])
    parent = f.add("parent", exclusive=True)
    sample = f.add("sample", exclusive=True, role="sampler", train_run="parent")
    self.assertEqual(f.s["reservations"], {"n": {"1": "parent"}})
    f.boot(parent)
    f.boot(sample)
    self.assertEqual(f.turn(parent)[1]["gpus"], [1])
    self.assertIsNone(f.turn(sample))
    f.release(parent, parent=10)
    f.step()
    self.assertEqual(f.turn(sample)[1]["gpus"], [1])
    self.assertIsNone(f.turn(parent))
    f.report(ordinary)
    f.step()
    self.assertEqual(f.turn(ordinary)[1]["gpus"], [0])
    g = Fleet(n=1)
    e = g.add("dirty")
    g.boot(e)
    g.release(e)
    g.report(e, ready=[])
    self.assertIsNone(g.add("dedicated", exclusive=True))
    g.s["live"].remove(e)
    g.step()
    self.assertIn("dedicated", g.s["claims"])


if __name__ == "__main__":
  unittest.main()
