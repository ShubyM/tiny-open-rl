import random
import unittest

from tinyrl.control import RULES, Event, Policy, State, check, decide, fold, to_json

GIB = 1024**3


class Log:
  """Appends facts the way workers and node daemons do, then decides the way the scheduler does, folding each decision
  back as it would read it from the log. Checks the invariants after every entry, and that nothing is decided twice."""

  def __init__(self, test: unittest.TestCase, policy: Policy = Policy()):
    self.test, self.s, self.t, self.entries = test, State(policy=policy), 0.0, []

  def append(self, kind: str, body: dict) -> None:
    self.entries.append(Event(str(len(self.entries)), self.t, kind, body))
    self.s = fold(self.s, self.entries[-1])
    self.test.assertEqual(check(self.s), [])

  def __call__(self, kind: str, /, at: float | None = None, **body) -> list[tuple[str, str, str]]:
    self.t = self.t if at is None else at
    self.append(kind, body)
    return self.tick()

  def tick(self, at: float | None = None) -> list[tuple[str, str, str]]:
    """What the scheduler decides at time `at`, as (kind, pool, rule)."""
    self.t = self.t if at is None else at
    decided = decide(self.s, self.t)
    for kind, body in decided:
      self.test.assertIn(body["rule"], RULES)
      self.append(kind, body)
    self.test.assertEqual(decide(self.s, self.t), [])
    return [(kind, body["pool"], body["rule"]) for kind, body in decided]

  def node(self, gpus: int, memory: int = 80 * GIB, node: str = "n") -> None:
    self("node_up", node=node, gpus=gpus, memory=memory)

  def pool(self, name: str, gpus: tuple[int, ...] = (0,), memory: int = 0, node: str = "n") -> None:
    self("started", pool=name, node=node, gpus=list(gpus), memory=memory)


GRANTED, IDLE, QUANTUM = "turn.granted", "turn.idle_holder_yields", "turn.quantum_expired"


class TestTurns(unittest.TestCase):
  def one_gpu(self, *names: str, policy: Policy = Policy()) -> Log:
    log = Log(self, policy)
    log.node(1)
    for name in names:
      log.pool(name)
    return log

  def test_waiters_get_the_gpu_in_first_ask_order(self):
    log = self.one_gpu("a", "b", "c")
    self.assertEqual(log("wants_gpu", pool="a"), [("granted", "a", GRANTED)])
    log("wants_gpu", pool="c")
    log("wants_gpu", pool="b")
    order = ["a"]
    for _ in range(2):
      self.assertEqual(log("idle", pool=order[-1]), [("revoked", order[-1], IDLE)])
      order += [pool for kind, pool, _ in log("released", pool=order[-1]) if kind == "granted"]
    self.assertEqual(order, ["a", "c", "b"])

  def test_one_pass_never_grants_two_pools_that_do_not_fit(self):
    log = self.one_gpu("a", "b")
    log.append("wants_gpu", {"pool": "a"})
    log.append("wants_gpu", {"pool": "b"})
    self.assertEqual(log.tick(), [("granted", "a", GRANTED)])

  def test_a_holder_with_nobody_waiting_keeps_the_gpu(self):
    log = self.one_gpu("a")
    log("wants_gpu", pool="a")
    log("idle", pool="a")
    for t in range(10, 300, 10):
      self.assertEqual(log.tick(at=t), [])

  def test_an_idle_holder_yields_as_soon_as_someone_waits(self):
    log = self.one_gpu("a", "b")
    log("wants_gpu", pool="a")
    log("idle", at=1, pool="a")
    self.assertEqual(log("wants_gpu", at=2, pool="b"), [("revoked", "a", IDLE)])

  def test_a_busy_holder_is_revoked_after_its_quantum(self):
    log = self.one_gpu("a", "b")
    log("wants_gpu", pool="a")
    self.assertEqual(log("wants_gpu", at=1, pool="b"), [])
    self.assertEqual(log.tick(at=29.9), [])
    self.assertEqual(log.tick(at=30), [("revoked", "a", QUANTUM)])

  def test_engine_boot_does_not_count_toward_the_quantum(self):
    log = self.one_gpu("a", "b")
    log("wants_gpu", pool="a")
    log("wants_gpu", at=1, pool="b")
    self.assertEqual(log("batch_done", at=35, pool="a", boot=35.0, seconds=36.0), [])
    self.assertEqual(log.tick(at=64.9), [])
    self.assertEqual(log.tick(at=65), [("revoked", "a", QUANTUM)])

  def test_pools_that_fit_together_hold_a_gpu_at_once_and_the_rest_take_turns(self):
    log = Log(self)
    log.node(1, memory=80 * GIB)
    for name, memory in (("s1", 33 * GIB), ("s2", 33 * GIB), ("t", 50 * GIB)):
      log.pool(name, memory=memory)
    self.assertEqual(log("wants_gpu", pool="s1"), [("granted", "s1", GRANTED)])
    self.assertEqual(log("wants_gpu", pool="s2"), [("granted", "s2", GRANTED)])
    self.assertEqual(log("wants_gpu", pool="t"), [])
    self.assertEqual(log("idle", at=5, pool="s1"), [("revoked", "s1", IDLE)])
    self.assertEqual(log("released", at=6, pool="s1"), [])  # t still doesn't fit next to s2
    self.assertEqual(log("idle", at=7, pool="s2"), [("revoked", "s2", IDLE)])
    self.assertEqual(log("released", at=8, pool="s2"), [("granted", "t", GRANTED)])

  def test_a_wide_pool_is_granted_only_with_every_gpu(self):
    log = Log(self)
    log.node(2)
    log.pool("w", gpus=(0, 1))
    log.pool("x")
    log("wants_gpu", pool="x")
    self.assertEqual(log("wants_gpu", pool="w"), [])
    log("idle", pool="x")
    self.assertEqual(log("released", pool="x"), [("granted", "w", GRANTED)])

  def test_a_dead_worker_loses_its_turn_and_the_next_in_line_gets_it(self):
    log = self.one_gpu("a", "b")
    log("wants_gpu", pool="a")
    log("wants_gpu", pool="b")
    self.assertEqual(log("exited", pool="a"), [("granted", "b", GRANTED)])
    log.pool("a")
    self.assertEqual(log("wants_gpu", pool="a"), [])

  def test_a_restarted_node_daemon_takes_its_processes_turns_with_it(self):
    log = self.one_gpu("a", "b")
    log("wants_gpu", pool="a")
    log("wants_gpu", pool="b")
    log.node(1)
    self.assertEqual(log.s.pools, {})


class TestLog(unittest.TestCase):
  def random_log(self, rng: random.Random) -> Log:
    """Random facts, faults included: pools restarting elsewhere, dead workers that never release, node restarts."""
    log = Log(self)
    nodes = {f"n{i}": rng.choice([1, 2, 4, 8]) for i in range(rng.choice([1, 1, 2, 4]))}
    for node, gpus in nodes.items():
      log.node(gpus, node=node)
    names = [f"p{i}" for i in range(rng.randint(1, 12))]
    for _ in range(rng.randint(20, 200)):
      name, roll = rng.choice(names), rng.random()
      if roll < 0.15:
        node = rng.choice(sorted(nodes))
        width = rng.choice([w for w in (1, 2, 4) if w <= nodes[node]])
        first = rng.randrange(0, nodes[node] - width + 1, width)
        log.pool(name, range(first, first + width), rng.choice([0, 20, 40, 90]) * GIB, node)
      elif roll < 0.4:
        log("wants_gpu", pool=name)
      elif roll < 0.55:
        log("batch_done", at=log.t + rng.random() * 20, pool=name, boot=rng.choice([0.0, 0.0, 30.0]), seconds=1.0)
      elif roll < 0.65:
        log("released", pool=name)
      elif roll < 0.75:
        log.tick(at=log.t + rng.random() * 40)
      elif roll < 0.8:
        log("exited", pool=name)
      elif roll < 0.82:
        node = rng.choice(sorted(nodes))
        log.node(nodes[node], node=node)
      else:
        log("idle", pool=name)
    return log

  def test_random_logs_keep_every_invariant(self):
    for seed in range(300):
      self.random_log(random.Random(seed))

  def test_replay_rebuilds_the_same_state(self):
    log = self.random_log(random.Random(7))
    s = State()
    for e in log.entries:
      s = fold(s, e)
    self.assertEqual(to_json(s), to_json(log.s))

  def test_every_waiter_gets_a_turn_within_its_place_in_line_times_the_quantum(self):
    log = Log(self, Policy(quantum=30.0))
    log.node(1)
    names = [f"p{i}" for i in range(6)]
    for name in names:
      log.pool(name)
    asked, granted = {}, {}

    def note(decisions: list) -> list:
      for kind, pool, _ in decisions:
        if kind == "granted":
          granted.setdefault(pool, log.t)
      return decisions

    for name in names:
      asked[name] = log.t = log.t + 1
      note(log("wants_gpu", pool=name))
    for _ in range(400):  # a bound, so a broken rule fails the test where it would wait forever
      for kind, pool, _ in note(log.tick(at=log.t + 1)):
        if kind == "revoked":
          note(log("released", at=log.t + 1, pool=pool))
    for place, name in enumerate(names):
      self.assertLessEqual(granted[name] - asked[name], (place + 1) * 32.0, name)


if __name__ == "__main__":
  unittest.main()
