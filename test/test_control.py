import random
import unittest

from tinyrl import schedule
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
    """A pool with a run, placed where the test says rather than where decide would, and its process started."""
    self.append("run_created", {"run": f"{name}-run", "pool": name})
    self.append("placed", {"pool": name, "node": node, "gpus": list(gpus), "memory": memory, "env": {}, "rule": "test"})
    self("started", pool=name, node=node)


GRANTED, IDLE, QUANTUM = "turn.granted", "turn.idle_holder_yields", "turn.quantum_expired"
PLACED, EMPTIED = "place.least_loaded_node", "place.emptied"
MIB = 2**20


class TestPlacement(unittest.TestCase):
  def test_a_pool_with_runs_is_placed_and_an_emptied_one_unplaced(self):
    log = Log(self)
    log.node(8)
    self.assertEqual(log("run_created", run="r1", pool="lora:a:0"), [("placed", "lora:a:0", PLACED)])
    self.assertEqual(log("run_created", run="r2", pool="lora:a:0"), [])
    self.assertEqual(log("run_deleted", run="r1", pool="lora:a:0"), [])
    self.assertEqual(log("run_deleted", run="r2", pool="lora:a:0"), [("unplaced", "lora:a:0", EMPTIED)])

  def test_a_dedicated_sampler_cools_with_its_run(self):
    log = Log(self)
    log.node(8)
    log("run_created", run="r", pool="r")
    self.assertEqual(log("sampler_warmed", pool="sampler:fft:r"), [("placed", "sampler:fft:r", PLACED)])
    self.assertEqual(
      log("run_deleted", run="r", pool="r"), [("unplaced", "r", EMPTIED), ("unplaced", "sampler:fft:r", EMPTIED)]
    )

  def test_pools_go_to_the_node_with_the_fewest_pools_per_gpu(self):
    log = Log(self)
    log.node(1, node="a")
    log.node(2, node="b")
    for run in ("r1", "r2", "r3"):
      log("run_created", run=run, pool=run)
    self.assertEqual([log.s.pools[run].node for run in ("r1", "r2", "r3")], ["a", "b", "b"])

  def test_a_lost_node_s_pools_move_to_a_live_one(self):
    log = Log(self)
    log.node(1, node="a")
    log("run_created", run="r", pool="r")
    log("started", pool="r", node="a")
    log.node(1, node="b")
    self.assertEqual(log("node_lost", node="a"), [("placed", "r", PLACED)])
    self.assertEqual((log.s.pools["r"].node, log.s.pools["r"].running), ("b", False))

  def test_each_trainer_gets_the_least_used_gpu(self):
    log = Log(self)
    log.node(8)
    for name in ("lora:a:0", "lora:b:0", "lora:c:0"):
      log("run_created", run=name, pool=name)
    self.assertEqual([log.s.pools[n].gpus for n in ("lora:a:0", "lora:b:0", "lora:c:0")], [(0,), (1,), (0,)])

  def test_a_wide_pool_gets_an_aligned_block_of_the_least_used_gpus(self):
    running = []
    for name in ("lora:a:0", "lora:tp:0", "lora:b:0"):
      running.append(schedule.RunningPool(name, schedule.choose_gpus(name, 2, 16, running).choice))
    self.assertEqual([r.gpus for r in running], [[0, 1], [2, 3], [0, 1]])

  def test_shared_fft_samplers_get_a_memory_budget_on_large_gpus(self):
    log = Log(self)
    log.node(8, memory=81559 * MIB)
    log("sampler_warmed", pool="sampler:fft:Qwen")
    log("sampler_warmed", pool="sampler:fft:Gemma")
    log("sampler_warmed", pool="sampler:Qwen")
    log("sampler_warmed", pool="sampler:fft:Llama", exclusive=True)
    budget = {"TINYRL_WARM_SAMPLER": "1", "TINYRL_GPU_MEM": "0.42", "TINYRL_KV_BYTES": str(16 * 1024**3)}
    alone = {"TINYRL_WARM_SAMPLER": "1", "TINYRL_KV_BYTES": str(24 * 1024**3)}
    placed = {name: (p.gpus, p.env) for name, p in log.s.pools.items()}
    self.assertEqual(
      placed,
      {
        "sampler:fft:Qwen": ((6,), budget),
        "sampler:fft:Gemma": ((6,), budget),
        "sampler:Qwen": ((4,), alone),
        "sampler:fft:Llama": ((7,), alone),
      },
    )
    self.assertEqual(log.s.pools["sampler:fft:Qwen"].memory, int(0.42 * 81559 * MIB))
    self.assertEqual(schedule.settings("sampler:fft:Qwen", 8, 23034), {"TINYRL_WARM_SAMPLER": "1"})
    self.assertEqual(schedule.settings("sampler:Qwen", 1, 81559), {"TINYRL_KV_BYTES": str(24 * 1024**3)})


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
    self.assertEqual([(p.running, p.since, p.asked) for p in log.s.pools.values()], [(False, None, 0)] * 2)
    self.assertEqual({p.node for p in log.s.pools.values()}, {"n"})  # still placed, so its node daemon restarts them


class TestLog(unittest.TestCase):
  def random_log(self, rng: random.Random) -> Log:
    """Random facts, faults included: pools restarting elsewhere, dead workers that never release, nodes restarting
    and lost, runs coming and going."""
    log = Log(self)
    nodes = {f"n{i}": rng.choice([1, 2, 4, 8]) for i in range(rng.choice([1, 1, 2, 4]))}
    for node, gpus in nodes.items():
      log.node(gpus, node=node)
    names = [f"p{i}" for i in range(rng.randint(1, 12))]
    for _ in range(rng.randint(20, 200)):
      name, roll = rng.choice(names), rng.random()
      if roll < 0.15 and log.s.nodes:
        node = rng.choice(sorted(log.s.nodes))
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
      elif roll < 0.84:
        log("node_lost", node=rng.choice(sorted(nodes)))
      elif roll < 0.88:
        log("run_created", run=f"r{rng.randint(0, 20)}", pool=name, exclusive=rng.random() < 0.2)
      elif roll < 0.92:
        log("run_deleted", run=f"r{rng.randint(0, 20)}", pool=name)
      elif roll < 0.94:
        log("sampler_warmed", pool=f"sampler:{name}")
      elif roll < 0.97 and (p := log.s.pools.get(name)) and p.node:
        log("started", pool=name, node=p.node)
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
