import random
import unittest

from tinyrl.bus import Event
from tinyrl.scheduler import (
  QUANTUM,
  RULES,
  State,
  choose_pool,
  choose_sampler_pool,
  decide,
  fold,
  holders,
  parse_pool,
  settings,
)

GIB = 1024**3


def check(s: State) -> list[str]:
  """Every broken invariant of the scheduler's state, as one message each."""
  bad = []
  for node, (gpus, gpu) in sorted(s.nodes.items()):
    for k in [(node, i) for i in range(gpus)]:
      held = holders(s, k)
      memory = [s.pools[n].uses for n in held]
      if len(held) > 1 and not (gpu and all(memory) and sum(memory) <= gpu):
        bad.append(f"{k} has holders that do not fit together")
  for name, p in sorted(s.pools.items()):
    if p.node is not None and p.node not in s.nodes:
      bad.append(f"{name} is placed on {p.node}, which is not up")
    if any(node not in s.nodes for node, _ in p.on):
      bad.append(f"{name} runs on a node that is not up")
    if p.asked and p.since is not None:
      bad.append(f"{name} holds its GPUs and waits for them")
    if p.revoked and p.since is None:
      bad.append(f"{name} is revoked but holds nothing")
  return bad


class Log:
  """Appends facts the way workers and node daemons do, then decides the way the scheduler does, folding each decision
  back as it would read it from the log. Checks the invariants after every entry, and that nothing is decided twice."""

  def __init__(self, test: unittest.TestCase, quantum: float = QUANTUM):
    self.test, self.s, self.t, self.entries = test, State(quantum=quantum), 0.0, []

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
    self.assertEqual((log.s.pools["r"].node, log.s.pools["r"].on), ("b", ()))

  def test_each_pool_gets_an_empty_gpu_while_there_is_one(self):
    log = Log(self)
    log.node(8)
    for name in ("lora:a:0", "lora:b:0", "lora:c:0"):
      log("run_created", run=name, pool=name)
    self.assertEqual([log.s.pools[n].gpus for n in ("lora:a:0", "lora:b:0", "lora:c:0")], [(0,), (1,), (2,)])

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
        "sampler:fft:Qwen": ((0,), budget),
        "sampler:fft:Gemma": ((0,), budget),
        "sampler:Qwen": ((1,), alone),
        "sampler:fft:Llama": ((2,), alone),
      },
    )
    self.assertEqual(log.s.pools["sampler:fft:Qwen"].memory, int(0.42 * 81559 * MIB))
    self.assertEqual(settings("sampler:fft:Qwen", 8, 23034), {"TINYRL_WARM_SAMPLER": "1"})
    self.assertEqual(settings("sampler:Qwen", 1, 81559), {"TINYRL_KV_BYTES": str(24 * 1024**3)})


class TestTurns(unittest.TestCase):
  def one_gpu(self, *names: str) -> Log:
    log = Log(self)
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

  def test_an_unplaced_pool_keeps_its_turn_until_its_process_exits(self):
    log = self.one_gpu("a", "b")
    log("wants_gpu", pool="a")
    log("wants_gpu", pool="b")
    self.assertEqual(log("run_deleted", run="a-run", pool="a"), [("unplaced", "a", EMPTIED)])
    self.assertEqual(log("exited", pool="a"), [("granted", "b", GRANTED)])
    self.assertNotIn("a", log.s.pools)

  def test_a_moved_pool_keeps_its_old_gpus_until_its_process_exits(self):
    log = Log(self)
    log.node(2)
    log.pool("a", gpus=(0,))
    log.pool("b", gpus=(0,))
    log("wants_gpu", pool="a")
    log("wants_gpu", pool="b")
    log.append("placed", {"pool": "a", "node": "n", "gpus": [1], "memory": 0, "env": {}, "rule": "test"})
    self.assertEqual(log.tick(), [])  # a's process is still on GPU 0
    self.assertEqual(log("exited", pool="a"), [("granted", "b", GRANTED)])

  def test_a_restarted_node_daemon_takes_its_processes_turns_with_it(self):
    log = self.one_gpu("a", "b")
    log("wants_gpu", pool="a")
    log("wants_gpu", pool="b")
    log.node(1)
    self.assertEqual([(p.on, p.since, p.asked) for p in log.s.pools.values()], [((), None, 0)] * 2)
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
    self.assertEqual(s, log.s)

  def test_every_waiter_gets_a_turn_within_its_place_in_line_times_the_quantum(self):
    log = Log(self, quantum=30.0)
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


class TestPoolsAndGpus(unittest.TestCase):
  def test_a_pool_s_role_and_model_come_from_its_name(self):
    self.assertEqual(parse_pool("lora:base:0"), ("trainer", True, "base"))
    self.assertEqual(parse_pool("lora:org/model"), ("trainer", True, "org/model"))
    self.assertEqual(parse_pool("sampler:base"), ("sampler", True, "base"))
    self.assertEqual(parse_pool("run"), ("trainer", False, "run"))
    self.assertEqual(parse_pool("sampler:fft:run"), ("sampler", False, "run"))
    self.assertEqual(parse_pool("sampler:lora:run"), ("sampler", False, "run"))

  def test_choose_pool(self):
    shards = [("lora:b:0", None, 0, 13)]
    d = choose_pool("r", "b", 1, False, shards, 8, target_util=0.7, new_run_util=0.04)
    self.assertEqual(d["choice"], "lora:b:0")
    self.assertEqual(d["rule"], "reuse_shard_under_target")

    shards[0] = ("lora:b:0", None, 18, 18)
    d = choose_pool("r", "b", 1, False, shards, 8, target_util=0.7, new_run_util=0.04)
    self.assertEqual(d["choice"], "lora:b:1")
    self.assertEqual(d["rule"], "open_new_shard")

    shards = [("lora:b:0", 0.1, 5, 10)]
    d = choose_pool("r", "b", 1, False, shards, 8, target_util=0.7, new_run_util=0.04)
    self.assertEqual(d["choice"], "lora:b:0")
    self.assertEqual(d["per_run"], 0.02)

    d = choose_pool("r", "b", 0, False, shards, 8)
    self.assertEqual(d["choice"], "r")
    self.assertEqual(d["rule"], "dedicated_fft")

    d = choose_pool("r", "b", 1, True, shards, 8)
    self.assertEqual(d["choice"], "r")
    self.assertEqual(d["rule"], "dedicated_exclusive")

  def test_choose_sampler_pool(self):
    d = choose_sampler_pool("b", False, "r")
    self.assertEqual(d["choice"], "sampler:fft:b")
    self.assertEqual(d["rule"], "shared_fft_sampler")

    d = choose_sampler_pool("b", False, "r", True)
    self.assertEqual(d["choice"], "sampler:fft:r")
    self.assertEqual(d["rule"], "dedicated_fft_sampler")

    d = choose_sampler_pool("", False, "r")
    self.assertEqual(d["choice"], "sampler:fft:r")
    self.assertEqual(d["rule"], "dedicated_fft_sampler")

    d = choose_sampler_pool("b", True, "r", True)
    self.assertEqual(d["choice"], "sampler:lora:r")
    self.assertEqual(d["rule"], "dedicated_exclusive_sampler")

    d = choose_sampler_pool("b", True, "r", False)
    self.assertEqual(d["choice"], "sampler:b")
    self.assertEqual(d["rule"], "shared_lora_sampler")

  def test_a_trainer_takes_turns_with_the_least_busy_trainers(self):
    log = Log(self)
    log.node(2)
    for pool, busy in (("busy", 50.0), ("light", 5.0)):
      log("run_created", run=pool, pool=pool)
      log("batch_done", pool=pool, seconds=busy)
    log("run_created", run="t", pool="t")
    self.assertEqual(log.s.pools["t"].gpus, log.s.pools["light"].gpus)

  def test_a_sampler_moves_the_least_busy_trainers_aside_and_never_takes_turns(self):
    log = Log(self)
    log.node(2)
    for pool, busy in (("busy", 50.0), ("light", 5.0)):
      log("run_created", run=pool, pool=pool)
      log("batch_done", pool=pool, seconds=busy)
    decided = log("sampler_warmed", pool="sampler:b")
    self.assertEqual([(k, p) for k, p, _ in decided], [("placed", "light"), ("placed", "sampler:b")])
    self.assertEqual(log.s.pools["light"].gpus, log.s.pools["busy"].gpus)
    self.assertNotEqual(log.s.pools["sampler:b"].gpus, log.s.pools["busy"].gpus)

  def test_a_pool_with_a_memory_budget_fills_a_gpu_it_fits_on_before_an_empty_one(self):
    log = Log(self)
    log.node(2, memory=80 * GIB)
    log("run_created", run="t", pool="t")
    log("sampler_warmed", pool="sampler:fft:a")
    log("run_deleted", run="t", pool="t")
    log("exited", pool="t")
    log("sampler_warmed", pool="sampler:fft:b")
    self.assertEqual(log.s.pools["sampler:fft:b"].gpus, log.s.pools["sampler:fft:a"].gpus)

  def test_no_pool_fits_next_to_an_exclusive_one_whatever_their_budgets(self):
    log = Log(self)
    log.node(2, memory=80 * GIB)
    log.append("run_created", {"run": "x", "pool": "x", "exclusive": True})
    log.append("placed", {"pool": "x", "node": "n", "gpus": [1], "memory": 30 * GIB, "env": {}, "rule": "test"})
    log("sampler_warmed", pool="sampler:fft:a")
    self.assertEqual(log.s.pools["sampler:fft:a"].gpus, (0,))

  def test_a_sampler_moves_as_few_trainers_as_it_can(self):
    log = Log(self)
    log.node(2)
    for pool, gpu, busy in (("a", 0, 1.0), ("b", 0, 1.0), ("c", 1, 5.0)):
      log.pool(pool, gpus=(gpu,))
      log("batch_done", pool=pool, seconds=busy)
    log("sampler_warmed", pool="sampler:s")
    self.assertEqual((log.s.pools["sampler:s"].gpus, log.s.pools["c"].gpus), ((1,), (0,)))

  def test_an_exclusive_pool_gets_a_gpu_to_itself_and_others_stay_off_it(self):
    log = Log(self)
    log.node(2)
    log("run_created", run="x", pool="x", exclusive=True)
    log("run_created", run="a", pool="a")
    log("run_created", run="b", pool="b")
    self.assertEqual([log.s.pools[n].gpus for n in ("x", "a", "b")], [(0,), (1,), (1,)])

  def test_the_benchmark_s_arrivals_end_with_light_trainers_together_and_samplers_alone(self):
    log = Log(self)
    log.node(8, memory=81559 * MIB)
    trainers = {"x": 4.0, "lora:g:0": 7.0, "lora:q:0": 7.7, "q1": 1.7, "q2": 2.4, "g1": 4.8}
    for pool, busy in trainers.items():
      log("run_created", run=pool, pool=pool, exclusive=pool == "x")
      log("batch_done", pool=pool, seconds=busy)
    for pool in ("sampler:fft:q", "sampler:q", "sampler:fft:g", "sampler:fft:x", "sampler:g"):
      log("sampler_warmed", pool=pool, exclusive=pool == "sampler:fft:x")
    gpu = {name: p.gpus[0] for name, p in log.s.pools.items()}
    self.assertEqual(gpu["q1"], gpu["q2"])
    self.assertEqual(gpu["q1"], gpu["g1"])
    self.assertEqual(gpu["sampler:fft:q"], gpu["sampler:fft:g"])
    alone = [gpu[n] for n in ("x", "lora:g:0", "lora:q:0", "sampler:q", "sampler:fft:x", "sampler:g")]
    self.assertEqual(sorted(alone + [gpu["q1"], gpu["sampler:fft:q"]]), list(range(8)))

  def test_random_arrivals_never_make_a_sampler_take_turns_while_trainers_could_move(self):
    rnd = random.Random(42)
    for _ in range(200):
      log, gpus = Log(self), rnd.choice([2, 4, 8])
      log.node(gpus)
      names = [f"lora:b-{i}:0" for i in range(3)] + [f"ts-{i}" for i in range(4)] + ["sampler:b-0", "sampler:c-0"]
      for name in rnd.sample(names, rnd.randint(1, len(names))):
        on = {g: [(q, p) for q, p in log.s.pools.items() if p.gpus == (g,)] for g in range(gpus)}
        movable = [g for g, ps in on.items() if ps and all(q.startswith(("lora:", "ts-")) for q, _ in ps)]
        before = len(log.entries)
        log("run_created" if not name.startswith("sampler:") else "sampler_warmed", run=name, pool=name)
        log("batch_done", pool=name, seconds=rnd.random() * 10)
        [mine] = [e for e in log.entries[before:] if e.kind == "placed" and e.body["pool"] == name]
        if name.startswith("sampler:") and movable:
          self.assertIn(mine.body["gpu_rule"], ("empty_gpu", "fits_next_to_others"), name)


if __name__ == "__main__":
  unittest.main()
