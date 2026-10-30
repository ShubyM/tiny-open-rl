import random
import unittest

from tinyrl.bus import Event
from tinyrl.scheduler import QUANTUM, RULES, Kind, State, budget, decide, fold, holders, settings, uses

GIB = 1024**3


def check(s: State) -> list[str]:
  """Every broken invariant of the scheduler's state, as one message each."""
  bad = []
  for node, (gpus, gpu) in sorted(s.nodes.items()):
    for k in [(node, i) for i in range(gpus)]:
      held = holders(s, k)
      memory = [uses(s, s.workers[n]) for n in held]
      if len(held) > 1 and not (gpu and all(memory) and sum(memory) <= gpu):
        bad.append(f"{k} has holders that do not fit together")
  for name, p in sorted(s.workers.items()):
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
    if (
      kind in ("run_created", "sampler_warmed") and "role" not in body
    ):  # name the worker's kind the way main's names did
      worker = body["worker"]
      role = "sampler" if worker.startswith("sampler:") else "trainer"
      rank = 0 if worker.startswith("sampler:fft:") or (role == "trainer" and not worker.startswith("lora:")) else 8
      kind, body = "run_created", {"run": f"{worker}-session", "role": role, "base": worker, "rank": rank} | body
    self.entries.append(Event(str(len(self.entries)), self.t, kind, body))
    self.s = fold(self.s, self.entries[-1])
    self.test.assertEqual(check(self.s), [])

  def __call__(self, kind: str, /, at: float | None = None, **body) -> list[tuple[str, str, str]]:
    self.t = self.t if at is None else at
    self.append(kind, body)
    return self.tick()

  def tick(self, at: float | None = None) -> list[tuple[str, str, str]]:
    """What the scheduler decides at time `at`, as (kind, worker, rule)."""
    self.t = self.t if at is None else at
    decided = decide(self.s, self.t)
    for kind, body in decided:
      self.test.assertIn(body["rule"], RULES)
      self.append(kind, body)
    self.test.assertEqual(decide(self.s, self.t), [])
    return [(kind, body["worker"], body["rule"]) for kind, body in decided]

  def node(self, gpus: int, memory: int = 80 * GIB, node: str = "n") -> None:
    self("node_up", node=node, gpus=gpus, memory=memory)

  def worker(self, name: str, gpus: tuple[int, ...] = (0,), node: str = "n") -> None:
    """A worker with a run, placed where the test says rather than where decide would, and its process started."""
    self.append("run_created", {"run": f"{name}-run", "worker": name})
    self.append("placed", {"worker": name, "node": node, "gpus": list(gpus), "rule": "test"})
    self("started", worker=name, node=node)


GRANTED, IDLE, QUANTUM = "turn.granted", "turn.idle_holder_yields", "turn.quantum_expired"
PLACED, EMPTIED = "place.least_loaded_node", "place.emptied"
MIB = 2**20


class TestPlacement(unittest.TestCase):
  def test_a_worker_with_runs_is_placed_and_an_emptied_one_unplaced(self):
    log = Log(self)
    log.node(8)
    self.assertEqual(log("run_created", run="r1", worker="lora:a:0"), [("placed", "lora:a:0", PLACED)])
    self.assertEqual(log("run_created", run="r2", worker="lora:a:0"), [])
    self.assertEqual(log("run_deleted", run="r1", worker="lora:a:0"), [])
    self.assertEqual(log("run_deleted", run="r2", worker="lora:a:0"), [("unplaced", "lora:a:0", EMPTIED)])

  def test_a_dedicated_sampler_cools_with_its_run(self):
    log = Log(self)
    log.node(8)
    log("run_created", run="r", worker="r")
    decided = log("sampler_warmed", worker="sampler:fft:r", train_run="r", exclusive=True)
    self.assertEqual(decided, [("placed", "sampler:fft:r", PLACED)])
    self.assertEqual(
      log("run_deleted", run="r", worker="r"), [("unplaced", "r", EMPTIED), ("unplaced", "sampler:fft:r", EMPTIED)]
    )

  def test_workers_go_to_the_node_with_the_fewest_workers_per_gpu(self):
    log = Log(self)
    log.node(1, node="a")
    log.node(2, node="b")
    for run in ("r1", "r2", "r3"):
      log("run_created", run=run, worker=run)
    self.assertEqual([log.s.workers[run].node for run in ("r1", "r2", "r3")], ["a", "b", "b"])

  def test_a_lost_node_s_workers_move_to_a_live_one(self):
    log = Log(self)
    log.node(1, node="a")
    log("run_created", run="r", worker="r")
    log("started", worker="r", node="a")
    log.node(1, node="b")
    self.assertEqual(log("node_lost", node="a"), [("placed", "r", PLACED)])
    self.assertEqual((log.s.workers["r"].node, log.s.workers["r"].on), ("b", ()))

  def test_each_worker_gets_an_empty_gpu_while_there_is_one(self):
    log = Log(self)
    log.node(8)
    for name in ("lora:a:0", "lora:b:0", "lora:c:0"):
      log("run_created", run=name, worker=name)
    self.assertEqual([log.s.workers[n].gpus for n in ("lora:a:0", "lora:b:0", "lora:c:0")], [(0,), (1,), (2,)])

  def test_shared_fft_samplers_get_a_memory_budget_on_large_gpus(self):
    log = Log(self)
    log.node(8, memory=81559 * MIB)
    log("sampler_warmed", worker="sampler:fft:Qwen")
    log("sampler_warmed", worker="sampler:fft:Gemma")
    log("sampler_warmed", worker="sampler:Qwen")
    log("sampler_warmed", worker="sampler:fft:Llama", exclusive=True)
    shares = {"TINYRL_WARM_SAMPLER": "1", "TINYRL_GPU_MEM": "0.42", "TINYRL_KV_BYTES": str(16 * 1024**3)}
    alone = {"TINYRL_WARM_SAMPLER": "1", "TINYRL_KV_BYTES": str(24 * 1024**3)}
    placed = {name: (p.gpus, settings(p.kind, 8, 81559 * MIB)) for name, p in log.s.workers.items()}
    self.assertEqual(
      placed,
      {
        "sampler:fft:Qwen": ((0,), shares),
        "sampler:fft:Gemma": ((0,), shares),
        "sampler:Qwen": ((1,), alone),
        "sampler:fft:Llama": ((2,), alone),
      },
    )
    self.assertEqual(budget(log.s, log.s.workers["sampler:fft:Qwen"].kind, "n"), int(0.42 * 81559 * MIB))
    self.assertEqual(settings(Kind("sampler", lora=False), 8, 23034 * MIB), {"TINYRL_WARM_SAMPLER": "1"})
    self.assertEqual(settings(Kind("sampler", lora=True), 1, 81559 * MIB), {"TINYRL_KV_BYTES": str(24 * 1024**3)})


class TestTurns(unittest.TestCase):
  def one_gpu(self, *names: str) -> Log:
    log = Log(self)
    log.node(1)
    for name in names:
      log.worker(name)
    return log

  def test_waiters_get_the_gpu_in_first_ask_order(self):
    log = self.one_gpu("a", "b", "c")
    self.assertEqual(log("wants_gpu", worker="a"), [("granted", "a", GRANTED)])
    log("wants_gpu", worker="c")
    log("wants_gpu", worker="b")
    order = ["a"]
    for _ in range(2):
      self.assertEqual(log("idle", worker=order[-1]), [("revoked", order[-1], IDLE)])
      order += [worker for kind, worker, _ in log("released", worker=order[-1]) if kind == "granted"]
    self.assertEqual(order, ["a", "c", "b"])

  def test_one_pass_never_grants_two_workers_that_do_not_fit(self):
    log = self.one_gpu("a", "b")
    log.append("wants_gpu", {"worker": "a"})
    log.append("wants_gpu", {"worker": "b"})
    self.assertEqual(log.tick(), [("granted", "a", GRANTED)])

  def test_a_holder_with_nobody_waiting_keeps_the_gpu(self):
    log = self.one_gpu("a")
    log("wants_gpu", worker="a")
    log("idle", worker="a")
    for t in range(10, 300, 10):
      self.assertEqual(log.tick(at=t), [])

  def test_an_idle_holder_yields_as_soon_as_someone_waits(self):
    log = self.one_gpu("a", "b")
    log("wants_gpu", worker="a")
    log("idle", at=1, worker="a")
    self.assertEqual(log("wants_gpu", at=2, worker="b"), [("revoked", "a", IDLE)])

  def test_a_busy_holder_is_revoked_after_its_quantum(self):
    log = self.one_gpu("a", "b")
    log("wants_gpu", worker="a")
    self.assertEqual(log("wants_gpu", at=1, worker="b"), [])
    self.assertEqual(log.tick(at=29.9), [])
    self.assertEqual(log.tick(at=30), [("revoked", "a", QUANTUM)])

  def test_engine_boot_does_not_count_toward_the_quantum(self):
    log = self.one_gpu("a", "b")
    log("wants_gpu", worker="a")
    log("wants_gpu", at=1, worker="b")
    self.assertEqual(log("batch_done", at=35, worker="a", boot=35.0, seconds=36.0), [])
    self.assertEqual(log.tick(at=64.9), [])
    self.assertEqual(log.tick(at=65), [("revoked", "a", QUANTUM)])

  def test_workers_that_fit_together_hold_a_gpu_at_once_and_the_rest_take_turns(self):
    log = Log(self)
    log.node(1, memory=80 * GIB)
    for name in ("sampler:fft:a", "sampler:fft:b", "t"):  # shared full fine-tune samplers each get 0.42 of a GPU
      log.worker(name)
    self.assertEqual(log("wants_gpu", worker="sampler:fft:a"), [("granted", "sampler:fft:a", GRANTED)])
    self.assertEqual(log("wants_gpu", worker="sampler:fft:b"), [("granted", "sampler:fft:b", GRANTED)])
    self.assertEqual(log("wants_gpu", worker="t"), [])
    self.assertEqual(log("idle", at=5, worker="sampler:fft:a"), [("revoked", "sampler:fft:a", IDLE)])
    self.assertEqual(log("released", at=6, worker="sampler:fft:a"), [])  # t still doesn't fit next to b
    self.assertEqual(log("idle", at=7, worker="sampler:fft:b"), [("revoked", "sampler:fft:b", IDLE)])
    self.assertEqual(log("released", at=8, worker="sampler:fft:b"), [("granted", "t", GRANTED)])

  def test_a_wide_worker_is_granted_only_with_every_gpu(self):
    log = Log(self)
    log.node(2)
    log.worker("w", gpus=(0, 1))
    log.worker("x")
    log("wants_gpu", worker="x")
    self.assertEqual(log("wants_gpu", worker="w"), [])
    log("idle", worker="x")
    self.assertEqual(log("released", worker="x"), [("granted", "w", GRANTED)])

  def test_a_dead_worker_loses_its_turn_and_the_next_in_line_gets_it(self):
    log = self.one_gpu("a", "b")
    log("wants_gpu", worker="a")
    log("wants_gpu", worker="b")
    self.assertEqual(log("exited", worker="a"), [("granted", "b", GRANTED)])
    log.worker("a")
    self.assertEqual(log("wants_gpu", worker="a"), [])

  def test_an_unplaced_worker_keeps_its_turn_until_its_process_exits(self):
    log = self.one_gpu("a", "b")
    log("wants_gpu", worker="a")
    log("wants_gpu", worker="b")
    self.assertEqual(log("run_deleted", run="a-run", worker="a"), [("unplaced", "a", EMPTIED)])
    self.assertEqual(log("exited", worker="a"), [("granted", "b", GRANTED)])
    self.assertNotIn("a", log.s.workers)

  def test_a_moved_worker_keeps_its_old_gpus_until_its_process_exits(self):
    log = Log(self)
    log.node(2)
    log.worker("a", gpus=(0,))
    log.worker("b", gpus=(0,))
    log("wants_gpu", worker="a")
    log("wants_gpu", worker="b")
    log.append("placed", {"worker": "a", "node": "n", "gpus": [1], "rule": "test"})
    self.assertEqual(log.tick(), [])  # a's process is still on GPU 0
    self.assertEqual(log("exited", worker="a"), [("granted", "b", GRANTED)])

  def test_a_restarted_node_daemon_takes_its_processes_turns_with_it(self):
    log = self.one_gpu("a", "b")
    log("wants_gpu", worker="a")
    log("wants_gpu", worker="b")
    log.node(1)
    self.assertEqual([(p.on, p.since, p.asked) for p in log.s.workers.values()], [((), None, 0)] * 2)
    self.assertEqual({p.node for p in log.s.workers.values()}, {"n"})  # still placed, so its node daemon restarts them


class TestLog(unittest.TestCase):
  def random_log(self, rng: random.Random) -> Log:
    """Random facts, faults included: workers restarting elsewhere, dead workers that never release, nodes restarting
    and lost, runs coming and going."""
    log = Log(self)
    nodes = {f"n{i}": rng.choice([1, 2, 4, 8]) for i in range(rng.choice([1, 1, 2, 4]))}
    for node, gpus in nodes.items():
      log.node(gpus, node=node)
    names = [rng.choice(("p", "sampler:fft:p")) + str(i) for i in range(rng.randint(1, 12))]  # those samplers share
    for _ in range(rng.randint(20, 200)):
      name, roll = rng.choice(names), rng.random()
      if roll < 0.15 and log.s.nodes:
        node = rng.choice(sorted(log.s.nodes))
        width = rng.choice([w for w in (1, 2, 4) if w <= nodes[node]])
        first = rng.randrange(0, nodes[node] - width + 1, width)
        log.worker(name, range(first, first + width), node)
      elif roll < 0.4:
        log("wants_gpu", worker=name)
      elif roll < 0.55:
        log("batch_done", at=log.t + rng.random() * 20, worker=name, boot=rng.choice([0.0, 0.0, 30.0]), seconds=1.0)
      elif roll < 0.65:
        log("released", worker=name)
      elif roll < 0.75:
        log.tick(at=log.t + rng.random() * 40)
      elif roll < 0.8:
        log("exited", worker=name)
      elif roll < 0.82:
        node = rng.choice(sorted(nodes))
        log.node(nodes[node], node=node)
      elif roll < 0.84:
        log("node_lost", node=rng.choice(sorted(nodes)))
      elif roll < 0.88:
        log("run_created", run=f"r{rng.randint(0, 20)}", worker=name, exclusive=rng.random() < 0.2)
      elif roll < 0.92:
        log("run_deleted", run=f"r{rng.randint(0, 20)}", worker=name)
      elif roll < 0.94:
        log("sampler_warmed", worker=f"sampler:{name}")
      elif roll < 0.97 and (p := log.s.workers.get(name)) and p.node:
        log("started", worker=name, node=p.node)
      else:
        log("idle", worker=name)
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
      log.worker(name)
    asked, granted = {}, {}

    def note(decisions: list) -> list:
      for kind, worker, _ in decisions:
        if kind == "granted":
          granted.setdefault(worker, log.t)
      return decisions

    for name in names:
      asked[name] = log.t = log.t + 1
      note(log("wants_gpu", worker=name))
    for _ in range(400):  # a bound, so a broken rule fails the test where it would wait forever
      for kind, worker, _ in note(log.tick(at=log.t + 1)):
        if kind == "revoked":
          note(log("released", at=log.t + 1, worker=worker))
    for place, name in enumerate(names):
      self.assertLessEqual(granted[name] - asked[name], (place + 1) * 32.0, name)


class TestWorkersAndGpus(unittest.TestCase):
  def test_a_trainer_takes_turns_with_the_least_busy_trainers(self):
    log = Log(self)
    log.node(2)
    for worker, busy in (("busy", 50.0), ("light", 5.0)):
      log("run_created", run=worker, worker=worker)
      log("batch_done", worker=worker, seconds=busy)
    log("run_created", run="t", worker="t")
    self.assertEqual(log.s.workers["t"].gpus, log.s.workers["light"].gpus)

  def test_a_sampler_moves_the_least_busy_trainers_aside_and_never_takes_turns(self):
    log = Log(self)
    log.node(2)
    for worker, busy in (("busy", 50.0), ("light", 5.0)):
      log("run_created", run=worker, worker=worker)
      log("batch_done", worker=worker, seconds=busy)
    decided = log("sampler_warmed", worker="sampler:b")
    self.assertEqual([(k, p) for k, p, _ in decided], [("placed", "light"), ("placed", "sampler:b")])
    self.assertEqual(log.s.workers["light"].gpus, log.s.workers["busy"].gpus)
    self.assertNotEqual(log.s.workers["sampler:b"].gpus, log.s.workers["busy"].gpus)

  def test_a_worker_with_a_memory_budget_fills_a_gpu_it_fits_on_before_an_empty_one(self):
    log = Log(self)
    log.node(2, memory=80 * GIB)
    log("run_created", run="t", worker="t")
    log("sampler_warmed", worker="sampler:fft:a")
    log("run_deleted", run="t", worker="t")
    log("exited", worker="t")
    log("sampler_warmed", worker="sampler:fft:b")
    self.assertEqual(log.s.workers["sampler:fft:b"].gpus, log.s.workers["sampler:fft:a"].gpus)

  def test_no_worker_fits_next_to_an_exclusive_one_whatever_their_budgets(self):
    log = Log(self)
    log.node(2, memory=80 * GIB)
    log.append("run_created", {"run": "x", "worker": "x", "exclusive": True})
    log.append("placed", {"worker": "x", "node": "n", "gpus": [1], "rule": "test"})
    log("sampler_warmed", worker="sampler:fft:a")
    self.assertEqual(log.s.workers["sampler:fft:a"].gpus, (0,))

  def test_a_sampler_moves_as_few_trainers_as_it_can(self):
    log = Log(self)
    log.node(2)
    for worker, gpu, busy in (("a", 0, 1.0), ("b", 0, 1.0), ("c", 1, 5.0)):
      log.worker(worker, gpus=(gpu,))
      log("batch_done", worker=worker, seconds=busy)
    log("sampler_warmed", worker="sampler:s")
    self.assertEqual((log.s.workers["sampler:s"].gpus, log.s.workers["c"].gpus), ((1,), (0,)))

  def test_a_sampler_moves_a_lora_trainer_aside_before_a_full_fine_tune(self):
    log = Log(self)
    log.node(2)
    for worker, gpu, busy in (("full", 0, 1.0), ("lora:a", 1, 9.0)):
      log.worker(worker, gpus=(gpu,))
      log("batch_done", worker=worker, seconds=busy)
    log("sampler_warmed", worker="sampler:s")
    self.assertEqual((log.s.workers["sampler:s"].gpus, log.s.workers["lora:a"].gpus), ((1,), (0,)))

  def test_an_exclusive_worker_gets_a_gpu_to_itself_and_others_stay_off_it(self):
    log = Log(self)
    log.node(2)
    log("run_created", run="x", worker="x", exclusive=True)
    log("run_created", run="a", worker="a")
    log("run_created", run="b", worker="b")
    self.assertEqual([log.s.workers[n].gpus for n in ("x", "a", "b")], [(0,), (1,), (1,)])

  def test_the_benchmark_s_arrivals_move_lora_trainers_and_leave_samplers_alone(self):
    log = Log(self)
    log.node(8, memory=81559 * MIB)
    trainers = {"x": 4.0, "lora:g:0": 7.0, "lora:q:0": 7.7, "q1": 1.7, "q2": 2.4, "g1": 4.8}
    for worker, busy in trainers.items():
      log("run_created", run=worker, worker=worker, exclusive=worker == "x")
      log("batch_done", worker=worker, seconds=busy)
    for worker in ("sampler:fft:q", "sampler:q", "sampler:fft:g", "sampler:fft:x", "sampler:g"):
      log("sampler_warmed", worker=worker, exclusive=worker == "sampler:fft:x")
    gpu = {name: p.gpus[0] for name, p in log.s.workers.items()}
    self.assertEqual({gpu["lora:g:0"], gpu["lora:q:0"]}, {gpu["q1"], gpu["q2"]})
    self.assertEqual(gpu["sampler:fft:q"], gpu["sampler:fft:g"])
    alone = [gpu[n] for n in ("x", "g1", "sampler:q", "sampler:fft:x", "sampler:g")]
    self.assertEqual(sorted(alone + [gpu["q1"], gpu["q2"], gpu["sampler:fft:q"]]), list(range(8)))

  def test_random_arrivals_never_make_a_sampler_take_turns_while_trainers_could_move(self):
    rnd = random.Random(42)
    for _ in range(200):
      log, gpus = Log(self), rnd.choice([2, 4, 8])
      log.node(gpus)
      names = [f"lora:b-{i}:0" for i in range(3)] + [f"ts-{i}" for i in range(4)] + ["sampler:b-0", "sampler:c-0"]
      for name in rnd.sample(names, rnd.randint(1, len(names))):
        on = {g: [(q, p) for q, p in log.s.workers.items() if p.gpus == (g,)] for g in range(gpus)}
        movable = [g for g, ps in on.items() if ps and all(q.startswith(("lora:", "ts-")) for q, _ in ps)]
        before = len(log.entries)
        log("run_created" if not name.startswith("sampler:") else "sampler_warmed", run=name, worker=name)
        log("batch_done", worker=name, seconds=rnd.random() * 10)
        [mine] = [e for e in log.entries[before:] if e.kind == "placed" and e.body["worker"] == name]
        if name.startswith("sampler:") and movable:
          self.assertIn(mine.body["gpu_rule"], ("empty_gpu", "fits_next_to_others"), name)


def workers_after(*runs: dict) -> dict[str, str]:
  """Each run's worker once the scheduler has decided, for runs created with these requirements."""
  s, n = State(), 0
  for r in runs:
    body = {"role": "trainer", "base": "a", "rank": 8, "exclusive": False, "train_run": None} | r
    s, n = fold(s, Event(str(n), 0.0, "run_created", body)), n + 1
  for kind, body in decide(s, 0.0):
    s = fold(s, Event("", 0.0, kind, body))
  return {name: r.worker for name, r in s.runs.items()}


class TestChooseWorker(unittest.TestCase):
  def test_lora_runs_of_one_base_share_a_worker_and_full_or_exclusive_runs_get_their_own(self):
    worker = workers_after(
      {"run": "A"}, {"run": "B"}, {"run": "C", "base": "b"}, {"run": "F", "rank": 0}, {"run": "X", "exclusive": True}
    )
    self.assertEqual(worker["A"], worker["B"])
    self.assertEqual(len(set(worker.values())), 4)

  def test_sampling_sessions_share_their_base_and_kinds_sampler(self):
    sampler = {"role": "sampler", "rank": 1}
    worker = workers_after(
      sampler | {"run": "s1", "train_run": "A"},
      sampler | {"run": "s2", "train_run": "B"},
      sampler | {"run": "f", "rank": 0, "train_run": "A"},
      sampler | {"run": "x1", "exclusive": True, "train_run": "X"},
      sampler | {"run": "x2", "exclusive": True, "train_run": "X"},
    )
    self.assertEqual(worker["s1"], worker["s2"])
    self.assertEqual(worker["x1"], worker["x2"])
    self.assertEqual(len(set(worker.values())), 3)


if __name__ == "__main__":
  unittest.main()
