import itertools
import random
import unittest
from dataclasses import asdict

from simulator.scheduling import (
  Job,
  SchedulerPolicy,
  exact_plan,
  fair_reference,
  fifo_policy,
  frozen_workloads,
  measure,
  replay,
  simulate,
)


class TestServiceBenchmark(unittest.TestCase):
  def test_reference_accounts_for_arrivals_weights_and_finished_demand(self):
    jobs = (Job("a", 0, 2), Job("b", 0, 20, 3), Job("late", 4, 20), Job("future", 20, 2))
    reference = fair_reference(jobs, 8)
    self.assertAlmostEqual(reference["a"], 1.8)
    self.assertAlmostEqual(reference["b"], 5.4)
    self.assertAlmostEqual(reference["late"], 0.8)
    self.assertEqual(reference["future"], 0)
    self.assertEqual(fair_reference((Job("short", 0, 1), Job("long", 0, 20)), 8), {"short": 1, "long": 7})

  def test_jain_can_look_excellent_while_a_job_gets_no_service(self):
    jobs = tuple(Job(str(i), 0, 100) for i in range(100))
    trace = simulate(jobs, 99, 1, 0, replay(tuple(range(99))))
    result = measure(jobs, trace, 99)
    self.assertAlmostEqual(result.jain, 0.99)
    self.assertEqual(result.minimum_service, 0)
    self.assertEqual(result.service["99"], 0)
    self.assertEqual(result.maximum_wait, 99)
    self.assertEqual(result.unfinished, 100)

  def test_complete_homogeneous_round_attains_the_fluid_bound(self):
    # One GPU supplies at most n*q compute per round. Equal service gives every
    # job q; improving the minimum would require more compute than exists.
    for n, quantum in itertools.product(range(1, 7), (1, 2, 5)):
      jobs = tuple(Job(str(i), 0, 1000) for i in range(n))
      horizon = 3 * n * quantum
      result = measure(jobs, simulate(jobs, horizon, quantum, 0, fifo_policy(jobs)), horizon)
      self.assertAlmostEqual(result.minimum_service, 1)
      self.assertEqual(set(result.service.values()), {3 * quantum})
      self.assertEqual(result.maximum_wait, (n - 1) * quantum)

  def test_switching_reduces_work_even_when_shares_remain_equal(self):
    jobs = (Job("a", 0, 100), Job("b", 0, 100))
    result = measure(jobs, simulate(jobs, 7, 1, 1, fifo_policy(jobs)), 7)
    self.assertEqual(result.jain, 1)
    self.assertEqual(result.useful_work, 4)
    self.assertEqual(result.switch_time, 3)
    self.assertAlmostEqual(result.minimum_service, 4 / 7)

  def test_oracle_matches_exhaustive_schedules_in_tiny_case(self):
    jobs = (Job("a", 0, 9), Job("b", 0, 9, 2))
    candidates = []
    for plan in itertools.product((0, 1, None), repeat=5):
      result = measure(jobs, simulate(jobs, 5, 1, 0, replay(plan)), 5)
      candidates.append((result.minimum_service, result.useful_work))
    result = measure(jobs, simulate(jobs, 5, 1, 0, replay(exact_plan(jobs, 5, 1, 0))), 5)
    self.assertEqual((result.minimum_service, result.useful_work), max(candidates))

  def test_oracle_exposes_equal_turns_that_ignore_weights(self):
    jobs = (Job("a", 0, 20), Job("b", 0, 20, 2))
    baseline = measure(jobs, simulate(jobs, 6, 1, 0, fifo_policy(jobs)), 6)
    optimum = measure(jobs, simulate(jobs, 6, 1, 0, replay(exact_plan(jobs, 6, 1, 0))), 6)
    self.assertEqual(baseline.service, {"a": 3, "b": 3})
    self.assertEqual(baseline.minimum_service, 0.75)
    self.assertEqual(optimum.service, {"a": 2, "b": 4})
    self.assertEqual(optimum.minimum_service, 1)

  def test_frozen_workloads_include_all_waiters_and_respect_oracle_bound(self):
    for jobs in frozen_workloads():
      with self.subTest(jobs=jobs):
        horizon = 10
        actual_trace = simulate(jobs, horizon, 2, 1, fifo_policy(jobs))
        actual = measure(jobs, actual_trace, horizon)
        oracle = measure(jobs, simulate(jobs, horizon, 2, 1, replay(exact_plan(jobs, horizon, 2, 1))), horizon)
        self.assertLessEqual((actual.minimum_service, actual.useful_work), (oracle.minimum_service, oracle.useful_work))
        self.assertEqual(set(actual.service), {job.name for job in jobs})
        self.assertEqual(actual.useful_work + actual.switch_time + actual.idle_time, horizon)
        for before, after in zip(actual_trace, actual_trace[1:]):
          self.assertEqual(before.end, after.start)
        for job in jobs:
          self.assertLessEqual(actual.service[job.name], job.work)
        self.assertEqual(frozen_workloads(), frozen_workloads())

  def test_horizon_cuts_switch_and_future_jobs_do_not_count_as_starved(self):
    jobs = (Job("a", 0, 2), Job("b", 0, 2), Job("future", 9, 2))
    trace = simulate(jobs, 3, 2, 5, fifo_policy(jobs))
    result = measure(jobs, trace, 3)
    self.assertEqual(result.useful_work, 2)
    self.assertEqual(result.switch_time, 1)
    self.assertEqual(result.unfinished, 1)
    self.assertNotIn("future", result.normalized_service)
    self.assertEqual(result.maximum_wait, 3)

  def test_rejects_impossible_selection_and_unbounded_oracle(self):
    jobs = (Job("late", 1, 2),)
    with self.assertRaises(ValueError):
      simulate(jobs, 3, 1, 0, lambda *args: 0)
    with self.assertRaises(ValueError):
      exact_plan(jobs, 100, 1, 0)


class TestActualScheduler(unittest.TestCase):
  def test_real_decisions_attain_the_homogeneous_round_bound(self):
    for n, quantum in itertools.product((1, 2, 4), (1, 3)):
      jobs = tuple(Job(str(i), 0, 100) for i in range(n))
      horizon = 3 * n * quantum
      result = measure(jobs, simulate(jobs, horizon, quantum, 0, SchedulerPolicy(jobs, quantum)), horizon)
      self.assertEqual(set(result.service.values()), {3 * quantum})
      self.assertAlmostEqual(result.minimum_service, 1)
      self.assertEqual(result.maximum_wait, (n - 1) * quantum)

  def test_real_weighted_turns_match_the_tiny_oracle(self):
    jobs = (Job("a", 0, 20), Job("b", 0, 20, 2))
    actual = measure(jobs, simulate(jobs, 6, 1, 0, SchedulerPolicy(jobs, 1)), 6)
    optimum = measure(jobs, simulate(jobs, 6, 1, 0, replay(exact_plan(jobs, 6, 1, 0))), 6)
    self.assertEqual(actual.service, optimum.service)
    self.assertEqual(actual.minimum_service, 1)

  def test_real_policy_on_frozen_arrivals_against_independent_oracle(self):
    for jobs in frozen_workloads():
      with self.subTest(jobs=jobs):
        actual = measure(jobs, simulate(jobs, 10, 2, 1, SchedulerPolicy(jobs, 2)), 10)
        optimum = measure(jobs, simulate(jobs, 10, 2, 1, replay(exact_plan(jobs, 10, 2, 1))), 10)
        self.assertLessEqual(
          (actual.minimum_service, actual.useful_work), (optimum.minimum_service, optimum.useful_work)
        )
        self.assertEqual(actual.useful_work + actual.switch_time + actual.idle_time, 10)
        self.assertEqual(set(actual.service), {j.name for j in jobs})

  def test_entitlement_starts_at_time_zero(self):
    policy = SchedulerPolicy((Job("a", 0, 10), Job("b", 0, 10)), 1)
    policy(0, (10, 10), None)
    policy(1, (9, 10), 0)
    self.assertEqual(policy.state.runs["a"].entitled, 0.5)
    self.assertEqual(policy.state.runs["b"].entitled, 0.5)

  def test_switch_cost_does_not_give_older_runs_unbounded_debt_over_newcomers(self):
    jobs = (Job("a", 0, 10000), Job("b", 0, 10000), Job("new", 300, 10000))
    for switch in (0, 1, 5):
      trace = simulate(jobs, 420, 1, switch, SchedulerPolicy(jobs, 1))
      first = next(part.start for part in trace if part.kind == "compute" and part.job == "new")
      # This bound is for these fixed jobs, unit batches, and known switching cost,
      # not a bound for arbitrary kernel duration or delayed release acknowledgements.
      self.assertLessEqual(first - 300, 3 * (1 + switch))
      self.assertGreater(measure(jobs, trace, 420).service["new"], 0)

  def test_revocation_does_not_free_gpu_before_acknowledgement(self):
    policy = SchedulerPolicy((Job("a", 0, 10), Job("b", 0, 10)), 1)
    policy(0, (10, 10), None)
    for kind, body in policy.scheduler.decide(policy.state, 1):
      policy.emit(1, kind, **body)
    self.assertTrue(policy.state.workers["worker:0"].revoked)
    self.assertTrue(policy.state.workers["worker:0"].held)
    self.assertFalse(policy.state.workers["worker:1"].held)
    self.assertEqual(policy.scheduler.decide(policy.state, 100), [])
    policy.emit(100, "released", worker="worker:0", turn=policy.state.workers["worker:0"].turn)
    decisions = policy.scheduler.decide(policy.state, 100)
    self.assertEqual([(kind, body["worker"]) for kind, body in decisions], [("granted", "worker:1")])

  def test_delayed_same_worker_release_cannot_end_its_new_turn(self):
    policy = SchedulerPolicy((Job("a", 0, 10),), 1)
    policy(0, (10,), None)
    old_turn = policy.state.workers["worker:0"].turn
    policy.emit(1, "released", worker="worker:0", turn=old_turn)
    policy.emit(1, "wants_gpu", worker="worker:0")
    for kind, body in policy.scheduler.decide(policy.state, 1):
      policy.emit(1, kind, **body)
    expected = asdict(policy.state.workers["worker:0"])
    self.assertNotEqual(expected["turn"], old_turn)
    self.assertTrue(expected["held"])
    for kind in ("released", "idle", "revoked"):
      policy.emit(1, kind, worker="worker:0", turn=old_turn)
      self.assertEqual(asdict(policy.state.workers["worker:0"]), expected)

  def test_delayed_old_worker_facts_cannot_change_replacement(self):
    facts = [
      ("started", {"worker": "worker:0", "node": "node"}),
      ("exited", {"worker": "worker:0", "node": "node"}),
      ("idle", {"worker": "worker:0"}),
      ("released", {"worker": "worker:0"}),
      ("service", {"worker": "worker:0", "run": "a", "seconds": 999}),
      ("worker_memory", {"worker": "worker:0", "memory": 999, "run_memory": {"a": 999}}),
      ("run_released", {"run": "a", "source": "worker:0", "token": "old"}),
    ]
    rng = random.Random(613)
    for _ in range(50):
      policy = SchedulerPolicy((Job("a", 0, 10),), 1)
      policy(0, (10,), None)
      policy.emit(1, "worker_created", worker="replacement", node="node", gpus=[0], base="a", rank=1)
      policy.emit(1, "run_move_requested", run="a", source="worker:0", target="replacement", token="old")
      policy.emit(1, "run_released", run="a", source="worker:0", token="old")
      policy.emit(1, "released", worker="worker:0", turn=policy.state.workers["worker:0"].turn)
      policy.emit(1, "run_assigned", run="a", worker="replacement")
      policy.emit(1, "started", worker="replacement", node="node")
      policy.emit(1, "wants_gpu", worker="replacement")
      policy.emit(1, "granted", worker="replacement", turn="replacement-turn")
      expected_run, expected_worker = asdict(policy.state.runs["a"]), asdict(policy.state.workers["replacement"])
      for kind, body in rng.sample(facts, len(facts)):
        policy.emit(1, kind, **body)
        self.assertEqual(asdict(policy.state.runs["a"]), expected_run)
        self.assertEqual(asdict(policy.state.workers["replacement"]), expected_worker)

  def test_eight_distinct_lora_workers_use_idle_gpus_without_fixed_lanes(self):
    from tinyrl.bus import Event
    from tinyrl.scheduler import State, decide, fold

    state = fold(State(), Event("0", 0, "node_up", {"node": "n", "gpus": 8, "memory": 80 * 1024**3}))
    for i in range(8):
      state = fold(state, Event("", 0, "run_created", {"run": f"r{i}", "base": f"base{i}", "rank": 4}))
      state = fold(state, Event("", 0, "run_queued", {"run": f"r{i}", "count": 1}))
    for kind, body in decide(state, 0):
      state = fold(state, Event("", 0, kind, body))
    for name in state.workers:
      state = fold(state, Event("", 0, "started", {"worker": name, "node": "n"}))
      state = fold(state, Event("", 0, "wants_gpu", {"worker": name}))
    for kind, body in decide(state, 0):
      state = fold(state, Event("", 0, kind, body))
    active = {gpu for worker in state.workers.values() if worker.held for gpu in worker.gpus}
    # Recorded former policy on HEAD a2d8871: eight shared LoRA workers on an
    # eight-GPU node alternate between trainer lanes 0 and 1, despite six idle GPUs.
    old_placement = (0, 1, 0, 1, 0, 1, 0, 1)
    self.assertEqual(len(set(old_placement)), 2)
    self.assertEqual(active, set(range(8)))

  def test_five_backlogged_runs_rotate_across_four_gpus_after_fenced_handoffs(self):
    from tinyrl.bus import Event
    from tinyrl.scheduler import State, decide, fold, keys, members

    state, moves, released = State(quantum=1), [], set()

    def emit(now, kind, **body):
      nonlocal state
      state = fold(state, Event(str(state.n), now, kind, body))
      held = [key for worker in state.workers.values() if worker.held for key in keys(worker)]
      self.assertEqual(len(held), len(set(held)), "overlapping physical GPU turns")

    def settle(now):
      for _ in range(100):
        for name, worker in list(state.workers.items()):
          if not worker.started and not worker.stopping:
            emit(now, "started", worker=name, node=worker.node)
          if members(state, name, ready=True) and not worker.held and not worker.stopping:
            emit(now, "wants_gpu", worker=name)
        decisions = decide(state, now)
        if not decisions:
          return
        for kind, body in decisions:
          if kind == "run_assigned" and state.runs[body["run"]].target:
            self.assertIn(body["run"], released, "target assigned before source checkpoint acknowledgement")
          emit(now, kind, **body)
          if kind == "revoked":
            emit(now, "released", worker=body["worker"], turn=body["turn"])
          elif kind == "worker_stopped":
            emit(now, "exited", worker=body["worker"], node=state.workers[body["worker"]].node)
          elif kind == "run_move_requested":
            moves.append((now, body["run"], body["source"], body["target"]))
            self.assertEqual(state.runs[body["run"]].worker, body["source"])
            # Acknowledge a completed checkpoint and physical park. The test uses
            # zero transfer/startup delay; it checks lifecycle, not migration speed.
            source = state.workers[body["source"]]
            emit(now, "released", worker=body["source"], turn=source.turn)
            emit(now, "run_released", run=body["run"], source=body["source"], token=body["token"])
            released.add(body["run"])
      self.fail("placement and handoff decisions failed to settle")

    emit(0, "node_up", node="node", gpus=4, memory=80 * 1024**3)
    for i in range(5):
      emit(0, "run_created", run=f"r{i}", base=f"base{i}", rank=1)
      emit(0, "run_queued", run=f"r{i}", count=1)
    settle(0)
    for now in range(1, 21):
      for worker, value in list(state.workers.items()):
        if value.held:
          running = members(state, worker, ready=True)
          self.assertEqual(len(running), 1)
          emit(now, "service", worker=worker, run=running[0], seconds=1)
      settle(now)
    self.assertTrue(moves, "fairness must include relocation when one GPU has the extra runnable job")
    service = [run.served for run in state.runs.values()]
    self.assertEqual(sum(service), 80)
    self.assertGreater(min(service), 10, "the extra job must not remain permanently confined to half a GPU")


if __name__ == "__main__":
  unittest.main()
