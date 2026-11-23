import unittest
from dataclasses import replace

from tinyrl.control.grants import QUANTUM_MS, plan_node, projected_lag
from tinyrl.control.scheduler import Job, State
from tinyrl.types import MAX_SEQS, Assignment, LocalWorker, NodeStatus, Op, Role, Run, WorkerSpec


class TestFleetGrants(unittest.TestCase):
  def test_one_sample_wave_reserves_the_same_gpu_time_as_one_trainer_turn(self):
    trainer = WorkerSpec(Role.TRAINER)
    sampler = WorkerSpec(Role.SAMPLER)
    state = State(nodes={"east": 1, "west": 1}, jobs={"a": Job(sampler, backlog=512), "b": Job(trainer, backlog=1)})
    wave = {("sampler", "east", "1"): ("a", (0,))}
    east = projected_lag(state, wave)
    self.assertEqual((east("a"), east("b")), (-QUANTUM_MS / 2, QUANTUM_MS / 2))
    both = projected_lag(state, wave | {("trainer", "west", "1"): ("b", (0,))})
    self.assertEqual((both("a"), both("b")), (0, 0))
    self.assertEqual(projected_lag(state, {})("a"), 0)
    self.assertEqual((state.service, state.lag, state.seen), ({}, {}, {}))

  def test_pending_work_reserves_its_exact_block_even_under_exclusive_contention(self):
    spec = WorkerSpec(Role.SAMPLER)
    assignments = [Assignment("w", spec), Assignment("x", WorkerSpec(exclusive=True), ready=True, lag=10000)]
    observed = {"w": LocalWorker(current=(1,), holding=("request",))}
    blocks, _, _ = plan_node(assignments, observed, 2)
    self.assertEqual(blocks, {"w": (1,), "x": (0,)})
    observed["closing"] = LocalWorker(current=(0,), busy=True)
    blocks, _, _ = plan_node(assignments, observed, 2)
    self.assertEqual(blocks, {"w": (1,)})

  def test_parked_wide_trainer_rebinds_only_with_an_alternative_block(self):
    assignments = [
      Assignment("exclusive", WorkerSpec(exclusive=True)),
      Assignment("wide", WorkerSpec(gpus=2), ready=True),
    ]
    observed = {"exclusive": LocalWorker(current=(0,)), "wide": LocalWorker(bound=(0, 1))}
    self.assertEqual(plan_node(assignments, observed, 2)[2], set())
    self.assertEqual(plan_node(assignments, observed, 4)[2], {"wide"})

  def test_only_ready_work_protects_a_residents_quantum(self):
    a = Assignment("resident", WorkerSpec(Role.SAMPLER), ready=True)
    b = Assignment("waiting", WorkerSpec(), ready=True, lag=10.0)
    local = {"resident": LocalWorker(current=(0,), resident_s=QUANTUM_MS / 1000 - 0.25)}
    self.assertEqual(plan_node([a, b], local, 1)[:2], ({"resident": (0,)}, 0.25))
    self.assertEqual(plan_node([replace(a, ready=False), b], local, 1)[:2], ({"waiting": (0,)}, None))


class TestNodePull(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    from tinyrl.control.ledger import Ledger

    self.ledger = Ledger(":memory:")
    self.addAsyncCleanup(self.ledger.aclose)

  async def test_second_node_sees_the_first_sample_wave_before_any_finish(self):
    status = NodeStatus(1)
    for node in ("east", "west"):
      await self.ledger.take(node, "boot", status)
    for run, backlog in (("a", MAX_SEQS * 2), ("b", MAX_SEQS)):
      self.ledger.create_run(Run(run, "base", 8, role=Role.SAMPLER))
      for _ in range(backlog):
        self.ledger.submit_command(run, Op.SAMPLE, b"{}")
    east = await self.ledger.take("east", "boot", status)
    west = await self.ledger.take("west", "boot", status)
    for plan, run in ((east, "a"), (west, "b")):
      [grant] = [g for g in plan.grants if g.work]
      self.assertEqual(grant.gpus, (0,))
      self.assertEqual([w.cmd.run for w in grant.work], [run] * MAX_SEQS)
    replay = await self.ledger.take("east", "boot", status)
    [retried] = [g for g in replay.grants if g.work]
    [original] = [g for g in east.grants if g.work]
    self.assertEqual((retried.worker, retried.gpus, retried.work), (original.worker, original.gpus, original.work))

  async def test_cold_start_while_peers_are_idle_does_not_delay_the_next_burst(self):
    from tinyrl.control.accounting import Charge
    from tinyrl.types import Reply

    for run in ("cold", "warm"):
      self.ledger.create_run(Run(run, "base", 8, account="team"))
    status = NodeStatus(1)
    ok = Reply(True, "application/json", b"{}")

    async def serve(ms):
      plan = await self.ledger.take("node", "boot", status)
      [grant] = [g for g in plan.grants if g.work]
      [work] = grant.work
      self.assertTrue(self.ledger.finish("node", "boot", grant.worker, work.cmd.request_id, ok, Charge(gpu_ms=ms)))
      return work.cmd.run

    self.ledger.submit_command("cold", Op.CREATE_MODEL, b"{}")
    await serve(60_000)
    for run in ("cold", "warm"):
      for _ in range(5):
        self.ledger.submit_command(run, Op.OPTIM_STEP, b"{}")
    self.assertEqual([await serve(1) for _ in range(4)], ["cold", "warm", "cold", "warm"])
    self.assertEqual(self.ledger.service["team"], 60_004)

  async def test_two_gpu_node_reconsiders_account_priority_after_its_first_grant(self):
    self.ledger.create_run(Run("first", "model1", 0, account="B"))
    self.ledger.create_run(Run("second", "model2", 0, account="A"))
    self.ledger.create_run(Run("third", "model3", 0, account="A"))
    for run in ("first", "second", "third"):
      self.ledger.submit_command(run, Op.CREATE_MODEL, b"{}")
    plan = await self.ledger.take("node", "boot", NodeStatus(2))
    grants = [g for g in plan.grants if g.work]
    self.assertEqual({g.gpus for g in grants}, {(0,), (1,)})
    self.assertEqual({self.ledger.plan_state.account(w.cmd.run) for g in grants for w in g.work}, {"A", "B"})

  async def test_busy_sampler_tops_up_for_its_account_and_drains_for_a_waiting_trainer(self):
    self.ledger.create_run(Run("rollout", "base", 8, account="A", role=Role.SAMPLER))
    first = self.ledger.submit_command("rollout", Op.SAMPLE, b"{}")
    [sampler] = (await self.ledger.take("node", "boot", NodeStatus(1))).grants
    second = self.ledger.submit_command("rollout", Op.SAMPLE, b"{}")
    observed = LocalWorker(current=sampler.gpus, holding=(first,))
    status = NodeStatus(1, workers={sampler.worker: observed})
    [topup] = (await self.ledger.take("node", "boot", status)).grants
    self.assertEqual([w.cmd.request_id for w in topup.work], [second])
    self.ledger.create_run(Run("train", "base", 8, account="B"))
    self.ledger.submit_command("train", Op.OPTIM_STEP, b"{}")
    self.ledger.submit_command("rollout", Op.SAMPLE, b"{}")
    status = replace(status, workers={sampler.worker: replace(observed, holding=(first, second))})
    plan = await self.ledger.take("node", "boot", status)
    self.assertFalse(any(g.work for g in plan.grants))
