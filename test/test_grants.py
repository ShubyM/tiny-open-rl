import unittest

from tinyrl.control.accounting import Charge
from tinyrl.control.controller import Controller
from tinyrl.control.policy import QUANTUM_MS, Demand, Worker, plan_gpus, projected_lag
from tinyrl.types import MAX_SEQS, Done, Effect, NodeMsg, Op, Reply, Role, Run, TaskKind, WorkerSpec


class TestFleetGrants(unittest.TestCase):
  def test_one_sample_wave_reserves_the_same_gpu_time_as_one_trainer_turn(self):
    runs = [("a", WorkerSpec(Role.SAMPLER), 512), ("b", WorkerSpec(Role.TRAINER), 1)]
    nodes = {"east": 1, "west": 1}
    east = projected_lag({}, {}, {}, {}, runs, nodes, [("a", 1)])
    self.assertEqual((east("a"), east("b")), (-QUANTUM_MS / 2, QUANTUM_MS / 2))
    both = projected_lag({}, {}, {}, {}, runs, nodes, [("a", 1), ("b", 1)])
    self.assertEqual((both("a"), both("b")), (0, 0))
    self.assertEqual(projected_lag({}, {}, {}, {}, runs, nodes, [])("a"), 0)

  def test_occupied_gpus_prevent_exclusive_overlap(self):
    cands = {"x": Worker(exclusive=True, lag=10_000)}
    self.assertEqual(plan_gpus(cands, 2, 0.0, locked={1})[0], {"x": (0,)})
    self.assertEqual(plan_gpus(cands, 2, 0.0, locked={0, 1})[0], {})

  def test_only_ready_work_protects_a_residents_quantum(self):
    res = Worker(current=(0,), hold_until=0.25)
    wait = Worker(lag=10.0)
    self.assertEqual(plan_gpus({"resident": res, "waiting": wait}, 1, 0.0), ({"resident": (0,)}, 0.25))
    idle = Worker(demand=Demand.IDLE, current=(0,), hold_until=0.25)
    self.assertEqual(plan_gpus({"resident": idle, "waiting": wait}, 1, 0.0), ({"waiting": (0,)}, None))


class TestNodeGrants(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.ctrl = Controller(":memory:")
    self.addAsyncCleanup(self.ctrl.aclose)

  async def test_second_node_sees_the_first_sample_wave_before_any_finish(self):
    self.ctrl.create_run(Run("a", "base", 8, role=Role.SAMPLER))
    for _ in range(MAX_SEQS * 2):
      self.ctrl.submit_command("a", Op.SAMPLE, b"{}")
    east = [t for t in self.ctrl.handle(NodeMsg("hello", "east", "boot", gpus=1)).tasks if t.kind == TaskKind.SAMPLE]
    self.ctrl.create_run(Run("b", "base", 8, role=Role.SAMPLER))
    for _ in range(MAX_SEQS):
      self.ctrl.submit_command("b", Op.SAMPLE, b"{}")
    west = [t for t in self.ctrl.handle(NodeMsg("hello", "west", "boot", gpus=1)).tasks if t.kind == TaskKind.SAMPLE]
    for wave, run in ((east, "a"), (west, "b")):
      self.assertEqual(wave[0].gpus, (0,))
      self.assertEqual([t.cmd.run for t in wave], [run] * MAX_SEQS)
    replay = [t for t in self.ctrl.handle(NodeMsg("hello", "east", "boot", gpus=1)).tasks if t.kind == TaskKind.SAMPLE]
    self.assertEqual(replay, east)

  async def test_cold_start_while_peers_are_idle_does_not_delay_the_next_burst(self):
    for run in ("cold", "warm"):
      self.ctrl.create_run(Run(run, "base", 8, account="team"))
    self.ctrl.handle(NodeMsg("hello", "node", "boot", gpus=1))
    ok = Reply(True, "application/json", b"{}")

    def serve(ms):
      [t] = [t for t in self.ctrl.nodes["node"].tasks.values() if t.kind == TaskKind.TRAIN]
      self.ctrl.handle(
        NodeMsg("events", "node", "boot", done=(Done(t.id, True, ok, Charge(gpu_ms=ms), Effect(True, True)),))
      )
      return t.cmd.run

    self.ctrl.submit_command("cold", Op.CREATE_MODEL, b"{}")
    serve(60_000)
    for run in ("cold", "warm"):
      for _ in range(5):
        self.ctrl.submit_command(run, Op.OPTIM_STEP, b"{}")
    self.assertEqual([serve(1) for _ in range(4)], ["cold", "warm", "cold", "warm"])
    self.assertEqual(self.ctrl.service["team"], 60_004)

  async def test_two_gpu_node_reconsiders_account_priority_after_its_first_grant(self):
    self.ctrl.create_run(Run("first", "model1", 0, account="B"))
    self.ctrl.create_run(Run("second", "model2", 0, account="A"))
    self.ctrl.create_run(Run("third", "model3", 0, account="A"))
    for run in ("first", "second", "third"):
      self.ctrl.submit_command(run, Op.CREATE_MODEL, b"{}")
    tasks = [t for t in self.ctrl.handle(NodeMsg("hello", "node", "boot", gpus=2)).tasks if t.kind == TaskKind.TRAIN]
    self.assertEqual({t.gpus for t in tasks}, {(0,), (1,)})
    self.assertEqual({self.ctrl.account(t.cmd.run) for t in tasks}, {"A", "B"})

  async def test_busy_sampler_tops_up_for_its_account_and_drains_for_a_waiting_trainer(self):
    self.ctrl.create_run(Run("rollout", "base", 8, account="A", role=Role.SAMPLER))
    first = self.ctrl.submit_command("rollout", Op.SAMPLE, b"{}")
    [t1] = [t for t in self.ctrl.handle(NodeMsg("hello", "node", "boot", gpus=1)).tasks if t.kind == TaskKind.SAMPLE]
    self.assertEqual(t1.cmd.request_id, first)
    second = self.ctrl.submit_command("rollout", Op.SAMPLE, b"{}")
    [t2] = self.ctrl.nodes["node"].drain_push().tasks
    self.assertEqual(t2.cmd.request_id, second)
    self.ctrl.create_run(Run("train", "base", 8, account="B"))
    self.ctrl.submit_command("train", Op.OPTIM_STEP, b"{}")
    self.ctrl.submit_command("rollout", Op.SAMPLE, b"{}")
    self.assertEqual(self.ctrl.nodes["node"].drain_push().tasks, ())
