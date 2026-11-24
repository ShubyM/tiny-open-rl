import sqlite3

from test.helpers import BusTest
from tinyrl.control.accounting import Charge
from tinyrl.types import Done, Effect, NodeMsg, Op, Reply, Role, Run, TaskKind

OK = Reply(True, "application/json", b"{}")


class TestClaimStorage(BusTest):
  async def test_sample_keeps_its_version_on_retry_and_reassignment(self):
    uri = "tinker://trainer/sampler_weights/latest"
    self.seed_checkpoint(uri, 1, version="first")
    self.bus.create_run(Run("s", "base", 8, role=Role.SAMPLER, model_path=uri))
    first = self.bus.submit_command("s", Op.SAMPLE, b"{}")
    staged = ("first", "second")

    [initial] = [
      t
      for t in self.bus.handle(NodeMsg("hello", "east", "boot1", gpus=1, staged=staged)).tasks
      if t.kind == TaskKind.SAMPLE
    ]
    self.seed_checkpoint(uri, 1, version="second")
    second = self.bus.submit_command("s", Op.SAMPLE, b"{}")
    [replay] = [
      t
      for t in self.bus.handle(NodeMsg("hello", "east", "boot1", gpus=1, staged=staged)).tasks
      if t.cmd.request_id == first
    ]
    self.assertEqual(replay, initial)
    self.assertIn("first", self.bus.live_versions())

    self.bus.nodes["east"].until = 0
    [retry] = [
      t
      for t in self.bus.handle(NodeMsg("hello", "west", "boot2", gpus=1, staged=staged)).tasks
      if t.cmd.request_id == first
    ]
    self.assertEqual((retry.cmd.request_id, retry.version), (first, "first"))
    self.bus.handle(NodeMsg("events", "west", "boot2", done=(Done(retry.id, True, OK),)))
    self.assertNotIn("first", self.bus.live_versions())
    done_second = [t for t in self.bus.nodes["west"].tasks.values() if t.cmd and t.cmd.request_id == second]
    self.assertEqual([(t.cmd.request_id, t.version) for t in done_second], [(second, "second")])

  async def test_redelivered_wave_preserves_interleaved_run_order(self):
    for run in ("a", "b"):
      self.bus.create_run(Run(run, "base", 8, role=Role.SAMPLER, account="owner"))
      for _ in range(2):
        self.bus.submit_command(run, Op.SAMPLE, b"{}")
    initial = [t for t in self.bus.handle(NodeMsg("hello", "n", "boot", gpus=1)).tasks if t.kind == TaskKind.SAMPLE]
    self.assertEqual([t.cmd.run for t in initial], ["a", "b", "a", "b"])
    replay = [t for t in self.bus.handle(NodeMsg("hello", "n", "boot", gpus=1)).tasks if t.kind == TaskKind.SAMPLE]
    self.assertEqual(replay, initial)

  async def test_failed_finish_preserves_trainer_claim_and_bills_once_on_retry(self):
    self.bus.create_run(Run("r", "base", 8, account="a"))
    req = self.bus.submit_command("r", Op.OPTIM_STEP, b"{}")
    [task] = [t for t in self.bus.handle(NodeMsg("hello", "n", "b", gpus=1)).tasks if t.kind == TaskKind.TRAIN]
    self.bus.db.execute(
      """CREATE TEMP TRIGGER fail_run_write BEFORE INSERT ON runs
         BEGIN SELECT RAISE(ABORT, 'injected write failure'); END"""
    )
    done = Done(task.id, True, OK, Charge(gpu_ms=5), Effect(loaded=True, mutated=True))
    try:
      with self.assertRaisesRegex(sqlite3.IntegrityError, "injected write failure"):
        self.bus.handle(NodeMsg("events", "n", "b", done=(done,)))
    finally:
      self.bus.db.execute("DROP TRIGGER fail_run_write")

    self.assertIsNone(await self.bus.wait_for_reply(req, 0))
    self.assertEqual([x["request_id"] for x in self.bus.queued_and_in_flight("r")["in_flight"]], [req])
    [replay] = [t for t in self.bus.handle(NodeMsg("hello", "n", "b", gpus=1)).tasks if t.kind == TaskKind.TRAIN]
    self.assertEqual(replay, task)
    ack = self.bus.handle(NodeMsg("events", "n", "b", done=(done,))).ack
    self.assertEqual(ack, (task.id,))
    self.assertEqual(await self.bus.wait_for_reply(req, 0), OK)
    self.assertEqual(self.bus.service["a"], 5)
    self.assertEqual(self.bus.get_run("r")["steps"], 1)
