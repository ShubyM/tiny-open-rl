import sqlite3

from test.helpers import BusTest
from tinyrl.control.accounting import Charge
from tinyrl.types import NodeStatus, Op, Reply, Role, Run

OK = Reply(True, "application/json", b"{}")


class TestClaimStorage(BusTest):
  async def test_sample_keeps_its_version_on_retry_and_reassignment(self):
    uri = "tinker://trainer/sampler_weights/latest"
    self.seed_checkpoint(uri, 1, version="first")
    self.bus.create_run(Run("s", "base", 8, role=Role.SAMPLER, model_path=uri))
    first = self.bus.submit_command("s", Op.SAMPLE, b"{}")
    staged = NodeStatus(1, staged=("first", "second"))

    [initial] = [g for g in (await self.bus.take("east", "boot1", staged)).grants if g.work]
    self.seed_checkpoint(uri, 1, version="second")
    second = self.bus.submit_command("s", Op.SAMPLE, b"{}")
    [replay] = [g for g in (await self.bus.take("east", "boot1", staged)).grants if g.work]
    self.assertEqual(replay, initial)
    self.assertIn("first", self.bus.live_versions())

    self.bus.nodes["east"].until = 0
    [retry] = [g for g in (await self.bus.take("west", "boot2", staged)).grants if g.work]
    self.assertEqual([(w.cmd.request_id, w.version) for w in retry.work], [(first, "first")])
    self.assertTrue(self.bus.finish("west", "boot2", retry.worker, first, OK, attempt=retry.work[0].attempt))
    self.assertNotIn("first", self.bus.live_versions())
    [new] = [g for g in (await self.bus.take("west", "boot2", staged)).grants if g.work]
    self.assertEqual([(w.cmd.request_id, w.version) for w in new.work], [(second, "second")])

  async def test_redelivered_wave_preserves_interleaved_run_order(self):
    for run in ("a", "b"):
      self.bus.create_run(Run(run, "base", 8, role=Role.SAMPLER, account="owner"))
      for _ in range(2):
        self.bus.submit_command(run, Op.SAMPLE, b"{}")
    [initial] = [g for g in (await self.bus.take("n", "boot", NodeStatus(1))).grants if g.work]
    self.assertEqual([w.cmd.run for w in initial.work], ["a", "b", "a", "b"])
    [replay] = [g for g in (await self.bus.take("n", "boot", NodeStatus(1))).grants if g.work]
    self.assertEqual(replay, initial)

  async def test_failed_finish_preserves_trainer_claim_and_bills_once_on_retry(self):
    self.bus.create_run(Run("r", "base", 8, account="a"))
    req = self.bus.submit_command("r", Op.OPTIM_STEP, b"{}")
    [grant] = [g for g in (await self.bus.take("n", "b", NodeStatus(1))).grants if g.work]
    # Fail the run update after the reply and billing writes have already succeeded.
    self.bus.db.conn.execute(
      """CREATE TEMP TRIGGER fail_run_write BEFORE INSERT ON kv WHEN NEW.ns='runs'
         BEGIN SELECT RAISE(ABORT, 'injected write failure'); END"""
    )
    try:
      with self.assertRaisesRegex(sqlite3.IntegrityError, "injected write failure"):
        self.bus.finish("n", "b", grant.worker, req, OK, Charge(gpu_ms=5), mutated=True)
    finally:
      self.bus.db.conn.execute("DROP TRIGGER fail_run_write")

    self.assertIsNone(await self.bus.wait_for_reply(req, 0))
    self.assertEqual([x["request_id"] for x in self.bus.queued_and_in_flight("r")["in_flight"]], [req])
    [replay] = [g for g in (await self.bus.take("n", "b", NodeStatus(1))).grants if g.work]
    self.assertEqual(replay, grant)
    self.assertTrue(self.bus.finish("n", "b", grant.worker, req, OK, Charge(gpu_ms=5), mutated=True))
    self.assertEqual(await self.bus.wait_for_reply(req, 0), OK)
    self.assertEqual(self.bus.service["a"], 5)
    self.assertEqual(self.bus.get_run("r")["steps"], 1)
