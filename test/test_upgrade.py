from test.helpers import BusTest
from tinyrl.control.ledger import Ledger
from tinyrl.types import NodeStatus, Op, Reply, Role, Run

OK = Reply(True, "application/json", b"{}")


class TestPopulatedUpgrade(BusTest):
  async def test_new_owner_fails_old_pending_command_and_keeps_the_queued_tail(self):
    self.bus.create_run(Run("t", "base", 8, role=Role.TRAINER))
    first = self.bus.submit_command("t", Op.OPTIM_STEP, b"\xff")
    second = self.bus.submit_command("t", Op.OPTIM_STEP, b"\xff")
    [a1], _ = await self.bus.heartbeat("n1", "b1", NodeStatus(1))
    [w1] = self.bus.take("n1", "b1", a1.worker)
    self.assertEqual(w1.cmd.request_id, first)

    self.bus.nodes["n1"].until = 0.0
    [a2], _ = await self.bus.heartbeat("n2", "b2", NodeStatus(1))
    self.assertFalse((await self.bus.wait_for_reply(first, 0)).ok)
    self.assertEqual(self.bus.get_run("t")["failed"], 1)
    [w2] = self.bus.take("n2", "b2", a2.worker)
    self.assertEqual((w2.cmd.request_id, w2.cmd.body), (second, b"\xff"))
    self.bus.finish("n2", "b2", a2.worker, w2.cmd.request_id, OK, mutated=True)
    self.assertEqual(await self.bus.wait_for_reply(second, 0), OK)

  async def test_sampler_recovers_abandoned_claim_alongside_fresh_entries(self):
    self.bus.create_run(Run("s", "base", 8, role=Role.SAMPLER))
    first = self.bus.submit_command("s", Op.SAMPLE, b"{}")
    [a1], _ = await self.bus.heartbeat("n1", "b1", NodeStatus(1))
    [w1] = self.bus.take("n1", "b1", a1.worker)
    self.assertEqual(w1.cmd.request_id, first)

    second = self.bus.submit_command("s", Op.SAMPLE, b"{}")
    self.bus.nodes["n1"].until = 0.0
    [a2], _ = await self.bus.heartbeat("n2", "b2", NodeStatus(1))
    claimed = self.bus.take("n2", "b2", a2.worker, slots=4)
    self.assertEqual([w.cmd.request_id for w in claimed], [first, second])
    for w in claimed:
      self.bus.finish("n2", "b2", a2.worker, w.cmd.request_id, OK)
    self.assertEqual(await self.bus.wait_for_reply(first, 0), OK)
    self.assertEqual(await self.bus.wait_for_reply(second, 0), OK)
    self.assertEqual(self.bus.queued_and_in_flight("s"), {"live": True, "queued": 0, "in_flight": []})

  async def test_persisted_metadata_and_trainer_replies_survive_db_reopen(self):
    self.bus.heartbeat_session("sess-1", {"sdk": "0.39"})
    self.bus.create_run(Run("t", "base", 8, session="sess-1", role=Role.TRAINER))
    self.seed_checkpoint("tinker://t/weights/v1", 42, version="ver-1")
    req = self.bus.submit_command("t", Op.OPTIM_STEP, b"{}", seq_id=7)
    [a1], _ = await self.bus.heartbeat("n1", "b1", NodeStatus(1))
    [w] = self.bus.take("n1", "b1", a1.worker)
    self.bus.finish("n1", "b1", a1.worker, w.cmd.request_id, OK, mutated=True)
    await self.bus.aclose()

    self.bus = Ledger(self.db_path)
    self.assertEqual(self.bus.get_session("sess-1")["metadata"], {"sdk": "0.39"})
    self.assertEqual(self.bus.checkpoint_version("tinker://t/weights/v1"), "ver-1")
    self.assertEqual(await self.bus.wait_for_reply(req, 0), OK)
    self.assertEqual(self.bus.seen_request("t", 7), req)
