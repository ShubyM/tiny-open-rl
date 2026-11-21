from test.helpers import BusTest, finished
from tinyrl.bus import Bus
from tinyrl.types import NodeReport, Op, Reply, Role, Run

OK = Reply(True, "application/json", b"{}")


class TestPopulatedUpgrade(BusTest):
  async def test_new_owner_fails_old_pending_command_and_keeps_the_queued_tail(self):
    await self.bus.create_run(Run("t", "base", 8, role=Role.TRAINER))
    first = await self.bus.submit_command("t", Op.OPTIM_STEP, b"\xff")
    second = await self.bus.submit_command("t", Op.OPTIM_STEP, b"\xff")
    reply1 = await self.bus.exchange(NodeReport("n1", "b1", 1))
    [t1] = reply1.turns
    self.assertEqual(t1.cmd.request_id, first)

    # n1 disappears; n2 takes over after n1 is reaped, first command fails and tail runs on n2.
    self.bus.nodes["n1"].until = 0.0
    reply2 = await self.bus.exchange(NodeReport("n2", "b2", 1))
    self.assertFalse((await self.bus.wait_for_reply(first, 0)).ok)
    self.assertEqual((await self.bus.get_run("t"))["failed"], 1)
    [t2] = reply2.turns
    self.assertEqual((t2.cmd.request_id, t2.cmd.body), (second, b"\xff"))
    await self.bus.exchange(NodeReport("n2", "b2", 1, finished=[finished(t2, OK, mutated=True)]))
    self.assertEqual(await self.bus.wait_for_reply(second, 0), OK)

  async def test_sampler_recovers_abandoned_claim_alongside_fresh_entries(self):
    await self.bus.create_run(Run("s", "base", 8, role=Role.SAMPLER))
    first = await self.bus.submit_command("s", Op.SAMPLE, b"{}")
    reply1 = await self.bus.exchange(NodeReport("n1", "b1", 1))
    [t1] = reply1.turns
    self.assertEqual(t1.cmd.request_id, first)

    second = await self.bus.submit_command("s", Op.SAMPLE, b"{}")
    self.bus.nodes["n1"].until = 0.0
    reply2 = await self.bus.exchange(NodeReport("n2", "b2", 1))
    self.assertEqual([t.cmd.request_id for t in reply2.turns], [first, second])
    await self.bus.exchange(NodeReport("n2", "b2", 1, finished=[finished(t, OK) for t in reply2.turns]))
    self.assertEqual(await self.bus.wait_for_reply(first, 0), OK)
    self.assertEqual(await self.bus.wait_for_reply(second, 0), OK)
    self.assertEqual(await self.bus.queued_and_in_flight("s"), {"live": True, "queued": 0, "in_flight": []})

  async def test_persisted_metadata_and_trainer_replies_survive_db_reopen(self):
    await self.bus.heartbeat_session("sess-1", {"sdk": "0.39"})
    await self.bus.create_run(Run("t", "base", 8, session="sess-1", role=Role.TRAINER))
    await self.seed_checkpoint("tinker://t/weights/v1", 42, version="ver-1")
    req = await self.bus.submit_command("t", Op.OPTIM_STEP, b"{}", seq_id=7)
    reply = await self.bus.exchange(NodeReport("n1", "b1", 1))
    [turn] = reply.turns
    await self.bus.exchange(NodeReport("n1", "b1", 1, finished=[finished(turn, OK, mutated=True)]))
    await self.bus.aclose()

    self.bus = Bus(self.db_path)
    self.assertEqual((await self.bus.get_session("sess-1"))["metadata"], {"sdk": "0.39"})
    self.assertEqual(await self.bus.checkpoint_version("tinker://t/weights/v1"), "ver-1")
    self.assertEqual(await self.bus.wait_for_reply(req, 0), OK)
    self.assertEqual(await self.bus.seen_request("t", 7), req)
