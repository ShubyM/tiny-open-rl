from test.helpers import BusTest
from tinyrl.bus import Bus
from tinyrl.control import scheduler
from tinyrl.types import NodeReport, Op, Reply, Role, Run, TurnResult

OK = Reply(True, "application/json", b"{}")


class TestPopulatedUpgrade(BusTest):
  async def test_scheduler_keeps_trainer_home_and_derives_existing_sampler_pool(self):
    await self.bus.create_run(Run("t", "base", 8, role=Role.TRAINER))
    await self.bus.create_run(Run("s", "base", 8, role=Role.SAMPLER))
    await self.bus.exchange(NodeReport("n", "b1", 2))
    old_plan = {
      "made": 2,
      "procs": {
        p: {"node": "n", "spec": {"role": role, "base": "base", "lora": True}}
        for p, role in [("p1", "trainer"), ("p2", "sampler")]
      },
      "home": {"t": ["p1"], "s": ["p2"]},
    }
    with self.bus.db:
      self.bus.meta["plan"] = old_plan
    for _ in range(2):
      await self.bus.maintain()
      raw = await self.bus.read_state()
      self.assertEqual(raw["plan"]["home"], {"t": "p1"})
      state = scheduler.from_state(raw)
      self.assertEqual(state.procs_of("t"), ["p1"])
      self.assertEqual(state.procs_of("s"), ["p2"])
      self.assertEqual(state.procs["p1"].node, "n")

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
    await self.bus.exchange(
      NodeReport("n2", "b2", 1, finished=[TurnResult(t2.id, "train", t2.worker, reply=OK, mutated=True)])
    )
    self.assertEqual(await self.bus.wait_for_reply(second, 0), OK)

  async def test_sampler_recovers_abandoned_claim_alongside_fresh_entries(self):
    await self.bus.create_run(Run("s", "base", 8, role=Role.SAMPLER))
    first = await self.bus.submit_command("s", Op.SAMPLE, b"{}")
    reply1 = await self.bus.exchange(NodeReport("n1", "b1", 1))
    [t1] = reply1.turns
    self.assertEqual([c.request_id for c in t1.cmds], [first])

    second = await self.bus.submit_command("s", Op.SAMPLE, b"{}")
    self.bus.nodes["n1"].until = 0.0
    reply2 = await self.bus.exchange(NodeReport("n2", "b2", 1))
    [t2] = reply2.turns
    self.assertEqual([c.request_id for c in t2.cmds], [first, second])
    samples = [(c.request_id, OK, None) for c in t2.cmds]
    await self.bus.exchange(
      NodeReport("n2", "b2", 1, finished=[TurnResult(t2.id, "sample", t2.worker, samples=samples)])
    )
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
    await self.bus.exchange(
      NodeReport("n1", "b1", 1, finished=[TurnResult(turn.id, "train", turn.worker, reply=OK, mutated=True)])
    )
    await self.bus.aclose()

    self.bus = Bus(self.db_path)
    self.assertEqual((await self.bus.get_session("sess-1"))["metadata"], {"sdk": "0.39"})
    self.assertEqual(await self.bus.checkpoint_version("tinker://t/weights/v1"), "ver-1")
    self.assertEqual(await self.bus.wait_for_reply(req, 0), OK)
    self.assertEqual(await self.bus.seen_request("t", 7), req)
