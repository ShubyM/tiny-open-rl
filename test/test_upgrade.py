from test.helpers import BusTest
from tinyrl.bus import Bus, Claim
from tinyrl.control import scheduler
from tinyrl.types import Cmd, Op, Reply, Role, Run

OK = Reply(True, "application/json", b"{}")


class TestPopulatedUpgrade(BusTest):
  async def test_scheduler_keeps_trainer_home_and_derives_existing_sampler_pool(self):
    await self.bus.create_run(Run("t", "base", 8, role=Role.TRAINER))
    await self.bus.create_run(Run("s", "base", 8, role=Role.SAMPLER))
    await self.bus.node_heartbeat("n", 2)
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
      await scheduler.tick(self.bus)
      raw = await self.bus.read_state()
      self.assertEqual(raw["plan"]["home"], {"t": "p1"})
      state = scheduler.from_state(raw)
      self.assertEqual(state.procs_of("t"), ["p1"])
      self.assertEqual(state.procs_of("s"), ["p2"])
      self.assertEqual(state.procs["p1"].node, "n")

  async def test_new_owner_fails_old_pending_command_and_keeps_the_queued_tail(self):
    await self.bus.create_run(Run("t", "base", 8, role=Role.TRAINER))
    snap = "tinker://t/parked/version\nv1"
    with self.bus.db:
      self.bus.runs["t"].update(owner="old:1", epoch=1, pos="1-0", snap=snap)
      self.bus.runs.save("t")
    await self.bus.touch_leases("old")
    first = await self.bus.submit_command("t", Op.OPTIM_STEP, b"\xff")
    second = await self.bus.submit_command("t", Op.OPTIM_STEP, b"\xff")
    await self.bus.claim_trainer("t", "old:1")

    await self.bus.touch_leases("new")
    self.assertIsNone(await self.bus.acquire_run("t", "new"))
    await self.bus.drop_lease("old")
    token, pos, restored = await self.bus.acquire_run("t", "new")
    self.assertEqual((token, pos, restored), ("new:2", "1-0", snap))
    await self.bus.fail_stranded_commands("t", token)
    await self.bus.fail_stranded_commands("t", token)
    self.assertFalse((await self.bus.wait_for_reply(first, 0)).ok)
    self.assertEqual((await self.bus.get_run("t"))["failed"], 1)
    self.assertIsNone(await self.bus.wait_for_reply(second, 0))
    tail = await self.bus.claim_trainer("t", token)
    self.assertEqual((tail.request_id, tail.body), (second, b"\xff"))
    self.assertTrue(await self.bus.commit_reply(tail, OK, token, mutates=True))
    self.assertEqual(await self.bus.wait_for_reply(second, 0), OK)

  async def test_sampler_recovers_abandoned_claim_alongside_fresh_entries(self):
    await self.bus.create_run(Run("s", "base", 8, role=Role.SAMPLER))
    first_cmd = Cmd("s", "1-0", Op.SAMPLE, b"{}")
    second_cmd = Cmd("s", "2-0", Op.SAMPLE, b"{}")
    self.bus.in_flight["s"]["1-0"] = Claim(first_cmd, "old", 0.0, 1)
    self.bus.queues["s"]["2-0"] = second_cmd
    [head] = await self.bus.peek_sampler(["s"])
    self.assertEqual(head.request_id, "s/1-0")
    batch = await self.bus.take_commands(["s"], "replacement", 2)
    self.assertEqual([c.entry_id for c in batch], ["1-0", "2-0"])
    for cmd in batch:
      self.assertTrue(await self.bus.commit_sample(cmd, OK))
    self.assertFalse(await self.bus.commit_sample(batch[0], OK))
    self.assertEqual(await self.bus.queued_and_in_flight("s"), {"live": True, "queued": 0, "in_flight": []})

  async def test_persisted_metadata_survives_db_reopen(self):
    await self.bus.heartbeat_session("sess-1", {"sdk": "0.39"})
    await self.bus.create_run(Run("t", "base", 8, session="sess-1", role=Role.TRAINER))
    await self.seed_checkpoint("tinker://t/weights/v1", 42, version="ver-1")
    await self.bus.write_plan({"made": 1, "procs": {}, "home": {}}, notify_nodes=False)
    await self.bus.aclose()

    self.bus = Bus(self.db_path)
    self.assertEqual((await self.bus.get_session("sess-1"))["metadata"], {"sdk": "0.39"})
    self.assertEqual(await self.bus.checkpoint_version("tinker://t/weights/v1"), "ver-1")
    self.assertEqual((await self.bus.read_state())["plan"]["made"], 1)
