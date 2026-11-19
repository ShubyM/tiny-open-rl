import json

from test.helpers import RedisTest
from tinyrl.bus import key
from tinyrl.control import scheduler
from tinyrl.types import Op, Reply

OK = Reply(True, "application/json", b"{}")


class TestPopulatedUpgrade(RedisTest):
  async def main_run(self, run, role):
    # These are the keys and stream group written by main before trainer and sampler placement diverged.
    p = self.bus.s.pipeline()
    p.sadd(key("runs"), run)
    p.hset(key("run", run), mapping={"record": json.dumps({"role": role, "base": "base", "rank": 8})})
    p.xgroup_create(key("q", run), "trainer", id="0", mkstream=True)
    await p.execute()

  async def test_scheduler_keeps_trainer_home_and_derives_existing_sampler_pool(self):
    await self.main_run("t", "trainer")
    await self.main_run("s", "sampler")
    await self.bus.node_heartbeat("n", 2)
    old_plan = {
      "made": 2,
      "procs": {
        p: {"node": "n", "spec": {"role": role, "base": "base", "lora": True}}
        for p, role in [("p1", "trainer"), ("p2", "sampler")]
      },
      "home": {"t": ["p1"], "s": ["p2"]},
    }
    await self.bus.s.set(key("plan"), json.dumps(old_plan))
    for _ in range(2):
      await scheduler.tick(self.bus)
      raw = await self.bus.read_state()
      self.assertEqual(raw["plan"]["home"], {"t": "p1"})
      state = scheduler.from_state(raw)
      self.assertEqual(state.procs_of("t"), ["p1"])
      self.assertEqual(state.procs_of("s"), ["p2"])
      self.assertEqual(state.procs["p1"].node, "n")

  async def test_new_owner_fails_old_pending_command_and_keeps_the_queued_tail(self):
    await self.main_run("t", "trainer")
    run, queue = key("run", "t"), key("q", "t")
    snap = "tinker://t/parked/version\nv1"
    await self.bus.s.hset(run, mapping={"owner": "old:1", "epoch": 1, "pos": "1-0", "snap": snap})
    await self.bus.s.set(key("lease", "old"), 1, px=60000)
    for eid in ("2-0", "3-0"):
      await self.bus.r.xadd(queue, {"op": Op.OPTIM_STEP, "body": b"\xff"}, id=eid)
    await self.bus.r.xreadgroup("trainer", "old", {queue: ">"}, count=1)

    await self.bus.touch_leases("new")
    self.assertIsNone(await self.bus.acquire_run("t", "new"))
    await self.bus.drop_lease("old")
    token, pos, restored = await self.bus.acquire_run("t", "new")
    self.assertEqual((token, pos, restored), ("new:2", "1-0", snap))
    await self.bus.fail_stranded_commands("t", token)
    await self.bus.fail_stranded_commands("t", token)
    self.assertFalse((await self.bus.wait_for_reply("t/2-0", 0)).ok)
    self.assertEqual((await self.bus.get_run("t"))["failed"], 1)
    self.assertIsNone(await self.bus.wait_for_reply("t/3-0", 0))
    tail = await self.bus.claim_trainer("t", token)
    self.assertEqual((tail.entry_id, tail.body), ("3-0", b"\xff"))
    self.assertTrue(await self.bus.commit_reply(tail, OK, token, mutates=True))
    self.assertEqual(await self.bus.wait_for_reply("t/3-0", 0), OK)

  async def test_sampler_recovers_main_pending_list_alongside_fresh_entries(self):
    await self.main_run("s", "sampler")
    q = key("q", "s")
    for eid in ("1-0", "2-0"):
      await self.bus.s.xadd(q, {"op": Op.SAMPLE, "body": "{}"}, id=eid)
    await self.bus.s.xreadgroup("trainer", "old", {q: ">"}, count=1)
    await self.bus.s.xclaim(q, "trainer", "old", 0, ["1-0"], idle=2000)
    [head] = await self.bus.peek_sampler(["s"])
    self.assertEqual(head.request_id, "s/1-0")
    batch = await self.bus.take_commands(["s"], "replacement", 2)
    self.assertEqual([c.entry_id for c in batch], ["1-0", "2-0"])
    for cmd in batch:
      self.assertTrue(await self.bus.commit_sample(cmd, OK))
    self.assertFalse(await self.bus.commit_sample(batch[0], OK))
    self.assertEqual(await self.bus.queued_and_in_flight("s"), {"live": True, "queued": 0, "in_flight": []})
