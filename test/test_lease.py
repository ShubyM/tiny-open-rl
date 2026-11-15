import asyncio
import copy
import json
import unittest

from test.helpers import RedisTest
from test.test_bus import ok
from tinyrl.bus import GROUP, Op, Run, key
from tinyrl.control import scheduler


class TestLease(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    await self.bus.create_run(Run("r", "p", "base", 0))
    old = scheduler.from_table(await self.bus.read_table())
    new = copy.deepcopy(old)
    new.workers["p"] = scheduler.Worker(old.jobs["r"].spec, "n", (0,), 1, {"r": 10}, {"r": 0}, "one")
    self.assertTrue(await scheduler.commit(self.bus, old, new))

  async def test_generation_and_owner_fence_replies_and_reclaim(self):
    owners = ["one", "two"]
    taken = await asyncio.gather(*(self.bus.take_lease("p", owner) for owner in owners))
    self.assertEqual(sum(taken), 1)
    owner = owners[taken.index(True)]
    for op in (Op.OPTIM_STEP, Op.SAMPLE, Op.SAVE_WEIGHTS):
      await self.bus.submit_command("r", op, b"{}")
    mutation, sample, save = await self.bus.take_commands({"r": 0}, owner)
    cp = {
      "uri": "tinker://r/weights/saved",
      "version": "saved",
      "boundary": save.entry_id,
      "meta": {"optimizer": True},
      "durable": False,
    }
    self.assertTrue(await self.bus.write_reply(save, ok(), "p", owner, checkpoint=cp, defer=True))
    await self.bus.release_lease("p", "stranger")
    self.assertTrue(await self.bus.renew_lease("p", owner))
    old = scheduler.from_table(await self.bus.read_table())
    new = copy.deepcopy(old)
    new.jobs["r"].generation = new.workers["p"].members["r"] = 1
    self.assertTrue(await scheduler.commit(self.bus, old, new))
    self.assertFalse(await self.bus.write_reply(mutation, ok(), "p", owner, mutated=True))
    self.assertEqual(await self.bus.reclaim_commands({"r": 0}, "p", owner), [])
    self.assertEqual(await self.bus.take_commands({"r": 0}, owner), [])
    await self.bus.release_lease("p", owner)
    self.assertTrue(await self.bus.take_lease("p", "next"))
    self.assertFalse(await self.bus.renew_lease("p", owner))
    self.assertFalse(await self.bus.write_reply(save, ok(), "p", owner, checkpoint=cp | {"durable": True}))
    self.assertEqual(await self.bus.reclaim_commands({"r": 1}, "p", owner), [])
    replay = await self.bus.reclaim_commands({"r": 1}, "p", "next")
    self.assertEqual(
      [(c.entry_id, c.generation, saved) for c, saved in replay], [(sample.entry_id, 1, None), (save.entry_id, 1, cp)]
    )
    self.assertFalse((await self.bus.wait_for_reply(mutation.request_id, 0)).ok)
    self.assertEqual(len((await self.bus.queued_and_in_flight("r"))["in_flight"]), 2)
    for cmd, saved in replay:
      self.assertTrue(
        await self.bus.write_reply(cmd, ok(), "p", "next", checkpoint=saved | {"durable": True} if saved else None)
      )
    self.assertEqual((await self.bus.r.xpending(key("q", "r"), GROUP))["pending"], 0)
    self.assertEqual(await self.registry.version(cp["uri"]), "saved")

  async def test_observations_fence_gpu_start_and_acknowledge_the_current_close(self):
    self.assertTrue(await self.bus.take_lease("p", "one"))
    obs = {"node": "n", "revision": 1, "gpus": [0], "released": False, "closed": False, "resident": {}}
    for bad in ({"revision": 0}, {"gpus": [1]}):
      self.assertFalse(await self.bus.write_observation("p", "one", obs | bad, starting=True))
    self.assertTrue(await self.bus.write_observation("p", "one", obs, starting=True))
    old = scheduler.from_table(await self.bus.read_table())
    new = copy.deepcopy(old)
    new.workers["p"].budgets["r"] = 20
    self.assertTrue(await self.bus.write_observation("p", "one", obs | {"usage": {"r": 1.0}}))
    self.assertTrue(await scheduler.commit(self.bus, old, new))  # accounting updates do not reject a safe grant
    old = scheduler.from_table(await self.bus.read_table())
    new = copy.deepcopy(old)
    new.workers["p"].revision += 1
    self.assertTrue(await self.bus.write_observation("p", "one", obs | {"released": True}))
    self.assertFalse(await scheduler.commit(self.bus, old, new))
    old = scheduler.from_table(await self.bus.read_table())
    new = copy.deepcopy(old)
    new.workers["p"].revision = 2
    self.assertTrue(await scheduler.commit(self.bus, old, new))
    self.assertTrue(await self.bus.write_observation("p", "one", obs | {"closed": True, "released": True}))
    actual = json.loads(await self.bus.r.hget(key("obs"), "p"))
    self.assertEqual(actual["revision"], 2)


if __name__ == "__main__":
  unittest.main()
