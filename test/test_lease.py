import asyncio
import json
import unittest

from test.helpers import RedisTest
from test.test_bus import grant, ok
from tinyrl.bus import Op, Role, Run, key


class TestLease(RedisTest):
  async def test_launch_and_release_use_the_original_incarnation_not_lease_expiry(self):
    await self.bus.create_run(Run("r", "base", 0))
    await self.bus.node_heartbeat("node", 1)
    await self.bus.r.hset(key("claims"), "r", json.dumps({"node": "node", "engine": "engine", "token": "claim"}))
    wins = await asyncio.gather(*(self.bus.start_engine("engine", "node") for _ in range(2)))
    self.assertEqual(sum(wins), 1)
    turn = {"node": "node", "engine": "engine", "gpus": [0], "work": {"r": ["claim", 10]}}
    await self.bus.r.hset(key("turns"), "turn", json.dumps(turn))
    obs = {"node": "node", "gpus": [0], "ready": ["r"], "closed": False}
    self.assertFalse(await self.bus.write_observation("engine", obs | {"gpus": [1]}, turn="turn"))
    self.assertTrue(await self.bus.write_observation("engine", obs, turn="turn"))
    self.assertFalse(await self.bus.release_turn("turn", "stranger", {"r": 1.5}))
    await self.bus.r.pexpire(key("live", "engine"), 0)
    self.assertFalse(await self.bus.renew_lease("engine"))
    self.assertFalse(await self.bus.start_engine("engine", "node"))
    self.assertTrue(await self.bus.r.hexists(key("turns"), "turn"))
    self.assertFalse(await self.bus.write_observation("engine", obs, turn="turn"))
    self.assertTrue(await self.bus.release_turn("turn", "engine", {"r": 1.5}))
    self.assertFalse(await self.bus.release_turn("turn", "engine", {"r": 1.5}))
    self.assertEqual((await self.bus.read_table())["service"], {"r": 1.5})
    self.assertTrue(await self.bus.write_observation("engine", obs | {"closed": True}))
    self.assertFalse(await self.bus.start_engine("engine", "node"))

  async def test_claim_fencing_fails_uncertain_training_then_reprepares_the_next_command(self):
    await self.bus.create_run(Run("r", "base", 0))
    await grant(self.bus, "r")
    first = await self.bus.submit_command("r", Op.OPTIM_STEP, b"{}")
    [old] = await self.bus.take_commands("turn", "r")
    await self.seed_checkpoint("tinker://source/weights/saved", version="saved")
    second = await self.bus.submit_command(
      "r", Op.LOAD_WEIGHTS, b'{"uri":"tinker://source/weights/saved","version":"saved"}'
    )
    await self.bus.r.pexpire(key("live", "engine"), 0)
    await grant(self.bus, "r", engine="fresh", node="other-node", token="new-claim", turn="new-turn")
    self.assertFalse(await self.bus.write_reply(old, ok(), "engine", mutated=True))
    self.assertEqual(await self.bus.take_commands("turn", "r"), [])
    self.assertEqual((await self.bus.head("r")).request_id, first)
    self.assertEqual(await self.bus.take_commands("new-turn", "r"), [])
    self.assertFalse((await self.bus.wait_for_reply(first, 0)).ok)
    self.assertEqual((await self.bus.head("r")).request_id, second)
    [load] = await self.bus.take_commands("new-turn", "r")
    cp = {
      "uri": "tinker://source/weights/saved",
      "version": "saved",
      "boundary": load.entry_id,
      "optimizer": True,
      "durable": True,
    }
    self.assertTrue(await self.bus.write_reply(load, ok(), "fresh", checkpoint=cp, mutated=True))
    self.assertTrue(await self.bus.release_turn("new-turn", "fresh", {}))
    await grant(self.bus, "r", engine="fresh", node="other-node", token="new-claim", turn="export", allowance=0)
    self.assertFalse(await self.bus.export("export", "r", cp | {"boundary": "0-0"}, "fresh"))
    self.assertTrue(await self.bus.export("export", "r", cp, "fresh"))
    self.assertFalse(await self.bus.r.hexists(key("claims"), "r"))
    self.assertTrue(await self.bus.r.hexists(key("turns"), "export"))
    self.assertEqual((await self.bus.get_run("r"))["state"]["working"]["version"], "saved")

  async def test_sampling_reclaims_its_pending_input_and_old_turns_cannot_publish(self):
    await self.seed_checkpoint("tinker://source/sampler_weights/saved", version="weights")
    await self.bus.create_run(
      Run("r", "base", 1, role=Role.SAMPLER, model_path="tinker://source/sampler_weights/saved", version="weights")
    )
    await grant(self.bus, "r")
    await self.bus.submit_command("r", Op.SAMPLE, b"{}")
    [old] = await self.bus.take_commands("turn", "r")
    self.assertTrue(await self.bus.release_turn("turn", "engine", {}))
    await grant(self.bus, "r", turn="next")
    [replay] = await self.bus.take_commands("next", "r")
    self.assertEqual((replay.request_id, replay.turn), (old.request_id, "next"))
    self.assertFalse(await self.bus.write_reply(old, ok(), "engine"))
    self.assertTrue(await self.bus.write_reply(replay, ok(), "engine"))
    self.assertEqual(await self.bus.r.hget(key("refs"), "job:r:weights"), b"weights")
    obs = {"node": "node", "gpus": [0], "ready": ["r"], "closed": False}
    self.assertFalse(await self.bus.write_observation("engine", obs, turn="turn"))


if __name__ == "__main__":
  unittest.main()
