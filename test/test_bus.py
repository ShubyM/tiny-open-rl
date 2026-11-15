import asyncio
import json
import unittest

from test.helpers import RedisTest, new_id
from tinyrl.bus import Op, Reply, Run, key


def ok(**body) -> Reply:
  return Reply(True, "application/json", json.dumps(body).encode())


class TestBus(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    await self.bus.take_lease("p", "owner")

  async def new_run(self, session=""):
    run = new_id()
    await self.bus.create_run(Run(run, "p", "base", 8, session=session))
    return run

  async def test_enqueue_pins_inputs_once_and_claims_only_the_budgeted_prefix(self):
    run, uri = await self.new_run(), "tinker://source/weights/input"
    await self.seed_checkpoint(uri, 1, version="input1")
    body = json.dumps({"uri": uri, "version": "input1"}).encode()
    requests = await asyncio.gather(
      *(self.bus.submit_command(run, Op.LOAD_WEIGHTS, body, seq_id=1, cost=3) for _ in range(4))
    )
    self.assertEqual(len(set(requests)), 1)
    self.assertEqual(await self.bus.r.xlen(key("q", run)), 1)
    await self.registry.unregister_checkpoints(uri)
    self.assertEqual(await self.bus.r.hget(key("refs"), f"cmd:{requests[0]}"), b"input1")
    second = await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}", cost=4)
    third = await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}", cost=9)
    head = await self.bus.head(run)
    self.assertEqual(head.request_id, requests[0])
    self.assertIsNone(head.generation)
    first, step = await self.bus.take_commands({run: 0}, "owner", limits={run: 7})
    self.assertEqual((first.request_id, step.request_id), (requests[0], second))
    queue = await self.bus.queued_and_in_flight(run)
    self.assertEqual((queue["queued"], len(queue["in_flight"])), (1, 2))
    self.assertEqual((await self.bus.head(run)).request_id, third)
    [oversized] = await self.bus.take_commands({run: 0}, "owner", limits={run: 1})
    self.assertEqual(oversized.request_id, third)
    waiters = [asyncio.create_task(self.bus.wait_for_reply(first.request_id, 5)) for _ in range(3)]
    self.assertTrue(await self.bus.write_reply(first, ok(), "p", "owner"))
    self.assertEqual(await asyncio.gather(*waiters), [ok()] * 3)
    self.assertFalse(await self.bus.write_reply(first, ok(), "p", "owner"))
    self.assertIsNone(await self.bus.r.hget(key("refs"), f"cmd:{first.request_id}"))
    self.assertTrue(await self.bus.write_reply(step, ok(), "p", "owner", mutated=True))
    self.assertEqual((await self.bus.get_run(run))["steps"], 1)
    self.assertEqual((await self.bus.get_run(run))["state"]["mutation"], step.entry_id)

  async def test_deferred_saves_keep_the_original_future_and_cannot_rewind_an_alias(self):
    run, uri = await self.new_run(), "tinker://another-run/weights/shared"
    for _ in range(2):
      await self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
    first, second = await self.bus.take_commands({run: 0}, "owner")
    saved = []
    for cmd, version in ((first, "first"), (second, "second")):
      cp = {"uri": uri, "version": version, "boundary": cmd.entry_id, "meta": {"optimizer": True}, "durable": False}
      self.assertTrue(await self.bus.write_reply(cmd, ok(), "p", "owner", checkpoint=cp, defer=True))
      self.assertIsNone(await self.bus.wait_for_reply(cmd.request_id, 0))
      self.assertEqual(await self.bus.r.hget(key("refs"), f"save:{cmd.request_id}"), version.encode())
      saved.append(cp)
    self.assertIsNone(await self.registry.get_checkpoint(uri))
    self.assertFalse(await self.bus.write_reply(first, ok(), "p", "owner", checkpoint=saved[0]))
    for cmd, cp in reversed(list(zip((first, second), saved))):
      self.assertTrue(await self.bus.write_reply(cmd, ok(), "p", "owner", checkpoint=cp | {"durable": True}))
      self.assertEqual(await self.bus.wait_for_reply(cmd.request_id, 0), ok())
      self.assertIsNone(await self.bus.r.hget(key("refs"), f"save:{cmd.request_id}"))
    self.assertEqual(await self.registry.version(uri), "second")
    state = (await self.bus.get_run(run))["state"]
    self.assertEqual([state[k]["version"] for k in ("working", "durable")], ["second", "second"])
    await self.bus.submit_command(run, Op.SAVE_WEIGHTS_FOR_SAMPLER, b"{}")
    [cmd] = await self.bus.take_commands({run: 0}, "owner")
    cp = {
      "uri": f"tinker://{run}/sampler_weights/export",
      "version": "sample",
      "boundary": cmd.entry_id,
      "meta": {"optimizer": False},
      "durable": False,
    }
    self.assertTrue(await self.bus.write_reply(cmd, ok(), "p", "owner", checkpoint=cp))
    self.assertEqual((await self.bus.get_run(run))["state"], state)
    self.assertEqual(await self.registry.version(cp["uri"]), "sample")

  async def test_delete_releases_inputs_and_pending_saves_but_keeps_public_checkpoints(self):
    run = await self.new_run()
    owned, public = [f"tinker://{run}/{path}" for path in ("sampler_weights/session", "weights/public")]
    await self.seed_checkpoint(owned, 1, owned=True, version="private")
    await self.seed_checkpoint(public, 1, version="public")
    await self.bus.pin_weights(run, "private")
    request = await self.bus.submit_command(
      run, Op.LOAD_WEIGHTS, json.dumps({"uri": owned, "version": "private"}).encode(), seq_id=1
    )
    await self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
    load, save = await self.bus.take_commands({run: 0}, "owner")
    cp = {
      "uri": f"tinker://{run}/weights/pending",
      "version": "pending",
      "boundary": save.entry_id,
      "meta": {"optimizer": True},
      "durable": False,
    }
    self.assertTrue(await self.bus.write_reply(save, ok(), "p", "owner", checkpoint=cp, defer=True))
    await self.bus.delete_run(run, "p")
    for cmd in (load, save):
      self.assertFalse((await self.bus.wait_for_reply(cmd.request_id, 0)).ok)
    self.assertEqual(await self.bus.r.hgetall(key("refs")), {f"ckpt:{public}".encode(): b"public"})
    self.assertIsNone(await self.registry.get_checkpoint(owned))
    self.assertEqual(await self.registry.version(public), "public")
    self.assertEqual(await self.bus.queued_and_in_flight(run), {"live": False})
    self.assertEqual(await self.bus.submit_command(run, Op.LOAD_WEIGHTS, b"{}", seq_id=1), request)
    self.assertIsNone(await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}", seq_id=2))
    self.assertFalse(await self.bus.r.exists(key("saving", run)))

  async def test_sessions_and_live_nodes(self):
    await self.bus.heartbeat_session("s", {"tag": "test"})
    run = await self.new_run("s")
    session = await self.bus.get_session("s")
    self.assertEqual((session["runs"], session["metadata"], session["live"]), ([run], {"tag": "test"}, True))
    await self.bus.finish_session("s")
    self.assertEqual(await self.bus.expired_sessions(), ["s"])
    await self.bus.end_session("s")
    self.assertEqual((await self.bus.head(run)).op, Op.DELETE_MODEL)
    await self.bus.delete_run(run, "p")
    await self.bus.end_session("s")
    self.assertIsNone(await self.bus.r.zscore(key("seen"), "s"))
    await self.bus.node_heartbeat("live", 2)
    await self.bus.r.hset(key("nodes"), "dead", '{"gpus": 8, "until": 0}')
    self.assertEqual({n: info["gpus"] for n, info in (await self.bus.read_table())["nodes"].items()}, {"live": 2})


if __name__ == "__main__":
  unittest.main()
