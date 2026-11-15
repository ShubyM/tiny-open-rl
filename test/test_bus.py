import asyncio
import json
import unittest

from test.helpers import RedisTest, new_id
from tinyrl.bus import GROUP, Op, Reply, Run, key


def ok(**body) -> Reply:
  return Reply(True, "application/json", json.dumps(body).encode())


async def grant(bus, run, *, engine="engine", node="node", token="claim", turn="turn", allowance=100):
  await bus.node_heartbeat(node, 2)
  await bus.r.hset(key("claims"), run, json.dumps({"node": node, "engine": engine, "token": token}))
  if not await bus.r.hexists(key("obs"), engine):
    assert await bus.start_engine(engine, node)
  await bus.r.hset(
    key("turns"), turn, json.dumps({"node": node, "engine": engine, "gpus": [0], "work": {run: [token, allowance]}})
  )


class TestBus(RedisTest):
  async def new_run(self, session=""):
    run = new_id()
    await self.bus.create_run(Run(run, "base", 8, session=session))
    await grant(self.bus, run)
    return run

  async def test_exact_inputs_deduplicate_and_claim_only_a_budgeted_prefix(self):
    run, uri = await self.new_run(), "tinker://source/weights/input"
    await self.seed_checkpoint(uri, version="input1")
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
    self.assertEqual((await self.bus.head(run)).turn, "")
    first, step = await self.bus.take_commands("turn", run, limit=7)
    self.assertEqual((first.request_id, step.request_id, step.turn), (requests[0], second, "turn"))
    self.assertEqual((await self.bus.head(run)).request_id, first.request_id)  # pending heads precede unclaimed work
    [oversized] = await self.bus.take_commands("turn", run, limit=1)
    self.assertEqual(oversized.request_id, third)
    self.assertEqual(await self.bus.take_commands("turn", run), [])  # same-turn work is never redelivered
    waiters = [asyncio.create_task(self.bus.wait_for_reply(first.request_id, 5)) for _ in range(3)]
    self.assertTrue(await self.bus.write_reply(first, ok(), "engine"))
    self.assertEqual(await asyncio.gather(*waiters), [ok()] * 3)
    self.assertFalse(await self.bus.write_reply(first, ok(), "engine"))
    self.assertIsNone(await self.bus.r.hget(key("refs"), f"cmd:{first.request_id}"))
    self.assertTrue(await self.bus.write_reply(step, ok(), "engine", mutated=True))
    self.assertEqual((await self.bus.get_run(run))["state"]["mutation"], step.entry_id)
    self.assertNotIn("worker", (await self.bus.read_table())["runs"][run])

  async def test_save_obligations_keep_the_original_future_and_version_after_engine_loss(self):
    run, uri = await self.new_run(), "tinker://another-run/weights/shared"
    for _ in range(2):
      await self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
    commands = await self.bus.take_commands("turn", run)
    for i, cmd in enumerate(commands):
      cp = {
        "uri": uri,
        "version": f"version{i}",
        "boundary": cmd.entry_id,
        "meta": {"optimizer": True},
        "durable": False,
      }
      self.assertFalse(await self.bus.write_reply(cmd, ok(), "engine", checkpoint=cp))
      self.assertTrue(
        await self.bus.write_reply(
          cmd, ok(type=Op.SAVE_WEIGHTS, model_id=run, path=uri), "engine", checkpoint=cp, defer=True
        )
      )
      self.assertIsNone(await self.bus.wait_for_reply(cmd.request_id, 0))
    self.assertEqual((await self.bus.r.xpending(key("q", run), GROUP))["pending"], 0)
    self.assertEqual(await self.bus.r.xlen(key("q", run)), 0)
    self.assertIsNone(await self.registry.version(uri))
    receipts = await self.bus.pending_saves()
    await self.bus.r.pexpire(key("live", "engine"), 0)
    for cmd in reversed(commands):
      receipt = receipts[cmd.request_id]
      corrupt = receipt | {"version": "wrong"}
      self.assertFalse(await self.bus.finish_save(cmd.request_id, corrupt))
      completed = await asyncio.gather(*(self.bus.finish_save(cmd.request_id, receipt) for _ in range(2)))
      self.assertEqual(sum(completed), 1)
      self.assertEqual(
        await self.bus.wait_for_reply(cmd.request_id, 0), ok(type=Op.SAVE_WEIGHTS, model_id=run, path=uri)
      )
      self.assertIsNone(await self.bus.r.hget(key("refs"), f"save:{cmd.request_id}"))
    self.assertEqual(await self.registry.version(uri), "version1")
    state = (await self.bus.get_run(run))["state"]
    self.assertEqual([state[k]["version"] for k in ("working", "durable")], ["version1", "version1"])
    self.assertEqual(await self.bus.pending_saves(), {})

  async def test_sampler_is_constructed_with_its_version_before_the_save_result(self):
    run, uri = await self.new_run("session"), "tinker://source/sampler_weights/export"
    args = {"sampling_session_id": "sampler", "sampler_gpus": 2}
    request = await self.bus.submit_command(run, Op.SAVE_WEIGHTS_FOR_SAMPLER, json.dumps(args).encode())
    [cmd] = await self.bus.take_commands("turn", run)
    cp = {
      "uri": uri,
      "version": "sample@peer:10",
      "boundary": cmd.entry_id,
      "owned": True,
      "meta": {"optimizer": False, "lora": True},
    }
    self.assertIsNone(await self.bus.get_run("sampler"))
    self.assertFalse(await self.bus.write_reply(cmd, ok(), "stale-engine", checkpoint=cp))
    self.assertIsNone(await self.bus.get_run("sampler"))
    reply = ok(sampling_session_id="sampler", path=uri)
    self.assertTrue(await self.bus.write_reply(cmd, reply, "engine", checkpoint=cp))
    self.assertEqual(await self.bus.wait_for_reply(request, 0), reply)
    sampler = await self.bus.get_run("sampler")
    self.assertEqual(
      (sampler["version"], sampler["session"], sampler["gpus"], sampler["train_run"]),
      (cp["version"], "session", 2, run),
    )
    self.assertEqual(await self.bus.r.hget(key("refs"), "job:sampler:weights"), b"sample")
    self.assertIsNone((await self.bus.get_run(run))["state"]["working"])
    with self.assertRaisesRegex(ValueError, "exact weight version"):
      await self.bus.create_run(Run("incomplete", "base", 0, model_path=uri))
    await self.bus.create_run(Run("standalone", "base", 1, model_path=uri, train_run=run, version=cp["version"]))
    queued = await self.bus.submit_command("sampler", Op.SAMPLE, b"{}")
    await self.bus.delete_run(run)
    self.assertFalse((await self.bus.wait_for_reply(queued, 0)).ok)
    self.assertEqual((await self.bus.get_run("sampler"))["state"], {})
    self.assertTrue((await self.bus.get_run("standalone"))["state"])
    self.assertIsNone(await self.bus.r.hget(key("refs"), "job:sampler:weights"))
    self.assertEqual(await self.bus.r.hget(key("refs"), "job:standalone:weights"), b"sample")

  async def test_delete_cancels_storage_obligations_but_never_releases_the_gpu_turn(self):
    run = await self.new_run()
    owned, public = [f"tinker://{run}/{p}" for p in ("sampler_weights/session", "weights/public")]
    await self.seed_checkpoint(owned, owned=True, version="private")
    await self.seed_checkpoint(public, version="public")
    request = await self.bus.submit_command(
      run, Op.LOAD_WEIGHTS, json.dumps({"uri": owned, "version": "private"}).encode(), seq_id=1
    )
    await self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
    load, save = await self.bus.take_commands("turn", run)
    cp = {
      "uri": f"tinker://{run}/weights/pending",
      "version": "pending",
      "boundary": save.entry_id,
      "meta": {"optimizer": True},
    }
    self.assertTrue(await self.bus.write_reply(save, ok(), "engine", checkpoint=cp, defer=True))
    receipt = (await self.bus.pending_saves())[save.request_id]
    await self.bus.delete_run(run)
    for cmd in (load, save):
      self.assertFalse((await self.bus.wait_for_reply(cmd.request_id, 0)).ok)
    self.assertFalse(await self.bus.finish_save(save.request_id, receipt))
    self.assertEqual(await self.bus.r.hgetall(key("refs")), {f"ckpt:{public}".encode(): b"public"})
    self.assertIsNone(await self.registry.get_checkpoint(owned))
    self.assertTrue(await self.bus.r.hexists(key("turns"), "turn"))
    self.assertFalse(await self.bus.r.hexists(key("claims"), run))
    self.assertEqual(await self.bus.submit_command(run, Op.LOAD_WEIGHTS, b"{}", seq_id=1), request)
    self.assertIsNone(await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}", seq_id=2))
    self.assertEqual(await self.bus.pending_saves(), {})

  async def test_sessions_and_live_nodes(self):
    await self.bus.heartbeat_session("s", {"tag": "test"})
    run = await self.new_run("s")
    session = await self.bus.get_session("s")
    self.assertEqual((session["runs"], session["metadata"], session["live"]), ([run], {"tag": "test"}, True))
    await self.bus.finish_session("s")
    self.assertEqual(await self.bus.expired_sessions(), ["s"])
    await self.bus.end_session("s")
    self.assertEqual((await self.bus.head(run)).op, Op.DELETE_MODEL)
    await self.bus.delete_run(run)
    await self.bus.end_session("s")
    self.assertIsNone(await self.bus.r.zscore(key("seen"), "s"))
    await self.bus.r.hset(key("nodes"), "dead", '{"gpus": 8, "until": 0}')
    self.assertEqual({n: v["gpus"] for n, v in (await self.bus.read_table())["nodes"].items()}, {"node": 2})


if __name__ == "__main__":
  unittest.main()
