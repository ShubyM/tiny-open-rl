import asyncio
import json
import tempfile
import time
import unittest

from test.helpers import RedisTest, new_id, start_redis
from tinyrl.bus import GROUP, Bus, Cmd, Op, Reply, Run, connect, key


def ok(**body) -> Reply:
  return Reply(True, "application/json", json.dumps(body).encode())


class TestBus(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    await self.bus.take_lease("p", "w0")  # only a worker's owner can answer for it

  async def new_run(self, worker: str = "p", session: str = "") -> str:
    run_id = new_id()
    await self.bus.create_run(Run(id=run_id, worker=worker, base="base", rank=8, session=session))
    return run_id

  async def test_a_command_goes_through_the_stream_and_its_reply_comes_back(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b'{"lr": 1}')

    [cmd] = await self.bus.take_commands([run_id], "w0", block_ms=10)
    self.assertEqual((cmd.run, cmd.op, cmd.body, cmd.request_id), (run_id, Op.OPTIM_STEP, b'{"lr": 1}', request_id))
    self.assertIsNone(await self.bus.wait_for_reply(request_id, 0))

    await self.bus.write_reply(cmd, ok(type="optim_step"), "p", "w0")
    self.assertEqual(await self.bus.wait_for_reply(request_id, 0), ok(type="optim_step"))
    self.assertEqual(await self.bus.take_commands([run_id], "w0", block_ms=10), [])

    run = await self.bus.get_run(run_id)
    self.assertEqual((run["worker"], run["base"], run["rank"], run["steps"]), ("p", "base", 8, 1))

  async def test_a_reply_wakes_every_waiter_and_reading_it_never_consumes_it(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    cmd = Cmd(run_id, request_id.split("/")[1], Op.OPTIM_STEP, b"")

    waiters = [asyncio.create_task(self.bus.wait_for_reply(request_id, 5)) for _ in range(3)]
    await asyncio.sleep(0.05)
    start = time.perf_counter()
    await self.bus.write_reply(cmd, ok(n=1), "p", "w0")

    self.assertEqual(await asyncio.gather(*waiters), [ok(n=1)] * 3)
    self.assertLess(time.perf_counter() - start, 0.5)
    self.assertEqual(await self.bus.wait_for_reply(request_id, 0), ok(n=1))

  async def test_hundreds_of_requests_wait_for_replies_at_once(self):
    """Each wait holds a Redis connection, as when 400 sampling requests are in flight through one gateway."""
    got = await asyncio.gather(*[self.bus.wait_for_reply(f"req-{i}", 0.5) for i in range(300)])
    self.assertEqual(got, [None] * 300)

  async def test_waiting_for_a_reply_outlasts_the_socket_timeout(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    [cmd] = await self.bus.take_commands([run_id], "w0", block_ms=10)

    async def answer_late():
      await asyncio.sleep(5.5)  # past redis-py's default socket timeout, but well inside retrieve_future's 30s
      await self.bus.write_reply(cmd, ok(n=1), "p", "w0")

    answering = asyncio.create_task(answer_late())
    self.assertEqual(await self.bus.wait_for_reply(request_id, 10), ok(n=1))
    await answering

  async def test_a_retried_call_is_queued_once_even_if_its_seq_key_is_gone(self):
    run_id = await self.new_run()
    first = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"", seq_id=5)
    await self.bus.r.delete(key("seq", run_id, "5"))  # as if the gateway died between queueing it and recording it
    self.assertEqual(await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"", seq_id=5), first)
    self.assertEqual(await self.bus.r.get(key("seq", run_id, "5")), first.encode())
    self.assertNotEqual(await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"", seq_id=6), first)
    self.assertEqual(await self.bus.r.xlen(key("q", run_id)), 2)

  async def test_a_retry_after_its_run_was_deleted_gets_the_first_request_id(self):
    run_id = await self.new_run()
    first = await self.bus.submit_command(run_id, Op.DELETE_MODEL, b"", seq_id=3)
    await self.bus.delete_run(run_id, "p")
    self.assertEqual(await self.bus.submit_command(run_id, Op.DELETE_MODEL, b"", seq_id=3), first)
    self.assertIsNone(await self.bus.submit_command(run_id, Op.DELETE_MODEL, b"", seq_id=4))

  async def test_unknown_and_deleted_runs_take_no_work(self):
    self.assertIsNone(await self.bus.submit_command(new_id(), Op.OPTIM_STEP, b""))

    run_id = await self.new_run()
    await self.bus.delete_run(run_id, "p")
    self.assertIsNone(await self.bus.submit_command(run_id, Op.OPTIM_STEP, b""))
    self.assertFalse(await self.bus.r.exists(key("worker", "p")))
    self.assertEqual(await self.bus.queued_and_in_flight(run_id), {"live": False})
    self.assertFalse(await self.bus.r.exists(key("q", run_id)))

  async def test_delete_removes_owned_checkpoints_and_keeps_saved_weights(self):
    run = await self.new_run()
    owned, saved = [f"tinker://{run}/{path}" for path in ("sampler_weights/session", "weights/save")]
    for uri in (owned, saved):
      await self.registry.register_checkpoint(uri, 1, ttl=60, owned=uri == owned)
    await self.bus.set_parked_checkpoint(run, owned, "p", "w0")
    request = await self.bus.submit_command(run, Op.DELETE_MODEL, b"{}")
    [cmd] = await self.bus.take_commands([run], "w0", count=1)
    publishing = asyncio.create_task(self.bus.write_reply(cmd, ok(), "p", "w0"))
    try:
      self.assertEqual(await self.bus.wait_for_reply(request, 5), ok())
      self.assertFalse(await self.bus.r.exists(key("q", run), key("parked", run)))
      self.assertFalse(await self.bus.r.sismember(key("runs"), run))
      self.assertIsNone(await self.registry.get_checkpoint(owned))
      self.assertIsNotNone(await self.registry.get_checkpoint(saved))
      for index in ("ckpts", "expiry"):
        self.assertIsNone(await self.bus.r.zscore(key(index), owned))
        self.assertIsNotNone(await self.bus.r.zscore(key(index), saved))
    finally:
      self.assertTrue(await publishing)

  async def test_the_queue_is_read_from_the_stream_itself(self):
    run_id = await self.new_run()
    request_ids = [await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"") for _ in range(3)]
    self.assertEqual(await self.bus.queued_and_in_flight(run_id), {"live": True, "queued": 3, "in_flight": []})

    [cmd] = await self.bus.take_commands([run_id], "w0", count=1, block_ms=10)
    queue = await self.bus.queued_and_in_flight(run_id)
    self.assertEqual(queue["queued"], 2)
    [in_flight] = queue["in_flight"]
    self.assertEqual(
      (in_flight["request_id"], in_flight["trainer"], in_flight["deliveries"]), (request_ids[0], "w0", 1)
    )

    await self.bus.write_reply(cmd, ok(), "p", "w0")
    self.assertEqual(await self.bus.queued_and_in_flight(run_id), {"live": True, "queued": 2, "in_flight": []})

  async def test_peeking_returns_each_unclaimed_head_without_taking_it(self):
    a, b, gone = await self.new_run(), await self.new_run(), await self.new_run()
    first = await self.bus.submit_command(a, Op.FORWARD_BACKWARD, b"first body")
    second = await self.bus.submit_command(a, Op.OPTIM_STEP, b"second body")
    other = await self.bus.submit_command(b, Op.SAVE_WEIGHTS_FOR_SAMPLER, b"other body")
    await self.bus.delete_run(gone, "p")

    heads = await self.bus.peek_commands([gone, b, a])
    by_run = {cmd.run: cmd for cmd in heads}
    self.assertEqual(set(by_run), {a, b})
    self.assertEqual((by_run[a].request_id, by_run[a].op, by_run[a].body), (first, Op.FORWARD_BACKWARD, b"first body"))
    self.assertEqual(by_run[b].request_id, other)
    self.assertEqual(await self.bus.peek_commands([gone, b, a]), heads)
    self.assertEqual(await self.bus.queued_and_in_flight(a), {"live": True, "queued": 2, "in_flight": []})
    self.assertEqual(await self.bus.queued_and_in_flight(b), {"live": True, "queued": 1, "in_flight": []})

    [claimed] = await self.bus.take_commands([gone, a], "w0", count=1)
    self.assertEqual(claimed, by_run[a])
    self.assertEqual({cmd.run: cmd.request_id for cmd in await self.bus.peek_commands([a, b])}, {a: second, b: other})
    pending = (await self.bus.queued_and_in_flight(a))["in_flight"]
    self.assertEqual([cmd["request_id"] for cmd in pending], [first])

  async def test_a_new_owner_fails_what_a_dead_one_took(self):
    run_id = await self.new_run()
    for _ in range(2):
      await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")

    taken = await self.bus.take_commands([run_id], "w0", block_ms=10)
    await self.bus.release_lease("p", "w0")
    self.assertTrue(await self.bus.take_lease("p", "new"))
    await self.bus.fail_stranded_commands([run_id], "p", "new")
    for cmd in taken:
      reply = await self.bus.wait_for_reply(cmd.request_id, 0)
      self.assertFalse(reply.ok)
      self.assertEqual(reply.content_type, "application/json")
      self.assertEqual(json.loads(reply.body)["category"], "server")
      self.assertIn(cmd.op, json.loads(reply.body)["error"])
    self.assertEqual(await self.bus.queued_and_in_flight(run_id), {"live": True, "queued": 0, "in_flight": []})
    self.assertEqual(await self.bus.take_commands([run_id], "new", block_ms=10), [])
    await self.bus.fail_stranded_commands([run_id], "p", "new")
    self.assertEqual((await self.bus.get_run(run_id))["failed"], 2)

  async def test_failing_stranded_commands_leaves_fresh_claims_alone(self):
    run_id = await self.new_run()
    for _ in range(2):
      await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    old, fresh = await self.bus.take_commands([run_id], "dead", block_ms=10)
    await self.bus.r.xclaim(key("q", run_id), GROUP, "dead", 0, [old.entry_id], idle=60_000)

    await self.bus.fail_stranded_commands([run_id], "p", "w0", min_idle_ms=30_000)
    self.assertFalse((await self.bus.wait_for_reply(old.request_id, 0)).ok)
    self.assertIsNone(await self.bus.wait_for_reply(fresh.request_id, 0))
    [pending] = (await self.bus.queued_and_in_flight(run_id))["in_flight"]
    self.assertEqual((pending["request_id"], pending["trainer"], pending["deliveries"]), (fresh.request_id, "dead", 1))
    self.assertEqual((await self.bus.get_run(run_id))["failed"], 1)

  async def test_a_stale_owner_cannot_fail_stranded_commands(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    await self.bus.take_commands([run_id], "w0", block_ms=10)
    await self.bus.release_lease("p", "w0")
    self.assertTrue(await self.bus.take_lease("p", "new"))

    await self.bus.fail_stranded_commands([run_id], "p", "w0")
    self.assertIsNone(await self.bus.wait_for_reply(request_id, 0))
    self.assertEqual((await self.bus.get_run(run_id))["failed"], 0)
    self.assertEqual(len((await self.bus.queued_and_in_flight(run_id))["in_flight"]), 1)

    await self.bus.fail_stranded_commands([run_id], "p", "new")
    self.assertFalse((await self.bus.wait_for_reply(request_id, 0)).ok)
    self.assertEqual((await self.bus.get_run(run_id))["failed"], 1)

  async def test_an_owner_can_recover_a_lease_that_expired_untaken(self):
    await self.bus.r.pexpire(key("owner", "p"), 0)
    self.assertFalse(await self.bus.renew_lease("p", "w0"))
    self.assertTrue(await self.bus.take_lease("p", "w0", recover=True))
    self.assertEqual(await self.bus.lease_owner("p"), "w0")
    self.assertEqual(await self.bus.r.pttl(key("last_owner", "p")), -1)

  async def test_concurrent_lease_takes_have_one_winner_and_matching_history(self):
    await self.bus.release_lease("p", "w0")
    owners = ["first", "second"]
    taken = await asyncio.gather(*(self.bus.take_lease("p", owner) for owner in owners))
    self.assertEqual(sum(taken), 1)
    winner = owners[taken.index(True)]
    self.assertEqual(await self.bus.lease_owner("p"), winner)
    self.assertEqual(await self.bus.r.get(key("last_owner", "p")), winner.encode())

  async def test_recovery_rejects_an_owner_after_another_owner_has_come_and_gone(self):
    await self.bus.r.pexpire(key("owner", "p"), 0)
    self.assertTrue(await self.bus.take_lease("p", "next"))
    await self.bus.release_lease("p", "next")
    self.assertFalse(await self.bus.take_lease("p", "w0", recover=True))

  async def test_only_the_workers_owner_can_answer_for_it(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    [cmd] = await self.bus.take_commands([run_id], "w0", block_ms=10)

    self.assertFalse(await self.bus.take_lease("p", "zombie"))  # w0 holds the lease
    self.assertFalse(await self.bus.renew_lease("p", "zombie"))
    self.assertFalse(await self.bus.write_reply(cmd, ok(by="zombie"), "p", "zombie"))
    self.assertIsNone(await self.bus.wait_for_reply(request_id, 0))

    self.assertTrue(await self.bus.write_reply(cmd, ok(by="w0"), "p", "w0"))
    self.assertEqual(await self.bus.wait_for_reply(request_id, 0), ok(by="w0"))

    await self.bus.release_lease("p", "zombie")  # not its lease, so this does nothing
    self.assertTrue(await self.bus.renew_lease("p", "w0"))
    await self.bus.release_lease("p", "w0")
    self.assertTrue(await self.bus.take_lease("p", "next"))

  async def test_a_dead_node_daemon_stops_counting(self):
    await self.bus.node_heartbeat("node-a", 2)
    await self.bus.r.hset("tinyrl:nodes", "node-b", '{"gpus": 8, "until": 0}')
    self.assertEqual(await self.bus.live_nodes(), {"node-a": 2})

  async def test_sessions(self):
    await self.bus.heartbeat_session("s")
    run_id = await self.new_run(session="s")

    session = await self.bus.get_session("s")
    self.assertTrue(session["live"])
    self.assertEqual(session["runs"], [run_id])
    self.assertIsNone(await self.bus.get_session("nope"))

  async def test_a_session_is_ended_again_until_its_training_runs_are_gone(self):
    await self.bus.heartbeat_session("s")
    run_id = await self.new_run(session="s")
    await self.bus.end_session("s")
    self.assertEqual(len(await self.bus.r.xrange(key("q", run_id))), 1)  # its delete, queued
    self.assertIsNotNone(await self.bus.r.zscore(key("seen"), "s"))  # and the session looked at again later
    await self.bus.r.xtrim(key("q", run_id), maxlen=0)  # a trainer took the delete and died
    await self.bus.end_session("s")
    self.assertEqual(len(await self.bus.r.xrange(key("q", run_id))), 1)
    await self.bus.delete_run(run_id, "p")
    await self.bus.end_session("s")
    self.assertIsNone(await self.bus.r.zscore(key("seen"), "s"))

  async def test_a_deleted_run_does_not_hide_another_runs_command(self):
    run_id, gone = await self.new_run(), await self.new_run()
    request = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    await self.bus.delete_run(gone, "p")
    [cmd] = await self.bus.take_commands([gone, run_id], "o", block_ms=10)
    self.assertEqual((cmd.run, cmd.request_id), (run_id, request))
    self.assertEqual(await self.bus.take_commands([run_id], "o", block_ms=10), [])


class TestRedisRestart(unittest.IsolatedAsyncioTestCase):
  async def test_a_call_waits_out_a_redis_restart(self):
    sock = f"{tempfile.mkdtemp()}/redis.sock"
    proc = start_redis(sock)
    bus = Bus(connect(f"unix://{sock}"))
    try:
      await bus.node_heartbeat("node-a", 1)
      proc.terminate()
      proc.wait()

      async def restart():
        await asyncio.sleep(2)
        return start_redis(sock)

      restarting = asyncio.create_task(restart())
      await bus.node_heartbeat("node-a", 1)  # retried until Redis is back, instead of raising
      proc = await restarting
      self.assertEqual(await bus.live_nodes(), {"node-a": 1})
    finally:
      await bus.r.aclose()
      proc.terminate()


if __name__ == "__main__":
  unittest.main()
