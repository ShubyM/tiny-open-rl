import asyncio
import json
import tempfile
import time
import unittest

from test.helpers import RedisTest, new_id, start_redis
from tinyrl.bus import GROUP, Bus, Cmd, Op, Reply, Role, Run, connect, key, unpack_snap
from tinyrl.control.accounting import Charge


def ok(**body) -> Reply:
  return Reply(True, "application/json", json.dumps(body).encode())


class TestBus(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    await self.bus.touch_leases("w0")

  async def new_run(self, session: str = "", role: Role = Role.TRAINER, account: str = "", claim: bool = True) -> str:
    run_id = new_id()
    await self.bus.create_run(Run(id=run_id, base="base", rank=8, session=session, role=role, account=account))
    if claim and role == Role.TRAINER:
      self.tokens = getattr(self, "tokens", {}) | {run_id: (await self.bus.acquire_run(run_id, "w0"))[0]}
    return run_id

  async def test_a_command_goes_through_the_stream_and_its_reply_comes_back(self):
    run_id = await self.new_run(account="alice")
    request_id = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b'{"lr": 1}', cost=42)

    [cmd] = await self.bus.take_commands([run_id], "w0")
    self.assertEqual(
      (cmd.run, cmd.op, cmd.body, cmd.request_id, cmd.cost), (run_id, Op.OPTIM_STEP, b'{"lr": 1}', request_id, 42)
    )
    self.assertIsNone(await self.bus.wait_for_reply(request_id, 0))

    charge = Charge(started_ms=int(cmd.entry_id.split("-")[0]) + 12, busy_ms=50, gpu_ms=100.5)
    committed, old = await self.bus.commit_reply(cmd, ok(type="optim_step"), self.tokens[run_id], "alice", True, charge)
    self.assertEqual((committed, old), (True, None))
    self.assertEqual(await self.bus.wait_for_reply(request_id, 0), ok(type="optim_step"))
    self.assertEqual(await self.bus.take_commands([run_id], "w0"), [])

    run = await self.bus.get_run(run_id)
    self.assertEqual(
      (run["base"], run["rank"], run["steps"], run["tokens"], run["pos"]), ("base", 8, 1, 42, cmd.entry_id)
    )
    self.assertEqual((run["wait_ms"], run["busy_ms"]), (12, 50))
    self.assertEqual((await self.bus.read_state())["service"]["alice"], 100.5)

  async def test_a_reply_wakes_every_waiter_and_reading_it_never_consumes_it(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    cmd = Cmd(run_id, request_id.split("/")[1], Op.OPTIM_STEP, b"")

    waiters = [asyncio.create_task(self.bus.wait_for_reply(request_id, 5)) for _ in range(3)]
    await asyncio.sleep(0.05)
    start = time.perf_counter()
    await self.bus.commit_reply(cmd, ok(n=1), self.tokens[run_id])

    self.assertEqual(await asyncio.gather(*waiters), [ok(n=1)] * 3)
    self.assertLess(time.perf_counter() - start, 0.5)
    self.assertEqual(await self.bus.wait_for_reply(request_id, 0), ok(n=1))

  async def test_hundreds_of_requests_wait_for_replies_at_once(self):
    got = await asyncio.gather(*[self.bus.wait_for_reply(f"req-{i}", 0.5) for i in range(300)])
    self.assertEqual(got, [None] * 300)

  async def test_waiting_for_a_reply_outlasts_the_socket_timeout(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    [cmd] = await self.bus.take_commands([run_id], "w0")

    async def answer_late():
      await asyncio.sleep(5.5)
      await self.bus.commit_reply(cmd, ok(n=1), self.tokens[run_id])

    answering = asyncio.create_task(answer_late())
    self.assertEqual(await self.bus.wait_for_reply(request_id, 10), ok(n=1))
    await answering

  async def test_a_retried_call_is_queued_once_even_if_its_seq_key_is_gone(self):
    run_id = await self.new_run()
    first = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"", seq_id=5)
    await self.bus.r.delete(key("seq", run_id, "5"))
    self.assertEqual(await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"", seq_id=5), first)
    self.assertEqual(await self.bus.r.get(key("seq", run_id, "5")), first.encode())
    self.assertNotEqual(await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"", seq_id=6), first)
    self.assertEqual(await self.bus.r.xlen(key("q", run_id)), 2)

  async def test_a_retry_after_its_run_was_deleted_gets_the_first_request_id(self):
    run_id = await self.new_run()
    first = await self.bus.submit_command(run_id, Op.DELETE_MODEL, b"", seq_id=3)
    await self.bus.delete_run(run_id)
    self.assertEqual(await self.bus.submit_command(run_id, Op.DELETE_MODEL, b"", seq_id=3), first)
    self.assertIsNone(await self.bus.submit_command(run_id, Op.DELETE_MODEL, b"", seq_id=4))

  async def test_unknown_and_deleted_runs_take_no_work(self):
    self.assertIsNone(await self.bus.submit_command(new_id(), Op.OPTIM_STEP, b""))

    run_id = await self.new_run()
    await self.bus.delete_run(run_id)
    self.assertIsNone(await self.bus.submit_command(run_id, Op.OPTIM_STEP, b""))
    self.assertEqual(await self.bus.queued_and_in_flight(run_id), {"live": False})
    self.assertFalse(await self.bus.r.exists(key("q", run_id)))

  async def test_delete_removes_owned_checkpoints_and_keeps_saved_weights(self):
    run = await self.new_run()
    owned, saved = [f"tinker://{run}/{path}" for path in ("sampler_weights/session", "weights/save")]
    for uri in (owned, saved):
      await self.seed_checkpoint(uri, 1, ttl=60, owned=uri == owned)
    self.assertTrue(await self.bus.detach_run(run, self.tokens[run], None, f"{owned}\nv1"))
    token, _, snap = await self.bus.acquire_run(run, "w0")
    self.assertEqual(unpack_snap(snap), (owned, "v1"))
    request = await self.bus.submit_command(run, Op.DELETE_MODEL, b"{}")
    [cmd] = await self.bus.take_commands([run], "w0", count=1)
    committed, _ = await self.bus.commit_reply(cmd, ok(), token)
    self.assertTrue(committed)
    self.assertEqual(await self.bus.wait_for_reply(request, 5), ok())
    self.assertFalse(await self.bus.r.exists(key("q", run)))
    self.assertIsNone((await self.bus.get_run(run)).get("snap"))
    self.assertIsNone((await self.bus.get_run(run)).get("owner"))
    self.assertFalse(await self.bus.r.sismember(key("runs"), run))
    self.assertIsNone(await self.bus.get_checkpoint(owned))
    self.assertIsNotNone(await self.bus.get_checkpoint(saved))
    for index in ("ckpts", "expiry"):
      self.assertIsNone(await self.bus.r.zscore(key(index), owned))
      self.assertIsNotNone(await self.bus.r.zscore(key(index), saved))

  async def test_the_queue_is_read_from_the_stream_itself(self):
    run_id = await self.new_run()
    request_ids = [await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"") for _ in range(3)]
    self.assertEqual(await self.bus.queued_and_in_flight(run_id), {"live": True, "queued": 3, "in_flight": []})

    [cmd] = await self.bus.take_commands([run_id], "w0", count=1)
    queue = await self.bus.queued_and_in_flight(run_id)
    self.assertEqual(queue["queued"], 2)
    [in_flight] = queue["in_flight"]
    self.assertEqual(
      (in_flight["request_id"], in_flight["trainer"], in_flight["deliveries"]), (request_ids[0], "w0", 1)
    )

    await self.bus.commit_reply(cmd, ok(), self.tokens[run_id])
    self.assertEqual(await self.bus.queued_and_in_flight(run_id), {"live": True, "queued": 2, "in_flight": []})

  async def test_peeking_returns_each_unclaimed_head_without_taking_it(self):
    a, b, gone = await self.new_run(), await self.new_run(), await self.new_run()
    first = await self.bus.submit_command(a, Op.FORWARD_BACKWARD, b"first body")
    second = await self.bus.submit_command(a, Op.OPTIM_STEP, b"second body")
    other = await self.bus.submit_command(b, Op.SAVE_WEIGHTS_FOR_SAMPLER, b"other body")
    await self.bus.delete_run(gone)

    heads = await self.bus.peek_trainer([gone, b, a])
    by_run = {cmd.run: cmd for cmd in heads}
    self.assertEqual(set(by_run), {a, b})
    self.assertEqual((by_run[a].request_id, by_run[a].op, by_run[a].body), (first, Op.FORWARD_BACKWARD, b""))
    self.assertEqual(by_run[b].request_id, other)
    self.assertEqual(await self.bus.peek_trainer([gone, b, a]), heads)
    self.assertEqual(await self.bus.queued_and_in_flight(a), {"live": True, "queued": 2, "in_flight": []})
    self.assertEqual(await self.bus.queued_and_in_flight(b), {"live": True, "queued": 1, "in_flight": []})

    [claimed] = await self.bus.take_commands([gone, a], "w0", count=1)
    self.assertEqual((claimed.request_id, claimed.op, claimed.body), (first, Op.FORWARD_BACKWARD, b"first body"))
    self.assertEqual({cmd.run: cmd.request_id for cmd in await self.bus.peek_sampler([a, b])}, {a: second, b: other})
    self.assertEqual([c["request_id"] for c in (await self.bus.queued_and_in_flight(a))["in_flight"]], [first])

  async def test_a_new_owner_fails_what_a_dead_one_took_and_stale_tokens_are_fenced(self):
    run_id = await self.new_run()
    old_token = self.tokens[run_id]
    for _ in range(2):
      await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    taken = await self.bus.take_commands([run_id], "w0")

    # Acquire refuses while the current owner's lease is live, and succeeds once it expires.
    await self.bus.touch_leases("new")
    self.assertIsNone(await self.bus.acquire_run(run_id, "new"))
    await self.bus.drop_lease("w0")
    new_token, _, _ = await self.bus.acquire_run(run_id, "new")
    self.assertNotEqual(new_token, old_token)

    # Stale token cannot commit, detach, re-acquire, or fail stranded commands — even if w0 re-opens a lease (ABA).
    await self.bus.touch_leases("w0")
    self.assertIsNone(await self.bus.acquire_run(run_id, "w0"))
    self.assertEqual(await self.bus.commit_reply(taken[0], ok(by="zombie"), old_token), (False, None))
    self.assertFalse(await self.bus.detach_run(run_id, old_token, None, "tinker://x/parked/1\nv1"))
    await self.bus.fail_stranded_commands(run_id, old_token)
    self.assertIsNone(await self.bus.wait_for_reply(taken[0].request_id, 0))
    self.assertEqual((await self.bus.get_run(run_id))["failed"], 0)

    await self.bus.fail_stranded_commands(run_id, new_token)
    for cmd in taken:
      reply = await self.bus.wait_for_reply(cmd.request_id, 0)
      self.assertFalse(reply.ok)
      self.assertEqual(json.loads(reply.body)["category"], "server")
      self.assertIn(cmd.op, json.loads(reply.body)["error"])
    self.assertEqual(await self.bus.queued_and_in_flight(run_id), {"live": True, "queued": 0, "in_flight": []})
    self.assertEqual((await self.bus.get_run(run_id))["failed"], 2)

  async def test_failing_stranded_commands_leaves_fresh_claims_alone(self):
    run_id = await self.new_run()
    for _ in range(2):
      await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    old, fresh = await self.bus.take_commands([run_id], "dead")
    await self.bus.r.xclaim(key("q", run_id), GROUP, "dead", 0, [old.entry_id], idle=60_000)

    await self.bus.fail_stranded_commands(run_id, self.tokens[run_id], idle_ms=30_000)
    self.assertFalse((await self.bus.wait_for_reply(old.request_id, 0)).ok)
    self.assertIsNone(await self.bus.wait_for_reply(fresh.request_id, 0))
    [pending] = (await self.bus.queued_and_in_flight(run_id))["in_flight"]
    self.assertEqual((pending["request_id"], pending["trainer"], pending["deliveries"]), (fresh.request_id, "dead", 1))
    self.assertEqual((await self.bus.get_run(run_id))["failed"], 1)

  async def test_old_epoch_cannot_claim_work_or_publish_a_named_or_parked_checkpoint(self):
    run = await self.new_run()
    old = self.tokens[run]
    await self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
    abandoned = await self.bus.take_trainer_command(run, old)
    await self.bus.drop_lease("w0")
    await self.bus.touch_leases("replacement")
    current, _, _ = await self.bus.acquire_run(run, "replacement")
    await self.bus.fail_stranded_commands(run, current)
    request = await self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
    with self.assertRaisesRegex(RuntimeError, "lost ownership"):
      await self.bus.take_trainer_command(run, old)
    cmd = await self.bus.take_trainer_command(run, current)
    self.assertEqual(cmd.request_id, request)
    uri = f"tinker://{run}/weights/step"
    saved = {"uri": uri, "size": 20, "version": "new"}
    self.assertTrue((await self.bus.commit_reply(cmd, ok(), current, checkpoint=saved))[0])

    stale = {"uri": uri, "size": 10, "ttl": 60, "version": "old"}
    self.assertFalse((await self.bus.commit_reply(abandoned, ok(), old, checkpoint=stale))[0])
    await self.bus.r.pexpire(key("f", request), 0)
    self.assertFalse((await self.bus.commit_reply(cmd, ok(), current, checkpoint=stale))[0])
    self.assertEqual(await self.bus.checkpoint_version(uri), "new")
    self.assertIsNone(await self.bus.r.zscore(key("expiry"), uri))
    parked = {"uri": f"tinker://{run}/parked/late", "size": 12, "owned": True}
    self.assertFalse(await self.bus.detach_run(run, old, None, parked["uri"], parked))
    self.assertFalse(await self.bus.detach_run(run, current, "0-1", parked["uri"], parked))
    self.assertIsNone(await self.bus.get_checkpoint(parked["uri"]))

  async def test_concurrent_acquires_pick_one_winner_and_detach_fences_exact_pos(self):
    run_id = await self.new_run(claim=False)
    await self.bus.touch_leases("a", "b", "c")
    got = await asyncio.gather(*(self.bus.acquire_run(run_id, life) for life in ("a", "b", "c")))
    winners = [g for g in got if g is not None]
    self.assertEqual(len(winners), 1)
    token = winners[0][0]

    # Advance pos and record a clean detach; then verify mutating commit supersedes snap while read-only keeps it.
    req1 = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    [c1] = await self.bus.take_commands([run_id], "a", 1)
    self.assertEqual(await self.bus.commit_reply(c1, ok(), token, mutates=True), (True, None))
    self.assertFalse(await self.bus.detach_run(run_id, token, "0-1", "tinker://r/parked/bad\nv0"))  # wrong pos
    self.assertTrue(await self.bus.detach_run(run_id, token, c1.entry_id, "tinker://r/parked/1\nv1"))

    token2, pos2, snap2 = await self.bus.acquire_run(run_id, "b")
    self.assertNotEqual(token2, token)
    self.assertEqual((pos2, snap2), (c1.entry_id, "tinker://r/parked/1\nv1"))
    self.assertIsNotNone(req1)

    # Read-only commit preserves snap; next mutating commit returns superseded snap and clears it.
    await self.bus.submit_command(run_id, Op.SAVE_WEIGHTS, b"{}")
    [ro] = await self.bus.take_commands([run_id], "b", 1)
    self.assertEqual(await self.bus.commit_reply(ro, ok(), token2, mutates=False), (True, None))
    self.assertEqual((await self.bus.get_run(run_id))["snap"], snap2)

    await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    [mut] = await self.bus.take_commands([run_id], "b", 1)
    self.assertEqual(await self.bus.commit_reply(mut, ok(), token2, mutates=True), (True, snap2))
    self.assertIsNone((await self.bus.get_run(run_id)).get("snap"))

  async def test_unreaped_lease_recovery_preserves_epoch_token(self):
    run_id = await self.new_run()
    token = self.tokens[run_id]
    await self.bus.r.pexpire(key("lease", "w0"), 0)
    self.assertFalse(await self.bus.r.exists(key("lease", "w0")))
    await self.bus.touch_leases("w0")  # recovered before anyone reaped
    self.assertTrue(await self.bus.r.exists(key("lease", "w0")))
    await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    [cmd] = await self.bus.take_commands([run_id], "w0", 1)
    self.assertEqual(await self.bus.commit_reply(cmd, ok(alive=1), token, mutates=True), (True, None))

  async def test_lease_renewal_does_not_resurrect_dropped_lease(self):
    await self.bus.touch_leases("live", "dropped")
    await self.bus.drop_lease("dropped")
    await self.bus.node_heartbeat("node-a", 1, leases=("live", "dropped"))
    await self.bus.touch_leases("live", "dropped", xx=True)
    self.assertTrue(await self.bus.r.exists(key("lease", "live")))
    self.assertFalse(await self.bus.r.exists(key("lease", "dropped")))

  async def test_competing_sampler_replicas_answer_and_charge_exactly_once(self):
    sid = await self.new_run(role=Role.SAMPLER, account="bob")
    req = await self.bus.submit_command(sid, Op.SAMPLE, b"{}", cost=100)
    [cmd] = await self.bus.take_commands([sid], "s1", 1)
    results = await asyncio.gather(
      *(self.bus.commit_sample(cmd, ok(replica=i), "bob", Charge(busy_ms=40, gpu_ms=40.5 + i)) for i in range(5))
    )
    self.assertEqual(sum(results), 1)
    winner = results.index(True)
    self.assertEqual(await self.bus.wait_for_reply(req, 0), ok(replica=winner))
    run = await self.bus.get_run(sid)
    self.assertEqual((run["cmds"], run["tokens"]), (1, 100))
    service = (await self.bus.read_state())["service"]["bob"]
    self.assertEqual(service, 40.5 + winner)

    await self.bus.r.pexpire(key("f", req), 0)
    self.assertFalse(await self.bus.commit_sample(cmd, ok(late=True), "bob", Charge(busy_ms=100, gpu_ms=100)))
    self.assertIsNone(await self.bus.wait_for_reply(req, 0))
    self.assertEqual(await self.bus.get_run(sid), run)
    self.assertEqual((await self.bus.read_state())["service"]["bob"], service)

  async def test_a_sample_in_flight_is_reclaimed_only_from_a_dead_replica(self):
    session = await self.new_run(role=Role.SAMPLER)
    requests = [await self.bus.submit_command(session, Op.SAMPLE, b"{}") for _ in range(2)]
    taken = await self.bus.take_commands([session], "w0")
    self.assertEqual([cmd.request_id for cmd in taken], requests)
    # However long a sample runs, another replica leaves it alone while its replica holds a lease.
    await self.bus.r.xclaim(key("q", session), GROUP, "w0", 0, [cmd.entry_id for cmd in taken], idle=3_600_000)
    self.assertEqual(await self.bus.take_commands([session], "other"), [])
    self.assertEqual(await self.bus.peek_sampler([session]), [])

    await self.bus.drop_lease("w0")  # its process died
    [head] = await self.bus.peek_sampler([session])
    self.assertEqual(head.request_id, requests[0])
    reclaimed = await self.bus.take_commands([session], "other")
    self.assertEqual([cmd.request_id for cmd in reclaimed], requests)

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
    self.assertEqual(len(await self.bus.r.xrange(key("q", run_id))), 1)
    self.assertIsNotNone(await self.bus.r.zscore(key("seen"), "s"))
    await self.bus.r.xtrim(key("q", run_id), maxlen=0)
    await self.bus.end_session("s")
    self.assertEqual(len(await self.bus.r.xrange(key("q", run_id))), 1)
    await self.bus.delete_run(run_id)
    await self.bus.end_session("s")
    self.assertIsNone(await self.bus.r.zscore(key("seen"), "s"))

  async def test_a_deleted_run_does_not_hide_another_runs_command(self):
    run_id, gone = await self.new_run(), await self.new_run()
    request = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    await self.bus.delete_run(gone)
    [cmd] = await self.bus.take_commands([gone, run_id], "o")
    self.assertEqual((cmd.run, cmd.request_id), (run_id, request))
    self.assertEqual(await self.bus.take_commands([run_id], "o"), [])


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
      await bus.node_heartbeat("node-a", 1)
      proc = await restarting
      self.assertEqual(await bus.live_nodes(), {"node-a": 1})
    finally:
      await bus.r.aclose()
      proc.terminate()


if __name__ == "__main__":
  unittest.main()
