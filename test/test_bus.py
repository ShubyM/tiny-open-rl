import asyncio
import json
import tempfile
import time
import unittest
from unittest import mock

from test.helpers import BusTest, new_id
from tinyrl.bus import (
  BackpressureError,
  Bus,
  Cmd,
  Op,
  PromiseGoneError,
  RemoteBus,
  Reply,
  Role,
  Run,
  serve_bus,
  unpack_snap,
)
from tinyrl.control.accounting import Charge


def ok(**body) -> Reply:
  return Reply(True, "application/json", json.dumps(body).encode())


class TestBus(BusTest):
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

    cmd = await self.bus.claim_trainer(run_id, self.tokens[run_id])
    self.assertEqual(
      (cmd.run, cmd.op, cmd.body, cmd.request_id, cmd.cost), (run_id, Op.OPTIM_STEP, b'{"lr": 1}', request_id, 42)
    )
    self.assertIsNone(await self.bus.wait_for_reply(request_id, 0))

    charge = Charge(started_ms=int(cmd.entry_id.split("-")[0]) + 12, busy_ms=50, gpu_ms=100.5)
    self.assertTrue(await self.bus.commit_reply(cmd, ok(type="optim_step"), self.tokens[run_id], "alice", True, charge))
    self.assertEqual(await self.bus.wait_for_reply(request_id, 0), ok(type="optim_step"))
    self.assertIsNone(await self.bus.claim_trainer(run_id, self.tokens[run_id]))

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
    cmd = await self.bus.claim_trainer(run_id, self.tokens[run_id])

    async def answer_late():
      await asyncio.sleep(1.0)
      await self.bus.commit_reply(cmd, ok(n=1), self.tokens[run_id])

    answering = asyncio.create_task(answer_late())
    self.assertEqual(await self.bus.wait_for_reply(request_id, 10), ok(n=1))
    await answering

  async def test_a_retried_call_is_queued_once(self):
    run_id = await self.new_run()
    first = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"", seq_id=5)
    self.assertEqual(await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"", seq_id=5), first)
    self.assertNotEqual(await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"", seq_id=6), first)
    self.assertEqual((await self.bus.queued_and_in_flight(run_id))["queued"], 2)

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
    self.assertNotIn(run_id, self.bus.queues)

  async def test_delete_removes_owned_checkpoints_and_keeps_saved_weights(self):
    run = await self.new_run()
    owned, saved = [f"tinker://{run}/{path}" for path in ("sampler_weights/session", "weights/save")]
    for uri in (owned, saved):
      await self.seed_checkpoint(uri, 1, ttl=60, owned=uri == owned)
    self.assertTrue(await self.bus.detach_run(run, self.tokens[run], None, f"{owned}\nv1"))
    token, _, snap = await self.bus.acquire_run(run, "w0")
    self.assertEqual(unpack_snap(snap), (owned, "v1"))
    request = await self.bus.submit_command(run, Op.DELETE_MODEL, b"{}")
    cmd = await self.bus.claim_trainer(run, token)
    self.assertTrue(await self.bus.commit_reply(cmd, ok(), token))
    self.assertEqual(await self.bus.wait_for_reply(request, 5), ok())
    self.assertNotIn(run, self.bus.queues)
    self.assertIsNone((await self.bus.get_run(run)).get("snap"))
    self.assertIsNone((await self.bus.get_run(run)).get("owner"))
    self.assertNotIn(run, (await self.bus.read_state())["runs"])
    self.assertIsNone(await self.bus.get_checkpoint(owned))
    self.assertIsNotNone(await self.bus.get_checkpoint(saved))
    self.assertEqual([c["tinker_path"] for c in (await self.bus.page_of_checkpoints(10, 0))[0]], [saved])

  async def test_the_queue_is_read_from_the_stream_itself(self):
    run_id = await self.new_run()
    request_ids = [await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"") for _ in range(3)]
    self.assertEqual(await self.bus.queued_and_in_flight(run_id), {"live": True, "queued": 3, "in_flight": []})

    cmd = await self.bus.claim_trainer(run_id, self.tokens[run_id])
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

    claimed = await self.bus.claim_trainer(a, self.tokens[a])
    self.assertEqual((claimed.request_id, claimed.op, claimed.body), (first, Op.FORWARD_BACKWARD, b"first body"))
    self.assertEqual([c["request_id"] for c in (await self.bus.queued_and_in_flight(a))["in_flight"]], [first])
    await self.bus.commit_reply(claimed, ok(), self.tokens[a])
    self.assertEqual({cmd.run: cmd.request_id for cmd in await self.bus.peek_trainer([a, b])}, {a: second, b: other})

  async def test_a_new_owner_fails_what_a_dead_one_took_and_stale_tokens_are_fenced(self):
    run_id = await self.new_run()
    old_token = self.tokens[run_id]
    first, second = [await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"") for _ in range(2)]
    claimed = await self.bus.claim_trainer(run_id, old_token)
    self.assertEqual(claimed.request_id, first)

    await self.bus.touch_leases("new")
    self.assertIsNone(await self.bus.acquire_run(run_id, "new"))
    await self.bus.drop_lease("w0")
    new_token, _, _ = await self.bus.acquire_run(run_id, "new")
    self.assertNotEqual(new_token, old_token)

    await self.bus.touch_leases("w0")
    self.assertIsNone(await self.bus.acquire_run(run_id, "w0"))
    self.assertFalse(await self.bus.commit_reply(claimed, ok(by="zombie"), old_token))
    self.assertFalse(await self.bus.detach_run(run_id, old_token, None, "tinker://x/parked/1\nv1"))
    await self.bus.fail_stranded_commands(run_id, old_token)
    self.assertIsNone(await self.bus.wait_for_reply(first, 0))
    self.assertEqual((await self.bus.get_run(run_id))["failed"], 0)

    await self.bus.fail_stranded_commands(run_id, new_token)
    reply = await self.bus.wait_for_reply(first, 0)
    self.assertFalse(reply.ok)
    self.assertEqual(json.loads(reply.body)["category"], "server")
    self.assertIn(claimed.op, json.loads(reply.body)["error"])
    self.assertIsNone(await self.bus.wait_for_reply(second, 0))
    self.assertEqual(await self.bus.queued_and_in_flight(run_id), {"live": True, "queued": 1, "in_flight": []})
    self.assertEqual((await self.bus.get_run(run_id))["failed"], 1)

  async def test_old_epoch_cannot_claim_work_or_publish_a_named_or_parked_checkpoint(self):
    run = await self.new_run()
    old = self.tokens[run]
    await self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
    abandoned = await self.bus.claim_trainer(run, old)
    await self.bus.drop_lease("w0")
    await self.bus.touch_leases("replacement")
    current, _, _ = await self.bus.acquire_run(run, "replacement")
    await self.bus.fail_stranded_commands(run, current)
    request = await self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
    with self.assertRaisesRegex(RuntimeError, "lost ownership"):
      await self.bus.claim_trainer(run, old)
    cmd = await self.bus.claim_trainer(run, current)
    self.assertEqual(cmd.request_id, request)
    uri = f"tinker://{run}/weights/step"
    saved = {"uri": uri, "size": 20, "version": "new"}
    self.assertTrue(await self.bus.commit_reply(cmd, ok(), current, checkpoint=saved))

    stale = {"uri": uri, "size": 10, "ttl": 60, "version": "old"}
    self.assertFalse(await self.bus.commit_reply(abandoned, ok(), old, checkpoint=stale))
    self.bus.replies.pop(request, None)
    self.assertFalse(await self.bus.commit_reply(cmd, ok(), current, checkpoint=stale))
    self.assertEqual(await self.bus.checkpoint_version(uri), "new")
    self.assertIsNone((await self.bus.get_checkpoint(uri))["expires_at"])
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

    req1 = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    c1 = await self.bus.claim_trainer(run_id, token)
    self.assertTrue(await self.bus.commit_reply(c1, ok(), token, mutates=True))
    self.assertFalse(await self.bus.detach_run(run_id, token, "0-1", "tinker://r/parked/bad\nv0"))
    parked_uri = f"tinker://{run_id}/parked/1"
    await self.seed_checkpoint(parked_uri, 1, owned=True, version="v1")
    self.assertTrue(await self.bus.detach_run(run_id, token, c1.entry_id, f"{parked_uri}\nv1"))

    token2, pos2, snap2 = await self.bus.acquire_run(run_id, "b")
    self.assertNotEqual(token2, token)
    self.assertEqual((pos2, snap2), (c1.entry_id, f"{parked_uri}\nv1"))
    self.assertIsNotNone(req1)

    await self.bus.submit_command(run_id, Op.SAVE_WEIGHTS, b"{}")
    ro = await self.bus.claim_trainer(run_id, token2)
    self.assertTrue(await self.bus.commit_reply(ro, ok(), token2, mutates=False))
    self.assertEqual((await self.bus.get_run(run_id))["snap"], snap2)
    self.assertIsNotNone(await self.bus.get_checkpoint(parked_uri))

    await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    mut = await self.bus.claim_trainer(run_id, token2)
    self.assertTrue(await self.bus.commit_reply(mut, ok(), token2, mutates=True))
    self.assertIsNone((await self.bus.get_run(run_id)).get("snap"))
    self.assertIsNone(await self.bus.get_checkpoint(parked_uri))

  async def test_unreaped_lease_recovery_preserves_epoch_token(self):
    run_id = await self.new_run()
    token = self.tokens[run_id]
    self.bus.leases.pop("w0", None)
    await self.bus.touch_leases("w0")
    self.assertIn("w0", self.bus.leases)
    await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    cmd = await self.bus.claim_trainer(run_id, token)
    self.assertTrue(await self.bus.commit_reply(cmd, ok(alive=1), token, mutates=True))

  async def test_lease_renewal_does_not_resurrect_dropped_lease(self):
    await self.bus.touch_leases("live", "dropped")
    await self.bus.drop_lease("dropped")
    lost = await self.bus.node_heartbeat("node-a", 1, leases=("live", "dropped"))
    self.assertEqual(lost, ["dropped"])
    await self.bus.touch_leases("live", "dropped", xx=True)
    self.assertIn("live", self.bus.leases)
    self.assertNotIn("dropped", self.bus.leases)

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

    self.bus.replies.pop(req, None)
    self.assertFalse(await self.bus.commit_sample(cmd, ok(late=True), "bob", Charge(busy_ms=100, gpu_ms=100)))
    with self.assertRaises(PromiseGoneError):
      await self.bus.wait_for_reply(req, 0)
    self.assertEqual(await self.bus.get_run(sid), run)
    self.assertEqual((await self.bus.read_state())["service"]["bob"], service)

  async def test_sampling_backpressure_and_unknown_promise_raise_typed_errors(self):
    sid = await self.new_run(role=Role.SAMPLER)
    with mock.patch("tinyrl.bus.SAMPLE_QUEUE_BYTES", 16):
      await self.bus.submit_command(sid, Op.SAMPLE, b"0123456789")
      with self.assertRaises(BackpressureError):
        await self.bus.submit_command(sid, Op.SAMPLE, b"0123456789")
    with self.assertRaises(PromiseGoneError):
      await self.bus.wait_for_reply("samp-missing/1-1", 0)

  async def test_a_sample_in_flight_is_reclaimed_only_from_a_dead_replica(self):
    session = await self.new_run(role=Role.SAMPLER)
    requests = [await self.bus.submit_command(session, Op.SAMPLE, b"{}") for _ in range(2)]
    taken = await self.bus.take_commands([session], "w0")
    self.assertEqual([cmd.request_id for cmd in taken], requests)
    for claim in self.bus.in_flight[session].values():
      claim.claimed_at = 0.0
    self.assertEqual(await self.bus.take_commands([session], "other"), [])
    self.assertEqual(await self.bus.peek_sampler([session]), [])

    await self.bus.drop_lease("w0")
    [head] = await self.bus.peek_sampler([session])
    self.assertEqual(head.request_id, requests[0])
    reclaimed = await self.bus.take_commands([session], "other")
    self.assertEqual([cmd.request_id for cmd in reclaimed], requests)

  async def test_a_dead_node_daemon_stops_counting(self):
    await self.bus.node_heartbeat("node-a", 2)
    self.bus.nodes["node-b"] = {"gpus": 8, "memory": 0, "until": 0}
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
    self.assertEqual((await self.bus.queued_and_in_flight(run_id))["queued"], 1)
    self.assertTrue(self.bus.sessions["s"]["active"])
    self.bus.queues[run_id].clear()
    await self.bus.end_session("s")
    self.assertEqual((await self.bus.queued_and_in_flight(run_id))["queued"], 1)
    await self.bus.delete_run(run_id)
    await self.bus.end_session("s")
    self.assertFalse(self.bus.sessions["s"]["active"])

  async def test_a_deleted_run_does_not_hide_another_runs_command(self):
    run_id, gone = await self.new_run(role=Role.SAMPLER), await self.new_run(role=Role.SAMPLER)
    request = await self.bus.submit_command(run_id, Op.SAMPLE, b"{}")
    await self.bus.delete_run(gone)
    [cmd] = await self.bus.take_commands([gone, run_id], "o")
    self.assertEqual((cmd.run, cmd.request_id), (run_id, request))
    self.assertEqual(await self.bus.take_commands([run_id], "o"), [])


class TestControlRestart(unittest.IsolatedAsyncioTestCase):
  async def test_remote_bus_waits_out_control_server_restart_and_preserves_durable_state(self):
    with tempfile.TemporaryDirectory() as tmp:
      db_path = f"{tmp}/tinyrl.db"
      bus = Bus(db_path)
      runner, url = await serve_bus(bus, host="127.0.0.1", port=0)
      port = int(url.rsplit(":", 1)[1])
      remote = RemoteBus(url)
      try:
        await remote.touch_leases("n1")
        await remote.heartbeat_session("s1", {"sdk": "0.39"})
        await remote.create_run(Run("tr-1", "base", 4, session="s1", account="alice"))
        token, _, _ = await remote.acquire_run("tr-1", "n1")
        await remote.submit_command("tr-1", Op.SAVE_WEIGHTS, b"\x00\xff", cost=5)
        cmd = await remote.claim_trainer("tr-1", token)
        self.assertEqual(cmd.body, b"\x00\xff")
        ckpt = {"uri": "tinker://tr-1/weights/v1", "size": 99, "version": "ver1"}
        self.assertTrue(await remote.commit_reply(cmd, ok(done=1), token, "alice", True, checkpoint=ckpt))
        await remote.node_heartbeat("node-a", 1, leases=("n1",))

        await runner.cleanup()
        await bus.aclose()

        async def restart():
          await asyncio.sleep(0.4)
          new_bus = Bus(db_path)
          new_runner, _ = await serve_bus(new_bus, host="127.0.0.1", port=port)
          return new_bus, new_runner

        restarting = asyncio.create_task(restart())
        await remote.node_heartbeat("node-a", 1)
        bus, runner = await restarting
        self.assertEqual(await remote.live_nodes(), {"node-a": 1})
        # Boot grace window prevents immediate reaping of n1 before LEASE_MS elapses.
        await remote.touch_leases("n2")
        self.assertIsNone(await remote.acquire_run("tr-1", "n2"))
        # Durable sessions, runs, checkpoints, and ownership survive control-server restart.
        self.assertEqual((await remote.get_session("s1"))["metadata"], {"sdk": "0.39"})
        self.assertEqual(await remote.checkpoint_version("tinker://tr-1/weights/v1"), "ver1")
        self.assertEqual((await remote.get_run("tr-1"))["owner"], token)
        # Lost sampling promises raise PromiseGoneError across HTTP RPC.
        with self.assertRaises(PromiseGoneError):
          await remote.wait_for_reply("samp-lost/1-1", 0)
      finally:
        await remote.aclose()
        await runner.cleanup()
        await bus.aclose()


if __name__ == "__main__":
  unittest.main()
