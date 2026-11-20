import asyncio
import json
import tempfile
import time
import unittest
from unittest import mock

from test.helpers import BusTest, new_id
from tinyrl.bus import BackpressureError, Bus, PromiseGoneError, serve_control
from tinyrl.control.accounting import Charge
from tinyrl.rpc import ControlClient
from tinyrl.types import NodeReport, Op, Reply, Role, Run, TurnResult, WorkerStatus, unpack_snap


def ok(**body) -> Reply:
  return Reply(True, "application/json", json.dumps(body).encode())


class TestBus(BusTest):
  async def new_run(self, session: str = "", role: Role = Role.TRAINER, account: str = "") -> str:
    run_id = new_id()
    await self.bus.create_run(Run(id=run_id, base="base", rank=8, session=session, role=role, account=account))
    return run_id

  async def test_a_command_goes_through_exchange_and_its_reply_comes_back(self):
    run_id = await self.new_run(account="alice")
    request_id = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b'{"lr": 1}', cost=42)

    reply = await self.bus.exchange(NodeReport("n1", "b1", 1))
    [turn] = reply.turns
    cmd = turn.cmd
    self.assertEqual(
      (cmd.run, cmd.op, cmd.body, cmd.request_id, cmd.cost), (run_id, Op.OPTIM_STEP, b'{"lr": 1}', request_id, 42)
    )
    self.assertIsNone(await self.bus.wait_for_reply(request_id, 0))

    charge = Charge(started_ms=int(cmd.entry_id.split("-")[0]) + 12, busy_ms=50, gpu_ms=100.5)
    res = TurnResult(
      turn.id, "train", turn.worker, reply=ok(type="optim_step"), charge=charge, mutated=True, loaded=True
    )
    done = await self.bus.exchange(NodeReport("n1", "b1", 1, finished=[res]))
    self.assertEqual(done.acknowledged, [turn.id])
    self.assertEqual(done.turns, [])
    self.assertEqual(await self.bus.wait_for_reply(request_id, 0), ok(type="optim_step"))

    # Repeated completion delivery is idempotent and still acknowledged without double-charging.
    again = await self.bus.exchange(NodeReport("n1", "b1", 1, finished=[res]))
    self.assertEqual(again.acknowledged, [turn.id])

    run = await self.bus.get_run(run_id)
    self.assertEqual(
      (run["base"], run["rank"], run["steps"], run["tokens"], run["pos"]), ("base", 8, 1, 42, cmd.entry_id)
    )
    self.assertEqual((run["wait_ms"], run["busy_ms"]), (12, 50))
    self.assertEqual((await self.bus.read_state())["service"]["alice"], 100.5)

  async def test_a_reply_wakes_every_waiter_and_reading_it_never_consumes_it(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    [turn] = (await self.bus.exchange(NodeReport("n1", "b1", 1))).turns

    waiters = [asyncio.create_task(self.bus.wait_for_reply(request_id, 5)) for _ in range(3)]
    await asyncio.sleep(0.05)
    start = time.perf_counter()
    await self.bus.exchange(
      NodeReport("n1", "b1", 1, finished=[TurnResult(turn.id, "train", turn.worker, reply=ok(n=1))])
    )

    self.assertEqual(await asyncio.gather(*waiters), [ok(n=1)] * 3)
    self.assertLess(time.perf_counter() - start, 0.5)
    self.assertEqual(await self.bus.wait_for_reply(request_id, 0), ok(n=1))

  async def test_hundreds_of_requests_wait_for_replies_at_once(self):
    got = await asyncio.gather(*[self.bus.wait_for_reply(f"req-{i}", 0.5) for i in range(300)])
    self.assertEqual(got, [None] * 300)

  async def test_waiting_for_a_reply_outlasts_the_socket_timeout(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    [turn] = (await self.bus.exchange(NodeReport("n1", "b1", 1))).turns

    async def answer_late():
      await asyncio.sleep(1.0)
      await self.bus.exchange(
        NodeReport("n1", "b1", 1, finished=[TurnResult(turn.id, "train", turn.worker, reply=ok(n=1))])
      )

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
    request = await self.bus.submit_command(run, Op.DELETE_MODEL, b"{}")
    [turn] = (await self.bus.exchange(NodeReport("n1", "b1", 1))).turns
    await self.bus.exchange(NodeReport("n1", "b1", 1, finished=[TurnResult(turn.id, "train", turn.worker, reply=ok())]))
    self.assertEqual(await self.bus.wait_for_reply(request, 5), ok())
    self.assertNotIn(run, self.bus.queues)
    self.assertIsNone((await self.bus.get_run(run)).get("snap"))
    self.assertIsNone((await self.bus.get_run(run)).get("owner"))
    self.assertNotIn(run, (await self.bus.read_state())["runs"])
    self.assertIsNone(await self.bus.get_checkpoint(owned))
    self.assertIsNotNone(await self.bus.get_checkpoint(saved))
    self.assertEqual([c["tinker_path"] for c in (await self.bus.page_of_checkpoints(10, 0))[0]], [saved])

  async def test_lost_exchange_reply_redelivers_the_same_turn_id(self):
    run_id = await self.new_run()
    request_ids = [await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"") for _ in range(3)]
    self.assertEqual(await self.bus.queued_and_in_flight(run_id), {"live": True, "queued": 3, "in_flight": []})

    [t1] = (await self.bus.exchange(NodeReport("n1", "b1", 1))).turns
    [t1_retry] = (await self.bus.exchange(NodeReport("n1", "b1", 1))).turns
    self.assertEqual(t1.id, t1_retry.id)
    queue = await self.bus.queued_and_in_flight(run_id)
    self.assertEqual(queue["queued"], 2)
    self.assertEqual([c["request_id"] for c in queue["in_flight"]], [request_ids[0]])

    reply2 = await self.bus.exchange(
      NodeReport("n1", "b1", 1, finished=[TurnResult(t1.id, "train", t1.worker, reply=ok(), loaded=True)])
    )
    self.assertEqual(reply2.acknowledged, [t1.id])
    self.assertEqual(reply2.turns[0].cmd.request_id, request_ids[1])

  async def test_reaped_boot_id_is_fenced_and_gets_reset_without_overwriting_checkpoint(self):
    run = await self.new_run()
    first = await self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
    [t1] = (await self.bus.exchange(NodeReport("n1", "b1", 1))).turns

    # n1 times out; its in-flight command fails and n2 takes over.
    self.bus.nodes["n1"].until = 0.0
    await self.bus.maintain()
    failed = await self.bus.wait_for_reply(first, 0)
    self.assertFalse(failed.ok)
    self.assertIn("optim_step" if False else "save_weights", json.loads(failed.body)["error"])
    self.assertEqual((await self.bus.get_run(run))["failed"], 1)

    second = await self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
    [t2] = (await self.bus.exchange(NodeReport("n2", "b2", 1))).turns
    self.assertEqual(t2.cmd.request_id, second)
    uri = f"tinker://{run}/weights/step"
    saved = {"uri": uri, "size": 20, "version": "new"}
    await self.bus.exchange(
      NodeReport("n2", "b2", 1, finished=[TurnResult(t2.id, "train", t2.worker, reply=ok(), saved=saved)])
    )

    # Stale n1/b1 returns with a late result; it gets reset=True and cannot overwrite reply or checkpoint.
    stale = {"uri": uri, "size": 10, "ttl": 60, "version": "old"}
    stale_reply = await self.bus.exchange(
      NodeReport("n1", "b1", 1, finished=[TurnResult(t1.id, "train", t1.worker, reply=ok(), saved=stale)])
    )
    self.assertTrue(stale_reply.reset)
    self.assertEqual(await self.bus.wait_for_reply(first, 0), failed)
    self.assertEqual(await self.bus.checkpoint_version(uri), "new")
    self.assertIsNone((await self.bus.get_checkpoint(uri))["expires_at"])

  async def test_snapshot_saves_parked_checkpoint_and_mutating_step_clears_it(self):
    run_id = await self.new_run()
    req1 = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    [t1] = (await self.bus.exchange(NodeReport("n1", "b1", 1))).turns
    await self.bus.exchange(
      NodeReport("n1", "b1", 1, finished=[TurnResult(t1.id, "train", t1.worker, reply=ok(), mutated=True, loaded=True)])
    )
    self.assertIsNotNone(req1)

    # Draining n1 issues an off-GPU snapshot turn.
    drain_reply = await self.bus.exchange(
      NodeReport("n1", "b1", 1, draining=True, workers=[WorkerStatus(t1.worker, (0,), loaded=(run_id,))])
    )
    [snap_turn] = drain_reply.turns
    self.assertEqual((snap_turn.kind, snap_turn.gpus, snap_turn.args["run"]), ("snapshot", (), run_id))
    snap_ver = snap_turn.args["version"]
    parked_uri = f"tinker://{run_id}/parked/{snap_ver}"
    saved = {"uri": parked_uri, "size": 1, "owned": True, "version": snap_ver}
    await self.bus.exchange(
      NodeReport(
        "n1", "b1", 1, draining=True, finished=[TurnResult(snap_turn.id, "snapshot", snap_turn.worker, saved=saved)]
      )
    )
    info = await self.bus.get_run(run_id)
    self.assertIsNone(info.get("owner"))
    self.assertEqual(unpack_snap(info["snap"]), (parked_uri, snap_ver))

    # Read-only SAVE_WEIGHTS on n2 keeps the parked snap; mutating OPTIM_STEP clears it.
    await self.bus.submit_command(run_id, Op.SAVE_WEIGHTS, b"{}")
    [ro] = (await self.bus.exchange(NodeReport("n2", "b2", 1))).turns
    self.assertEqual(ro.args["saved"], snap_ver)
    await self.bus.exchange(
      NodeReport(
        "n2", "b2", 1, finished=[TurnResult(ro.id, "train", ro.worker, reply=ok(), mutated=False, loaded=True)]
      )
    )
    self.assertIsNotNone(await self.bus.get_checkpoint(parked_uri))

    await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    [mut] = (
      await self.bus.exchange(NodeReport("n2", "b2", 1, workers=[WorkerStatus(ro.worker, (0,), loaded=(run_id,))]))
    ).turns
    await self.bus.exchange(
      NodeReport(
        "n2", "b2", 1, finished=[TurnResult(mut.id, "train", mut.worker, reply=ok(), mutated=True, loaded=True)]
      )
    )
    self.assertIsNone((await self.bus.get_run(run_id)).get("snap"))
    self.assertIsNone(await self.bus.get_checkpoint(parked_uri))

  async def test_competing_sampler_replicas_answer_and_charge_exactly_once(self):
    sid = await self.new_run(role=Role.SAMPLER, account="bob")
    req = await self.bus.submit_command(sid, Op.SAMPLE, b"{}", cost=100)
    [turn] = (await self.bus.exchange(NodeReport("n1", "b1", 1))).turns
    cmd = turn.cmds[0]
    results = [
      self.bus.commit_sample_cmd(cmd, ok(replica=i), turn.id, Charge(busy_ms=40, gpu_ms=40.5 + i)) for i in range(5)
    ]
    self.assertEqual(sum(results), 1)
    winner = results.index(True)
    self.assertEqual(await self.bus.wait_for_reply(req, 0), ok(replica=winner))
    run = await self.bus.get_run(sid)
    self.assertEqual((run["cmds"], run["tokens"]), (1, 100))
    service = (await self.bus.read_state())["service"]["bob"]
    self.assertEqual(service, 40.5 + winner)

    self.bus.sample_replies.pop(req, None)
    self.assertFalse(self.bus.commit_sample_cmd(cmd, ok(late=True), turn.id, Charge(busy_ms=100, gpu_ms=100)))
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

  async def test_a_sample_in_flight_is_reclaimed_only_from_a_dead_node(self):
    session = await self.new_run(role=Role.SAMPLER)
    requests = [await self.bus.submit_command(session, Op.SAMPLE, b"{}") for _ in range(2)]
    [t1] = (await self.bus.exchange(NodeReport("n1", "b1", 1))).turns
    self.assertEqual([c.request_id for c in t1.cmds], requests)
    # While n1 is alive, n2 gets nothing for those in-flight samples.
    self.assertEqual((await self.bus.exchange(NodeReport("n2", "b2", 1))).turns, [])
    # Once n1 dies, n2 reclaims both in FIFO order.
    self.bus.nodes["n1"].until = 0.0
    [t2] = (await self.bus.exchange(NodeReport("n2", "b2", 1))).turns
    self.assertEqual([c.request_id for c in t2.cmds], requests)

  async def test_a_dead_node_daemon_stops_counting(self):
    await self.bus.exchange(NodeReport("node-a", "ba", 2))
    await self.bus.exchange(NodeReport("node-b", "bb", 8))
    self.bus.nodes["node-b"].until = 0.0
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
    [turn] = (await self.bus.exchange(NodeReport("n1", "b1", 1))).turns
    self.assertEqual([(c.run, c.request_id) for c in turn.cmds], [(run_id, request)])


class TestControlRestart(unittest.IsolatedAsyncioTestCase):
  async def test_exchange_client_survives_control_server_restart_and_preserves_durable_state(self):
    with tempfile.TemporaryDirectory() as tmp:
      db_path = f"{tmp}/tinyrl.db"
      bus = Bus(db_path)
      runner, url = await serve_control(bus, host="127.0.0.1", port=0)
      port = int(url.rsplit(":", 1)[1])
      client = ControlClient(url)
      try:
        await bus.heartbeat_session("s1", {"sdk": "0.39"})
        await bus.create_run(Run("tr-1", "base", 4, session="s1", account="alice"))
        req = await bus.submit_command("tr-1", Op.SAVE_WEIGHTS, b"\x00\xff", cost=5)
        reply = await client.exchange(NodeReport("node-a", "boot-a", 1))
        [turn] = reply.turns
        self.assertEqual(turn.cmd.body, b"\x00\xff")
        ckpt = {"uri": "tinker://tr-1/weights/v1", "size": 99, "version": "ver1"}
        done = TurnResult(turn.id, "train", turn.worker, reply=ok(done=1), mutated=True, loaded=True, saved=ckpt)

        # Restart control server BEFORE node delivers TurnResult!
        await runner.cleanup()
        await bus.aclose()
        bus = Bus(db_path)
        runner, _ = await serve_control(bus, host="127.0.0.1", port=port)

        # Node delivers TurnResult over HTTP to restarted controller -> committed durably!
        ack = await client.exchange(NodeReport("node-a", "boot-a", 1, finished=[done]))
        self.assertEqual(ack.acknowledged, [turn.id])
        self.assertEqual(await bus.wait_for_reply(req, 0), ok(done=1))
        self.assertEqual((await bus.get_session("s1"))["metadata"], {"sdk": "0.39"})
        self.assertEqual(await bus.checkpoint_version("tinker://tr-1/weights/v1"), "ver1")
        with self.assertRaises(PromiseGoneError):
          await bus.wait_for_reply("samp-lost/1-1", 0)
      finally:
        await client.aclose()
        await runner.cleanup()
        await bus.aclose()


if __name__ == "__main__":
  unittest.main()
