import asyncio
import json
import tempfile
import time
import unittest
from unittest import mock

from test.helpers import BusTest, new_id, serve_control
from tinyrl.control.accounting import Charge
from tinyrl.control.ledger import Ledger
from tinyrl.control.node_daemon import ControlClient
from tinyrl.control.samples import BackpressureError, PromiseGoneError
from tinyrl.types import Ev, FencedError, Op, Reply, Role, Run, RunStatus


def ok(**body) -> Reply:
  return Reply(True, "application/json", json.dumps(body).encode())


class TestBus(BusTest):
  def new_run(self, session: str = "", role: Role = Role.TRAINER, account: str = "") -> str:
    run_id = new_id()
    self.bus.create_run(Run(id=run_id, base="base", rank=8, session=session, role=role, account=account))
    return run_id

  async def take_one(self, node: str = "n1", boot: str = "b1", gpus: int = 1, **opts):
    plan, _ = await self.bus.heartbeat(node, boot, gpus)
    for a in plan:
      if items := self.bus.take(node, boot, a.worker, **opts):
        return a.worker, items[0]
    raise AssertionError("expected ready work")

  async def test_a_command_goes_through_take_and_finish_and_its_reply_comes_back(self):
    run_id = self.new_run(account="alice")
    request_id = self.bus.submit_command(run_id, Op.OPTIM_STEP, b'{"lr": 1}', cost=42)

    wid, w = await self.take_one()
    cmd = w.cmd
    self.assertEqual(
      (cmd.run, cmd.op, cmd.body, cmd.request_id, cmd.cost), (run_id, Op.OPTIM_STEP, b'{"lr": 1}', request_id, 42)
    )
    self.assertIsNone(await self.bus.wait_for_reply(request_id, 0))

    charge = Charge(started_ms=int(cmd.entry_id.split("-")[0]) + 12, busy_ms=50, gpu_ms=100.5)
    self.assertTrue(self.bus.finish("n1", "b1", wid, request_id, ok(type="optim_step"), charge, mutated=True))
    self.assertEqual(await self.bus.wait_for_reply(request_id, 0), ok(type="optim_step"))

    # Repeated finish is idempotent and does not double-charge.
    self.assertFalse(self.bus.finish("n1", "b1", wid, request_id, ok(type="optim_step"), charge, mutated=True))

    run = self.bus.get_run(run_id)
    self.assertEqual(
      (run["base"], run["rank"], run["steps"], run["tokens"], run["status"]), ("base", 8, 1, 42, RunStatus.LIVE)
    )
    self.assertEqual((run["wait_ms"], run["busy_ms"]), (12, 50))
    self.assertEqual(self.bus.service["alice"], 100.5)

  async def test_a_reply_wakes_every_waiter_and_reading_it_never_consumes_it(self):
    run_id = self.new_run()
    request_id = self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    wid, _ = await self.take_one()

    waiters = [asyncio.create_task(self.bus.wait_for_reply(request_id, 5)) for _ in range(3)]
    await asyncio.sleep(0.05)
    start = time.perf_counter()
    self.bus.finish("n1", "b1", wid, request_id, ok(n=1))

    self.assertEqual(await asyncio.gather(*waiters), [ok(n=1)] * 3)
    self.assertLess(time.perf_counter() - start, 0.5)
    self.assertEqual(await self.bus.wait_for_reply(request_id, 0), ok(n=1))

  async def test_hundreds_of_requests_wait_for_replies_at_once(self):
    got = await asyncio.gather(*[self.bus.wait_for_reply(f"req-{i}", 0.2) for i in range(300)])
    self.assertEqual(got, [None] * 300)

  async def test_waiting_for_a_reply_outlasts_a_delay(self):
    run_id = self.new_run()
    request_id = self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    wid, _ = await self.take_one()

    async def answer_late():
      await asyncio.sleep(0.2)
      self.bus.finish("n1", "b1", wid, request_id, ok(n=1))

    answering = asyncio.create_task(answer_late())
    self.assertEqual(await self.bus.wait_for_reply(request_id, 10), ok(n=1))
    await answering

  async def test_a_retried_call_is_queued_once(self):
    run_id = self.new_run()
    first = self.bus.submit_command(run_id, Op.OPTIM_STEP, b"", seq_id=5)
    self.assertEqual(self.bus.submit_command(run_id, Op.OPTIM_STEP, b"", seq_id=5), first)
    self.assertNotEqual(self.bus.submit_command(run_id, Op.OPTIM_STEP, b"", seq_id=6), first)
    self.assertEqual(self.bus.queued_and_in_flight(run_id)["queued"], 2)

  async def test_a_retry_after_its_run_was_deleted_gets_the_first_request_id(self):
    run_id = self.new_run()
    first = self.bus.submit_command(run_id, Op.DELETE_MODEL, b"", seq_id=3)
    self.bus.delete_run(run_id)
    self.assertEqual(self.bus.submit_command(run_id, Op.DELETE_MODEL, b"", seq_id=3), first)
    self.assertIsNone(self.bus.submit_command(run_id, Op.DELETE_MODEL, b"", seq_id=4))

  async def test_unknown_and_deleted_runs_take_no_work(self):
    self.assertIsNone(self.bus.submit_command(new_id(), Op.OPTIM_STEP, b""))
    run_id = self.new_run()
    self.bus.delete_run(run_id)
    self.assertIsNone(self.bus.submit_command(run_id, Op.OPTIM_STEP, b""))
    self.assertEqual(self.bus.queued_and_in_flight(run_id), {"live": False})
    self.assertNotIn(run_id, self.bus.train_q)

  async def test_delete_removes_owned_checkpoints_and_keeps_saved_weights(self):
    run = self.new_run()
    owned, saved = [f"tinker://{run}/{path}" for path in ("sampler_weights/session", "weights/save")]
    for uri in (owned, saved):
      self.seed_checkpoint(uri, 1, ttl=60, owned=uri == owned)
    request = self.bus.submit_command(run, Op.DELETE_MODEL, b"{}")
    wid, _ = await self.take_one()
    self.bus.finish("n1", "b1", wid, request, ok())
    self.assertEqual(await self.bus.wait_for_reply(request, 5), ok())
    self.assertNotIn(run, self.bus.train_q)
    self.assertIsNone(self.bus.get_run(run).get("snap"))
    self.assertIsNone(self.bus.get_run(run).get("owner"))
    self.assertNotIn(run, self.bus.state().jobs)
    self.assertIsNone(self.bus.get_checkpoint(owned))
    self.assertIsNotNone(self.bus.get_checkpoint(saved))

  async def test_lost_take_reply_redelivers_same_command_unless_held(self):
    run_id = self.new_run()
    request_ids = [self.bus.submit_command(run_id, Op.OPTIM_STEP, b"") for _ in range(3)]
    self.assertEqual(self.bus.queued_and_in_flight(run_id), {"live": True, "queued": 3, "in_flight": []})

    wid, w1 = await self.take_one()
    [w1_retry] = self.bus.take("n1", "b1", wid)
    self.assertEqual(w1, w1_retry)
    self.assertEqual(self.bus.take("n1", "b1", wid, holding=(w1.cmd.request_id,)), [])
    queue = self.bus.queued_and_in_flight(run_id)
    self.assertEqual(queue["queued"], 2)
    self.assertEqual([c["request_id"] for c in queue["in_flight"]], [request_ids[0]])

    self.assertTrue(self.bus.finish("n1", "b1", wid, w1.cmd.request_id, ok(), mutated=True))
    [w2] = self.bus.take("n1", "b1", wid)
    self.assertEqual(w2.cmd.request_id, request_ids[1])

  async def test_the_planner_runs_on_heartbeat_only_when_its_inputs_changed(self):
    with (
      mock.patch("tinyrl.control.scheduler.TICK_MS", 10**9),
      mock.patch.object(self.bus, "reconcile", wraps=self.bus.reconcile) as plan,
    ):
      run_id = self.new_run()
      await self.bus.heartbeat("n1", "b1", 1)
      self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
      await self.bus.heartbeat("n1", "b1", 1)
      self.assertEqual(plan.call_count, 1)
      self.new_run()
      await self.bus.heartbeat("n1", "b1", 1)
      self.assertEqual(plan.call_count, 2)

  async def test_expired_dedupe_keys_are_swept_less_often_than_the_tick(self):
    self.bus.maintain()
    with self.bus.db:
      self.bus.seen["run:1"] = ["run/1-1", 0.0]
    self.bus.maintain()
    self.assertIn("run:1", self.bus.seen)
    self.bus.gc_due = 0.0
    self.bus.maintain()
    self.assertNotIn("run:1", self.bus.seen)

  async def test_state_prunes_deleted_runs_and_late_finish_is_ignored(self):
    run_id = self.new_run()
    req = self.bus.submit_command(run_id, Op.FORWARD_BACKWARD, b"")
    wid, _ = await self.take_one()
    self.bus.delete_run(run_id)
    st = self.bus.state()
    self.assertNotIn(run_id, st.home)
    self.assertFalse(st.busy(wid))
    self.assertFalse((await self.bus.wait_for_reply(req, 0)).ok)
    self.assertFalse(self.bus.finish("n1", "b1", wid, req, ok(), mutated=True))

  async def test_sample_finish_ignores_trainer_checkpoint_effects(self):
    sid = self.new_run(role=Role.SAMPLER)
    req = self.bus.submit_command(sid, Op.SAMPLE, b"{}")
    wid, _ = await self.take_one()
    saved = {"uri": f"tinker://{sid}/weights/bad", "size": 1, "version": "v1"}
    self.bus.finish("n1", "b1", wid, req, ok(), saved=saved)
    self.assertFalse(self.bus.db.conn.in_transaction)
    self.assertIsNone(self.bus.get_checkpoint(saved["uri"]))

  async def test_reaped_boot_id_is_fenced_and_gets_reset_without_overwriting_checkpoint(self):
    run = self.new_run()
    first = self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
    w1, _ = await self.take_one("n1", "b1", 1)

    self.bus.nodes["n1"]["until"] = 0.0
    self.bus.maintain()
    failed = await self.bus.wait_for_reply(first, 0)
    self.assertFalse(failed.ok)
    self.assertIn("save_weights", json.loads(failed.body)["error"])
    self.assertEqual(self.bus.get_run(run)["failed"], 1)

    second = self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
    w2, item2 = await self.take_one("n2", "b2", 1)
    self.assertEqual(item2.cmd.request_id, second)
    uri = f"tinker://{run}/weights/step"
    self.bus.finish("n2", "b2", w2, second, ok(), saved={"uri": uri, "size": 20, "version": "new"})

    stale = {"uri": uri, "size": 10, "ttl": 60, "version": "old"}
    with self.assertRaises(FencedError):
      self.bus.finish("n1", "b1", w1, first, ok(), saved=stale)
    self.assertEqual(await self.bus.wait_for_reply(first, 0), failed)
    self.assertEqual(self.bus.checkpoint_version(uri), "new")
    self.assertIsNone(self.bus.get_checkpoint(uri)["expires_at"])

  async def test_detach_saves_parked_checkpoint_and_mutating_step_clears_it(self):
    run_id = self.new_run()
    req1 = self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    w1, _ = await self.take_one("n1", "b1", 1)
    self.bus.finish("n1", "b1", w1, req1, ok(), mutated=True)

    plan, _ = await self.bus.heartbeat("n1", "b1", 1, draining=True, live_workers=(w1,))
    self.assertEqual(plan, [])
    snap_ver = "a" * 32
    parked_uri = f"tinker://{run_id}/parked/{snap_ver}"
    self.assertTrue(
      self.bus.detach("n1", "b1", w1, run_id, saved={"uri": parked_uri, "size": 1, "owned": True, "version": snap_ver})
    )
    info = self.bus.get_run(run_id)
    self.assertIsNone(info.get("owner"))
    self.assertEqual(info["snap"], [parked_uri, snap_ver])
    self.assertEqual(info["status"], RunStatus.SAVED)

    # Read-only SAVE_WEIGHTS on n2 keeps parked snap; mutating OPTIM_STEP clears it.
    req2 = self.bus.submit_command(run_id, Op.SAVE_WEIGHTS, b"{}")
    w2, ro = await self.take_one("n2", "b2", 1)
    self.assertEqual(ro.snap, snap_ver)
    self.bus.finish("n2", "b2", w2, req2, ok(), mutated=False)
    self.assertIsNotNone(self.bus.get_checkpoint(parked_uri))

    req3 = self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    [mut] = self.bus.take("n2", "b2", w2, loaded=(run_id,))
    self.assertIsNone(mut.snap)
    self.bus.finish("n2", "b2", w2, req3, ok(), mutated=True)
    self.assertIsNone(self.bus.get_run(run_id).get("snap"))
    self.assertIsNone(self.bus.get_checkpoint(parked_uri))

  async def test_competing_sampler_replicas_answer_and_charge_exactly_once(self):
    sid = self.new_run(role=Role.SAMPLER, account="bob")
    req = self.bus.submit_command(sid, Op.SAMPLE, b"{}", cost=100)
    wid, _ = await self.take_one("n1", "b1", 1)
    # A competing replica that did not claim `req` cannot answer or bill it.
    self.assertFalse(self.bus.finish("n1", "b1", "other", req, ok(replica=-1), Charge(busy_ms=99, gpu_ms=99)))
    results = [
      self.bus.finish("n1", "b1", wid, req, ok(replica=i), Charge(busy_ms=40, gpu_ms=40.5 + i)) for i in range(5)
    ]
    self.assertEqual(sum(results), 1)
    winner = results.index(True)
    self.assertEqual(await self.bus.wait_for_reply(req, 0), ok(replica=winner))
    run = self.bus.get_run(sid)
    self.assertEqual((run["cmds"], run["tokens"]), (1, 100))
    service = self.bus.service["bob"]
    self.assertEqual(service, 40.5 + winner)

    self.bus.samples.replies.pop(req, None)
    self.assertFalse(self.bus.finish("n1", "b1", wid, req, ok(late=True), Charge(busy_ms=100, gpu_ms=100)))
    with self.assertRaises(PromiseGoneError):
      await self.bus.wait_for_reply(req, 0)
    self.assertEqual(self.bus.get_run(sid), run)
    self.assertEqual(self.bus.service["bob"], service)

  async def test_sampling_backpressure_and_unknown_promise_raise_typed_errors(self):
    sid = self.new_run(role=Role.SAMPLER)
    with mock.patch("tinyrl.control.samples.SAMPLE_QUEUE_BYTES", 16):
      self.bus.submit_command(sid, Op.SAMPLE, b"0123456789")
      with self.assertRaises(BackpressureError):
        self.bus.submit_command(sid, Op.SAMPLE, b"0123456789")
    with self.assertRaises(PromiseGoneError):
      await self.bus.wait_for_reply("samp-missing/1-1", 0)

  async def test_a_sample_in_flight_is_reclaimed_only_from_a_dead_node(self):
    session = self.new_run(role=Role.SAMPLER)
    requests = [self.bus.submit_command(session, Op.SAMPLE, b"{}") for _ in range(2)]
    [a1], _ = await self.bus.heartbeat("n1", "b1", 1)
    t1 = self.bus.take("n1", "b1", a1.worker, slots=4)
    self.assertEqual([w.cmd.request_id for w in t1], requests)
    plan2, _ = await self.bus.heartbeat("n2", "b2", 1)
    self.assertEqual([w for a in plan2 for w in self.bus.take("n2", "b2", a.worker, slots=4)], [])
    self.bus.nodes["n1"]["until"] = 0.0
    [a2], _ = await self.bus.heartbeat("n2", "b2", 1)
    t2 = self.bus.take("n2", "b2", a2.worker, slots=4)
    self.assertEqual([w.cmd.request_id for w in t2], requests)

  async def test_a_dead_node_daemon_stops_counting(self):
    await self.bus.heartbeat("node-a", "ba", 2)
    await self.bus.heartbeat("node-b", "bb", 8)
    self.bus.nodes["node-b"]["until"] = 0.0
    self.assertEqual(self.bus.live_nodes(), {"node-a": 2})

  async def test_sessions(self):
    self.bus.heartbeat_session("s")
    run_id = self.new_run(session="s")
    session = self.bus.get_session("s")
    self.assertTrue(session["live"])
    self.assertEqual(session["runs"], [run_id])
    self.assertIsNone(self.bus.get_session("nope"))

  async def test_a_session_is_ended_again_until_its_training_runs_are_gone(self):
    self.bus.heartbeat_session("s")
    run_id = self.new_run(session="s")
    self.bus.end_session("s")
    self.assertEqual(self.bus.queued_and_in_flight(run_id)["queued"], 1)
    self.assertTrue(self.bus.sessions["s"]["active"])
    self.bus.train_q[run_id].clear()
    self.bus.end_session("s")
    self.assertEqual(self.bus.queued_and_in_flight(run_id)["queued"], 1)
    self.bus.delete_run(run_id)
    self.bus.end_session("s")
    self.assertFalse(self.bus.sessions["s"]["active"])

  async def test_a_deleted_run_does_not_hide_another_runs_command(self):
    run_id, gone = self.new_run(role=Role.SAMPLER), self.new_run(role=Role.SAMPLER)
    request = self.bus.submit_command(run_id, Op.SAMPLE, b"{}")
    self.bus.delete_run(gone)
    _, w = await self.take_one()
    self.assertEqual((w.cmd.run, w.cmd.request_id), (run_id, request))


class TestControlRestart(unittest.IsolatedAsyncioTestCase):
  async def test_a_node_that_never_returns_after_a_restart_is_lost_once(self):
    with tempfile.TemporaryDirectory() as tmp:
      bus = Ledger(f"{tmp}/tinyrl.db")
      for name in ("a", "b"):
        bus.create_run(Run(name, f"base-{name}", 0))
        bus.submit_command(name, Op.OPTIM_STEP, b"")
      plan, _ = await bus.heartbeat("n1", "b1", 2)
      self.assertEqual(sum(len(bus.take("n1", "b1", a.worker)) for a in plan), 2)
      await bus.aclose()
      with mock.patch("tinyrl.control.ledger.LEASE_MS", 0):
        bus = Ledger(f"{tmp}/tinyrl.db")
      bus.maintain()
      self.assertEqual([e.kind for e in bus.read_events()].count(Ev.NODE_LOST), 1)
      await bus.aclose()

  async def test_rpc_client_survives_control_server_restart_and_preserves_durable_state(self):
    with tempfile.TemporaryDirectory() as tmp:
      db_path = f"{tmp}/tinyrl.db"
      bus = Ledger(db_path)
      runner, url = await serve_control(bus, host="127.0.0.1", port=0)
      port = int(url.rsplit(":", 1)[1])
      client = ControlClient(url)
      try:
        bus.heartbeat_session("s1", {"sdk": "0.39"})
        bus.create_run(Run("tr-1", "base", 4, session="s1", account="alice"))
        req = bus.submit_command("tr-1", Op.SAVE_WEIGHTS, b"\x00\xff", cost=5)
        [a], _ = await client.heartbeat("node-a", "boot-a", 1)
        [w] = await client.take("node-a", "boot-a", a.worker)
        self.assertEqual(w.cmd.body, b"\x00\xff")
        ckpt = {"uri": "tinker://tr-1/weights/v1", "size": 99, "version": "ver1"}

        await runner.cleanup()
        await bus.aclose()
        bus = Ledger(db_path)
        runner, _ = await serve_control(bus, host="127.0.0.1", port=port)

        self.assertTrue(await client.finish("node-a", "boot-a", a.worker, req, ok(done=1), mutated=True, saved=ckpt))
        self.assertEqual(await bus.wait_for_reply(req, 0), ok(done=1))
        self.assertEqual(bus.get_session("s1")["metadata"], {"sdk": "0.39"})
        self.assertEqual(bus.checkpoint_version("tinker://tr-1/weights/v1"), "ver1")
        with self.assertRaises(PromiseGoneError):
          await bus.wait_for_reply("samp-lost/1-1", 0)
      finally:
        await client.aclose()
        await runner.cleanup()
        await bus.aclose()

  async def test_reopen_redelivers_exact_trainer_claim_and_preserves_drain_ownership(self):
    from test.execution import TestNode
    from tinyrl.storage.store import Store

    with tempfile.TemporaryDirectory() as tmp:
      db_path = f"{tmp}/tinyrl.db"
      store = Store(root=f"{tmp}/ckpt")
      bus = Ledger(db_path)
      node = TestNode(bus, gpus=1, node="n1", store=store)
      try:
        rid = new_id()
        bus.create_run(Run(rid, "tiny", 4, account="alice"))
        bus.submit_command(rid, Op.CREATE_MODEL, b'{"base_model":"tiny","rank":4,"lora":{}}')
        await node.drain(rid)
        self.assertEqual(node.workers["p1"].resident, (0,))

        # Reopen SQLite: node keeps local GPU residency & dirty state; sampler parks trainer cleanly without overlap.
        await bus.aclose()
        bus = Ledger(db_path)
        node.control = bus
        bus.create_run(Run("samp-1", "other", 4, role=Role.SAMPLER, account="bob"))
        sreq = bus.submit_command(
          "samp-1",
          Op.SAMPLE,
          b'{"prompt":[1],"num_samples":1,"max_tokens":2,"temperature":1,"top_k":-1,"top_p":1,"seed":0,"stop":null,"prompt_logprobs":false,"prompt_logprobs_last_n":null,"topk_prompt":0,"topk_sample":0}',
        )
        await node.drain("samp-1")
        self.assertTrue((await bus.wait_for_reply(sreq, 1)).ok)
        self.assertEqual(node.workers["p1"].resident, ())

        # Draining the node after controller restart snapshots the dirty trainer and publishes `snap`.
        node.begin_drain()
        async with asyncio.timeout(5):
          while node.workers or node.busy():
            await node.tick()
            await asyncio.sleep(0.005)
        info = bus.get_run(rid)
        self.assertIsNone(info.get("owner"))
        self.assertEqual(info["status"], RunStatus.SAVED)
        self.assertIsNotNone(info.get("snap"))
      finally:
        await node.aclose()
        await bus.aclose()


if __name__ == "__main__":
  unittest.main()
