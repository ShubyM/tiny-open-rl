import asyncio
import json
import sqlite3
import tempfile
import time
import unittest
from unittest import mock

import msgspec

from test.helpers import BusTest, new_id, serve_control
from tinyrl.control.accounting import Charge
from tinyrl.control.controller import Controller
from tinyrl.control.db import SCHEMA_VERSION
from tinyrl.control.node_daemon import ControlClient
from tinyrl.types import (
  Checkpoint,
  ControlMsg,
  Done,
  Effect,
  Ev,
  NodeMsg,
  Op,
  QueueFullError,
  Reply,
  RequestGoneError,
  Role,
  Run,
  RunStatus,
  TaskKind,
  WorkerInfo,
  WorkerSpec,
)


def ok(**body) -> Reply:
  return Reply(True, "application/json", json.dumps(body).encode())


class TestLedger(BusTest):
  def new_run(self, session: str = "", role: Role = Role.TRAINER, account: str = "") -> str:
    run_id = new_id()
    self.bus.create_run(Run(id=run_id, base="base", rank=8, session=session, role=role, account=account))
    return run_id

  def take_one(self, node: str = "n1", boot: str = "b1", gpus: int = 1):
    cmsg = self.bus.handle(NodeMsg("hello", node, boot, gpus=gpus))
    [t] = [t for t in cmsg.tasks if t.kind in (TaskKind.TRAIN, TaskKind.SAMPLE)]
    return t

  def done(
    self, node: str, boot: str, task_id: str, reply: Reply, charge=None, *, mutated=False, loaded=True, checkpoint=None
  ):
    eff = Effect(loaded=loaded, mutated=mutated, checkpoint=checkpoint)
    return self.bus.handle(NodeMsg("events", node, boot, done=(Done(task_id, True, reply, charge, eff),)))

  async def test_a_command_goes_through_take_and_finish_and_its_reply_comes_back(self):
    run_id = self.new_run(account="alice")
    request_id = self.bus.submit_command(run_id, Op.OPTIM_STEP, b'{"lr": 1}', cost=42)

    t = self.take_one()
    cmd = t.cmd
    self.assertEqual(
      (cmd.run, cmd.op, cmd.body, cmd.request_id, cmd.cost), (run_id, Op.OPTIM_STEP, b'{"lr": 1}', request_id, 42)
    )
    self.assertIsNone(await self.bus.wait_for_reply(request_id, 0))

    charge = Charge(started_ms=int(cmd.entry_id.split("-")[0]) + 12, busy_ms=50, gpu_ms=100.5)
    self.done("n1", "b1", t.id, ok(type="optim_step"), charge, mutated=True)
    self.assertEqual(await self.bus.wait_for_reply(request_id, 0), ok(type="optim_step"))

    # Repeated Done is idempotent and does not double-charge.
    self.done("n1", "b1", t.id, ok(type="optim_step"), charge, mutated=True)

    run = self.bus.get_run(run_id)
    self.assertEqual(
      (run["base"], run["rank"], run["steps"], run["tokens"], run["status"]), ("base", 8, 1, 42, RunStatus.LIVE)
    )
    self.assertEqual((run["wait_ms"], run["busy_ms"]), (12, 50))
    self.assertEqual(self.bus.service["alice"], 100.5)

  async def test_a_reply_wakes_every_waiter_and_reading_it_never_consumes_it(self):
    run_id = self.new_run()
    request_id = self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    t = self.take_one()

    waiters = [asyncio.create_task(self.bus.wait_for_reply(request_id, 5)) for _ in range(3)]
    await asyncio.sleep(0.05)
    start = time.perf_counter()
    self.done("n1", "b1", t.id, ok(n=1))

    self.assertEqual(await asyncio.gather(*waiters), [ok(n=1)] * 3)
    self.assertLess(time.perf_counter() - start, 0.5)
    self.assertEqual(await self.bus.wait_for_reply(request_id, 0), ok(n=1))

  async def test_hundreds_of_requests_wait_for_replies_at_once(self):
    got = await asyncio.gather(*[self.bus.wait_for_reply(f"req-{i}", 0.2) for i in range(300)])
    self.assertEqual(got, [None] * 300)

  async def test_waiting_for_a_reply_outlasts_a_delay(self):
    run_id = self.new_run()
    request_id = self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    t = self.take_one()

    async def answer_late():
      await asyncio.sleep(0.2)
      self.done("n1", "b1", t.id, ok(n=1))

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
    self.assertEqual(self.bus.runs[run_id].queue, {})

  async def test_delete_removes_owned_checkpoints_and_keeps_saved_weights(self):
    run = self.new_run()
    owned, saved = [f"tinker://{run}/{path}" for path in ("sampler_weights/session", "weights/save")]
    for uri in (owned, saved):
      self.seed_checkpoint(uri, 1, ttl=60, owned=uri == owned)
    request = self.bus.submit_command(run, Op.DELETE_MODEL, b"{}")
    t = self.take_one()
    self.done("n1", "b1", t.id, ok(), loaded=False)
    self.assertEqual(await self.bus.wait_for_reply(request, 5), ok())
    self.assertEqual(self.bus.runs[run].queue, {})
    self.assertIsNone(self.bus.get_run(run).get("snapshot"))
    self.assertIsNone(self.bus.get_run(run).get("owner"))
    self.assertFalse(self.bus.runs[run].live)
    self.assertIsNone(self.bus.get_checkpoint(owned))
    self.assertIsNotNone(self.bus.get_checkpoint(saved))

  async def test_lost_hello_reply_redelivers_same_command_unless_held(self):
    run_id = self.new_run()
    request_ids = [self.bus.submit_command(run_id, Op.OPTIM_STEP, b"") for _ in range(3)]
    self.assertEqual(self.bus.queued_and_in_flight(run_id), {"live": True, "queued": 3, "in_flight": []})

    t1 = self.take_one()
    t1_retry = self.take_one()
    self.assertEqual(t1, t1_retry)
    wi = WorkerInfo(t1.worker, t1.spec, t1.gpus)
    held = self.bus.handle(NodeMsg("hello", "n1", "b1", gpus=1, workers=(wi,), active=(t1.id,)))
    self.assertEqual(held.tasks, ())
    queue = self.bus.queued_and_in_flight(run_id)
    self.assertEqual(queue["queued"], 2)
    self.assertEqual([c["request_id"] for c in queue["in_flight"]], [request_ids[0]])

    next_msg = self.done("n1", "b1", t1.id, ok(), mutated=True)
    [t2] = [t for t in next_msg.tasks if t.kind == TaskKind.TRAIN]
    self.assertEqual(t2.cmd.request_id, request_ids[1])

  async def test_expired_dedupe_keys_are_swept_less_often_than_the_tick(self):
    self.bus.maintain()
    with self.bus.db:
      self.bus.db.execute("INSERT INTO dedup VALUES ('run', 1, 'run/1-1', 0.0)")
    self.bus.maintain()
    self.assertIsNotNone(self.bus.db.execute("SELECT 1 FROM dedup WHERE run_id='run'").fetchone())
    self.bus.gc_due = 0.0
    self.bus.maintain()
    self.assertIsNone(self.bus.db.execute("SELECT 1 FROM dedup WHERE run_id='run'").fetchone())

  async def test_delete_run_cancels_inflight_and_late_finish_is_ignored(self):
    run_id = self.new_run()
    req = self.bus.submit_command(run_id, Op.FORWARD_BACKWARD, b"")
    t = self.take_one()
    self.bus.delete_run(run_id)
    self.assertFalse((await self.bus.wait_for_reply(req, 0)).ok)
    self.assertFalse(self.bus.finish_train(req, (t.worker, "n1", "b1"), ok(), mutated=True))

  async def test_sample_finish_ignores_trainer_checkpoint_effects(self):
    sid = self.new_run(role=Role.SAMPLER)
    self.bus.submit_command(sid, Op.SAMPLE, b"{}")
    t = self.take_one()
    saved = Checkpoint(f"tinker://{sid}/weights/bad", "v1", 1, 1.0)
    self.done("n1", "b1", t.id, ok(), checkpoint=saved)
    self.assertFalse(self.bus.db.conn.in_transaction)
    self.assertIsNone(self.bus.get_checkpoint(saved.uri))

  async def test_reaped_boot_id_is_fenced_and_gets_reset_without_overwriting_checkpoint(self):
    run = self.new_run()
    first = self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
    t1 = self.take_one("n1", "b1", 1)

    self.bus.nodes["n1"].until = 0.0
    self.bus.maintain()
    failed = await self.bus.wait_for_reply(first, 0)
    self.assertFalse(failed.ok)
    self.assertIn("save_weights", json.loads(failed.body)["error"])
    self.assertEqual(self.bus.get_run(run)["failed"], 1)

    second = self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
    t2 = self.take_one("n2", "b2", 1)
    self.assertEqual(t2.cmd.request_id, second)
    uri = f"tinker://{run}/weights/step"
    self.done("n2", "b2", t2.id, ok(), checkpoint=Checkpoint(uri, "new", 20, 1.0))

    stale = Checkpoint(uri, "old", 10, 1.0, expires_at=61.0)
    res = self.done("n1", "b1", t1.id, ok(), checkpoint=stale)
    self.assertTrue(res.reset)
    self.assertEqual(await self.bus.wait_for_reply(first, 0), failed)
    self.assertEqual(self.bus.checkpoint_version(uri), "new")
    self.assertIsNone(self.bus.get_checkpoint(uri).expires_at)

  async def test_detach_saves_parked_snapshot_and_mutating_step_clears_it(self):
    run_id = self.new_run()
    self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    t1 = self.take_one("n1", "b1", 1)
    self.done("n1", "b1", t1.id, ok(), mutated=True)

    drain_msg = self.bus.handle(NodeMsg("events", "n1", "b1", draining=True))
    [save_t] = drain_msg.tasks
    self.assertEqual(save_t.kind, TaskKind.SAVE)
    snap_ver = "a" * 32
    self.bus.handle(NodeMsg("events", "n1", "b1", draining=True, done=(Done(save_t.id, saved=snap_ver),)))
    info = self.bus.get_run(run_id)
    self.assertIsNone(info.get("owner"))
    self.assertEqual(info["snapshot"], snap_ver)
    self.assertEqual(info["status"], RunStatus.SAVED)
    self.assertIn(snap_ver, self.bus.live_versions())
    self.assertEqual(self.bus.list_checkpoints(run_id), [])

    # Read-only SAVE_WEIGHTS on n2 keeps parked snap; mutating OPTIM_STEP clears it.
    self.bus.submit_command(run_id, Op.SAVE_WEIGHTS, b"{}")
    ro = self.take_one("n2", "b2", 1)
    self.assertEqual(ro.snapshot, snap_ver)
    self.done("n2", "b2", ro.id, ok(), mutated=False)
    self.assertIn(snap_ver, self.bus.live_versions())

    req3 = self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    [mut] = self.bus.nodes["n2"].drain_push().tasks
    self.assertEqual(mut.cmd.request_id, req3)
    self.done("n2", "b2", mut.id, ok(), mutated=True)
    self.assertIsNone(self.bus.get_run(run_id).get("snapshot"))
    self.assertNotIn(snap_ver, self.bus.live_versions())

  async def test_competing_sampler_replicas_answer_and_charge_exactly_once(self):
    sid = self.new_run(role=Role.SAMPLER, account="bob")
    req = self.bus.submit_command(sid, Op.SAMPLE, b"{}", cost=100)
    t = self.take_one("n1", "b1", 1)
    self.assertFalse(
      self.bus.finish_sample(req, ("other", "n1", "b1"), ok(replica=-1), Charge(busy_ms=99, gpu_ms=99), t.attempt)
    )
    results = [
      self.bus.finish_sample(req, (t.worker, "n1", "b1"), ok(replica=i), Charge(busy_ms=40, gpu_ms=40.5 + i), t.attempt)
      for i in range(5)
    ]
    self.assertEqual(sum(results), 1)
    winner = results.index(True)
    self.assertEqual(await self.bus.wait_for_reply(req, 0), ok(replica=winner))
    run = self.bus.get_run(sid)
    self.assertEqual((run["cmds"], run["tokens"]), (1, 100))
    self.assertEqual(self.bus.service["bob"], 40.5 + winner)

    self.bus.samples.replies.pop(req, None)
    self.assertFalse(
      self.bus.finish_sample(req, (t.worker, "n1", "b1"), ok(late=True), Charge(busy_ms=100, gpu_ms=100), t.attempt)
    )
    with self.assertRaises(RequestGoneError):
      await self.bus.wait_for_reply(req, 0)
    self.assertEqual(self.bus.service["bob"], 40.5 + winner)

  async def test_sampling_backpressure_and_unknown_promise_raise_typed_errors(self):
    sid = self.new_run(role=Role.SAMPLER)
    with mock.patch("tinyrl.control.ledger.SAMPLE_QUEUE_BYTES", 16):
      self.bus.submit_command(sid, Op.SAMPLE, b"0123456789")
      with self.assertRaises(QueueFullError):
        self.bus.submit_command(sid, Op.SAMPLE, b"0123456789")
    with self.assertRaises(RequestGoneError):
      await self.bus.wait_for_reply("samp-missing/1-1", 0)

  async def test_a_sample_in_flight_is_reclaimed_only_from_a_dead_node(self):
    session = self.new_run(role=Role.SAMPLER)
    requests = [self.bus.submit_command(session, Op.SAMPLE, b"{}") for _ in range(2)]
    t1 = [t for t in self.bus.handle(NodeMsg("hello", "n1", "b1", gpus=1)).tasks if t.kind == TaskKind.SAMPLE]
    self.assertEqual([t.cmd.request_id for t in t1], requests)
    t_empty = [t for t in self.bus.handle(NodeMsg("hello", "n2", "b2", gpus=1)).tasks if t.kind == TaskKind.SAMPLE]
    self.assertEqual(t_empty, [])
    self.bus.nodes["n1"].until = 0.0
    t2 = [t for t in self.bus.handle(NodeMsg("hello", "n2", "b2", gpus=1)).tasks if t.kind == TaskKind.SAMPLE]
    self.assertEqual([t.cmd.request_id for t in t2], requests)

  async def test_a_dead_node_daemon_stops_counting(self):
    self.bus.handle(NodeMsg("hello", "node-a", "ba", gpus=2))
    self.bus.handle(NodeMsg("hello", "node-b", "bb", gpus=8))
    self.bus.nodes["node-b"].until = 0.0
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
    self.assertEqual(self.bus.db.execute("SELECT active FROM sessions WHERE id='s'").fetchone()[0], 1)
    with self.bus.db:
      self.bus.db.execute("DELETE FROM commands WHERE run_id=?", (run_id,))
    self.bus.runs[run_id].queue.clear()
    self.bus.end_session("s")
    self.assertEqual(self.bus.queued_and_in_flight(run_id)["queued"], 1)
    self.bus.delete_run(run_id)
    self.bus.end_session("s")
    self.assertEqual(self.bus.db.execute("SELECT active FROM sessions WHERE id='s'").fetchone()[0], 0)

  async def test_a_deleted_run_does_not_hide_another_runs_command(self):
    run_id, gone = self.new_run(role=Role.SAMPLER), self.new_run(role=Role.SAMPLER)
    request = self.bus.submit_command(run_id, Op.SAMPLE, b"{}")
    self.bus.delete_run(gone)
    t = self.take_one()
    self.assertEqual((t.cmd.run, t.cmd.request_id), (run_id, request))

  async def test_deleting_run_with_inflight_cmd_keeps_sibling_lora_run_live(self):
    keep, drop = self.new_run(), self.new_run()
    self.bus.submit_command(keep, Op.OPTIM_STEP, b"{}")
    t1 = self.take_one("n1", "b1", 1)
    self.done("n1", "b1", t1.id, ok(), mutated=True)
    self.assertEqual(self.bus.get_run(keep)["status"], RunStatus.LIVE)
    self.bus.submit_command(drop, Op.FORWARD_BACKWARD, b"{}")
    [taken] = self.bus.nodes["n1"].drain_push().tasks
    self.assertEqual((taken.worker, taken.cmd.run), (t1.worker, drop))
    self.bus.delete_run(drop)
    self.assertEqual(self.bus.get_run(keep)["status"], RunStatus.LIVE)
    self.assertEqual(self.bus.get_run(keep)["owner"], [t1.worker, "n1", "b1"])

  async def test_unstaged_samples_wait_while_base_samples_can_run(self):
    base_sid = self.new_run(role=Role.SAMPLER)
    ckpt_uri = "tinker://tr/sampler_weights/s1"
    self.seed_checkpoint(ckpt_uri, 1, version="badver")
    lora_sid = f"samp-{new_id()}"
    self.bus.create_run(Run(lora_sid, "base", 1, model_path=ckpt_uri, role=Role.SAMPLER))
    base_req = self.bus.submit_command(base_sid, Op.SAMPLE, b"{}")
    r1 = self.bus.submit_command(lora_sid, Op.SAMPLE, b"{}")
    r2 = self.bus.submit_command(lora_sid, Op.SAMPLE, b"{}")
    cmsg = self.bus.handle(NodeMsg("hello", "n1", "b1", gpus=1))
    self.assertIn("badver", set(cmsg.stage))
    [s0] = [t for t in cmsg.tasks if t.kind == TaskKind.SAMPLE]
    self.assertEqual(s0.cmd.request_id, base_req)
    self.done("n1", "b1", s0.id, ok())
    for req in (r1, r2):
      self.assertIsNone(await self.bus.wait_for_reply(req, 0))
    cmsg2 = self.bus.handle(NodeMsg("events", "n1", "b1", staged=("badver",)))
    self.assertEqual([t.cmd.request_id for t in cmsg2.tasks if t.kind == TaskKind.SAMPLE], [r1, r2])

  async def test_sampler_billing_persists_across_maintain_and_reopen(self):
    self.bus.maintain()
    sid = self.new_run(role=Role.SAMPLER, account="carol")
    self.bus.submit_command(sid, Op.SAMPLE, b"{}")
    t = self.take_one("n1", "b1", 1)
    self.done("n1", "b1", t.id, ok(), Charge(busy_ms=10, gpu_ms=77.5))
    self.bus.maintain()
    reopened = Controller(self.db_path)
    try:
      self.assertEqual(reopened.service["carol"], 77.5)
    finally:
      await reopened.aclose()

  async def test_future_schema_version_is_rejected_without_modification(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = f"{tmp}/future.db"
      conn = sqlite3.connect(path)
      conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
      conn.close()
      with self.assertRaisesRegex(RuntimeError, "newer"):
        Controller(path)

  async def test_sample_reply_lru_evicts_read_replies_before_unread(self):
    sid = self.new_run(role=Role.SAMPLER)
    reqs = [self.bus.submit_command(sid, Op.SAMPLE, b"{}") for _ in range(3)]
    tasks = [t for t in self.bus.handle(NodeMsg("hello", "n1", "b1", gpus=1)).tasks if t.kind == TaskKind.SAMPLE]
    self.assertEqual([t.cmd.request_id for t in tasks], reqs)
    big = Reply(True, "application/json", b"x" * 60)
    with mock.patch("tinyrl.control.ledger.SAMPLE_REPLY_BYTES", 120):
      self.done("n1", "b1", tasks[0].id, big)
      self.done("n1", "b1", tasks[1].id, big)
      self.assertEqual(await self.bus.wait_for_reply(reqs[1], 0), big)
      self.done("n1", "b1", tasks[2].id, big)
      self.assertEqual(await self.bus.wait_for_reply(reqs[0], 0), big)
      self.assertEqual(await self.bus.wait_for_reply(reqs[2], 0), big)
      with self.assertRaises(RequestGoneError):
        await self.bus.wait_for_reply(reqs[1], 0)


class TestControlRestart(unittest.IsolatedAsyncioTestCase):
  async def test_a_node_that_never_returns_after_a_restart_is_lost_once(self):
    with tempfile.TemporaryDirectory() as tmp:
      bus = Controller(f"{tmp}/tinyrl.db")
      for name in ("a", "b"):
        bus.create_run(Run(name, f"base-{name}", 0))
        bus.submit_command(name, Op.OPTIM_STEP, b"")
      cmsg = bus.handle(NodeMsg("hello", "n1", "b1", gpus=2))
      self.assertEqual(len([t for t in cmsg.tasks if t.kind == TaskKind.TRAIN]), 2)
      await bus.aclose()
      with mock.patch("tinyrl.control.controller.LEASE_MS", 0):
        bus = Controller(f"{tmp}/tinyrl.db")
      bus.maintain()
      self.assertEqual([e.kind for e in bus.read_events()].count(Ev.NODE_LOST), 1)
      await bus.aclose()
      # Reaped boot_id stays fenced even across another controller restart.
      bus = Controller(f"{tmp}/tinyrl.db")
      self.assertTrue(bus.handle(NodeMsg("hello", "n1", "b1", gpus=2)).reset)
      await bus.aclose()

  async def test_ws_client_survives_control_server_restart_and_preserves_durable_state(self):
    with tempfile.TemporaryDirectory() as tmp:
      db_path = f"{tmp}/tinyrl.db"
      bus = Controller(db_path)
      runner, url = await serve_control(bus, host="127.0.0.1", port=0)
      port = int(url.rsplit(":", 1)[1])
      client = ControlClient(url)
      try:
        bus.heartbeat_session("s1", {"sdk": "0.39"})
        bus.create_run(Run("tr-1", "base", 4, session="s1", account="alice"))
        req = bus.submit_command("tr-1", Op.SAVE_WEIGHTS, b"\x00\xff", cost=5)
        async with client.connect() as ws:
          await ws.send_bytes(msgspec.msgpack.encode(NodeMsg("hello", "node-a", "boot-a", gpus=1)))
          init = msgspec.msgpack.decode(await ws.receive_bytes(), type=ControlMsg)
        [t1] = init.tasks
        self.assertEqual((t1.gpus, t1.cmd.body), ((0,), b"\x00\xff"))
        ckpt = Checkpoint("tinker://tr-1/weights/v1", "ver1", 99, 1.0)

        await runner.cleanup()
        await bus.aclose()
        bus = Controller(db_path)
        runner, _ = await serve_control(bus, host="127.0.0.1", port=port)

        async with client.connect() as ws:
          await ws.send_bytes(msgspec.msgpack.encode(NodeMsg("hello", "node-a", "boot-a", gpus=1)))
          replay = msgspec.msgpack.decode(await ws.receive_bytes(), type=ControlMsg)
          self.assertEqual(replay.tasks, (t1,))
          wi = WorkerInfo(t1.worker, WorkerSpec(lora=True, base="base"), (0,))
          await ws.send_bytes(
            msgspec.msgpack.encode(NodeMsg("hello", "node-a", "boot-a", gpus=1, workers=(wi,), active=(t1.id,)))
          )
          held = msgspec.msgpack.decode(await ws.receive_bytes(), type=ControlMsg)
          self.assertEqual(held.tasks, ())
          done = Done(t1.id, True, ok(done=1), effect=Effect(loaded=True, mutated=True, checkpoint=ckpt))
          await ws.send_bytes(msgspec.msgpack.encode(NodeMsg("events", "node-a", "boot-a", done=(done,))))
          acked = msgspec.msgpack.decode(await ws.receive_bytes(), type=ControlMsg)
          self.assertEqual(acked.ack, (t1.id,))

        self.assertEqual(await bus.wait_for_reply(req, 0), ok(done=1))
        self.assertEqual(bus.get_session("s1")["metadata"], {"sdk": "0.39"})
        self.assertEqual(bus.checkpoint_version("tinker://tr-1/weights/v1"), "ver1")
        with self.assertRaises(RequestGoneError):
          await bus.wait_for_reply("samp-lost/1-1", 0)
      finally:
        await runner.cleanup()
        await bus.aclose()

  async def test_reopen_redelivers_exact_trainer_claim_and_preserves_drain_ownership(self):
    from test.execution import TestNode
    from tinyrl.storage.store import Store

    with tempfile.TemporaryDirectory() as tmp:
      db_path = f"{tmp}/tinyrl.db"
      store = Store(root=f"{tmp}/ckpt")
      bus = Controller(db_path)
      node = TestNode(bus, gpus=1, node="n1", store=store)
      try:
        rid = new_id()
        bus.create_run(Run(rid, "tiny", 4, account="alice"))
        bus.submit_command(rid, Op.CREATE_MODEL, b'{"base_model":"tiny","rank":4,"lora":{}}')
        await node.drain(rid)
        [tr_wid] = list(node.workers)
        self.assertEqual(node.workers[tr_wid].resident, (0,))

        await bus.aclose()
        bus = Controller(db_path)
        node.control = bus
        node.connected = False
        bus.create_run(Run("samp-1", "other", 4, role=Role.SAMPLER, account="bob"))
        sreq = bus.submit_command(
          "samp-1",
          Op.SAMPLE,
          b'{"prompt":[1],"num_samples":1,"max_tokens":2,"temperature":1,"top_k":-1,"top_p":1,"seed":0,"stop":null,"prompt_logprobs":false,"prompt_logprobs_last_n":null,"topk_prompt":0,"topk_sample":0}',
        )
        await node.drain("samp-1")
        self.assertTrue((await bus.wait_for_reply(sreq, 1)).ok)
        self.assertEqual(node.workers[tr_wid].resident, ())

        node.begin_drain()
        async with asyncio.timeout(5):
          while node.workers or node.busy():
            await node.tick()
            await asyncio.sleep(0.005)
        info = bus.get_run(rid)
        self.assertIsNone(info.get("owner"))
        self.assertEqual(info["status"], RunStatus.SAVED)
        self.assertIsNotNone(info.get("snapshot"))
      finally:
        await node.aclose()
        await bus.aclose()


if __name__ == "__main__":
  unittest.main()
