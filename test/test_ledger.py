import asyncio
import json
import sqlite3
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import msgspec

from test.helpers import BusTest, new_id, serve_control
from tinyrl.control.accounting import Charge
from tinyrl.control.controller import Controller
from tinyrl.control.db import DB, SCHEMA_VERSION
from tinyrl.control.node_daemon import ControlClient, NodeDaemon, WorkerHandle
from tinyrl.storage.store import Store
from tinyrl.types import (
  MAX_SEQS,
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

    self.bus.sample_replies.pop(req, None)
    self.assertFalse(
      self.bus.finish_sample(req, (t.worker, "n1", "b1"), ok(late=True), Charge(busy_ms=100, gpu_ms=100), t.attempt)
    )
    with self.assertRaises(RequestGoneError):
      await self.bus.wait_for_reply(req, 0)
    self.assertEqual(self.bus.service["bob"], 40.5 + winner)

  async def test_sampling_backpressure_and_unknown_promise_raise_typed_errors(self):
    sid = self.new_run(role=Role.SAMPLER)
    with mock.patch("tinyrl.control.controller.SAMPLE_QUEUE_BYTES", 16):
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
    with mock.patch("tinyrl.control.controller.SAMPLE_REPLY_BYTES", 120):
      self.done("n1", "b1", tasks[0].id, big)
      self.done("n1", "b1", tasks[1].id, big)
      self.assertEqual(await self.bus.wait_for_reply(reqs[1], 0), big)
      self.done("n1", "b1", tasks[2].id, big)
      self.assertEqual(await self.bus.wait_for_reply(reqs[0], 0), big)
      self.assertEqual(await self.bus.wait_for_reply(reqs[2], 0), big)
      with self.assertRaises(RequestGoneError):
        await self.bus.wait_for_reply(reqs[1], 0)

  async def test_sql_prefix_delete_does_not_match_like_wildcards(self):
    self.bus.create_run(Run("run_1", "base", 4))
    self.bus.create_run(Run("runA1", "base", 4))
    self.bus.put_checkpoint(Checkpoint("tinker://run_1/sampler_weights/s0", "v1", 10, 1.0, owned=True))
    self.bus.put_checkpoint(Checkpoint("tinker://runA1/sampler_weights/s0", "v2", 10, 1.0, owned=True))
    self.bus.submit_command("run_1", Op.DELETE_MODEL, b"{}")
    [t] = self.bus.handle(NodeMsg("hello", "n1", "b1", gpus=1)).tasks
    self.bus.handle(NodeMsg("events", "n1", "b1", done=(Done(t.id, True, ok()),)))
    self.assertIsNone(self.bus.get_checkpoint("tinker://run_1/sampler_weights/s0"))
    self.assertIsNotNone(self.bus.get_checkpoint("tinker://runA1/sampler_weights/s0"))

  async def test_delete_run_keeps_inflight_trainer_tracked_and_replies_queued_cmds(self):
    self.bus.create_run(Run("r1", "base", 0))
    req1 = self.bus.submit_command("r1", Op.CREATE_MODEL, b"{}")
    req2 = self.bus.submit_command("r1", Op.OPTIM_STEP, b"{}")
    [tr_task] = self.bus.handle(NodeMsg("hello", "n1", "b1", gpus=1)).tasks
    self.bus.delete_run("r1")
    self.assertIn(tr_task.id, self.bus.nodes["n1"].tasks)
    self.assertFalse((await self.bus.wait_for_reply(req1, 0)).ok)
    self.assertFalse((await self.bus.wait_for_reply(req2, 0)).ok)
    after = self.bus.handle(
      NodeMsg("events", "n1", "b1", done=(Done(tr_task.id, True, ok(), effect=Effect(True, True)),))
    )
    self.assertEqual([t.kind for t in after.tasks], [TaskKind.STOP])

  async def test_same_frame_cancel_and_hello_stale_active_cancel(self):
    self.bus.create_run(Run("s1", "base", 4, role=Role.SAMPLER))
    self.bus.handle(NodeMsg("hello", "n1", "b1", gpus=1))
    self.bus.submit_command("s1", Op.SAMPLE, b"{}")
    self.bus.delete_run("s1")
    self.assertEqual(self.bus.nodes["n1"].drain_push().tasks, ())
    hello_reply = self.bus.handle(NodeMsg("hello", "n1", "b1", gpus=1, active=("stale-task-id",)))
    self.assertIn("stale-task-id", hello_reply.cancel)

  async def test_waiting_unowned_trainer_reuses_unstarted_worker_without_churn(self):
    self.bus.create_run(Run("r1", "base-a", 0))
    self.bus.submit_command("r1", Op.CREATE_MODEL, b"{}")
    self.bus.handle(NodeMsg("hello", "n1", "b1", gpus=1))
    self.bus.create_run(Run("r2", "base-b", 0))
    self.bus.submit_command("r2", Op.CREATE_MODEL, b"{}")
    for _ in range(5):
      self.bus.handle(NodeMsg("events", "n1", "b1"))
    self.assertEqual(len([e for e in self.bus.read_events() if e.kind == Ev.PLACED]), 2)

  async def test_empty_live_versions_still_triggers_sweep(self):
    cmsg = self.bus.handle(NodeMsg("hello", "n1", "b1", gpus=1, want_sweep=True))
    self.assertEqual(cmsg.sweep, ())
    with tempfile.TemporaryDirectory() as tmp:
      store = Store(root=tmp)
      store.put_blob(b"orphan", version="a" * 32)
      daemon = NodeDaemon(self.bus, gpus=1, node="n1", gpu_mib=0, store=store)
      self.addAsyncCleanup(daemon.close)
      with mock.patch("tinyrl.control.node_daemon.SWEEP_GRACE_S", 0):
        daemon.apply(ControlMsg(sweep=()))
        await daemon.sweep_task
      self.assertEqual(store.versions(), set())

  async def test_failed_saved_restore_clears_owner_so_empty_worker_evicts(self):
    self.bus.create_run(Run("r1", "base", 4))
    rs = self.bus.runs["r1"]
    rs.status, rs.snapshot = RunStatus.SAVED, "missing-snap"
    self.bus.submit_command("r1", Op.OPTIM_STEP, b"{}")
    [t] = self.bus.handle(NodeMsg("hello", "n1", "b1", gpus=1)).tasks
    err = Reply.error(ValueError("no checkpoint for version missing-snap"))
    cmsg = self.bus.handle(NodeMsg("events", "n1", "b1", done=(Done(t.id, True, err, effect=Effect(False, False)),)))
    self.assertIsNone(self.bus.get_run("r1").get("owner"))
    self.assertEqual(self.bus.get_run("r1")["status"], RunStatus.SAVED)
    self.assertEqual([x.kind for x in cmsg.tasks], [TaskKind.STOP])

  async def test_custom_swappable_scheduler_drives_train_and_sample(self):
    def fifo_scheduler(ctrl: Controller, now: float) -> list[float]:
      [node] = ctrl.healthy_nodes().values()
      if node.tasks:
        return []
      for run_id, state in ctrl.runs.items():
        if ctrl.train_head(run_id) is not None:
          wid = ctrl.spawn_worker(node.id, state.run.spec)
          ctrl.dispatch_train(node, node.workers[wid], (0,), run_id, now)
          return []
        if state.live and state.run.role == Role.SAMPLER and (heads := ctrl.sample_heads(state.run.spec, node.staged)):
          wid = ctrl.spawn_worker(node.id, state.run.spec)
          ctrl.dispatch_samples(node, node.workers[wid], (0,), [heads[run_id].cmd], now)
          return []
      return []

    ctrl = Controller(":memory:", scheduler=fifo_scheduler)
    self.addAsyncCleanup(ctrl.aclose)
    ctrl.create_run(Run("tr", "base", 4))
    req_tr = ctrl.submit_command("tr", Op.OPTIM_STEP, b"{}")
    [t_tr] = ctrl.handle(NodeMsg("hello", "n1", "b1", gpus=1)).tasks
    self.assertEqual((t_tr.kind, t_tr.run_id), (TaskKind.TRAIN, "tr"))
    ctrl.handle(NodeMsg("events", "n1", "b1", done=(Done(t_tr.id, True, ok(), effect=Effect(True, True)),)))
    self.assertTrue((await ctrl.wait_for_reply(req_tr, 0)).ok)

    ctrl.create_run(Run("samp", "base", 4, role=Role.SAMPLER))
    req_s = ctrl.submit_command("samp", Op.SAMPLE, b"{}")
    [t_s] = ctrl.nodes["n1"].drain_push().tasks
    self.assertEqual((t_s.kind, t_s.run_id), (TaskKind.SAMPLE, "samp"))
    ctrl.handle(NodeMsg("events", "n1", "b1", done=(Done(t_s.id, True, ok()),)))
    self.assertTrue((await ctrl.wait_for_reply(req_s, 0)).ok)

    # Kernel drain mechanism saves dirty trainer runs even with a trivial scheduler.
    draining = ctrl.handle(NodeMsg("events", "n1", "b1", draining=True))
    [t_save] = [t for t in draining.tasks if t.kind == TaskKind.SAVE]
    self.assertEqual(t_save.run_id, "tr")
    after = ctrl.handle(NodeMsg("events", "n1", "b1", draining=True, done=(Done(t_save.id, True, saved="snap-tr"),)))
    self.assertEqual([t.kind for t in after.tasks], [TaskKind.STOP])
    self.assertEqual(ctrl.get_run("tr")["snapshot"], "snap-tr")


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

  def test_legacy_kv_database_fails_closed(self):
    with tempfile.TemporaryDirectory() as tmp:
      for name, insert in (("legacy.db", True), ("empty.db", False)):
        path = f"{tmp}/{name}"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE kv (ns TEXT, k TEXT, v BLOB, PRIMARY KEY (ns, k))")
        if insert:
          conn.execute("INSERT INTO kv VALUES ('runs', 'r1', X'01')")
        conn.commit()
        conn.close()
        if insert:
          with self.assertRaisesRegex(RuntimeError, "legacy KV"):
            DB(path)
        else:
          DB(path).close()

  async def test_restart_with_claimed_task_preserves_sibling_lora_run_on_drain(self):
    with tempfile.TemporaryDirectory() as tmp:
      db_path = f"{tmp}/tinyrl.db"
      ctrl = Controller(db_path)
      ctrl.create_run(Run("a", "base", 4))
      ctrl.create_run(Run("b", "base", 4))
      ctrl.submit_command("a", Op.OPTIM_STEP, b"{}")
      [t_a1] = ctrl.handle(NodeMsg("hello", "n1", "b1", gpus=1)).tasks
      ctrl.handle(NodeMsg("events", "n1", "b1", done=(Done(t_a1.id, True, ok(), effect=Effect(True, True)),)))
      ctrl.submit_command("b", Op.OPTIM_STEP, b"{}")
      [t_b1] = ctrl.nodes["n1"].drain_push().tasks
      ctrl.handle(NodeMsg("events", "n1", "b1", done=(Done(t_b1.id, True, ok(), effect=Effect(True, True)),)))
      ctrl.submit_command("a", Op.OPTIM_STEP, b"{}")
      [t_a2] = ctrl.nodes["n1"].drain_push().tasks
      wid = t_a1.worker
      await ctrl.aclose()

      ctrl = Controller(db_path)
      self.addAsyncCleanup(ctrl.aclose)
      wi = WorkerInfo(wid, WorkerSpec(lora=True, base="base"), (0,), loaded={"a": False, "b": False})
      ctrl.handle(NodeMsg("hello", "n1", "b1", gpus=1, workers=(wi,), active=(t_a2.id,)))
      ctrl.handle(NodeMsg("events", "n1", "b1", done=(Done(t_a2.id, True, ok(), effect=Effect(True, True)),)))
      saved_runs, cmsg = [], ctrl.handle(NodeMsg("events", "n1", "b1", draining=True))
      while cmsg.tasks and cmsg.tasks[0].kind == TaskKind.SAVE:
        st = cmsg.tasks[0]
        saved_runs.append(st.run_id)
        cmsg = ctrl.handle(
          NodeMsg("events", "n1", "b1", draining=True, done=(Done(st.id, True, saved=f"snap-{st.run_id}"),))
        )
      self.assertEqual(sorted(saved_runs), ["a", "b"])
      self.assertEqual([t.kind for t in cmsg.tasks], [TaskKind.STOP])
      self.assertEqual(ctrl.get_run("b")["snapshot"], "snap-b")


class TestClaimStorage(BusTest):
  async def test_sample_keeps_its_version_on_retry_and_reassignment(self):
    uri = "tinker://trainer/sampler_weights/latest"
    self.seed_checkpoint(uri, 1, version="first")
    self.bus.create_run(Run("s", "base", 8, role=Role.SAMPLER, model_path=uri))
    first = self.bus.submit_command("s", Op.SAMPLE, b"{}")
    staged = ("first", "second")

    [initial] = [
      t
      for t in self.bus.handle(NodeMsg("hello", "east", "boot1", gpus=1, staged=staged)).tasks
      if t.kind == TaskKind.SAMPLE
    ]
    self.seed_checkpoint(uri, 1, version="second")
    second = self.bus.submit_command("s", Op.SAMPLE, b"{}")
    [replay] = [
      t
      for t in self.bus.handle(NodeMsg("hello", "east", "boot1", gpus=1, staged=staged)).tasks
      if t.cmd.request_id == first
    ]
    self.assertEqual(replay, initial)
    self.assertIn("first", self.bus.live_versions())

    self.bus.nodes["east"].until = 0
    [retry] = [
      t
      for t in self.bus.handle(NodeMsg("hello", "west", "boot2", gpus=1, staged=staged)).tasks
      if t.cmd.request_id == first
    ]
    self.assertEqual((retry.cmd.request_id, retry.version), (first, "first"))
    self.bus.handle(NodeMsg("events", "west", "boot2", done=(Done(retry.id, True, ok()),)))
    self.assertNotIn("first", self.bus.live_versions())
    done_second = [t for t in self.bus.nodes["west"].tasks.values() if t.cmd and t.cmd.request_id == second]
    self.assertEqual([(t.cmd.request_id, t.version) for t in done_second], [(second, "second")])

  async def test_redelivered_wave_preserves_interleaved_run_order(self):
    for run in ("a", "b"):
      self.bus.create_run(Run(run, "base", 8, role=Role.SAMPLER, account="owner"))
      for _ in range(2):
        self.bus.submit_command(run, Op.SAMPLE, b"{}")
    initial = [t for t in self.bus.handle(NodeMsg("hello", "n", "boot", gpus=1)).tasks if t.kind == TaskKind.SAMPLE]
    self.assertEqual([t.cmd.run for t in initial], ["a", "b", "a", "b"])
    replay = [t for t in self.bus.handle(NodeMsg("hello", "n", "boot", gpus=1)).tasks if t.kind == TaskKind.SAMPLE]
    self.assertEqual(replay, initial)

  async def test_failed_finish_preserves_trainer_claim_and_bills_once_on_retry(self):
    self.bus.create_run(Run("r", "base", 8, account="a"))
    req = self.bus.submit_command("r", Op.OPTIM_STEP, b"{}")
    [task] = [t for t in self.bus.handle(NodeMsg("hello", "n", "b", gpus=1)).tasks if t.kind == TaskKind.TRAIN]
    self.bus.db.execute(
      """CREATE TEMP TRIGGER fail_run_write BEFORE INSERT ON runs
         BEGIN SELECT RAISE(ABORT, 'injected write failure'); END"""
    )
    done = Done(task.id, True, ok(), Charge(gpu_ms=5), Effect(loaded=True, mutated=True))
    try:
      with self.assertRaisesRegex(sqlite3.IntegrityError, "injected write failure"):
        self.bus.handle(NodeMsg("events", "n", "b", done=(done,)))
    finally:
      self.bus.db.execute("DROP TRIGGER fail_run_write")

    self.assertIsNone(await self.bus.wait_for_reply(req, 0))
    self.assertEqual([x["request_id"] for x in self.bus.queued_and_in_flight("r")["in_flight"]], [req])
    [replay] = [t for t in self.bus.handle(NodeMsg("hello", "n", "b", gpus=1)).tasks if t.kind == TaskKind.TRAIN]
    self.assertEqual(replay, task)
    ack = self.bus.handle(NodeMsg("events", "n", "b", done=(done,))).ack
    self.assertEqual(ack, (task.id,))
    self.assertEqual(await self.bus.wait_for_reply(req, 0), ok())
    self.assertEqual(self.bus.service["a"], 5)
    self.assertEqual(self.bus.get_run("r")["steps"], 1)


class TestDetachRetry(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.ctrl = Controller(":memory:")
    self.addAsyncCleanup(self.ctrl.aclose)
    store = mock.Mock()
    store.env.return_value = {}
    store.versions.return_value = set()
    self.node = NodeDaemon(self.ctrl, gpus=1, node="n", gpu_mib=0, store=store)
    self.model = SimpleNamespace(call=mock.AsyncMock(), close=mock.AsyncMock(), proc=None)
    spec = WorkerSpec(lora=True, base="base")
    self.worker = self.node.workers["w"] = WorkerHandle("w", spec, self.model, resident=(0,))
    self.worker.loaded = {"a": False, "b": False}
    for r in ("a", "b"):
      self.ctrl.create_run(Run(r, "base", 8))
      rs = self.ctrl.runs[r]
      rs.status, rs.owner = RunStatus.LIVE, ("w", "n", self.node.boot_id)
    await self.node.step_once()

  async def test_stop_saves_other_runs_and_retries_only_the_failed_snapshot(self):
    self.model.call.side_effect = [OSError("storage unavailable"), "snap-b", "snap-a"]
    self.node.begin_drain()
    await self.node.step_once()
    await asyncio.sleep(0.01)
    await self.node.step_once()
    await asyncio.sleep(0.01)
    await self.node.step_once()
    self.assertEqual(self.ctrl.get_run("b")["snapshot"], "snap-b")
    self.assertIsNone(self.ctrl.get_run("a").get("snapshot"))
    self.model.close.assert_not_awaited()

    await asyncio.sleep(0.06)
    await self.node.step_once()
    await asyncio.sleep(0.01)
    await self.node.step_once()
    await asyncio.sleep(0.01)
    await self.node.step_once()
    self.assertEqual([c.args[:2] for c in self.model.call.await_args_list], [("save_run", r) for r in "aba"])
    self.assertEqual(self.ctrl.get_run("a")["snapshot"], "snap-a")
    self.assertEqual(self.node.workers, {})
    self.model.close.assert_awaited_once()

  async def test_detach_keeps_source_until_controller_acknowledges_snapshot(self):
    self.model.call.return_value = "snap-b"
    self.ctrl.detach_run(self.ctrl.nodes["n"], self.ctrl.nodes["n"].workers["w"], "b", 0.0)
    cmsg = self.ctrl.nodes["n"].drain_push()
    self.assertEqual([t.kind for t in cmsg.tasks], [TaskKind.SAVE])
    self.node.apply(cmsg)
    await asyncio.sleep(0.01)
    self.assertIn("b", self.worker.loaded)
    self.assertEqual([c.args[0] for c in self.model.call.await_args_list], ["save_run"])

    await self.node.step_once()
    self.assertEqual(self.ctrl.get_run("b")["snapshot"], "snap-b")
    await asyncio.sleep(0.01)
    await self.node.step_once()
    self.model.call.assert_awaited_with("drop_run", "b")
    self.assertEqual(set(self.worker.loaded), {"a"})
    self.model.close.assert_not_awaited()

  async def test_permanent_save_error_never_publishes_a_snapshot_during_forced_drain(self):
    self.model.call.side_effect = OSError("storage unavailable")
    self.node.begin_drain()
    with mock.patch("tinyrl.control.node_daemon.TICK", 0.01), mock.patch("tinyrl.control.node_daemon.DRAIN_S", 0.15):
      await asyncio.wait_for(self.node.serve(), 2)
    self.assertGreater(self.model.call.await_count, 2)
    self.assertIsNone(self.ctrl.get_run("a").get("snapshot"))
    self.assertIsNone(self.ctrl.get_run("b").get("snapshot"))
    self.model.close.assert_awaited_once()

  async def test_scale_out_detach_respects_failed_saves_backoff(self):
    self.ctrl.submit_command("a", Op.OPTIM_STEP, b"{}")
    [t] = self.ctrl.nodes["n"].drain_push().tasks
    self.ctrl.submit_command("a", Op.OPTIM_STEP, b"{}")
    self.ctrl.submit_command("b", Op.OPTIM_STEP, b"{}")
    self.ctrl.handle(NodeMsg("hello", "n2", "b2", gpus=1))
    self.ctrl.handle(
      NodeMsg("events", "n", self.node.boot_id, done=(Done(t.id, True, ok(), effect=Effect(True, True)),))
    )
    save_tasks = [x for x in self.ctrl.nodes["n"].tasks.values() if x.kind == TaskKind.SAVE]
    self.assertEqual([x.run_id for x in save_tasks], ["b"])
    cmsg = self.ctrl.handle(NodeMsg("events", "n", self.node.boot_id, done=(Done(save_tasks[0].id, ok=False),)))
    self.assertEqual([x for x in cmsg.tasks if x.kind == TaskKind.SAVE and x.run_id == "b"], [])


class TestSampleFailover(unittest.IsolatedAsyncioTestCase):
  async def test_unstarted_claim_can_move_on_cache_loss_and_fences_all_earlier_attempts(self):
    ctrl = Controller(":memory:")
    self.addAsyncCleanup(ctrl.aclose)
    uri, version = "tinker://train/sampler_weights/latest", "v1"
    ctrl.put_checkpoint(Checkpoint(uri, version, 1))
    for node in ("east", "west"):
      ctrl.handle(NodeMsg("hello", node, node, gpus=1))
    ctrl.create_run(Run("sample", "base", 8, role=Role.SAMPLER, model_path=uri))
    ids = [ctrl.submit_command("sample", Op.SAMPLE, b"{}") for _ in range(MAX_SEQS + 1)]
    first = [
      t
      for t in ctrl.handle(NodeMsg("hello", "east", "east", gpus=1, staged=(version,))).tasks
      if t.kind == TaskKind.SAMPLE
    ]
    running, unseen = first[:2]
    self.assertEqual(running.cmd.request_id, ids[0])

    wi = WorkerInfo(running.worker, running.spec, running.gpus, running.gpus)
    evicted = ctrl.handle(NodeMsg("hello", "east", "east", gpus=1, workers=(wi,), active=(running.id,)))
    self.assertIn(unseen.id, evicted.cancel)
    west = [
      t
      for t in ctrl.handle(NodeMsg("hello", "west", "west", gpus=1, staged=(version,))).tasks
      if t.kind == TaskKind.SAMPLE
    ]
    west_claim = next(t for t in west if t.cmd.request_id == unseen.cmd.request_id)
    self.assertNotIn(running.cmd.request_id, [t.cmd.request_id for t in west])
    ctrl.handle(NodeMsg("events", "east", "east", done=(Done(running.id, True, ok()),)))
    self.assertEqual(await ctrl.wait_for_reply(running.cmd.request_id, 0), ok())
    self.assertFalse(
      ctrl.finish_sample(unseen.cmd.request_id, (unseen.worker, "east", "east"), ok(), None, unseen.attempt)
    )

    ctrl.handle(NodeMsg("hello", "west", "west", gpus=1))
    east = [
      t
      for t in ctrl.handle(NodeMsg("hello", "east", "east", gpus=1, workers=(wi,), staged=(version,))).tasks
      if t.kind == TaskKind.SAMPLE
    ]
    current = next(t for t in east if t.cmd.request_id == unseen.cmd.request_id)
    self.assertEqual((current.worker, current.version), (running.worker, version))
    self.assertLess(unseen.attempt, west_claim.attempt)
    self.assertLess(west_claim.attempt, current.attempt)
    self.assertFalse(ctrl.finish_sample(unseen.cmd.request_id, (current.worker, "east", "east"), ok(), None, 0))
    self.assertFalse(
      ctrl.finish_sample(unseen.cmd.request_id, (west_claim.worker, "west", "west"), ok(), None, west_claim.attempt)
    )
    self.assertTrue(
      ctrl.finish_sample(unseen.cmd.request_id, (current.worker, "east", "east"), ok(), None, current.attempt)
    )
    self.assertEqual(await ctrl.wait_for_reply(unseen.cmd.request_id, 0), ok())

  async def test_unstaged_old_revision_does_not_block_later_samples_in_same_run(self):
    ctrl = Controller(":memory:")
    self.addAsyncCleanup(ctrl.aclose)
    uri = "tinker://train/sampler_weights/latest"
    ctrl.create_run(Run("sample", "base", 8, role=Role.SAMPLER, model_path=uri))
    ids = []
    for version in ("v1", "v2", "v1", "v2"):
      ctrl.put_checkpoint(Checkpoint(uri, version, 1))
      ids.append(ctrl.submit_command("sample", Op.SAMPLE, b"{}"))
    r1 = ctrl.handle(NodeMsg("hello", "n", "boot", gpus=1))
    self.assertEqual(set(r1.stage), {"v1", "v2"})
    r2 = ctrl.handle(NodeMsg("hello", "n", "boot", gpus=1, staged=("v2",)))
    samples = [t for t in r2.tasks if t.kind == TaskKind.SAMPLE]
    self.assertEqual([t.cmd.request_id for t in samples], [ids[1], ids[3]])
    self.assertEqual(set(r2.stage), {"v1"})


if __name__ == "__main__":
  unittest.main()
