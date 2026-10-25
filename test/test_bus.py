import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from redis.asyncio.client import Pipeline
from redis.exceptions import ConnectionError as RedisConnectionError

from test.helpers import CoordinatorTest, new_id
from tinyrl.bus import Bus, CoordinatorUnavailable, Op, Reply, Run


def ok(*, artifact="", **body) -> Reply:
  return Reply(True, "application/json", json.dumps(body).encode(), artifact=artifact)


class TestBus(CoordinatorTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    await self.bus.register("w0", "p")
    await self.bus.register("w1", "other")

  async def new_run(self, pool="p", session="", *, initialized=False):
    run = new_id()
    body = {"base": "base", "rank": 8, "lora": {}} if initialized else None
    await self.bus.create(Run(run, pool, "base", 8, session), body=body)
    if initialized:
      [cmd] = await self.bus.take(pool, "w0" if pool == "p" else "w1", block_ms=0)
      await self.bus.finish(cmd, ok())
    return run

  async def take(self, pool="p", worker="w0"):
    [cmd] = await self.bus.take(pool, worker, block_ms=0)
    return cmd

  async def test_command_reply_and_all_waiters(self):
    run = await self.new_run()
    request = await self.bus.submit(run, Op.OPTIM_STEP, b'{"lr":1}')
    cmd = await self.take()
    self.assertEqual((cmd.run, cmd.op, cmd.request_id), (run, Op.OPTIM_STEP, request))
    waiters = [asyncio.create_task(self.bus.result(request, 2)) for _ in range(3)]
    await asyncio.sleep(0)
    self.assertEqual(await self.bus.finish(cmd, ok(n=1)), "committed")
    self.assertEqual(await asyncio.gather(*waiters), [ok(n=1)] * 3)
    self.assertEqual(await self.bus.result(request, 0), ok(n=1))
    self.assertEqual((await self.bus.run(run))["steps"], 1)
    self.assertEqual(await self.bus.take("p", "w0", block_ms=0), [])

  async def test_create_atomic_idempotency_and_bootstrap_does_not_consume_sequence_zero(self):
    run = Run(new_id(), "p", "base", 8)
    body = {"base": "base", "rank": 8, "lora": {"seed": 123}}
    first = await self.bus.create(run, body=body, idempotency_key="session/0")
    second = await self.bus.create(run, body=body, idempotency_key="session/0")
    self.assertEqual(first, second)
    with self.assertRaisesRegex(ValueError, "configuration"):
      await self.bus.create(run, body={**body, "rank": 2}, idempotency_key="session/0")
    request = await self.bus.submit(run.id, Op.OPTIM_STEP, b"{}", seq_id=0)
    create = await self.take()
    self.assertEqual(create.request_id, first["request_id"])
    await self.bus.finish(create, ok())
    self.assertEqual((await self.take()).request_id, request)

  async def test_sequence_gap_blocks_and_retries_are_durable(self):
    run = await self.new_run()
    second = await self.bus.submit(run, Op.OPTIM_STEP, b'{"lr":2}', seq_id=1)
    self.assertEqual(await self.bus.take("p", "w0", block_ms=0), [])
    first = await self.bus.submit(run, Op.OPTIM_STEP, b'{"lr":1}', seq_id=0)
    self.assertEqual(first, await self.bus.submit(run, Op.OPTIM_STEP, b'{"lr":1}', seq_id=0, idempotency_key="first"))
    self.assertEqual(first, await self.bus.submit(run, Op.OPTIM_STEP, b'{"lr":1}', idempotency_key="first"))
    with self.assertRaisesRegex(ValueError, "different"):
      await self.bus.submit(run, Op.OPTIM_STEP, b'{"lr":1}', seq_id=0, idempotency_key="different")
    with self.assertRaisesRegex(ValueError, "different content"):
      await self.bus.submit(run, Op.OPTIM_STEP, b'{"lr":3}', seq_id=0)
    cmd = await self.take()
    self.assertEqual(cmd.request_id, first)
    self.assertEqual([cmd], await self.bus.take("p", "w0", block_ms=0))
    await self.bus.close()
    self.bus = Bus(url=self.url, namespace=self.namespace)
    self.assertEqual([cmd], await self.bus.take("p", "w0", block_ms=0))
    self.assertEqual(first, await self.bus.submit(run, Op.OPTIM_STEP, b'{"lr":1}', seq_id=0))
    await self.bus.finish(cmd, ok())
    self.assertEqual((await self.take()).request_id, second)

  async def test_parallel_runs_have_one_outstanding_each(self):
    runs = [await self.new_run() for _ in range(3)]
    for run in runs:
      for seq in range(2):
        await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=seq)
    commands = await self.bus.take("p", "w0", count=16, block_ms=0)
    self.assertEqual(sorted(c.run for c in commands), sorted(runs))
    self.assertEqual(commands, await self.bus.take("p", "w0", count=16, block_ms=0))
    for cmd in commands:
      await self.bus.finish(cmd, ok())
    self.assertEqual(len(await self.bus.take("p", "w0", count=16, block_ms=0)), 3)

  async def test_committed_artifact_identity_and_dispatch_time_load_resolution(self):
    run = await self.new_run()
    save = await self.bus.submit(run, Op.SAVE_WEIGHTS, b'{"uri":"tinyrl://run/weights/a"}', seq_id=0)
    await self.bus.submit(run, Op.LOAD_WEIGHTS, b'{"uri":"tinyrl://run/weights/a","optimizer":true}', seq_id=1)
    cmd = await self.take()
    self.assertEqual(cmd.request_id, save)
    with self.assertRaisesRegex(ValueError, "artifact"):
      await self.bus.finish(cmd, ok(artifact="../unsafe"))
    reply = ok(artifact="a" * 32)
    forged = replace(cmd, op=Op.DELETE_MODEL, body=b'{"uri":"tinyrl://attacker"}')
    self.assertEqual(await self.bus.finish(forged, reply), "committed")
    self.assertEqual(await self.bus.ckpt("tinyrl://run/weights/a"), "a" * 32)
    self.assertIsNone(await self.bus.ckpt("tinyrl://attacker"))
    self.assertEqual(await self.bus.finish(cmd, reply), "already_committed")
    with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
      await self.bus.finish(cmd, ok(artifact="b" * 32))
    load = await self.take()
    self.assertEqual(json.loads(load.body)["ref"], "a" * 32)

  async def test_delete_accepts_missing_predecessors_and_terminates_tail(self):
    run = await self.new_run()
    tail = await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=2)
    deletion = await self.bus.submit(run, Op.DELETE_MODEL, b"{}", seq_id=1)
    with self.assertRaisesRegex(ValueError, "closing"):
      await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=3)
    with self.assertRaisesRegex(ValueError, "closing"):
      await self.bus.submit(run, Op.OPTIM_STEP, b"{}")
    first = await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=0)
    self.assertEqual((await self.take()).request_id, first)
    await self.bus.finish(await self.take(), ok())
    cmd = await self.take()
    self.assertEqual(cmd.request_id, deletion)
    await self.bus.finish(cmd, ok())
    self.assertFalse((await self.bus.result(tail, 0)).ok)
    self.assertEqual(await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=2), tail)
    self.assertIsNone(await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=3))
    self.assertEqual(await self.bus.queue(run), {"live": False})

  async def test_unknown_and_expired_futures_are_terminal_but_identity_survives_pruning(self):
    self.assertFalse((await self.bus.result("missing/future", 0)).ok)
    run = await self.new_run()
    request = await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=0)
    cmd = await self.take()
    await self.bus.finish(cmd, ok())
    with patch("tinyrl.bus.REPLY_TTL", -1):
      await self.bus.expire_workers()
      reply = await self.bus.result(request, 0)
    self.assertFalse(reply.ok)
    self.assertIn(b"expired", reply.body)
    self.assertEqual(request, await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=0))
    self.assertEqual(await self.bus.finish(cmd, ok()), "already_committed")

  async def test_worker_loss_terminates_futures_without_replay_and_allows_explicit_recovery(self):
    run = await self.new_run(initialized=True)
    requests = [await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=i) for i in range(2)]
    uncertain = await self.take()
    self.assertFalse((await self.bus.register("replacement", "p"))["registered"])
    self.assertEqual(await self.bus.worker_lost("w0"), 1)
    self.assertFalse(await self.bus.heartbeat("w0"))
    self.assertFalse((await self.bus.register("w0", "p"))["registered"])
    for request in requests:
      self.assertFalse((await self.bus.result(request, 0)).ok)
    self.assertEqual(await self.bus.finish(uncertain, ok()), "stale")
    self.assertTrue((await self.bus.register("replacement", "p"))["registered"])
    self.assertEqual(await self.bus.take("p", "replacement", block_ms=0), [])
    await self.bus.recover(run, "c" * 32)
    restore = await self.take("p", "replacement")
    self.assertEqual(restore.op, Op.RESTORE)
    self.assertTrue(json.loads(restore.body)["recovery"])
    self.assertEqual(json.loads(restore.body)["boundary"], 2)
    await self.bus.finish(restore, ok())
    activate = await self.take("p", "replacement")
    self.assertEqual(activate.op, Op.ACTIVATE)
    await self.bus.finish(activate, ok())
    new = await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=2)
    self.assertEqual((await self.take("p", "replacement")).request_id, new)

  async def test_move_uses_quiescent_boundary_keeps_source_until_target_active_and_fences_old_loss(self):
    run = await self.new_run(initialized=True)
    await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=0)
    executing = await self.take()
    await self.bus.request_move(run, "other")
    await self.bus.finish(executing, ok())
    snapshot = await self.take()
    self.assertEqual((snapshot.op, json.loads(snapshot.body)["boundary"]), (Op.SNAPSHOT, 1))
    future = await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=1)
    snapshot_reply = ok(artifact="d" * 32)
    await self.bus.finish(snapshot, snapshot_reply)
    self.assertEqual(await self.bus.take("p", "w0", block_ms=0), [])
    restore = await self.take("other", "w1")
    self.assertEqual(restore.op, Op.RESTORE)
    self.assertEqual((await self.bus.run(run))["owner"], "w0")
    await self.bus.finish(restore, ok())
    state = await self.bus.run(run)
    self.assertEqual((state["owner"], state["pool"], state["phase"]), ("w1", "other", "activating"))
    self.assertEqual(await self.bus.take("p", "w0", block_ms=0), [])
    activate = await self.take("other", "w1")
    self.assertEqual(activate.op, Op.ACTIVATE)
    await self.bus.finish(activate, ok())
    self.assertFalse(await self.bus.lose_worker(run, "w0", snapshot.generation))
    self.assertEqual(await self.bus.finish(snapshot, snapshot_reply), "already_committed")
    detach = await self.take()
    self.assertEqual((detach.op, detach.generation), (Op.DETACH, snapshot.generation))
    await self.bus.finish(detach, ok())
    self.assertEqual((await self.take("other", "w1")).request_id, future)

  async def test_target_loss_before_ownership_commit_resumes_source(self):
    run = await self.new_run(initialized=True)
    await self.bus.request_move(run, "other")
    await self.bus.finish(await self.take(), ok(artifact="e" * 32))
    restore = await self.take("other", "w1")
    await self.bus.worker_lost("w1")
    state = await self.bus.run(run)
    self.assertEqual((state["owner"], state["phase"]), ("w0", "active"))
    self.assertEqual(await self.bus.finish(restore, ok()), "stale")
    await self.bus.submit(run, Op.OPTIM_STEP, b"{}")
    self.assertEqual((await self.take()).op, Op.OPTIM_STEP)

  async def test_park_has_durable_snapshot_and_restores_same_identity(self):
    run = await self.new_run(initialized=True)
    await self.bus.request_move(run, "")
    await self.bus.finish(await self.take(), ok(artifact="f" * 32))
    self.assertEqual((await self.bus.run(run))["phase"], "parked")
    detach = await self.take()
    self.assertEqual(detach.op, Op.DETACH)
    await self.bus.finish(detach, ok())
    future = await self.bus.submit(run, Op.OPTIM_STEP, b"{}")
    await self.bus.request_move(run, "other")
    restore = await self.take("other", "w1")
    self.assertEqual(json.loads(restore.body)["ref"], "f" * 32)
    await self.bus.finish(restore, ok())
    await self.bus.finish(await self.take("other", "w1"), ok())
    self.assertEqual((await self.take("other", "w1")).request_id, future)

  async def test_delete_parked_or_failed_run_without_restoring_it(self):
    run = await self.new_run(initialized=True)
    await self.bus.request_move(run, "")
    await self.bus.finish(await self.take(), ok(artifact="f" * 32))
    await self.bus.finish(await self.take(), ok())  # detach parked source
    waiting = await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=0)
    deletion = await self.bus.submit(run, Op.DELETE_MODEL, b"{}", seq_id=1)
    self.assertTrue((await self.bus.result(deletion, 0)).ok)
    self.assertFalse((await self.bus.result(waiting, 0)).ok)
    self.assertEqual((await self.bus.run(run))["phase"], "deleted")

    run = await self.new_run(initialized=True)
    await self.bus.submit(run, Op.DELETE_MODEL, b"{}", seq_id=0)
    cmd = await self.take()
    await self.bus.finish(cmd, Reply(False, "application/json", b'{"error":"unknown"}', recover=True))
    deletion = await self.bus.submit(run, Op.DELETE_MODEL, b"{}", seq_id=1)
    self.assertTrue((await self.bus.result(deletion, 0)).ok)
    detach = await self.take()
    self.assertEqual((detach.op, detach.generation), (Op.DETACH, cmd.generation))

  async def test_recovery_to_other_pool_detaches_poisoned_live_source_without_rollback(self):
    run = await self.new_run(initialized=True)
    await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=0)
    cmd = await self.take()
    await self.bus.finish(cmd, Reply(False, "application/json", b'{"error":"unknown"}', recover=True))
    await self.bus.recover(run, "a" * 32, "other")
    detach = await self.take()
    self.assertEqual((detach.op, detach.generation), (Op.DETACH, cmd.generation))
    await self.bus.finish(detach, ok())
    restore = await self.take("other", "w1")
    await self.bus.finish(restore, Reply(False, "application/json", b'{"error":"invalid checkpoint"}'))
    self.assertEqual((await self.bus.run(run))["phase"], "recovery")

  async def test_failed_activation_recovery_cleans_both_resident_generations(self):
    run = await self.new_run(initialized=True)
    await self.bus.request_move(run, "other")
    await self.bus.finish(await self.take(), ok(artifact="a" * 32))
    await self.bus.finish(await self.take("other", "w1"), ok())
    activation = await self.take("other", "w1")
    await self.bus.finish(activation, Reply(False, "application/json", b'{"error":"activation"}'))
    await self.bus.recover(run, "a" * 32, "other")
    source = await self.take()
    target = await self.take("other", "w1")
    self.assertEqual((source.op, source.generation), (Op.DETACH, activation.generation - 1))
    self.assertEqual((target.op, target.generation), (Op.DETACH, activation.generation))
    await self.bus.finish(source, ok())
    await self.bus.finish(target, ok())
    self.assertEqual((await self.take("other", "w1")).op, Op.RESTORE)

  async def test_lost_source_after_handoff_never_leaves_queued_cleanup(self):
    run = await self.new_run(initialized=True)
    await self.bus.request_move(run, "other")
    await self.bus.finish(await self.take(), ok(artifact="a" * 32))
    await self.bus.finish(await self.take("other", "w1"), ok())
    self.assertEqual(await self.bus.worker_lost("w0"), 0)
    await self.bus.finish(await self.take("other", "w1"), ok())
    self.assertEqual((await self.bus.queue(run))["queued"], 0)

  async def test_out_of_order_queue_reserves_room_for_missing_predecessor(self):
    run = await self.new_run()
    with patch("tinyrl.bus.MAX_PENDING_COMMANDS", 2):
      await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=2)
      with self.assertRaisesRegex(ValueError, "reserved"):
        await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=3)
      first = await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=0)
      self.assertEqual((await self.take()).request_id, first)
      await self.bus.finish(await self.take(), ok())
      second = await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=1)
      self.assertEqual((await self.take()).request_id, second)

  async def test_restart_grace_and_expiry(self):
    run = await self.new_run(initialized=True)
    await self.bus.submit(run, Op.OPTIM_STEP, b"{}")
    await self.take()
    await self.bus.r.zadd(self.bus.key("workers"), {"w0": time.time() - 100})
    self.assertEqual(await self.bus.expire_workers(), 0)
    self.bus._started -= 100
    self.assertEqual(await self.bus.expire_workers(), 1)
    self.assertEqual((await self.bus.run(run))["phase"], "recovery")

  async def test_queue_admission_limits_and_sessions(self):
    await self.bus.session("session")
    run = await self.new_run(session="session")
    with patch("tinyrl.bus.MAX_COMMAND_BYTES", 3):
      with self.assertRaisesRegex(ValueError, "MAX_COMMAND_BYTES"):
        await self.bus.submit(run, Op.OPTIM_STEP, b"1234")
    with patch("tinyrl.bus.MAX_PENDING_COMMANDS", 1):
      await self.bus.submit(run, Op.OPTIM_STEP, b"{}")
      with self.assertRaisesRegex(ValueError, "capacity"):
        await self.bus.submit(run, Op.OPTIM_STEP, b"{}")
    session = await self.bus.session_info("session")
    self.assertTrue(session["live"])
    self.assertEqual(session["runs"], [run])
    self.assertIsNone(await self.bus.session_info("absent"))

  async def test_exclusive_owner_and_committed_work_survives_process_exit(self):
    other = Bus(url=self.url, namespace=self.namespace)
    try:
      with self.assertRaisesRegex(CoordinatorUnavailable, "another coordinator"):
        await other.start()
    finally:
      await other.close()
    await self.bus.close()
    script = """
import asyncio, os, sys
from tinyrl.bus import Bus, Op, Run
async def main():
  bus = Bus(url=sys.argv[1], namespace=sys.argv[2], lease_seconds=.2)
  await bus.create(Run('durable', 'p', 'base', 8))
  request = await bus.submit('durable', Op.OPTIM_STEP, b'{}', seq_id=0)
  print(request, flush=True)
  async with bus._transaction() as tx:
    run = await tx.record(bus.key('run', 'durable'))
    run['cursor'] = 99
    tx.save(bus.key('run', 'durable'), run)
    os._exit(0)  # no EXEC for the cursor update
asyncio.run(main())
"""
    process = subprocess.run(
      [sys.executable, "-c", script, self.url, self.namespace],
      cwd=os.getcwd(),
      capture_output=True,
      text=True,
      check=True,
    )
    request = process.stdout.strip()
    self.redis_server.stop(kill=True)
    self.redis_server.start()  # reload the AOF after an actual Redis process crash
    await asyncio.sleep(0.25)
    self.bus = Bus(url=self.url, namespace=self.namespace)
    self.assertEqual((await self.bus.run("durable"))["cursor"], 0)
    self.assertEqual(await self.bus.submit("durable", Op.OPTIM_STEP, b"{}", seq_id=0), request)
    self.assertEqual((await self.take()).request_id, request)

  async def test_lost_exec_response_reconciles_admission_and_completion(self):
    run = await self.new_run()
    original = Pipeline.execute

    async def committed_but_disconnected(pipe, *args, **kwargs):
      await original(pipe, *args, **kwargs)
      raise RedisConnectionError("connection lost after EXEC committed")

    with patch.object(Pipeline, "execute", committed_but_disconnected):
      with self.assertRaises(CoordinatorUnavailable):
        await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=0)
    request = await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=0)
    self.assertEqual((await self.bus.queue(run))["queued"], 1)
    cmd = await self.take()
    self.assertEqual(cmd.request_id, request)
    with patch.object(Pipeline, "execute", committed_but_disconnected):
      with self.assertRaises(CoordinatorUnavailable):
        await self.bus.finish(cmd, ok())
    self.assertEqual(await self.bus.finish(cmd, ok()), "already_committed")
    self.assertEqual((await self.bus.run(run))["steps"], 1)

  async def test_expired_coordinator_cannot_commit_or_release_successors_lease(self):
    run = await self.new_run()
    successor = Bus(url=self.url, namespace=self.namespace)
    try:
      with self.assertRaises(CoordinatorUnavailable):
        async with self.bus._transaction() as tx:
          r = await tx.record(self.bus.key("run", run))
          r["cursor"] = 99
          tx.save(self.bus.key("run", run), r)
          await self.bus.r.pexpire(self.bus.key("coordinator"), 1)
          await asyncio.sleep(0.02)
          await successor.start()
      with self.assertRaises(CoordinatorUnavailable):
        await self.bus.submit(run, Op.OPTIM_STEP, b"{}")
      await self.bus.close()
      self.assertEqual(await successor.r.get(successor.key("coordinator")), successor.token.encode())
      self.assertEqual((await successor.run(run))["cursor"], 0)
    finally:
      await successor.close()

  async def test_redis_runtime_transaction_error_fences_and_reports_required_repair(self):
    run = await self.new_run()
    wrong_type = self.bus.key("wrong_type")
    await self.bus.r.set(wrong_type, "outside writer corrupted a hash")
    with self.assertLogs("tinyrl.bus", level="ERROR") as logs:
      with self.assertRaisesRegex(CoordinatorUnavailable, "inspect and repair"):
        async with self.bus._transaction() as tx:
          r = await tx.record(self.bus.key("run", run))
          r["cursor"] = 99
          tx.save(self.bus.key("run", run), r)
          tx.write("hset", wrong_type, "field", "value")
    # This is a real Redis runtime error: MULTI did not roll back the first
    # command. The current process must halt writes and request explicit repair.
    self.assertEqual((await self.bus.run(run))["cursor"], 99)
    with self.assertRaises(CoordinatorUnavailable):
      await self.bus.submit(run, Op.OPTIM_STEP, b"{}")
    self.assertIn("repair this namespace before restarting", " ".join(logs.output))

  async def test_payload_identity_is_independent_of_transport_and_logical_size_bounds_admission(self):
    run = await self.new_run()
    data = b"batch input"
    ref = hashlib.sha256(data).hexdigest()
    request = await self.bus.submit(run, Op.FORWARD_BACKWARD, data, seq_id=0)
    self.assertEqual(
      request, await self.bus.submit(run, Op.FORWARD_BACKWARD, b"", seq_id=0, body_ref=ref, body_size=len(data))
    )
    cmd = await self.take()
    reply = ok(n=1)
    self.assertEqual(await self.bus.finish(cmd, reply), "committed")
    alternate = replace(reply, body=b"", body_ref=hashlib.sha256(reply.body).hexdigest(), body_size=len(reply.body))
    self.assertEqual(await self.bus.finish(cmd, alternate), "already_committed")
    big = b"x" * (1 << 20)
    request = await self.bus.submit(
      run, Op.FORWARD_BACKWARD, b"", seq_id=1, body_ref=hashlib.sha256(big).hexdigest(), body_size=len(big)
    )
    cmd = await self.take()
    self.assertEqual((cmd.body, cmd.body_ref, cmd.body_size), (b"", hashlib.sha256(big).hexdigest(), len(big)))
    self.assertLess(self.redis_server.client.memory_usage(self.bus.key("cmd", cmd.entry_id)), 4096)
    await self.bus.finish(cmd, Reply(True, "application/octet-stream", b"", body_ref=cmd.body_ref, body_size=len(big)))
    result = await self.bus.result(request, 0)
    self.assertEqual((result.body, result.body_ref, result.body_size), (b"", cmd.body_ref, len(big)))
    with patch("tinyrl.bus.MAX_COMMAND_BYTES", 100):
      with self.assertRaisesRegex(ValueError, "MAX_COMMAND_BYTES"):
        await self.bus.submit(run, Op.FORWARD_BACKWARD, b"", body_ref=ref, body_size=101)

  async def test_single_command_prefetch_rotates_ready_runs_fairly(self):
    runs = [await self.new_run() for _ in range(3)]
    for run in runs:
      for seq in range(2):
        await self.bus.submit(run, Op.OPTIM_STEP, b"{}", seq_id=seq)
    seen = []
    for _ in range(6):
      [cmd] = await self.bus.take("p", "w0", count=1, block_ms=0)
      seen.append(cmd.run)
      await self.bus.finish(cmd, ok())
    self.assertEqual(seen, runs + runs)


if __name__ == "__main__":
  unittest.main()
