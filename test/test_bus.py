import asyncio
import json
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

from test.helpers import RedisTest, new_id, start_redis
from tinyrl.bus import Bus, Cmd, Op, Reply, Run, connect, key


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

    [cmd] = await self.bus.take_commands("p", "w0", block_ms=10)
    self.assertEqual((cmd.run, cmd.op, cmd.body, cmd.request_id), (run_id, Op.OPTIM_STEP, b'{"lr": 1}', request_id))
    self.assertIsNone(await self.bus.wait_for_reply(request_id, 0))

    await self.bus.write_reply(cmd, ok(type="optim_step"), "p", "w0")
    self.assertEqual(await self.bus.wait_for_reply(request_id, 0), ok(type="optim_step"))
    self.assertEqual(await self.bus.take_commands("p", "w0", block_ms=10), [])

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
    [cmd] = await self.bus.take_commands("p", "w0", block_ms=10)

    async def answer_late():
      await asyncio.sleep(5.5)  # past redis-py's default socket timeout, but well inside retrieve_future's 30s
      await self.bus.write_reply(cmd, ok(n=1), "p", "w0")

    answering = asyncio.create_task(answer_late())
    self.assertEqual(await self.bus.wait_for_reply(request_id, 10), ok(n=1))
    await answering

  async def test_unknown_and_deleted_runs_take_no_work(self):
    self.assertIsNone(await self.bus.submit_command(new_id(), Op.OPTIM_STEP, b""))

    run_id = await self.new_run()
    await self.bus.delete_run(run_id, "p")
    self.assertIsNone(await self.bus.submit_command(run_id, Op.OPTIM_STEP, b""))
    self.assertEqual(await self.bus.runs_in_worker("p"), [])
    self.assertEqual(await self.bus.queued_and_in_flight(run_id), {"live": False})
    self.assertFalse(await self.bus.r.exists(key("q", run_id)))

  async def test_the_queue_is_read_from_the_stream_itself(self):
    run_id = await self.new_run()
    request_ids = [await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"") for _ in range(3)]
    self.assertEqual(await self.bus.queued_and_in_flight(run_id), {"live": True, "queued": 3, "in_flight": []})

    [cmd] = await self.bus.take_commands("p", "w0", count=1, block_ms=10)
    queue = await self.bus.queued_and_in_flight(run_id)
    self.assertEqual(queue["queued"], 2)
    [in_flight] = queue["in_flight"]
    self.assertEqual(
      (in_flight["request_id"], in_flight["trainer"], in_flight["deliveries"]), (request_ids[0], "w0", 1)
    )

    await self.bus.write_reply(cmd, ok(), "p", "w0")
    self.assertEqual(await self.bus.queued_and_in_flight(run_id), {"live": True, "queued": 2, "in_flight": []})

  async def test_a_new_trainer_reclaims_what_a_dead_one_took(self):
    run_id = await self.new_run()
    for _ in range(2):
      await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")

    taken = await self.bus.take_commands("p", "dead", block_ms=10)
    reclaimed = await self.bus.reclaim_stranded_commands("p", "new")
    self.assertEqual([cmd.request_id for cmd in reclaimed], [cmd.request_id for cmd in taken])
    self.assertEqual(
      {entry["trainer"] for entry in (await self.bus.queued_and_in_flight(run_id))["in_flight"]}, {"new"}
    )
    self.assertEqual(await self.bus.take_commands("p", "new", block_ms=10), [])

  async def test_planned_handoff_preserves_unstarted_commands_before_newer_work(self):
    run = await self.new_run()
    requests = [await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}", i) for i in range(3)]
    first, second = await self.bus.take_commands("p", "w0", count=2, block_ms=0)
    await self.bus.write_reply(first, ok(), "p", "w0")
    self.assertTrue(await self.bus.return_commands("p", "w0", [second]))
    await self.bus.release_lease("p", "w0")
    await self.bus.take_lease("p", "next")
    self.assertEqual(await self.bus.reclaim_stranded_commands("p", "next"), [])
    [resumed] = await self.bus.take_commands("p", "next", block_ms=0)
    self.assertEqual(resumed.request_id, requests[1])
    await self.bus.write_reply(resumed, ok(), "p", "next")
    [last] = await self.bus.take_commands("p", "next", block_ms=0)
    self.assertEqual(last.request_id, requests[2])

  async def test_a_stale_owner_cannot_return_someone_elses_claim(self):
    run = await self.new_run()
    await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}")
    cmds = await self.bus.take_commands("p", "w0", block_ms=0)
    await self.bus.release_lease("p", "w0")
    await self.bus.take_lease("p", "next")
    self.assertFalse(await self.bus.return_commands("p", "w0", cmds))
    self.assertEqual(await self.bus.reclaim_stranded_commands("p", "next"), cmds)

  async def test_checkpoint_publication_and_reply_require_the_same_current_owner(self):
    run = await self.new_run()
    request = await self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
    [cmd] = await self.bus.take_commands("p", "w0", block_ms=0)
    checkpoint = {"uri": f"tinker://{run}/weights/s", "size": 12, "version": "v1"}
    await self.bus.release_lease("p", "w0")
    await self.bus.take_lease("p", "next")
    self.assertFalse(await self.bus.write_reply(cmd, ok(), "p", "w0", checkpoint=checkpoint))
    self.assertIsNone(await self.registry.get_checkpoint(checkpoint["uri"]))
    self.assertIsNone(await self.bus.wait_for_reply(request, 0))
    self.assertTrue(await self.bus.write_reply(cmd, ok(), "p", "next", checkpoint=checkpoint))
    self.assertEqual((await self.registry.get_checkpoint(checkpoint["uri"]))["version"], "v1")
    self.assertIsNotNone(await self.bus.wait_for_reply(request, 0))

  async def test_all_fenced_writes_retry_a_renewal_without_duplicating_effects(self):
    for operation in ("reply", "park", "return"):
      with self.subTest(operation=operation):
        run = await self.new_run()
        await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}")
        [cmd] = await self.bus.take_commands("p", "w0", block_ms=0)
        checkpoint = {"uri": f"tinker://{run}/parked/s", "size": 12, "version": "v1"}
        pipeline = self.bus.r.pipeline(transaction=True)
        get = pipeline.get

        async def renewed(k, get=get):
          value = await get(k)
          await self.bus.renew_lease("p", "w0")
          return value

        pipeline.get = renewed
        with patch.object(self.bus.r, "pipeline", side_effect=[pipeline, self.bus.r.pipeline(transaction=True)]):
          if operation == "reply":
            self.assertTrue(await self.bus.write_reply(cmd, ok(), "p", "w0", checkpoint=checkpoint))
          elif operation == "park":
            self.assertTrue(await self.bus.set_parked_checkpoint(run, checkpoint["uri"], "p", "w0", checkpoint))
          else:
            self.assertTrue(await self.bus.return_commands("p", "w0", [cmd]))
        if operation == "reply":
          self.assertEqual((await self.bus.get_run(run))["steps"], 1)
          self.assertEqual((await self.registry.get_checkpoint(checkpoint["uri"]))["version"], "v1")
        elif operation == "park":
          self.assertEqual(await self.bus.parked_checkpoint(run), checkpoint["uri"])
          self.assertEqual((await self.registry.get_checkpoint(checkpoint["uri"]))["version"], "v1")
          await self.bus.write_reply(cmd, ok(), "p", "w0")
        else:
          [returned] = await self.bus.take_commands("p", "w0", block_ms=0)
          self.assertEqual(returned, cmd)
          await self.bus.write_reply(cmd, ok(), "p", "w0")

  async def test_replaced_lease_during_publication_does_not_publish_a_parked_checkpoint(self):
    run = await self.new_run()
    checkpoint = {"uri": f"tinker://{run}/parked/s", "size": 12, "version": "v1"}
    pipeline = self.bus.r.pipeline(transaction=True)
    get = pipeline.get

    async def replaced(k):
      value = await get(k)
      await self.bus.r.set(key("owner", "p"), "next")
      return value

    pipeline.get = replaced
    with patch.object(self.bus.r, "pipeline", side_effect=[pipeline, self.bus.r.pipeline(transaction=True)]):
      self.assertFalse(await self.bus.set_parked_checkpoint(run, checkpoint["uri"], "p", "w0", checkpoint))
    self.assertIsNone(await self.bus.parked_checkpoint(run))
    self.assertIsNone(await self.registry.get_checkpoint(checkpoint["uri"]))

  async def test_every_run_in_a_worker_is_served_each_take(self):
    a, b, other = await self.new_run(), await self.new_run(), await self.new_run("other")
    for run_id in (a, a, a, b, other):
      await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")

    first = await self.bus.take_commands("p", "w0", count=2, block_ms=10)
    self.assertEqual(sorted(cmd.run for cmd in first), sorted([a, a, b]))
    second = await self.bus.take_commands("p", "w0", count=2, block_ms=10)
    self.assertEqual([cmd.run for cmd in second], [a])

  async def test_a_failure_is_counted(self):
    run_id = await self.new_run()
    await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    [cmd] = await self.bus.take_commands("p", "w0", block_ms=10)
    await self.bus.write_reply(cmd, Reply(False, "application/json", b'{"error": "x"}'), "p", "w0")

    run = await self.bus.get_run(run_id)
    self.assertEqual((run["steps"], run["failed"]), (0, 1))

  async def test_only_the_workers_owner_can_answer_for_it(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"")
    [cmd] = await self.bus.take_commands("p", "w0", block_ms=10)

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

  async def test_a_paused_owner_cannot_recover_after_its_replacement_dies_unsaved(self):
    run = await self.new_run()
    request = await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}")
    [cmd] = await self.bus.take_commands("p", "w0", block_ms=0)
    await self.bus.r.delete(key("owner", "p"))
    self.assertTrue(await self.bus.take_lease("p", "next"))
    failed = Reply(False, "application/json", b'{"error":"previous owner lost the worker"}')
    self.assertTrue(await self.bus.write_reply(cmd, failed, "p", "next"))
    await self.bus.r.delete(key("owner", "p"))  # next dies without parking anything
    self.assertFalse(await self.bus.take_lease("p", "w0", recover=True))
    self.assertFalse(await self.bus.write_reply(cmd, ok(), "p", "w0"))
    self.assertEqual(await self.bus.wait_for_reply(request, 0), failed)
    self.assertIsNone(await self.bus.lease_owner("p"))

  async def test_lease_recovery_survives_untaken_expiry_and_redis_flush(self):
    self.assertFalse(await self.bus.take_lease("p", "zombie"))  # a failed contender must not replace history
    await self.bus.r.delete(key("owner", "p"))
    self.assertTrue(await self.bus.take_lease("p", "w0", recover=True))
    self.assertEqual(await self.bus.r.ttl(key("last_owner", "p")), -1)
    await self.bus.r.flushdb()
    self.assertTrue(await self.bus.take_lease("p", "w0", recover=True))
    self.assertEqual(await self.bus.lease_owner("p"), "w0")

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

  async def test_a_run_deleted_after_its_worker_was_listed_takes_nothing_until_the_next_take(self):
    run_id, gone = await self.new_run(), await self.new_run()
    await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    await self.bus.delete_run(gone, "p")
    with patch.object(self.bus, "runs_in_worker", AsyncMock(return_value=[run_id, gone])):  # listed before it went
      self.assertEqual(await self.bus.take_commands("p", "o", block_ms=10), [])
    self.assertEqual([c.run for c in await self.bus.take_commands("p", "o", block_ms=10)], [run_id])

  async def test_deleting_another_run_does_not_drop_a_claim_resumed_from_handoff(self):
    run, gone = await self.new_run(), await self.new_run()
    await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}")
    [cmd] = await self.bus.take_commands("p", "w0", block_ms=0)
    await self.bus.return_commands("p", "w0", [cmd])
    await self.bus.delete_run(gone, "p")
    with (
      patch.object(self.bus, "runs_in_worker", AsyncMock(return_value=[run, gone])),
      patch("tinyrl.bus.random.shuffle"),
    ):
      self.assertEqual(await self.bus.take_commands("p", "w0", block_ms=0), [cmd])

  async def test_a_run_deleted_after_its_worker_was_listed_is_skipped_when_reclaiming(self):
    run_id, gone = await self.new_run(), await self.new_run()
    await self.bus.submit_command(run_id, Op.OPTIM_STEP, b"{}")
    await self.bus.take_commands("p", "dead", block_ms=10)  # taken, and never answered
    await self.bus.delete_run(gone, "p")
    with patch.object(self.bus, "runs_in_worker", AsyncMock(return_value=[gone, run_id])):  # listed before it went
      self.assertEqual([c.run for c in await self.bus.reclaim_stranded_commands("p", "o")], [run_id])

  async def test_stats_and_workers_cleanup(self):
    run_unassigned = new_id()
    await self.bus.create_run(Run(id=run_unassigned, worker="", base="base", rank=1))

    run_assigned = new_id()
    await self.bus.create_run(Run(id=run_assigned, worker="", base="base", rank=1))
    await self.bus.assign_run_to_worker(run_assigned, "w1")

    run_preassigned = new_id()
    await self.bus.create_run(Run(id=run_preassigned, worker="w2", base="base", rank=1))

    stats = await self.bus.run_counters()
    self.assertIn(("w1", run_assigned), stats)
    self.assertIn(("w2", run_preassigned), stats)
    self.assertEqual(len(stats), 2)
    self.assertEqual(stats[("w1", run_assigned)], {"steps": 0, "failed": 0, "cmds": 0, "busy_ms": 0, "wait_ms": 0})
    self.assertEqual(stats[("w2", run_preassigned)], {"steps": 0, "failed": 0, "cmds": 0, "busy_ms": 0, "wait_ms": 0})

    await self.bus.delete_run(run_assigned, "w1")
    stats_after = await self.bus.run_counters()
    self.assertNotIn(("w1", run_assigned), stats_after)

    self.assertFalse(await self.bus.r.exists(key("workers")))
    runs_in_redis = {r.decode() for r in await self.bus.r.smembers(key("runs"))}
    self.assertIn(run_preassigned, runs_in_redis)
    self.assertIn(run_unassigned, runs_in_redis)
    self.assertNotIn(run_assigned, runs_in_redis)

    self.assertFalse(hasattr(self.bus, "workers"))


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
