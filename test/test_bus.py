import asyncio
import json
import tempfile
import time
import unittest
from unittest.mock import patch

from test.helpers import RedisTest, new_id, start_redis
from tinyrl.bus import Bus, Cmd, Op, Reply, Run, connect, key


def ok(**body) -> Reply:
  return Reply(True, "application/json", json.dumps(body).encode())


class TestBus(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    await self.bus.own("p", "w0")  # only a pool's owner can answer for it

  async def new_run(self, pool: str = "p", session: str = "") -> str:
    run_id = new_id()
    await self.bus.create(Run(run_id, pool, "base", 8, session))
    return run_id

  async def test_a_command_goes_through_the_stream_and_its_reply_comes_back(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit(run_id, Op.OPTIM_STEP, b'{"lr": 1}')

    [cmd] = await self.bus.take("p", "w0", block_ms=10)
    self.assertEqual((cmd.run, cmd.op, cmd.body, cmd.request_id), (run_id, Op.OPTIM_STEP, b'{"lr": 1}', request_id))
    self.assertIsNone(await self.bus.result(request_id, 0))

    await self.bus.finish(cmd, ok(type="optim_step"), "p", "w0")
    self.assertEqual(await self.bus.result(request_id, 0), ok(type="optim_step"))
    self.assertEqual(await self.bus.take("p", "w0", block_ms=10), [])

    run = await self.bus.run(run_id)
    self.assertEqual((run["pool"], run["base"], run["rank"], run["steps"]), ("p", "base", 8, 1))

  async def test_a_reply_wakes_every_waiter_and_reading_it_never_consumes_it(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit(run_id, Op.OPTIM_STEP, b"")
    cmd = Cmd(run_id, request_id.split("/")[1], Op.OPTIM_STEP, b"")

    waiters = [asyncio.create_task(self.bus.result(request_id, 5)) for _ in range(3)]
    await asyncio.sleep(0.05)
    start = time.perf_counter()
    await self.bus.finish(cmd, ok(n=1), "p", "w0")

    self.assertEqual(await asyncio.gather(*waiters), [ok(n=1)] * 3)
    self.assertLess(time.perf_counter() - start, 0.5)
    self.assertEqual(await self.bus.result(request_id, 0), ok(n=1))

  async def test_hundreds_of_requests_wait_for_replies_at_once(self):
    """Each wait holds a Redis connection, as when 400 sampling requests are in flight through one gateway."""
    got = await asyncio.gather(*[self.bus.result(f"req-{i}", 0.5) for i in range(300)])
    self.assertEqual(got, [None] * 300)

  async def test_waiting_for_a_reply_outlasts_the_socket_timeout(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit(run_id, Op.OPTIM_STEP, b"")
    [cmd] = await self.bus.take("p", "w0", block_ms=10)

    async def answer_late():
      await asyncio.sleep(5.5)  # past redis-py's default socket timeout, but well inside retrieve_future's 30s
      await self.bus.finish(cmd, ok(n=1), "p", "w0")

    answering = asyncio.create_task(answer_late())
    self.assertEqual(await self.bus.result(request_id, 10), ok(n=1))
    await answering

  async def test_unknown_and_deleted_runs_take_no_work(self):
    self.assertIsNone(await self.bus.submit(new_id(), Op.OPTIM_STEP, b""))

    run_id = await self.new_run()
    await self.bus.drop(run_id, "p")
    self.assertIsNone(await self.bus.submit(run_id, Op.OPTIM_STEP, b""))
    self.assertEqual(await self.bus.runs("p"), [])
    self.assertEqual(await self.bus.queue(run_id), {"live": False})
    self.assertFalse(await self.bus.r.exists(key("q", run_id)))

  async def test_the_queue_is_read_from_the_stream_itself(self):
    run_id = await self.new_run()
    request_ids = [await self.bus.submit(run_id, Op.OPTIM_STEP, b"") for _ in range(3)]
    self.assertEqual(await self.bus.queue(run_id), {"live": True, "queued": 3, "in_flight": []})

    [cmd] = await self.bus.take("p", "w0", count=1, block_ms=10)
    queue = await self.bus.queue(run_id)
    self.assertEqual(queue["queued"], 2)
    [in_flight] = queue["in_flight"]
    self.assertEqual(
      (in_flight["request_id"], in_flight["trainer"], in_flight["deliveries"]), (request_ids[0], "w0", 1)
    )

    await self.bus.finish(cmd, ok(), "p", "w0")
    self.assertEqual(await self.bus.queue(run_id), {"live": True, "queued": 2, "in_flight": []})

  async def test_a_new_trainer_reclaims_what_a_dead_one_took(self):
    run_id = await self.new_run()
    for _ in range(2):
      await self.bus.submit(run_id, Op.OPTIM_STEP, b"")

    await self.bus.release("p", "w0")
    await self.bus.own("p", "dead")
    taken = await self.bus.take("p", "dead", block_ms=10)
    await self.bus.release("p", "dead")
    await self.bus.own("p", "new")
    reclaimed = await self.bus.reclaim("p", "new")
    self.assertEqual([cmd.request_id for cmd in reclaimed], [cmd.request_id for cmd in taken])
    self.assertEqual({entry["trainer"] for entry in (await self.bus.queue(run_id))["in_flight"]}, {"new"})
    self.assertEqual(await self.bus.take("p", "new", block_ms=10), [])

  async def test_every_run_in_a_pool_is_served_each_take(self):
    a, b, other = await self.new_run(), await self.new_run(), await self.new_run("other")
    for run_id in (a, a, a, b, other):
      await self.bus.submit(run_id, Op.OPTIM_STEP, b"")

    first = await self.bus.take("p", "w0", count=2, block_ms=10)
    self.assertEqual(sorted(cmd.run for cmd in first), sorted([a, a, b]))
    second = await self.bus.take("p", "w0", count=2, block_ms=10)
    self.assertEqual([cmd.run for cmd in second], [a])

  async def test_run_facts_commit_with_metadata_without_a_second_emit(self):
    with patch.object(self.bus, "emit", side_effect=RuntimeError("separate emission is unavailable")):
      await self.bus.create(Run("new", "", "base", 4, role="trainer", exclusive=True))
      await self.bus.drop("new", "")
    events = await self.bus.events()
    self.assertEqual([e.kind for e in events], ["run_created", "run_deleted"])
    self.assertEqual((events[0].body["base"], events[0].body["rank"], events[0].body["exclusive"]), ("base", 4, True))
    self.assertFalse(await self.bus.r.exists(key("q", "new")))

  async def test_invalid_requirements_do_not_create_partial_runs(self):
    with self.assertRaises(ValueError):
      await self.bus.create(Run("invalid", "", "base", 4, role="unknown"))
    self.assertIsNone(await self.bus.run("invalid"))
    self.assertEqual(await self.bus.events(), [])

  async def test_assignment_and_reply_fence_late_source_actions_after_exit(self):
    run = await self.new_run()
    await self.bus.submit(run, Op.OPTIM_STEP, b"")
    [cmd] = await self.bus.take("p", "w0", block_ms=0)
    await self.bus.unassign_worker("p")
    self.assertTrue(await self.bus.assign(run, "target"))
    self.assertFalse(await self.bus.finish(cmd, ok(stale=True), "p", "w0"))
    self.assertEqual(await self.bus.take("p", "w0", block_ms=0), [])
    await self.bus.own("target", "w1")
    [stranded] = await self.bus.reclaim("target", "w1")
    self.assertEqual(stranded.request_id, cmd.request_id)

  async def test_retirement_waits_for_lease_expiry_and_prevents_reacquisition(self):
    run = await self.new_run()
    await self.bus.submit(run, Op.OPTIM_STEP, b"")
    [cmd] = await self.bus.take("p", "w0", block_ms=0)
    after = (await self.bus.events())[-1].id
    self.assertFalse(await self.bus.fence_worker("p", after))
    await self.bus.r.pexpire(key("owner", "p"), 1)
    await asyncio.sleep(0.01)
    self.assertTrue(await self.bus.fence_worker("p", after))
    self.assertFalse(await self.bus.own("p", "w0"))
    self.assertFalse(await self.bus.own("p", "other"))
    self.assertEqual(await self.bus.take("p", "w0", block_ms=0), [])
    self.assertFalse(await self.bus.finish(cmd, ok(stale=True), "p", "w0"))
    self.assertEqual([e.kind for e in await self.bus.events()][-1], "worker_fenced")

  async def test_retirement_wins_against_an_acquisition_that_already_read_the_old_state(self):
    await self.bus.release("p", "w0")
    after = await self.bus.emit("node_lost", node="n")
    pipeline_type = type(self.bus.r.pipeline())
    execute, inserted = pipeline_type.execute, False

    async def race(pipeline, *args, **kwargs):
      nonlocal inserted
      if not inserted:
        inserted = True
        self.assertTrue(await self.bus.fence_worker("p", after))
      return await execute(pipeline, *args, **kwargs)

    with patch.object(pipeline_type, "execute", race):
      self.assertFalse(await self.bus.own("p", "w0"))
    self.assertIsNone(await self.bus.owner("p"))

  async def test_decision_tail_cas_rejects_an_intervening_fact_without_projecting_assignment(self):
    await self.bus.create(Run("new", "", "base", 4))
    after = (await self.bus.events())[-1].id
    await self.bus.emit("exited", worker="target", node="n")
    self.assertFalse(await self.bus.assign("new", "target", after=after))
    self.assertFalse(await self.bus.emit_decision("granted", after, worker="target"))
    self.assertEqual((await self.bus.run("new"))["pool"], "")
    self.assertEqual(await self.bus.runs("target"), [])
    tail = (await self.bus.events())[-1].id
    self.assertTrue(await self.bus.assign("new", "other", after=tail))

  async def test_a_fact_during_decision_commit_invalidates_its_watch(self):
    after = await self.bus.emit("wants_gpu", worker="w")
    pipeline_type = type(self.bus.r.pipeline())
    execute, inserted = pipeline_type.execute, False

    async def race(pipeline, *args, **kwargs):
      nonlocal inserted
      if not inserted:
        inserted = True
        await self.bus.emit("exited", worker="w", node="n")
      return await execute(pipeline, *args, **kwargs)

    with patch.object(pipeline_type, "execute", race):
      self.assertFalse(await self.bus.emit_decision("granted", after, worker="w"))
    self.assertEqual([e.kind for e in await self.bus.events()], ["wants_gpu", "exited"])

  async def test_a_failure_is_counted(self):
    run_id = await self.new_run()
    await self.bus.submit(run_id, Op.OPTIM_STEP, b"")
    [cmd] = await self.bus.take("p", "w0", block_ms=10)
    await self.bus.finish(cmd, Reply(False, "application/json", b'{"error": "x"}'), "p", "w0")

    run = await self.bus.run(run_id)
    self.assertEqual((run["steps"], run["failed"]), (0, 1))

  async def test_only_the_pools_owner_can_answer_for_it(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit(run_id, Op.OPTIM_STEP, b"")
    [cmd] = await self.bus.take("p", "w0", block_ms=10)

    self.assertFalse(await self.bus.own("p", "zombie"))  # w0 holds the lease
    self.assertFalse(await self.bus.renew("p", "zombie"))
    self.assertFalse(await self.bus.finish(cmd, ok(by="zombie"), "p", "zombie"))
    self.assertIsNone(await self.bus.result(request_id, 0))

    self.assertTrue(await self.bus.finish(cmd, ok(by="w0"), "p", "w0"))
    self.assertEqual(await self.bus.result(request_id, 0), ok(by="w0"))

    await self.bus.release("p", "zombie")  # not its lease, so this does nothing
    self.assertTrue(await self.bus.renew("p", "w0"))
    await self.bus.release("p", "w0")
    self.assertTrue(await self.bus.own("p", "next"))

  async def test_sessions(self):
    await self.bus.session("s")
    run_id = await self.new_run(session="s")

    session = await self.bus.session_info("s")
    self.assertTrue(session["live"])
    self.assertEqual(session["runs"], [run_id])
    self.assertIsNone(await self.bus.session_info("nope"))


class TestRedisRestart(unittest.IsolatedAsyncioTestCase):
  async def test_a_call_waits_out_a_redis_restart(self):
    sock = f"{tempfile.mkdtemp()}/redis.sock"
    proc = start_redis(sock)
    bus = Bus(connect(f"unix://{sock}"))
    try:
      await bus.heartbeat("node-a", 1)
      proc.terminate()
      proc.wait()

      async def restart():
        await asyncio.sleep(2)
        return start_redis(sock)

      restarting = asyncio.create_task(restart())
      await bus.heartbeat("node-a", 1)  # retried until Redis is back, instead of raising
      proc = await restarting
      self.assertEqual(await bus.gpus(), 1)
    finally:
      await bus.r.aclose()
      proc.terminate()


if __name__ == "__main__":
  unittest.main()
