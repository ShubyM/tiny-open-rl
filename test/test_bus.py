import asyncio
import json
import time
import unittest

from test.helpers import RedisTest, new_id
from tinyrl.bus import Cmd, Reply, Run, key


def ok(**body) -> Reply:
  return Reply(True, "application/json", json.dumps(body).encode())


class TestBus(RedisTest):
  async def new_run(self, pool: str = "p", session: str = "") -> str:
    run_id = new_id()
    await self.bus.create(Run(run_id, pool, "base", 8, session))
    return run_id

  async def test_a_command_goes_through_the_stream_and_its_reply_comes_back(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit(run_id, "optim_step", b'{"lr": 1}')

    [cmd] = await self.bus.take("p", "w0", block_ms=10)
    self.assertEqual((cmd.run, cmd.op, cmd.body, cmd.request_id), (run_id, "optim_step", b'{"lr": 1}', request_id))
    self.assertIsNone(await self.bus.result(request_id, 0))

    await self.bus.finish(cmd, ok(type="optim_step"))
    self.assertEqual(await self.bus.result(request_id, 0), ok(type="optim_step"))
    self.assertEqual(await self.bus.take("p", "w0", block_ms=10), [])

    run = await self.bus.run(run_id)
    self.assertEqual((run["pool"], run["base"], run["rank"], run["steps"]), ("p", "base", 8, 1))

  async def test_a_reply_wakes_every_waiter_and_reading_it_never_consumes_it(self):
    run_id = await self.new_run()
    request_id = await self.bus.submit(run_id, "x", b"")
    cmd = Cmd(run_id, request_id.split("/")[1], "x", b"")

    waiters = [asyncio.create_task(self.bus.result(request_id, 5)) for _ in range(3)]
    await asyncio.sleep(0.05)
    start = time.perf_counter()
    await self.bus.finish(cmd, ok(n=1))

    self.assertEqual(await asyncio.gather(*waiters), [ok(n=1)] * 3)
    self.assertLess(time.perf_counter() - start, 0.5)
    self.assertEqual(await self.bus.result(request_id, 0), ok(n=1))

  async def test_unknown_and_deleted_runs_take_no_work(self):
    self.assertIsNone(await self.bus.submit(new_id(), "x", b""))

    run_id = await self.new_run()
    await self.bus.drop(run_id, "p")
    self.assertIsNone(await self.bus.submit(run_id, "x", b""))
    self.assertEqual(await self.bus.runs("p"), [])
    self.assertEqual(await self.bus.queue(run_id), {"live": False})
    self.assertFalse(await self.bus.r.exists(key("q", run_id)))

  async def test_the_queue_is_read_from_the_stream_itself(self):
    run_id = await self.new_run()
    request_ids = [await self.bus.submit(run_id, "x", b"") for _ in range(3)]
    self.assertEqual(await self.bus.queue(run_id), {"live": True, "queued": 3, "in_flight": []})

    [cmd] = await self.bus.take("p", "w0", count=1, block_ms=10)
    queue = await self.bus.queue(run_id)
    self.assertEqual(queue["queued"], 2)
    [in_flight] = queue["in_flight"]
    self.assertEqual(
      (in_flight["request_id"], in_flight["trainer"], in_flight["deliveries"]), (request_ids[0], "w0", 1)
    )

    await self.bus.finish(cmd, ok())
    self.assertEqual(await self.bus.queue(run_id), {"live": True, "queued": 2, "in_flight": []})

  async def test_a_new_trainer_reclaims_what_a_dead_one_took(self):
    run_id = await self.new_run()
    for _ in range(2):
      await self.bus.submit(run_id, "x", b"")

    taken = await self.bus.take("p", "dead", block_ms=10)
    reclaimed = await self.bus.reclaim("p", "new")
    self.assertEqual([cmd.request_id for cmd in reclaimed], [cmd.request_id for cmd in taken])
    self.assertEqual({entry["trainer"] for entry in (await self.bus.queue(run_id))["in_flight"]}, {"new"})
    self.assertEqual(await self.bus.take("p", "new", block_ms=10), [])

  async def test_every_run_in_a_pool_is_served_each_take(self):
    a, b, other = await self.new_run(), await self.new_run(), await self.new_run("other")
    for run_id in (a, a, a, b, other):
      await self.bus.submit(run_id, "x", b"")

    first = await self.bus.take("p", "w0", count=2, block_ms=10)
    self.assertEqual(sorted(cmd.run for cmd in first), sorted([a, a, b]))
    second = await self.bus.take("p", "w0", count=2, block_ms=10)
    self.assertEqual([cmd.run for cmd in second], [a])

  async def test_a_failure_is_counted(self):
    run_id = await self.new_run()
    await self.bus.submit(run_id, "optim_step", b"")
    [cmd] = await self.bus.take("p", "w0", block_ms=10)
    await self.bus.finish(cmd, Reply(False, "application/json", b'{"error": "x"}'))

    run = await self.bus.run(run_id)
    self.assertEqual((run["steps"], run["failed"]), (0, 1))

  async def test_sessions(self):
    await self.bus.session("s")
    run_id = await self.new_run(session="s")

    session = await self.bus.session_info("s")
    self.assertTrue(session["live"])
    self.assertEqual(session["runs"], [run_id])
    self.assertIsNone(await self.bus.session_info("nope"))


if __name__ == "__main__":
  unittest.main()
