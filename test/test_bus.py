import asyncio, json, time, unittest
from tinyrl.bus import Cmd, Reply, Run, key
from test.helpers import RedisTest, rid

def ok(**d) -> Reply: return Reply(True, "application/json", json.dumps(d).encode())

class TestBus(RedisTest):
  async def new_run(self, pool:str="p", session:str="") -> str:
    await self.bus.create(Run(r:=rid(), pool, "base", 8, session))
    return r

  async def test_a_command_goes_through_the_stream_and_its_reply_comes_back(self):
    r = await self.new_run()
    rid1 = await self.bus.submit(r, "optim_step", b'{"lr": 1}')
    [c] = await self.bus.take("p", "w0", block=10)
    self.assertEqual((c.run, c.op, c.body, c.rid), (r, "optim_step", b'{"lr": 1}', rid1))
    self.assertIsNone(await self.bus.result(rid1, 0))
    await self.bus.finish(c, ok(type="optim_step"))
    self.assertEqual(await self.bus.result(rid1, 0), ok(type="optim_step"))
    rec = await self.bus.run(r)
    self.assertEqual((rec["pool"], rec["base"], rec["rank"], rec["steps"]), ("p", "base", 8, 1))
    self.assertEqual(await self.bus.take("p", "w0", block=10), [])

  async def test_a_reply_wakes_every_waiter_and_reading_it_never_consumes_it(self):
    r = await self.new_run()
    c = Cmd(r, (await self.bus.submit(r, "x", b"")).split("/")[1], "x", b"")
    waiters = [asyncio.create_task(self.bus.result(c.rid, 5)) for _ in range(3)]
    await asyncio.sleep(0.05)
    st = time.perf_counter()
    await self.bus.finish(c, ok(n=1))
    self.assertEqual(await asyncio.gather(*waiters), [ok(n=1)]*3)
    self.assertLess(time.perf_counter()-st, 0.5)
    self.assertEqual(await self.bus.result(c.rid, 0), ok(n=1))

  async def test_unknown_and_dropped_runs_take_no_work(self):
    self.assertIsNone(await self.bus.submit(rid(), "x", b""))
    r = await self.new_run()
    await self.bus.drop(r, "p")
    self.assertIsNone(await self.bus.submit(r, "x", b""))
    self.assertEqual((await self.bus.runs("p"), await self.bus.queue(r)), ([], {"live": False}))
    self.assertFalse(await self.bus.r.exists(key("q", r)))

  async def test_the_queue_is_read_from_the_stream_itself(self):
    r = await self.new_run()
    rids = [await self.bus.submit(r, "x", b"") for _ in range(3)]
    self.assertEqual(await self.bus.queue(r), {"live": True, "queued": 3, "in_flight": []})
    [c] = await self.bus.take("p", "w0", count=1, block=10)
    q = await self.bus.queue(r)
    self.assertEqual((q["queued"], [(f["request_id"], f["trainer"], f["deliveries"]) for f in q["in_flight"]]), (2, [(rids[0], "w0", 1)]))
    await self.bus.finish(c, ok())
    self.assertEqual(await self.bus.queue(r), {"live": True, "queued": 2, "in_flight": []})

  async def test_a_new_trainer_reclaims_what_a_dead_one_took(self):
    r = await self.new_run()
    for _ in range(2): await self.bus.submit(r, "x", b"")
    taken = await self.bus.take("p", "dead", block=10)
    reclaimed = await self.bus.reclaim("p", "new")
    self.assertEqual([c.rid for c in reclaimed], [c.rid for c in taken])
    self.assertEqual({f["trainer"] for f in (await self.bus.queue(r))["in_flight"]}, {"new"})
    self.assertEqual(await self.bus.take("p", "new", block=10), [])

  async def test_every_run_in_a_pool_is_served_each_take(self):
    a, b, other = await self.new_run(), await self.new_run(), await self.new_run("other")
    for r in (a, a, a, b, other): await self.bus.submit(r, "x", b"")
    self.assertEqual(sorted(c.run for c in await self.bus.take("p", "w0", count=2, block=10)), sorted([a, a, b]))
    self.assertEqual([c.run for c in await self.bus.take("p", "w0", count=2, block=10)], [a])

  async def test_a_failure_is_counted(self):
    r = await self.new_run()
    await self.bus.submit(r, "optim_step", b"")
    [c] = await self.bus.take("p", "w0", block=10)
    await self.bus.finish(c, Reply(False, "application/json", b'{"error": "x"}'))
    self.assertEqual(((rec:=await self.bus.run(r))["steps"], rec["failed"]), (0, 1))

  async def test_sessions(self):
    await self.bus.session("s")
    r = await self.new_run(session="s")
    s = await self.bus.session_info("s")
    self.assertEqual((s["live"], s["runs"]), (True, [r]))
    self.assertIsNone(await self.bus.session_info("nope"))

if __name__ == "__main__":
  unittest.main()
