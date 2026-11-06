import asyncio

from test.helpers import RedisTest
from tinyrl.bus import Run
from tinyrl.control import scheduler


class TestEvents(RedisTest):
  async def test_round_trip(self):
    first = await self.bus.append_event("node_up", node="n", gpus=2, memory=0)
    second = await self.bus.append_event("node_lost", node="n")
    events = await self.bus.read_events()
    self.assertEqual([(e.id, e.kind) for e in events], [(first, "node_up"), (second, "node_lost")])
    self.assertEqual(events[0].body, {"node": "n", "gpus": 2, "memory": 0})
    self.assertLessEqual(events[0].t, events[1].t)
    self.assertEqual([e.id for e in await self.bus.read_events(first)], [second])

  async def test_the_scheduler_assigns_new_runs_and_the_log_has_the_worker_membership(self):
    self.renewal_tasks.append(asyncio.create_task(scheduler.run(self.bus)))
    for run in ("a", "b"):
      await self.bus.create_run(Run(run, "", "tiny", 4))
    await self.bus.create_run(Run("f", "", "tiny", 0))
    for _ in range(100):
      runs = {r: (await self.bus.get_run(r))["worker"] for r in ("a", "b", "f")}
      if all(runs.values()):
        break
      await asyncio.sleep(0.02)
    self.assertEqual(runs["a"], runs["b"])
    self.assertNotEqual(runs["a"], runs["f"])
    state = scheduler.State()
    for event in await self.bus.read_events():
      state = scheduler.fold(state, event)
    self.assertEqual(sorted(scheduler.members(state, runs["a"])), ["a", "b"])
    self.assertEqual(scheduler.members(state, runs["f"]), ["f"])
