from test.helpers import BusTest
from tinyrl.bus import Run
from tinyrl.control import scheduler
from tinyrl.types import Ev


class TestEvents(BusTest):
  async def test_round_trip(self):
    first = self.bus.record_event(Ev.NODE_UP, {"node": "n", "gpus": 2, "memory": 0}).id
    second = self.bus.record_event(Ev.NODE_LOST, {"node": "n"}).id
    events = await self.bus.read_events()
    self.assertEqual([(e.id, e.kind) for e in events], [(first, "node_up"), (second, "node_lost")])
    self.assertEqual(events[0].body, {"node": "n", "gpus": 2, "memory": 0})
    self.assertLessEqual(events[0].t, events[1].t)
    self.assertEqual([e.id for e in await self.bus.read_events(first)], [second])

  async def test_the_scheduler_assigns_new_runs_and_the_log_has_the_process_membership(self):
    for run in ("a", "b"):
      await self.bus.create_run(Run(run, "tiny", 4))
    await self.bus.create_run(Run("f", "tiny", 0))
    await self.bus.maintain()
    state = scheduler.from_state(await self.bus.read_state())
    home = state.home
    self.assertEqual(home["a"], home["b"])
    self.assertNotEqual(home["a"], home["f"])
    self.assertEqual(state.runs(home["a"]), ["a", "b"])
    self.assertEqual(state.runs(home["f"]), ["f"])
