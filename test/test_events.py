from test.helpers import BusTest
from tinyrl.types import Ev, NodeMsg, Op, Run


class TestEvents(BusTest):
  async def test_round_trip(self):
    first = self.bus.record_event(Ev.NODE_UP, {"node": "n", "gpus": 2, "memory": 0}).id
    second = self.bus.record_event(Ev.NODE_LOST, {"node": "n"}).id
    events = self.bus.read_events()
    self.assertEqual([(e.id, e.kind) for e in events], [(first, "node_up"), (second, "node_lost")])
    self.assertEqual(events[0].body, {"node": "n", "gpus": 2, "memory": 0})
    self.assertLessEqual(events[0].t, events[1].t)
    self.assertEqual([e.id for e in self.bus.read_events(first)], [second])

  async def test_compatible_lora_runs_share_a_worker_while_full_finetune_gets_its_own(self):
    self.bus.handle(NodeMsg("hello", "n", "b", gpus=1))
    for run in ("a", "b"):
      self.bus.create_run(Run(run, "tiny", 4))
      self.bus.submit_command(run, Op.CREATE_MODEL, b"{}")
    self.bus.create_run(Run("f", "tiny", 0))
    self.bus.submit_command("f", Op.CREATE_MODEL, b"{}")
    ws = self.bus.nodes["n"].workers
    self.assertEqual(len(ws), 2)
    self.assertEqual(self.bus.runs["a"].owner[0], [w.id for w in ws.values() if w.spec.lora][0])
    self.assertNotEqual([w.id for w in ws.values() if w.spec.lora], [w.id for w in ws.values() if not w.spec.lora])
