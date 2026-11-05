import asyncio
import json
import tempfile

from test.execution import TestNode
from test.helpers import RedisTest
from tinyrl.bus import Ev, Op, Run
from tinyrl.control import scheduler
from tinyrl.storage.store import Store


class TestEvents(RedisTest):
  async def test_round_trip(self):
    first = await self.bus.append_event("node_up", node="n", gpus=2, memory=0)
    second = await self.bus.append_event("node_lost", node="n")
    events = await self.bus.read_events()
    self.assertEqual([(e.id, e.kind) for e in events], [(first, "node_up"), (second, "node_lost")])
    self.assertEqual(events[0].body, {"node": "n", "gpus": 2, "memory": 0})
    self.assertLessEqual(events[0].t, events[1].t)
    self.assertEqual([e.id for e in await self.bus.read_events(first)], [second])

  async def test_a_kind_nothing_folds_cant_be_appended(self):
    with self.assertRaises(ValueError):
      await self.bus.append_event("grantd", worker="p")
    self.assertEqual(await self.bus.read_events(), [])

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

  async def test_gpu_turns_and_batches_stay_in_the_node_that_executes_them(self):
    node = TestNode(self.bus, store=Store(tempfile.mkdtemp()), gpus=1)
    try:
      await self.bus.create_run(Run("r", "p", "tiny", 4))
      await node.add("p", gpus=[0])
      request = await self.bus.submit_command(
        "r", Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": {}}).encode()
      )
      self.assertEqual(await node.step("p", block_ms=10), 1)
      self.assertTrue((await self.bus.wait_for_reply(request, 1)).ok)
      self.assertEqual(await node.step("p", block_ms=10), 0)
      worker = node.state.workers["p"]
      self.assertEqual(worker.on, ((node.id, 0),))
      self.assertTrue(worker.idle)
      self.assertGreater(worker.busy, 0)
      local = {Ev.STARTED, Ev.EXITED, Ev.WANTS_GPU, Ev.GRANTED, Ev.REVOKED, Ev.RELEASED, Ev.BATCH_DONE, Ev.IDLE}
      self.assertFalse([e for e in await self.bus.read_events() if e.kind in local])
    finally:
      await node.aclose()

  async def test_cluster_placement_names_the_node_and_leaves_gpu_selection_to_it(self):
    await self.bus.node_heartbeat("n", 2)
    await self.bus.append_event(Ev.NODE_UP, node="n", gpus=2, memory=0)
    await self.bus.create_run(Run("r", "p", "tiny", 4))
    self.renewal_tasks.append(asyncio.create_task(scheduler.run(self.bus)))
    for _ in range(100):
      placed = [e for e in await self.bus.read_events() if e.kind == Ev.PLACED]
      if placed:
        break
      await asyncio.sleep(0.02)
    self.assertEqual(len(placed), 1)
    self.assertEqual((placed[0].body["worker"], placed[0].body["node"]), ("p", "n"))
    self.assertNotIn("gpus", placed[0].body)
