import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from tinyrl.control.controller import Controller
from tinyrl.control.node_daemon import NodeDaemon, WorkerHandle
from tinyrl.types import Run, RunStatus, TaskKind, WorkerSpec


class TestDetachRetry(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.ctrl = Controller(":memory:")
    self.addAsyncCleanup(self.ctrl.aclose)
    store = Mock()
    store.env.return_value = {}
    store.versions.return_value = set()
    self.node = NodeDaemon(self.ctrl, gpus=1, node="n", gpu_mib=0, store=store)
    self.model = SimpleNamespace(call=AsyncMock(), close=AsyncMock(), proc=None)
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
    await self.node.step_once()  # issues SAVE a -> fails
    await asyncio.sleep(0.01)
    await self.node.step_once()  # reports failed a -> immediately issues SAVE b
    await asyncio.sleep(0.01)
    await self.node.step_once()  # commits snap-b; b detached, a still owned & dirty
    self.assertEqual(self.ctrl.get_run("b")["snapshot"], "snap-b")
    self.assertIsNone(self.ctrl.get_run("a").get("snapshot"))
    self.model.close.assert_not_awaited()

    await asyncio.sleep(0.06)
    await self.node.step_once()  # retries SAVE a -> succeeds
    await asyncio.sleep(0.01)
    await self.node.step_once()  # commits snap-a -> issues STOP
    await asyncio.sleep(0.01)
    await self.node.step_once()  # STOP finishes
    self.assertEqual([c.args[:2] for c in self.model.call.await_args_list], [("save_run", r) for r in "aba"])
    self.assertEqual(self.ctrl.get_run("a")["snapshot"], "snap-a")
    self.assertEqual(self.node.workers, {})
    self.model.close.assert_awaited_once()

  async def test_detach_keeps_source_until_controller_acknowledges_snapshot(self):
    self.model.call.return_value = "snap-b"
    self.ctrl._detach_one(self.ctrl.nodes["n"], self.ctrl.nodes["n"].workers["w"], "b", 0.0)
    cmsg = self.ctrl.nodes["n"].drain_push()
    self.assertEqual([t.kind for t in cmsg.tasks], [TaskKind.SAVE])
    self.node.apply(cmsg)
    await asyncio.sleep(0.01)
    self.assertIn("b", self.worker.loaded)
    self.assertEqual([c.args[0] for c in self.model.call.await_args_list], ["save_run"])

    await self.node.step_once()  # commits snap-b -> issues DROP b
    self.assertEqual(self.ctrl.get_run("b")["snapshot"], "snap-b")
    await asyncio.sleep(0.01)
    await self.node.step_once()  # DROP b completes
    self.model.call.assert_awaited_with("drop_run", "b")
    self.assertEqual(set(self.worker.loaded), {"a"})
    self.model.close.assert_not_awaited()

  async def test_permanent_save_error_never_publishes_a_snapshot_during_forced_drain(self):
    self.model.call.side_effect = OSError("storage unavailable")
    self.node.begin_drain()
    with patch("tinyrl.control.node_daemon.TICK", 0.01), patch("tinyrl.control.node_daemon.DRAIN_S", 0.15):
      await asyncio.wait_for(self.node.serve(), 2)
    self.assertGreater(self.model.call.await_count, 2)
    self.assertIsNone(self.ctrl.get_run("a").get("snapshot"))
    self.assertIsNone(self.ctrl.get_run("b").get("snapshot"))
    self.model.close.assert_awaited_once()
