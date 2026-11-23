import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from tinyrl.control.node_daemon import LoadedRun, NodeDaemon, WorkerHandle
from tinyrl.types import NodePlan, WorkerSpec


class TestDetachRetry(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.control = SimpleNamespace(detach=AsyncMock(return_value=True))
    self.node = NodeDaemon(self.control, node="n", gpu_mib=0, store=Mock())
    self.model = SimpleNamespace(call=AsyncMock(), close=AsyncMock())
    self.worker = self.node.workers["w"] = WorkerHandle("w", WorkerSpec(lora=True), self.model)
    self.worker.loaded = {r: LoadedRun(dirty=True) for r in ("a", "b")}

  async def test_stop_saves_other_runs_and_retries_only_the_failed_snapshot(self):
    self.model.call.side_effect = [OSError("storage unavailable"), "snapshot-b", "snapshot-a"]
    with patch("tinyrl.control.node_daemon.TICK", 0):
      await self.node.run_op(self.worker, "stop")

    self.assertIs(self.node.workers["w"], self.worker)
    self.assertEqual(self.worker.loaded, {"a": LoadedRun(dirty=True)})
    self.model.close.assert_not_awaited()
    self.control.detach.assert_awaited_once_with("n", self.node.boot_id, "w", "b", "snapshot-b")

    await self.node.run_op(self.worker, "stop")
    self.assertEqual([call.args[:2] for call in self.model.call.await_args_list], [("save_run", r) for r in "aba"])
    self.assertEqual(self.control.detach.await_args.args[-2:], ("a", "snapshot-a"))
    self.assertEqual(self.node.workers, {})
    self.model.close.assert_awaited_once()

  async def test_detach_keeps_source_until_controller_acknowledges_snapshot(self):
    publishing, allow = asyncio.Event(), asyncio.Event()

    async def acknowledge(*args):
      publishing.set()
      await allow.wait()
      return True

    self.control.detach.side_effect = acknowledge
    self.model.call.return_value = "snapshot-a"
    detaching = asyncio.create_task(self.node.run_op(self.worker, "detach", "a"))
    try:
      await asyncio.wait_for(publishing.wait(), 1)
      self.assertTrue(self.worker.loaded["a"].dirty)
      self.assertEqual(self.model.call.await_count, 1)
    finally:
      allow.set()
      await detaching
    self.model.call.assert_awaited_with("drop_run", "a")
    self.assertEqual(self.worker.loaded, {"b": LoadedRun(dirty=True)})
    self.model.close.assert_not_awaited()

  async def test_permanent_save_error_never_publishes_a_snapshot_during_forced_drain(self):
    self.model.call.side_effect = OSError("storage unavailable")
    self.control.take = AsyncMock(return_value=NodePlan())
    self.node.begin_drain()
    with patch("tinyrl.control.node_daemon.TICK", 0.01), patch("tinyrl.control.node_daemon.DRAIN_S", 0.08):
      await asyncio.wait_for(self.node.serve(), 2)
    self.assertGreater(self.model.call.await_count, 2)
    self.control.detach.assert_not_awaited()
    self.model.close.assert_awaited_once()
