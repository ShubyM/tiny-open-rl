import asyncio
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from tinyrl.trace import trace


class TestTrace(unittest.IsolatedAsyncioTestCase):
  def test_disabled_needs_no_nvtx_and_preserves_errors(self):
    with patch("tinyrl.trace.getenv", return_value=0), patch.dict(sys.modules, {"nvtx": None}):
      with trace("execute", "run/1"):
        pass
      with self.assertRaisesRegex(ValueError, "execution failed"), trace("execute", "run/2"):
        raise ValueError("execution failed")

  def test_same_request_correlates_across_process_labels(self):
    nvtx = SimpleNamespace(start_range=Mock(side_effect=[(1, 7), (2, 7)]), end_range=Mock())
    with patch("tinyrl.trace.getenv", return_value=1), patch.dict(sys.modules, {"nvtx": nvtx}):
      with trace("node.execute", "run/1"):
        pass
      with self.assertRaisesRegex(ValueError, "execution failed"), trace("trainer.execute", "run/1"):
        raise ValueError("execution failed")
    self.assertEqual(
      nvtx.start_range.call_args_list,
      [
        call("node.execute", domain="tinyrl", payload=b"run/1"),
        call("trainer.execute", domain="tinyrl", payload=b"run/1"),
      ],
    )
    self.assertEqual(nvtx.end_range.call_args_list, [call((1, 7)), call((2, 7))])

  async def test_overlapping_tasks_close_their_own_ranges_on_cancel(self):
    nvtx = SimpleNamespace(start_range=Mock(side_effect=[(1, 7), (2, 7)]), end_range=Mock())
    started, finish = [asyncio.Event(), asyncio.Event()], asyncio.Event()

    async def work(i):
      with trace("execute", f"run/{i}"):
        started[i].set()
        await finish.wait()

    with patch("tinyrl.trace.getenv", return_value=1), patch.dict(sys.modules, {"nvtx": nvtx}):
      a = asyncio.create_task(work(0))
      await started[0].wait()
      b = asyncio.create_task(work(1))
      await started[1].wait()
      a.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await a
      self.assertFalse(b.done())
      nvtx.end_range.assert_called_once_with((1, 7))
      finish.set()
      await b
    self.assertEqual(nvtx.end_range.call_args_list, [call((1, 7)), call((2, 7))])


if __name__ == "__main__":
  unittest.main()
