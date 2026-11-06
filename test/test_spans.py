import asyncio
import time
import unittest

from tinyrl import spans


class TestSpans(unittest.TestCase):
  def setUp(self):
    spans.report()

  def test_span_records_name_start_seconds_and_worker_and_nests(self):
    before = time.time()
    with spans.span("outer"):
      with spans.span("inner"):
        time.sleep(0.01)
    inner, outer = spans.report()
    self.assertEqual([inner[0], outer[0], inner[3]], ["inner", "outer", ""])
    self.assertGreaterEqual(inner[1], round(before, 4) - 1e-3)
    self.assertGreaterEqual(outer[2], inner[2] - 1e-3)
    self.assertGreaterEqual(inner[2], 0.009)
    self.assertEqual(spans.report(), [])

  def test_span_ends_even_when_its_body_raises(self):
    with self.assertRaises(ValueError), spans.span("failing"):
      raise ValueError
    self.assertEqual([s[0] for s in spans.report()], ["failing"])

  def test_timed_names_spans_after_functions_and_coroutines(self):
    @spans.timed
    def work(x):
      return x + 1

    @spans.timed
    async def awaited(x):
      await asyncio.sleep(0)
      return x * 2

    self.assertEqual(work(1), 2)
    self.assertEqual(asyncio.run(awaited(3)), 6)
    self.assertEqual([s[0] for s in spans.report()], ["work", "awaited"])

  def test_a_task_s_spans_carry_its_worker_even_from_threads_and_absorbed_reports(self):
    async def serve(worker):
      spans.worker.set(worker)
      await asyncio.to_thread(spans.timed(lambda: None))
      spans.absorb([["child", time.time(), 0.0, ""], ["theirs", time.time(), 0.0, "w9"]])

    async def run():
      await asyncio.gather(asyncio.create_task(serve("w1")), asyncio.create_task(serve("w2")))

    asyncio.run(run())
    self.assertEqual(
      sorted((s[0], s[3]) for s in spans.report()),
      sorted(
        [("<lambda>", "w1"), ("child", "w1"), ("theirs", "w9")]
        + [("<lambda>", "w2"), ("child", "w2"), ("theirs", "w9")]
      ),
    )


if __name__ == "__main__":
  unittest.main()
