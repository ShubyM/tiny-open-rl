import itertools
import unittest

from tinyrl.bus import Cmd, Op
from tinyrl.control.scheduler import select_runs


def head(run, entry="1-0", op=Op.FORWARD_BACKWARD):
  return Cmd(run, entry, op, b"")


class TestWorkPolicy(unittest.TestCase):
  def test_ties_use_numeric_stream_ids_then_run_ids(self):
    heads = [head("b", "10-0"), head("a", "9-10"), head("d", "9-2"), head("c", "9-2")]
    service = {cmd.run: 4.0 for cmd in heads}
    for order in itertools.permutations(heads):
      supplied = list(order)
      self.assertEqual(select_runs(supplied, service, 4), ["c", "d", "a", "b"])
      self.assertEqual(supplied, list(order))
    self.assertEqual(service, {cmd.run: 4.0 for cmd in heads})

  def test_width_limits_unique_runs_and_empty_ready_is_valid(self):
    heads = [head("a"), head("a"), head("b"), head("c")]
    self.assertEqual(select_runs(heads, {}, 2), ["a", "b"])
    self.assertEqual(select_runs(heads, {}, 10), ["a", "b", "c"])
    self.assertEqual(select_runs([], {}, 1), [])
    for width in (0, -1):
      with self.assertRaisesRegex(ValueError, "width"):
        select_runs(heads, {}, width)

  def test_costly_run_gets_fewer_turns_for_the_same_service(self):
    heads, service, chosen = [head("slow", "1-0"), head("fast", "2-0")], {"slow": 0.0, "fast": 0.0}, []
    for _ in range(11):
      run = select_runs(heads, service, 1)[0]
      chosen.append(run)
      service[run] += 10.0 if run == "slow" else 1.0
    self.assertEqual(chosen, ["slow"] + ["fast"] * 10)
    self.assertEqual(service, {"slow": 10.0, "fast": 10.0})

  def test_continuation_ends_when_caller_removes_preference(self):
    heads, service = [head("a"), head("b"), head("c")], {"a": 10.0, "b": 0.0, "c": 1.0}
    self.assertEqual(select_runs(heads, service, 2, prefer="a"), ["a", "b"])
    self.assertEqual(select_runs(heads, service, 2), ["b", "c"])
    self.assertEqual(select_runs(heads, service, 1, prefer="absent"), ["b"])
    self.assertEqual(service, {"a": 10.0, "b": 0.0, "c": 1.0})

  def test_saves_break_service_ties_and_cannot_skip_a_run_head(self):
    heads = [head("a"), head("b", "4-0", Op.SAVE_WEIGHTS_FOR_SAMPLER)]
    self.assertEqual(select_runs(heads, {"a": 0.0, "b": 1.0}, 1), ["a"])
    self.assertEqual(select_runs(heads, {"a": 1.0, "b": 1.0}, 1), ["b"])
    heads.append(head("a", "3-0", Op.SAVE_WEIGHTS_FOR_SAMPLER))
    self.assertEqual(select_runs(heads, {}, 2), ["b", "a"])

  def test_new_run_joins_active_service_floor(self):
    heads = [head("a", "3-0"), head("new", "4-0"), head("c", "2-0")]
    service = {"a": 10.0, "c": 20.0, "inactive": 0.0}
    self.assertEqual(select_runs(heads, service, 3), ["a", "new", "c"])
    self.assertNotIn("new", service)
    self.assertEqual(select_runs([head("new")], service, 1), ["new"])


if __name__ == "__main__":
  unittest.main()
