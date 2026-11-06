import copy
import json
import unittest
from unittest.mock import patch

from k8s import e2e


class TestChaosDiagnostics(unittest.TestCase):
  def test_live_workers_report_physical_location_and_ignore_expired_nodes(self):
    state = {"kind": {"role": "trainer"}, "pid": 40, "node": "new-pod-1"}
    rows = [
      "old-pod-1",
      json.dumps({"until": 101, "workers": {"worker": state}}),
      "dead-pod-1",
      json.dumps({"until": 99, "workers": {"worker": state}}),
    ]
    with patch.object(e2e, "redis", return_value="\n".join(rows)), patch.object(e2e.time, "time", return_value=100):
      self.assertEqual(e2e.live_workers(), {("old-pod", "worker"): state})

  def test_process_listing_keeps_parent_and_group_for_detached_ranks(self):
    output = "    40 1 40 python -m tinyrl.control.process python -m tinyrl.workers.trainer /tmp/model.sock\n"
    output += "   42 41 42 python -m tinyrl.workers.trainer /tmp/model.sock\n"
    with patch.object(e2e, "kubectl", return_value=output):
      ps = e2e.processes("pod")
    self.assertEqual(ps[40][:2], (1, 40))
    self.assertEqual(ps[42][:2], (41, 42))
    self.assertEqual(ps[42][2], "python -m tinyrl.workers.trainer /tmp/model.sock")

  def test_trainer_target_is_the_reported_supervisor_and_rejects_a_reused_pid(self):
    state = {"kind": {"role": "trainer"}, "pid": 40}
    with (
      patch.object(e2e, "serving", return_value="pod"),
      patch.object(e2e, "live_workers", return_value={("pod", "worker"): state}),
      patch.object(e2e, "processes") as ps,
    ):
      ps.return_value = {
        40: (1, 40, "python -m tinyrl.control.process python -m tinyrl.workers.trainer /tmp/model.sock")
      }
      self.assertEqual(e2e.trainer_process("worker"), ("pod", 40))
      ps.return_value = {40: (1, 40, "python -m unrelated_service")}
      with self.assertRaises(AssertionError):
        e2e.trainer_process("worker")

  def setUp(self):
    self.placed = {"trainer": "pod", "sampler": "pod", "idle-sampler": "pod"}
    self.workers = {
      ("pod", "trainer"): {"kind": {"role": "trainer"}, "pid": 10},
      ("pod", "sampler"): {"kind": {"role": "sampler"}, "pid": 20},
      ("pod", "idle-sampler"): {"kind": {"role": "sampler"}, "pid": None},
    }
    self.procs = {
      "pod": {
        1: (0, 1, "python -m tinyrl.control.node_daemon"),
        10: (1, 10, "python -m tinyrl.control.process python -m torch.distributed.run -m tinyrl.workers.trainer sock"),
        11: (10, 10, "python -m torch.distributed.run -m tinyrl.workers.trainer sock"),
        12: (11, 12, "python -m tinyrl.workers.trainer sock"),
        13: (11, 13, "python -m tinyrl.workers.trainer sock"),
        20: (1, 20, "python -m tinyrl.control.process unshare --pid python -m tinyrl.workers.snapshot saved gpu"),
        21: (20, 20, "unshare --pid python -m tinyrl.workers.snapshot saved gpu"),
        22: (21, 22, "vllm serve model"),
        23: (22, 23, "VLLM::EngineCore"),
      }
    }

  def test_multi_rank_trainers_and_restored_engines_are_one_owned_group_each(self):
    e2e.check_processes(self.placed, self.workers, self.procs, list(self.placed))

  def test_missing_or_shared_supervisors_and_misplaced_workers_fail(self):
    variants = []
    missing = copy.deepcopy(self.workers)
    missing[("pod", "trainer")]["pid"] = None
    variants.append(missing)
    shared = copy.deepcopy(self.workers)
    shared[("pod", "sampler")]["pid"] = 10
    variants.append(shared)
    misplaced = copy.deepcopy(self.workers)
    misplaced[("other-pod", "trainer")] = misplaced.pop(("pod", "trainer"))
    variants.append(misplaced)
    for workers in variants:
      with self.subTest(workers=workers), self.assertRaises(AssertionError):
        e2e.check_processes(self.placed, workers, self.procs, list(self.placed))

  def test_unreported_supervisors_and_orphaned_model_processes_fail(self):
    for args in (
      "python -m tinyrl.control.process vllm serve other",
      "VLLM::EngineCore",
      "python -m tinyrl.workers.trainer sock",
    ):
      procs = copy.deepcopy(self.procs)
      procs["pod"][30] = (1, 30, args)
      with self.subTest(args=args), self.assertRaises(AssertionError):
        e2e.check_processes(self.placed, self.workers, procs, list(self.placed))

  def test_missing_pods_and_stray_leases_fail(self):
    with self.assertRaises(AssertionError):
      e2e.check_processes(self.placed, self.workers, {}, list(self.placed))
    with self.assertRaises(AssertionError):
      e2e.check_processes(self.placed, self.workers, self.procs, [*self.placed, "old-worker"])


if __name__ == "__main__":
  unittest.main()
