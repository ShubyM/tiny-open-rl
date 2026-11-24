import unittest

from tinyrl.control.controller import Controller
from tinyrl.types import MAX_SEQS, Checkpoint, Done, NodeMsg, Op, Reply, Role, Run, TaskKind, WorkerInfo


class TestSampleFailover(unittest.IsolatedAsyncioTestCase):
  async def test_unstarted_claim_can_move_on_cache_loss_and_fences_all_earlier_attempts(self):
    ctrl = Controller(":memory:")
    self.addAsyncCleanup(ctrl.aclose)
    uri, version = "tinker://train/sampler_weights/latest", "v1"
    ctrl.put_checkpoint(Checkpoint(uri, version, 1))
    for node in ("east", "west"):
      ctrl.handle(NodeMsg("hello", node, node, gpus=1))
    ctrl.create_run(Run("sample", "base", 8, role=Role.SAMPLER, model_path=uri))
    ids = [ctrl.submit_command("sample", Op.SAMPLE, b"{}") for _ in range(MAX_SEQS + 1)]
    first = [
      t
      for t in ctrl.handle(NodeMsg("hello", "east", "east", gpus=1, staged=(version,))).tasks
      if t.kind == TaskKind.SAMPLE
    ]
    running, unseen = first[:2]
    self.assertEqual(running.cmd.request_id, ids[0])

    # Cache eviction leaves one sample executing; unstarted siblings fail over to the staged replica.
    wi = WorkerInfo(running.worker, running.spec, running.gpus, running.gpus)
    evicted = ctrl.handle(NodeMsg("hello", "east", "east", gpus=1, workers=(wi,), active=(running.id,)))
    self.assertIn(unseen.id, evicted.cancel)
    west = [
      t
      for t in ctrl.handle(NodeMsg("hello", "west", "west", gpus=1, staged=(version,))).tasks
      if t.kind == TaskKind.SAMPLE
    ]
    west_claim = next(t for t in west if t.cmd.request_id == unseen.cmd.request_id)
    self.assertNotIn(running.cmd.request_id, [t.cmd.request_id for t in west])
    ok = Reply(True, "application/json", b"{}")
    ctrl.handle(NodeMsg("events", "east", "east", done=(Done(running.id, True, ok),)))
    self.assertEqual(await ctrl.wait_for_reply(running.cmd.request_id, 0), ok)
    self.assertFalse(
      ctrl.finish_sample(unseen.cmd.request_id, (unseen.worker, "east", "east"), ok, None, unseen.attempt)
    )

    # Return to the original boot and worker; every earlier attempt stays fenced.
    ctrl.handle(NodeMsg("hello", "west", "west", gpus=1))
    east = [
      t
      for t in ctrl.handle(NodeMsg("hello", "east", "east", gpus=1, workers=(wi,), staged=(version,))).tasks
      if t.kind == TaskKind.SAMPLE
    ]
    current = next(t for t in east if t.cmd.request_id == unseen.cmd.request_id)
    self.assertEqual(current.worker, running.worker)
    self.assertEqual(current.version, version)
    self.assertLess(unseen.attempt, west_claim.attempt)
    self.assertLess(west_claim.attempt, current.attempt)
    self.assertFalse(ctrl.finish_sample(unseen.cmd.request_id, (current.worker, "east", "east"), ok, None, 0))
    self.assertFalse(
      ctrl.finish_sample(unseen.cmd.request_id, (west_claim.worker, "west", "west"), ok, None, west_claim.attempt)
    )
    self.assertTrue(
      ctrl.finish_sample(unseen.cmd.request_id, (current.worker, "east", "east"), ok, None, current.attempt)
    )
    self.assertEqual(await ctrl.wait_for_reply(unseen.cmd.request_id, 0), ok)

  async def test_unstaged_old_revision_does_not_block_later_samples_in_same_run(self):
    ctrl = Controller(":memory:")
    self.addAsyncCleanup(ctrl.aclose)
    uri = "tinker://train/sampler_weights/latest"
    ctrl.create_run(Run("sample", "base", 8, role=Role.SAMPLER, model_path=uri))
    ids = []
    for version in ("v1", "v2", "v1", "v2"):
      ctrl.put_checkpoint(Checkpoint(uri, version, 1))
      ids.append(ctrl.submit_command("sample", Op.SAMPLE, b"{}"))
    r1 = ctrl.handle(NodeMsg("hello", "n", "boot", gpus=1))
    self.assertEqual(set(r1.stage), {"v1", "v2"})
    r2 = ctrl.handle(NodeMsg("hello", "n", "boot", gpus=1, staged=("v2",)))
    samples = [t for t in r2.tasks if t.kind == TaskKind.SAMPLE]
    self.assertEqual([t.cmd.request_id for t in samples], [ids[1], ids[3]])
    self.assertEqual(set(r2.stage), {"v1"})
