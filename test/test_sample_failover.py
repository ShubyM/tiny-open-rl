import unittest

from tinyrl.control.ledger import Ledger
from tinyrl.types import MAX_SEQS, Checkpoint, LocalWorker, NodeStatus, Op, Reply, Role, Run


class TestSampleFailover(unittest.IsolatedAsyncioTestCase):
  async def test_unstarted_claim_can_move_on_cache_loss_and_fences_all_earlier_attempts(self):
    ledger = Ledger(":memory:")
    self.addAsyncCleanup(ledger.aclose)
    uri, version = "tinker://train/sampler_weights/latest", "v1"
    with ledger.db:
      ledger.checkpoints[uri] = Checkpoint(uri, version, 1)
    cold, warm = NodeStatus(1), NodeStatus(1, staged=(version,))
    for node in ("east", "west"):
      await ledger.take(node, node, cold)
    ledger.create_run(Run("sample", "base", 8, role=Role.SAMPLER, model_path=uri))
    ids = [ledger.submit_command("sample", Op.SAMPLE, b"{}") for _ in range(MAX_SEQS + 1)]
    [first] = [g for g in (await ledger.take("east", "east", warm)).grants if g.work]
    running, unseen = first.work[:2]
    self.assertEqual(running.cmd.request_id, ids[0])

    # Cache eviction leaves one sample executing. Its unstarted siblings can run on the staged replica.
    local = LocalWorker(current=first.gpus, holding=(running.cmd.request_id,))
    await ledger.take("east", "east", NodeStatus(1, workers={first.worker: local}))
    [west] = [g for g in (await ledger.take("west", "west", warm)).grants if g.work]
    west_claim = next(w for w in west.work if w.cmd.request_id == unseen.cmd.request_id)
    self.assertNotIn(running.cmd.request_id, [w.cmd.request_id for w in west.work])
    ok = Reply(True, "application/json", b"{}")
    self.assertTrue(ledger.finish("east", "east", first.worker, running.cmd.request_id, ok, attempt=running.attempt))
    self.assertFalse(ledger.finish("east", "east", first.worker, unseen.cmd.request_id, ok, attempt=unseen.attempt))

    # Return to the same boot and worker that owned the initial attempt.
    await ledger.take("west", "west", cold)
    [east] = [g for g in (await ledger.take("east", "east", warm)).grants if g.work]
    current = next(w for w in east.work if w.cmd.request_id == unseen.cmd.request_id)
    self.assertEqual(east.worker, first.worker)
    self.assertEqual(current.version, version)
    self.assertLess(unseen.attempt, west_claim.attempt)
    self.assertLess(west_claim.attempt, current.attempt)
    self.assertFalse(ledger.finish("east", "east", east.worker, unseen.cmd.request_id, ok))
    self.assertFalse(ledger.finish("west", "west", west.worker, unseen.cmd.request_id, ok, attempt=west_claim.attempt))
    self.assertTrue(ledger.finish("east", "east", east.worker, unseen.cmd.request_id, ok, attempt=current.attempt))
    self.assertEqual(await ledger.wait_for_reply(unseen.cmd.request_id, 0), ok)

  async def test_unstaged_old_revision_does_not_block_later_samples_in_same_run(self):
    ledger = Ledger(":memory:")
    self.addAsyncCleanup(ledger.aclose)
    uri = "tinker://train/sampler_weights/latest"
    ledger.create_run(Run("sample", "base", 8, role=Role.SAMPLER, model_path=uri))
    ids = []
    for version in ("v1", "v2", "v1", "v2"):
      with ledger.db:
        ledger.checkpoints[uri] = Checkpoint(uri, version, 1)
      ids.append(ledger.submit_command("sample", Op.SAMPLE, b"{}"))
    plan = await ledger.take("n", "boot", NodeStatus(1))
    self.assertEqual(set(plan.stage), {"v1", "v2"})
    plan = await ledger.take("n", "boot", NodeStatus(1, staged=("v2",)))
    [grant] = [g for g in plan.grants if g.work]
    self.assertEqual([w.cmd.request_id for w in grant.work], [ids[1], ids[3]])
    self.assertEqual(plan.stage, ("v1",))
