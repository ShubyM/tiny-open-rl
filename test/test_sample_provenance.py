import json
import tempfile
import time
import unittest
from unittest.mock import AsyncMock

import numpy as np

from tinyrl.api import wire
from tinyrl.api.proto import tinker_public_pb2 as pb
from tinyrl.control.controller import Controller
from tinyrl.control.db import DB
from tinyrl.storage.store import Store
from tinyrl.types import NodeMsg, Op, Role, Run, TaskKind
from tinyrl.workers.sampler import Out, Sampler, Seq


class TestSampleProvenance(unittest.IsolatedAsyncioTestCase):
  def test_stability_budget_evicts_oldest_and_keeps_one_oversized_record(self):
    db = DB(":memory:")
    self.addCleanup(db.close)
    now = time.time()
    with db:
      db.record_stability({"a": ("r1", 0, 3), "b": ("r1", 3, 3), "c": ("r1", 6, 1)}, now + 60, 5)
      self.assertEqual(db.read_stability({"a", "b", "c"}, now), {"b": ("r1", 3, 3), "c": ("r1", 6, 1)})
      db.record_stability({"large": ("r2", 0, 6)}, now + 60, 5)
      self.assertEqual(db.read_stability({"b", "c", "large"}, now), {"large": ("r2", 0, 6)})

  async def test_late_attempt_cannot_publish_its_tokens_as_training_inputs(self):
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    store = Store(root=tmp.name)
    ctrl = Controller(":memory:")
    self.addAsyncCleanup(ctrl.aclose)
    ctrl.create_run(Run("sample", "base", 8, role=Role.SAMPLER))
    job = {
      "prompt": [0],
      "prompt_logprobs_last_n": None,
      "prompt_logprobs": False,
      "topk_prompt": 0,
      "topk_sample": 0,
      "record_stability_info": True,
    }
    rid = ctrl.submit_command("sample", Op.SAMPLE, json.dumps(job).encode())
    worker = Sampler("base", True, engine=AsyncMock(), store=store)
    worker.running = AsyncMock()
    worker.running.sample.side_effect = [Out([Seq([t], [-1.0], [{t: -1.0}], False)], []) for t in (1, 2)]

    [t1] = [t for t in ctrl.handle(NodeMsg("hello", "a", "b1", gpus=1)).tasks if t.kind == TaskKind.SAMPLE]
    early = await worker.execute(t1.cmd)
    ctrl.expire_node("a", "b1")
    [t2] = [t for t in ctrl.handle(NodeMsg("hello", "b", "b2", gpus=1)).tasks if t.kind == TaskKind.SAMPLE]
    winner = await worker.execute(t2.cmd)
    self.assertTrue(ctrl.finish_sample(rid, (t2.worker, "b", "b2"), winner, None, t2.attempt))
    self.assertFalse(ctrl.finish_sample(rid, (t1.worker, "a", "b1"), early, None, t1.attempt))

    reply = await ctrl.wait_for_reply(rid, 0)
    self.assertIsNone(reply.stability)
    returned = pb.SampleResponse.FromString(reply.body).sequences[0].tokens
    self.assertEqual(np.frombuffer(returned, np.int32).tolist(), [2])
    sid = f"{rid}:0"
    ctrl.create_run(Run("train", "base", 8))
    req = ctrl.submit_command("train", Op.FORWARD_BACKWARD, b"", stability_refs={sid})
    ref, off, ln = ctrl.hydrate_cmd(req).stability[sid]
    ids, _ = wire.unpack_stability(store.get_blob(ref)[off : off + ln])
    self.assertEqual(ids.tolist(), [[2]])
