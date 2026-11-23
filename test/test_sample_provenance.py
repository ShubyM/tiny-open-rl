import json
import time
import unittest
from unittest.mock import AsyncMock

import numpy as np

from tinyrl.api import wire
from tinyrl.api.proto import tinker_public_pb2 as pb
from tinyrl.control.db import DB
from tinyrl.control.ledger import Ledger
from tinyrl.types import Op, Role, Run
from tinyrl.workers.sampler import Out, Sampler, Seq


class TestSampleProvenance(unittest.IsolatedAsyncioTestCase):
  def test_stability_budget_evicts_oldest_and_keeps_one_oversized_record(self):
    db = DB(":memory:")
    self.addCleanup(db.conn.close)
    now = time.time()
    with db:
      db.record_stability({"a": b"aaa", "b": b"bbb", "c": b"c"}, now + 60, 5)
      self.assertEqual(db.read_stability({"a", "b", "c"}, now), {"b": b"bbb", "c": b"c"})
      db.record_stability({"large": b"xxxxxx"}, now + 60, 5)
      self.assertEqual(db.read_stability({"b", "c", "large"}, now), {"large": b"xxxxxx"})

  async def test_late_attempt_cannot_publish_its_tokens_as_training_inputs(self):
    ledger = Ledger(":memory:")
    self.addAsyncCleanup(ledger.aclose)
    ledger.create_run(Run("sample", "base", 8, role=Role.SAMPLER))
    job = {
      "prompt": [0],
      "prompt_logprobs_last_n": None,
      "prompt_logprobs": False,
      "topk_prompt": 0,
      "topk_sample": 0,
      "record_stability_info": True,
    }
    rid = ledger.submit_command("sample", Op.SAMPLE, json.dumps(job).encode())
    cmd = next(ledger.samples.queued("sample", 1))
    worker = Sampler("base", True, engine=AsyncMock())
    worker.running = AsyncMock()
    worker.running.sample.side_effect = [Out([Seq([t], [-1.0], [{t: -1.0}], False)], []) for t in (1, 2)]
    previous, current = ("w1", "a", "b1"), ("w2", "b", "b2")

    ledger.samples.claim(cmd, previous, time.monotonic())
    early = await worker.execute(cmd)
    ledger.samples.reclaim(lambda entry: entry.owner == previous)
    retried = ledger.samples.claim(cmd, current, time.monotonic())
    winner = await worker.execute(cmd)
    self.assertTrue(ledger.finish_sample(rid, current, winner, attempt=retried.attempt))
    self.assertFalse(ledger.finish_sample(rid, previous, early))

    reply = await ledger.wait_for_reply(rid, 0)
    returned = pb.SampleResponse.FromString(reply.body).sequences[0].tokens
    self.assertIsNone(reply.stability)
    self.assertEqual(np.frombuffer(returned, np.int32).tolist(), [2])
    sid = f"{rid}:0"
    ledger.create_run(Run("train", "base", 8))
    req = ledger.submit_command("train", Op.FORWARD_BACKWARD, b"", stability_refs={sid})
    ids, _ = wire.unpack_stability(ledger.trains.cmd(req).stability[sid])
    self.assertEqual(ids.tolist(), [[2]])
