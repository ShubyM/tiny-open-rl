import json
from unittest.mock import patch

from test.helpers import RedisTest, new_id, tiny
from test.test_scheduler import check
from tinyrl import scheduler
from tinyrl.bus import Op, Run
from tinyrl.sampler import Sampler
from tinyrl.trainer import Trainer


class TestEvents(RedisTest):
  async def test_round_trip(self):
    first = await self.bus.emit("started", pool="p", node="n", gpus=[0], memory=0)
    second = await self.bus.emit("exited", pool="p")
    events = await self.bus.events()
    self.assertEqual([(e.id, e.kind) for e in events], [(first, "started"), (second, "exited")])
    self.assertEqual(events[0].body, {"pool": "p", "node": "n", "gpus": [0], "memory": 0})
    self.assertLessEqual(events[0].t, events[1].t)
    self.assertEqual([e.id for e in await self.bus.events(first)], [second])

  async def test_each_worker_appends_its_facts_in_order_and_the_log_folds_soundly(self):
    run_id = new_id()
    await self.run_scheduler({"lora:tiny:1": [0], "sampler:tiny": [1]})
    await self.bus.create(Run(run_id, "lora:tiny:1", "tiny", 4, exclusive=False))

    with patch("tinyrl.trainer.load", side_effect=tiny), patch("tinyrl.trainer.GPU", "0"):
      trainer = Trainer(self.bus, "lora:tiny:1", worker="tw1")
      await self.own(trainer)

      await self.bus.submit(run_id, Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": {}}).encode())

      await trainer.step()

      await trainer.step()

    with patch("tinyrl.sampler.ENGINE", "test.helpers:FakeEngine"), patch("tinyrl.sampler.GPU", "1"):
      sampler = Sampler(self.bus, "sampler:tiny", worker="sw1")
      run_id2 = new_id()
      await self.bus.create(Run(run_id2, "sampler:tiny", "tiny", 0))
      req = {
        "prompt": [1],
        "max_tokens": 1,
        "num_samples": 1,
        "prompt_logprobs": False,
        "prompt_logprobs_last_n": 0,
        "topk_sample": 1,
        "topk_prompt": 0,
      }
      await self.bus.submit(run_id2, Op.SAMPLE, json.dumps(req).encode())
      await self.own(sampler)

      await sampler.step()
      await sampler.step()

    events = await self.bus.events()
    turns = ("wants_gpu", "granted", "batch_done", "idle")
    for pool in ("lora:tiny:1", "sampler:tiny"):
      self.assertEqual([e.kind for e in events if e.body.get("pool") == pool and e.kind in turns], list(turns))
    s = scheduler.State()
    for e in events:
      s = scheduler.fold(s, e)
    self.assertEqual(check(s), [])
