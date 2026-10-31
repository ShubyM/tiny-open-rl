import asyncio
import json
from unittest.mock import patch

from test.helpers import RedisTest, new_id, tiny
from test.test_scheduler import check
from tinyrl.bus import Op, Run
from tinyrl.control import scheduler
from tinyrl.workers.sampler import Sampler
from tinyrl.workers.trainer import Trainer


class TestEvents(RedisTest):
  async def test_round_trip(self):
    first = await self.bus.emit("started", worker="p", node="n", gpus=[0], memory=0)
    second = await self.bus.emit("exited", worker="p")
    events = await self.bus.events()
    self.assertEqual([(e.id, e.kind) for e in events], [(first, "started"), (second, "exited")])
    self.assertEqual(events[0].body, {"worker": "p", "node": "n", "gpus": [0], "memory": 0})
    self.assertLessEqual(events[0].t, events[1].t)
    self.assertEqual([e.id for e in await self.bus.events(first)], [second])

  async def test_the_scheduler_assigns_new_runs_and_workers_find_them_in_redis(self):
    self.renewal_tasks.append(asyncio.create_task(scheduler.run(self.bus)))
    for run in ("a", "b"):
      await self.bus.create(Run(run, "", "tiny", 4))
    await self.bus.create(Run("f", "", "tiny", 0))
    for _ in range(100):
      runs = {r: (await self.bus.run(r))["worker"] for r in ("a", "b", "f")}
      if all(runs.values()):
        break
      await asyncio.sleep(0.02)
    self.assertEqual(runs["a"], runs["b"])
    self.assertNotEqual(runs["a"], runs["f"])
    self.assertEqual(sorted(await self.bus.runs(runs["a"])), ["a", "b"])
    self.assertEqual(await self.bus.runs(runs["f"]), ["f"])

  async def test_each_worker_appends_its_facts_in_order_and_the_log_folds_soundly(self):
    run_id = new_id()
    await self.run_scheduler({"lora:tiny:1": [0], "sampler:tiny": [1]})
    await self.bus.create(Run(run_id, "lora:tiny:1", "tiny", 4, exclusive=False))

    with patch("tinyrl.workers.model.load", side_effect=tiny), patch("tinyrl.workers.trainer.GPU", "0"):
      trainer = Trainer(self.bus, "lora:tiny:1", owner="tw1")
      await self.own(trainer)

      await self.bus.submit(run_id, Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": {}}).encode())

      await trainer.step()

      await trainer.step()

    with patch("tinyrl.workers.sampler.ENGINE", "test.helpers:FakeEngine"), patch("tinyrl.workers.sampler.GPU", "1"):
      sampler = Sampler(self.bus, "sampler:tiny", owner="sw1")
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
    for worker in ("lora:tiny:1", "sampler:tiny"):
      self.assertEqual([e.kind for e in events if e.body.get("worker") == worker and e.kind in turns], list(turns))
    s = scheduler.State()
    for e in events:
      s = scheduler.fold(s, e)
    self.assertEqual(check(s), [])
