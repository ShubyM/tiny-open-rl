import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from test.helpers import RedisTest, new_id, tiny
from tinyrl.agent import Agent
from tinyrl.bus import Op, Run

LEASE_MS = 2000
LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


class TestAgent(RedisTest):
  async def test_the_agent_starts_a_trainer_for_a_pool_with_runs_and_again_when_it_dies(self):
    base = f"{tempfile.mkdtemp()}/tiny"
    tiny().save_pretrained(base)
    pool, run = f"lora:{base}", new_id()
    await self.bus.create(Run(run, pool, base, 4))
    await self.bus.create(Run(new_id(), "empty", base, 4))
    await self.bus.drop((await self.bus.runs("empty"))[0], "empty")

    with patch.dict(os.environ, {"REDIS_URL": self.url, "LEASE_MS": str(LEASE_MS)}):
      agent = Agent(self.bus, gpus=2)
      try:
        await agent.tick()
        self.assertEqual(list(agent.trainers), [pool])  # not the pool whose runs are all gone
        request_id = await self.bus.submit(
          run, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 4, "lora": LORA}).encode()
        )
        self.assertTrue((await self.bus.result(request_id, 60)).ok)

        gpu, first = agent.trainers[pool]
        await agent.tick()
        self.assertIs(agent.trainers[pool][1], first)  # one trainer per pool

        first.kill()
        await asyncio.to_thread(first.wait)
        while await self.bus.owner(pool):  # its lease runs out
          await asyncio.sleep(0.1)
        await agent.tick()
        self.assertIsNot(agent.trainers[pool][1], first)
        request_id = await self.bus.submit(
          run, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 4, "lora": LORA}).encode()
        )
        self.assertTrue((await self.bus.result(request_id, 60)).ok)
      finally:
        for _, proc in agent.trainers.values():
          proc.terminate()
          proc.wait()

  def test_each_trainer_gets_the_least_used_gpu(self):
    agent = Agent(None, gpus=2)
    with patch("subprocess.Popen") as popen:
      agent.trainers = {"a": agent.start("a")}
      agent.trainers["b"] = agent.start("b")
      agent.trainers["c"] = agent.start("c")
    gpus = [call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for call in popen.call_args_list]
    self.assertEqual(gpus, ["0", "1", "0"])


if __name__ == "__main__":
  unittest.main()
