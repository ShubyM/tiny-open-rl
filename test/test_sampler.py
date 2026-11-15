import asyncio
import json
import tempfile
import unittest

import httpx

from test.execution import TestNode
from test.helpers import FakeEngine, RedisTest, tiny
from tinyrl.api import wire
from tinyrl.api.gateway import app
from tinyrl.bus import Op, Role, Run, key
from tinyrl.storage.store import Store
from tinyrl.workers.sampler import Out, Seq, encode

JOB = {
  "prompt": [1, 2, 3],
  "num_samples": 1,
  "max_tokens": 3,
  "temperature": 0.0,
  "top_k": -1,
  "top_p": 1.0,
  "seed": None,
  "stop": None,
  "prompt_logprobs": False,
  "prompt_logprobs_last_n": None,
  "topk_prompt": 0,
  "topk_sample": 0,
}


class TestSampler(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.nodes = []

  async def asyncTearDown(self):
    for node in self.nodes:
      await node.aclose()
    await super().asyncTearDown()

  def node(self, store=None, **kwargs):
    node = TestNode(self.bus, store=store, **kwargs)
    self.nodes.append(node)
    return node

  def test_encode_masks_the_prompt_before_last_n_and_pads_top_k(self):
    out = Out(
      [Seq([9], [-3.0], [{9: -3.0, 7: -0.5}], False)],
      [None, {2: -1.0, 5: -0.1}, {3: -2.0}],
    )
    job = JOB | {"prompt_logprobs": True, "prompt_logprobs_last_n": 1, "topk_prompt": 2, "topk_sample": 1}
    res = wire.sample_json(encode(job, out))
    self.assertEqual(res["prompt_logprobs"], [None, None, -2.0])
    self.assertEqual(res["topk_prompt_logprobs"], [None, None, [[3, -2.0]]])
    self.assertEqual(res["sequences"][0]["topk_logprobs"], [[[7, -0.5]]])
    self.assertEqual(res["sequences"][0]["stop_reason"], "length")

  async def test_api_samples_base_lora_and_full_finetunes(self):
    base = f"{tempfile.mkdtemp()}/tiny"
    tiny().save_pretrained(base)
    store = Store(tempfile.mkdtemp())
    node = self.node(store, gpus=1)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app(self.bus, store)), base_url="http://test")
    self.addAsyncCleanup(client.aclose)

    async def call(path: str, body: dict) -> dict:
      return (await client.post(f"/api/v1/{path}", json=body)).json()

    prompt = {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]}
    samplers = []
    for options in ({"lora_config": {"rank": 4}}, {"user_metadata": {"openrl.fine_tuning_type": "full"}}):
      run = (await call("create_model", {"base_model": base, **options}))["model_id"]
      await node.add(await node.worker_for(run))
      request_id = (await call("save_weights_for_sampler", {"model_id": run, "sampling_session_seq_id": 0}))[
        "request_id"
      ]
      sid = json.loads((await self.bus.wait_for_reply(request_id, 5)).body)["sampling_session_id"]
      samplers.append(await node.worker_for(sid))
      await node.add(samplers[-1])
      request_id = (await call("asample", {"sampling_session_id": sid, "prompt": prompt, "prompt_logprobs": True}))[
        "request_id"
      ]
      self.assertTrue((await self.bus.wait_for_reply(request_id, 5)).ok)
      reply = await call("retrieve_future", {"request_id": request_id})
      self.assertEqual(reply["prompt_logprobs"], [None, -1.0, -1.0])
      self.assertEqual(len(reply["sequences"]), 1)
    self.assertNotEqual(*samplers)

    request_id = (await call("asample", {"base_model": base, "prompt": prompt}))["request_id"]
    self.assertEqual(await node.worker_for(request_id.split("/")[0]), samplers[0])
    self.assertTrue((await self.bus.wait_for_reply(request_id, 5)).ok)
    self.assertEqual(len((await call("retrieve_future", {"request_id": request_id}))["sequences"]), 1)

  async def test_a_sampler_uses_its_fleet_assigned_two_gpu_bundle(self):
    await self.bus.create_run(Run("t1", "base", 0))
    await self.bus.create_run(Run("samp-1", "base", 1, role=Role.SAMPLER, gpus=2))
    node = self.node(gpus=2)
    for run, op, body in [
      ("t1", Op.CREATE_MODEL, {"base_model": "base", "rank": 0, "lora": {}}),
      ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
      ("samp-1", Op.SAMPLE, JOB),
      ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
    ]:
      reply = await self.bus.wait_for_reply(await self.bus.submit_command(run, op, json.dumps(body).encode()), 5)
      self.assertIsNotNone(reply, f"{run}: {op}")
      self.assertTrue(reply.ok, reply.body)
    self.assertEqual(node.samplers[await node.worker_for("samp-1")].env["CUDA_VISIBLE_DEVICES"], "0,1")

  async def test_samples_run_concurrently_in_bounded_waves(self):
    store = Store(tempfile.mkdtemp())
    for i in range(4):
      uri = f"tinker://r{i}/sampler_weights/s0"
      version = store.put(uri, {}, {"base": "base", "lora": True, "config": {}})
      await self.seed_checkpoint(uri, 0, version=version)
      await self.bus.create_run(Run(f"s{i}", "base", 4, role=Role.SAMPLER, model_path=uri, version=version))
    active, peak, parallel = {}, 0, 0

    class WaveEngine(FakeEngine):
      async def sample(self, job, name, request_id):
        nonlocal peak, parallel
        active[name] = active.get(name, 0) + 1
        peak = max(peak, len(active))
        parallel = max(parallel, sum(active.values()))
        try:
          await asyncio.sleep(0)
          return await super().sample(job, name, request_id)
        finally:
          active[name] -= 1
          if not active[name]:
            del active[name]

    node = self.node(store, gpus=1, engine=WaveEngine)
    requests = [
      await self.bus.submit_command(f"s{i}", Op.SAMPLE, json.dumps(JOB).encode()) for i in range(4) for _ in range(2)
    ]
    await node.add(await node.worker_for("s0"))
    self.assertTrue(all([(await self.bus.wait_for_reply(request, 5)).ok for request in requests]))
    self.assertLessEqual(peak, 3)
    self.assertGreater(parallel, 1)

  async def test_failed_boot_answers_the_request_and_releases_its_gpus(self):
    await self.bus.create_run(Run("first", "base", 4, role=Role.SAMPLER))
    closed = []

    class Broken(FakeEngine):
      async def ready(self):
        raise RuntimeError("engine boot failed")

      async def close(self):
        await super().close()
        closed.append(True)

    node = self.node(gpus=1, engine=Broken)
    first = await self.bus.submit_command("first", Op.SAMPLE, json.dumps(JOB).encode())
    await node.add(await node.worker_for("first"))
    reply = await self.bus.wait_for_reply(first, 5)
    self.assertIsNotNone(reply)
    self.assertFalse(reply.ok)
    self.assertIn("engine boot failed", json.loads(reply.body)["error"])
    async with asyncio.timeout(5):
      while node.state["turns"]:
        await asyncio.sleep(0.01)
    self.assertIsNone(node.samplers[await node.worker_for("first")].running)
    self.assertEqual(closed, [True])

  async def test_fft_sessions_do_not_overlap_and_keep_their_pinned_versions(self):
    store = Store(tempfile.mkdtemp())
    paths = {f"s{i}": f"tinker://r{i}/sampler_weights/s0" for i in (1, 2)}
    for run, uri in paths.items():
      version = store.put(uri, {}, {"base": "base", "lora": False, "config": {}})
      await self.seed_checkpoint(uri, 0, version=version)
      await self.bus.create_run(Run(run, "base", 0, role=Role.SAMPLER, model_path=uri, version=version))
    case = self

    class UnmixedEngine(FakeEngine):
      async def sample(self, job, name, request_id):
        case.assertEqual(self.loaded, {name})
        await asyncio.sleep(0.01)
        case.assertEqual(self.loaded, {name})
        return await super().sample(job, name, request_id)

    node = self.node(store, gpus=1, engine=UnmixedEngine)
    requests = []
    for run in ("s1", "s2", "s1", "s2"):
      requests.append(await self.bus.submit_command(run, Op.SAMPLE, json.dumps(JOB).encode()))
    await node.add(await node.worker_for("s1"))
    self.assertTrue(all([(await self.bus.wait_for_reply(request, 5)).ok for request in requests]))
    engine = node.samplers[await node.worker_for("s1")].running
    request = await self.bus.submit_command("s1", Op.SAMPLE, json.dumps(JOB).encode())
    self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
    before = set(engine.loaded)
    version = store.put(paths["s1"], {}, {"base": "base", "lora": False, "config": {}})
    await self.seed_checkpoint(paths["s1"], 0, version=version)
    request = await self.bus.submit_command("s1", Op.SAMPLE, json.dumps(JOB).encode())
    self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
    self.assertEqual(engine.loaded, before)

  async def test_invalid_backlog_does_not_starve_a_ready_sibling(self):
    await self.seed_checkpoint("tinker://x/sampler_weights/gone", 0, version="gone")
    await self.bus.create_run(
      Run("broken", "base", 4, role=Role.SAMPLER, model_path="tinker://x/sampler_weights/gone", version="gone")
    )
    await self.bus.create_run(Run("ready", "base", 4, role=Role.SAMPLER))
    requests = [await self.bus.submit_command("broken", Op.SAMPLE, json.dumps(JOB).encode()) for _ in range(32)]
    ready = await self.bus.submit_command("ready", Op.SAMPLE, json.dumps(JOB).encode())
    bus, queued = self.bus, []

    class Ready(FakeEngine):
      async def sample(self, job, name, request_id):
        queued.append((await bus.queued_and_in_flight("broken"))["queued"])
        return await super().sample(job, name, request_id)

    node = self.node(gpus=1, engine=Ready)
    await node.add(await node.worker_for("ready"))
    self.assertTrue((await self.bus.wait_for_reply(ready, 5)).ok)
    self.assertGreater(queued[0], 0, "ready work waited for the whole invalid backlog")
    replies = await asyncio.gather(*(self.bus.wait_for_reply(request, 5) for request in requests))
    self.assertTrue(all(reply is not None and not reply.ok for reply in replies))
    self.assertEqual(int(await self.bus.r.hget(key("run", "broken"), "failed")), len(requests))


if __name__ == "__main__":
  unittest.main()
