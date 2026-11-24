import asyncio
import json
import tempfile
import unittest
from unittest import mock

import httpx
import numpy as np

from test.execution import TestNode
from test.helpers import BusTest, FakeEngine, tiny
from tinyrl.api import wire
from tinyrl.api.gateway import app
from tinyrl.api.proto import tinker_public_pb2 as pb
from tinyrl.storage.store import Store
from tinyrl.types import Op, Role, Run
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


class TestSampler(BusTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.nodes, self.tasks = [], []

  async def asyncTearDown(self):
    for task in self.tasks:
      task.cancel()
    await asyncio.gather(*self.tasks, return_exceptions=True)
    for node in self.nodes:
      await node.aclose()
    await super().asyncTearDown()

  def node(self, store=None, **kwargs):
    node = TestNode(self.bus, store=store, **kwargs)
    self.nodes.append(node)
    return node

  def pump(self, node, sched=False):
    async def run():
      while True:
        await node.tick()
        await asyncio.sleep(0.005)

    self.tasks.append(asyncio.create_task(run()))

  async def until(self, done):
    async with asyncio.timeout(5):
      while not done():
        await asyncio.sleep(0.01)

  def test_encode_masks_the_prompt_before_last_n_and_pads_top_k(self):
    out = Out(
      [Seq([9], [-3.0], [{9: -3.0, 7: -0.5}], False)],
      [None, {2: -1.0, 5: -0.1}, {3: -2.0}],
    )
    job = JOB | {"prompt_logprobs": True, "prompt_logprobs_last_n": 1, "topk_prompt": 2, "topk_sample": 1}
    res = pb.SampleResponse.FromString(encode(job, out))
    plp = np.frombuffer(res.prompt_logprobs, np.float32)
    self.assertTrue(np.isnan(plp[0]) and np.isnan(plp[1]))
    self.assertEqual(plp[2], -2.0)
    ptk = res.topk_prompt_logprobs
    p_ids = np.frombuffer(ptk.token_ids, np.int32).reshape(ptk.length, ptk.k)
    p_lps = np.frombuffer(ptk.logprobs, np.float32).reshape(ptk.length, ptk.k)
    self.assertEqual((p_ids[0, 0], p_lps[0, 0]), (0, wire.MASK_LOGPROB))
    self.assertEqual((p_ids[2, 0], p_lps[2, 0], p_lps[2, 1]), (3, -2.0, wire.MASK_LOGPROB))
    stk = res.sequences[0].topk_sampled_logprobs
    self.assertEqual(np.frombuffer(stk.token_ids, np.int32).tolist(), [7])
    self.assertEqual(np.frombuffer(stk.logprobs, np.float32).tolist(), [-0.5])
    self.assertEqual(res.sequences[0].stop_reason, pb.STOP_REASON_LENGTH)

  async def test_api_samples_base_lora_and_full_finetunes(self):
    base = f"{tempfile.mkdtemp()}/tiny"
    tiny().save_pretrained(base)
    store = Store(tempfile.mkdtemp())
    node = self.node(store)
    self.pump(node)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app(self.bus)), base_url="http://test")
    self.addAsyncCleanup(client.aclose)

    async def call(path: str, body: dict) -> dict:
      return (await client.post(f"/api/v1/{path}", json=body)).json()

    prompt = {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]}
    samplers = []
    for options in ({"lora_config": {"rank": 4}}, {"user_metadata": {"openrl.fine_tuning_type": "full"}}):
      run = (await call("create_model", {"base_model": base, **options}))["model_id"]
      request_id = (await call("save_weights_for_sampler", {"model_id": run, "sampling_session_seq_id": 0}))[
        "request_id"
      ]
      sid = json.loads((await self.bus.wait_for_reply(request_id, 5)).body)["sampling_session_id"]
      samplers.append(await node.worker_for(sid))
      request_id = (await call("asample", {"sampling_session_id": sid, "prompt": prompt, "prompt_logprobs": True}))[
        "request_id"
      ]
      self.assertTrue((await self.bus.wait_for_reply(request_id, 5)).ok)
      raw = await client.post("/api/v1/retrieve_future", json={"request_id": request_id})
      reply = pb.SampleResponse.FromString(raw.content)
      plp = np.frombuffer(reply.prompt_logprobs, np.float32)
      self.assertTrue(np.isnan(plp[0]))
      self.assertEqual(plp[1:].tolist(), [-1.0, -1.0])
      self.assertEqual(len(reply.sequences), 1)
    self.assertNotEqual(*samplers)

    request_id = (await call("asample", {"base_model": base, "prompt": prompt}))["request_id"]
    self.assertEqual(await node.worker_for(request_id.split("/")[0]), samplers[0])
    self.assertTrue((await self.bus.wait_for_reply(request_id, 5)).ok)
    raw = await client.post("/api/v1/retrieve_future", json={"request_id": request_id})
    self.assertEqual(len(pb.SampleResponse.FromString(raw.content).sequences), 1)

  async def test_a_tensor_parallel_sampler_takes_turns_on_both_its_gpus(self):
    self.bus.create_run(Run("t1", "base", 0))
    self.bus.create_run(Run("samp-1", "base", 1, role=Role.SAMPLER, gpus=2))
    node = self.node(gpus=2)
    self.pump(node, sched=True)
    create = self.bus.submit_command("t1", Op.CREATE_MODEL, b'{"base_model":"base","rank":0,"lora":{}}')
    self.assertTrue((await self.bus.wait_for_reply(create, 5)).ok)
    for run, op, body in [
      ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
      ("samp-1", Op.SAMPLE, JOB),
      ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
    ]:
      reply = await self.bus.wait_for_reply(self.bus.submit_command(run, op, json.dumps(body).encode()), 5)
      self.assertIsNotNone(reply)
      self.assertTrue(reply.ok, reply.body)

  async def test_samples_run_concurrently_in_bounded_waves(self):
    store = Store(tempfile.mkdtemp())
    for i in range(4):
      uri = f"tinker://r{i}/sampler_weights/s0"
      self.seed_checkpoint(uri, 0, version=store.put({}, {"base": "base", "lora": True, "config": {}}))
      self.bus.create_run(Run(f"s{i}", "base", 4, role=Role.SAMPLER, model_path=uri, account="alice"))
    active, peak, parallel = {}, 0, 0

    class WaveEngine(FakeEngine):
      async def sample(self, job, name):
        nonlocal peak, parallel
        active[name] = active.get(name, 0) + 1
        peak = max(peak, len(active))
        parallel = max(parallel, sum(active.values()))
        try:
          await asyncio.sleep(0)
          return await super().sample(job, name)
        finally:
          active[name] -= 1
          if not active[name]:
            del active[name]

    node = self.node(store, gpus=1, engine=WaveEngine)
    requests = [
      self.bus.submit_command(f"s{i}", Op.SAMPLE, json.dumps(JOB).encode()) for i in range(4) for _ in range(2)
    ]
    with mock.patch("tinyrl.types.MAX_LORAS", 3):
      await node.bind("sampler", "s0", "s1", "s2", "s3")
      self.pump(node)
      self.assertTrue(all([(await self.bus.wait_for_reply(request, 5)).ok for request in requests]))
    self.assertLessEqual(peak, 3)
    self.assertGreater(parallel, 1)

  async def test_failed_boot_answers_waiting_heads_and_leaves_later_work_unclaimed(self):
    store = Store(tempfile.mkdtemp())
    uri = "tinker://r/sampler_weights/new"
    self.seed_checkpoint(uri, 0, version=store.put({}, {"base": "base", "lora": True, "config": {}}))
    self.bus.create_run(Run("first", "base", 4, role=Role.SAMPLER))
    self.bus.create_run(Run("later", "base", 4, role=Role.SAMPLER, model_path=uri))
    booting, finish_boot = asyncio.Event(), asyncio.Event()

    class Broken(FakeEngine):
      @classmethod
      async def start(cls, base, lora, env=None):
        booting.set()
        await finish_boot.wait()
        raise RuntimeError("engine boot failed")

    node = self.node(store, gpus=1, engine=Broken)
    first = self.bus.submit_command("first", Op.SAMPLE, json.dumps(JOB).encode())
    await node.bind("sampler", "first", "later")
    try:
      await asyncio.wait_for(booting.wait(), 5)
      later = self.bus.submit_command("later", Op.SAMPLE, json.dumps(JOB).encode())
      self.assertEqual(self.bus.queued_and_in_flight("later"), {"live": True, "queued": 1, "in_flight": []})
      finish_boot.set()
      await asyncio.gather(*node.tasks())
      await node.tick()
      if ts := node.tasks():
        await asyncio.gather(*ts)
      reply = await self.bus.wait_for_reply(first, 5)
      self.assertIsNotNone(reply)
      self.assertFalse(reply.ok)
      self.assertIn("engine boot failed", json.loads(reply.body)["error"])
      self.assertNotIn("sampler", node.workers)
      self.assertIsNone(await self.bus.wait_for_reply(later, 0))
    finally:
      finish_boot.set()
    node.engine_type = FakeEngine
    self.pump(node)
    self.assertTrue((await self.bus.wait_for_reply(later, 5)).ok)

  async def test_fft_sessions_do_not_overlap_and_reload_changed_checkpoints(self):
    store = Store(tempfile.mkdtemp())
    paths = {f"s{i}": f"tinker://r{i}/sampler_weights/s0" for i in (1, 2)}
    for run, uri in paths.items():
      self.seed_checkpoint(uri, 0, version=store.put({}, {"base": "base", "lora": False, "config": {}}))
      self.bus.create_run(Run(run, "base", 0, role=Role.SAMPLER, model_path=uri))
    case = self

    class UnmixedEngine(FakeEngine):
      async def sample(self, job, name):
        case.assertEqual(self.loaded, {name})
        await asyncio.sleep(0.01)
        case.assertEqual(self.loaded, {name})
        return await super().sample(job, name)

    node = self.node(store, engine=UnmixedEngine, gpus=1)
    requests = [self.bus.submit_command(run, Op.SAMPLE, json.dumps(JOB).encode()) for run in ("s1", "s2", "s1", "s2")]
    await node.bind("sampler:fft:base", "s1", "s2")
    self.pump(node)
    self.assertTrue(all([(await self.bus.wait_for_reply(request, 5)).ok for request in requests]))
    engine = node.samplers["sampler:fft:base"].running
    request = self.bus.submit_command("s1", Op.SAMPLE, json.dumps(JOB).encode())
    self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
    before = set(engine.loaded)
    self.seed_checkpoint(paths["s1"], 0, version=store.put({}, {"base": "base", "lora": False, "config": {}}))
    request = self.bus.submit_command("s1", Op.SAMPLE, json.dumps(JOB).encode())
    self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
    self.assertNotEqual(engine.loaded, before)

  async def test_a_new_preparation_error_replies_while_its_process_is_waiting_for_gpu(self):
    store = Store(tempfile.mkdtemp())
    self.bus.create_run(Run("busy", "busy-base", 4, role=Role.SAMPLER))
    for run, path in (("ready", None), ("broken", "tinker://r/sampler_weights/missing")):
      self.bus.create_run(Run(run, "base", 4, role=Role.SAMPLER, model_path=path))
    sampling, finish = asyncio.Event(), asyncio.Event()

    class Occupied(FakeEngine):
      async def sample(self, job, name):
        if job["prompt"] == [11]:
          sampling.set()
          await finish.wait()
        return await super().sample(job, name)

    node = self.node(store, gpus=1, engine=Occupied)
    await node.bind("busy", "busy")
    await node.bind("waiting", "ready", "broken")
    self.pump(node)
    holding = self.bus.submit_command("busy", Op.SAMPLE, json.dumps(JOB | {"prompt": [11]}).encode())
    try:
      await asyncio.wait_for(sampling.wait(), 5)
      ready = self.bus.submit_command("ready", Op.SAMPLE, json.dumps(JOB).encode())
      broken = self.bus.submit_command("broken", Op.SAMPLE, json.dumps(JOB).encode())
      reply = await self.bus.wait_for_reply(broken, 3)
      self.assertIsNotNone(reply, "a healthy head waiting for GPU must not hold up a preparation error")
      self.assertFalse(reply.ok)
      self.assertIn("no checkpoint at", json.loads(reply.body)["error"])
      self.assertEqual(self.bus.queued_and_in_flight("ready"), {"live": True, "queued": 1, "in_flight": []})
      self.assertIsNone(await self.bus.wait_for_reply(holding, 0))
    finally:
      finish.set()
    self.assertTrue((await self.bus.wait_for_reply(holding, 5)).ok)
    self.assertTrue((await self.bus.wait_for_reply(ready, 5)).ok)

  async def test_a_busy_holder_yields_its_gpu_after_its_quantum(self):
    self.bus.create_run(Run("busy", "base-a", 4, role=Role.SAMPLER, account="a"))
    self.bus.create_run(Run("ready", "base-b", 4, role=Role.SAMPLER, account="b"))
    order, sampling, finish = [], asyncio.Event(), asyncio.Event()

    class Recording(FakeEngine):
      async def sample(self, job, name):
        order.append(job["prompt"])
        if job["prompt"] == [11]:
          sampling.set()
          await finish.wait()
        else:
          await asyncio.sleep(0.02)
        return await super().sample(job, name)

    node = self.node(Store(tempfile.mkdtemp()), gpus=1, engine=Recording)
    with (
      mock.patch("tinyrl.control.policy.QUANTUM_MS", 15),
      mock.patch("tinyrl.types.MAX_SEQS", 1),
    ):
      first = self.bus.submit_command("busy", Op.SAMPLE, json.dumps(JOB | {"prompt": [11]}).encode(), cost=10)
      self.pump(node, sched=True)
      await asyncio.wait_for(sampling.wait(), 5)
      ready = self.bus.submit_command("ready", Op.SAMPLE, json.dumps(JOB | {"prompt": [22]}).encode(), cost=10)
      tail = [
        self.bus.submit_command("busy", Op.SAMPLE, json.dumps(JOB | {"prompt": [p]}).encode(), cost=10)
        for p in (33, 44, 55)
      ]
      await asyncio.sleep(0.03)  # let busy's 15ms quantum expire
      finish.set()
      for req in (first, ready, *tail):
        self.assertTrue((await self.bus.wait_for_reply(req, 5)).ok)
    self.assertEqual(order[0], [11])
    self.assertLess(order.index([22]), order.index([55]))

  async def test_missing_checkpoint_is_rejected_before_healthy_sibling_uses_gpu(self):
    self.bus.create_run(Run("broken", "base", 4, role=Role.SAMPLER, model_path="tinker://x/sampler_weights/gone"))
    self.bus.create_run(Run("ready", "base", 4, role=Role.SAMPLER))
    requests = [self.bus.submit_command("broken", Op.SAMPLE, json.dumps(JOB).encode()) for _ in range(32)]
    ready = self.bus.submit_command("ready", Op.SAMPLE, json.dumps(JOB).encode())
    self.assertEqual(self.bus.queued_and_in_flight("broken"), {"live": True, "queued": 0, "in_flight": []})
    for request in requests:
      reply = await self.bus.wait_for_reply(request, 0)
      self.assertFalse(reply.ok)
      self.assertIn("no checkpoint at", json.loads(reply.body)["error"])
    self.assertEqual(self.bus.get_run("broken")["failed"], len(requests))

    node = self.node(gpus=1)
    await node.bind("sampler", "broken", "ready")
    self.pump(node)
    self.assertTrue((await self.bus.wait_for_reply(ready, 5)).ok)

  async def test_high_priority_sampler_account_runs_before_overserved_sibling(self):
    self.bus.create_run(Run("high", "base", 4, role=Role.SAMPLER, account="alice"))
    self.bus.create_run(Run("low", "base", 4, role=Role.SAMPLER, account="bob"))
    order = []

    class OrderEngine(FakeEngine):
      async def sample(self, job, name):
        order.append(job["prompt"][0])
        return await super().sample(job, name)

    self.bus.service["bob"] = 1000.0
    req_low = self.bus.submit_command("low", Op.SAMPLE, json.dumps(JOB | {"prompt": [1]}).encode())
    req_high = self.bus.submit_command("high", Op.SAMPLE, json.dumps(JOB | {"prompt": [2]}).encode())
    node = self.node(gpus=1, engine=OrderEngine)
    self.pump(node)
    self.assertTrue((await self.bus.wait_for_reply(req_high, 5)).ok)
    self.assertTrue((await self.bus.wait_for_reply(req_low, 5)).ok)
    self.assertEqual(order, [2, 1])


if __name__ == "__main__":
  unittest.main()
