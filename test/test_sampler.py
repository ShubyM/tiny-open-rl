import asyncio
import json
import os
import tempfile
import unittest
from unittest import mock

import httpx

from test.execution import TestNode
from test.helpers import FakeEngine, RedisTest, tiny
from tinyrl.api import wire
from tinyrl.api.gateway import app
from tinyrl.bus import Op, Role, Run, key
from tinyrl.control import scheduler
from tinyrl.storage.store import Store
from tinyrl.workers import vllm
from tinyrl.workers.sampler import Out, SamplingAbortedError, Seq, encode

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

  def pump(self, node):
    async def run():
      while True:
        await node.tick()
        await asyncio.sleep(0.01)

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
    res = wire.sample_json(encode(job, out))
    self.assertEqual(res["prompt_logprobs"], [None, None, -2.0])
    self.assertEqual(res["topk_prompt_logprobs"], [None, None, [[3, -2.0]]])
    self.assertEqual(res["sequences"][0]["topk_logprobs"], [[[7, -0.5]]])
    self.assertEqual(res["sequences"][0]["stop_reason"], "length")

  async def test_api_samples_base_lora_and_full_finetunes(self):
    base = f"{tempfile.mkdtemp()}/tiny"
    tiny().save_pretrained(base)
    store = Store(tempfile.mkdtemp())
    node = self.node(store)
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

  async def test_a_tensor_parallel_sampler_takes_turns_on_both_its_gpus(self):
    await self.bus.create_run(Run("t1", "", "base", 0))
    await self.bus.create_run(Run("samp-1", "", "base", 1, role=Role.SAMPLER, gpus=2))
    node = self.node(gpus=2)
    await node.announce_node()
    self.tasks.append(asyncio.create_task(scheduler.run(self.bus)))
    self.pump(node)
    for run, op, body in [
      ("t1", Op.CREATE_MODEL, {"base_model": "base", "rank": 0, "lora": {}}),
      ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
      ("samp-1", Op.SAMPLE, JOB),
      ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
    ]:
      reply = await self.bus.wait_for_reply(await self.bus.submit_command(run, op, json.dumps(body).encode()), 5)
      self.assertIsNotNone(reply)
      self.assertTrue(reply.ok, reply.body)

  async def test_samples_run_concurrently_in_bounded_waves(self):
    store = Store(tempfile.mkdtemp())
    for i in range(4):
      uri = f"tinker://r{i}/sampler_weights/s0"
      store.put(uri, {}, {"base": "base", "lora": True, "config": {}})
      await self.bus.create_run(Run(f"s{i}", "sampler", "base", 4, role=Role.SAMPLER, model_path=uri))
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

    node = self.node(store, engine=WaveEngine)
    requests = [
      await self.bus.submit_command(f"s{i}", Op.SAMPLE, json.dumps(JOB).encode()) for i in range(4) for _ in range(2)
    ]
    await node.add("sampler")
    self.assertTrue(all([(await self.bus.wait_for_reply(request, 5)).ok for request in requests]))
    self.assertLessEqual(peak, 3)
    self.assertGreater(parallel, 1)

  async def test_failed_boot_answers_waiting_heads_and_leaves_later_work_unclaimed(self):
    store = Store(tempfile.mkdtemp())
    uri = "tinker://r/sampler_weights/new"
    store.put(uri, {}, {"base": "base", "lora": True, "config": {}})
    await self.bus.create_run(Run("first", "sampler", "base", 4, role=Role.SAMPLER))
    await self.bus.create_run(Run("later", "sampler", "base", 4, role=Role.SAMPLER, model_path=uri))
    booting, finish_boot = asyncio.Event(), asyncio.Event()

    class Broken(FakeEngine):
      @classmethod
      async def start(cls, base, lora, env=None):
        booting.set()
        await finish_boot.wait()
        raise RuntimeError("engine boot failed")

    node = self.node(store, gpus=1, engine=Broken)
    first = await self.bus.submit_command("first", Op.SAMPLE, json.dumps(JOB).encode())
    reservation = await node.add("sampler")
    sampler, task = node.samplers["sampler"], node.tasks["sampler"]
    try:
      await asyncio.wait_for(booting.wait(), 5)
      later = await self.bus.submit_command("later", Op.SAMPLE, json.dumps(JOB).encode())
      future = await self.bus.submit_command("later", Op.SAMPLE, json.dumps(JOB).encode())
      self.assertEqual(await self.bus.queued_and_in_flight("later"), {"live": True, "queued": 2, "in_flight": []})
      finish_boot.set()
      for request in (first, later):
        reply = await self.bus.wait_for_reply(request, 5)
        self.assertIsNotNone(reply)
        self.assertFalse(reply.ok)
        self.assertIn("engine boot failed", json.loads(reply.body)["error"])
      [error] = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
      self.assertIsInstance(error, RuntimeError)
      self.assertIsNone(sampler.running)
      self.assertTrue(reservation.gpu_released)
      self.assertEqual(reservation.gpus_held, ())
      self.assertIsNone(await self.bus.lease_owner("sampler"))
      self.assertIsNone(await self.bus.wait_for_reply(future, 0))
      self.assertEqual(await self.bus.queued_and_in_flight("later"), {"live": True, "queued": 1, "in_flight": []})
    finally:
      finish_boot.set()
    node.engine_type = FakeEngine
    await node.add("sampler")
    self.assertTrue((await self.bus.wait_for_reply(future, 5)).ok)

  async def test_fft_sessions_do_not_overlap_and_reload_changed_checkpoints(self):
    store = Store(tempfile.mkdtemp())
    paths = {f"s{i}": f"tinker://r{i}/sampler_weights/s0" for i in (1, 2)}
    for run, uri in paths.items():
      store.put(uri, {}, {"base": "base", "lora": False, "config": {}})
      await self.bus.create_run(Run(run, "sampler:fft:base", "base", 0, role=Role.SAMPLER, model_path=uri))
    case = self

    class UnmixedEngine(FakeEngine):
      async def sample(self, job, name):
        case.assertEqual(self.loaded, {name})
        await asyncio.sleep(0.01)
        case.assertEqual(self.loaded, {name})
        return await super().sample(job, name)

    node = self.node(store, engine=UnmixedEngine)
    requests = []
    for run in ("s1", "s2", "s1", "s2"):
      requests.append(await self.bus.submit_command(run, Op.SAMPLE, json.dumps(JOB).encode()))
    await node.add("sampler:fft:base")
    self.assertTrue(all([(await self.bus.wait_for_reply(request, 5)).ok for request in requests]))
    engine = node.samplers["sampler:fft:base"].running
    request = await self.bus.submit_command("s1", Op.SAMPLE, json.dumps(JOB).encode())
    self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
    before = set(engine.loaded)
    store.put(paths["s1"], {}, {"base": "base", "lora": False, "config": {}})
    request = await self.bus.submit_command("s1", Op.SAMPLE, json.dumps(JOB).encode())
    self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
    self.assertNotEqual(engine.loaded, before)

  async def test_a_new_preparation_error_replies_while_its_worker_is_waiting_for_gpu(self):
    store = Store(tempfile.mkdtemp())
    for run, base, path in (
      ("busy", "base-a", None),
      ("ready", "base-b", None),
      ("broken", "base-b", "tinker://r/sampler_weights/missing"),
    ):
      await self.bus.create_run(Run(run, "", base, 4, role=Role.SAMPLER, model_path=path))
    sampling, finish = asyncio.Event(), asyncio.Event()

    class Occupied(FakeEngine):
      async def sample(self, job, name):
        if job["prompt"] == [11]:
          sampling.set()
          await finish.wait()
        return await super().sample(job, name)

    node = self.node(store, gpus=1, engine=Occupied)
    await node.announce_node()
    self.tasks.append(asyncio.create_task(scheduler.run(self.bus)))
    self.pump(node)
    holding = await self.bus.submit_command("busy", Op.SAMPLE, json.dumps(JOB | {"prompt": [11]}).encode())
    try:
      await asyncio.wait_for(sampling.wait(), 5)
      ready = await self.bus.submit_command("ready", Op.SAMPLE, json.dumps(JOB).encode())
      await self.until(lambda: "ready" in node.state.jobs and node.state.jobs["ready"].worker in node.running)
      broken = await self.bus.submit_command("broken", Op.SAMPLE, json.dumps(JOB).encode())
      reply = await self.bus.wait_for_reply(broken, 3)
      self.assertIsNotNone(reply, "a healthy head waiting for GPU must not hold up a preparation error")
      self.assertFalse(reply.ok)
      self.assertIn("no checkpoint at", json.loads(reply.body)["error"])
      self.assertEqual(await self.bus.queued_and_in_flight("ready"), {"live": True, "queued": 1, "in_flight": []})
      self.assertIsNone(await self.bus.wait_for_reply(holding, 0))
    finally:
      finish.set()
    self.assertTrue((await self.bus.wait_for_reply(holding, 5)).ok)
    self.assertTrue((await self.bus.wait_for_reply(ready, 5)).ok)

  async def test_a_busy_holder_yields_its_gpu_after_its_turn_of_tokens(self):
    for run, base in (("busy", "base-a"), ("ready", "base-b")):
      await self.bus.create_run(Run(run, "", base, 4, role=Role.SAMPLER))
    order, sampling, finish = [], asyncio.Event(), asyncio.Event()

    class Recording(FakeEngine):
      async def sample(self, job, name):
        order.append(job["prompt"])
        if job["prompt"] == [11]:
          sampling.set()
          await finish.wait()
        return await super().sample(job, name)

    node = self.node(Store(tempfile.mkdtemp()), gpus=1, engine=Recording)
    await node.announce_node()
    self.pump(node)
    with mock.patch.object(scheduler, "BUDGET_S", 0.05):
      self.tasks.append(asyncio.create_task(scheduler.run(self.bus)))
      first = await self.bus.submit_command("busy", Op.SAMPLE, json.dumps(JOB | {"prompt": [11]}).encode())
      await asyncio.wait_for(sampling.wait(), 5)
      ready = await self.bus.submit_command("ready", Op.SAMPLE, json.dumps(JOB | {"prompt": [22]}).encode())
      await self.until(lambda: "ready" in node.state.jobs and node.state.jobs["ready"].worker in node.running)
      more = await self.bus.submit_command("busy", Op.SAMPLE, json.dumps(JOB | {"prompt": [33]}).encode())
      await asyncio.sleep(0.15)  # exceed BUDGET_S so the holder yields GPU 0 after completing its active command
      finish.set()
      for request in (first, ready, more):
        self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
    self.assertEqual(order, [[11], [22], [33]])

  async def test_vllm_abort_does_not_publish_truncated_output_and_replays_cleanly(self):
    aborted_payload = json.dumps(
      {
        "choices": [
          {
            "index": 0,
            "token_ids": [1],
            "finish_reason": "abort",
            "logprobs": {"token_logprobs": [-1.0], "top_logprobs": [{}]},
          }
        ]
      }
    ).encode()
    with self.assertRaises(SamplingAbortedError):
      vllm.parse(aborted_payload)
    with self.assertRaises(SamplingAbortedError):
      vllm.parse(json.dumps({"choices": []}).encode())

    await self.bus.create_run(Run("s1", "sampler", "base", 4, role=Role.SAMPLER))
    aborted = asyncio.Event()

    class AbortDuringDrain(FakeEngine):
      async def sample(self, job, name):
        aborted.set()
        raise SamplingAbortedError("paused during drain")

    node = self.node(Store(tempfile.mkdtemp()), gpus=1, engine=AbortDuringDrain)
    req = await self.bus.submit_command("s1", Op.SAMPLE, json.dumps(JOB).encode())
    await node.add("sampler")
    await asyncio.wait_for(aborted.wait(), 5)
    # Aborted attempt must not publish a truncated reply; command remains pending until takeover/replay.
    self.assertIsNone(await self.bus.wait_for_reply(req, 0))
    await node.close(graceful=False)
    self.assertIsNone(await self.bus.wait_for_reply(req, 0))
    node2 = self.node(Store(tempfile.mkdtemp()), gpus=1, engine=FakeEngine)
    await node2.add("sampler")
    reply = await self.bus.wait_for_reply(req, 5)
    self.assertIsNotNone(reply)
    self.assertTrue(reply.ok, reply.body)

  def test_compiled_artifact_publish_stage_and_reject_mismatch(self):
    with tempfile.TemporaryDirectory() as folder:
      store = Store(f"{folder}/store")
      src = f"{folder}/src_cache"
      os.makedirs(f"{src}/vllm/sub", exist_ok=True)
      with open(f"{src}/vllm/sub/kernel.bin", "wb") as f:
        f.write(b"compiled-kernel")
      env = {"TINYRL_GPU_ARCH": "sm_90", "TINYRL_VLLM_VERSION": "0.10.0", "TINYRL_TORCH_VERSION": "2.7.0"}
      key = vllm.compiled_artifact_key("base/m", True, 2, ["--enable-sleep-mode"], env)
      uri = "tinker://artifacts/compiled/v1"
      ver, size = vllm.publish_compiled_artifact(store, uri, key, src)
      self.assertGreater(size, 0)
      dst = f"{folder}/dst_cache"
      self.assertTrue(vllm.stage_compiled_artifact(store, uri, key, dst, version=ver))
      with open(f"{dst}/vllm/sub/kernel.bin", "rb") as f:
        self.assertEqual(f.read(), b"compiled-kernel")
      wrong_arch = vllm.compiled_artifact_key(
        "base/m", True, 2, ["--enable-sleep-mode"], env | {"TINYRL_GPU_ARCH": "sm_80"}
      )
      self.assertFalse(vllm.stage_compiled_artifact(store, uri, wrong_arch, f"{folder}/dst_wrong", version=ver))

  async def test_invalid_backlog_does_not_starve_a_ready_sibling(self):
    await self.bus.create_run(
      Run("broken", "sampler", "base", 4, role=Role.SAMPLER, model_path="tinker://x/sampler_weights/gone")
    )
    await self.bus.create_run(Run("ready", "sampler", "base", 4, role=Role.SAMPLER))
    requests = [await self.bus.submit_command("broken", Op.SAMPLE, json.dumps(JOB).encode()) for _ in range(32)]
    ready = await self.bus.submit_command("ready", Op.SAMPLE, json.dumps(JOB).encode())
    bus, queued = self.bus, []

    class Ready(FakeEngine):
      async def sample(self, job, name):
        queued.append((await bus.queued_and_in_flight("broken"))["queued"])
        return await super().sample(job, name)

    node = self.node(gpus=1, engine=Ready)
    await node.add("sampler")
    self.assertTrue((await self.bus.wait_for_reply(ready, 5)).ok)
    self.assertGreater(queued[0], 0, "ready work waited for the whole invalid backlog")
    self.assertFalse((await self.bus.wait_for_reply(requests[-1], 5)).ok)
    self.assertEqual(int(await self.bus.r.hget(key("run", "broken"), "failed")), len(requests))


if __name__ == "__main__":
  unittest.main()
