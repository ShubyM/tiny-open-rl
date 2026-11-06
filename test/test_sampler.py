import asyncio
import json
import os
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import httpx
import safetensors.torch
import torch

from test.execution import TestNode
from test.helpers import FakeEngine, RedisTest, tiny
from tinyrl.api import wire
from tinyrl.api.gateway import app
from tinyrl.bus import Ev, Op, Role, Run, key
from tinyrl.storage.store import Store
from tinyrl.workers.sampler import Out, Seq, encode
from tinyrl.workers.vllm import VLLM

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


def tokens(chunks: list[int]) -> dict:
  return {"chunks": [{"type": "encoded_text", "tokens": chunks}]}


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

  async def test_vllm_speaks_the_completions_api(self):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
      calls.append((request.url.path, json.loads(request.content) if request.content else None))
      if request.url.path != "/v1/completions":
        return httpx.Response(200)
      choice = {
        "index": 0,
        "finish_reason": "stop",
        "token_ids": [9],
        "logprobs": {"token_logprobs": [-3.0], "top_logprobs": [{"token_id:9": -3.0, "token_id:7": -0.5}]},
        "prompt_logprobs": [None, {"2": {"logprob": -1.0}}, {"3": {"logprob": -2.0}}],
      }
      return httpx.Response(200, json={"choices": [choice]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://vllm")
    full = VLLM("http://vllm", "base", lora=False, client=client)
    await full.load_checkpoint("c0", "/ckpt/w")
    await full.load_checkpoint("c0", "/ckpt/w")  # already in place
    engine = VLLM("http://vllm", "base", lora=True, client=client)
    await engine.load_checkpoint("c0", "/ckpt/a")
    await engine.load_checkpoint("c0", "/ckpt/a")
    await engine.sleep()
    await engine.wake()
    paths = [path for path, _ in calls]
    self.assertEqual(paths, ["/collective_rpc", "/reset_prefix_cache", "/v1/load_lora_adapter", "/sleep", "/wake_up"])

    job = JOB | {"stop": ["\n", 4], "prompt_logprobs": True, "topk_sample": 1}
    out = await engine.sample(job, "c0")
    req = calls[-1][1]
    self.assertEqual((req["model"], req["stop"], req["stop_token_ids"]), ("c0", ["\n"], [4]))
    self.assertEqual((req["logprobs"], req["prompt_logprobs"]), (1, 1))
    self.assertEqual(out.seqs, [Seq([9], [-3.0], [{9: -3.0, 7: -0.5}], True)])
    self.assertEqual(out.prompt, [None, {2: -1.0}, {3: -2.0}])

  async def test_the_sdk_calls_sample_lora_base_and_full_finetunes(self):
    base = f"{tempfile.mkdtemp()}/tiny"
    tiny().save_pretrained(base)
    store = Store(tempfile.mkdtemp())
    node = self.node(store)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app(self.bus, store)), base_url="http://test")
    self.addAsyncCleanup(client.aclose)

    async def call(path: str, body: dict) -> dict:
      return (await client.post(f"/api/v1/{path}", json=body)).json()

    async def save_for_sampler(run: str) -> str:
      request_id = (await call("save_weights_for_sampler", {"model_id": run, "sampling_session_seq_id": 0}))[
        "request_id"
      ]
      return json.loads((await self.bus.wait_for_reply(request_id, 5)).body)["sampling_session_id"]

    run = (await call("create_model", {"base_model": base, "lora_config": {"rank": 4}}))["model_id"]
    trainer_worker = await node.worker_for(run)
    await node.add(trainer_worker)
    sid = await save_for_sampler(run)
    sampler_worker = await node.worker_for(sid)
    await node.add(sampler_worker)
    sampler = node.samplers[sampler_worker]
    self.assertTrue((await client.get(f"/api/v1/samplers/{sid}")).json()["model_path"].startswith("tinker://"))
    first = await call("join_sampling_session", {"sampling_session_id": sid})
    second = await call("join_sampling_session", {"sampling_session_id": sid})
    self.assertEqual(second["client_counter"], first["client_counter"] + 1)

    job = {"sampling_session_id": sid, "seq_id": 7, "prompt": tokens([1, 2, 3]), "prompt_logprobs": True}
    replies = []
    for _ in range(2):  # the SDK retrying with the same seq_id gets the first answer
      request_id = (await call("asample", job))["request_id"]
      self.assertTrue((await self.bus.wait_for_reply(request_id, 5)).ok)
      replies.append(await call("retrieve_future", {"request_id": request_id}))
    self.assertEqual(replies[0], replies[1])
    self.assertEqual(replies[0]["prompt_logprobs"], [None, -1.0, -1.0])

    request_id = (await call("asample", {"base_model": base, "prompt": tokens([1, 2])}))["request_id"]
    self.assertEqual(await node.worker_for(request_id.split("/")[0]), sampler_worker)
    self.assertTrue((await self.bus.wait_for_reply(request_id, 5)).ok)
    self.assertEqual(len((await call("retrieve_future", {"request_id": request_id}))["sequences"]), 1)
    calls = sampler.running.calls
    self.assertEqual(calls[0], ("start", base, True))
    loads = [c for c in calls if c[0] == "load"]
    self.assertEqual(len(loads), 1)
    self.assertEqual(loads[0][2], store.path(f"tinker://{run}/sampler_weights/step-0"))
    self.assertEqual([c for c in calls if c[0] == "sample"], [("sample", loads[0][1]), ("sample", None)])

    fft = (await call("create_model", {"base_model": base, "user_metadata": {"openrl.fine_tuning_type": "full"}}))[
      "model_id"
    ]
    fft_worker = await node.worker_for(fft)
    await node.add(fft_worker)
    fft_sid = await save_for_sampler(fft)
    fft_sampler_worker = await node.worker_for(fft_sid)
    self.assertNotEqual(fft_sampler_worker, sampler_worker)
    await node.add(fft_sampler_worker)
    fft_sampler = node.samplers[fft_sampler_worker]
    request_id = (await call("asample", {"sampling_session_id": fft_sid, "prompt": tokens([1, 2])}))["request_id"]
    self.assertTrue((await self.bus.wait_for_reply(request_id, 5)).ok)
    res = await client.post("/api/v1/retrieve_future", json={"request_id": request_id}, headers={"Accept": wire.PROTO})
    self.assertEqual(res.headers["content-type"], wire.PROTO)
    self.assertEqual(
      [c[2] for c in fft_sampler.running.calls if c[0] == "load"],
      [store.path(f"tinker://{fft}/sampler_weights/step-0")],
    )

  async def test_a_trainer_and_a_sampler_take_turns_on_one_gpu(self):
    await self.take_turns(1, [0], [0])

  async def test_a_tensor_parallel_sampler_takes_turns_on_both_its_gpus(self):
    await self.take_turns(2, [1], [0, 1])

  async def take_turns(self, gpus, trainer_gpus, sampler_gpus):
    await self.bus.create_run(Run("t1", "t1", "base", 0))
    await self.bus.create_run(Run("samp-1", "sampler:base", "base", 1, role=Role.SAMPLER, gpus=len(sampler_gpus)))
    node = self.node(gpus=gpus)
    await node.add("t1", gpus=trainer_gpus)
    node.processes["t1"].trainer.model.create_run("t1", "base", 0, {})
    await node.add("sampler:base", gpus=sampler_gpus)
    self.pump(node)
    for run, op, body in [
      ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
      ("samp-1", Op.SAMPLE, JOB),
      ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
    ]:
      reply = await self.bus.wait_for_reply(await self.bus.submit_command(run, op, json.dumps(body).encode()), 5)
      self.assertIsNotNone(reply)
      self.assertTrue(reply.ok, reply.body)
    self.assertIn(("sleep",), node.samplers["sampler:base"].running.calls)

  async def test_compatible_fft_samplers_share_a_gpu_without_parking_the_idle_holder(self):
    for run in ("a", "b"):
      await self.bus.create_run(Run(run, run, f"base-{run}", 0, role=Role.SAMPLER))
    node = self.node(gpus=1, gpu_mib=80 * 1024)
    with patch.dict(os.environ, {"TINYRL_WARM_SAMPLER": "0"}):
      first, second = await node.add("a", gpus=[0]), await node.add("b", gpus=[0])
    for sampler in node.samplers.values():
      self.assertEqual(sampler.env["TINYRL_GPU_MEM"], "0.42")

    request = await self.bus.submit_command("a", Op.SAMPLE, json.dumps(JOB).encode())
    self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
    reserved = first.since
    self.assertIsNotNone(reserved)
    self.assertFalse((await self.bus.queued_and_in_flight("a"))["in_flight"])

    request = await self.bus.submit_command("b", Op.SAMPLE, json.dumps(JOB).encode())
    self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
    self.assertEqual(first.since, reserved)
    self.assertIsNotNone(second.since)
    self.assertEqual((first.gpus, second.gpus), ((0,), (0,)))
    self.assertNotIn(("sleep",), node.samplers["a"].running.calls)
    self.assertNotIn(("sleep",), node.samplers["b"].running.calls)

  async def test_a_sampler_batches_queued_commands_concurrently_and_loads_once(self):
    store = Store(tempfile.mkdtemp())
    store.put("tinker://r1/sampler_weights/s0", {}, {"base": "base", "lora": True, "config": {}})
    await self.bus.create_run(
      Run("samp-1", "sampler:base", "base", 1, role=Role.SAMPLER, model_path="tinker://r1/sampler_weights/s0")
    )
    active = peak = starts = loads = 0

    class BatchEngine(FakeEngine):
      @classmethod
      async def start(cls, base: str, lora: bool, env=None):
        nonlocal starts
        await asyncio.sleep(0.01)
        starts += 1
        return await super().start(base, lora, env)

      async def load_checkpoint(self, name: str, path: str) -> None:
        nonlocal loads
        if name not in self.loaded:
          await asyncio.sleep(0.01)
          loads += 1
        await super().load_checkpoint(name, path)

      async def sample(self, job: dict, name: str | None) -> Out:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return await super().sample(job, name)

    store.local_folder = MagicMock(wraps=store.local_folder)
    node = self.node(store, engine=BatchEngine)
    ids = [await self.bus.submit_command("samp-1", Op.SAMPLE, json.dumps(JOB).encode()) for _ in range(8)]
    await node.add("sampler:base")
    self.assertTrue(all([(await self.bus.wait_for_reply(i, 5)).ok for i in ids]))
    self.assertEqual((starts, loads, peak, store.local_folder.call_count), (1, 1, 8, 1))

  async def test_a_lora_sampler_runs_three_sessions_at_once_before_starting_the_next_wave(self):
    store = Store(tempfile.mkdtemp())
    for i in range(4):
      uri = f"tinker://r{i}/sampler_weights/s0"
      store.put(uri, {}, {"base": "base", "lora": True, "config": {}})
      await self.bus.create_run(Run(f"s{i}", "sampler", "base", 4, role=Role.SAMPLER, model_path=uri))
    active, peak = {}, 0
    entered, finish = asyncio.Event(), asyncio.Event()

    class WaveEngine(FakeEngine):
      async def sample(self, job, name):
        nonlocal peak
        active[name] = active.get(name, 0) + 1
        peak = max(peak, len(active))
        if sum(active.values()) == 6:
          entered.set()
        try:
          await finish.wait()
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
    try:
      await asyncio.wait_for(entered.wait(), 5)
      self.assertEqual(sorted(active.values()), [2, 2, 2])
      self.assertEqual(len([c for c in node.samplers["sampler"].running.calls if c[0] == "load"]), 3)
    finally:
      finish.set()
    self.assertTrue(all([(await self.bus.wait_for_reply(request, 5)).ok for request in requests]))
    self.assertEqual(peak, 3)
    self.assertEqual(len([c for c in node.samplers["sampler"].running.calls if c[0] == "load"]), 4)

  async def test_empty_peeks_never_fail_a_sample_that_is_still_running(self):
    await self.bus.create_run(Run("s", "sampler", "base", 4, role=Role.SAMPLER))
    sampling, finish, peeked = asyncio.Event(), asyncio.Event(), asyncio.Event()
    peek = self.bus.peek_commands

    async def read(*args, **kwargs):
      commands = await peek(*args, **kwargs)
      if not commands and sampling.is_set():
        peeked.set()
      return commands

    class SlowSample(FakeEngine):
      async def sample(self, job, name):
        sampling.set()
        await finish.wait()
        return await super().sample(job, name)

    node = self.node(gpus=1, engine=SlowSample)
    await node.add("sampler")
    with (
      patch("tinyrl.control.node_daemon.STRANDED_MS", 0),
      patch("tinyrl.control.node_daemon.IDLE_MS", 5),
      patch.object(self.bus, "peek_commands", side_effect=read),
    ):
      self.pump(node)
      request = await self.bus.submit_command("s", Op.SAMPLE, json.dumps(JOB).encode())
      try:
        await asyncio.wait_for(peeked.wait(), 5)
        self.assertIsNone(await self.bus.wait_for_reply(request, 0))
        run = await self.bus.get_run("s")
        self.assertEqual((run["cmds"], run["failed"]), (0, 0))
      finally:
        finish.set()
      self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
      run = await self.bus.get_run("s")
      self.assertEqual((run["cmds"], run["failed"]), (1, 0))

  async def test_vllm_retries_a_dropped_keepalive_socket(self):
    import httpx

    from tinyrl.workers.vllm import VLLM

    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
      nonlocal calls
      calls += 1
      if calls == 1:
        raise httpx.ReadError("", request=req)
      body = {
        "choices": [
          {
            "index": 0,
            "token_ids": [9],
            "finish_reason": "stop",
            "logprobs": {"token_logprobs": [-0.2], "top_logprobs": [{"token_id:9": -0.2}]},
          }
        ]
      }
      return httpx.Response(200, json=body)

    engine = VLLM(
      "http://vllm",
      "base",
      True,
      client=httpx.AsyncClient(base_url="http://vllm", transport=httpx.MockTransport(handler)),
    )
    out = await engine.sample(JOB, "c0")
    await engine.close()
    self.assertEqual(calls, 2)
    self.assertEqual(out.seqs[0].tokens, [9])

  async def test_vllm_load_skips_untouched_fft_checkpoint(self):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
      calls.append(request.url.path)
      return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://vllm")
    full = VLLM("http://vllm", "base", lora=False, client=client)

    dir1 = tempfile.mkdtemp()
    safetensors.torch.save_file(
      {"w": torch.ones(1)}, f"{dir1}/model.safetensors", metadata={"meta": json.dumps({"untouched": True})}
    )

    dir2 = tempfile.mkdtemp()
    safetensors.torch.save_file(
      {"w": torch.ones(1)}, f"{dir2}/model.safetensors", metadata={"meta": json.dumps({"untouched": True})}
    )

    dir3 = tempfile.mkdtemp()
    safetensors.torch.save_file(
      {"w": torch.ones(1)}, f"{dir3}/model.safetensors", metadata={"meta": json.dumps({"untouched": False})}
    )

    await full.load_checkpoint("c0", dir1)
    self.assertEqual(calls, [])

    await full.load_checkpoint("c1", dir2)
    self.assertEqual(calls, [])

    await full.load_checkpoint("c2", dir3)
    self.assertEqual(calls, ["/collective_rpc", "/reset_prefix_cache"])

    await full.load_checkpoint("c0", dir1)
    self.assertEqual(calls, ["/collective_rpc", "/reset_prefix_cache", "/collective_rpc", "/reset_prefix_cache"])
    await full.close()

  async def test_a_sampler_takes_the_commands_that_arrive_while_its_engine_boots(self):
    store = Store(tempfile.mkdtemp())
    store.put("tinker://r1/sampler_weights/s0", {}, {"base": "base", "lora": True, "config": {}})
    await self.bus.create_run(
      Run("sess", "sampler:base", "base", 1, role=Role.SAMPLER, model_path="tinker://r1/sampler_weights/s0")
    )
    booting, finish_boot = asyncio.Event(), asyncio.Event()
    active = peak = 0

    class SlowStartEngine(FakeEngine):
      @classmethod
      async def start(cls, base: str, lora: bool, env=None):
        booting.set()
        await finish_boot.wait()
        return await super().start(base, lora, env)

      async def sample(self, job, name):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return await super().sample(job, name)

    node = self.node(store, gpus=1, engine=SlowStartEngine)
    requests = [await self.bus.submit_command("sess", Op.SAMPLE, json.dumps(JOB).encode())]
    await node.add("sampler:base")
    try:
      await asyncio.wait_for(booting.wait(), 5)
      requests += [await self.bus.submit_command("sess", Op.SAMPLE, json.dumps(JOB).encode()) for _ in range(2)]
      self.assertEqual(await self.bus.queued_and_in_flight("sess"), {"live": True, "queued": 3, "in_flight": []})
    finally:
      finish_boot.set()
    self.assertTrue(all([(await self.bus.wait_for_reply(request, 5)).ok for request in requests]))
    self.assertEqual(peak, 3)

  async def test_failed_boot_answers_waiting_heads_and_leaves_later_work_unclaimed(self):
    store = Store(tempfile.mkdtemp())
    uri = "tinker://r/sampler_weights/new"
    store.put(uri, {}, {"base": "base", "lora": True, "config": {}})
    await self.bus.create_run(Run("first", "sampler", "base", 4, role=Role.SAMPLER))
    await self.bus.create_run(Run("later", "sampler", "base", 4, role=Role.SAMPLER, model_path=uri))
    booting, finish_boot = asyncio.Event(), asyncio.Event()
    starts = 0

    class Broken(FakeEngine):
      @classmethod
      async def start(cls, base, lora, env=None):
        nonlocal starts
        starts += 1
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
      self.assertEqual(starts, 1)
      [error] = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
      self.assertIsInstance(error, RuntimeError)
      self.assertIsNone(sampler.running)
      self.assertIsNone(reservation.since)
      self.assertIsNone(await self.bus.lease_owner("sampler"))
      self.assertIsNone(await self.bus.wait_for_reply(future, 0))
      self.assertEqual(await self.bus.queued_and_in_flight("later"), {"live": True, "queued": 1, "in_flight": []})
    finally:
      finish_boot.set()

  async def test_sampler_step_groups_non_exclusive_fft_interleaved(self):
    store = Store(tempfile.mkdtemp())
    store.put("tinker://r1/sampler_weights/s0", {}, {"base": "base", "lora": False, "config": {}})
    store.put("tinker://r2/sampler_weights/s0", {}, {"base": "base", "lora": False, "config": {}})
    await self.bus.create_run(
      Run("s1", "sampler:fft:base", "base", 0, role=Role.SAMPLER, model_path="tinker://r1/sampler_weights/s0")
    )
    await self.bus.create_run(
      Run("s2", "sampler:fft:base", "base", 0, role=Role.SAMPLER, model_path="tinker://r2/sampler_weights/s0")
    )
    active = 0
    case = self

    class UnmixedEngine(FakeEngine):
      async def load_checkpoint(self, name, path):
        if name not in self.loaded:
          case.assertEqual(active, 0, "FFT weights changed while the previous session was still sampling")
        await super().load_checkpoint(name, path)

      async def sample(self, job, name):
        nonlocal active
        expected = set(self.loaded)
        active += 1
        try:
          await asyncio.sleep(0.01)
          case.assertEqual(self.loaded, expected)
          return await super().sample(job, name)
        finally:
          active -= 1

    node = self.node(store, engine=UnmixedEngine)
    requests = []
    for run in ("s1", "s2", "s1", "s2"):
      requests.append(await self.bus.submit_command(run, Op.SAMPLE, json.dumps(JOB).encode()))
    await node.add("sampler:fft:base")
    self.assertTrue(all([(await self.bus.wait_for_reply(request, 5)).ok for request in requests]))
    engine = node.samplers["sampler:fft:base"].running
    loads = [c for c in engine.calls if c[0] == "load"]
    self.assertEqual(len(loads), 2)
    self.assertNotEqual(loads[0][1], loads[1][1])
    first_sess = "s1" if "/r1/" in loads[0][2] else "s2"
    engine.calls.clear()
    request = await self.bus.submit_command(first_sess, Op.SAMPLE, json.dumps(JOB).encode())
    self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
    self.assertEqual([c for c in engine.calls if c[0] == "load"], [loads[0]])

  async def test_warm_sampler_starts_engine_before_sample(self):
    await self.bus.create_run(Run("s", "sampler:base", "base", 1, role=Role.SAMPLER))
    node = self.node(gpus=2)
    await node.add("sampler:base")
    self.pump(node)
    await self.until(lambda: node.samplers["sampler:base"].running is not None)
    self.assertEqual(node.samplers["sampler:base"].running.calls[0], ("start", "base", True))

  async def test_cold_boot_longer_than_a_quantum_still_gets_a_turn_to_sample(self):
    for run in ("a", "b"):
      await self.bus.create_run(Run(run, run, "base", 4, role=Role.SAMPLER))
    booting, finish_boot = asyncio.Event(), asyncio.Event()

    class SlowBoot(FakeEngine):
      @classmethod
      async def start(cls, base, lora, env=None):
        booting.set()
        await finish_boot.wait()
        return await super().start(base, lora, env)

    node = self.node(gpus=1)
    node.state.quantum = 0.1
    await node.add("a", gpus=[0])
    waiter = await node.add("b", gpus=[0])
    node.samplers["a"].engine_type = SlowBoot
    self.pump(node)
    request = await self.bus.submit_command("a", Op.SAMPLE, json.dumps(JOB).encode())
    try:
      await asyncio.wait_for(booting.wait(), 5)
      other = await self.bus.submit_command("b", Op.SAMPLE, json.dumps(JOB).encode())
      await self.until(lambda: bool(waiter.ticket))
      await asyncio.sleep(0.2)
      self.assertIsNone(node.samplers["b"].running)
    finally:
      finish_boot.set()
    self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
    calls = node.samplers["a"].running.calls
    sampled = next(i for i, call in enumerate(calls) if call[0] == "sample")
    self.assertNotIn(("sleep",), calls[:sampled])
    self.assertTrue((await self.bus.wait_for_reply(other, 5)).ok)

  async def test_failed_warm_start_replies_with_error_and_a_replacement_sampler_recovers(self):
    class Broken(FakeEngine):
      @classmethod
      async def start(cls, base: str, lora: bool, env=None):
        raise RuntimeError("no GPU memory")

    await self.bus.create_run(Run("s", "sampler:base", "base", 1, role=Role.SAMPLER))
    node = self.node(gpus=2, engine=Broken)
    command = await self.bus.submit_command("s", Op.SAMPLE, json.dumps(JOB).encode())
    with patch.dict(os.environ, {"TINYRL_WARM_SAMPLER": "1"}):
      reservation = await node.add("sampler:base")
    sampler, task = node.samplers["sampler:base"], node.tasks["sampler:base"]
    reply = await self.bus.wait_for_reply(command, 5)
    self.assertIsNotNone(reply)
    self.assertFalse(reply.ok)
    self.assertIn("no GPU memory", json.loads(reply.body)["error"])
    [error] = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
    self.assertIsInstance(error, RuntimeError)
    self.assertIsNone(sampler.running)
    self.assertIsNone(reservation.since)
    self.assertIsNone(await self.bus.lease_owner("sampler:base"))
    node.engine_type = FakeEngine
    command = await self.bus.submit_command("s", Op.SAMPLE, json.dumps(JOB).encode())
    await node.add("sampler:base")
    self.assertTrue((await self.bus.wait_for_reply(command, 5)).ok)
    self.assertIsNot(node.samplers["sampler:base"], sampler)

  async def test_unplacing_a_waiting_warm_sampler_closes_it_without_booting(self):
    for run in ("a", "b"):
      await self.bus.create_run(Run(run, run, "base", 4, role=Role.SAMPLER))
    sampling, finish = asyncio.Event(), asyncio.Event()

    class Occupied(FakeEngine):
      async def sample(self, job, name):
        sampling.set()
        await finish.wait()
        return await super().sample(job, name)

    node = self.node(gpus=1, engine=Occupied)
    await node.add("a", gpus=[0])
    request = await self.bus.submit_command("a", Op.SAMPLE, json.dumps(JOB).encode())
    try:
      await asyncio.wait_for(sampling.wait(), 5)
      with patch.dict(os.environ, {"TINYRL_WARM_SAMPLER": "1"}):
        waiting = await node.add("b", gpus=[0])
      sampler, task = node.samplers["b"], node.tasks["b"]
      await self.until(lambda: bool(waiting.ticket))
      await self.bus.append_event(Ev.UNPLACED, worker="b")
      await node.tick()
      await asyncio.wait_for(asyncio.shield(task), 5)
      self.assertIsNone(sampler.running)
      self.assertIsNone(waiting.since)
      self.assertIsNone(await self.bus.lease_owner("b"))
      self.assertIsNone(await self.bus.wait_for_reply(request, 0))
    finally:
      finish.set()
    self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)

  async def test_a_failed_wake_keeps_its_gpu_until_the_engine_has_closed(self):
    for run in ("a", "b"):
      await self.bus.create_run(Run(run, run, "base", 4, role=Role.SAMPLER))
    closing, finish_close = asyncio.Event(), asyncio.Event()

    class BrokenWake(FakeEngine):
      async def wake(self):
        raise RuntimeError("wake failed after allocating GPU memory")

      async def close(self):
        closing.set()
        await finish_close.wait()
        await super().close()

    node = self.node(gpus=1)
    first, other = await node.add("a", gpus=[0]), await node.add("b", gpus=[0])
    node.samplers["a"].engine_type = BrokenWake
    self.pump(node)

    async def submit(run):
      return await self.bus.submit_command(run, Op.SAMPLE, json.dumps(JOB).encode())

    try:
      self.assertTrue((await self.bus.wait_for_reply(await submit("a"), 5)).ok)
      self.assertTrue((await self.bus.wait_for_reply(await submit("b"), 5)).ok)
      self.assertIn(("sleep",), node.samplers["a"].running.calls)
      failed = await submit("a")
      await asyncio.wait_for(closing.wait(), 5)
      waiting = await submit("b")
      await self.until(lambda: bool(other.ticket))
      self.assertIsNotNone(first.since)
      self.assertIsNone(await self.bus.wait_for_reply(waiting, 0))
    finally:
      finish_close.set()
    self.assertFalse((await self.bus.wait_for_reply(failed, 5)).ok)
    self.assertTrue((await self.bus.wait_for_reply(waiting, 5)).ok)

  def test_sampler_imports_are_clean(self):
    import subprocess
    import sys

    out = subprocess.run(
      [
        sys.executable,
        "-c",
        "import sys, tinyrl.workers.sampler, tinyrl.workers.vllm; "
        "assert not ({'torch', 'transformers', 'peft', 'safetensors'} & set(sys.modules))",
      ],
      capture_output=True,
      text=True,
    )
    self.assertEqual(out.returncode, 0, out.stderr)

  async def test_a_sampler_stages_checkpoints_before_it_is_ready_for_a_turn(self):
    store = Store(tempfile.mkdtemp())
    uri = "tinker://r1/sampler_weights/s0"
    store.put(uri, {}, {"base": "base", "lora": True, "config": {}})
    await self.bus.create_run(Run("samp-1", "sampler:base", "base", 1, role=Role.SAMPLER, model_path=uri))
    real_local = store.local_folder
    blocked, blocking = threading.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()

    def blocking_local_folder(uri, version):
      self.assertNotEqual(threading.get_ident(), loop_thread)
      loop.call_soon_threadsafe(blocking.set)
      blocked.wait(5)
      return real_local(uri, version)

    store.local_folder = MagicMock(side_effect=blocking_local_folder)
    node = self.node(store, gpus=1)
    running = await node.add("sampler:base")
    command = await self.bus.submit_command("samp-1", Op.SAMPLE, json.dumps(JOB).encode())
    try:
      await asyncio.wait_for(blocking.wait(), timeout=2)
      self.assertEqual(running.ticket, 0)
      self.assertIsNone(running.since)
      self.assertIsNone(node.samplers["sampler:base"].running)
    finally:
      blocked.set()
    self.assertTrue((await self.bus.wait_for_reply(command, 5)).ok)

  async def test_preparation_errors_reply_while_another_worker_owns_the_gpu(self):
    store = Store(tempfile.mkdtemp())
    store.local_folder = MagicMock(side_effect=RuntimeError("bucket error"))
    for run, path in (("busy", None), ("broken", "tinker://r/sampler_weights/missing"), ("missing", None)):
      await self.bus.create_run(Run(run, run, "base", 4, role=Role.SAMPLER, model_path=path))
    sampling, finish = asyncio.Event(), asyncio.Event()

    class Occupied(FakeEngine):
      async def sample(self, job, name):
        if job["prompt"] == [11]:
          sampling.set()
          await finish.wait()
        return await super().sample(job, name)

    node = self.node(store, gpus=1, engine=Occupied)
    for worker in ("busy", "broken", "missing"):
      await node.add(worker, gpus=[0])
    holding = await self.bus.submit_command("busy", Op.SAMPLE, json.dumps(JOB | {"prompt": [11]}).encode())
    try:
      await asyncio.wait_for(sampling.wait(), 5)
      await self.bus.r.delete(key("run", "missing"))
      for run, error in (("broken", "RuntimeError: bucket error"), ("missing", "no sampling session missing")):
        request = await self.bus.submit_command(run, Op.SAMPLE, json.dumps(JOB).encode())
        reply = await self.bus.wait_for_reply(request, 5)
        self.assertIsNotNone(reply, "preparation errors must not wait for an occupied GPU")
        self.assertFalse(reply.ok)
        self.assertIn(error, json.loads(reply.body)["error"])
        self.assertEqual(node.running[run].ticket, 0)
        self.assertIsNone(node.running[run].since)
        self.assertIsNone(node.samplers[run].running)
      self.assertIsNone(await self.bus.wait_for_reply(holding, 0))
    finally:
      finish.set()
    self.assertTrue((await self.bus.wait_for_reply(holding, 5)).ok)

    await self.bus.create_run(Run("base", "broken", "base", 4, role=Role.SAMPLER))
    await node.sync()
    cmd_id = await self.bus.submit_command("base", Op.SAMPLE, json.dumps(JOB).encode())
    self.assertTrue((await self.bus.wait_for_reply(cmd_id, 5)).ok)

  async def test_a_new_preparation_error_replies_while_its_worker_is_waiting_for_gpu(self):
    store = Store(tempfile.mkdtemp())
    store.local_folder = MagicMock(side_effect=RuntimeError("bucket error"))
    for run, worker, path in (
      ("busy", "busy", None),
      ("ready", "waiting", None),
      ("broken", "waiting", "tinker://r/sampler_weights/missing"),
    ):
      await self.bus.create_run(Run(run, worker, "base", 4, role=Role.SAMPLER, model_path=path))
    sampling, finish = asyncio.Event(), asyncio.Event()

    class Occupied(FakeEngine):
      async def sample(self, job, name):
        if job["prompt"] == [11]:
          sampling.set()
          await finish.wait()
        return await super().sample(job, name)

    node = self.node(store, gpus=1, engine=Occupied)
    node.state.quantum = 0.01
    await node.add("busy", gpus=[0])
    waiting = await node.add("waiting", gpus=[0])
    self.pump(node)
    holding = await self.bus.submit_command("busy", Op.SAMPLE, json.dumps(JOB | {"prompt": [11]}).encode())
    try:
      await asyncio.wait_for(sampling.wait(), 5)
      ready = await self.bus.submit_command("ready", Op.SAMPLE, json.dumps(JOB).encode())
      await self.until(lambda: bool(waiting.ticket))
      ticket = waiting.ticket
      broken = await self.bus.submit_command("broken", Op.SAMPLE, json.dumps(JOB).encode())
      reply = await self.bus.wait_for_reply(broken, 3)
      self.assertIsNotNone(reply, "a healthy head waiting for GPU must not hold up a preparation error")
      self.assertFalse(reply.ok)
      self.assertIn("bucket error", json.loads(reply.body)["error"])
      self.assertEqual(waiting.ticket, ticket)
      self.assertIsNone(waiting.since)
      self.assertIsNone(node.samplers["waiting"].running)
      self.assertEqual(await self.bus.queued_and_in_flight("ready"), {"live": True, "queued": 1, "in_flight": []})
      self.assertIsNone(await self.bus.wait_for_reply(holding, 0))
    finally:
      finish.set()
    self.assertTrue((await self.bus.wait_for_reply(holding, 5)).ok)
    self.assertTrue((await self.bus.wait_for_reply(ready, 5)).ok)

  async def test_invalid_backlog_does_not_starve_a_ready_sibling(self):
    for run in ("broken", "ready"):
      await self.bus.create_run(Run(run, "sampler", "base", 4, role=Role.SAMPLER))
    await self.bus.r.delete(key("run", "broken"))
    requests = [await self.bus.submit_command("broken", Op.SAMPLE, json.dumps(JOB).encode()) for _ in range(32)]
    ready = await self.bus.submit_command("ready", Op.SAMPLE, json.dumps(JOB).encode())
    completed, write = [], self.bus.write_reply

    async def publish(cmd, *args, **kwargs):
      result = await write(cmd, *args, **kwargs)
      completed.append(cmd.run)
      if cmd.run == "broken":
        await asyncio.sleep(0.005)  # the next invalid head is ready while this publication finishes
      return result

    node = self.node(gpus=1)
    with patch.object(self.bus, "write_reply", side_effect=publish):
      await node.add("sampler")
      reply = await self.bus.wait_for_reply(ready, 5)
      self.assertIsNotNone(reply)
      self.assertTrue(reply.ok)
      await self.until(lambda: "ready" in completed)
      self.assertEqual(completed[0], "broken")
      self.assertLess(completed.index("ready"), len(requests), "ready work waited for the whole invalid backlog")
      self.assertFalse((await self.bus.wait_for_reply(requests[-1], 5)).ok)
    self.assertEqual(int(await self.bus.r.hget(key("run", "broken"), "failed")), len(requests))

  async def test_a_preparation_error_cannot_overwrite_the_new_lease_owners_reply(self):
    store = Store(tempfile.mkdtemp())
    await self.bus.create_run(
      Run("s", "sampler", "base", 4, role=Role.SAMPLER, model_path="tinker://r/sampler_weights/missing")
    )
    staging, finish = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def failing_download(uri, version):
      loop.call_soon_threadsafe(staging.set)
      if not finish.wait(5):
        raise TimeoutError("test never released checkpoint staging")
      raise RuntimeError("old owner's download failed")

    store.local_folder = failing_download
    first = self.node(store, gpus=1, node="first")
    await first.add("sampler")
    sampler, task = first.samplers["sampler"], first.tasks["sampler"]
    request = await self.bus.submit_command("s", Op.SAMPLE, json.dumps(JOB).encode())
    try:
      await asyncio.wait_for(staging.wait(), 5)
      self.assertEqual(await self.bus.queued_and_in_flight("s"), {"live": True, "queued": 1, "in_flight": []})
      await self.bus.r.delete(key("owner", "sampler"))  # expiry while the old node is still preparing
      replacement_store = Store(tempfile.mkdtemp())
      replacement_store.local_folder = MagicMock(side_effect=RuntimeError("new owner's download failed"))
      second = self.node(replacement_store, gpus=1, node="second")
      await second.add("sampler")
      owner = await self.bus.lease_owner("sampler")
      expected = await self.bus.wait_for_reply(request, 5)
      self.assertIsNotNone(expected)
      self.assertFalse(expected.ok)
      self.assertIn("new owner's download failed", json.loads(expected.body)["error"])
      finish.set()
      async with asyncio.timeout(5):
        [error] = await asyncio.gather(task, return_exceptions=True)
      self.assertIsInstance(error, RuntimeError)
      self.assertIn("lost worker", str(error))
      self.assertIsNone(sampler.running)
      self.assertEqual(await self.bus.wait_for_reply(request, 0), expected)
      self.assertEqual((await self.bus.get_run("s"))["failed"], 1)
      self.assertEqual(await self.bus.lease_owner("sampler"), owner)
    finally:
      finish.set()

  async def test_a_new_checkpoint_can_stage_while_both_gpus_and_heartbeats_keep_progressing(self):
    store = Store(tempfile.mkdtemp())
    uri = "tinker://r1/sampler_weights/new"
    store.put(uri, {}, {"base": "base", "lora": True, "config": {}})
    for run, worker, path in (("first", "a", None), ("later", "a", uri), ("waiter", "b", None), ("other", "c", None)):
      await self.bus.create_run(Run(run, worker, "base", 4, role=Role.SAMPLER, model_path=path))
    sampling, finish_sample, staging = asyncio.Event(), asyncio.Event(), asyncio.Event()
    finish_stage = threading.Event()
    loop, loop_thread = asyncio.get_running_loop(), threading.get_ident()
    local = store.local_folder

    def slow_checkpoint(uri, version):
      self.assertNotEqual(threading.get_ident(), loop_thread)
      loop.call_soon_threadsafe(staging.set)
      if not finish_stage.wait(10):
        raise TimeoutError("test never released checkpoint staging")
      return local(uri, version)

    class ControlledEngine(FakeEngine):
      async def sample(self, job, name):
        if job["prompt"] == [11]:
          sampling.set()
          await finish_sample.wait()
        return await super().sample(job, name)

    store.local_folder = MagicMock(side_effect=slow_checkpoint)
    node = self.node(store, gpus=2, engine=ControlledEngine)
    with patch.dict(os.environ, {"TINYRL_WARM_SAMPLER": "0"}):
      for worker, gpus in (("a", [0]), ("b", [0]), ("c", [1])):
        await node.add(worker, gpus=gpus)
    self.pump(node)
    first = await self.bus.submit_command("first", Op.SAMPLE, json.dumps(JOB | {"prompt": [11]}).encode())
    try:
      await asyncio.wait_for(sampling.wait(), 5)
      later = await self.bus.submit_command("later", Op.SAMPLE, json.dumps(JOB).encode())
      await asyncio.wait_for(staging.wait(), 5)
      heartbeat = (await self.bus.live_nodes(details=True))[node.id]["until"]
      other = await self.bus.submit_command("other", Op.SAMPLE, json.dumps(JOB).encode())
      reply = await self.bus.wait_for_reply(other, 5)
      self.assertIsNotNone(reply, "GPU1 was blocked by GPU0's work or checkpoint staging")
      self.assertTrue(reply.ok, reply.body)
      self.assertFalse(finish_stage.is_set())
      self.assertGreater((await self.bus.live_nodes(details=True))[node.id]["until"], heartbeat)
      waiter = await self.bus.submit_command("waiter", Op.SAMPLE, json.dumps(JOB).encode())
      finish_sample.set()
      self.assertTrue((await self.bus.wait_for_reply(first, 5)).ok)
      reply = await self.bus.wait_for_reply(waiter, 5)
      self.assertIsNotNone(reply, "GPU0 stayed reserved until an unrelated checkpoint finished staging")
      self.assertTrue(reply.ok, reply.body)
      self.assertIn(("sleep",), node.samplers["a"].running.calls)
      self.assertIsNone(await self.bus.wait_for_reply(later, 0))
    finally:
      finish_sample.set()
      finish_stage.set()
    self.assertTrue((await self.bus.wait_for_reply(later, 5)).ok)
    self.assertEqual(store.local_folder.call_count, 1)


if __name__ == "__main__":
  unittest.main()
