import asyncio
import contextlib
import hashlib
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock

import httpx
import safetensors.torch
import torch

from test.execution import TestNode
from test.helpers import FakeEngine, RedisTest, tiny
from tinyrl.api import wire
from tinyrl.api.gateway import app
from tinyrl.bus import Ev, Op, Role, Run
from tinyrl.control import scheduler
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


class Slow(FakeEngine):
  async def sample(self, job: dict, name: str | None):
    await asyncio.sleep(0.02)
    return await super().sample(job, name)


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

  async def sample(self, run: str) -> str:
    return await self.bus.submit_command(run, Op.SAMPLE, json.dumps(JOB).encode())

  @contextlib.asynccontextmanager
  async def traffic(self, run: str):
    stop = asyncio.Event()

    async def submit():
      while not stop.is_set():
        await self.sample(run)
        await asyncio.sleep(0.005)

    task = asyncio.create_task(submit())
    try:
      yield
    finally:
      stop.set()
      await task

  async def test_a_name_saved_again_is_staged_again_while_the_sampler_stays_busy(self):
    store, uri = Store(tempfile.mkdtemp()), "tinker://r1/sampler_weights/latest"

    async def save(value: float) -> str:
      store.put(uri, {"w": torch.full((4,), value)}, {"base": "base", "lora": True, "config": {"r": 4}})
      await self.registry.register_checkpoint(uri, 16)
      return hashlib.sha256(f"{uri}\0{store.local_folder(uri)[1]}".encode()).hexdigest()

    await save(1.0)
    await self.bus.create_run(Run("s1", "sampler:base", "base", 1, role=Role.SAMPLER, model_path=uri))
    node = self.node(store, engine=Slow)
    lw = await node.add("sampler:base")
    async with self.traffic("s1"):
      await asyncio.sleep(0.3)
      second = await save(2.0)
      await self.bus.create_run(Run("s2", "sampler:base", "base", 1, role=Role.SAMPLER, model_path=uri))
      await node.sync()
      self.assertTrue((await self.bus.wait_for_reply(await self.sample("s2"), 10)).ok)
    loads = [name for call, name, *_ in lw.engine.running.calls if call == "load"]
    self.assertIn(second, loads, "a session opened on the re-saved name sampled the weights staged earlier")

  async def test_a_staging_error_fails_only_the_commands_of_its_step(self):
    store, uris = Store(tempfile.mkdtemp()), ("tinker://r1/sampler_weights/a", "tinker://r2/sampler_weights/b")
    for uri in uris:
      store.put(uri, {"w": torch.zeros(4)}, {"base": "base", "lora": True, "config": {"r": 4}})
    local_folder, failing = store.local_folder, []

    def flaky(uri: str, version=None):
      if uri == uris[1] and failing:
        raise OSError(failing.pop())
      return local_folder(uri, version)

    store.local_folder = flaky
    for run, uri in zip(("s1", "s2"), uris):
      await self.bus.create_run(Run(run, "sampler:base", "base", 1, role=Role.SAMPLER, model_path=uri))
    node = self.node(store, engine=Slow)
    await node.add("sampler:base")
    async with self.traffic("s1"):
      await asyncio.sleep(0.3)
      failing.append("503 from the bucket")
      replies = [await self.bus.wait_for_reply(await self.sample("s2"), 10) for _ in range(5)]
    self.assertEqual(failing, [])
    self.assertLessEqual([r.ok for r in replies].count(False), 1, "the error outlived the step that hit it")
    self.assertTrue(replies[-1].ok)

  async def test_a_stale_claim_is_failed_while_the_sampler_stays_busy(self):
    await self.bus.create_run(Run("s1", "sampler:base", "base", 1, role=Role.SAMPLER))
    node = self.node(engine=Slow)
    await node.add("sampler:base")
    async with self.traffic("s1"):
      await asyncio.sleep(0.1)
      stranded = await self.sample("s1")
      self.assertIn(stranded, [c.request_id for c in await self.bus.take_commands(["s1"], "dead", 1000)])
      reply = await self.bus.wait_for_reply(stranded, 1)
    self.assertIsNotNone(reply, "a dead owner's claim was not failed by the step after it while the worker had work")
    self.assertFalse(reply.ok)

  async def test_an_idle_session_s_slow_staging_does_not_delay_a_busy_one(self):
    store, uri = Store(tempfile.mkdtemp()), "tinker://r2/sampler_weights/slow"
    store.put(uri, {"w": torch.zeros(4)}, {"base": "base", "lora": True, "config": {"r": 4}})
    local_folder = store.local_folder

    def waiting(u: str, version=None):
      if u == uri:
        time.sleep(1.0)
      return local_folder(u, version)

    store.local_folder = waiting
    await self.bus.create_run(Run("s1", "sampler:base", "base", 1, role=Role.SAMPLER))
    await self.bus.create_run(Run("s2", "sampler:base", "base", 1, role=Role.SAMPLER, model_path=uri))
    await self.node(store).add("sampler:base")
    for _ in range(3):
      start = time.monotonic()
      self.assertTrue((await self.bus.wait_for_reply(await self.sample("s1"), 10)).ok)
      self.assertLess(time.monotonic() - start, 0.5, "an idle session's staging held up a busy one")

  async def test_a_burst_over_three_sessions_reaches_the_engine_together(self):
    in_flight, peaks = [0], []

    class Overlapping(Slow):
      async def sample(self, job: dict, name: str | None):
        in_flight[0] += 1
        peaks.append(in_flight[0])
        try:
          return await super().sample(job, name)
        finally:
          in_flight[0] -= 1

    sessions = [f"s{i}" for i in range(5)]
    for session in sessions:
      await self.bus.create_run(Run(session, "sampler:base", "base", 1, role=Role.SAMPLER))
    await self.node(engine=Overlapping).add("sampler:base")
    for _ in range(3):
      await asyncio.sleep(0.4)
      peaks.clear()
      ids = []
      for session in sessions[:3]:
        ids.append(await self.sample(session))
        await asyncio.sleep(0.002)
      self.assertTrue(all([(await self.bus.wait_for_reply(i, 10)).ok for i in ids]))
      self.assertEqual(max(peaks), 3, f"the burst reached the engine as {peaks}")

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
      return json.loads((await self.bus.wait_for_reply(request_id, 10)).body)["sampling_session_id"]

    run = (await call("create_model", {"base_model": base, "lora_config": {"rank": 4}}))["model_id"]
    await node.drain(run)
    sid = await save_for_sampler(run)
    sampler_worker = await node.worker_for(sid)
    sampler = (await node.add(sampler_worker)).engine
    self.assertTrue((await client.get(f"/api/v1/samplers/{sid}")).json()["model_path"].startswith("tinker://"))
    first = await call("join_sampling_session", {"sampling_session_id": sid})
    second = await call("join_sampling_session", {"sampling_session_id": sid})
    self.assertEqual(second["client_counter"], first["client_counter"] + 1)

    job = {"sampling_session_id": sid, "seq_id": 7, "prompt": tokens([1, 2, 3]), "prompt_logprobs": True}
    replies = []
    for _ in range(2):  # the SDK retrying with the same seq_id gets the first answer
      request_id = (await call("asample", job))["request_id"]
      replies.append(await call("retrieve_future", {"request_id": request_id}))
    self.assertEqual(replies[0], replies[1])
    self.assertEqual(replies[0]["prompt_logprobs"], [None, -1.0, -1.0])

    request_id = (await call("asample", {"base_model": base, "prompt": tokens([1, 2])}))["request_id"]
    self.assertEqual(await node.worker_for(request_id.split("/")[0]), sampler_worker)
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
    await node.drain(fft)
    fft_sid = await save_for_sampler(fft)
    fft_sampler_worker = await node.worker_for(fft_sid)
    self.assertNotEqual(fft_sampler_worker, sampler_worker)
    fft_sampler = (await node.add(fft_sampler_worker)).engine
    request_id = (await call("asample", {"sampling_session_id": fft_sid, "prompt": tokens([1, 2])}))["request_id"]
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
    trainer = await node.add("t1", gpus=trainer_gpus)
    trainer.engine.process.trainer.model.create_run("t1", "base", 0, {})
    sampler = await node.add("sampler:base", gpus=sampler_gpus)
    for run, op, body in [
      ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
      ("samp-1", Op.SAMPLE, JOB),
      ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
    ]:
      reply = await self.bus.wait_for_reply(await self.bus.submit_command(run, op, json.dumps(body).encode()), 10)
      self.assertTrue(reply.ok, reply.body)
    engine = sampler.engine.running
    self.assertIn(("sleep",), engine.calls)
    await node.aclose()
    self.assertEqual(engine.calls[-1], ("close",))

  async def test_engine_boot_does_not_consume_a_sampler_s_turn(self):
    class SlowBoot(FakeEngine):
      @classmethod
      async def start(cls, base: str, lora: bool, env=None):
        await asyncio.sleep(0.6)
        return await super().start(base, lora, env)

    await self.bus.create_run(Run("s1", "sampler:base", "base", 1, role=Role.SAMPLER))
    await self.bus.create_run(Run("rt", "t", "base", 0))
    node = self.node(gpus=1, engine=SlowBoot)
    node.state.quantum = 0.2
    notes = []
    note = node.note
    node.note = lambda kind, **body: (notes.append((kind, dict(body))), note(kind, **body))
    await node.add("sampler:base", gpus=[0])
    trainer = await node.add("t", gpus=[0])
    trainer.engine.process.trainer.model.create_run("rt", "base", 0, {})
    sampled = await self.bus.submit_command("s1", Op.SAMPLE, json.dumps(JOB).encode())
    for _ in range(100):
      if scheduler.holders(node.state, (node.id, 0)) == ["sampler:base"]:
        break
      await asyncio.sleep(0.02)
    trained = await self.bus.submit_command("rt", Op.OPTIM_STEP, json.dumps({"adam_params": {}}).encode())
    self.assertTrue((await self.bus.wait_for_reply(sampled, 10)).ok)
    self.assertTrue((await self.bus.wait_for_reply(trained, 10)).ok)
    boot = notes.index(next(n for n in notes if n[0] == Ev.BATCH_DONE and n[1].get("boot")))
    revoked = [n for n in notes[:boot] if n[0] == Ev.REVOKED and n[1]["worker"] == "sampler:base"]
    self.assertEqual(revoked, [])

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
    self.assertTrue(all([(await self.bus.wait_for_reply(i, 10)).ok for i in ids]))
    self.assertEqual((starts, loads, peak, store.local_folder.call_count), (1, 1, 8, 1))

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
    booting, queued = asyncio.Event(), asyncio.Event()
    active = peak = 0

    class SlowStartEngine(FakeEngine):
      @classmethod
      async def start(cls, base: str, lora: bool, env=None):
        booting.set()
        await queued.wait()
        return await super().start(base, lora, env)

      async def sample(self, job: dict, name: str | None) -> Out:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return await super().sample(job, name)

    node = self.node(store, engine=SlowStartEngine, gpus=1)
    ids = [await self.bus.submit_command("sess", Op.SAMPLE, json.dumps(JOB).encode())]
    await node.add("sampler:base")
    await asyncio.wait_for(booting.wait(), 5)
    ids += [await self.bus.submit_command("sess", Op.SAMPLE, json.dumps(JOB).encode()) for _ in range(2)]
    queued.set()
    self.assertTrue(all([(await self.bus.wait_for_reply(i, 10)).ok for i in ids]))
    self.assertEqual(peak, 3)

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
    node = self.node(store)
    ids = [await self.bus.submit_command(run, Op.SAMPLE, json.dumps(JOB).encode()) for run in ("s1", "s2", "s1", "s2")]
    lw = await node.add("sampler:fft:base")
    self.assertTrue(all([(await self.bus.wait_for_reply(i, 10)).ok for i in ids]))
    engine = lw.engine.running
    loads = [c for c in engine.calls if c[0] == "load"]
    self.assertEqual(len(loads), 2)
    self.assertNotEqual(loads[0][1], loads[1][1])
    first_sess = "s1" if "/r1/" in loads[0][2] else "s2"
    engine.calls.clear()
    request = await self.bus.submit_command(first_sess, Op.SAMPLE, json.dumps(JOB).encode())
    self.assertTrue((await self.bus.wait_for_reply(request, 10)).ok)
    self.assertEqual([c for c in engine.calls if c[0] == "load"], [loads[0]])

  async def test_warm_sampler_starts_engine_before_sample(self):
    await self.bus.create_run(Run("s", "sampler:base", "base", 1, role=Role.SAMPLER))
    node = self.node(gpus=2)
    lw = await node.add("sampler:base")
    for _ in range(50):
      if lw.engine.started:
        break
      await asyncio.sleep(0.02)
    self.assertTrue(lw.engine.started)
    self.assertEqual(lw.engine.running.calls[0], ("start", "base", True))

  async def test_failed_warm_start_replies_with_error_and_the_sampler_keeps_going(self):
    class Broken(FakeEngine):
      @classmethod
      async def start(cls, base: str, lora: bool, env=None):
        raise RuntimeError("no GPU memory")

    await self.bus.create_run(Run("s", "sampler:base", "base", 1, role=Role.SAMPLER))
    node = self.node(gpus=2, engine=Broken)
    lw = await node.add("sampler:base")
    command = await self.bus.submit_command("s", Op.SAMPLE, json.dumps(JOB).encode())
    reply = await self.bus.wait_for_reply(command, 5)
    self.assertFalse(reply.ok)
    self.assertIn("no GPU memory", json.loads(reply.body)["error"])
    self.assertFalse(lw.task.done())
    lw.engine.engine_type = FakeEngine
    command = await self.bus.submit_command("s", Op.SAMPLE, json.dumps(JOB).encode())
    self.assertTrue((await self.bus.wait_for_reply(command, 5)).ok)

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

  async def test_a_sampler_stages_checkpoints_before_asking_for_its_turn(self):
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
    command = await self.bus.submit_command("samp-1", Op.SAMPLE, json.dumps(JOB).encode())
    await node.add("sampler:base")
    try:
      await asyncio.wait_for(blocking.wait(), timeout=5)
      worker = node.state.workers["sampler:base"]
      self.assertEqual(worker.asked, 0)
      self.assertIsNone(worker.since)
    finally:
      blocked.set()
    self.assertTrue((await self.bus.wait_for_reply(command, 5)).ok)

  async def test_a_new_session_s_sample_sent_after_the_sampler_peeked_runs_on_its_own_checkpoint(self):
    store = Store(tempfile.mkdtemp())
    uris = {"s1": "tinker://r1/sampler_weights/s0", "s2": "tinker://r2/sampler_weights/s0"}
    for run, uri in uris.items():
      store.put(uri, {"w": torch.zeros(4)}, {"base": "base", "lora": True, "config": {"r": 4}})
      await self.bus.create_run(Run(run, "sampler:base", "base", 1, role=Role.SAMPLER, model_path=uri))
    real_local, loop = store.local_folder, asyncio.get_running_loop()
    staging, go = asyncio.Event(), threading.Event()

    def held(uri, version):
      if uri == uris["s1"]:
        loop.call_soon_threadsafe(staging.set)
        go.wait(5)
      return real_local(uri, version)

    store.local_folder = held
    lw = await self.node(store).add("sampler:base")
    await self.sample("s1")
    await asyncio.wait_for(staging.wait(), 5)
    command = await self.sample("s2")
    go.set()
    reply = await self.bus.wait_for_reply(command, 10)
    self.assertTrue(reply.ok, reply.body)
    loads = [call[2] for call in lw.engine.running.calls if call[0] == "load"]
    self.assertIn(store.path(uris["s2"]), loads, "s2 sampled without its checkpoint")

  async def test_a_run_whose_checkpoint_fails_to_stage_gets_an_error_and_the_sampler_keeps_serving(self):
    store = Store(tempfile.mkdtemp())
    await self.bus.create_run(
      Run("samp-1", "sampler:base", "base", 1, role=Role.SAMPLER, model_path="tinker://r1/sampler_weights/s0")
    )
    store.local_folder = MagicMock(side_effect=RuntimeError("bucket error"))
    node = self.node(store)
    await node.add("sampler:base")
    cmd_id = await self.bus.submit_command("samp-1", Op.SAMPLE, json.dumps(JOB).encode())
    reply = await self.bus.wait_for_reply(cmd_id, 5)
    self.assertFalse(reply.ok)
    self.assertIn("RuntimeError: bucket error", json.loads(reply.body)["error"])

    await self.bus.create_run(Run("base", "sampler:base", "base", 1, role=Role.SAMPLER))
    await node.sync()
    cmd_id = await self.bus.submit_command("base", Op.SAMPLE, json.dumps(JOB).encode())
    self.assertTrue((await self.bus.wait_for_reply(cmd_id, 5)).ok)


if __name__ == "__main__":
  unittest.main()
