import asyncio
import json
import tempfile
import threading
import unittest
from unittest.mock import MagicMock

import httpx
import safetensors.torch
import torch

from test.execution import TestNode
from test.helpers import FakeEngine, RedisTest, tiny
from tinyrl.api import wire
from tinyrl.api.gateway import app
from tinyrl.bus import Op, Role, Run
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

    async def save_for_sampler(worker: str, run: str) -> str:
      request_id = (await call("save_weights_for_sampler", {"model_id": run, "sampling_session_seq_id": 0}))[
        "request_id"
      ]
      await node.step(worker, block_ms=0)
      return json.loads((await self.bus.wait_for_reply(request_id, 1)).body)["sampling_session_id"]

    run = (await call("create_model", {"base_model": base, "lora_config": {"rank": 4}}))["model_id"]
    trainer_worker = await node.worker_for(run)
    await node.add(trainer_worker)
    await node.step(trainer_worker, block_ms=0)
    sid = await save_for_sampler(trainer_worker, run)
    sampler_worker = await node.worker_for(sid)
    sampler = (await node.add(sampler_worker)).sampler
    self.assertTrue((await client.get(f"/api/v1/samplers/{sid}")).json()["model_path"].startswith("tinker://"))
    first = await call("join_sampling_session", {"sampling_session_id": sid})
    second = await call("join_sampling_session", {"sampling_session_id": sid})
    self.assertEqual(second["client_counter"], first["client_counter"] + 1)

    job = {"sampling_session_id": sid, "seq_id": 7, "prompt": tokens([1, 2, 3]), "prompt_logprobs": True}
    replies = []
    for _ in range(2):  # the SDK retrying with the same seq_id gets the first answer
      request_id = (await call("asample", job))["request_id"]
      await node.step(sampler_worker, block_ms=0)
      replies.append(await call("retrieve_future", {"request_id": request_id}))
    self.assertEqual(replies[0], replies[1])
    self.assertEqual(replies[0]["prompt_logprobs"], [None, -1.0, -1.0])

    request_id = (await call("asample", {"base_model": base, "prompt": tokens([1, 2])}))["request_id"]
    self.assertEqual(await node.worker_for(request_id.split("/")[0]), sampler_worker)
    await node.step(sampler_worker, block_ms=0)
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
    await node.step(fft_worker, block_ms=0)
    fft_sid = await save_for_sampler(fft_worker, fft)
    fft_sampler_worker = await node.worker_for(fft_sid)
    self.assertNotEqual(fft_sampler_worker, sampler_worker)
    fft_sampler = (await node.add(fft_sampler_worker)).sampler
    request_id = (await call("asample", {"sampling_session_id": fft_sid, "prompt": tokens([1, 2])}))["request_id"]
    await node.step(fft_sampler_worker, block_ms=0)
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
    trainer.model.trainer.model.create_run("t1", "base", 0, {})
    sampler = await node.add("sampler:base", gpus=sampler_gpus)
    trainer.task = asyncio.create_task(node.serve_worker("t1"))
    sampler.task = asyncio.create_task(node.serve_worker("sampler:base"))
    try:
      for run, op, body in [
        ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
        ("samp-1", Op.SAMPLE, JOB),
        ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
      ]:
        reply = await self.bus.wait_for_reply(await self.bus.submit_command(run, op, json.dumps(body).encode()), 5)
        self.assertTrue(reply.ok, reply.body)
      engine = sampler.sampler.running
      self.assertIn(("sleep",), engine.calls)
    finally:
      trainer.stopping = sampler.stopping = True
      await asyncio.gather(trainer.task, sampler.task)
    self.assertEqual(engine.calls[-1], ("close",))

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
    await node.add("sampler:base")
    ids = [await self.bus.submit_command("samp-1", Op.SAMPLE, json.dumps(JOB).encode()) for _ in range(8)]
    self.assertEqual(await node.step("sampler:base", block_ms=0), 8)
    self.assertEqual((starts, loads, peak, store.local_folder.call_count), (1, 1, 8, 1))
    self.assertTrue(all([(await self.bus.wait_for_reply(i, 1)).ok for i in ids]))

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

    class SlowStartEngine(FakeEngine):
      @classmethod
      async def start(cls, base: str, lora: bool, env=None):
        booting.set()
        await queued.wait()
        return await super().start(base, lora, env)

    node = self.node(store, engine=SlowStartEngine)
    await node.add("sampler:base")

    await self.bus.submit_command("sess", Op.SAMPLE, json.dumps(JOB).encode())

    async def push_more():
      await booting.wait()
      await self.bus.submit_command("sess", Op.SAMPLE, json.dumps(JOB).encode())
      await self.bus.submit_command("sess", Op.SAMPLE, json.dumps(JOB).encode())
      queued.set()

    pushing = asyncio.create_task(push_more())

    handled = await node.step("sampler:base", block_ms=200)
    await pushing
    self.assertEqual(handled, 3)

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
    running = await node.add("sampler:fft:base")
    for run in ("s1", "s2", "s1", "s2"):
      await self.bus.submit_command(run, Op.SAMPLE, json.dumps(JOB).encode())
    self.assertEqual(await node.step("sampler:fft:base", block_ms=10), 4)
    engine = running.sampler.running
    loads = [c for c in engine.calls if c[0] == "load"]
    self.assertEqual(len(loads), 2)
    self.assertNotEqual(loads[0][1], loads[1][1])
    first_sess = "s1" if "/r1/" in loads[0][2] else "s2"
    engine.calls.clear()
    await self.bus.submit_command(first_sess, Op.SAMPLE, json.dumps(JOB).encode())
    self.assertEqual(await node.step("sampler:fft:base", block_ms=10), 1)
    self.assertEqual([c for c in engine.calls if c[0] == "load"], [loads[0]])

  async def test_warm_sampler_starts_engine_before_sample(self):
    await self.bus.create_run(Run("s", "sampler:base", "base", 1, role=Role.SAMPLER))
    node = self.node(gpus=2)
    running = await node.add("sampler:base")
    running.task = asyncio.create_task(node.serve_worker("sampler:base"))
    try:
      for _ in range(50):
        if running.sampler.running is not None:
          break
        await asyncio.sleep(0.02)
      self.assertIsNotNone(running.sampler.running)
      self.assertEqual(running.sampler.running.calls[0], ("start", "base", True))
    finally:
      running.stopping = True
      await running.task

  async def test_failed_warm_start_replies_with_error_and_the_sampler_keeps_going(self):
    class Broken(FakeEngine):
      @classmethod
      async def start(cls, base: str, lora: bool, env=None):
        raise RuntimeError("no GPU memory")

    await self.bus.create_run(Run("s", "sampler:base", "base", 1, role=Role.SAMPLER))
    node = self.node(gpus=2, engine=Broken)
    running = await node.add("sampler:base")
    running.task = asyncio.create_task(node.serve_worker("sampler:base"))
    try:
      command = await self.bus.submit_command("s", Op.SAMPLE, json.dumps(JOB).encode())
      reply = await self.bus.wait_for_reply(command, 2)
      self.assertFalse(reply.ok)
      self.assertIn("no GPU memory", json.loads(reply.body)["error"])
      self.assertFalse(running.task.done())
      running.sampler.engine_type = FakeEngine
      command = await self.bus.submit_command("s", Op.SAMPLE, json.dumps(JOB).encode())
      self.assertTrue((await self.bus.wait_for_reply(command, 2)).ok)
    finally:
      running.stopping = True
      await running.task

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
    running = await node.add("sampler:base")
    command = await self.bus.submit_command("samp-1", Op.SAMPLE, json.dumps(JOB).encode())
    running.task = asyncio.create_task(node.step("sampler:base", block_ms=0))
    try:
      await asyncio.wait_for(blocking.wait(), timeout=2)
      worker = node.state.workers["sampler:base"]
      self.assertEqual(worker.asked, 0)
      self.assertIsNone(worker.since)
    finally:
      blocked.set()
    self.assertEqual(await asyncio.wait_for(running.task, 2), 1)
    self.assertTrue((await self.bus.wait_for_reply(command, 1)).ok)

  async def test_a_run_whose_checkpoint_fails_to_stage_gets_an_error_and_the_sampler_keeps_serving(self):
    store = Store(tempfile.mkdtemp())
    await self.bus.create_run(
      Run("samp-1", "sampler:base", "base", 1, role=Role.SAMPLER, model_path="tinker://r1/sampler_weights/s0")
    )
    store.local_folder = MagicMock(side_effect=RuntimeError("bucket error"))
    node = self.node(store)
    await node.add("sampler:base")
    cmd_id = await self.bus.submit_command("samp-1", Op.SAMPLE, json.dumps(JOB).encode())
    self.assertEqual(await node.step("sampler:base", block_ms=0), 1)
    reply = await self.bus.wait_for_reply(cmd_id, 1)
    self.assertFalse(reply.ok)
    self.assertIn("RuntimeError: bucket error", json.loads(reply.body)["error"])

    await self.bus.create_run(Run("base", "sampler:base", "base", 1, role=Role.SAMPLER))
    await node.sync()
    cmd_id = await self.bus.submit_command("base", Op.SAMPLE, json.dumps(JOB).encode())
    self.assertEqual(await node.step("sampler:base", block_ms=0), 1)
    self.assertTrue((await self.bus.wait_for_reply(cmd_id, 1)).ok)


if __name__ == "__main__":
  unittest.main()
