import asyncio
import io
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import httpx
import safetensors.torch
import torch

from test.helpers import FakeEngine, RedisTest, tiny
from tinyrl.api import wire
from tinyrl.api.gateway import app
from tinyrl.bus import Op, Run
from tinyrl.storage.store import Store
from tinyrl.workers.sampler import Out, Sampler, Seq, encode
from tinyrl.workers.trainer import Trainer
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
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app(self.bus, store)), base_url="http://test")

    async def call(path: str, body: dict) -> dict:
      return (await client.post(f"/api/v1/{path}", json=body)).json()

    async def save_for_sampler(trainer: Trainer, run: str) -> str:
      request_id = (await call("save_weights_for_sampler", {"model_id": run, "sampling_session_seq_id": 0}))[
        "request_id"
      ]
      await trainer.step(block_ms=0)
      return json.loads((await self.bus.wait_for_reply(request_id, 1)).body)["sampling_session_id"]

    run = (await call("create_model", {"base_model": base, "lora_config": {"rank": 4}}))["model_id"]
    trainer = Trainer(self.bus, await self.worker(run), store=store)
    await self.own(trainer)
    await trainer.step(block_ms=0)

    sid = await save_for_sampler(trainer, run)
    sampler = Sampler(self.bus, await self.worker(sid), engine=FakeEngine, store=store)
    await self.own(sampler)
    self.assertTrue((await client.get(f"/api/v1/samplers/{sid}")).json()["model_path"].startswith("tinker://"))
    first = await call("join_sampling_session", {"sampling_session_id": sid})
    second = await call("join_sampling_session", {"sampling_session_id": sid})
    self.assertEqual(second["client_counter"], first["client_counter"] + 1)

    job = {"sampling_session_id": sid, "seq_id": 7, "prompt": tokens([1, 2, 3]), "prompt_logprobs": True}
    replies = []
    for _ in range(2):  # the SDK retrying with the same seq_id gets the first answer
      request_id = (await call("asample", job))["request_id"]
      await sampler.step(block_ms=0)
      replies.append(await call("retrieve_future", {"request_id": request_id}))
    self.assertEqual(replies[0], replies[1])
    self.assertEqual(replies[0]["prompt_logprobs"], [None, -1.0, -1.0])

    request_id = (await call("asample", {"base_model": base, "prompt": tokens([1, 2])}))["request_id"]
    # base sampling joins the same sampler
    self.assertEqual(await self.worker(request_id.split("/")[0]), sampler.worker)
    await sampler.step(block_ms=0)
    self.assertEqual(len((await call("retrieve_future", {"request_id": request_id}))["sequences"]), 1)
    calls = sampler.engine.calls
    self.assertEqual(calls[0], ("start", base, True))
    self.assertEqual(
      [c for c in calls if c[0] == "load"],
      [("load", "c0", store.path(f"tinker://{run}/sampler_weights/step-0"))],
    )
    self.assertEqual([c for c in calls if c[0] == "sample"], [("sample", "c0"), ("sample", None)])  # the retry ran once

    fft = (await call("create_model", {"base_model": base, "user_metadata": {"openrl.fine_tuning_type": "full"}}))[
      "model_id"
    ]
    fft_trainer = Trainer(self.bus, await self.worker(fft), store=store)
    await self.own(fft_trainer)
    await fft_trainer.step(block_ms=0)
    fft_sid = await save_for_sampler(fft_trainer, fft)
    self.assertNotEqual(await self.worker(fft_sid), await self.worker(sid))
    with patch.dict(os.environ, {"TINYRL_LORA": "0"}):
      fft_sampler = Sampler(self.bus, await self.worker(fft_sid), engine=FakeEngine, store=store)
    await self.own(fft_sampler)
    request_id = (await call("asample", {"sampling_session_id": fft_sid, "prompt": tokens([1, 2])}))["request_id"]
    await fft_sampler.step(block_ms=0)
    res = await client.post("/api/v1/retrieve_future", json={"request_id": request_id}, headers={"Accept": wire.PROTO})
    self.assertEqual(res.headers["content-type"], wire.PROTO)
    self.assertIn(("load", "c0", store.path(f"tinker://{fft}/sampler_weights/step-0")), fft_sampler.engine.calls)
    await client.aclose()

  @patch("tinyrl.workers.trainer.GPU", "node-a:0")
  @patch("tinyrl.workers.sampler.GPU", "node-a:0")
  async def test_a_trainer_and_a_sampler_take_turns_on_one_gpu(self):
    await self.run_scheduler({"t1": [0], "sampler:base": [0]})
    await self.take_turns()

  @patch("tinyrl.workers.trainer.GPU", "node-a:1")
  @patch("tinyrl.workers.sampler.GPU", "node-a:0,node-a:1")
  async def test_a_tensor_parallel_sampler_takes_turns_on_both_its_gpus(self):
    await self.run_scheduler({"t1": [1], "sampler:base": [0, 1]})
    await self.take_turns()

  async def take_turns(self):
    await self.bus.create_run(Run("t1", "t1", "base", 0))
    await self.bus.create_run(Run("samp-1", "sampler:base", "base", 1))
    model = MagicMock()
    model.optim_step.return_value = {"grad_norm:mean": 0.5}
    model.params = {"t1": {}}  # the trainer holds run t1
    trainer = Trainer(self.bus, "t1", model=model)
    sampler = Sampler(self.bus, "sampler:base", engine=FakeEngine)

    tasks = [asyncio.create_task(trainer.serve()), asyncio.create_task(sampler.serve())]
    try:
      for run, op, body in [
        ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
        ("samp-1", Op.SAMPLE, JOB),
        ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
      ]:
        reply = await self.bus.wait_for_reply(await self.bus.submit_command(run, op, json.dumps(body).encode()), 5)
        self.assertTrue(reply.ok, reply.body)
      self.assertIn(("sleep",), sampler.engine.calls)
    finally:
      trainer.done = sampler.done = True
      await asyncio.gather(*tasks)
    self.assertEqual(sampler.engine.calls[-1], ("close",))

  async def test_a_sampler_batches_queued_commands_concurrently_and_loads_once(self):
    store = Store(tempfile.mkdtemp())
    store.put("tinker://r1/sampler_weights/s0", {}, {"base": "base", "lora": True, "config": {}})
    await self.bus.create_run(Run("samp-1", "sampler:base", "base", 1, model_path="tinker://r1/sampler_weights/s0"))
    active = peak = starts = loads = 0

    class BatchEngine(FakeEngine):
      @classmethod
      async def start(cls, base: str, lora: bool):
        nonlocal starts
        await asyncio.sleep(0.01)
        starts += 1
        return await super().start(base, lora)

      async def load_checkpoint(self, name: str, path: str) -> None:
        nonlocal loads
        await asyncio.sleep(0.01)
        if ("load", name, path) not in self.calls:
          loads += 1
        await super().load_checkpoint(name, path)

      async def sample(self, job: dict, name: str | None) -> Out:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return await super().sample(job, name)

    sampler = Sampler(self.bus, "sampler:base", engine=BatchEngine, store=store)
    await self.own(sampler)
    ids = [await self.bus.submit_command("samp-1", Op.SAMPLE, json.dumps(JOB).encode()) for _ in range(8)]
    self.assertEqual(await sampler.step(block_ms=0), 8)
    self.assertEqual((starts, loads, peak), (1, 1, 8))
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

  @patch("tinyrl.workers.vllm.subprocess.Popen")
  @patch("tinyrl.workers.vllm.httpx.AsyncClient.get")
  async def test_vllm_start_compilation_config(self, mock_get, mock_popen):
    mock_popen.return_value.poll.return_value = None
    mock_get.return_value.is_success = True

    from tinyrl.workers.vllm import CUDAGRAPH_CONFIG, VLLM

    with patch("tinyrl.workers.vllm.VLLM_ARGS", ""):
      await (await VLLM.start("base", False)).http.aclose()
      args = mock_popen.call_args[0][0]
      self.assertIn("--compilation-config", args)
      idx = args.index("--compilation-config")
      self.assertEqual(args[idx + 1], CUDAGRAPH_CONFIG)

    with patch("tinyrl.workers.vllm.VLLM_ARGS", "--compilation-config '{}'"):
      await (await VLLM.start("base", False)).http.aclose()
      args = mock_popen.call_args[0][0]
      self.assertIn("--compilation-config", args)
      idx = args.index("--compilation-config")
      self.assertEqual(args[idx + 1], "{}")
      self.assertEqual(args.count("--compilation-config"), 1)
      self.assertNotIn(CUDAGRAPH_CONFIG, args)

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

  async def test_a_sampler_takes_the_commands_that_arrive_while_its_engine_boots(self):
    store = Store(tempfile.mkdtemp())
    store.put("tinker://r1/sampler_weights/s0", {}, {"base": "base", "lora": True, "config": {}})
    await self.bus.create_run(Run("sess", "sampler:base", "base", 1, model_path="tinker://r1/sampler_weights/s0"))

    class SlowStartEngine(FakeEngine):
      @classmethod
      async def start(cls, base: str, lora: bool):
        await asyncio.sleep(0.05)
        return await super().start(base, lora)

    sampler = Sampler(self.bus, "sampler:base", engine=SlowStartEngine, store=store)
    await self.own(sampler)

    await self.bus.submit_command("sess", Op.SAMPLE, json.dumps(JOB).encode())

    async def push_more():
      await asyncio.sleep(0.01)
      await self.bus.submit_command("sess", Op.SAMPLE, json.dumps(JOB).encode())
      await self.bus.submit_command("sess", Op.SAMPLE, json.dumps(JOB).encode())

    asyncio.create_task(push_more())

    handled = await sampler.step(block_ms=200)
    self.assertEqual(handled, 3)

  async def test_sampler_step_groups_non_exclusive_fft_interleaved(self):
    store = Store(tempfile.mkdtemp())
    store.put("tinker://r1/sampler_weights/s0", {}, {"base": "base", "lora": False, "config": {}})
    store.put("tinker://r2/sampler_weights/s0", {}, {"base": "base", "lora": False, "config": {}})
    await self.bus.create_run(Run("s1", "sampler:fft:base", "base", 0, model_path="tinker://r1/sampler_weights/s0"))
    await self.bus.create_run(Run("s2", "sampler:fft:base", "base", 0, model_path="tinker://r2/sampler_weights/s0"))

    loads = []

    class RecordEngine(FakeEngine):
      async def load_checkpoint(self, name: str, path: str) -> None:
        loads.append((name, path))
        await super().load_checkpoint(name, path)

    with patch.dict(os.environ, {"TINYRL_LORA": "0"}):
      sampler = Sampler(self.bus, "sampler:fft:base", engine=RecordEngine, store=store)
    await self.own(sampler)

    await self.bus.submit_command("s1", Op.SAMPLE, json.dumps(JOB).encode())
    await self.bus.submit_command("s2", Op.SAMPLE, json.dumps(JOB).encode())
    await self.bus.submit_command("s1", Op.SAMPLE, json.dumps(JOB).encode())
    await self.bus.submit_command("s2", Op.SAMPLE, json.dumps(JOB).encode())

    handled = await sampler.step(block_ms=10)
    self.assertEqual(handled, 4)

    self.assertEqual([n for n, _ in loads], ["c0", "c1"])
    first_sess = "s1" if "/r1/" in loads[0][1] else "s2"
    loads.clear()

    await self.bus.submit_command(first_sess, Op.SAMPLE, json.dumps(JOB).encode())
    handled = await sampler.step(block_ms=10)
    self.assertEqual(handled, 1)
    self.assertEqual([n for n, _ in loads], ["c0"])

  @patch("subprocess.Popen")
  async def test_vllm_start_appends_memory_and_kv_args_from_environ(self, mock_popen):
    mock_popen.return_value.poll.return_value = 0
    import os

    from tinyrl.workers.vllm import VLLM

    with patch.dict(os.environ, {"TINYRL_GPU_MEM": "0.42", "TINYRL_KV_BYTES": str(16 * 1024**3)}):
      with (
        patch("tinyrl.workers.vllm.VLLM_ARGS", "--gpu-memory-utilization 0.8"),
        patch("tinyrl.workers.vllm.GPU_MEM", "0.42"),
        patch("tinyrl.workers.vllm.KV_BYTES", str(16 * 1024**3)),
      ):
        with self.assertRaises(RuntimeError):
          await VLLM.start("base", False)
        args = mock_popen.call_args[0][0]
        child_env = mock_popen.call_args[1]["env"]
        self.assertEqual(args[-4:], ["--gpu-memory-utilization", "0.42", "--kv-cache-memory-bytes", str(16 * 1024**3)])
        self.assertNotIn("TINYRL_GPU_MEM", child_env)
        self.assertNotIn("TINYRL_KV_BYTES", child_env)

  async def test_warm_sampler_starts_engine_before_sample_and_strips_env_var(self):
    import os

    sampler = Sampler(self.bus, "sampler:base", engine=FakeEngine)
    with (
      patch.dict(os.environ, {"TINYRL_WARM_SAMPLER": "1", "TINYRL_BASE": "base"}),
      patch("tinyrl.workers.sampler.WARM", 1),
    ):
      task = asyncio.create_task(sampler.serve())
      try:
        for _ in range(50):
          if sampler.engine is not None:
            break
          await asyncio.sleep(0.02)
        self.assertIsNotNone(sampler.engine)
        self.assertEqual(sampler.engine.calls[0], ("start", "base", True))
      finally:
        sampler.done = True
        await task

      with patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value.poll.return_value = 0
        with self.assertRaises(RuntimeError):
          await VLLM.start("base", False)
        self.assertNotIn("TINYRL_WARM_SAMPLER", mock_popen.call_args[1]["env"])

  async def test_an_engine_that_fails_to_start_early_is_logged_and_the_sampler_keeps_going(self):
    class Broken(FakeEngine):
      @classmethod
      async def start(cls, base: str, lora: bool):
        raise RuntimeError("no GPU memory")

    sampler = Sampler(self.bus, "sampler:base", engine=Broken)
    with patch.dict(os.environ, {"TINYRL_BASE": "base"}), patch("sys.stderr", new_callable=io.StringIO) as err:
      await sampler.warm_engine()
    self.assertIsNone(sampler.engine)
    self.assertIn("RuntimeError: no GPU memory", err.getvalue())

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

  @patch("tinyrl.workers.vllm.subprocess.Popen")
  @patch("tinyrl.workers.vllm.httpx.AsyncClient.get")
  async def test_vllm_start_kernel_config(self, mock_get, mock_popen):
    mock_popen.return_value.poll.return_value = None
    mock_get.return_value.is_success = True

    from tinyrl.workers.vllm import KERNEL_CONFIG, VLLM

    with patch("tinyrl.workers.vllm.VLLM_ARGS", ""):
      await (await VLLM.start("base", False)).http.aclose()
      args = mock_popen.call_args[0][0]
      self.assertIn("--kernel-config", args)
      idx = args.index("--kernel-config")
      self.assertEqual(args[idx + 1], KERNEL_CONFIG)

    with patch("tinyrl.workers.vllm.VLLM_ARGS", "--kernel-config '{\"enable_flashinfer_autotune\":true}'"):
      await (await VLLM.start("base", False)).http.aclose()
      args = mock_popen.call_args[0][0]
      self.assertIn("--kernel-config", args)
      idx = args.index("--kernel-config")
      self.assertEqual(args[idx + 1], '{"enable_flashinfer_autotune":true}')
      self.assertEqual(args.count("--kernel-config"), 1)
      self.assertNotIn(KERNEL_CONFIG, args)


if __name__ == "__main__":
  unittest.main()
