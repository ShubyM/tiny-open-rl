import asyncio
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import httpx

from test.helpers import FakeEngine, RedisTest, tiny
from tinyrl import wire
from tinyrl.bus import Op, Run
from tinyrl.gateway import app
from tinyrl.sampler import Out, Sampler, Seq, encode
from tinyrl.store import Store
from tinyrl.trainer import Trainer
from tinyrl.vllm import VLLM

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
    await full.load("c0", "/ckpt/w")
    await full.load("c0", "/ckpt/w")  # already in place
    engine = VLLM("http://vllm", "base", lora=True, client=client)
    await engine.load("c0", "/ckpt/a")
    await engine.load("c0", "/ckpt/a")
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
      return json.loads((await self.bus.result(request_id, 1)).body)["sampling_session_id"]

    run = (await call("create_model", {"base_model": base, "lora_config": {"rank": 4}}))["model_id"]
    trainer = Trainer(self.bus, f"lora:{base}:0", store=store)
    await self.own(trainer)
    await trainer.step(block_ms=0)
    sampler = Sampler(self.bus, f"sampler:{base}", engine=FakeEngine, store=store)
    await self.own(sampler)

    sid = await save_for_sampler(trainer, run)
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
    fft_trainer = Trainer(self.bus, fft, store=store)
    await self.own(fft_trainer)
    await fft_trainer.step(block_ms=0)
    fft_sid = await save_for_sampler(fft_trainer, fft)
    self.assertEqual((await self.bus.run(fft_sid))["pool"], f"sampler:fft:{fft}")
    fft_sampler = Sampler(self.bus, f"sampler:fft:{fft}", engine=FakeEngine, store=store)
    await self.own(fft_sampler)
    request_id = (await call("asample", {"sampling_session_id": fft_sid, "prompt": tokens([1, 2])}))["request_id"]
    await fft_sampler.step(block_ms=0)
    res = await client.post("/api/v1/retrieve_future", json={"request_id": request_id}, headers={"Accept": wire.PROTO})
    self.assertEqual(res.headers["content-type"], wire.PROTO)
    self.assertIn(("load", "c0", store.path(f"tinker://{fft}/sampler_weights/step-0")), fft_sampler.engine.calls)
    await client.aclose()

  @patch("tinyrl.trainer.HOLD_S", 0.0)
  @patch("tinyrl.sampler.HOLD_S", 0.0)
  @patch("tinyrl.trainer.GPU", "node-a:0")
  @patch("tinyrl.sampler.GPUS", ["node-a:0"])
  async def test_a_trainer_and_a_sampler_take_turns_on_one_gpu(self):
    await self.take_turns()

  @patch("tinyrl.trainer.HOLD_S", 0.0)
  @patch("tinyrl.sampler.HOLD_S", 0.0)
  @patch("tinyrl.trainer.GPU", "node-a:1")
  @patch("tinyrl.sampler.GPUS", ["node-a:0", "node-a:1"])
  async def test_a_tensor_parallel_sampler_takes_turns_on_both_its_gpus(self):
    await self.take_turns()
    self.assertIsNone(await self.bus.owner("gpu:node-a:1"))

  async def take_turns(self):
    await self.bus.create(Run("t1", "t1", "base", 0))
    await self.bus.create(Run("samp-1", "sampler:base", "base", 1))
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
        reply = await self.bus.result(await self.bus.submit(run, op, json.dumps(body).encode()), 5)
        self.assertTrue(reply.ok, reply.body)
      self.assertIn(("sleep",), sampler.engine.calls)
    finally:
      trainer.done = sampler.done = True
      await asyncio.gather(*tasks)
    self.assertIsNone(await self.bus.owner("gpu:node-a:0"))
    self.assertEqual(sampler.engine.calls[-1], ("close",))

  async def test_a_sampler_batches_queued_commands_concurrently_and_loads_once(self):
    store = Store(tempfile.mkdtemp())
    store.put("tinker://r1/sampler_weights/s0", {}, {"base": "base", "lora": True, "config": {}})
    await self.bus.create(Run("samp-1", "sampler:base", "base", 1, model_path="tinker://r1/sampler_weights/s0"))
    active = peak = starts = loads = 0

    class BatchEngine(FakeEngine):
      @classmethod
      async def start(cls, base: str, lora: bool):
        nonlocal starts
        await asyncio.sleep(0.01)
        starts += 1
        return await super().start(base, lora)

      async def load(self, name: str, path: str) -> None:
        nonlocal loads
        await asyncio.sleep(0.01)
        if ("load", name, path) not in self.calls:
          loads += 1
        await super().load(name, path)

      async def sample(self, job: dict, name: str | None) -> Out:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return await super().sample(job, name)

    sampler = Sampler(self.bus, "sampler:base", engine=BatchEngine, store=store)
    await self.own(sampler)
    ids = [await self.bus.submit("samp-1", Op.SAMPLE, json.dumps(JOB).encode()) for _ in range(8)]
    self.assertEqual(await sampler.step(block_ms=0), 8)
    self.assertEqual((starts, loads, peak), (1, 1, 8))
    self.assertTrue(all([(await self.bus.result(i, 1)).ok for i in ids]))

  async def test_vllm_retries_a_dropped_keepalive_socket(self):
    import httpx

    from tinyrl.vllm import VLLM

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


if __name__ == "__main__":
  unittest.main()
