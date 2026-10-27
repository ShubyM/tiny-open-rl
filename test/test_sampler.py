import asyncio
import json
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from test.helpers import RedisTest, tiny
from tinyrl import wire
from tinyrl.bus import Op, Run
from tinyrl.gateway import app
from tinyrl.sampler import HFEngine, Out, Sampler, Seq, VLLMServer, encode
from tinyrl.store import Store
from tinyrl.trainer import Trainer

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
  async def test_hf_engine_samples_and_scores(self):
    base = f"{tempfile.mkdtemp()}/tiny"
    tiny().save_pretrained(base)
    engine = HFEngine(base)

    async def sample(**job) -> dict:
      job = JOB | job
      return wire.sample_json(encode(job, await engine.sample(job, None)))

    out = await sample(num_samples=2, prompt_logprobs=True, prompt_logprobs_last_n=1, topk_prompt=2, topk_sample=2)
    a, b = out["sequences"]
    self.assertEqual(a["tokens"], b["tokens"])
    self.assertEqual(out["prompt_logprobs"][:2], [None, None])
    self.assertIsInstance(out["prompt_logprobs"][2], float)
    self.assertEqual(out["topk_prompt_logprobs"][:2], [None, None])
    self.assertEqual(len(out["topk_prompt_logprobs"][2]), 2)
    self.assertEqual(len(a["topk_logprobs"]), len(a["tokens"]))

    seeded = {"temperature": 1.0, "seed": 42, "max_tokens": 5}
    self.assertEqual((await sample(**seeded))["sequences"], (await sample(**seeded))["sequences"])

    stopped = (await sample(max_tokens=10, stop=[a["tokens"][0]]))["sequences"][0]
    self.assertEqual((stopped["tokens"], stopped["stop_reason"]), ([a["tokens"][0]], "stop"))

  async def test_vllm_server_speaks_the_completions_api(self):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
      calls.append((request.url.path, json.loads(request.content) if request.content else None))
      if request.url.path != "/v1/completions":
        return httpx.Response(200)
      choice = {
        "index": 0,
        "finish_reason": "length",
        "token_ids": [9],
        "logprobs": {"token_logprobs": [-3.0], "top_logprobs": [{"token_id:9": -3.0, "token_id:7": -0.5}]},
        "prompt_logprobs": [None, {"2": {"logprob": -1.0}, "5": {"logprob": -0.1}}, {"3": {"logprob": -2.0}}],
      }
      return httpx.Response(200, json={"choices": [choice]})

    server = VLLMServer(
      "http://vllm", "base", httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://vllm")
    )
    await server.load_lora("a0", "/ckpt/a")
    await server.load_weights("/ckpt/w")
    await server.sleep()
    await server.wake()
    self.assertEqual(
      [path for path, _ in calls],
      ["/v1/load_lora_adapter", "/collective_rpc", "/reset_prefix_cache", "/sleep", "/wake_up"],
    )

    job = JOB | {
      "stop": ["\n", 4],
      "prompt_logprobs": True,
      "prompt_logprobs_last_n": 1,
      "topk_prompt": 2,
      "topk_sample": 1,
    }
    out = wire.sample_json(encode(job, await server.sample(job, "a0")))
    req = calls[-1][1]
    self.assertEqual((req["model"], req["stop"], req["stop_token_ids"]), ("a0", ["\n"], [4]))
    self.assertEqual((req["logprobs"], req["prompt_logprobs"]), (1, 2))
    self.assertEqual(out["sequences"][0]["tokens"], [9])
    self.assertEqual(out["sequences"][0]["stop_reason"], "length")
    self.assertEqual(out["sequences"][0]["topk_logprobs"], [[[7, -0.5]]])
    self.assertEqual(out["prompt_logprobs"], [None, None, -2.0])
    self.assertEqual(out["topk_prompt_logprobs"], [None, None, [[3, -2.0]]])

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
    sampler = Sampler(self.bus, f"sampler:{base}", store=store)
    await self.own(sampler)

    sid = await save_for_sampler(trainer, run)
    self.assertTrue((await client.get(f"/api/v1/samplers/{sid}")).json()["model_path"].startswith("tinker://"))
    first = await call("join_sampling_session", {"sampling_session_id": sid})
    self.assertEqual(
      (await call("join_sampling_session", {"sampling_session_id": sid}))["client_counter"], first["client_counter"] + 1
    )

    job = {"sampling_session_id": sid, "seq_id": 7, "prompt": tokens([1, 2, 3]), "prompt_logprobs": True}
    replies = []
    for _ in range(2):  # the SDK retrying with the same seq_id gets the first answer
      request_id = (await call("asample", job))["request_id"]
      await sampler.step(block_ms=0)
      replies.append(await call("retrieve_future", {"request_id": request_id}))
    self.assertEqual(replies[0], replies[1])
    self.assertEqual(replies[0]["prompt_logprobs"][0], None)

    request_id = (await call("asample", {"base_model": base, "prompt": tokens([1, 2])}))["request_id"]
    await sampler.step(block_ms=0)
    self.assertEqual(len((await call("retrieve_future", {"request_id": request_id}))["sequences"]), 1)

    fft = await call("create_model", {"base_model": base, "user_metadata": {"openrl.fine_tuning_type": "full"}})
    fft_trainer = Trainer(self.bus, fft["model_id"], store=store)
    await self.own(fft_trainer)
    await fft_trainer.step(block_ms=0)
    fft_sid = await save_for_sampler(fft_trainer, fft["model_id"])
    self.assertEqual((await self.bus.run(fft_sid))["pool"], f"sampler:fft:{fft['model_id']}")
    fft_sampler = Sampler(self.bus, f"sampler:fft:{fft['model_id']}", store=store)
    await self.own(fft_sampler)
    request_id = (await call("asample", {"sampling_session_id": fft_sid, "prompt": tokens([1, 2])}))["request_id"]
    await fft_sampler.step(block_ms=0)
    res = await client.post("/api/v1/retrieve_future", json={"request_id": request_id}, headers={"Accept": wire.PROTO})
    self.assertEqual(res.headers["content-type"], wire.PROTO)
    await client.aclose()

  @patch("tinyrl.trainer.HOLD_S", 0.0)
  @patch("tinyrl.sampler.HOLD_S", 0.0)
  @patch("tinyrl.trainer.GPU", "node-a:0")
  @patch("tinyrl.sampler.GPU", "node-a:0")
  async def test_a_trainer_and_a_sampler_take_turns_on_one_gpu(self):
    await self.bus.create(Run("t1", "t1", "base", 0))
    await self.bus.create(Run("samp-1", "sampler:base", "base", 1))
    model = MagicMock()
    model.optim_step.return_value = {"grad_norm:mean": 0.5}
    trainer = Trainer(self.bus, "t1", model=model)
    engine = AsyncMock()
    engine.sample.return_value = Out([Seq([5], [-0.2], [{5: -0.2}], True)], [])
    sampler = Sampler(self.bus, "sampler:base", engine=engine)

    tasks = [asyncio.create_task(trainer.serve()), asyncio.create_task(sampler.serve())]
    try:
      for run, op, body in [
        ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
        ("samp-1", Op.SAMPLE, JOB),
        ("t1", Op.OPTIM_STEP, {"adam_params": {}}),
      ]:
        reply = await self.bus.result(await self.bus.submit(run, op, json.dumps(body).encode()), 5)
        self.assertTrue(reply.ok, reply.body)
      self.assertGreaterEqual(engine.sleep.call_count, 1)
    finally:
      trainer.done = sampler.done = True
      await asyncio.gather(*tasks)
    self.assertIsNone(await self.bus.owner("gpu:node-a:0"))


if __name__ == "__main__":
  unittest.main()
