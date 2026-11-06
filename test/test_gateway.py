import unittest

import httpx

from test.helpers import RedisTest
from tinyrl.api import gateway


class TestGatewayDedupe(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app(self.bus)), base_url="http://tinyrl")

  async def asyncTearDown(self):
    await self.http.aclose()
    await super().asyncTearDown()

  async def test_b1_create_model_dedupes(self):
    body = {"base_model": "tiny", "session_id": "sess-1", "model_seq_id": 1, "lora_config": {"rank": 4}}
    r1 = await self.http.post("/api/v1/create_model", json=body)
    r2 = await self.http.post("/api/v1/create_model", json=body)

    self.assertEqual(r1.status_code, 200)
    self.assertEqual(r2.status_code, 200)
    self.assertEqual(r1.json()["model_id"], r2.json()["model_id"])

    runs = await self.bus.r.smembers("tinyrl:runs")
    self.assertEqual(len(runs), 1)

  async def test_b2_save_weights_for_sampler_dedupes(self):
    body = {"base_model": "tiny", "session_id": "sess-1", "model_seq_id": 1}
    r1 = await self.http.post("/api/v1/create_model", json=body)
    run_id = r1.json()["model_id"]

    save_req = {"model_id": run_id, "sampling_session_seq_id": 0, "seq_id": 10}
    s1 = await self.http.post("/api/v1/save_weights_for_sampler", json=save_req)
    s2 = await self.http.post("/api/v1/save_weights_for_sampler", json=save_req)

    self.assertEqual(s1.status_code, 200)
    self.assertEqual(s2.status_code, 200)
    self.assertEqual(s1.json()["request_id"], s2.json()["request_id"])

    runs = await self.bus.r.smembers("tinyrl:runs")
    samplers = [r for r in runs if r.decode().startswith("samp-")]
    self.assertEqual(len(samplers), 1)


if __name__ == "__main__":
  unittest.main()
