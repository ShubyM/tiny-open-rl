import json
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from test.helpers import RedisTest
from tinyrl.api import gateway
from tinyrl.control import estimator


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


class TestEstimator(unittest.TestCase):
  def test_estimates_full_and_vllm_without_importing_ml_frameworks(self):
    meta = SimpleNamespace(parameter_count={"BF16": 8_000_000_000})
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
      json.dump({"text_config": {"num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32}}, f)
      f.flush()
      with (
        patch.object(estimator, "get_safetensors_metadata", return_value=meta),
        patch.object(estimator, "hf_hub_download", return_value=f.name),
      ):
        full_bytes, full_gpus = estimator.estimate("org/model", "full")
        vllm_bytes, vllm_gpus = estimator.estimate("org/model", "vllm")
    self.assertEqual((full_bytes, full_gpus), (8_000_000_000 * 16 + estimator.OVERHEAD, 2))
    self.assertEqual(vllm_gpus, 1)
    self.assertGreater(vllm_bytes, 16_000_000_000)

    check = "import sys, tinyrl.api.gateway, tinyrl.control.node_daemon; assert not ({'torch', 'transformers', 'peft'} & sys.modules.keys())"
    res = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)
    self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
  unittest.main()
