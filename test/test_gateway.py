import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import msgspec

from test.helpers import BusTest, serve_control
from tinyrl.api import gateway
from tinyrl.control import estimator
from tinyrl.control.node_daemon import ControlClient
from tinyrl.types import ControlMsg, NodeMsg, Op, Role, Run


class TestGatewayDedupe(BusTest):
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

    runs = {r for r, rs in self.bus.runs.items() if rs.live}
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

    runs = {r for r, rs in self.bus.runs.items() if rs.live}
    samplers = [r for r in runs if r.startswith("samp-")]
    self.assertEqual(len(samplers), 1)

  async def test_rejects_invalid_gpus(self):
    body = {"base_model": "tiny", "user_metadata": {"openrl.gpus": 0}}
    res = await self.http.post("/api/v1/create_model", json=body)
    self.assertEqual(res.status_code, 400)

  async def test_rejects_slash_in_session_and_deleted_parent_sampler(self):
    bad = await self.http.post("/api/v1/create_model", json={"base_model": "tiny", "session_id": "bad/sess"})
    self.assertEqual(bad.status_code, 400)

    self.bus.create_run(Run("r1", "tiny", 4))
    orig_to_thread = asyncio.to_thread

    async def delete_during_estimate(fn, *args, **kwargs):
      self.bus.delete_run("r1")
      return await orig_to_thread(fn, *args, **kwargs)

    with patch("tinyrl.api.gateway.asyncio.to_thread", side_effect=delete_during_estimate):
      res = await self.http.post(
        "/api/v1/save_weights_for_sampler", json={"model_id": "r1", "sampling_session_seq_id": 0}
      )
    self.assertEqual(res.status_code, 404)
    self.assertEqual([r for r, s in self.bus.runs.items() if s.run.role == Role.SAMPLER], [])

  async def test_reconnect_same_boot_id_supersedes_stale_ws_pusher(self):
    runner, url = await serve_control(self.bus, host="127.0.0.1", port=0)
    client = ControlClient(url)
    try:
      async with client.connect() as ws1:
        await ws1.send_bytes(msgspec.msgpack.encode(NodeMsg("hello", "n1", "b1", gpus=1)))
        msgspec.msgpack.decode(await ws1.receive_bytes(), type=ControlMsg)
        async with client.connect() as ws2:
          await ws2.send_bytes(msgspec.msgpack.encode(NodeMsg("hello", "n1", "b1", gpus=1)))
          msgspec.msgpack.decode(await ws2.receive_bytes(), type=ControlMsg)
          self.bus.create_run(Run("r1", "base", 4))
          self.bus.submit_command("r1", Op.CREATE_MODEL, b"{}")
          pushed = msgspec.msgpack.decode(await asyncio.wait_for(ws2.receive_bytes(), 2.0), type=ControlMsg)
          self.assertEqual([t.cmd.run for t in pushed.tasks], ["r1"])
    finally:
      await runner.cleanup()

  async def test_superseded_ws_cannot_reclaim_live_ws_during_send(self):
    from fastapi import WebSocket

    paused, release = asyncio.Event(), asyncio.Event()
    orig_send = WebSocket.send_bytes
    calls = 0

    async def slow_first_send(ws, data):
      nonlocal calls
      calls += 1
      if calls == 1:
        paused.set()
        await release.wait()
      return await orig_send(ws, data)

    runner, url = await serve_control(self.bus, host="127.0.0.1", port=0)
    client = ControlClient(url)
    try:
      with patch.object(WebSocket, "send_bytes", slow_first_send):
        async with client.connect() as ws1, client.connect() as ws2:
          await ws1.send_bytes(msgspec.msgpack.encode(NodeMsg("hello", "n1", "b1", gpus=1)))
          await asyncio.wait_for(paused.wait(), 2.0)
          await ws2.send_bytes(msgspec.msgpack.encode(NodeMsg("hello", "n1", "b1", gpus=1)))
          msgspec.msgpack.decode(await asyncio.wait_for(ws2.receive_bytes(), 2.0), type=ControlMsg)
          release.set()
          await asyncio.sleep(0.05)
          self.bus.create_run(Run("r1", "base", 4))
          self.bus.submit_command("r1", Op.CREATE_MODEL, b"{}")
          pushed = msgspec.msgpack.decode(await asyncio.wait_for(ws2.receive_bytes(), 2.0), type=ControlMsg)
          self.assertEqual([t.cmd.run for t in pushed.tasks], ["r1"])
    finally:
      await runner.cleanup()


class TestEstimator(unittest.TestCase):
  def test_estimates_full_and_vllm_without_importing_ml_frameworks(self):
    meta = SimpleNamespace(parameter_count={"BF16": 8_000_000_000})
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
      json.dump({"text_config": {"num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32}}, f)
      f.flush()
      with (
        patch("huggingface_hub.get_safetensors_metadata", return_value=meta),
        patch("huggingface_hub.hf_hub_download", return_value=f.name),
      ):
        full_bytes, full_gpus = estimator.estimate("org/model", "full")
        vllm_bytes, vllm_gpus = estimator.estimate("org/model", "vllm")
    self.assertEqual((full_bytes, full_gpus), (8_000_000_000 * 16 + estimator.OVERHEAD, 2))
    self.assertEqual(vllm_gpus, 1)
    self.assertGreater(vllm_bytes, 16_000_000_000)

    check = "import sys, tinyrl.api.gateway, tinyrl.control.node_daemon; assert not ({'torch', 'transformers', 'peft', 'safetensors', 'huggingface_hub', 'redis'} & sys.modules.keys())"
    res = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)
    self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
  unittest.main()
