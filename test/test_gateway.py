import asyncio
import json
import tempfile
import unittest
from unittest.mock import patch

import httpx

from test.helpers import RedisTest
from tinyrl.api import gateway
from tinyrl.bus import Op, Reply, key
from tinyrl.storage.store import Store


class TestAcceptance(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.http = httpx.AsyncClient(
      transport=httpx.ASGITransport(app=gateway.app(self.bus, Store(tempfile.mkdtemp()))), base_url="http://tinyrl"
    )

  async def asyncTearDown(self):
    await self.http.aclose()
    await super().asyncTearDown()

  async def create(self, seq=0):
    response = await self.http.post(
      "/api/v1/create_model", json={"base_model": "tiny", "session_id": "s", "model_seq_id": seq}
    )
    self.assertEqual(response.status_code, 200)
    return response.json()

  async def test_concurrent_create_accepts_one_run_and_one_command(self):
    results = await asyncio.gather(*(self.create(1) for _ in range(20)))
    self.assertEqual(len({r["request_id"] for r in results}), 1)
    self.assertEqual(len({r["model_id"] for r in results}), 1)
    self.assertEqual(await self.bus.r.scard(key("runs")), 1)
    self.assertEqual(await self.bus.r.xlen(key("q", results[0]["model_id"])), 1)
    self.assertEqual([e.kind for e in await self.bus.read_events()], ["run_created"])

  async def test_each_models_first_operations_have_their_own_sequence_numbers(self):
    for model_seq in range(3):
      created = await self.create(model_seq)
      futures = [created["request_id"]]
      for seq in (1, 2):
        response = await self.http.post("/api/v1/optim_step", json={"model_id": created["model_id"], "seq_id": seq})
        self.assertEqual(response.status_code, 200)
        futures.append(response.json()["request_id"])
      self.assertEqual(len(set(futures)), 3)
      self.assertEqual(await self.bus.r.xlen(key("q", created["model_id"])), 3)

  async def test_concurrent_sampler_save_accepts_one_resource_and_one_command(self):
    created = await self.create(1)
    body = {"model_id": created["model_id"], "seq_id": 7, "sampling_session_seq_id": 0}
    responses = await asyncio.gather(
      *(self.http.post("/api/v1/save_weights_for_sampler", json=body) for _ in range(20))
    )
    self.assertTrue(all(r.status_code == 200 for r in responses))
    self.assertEqual(len({r.json()["request_id"] for r in responses}), 1)
    runs = await self.bus.r.smembers(key("runs"))
    self.assertEqual(len(runs), 2)
    stream = await self.bus.r.xrange(key("q", created["model_id"]))
    self.assertEqual([f[b"op"] for _, f in stream], [b"create_model", b"save_weights_for_sampler"])
    self.assertIn(json.loads(stream[-1][1][b"body"])["sampling_session_id"].encode(), runs)

  async def test_sampler_creation_is_idempotent_for_the_sdk_identity(self):
    body = {"base_model": "tiny", "session_id": "s", "sampling_session_seq_id": 3}
    responses = await asyncio.gather(*(self.http.post("/api/v1/create_sampling_session", json=body) for _ in range(8)))
    self.assertEqual(len({r.json()["sampling_session_id"] for r in responses}), 1)
    self.assertEqual(await self.bus.r.scard(key("runs")), 1)

  async def test_rejected_sampler_save_creates_no_orphan(self):
    created = await self.create()
    await self.bus.delete_run(created["model_id"], "")
    response = await self.http.post(
      "/api/v1/save_weights_for_sampler", json={"model_id": created["model_id"], "seq_id": 9}
    )
    self.assertEqual(response.status_code, 404)
    self.assertEqual(await self.bus.r.scard(key("runs")), 0)

  async def test_expired_reply_is_terminal_and_does_not_reexecute(self):
    created = await self.create()
    run = created["model_id"]
    await self.bus.assign_run_to_worker(run, "w")
    await self.bus.take_lease("w", "owner")
    [cmd] = await self.bus.take_commands("w", "owner", block_ms=0)
    await self.bus.write_reply(cmd, Reply(True, "application/json", b"{}"), "w", "owner")
    await self.bus.r.delete(key("f", cmd.request_id))
    self.assertEqual((await self.create())["request_id"], cmd.request_id)
    self.assertEqual(await self.bus.r.xlen(key("q", run)), 0)
    self.assertEqual(await self.bus.r.ttl(key("seq", run)), -1)
    with patch.object(gateway, "WAIT", 0):
      expired = await self.http.post("/api/v1/retrieve_future", json={"request_id": cmd.request_id})
      pending_id = await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}", 1)
      pending = await self.http.post("/api/v1/retrieve_future", json={"request_id": pending_id})
    self.assertEqual(expired.status_code, 404)
    self.assertEqual(pending.json()["type"], "try_again")


if __name__ == "__main__":
  unittest.main()
