import json
from unittest.mock import patch

import httpx

from test.helpers import CoordinatorTest, new_id, tiny
from test.test_api import proto_body
from tinyrl import gateway
from tinyrl.bus import Op, Run
from tinyrl.client import WorkerClient
from tinyrl.store import Store
from tinyrl.trainer import Model, Trainer


class LostResponses(httpx.AsyncBaseTransport):
  """Drop the first response after each coordinator transition has committed."""

  def __init__(self, app):
    self.inner = httpx.ASGITransport(app=app)
    self.drop = {"/internal/register", "/internal/take", "/internal/finish"}

  async def handle_async_request(self, request):
    response = await self.inner.handle_async_request(request)
    completion = request.url.path == "/internal/finish"
    optimizer_reply = completion and json.loads(request.content)["command"]["op"] == Op.OPTIM_STEP
    if request.url.path in self.drop and (not completion or optimizer_reply):
      self.drop.remove(request.url.path)
      await response.aclose()
      raise httpx.ReadError("response connection lost after commit", request=request)
    return response

  async def aclose(self):
    await self.inner.aclose()


class TestWorkerClient(CoordinatorTest):
  async def test_unknown_optimizer_is_rejected_instead_of_silently_using_adam(self):
    transport = httpx.ASGITransport(app=gateway.app(self.bus))
    async with httpx.AsyncClient(transport=transport, base_url="http://coordinator") as connection:
      response = await connection.post(
        "/api/v1/create_model", json={"base_model": "tiny", "optimizer_config": {"type": "dimuon"}}
      )
      self.assertEqual(response.status_code, 400)
      for field in ("adam_params", "optim_params", "optimizer_params"):
        response = await connection.post("/api/v1/optim_step", json={"model_id": "unused", field: {"type": "dimuon"}})
        self.assertEqual(response.status_code, 400)

  async def test_lost_http_responses_reconcile_without_reexecuting_training(self):
    run = new_id()
    await self.bus.create(
      Run(run, "p", "tiny", 4),
      body={"base_model": "tiny", "rank": 4, "lora": {"seed": 0, "lora_dropout": 0}},
    )
    await self.bus.submit(run, Op.FORWARD_BACKWARD, proto_body(run), seq_id=0)
    await self.bus.submit(run, Op.OPTIM_STEP, b'{"adam_params":{}}', seq_id=1)
    transport = LostResponses(gateway.app(self.bus, worker_token="test-token"))
    async with httpx.AsyncClient(
      transport=transport, base_url="http://coordinator", headers={"Authorization": "Bearer test-token"}
    ) as connection:
      trainer = Trainer(WorkerClient(http=connection), "p", Model(tiny), store=Store(root=f"{self.tmpdir.name}/ckpt"))
      with patch.object(trainer.model, "optim_step", wraps=trainer.model.optim_step) as update:
        for _ in range(3):
          await trainer.step(block_ms=0)
        self.assertEqual(update.call_count, 1)
      self.assertFalse(trainer.pending)
      self.assertFalse(transport.drop)
      self.assertEqual((await self.bus.run(run))["steps"], 1)
      self.assertEqual((await self.bus.queue(run))["in_flight"], [])

  async def test_recovery_accepts_the_immutable_migration_snapshot(self):
    await self.bus.register("old", "p")
    await self.bus.create(Run("run", "p", "tiny", 4))
    await self.bus.submit("run", Op.OPTIM_STEP, b"{}")
    await self.bus.take("p", "old", block_ms=0)
    await self.bus.worker_lost("old")
    transport = httpx.ASGITransport(app=gateway.app(self.bus, worker_token="test-token"))
    async with httpx.AsyncClient(
      transport=transport, base_url="http://coordinator", headers={"Authorization": "Bearer test-token"}
    ) as connection:
      rejected = await connection.post("/internal/recover", json={"run_id": "run"})
      self.assertEqual(rejected.status_code, 400)
      accepted = await connection.post("/internal/recover", json={"run_id": "run", "ref": "a" * 32})
      self.assertEqual(accepted.status_code, 200)
    await self.bus.register("new", "p")
    [restore] = await self.bus.take("p", "new", block_ms=0)
    self.assertEqual(restore.op, Op.RESTORE)
    self.assertEqual(json.loads(restore.body)["ref"], "a" * 32)
