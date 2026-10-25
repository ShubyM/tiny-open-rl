import json
from unittest.mock import patch

import httpx

from test.helpers import CoordinatorTest, new_id, tiny
from test.test_api import proto_body
from tinyrl import gateway
from tinyrl.bus import Op, Reply, Run
from tinyrl.client import WorkerClient
from tinyrl.payloads import INLINE_BYTES, PayloadStore
from tinyrl.store import Store
from tinyrl.trainer import Model, Trainer


class LostResponses(httpx.AsyncBaseTransport):
  """Drop the first response after each coordinator transition has committed."""

  def __init__(self, app, completion_op=Op.OPTIM_STEP):
    self.inner = httpx.ASGITransport(app=app)
    self.drop = {"/internal/register", "/internal/take", "/internal/finish"}
    self.completion_op, self.completions = completion_op, []

  async def handle_async_request(self, request):
    response = await self.inner.handle_async_request(request)
    completion = request.url.path == "/internal/finish"
    matching_reply = completion and json.loads(request.content)["command"]["op"] == self.completion_op
    if completion:
      self.completions.append(json.loads(request.content))
    if request.url.path in self.drop and (not completion or matching_reply):
      self.drop.remove(request.url.path)
      await response.aclose()
      raise httpx.ReadError("response connection lost after commit", request=request)
    return response

  async def aclose(self):
    await self.inner.aclose()


class TestWorkerClient(CoordinatorTest):
  async def test_corrupt_input_fails_before_numerical_execution(self):
    payloads = PayloadStore(f"{self.tmpdir.name}/payloads")
    await self.bus.create(Run("run", "p", "tiny", 4))
    body = b"input" * INLINE_BYTES
    ref = payloads.put(body)
    future = await self.bus.submit("run", Op.FORWARD_BACKWARD, b"", seq_id=0, body_ref=ref, body_size=len(body))
    payloads.path(ref).write_bytes(b"X" * len(body))
    transport = httpx.ASGITransport(app=gateway.app(self.bus, worker_token="test-token", payloads=payloads))
    async with httpx.AsyncClient(
      transport=transport, base_url="http://coordinator", headers={"Authorization": "Bearer test-token"}
    ) as connection:
      trainer = Trainer(WorkerClient(http=connection, payloads=payloads), "p", Model(tiny))
      with patch.object(trainer, "run", side_effect=AssertionError("corrupt input reached numerical execution")):
        self.assertEqual(await trainer.step(block_ms=0), 0)
    result = await self.bus.result(future, 0)
    self.assertFalse(result.ok)
    self.assertIn("input payload unavailable", result.body.decode())
    self.assertEqual((await self.bus.queue("run"))["in_flight"], [])

  async def test_large_completion_retry_reuses_refs_and_missing_output_never_requeues(self):
    payloads = PayloadStore(f"{self.tmpdir.name}/payloads")
    await self.bus.create(Run("run", "p", "tiny", 4))
    body = b"input" * INLINE_BYTES
    ref = payloads.put(body)
    future = await self.bus.submit("run", Op.FORWARD_BACKWARD, b"", seq_id=0, body_ref=ref, body_size=len(body))
    transport = LostResponses(
      gateway.app(self.bus, worker_token="test-token", payloads=payloads), completion_op=Op.FORWARD_BACKWARD
    )
    async with httpx.AsyncClient(
      transport=transport, base_url="http://coordinator", headers={"Authorization": "Bearer test-token"}
    ) as connection:
      worker = WorkerClient(http=connection, payloads=payloads)
      await worker.register("worker", "p")
      [cmd] = await worker.take("p", "worker", block_ms=0)
      self.assertEqual(cmd.body, body)
      output = json.dumps({"payload": "output" * INLINE_BYTES}).encode()
      original_put = payloads.put
      attempts = 0

      def put_after_outage(data):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
          raise OSError("temporary storage outage")
        return original_put(data)

      with patch.object(payloads, "put", side_effect=put_after_outage):
        self.assertEqual(await worker.finish(cmd, Reply(True, "application/json", output)), "already_committed")
      self.assertEqual(attempts, 2)
      self.assertEqual(len(transport.completions), 2)
      self.assertEqual(transport.completions[0], transport.completions[1])
      for completion in transport.completions:
        self.assertEqual(completion["command"]["body"], "")
        self.assertEqual(completion["reply"]["body"], "")
      stored = await self.bus.result(future, 0)
      payloads.path(stored.body_ref).unlink()
      unavailable = await connection.post("/api/v1/retrieve_future", json={"request_id": future})
      self.assertEqual(unavailable.status_code, 503)
      self.assertEqual(await worker.take("p", "worker", block_ms=0), [])
      payloads.put(output)
      restored = await connection.post("/api/v1/retrieve_future", json={"request_id": future})
      self.assertEqual(restored.content, output)
      self.assertEqual((await self.bus.run("run"))["cursor"], 1)

  async def test_oversized_result_reports_recovery_without_publishing_blob(self):
    payloads = PayloadStore(f"{self.tmpdir.name}/payloads")
    await self.bus.create(Run("run", "p", "tiny", 4))
    future = await self.bus.submit("run", Op.FORWARD_BACKWARD, b"input", seq_id=0)
    transport = httpx.ASGITransport(app=gateway.app(self.bus, worker_token="test-token", payloads=payloads))
    async with httpx.AsyncClient(
      transport=transport, base_url="http://coordinator", headers={"Authorization": "Bearer test-token"}
    ) as connection:
      worker = WorkerClient(http=connection, payloads=payloads)
      await worker.register("worker", "p")
      [cmd] = await worker.take("p", "worker", block_ms=0)
      with patch("tinyrl.client.MAX_REPLY_BYTES", 256):
        self.assertEqual(await worker.finish(cmd, Reply(True, "application/json", b"X" * 257)), "committed")
    result = await self.bus.result(future, 0)
    self.assertFalse(result.ok)
    self.assertTrue(result.recover)
    self.assertEqual(list(payloads.root.iterdir()), [])
    self.assertEqual((await self.bus.run("run"))["phase"], "recovery")

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
