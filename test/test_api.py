import json
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from test.helpers import CoordinatorTest, tiny
from test.test_wire import json_body
from tinyrl import gateway, wire
from tinyrl.bus import Reply
from tinyrl.client import WorkerClient
from tinyrl.proto import tinker_public_pb2 as pb
from tinyrl.store import Store
from tinyrl.trainer import Model, Trainer

PROTO = {"content-type": wire.PROTO}


def proto_body(model_id: str, n: int = 12, loss_fn: str = "cross_entropy", seq_id: int = 0) -> bytes:
  """A forward_backward the way the SDK sends it: two sequences that count up."""
  req = pb.ForwardBackwardRequest(model_id=model_id, loss_fn=loss_fn, seq_id=seq_id)
  for start in (0, 40):
    tokens = np.arange(start, start + n + 1)
    datum = req.data.add()
    datum.model_input.add().encoded_text.tokens = tokens[:-1].astype(np.int32).tobytes()
    targets = pb.Tensor(dense=tokens[1:].astype(np.int64).tobytes(), dtype=pb.DTYPE_INT64, shape=[n])
    weights = pb.Tensor(dense=np.ones(n, np.float32).tobytes(), dtype=pb.DTYPE_FLOAT32, shape=[n])
    datum.loss_fn_inputs["target_tokens"].CopyFrom(targets)
    datum.loss_fn_inputs["weights"].CopyFrom(weights)
  return req.SerializeToString()


class TestAPI(CoordinatorTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.store = Store(root=f"{self.tmpdir.name}/ckpt")
    self.api = gateway.app(self.bus, worker_token="test-token")
    self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.api), base_url="http://tinyrl")
    self.trainers: dict[str, Trainer] = {}

  async def asyncTearDown(self):
    await self.http.aclose()
    await super().asyncTearDown()

  async def post(self, path: str, **kwargs) -> httpx.Response:
    return await self.http.post(f"/api/v1/{path}", **kwargs)

  async def drain(self, run_id: str):
    """Let the trainer for the run's pool do all of the pool's queued work."""
    pool = (await self.bus.run(run_id))["pool"]
    if pool not in self.trainers:
      self.trainers[pool] = Trainer(self.bus, pool, Model(tiny), store=self.store)
    while await self.trainers[pool].step(block_ms=10):
      pass

  async def result(self, submitted: httpx.Response, **kwargs) -> httpx.Response:
    """Run a submitted call and retrieve its result."""
    await self.drain(submitted.json()["model_id"])
    return await self.post("retrieve_future", json={"request_id": submitted.json()["request_id"]}, **kwargs)

  async def create_model(self, **kwargs) -> str:
    body = {"base_model": "tiny", "lora_config": {"rank": 4, "seed": 0, "lora_dropout": 0}, **kwargs}
    created = await self.result(await self.post("create_model", json=body))
    return created.json()["model_id"]

  async def test_a_tinker_training_loop(self):
    session = await self.post("create_session", json={"tags": [], "user_metadata": None, "sdk_version": "0.29.0"})
    session_id = session.json()["session_id"]
    heartbeat = await self.post("session_heartbeat", json={"session_id": session_id})
    self.assertEqual(heartbeat.json(), {"type": "session_heartbeat"})

    run_id = await self.create_model(session_id=session_id, model_seq_id=0)
    losses = []
    for _ in range(8):
      fwdbwd = await self.post("forward_backward", content=proto_body(run_id), headers=PROTO)
      optim = await self.post("optim_step", json={"model_id": run_id, "adam_params": {"learning_rate": 1e-2}})

      out = await self.result(fwdbwd, headers={"accept": wire.PROTO})
      self.assertEqual(out.headers["content-type"], wire.PROTO)
      output = pb.ForwardBackwardOutput.FromString(out.content)
      self.assertEqual(output.loss_fn_outputs[0].num_datums, 2)
      losses.append(output.metrics["loss:sum"])
      self.assertEqual((await self.result(optim)).json()["type"], "optim_step")
    self.assertLess(losses[-1], losses[0] * 0.7)

    run = (await self.http.get(f"/api/v1/runs/{run_id}")).json()
    self.assertEqual((run["steps"], run["failed"], run["session"]), (8, 0, session_id))
    self.assertEqual((run["live"], run["queued"], run["in_flight"]), (True, 0, []))

    session = (await self.http.get(f"/api/v1/sessions/{session_id}")).json()
    self.assertEqual((session["runs"], session["live"]), ([run_id], True))

    info = (await self.post("get_info", json={"model_id": run_id})).json()
    self.assertEqual(info["model_data"], {"model_name": "tiny", "tokenizer_id": "tiny"})

  async def test_json_clients_get_json(self):
    run_id = await self.create_model()
    fwdbwd = await self.result(await self.post("forward_backward", json={**json_body(), "model_id": run_id}))
    forward_body = {"model_id": run_id, "forward_input": json_body()["forward_backward_input"]}
    forward = await self.result(await self.post("forward", json=forward_body))

    fwdbwd, forward = fwdbwd.json(), forward.json()
    self.assertEqual(fwdbwd["type"], "forward_backward")
    self.assertEqual(len(fwdbwd["loss_fn_outputs"]), 2)
    self.assertEqual(fwdbwd["loss_fn_outputs"][0]["logprobs"]["shape"], [3])
    # dropout is off, so the forward pass gives the same logprobs as the forward_backward pass
    self.assertEqual(
      [row["logprobs"]["data"] for row in forward["loss_fn_outputs"]],
      [row["logprobs"]["data"] for row in fwdbwd["loss_fn_outputs"]],
    )

  async def test_checkpoints(self):
    run_id = await self.create_model()
    saved = (await self.result(await self.post("save_weights", json={"model_id": run_id, "path": "final"}))).json()
    self.assertEqual(saved, {"type": "save_weights", "model_id": run_id, "path": f"tinker://{run_id}/weights/final"})

    load = {"model_id": run_id, "path": saved["path"], "optimizer": True}
    loaded = (await self.result(await self.post("load_weights", json=load))).json()
    self.assertEqual((loaded["type"], loaded["path"]), ("load_weights", saved["path"]))

    save = {"model_id": run_id, "sampling_session_seq_id": 0}
    sampler = (await self.result(await self.post("save_weights_for_sampler", json=save))).json()
    self.assertEqual(sampler["path"], f"tinker://{run_id}/sampler_weights/step-0")
    resolved = await self.post("resolve_checkpoint", json={"path": sampler["path"]})
    self.assertEqual(resolved.status_code, 200)
    with open(f"{self.store.export(resolved.json()['ref'])}/adapter_config.json") as f:
      self.assertEqual(json.load(f)["r"], 4)

    load = {"model_id": run_id, "path": "nope", "optimizer": False}
    missing = (await self.result(await self.post("load_weights", json=load))).json()
    self.assertEqual(missing["category"], "user")

    for bad in ("../../etc", "/etc", f"tinker://{run_id}/weights/../../x"):
      response = await self.post("save_weights", json={"model_id": run_id, "path": bad})
      self.assertEqual(response.status_code, 400)

  async def test_errors(self):
    run_id = await self.create_model()
    for path, kwargs in [
      ("optim_step", {"json": {"model_id": "nope"}}),
      ("forward_backward", {"content": proto_body("nope"), "headers": PROTO}),
      ("get_info", {"json": {"model_id": "nope"}}),
    ]:
      self.assertEqual((await self.post(path, **kwargs)).status_code, 404)

    self.assertEqual((await self.post("forward_backward", content=b"\xff\xff", headers=PROTO)).status_code, 400)
    self.assertEqual((await self.post("forward_backward", json={"forward_backward_input": {}})).status_code, 400)

    # a call that fails in the trainer comes back as a 200 with the error, which is what the SDK reads
    failed = await self.result(
      await self.post("forward_backward", content=proto_body(run_id, loss_fn="dpo"), headers=PROTO)
    )
    self.assertEqual(failed.status_code, 200)
    self.assertEqual(failed.json()["category"], "user")
    self.assertIn("dpo", failed.json()["error"])
    self.assertEqual((await self.bus.run(run_id))["failed"], 1)

    healthy_run = await self.create_model()
    queued = await self.post("optim_step", json={"model_id": healthy_run})
    with patch.object(gateway, "WAIT", 0.01):
      pending = await self.post("retrieve_future", json={"request_id": queued.json()["request_id"]})
      self.assertEqual(pending.json()["type"], "try_again")
      unknown = await self.post("retrieve_future", json={"request_id": "unknown-request"})
      self.assertIn("error", unknown.json())

  async def test_sequence_order_and_retries_preserve_one_optimizer_step(self):
    run_id = await self.create_model()
    body = {"model_id": run_id, "seq_id": 2, "optim_params": {"learning_rate": 1e-2}}
    optim = await self.post("optim_step", json=body)
    self.assertEqual(await self.trainers["lora:tiny"].step(block_ms=0), 0)

    forward_body = proto_body(run_id, seq_id=1)
    forward = await self.post("forward_backward", content=forward_body, headers=PROTO)
    retry = await self.post("forward_backward", content=forward_body, headers=PROTO)
    self.assertEqual(retry.json(), forward.json())
    conflict = await self.post("forward_backward", content=proto_body(run_id, n=8, seq_id=1), headers=PROTO)
    self.assertEqual(conflict.status_code, 400)

    await self.result(forward)
    first = await self.result(optim)
    retry = await self.post("optim_step", json=body)
    self.assertEqual(retry.json(), optim.json())
    self.assertEqual((await self.result(retry)).json(), first.json())
    self.assertGreater(first.json()["metrics"]["grad_norm:mean"], 0)
    self.assertEqual((await self.bus.run(run_id))["steps"], 1)

  async def test_creation_and_header_retry_ids_are_stable(self):
    session = (await self.post("create_session", json={})).json()["session_id"]
    body = {"base_model": "tiny", "session_id": session, "model_seq_id": 7, "lora_config": {"rank": 4}}
    created = await self.post("create_model", json=body)
    self.assertEqual((await self.post("create_model", json=body)).json(), created.json())
    conflict = await self.post("create_model", json={**body, "base_model": "different"})
    self.assertEqual(conflict.status_code, 400)
    run_id = created.json()["model_id"]
    await self.result(created)

    headers = {"Idempotency-Key": "one-step"}
    step = await self.post("optim_step", json={"model_id": run_id}, headers=headers)
    retry = await self.post("optim_step", json={"model_id": run_id}, headers=headers)
    self.assertEqual(retry.json(), step.json())
    await self.result(step)
    self.assertEqual((await self.bus.run(run_id))["steps"], 1)

  async def test_worker_http_requires_authentication_and_completion_is_idempotent(self):
    denied = await self.http.post("/internal/register", json={"worker": "worker", "pool": "lora:tiny"})
    self.assertEqual(denied.status_code, 401)
    created = await self.post("create_model", json={"base_model": "tiny", "lora_config": {"rank": 4}})
    async with httpx.AsyncClient(
      transport=httpx.ASGITransport(app=self.api),
      base_url="http://tinyrl",
      headers={"Authorization": "Bearer test-token"},
    ) as connection:
      worker = WorkerClient(http=connection)
      self.assertTrue((await worker.register("worker", "lora:tiny"))["registered"])
      [command] = await worker.take("lora:tiny", "worker", block_ms=0)
      self.assertEqual(command.request_id, created.json()["request_id"])
      outcome = Reply(True, wire.JSON, b'{"type":"create_model"}')
      self.assertEqual(await worker.finish(command, outcome), "committed")
      self.assertEqual(await worker.finish(command, outcome), "already_committed")
      self.assertEqual(await worker.take("lora:tiny", "worker", block_ms=0), [])
    self.assertEqual(await self.bus.result(command.request_id, 0), outcome)

  async def test_pools(self):
    shared, also_shared = await self.create_model(), await self.create_model()
    full = await self.create_model(user_metadata={"openrl.fine_tuning_type": "full"})
    exclusive = await self.create_model(user_metadata={"openrl.exclusive": "true"})

    pools = [(await self.bus.run(run_id))["pool"] for run_id in (shared, also_shared, full, exclusive)]
    self.assertEqual(pools, ["lora:tiny", "lora:tiny", full, exclusive])
    self.assertEqual((await self.bus.run(full))["rank"], 0)
    self.assertEqual((await self.post("get_info", json={"model_id": full})).json()["lora_rank"], 16)

  async def test_delete(self):
    run_id = await self.create_model()
    deleted = await self.post("delete_model", json={"model_id": run_id})
    self.assertEqual(deleted.json(), {"status": "ok"})

    await self.drain(run_id)
    self.assertEqual((await self.post("optim_step", json={"model_id": run_id})).status_code, 404)
    self.assertFalse((await self.http.get(f"/api/v1/runs/{run_id}")).json()["live"])


if __name__ == "__main__":
  unittest.main()
