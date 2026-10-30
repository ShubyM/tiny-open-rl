import asyncio
import glob
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np
import safetensors.torch
import torch

from test.helpers import RedisTest, tiny
from test.test_wire import json_body
from tinyrl import gateway, scheduler, wire
from tinyrl.proto import tinker_public_pb2 as pb
from tinyrl.store import Store
from tinyrl.trainer import Model, Trainer

PROTO = {"content-type": wire.PROTO}


def proto_body(model_id: str, n: int = 12, loss_fn: str = "cross_entropy") -> bytes:
  """A forward_backward the way the SDK sends it: two sequences that count up."""
  req = pb.ForwardBackwardRequest(model_id=model_id, loss_fn=loss_fn)
  for start in (0, 40):
    tokens = np.arange(start, start + n + 1)
    datum = req.data.add()
    datum.model_input.add().encoded_text.tokens = tokens[:-1].astype(np.int32).tobytes()
    targets = pb.Tensor(dense=tokens[1:].astype(np.int64).tobytes(), dtype=pb.DTYPE_INT64, shape=[n])
    weights = pb.Tensor(dense=np.ones(n, np.float32).tobytes(), dtype=pb.DTYPE_FLOAT32, shape=[n])
    datum.loss_fn_inputs["target_tokens"].CopyFrom(targets)
    datum.loss_fn_inputs["weights"].CopyFrom(weights)
  return req.SerializeToString()


class TestAPI(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.store = Store(root=tempfile.mkdtemp())
    self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app(self.bus)), base_url="http://tinyrl")
    self.trainers: dict[str, Trainer] = {}

  async def asyncTearDown(self):
    await self.http.aclose()
    await super().asyncTearDown()

  async def post(self, path: str, **kwargs) -> httpx.Response:
    return await self.http.post(f"/api/v1/{path}", **kwargs)

  async def drain(self, run_id: str):
    """Let the trainer for the run's pool do all of the pool's queued work."""
    pool = await self.pool(run_id)
    if pool not in self.trainers:
      self.trainers[pool] = Trainer(self.bus, pool, Model(tiny), store=self.store)
      await self.own(self.trainers[pool])
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
    self.assertEqual(run["cmds"], 17)
    metrics, pool = (await self.http.get("/metrics")).text, run["pool"]
    self.assertIn(f'tinyrl_steps_total{{pool="{pool}",run="{run_id}"}} 8\n', metrics)
    self.assertIn(f'tinyrl_busy_seconds_total{{pool="{pool}",run="{run_id}"}} {run["busy_ms"] / 1000}\n', metrics)

    session = (await self.http.get(f"/api/v1/sessions/{session_id}")).json()
    self.assertEqual((session["runs"], session["live"]), ([run_id], True))

    info = (await self.post("get_info", json={"model_id": run_id})).json()
    self.assertEqual(info["model_data"], {"model_name": "tiny", "tokenizer_id": "tiny"})

  async def test_a_call_counts_its_time_in_the_queue_and_running(self):
    run_id = await self.create_model()
    before = (await self.http.get(f"/api/v1/runs/{run_id}")).json()
    submitted = await self.post("forward_backward", content=proto_body(run_id), headers=PROTO)
    await asyncio.sleep(0.3)
    await self.result(submitted)
    after = (await self.http.get(f"/api/v1/runs/{run_id}")).json()
    self.assertEqual(after["cmds"] - before["cmds"], 1)
    self.assertGreaterEqual(after["wait_ms"] - before["wait_ms"], 300)
    self.assertGreater(after["busy_ms"] - before["busy_ms"], 0)

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
    with open(f"{self.store.path(sampler['path'])}/adapter_config.json") as f:
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

    with patch.object(gateway, "WAIT", 0.01):
      pending = await self.post("retrieve_future", json={"request_id": f"{run_id}/1-0"})
      self.assertEqual(pending.json()["type"], "try_again")

  async def test_pools(self):
    shared, also_shared = await self.create_model(), await self.create_model()
    full = await self.create_model(user_metadata={"openrl.fine_tuning_type": "full"})
    exclusive = await self.create_model(user_metadata={"openrl.exclusive": "true"})

    pools = [(await self.bus.run(run_id))["pool"] for run_id in (shared, also_shared, full, exclusive)]
    self.assertEqual(pools[0], pools[1])
    self.assertEqual(len(set(pools)), 3)
    self.assertEqual((await self.bus.run(full))["rank"], 0)
    self.assertEqual((await self.post("get_info", json={"model_id": full})).json()["lora_rank"], 16)

  async def test_session_tags_apply_to_its_models(self):
    """TINKER_TAGS reach the server on create_session, which is how open-rl clients ask for a full fine-tune."""
    tags = ["openrl.fine_tuning_type=full", "openrl.recipe=rl"]
    session = await self.post("create_session", json={"tags": tags, "user_metadata": None, "sdk_version": "0.29.0"})
    full = await self.create_model(session_id=session.json()["session_id"])
    self.assertEqual((await self.bus.run(full))["rank"], 0)
    self.assertEqual(await self.bus.runs(await self.pool(full)), [full])

  async def test_delete(self):
    run_id = await self.create_model()
    deleted = await self.post("delete_model", json={"model_id": run_id})
    self.assertEqual(deleted.json(), {"status": "ok"})

    await self.drain(run_id)
    self.assertEqual((await self.post("optim_step", json={"model_id": run_id})).status_code, 404)
    self.assertFalse((await self.http.get(f"/api/v1/runs/{run_id}")).json()["live"])

  async def test_training_checkpoints_are_safetensors_with_adam_moments(self):
    run_id = await self.create_model()
    await self.result(await self.post("forward_backward", content=proto_body(run_id), headers=PROTO))
    await self.result(await self.post("optim_step", json={"model_id": run_id, "adam_params": {"learning_rate": 1e-2}}))

    saved = (await self.result(await self.post("save_weights", json={"model_id": run_id, "path": "final_pin"}))).json()

    folder = self.store.path(saved["path"])
    files = os.listdir(folder)
    self.assertTrue(all(f.endswith(".safetensors") for f in files), f"Found non-safetensors files: {files}")

    found_exp_avg = False
    for f in glob.glob(f"{folder}/*.safetensors"):
      tensors = safetensors.torch.load_file(f)
      for k in tensors:
        if k.endswith(".exp_avg"):
          found_exp_avg = True
          break
    self.assertTrue(found_exp_avg, "No .exp_avg tensors found in safetensors")

  async def test_sampler_weights_preserve_base_model_dtype(self):
    async def drain_bf16(run_id: str):
      pool = await self.pool(run_id)
      if pool not in self.trainers:

        def bf16(name: str):
          return tiny(name).to(torch.bfloat16)

        self.trainers[pool] = Trainer(self.bus, pool, Model(bf16), store=self.store)
        await self.own(self.trainers[pool])
      while await self.trainers[pool].step(block_ms=10):
        pass

    with patch.object(self, "drain", drain_bf16):
      full_run = await self.create_model(user_metadata={"openrl.fine_tuning_type": "full"})
      saved = (
        await self.result(await self.post("save_weights_for_sampler", json={"model_id": full_run, "path": "sampler"}))
      ).json()
    folder = self.store.path(saved["path"])

    tensor_count = 0
    for f in glob.glob(f"{folder}/*.safetensors"):
      tensors = safetensors.torch.load_file(f)
      for k, v in tensors.items():
        self.assertEqual(v.dtype, torch.bfloat16, f"Tensor {k} is {v.dtype}, expected bf16")
        tensor_count += 1
    self.assertGreater(tensor_count, 0, "No tensors found in sampler safetensors")

  async def test_a_retried_call_runs_once_and_returns_the_first_reply(self):
    run_id = await self.create_model()
    pb1 = pb.ForwardBackwardRequest.FromString(proto_body(run_id))
    pb1.seq_id = 1
    req1 = pb1.SerializeToString()

    r1 = await self.post("forward_backward", content=req1, headers=PROTO)
    r2 = await self.post("forward_backward", content=req1, headers=PROTO)

    r3 = await self.post("optim_step", json={"model_id": run_id, "adam_params": {}, "seq_id": 2})
    r4 = await self.post("optim_step", json={"model_id": run_id, "adam_params": {}, "seq_id": 2})

    await self.drain(run_id)

    out1 = await self.result(r1)
    out2 = await self.result(r2)
    self.assertEqual(out1.content, out2.content)

    out3 = await self.result(r3)
    out4 = await self.result(r4)
    self.assertEqual(out3.content, out4.content)

    run = (await self.http.get(f"/api/v1/runs/{run_id}")).json()
    self.assertEqual(run["steps"], 1)

  async def test_a_save_never_takes_another_calls_reply(self):
    run_id = await self.create_model()
    pb1 = pb.ForwardBackwardRequest.FromString(proto_body(run_id))
    pb1.seq_id = 1
    r1 = await self.post("forward_backward", content=pb1.SerializeToString(), headers=PROTO)

    save = {"model_id": run_id, "sampling_session_seq_id": 1, "seq_id": 2}
    r2 = await self.post("save_weights_for_sampler", json=save)

    await self.drain(run_id)

    out1 = (await self.result(r1)).json()
    out2 = (await self.result(r2)).json()
    self.assertEqual(out1["type"], "forward_backward")
    self.assertEqual(out2["type"], "save_weights_for_sampler")
    self.assertIn("path", out2)

  async def test_chunks_of_one_call_run_even_out_of_order(self):
    run_id = await self.create_model()
    pb1 = pb.ForwardBackwardRequest.FromString(proto_body(run_id))
    pb1.seq_id = 2
    res2 = await self.post("forward_backward", content=pb1.SerializeToString(), headers=PROTO)
    r2 = res2.json()

    pb1.seq_id = 1
    res1 = await self.post("forward_backward", content=pb1.SerializeToString(), headers=PROTO)
    r1 = res1.json()
    self.assertNotEqual(r1["request_id"], r2["request_id"])

    out1 = (await self.result(res1)).json()
    out2 = (await self.result(res2)).json()
    self.assertEqual(out1["type"], "forward_backward")
    self.assertEqual(out2["type"], "forward_backward")

  async def test_a_call_on_a_deleted_run_is_a_404_with_or_without_a_seq_id(self):
    run_id = await self.create_model()
    await self.post("delete_model", json={"model_id": run_id, "seq_id": 1})
    await self.drain(run_id)

    res = await self.post("optim_step", json={"model_id": run_id, "seq_id": 2})
    self.assertEqual(res.status_code, 404)
    self.assertEqual((await self.post("optim_step", json={"model_id": run_id})).status_code, 404)

  async def test_session_finish_and_telemetry_succeed(self):
    sid = (await self.post("create_session")).json()["session_id"]
    self.assertEqual(
      (await self.post(f"sessions/{sid}/finish", json={"reason": {"type": "complete"}})).status_code, 200
    )
    self.assertEqual((await self.post("telemetry", json={"events": []})).json(), {"status": "accepted"})

  async def test_end_to_end_scheduling_inspection_and_cleanup(self):
    run_id = await self.create_model()
    excl_id = await self.create_model(user_metadata={"openrl.exclusive": "true"})
    saved = await self.result(
      await self.post("save_weights_for_sampler", json={"model_id": excl_id, "sampling_session_seq_id": 0})
    )
    samp_id = saved.json()["sampling_session_id"]

    await self.bus.heartbeat("node-1", 8)
    await self.bus.emit("node_up", node="node-1", gpus=8, memory=0)
    self.renewal_tasks.append(asyncio.create_task(scheduler.run(self.bus)))
    pool = await self.pool(run_id)
    data = await self.scheduling_until(lambda d: d["state"]["pools"].get(pool, {}).get("node"))
    self.assertEqual(data["state"]["pools"][pool]["node"], "node-1")
    sampler = data["state"]["pools"][await self.pool(samp_id)]
    self.assertEqual((sampler["role"], sampler["exclusive"], sampler["train_run"]), ("sampler", True, excl_id))
    self.assertIn("node-1", data["nodes"])

    await self.post("delete_model", json={"model_id": run_id})
    await self.drain(run_id)
    data = await self.scheduling_until(lambda d: pool not in d["state"]["pools"])
    self.assertNotIn(run_id, data["state"]["runs"])

  async def scheduling_until(self, done, tries: int = 100) -> dict:
    """GET /api/v1/scheduling until `done` says yes about it."""
    for _ in range(tries):
      data = (await self.http.get("/api/v1/scheduling")).json()
      if data["state"] and done(data):
        return data
      await asyncio.sleep(0.05)
    self.fail(f"never: {data}")

  async def sampler_runs(self) -> list[str]:
    return [e.body["run"] for e in await self.bus.events() if e.kind == "run_created" and e.body["role"] == "sampler"]

  async def test_only_sampling_sessions_get_a_sampler_and_it_ends_with_its_training_run(self):
    sft = await self.create_model()
    await self.result(await self.post("forward_backward", content=proto_body(sft), headers=PROTO))
    await self.result(await self.post("optim_step", json={"model_id": sft, "adam_params": {}}))
    await self.result(await self.post("save_weights", json={"model_id": sft, "path": "sft-ckpt"}))
    self.assertEqual(
      (await self.post("save_weights_for_sampler", json={"model_id": sft, "path": "../bad"})).status_code, 400
    )
    await self.post("save_weights_for_sampler", json={"model_id": sft, "path": "000000"})
    self.assertEqual(await self.sampler_runs(), [])

    excl = await self.create_model(user_metadata={"openrl.exclusive": "true"})
    saved = await self.post("save_weights_for_sampler", json={"model_id": excl, "sampling_session_seq_id": 0})
    sid = (await self.result(saved)).json()["sampling_session_id"]
    self.assertEqual(await self.sampler_runs(), [sid])
    sampler = await self.pool(sid)

    await self.bus.heartbeat("n8", 8)
    await self.bus.emit("node_up", node="n8", gpus=8, memory=0)
    self.renewal_tasks.append(asyncio.create_task(scheduler.run(self.bus)))
    data = await self.scheduling_until(lambda d: d["state"]["pools"].get(sampler, {}).get("node"))
    self.assertEqual(data["state"]["pools"][sampler]["env"].get("TINYRL_WARM_SAMPLER"), "1")
    await self.post("delete_model", json={"model_id": excl})
    await self.drain(excl)
    await self.scheduling_until(lambda d: sampler not in d["state"]["pools"])


if __name__ == "__main__":
  unittest.main()
