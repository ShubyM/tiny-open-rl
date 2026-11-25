import asyncio
import datetime
import glob
import json
import os
import tempfile
import time
import unittest

import httpx
import numpy as np
import safetensors.torch
import torch

from test.execution import TestNode
from test.helpers import BusTest, tiny
from tinyrl.api import gateway, wire
from tinyrl.api.proto import tinker_public_pb2 as pb
from tinyrl.control import controller as bus
from tinyrl.storage.store import Store
from tinyrl.types import NodeMsg

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


class TestAPI(BusTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.store = Store(root=tempfile.mkdtemp())
    self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app(self.bus)), base_url="http://tinyrl")
    self.node = TestNode(self.bus, store=self.store)

  async def asyncTearDown(self):
    await self.node.aclose()
    await self.http.aclose()
    await super().asyncTearDown()

  async def post(self, path: str, **kwargs) -> httpx.Response:
    return await self.http.post(f"/api/v1/{path}", **kwargs)

  async def drain(self, run_id: str):
    """Let the node execute the run's worker until its queue is empty."""
    await self.node.drain(run_id)

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
    metrics, worker = (await self.http.get("/metrics")).text, run["worker"]
    self.assertIn(f'tinyrl_steps_total{{worker="{worker}",run="{run_id}"}} 8\n', metrics)
    self.assertIn(f'tinyrl_busy_seconds_total{{worker="{worker}",run="{run_id}"}} {run["busy_ms"] / 1000}\n', metrics)
    self.assertIn("tinyrl_node_loop_lag_seconds{node=", metrics)

    session = (await self.http.get(f"/api/v1/sessions/{session_id}")).json()
    self.assertEqual((session["runs"], session["live"]), ([run_id], True))

    info = (await self.post("get_info", json={"model_id": run_id})).json()
    self.assertEqual(info["model_data"], {"model_name": "tiny", "tokenizer_id": "tiny"})

  async def test_forward_only_matches_forward_backward_logprobs(self):
    run_id = await self.create_model()
    fb_req = pb.ForwardBackwardRequest.FromString(proto_body(run_id, n=3))
    fwdbwd = await self.result(await self.post("forward_backward", content=fb_req.SerializeToString(), headers=PROTO))
    fb_req.forward_only = True
    forward = await self.result(await self.post("forward_backward", content=fb_req.SerializeToString(), headers=PROTO))

    out_fb = pb.ForwardBackwardOutput.FromString(fwdbwd.content)
    out_fwd = pb.ForwardBackwardOutput.FromString(forward.content)
    self.assertEqual(out_fb.loss_fn_outputs[0].num_datums, 2)
    # dropout is off, so the forward pass gives the same logprobs as the forward_backward pass
    self.assertEqual(
      out_fwd.loss_fn_outputs[0].fields["logprobs"].data, out_fb.loss_fn_outputs[0].fields["logprobs"].data
    )

  async def test_checkpoints(self):
    run_id = await self.create_model()
    saved = (await self.result(await self.post("save_weights", json={"model_id": run_id, "path": "final"}))).json()
    self.assertEqual(saved, {"type": "save_weights", "model_id": run_id, "path": f"tinker://{run_id}/weights/final"})

    load = {"model_id": run_id, "path": saved["path"], "optimizer": True}
    loaded = (await self.result(await self.post("load_weights", json=load))).json()
    self.assertEqual((loaded["type"], loaded["path"]), ("load_weights", saved["path"]))

    # what a client resuming from the checkpoint asks first, answered without the run
    info = (await self.post("weights_info", json={"tinker_path": saved["path"]})).json()
    trained = {"train_unembed": True, "train_mlp": True, "train_attn": True}
    self.assertEqual(info, {"base_model": "tiny", "is_lora": True, "lora_rank": 4} | trained)
    self.assertEqual(
      (await self.post("weights_info", json={"tinker_path": f"tinker://{run_id}/weights/nope"})).status_code, 404
    )

    save = {"model_id": run_id, "sampling_session_seq_id": 0}
    sampler = (await self.result(await self.post("save_weights_for_sampler", json=save))).json()
    self.assertEqual(sampler["path"], f"tinker://{run_id}/sampler_weights/step-0")
    with open(f"{self.store.local_folder(self.bus.checkpoint_version(sampler['path']))[0]}/adapter_config.json") as f:
      self.assertEqual(json.load(f)["r"], 4)

    load = {"model_id": run_id, "path": "nope", "optimizer": False}
    missing = (await self.result(await self.post("load_weights", json=load))).json()
    self.assertEqual(missing["category"], "user")

    for bad in ("../../etc", "/etc", f"tinker://{run_id}/weights/../../x"):
      response = await self.post("save_weights", json={"model_id": run_id, "path": bad})
      self.assertEqual(response.status_code, 400)

  async def test_checkpoints_are_listed_expire_and_delete(self):
    run_id = await self.create_model()
    await self.result(await self.post("save_weights", json={"model_id": run_id, "path": "final", "ttl_seconds": 7200}))
    save = {"model_id": run_id, "sampling_session_seq_id": 0}
    await self.result(await self.post("save_weights_for_sampler", json=save))
    listed = (await self.http.get(f"/api/v1/training_runs/{run_id}/checkpoints")).json()
    got = [(c["checkpoint_id"], c["checkpoint_type"], c["expires_at"] is None) for c in listed["checkpoints"]]
    self.assertEqual(got, [("weights/final", "training", False), ("sampler_weights/step-0", "sampler", True)])
    self.assertGreater(listed["checkpoints"][0]["size_bytes"], 0)
    every = (await self.http.get("/api/v1/checkpoints", params={"limit": 1})).json()
    self.assertEqual((len(every["checkpoints"]), every["cursor"]["total_count"]), (1, 2))

    final = f"/api/v1/training_runs/{run_id}/checkpoints/weights/final"
    self.assertEqual((await self.http.put(f"{final}/ttl", json={"ttl_seconds": 60})).status_code, 400)
    self.assertEqual((await self.http.put(f"{final}/ttl", json={"ttl_seconds": None})).status_code, 200)
    self.bus.maintain()
    self.assertEqual(len(self.bus.list_checkpoints()), 2)
    self.assertEqual((await self.http.delete(final)).status_code, 200)
    self.assertEqual((await self.http.delete(final)).status_code, 404)

    await self.post("delete_model", json={"model_id": run_id})
    await self.drain(run_id)  # the sampling session's weights are the run's, so they go with it
    self.assertEqual((await self.http.get(f"/api/v1/training_runs/{run_id}/checkpoints")).json()["checkpoints"], [])

  async def test_a_checkpoint_leaves_the_registry_when_its_ttl_runs_out(self):
    self.seed_checkpoint("tinker://r/weights/soon", 10, ttl=0.2)
    self.seed_checkpoint("tinker://r/weights/later", 10, ttl=3600)
    self.start_maintenance()
    for _ in range(100):
      if self.bus.get_checkpoint("tinker://r/weights/soon") is None:
        break
      await asyncio.sleep(0.05)
    self.assertEqual([c.uri for c in self.bus.list_checkpoints()], ["tinker://r/weights/later"])

  async def test_errors(self):
    run_id = await self.create_model()
    for path, kwargs in [
      ("optim_step", {"json": {"model_id": "nope"}}),
      ("forward_backward", {"content": proto_body("nope"), "headers": PROTO}),
      ("get_info", {"json": {"model_id": "nope"}}),
    ]:
      self.assertEqual((await self.post(path, **kwargs)).status_code, 404)

    self.assertEqual((await self.post("forward_backward", content=b"\xff\xff", headers=PROTO)).status_code, 400)

    # a call that fails in the trainer comes back as a 200 with the error, which is what the SDK reads
    failed = await self.result(
      await self.post("forward_backward", content=proto_body(run_id, loss_fn="dpo"), headers=PROTO)
    )
    self.assertEqual(failed.status_code, 200)
    self.assertEqual(failed.json()["category"], "user")
    self.assertIn("dpo", failed.json()["error"])
    self.assertEqual((self.bus.get_run(run_id))["failed"], 1)

  async def test_workers(self):
    shared, also_shared = await self.create_model(), await self.create_model()
    full = await self.create_model(user_metadata={"openrl.fine_tuning_type": "full"})
    exclusive = await self.create_model(user_metadata={"openrl.exclusive": "true"})

    workers = [self.worker(run_id) for run_id in (shared, also_shared, full, exclusive)]
    self.assertEqual(workers[0], workers[1])
    self.assertEqual(len(set(workers)), 3)
    self.assertEqual((self.bus.get_run(full))["rank"], 0)
    self.assertEqual((await self.post("get_info", json={"model_id": full})).json()["lora_rank"], 16)

  async def test_session_tags_apply_to_its_models(self):
    """TINKER_TAGS reach the server on create_session, which is how open-rl clients ask for a full fine-tune."""
    tags = ["openrl.fine_tuning_type=full", "openrl.recipe=rl"]
    session = await self.post("create_session", json={"tags": tags, "user_metadata": None, "sdk_version": "0.29.0"})
    full = await self.create_model(session_id=session.json()["session_id"])
    self.assertEqual((self.bus.get_run(full))["rank"], 0)
    wid = self.worker(full)
    self.assertEqual([r for r, rs in self.bus.runs.items() if rs.live and rs.owner and rs.owner[0] == wid], [full])

  async def test_delete(self):
    run_id = await self.create_model()
    deleted = await self.post("delete_model", json={"model_id": run_id})
    self.assertEqual(deleted.json(), {"status": "ok"})

    await self.drain(run_id)
    self.assertEqual((await self.post("optim_step", json={"model_id": run_id})).status_code, 404)
    self.assertFalse((await self.http.get(f"/api/v1/runs/{run_id}")).json()["live"])

  async def test_a_session_whose_client_stops_heartbeating_loses_its_runs(self):
    session = await self.post("create_session", json={"tags": [], "user_metadata": None, "sdk_version": "0.29.0"})
    session_id = session.json()["session_id"]
    run_id = await self.create_model(session_id=session_id)
    sampling = await self.post(
      "create_sampling_session", json={"session_id": session_id, "base_model": "tiny", "model_path": ""}
    )
    samp_id = sampling.json()["sampling_session_id"]
    self.assertIsNone((self.bus.get_run(samp_id))["model_path"])
    self.start_maintenance()
    await self.scheduling_until(lambda d: samp_id in d["state"]["jobs"])

    stale = time.time() - bus.SESSION_GRACE - 1
    with self.bus.db:
      self.bus.db.execute("UPDATE sessions SET seen=? WHERE id=?", (stale, session_id))
    await self.post("session_heartbeat", json={"session_id": session_id})  # a heartbeat in time keeps it
    await asyncio.sleep(0.3)
    self.assertTrue((await self.http.get(f"/api/v1/runs/{samp_id}")).json()["live"])

    with self.bus.db:
      self.bus.db.execute("UPDATE sessions SET seen=? WHERE id=?", (stale, session_id))  # longer ago than the grace
    await self.scheduling_until(lambda d: samp_id not in d["state"]["jobs"])
    await self.drain(run_id)  # its training run goes once its queued work is done
    self.assertFalse((await self.http.get(f"/api/v1/runs/{run_id}")).json()["live"])
    self.assertFalse((await self.http.get(f"/api/v1/runs/{samp_id}")).json()["live"])
    cutoff = time.time() - bus.SESSION_GRACE
    self.assertEqual(
      self.bus.db.execute("SELECT id FROM sessions WHERE active=1 AND seen<=?", (cutoff,)).fetchall(), []
    )

  async def test_a_finished_session_loses_its_runs_without_waiting_for_its_heartbeats_to_stop(self):
    session_id = (await self.post("create_session")).json()["session_id"]
    run_id = await self.create_model(session_id=session_id)
    sampling = await self.post("create_sampling_session", json={"session_id": session_id, "base_model": "tiny"})
    samp_id = sampling.json()["sampling_session_id"]
    self.start_maintenance()
    await self.scheduling_until(lambda d: samp_id in d["state"]["jobs"])

    await self.post(f"sessions/{session_id}/finish", json={"reason": {"type": "success"}})
    await self.scheduling_until(lambda d: samp_id not in d["state"]["jobs"])
    await self.drain(run_id)  # its training run goes once its queued work is done
    self.assertFalse((await self.http.get(f"/api/v1/runs/{run_id}")).json()["live"])
    cutoff = time.time() - bus.SESSION_GRACE
    self.assertEqual(
      self.bus.db.execute("SELECT id FROM sessions WHERE active=1 AND seen<=?", (cutoff,)).fetchall(), []
    )

  async def test_a_checkpoint_saved_without_a_ttl_gets_the_deployment_default_unless_its_run_owns_it(self):
    run_id = await self.create_model()
    await self.result(await self.post("save_weights", json={"model_id": run_id, "path": "kept"}))
    await self.result(
      await self.post("save_weights_for_sampler", json={"model_id": run_id, "sampling_session_seq_id": 0})
    )
    await self.result(await self.post("save_weights", json={"model_id": run_id, "path": "own", "ttl_seconds": 7200}))
    ckpts = {
      c["checkpoint_id"]: c
      for c in (await self.http.get(f"/api/v1/training_runs/{run_id}/checkpoints")).json()["checkpoints"]
    }
    left = {
      k: datetime.datetime.fromisoformat(c["expires_at"]).timestamp() - time.time()
      for k, c in ckpts.items()
      if c["expires_at"]
    }
    if gateway.CKPT_TTL:
      self.assertAlmostEqual(left["weights/kept"], gateway.CKPT_TTL, delta=60)
    else:
      self.assertIsNone(ckpts["weights/kept"]["expires_at"])
    self.assertAlmostEqual(left["weights/own"], 7200, delta=60)
    self.assertIsNone(next(c for k, c in ckpts.items() if k.startswith("sampler_weights/"))["expires_at"])

  async def test_training_checkpoints_are_safetensors_with_adam_moments(self):
    run_id = await self.create_model()
    await self.result(await self.post("forward_backward", content=proto_body(run_id), headers=PROTO))
    await self.result(await self.post("optim_step", json={"model_id": run_id, "adam_params": {"learning_rate": 1e-2}}))

    saved = (await self.result(await self.post("save_weights", json={"model_id": run_id, "path": "final_pin"}))).json()

    folder = self.store.local_folder(self.bus.checkpoint_version(saved["path"]))[0]
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
    self.node.default_model = lambda name: tiny(name).to(torch.bfloat16)
    run = await self.create_model(user_metadata={"openrl.fine_tuning_type": "full"})
    saved = (
      await self.result(await self.post("save_weights_for_sampler", json={"model_id": run, "path": "sampler"}))
    ).json()
    files = glob.glob(f"{self.store.local_folder(self.bus.checkpoint_version(saved['path']))[0]}/*.safetensors")
    self.assertTrue(files)
    for path in files:
      self.assertTrue(all(t.dtype == torch.bfloat16 for t in safetensors.torch.load_file(path).values()))

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

    out1 = await self.result(r1)
    out2 = (await self.result(r2)).json()
    self.assertEqual(out1.headers["content-type"], wire.PROTO)
    self.assertEqual(pb.ForwardBackwardOutput.FromString(out1.content).loss_fn_outputs[0].num_datums, 2)
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

    out1 = await self.result(res1)
    out2 = await self.result(res2)
    self.assertEqual(pb.ForwardBackwardOutput.FromString(out1.content).loss_fn_outputs[0].num_datums, 2)
    self.assertEqual(pb.ForwardBackwardOutput.FromString(out2.content).loss_fn_outputs[0].num_datums, 2)

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

    self.bus.handle(NodeMsg("hello", "node-1", "b1", gpus=8))
    self.start_maintenance()
    worker = self.worker(run_id)
    data = await self.scheduling_until(lambda d: d["state"]["procs"].get(worker, {}).get("node"))
    self.assertIn(data["state"]["procs"][worker]["node"], ("test-node", "node-1"))
    sampler = data["state"]["procs"][self.worker(samp_id)]
    self.assertEqual(
      sampler["spec"],
      {"role": "sampler", "base": "tiny", "lora": True, "exclusive": True, "train_run": excl_id, "gpus": 1},
    )
    self.assertIn("node-1", data["nodes"])

    await self.post("delete_model", json={"model_id": run_id})
    await self.drain(run_id)
    data = await self.scheduling_until(lambda d: worker not in d["state"]["procs"])
    self.assertNotIn(run_id, data["state"]["jobs"])

  async def scheduling_until(self, done, tries: int = 100) -> dict:
    """GET /api/v1/scheduling until `done` says yes about it."""
    for _ in range(tries):
      data = (await self.http.get("/api/v1/scheduling")).json()
      if data["state"] and done(data):
        return data
      await asyncio.sleep(0.05)
    self.fail(f"never: {data}")

  async def sampler_runs(self) -> list[str]:
    return [e.body["run"] for e in self.bus.read_events() if e.kind == "run_created" and e.body["role"] == "sampler"]

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
    sampler = self.worker(sid)

    self.bus.handle(NodeMsg("hello", "n8", "b8", gpus=8))
    self.start_maintenance()
    await self.scheduling_until(lambda d: d["state"]["procs"].get(sampler, {}).get("node"))
    await self.post("delete_model", json={"model_id": excl})
    await self.drain(excl)
    await self.scheduling_until(lambda d: sampler not in d["state"]["procs"])

  async def test_lost_sampling_promise_returns_410_and_backpressure_returns_429(self):
    from unittest.mock import patch

    cfg = await self.post("client/config", json={})
    self.assertEqual(cfg.json(), {"parallel_fwdbwd_chunks": False})
    gone = await self.post("retrieve_future", json={"request_id": "samp-missing/1-1"})
    self.assertEqual(gone.status_code, 410)

    session = await self.post("create_sampling_session", json={"base_model": "tiny"})
    sid = session.json()["sampling_session_id"]
    with patch("tinyrl.control.ledger.SAMPLE_QUEUE_BYTES", 8):
      res = await self.post(
        "asample",
        json={
          "sampling_session_id": sid,
          "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
          "sampling_params": {"max_tokens": 2},
          "num_samples": 1,
        },
      )
    self.assertEqual(res.status_code, 429)

  async def _sample_for_stable_reinforce(self, run_id: str, prompt: list[int]):
    saved = await self.post("save_weights_for_sampler", json={"model_id": run_id, "sampling_session_seq_id": 0})
    sid = (await self.result(saved)).json()["sampling_session_id"]
    sub = await self.post(
      "asample",
      json={
        "sampling_session_id": sid,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt}]},
        "sampling_params": {"max_tokens": 6},
        "num_samples": 2,
        "record_stability_info": True,
      },
    )
    self.assertEqual(sub.status_code, 200)
    seq_ids = sub.json()["sample_sequence_ids"]
    self.assertEqual(len(seq_ids), 2)
    await self.drain(sid)
    out = await self.post(
      "retrieve_future", json={"request_id": sub.json()["request_id"]}, headers={"accept": wire.PROTO}
    )
    return seq_ids, pb.SampleResponse.FromString(out.content)

  def _stable_reinforce_req(
    self,
    run_id: str,
    prompt: list[int],
    seq_pb,
    seq_id: str,
    *,
    include_spans: bool = True,
    offset: int = 0,
    prompt_adv: float = 0.0,
  ) -> bytes:
    sampled_tokens = np.frombuffer(seq_pb.tokens, np.int32)
    sampled_lps = np.frombuffer(seq_pb.logprobs, np.float32)
    full = np.concatenate([np.array(prompt, np.int32), sampled_tokens])
    n_prompt, n_samp = len(prompt) - 1, len(sampled_tokens)
    req = pb.ForwardBackwardRequest(model_id=run_id, loss_fn="stable_reinforce")
    req.loss_fn_config_v2["kl_coeff"].number = 0.01
    req.loss_fn_config_v2["kl_estimator"].text = "k3"
    d = req.data.add()
    d.model_input.add().encoded_text.tokens = full[:-1].astype(np.int32).tobytes()
    targets = full[1:].astype(np.int64)
    lps = np.concatenate([np.zeros(n_prompt, np.float32), sampled_lps])
    advs = np.concatenate([np.full(n_prompt, prompt_adv, np.float32), np.ones(n_samp, np.float32)])
    d.loss_fn_inputs["target_tokens"].CopyFrom(
      pb.Tensor(dense=targets.tobytes(), dtype=pb.DTYPE_INT64, shape=[len(targets)])
    )
    d.loss_fn_inputs["logprobs"].CopyFrom(pb.Tensor(dense=lps.tobytes(), dtype=pb.DTYPE_FLOAT32, shape=[len(lps)]))
    d.loss_fn_inputs["advantages"].CopyFrom(pb.Tensor(dense=advs.tobytes(), dtype=pb.DTYPE_FLOAT32, shape=[len(advs)]))
    if include_spans:
      d.model_input_spans.add().prompt_tokens.CopyFrom(
        pb.PromptProvenanceSpan(sequence_id=seq_id, offset=0, length=len(prompt))
      )
      d.model_input_spans.add().sampled_tokens.CopyFrom(
        pb.SampledProvenanceSpan(sequence_id=seq_id, offset=0, length=n_samp - 1)
      )
      d.loss_fn_input_spans.add().prompt_tokens.CopyFrom(
        pb.PromptProvenanceSpan(sequence_id=seq_id, offset=1, length=n_prompt)
      )
      d.loss_fn_input_spans.add().sampled_tokens.CopyFrom(
        pb.SampledProvenanceSpan(sequence_id=seq_id, offset=offset, length=n_samp)
      )
    return req.SerializeToString()

  async def test_stable_reinforce_records_stability_info_and_trains_end_to_end(self):
    run_id, prompt = await self.create_model(), [1, 2, 3, 4]
    seq_ids, resp = await self._sample_for_stable_reinforce(run_id, prompt)
    # Client did not ask for topk_sample_logprobs, so only the training input carries the recorded top-k values.
    self.assertFalse(resp.sequences[0].HasField("topk_sampled_logprobs"))
    self.assertIn(seq_ids[0], self.bus.db.read_stability({seq_ids[0]}, time.time()))

    # Valid provenance -> succeeds and runs optim_step.
    fb = await self.post(
      "forward_backward",
      content=self._stable_reinforce_req(run_id, prompt, resp.sequences[0], seq_ids[0]),
      headers=PROTO,
    )
    fb_out = await self.result(fb, headers={"accept": wire.PROTO})
    self.assertEqual(fb_out.status_code, 200)
    self.assertIn("loss:sum", pb.ForwardBackwardOutput.FromString(fb_out.content).metrics)
    opt = await self.post("optim_step", json={"model_id": run_id, "adam_params": {"learning_rate": 1e-2}})
    self.assertEqual((await self.result(opt)).json()["type"], "optim_step")

    # Validation checks: missing spans, unknown sequence_id, nonzero prompt advantage, nonzero offset all fail cleanly.
    for bad in (
      self._stable_reinforce_req(run_id, prompt, resp.sequences[0], seq_ids[0], include_spans=False),
      self._stable_reinforce_req(run_id, prompt, resp.sequences[0], "unknown-seq-id"),
      self._stable_reinforce_req(run_id, prompt, resp.sequences[0], seq_ids[0], prompt_adv=1.0),
      self._stable_reinforce_req(run_id, prompt, resp.sequences[0], seq_ids[0], offset=1),
    ):
      res = await self.result(await self.post("forward_backward", content=bad, headers=PROTO))
      self.assertIn("error", res.json())

  async def test_stable_reinforce_survives_sqlite_reopen_after_submission_and_dispatch(self):
    run_id, prompt = await self.create_model(), [1, 2, 3, 4]
    seq_ids, resp = await self._sample_for_stable_reinforce(run_id, prompt)

    fb0 = await self.post(
      "forward_backward",
      content=self._stable_reinforce_req(run_id, prompt, resp.sequences[0], seq_ids[0]),
      headers=PROTO,
    )
    fb1 = await self.post(
      "forward_backward",
      content=self._stable_reinforce_req(run_id, prompt, resp.sequences[1], seq_ids[1]),
      headers=PROTO,
    )
    id0, id1 = fb0.json()["request_id"], fb1.json()["request_id"]

    # Leave the first command claimed and the second queued when the controller restarts.
    hello = NodeMsg(
      "hello",
      self.node.id,
      self.node.boot_id,
      gpus=self.node.gpus,
      workers=tuple(w.info() for w in self.node.workers.values()),
    )
    [first] = [t for t in self.bus.handle(hello).tasks if t.cmd and t.cmd.run == run_id]
    self.assertEqual(first.cmd.request_id, id0)

    await self.bus.aclose()
    self.bus = bus.Controller(self.db_path)
    self.node.control = self.bus
    self.node.connected = False
    with self.bus.db:
      self.bus.db.execute("DELETE FROM stability WHERE expires<=?", (float("inf"),))
    self.assertFalse(self.bus.db.read_stability(set(seq_ids), time.time()))
    # Store sweep still preserves stability blobs pinned by queued/claimed commands!
    self.store.sweep(self.store.versions() - set(self.bus.live_versions()), 0)

    # The queued commands retain their own inputs across a restart, sample expiry, and store GC sweep.
    [redelivered] = [t for t in self.bus.handle(hello).tasks if t.cmd and t.cmd.run == run_id]
    self.assertEqual(redelivered, first)
    await self.node.drain(run_id)
    for req_id in (id0, id1):
      reply = await self.bus.wait_for_reply(req_id, 1)
      self.assertTrue(reply.ok, reply.body)
      self.assertIn("loss:sum", pb.ForwardBackwardOutput.FromString(reply.body).metrics)

  async def test_join_sampler_info_weights_info_and_asample_validation(self):
    created = await self.post(
      "create_model", json={"base_model": "tiny", "session_id": None, "lora_config": {"rank": 4}}
    )
    run_id = (await self.result(created)).json()["model_id"]
    save_res = await self.post("save_weights_for_sampler", json={"model_id": run_id, "sampling_session_seq_id": 0})
    saved = (await self.result(save_res)).json()
    sid, path = saved["sampling_session_id"], saved["path"]

    info = (await self.http.get(f"/api/v1/samplers/{sid}")).json()
    self.assertEqual(info, {"sampler_id": sid, "base_model": "tiny", "model_path": path})
    self.assertEqual(
      (await self.post("join_sampling_session", json={"sampling_session_id": sid})).json()["client_counter"], 1
    )
    self.assertEqual(
      (await self.post("join_sampling_session", json={"sampling_session_id": sid})).json()["client_counter"], 2
    )

    winfo = await self.post("weights_info", json={"tinker_path": path})
    self.assertEqual(winfo.status_code, 200)
    self.assertEqual(winfo.json()["base_model"], "tiny")

    bad = await self.post(
      "asample", json={"base_model": "tiny", "prompt": {"chunks": [{"tokens": [1, 2]}]}, "target_prompt_logprobs": True}
    )
    self.assertEqual(bad.status_code, 400)


if __name__ == "__main__":
  unittest.main()
