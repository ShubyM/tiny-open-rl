import json, tempfile, unittest
from unittest.mock import patch
import httpx, numpy as np
from tinyrl import gateway, wire
from tinyrl.proto import tinker_public_pb2 as pb
from tinyrl.trainer import Model, Trainer
from test.helpers import RedisTest, tiny
from test.test_wire import body

def proto(model_id:str, n:int=12, loss_fn:str="cross_entropy") -> bytes:  # forward_backward the way the sdk sends it
  req = pb.ForwardBackwardRequest(model_id=model_id, loss_fn=loss_fn)
  for shift in (0, 40):
    d, t = req.data.add(), np.arange(shift, shift+n+1)
    d.model_input.add().encoded_text.tokens = t[:-1].astype(np.int32).tobytes()
    d.loss_fn_inputs["target_tokens"].CopyFrom(pb.Tensor(dense=t[1:].astype(np.int64).tobytes(), dtype=pb.DTYPE_INT64, shape=[n]))
    d.loss_fn_inputs["weights"].CopyFrom(pb.Tensor(dense=np.ones(n, np.float32).tobytes(), dtype=pb.DTYPE_FLOAT32, shape=[n]))
  return req.SerializeToString()

class TestAPI(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.enterContext(patch.object(gateway, "CKPT", tempfile.mkdtemp()))
    self.http, self.trainers = httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app(self.bus)), base_url="http://tinyrl"), {}
  async def asyncTearDown(self):
    await self.http.aclose()
    await super().asyncTearDown()

  async def post(self, path:str, **kw) -> httpx.Response: return await self.http.post(f"/api/v1/{path}", **kw)
  async def drain(self, run:str):  # the run's pool trainer does all its pool's queued work
    t = self.trainers.setdefault(pool:=(await self.bus.run(run))["pool"], Trainer(self.bus, pool, Model(tiny)))
    while await t.step(block=10): pass
  async def result(self, r:httpx.Response, **kw) -> httpx.Response:
    await self.drain(r.json()["model_id"])
    return await self.post("retrieve_future", json={"request_id": r.json()["request_id"]}, **kw)
  async def model(self, **kw) -> str:
    r = await self.post("create_model", json={"base_model": "tiny", "lora_config": {"rank": 4, "seed": 0, "lora_dropout": 0}, **kw})
    return (await self.result(r)).json()["model_id"]

  async def test_a_tinker_training_loop(self):
    sid = (await self.post("create_session", json={"tags": [], "user_metadata": None, "sdk_version": "0.29.0"})).json()["session_id"]
    self.assertEqual((await self.post("session_heartbeat", json={"session_id": sid})).json(), {"type": "session_heartbeat"})
    run, losses = await self.model(session_id=sid, model_seq_id=0), []
    for _ in range(8):
      fb = await self.post("forward_backward", content=proto(run), headers={"content-type": wire.PROTO})
      opt = await self.post("optim_step", json={"model_id": run, "adam_params": {"learning_rate": 1e-2, "grad_clip_norm": 0.0}})
      out = await self.result(fb, headers={"accept": wire.PROTO})
      self.assertEqual(out.headers["content-type"], wire.PROTO)
      losses.append((o:=pb.ForwardBackwardOutput.FromString(out.content)).metrics["loss:sum"])
      self.assertEqual((o.loss_fn_outputs[0].num_datums, (await self.result(opt)).json()["type"]), (2, "optim_step"))
    self.assertLess(losses[-1], losses[0]*0.7)
    info = (await self.http.get(f"/api/v1/runs/{run}")).json()
    self.assertEqual((info["steps"], info["failed"], info["live"], info["queued"], info["in_flight"], info["session"]), (8, 0, True, 0, [], sid))
    self.assertEqual((s:=(await self.http.get(f"/api/v1/sessions/{sid}")).json())["runs"], [run])
    self.assertTrue(s["live"])
    self.assertEqual((await self.post("get_info", json={"model_id": run})).json()["model_data"], {"model_name": "tiny", "tokenizer_id": "tiny"})

  async def test_json_clients_get_json(self):
    run = await self.model()
    j = (await self.result(await self.post("forward_backward", json=body() | {"model_id": run}))).json()
    fwd = (await self.result(await self.post("forward", json={"model_id": run, "forward_input": body()["forward_backward_input"]}))).json()
    self.assertEqual((j["type"], len(j["loss_fn_outputs"]), j["loss_fn_outputs"][0]["logprobs"]["shape"]), ("forward_backward", 2, [3]))
    self.assertEqual([o["logprobs"]["data"] for o in fwd["loss_fn_outputs"]], [o["logprobs"]["data"] for o in j["loss_fn_outputs"]])

  async def test_checkpoints(self):
    run = await self.model()
    saved = (await self.result(await self.post("save_weights", json={"model_id": run, "path": "final"}))).json()
    self.assertEqual(saved, {"type": "save_weights", "model_id": run, "path": f"tinker://{run}/weights/final"})
    loaded = (await self.result(await self.post("load_weights", json={"model_id": run, "path": saved["path"], "optimizer": True}))).json()
    self.assertEqual((loaded["type"], loaded["path"]), ("load_weights", saved["path"]))
    sampler = (await self.result(await self.post("save_weights_for_sampler", json={"model_id": run, "sampling_session_seq_id": 0}))).json()
    self.assertEqual(sampler["path"], f"tinker://{run}/sampler_weights/step-0")
    with open(f"{gateway.CKPT}/{run}/sampler_weights/step-0/adapter_config.json") as f: self.assertEqual(json.load(f)["r"], 4)
    missing = (await self.result(await self.post("load_weights", json={"model_id": run, "path": "nope", "optimizer": False}))).json()
    self.assertEqual(missing["category"], "user")
    for bad in ("../../etc", "/etc", f"tinker://{run}/weights/../../x"):
      self.assertEqual((await self.post("save_weights", json={"model_id": run, "path": bad})).status_code, 400)

  async def test_errors(self):
    run = await self.model()
    nope = {"content": proto("nope"), "headers": {"content-type": wire.PROTO}}
    for path, kw in [("optim_step", {"json": {"model_id": "nope"}}), ("forward_backward", nope), ("get_info", {"json": {"model_id": "nope"}})]:
      self.assertEqual((await self.post(path, **kw)).status_code, 404)
    self.assertEqual((await self.post("forward_backward", content=b"\xff\xff", headers={"content-type": wire.PROTO})).status_code, 400)
    self.assertEqual((await self.post("forward_backward", json={"forward_backward_input": {}})).status_code, 400)
    failed = await self.result(await self.post("forward_backward", content=proto(run, loss_fn="dpo"), headers={"content-type": wire.PROTO}))
    self.assertEqual((failed.status_code, failed.json()["category"]), (200, "user"))
    self.assertIn("dpo", failed.json()["error"])
    with patch.object(gateway, "WAIT", 0.01):
      self.assertEqual((await self.post("retrieve_future", json={"request_id": f"{run}/1-0"})).json()["type"], "try_again")
    self.assertEqual((await self.bus.run(run))["failed"], 1)

  async def test_pools(self):
    shared, other = await self.model(), await self.model()
    full = await self.model(user_metadata={"openrl.fine_tuning_type": "full"})
    exclusive = await self.model(user_metadata={"openrl.exclusive": "true"})
    self.assertEqual([(await self.bus.run(r))["pool"] for r in (shared, other, full, exclusive)], ["lora:tiny", "lora:tiny", full, exclusive])
    self.assertEqual(((await self.bus.run(full))["rank"], (await self.post("get_info", json={"model_id": full})).json()["lora_rank"]), (0, 16))

  async def test_delete(self):
    run = await self.model()
    self.assertEqual((await self.post("delete_model", json={"model_id": run})).json(), {"status": "ok"})
    await self.drain(run)
    self.assertEqual((await self.post("optim_step", json={"model_id": run})).status_code, 404)
    self.assertFalse((await self.http.get(f"/api/v1/runs/{run}")).json()["live"])

if __name__ == "__main__":
  unittest.main()
