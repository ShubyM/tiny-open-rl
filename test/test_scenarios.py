"""Real end-to-end validation scenarios covering storage lifecycle, training continuation, fencing, and fleet replay."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

import httpx
import torch

from test.execution import TestNode
from test.helpers import FakeEngine, RedisTest, new_id, tiny
from test.test_process import require_unix
from test.test_sampler import JOB
from tinyrl.api import wire
from tinyrl.api.gateway import app
from tinyrl.bus import Op, Role, Run, key
from tinyrl.control import scheduler
from tinyrl.control.node_daemon import NodeDaemon
from tinyrl.storage.store import Store, parse_version
from tinyrl.workers.model import Model
from tinyrl.workers.sampler import SamplingAbortedError
from tinyrl.workers.trainer import Trainer

GIB = 2**30


def _sample_datum(tokens: list[int], model_id: str = "m") -> bytes:
  datum = {
    "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens[:-1]}]},
    "loss_fn_inputs": {"target_tokens": tokens[1:], "weights": [1.0] * (len(tokens) - 1)},
  }
  return wire.from_json({"model_id": model_id, "forward_backward_input": {"data": [datum], "loss_fn": "cross_entropy"}})


class TestScenarioStorageLifecycle(RedisTest):
  """Scenario 1: versioned reads, active-reader retention/eviction protection, source loss, and incomplete rejection."""

  async def test_storage_lifecycle_version_pins_durability_and_source_loss(self):
    with tempfile.TemporaryDirectory() as folder:
      bucket = f"file://{folder}/bucket"
      writer = Store(bucket, cache=f"{folder}/writer")
      writer.start()
      reader = Store(bucket, cache=f"{folder}/reader")
      try:
        uri = "tinker://run1/weights/ckpt"
        # 1. Publish version A (durable), then overwrite same URI with version B (_latest -> B).
        ver_a = writer.put(uri, {"w": torch.full((4, 4), 1.0)}, {"step": 1})
        ver_a_id, _ = parse_version(ver_a)
        self.assertTrue(writer.wait_durable(uri, ver_a, timeout=10.0))
        await self.registry.register_checkpoint(uri, 64, version=ver_a, durable=True, node="n1", meta={"step": 1})

        ver_b = writer.put(uri, {"w": torch.full((4, 4), 2.0)}, {"step": 2})
        ver_b_id, _ = parse_version(ver_b)
        self.assertTrue(writer.wait_durable(uri, ver_b, timeout=10.0))
        await self.registry.register_checkpoint(uri, 64, version=ver_b, durable=True, node="n1", meta={"step": 2})
        self.assertNotEqual(ver_a_id, ver_b_id)

        # Read A directly by exact version after B became _latest.
        tensors_a, meta_a = reader.get(uri, version=ver_a_id)
        self.assertTrue(torch.equal(tensors_a["w"], torch.full((4, 4), 1.0)))
        self.assertEqual(meta_a["step"], 1)

        # 2. Protect active reader via registry read ref + cross-process flock while sweep & tight eviction run.
        self.assertTrue(await self.registry.acquire_ref(ver_a_id, "read:reader:1", ttl=60.0))
        await self.registry.unregister_checkpoints(uri)
        with writer.pin_local(uri, ver_a_id) as (pinned_dir, _):
          self.assertFalse(await self.registry.claim_delete(ver_a_id))
          await self.registry.sweep(writer, grace=0)
          with mock.patch("tinyrl.storage.store.CACHE_BYTES", 1):
            writer.trim_cache()
          self.assertTrue(os.path.exists(os.path.join(pinned_dir, "manifest.json")))
          loaded, _ = writer.get(uri, version=ver_a_id)
          self.assertTrue(torch.equal(loaded["w"], torch.full((4, 4), 1.0)))

        # Once active reader releases ref and lock, version A can be claimed and deleted cleanly.
        await self.registry.release_ref(ver_a_id, "read:reader:1")
        self.assertTrue(await self.registry.claim_delete(ver_a_id))
        writer.delete_version(uri, ver_a_id)
        with self.assertRaises(ValueError):
          writer.get(uri, version=ver_a_id)

        # 3. Un-uploaded local/peer copy survives peer read while writer lives, fails cleanly after source loss.
        writer_nodur = Store(bucket, cache=f"{folder}/writer2")
        with mock.patch.object(writer_nodur, "upload_pending_saves"):
          writer_nodur.start()
          try:
            uri_c = "tinker://run1/weights/ephemeral"
            ver_c = writer_nodur.put(uri_c, {"w": torch.full((2, 2), 3.0)}, {"step": 3})
            ver_c_id, peer_c = parse_version(ver_c)
            self.assertTrue(peer_c)
            await self.registry.register_checkpoint(uri_c, 32, version=ver_c, durable=False, node="n2")
            t_peer, _ = reader.get(uri_c, version=ver_c_id, peers=[peer_c])
            self.assertTrue(torch.equal(t_peer["w"], torch.full((2, 2), 3.0)))
          finally:
            writer_nodur.close()  # peer server stops without uploading

        # Source died before durable upload -> fresh reader fails cleanly (never silent fallback).
        reader_fresh = Store(bucket, cache=f"{folder}/reader_fresh")
        with self.assertRaises(ValueError):
          reader_fresh.get(uri_c, version=ver_c_id, peers=[peer_c])

        # 4. Version advertised as durable survives total source loss (peer down + local cache wiped).
        uri_d = "tinker://run1/weights/durable"
        ver_d = writer.put(uri_d, {"w": torch.full((2, 2), 4.0)}, {"step": 4})
        ver_d_id, peer_d = parse_version(ver_d)
        self.assertTrue(writer.wait_durable(uri_d, ver_d, timeout=10.0))
        await self.registry.register_checkpoint(uri_d, 32, version=ver_d, durable=True, node="n1")
        writer.close()
        writer = None
        t_dur, _ = reader_fresh.get(uri_d, version=ver_d_id, peers=[peer_d])
        self.assertTrue(torch.equal(t_dur["w"], torch.full((2, 2), 4.0)))

        # 5. Incomplete version directory (without manifest.json) is refused by peer HTTP (404) and local reader.
        peer_only = Store(bucket, cache=f"{folder}/peer_partial")
        with mock.patch.object(peer_only, "upload_pending_saves"):
          peer_only.start()
          try:
            partial_dir = os.path.join(peer_only.cache, "vpartial")
            os.makedirs(partial_dir, exist_ok=True)
            with open(os.path.join(partial_dir, "model.safetensors"), "wb") as f:
              f.write(b"partial-bytes")
            with self.assertRaises(urllib.error.HTTPError) as cm:
              urllib.request.urlopen(f"http://{peer_only.peer}/vpartial/manifest.json", timeout=2)
            self.assertEqual(cm.exception.code, 404)
            with self.assertRaises(ValueError):
              peer_only.local_folder("tinker://run1/weights/partial", "vpartial")
          finally:
            peer_only.close()
      finally:
        if writer is not None:
          writer.close()


class TestScenarioTrainingContinuation(RedisTest):
  """Scenario 2: exact training continuation across nondefault LoRA, dropout RNG isolation, full-FT, and 2-rank FSDP."""

  async def test_nondefault_lora_dropout_rng_isolation_accumulated_grad_and_fresh_session_restore(self):
    with tempfile.TemporaryDirectory() as folder:
      base = f"{folder}/tiny"
      tiny().save_pretrained(base)
      store = Store(f"{folder}/ckpt")

      lora_cfg = {
        "rank": 8,
        "seed": 42,
        "lora_alpha": 16.0,
        "lora_dropout": 0.25,
        "train_attn": True,
        "train_mlp": False,
        "train_unembed": False,
      }
      b1 = _sample_datum(list(range(1, 17)))
      b2 = _sample_datum(list(range(5, 21)))
      b3 = _sample_datum(list(range(9, 25)))
      adam = json.dumps({"adam_params": {"learning_rate": 1e-3}}).encode()

      # 1. Reference uninterrupted trainer (no interleaving, no restart).
      ref = Trainer(Model(tiny), store)
      ref_id = new_id()
      ref.model.create_run(ref_id, base, 8, lora_cfg)
      ref.model.forward_backward(ref_id, wire.datums(wire.parse(b1)), "cross_entropy", {}, False)
      ref.model.optim_step(ref_id, {"learning_rate": 1e-3})
      ref.model.forward_backward(ref_id, wire.datums(wire.parse(b2)), "cross_entropy", {}, False)  # accumulated grad
      ref_step2 = ref.model.optim_step(ref_id, {"learning_rate": 1e-3})
      _, ref_loss3 = ref.model.forward_backward(ref_id, wire.datums(wire.parse(b3)), "cross_entropy", {}, False)
      ref_step3 = ref.model.optim_step(ref_id, {"learning_rate": 1e-3})
      ref_state, _ = ref.model.training_state(ref_id)

      # 2. Shared trainer with interleaved sibling job B, save with accumulated grad, restore into a FRESH session via gateway!
      node1 = TestNode(self.bus, store=store)
      client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app(self.bus, store)), base_url="http://test")
      self.addAsyncCleanup(client.aclose)
      try:
        res_a = (await client.post("/api/v1/create_model", json={"base_model": base, "lora_config": lora_cfg})).json()
        res_sib = (
          await client.post(
            "/api/v1/create_model",
            json={"base_model": base, "lora_config": {"rank": 4, "seed": 7, "lora_dropout": 0.3}},
          )
        ).json()
        run_a, run_sib = res_a["model_id"], res_sib["model_id"]
        w = await node1.worker_for(run_a)
        self.assertEqual(w, await node1.worker_for(run_sib))  # shared LoRA engine
        await node1.add(w)
        await node1.drain(run_a)

        # Interleave sibling before and between A's steps; A's per-job RNG stream must stay isolated!
        await self.bus.submit_command(run_sib, Op.FORWARD_BACKWARD, b1)
        await self.bus.submit_command(run_a, Op.FORWARD_BACKWARD, b1)
        await self.bus.submit_command(run_sib, Op.OPTIM_STEP, adam)
        await self.bus.submit_command(run_a, Op.OPTIM_STEP, adam)
        # Explicit save after step 1 for fresh-session /api/v1/load_weights(optimizer=True).
        save_req = await self.bus.submit_command(
          run_a, Op.SAVE_WEIGHTS, json.dumps({"uri": f"tinker://{run_a}/weights/step1"}).encode()
        )
        await node1.drain(run_a)
        save_rep = await self.bus.wait_for_reply(save_req, 5)
        self.assertTrue(save_rep.ok, save_rep.body)
        ckpt_uri = json.loads(save_rep.body)["path"]

        # Also accumulate pending gradient on run_a, then park/drain node1 gracefully so handoff preserves pending grad!
        await self.bus.submit_command(run_sib, Op.FORWARD_BACKWARD, b2)
        await self.bus.submit_command(run_a, Op.FORWARD_BACKWARD, b2)
        await node1.drain(run_a)

        # Fresh session inspects checkpoint metadata via /api/v1/weights_info.
        info = (await client.post("/api/v1/weights_info", json={"tinker_path": ckpt_uri})).json()
        self.assertEqual(info["base_model"], base)
        self.assertTrue(info["is_lora"])
        self.assertEqual(info["lora_rank"], 8)
        self.assertFalse(info["train_mlp"])
        self.assertTrue(info["train_attn"])

        # Gracefully close node1 (parking run_a with its pending accumulated gradient + RNG).
        await node1.close(graceful=True)
      finally:
        await node1.aclose()

      node2 = TestNode(self.bus, store=store, node="node-2")
      try:
        # Part A: Planned handoff of run_a to fresh process node2 preserves pending accumulated grad + optimizer + RNG!
        w_a = await node2.worker_for(run_a)
        await node2.add(w_a)
        h_step2 = await self.bus.wait_for_reply(await self.bus.submit_command(run_a, Op.OPTIM_STEP, adam), 5)
        h_loss3 = await self.bus.wait_for_reply(await self.bus.submit_command(run_a, Op.FORWARD_BACKWARD, b3), 5)
        h_step3 = await self.bus.wait_for_reply(await self.bus.submit_command(run_a, Op.OPTIM_STEP, adam), 5)
        self.assertTrue(h_step2.ok and h_loss3.ok and h_step3.ok)
        self.assertAlmostEqual(
          json.loads(h_step2.body)["metrics"]["grad_norm:mean"], ref_step2["grad_norm:mean"], places=6
        )
        self.assertAlmostEqual(wire.output_json(h_loss3.body)["metrics"]["loss:sum"], ref_loss3["loss:sum"], places=6)
        self.assertAlmostEqual(
          json.loads(h_step3.body)["metrics"]["grad_norm:mean"], ref_step3["grad_norm:mean"], places=6
        )

        # Part B: Fresh client session reconstructs from explicit checkpoint via /api/v1/load_weights(optimizer=True).
        sess_id = (await client.post("/api/v1/create_session", json={})).json()["session_id"]
        load_res = (
          await client.post(
            "/api/v1/load_weights",
            json={"session_id": sess_id, "model_seq_id": 0, "path": ckpt_uri, "optimizer": True},
          )
        ).json()
        restored_id = load_res["model_id"]
        await node2.add(await node2.worker_for(restored_id))
        self.assertTrue((await self.bus.wait_for_reply(load_res["request_id"], 5)).ok)

        r_fb2 = await self.bus.wait_for_reply(await self.bus.submit_command(restored_id, Op.FORWARD_BACKWARD, b2), 5)
        r_step2 = await self.bus.wait_for_reply(await self.bus.submit_command(restored_id, Op.OPTIM_STEP, adam), 5)
        r_loss3 = await self.bus.wait_for_reply(await self.bus.submit_command(restored_id, Op.FORWARD_BACKWARD, b3), 5)
        r_step3 = await self.bus.wait_for_reply(await self.bus.submit_command(restored_id, Op.OPTIM_STEP, adam), 5)
        self.assertTrue(r_fb2.ok and r_step2.ok and r_loss3.ok and r_step3.ok)

        self.assertAlmostEqual(
          json.loads(r_step2.body)["metrics"]["grad_norm:mean"], ref_step2["grad_norm:mean"], places=6
        )
        self.assertAlmostEqual(wire.output_json(r_loss3.body)["metrics"]["loss:sum"], ref_loss3["loss:sum"], places=6)
        self.assertAlmostEqual(
          json.loads(r_step3.body)["metrics"]["grad_norm:mean"], ref_step3["grad_norm:mean"], places=6
        )

        restored_w = await node2.worker_for(restored_id)
        restored_state, _ = node2.processes[restored_w].trainer.model.training_state(restored_id)
        for k in ref_state:
          if k.startswith("rng."):
            self.assertTrue(torch.equal(restored_state[k], ref_state[k]), f"RNG mismatch on {k}")
          else:
            self.assertTrue(torch.allclose(restored_state[k], ref_state[k], atol=1e-6), f"tensor mismatch on {k}")
      finally:
        await node2.aclose()

  def test_two_rank_fsdp_host_parked_save_and_restore_without_gpu(self):
    """2-rank CPU/Gloo FSDP full-FT parks on host, serializes full_tensor() without GPU, and restores bit-for-bit."""
    require_unix()
    with tempfile.TemporaryDirectory() as folder:
      base = f"{folder}/tiny"
      tiny().save_pretrained(base)
      script = f"{folder}/run_fsdp_park.py"
      with open(script, "w") as f:
        f.write(
          """
import os, sys, numpy as np, torch
import torch.distributed as dist
from tinyrl.api.wire import Datum
from tinyrl.storage.store import Store
from tinyrl.workers.model import Model
from test.helpers import tiny

folder, base = sys.argv[1], sys.argv[2]
dist.init_process_group("gloo")
rank = dist.get_rank()
store = Store(f"{folder}/ckpt")
m = Model(tiny)
m.create_run("r1", base, 0, {})
batch = [Datum(np.arange(1, 16, dtype=np.int32), {"target_tokens": np.arange(2, 17, dtype=np.int64), "weights": np.ones(15, dtype=np.float32)})]
m.forward_backward("r1", batch, "cross_entropy", {}, False)
m.optim_step("r1", {"learning_rate": 1e-3})
m.forward_backward("r1", batch, "cross_entropy", {}, False)  # accumulated grad before parking
m.park_on_host()
assert m.parked is True
tensors, meta = m.training_state("r1")
if rank == 0:
  store.put("tinker://r1/weights/parked", tensors, meta)
dist.barrier()

# Continue on original after resuming from host
m.resume_on_gpu()
s_orig = m.optim_step("r1", {"learning_rate": 1e-3})
_, l_orig = m.forward_backward("r1", batch, "cross_entropy", {}, False)

# Fresh 2-rank model restores directly from the parked checkpoint
m2 = Model(tiny)
m2.create_run("r2", base, 0, {})
folder_path, _ = store.local_folder("tinker://r1/weights/parked")
m2.restore_from_folder("r2", folder_path, optimizer=True, gradients=True)
s_rest = m2.optim_step("r2", {"learning_rate": 1e-3})
_, l_rest = m2.forward_backward("r2", batch, "cross_entropy", {}, False)

if rank == 0:
  assert abs(s_orig["grad_norm:mean"] - s_rest["grad_norm:mean"]) < 1e-5, (s_orig, s_rest)
  assert abs(l_orig["loss:sum"] - l_rest["loss:sum"]) < 1e-5, (l_orig, l_rest)
dist.destroy_process_group()
"""
        )
      res = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", script, folder, base],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
      )
      self.assertEqual(res.returncode, 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")


class TestScenarioSuspensionAndFencing(RedisTest):
  """Scenario 3: single-job shared-worker drain, generation fencing, crash rollback, and sampling abort replay."""

  async def test_single_job_drain_from_shared_lora_worker_and_generation_fence(self):
    with tempfile.TemporaryDirectory() as folder:
      base = f"{folder}/tiny"
      tiny().save_pretrained(base)
      store = Store(f"{folder}/ckpt")
      node = TestNode(self.bus, store=store, gpus=2, node="node-1")
      tasks = [asyncio.create_task(scheduler.run(self.bus))]

      async def pump():
        while True:
          await node.tick()
          await asyncio.sleep(0.01)

      tasks.append(asyncio.create_task(pump()))
      try:
        await node.announce_node()
        a, b = new_id(), new_id()
        lora = {"rank": 4, "seed": 0, "lora_dropout": 0.0}
        for run in (a, b):
          await self.bus.create_run(Run(run, "", base, 4, lora=lora))
          req = await self.bus.submit_command(
            run, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 4, "lora": lora}).encode()
          )
          self.assertTrue((await self.bus.wait_for_reply(req, 10)).ok)
        w = (await self.bus.get_run(a))["worker"]
        self.assertEqual(w, (await self.bus.get_run(b))["worker"])
        gen_a_old = int((await self.bus.get_run(a))["gen"])
        gen_b_old = int((await self.bus.get_run(b))["gen"])

        # Step both runs once.
        for run in (a, b):
          rep = await self.bus.wait_for_reply(
            await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}'), 10
          )
          self.assertTrue(rep.ok)

        # Trigger single-job drain of job A while job B keeps running on worker w!
        # Pause scheduler briefly while injecting the drain command so a concurrent write_table cannot overwrite it.
        tasks[0].cancel()
        await asyncio.gather(tasks[0], return_exceptions=True)
        raw = json.loads(await self.bus.r.hget(key("workers"), w))
        raw["drain_jobs"] = {a: ""}
        raw["rev"] = int(raw.get("rev", 0)) + 1
        await self.bus.r.hset(key("workers"), w, json.dumps(raw))
        tasks[0] = asyncio.create_task(scheduler.run(self.bus))
        await self.bus.wake(f"node:{node.id}", "scheduler")

        # Job B continues executing on the same worker uninterrupted.
        rep_b = await self.bus.wait_for_reply(
          await self.bus.submit_command(b, Op.OPTIM_STEP, b'{"adam_params":{}}'), 10
        )
        self.assertTrue(rep_b.ok)

        # Wait until scheduler reassigns A to a new worker with bumped generation.
        async with asyncio.timeout(10):
          while (await self.bus.get_run(a))["worker"] == w:
            await asyncio.sleep(0.02)
        run_a_now, run_b_now = await self.bus.get_run(a), await self.bus.get_run(b)
        self.assertNotEqual(run_a_now["worker"], w)
        self.assertEqual(run_b_now["worker"], w)
        self.assertEqual(int(run_a_now["gen"]), gen_a_old + 1)
        self.assertEqual(int(run_b_now["gen"]), gen_b_old)

        # Verify stale generation CANNOT commit state or publish a reply for A.
        stale_commit = await self.bus.commit_job_state(
          a, "tinker://stale/weights/x", "v-stale", "999-0", durable=True, worker=w, owner="old", generation=gen_a_old
        )
        self.assertFalse(stale_commit)
        # Job A seamlessly restores and executes on its new worker.
        rep_a2 = await self.bus.wait_for_reply(
          await self.bus.submit_command(a, Op.OPTIM_STEP, b'{"adam_params":{}}'), 10
        )
        self.assertTrue(rep_a2.ok, rep_a2.body)
        self.assertEqual((await self.bus.get_run(a))["steps"], 2)
        self.assertEqual((await self.bus.get_run(b))["steps"], 2)
      finally:
        for t in tasks:
          t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await node.aclose()

  async def test_unexpected_crash_exposes_rollback_and_recovers_on_explicit_load(self):
    require_unix()
    with tempfile.TemporaryDirectory() as folder:
      base = f"{folder}/tiny"
      tiny().save_pretrained(base)
      store = Store(f"{folder}/ckpt")
      node = NodeDaemon(self.bus, gpus=1, node="node-crash", store=store, gpu_mib=0)
      tasks = [asyncio.create_task(scheduler.run(self.bus))]

      async def pump():
        while True:
          await node.tick()
          await asyncio.sleep(0.02)

      tasks.append(asyncio.create_task(pump()))
      try:
        await node.announce_node()
        run = new_id()
        lora = {"rank": 4, "seed": 0, "lora_dropout": 0.0}
        await self.bus.create_run(Run(run, "", base, 4, lora=lora))
        self.assertTrue(
          (
            await self.bus.wait_for_reply(
              await self.bus.submit_command(
                run, Op.CREATE_MODEL, json.dumps({"base_model": base, "rank": 4, "lora": lora}).encode()
              ),
              30,
            )
          ).ok
        )
        save_rep = await self.bus.wait_for_reply(
          await self.bus.submit_command(
            run, Op.SAVE_WEIGHTS, json.dumps({"uri": f"tinker://{run}/weights/safe"}).encode()
          ),
          30,
        )
        self.assertTrue(save_rep.ok)
        safe_uri = json.loads(save_rep.body)["path"]
        # One unsaved mutation after the durable checkpoint.
        self.assertTrue(
          (
            await self.bus.wait_for_reply(await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}'), 30)
          ).ok
        )
        w = (await self.bus.get_run(run))["worker"]
        gen_before = int((await self.bus.get_run(run))["gen"])
        pid = node.running[w].pid
        os.killpg(pid, signal.SIGKILL)
        async with asyncio.timeout(15):
          while int((await self.bus.get_run(run))["gen"]) == gen_before:
            await asyncio.sleep(0.02)

        # Next mutation must expose rollback instead of silently resuming stale weights.
        fail_rep = await self.bus.wait_for_reply(
          await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}'), 30
        )
        self.assertIsNotNone(fail_rep)
        self.assertFalse(fail_rep.ok)
        self.assertIn("lost its unsaved steps", fail_rep.body.decode())
        self.assertTrue((await self.bus.get_run(run))["rollback_required"])

        # Explicit LOAD_WEIGHTS clears rollback_required and resumes cleanly.
        load_rep = await self.bus.wait_for_reply(
          await self.bus.submit_command(
            run, Op.LOAD_WEIGHTS, json.dumps({"uri": safe_uri, "optimizer": True}).encode()
          ),
          30,
        )
        self.assertTrue(load_rep.ok, load_rep.body)
        ok_rep = await self.bus.wait_for_reply(
          await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}'), 30
        )
        self.assertTrue(ok_rep.ok, ok_rep.body)
      finally:
        for t in tasks:
          t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await node.close(graceful=False)

  async def test_inflight_sampling_drain_aborts_and_replays_cleanly(self):
    in_sample, paused = asyncio.Event(), asyncio.Event()

    class PauseAbortEngine(FakeEngine):
      async def pause(self, mode="wait"):
        paused.set()

      async def sample(self, job, name):
        in_sample.set()
        await paused.wait()
        raise SamplingAbortedError("drain abort")

    store = Store(tempfile.mkdtemp())
    node1 = TestNode(self.bus, store=store, gpus=1, node="n1", engine=PauseAbortEngine)
    node2 = TestNode(self.bus, store=store, gpus=1, node="n2", engine=FakeEngine)
    try:
      await self.bus.create_run(Run("s1", "w-samp", "base", 4, role=Role.SAMPLER))
      req = await self.bus.submit_command("s1", Op.SAMPLE, json.dumps(JOB).encode())
      await node1.add("w-samp", gpus=[0])
      await asyncio.wait_for(in_sample.wait(), 5)
      # Trigger graceful drain on node1 while sample is in flight -> watch_drain calls pause("abort").
      close_task = asyncio.create_task(node1.close(graceful=True))
      await asyncio.wait_for(close_task, 5)
      self.assertTrue(paused.is_set())
      self.assertIsNone(await self.bus.wait_for_reply(req, 0))
      # Replacement node2 takes over and replays the aborted sampling command cleanly.
      await node2.add("w-samp", gpus=[0])
      rep = await self.bus.wait_for_reply(req, 5)
      self.assertIsNotNone(rep)
      self.assertTrue(rep.ok, rep.body)
    finally:
      await node1.aclose()
      await node2.aclose()


class TestScenarioFleetReplay(unittest.TestCase):
  """Scenario 4: deterministic multi-node fleet replay with wide jobs, fairness, exclusive cap, host-RAM demotion, and restarts."""

  def test_multi_node_fleet_replay_with_scheduler_restarts(self):
    nodes = {
      "n1": {
        "gpus": 4,
        "memory": 80 * GIB,
        "host_ram": 100 * GIB,
        "host_ram_used": 0,
        "peer": "10.0.0.1:8000",
        "workers": {},
        "versions": {},
      },
      "n2": {
        "gpus": 4,
        "memory": 80 * GIB,
        "host_ram": 100 * GIB,
        "host_ram_used": 0,
        "peer": "10.0.0.2:8000",
        "workers": {},
        "versions": {},
      },
      "n3": {
        "gpus": 2,
        "memory": 80 * GIB,
        "host_ram": 100 * GIB,
        "host_ram_used": 0,
        "peer": "10.0.0.3:8000",
        "workers": {},
        "versions": {},
      },
    }
    # Mixed workload: narrow trainers, paired samplers, wide 2-GPU & 4-GPU trainers, and exclusive jobs.
    jobs = {
      "narrow-1": {"role": "trainer", "base": "m1", "rank": 0, "gpus": 1},
      "narrow-2": {"role": "trainer", "base": "m2", "rank": 0, "gpus": 1},
      "narrow-3": {"role": "trainer", "base": "m3", "rank": 0, "gpus": 1},
      "narrow-1-s": {"role": "sampler", "base": "m1", "rank": 0, "gpus": 1, "train_run": "narrow-1"},
      "wide-2": {"role": "trainer", "base": "mw2", "rank": 0, "gpus": 2},
      "wide-4": {"role": "trainer", "base": "mw4", "rank": 0, "gpus": 4},
      "excl-1": {"role": "trainer", "base": "me1", "rank": 8, "gpus": 2, "exclusive": True},
    }
    remaining = {j: 120.0 for j in jobs}
    completed_usage = {j: 0.0 for j in jobs}
    s = scheduler.Assignment(exclusive_cap=0.5)
    meta = {j: {"gen": 1} for j in jobs}
    versions = {"v-wide4": {"locations": {"n2": "10.0.0.2:8000"}, "durable": True, "size_bytes": 4 * GIB}}
    meta["wide-4"] = {
      "gen": 1,
      "restore_uri": "tinker://wide-4/weights/c0",
      "restore_version": "v-wide4",
      "restore_boundary": "1-0",
    }
    nodes["n2"]["versions"]["v-wide4"] = {"ready": True, "progress": 1.0, "bytes": 4 * GIB}
    dt = 4.0  # simulated GPU-seconds per tick per GPU

    def build_table(cur: scheduler.Assignment) -> dict:
      return {
        "nodes": nodes,
        "runs": {j: (rec, cur.jobs[j].worker if j in cur.jobs else "") for j, rec in jobs.items()},
        "meta": {j: {**meta[j], "gen": cur.jobs[j].gen if j in cur.jobs else 1} for j in jobs},
        "queues": {j: {"queued": 1 if remaining[j] > 0 else 0, "in_flight": 0} for j in jobs},
        "workers": {n: scheduler.worker_to_dict(w) for n, w in cur.workers.items()},
        "workers_made": cur.workers_made,
        "table_rev": cur.table_rev,
        "service": dict(cur.service),
        "service_obs": dict(cur.service_obs),
        "service_ready": list(cur.service_ready),
        "versions": versions,
        "ckpt_versions": {},
      }

    restarts = 0
    for tick in range(180):
      # Every 10 ticks, simulate a scheduler restart reconstructing purely from serialized Redis table.
      table = build_table(s)
      rebuilt = scheduler.from_table(table)
      rebuilt.exclusive_cap = 0.5
      if tick > 0 and tick % 10 == 0:
        self.assertEqual(rebuilt.workers, s.workers)
        self.assertEqual(rebuilt.service, s.service)
        self.assertEqual(rebuilt.service_obs, s.service_obs)
        self.assertEqual(rebuilt.service_ready, s.service_ready)
        self.assertEqual(rebuilt.table_rev, s.table_rev)
        restarts += 1
      s, _ = scheduler.decide(rebuilt)

      # Invariant 1: zero physical GPU overcommit on every node & GPU.
      for node_name, ni in s.nodes.items():
        for gpu in range(ni.gpus):
          holders = [n for n, w in s.workers.items() if w.node == node_name and gpu in scheduler.gpus_still_held(s, n)]
          if len(holders) > 1:
            self.assertFalse(any(s.workers[h].spec.exclusive for h in holders))
            budgets = [scheduler.memory_budget(s, s.workers[h].spec, node_name) for h in holders]
            self.assertTrue(
              scheduler.fits(budgets, ni.memory), f"tick {tick} overcommit on {node_name}:{gpu}: {holders}"
            )

      # Invariant 2: exclusive GPU cap never exceeded.
      excl_gpus = sum(w.spec.gpus for w in s.workers.values() if w.node and w.intent != "drain" and w.spec.exclusive)
      self.assertLessEqual(excl_gpus, 5)  # 50% of 10 total GPUs

      # Simulate node execution & release acknowledgments for one tick.
      for nd in nodes.values():
        nd["host_ram_used"] = 0
      for wname, w in s.workers.items():
        if not w.node or w.node not in nodes:
          continue
        nd = nodes[w.node]
        cur = nd["workers"].get(wname, {"owner": f"{w.node}:{wname}", "pid": 100, "jobs": {}, "host_ram_bytes": 0})
        if w.intent == "run":
          cur.update(
            rev=w.rev, gpus=list(w.gpus), gpus_held=list(w.gpus), gpu_released=False, phase="running", host_ram_bytes=0
          )
          active = [j for j in scheduler.active_jobs_of(s, wname) if remaining[j] > 0]
          if active:
            step = min(remaining[active[0]], dt * len(w.gpus))
            remaining[active[0]] -= step
            completed_usage[active[0]] += step
            je = cur["jobs"].setdefault(
              active[0], {"gen": s.jobs[active[0]].gen, "usage": 0.0, "tokens": 0, "state": "active"}
            )
            je["usage"] += step
        elif w.intent == "park":
          ram = 30 * GIB if w.spec.role == Role.TRAINER and cur.get("pid") else 0
          cur.update(
            rev=w.rev,
            gpus=list(w.gpus),
            gpus_held=[],
            gpu_released=True,
            phase="parked" if cur.get("pid") else "idle",
            host_ram_bytes=ram,
          )
          nd["host_ram_used"] += ram
        elif w.intent == "drain":
          cur.update(
            rev=w.rev, owner="", pid=None, gpus=[], gpus_held=[], gpu_released=True, phase="drained", host_ram_bytes=0
          )
        nd["workers"][wname] = cur
        # Complete any requested prefetches in one tick.
        for pf in w.prefetch:
          nd["versions"][pf["version"]] = {"ready": True, "progress": 1.0, "bytes": 1 * GIB}

      if all(v <= 0 for v in remaining.values()):
        break

    self.assertGreaterEqual(restarts, 1)
    # Every job including wide 2-GPU and 4-GPU jobs finished without starvation!
    for j, rem in remaining.items():
      self.assertEqual(rem, 0.0, f"job {j} starved with {rem} remaining")
    # Storage-local wide-4 job was placed on n2 where v-wide4 lived.
    self.assertEqual(s.workers[s.jobs["wide-4"].worker].node, "n2")


if __name__ == "__main__":
  unittest.main()
