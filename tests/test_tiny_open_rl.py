"""End-to-end test and benchmark suite for Tiny-Open-RL."""

from __future__ import annotations

import asyncio
import json
import unittest

import httpx

from tiny_open_rl.bus import RuntimeBus
from tiny_open_rl.codec import (
  PROTO_CONTENT_TYPE,
  build_forward_backward_proto,
  decode_forward_backward_output_proto,
  decode_forward_backward_proto,
)
from tiny_open_rl.gateway import create_app
from tiny_open_rl.trainer import TrainerRunner


class TinyOpenRLTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    self.bus = RuntimeBus()
    self.app = create_app(self.bus)
    self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")

  async def asyncTearDown(self) -> None:
    await self.client.aclose()

  async def test_e2e_protobuf_training_loop_decreases_loss(self) -> None:
    sess = (await self.client.post("/api/v1/create_session", json={"user_metadata": {"recipe": "gsm8k", "api_key": "sk-secret"}})).json()
    session_id = sess["session_id"]

    created = (await self.client.post("/api/v1/create_model", json={"base_model": "tiny-gpt", "session_id": session_id})).json()
    run_id = created["model_id"]
    pool_id = created["pool_id"]

    runner = TrainerRunner(self.bus, pool_id=pool_id, worker_id="trainer-0")
    # Process create_model
    await runner.step_once()
    create_res = (await self.client.post("/api/v1/retrieve_future", json={"request_id": run_id})).json()
    self.assertEqual(create_res["type"], "model_created")

    losses_over_steps: list[float] = []
    sequences = [
      ([10, 20, 30, 40], [20, 30, 40, 50], [1.0, 1.0, 1.0, 1.0]),
      ([15, 25, 35, 45], [25, 35, 45, 55], [1.0, 1.0, 1.0, 1.0]),
    ]

    for step in range(6):
      raw_proto = build_forward_backward_proto(run_id, sequences, seq_id=step + 1)
      fb_resp = (
        await self.client.post(
          "/api/v1/forward_backward",
          content=raw_proto,
          headers={"Content-Type": PROTO_CONTENT_TYPE},
        )
      ).json()
      opt_resp = (
        await self.client.post(
          "/api/v1/optim_step",
          json={"model_id": run_id, "session_id": session_id, "adam_params": {"learning_rate": 0.05}},
        )
      ).json()

      await runner.step_once()

      # Retrieve forward_backward result as raw binary Protobuf (zero-encode gateway pass-through)
      fb_future = await self.client.post(
        "/api/v1/retrieve_future",
        json={"request_id": fb_resp["request_id"]},
        headers={"Accept": PROTO_CONTENT_TYPE},
      )
      self.assertEqual(fb_future.headers["content-type"], PROTO_CONTENT_TYPE)
      decoded = decode_forward_backward_output_proto(fb_future.content)
      losses_over_steps.append(decoded["metrics"]["loss:sum"])
      self.assertEqual(len(decoded["loss_fn_outputs"]), 2)
      self.assertEqual(len(decoded["loss_fn_outputs"][0]["logprobs"]), 4)

      opt_future = (await self.client.post("/api/v1/retrieve_future", json={"request_id": opt_resp["request_id"]})).json()
      self.assertEqual(opt_future["type"], "optim_step_completed")

    # Loss strictly decreases over the 6 optimization steps
    self.assertLess(losses_over_steps[-1], losses_over_steps[0] * 0.5)

    # Inspect by run_id, by single-run session_id, and via /api/v1/models/{id}/status alias
    by_run = (await self.client.get(f"/api/v1/inspect/runs/{run_id}/summary")).json()
    by_sess = (await self.client.get(f"/api/v1/inspect/runs/{session_id}/summary")).json()
    by_status = (await self.client.get(f"/api/v1/models/{run_id}/status")).json()
    self.assertEqual((by_run["steps"], by_run["queue"]["queued"], by_run["queue"]["in_flight"]), (6, 0, []))
    self.assertEqual(by_sess["run_id"], run_id)
    self.assertEqual(by_status["steps"], 6)

  async def test_in_flight_xpending_and_queued_lag_without_external_ledger(self) -> None:
    created = (await self.client.post("/api/v1/create_model", json={"base_model": "tiny-gpt"})).json()
    run_id, pool_id = created["model_id"], created["pool_id"]
    runner = TrainerRunner(self.bus, pool_id=pool_id, worker_id="trainer-pel")
    await runner.step_once()  # drain create_model

    # Enqueue 3 requests without running the trainer yet
    reqs = []
    for i in range(3):
      raw = build_forward_backward_proto(run_id, [([1, 2, 3], [2, 3, 4], [1.0, 1.0, 1.0])], seq_id=i + 1)
      r = (await self.client.post("/api/v1/forward_backward", content=raw, headers={"Content-Type": PROTO_CONTENT_TYPE})).json()
      reqs.append(r["request_id"])

    before = (await self.client.get(f"/api/v1/inspect/runs/{run_id}/summary")).json()["queue"]
    self.assertEqual((before["queued"], before["in_flight"]), (3, []))

    # Worker claims 1 entry into the Redis Stream PEL (simulating mid-GPU execution)
    claimed = await self.bus.read_batch(pool_id, "trainer-pel", max_count=1)
    self.assertEqual(len(claimed), 1)

    during = (await self.client.get(f"/api/v1/inspect/runs/{run_id}/summary")).json()["queue"]
    self.assertEqual(during["queued"], 2)
    self.assertEqual(len(during["in_flight"]), 1)
    self.assertEqual((during["in_flight"][0]["req_id"], during["in_flight"][0]["worker_id"]), (reqs[0], "trainer-pel"))

  async def test_worker_crash_recovery_via_xautoclaim_fails_orphans_fast(self) -> None:
    created = (await self.client.post("/api/v1/create_model", json={"base_model": "tiny-gpt"})).json()
    run_id, pool_id = created["model_id"], created["pool_id"]
    runner_1 = TrainerRunner(self.bus, pool_id=pool_id, worker_id="pod-dead")
    await runner_1.step_once()  # drain create_model

    raw = build_forward_backward_proto(run_id, [([1, 2], [2, 3], [1.0, 1.0])], seq_id=1)
    fb = (await self.client.post("/api/v1/forward_backward", content=raw, headers={"Content-Type": PROTO_CONTENT_TYPE})).json()

    # pod-dead claims the batch into its PEL, then crashes before completing it
    claimed = await self.bus.read_batch(pool_id, "pod-dead", max_count=10)
    self.assertEqual(len(claimed), 1)

    # Replacement pod-new starts up and runs claim_orphans()
    runner_2 = TrainerRunner(self.bus, pool_id=pool_id, worker_id="pod-new")
    failed = await runner_2.claim_orphans(min_idle_ms=0)
    self.assertEqual(failed, 1)

    # Client waiting on retrieve_future immediately receives RequestFailedResponse instead of hanging
    reply = (await self.client.post("/api/v1/retrieve_future", json={"request_id": fb["request_id"], "timeout": 1.0})).json()
    self.assertEqual(reply["type"], "RequestFailedResponse")
    self.assertIn("pod-new restarted", reply["error_message"])

    after = (await self.client.get(f"/api/v1/inspect/runs/{run_id}/summary")).json()["queue"]
    self.assertEqual((after["queued"], after["in_flight"]), (0, []))

  async def test_blmove_future_wakes_immediately_and_is_idempotent_on_retries(self) -> None:
    req_id = "req-blmove-test"

    async def delayed_writer() -> None:
      await asyncio.sleep(0.03)
      await self.bus.set_result(req_id, {"type": "optim_step_completed", "step": 1}, wire="json")

    writer_task = asyncio.create_task(delayed_writer())
    # First read blocks on BLMOVE until delayed_writer pushes the result
    first = (await self.client.post("/api/v1/retrieve_future", json={"request_id": req_id, "timeout": 2.0})).json()
    await writer_task
    # Subsequent retries read the exact same result without blocking or consuming it
    second = (await self.client.post("/api/v1/retrieve_future", json={"request_id": req_id, "timeout": 0.1})).json()
    third = (await self.client.post("/api/v1/retrieve_future", json={"request_id": req_id, "timeout": 0.1})).json()
    self.assertEqual(first, {"type": "optim_step_completed", "step": 1})
    self.assertEqual(second, first)
    self.assertEqual(third, first)

  async def test_shared_lora_pool_round_robin_without_active_tenants(self) -> None:
    m1 = (await self.client.post("/api/v1/create_model", json={"base_model": "Qwen/Qwen3-8B"})).json()
    m2 = (await self.client.post("/api/v1/create_model", json={"base_model": "Qwen/Qwen3-8B"})).json()
    self.assertEqual(m1["pool_id"], m2["pool_id"])
    pool_id = m1["pool_id"]

    # Round-robin across the two runs in the shared LoRA pool
    b1 = await self.bus.read_batch(pool_id, "lora-worker", max_count=10)
    b2 = await self.bus.read_batch(pool_id, "lora-worker", max_count=10)
    self.assertEqual({b1[0].run_id, b2[0].run_id}, {m1["model_id"], m2["model_id"]})

  async def test_lock_free_session_concurrency_and_multi_run_409_resolution(self) -> None:
    sess = (await self.client.post("/api/v1/create_session", json={"user_metadata": {"owner": "shuby", "api_ token": "secret"}})).json()
    sid = sess["session_id"]

    m1 = (await self.client.post("/api/v1/create_model", json={"base_model": "tiny-gpt", "session_id": sid})).json()["model_id"]
    m2 = (await self.client.post("/api/v1/create_model", json={"base_model": "tiny-gpt", "session_id": sid})).json()["model_id"]

    # Race 10 concurrent heartbeats and attaches with zero asyncio.Lock
    await asyncio.gather(
      *(self.bus.upsert_session(sid) for _ in range(10)),
      self.bus.upsert_session(sid, run_id=m1),
      self.bus.upsert_session(sid, run_id=m2),
    )

    sess_summary = (await self.client.get(f"/api/v1/inspect/sessions/{sid}/summary")).json()
    self.assertEqual(sorted(sess_summary["model_ids"]), sorted([m1, m2]))
    self.assertNotIn("secret", json.dumps(sess_summary))

    # Calling /runs/{session_id}/summary when the session has 2 runs returns HTTP 409 listing both
    conflict = await self.client.get(f"/api/v1/inspect/runs/{sid}/summary")
    self.assertEqual(conflict.status_code, 409)
    self.assertIn(m1, conflict.json()["detail"])
    self.assertIn(m2, conflict.json()["detail"])
    self.assertEqual((await self.client.get("/api/v1/inspect/runs/unknown-id/summary")).status_code, 404)

  async def test_protobuf_wire_size_vs_json_list_encoding(self) -> None:
    inp_tokens = [100000 + i for i in range(256)]
    tgt_tokens = [100001 + i for i in range(256)]
    weights = [-1.23456789 + i * 0.001234567 for i in range(256)]
    sequences = [(inp_tokens, tgt_tokens, weights) for _ in range(16)]
    raw_proto = build_forward_backward_proto("run-bench", sequences)
    legacy_json = json.dumps(
      {
        "model_id": "run-bench",
        "seq_id": 1,
        "forward_backward_input": {
          "loss_fn": "cross_entropy",
          "data": [
            {
              "model_input": {"chunks": [{"type": "encoded_text", "tokens": inp}]},
              "loss_fn_inputs": {
                "target_tokens": {"data": tgt, "dtype": "int64", "shape": [len(tgt)]},
                "weights": {"data": wts, "dtype": "float32", "shape": [len(wts)]},
              },
            }
            for inp, tgt, wts in sequences
          ],
        },
      }
    ).encode("utf-8")

    # Protobuf binary representation is >2x smaller than JSON lists in Redis
    self.assertLess(len(raw_proto) * 2.0, len(legacy_json))
    decoded = decode_forward_backward_proto(raw_proto)
    self.assertEqual(len(decoded.data), 16)
    self.assertEqual(tuple(decoded.data[0].model_input.shape), (256,))


if __name__ == "__main__":
  unittest.main()
