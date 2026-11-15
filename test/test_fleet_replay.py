"""Validation Scenario 4: end-to-end multi-node fleet replay over real Bus, Registry, Store, scheduler, and nodes."""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
import unittest
from unittest import mock

import httpx
import torch

from test.execution import TestNode
from test.helpers import RedisTest
from test.test_api import proto_body
from test.test_sampler import JOB as SAMPLE_JOB
from tinyrl.api import gateway
from tinyrl.bus import Op, Role, Run
from tinyrl.control import scheduler
from tinyrl.storage.registry import Registry
from tinyrl.storage.store import Store, TransferCancelled, split_version

LORA = {"rank": 4, "seed": 3, "lora_dropout": 0.0}


class TestFleetReplay(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.tmp = tempfile.TemporaryDirectory()
    self.root = f"file://{self.tmp.name}/bucket"
    self.registry = Registry(self.bus)
    self.tasks: list[asyncio.Task] = []
    self.nodes: list[TestNode] = []
    self.stores: list[Store] = []

  async def asyncTearDown(self):
    for t in reversed(self.tasks):
      t.cancel()
    await asyncio.gather(*self.tasks, return_exceptions=True)
    for n in self.nodes:
      await n.close()
    for s in self.stores:
      await asyncio.to_thread(s.close, False, 1.0)
    self.tmp.cleanup()
    await super().asyncTearDown()

  def bg(self, coro) -> asyncio.Task:
    t = asyncio.create_task(coro)
    self.tasks.append(t)
    return t

  def make_node(self, name: str, gpus: int, *, gpu_class: str = "H100", rate: float = 1.0) -> TestNode:
    store = Store(self.root, cache=f"{self.tmp.name}/{name}-cache")
    store.start()
    self.stores.append(store)
    node = TestNode(self.bus, store=store, gpus=gpus, node=name, gpu_class=gpu_class, rate=rate, auto_schedule=False)
    self.nodes.append(node)
    return node

  async def pump(self, node: TestNode, stop: asyncio.Event | None = None) -> None:
    while not (stop and stop.is_set()):
      await node.tick()
      await asyncio.sleep(0.015)

  async def call(self, run_id: str, op: Op, body, *, cost: int = 1, timeout: float = 25.0):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = await self.bus.submit_command(run_id, op, raw, cost=cost)
    self.assertIsNotNone(req)
    rep = await self.bus.wait_for_reply(req, timeout)
    self.assertIsNotNone(rep, f"timed out waiting for {run_id} {op}")
    self.assertTrue(rep.ok, rep.body)
    return rep

  async def test_alternating_rl_shared_accounts_wide_contention_and_prefetch(self):
    """Multi-node replay covering alternating train/export/sample/sandbox-wait, shared accounts, 2-GPU reservation, cold-vs-warm fairness, and prefetch/cancellation under bounded cache."""
    n1 = self.make_node("node-a", gpus=2, gpu_class="H100", rate=1.0)
    n2 = self.make_node("node-b", gpus=1, gpu_class="H100", rate=1.0)
    await n1.announce_node()
    await n2.announce_node()
    sched = self.bg(scheduler.run(self.bus))
    self.bg(self.pump(n1))
    self.bg(self.pump(n2))

    # 1. Alternating RL jobs sharing LoRA engines: job_a has 1 trainer + 2 sampler replicas; job_b has 1 trainer + 1 sampler.
    t0 = time.monotonic()
    job_a, job_b = "rl-a", "rl-b"
    samplers_a = ["rl-a-s0", "rl-a-s1"]
    samplers_b = ["rl-b-s0"]
    await self.bus.create_run(Run(job_a, "", "tiny", 4, lora=LORA))
    await self.bus.create_run(Run(job_b, "", "tiny", 4, lora=LORA))
    await asyncio.gather(
      self.call(job_a, Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA}),
      self.call(job_b, Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA}),
    )

    async def rl_loop(train_id: str, sids: list[str], rounds: int, sandbox_s: float) -> dict:
      loop_t0 = time.monotonic()
      idle_deltas: list[float] = []
      for step in range(rounds):
        await self.call(train_id, Op.FORWARD_BACKWARD, proto_body(train_id), cost=32)
        await self.call(train_id, Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}}, cost=16)
        exp_uri = f"tinker://{train_id}/sampler_weights/step-{step}"
        await self.call(train_id, Op.SAVE_WEIGHTS_FOR_SAMPLER, {"uri": exp_uri, "owned": True})
        if step == 0:
          for sid in sids:
            await self.bus.create_run(
              Run(sid, "", "tiny", 4, role=Role.SAMPLER, train_run=train_id, model_path=exp_uri)
            )
        else:
          ver, _, _, _ = await self.registry.resolve(exp_uri)
          for sid in sids:
            await self.bus.pin_sampler_weights(sid, ver)
        await asyncio.gather(*(self.call(sid, Op.SAMPLE, SAMPLE_JOB, cost=16) for sid in sids))
        # External tool/sandbox wait: account consumes zero active GPU service while idle.
        await asyncio.sleep(0.05)
        svc_before = scheduler.from_table(await self.bus.read_table()).service.get(train_id, 0.0)
        await asyncio.sleep(sandbox_s)
        svc_after = scheduler.from_table(await self.bus.read_table()).service.get(train_id, 0.0)
        idle_deltas.append(round(svc_after - svc_before, 6))
      return {"elapsed_s": round(time.monotonic() - loop_t0, 3), "idle_deltas": idle_deltas}

    res_a, res_b = await asyncio.gather(rl_loop(job_a, samplers_a, 3, 0.12), rl_loop(job_b, samplers_b, 3, 0.06))
    # Sampler replicas share their parent trainer's fair-service account (no separate replica accounts minted).
    tbl = scheduler.from_table(await self.bus.read_table())
    for sid in [*samplers_a, *samplers_b]:
      self.assertNotIn(sid, tbl.service)
      self.assertEqual(tbl.jobs[sid].account, job_a if sid in samplers_a else job_b)
    self.assertGreater(tbl.service.get(job_a, 0.0), 0.0)
    self.assertGreater(tbl.service.get(job_b, 0.0), 0.0)
    # Idle sandbox waits never accrue unbounded runaway service.
    self.assertEqual(min(res_a["idle_deltas"]), 0.0)
    # While uncontended after finishing, resident trainer/sampler switches to intent="idle" (opportunistic warm residency).
    await asyncio.sleep(0.08)
    tbl_idle = scheduler.from_table(await self.bus.read_table())
    idle_warm = [
      name
      for name, w in tbl_idle.workers.items()
      if w.intent == "idle" and w.gpus and (o := tbl_idle.observations.get(name)) and o.gpu_state == "active"
    ]
    self.assertTrue(idle_warm)

    # Returning idle job gets zero idle credit: advance job_b while job_a sits idle, then submit on job_a -> lifted to floor!
    for _ in range(3):
      await self.call(job_b, Op.FORWARD_BACKWARD, proto_body(job_b), cost=64)
      await self.call(job_b, Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}}, cost=32)
    b_svc = scheduler.from_table(await self.bus.read_table()).service[job_b]
    await self.call(job_a, Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}}, cost=16)
    a_svc = scheduler.from_table(await self.bus.read_table()).service[job_a]
    self.assertGreaterEqual(a_svc, b_svc - 1e-4)

    # 2. Wide 2-GPU full-FT contending with two steady 1-GPU full-FT holders on node-a (2 GPUs).
    s1, s2, wide = "fft-1a", "fft-1b", "fft-wide-2gpu"
    for r in (s1, s2):
      await self.bus.create_run(Run(r, "", f"tiny-{r}", 0, gpus=1))
      await self.call(r, Op.CREATE_MODEL, {"base_model": f"tiny-{r}", "rank": 0, "lora": {}})
    # Submit a steady stream of 1-GPU commands concurrently with the arriving 2-GPU job.
    steady_futs = [
      asyncio.create_task(self.call(r, Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-3}}, cost=32))
      for _ in range(3)
      for r in (s1, s2)
    ]
    wide_t0 = time.monotonic()
    await self.bus.create_run(Run(wide, "", "tiny-wide", 0, gpus=2))
    await self.call(wide, Op.CREATE_MODEL, {"base_model": "tiny-wide", "rank": 0, "lora": {}})
    await self.call(wide, Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-3}}, cost=64)
    wide_wait_s = round(time.monotonic() - wide_t0, 3)
    await asyncio.gather(*steady_futs)
    wide_w = (await self.bus.get_run(wide))["worker"]
    self.assertEqual(scheduler.from_table(await self.bus.read_table()).workers[wide_w].node, "node-a")
    self.assertEqual((await self.bus.get_run(wide))["steps"], 1)
    self.assertEqual((await self.bus.get_run(s1))["steps"], 3)
    self.assertEqual((await self.bus.get_run(s2))["steps"], 3)

    # 3. Overlapped storage prefetch, cold-job fairness, obsolete-prefetch cancellation, and bounded cache trim safety.
    cold_uri = "tinker://seed/sampler_weights/cold"
    big = {"w": torch.arange(256 * 1024, dtype=torch.float32)}  # 1 MiB
    cold_v1, _ = split_version(
      n1.store.put(cold_uri, big, {"base": "tiny-cold", "lora": True, "rank": 4, "config": {}})
    )
    await asyncio.to_thread(n1.store.upload_version, cold_uri, cold_v1)
    await self.registry.register_checkpoint(
      cold_uri, 256 * 1024 * 4, version=cold_v1, durable=True, location=n1.store.peer
    )

    # Cold sampler pinned to cold_v1 migrated onto node-b pulls cold_v1 via peer/bucket and finishes without starvation.
    await self.bus.create_run(
      Run("cold-s", "", "tiny-cold", 4, role=Role.SAMPLER, model_path=cold_uri, lora={"weights_version": cold_v1})
    )
    await self.bus.request_job_migration("cold-s", "node-b")
    cold_t0 = time.monotonic()
    await asyncio.gather(
      self.call("cold-s", Op.SAMPLE, SAMPLE_JOB, cost=32),
      self.call(job_b, Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-3}}, cost=32),
    )
    cold_s = round(time.monotonic() - cold_t0, 3)
    self.assertTrue(n2.store.has_local(cold_uri, cold_v1))

    # Obsolete prefetch cancellation records cancelled_bytes cleanly and leaves zero corrupt partial versions.
    obs_uri = "tinker://seed/sampler_weights/obsolete"
    obs_v, _ = split_version(
      n1.store.put(
        obs_uri,
        {"w": torch.zeros(1024 * 1024, dtype=torch.float32)},
        {"base": "tiny-cold", "lora": True, "rank": 4, "config": {}},
      )
    )
    await asyncio.to_thread(n1.store.upload_version, obs_uri, obs_v)
    cancel_ev = threading.Event()
    orig_chunk = n2.store._on_chunk

    def cancel_after_first_chunk(tr, cancel, n):
      orig_chunk(tr, cancel, n)
      if tr.version == obs_v and tr.bytes_done > 0:
        cancel_ev.set()

    n2.store._on_chunk = cancel_after_first_chunk
    try:
      with self.assertRaises(TransferCancelled):
        await asyncio.to_thread(n2.store.local_folder, obs_uri, obs_v, (), cancel_ev, "prefetch")
    finally:
      n2.store._on_chunk = orig_chunk
    self.assertFalse(n2.store.has_local(obs_uri, obs_v))

    # Bounded cache trim retains pinned active version while evicting unpinned durable versions.
    extra_v, _ = split_version(
      n1.store.put(
        "tinker://seed/sampler_weights/extra", big, {"base": "tiny-cold", "lora": True, "rank": 4, "config": {}}
      )
    )
    await asyncio.to_thread(n1.store.upload_version, "tinker://seed/sampler_weights/extra", extra_v)
    await asyncio.to_thread(n2.store.local_folder, "tinker://seed/sampler_weights/extra", extra_v)
    with n2.store.pin(cold_v1):
      n2.store.trim_cache(limit=1024)
    self.assertTrue(n2.store.has_local(cold_uri, cold_v1))
    self.assertFalse(n2.store.has_local("tinker://seed/sampler_weights/extra", extra_v))

    c1, c2 = n1.store.cache_status(), n2.store.cache_status()
    transferred = c1["transferred_bytes"] + c2["transferred_bytes"]
    cancelled = c1["cancelled_bytes"] + c2["cancelled_bytes"]
    self.assertGreater(transferred, 0)
    self.assertGreater(cancelled, 0)

    total_s = round(time.monotonic() - t0, 3)
    sched.cancel()
    print(
      f"\n[fleet-replay:1] total_s={total_s} rl_a_s={res_a['elapsed_s']} rl_b_s={res_b['elapsed_s']} "
      f"svc_a={round(a_svc, 4)} svc_b={round(b_svc, 4)} wide_2gpu_s={wide_wait_s} cold_s={cold_s} "
      f"transferred_bytes={transferred} cancelled_bytes={cancelled}",
      flush=True,
    )

  async def test_exclusive_cap_scheduler_outage_live_restart_and_node_loss_recovery(self):
    """Replay bounded exclusive admission + gateway rejection, bounded execution during scheduler outage, live scheduler restart, and mid-run node loss."""
    with mock.patch.object(scheduler, "EXCLUSIVE_MAX_GPUS", 1):
      n1 = self.make_node("ex-n1", gpus=2, gpu_class="H100")
      n2 = self.make_node("ex-n2", gpus=1, gpu_class="H100")
      stop_n2 = asyncio.Event()
      await n1.announce_node()
      await n2.announce_node()
      sched1 = self.bg(scheduler.run(self.bus))
      self.bg(self.pump(n1))
      pump_n2 = self.bg(self.pump(n2, stop_n2))

      # 1. Gateway rejects infeasible exclusive width / unknown GPU class with HTTP 400.
      async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway.app(self.bus, n1.store)), base_url="http://tinyrl"
      ) as client:
        sid = (await client.post("/api/v1/create_session", json={"tags": [], "sdk_version": "0.1"})).json()[
          "session_id"
        ]
        bad_w = await client.post(
          "/api/v1/create_model",
          json={
            "session_id": sid,
            "base_model": "tiny",
            "user_metadata": {"openrl.exclusive": "true", "openrl.gpus": "8"},
          },
        )
        self.assertEqual(bad_w.status_code, 400)
        bad_cls = await client.post(
          "/api/v1/create_model",
          json={"session_id": sid, "base_model": "tiny", "user_metadata": {"openrl.gpu_classes": "B200"}},
        )
        self.assertEqual(bad_cls.status_code, 400)

      # 2. Bounded exclusive admission: EXCLUSIVE_MAX_GPUS=1 admits ex1 immediately on an isolated GPU while ex2 queues.
      await self.bus.create_run(Run("ex1", "", "tiny-ex1", 0, exclusive=True, gpus=1))
      await self.bus.create_run(Run("ex2", "", "tiny-ex2", 0, exclusive=True, gpus=1))
      await self.bus.create_run(Run("shared", "", "tiny-sh", 4, lora=LORA))
      await self.call("ex1", Op.CREATE_MODEL, {"base_model": "tiny-ex1", "rank": 0, "lora": {}})
      await self.call("shared", Op.CREATE_MODEL, {"base_model": "tiny-sh", "rank": 4, "lora": LORA})
      ex2_create = await self.bus.submit_command(
        "ex2", Op.CREATE_MODEL, b'{"base_model":"tiny-ex2","rank":0,"lora":{}}'
      )
      await asyncio.sleep(0.12)
      st = scheduler.from_table(await self.bus.read_table())
      w_ex1, w_ex2, w_sh = st.jobs["ex1"].worker, st.jobs["ex2"].worker, st.jobs["shared"].worker
      self.assertIsNotNone(st.workers[w_ex1].node)
      self.assertIsNone(st.workers[w_ex2].node)  # queued cleanly waiting for exclusive quota
      if st.workers[w_sh].node == st.workers[w_ex1].node:
        self.assertFalse(set(st.workers[w_sh].gpus) & set(st.workers[w_ex1].gpus))  # exclusive GPU never co-occupied!
      # Delete ex1 -> ex2 is immediately admitted onto a dedicated GPU and finishes!
      await self.call("ex1", Op.DELETE_MODEL, {})
      ex2_rep = await self.bus.wait_for_reply(ex2_create, 15)
      self.assertIsNotNone(ex2_rep)
      self.assertTrue(ex2_rep.ok, ex2_rep.body)
      await self.call("ex2", Op.DELETE_MODEL, {})

      # 3. Scheduler loss & live restart: pause scheduler, push > TURN budget of work -> worker stops at TURN boundary!
      sched1.cancel()
      await asyncio.gather(sched1, return_exceptions=True)
      await self.bus.release_scheduler("any")
      # Set a tight explicit remaining budget of 2 commands on shared's worker while scheduler is offline.
      snap = scheduler.from_table(await self.bus.read_table())
      ws = snap.workers[w_sh]
      ws.budgets, ws.intent, ws.drain_mode = {"shared": 2}, "run", ""
      ws.gpus = ws.gpus or (0,)
      ws.revision += 1
      await self.bus.write_table({}, {w_sh: scheduler.worker_to_dict(ws)}, [], snap.workers_made, [])
      burst = [
        await self.bus.submit_command("shared", Op.OPTIM_STEP, b'{"adam_params":{"learning_rate":0.01}}', cost=1)
        for _ in range(5)
      ]
      await asyncio.sleep(0.2)
      self.assertEqual((await self.bus.get_run("shared"))["steps"], 2)  # strictly bounded while scheduler is down!
      self.assertEqual((await self.bus.queued_and_in_flight("shared"))["queued"], 3)

      # Bring up a fresh scheduler instance (new epoch): reconciles live worker & drains the remaining 3 steps with zero loss!
      restart_t0 = time.monotonic()
      sched2 = self.bg(scheduler.run(self.bus))
      for req in burst:
        self.assertTrue((await self.bus.wait_for_reply(req, 15)).ok)
      restart_s = round(time.monotonic() - restart_t0, 3)
      self.assertEqual((await self.bus.get_run("shared"))["steps"], 5)

      # 4. Mid-run node loss: put survivor job on ex-n2, commit durable save, drop ex-n2 -> migrates & continues on ex-n1!
      await self.bus.create_run(Run("survivor", "", "tiny-surv", 4, lora=LORA))
      await self.bus.request_job_migration("survivor", "ex-n2")
      await self.call("survivor", Op.CREATE_MODEL, {"base_model": "tiny-surv", "rank": 4, "lora": LORA})
      await self.call("survivor", Op.FORWARD_BACKWARD, proto_body("survivor"), cost=16)
      await self.call("survivor", Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}}, cost=16)
      save_uri = "tinker://survivor/weights/durable-0"
      await self.call("survivor", Op.SAVE_WEIGHTS, {"uri": save_uri})
      gen_before = await self.bus.get_generation("survivor")
      self.assertEqual(
        scheduler.from_table(await self.bus.read_table()).workers[(await self.bus.get_run("survivor"))["worker"]].node,
        "ex-n2",
      )

      loss_t0 = time.monotonic()
      stop_n2.set()
      await asyncio.gather(pump_n2, return_exceptions=True)
      await n2.close()
      self.nodes.remove(n2)
      await self.bus.r.hdel("tinyrl:nodes", "ex-n2")
      await self.call("survivor", Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}}, cost=16)
      loss_recovery_s = round(time.monotonic() - loss_t0, 3)
      gen_after = await self.bus.get_generation("survivor")
      self.assertGreater(gen_after, gen_before)
      self.assertEqual(
        scheduler.from_table(await self.bus.read_table()).workers[(await self.bus.get_run("survivor"))["worker"]].node,
        "ex-n1",
      )
      self.assertEqual((await self.bus.get_run("survivor"))["steps"], 2)
      sched2.cancel()
      print(
        f"\n[fleet-replay:2] scheduler_restart_s={restart_s} node_loss_recovery_s={loss_recovery_s} "
        f"gen_transition={gen_before}->{gen_after}",
        flush=True,
      )


if __name__ == "__main__":
  unittest.main()
