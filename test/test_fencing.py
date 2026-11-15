"""Scenario 3 + PR 6 gate: concurrent generation fencing, single-job migration from shared workers, crash rollback, multi-rank failure, verified GPU release, and compiled-artifact startup contract."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest

import torch

from test.execution import TestNode
from test.helpers import FakeEngine, RedisTest, new_id, tiny
from test.test_api import proto_body
from test.test_process import require_unix
from test.test_sampler import JOB
from tinyrl.bus import Cmd, Op, Reply, Role, Run
from tinyrl.control import scheduler
from tinyrl.control.node_daemon import NodeDaemon
from tinyrl.control.process import Process, gpu_pids_clear
from tinyrl.storage.store import Store, split_version
from tinyrl.workers import vllm

LORA = {"rank": 4, "seed": 3, "lora_dropout": 0.0}


class TestFencingAndRecovery(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.nodes, self.tasks, self.procs = [], [], []
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)

  async def asyncTearDown(self):
    for task in self.tasks:
      task.cancel()
    await asyncio.gather(*self.tasks, return_exceptions=True)
    for node in self.nodes:
      await node.close(graceful=False)
    for proc in self.procs:
      if proc.poll() is None:
        proc.kill()
      await asyncio.to_thread(proc.wait)
    await super().asyncTearDown()

  def bg(self, coro):
    t = asyncio.create_task(coro)
    self.tasks.append(t)
    return t

  async def until(self, pred, timeout: float = 30.0):
    async with asyncio.timeout(timeout):
      while True:
        res = pred()
        if (await res) if asyncio.iscoroutine(res) else res:
          return
        await asyncio.sleep(0.02)

  async def test_shared_trainer_single_job_migration_fences_delayed_stale_commit_and_reply(self):
    """Job A moves out of a shared trainer while job B stays active; stale generation g is fenced on state, claims and replies."""
    root = f"file://{self.tmp.name}/bucket"
    s1 = Store(root, cache=f"{self.tmp.name}/c1")
    s2 = Store(root, cache=f"{self.tmp.name}/c2")
    s1.start()
    s2.start()
    n1 = TestNode(self.bus, store=s1, gpus=1, node="n1", auto_schedule=False)
    n2 = TestNode(self.bus, store=s2, gpus=1, node="n2", auto_schedule=False)
    self.nodes += [n1, n2]
    await n1.announce_node()
    await n2.announce_node()
    sched = self.bg(scheduler.run(self.bus))
    self.bg(self._pump(n1))
    self.bg(self._pump(n2))

    a, b = new_id(), new_id()
    await self.bus.create_run(Run(a, "", "tiny", 4, lora=LORA))
    await self.bus.create_run(Run(b, "", "tiny", 4, lora=LORA))
    for r in (a, b):
      req = await self.bus.submit_command(
        r, Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode()
      )
      self.assertTrue((await self.bus.wait_for_reply(req, 10)).ok)
    src_worker = (await self.bus.get_run(a))["worker"]
    self.assertEqual((await self.bus.get_run(b))["worker"], src_worker)
    gen_a1 = await self.bus.get_generation(a)
    gen_b1 = await self.bus.get_generation(b)
    src_node = n1 if src_worker in n1.running else n2
    dst_node = n2 if src_node is n1 else n1
    src_owner = await self.bus.lease_owner(src_worker)

    # Train one step on A and B on the shared source trainer.
    for r in (a, b):
      fb = await self.bus.submit_command(r, Op.FORWARD_BACKWARD, proto_body(r))
      op = await self.bus.submit_command(r, Op.OPTIM_STEP, b'{"adam_params":{"learning_rate":0.01}}')
      self.assertTrue((await self.bus.wait_for_reply(fb, 10)).ok)
      self.assertTrue((await self.bus.wait_for_reply(op, 10)).ok)

    # Request single-job migration of A to dst_node while B continues uninterrupted on src_worker!
    self.assertTrue(await self.bus.request_job_migration(a, dst_node.id))
    b_during = await self.bus.submit_command(b, Op.OPTIM_STEP, b'{"adam_params":{"learning_rate":0.01}}')
    await self.until(lambda: self._migrated(a, src_worker, dst_node.id))
    self.assertTrue((await self.bus.wait_for_reply(b_during, 10)).ok)
    self.assertEqual((await self.bus.get_run(b))["worker"], src_worker)
    self.assertEqual(await self.bus.get_generation(b), gen_b1)
    gen_a2 = await self.bus.get_generation(a)
    self.assertGreater(gen_a2, gen_a1)
    # src_worker STILL holds a valid worker lease for B, proving job fencing is independent of worker lease!
    self.assertEqual(await self.bus.lease_owner(src_worker), src_owner)

    # Hold a delayed stale snapshot & reply from src_worker under old generation gen_a1 behind a barrier, then release:
    stale_uri = f"tinker://{a}/weights/.stale"
    stale_raw = src_node.store.put(stale_uri, {"stale": torch.zeros(4)}, {"optimizer": True, "base": "tiny"})
    stale_ver, stale_loc = split_version(stale_raw)
    os.utime(f"{src_node.store.cache}/{stale_ver}.manifest.json", (1, 1))
    if os.path.exists(f"{src_node.store.cache}/{stale_ver}.upload"):
      os.utime(f"{src_node.store.cache}/{stale_ver}.upload", (1, 1))
    gate = asyncio.Event()

    async def delayed_stale_actor():
      await gate.wait()
      c_ok = await self.bus.commit_job_state(
        a, gen_a1, stale_uri, stale_ver, "9999999999999-0", 16, durable=True, location=stale_loc, optimizer=True
      )
      claimed = await self.bus.take_commands({a: gen_a1}, src_owner, 1)
      r_ok = await self.bus.write_reply(
        Cmd(a, "9999999999999-0", Op.OPTIM_STEP, b"{}"),
        Reply(True, "application/json", b'{"stale":true}'),
        src_worker,
        src_owner,
        generation=gen_a1,
        mutates=True,
      )
      return c_ok, claimed, r_ok

    stale_task = asyncio.create_task(delayed_stale_actor())
    # A restores and executes on dst_node under gen_a2 while B keeps executing on src_node under gen_b1.
    a_next = await self.bus.submit_command(a, Op.OPTIM_STEP, b'{"adam_params":{"learning_rate":0.01}}')
    b_next = await self.bus.submit_command(b, Op.OPTIM_STEP, b'{"adam_params":{"learning_rate":0.01}}')
    gate.set()
    c_ok, claimed, r_ok = await stale_task
    self.assertEqual((c_ok, claimed, r_ok), (False, [], False))
    self.assertTrue((await self.bus.wait_for_reply(a_next, 15)).ok)
    self.assertTrue((await self.bus.wait_for_reply(b_next, 15)).ok)
    self.assertEqual((await self.bus.get_run(a))["steps"], 2)
    self.assertEqual((await self.bus.get_run(b))["steps"], 3)
    self.assertNotEqual(((await self.bus.get_run(a))["state"].get("working") or {}).get("version"), stale_ver)

    # Unreferenced stale snapshot is swept cleanly without touching A's committed state.
    a_ver = ((await self.bus.get_run(a))["state"].get("working") or {}).get("version")
    await self.bus.r.set("tinyrl:ckpt_since", 0)
    await self.registry.sweep(src_node.store, grace=0)
    src_node.store.trim_cache(protected={a_ver} if a_ver else set(), limit=0)
    self.assertFalse(os.path.exists(f"{src_node.store.cache}/{stale_ver}"))
    self.assertFalse(src_node.store.is_durable(stale_uri, stale_ver))
    sched.cancel()

  async def test_shared_sampler_aborts_migrating_job_and_replays_on_pinned_version_without_disturbing_sibling(self):
    """Moving sampler session A out of a shared sampler aborts A's in-flight request, leaves PEL unacked for g+1 replay against A's pinned weights_version even after _latest is overwritten, while sibling B succeeds continuously."""
    root = f"file://{self.tmp.name}/sbucket"
    writer = Store(root, cache=f"{self.tmp.name}/sw")
    uri = "tinker://train/sampler_weights/step"
    v1_raw = writer.put(uri, {"w": torch.ones(8)}, {"base": "base", "lora": True, "rank": 4, "config": {}})
    v1, _ = split_version(v1_raw)
    writer.upload_version(uri, v1)
    await self.registry.register_checkpoint(uri, 32, version=v1, durable=True)

    # Overwrite _latest with v2 AFTER pinning A to v1 so replay on destination must honor v1, never _latest!
    v2, _ = split_version(
      writer.put(uri, {"w": torch.full((8,), 2.0)}, {"base": "base", "lora": True, "rank": 4, "config": {}})
    )
    writer.upload_version(uri, v2)
    await self.registry.register_checkpoint(uri, 32, version=v2, durable=True)

    a_in_flight, b_done = asyncio.Event(), asyncio.Event()
    aborted_ids: list[str] = []
    seen_versions: list[str] = []

    class GatedEngine(FakeEngine):
      async def load_checkpoint(self, name, path):
        seen_versions.append(os.path.basename(path))
        await super().load_checkpoint(name, path)

      async def abort(self, request_ids):
        aborted_ids.extend(request_ids)
        await super().abort(request_ids)

      async def sample(self, job, name, request_id=""):
        if job["prompt"] == [77] and not aborted_ids:
          a_in_flight.set()
          await asyncio.sleep(30)  # waits until sampler.cancel_run aborts and cancels it
        return await super().sample(job, name, request_id)

    n1 = TestNode(
      self.bus, Store(root, cache=f"{self.tmp.name}/sc1"), gpus=1, node="sn1", engine=GatedEngine, auto_schedule=False
    )
    n2 = TestNode(
      self.bus, Store(root, cache=f"{self.tmp.name}/sc2"), gpus=1, node="sn2", engine=GatedEngine, auto_schedule=False
    )
    self.nodes += [n1, n2]
    await n1.announce_node()
    await n2.announce_node()
    self.bg(scheduler.run(self.bus))
    self.bg(self._pump(n1))
    self.bg(self._pump(n2))

    await self.bus.create_run(Run("sa", "", "base", 1, role=Role.SAMPLER, model_path=uri, lora={"weights_version": v1}))
    await self.bus.create_run(Run("sb", "", "base", 1, role=Role.SAMPLER))
    await self.registry.acquire_ref(v1, "pin:test")  # keep v1 alive alongside v2

    req_a = await self.bus.submit_command("sa", Op.SAMPLE, json.dumps(JOB | {"prompt": [77]}).encode())
    await asyncio.wait_for(a_in_flight.wait(), 10)
    src_w = (await self.bus.get_run("sa"))["worker"]
    self.assertEqual((await self.bus.get_run("sb"))["worker"], src_w)
    dst_id = "sn2" if src_w in n1.running else "sn1"

    # Sibling sb completes cleanly while sa is in flight and while sa is migrated off src_w!
    req_b1 = await self.bus.submit_command("sb", Op.SAMPLE, json.dumps(JOB | {"prompt": [88]}).encode())
    self.assertTrue((await self.bus.wait_for_reply(req_b1, 10)).ok)

    # Migrate sa onto dst_id: source sampler aborts req_a, fences sa without acking PEL, and dst_id replays req_a cleanly on v1!
    self.assertTrue(await self.bus.request_job_migration("sa", dst_id))
    rep_a = await self.bus.wait_for_reply(req_a, 15)
    self.assertIsNotNone(rep_a)
    self.assertTrue(rep_a.ok, rep_a.body)
    self.assertIn(req_a, aborted_ids)
    self.assertEqual(seen_versions, [v1, v1])  # both initial attempt and replayed attempt staged exact v1, never v2!
    b_done.set()
    req_b2 = await self.bus.submit_command("sb", Op.SAMPLE, json.dumps(JOB | {"prompt": [99]}).encode())
    self.assertTrue((await self.bus.wait_for_reply(req_b2, 10)).ok)
    self.assertEqual((await self.bus.get_run("sb"))["worker"], src_w)

  async def test_sigkill_auto_resumes_at_durable_boundary_exposes_rollback_and_allows_delete(self):
    """Abrupt subprocess SIGKILL auto-resumes when durable boundary covers acked work, and enters explicit rollback when acked mutations were lost."""
    require_unix()
    base = f"{self.tmp.name}/base"
    tiny().save_pretrained(base)
    store = Store(root=f"{self.tmp.name}/local_ckpt")
    node = NodeDaemon(self.bus, gpus=1, node="real-node", gpu_mib=0, store=store)
    self.nodes.append(node)
    await node.announce_node()
    self.bg(scheduler.run(self.bus))
    self.bg(self._pump(node))

    run = new_id()
    save_uri = f"tinker://{run}/weights/durable-1"
    await self.bus.create_run(Run(run, "", base, 4, lora=LORA))
    for op, body in (
      (Op.CREATE_MODEL, {"base_model": base, "rank": 4, "lora": LORA}),
      (Op.FORWARD_BACKWARD, proto_body(run)),
      (Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}}),
      (Op.SAVE_WEIGHTS, {"uri": save_uri}),
    ):
      raw = body if isinstance(body, bytes) else json.dumps(body).encode()
      rep = await self.bus.wait_for_reply(await self.bus.submit_command(run, op, raw), 30)
      self.assertTrue(rep.ok, rep.body)

    # 1. SIGKILL right at durable boundary -> automatic recovery from save_uri without client intervention!
    worker = (await self.bus.get_run(run))["worker"]
    pid1 = node.running[worker]["pid"]
    os.killpg(pid1, signal.SIGKILL)
    auto_step = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{"learning_rate":0.01}}')
    auto_rep = await self.bus.wait_for_reply(auto_step, 30)
    self.assertTrue(auto_rep.ok, auto_rep.body)
    pid2 = node.running[worker]["pid"]
    self.assertNotEqual(pid1, pid2)

    # 2. auto_step mutated state past save_uri without a new durable save -> SIGKILL now triggers rollback!
    os.killpg(pid2, signal.SIGKILL)
    for op, body in (
      (Op.FORWARD_BACKWARD, proto_body(run)),
      (Op.OPTIM_STEP, b'{"adam_params":{}}'),
      (Op.SAVE_WEIGHTS, json.dumps({"uri": f"tinker://{run}/weights/blocked"}).encode()),
    ):
      rep = await self.bus.wait_for_reply(await self.bus.submit_command(run, op, body), 30)
      self.assertFalse(rep.ok)
      self.assertIn("rolled back", rep.body.decode())

    # Explicit LOAD_WEIGHTS restores save_uri and clears rollback so mutations succeed again.
    load_rep = await self.bus.wait_for_reply(
      await self.bus.submit_command(run, Op.LOAD_WEIGHTS, json.dumps({"uri": save_uri, "optimizer": True}).encode()), 30
    )
    self.assertTrue(load_rep.ok, load_rep.body)
    ok_rep = await self.bus.wait_for_reply(
      await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{"learning_rate":0.01}}'), 30
    )
    self.assertTrue(ok_rep.ok, ok_rep.body)

    # 3. A rolled-back job can always be deleted cleanly without restoring first.
    os.killpg(node.running[worker]["pid"], signal.SIGKILL)
    denied = await self.bus.wait_for_reply(await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}'), 30)
    self.assertFalse(denied.ok)
    del_rep = await self.bus.wait_for_reply(await self.bus.submit_command(run, Op.DELETE_MODEL, b"{}"), 30)
    self.assertTrue(del_rep.ok, del_rep.body)
    self.assertEqual(await self.bus.queued_and_in_flight(run), {"live": False})
    self.assertIsNone(await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}"))

  async def test_multi_rank_follower_death_during_suspension_enters_failure_not_parked(self):
    """Killing a non-zero FSDP training rank prevents a clean park and surfaces failure + verified process cleanup."""
    require_unix()
    import psutil

    base, ckpt = f"{self.tmp.name}/mbase", f"{self.tmp.name}/mckpt"
    tiny().save_pretrained(base)
    env = {**os.environ, "CKPT": ckpt, "CUDA_VISIBLE_DEVICES": "", "COLL_TIMEOUT_S": "3"}
    args = [
      sys.executable,
      "-m",
      "torch.distributed.run",
      "--standalone",
      "--nproc-per-node=2",
      "-m",
      "tinyrl.workers.trainer",
    ]
    proc = Process(args, env)
    try:
      await asyncio.wait_for(proc.call("ensure_run", "mr", base, 4, LORA), 90)
      # Find the two rank worker children under torchrun and SIGKILL rank 1 before calling park.
      children = sorted(psutil.Process(proc.pid).children(recursive=True), key=lambda p: p.pid)
      self.assertGreaterEqual(len(children), 2)
      children[-1].kill()
      with self.assertRaises((EOFError, RuntimeError, TimeoutError, OSError)):
        await asyncio.wait_for(proc.call("park"), timeout=10)
    finally:
      await proc.close(graceful=False)
    self.assertTrue(gpu_pids_clear(proc.pid))

  async def test_expired_lease_never_proves_gpu_release_while_child_still_lives(self):
    """Even after a worker's lease key expires in Redis, a live child PID keeps the GPU unavailable until terminated."""
    require_unix()
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    self.procs.append(sleeper)
    self.assertFalse(gpu_pids_clear(sleeper.pid))

    await self.bus.node_heartbeat("n1", 1, 0)
    await self.bus.create_run(Run("old", "", "tiny-a", 0))
    await self.bus.create_run(Run("new", "", "tiny-b", 0))
    await self.bus.submit_command("old", Op.CREATE_MODEL, b'{"base_model":"tiny-a","rank":0,"lora":{}}')
    await self.bus.submit_command("new", Op.CREATE_MODEL, b'{"base_model":"tiny-b","rank":0,"lora":{}}')
    await scheduler.step(self.bus)
    s = scheduler.from_table(await self.bus.read_table())
    w_old, w_new = s.jobs["old"].worker, s.jobs["new"].worker
    # Report active observation with real live child PID; lease key is completely absent (expired), and w_old is draining.
    await self.bus.write_observation(
      w_old,
      {
        "worker": w_old,
        "node": "n1",
        "revision": s.workers[w_old].revision,
        "generations": {"old": 1},
        "gpus": [0],
        "phase": "running",
        "gpu_state": "active",
        "pids_clear": gpu_pids_clear(sleeper.pid),
        "usage": {"old": 50.0},
        "served": {"old": scheduler.TURN},
        "pid": sleeper.pid,
      },
    )
    self.assertIsNone(await self.bus.lease_owner(w_old))
    await scheduler.step(self.bus)
    s2 = scheduler.from_table(await self.bus.read_table())
    self.assertEqual(s2.workers[w_old].intent, "drain")
    self.assertEqual(s2.workers[w_new].gpus, ())  # GPU 0 is NOT granted to w_new despite expired lease!

    # Kill the child, confirm gpu_pids_clear(pid) is True, publish verified release -> w_new immediately gets GPU 0.
    sleeper.kill()
    await asyncio.to_thread(sleeper.wait)
    self.assertTrue(gpu_pids_clear(sleeper.pid))
    await self.bus.write_observation(
      w_old,
      {
        "worker": w_old,
        "node": "n1",
        "revision": s2.workers[w_old].revision,
        "generations": {"old": 1},
        "gpus": [],
        "phase": "released",
        "gpu_state": "released",
        "pids_clear": gpu_pids_clear(sleeper.pid),
        "usage": {"old": 50.0},
        "pid": sleeper.pid,
      },
    )
    await scheduler.step(self.bus)
    s3 = scheduler.from_table(await self.bus.read_table())
    self.assertEqual((s3.workers[w_new].gpus, s3.workers[w_new].intent), ((0,), "run"))

  async def _migrated(self, run_id: str, old_worker: str, dst_node: str) -> bool:
    s = scheduler.from_table(await self.bus.read_table())
    j = s.jobs.get(run_id)
    if not j or j.worker == old_worker:
      return False
    w = s.workers.get(j.worker)
    return bool(w and w.node == dst_node)

  async def _pump(self, node: NodeDaemon):
    while True:
      await node.tick()
      await asyncio.sleep(0.02)


class TestCompiledArtifactContract(unittest.TestCase):
  """PR 6 gate: compiled-artifact compatibility key stability and p95 startup replacement gate."""

  def test_artifact_key_ignores_job_id_port_and_checkpoint_version_and_separates_incompatible_profiles(self):
    base_flags = ["--enable-sleep-mode", "--enable-lora", "--max-lora-rank", "64"]
    k1 = vllm.artifact_key(
      "Qwen/Qwen3-4B", True, 1, [*base_flags, "--port", "8001", "--gpu-memory-utilization", "0.85"], "H100"
    )
    k2 = vllm.artifact_key(
      "Qwen/Qwen3-4B",
      True,
      1,
      [*base_flags, "--port", "9002", "--gpu-memory-utilization=0.42", "--served-model-name", "alt"],
      "H100",
    )
    self.assertEqual(k1, k2)
    self.assertNotEqual(k1, vllm.artifact_key("Qwen/Qwen3-4B", False, 1, ["--enable-sleep-mode"], "H100"))
    self.assertNotEqual(k1, vllm.artifact_key("Qwen/Qwen3-4B", True, 1, base_flags, "A100"))
    self.assertNotEqual(k1, vllm.artifact_key("Qwen/Qwen3-4B", True, 2, base_flags, "H100"))
    self.assertNotEqual(k1, vllm.artifact_key("Qwen/Qwen3-8B", True, 1, base_flags, "H100"))
    self.assertNotEqual(k1, vllm.artifact_key("Qwen/Qwen3-4B", True, 1, [*base_flags, "--dtype", "float16"], "H100"))

  def test_startup_gate_evaluates_p95_against_budget_and_snapshot_baseline(self):
    fast = [4.1, 4.3, 4.2, 4.5, 4.4, 4.6, 4.3, 4.2, 4.7, 4.8]
    passed = vllm.startup_gate(fast, target_p95_s=10.0)
    self.assertTrue(passed["passed"])
    self.assertLessEqual(passed["p95_s"], 4.8)
    self.assertFalse(vllm.startup_gate(fast[:3], target_p95_s=10.0, min_samples=5)["passed"])
    slow = [11.0, 12.5, 13.0, 14.0, 15.0]
    self.assertFalse(vllm.startup_gate(slow, target_p95_s=10.0)["passed"])
    self.assertFalse(vllm.startup_gate([], target_p95_s=10.0)["passed"])


if __name__ == "__main__":
  unittest.main()
