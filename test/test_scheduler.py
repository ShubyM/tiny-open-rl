import copy
import dataclasses
import random
import unittest
from unittest import mock

from test.helpers import RedisTest
from tinyrl.bus import Cmd, Ev, Op, Run
from tinyrl.control import scheduler
from tinyrl.control.scheduler import (
  Assignment,
  Observation,
  Worker,
  WorkerSpec,
  decide,
  from_table,
  select_runs,
  worker_to_dict,
)

GIB, MIB = 2**30, 2**20


class Cluster:
  """In-memory mirror of the Redis table that `scheduler.decide` consumes and commits."""

  def __init__(self, now: float = 1000.0):
    self.nodes: dict[str, dict] = {}
    self.records: dict[str, dict] = {}
    self.states: dict[str, dict] = {}
    self.queues: dict[str, int] = {}
    self.obs: dict[str, dict] = {}
    self.versions: dict[str, dict] = {}
    self.s = Assignment()
    self.now = now

  def table(self) -> dict:
    runs = {}
    for r, rec in self.records.items():
      st = dict(self.states.get(r, {}))
      if r in self.s.jobs:
        st["generation"] = self.s.jobs[r].generation
      runs[r] = (rec, self.s.jobs[r].worker if r in self.s.jobs else "", st)
    return {
      "nodes": copy.deepcopy(self.nodes),
      "runs": runs,
      "queues": dict(self.queues),
      "workers": {n: worker_to_dict(w) for n, w in self.s.workers.items()},
      "workers_made": self.s.workers_made,
      "observations": copy.deepcopy(self.obs),
      "service": dict(self.s.service),
      "settled": copy.deepcopy(self.s.settled),
      "reserved": copy.deepcopy(self.s.reserved),
      "versions": copy.deepcopy(self.versions),
    }

  def settle(self, dt: float = 0.0) -> list[tuple[Ev, dict]]:
    self.now += dt
    old = from_table(self.table())
    self.s, decisions, bumps = decide(old, self.now)
    for gone in set(self.obs) - set(self.s.observations):
      self.obs.pop(gone, None)
    for r, gen in bumps.items():
      if r in self.states:
        self.states[r]["generation"] = gen
    return decisions

  def node(self, name: str, gpus: int, memory: int = 80 * GIB, **extra):
    self.nodes[name] = {"gpus": gpus, "memory": memory, **extra}
    return self.settle()

  def lose(self, name: str):
    self.nodes.pop(name, None)

  def run(self, name: str, queued: int = 0, state: dict | None = None, **body) -> str:
    rec = {"role": "trainer", "base": "a", "rank": 0, "created": self.now} | body
    self.records[name] = rec
    acct = rec["train_run"] if rec["role"] == "sampler" and rec.get("train_run") else name
    self.states[name] = {"generation": 0, "account": acct} | (state or {})
    self.queues[name] = queued
    self.settle()
    return self.s.jobs[name].worker

  def observe(self, worker: str, **fields):
    w = self.s.workers.get(worker)
    base = Observation(
      worker=worker,
      node=fields.pop("node", w.node if w else ""),
      revision=fields.pop("revision", w.revision if w else 1),
      generations=fields.pop("generations", dict(w.generations) if w else {}),
      gpus=tuple(fields.pop("gpus", w.gpus if w else ())),
      at=self.now,
      **fields,
    )
    self.obs[worker] = dataclasses.asdict(base)

  def delete(self, run: str):
    self.records.pop(run, None)
    self.states.pop(run, None)
    self.queues.pop(run, None)


class TestFleetAllocation(unittest.TestCase):
  def test_runs_share_eligible_workers_receive_physical_bundles_and_clean_up_when_empty(self):
    c = Cluster()
    c.node("n", 4)
    a = c.run("a", queued=1, rank=8)
    self.assertEqual(c.run("b", queued=1, rank=8), a)
    w = c.s.workers[a]
    self.assertEqual((w.node, w.gpus, w.intent), ("n", (0,), "run"))
    self.assertEqual(set(w.generations), {"a", "b"})
    self.assertTrue(all(w.budgets[r] > 0 for r in ("a", "b")))
    c.delete("a")
    c.settle()
    self.assertIn(a, c.s.workers)
    c.delete("b")
    events = c.settle()
    self.assertTrue(any(ev == Ev.UNPLACED and body["worker"] == a for ev, body in events))
    self.assertNotIn(a, c.s.workers)

  def test_new_worker_never_reuses_existing_name(self):
    c = Cluster()
    c.node("n", 2)
    c.s.workers["w1"] = Worker(WorkerSpec(lora=True), "n", (0,))
    self.assertNotEqual(c.run("f", queued=1), "w1")

  def test_lora_and_sampler_sharing_boundaries(self):
    c = Cluster()
    c.node("n", 8)
    a, b = c.run("a", rank=8), c.run("b", rank=8)
    others = [c.run("c", rank=8, base="b"), c.run("f"), c.run("x", rank=8, exclusive=True)]
    self.assertEqual(a, b)
    self.assertEqual(len({a, b, *others}), 4)

    c.run("Z", rank=8, exclusive=True)
    sa = c.run("sa", role="sampler", rank=1, train_run="a")
    sb = c.run("sb", role="sampler", rank=1, train_run="b")
    sf = c.run("sf", role="sampler", rank=0, train_run="f")
    sx = c.run("sx", role="sampler", rank=1, train_run="x", exclusive=True)
    sy = c.run("sy", role="sampler", rank=1, train_run="x", exclusive=True)
    sz = c.run("sz", role="sampler", rank=1, train_run="Z", exclusive=True)
    self.assertEqual(sa, sb)
    self.assertEqual(sx, sy)
    self.assertEqual(len({sa, sb, sf, sx, sy, sz}), 4)

  def test_gpu_class_and_intra_node_bundle_constraints(self):
    c = Cluster()
    c.node("a100", 4, gpu_class="A100", bundles=[[0, 1], [2, 3]])
    c.node("h100", 4, gpu_class="H100", bundles=[[0, 1], [2, 3]])
    wh = c.run("h-job", queued=1, gpus=2, state={"gpu_classes": ["H100"]})
    self.assertEqual((c.s.workers[wh].node, c.s.workers[wh].gpus), ("h100", (0, 1)))
    c.observe(wh, phase="running", gpu_state="active")
    # Occupy GPU 2 on h100 so only GPU 3 is idle -> second 2-GPU H100 job waits for a valid pair instead of taking (1, 3) or A100.
    w1 = c.run("h-one", queued=1, gpus=1, state={"gpu_classes": ["H100"]})
    self.assertEqual((c.s.workers[w1].node, c.s.workers[w1].gpus), ("h100", (2,)))
    c.observe(w1, phase="running", gpu_state="active")
    wh2 = c.run("h-job2", queued=1, gpus=2, state={"gpu_classes": ["H100"]})
    self.assertEqual((c.s.workers[wh2].node, c.s.workers[wh2].gpus), ("h100", ()))
    # An incompatible GPU class stays unplaced until a matching node joins.
    wb = c.run("b200-job", queued=1, state={"gpu_classes": ["B200"]})
    self.assertIsNone(c.s.workers[wb].node)

  def test_memory_copacking_and_host_memory_eviction(self):
    c = Cluster()
    c.node("n", 2, memory=80 * GIB, host_memory=4 * GIB)
    # Two non-exclusive 1-GPU full-FT samplers co-pack on GPU 0 (each 0.42 * 80 GiB), leaving GPU 1 free for a trainer.
    s1 = c.run("s1", queued=1, role="sampler", rank=0, base="m1")
    s2 = c.run("s2", queued=1, role="sampler", rank=0, base="m2")
    t1 = c.run("t1", queued=1, rank=0, base="m1")
    self.assertEqual(c.s.workers[s1].gpus, (0,))
    self.assertEqual(c.s.workers[s2].gpus, (0,))
    self.assertEqual(c.s.workers[t1].gpus, (1,))

    # Two offloaded workers exceeding host_memory trigger export drain on the lower-priority holder.
    p1 = c.run("p1", queued=0, base="p1")
    p2 = c.run("p2", queued=0, base="p2")
    c.s.service["p1"], c.s.service["p2"] = 1.0, 20.0
    c.observe(p1, gpus=(1,), phase="parked", gpu_state="offloaded", host_bytes=3 * GIB)
    c.observe(p2, gpus=(1,), phase="parked", gpu_state="offloaded", host_bytes=3 * GIB)
    c.settle(1.0)
    self.assertEqual((c.s.workers[p2].intent, c.s.workers[p2].drain_mode), ("drain", "export"))
    self.assertNotEqual(c.s.workers[p1].drain_mode, "export")

  def test_exclusive_admission_cap_and_isolation(self):
    with mock.patch.object(scheduler, "EXCLUSIVE_MAX_GPUS", 2):
      c = Cluster()
      c.node("n", 4)
      x1 = c.run("x1", queued=1, exclusive=True)
      xs = c.run("xs", queued=1, role="sampler", rank=1, train_run="x1", exclusive=True)
      x2 = c.run("x2", queued=1, exclusive=True)  # exceeds EXCLUSIVE_MAX_GPUS=2 -> queued unplaced
      shared = c.run("sh", queued=1, rank=8)
      self.assertEqual((c.s.workers[x1].node, c.s.workers[x1].gpus), ("n", (0,)))
      self.assertEqual((c.s.workers[xs].node, c.s.workers[xs].gpus), ("n", (1,)))
      self.assertIsNone(c.s.workers[x2].node)
      self.assertNotIn(c.s.workers[shared].gpus[0], (0, 1))
      # Once x1 + xs finish and leave, x2 is admitted and receives a dedicated bundle.
      c.delete("xs")
      c.delete("x1")
      c.settle(1.0)
      self.assertEqual((c.s.workers[x2].node, len(c.s.workers[x2].gpus)), ("n", 1))

  def test_shared_job_account_and_idle_return_floor(self):
    c = Cluster()
    c.node("n", 1)
    # Job A has a trainer plus a sampler replica; Job B has only a trainer. A's sampler charges account "a".
    wa = c.run("a", queued=1, base="ma")
    wsa = c.run("sa", queued=1, role="sampler", rank=1, train_run="a", base="ma")
    wb = c.run("b", queued=1, base="mb")
    self.assertEqual(c.s.jobs["sa"].account, "a")
    c.observe(wsa, phase="running", gpu_state="active", usage={"a": 12.0})
    c.observe(wa, phase="running", gpu_state="active", usage={"a": 3.0}, served={"a": scheduler.TURN})
    c.settle(1.0)
    self.assertAlmostEqual(c.s.service["a"], 15.0)
    # B has 0 service vs A's 15 -> A is told to drain so B gets the GPU.
    self.assertEqual((c.s.workers[wa].intent, c.s.workers[wa].drain_mode), ("drain", "park"))
    c.observe(wa, phase="parked", gpu_state="offloaded", usage={"a": 3.0})
    c.settle(0.5)
    self.assertEqual((c.s.workers[wb].intent, c.s.workers[wb].gpus), ("run", (0,)))

    # B accrues 10 service; idle job C wakes up -> lifted to floor (10.0), zero idle credit; A keeps prior debt (15.0).
    c.observe(wb, phase="running", gpu_state="active", usage={"b": 10.0})
    c.run("idle-c", queued=0, base="mc")
    c.settle(1.0)
    c.queues["idle-c"] = 1
    c.settle(1.0)
    self.assertAlmostEqual(c.s.service["idle-c"], 10.0)
    self.assertAlmostEqual(c.s.service["a"], 15.0)

    # Opportunistic idle residency charges zero service.
    c.queues["a"] = c.queues["sa"] = c.queues["b"] = c.queues["idle-c"] = 0
    c.settle(5.0)
    self.assertEqual(c.s.workers[wb].intent, "idle")
    self.assertEqual(c.s.workers[wb].gpus, (0,))
    self.assertAlmostEqual(c.s.service["b"], 10.0)

  def test_node_loss_settles_service_conservatively_and_advances_generation(self):
    c = Cluster()
    c.node("a", 2)
    c.node("b", 2)
    w = c.run("r", queued=1)
    gen1 = c.s.jobs["r"].generation
    pred = c.s.workers[w].predicted["r"]
    self.assertGreater(pred, 0.0)
    c.lose("a")
    events = c.settle(dt=100.0)  # even after a long outage, loss charge is capped by assigned predicted work
    self.assertTrue(any(ev == Ev.UNPLACED and body["rule"] == "place.node_lost" for ev, body in events))
    self.assertAlmostEqual(c.s.service["r"], pred, places=4)
    self.assertEqual(c.s.workers[w].node, "b")
    self.assertGreater(c.s.jobs["r"].generation, gen1)

  def test_unreleased_gpu_blocks_reuse_until_verified_release(self):
    c = Cluster()
    c.node("n", 1)
    wa = c.run("a", queued=1, base="ma")
    wb = c.run("b", queued=1, base="mb")
    # Force wa to drain, but report gpu_state="released" with pids_clear=False (lingering GPU process!).
    c.observe(wa, phase="running", gpu_state="active", usage={"a": 5.0}, served={"a": scheduler.TURN})
    c.settle(1.0)
    self.assertEqual(c.s.workers[wa].intent, "drain")
    c.observe(wa, phase="released", gpu_state="released", pids_clear=False, usage={"a": 5.0})
    c.settle(1.0)
    self.assertEqual(c.s.workers[wb].gpus, ())
    # Once pids_clear=True is reported, wb immediately gets GPU 0.
    c.observe(wa, phase="released", gpu_state="released", pids_clear=True, usage={"a": 5.0})
    c.settle(1.0)
    self.assertEqual((c.s.workers[wb].gpus, c.s.workers[wb].intent), ((0,), "run"))

  def test_wide_job_reserves_bundle_and_drains_conflicting_single_gpu_holders(self):
    c = Cluster()
    c.node("n", 2)
    w0 = c.run("n0", queued=1, base="m0")
    w1 = c.run("n1", queued=1, base="m1")
    self.assertEqual((c.s.workers[w0].gpus, c.s.workers[w1].gpus), ((0,), (1,)))
    c.observe(w0, phase="running", gpu_state="active", usage={"n0": 5.0})
    c.observe(w1, phase="running", gpu_state="active", usage={"n1": 5.0})
    ww = c.run("wide", queued=1, gpus=2, base="mw")
    # Wide job reserves (0, 1) and orders both w0 and w1 to park; even if w0 parks first, another 1-GPU job cannot steal GPU 0.
    self.assertEqual(c.s.reserved["n"]["worker"], ww)
    self.assertEqual(c.s.workers[w0].intent, "drain")
    self.assertEqual(c.s.workers[w1].intent, "drain")
    c.observe(w0, phase="parked", gpu_state="offloaded", usage={"n0": 5.0})
    w2 = c.run("late", queued=1, base="m2")
    self.assertEqual(c.s.workers[w2].gpus, ())
    self.assertEqual(c.s.workers[ww].gpus, ())
    c.observe(w1, phase="parked", gpu_state="offloaded", usage={"n1": 5.0})
    c.settle(0.5)
    self.assertEqual((c.s.workers[ww].gpus, c.s.workers[ww].intent), ((0, 1), "run"))
    self.assertNotIn("n", c.s.reserved)

  def test_makespan_migration_respects_transfer_hysteresis_and_sets_prepare(self):
    c = Cluster()
    c.versions["v1"] = {
      "uri": "tinker://j/weights/s1",
      "version": "v1",
      "size": 100 * MIB,
      "durable": True,
      "locations": ["10.0.0.1:9"],
      "state": "live",
    }
    c.node("slow", 1, peer="10.0.0.1:9", peer_bps=200 * MIB, bucket_bps=100 * MIB)
    c.node("fast", 1, peer="10.0.0.2:9", peer_bps=200 * MIB, bucket_bps=100 * MIB)
    blocker = c.run("block", queued=1, base="mb")
    self.assertEqual(c.s.workers[blocker].node, "fast")
    # Place j on slow while it has cached v1, then block slow and free fast so migrating beats transfer + RESTORE_S.
    c.nodes["slow"]["cached"] = ["v1"]
    wj = c.run("j", queued=1, base="mj", state={"working": {"uri": "tinker://j/weights/s1", "version": "v1"}})
    self.assertEqual(c.s.workers[wj].node, "slow")
    self.assertEqual(c.s.workers[wj].prepare, [])
    # Move j to a node without v1 -> prepare lists v1; once cached, prepare clears.
    c.lose("slow")
    c.delete("block")
    c.settle(1.0)
    self.assertEqual(c.s.workers[wj].node, "fast")
    self.assertEqual(c.s.workers[wj].prepare, ["v1"])
    c.nodes["fast"]["cached"] = ["v1"]
    c.settle(0.5)
    self.assertEqual(c.s.workers[wj].prepare, [])

  def test_in_assignment_select_runs_balances_heterogeneous_command_costs(self):
    heads = [Cmd(run, "1-0", Op.FORWARD_BACKWARD, b"") for run in ("slow", "fast")]
    service = {"slow": 0.0, "fast": 0.0}
    for _ in range(11):
      [run] = select_runs(heads, service, 1)
      service[run] += 10 if run == "slow" else 1
    self.assertEqual(service, {"slow": 10.0, "fast": 10.0})

  def test_random_clusters_never_double_book_physical_gpus_and_survive_roundtrip(self):
    for seed in range(50):
      rng, c = random.Random(seed), Cluster()
      for _ in range(60):
        roll, name = rng.random(), str(rng.randrange(8))
        if roll < 0.15:
          c.node(name, rng.choice([1, 2, 4]))
        elif roll < 0.25:
          c.lose(name)
        elif roll < 0.75 and name not in c.s.jobs:
          c.run(
            name,
            queued=rng.choice([0, 1, 2]),
            rank=rng.choice([0, 8]),
            base=str(rng.randrange(3)),
            gpus=rng.choice([1, 2]),
          )
        else:
          c.delete(name)
        c.settle(0.2)
        for node, info in c.s.nodes.items():
          seen: dict[int, list[str]] = {}
          for wn, w in c.s.workers.items():
            if w.node == node and w.intent in ("run", "idle") and w.gpus:
              self.assertEqual(len(w.gpus), w.spec.gpus)
              for g in w.gpus:
                self.assertIn(g, info.gpu_ids)
                seen.setdefault(g, []).append(wn)
          for holders in seen.values():
            if len(holders) > 1:
              self.assertFalse(any(c.s.workers[h].spec.exclusive for h in holders))
        rt = from_table(c.table())
        self.assertEqual(rt.workers, c.s.workers)
        self.assertEqual(rt.service, c.s.service)


class TestSchedulerFencing(RedisTest):
  async def test_single_writer_lease_and_live_restart_preserve_state_and_service(self):
    await self.bus.node_heartbeat("n1", 2, 80 * GIB)
    await self.bus.create_run(Run("r1", "", "tiny", 4))
    await self.bus.submit_command("r1", Op.CREATE_MODEL, b'{"base_model":"tiny","rank":4,"lora":{}}')

    epoch1 = await self.bus.claim_scheduler("sched-a")
    self.assertIsNotNone(epoch1)
    self.assertIsNone(await self.bus.claim_scheduler("sched-b"))

    await scheduler.step(self.bus, "sched-a")
    s1 = from_table(await self.bus.read_table())
    w1 = s1.jobs["r1"].worker
    gen1 = s1.jobs["r1"].generation
    self.assertEqual((s1.workers[w1].node, s1.workers[w1].gpus, s1.workers[w1].intent), ("n1", (0,), "run"))
    await self.bus.write_observation(
      w1,
      dataclasses.asdict(
        Observation(w1, "n1", s1.workers[w1].revision, {"r1": gen1}, (0,), "running", "active", usage={"r1": 4.5})
      ),
    )
    await scheduler.step(self.bus, "sched-a")

    # Stale scheduler cannot overwrite the table after sched-b acquires the next epoch.
    await self.bus.release_scheduler("sched-a")
    epoch2 = await self.bus.claim_scheduler("sched-b")
    self.assertIsNotNone(epoch2)
    self.assertGreater(epoch2, epoch1)
    self.assertFalse(await self.bus.write_table({}, {}, [], 99, [], scheduler_owner="sched-a"))

    # Fresh scheduler reconciles live node + running worker without disrupting its assignment, generation, or service.
    await scheduler.step(self.bus, "sched-b")
    s2 = from_table(await self.bus.read_table())
    self.assertEqual(s2.jobs["r1"].worker, w1)
    self.assertEqual(s2.jobs["r1"].generation, gen1)
    self.assertEqual((s2.workers[w1].node, s2.workers[w1].gpus), ("n1", (0,)))
    self.assertAlmostEqual(s2.service["r1"], 4.5)


if __name__ == "__main__":
  unittest.main()
