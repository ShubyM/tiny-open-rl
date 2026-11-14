import random
import unittest

from tinyrl.bus import Ev
from tinyrl.control.scheduler import (
  BUDGET_S,
  Assignment,
  Worker,
  WorkerSpec,
  decide,
  fits,
  from_table,
  gpus_still_held,
  memory_budget,
  worker_to_dict,
)

GIB, MIB = 2**30, 2**20


class Cluster:
  """In-memory mirror of the Redis table consumed by `from_table -> decide`."""

  def __init__(self, exclusive_cap: float = 0.5):
    self.nodes: dict[str, dict] = {}
    self.records: dict[str, dict] = {}
    self.meta: dict[str, dict] = {}
    self.queues: dict[str, dict] = {}
    self.versions: dict[str, dict] = {}
    self.ckpt_versions: dict[str, str] = {}
    self.s = Assignment(exclusive_cap=exclusive_cap)

  def table(self) -> dict:
    return {
      "nodes": self.nodes,
      "runs": {r: (rec, self.s.jobs[r].worker if r in self.s.jobs else "") for r, rec in self.records.items()},
      "meta": {
        r: {
          **self.meta.get(r, {}),
          "gen": self.s.jobs[r].gen if r in self.s.jobs else int(self.meta.get(r, {}).get("gen", 0)),
        }
        for r in self.records
      },
      "queues": self.queues,
      "workers": {n: worker_to_dict(w) for n, w in self.s.workers.items()},
      "workers_made": self.s.workers_made,
      "table_rev": self.s.table_rev,
      "service": dict(self.s.service),
      "service_obs": dict(self.s.service_obs),
      "service_ready": list(self.s.service_ready),
      "versions": dict(self.versions),
      "ckpt_versions": dict(self.ckpt_versions),
    }

  def settle(self) -> list[tuple[Ev, dict]]:
    snap = from_table(self.table())
    snap.exclusive_cap = self.s.exclusive_cap
    self.s, decisions = decide(snap)
    return decisions

  def node(self, name: str, gpus: int, memory: int = 80 * GIB, host_ram: int = 0, peer: str = "") -> None:
    self.nodes[name] = {
      "gpus": gpus,
      "memory": memory,
      "host_ram": host_ram,
      "host_ram_used": 0,
      "peer": peer,
      "workers": {},
      "versions": {},
    }
    self.settle()

  def lose(self, node: str) -> None:
    self.nodes.pop(node, None)

  def run(self, name: str, *, queued: int = 0, meta: dict | None = None, **body) -> str:
    self.records[name] = {"role": "trainer", "base": "a", "rank": 0} | body
    if meta:
      self.meta[name] = dict(meta)
    self.queues[name] = {"queued": queued, "in_flight": 0}
    self.settle()
    return self.s.jobs[name].worker

  def queue(self, name: str, queued: int, in_flight: int = 0) -> None:
    self.queues[name] = {"queued": queued, "in_flight": in_flight}

  def delete(self, run: str) -> None:
    self.records.pop(run, None)
    self.meta.pop(run, None)
    self.queues.pop(run, None)

  def obs(self, worker: str, **fields) -> None:
    w = self.s.workers[worker]
    assert w.node is not None and w.node in self.nodes
    cur = self.nodes[w.node]["workers"].get(worker, {})
    cur.update(fields)
    self.nodes[w.node]["workers"][worker] = cur

  def ack_all(self) -> None:
    """Simulate every node applying its current worker intent & confirming GPU hold/release."""
    for name, w in list(self.s.workers.items()):
      if not w.node or w.node not in self.nodes:
        continue
      cur = self.nodes[w.node]["workers"].get(name, {})
      jobs = cur.get("jobs", {})
      if w.intent == "run":
        self.obs(
          name,
          rev=w.rev,
          owner=cur.get("owner") or f"{w.node}:{name}",
          pid=cur.get("pid") or 100,
          gpus=list(w.gpus),
          gpus_held=list(w.gpus),
          gpu_released=False,
          phase="running",
          jobs=jobs,
        )
      elif w.intent == "park":
        self.obs(
          name,
          rev=w.rev,
          owner=cur.get("owner") or f"{w.node}:{name}",
          pid=cur.get("pid"),
          gpus=list(w.gpus),
          gpus_held=[],
          gpu_released=True,
          phase="parked" if cur.get("pid") else "idle",
          jobs=jobs,
        )
      elif w.intent == "drain":
        self.obs(
          name,
          rev=w.rev,
          owner="",
          pid=None,
          gpus=[],
          gpus_held=[],
          gpu_released=True,
          phase="drained",
          host_ram_bytes=0,
          jobs=jobs,
        )

  def add_usage(self, worker: str, job: str, delta: float) -> None:
    w = self.s.workers[worker]
    assert w.node is not None
    cur = self.nodes[w.node]["workers"].setdefault(
      worker,
      {
        "rev": w.rev,
        "owner": f"{w.node}:{worker}",
        "pid": 100,
        "gpus": list(w.gpus),
        "gpus_held": list(w.gpus) if w.intent == "run" else [],
        "gpu_released": w.intent != "run",
        "phase": "running" if w.intent == "run" else "parked",
        "jobs": {},
      },
    )
    je = cur["jobs"].setdefault(job, {"gen": self.s.jobs[job].gen, "usage": 0.0, "tokens": 0, "state": "active"})
    je["usage"] += delta

  def check_no_gpu_overcommit(self, test: unittest.TestCase) -> None:
    for node_name, ni in self.s.nodes.items():
      for gpu in range(ni.gpus):
        holders = [n for n, w in self.s.workers.items() if w.node == node_name and gpu in gpus_still_held(self.s, n)]
        if len(holders) > 1:
          test.assertFalse(any(self.s.workers[h].spec.exclusive for h in holders))
          budgets = [memory_budget(self.s, self.s.workers[h].spec, node_name) for h in holders]
          test.assertTrue(fits(budgets, ni.memory), f"overcommit on {node_name}:{gpu}: {holders}")


class TestCluster(unittest.TestCase):
  def test_runs_are_assigned_and_empty_workers_require_release_before_removal(self):
    c = Cluster()
    c.node("n", 8)
    a = c.run("a", rank=8, queued=1)
    self.assertEqual(c.run("b", rank=8, queued=1), a)
    self.assertEqual(c.s.workers[a].node, "n")
    self.assertEqual(c.s.workers[a].intent, "run")
    self.assertEqual(len(c.s.workers[a].gpus), 1)
    c.ack_all()
    c.delete("a")
    c.settle()
    self.assertIn(a, c.s.workers)
    c.delete("b")
    decisions = c.settle()
    self.assertEqual(decisions[0][0], Ev.DRAINING)
    self.assertIn(a, c.s.workers)  # waits for GPU release confirmation
    c.ack_all()
    decisions = c.settle()
    self.assertEqual(decisions, [(Ev.UNPLACED, {"worker": a, "rule": "place.emptied"})])
    self.assertNotIn(a, c.s.workers)

  def test_a_new_worker_never_takes_a_name_in_use(self):
    c = Cluster()
    c.node("n", 1)
    c.s.workers["w1"] = Worker(WorkerSpec(lora=True), "n")
    self.assertNotEqual(c.run("f"), "w1")

  def test_workers_spread_across_nodes_and_multi_gpu_waits_for_eligible_node(self):
    c = Cluster()
    c.node("a", 1)
    w = c.run("wide", gpus=2, queued=1)
    self.assertIsNone(c.s.workers[w].node)
    c.node("b", 4)
    self.assertEqual(c.s.workers[w].node, "b")
    self.assertEqual(len(c.s.workers[w].gpus), 2)

  def test_lost_node_settles_remaining_budget_bumps_generation_and_reassigns(self):
    c = Cluster()
    c.node("a", 1)
    w = c.run("r", queued=1)
    self.assertEqual(c.s.workers[w].intent, "run")
    gen_before = c.s.jobs["r"].gen
    budget_before = c.s.workers[w].budgets["r"]
    c.node("b", 1)
    c.lose("a")
    decisions = c.settle()
    self.assertEqual(decisions[0], (Ev.UNPLACED, {"worker": w, "rule": "place.node_lost"}))
    self.assertEqual(c.s.jobs["r"].gen, gen_before + 1)
    self.assertGreaterEqual(c.s.service["r"], budget_before)
    self.assertEqual(c.s.workers[w].node, "b")
    self.assertEqual(c.s.workers[w].generations["r"], gen_before + 1)

  def test_lora_runs_share_but_full_or_exclusive_runs_get_their_own_workers(self):
    c = Cluster()
    a, b = c.run("a", rank=8), c.run("b", rank=8)
    others = [c.run("c", rank=8, base="b"), c.run("f"), c.run("x", rank=8, exclusive=True)]
    self.assertEqual(a, b)
    self.assertEqual(len({a, b, *others}), 4)

  def test_single_job_drain_moves_one_job_and_bumps_its_generation_while_sibling_stays(self):
    c = Cluster()
    c.node("n1", 2)
    c.node("n2", 2)
    w = c.run("a", rank=8, queued=1)
    self.assertEqual(c.run("b", rank=8, queued=1), w)
    c.ack_all()
    gen_a, gen_b = c.s.jobs["a"].gen, c.s.jobs["b"].gen
    c.s.workers[w].drain_jobs["a"] = ""
    c.s.workers[w].rev += 1
    c.settle()
    self.assertEqual(c.s.jobs["a"].worker, w)  # waits for drain observation
    c.obs(w, rev=c.s.workers[w].rev, jobs={"a": {"gen": gen_a, "usage": 0.0, "tokens": 0, "state": "drained"}})
    c.settle()
    self.assertNotEqual(c.s.jobs["a"].worker, w)
    self.assertEqual(c.s.jobs["b"].worker, w)
    self.assertEqual(c.s.jobs["a"].gen, gen_a + 1)
    self.assertEqual(c.s.jobs["b"].gen, gen_b)

  def test_random_clusters_settle_idempotently_survive_table_and_never_overcommit(self):
    for seed in range(60):
      rng, c = random.Random(seed), Cluster()
      for _ in range(80):
        roll, name = rng.random(), str(rng.randrange(8))
        if roll < 0.15 and name not in c.nodes:
          c.node(name, rng.choice([1, 2, 4]), memory=rng.choice([0, 80 * GIB]))
        elif roll < 0.22:
          c.lose(name)
        elif roll < 0.72 and name not in c.records:
          c.run(
            name,
            queued=rng.choice([0, 1]),
            rank=rng.choice([0, 8]),
            base=str(rng.randrange(3)),
            role=rng.choice(["trainer", "sampler"]),
            gpus=rng.choice([1, 2]),
            exclusive=rng.random() < 0.15,
          )
        elif roll < 0.85:
          c.ack_all()
        else:
          c.delete(name)
        c.settle()
        c.check_no_gpu_overcommit(self)
        self.assertTrue(all(w.node is None or w.node in c.s.nodes for w in c.s.workers.values()))
        # Idempotency and lossless table round-trip.
        snap = from_table(c.table())
        snap.exclusive_cap = c.s.exclusive_cap
        self.assertEqual(snap, c.s)
        self.assertEqual(decide(snap)[1], [])


class TestFleetAllocation(unittest.TestCase):
  def test_gpu_is_never_reused_until_previous_holder_confirms_release(self):
    c = Cluster()
    c.node("n", 1)
    wa = c.run("a", base="m1", queued=1)
    c.ack_all()
    wb = c.run("b", base="m2", queued=1)
    self.assertEqual((c.s.workers[wa].intent, c.s.workers[wb].intent), ("run", "park"))
    # Exhaust a's budget -> scheduler asks a to park, but must NOT grant GPU 0 to b until a reports gpu_released=True.
    c.add_usage(wa, "a", BUDGET_S + 1.0)
    c.settle()
    self.assertEqual(c.s.workers[wa].intent, "park")
    self.assertEqual(c.s.workers[wb].intent, "park")
    c.check_no_gpu_overcommit(self)
    # Stale rev release does not count; only matching rev + gpu_released unlocks b.
    c.obs(wa, rev=c.s.workers[wa].rev - 1, gpus_held=[], gpu_released=True, phase="parked")
    c.settle()
    self.assertEqual(c.s.workers[wb].intent, "park")
    c.obs(wa, rev=c.s.workers[wa].rev, gpus_held=[], gpu_released=True, phase="parked")
    c.settle()
    self.assertEqual(c.s.workers[wb].intent, "run")
    self.assertEqual(c.s.workers[wb].gpus, (0,))
    c.check_no_gpu_overcommit(self)

  def test_memory_budget_packing_and_unknown_gpu_memory_guard(self):
    c = Cluster()
    c.node("n", 1, memory=81559 * MIB)
    w1 = c.run("s1", role="sampler", base="m1", queued=1)
    w2 = c.run("s2", role="sampler", base="m2", queued=1)
    w3 = c.run("s3", role="sampler", base="m3", queued=1)
    # Two FFT samplers (0.42 * 80GiB each = 0.84) pack onto GPU 0; the third must wait.
    self.assertEqual([c.s.workers[w].intent for w in (w1, w2, w3)], ["run", "run", "park"])
    c.check_no_gpu_overcommit(self)

    # Unknown GPU memory (0) never allows concurrent sharing on one GPU.
    c0 = Cluster()
    c0.node("n", 1, memory=0)
    a = c0.run("s1", role="sampler", base="m1", queued=1)
    b = c0.run("s2", role="sampler", base="m2", queued=1)
    self.assertEqual([c0.s.workers[w].intent for w in (a, b)], ["run", "park"])

  def test_exclusive_admission_cap_dedicated_bundle_and_paired_sampler(self):
    c = Cluster(exclusive_cap=0.5)
    c.node("n", 2)
    ex1 = c.run("ex1", rank=8, exclusive=True, queued=1)
    self.assertEqual(c.s.workers[ex1].node, "n")
    self.assertEqual(c.s.workers[ex1].gpus, (0,))
    # Second exclusive trainer exceeds 50% cap (1 of 2 GPUs) and stays unplaced.
    ex2 = c.run("ex2", rank=8, exclusive=True, queued=1)
    self.assertIsNone(c.s.workers[ex2].node)
    # Paired sampler for admitted ex1 is placed on ex1's node so it never deadlocks.
    ex1_samp = c.run("ex1-s", role="sampler", rank=8, train_run="ex1", exclusive=True, queued=0)
    self.assertEqual(c.s.workers[ex1_samp].node, "n")
    # Regular shared job stays off ex1's dedicated GPU 0 and uses GPU 1.
    reg = c.run("reg", rank=8, queued=1)
    self.assertEqual(c.s.workers[reg].gpus, (1,))
    c.check_no_gpu_overcommit(self)

  def test_wide_job_reservation_prevents_starvation_and_grants_atomically(self):
    c = Cluster()
    c.node("n", 2)
    w1 = c.run("s1", base="m1", queued=1)
    c.ack_all()
    wide = c.run("wide", base="mw", gpus=2, queued=1)
    c.s.service["s1"] = 50.0  # s1 has received more service, so wide has priority
    c.settle()
    self.assertEqual(c.s.workers[wide].reserve_node, "n")
    # Another single-GPU job arriving on GPU 1 must NOT bypass wide's reservation!
    w2 = c.run("s2", base="m2", queued=1)
    self.assertEqual(c.s.workers[w2].intent, "park")
    # Once s1 finishes its budget and confirms release, wide atomically gets (0, 1).
    c.add_usage(w1, "s1", BUDGET_S + 1.0)
    c.settle()
    self.assertEqual(c.s.workers[w1].intent, "park")
    c.ack_all()
    c.settle()
    self.assertEqual(c.s.workers[wide].intent, "run")
    self.assertEqual(c.s.workers[wide].gpus, (0, 1))
    self.assertIsNone(c.s.workers[wide].reserve_node)
    c.check_no_gpu_overcommit(self)

  def test_persisted_fair_service_shared_account_and_idle_floor_snap(self):
    c = Cluster()
    c.node("n", 1)
    wt = c.run("jobA", base="m1", queued=1)
    ws = c.run("jobA-s", role="sampler", base="ms", train_run="jobA", queued=0)
    wb = c.run("jobB", base="m2", queued=1)
    c.ack_all()
    # Usage on jobA's trainer charges logical account "jobA" (shared by jobA-s).
    c.add_usage(wt, "jobA", BUDGET_S + 2.0)
    c.settle()
    self.assertGreaterEqual(c.s.service["jobA"], BUDGET_S + 2.0)
    c.ack_all()
    c.settle()
    # Now jobB (service 0) wins GPU 0 over jobA-s (whose account "jobA" already has service > BUDGET_S).
    c.queue("jobA-s", 1)
    c.settle()
    self.assertEqual(c.s.workers[wb].intent, "run")
    self.assertEqual(c.s.workers[ws].intent, "park")
    # Idle account C waking up snaps to active service floor (cannot bank credit while idle).
    c.run("jobC", base="m3", queued=0)
    c.add_usage(wb, "jobB", 5.0)
    c.settle()
    c.queue("jobC", 1)
    c.settle()
    self.assertGreaterEqual(c.s.service["jobC"], min(c.s.service["jobA"], c.s.service["jobB"]))

  def test_host_ram_pressure_demotes_coldest_parked_trainer(self):
    c = Cluster()
    c.node("n", 1, host_ram=100 * GIB)
    w_cold = c.run("cold", base="m1", queued=1)
    c.ack_all()
    c.queue("cold", 0)
    w_warm = c.run("warm", base="m2", queued=1)
    c.settle()
    c.ack_all()
    c.settle()
    c.ack_all()
    c.queue("warm", 0)
    w_active = c.run("active", base="m3", queued=1)
    c.settle()
    c.ack_all()
    c.settle()
    c.ack_all()
    # Both cold and warm are parked on host RAM; together with active they exceed 80% host RAM.
    c.obs(w_cold, phase="parked", gpu_released=True, gpus_held=[], host_ram_bytes=45 * GIB)
    c.obs(w_warm, phase="parked", gpu_released=True, gpus_held=[], host_ram_bytes=45 * GIB)
    c.nodes["n"]["host_ram_used"] = 90 * GIB
    decisions = c.settle()
    self.assertIn((Ev.DRAINING, {"worker": w_cold, "node": "n", "rule": "drain.host_ram"}), decisions)
    self.assertEqual(c.s.workers[w_cold].intent, "drain")
    self.assertEqual(c.s.workers[w_warm].intent, "park")
    self.assertEqual(c.s.workers[w_active].intent, "run")

  def test_storage_aware_placement_prefetch_hysteresis_and_restore_refs(self):
    c = Cluster()
    c.node("n1", 1, peer="10.0.0.1:8000")
    c.node("n2", 1, peer="10.0.0.2:8000")
    c.versions["ver-big"] = {"locations": {"n2": "10.0.0.2:8000"}, "durable": True, "size_bytes": 5 * GIB}
    w = c.run(
      "r1",
      queued=1,
      meta={"restore_uri": "tinker://r1/weights/c1", "restore_version": "ver-big", "restore_boundary": "1-0"},
    )
    # n2 already has ver-big locally -> placement picks n2 over n1 and populates restore_refs (no prefetch needed).
    self.assertEqual(c.s.workers[w].node, "n2")
    self.assertEqual(c.s.workers[w].restore_refs["r1"]["version"], "ver-big")
    self.assertEqual(c.s.workers[w].prefetch, [])

    # When a version is NOT local on the placed node, scheduler populates prefetch with exact version and peers.
    c.versions["ver-remote"] = {"locations": {"n2": "10.0.0.2:8000"}, "durable": True, "size_bytes": 2 * GIB}
    w2 = c.run(
      "r2",
      base="m2",
      queued=1,
      meta={"restore_uri": "tinker://r2/weights/c1", "restore_version": "ver-remote", "restore_boundary": "1-0"},
    )
    # n2 is busy with r1 (gpu_avail > 0) while n1 is free; check whichever node got it has accurate prefetch/refs.
    self.assertIn("r2", c.s.workers[w2].restore_refs)
    self.assertEqual(c.s.workers[w2].restore_refs["r2"]["version"], "ver-remote")
    if c.s.workers[w2].node == "n1":
      self.assertEqual(len(c.s.workers[w2].prefetch), 1)
      self.assertEqual(c.s.workers[w2].prefetch[0]["version"], "ver-remote")
      self.assertIn("10.0.0.2:8000", c.s.workers[w2].prefetch[0]["peers"])


if __name__ == "__main__":
  unittest.main()
