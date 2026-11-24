import asyncio
import json
import random
import tempfile
import time
import types
import unittest
from dataclasses import replace
from unittest import mock

import torch

from test.execution import InlineProcess, TestNode
from test.helpers import BusTest, FakeEngine, new_id, tiny
from test.test_api import proto_body
from test.test_sampler import JOB
from test.test_trainer import LORA, counting
from tinyrl.api.proto import tinker_public_pb2 as pb
from tinyrl.control import policy
from tinyrl.control.node_daemon import NodeDaemon, WorkerHandle
from tinyrl.control.policy import (
  REBALANCE_MS,
  Demand,
  Placed,
  Worker,
  account_demand,
  admit_samples,
  can_host,
  choose_node,
  node_stats,
  plan_gpus,
  rank_runs,
  ready_lag,
  rebalance_target,
  sampler_replicas,
  update_lag,
  waterfill,
)
from tinyrl.storage.store import Store
from tinyrl.types import (
  MAX_SEQS,
  Cmd,
  Done,
  Effect,
  NodeMsg,
  Op,
  Reply,
  Role,
  Run,
  TaskKind,
  WorkerInfo,
  WorkerSpec,
)
from tinyrl.workers.model import Model

GIB = 2**30


class FleetSim:
  """Drives the real `Controller.schedule()` loop with instant fake nodes to test fleet paging deterministically."""

  def __init__(self):
    from tinyrl.control.controller import Controller

    self.ctrl = Controller(":memory:")
    self.boots: dict[str, str] = {}
    self.pinned_owners: set[str] = set()

  def close(self) -> None:
    self.ctrl.db.close()

  def node(self, name: str, gpus: int) -> None:
    bid = self.boots.setdefault(name, f"b-{name}")
    n = self.ctrl.nodes.get(name)
    if n is not None:
      n.gpus = gpus
      self.settle()
      return
    self.ctrl.handle(NodeMsg("hello", name, bid, gpus=gpus))
    self.settle()

  def lose(self, name: str) -> None:
    if bid := self.boots.pop(name, None):
      self.ctrl.expire_node(name, bid)
    self.settle()

  def run(self, name: str, *, backlog: int = 0, **body) -> str | None:
    role = Role(body.pop("role", "trainer"))
    base = body.pop("base", "a")
    rank = int(body.pop("rank", 0))
    self.ctrl.create_run(Run(name, base, rank, role=role, **body))
    self.queue(name, backlog)
    if role == Role.TRAINER and backlog == 0:
      # Materialize idle trainer onto a worker by running one instant CREATE_MODEL.
      self.ctrl.submit_command(name, Op.CREATE_MODEL, b"{}")
      self._step(run_one_create=name)
    self.settle()
    ws = self.workers_of(name)
    return ws[0] if ws else None

  def queue(self, run: str, n: int) -> None:
    rs = self.ctrl.runs[run]
    if rs.run.role == Role.TRAINER:
      with self.ctrl.db:
        self.ctrl.db.execute("DELETE FROM commands WHERE run_id=?", (run,))
    rs.queue.clear()
    op = Op.SAMPLE if rs.run.role == Role.SAMPLER else Op.OPTIM_STEP
    for _ in range(n):
      self.ctrl.submit_command(run, op, b"{}")

  def serve(self, run: str, gpu_ms: float) -> None:
    acct = self.ctrl.account(run)
    self.ctrl.service[acct] = self.ctrl.service.get(acct, 0.0) + gpu_ms
    self.ctrl.advance_lag()

  def delete(self, run: str) -> None:
    self.queue(run, 0)
    self.ctrl.delete_run(run)

  def workers_of(self, run: str) -> list[str]:
    return self.ctrl.workers_of(run)

  def node_of(self, wid: str) -> str | None:
    for nid, n in self.ctrl.nodes.items():
      if wid in n.workers and not n.workers[wid].stopping:
        return nid
    return None

  def home(self, run: str) -> str | None:
    rs = self.ctrl.runs.get(run)
    if not rs or not rs.live:
      return None
    if rs.owner:
      return rs.owner[0]
    for _nid, n in sorted(self.ctrl.nodes.items()):
      for t in n.tasks.values():
        if t.kind == TaskKind.TRAIN and t.run_id == run:
          return t.worker
    return None

  def _step(self, run_one_create: str | None = None) -> None:
    for _ in range(12):
      for n in self.ctrl.nodes.values():
        for w in n.workers.values():
          w.hold_until = w.idle_until = 0.0
      self.ctrl.lag_due = 0.0
      self.ctrl.schedule()
      progressed = False
      for nid, n in list(self.ctrl.nodes.items()):
        dones: list[Done] = []
        for tid, t in list(n.tasks.items()):
          if t.kind in (TaskKind.SAVE, TaskKind.DROP, TaskKind.STOP, TaskKind.PARK):
            if t.run_id in self.pinned_owners and t.kind in (TaskKind.SAVE, TaskKind.DROP):
              continue
            dones.append(Done(tid, saved=t.kind == TaskKind.SAVE, dead=t.kind == TaskKind.STOP))
          elif t.kind == TaskKind.TRAIN and t.run_id == run_one_create and t.cmd and t.cmd.op == Op.CREATE_MODEL:
            dones.append(
              Done(tid, reply=Reply(True, "application/json", b"{}"), effect=Effect(loaded=True, mutated=True))
            )
          elif t.kind == TaskKind.TRAIN and (rs := self.ctrl.runs.get(t.run_id or "")):
            w = n.workers[t.worker]
            w.started = True
            if rs.owner is None:
              w.loaded[t.run_id] = False
              rs.owner = (t.worker, nid, n.boot_id)
              progressed = True
            elif any(
              o.live and o.run.role == Role.TRAINER and o.owner is None and o.queue for o in self.ctrl.runs.values()
            ):
              progressed = True
            rs.fair_ms = (rs.fair_ms or 0.0) + 1.0
            w.gpu_ms += 1.0
            if it := rs.queue.get(t.cmd.request_id):
              it.owner = None
            n.tasks.pop(tid, None)
            n.new_tasks = [x for x in n.new_tasks if x.id != tid]
          elif t.kind == TaskKind.SAMPLE:
            w = n.workers[t.worker]
            w.started = True
            if it := self.ctrl.runs[t.cmd.run].queue.get(t.cmd.request_id):
              it.owner = None
            n.tasks.pop(tid, None)
            n.new_tasks = [x for x in n.new_tasks if x.id != tid]
            progressed = True
        if dones:
          self.ctrl.handle(NodeMsg("events", nid, n.boot_id, done=tuple(dones)))
          progressed = True
      if not progressed:
        break

  def settle(self) -> None:
    self._step()


def apply_gpus(
  workers: dict[str, Worker],
  quanta: dict[str, float],
  lag_of,
  total: int,
  now: float,
) -> tuple[dict[str, tuple[int, ...]], float | None]:
  ranked = {p: replace(w, lag=lag_of(p) if w.demand != Demand.IDLE else -w.gpu_ms) for p, w in workers.items()}
  keep, wake_at = plan_gpus(ranked, total, now)
  for p, w in list(workers.items()):
    want = keep.get(p, ())
    until = now + quanta[p] if want and want != w.current else w.hold_until
    workers[p] = replace(w, current=want, hold_until=until)
  return keep, wake_at


def check_fleet_invariants(test: unittest.TestCase, f: FleetSim) -> None:
  for n in f.ctrl.nodes.values():
    live = [w for w in n.workers.values() if not w.stopping]
    test.assertLessEqual(sum(w.spec.gpus for w in live if w.spec.exclusive), n.gpus)
    for w in live:
      test.assertLessEqual(w.spec.gpus, n.gpus)
  for _r, rs in f.ctrl.runs.items():
    if rs.live and rs.owner:
      wid, nid, _ = rs.owner
      test.assertEqual(rs.run.role, Role.TRAINER)
      test.assertIn(nid, f.ctrl.nodes)
      test.assertIn(wid, f.ctrl.nodes[nid].workers)


def check_gpu_invariants(test: unittest.TestCase, keep: dict, workers: dict[str, Worker], total: int) -> None:
  seen = []
  for p, gs in keep.items():
    test.assertEqual(len(gs), workers[p].gpus)
    test.assertEqual(len(set(gs)), workers[p].gpus)
    test.assertTrue(all(0 <= g < total for g in gs))
    seen.extend(gs)
  test.assertEqual(len(seen), len(set(seen)))


class TestCluster(unittest.TestCase):
  def make_fleet(self) -> FleetSim:
    f = FleetSim()
    self.addCleanup(f.close)
    return f

  def test_runs_are_assigned_and_empty_workers_stop_after_detach(self):
    f = self.make_fleet()
    f.node("n", 8)
    a = f.run("a", rank=8)
    self.assertEqual(f.run("b", rank=8), a)
    self.assertEqual(f.node_of(a), "n")
    f.delete("a")
    f.settle()
    self.assertEqual(f.node_of(a), "n")
    f.delete("b")
    f.settle()
    self.assertIsNone(f.node_of(a))

  def test_procs_go_to_the_node_with_the_fewest_procs_per_gpu(self):
    f = self.make_fleet()
    f.node("a", 1)
    f.node("b", 2)
    procs = [f.run(name, rank=4, base=name) for name in ("r1", "r2", "r3")]
    self.assertEqual([f.node_of(p) for p in procs], ["a", "b", "b"])

  def test_full_fine_tune_sampler_stays_near_its_training_run(self):
    f = self.make_fleet()
    f.node("a", 1)
    f.node("b", 1)
    a, b = f.run("a"), f.run("b")
    self.assertEqual([f.node_of(p) for p in (a, b)], ["a", "b"])
    sampler = f.run("sample", role="sampler", train_run="b")
    self.assertEqual(f.node_of(sampler), "b")

  def test_full_fine_tune_sampler_goes_elsewhere_when_its_trainers_node_is_full(self):
    f = self.make_fleet()
    f.node("a", 2)
    f.node("b", 8)
    x = f.run("x", exclusive=True)
    self.assertEqual(f.node_of(x), "a")
    y = f.run("y", rank=4, exclusive=True)
    self.assertEqual(f.node_of(y), "b")
    # Fill remaining slot on `a` with an exclusive worker so `x`'s exclusive sampler must spill to `b`.
    f.ctrl.nodes["a"].workers[y] = f.ctrl.nodes["b"].workers.pop(y)
    f.ctrl.runs["y"].owner = (y, "a", f.boots["a"])
    sampler = f.run("sample", role="sampler", train_run="x", exclusive=True)
    self.assertEqual(f.node_of(sampler), "b")

  def test_a_full_fine_tunes_trainer_keeps_room_beside_it_for_its_sampler(self):
    f = self.make_fleet()
    f.node("a", 1)
    f.node("b", 2)
    f.node("c", 8)
    t = f.run("t")
    others = [f.run(f"r{i}", rank=4, base=f"m{i}") for i in range(5)]
    self.assertEqual([f.node_of(p) for p in [t, *others]], ["b", "a", "c", "c", "c", "c"])
    lora = f.run("q", role="sampler", rank=4, base="m0")
    sampler = f.run("sample", role="sampler", train_run="t")
    self.assertEqual([f.node_of(p) for p in (lora, sampler)], ["c", "b"])

  def test_a_second_sampler_stays_near_its_trainer_when_other_free_gpus_are_owed(self):
    f = self.make_fleet()
    f.node("a", 2)
    f.node("b", 8)
    t = f.run("t")
    others = [f.run(f"r{i}") for i in range(4)]
    self.assertEqual([f.node_of(p) for p in [t, *others]], ["a", "b", "b", "b", "b"])
    q = f.run("q", role="sampler", train_run="t")
    g = f.run("g", role="sampler", train_run="t", base="b")
    self.assertEqual([f.node_of(p) for p in (q, g)], ["a", "a"])

  def test_lost_node_and_multi_gpu_waiting(self):
    f = self.make_fleet()
    f.node("a", 1)
    self.assertIsNone(f.run("wide", gpus=2, backlog=1))
    f.node("b", 4)
    p = f.home("wide")
    self.assertIsNotNone(p)
    self.assertEqual(f.node_of(p), "b")
    f.node("c", 2)
    f.ctrl.runs["wide"].status = "saved"  # type: ignore[assignment]
    f.ctrl.runs["wide"].snapshot = "snap"
    f.lose("b")
    p2 = f.home("wide")
    self.assertIsNotNone(p2)
    self.assertEqual(f.node_of(p2), "c")

  def test_lora_runs_share_when_idle_split_when_busy_and_merge_when_squeezed(self):
    f = self.make_fleet()
    f.node("a", 1)
    f.node("b", 1)
    p1 = f.run("r1", rank=8)
    self.assertEqual(f.run("r2", rank=8), p1)
    others = [f.run("r3", rank=8, base="b"), f.run("f"), f.run("x", rank=8, exclusive=True)]
    self.assertEqual(len({p1, *others}), 4)

    f = self.make_fleet()
    f.node("a", 1)
    p1 = f.run("r1", rank=8, backlog=1)
    self.assertEqual(f.run("r2", rank=8, backlog=1), p1)
    f.node("b", 1)
    p2 = f.home("r2")
    self.assertNotEqual(p1, p2)
    self.assertEqual({f.node_of(p1), f.node_of(p2)}, {"a", "b"})
    f.run("excl", exclusive=True, backlog=1)
    f.settle()
    self.assertEqual(f.home("r1"), f.home("r2"))
    check_fleet_invariants(self, f)

  def test_subscribed_node_with_idle_peers_does_not_split_or_scale_out_thrash(self):
    f = self.make_fleet()
    f.node("a", 2)
    s1 = f.run("s1", role="sampler", base="m1")
    f.run("s2", role="sampler", base="m2")
    p1 = f.run("r1", rank=8, backlog=1)
    self.assertEqual(f.run("r2", rank=8, backlog=1), p1)
    self.assertEqual(f.home("r1"), f.home("r2"))
    f.queue("r1", 0)
    f.queue("r2", 0)
    f.queue("s1", 12)
    f.settle()
    self.assertEqual(f.workers_of("s1"), [s1])
    check_fleet_invariants(self, f)

  def test_sampling_sessions_share_only_their_kind_and_exclusive_training_run(self):
    f = self.make_fleet()
    f.node("n", 8)
    f.run("A", rank=8)
    f.run("B", rank=8)
    f.run("X", rank=8, exclusive=True)
    f.run("Z", rank=8, exclusive=True)
    a = f.run("a", role="sampler", rank=8, train_run="A")
    b = f.run("b", role="sampler", rank=8, train_run="B")
    fft = f.run("f", role="sampler", train_run="A")
    x = f.run("x", role="sampler", rank=8, train_run="X", exclusive=True)
    y = f.run("y", role="sampler", rank=8, train_run="X", exclusive=True)
    z = f.run("z", role="sampler", rank=8, train_run="Z", exclusive=True)
    self.assertEqual(a, b)
    self.assertEqual(x, y)
    self.assertEqual(len({a, b, fft, x, y, z}), 4)
    self.assertNotEqual(f.home("X"), f.home("Z"))

  def test_a_sampling_session_without_an_account_bills_its_training_run(self):
    f = self.make_fleet()
    f.ctrl.create_run(Run("t", "a", 0, session="s1"))
    f.ctrl.create_run(Run("q", "a", 0, role=Role.SAMPLER, train_run="t", session="s2"))
    f.ctrl.create_run(Run("base", "a", 0, role=Role.SAMPLER, session="s3"))
    self.assertEqual([f.ctrl.account(r) for r in ("t", "q", "base")], ["s1", "s1", "s3"])

  def test_demand_is_what_each_account_could_run_now(self):
    nodes = {"a": 2, "b": 2}
    runs = [
      ("idle", WorkerSpec(Role.TRAINER, "z", True), 0),
      ("wide", WorkerSpec(Role.TRAINER, "a", False, gpus=4), 1),
      ("t", WorkerSpec(Role.TRAINER, "a", True), 9),
      ("x", WorkerSpec(Role.SAMPLER, "a", False, True, "t"), 9),
      ("s", WorkerSpec(Role.SAMPLER, "m", False), 3 * MAX_SEQS),
      ("big", WorkerSpec(Role.SAMPLER, "n", False), 99 * MAX_SEQS),
    ]
    self.assertEqual(account_demand(runs, nodes), {"t": 1, "x": 1, "s": 3, "big": 4})

  def test_a_backlogged_shared_sampler_gets_a_replica_per_batch_and_an_exclusive_one_does_not(self):
    f = self.make_fleet()
    for node in "abcde":
      f.node(node, 1)
    f.run("t")
    f.run("x", role="sampler", train_run="t", exclusive=True, backlog=3 * MAX_SEQS)
    f.run("s", role="sampler", backlog=3 * MAX_SEQS)
    self.assertEqual((len(f.workers_of("x")), len(set(f.workers_of("s")))), (1, 3))

  def test_unplaced_sampler_pool_does_not_scale_out_onto_busy_single_gpu_node(self):
    f = self.make_fleet()
    f.node("n1", 1)
    f.run("s1", role="sampler", rank=4, backlog=4 * MAX_SEQS, account="alice")
    self.assertEqual(len(f.workers_of("s1")), 1)

  def test_two_busy_sessions_do_not_split_their_lora_sampler_while_fft_scales_out(self):
    f = self.make_fleet()
    f.node("a", 1)
    f.node("b", 1)
    self.assertEqual(f.run("s1", role="sampler", rank=4), f.run("s2", role="sampler", rank=4))
    f.queue("s1", 1)
    f.queue("s2", 1)
    f.settle()
    self.assertEqual(len(f.workers_of("s1")), 1)

    f2 = self.make_fleet()
    f2.node("a", 1)
    f2.node("b", 1)
    f2.run("f1", role="sampler", rank=0, backlog=1)
    f2.run("f2", role="sampler", rank=0, backlog=1)
    self.assertEqual(len(f2.workers_of("f1")), 2)

  def test_idle_lora_trainers_stay_split_while_their_node_has_room(self):
    f = self.make_fleet()
    f.node("a", 2)
    self.assertNotEqual(f.run("r1", rank=4, backlog=1), f.run("r2", rank=4, backlog=1))
    f.queue("r1", 0)
    f.queue("r2", 0)
    f.settle()
    self.assertNotEqual(f.home("r1"), f.home("r2"))

  def test_idle_and_crowded_rebalance_rules(self):
    f = self.make_fleet()
    f.node("a", 1)
    idle = f.run("idle", rank=4, base="x")
    for i in range(2):
      f.run(f"busy{i}", rank=4, base=f"y{i}", backlog=1)
    f.node("b", 1)
    self.assertEqual(f.node_of(idle), "a")

    f2 = self.make_fleet()
    f2.node("a", 1)
    f2.run("idle", rank=4, base="x")
    busy = f2.run("busy", rank=4, base="y", backlog=1)
    f2.node("b", 1)
    self.assertEqual(f2.node_of(busy), "a")

    f3 = self.make_fleet()
    f3.node("a", 2)
    x = f3.run("x", exclusive=True, backlog=1)
    shared = [f3.run(f"r{i}", rank=4, base=f"m{i}", backlog=1) for i in range(2)]
    f3.node("b", 1)
    self.assertEqual(f3.node_of(x), "a")
    self.assertEqual(sorted(f3.node_of(f3.home(f"r{i}")) for i in range(2)), ["a", "b"])
    self.assertEqual(len(shared), 2)

  def test_rotation_weighs_a_mover_against_the_busy_shared_procs_on_the_other_node(self):
    def target(c_lag: float, n2: int = 1, extra: list[Placed] = ()) -> str | None:
      me = Placed("n1", "me", WorkerSpec(Role.TRAINER, "me", True), True, 30000.0)
      placed = [
        me,
        Placed("n1", "b", WorkerSpec(Role.TRAINER, "b", True), True, 25000.0),
        Placed("n2", "c", WorkerSpec(Role.TRAINER, "c", True), True, c_lag),
        *extra,
      ]
      return rebalance_target(me, {"n1": 1, "n2": n2}, placed)

    self.assertEqual(target(5000.0), "n2")
    self.assertIsNone(target(15000.0))
    self.assertIsNone(target(15000.0, extra=[Placed("n2", "d", WorkerSpec(Role.TRAINER, "d", True), False, 0.0)]))
    self.assertIsNone(
      target(15000.0, 2, [Placed("n2", "x", WorkerSpec(Role.TRAINER, "x", False, True), True, -50000.0)])
    )

  def test_only_a_trainer_is_owed_room_for_a_sampler(self):
    f = self.make_fleet()
    f.node("a", 2)
    q = f.run("q", role="sampler")
    f.node("b", 2)
    r1 = f.run("r1", rank=4, base="m1")
    self.assertEqual([f.node_of(q), f.node_of(r1)], ["a", "b"])
    self.assertEqual(f.node_of(f.run("r2", rank=4, base="m2")), "a")

  def test_sampler_settings_follow_the_node(self):
    seen = {}
    for mib in (24 * 1024, 80 * 1024):
      d = NodeDaemon(types.SimpleNamespace(), gpus=1, node="n", gpu_mib=mib, store=Store(root=tempfile.mkdtemp()))
      d.open_sampler = lambda wid, spec, env, m=mib: seen.setdefault(m, env.get("TINYRL_KV_BYTES"))  # type: ignore[assignment]
      d.spawn("s", WorkerSpec(role=Role.SAMPLER, base="b"), (0,))
    self.assertIsNone(seen[24 * 1024])
    self.assertEqual(seen[80 * 1024], str(24 * GIB))

  def test_exclusive_evacuates_squeezed_shareable_procs_and_never_overcommits(self):
    f = self.make_fleet()
    f.node("a", 1)
    f.node("b", 1)
    f.run("ra", backlog=1)
    f.run("rb", backlog=1)
    self.assertEqual({f.node_of(f.home("ra")), f.node_of(f.home("rb"))}, {"a", "b"})
    x = f.run("x", exclusive=True, backlog=1)
    f.settle()
    x_node = f.node_of(x)
    other = "b" if x_node == "a" else "a"
    self.assertEqual((f.node_of(f.home("ra")), f.node_of(f.home("rb"))), (other, other))
    f.ctrl.lag["ra"] = f.ctrl.lag["rb"] = 100 * REBALANCE_MS
    f.settle()
    self.assertEqual((f.node_of(f.home("ra")), f.node_of(f.home("rb"))), (other, other))
    y = f.run("y", exclusive=True, backlog=1)
    z = f.run("z", exclusive=True, backlog=1)
    self.assertEqual({f.node_of(x), f.node_of(y)}, {"a", "b"})
    self.assertIsNone(z)
    check_fleet_invariants(self, f)

  def test_random_clusters_preserve_invariants(self):
    for seed in range(40):
      rng, f = random.Random(seed), FleetSim()
      try:
        for _ in range(40):
          roll, name = rng.random(), f"x{rng.randrange(8)}"
          if roll < 0.15:
            f.node(name, rng.choice([1, 2, 4]))
          elif roll < 0.23:
            f.lose(name)
          elif roll < 0.65 and (name not in f.ctrl.runs or not f.ctrl.runs[name].live):
            role = rng.choice(["trainer", "trainer", "sampler"])
            excl = rng.random() < 0.15 and role == "trainer"
            f.run(
              name,
              role=role,
              rank=rng.choice([0, 8]),
              base=str(rng.randrange(3)),
              gpus=rng.choice([1, 2]),
              exclusive=excl,
              backlog=rng.choice([0, 1, 2, 8]),
            )
          elif roll < 0.82:
            live = [r for r, rs in f.ctrl.runs.items() if rs.live]
            if live:
              r = rng.choice(live)
              f.queue(r, rng.choice([0, 1, 8]))
              if rng.random() < 0.5:
                f.serve(r, rng.choice([500.0, 2000.0, 25_000.0]))
          elif name in f.ctrl.runs and f.ctrl.runs[name].live:
            f.delete(name)
          f.settle()
          check_fleet_invariants(self, f)
      finally:
        f.close()


class TestLocalQuantum(unittest.TestCase):
  def make_daemon(self, gpus: int) -> NodeDaemon:
    store = Store(tempfile.mkdtemp())
    node = NodeDaemon(types.SimpleNamespace(), gpus=gpus, node="n", gpu_mib=0, store=store)
    self.addCleanup(store.close)
    return node

  def test_ready_runs_share_service_even_when_commands_have_different_costs(self):
    heads = [Cmd(run, "1-0", Op.FORWARD_BACKWARD, b"") for run in ("slow", "fast")]
    service = {"slow": 0.0, "fast": 0.0}
    for _ in range(11):
      run = rank_runs(heads, lambda _: 0.0, service)[0]
      service[run] += 10 if run == "slow" else 1
    self.assertEqual(service, {"slow": 10.0, "fast": 10.0})

  def test_equal_service_uses_numeric_stream_arrival_order(self):
    heads = [
      Cmd("later", "10-0", Op.OPTIM_STEP, b""),
      Cmd("same_ms_later", "9-10", Op.OPTIM_STEP, b""),
      Cmd("first", "9-2", Op.OPTIM_STEP, b""),
    ]
    self.assertEqual(rank_runs(heads, lambda _: 0.0, {}), ["first", "same_ms_later", "later"])
    tiny_drift = {"later": 1e-12, "same_ms_later": 0.0, "first": 0.0}
    self.assertEqual(rank_runs(heads, tiny_drift.__getitem__, {})[0], "first")

  def test_account_lag_precedes_run_service_and_arrival(self):
    accts = {"current": "alice", "save": "alice", "waiting": "bob"}
    lag, seen, svc = {"alice": 10.0, "bob": 5.0}, {"alice": 100.0}, {"alice": 110.0}
    heads = [
      Cmd("current", "1-0", Op.OPTIM_STEP, b""),
      Cmd("save", "2-0", Op.SAVE_WEIGHTS_FOR_SAMPLER, b""),
      Cmd("waiting", "3-0", Op.OPTIM_STEP, b""),
    ]
    self.assertEqual(rank_runs(heads, lambda r: ready_lag(lag, seen, svc, accts[r]), {"waiting": 1000})[0], "waiting")

  def test_continuous_sampler_saves_share_a_trainer_with_training_work(self):
    heads = [Cmd("saving", "1-0", Op.SAVE_WEIGHTS_FOR_SAMPLER, b""), Cmd("training", "2-0", Op.OPTIM_STEP, b"")]
    service = {"saving": 0, "training": 0}
    for _ in range(12):
      run = rank_runs(heads, lambda _: 0.0, service)[0]
      service[run] += 3 if run == "saving" else 1
    self.assertEqual(service, {"saving": 9, "training": 9})

  def test_sampler_wave_obeys_account_and_version_limits_and_rotates_within_them(self):
    queues = {
      run: [Cmd(run, f"{i}-{j}", Op.SAMPLE, b"{}") for j in range(3)] for i, run in enumerate(("a1", "a2", "a3", "b1"))
    }
    heads = [q[0] for q in queues.values()]
    versions = {"a1": "v1", "a2": "v1", "a3": "v2", "b1": "v1"}
    # max_loras=1 skips a3 (v2), splits 4 slots fairly 2/2 across accounts a and b, and rotates a1/a2 inside a.
    admitted = admit_samples(heads, lambda r, n: queues[r][:n], lambda r: r[0], lambda r: 0, {}, versions.get, 4, 1)
    self.assertEqual([c.run for c in admitted], ["a1", "a2", "b1", "b1"])
    # Existing in-flight slots for account a tilt new slots toward account b until balanced.
    topped = admit_samples(
      heads, lambda r, n: queues[r][:n], lambda r: r[0], lambda r: 0, {}, versions.get, 3, 2, {"a": 2}
    )
    self.assertEqual([c.run for c in topped], ["a1", "b1", "b1"])

  def test_local_guard_enforces_gpu_exclusivity_under_concurrent_handoffs(self):
    async def run():
      node = self.make_daemon(4)
      active_gpus: dict[str, tuple[int, ...]] = {}
      violations: list[str] = []

      class FakeModel:
        proc = None

        def __init__(self, wid):
          self.wid = wid

        async def resume(self, gpus):
          await asyncio.sleep(0)
          if overlap := {k: v for k, v in active_gpus.items() if k != self.wid and set(v) & set(gpus)}:
            violations.append(f"{self.wid} resumed on {gpus} while {overlap}")
          active_gpus[self.wid] = gpus

        async def park(self):
          await asyncio.sleep(0)
          active_gpus.pop(self.wid, None)

        async def close(self):
          active_gpus.pop(self.wid, None)

      rng = random.Random(7)
      specs = [
        (f"p{i}", WorkerSpec(Role.TRAINER, "b", gpus=w), tuple(range(g, g + w)))
        for i, (g, w) in enumerate([(0, 2), (2, 2), (0, 1), (1, 1), (2, 1), (3, 1)])
      ]

      for _ in range(40):
        cands = {}
        for wid, spec, gpus in rng.sample(specs, 3):
          h = node.workers.setdefault(wid, WorkerHandle(wid, spec, FakeModel(wid)))
          cands[wid] = Worker(spec.gpus, False, Demand.READY, h.resident, gpus, spec.gpus > 1)
        keep, _ = plan_gpus(cands, 4, 1.0)
        for wid, h in list(node.workers.items()):
          if h.resident and keep.get(wid) != h.resident:
            await h.park()
        await asyncio.gather(*(node.workers[wid].ensure_gpu(gpus) for wid, gpus in keep.items()))
        flat = [g for h in node.workers.values() for g in h.resident]
        self.assertEqual(len(flat), len(set(flat)))
      self.assertEqual(violations, [])

    asyncio.run(run())

  def test_active_quantum_prevents_thrash_and_overtime_hands_off_with_zero_wait(self):
    lag, seen, svc = {}, {}, {}
    workers, quanta = {"a": Worker(hold_until=2.0)}, {"a": 2.0}
    for t in (0.0, 0.5, 1.9, 2.5, 10.0):
      keep, wake = apply_gpus(workers, quanta, lambda p, lg=lag, sn=seen: ready_lag(lg, sn, svc, p), 1, t)
      self.assertEqual((keep, wake, workers["a"].hold_until), ({"a": (0,)}, None, 2.0))
      workers["a"] = replace(workers["a"], gpu_ms=workers["a"].gpu_ms + 500.0)
      svc["a"] = svc.get("a", 0.0) + 500.0
      lag, seen = update_lag(lag, seen, svc, 1.0, {"a": 1.0})

    workers["b"], quanta["b"] = Worker(hold_until=2.0), 2.0
    keep, wake = apply_gpus(workers, quanta, lambda p, lg=lag, sn=seen: ready_lag(lg, sn, svc, p), 1, 10.0)
    self.assertEqual((keep, wake), ({"b": (0,)}, None))
    for t in (10.1, 10.4, 10.9, 11.5, 11.9):
      keep, wake = apply_gpus(workers, quanta, lambda p, lg=lag, sn=seen: ready_lag(lg, sn, svc, p), 1, t)
      self.assertEqual((keep, wake, workers["b"].hold_until), ({"b": (0,)}, 12.0, 12.0))
    workers["b"] = replace(workers["b"], demand=Demand.IDLE)
    keep, wake = apply_gpus(workers, quanta, lambda p, lg=lag, sn=seen: ready_lag(lg, sn, svc, p), 1, 11.95)
    self.assertEqual((keep, wake), ({"a": (0,)}, None))
    workers["a"] = replace(workers["a"], demand=Demand.IDLE)
    keep, wake = apply_gpus(workers, quanta, lambda p, lg=lag, sn=seen: ready_lag(lg, sn, svc, p), 1, 12.1)
    self.assertEqual((keep, wake), ({"a": (0,)}, None))

  def test_asymmetric_switch_cost_and_account_grouping_share_gpu_ms_fairly(self):
    lag, seen, svc = {}, {}, {}
    quanta = {"train": 1.0, "samp": 6.0}
    workers = {p: Worker(hold_until=q) for p, q in quanta.items()}
    switches, last, dt = 0, None, 0.5
    for i in range(240):
      keep, _ = apply_gpus(workers, quanta, lambda p, lg=lag, sn=seen: ready_lag(lg, sn, svc, p), 1, i * dt)
      [h] = keep
      switches += h != last
      last = h
      workers[h] = replace(workers[h], gpu_ms=workers[h].gpu_ms + dt * 1000)
      svc[h] = svc.get(h, 0.0) + dt * 1000
      lag, seen = update_lag(lag, seen, svc, 1.0, {"train": 1.0, "samp": 1.0})
    self.assertAlmostEqual(svc["train"] / sum(svc.values()), 0.5, delta=0.05)
    self.assertLessEqual(switches, 24)

    lag, seen, svc = {}, {}, {}
    acct = {"t1": "alice", "s1": "alice", "t2": "bob"}
    quanta = {"t1": 1.0, "s1": 4.0, "t2": 1.0}
    workers = {p: Worker(hold_until=q) for p, q in quanta.items()}
    for i in range(240):
      keep, _ = apply_gpus(workers, quanta, lambda p, lg=lag, sn=seen: ready_lag(lg, sn, svc, acct[p]), 1, i * dt)
      [h] = keep
      workers[h] = replace(workers[h], gpu_ms=workers[h].gpu_ms + dt * 1000)
      svc[acct[h]] = svc.get(acct[h], 0.0) + dt * 1000
      lag, seen = update_lag(lag, seen, svc, 1.0, {"alice": 2.0, "bob": 1.0})
    self.assertAlmostEqual(svc["alice"] / (svc["alice"] + svc["bob"]), 0.5, delta=0.06)
    self.assertAlmostEqual(workers["t1"].gpu_ms / (workers["t1"].gpu_ms + workers["s1"].gpu_ms), 0.5, delta=0.08)

  def test_gang_hold_prevents_leapfrog_starvation_and_achieves_waterfill_shares(self):
    quanta = dict.fromkeys(("a", "b", "d", "wide"), 2.0)
    workers = {
      "a": Worker(current=(0,), hold_until=2.0),
      "b": Worker(current=(1,), hold_until=3.0),
      "d": Worker(hold_until=2.0),
      "wide": Worker(gpus=2, hold_until=2.0),
    }
    lag = {"wide": 5000.0, "d": 1000.0, "a": 0.0, "b": 0.0}
    keep, wake = apply_gpus(workers, quanta, lag.__getitem__, 2, 2.2)
    self.assertEqual((keep, wake, workers["a"].hold_until), ({"a": (0,), "b": (1,)}, 3.0, 2.0))
    keep, _ = apply_gpus(workers, quanta, lag.__getitem__, 2, 3.0)
    self.assertEqual(keep, {"wide": (0, 1)})

    lag, seen, svc = {}, {}, {}
    quanta = dict.fromkeys(("a", "b", "wide"), 1.0)
    workers = {"a": Worker(hold_until=1.0), "b": Worker(hold_until=1.0), "wide": Worker(gpus=2, hold_until=1.0)}
    both_ticks = gang_ticks = 0
    for i in range(180):
      keep, _ = apply_gpus(workers, quanta, lambda p, lg=lag, sn=seen: ready_lag(lg, sn, svc, p), 2, i * 0.5)
      check_gpu_invariants(self, keep, workers, 2)
      gang_ticks += keep == {"wide": (0, 1)}
      both_ticks += "a" in keep and "b" in keep
      for h, gs in keep.items():
        ms = 500.0 * len(gs)
        workers[h] = replace(workers[h], gpu_ms=workers[h].gpu_ms + ms)
        svc[h] = svc.get(h, 0.0) + ms
      lag, seen = update_lag(lag, seen, svc, 2.0, {"a": 1.0, "b": 1.0, "wide": 2.0})
    self.assertGreater(gang_ticks, 25)
    self.assertGreater(both_ticks, 25)
    total = sum(svc.values())
    for a in ("a", "b", "wide"):
      self.assertAlmostEqual(svc[a] / total, 1 / 3, delta=0.05)

  def test_random_local_quantum_schedules_never_overlap_or_violate_exclusives(self):
    for seed in range(100):
      rng, now = random.Random(seed), 0.0
      total = rng.choice([1, 2, 4])
      workers, quanta = {}, {}
      for i in range(6):
        w = rng.choice([1, 1, 2]) if total >= 2 else 1
        quanta[f"p{i}"] = q = rng.choice([0.5, 2.0])
        workers[f"p{i}"] = Worker(gpus=w, exclusive=rng.random() < 0.15, hold_until=q)
      for _ in range(60):
        now += rng.uniform(0.0, 1.5)
        for p, w in list(workers.items()):
          demand = Demand.READY if rng.random() < 0.7 else Demand.IDLE
          ms = w.gpu_ms + (rng.uniform(10.0, 500.0) if w.current else 0.0)
          workers[p] = replace(w, demand=demand, gpu_ms=ms)
        keep, wake = apply_gpus(workers, quanta, lambda _: 0.0, total, now)
        check_gpu_invariants(self, keep, workers, total)
        if wake is not None:
          self.assertGreater(wake, now)


class TestMultiNodeFairness(unittest.TestCase):
  def test_waterfill_equal_capped_and_weighted(self):
    self.assertEqual(waterfill(2, {"a": 1, "b": 1, "c": 1}), {"a": 2 / 3, "b": 2 / 3, "c": 2 / 3})
    self.assertEqual(waterfill(4, {"a": 1, "b": 4, "c": 4}), {"a": 1.0, "b": 1.5, "c": 1.5})
    self.assertEqual(waterfill(4, {"a": 4, "b": 4}, {"a": 3.0, "b": 1.0}), {"a": 3.0, "b": 1.0})
    self.assertEqual(waterfill(8, {"a": 1, "b": 2}), {"a": 1.0, "b": 2.0})

  def test_update_lag_conserves_service_blocks_flicker_and_handles_async_deltas(self):
    lag, seen = update_lag({}, {}, {"a": 300.0, "b": 0.0}, 1.0, {"a": 1.0, "b": 1.0})
    self.assertAlmostEqual(lag["a"], -150.0)
    self.assertAlmostEqual(lag["b"], 150.0)

    lag, seen = update_lag({}, {}, {"a": 200.0, "b": 0.0}, 1.0, {"b": 1.0})
    self.assertAlmostEqual(lag["a"], -100.0)
    self.assertAlmostEqual(lag["b"], 100.0)

    for _ in range(5):
      lag, seen = update_lag(lag, seen, {"a": 200.0, "b": 0.0}, 1.0, {"a": 1.0, "b": 1.0})
      lag, seen = update_lag(lag, seen, {"a": 200.0, "b": 0.0}, 1.0, {"b": 1.0})
    self.assertAlmostEqual(lag["a"], -100.0)

    lag2, _ = update_lag(lag, seen, {"a": 200.0, "b": 0.0}, 1.0, {})
    self.assertNotIn("b", lag2)
    lag, seen = update_lag(lag, seen, {"a": 200.0, "b": 200.0}, 1.0, {"b": 1.0})
    self.assertNotIn("a", lag)

    lag, seen = update_lag({}, {}, {"a": 100.0, "b": 0.0}, 1.0, {"a": 1.0, "b": 1.0})
    lag, seen = update_lag(lag, seen, {"a": 250.0, "b": 50.0}, 1.0, {"a": 1.0, "b": 1.0})
    self.assertAlmostEqual(lag["a"], -100.0)
    self.assertAlmostEqual(lag["b"], 100.0)
    lag, seen = update_lag(lag, seen, {"a": 250.0, "b": 80.0}, 1.0, {"a": 1.0})
    self.assertAlmostEqual(lag["a"], -85.0)
    self.assertAlmostEqual(lag["b"], 85.0)

  def simulate(
    self,
    nodes: dict[str, int],
    placement: dict[str, str],
    accts: dict[str, str],
    steps: int,
    *,
    weights: dict[str, float] | None = None,
    dt: float = 0.5,
    quantum: float = 1.0,
  ) -> tuple[int, dict[str, float], dict[str, str]]:
    """Co-simulate per-node `plan_gpus` and cross-node `rebalance_target` + `update_lag`."""
    placement = dict(placement)
    workers: dict[str, tuple[str, Worker]] = {}
    lag, seen, svc = {}, {}, {}
    demand: dict[str, float] = {}
    for p in placement:
      demand[accts[p]] = demand.get(accts[p], 0.0) + 1.0
    moves = 0
    for i in range(steps):
      now = i * dt
      for n, total in nodes.items():
        local = {}
        for p, pn in placement.items():
          if pn == n:
            local[p] = workers[p][1] if p in workers and workers[p][0] == n else Worker(hold_until=quantum)
        keep, _ = apply_gpus(
          local, dict.fromkeys(local, quantum), lambda p, lg=lag, sn=seen: ready_lag(lg, sn, svc, accts[p]), total, now
        )
        for p, w in local.items():
          if keep.get(p):
            ms = dt * 1000.0 * len(keep[p])
            w = replace(w, gpu_ms=w.gpu_ms + ms)
            svc[accts[p]] = svc.get(accts[p], 0.0) + ms
          workers[p] = (n, w)
      lag, seen = update_lag(lag, seen, svc, float(sum(nodes.values())), demand, weights)
      placed = [
        Placed(pn, p, WorkerSpec(Role.TRAINER, p, True), True, ready_lag(lag, seen, svc, accts[p]))
        for p, pn in sorted(placement.items())
      ]
      for pl in sorted(placed, key=lambda x: (-x.lag, x.worker)):
        if dst := rebalance_target(pl, nodes, placed):
          placement[pl.worker] = dst
          moves += 1
          break
    return moves, svc, placement

  def test_four_on_two_weighted_and_sybil_settle_without_cross_node_thrash(self):
    moves, svc, place = self.simulate(
      {"a": 1, "b": 1}, {"r0": "a", "r1": "a", "r2": "a", "r3": "b"}, {f"r{i}": f"r{i}" for i in range(4)}, 240
    )
    self.assertEqual(moves, 1)
    self.assertEqual(sorted(sum(v == n for v in place.values()) for n in ("a", "b")), [2, 2])
    total = sum(svc.values())
    for i in range(4):
      self.assertAlmostEqual(svc[f"r{i}"] / total, 0.25, delta=0.05)

    moves, _, _ = self.simulate(
      {"a": 1, "b": 1},
      {"r0": "a", "r1": "a", "r2": "b"},
      {f"r{i}": f"r{i}" for i in range(3)},
      200,
      weights={"r0": 1.0, "r1": 1.0, "r2": 2.0},
    )
    self.assertEqual(moves, 0)

    moves, svc, _ = self.simulate(
      {"a": 1, "b": 1},
      {"s0": "a", "s1": "a", "s2": "a", "b0": "b"},
      {"s0": "alice", "s1": "alice", "s2": "alice", "b0": "bob"},
      200,
    )
    self.assertEqual(moves, 0)
    self.assertAlmostEqual(svc["alice"] / (svc["alice"] + svc["bob"]), 0.5, delta=0.02)

  def test_three_busy_trainers_on_two_nodes_rotate_to_one_third_each(self):
    with mock.patch.object(policy, "REBALANCE_MS", 5_000):
      moves, svc, _ = self.simulate(
        {"a": 1, "b": 1}, {"r0": "a", "r1": "b", "r2": "a"}, {f"r{i}": f"r{i}" for i in range(3)}, 360
      )
    total = sum(svc.values())
    for i in range(3):
      self.assertAlmostEqual(svc[f"r{i}"] / total, 1 / 3, delta=0.04)
    self.assertTrue(4 <= moves <= 30, moves)

  def test_pure_placement_helpers(self):
    spec = WorkerSpec(Role.SAMPLER, "b", True)
    self.assertEqual(sampler_replicas(spec, [0, 0]), 1)
    self.assertEqual(sampler_replicas(WorkerSpec(Role.SAMPLER, "b", True, True, "t"), [99 * MAX_SEQS]), 1)
    placed = [Placed("a", "w1", WorkerSpec(Role.TRAINER, "x", False, True), True)]
    self.assertEqual(node_stats("a", 2, placed).spare, 1)
    self.assertFalse(can_host("missing", {"a": 2}, placed, spec))
    self.assertEqual(choose_node(spec, {"a": 2}, placed), "a")


class TestMultiNodeExecution(BusTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.bucket = f"file://{tempfile.mkdtemp()}"
    self.nodes: list[TestNode] = []
    self.tasks: list[asyncio.Task] = []

  async def asyncTearDown(self):
    for task in self.tasks:
      task.cancel()
    await asyncio.gather(*self.tasks, return_exceptions=True)
    for node in self.nodes:
      await node.aclose()
    await super().asyncTearDown()

  async def add_node(self, name: str, gpus: int = 1, engine=FakeEngine) -> TestNode:
    store = Store(self.bucket, cache=tempfile.mkdtemp())
    node = TestNode(self.bus, store=store, gpus=gpus, node=name, engine=engine)
    self.nodes.append(node)
    await node.tick()

    async def loop():
      while True:
        await node.tick()
        await asyncio.sleep(0.01)

    self.tasks.append(asyncio.create_task(loop()))
    return node

  def start_scheduler(self) -> None:
    self.start_maintenance(0.01)

  async def submit(self, run: str, op: Op, body: bytes = b"", cost: int = 100) -> str:
    req = self.bus.submit_command(run, op, body, cost=cost)
    assert req is not None
    return req

  async def run_cmd(self, run: str, op: Op, body: bytes = b"", cost: int = 100) -> bytes:
    reply = await self.bus.wait_for_reply(await self.submit(run, op, body, cost), 10)
    self.assertTrue(reply.ok, reply.body)
    return reply.body

  async def create(self, run: str, rank: int = 0, lora: dict | None = None, **kw) -> None:
    cfg = {} if lora is None else lora
    self.bus.create_run(Run(run, "tiny", rank, lora=cfg, **kw))
    await self.submit(run, Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": rank, "lora": cfg}).encode())

  async def until(self, pred, timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
      while not await pred():
        await asyncio.sleep(0.01)

  def live_params(self, run: str) -> dict[str, torch.Tensor]:
    for node in self.nodes:
      for proc, h in node.workers.items():
        if run in h.loaded and proc in node.processes:
          return node.processes[proc].trainer.model.params[run]
    raise AssertionError(f"no live owner found for {run}")

  async def test_idle_maintain_writes_zero_sql_and_accounts_flush_only_when_dirty(self):
    self.bus.handle(NodeMsg("hello", "n", "bn", gpus=1, workers=(WorkerInfo("w", WorkerSpec(Role.TRAINER, "tiny")),)))
    self.bus.maintain()
    flushes, orig = 0, self.bus.flush_sql

    def count():
      nonlocal flushes
      flushes += 1
      orig()

    with mock.patch.object(self.bus, "flush_sql", count):
      self.bus.maintain()
      self.bus.maintain()
      self.assertEqual(flushes, 0)
      self.bus.service["acct"] = 5.0
      self.bus.maintain()  # records delta
      self.bus.maintain()  # clears zero-demand entry
      self.assertEqual(flushes, 2)
      self.bus.maintain()  # steady-state idle: zero further SQL writes
      self.bus.maintain()
      self.assertEqual(flushes, 2)

  async def test_new_node_rebalances_mid_accumulation_with_exact_model_parameters(self):
    opt = json.dumps({"adam_params": {"learning_rate": 1e-2}}).encode()
    await self.add_node("node-a", gpus=1)
    self.start_scheduler()
    r1, r2, ctrl = new_id(), new_id(), new_id()
    for r in (r1, r2):
      await self.create(r)
      await self.run_cmd(r, Op.FORWARD_BACKWARD, proto_body(r, 10))
      await self.run_cmd(r, Op.OPTIM_STEP, opt)
      await self.run_cmd(r, Op.FORWARD_BACKWARD, proto_body(r, 10))

    await self.add_node("node-b", gpus=1)
    reqs = [
      (await self.submit(r, Op.FORWARD_BACKWARD, proto_body(r, 8)), await self.submit(r, Op.OPTIM_STEP, opt))
      for r in (r1, r2)
    ]
    for a, b in reqs:
      self.assertTrue((await self.bus.wait_for_reply(a, 10)).ok)
      self.assertTrue((await self.bus.wait_for_reply(b, 10)).ok)

    snap = self.bus.snapshot_scheduling()["state"]
    self.assertEqual({snap["procs"][snap["home"][r]]["node"] for r in (r1, r2)}, {"node-a", "node-b"})

    ref = Model(tiny)
    ref.create_run(ctrl, "tiny", 0, {})
    for n, step_after in ((10, True), (10, False), (8, True)):
      ref.forward_backward(ctrl, [counting(n, 0), counting(n, 40)], "cross_entropy", {}, False)
      if step_after:
        ref.optim_step(ctrl, {"learning_rate": 1e-2})
    expected, _ = ref.training_state(ctrl)
    for r in (r1, r2):
      uri = f"tinker://{r}/weights/after-rebalance"
      await self.run_cmd(r, Op.SAVE_WEIGHTS, json.dumps({"uri": uri}).encode())
      version = self.bus.checkpoint_version(uri)
      actual, _ = await asyncio.to_thread(self.nodes[0].store.get, version)
      torch.testing.assert_close(actual, expected)

  async def test_ready_trainer_runs_while_sampler_stays_backlogged_on_one_gpu(self):
    sampling, release = asyncio.Event(), asyncio.Event()

    class BusySampler(FakeEngine):
      async def sample(self, job, name):
        sampling.set()
        await release.wait()
        await asyncio.sleep(0.01)
        return await super().sample(job, name)

    with (
      mock.patch("tinyrl.control.policy.QUANTUM_MS", 15),
      mock.patch("tinyrl.types.MAX_SEQS", 2),
    ):
      await self.add_node("shared", gpus=1, engine=BusySampler)
      self.start_scheduler()
      await self.create("train", account="team")
      await self.run_cmd("train", Op.FORWARD_BACKWARD, proto_body("train", 6))
      self.bus.create_run(Run("sample", "tiny", 4, role=Role.SAMPLER, account="team"))
      for _ in range(64):
        await self.submit("sample", Op.SAMPLE, json.dumps(JOB).encode())
      try:
        await asyncio.wait_for(sampling.wait(), 5)
        requests = [await self.submit("train", Op.OPTIM_STEP, b'{"adam_params":{}}') for _ in range(3)]
      finally:
        release.set()
      for request in requests:
        self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
      self.assertGreater((self.bus.queued_and_in_flight("sample"))["queued"], 0)

  async def test_shared_trainer_splits_gpu_time_by_account_not_run_count(self):
    node = await self.add_node("single-gpu")
    accounts = {new_id(): account for account in ("alice", "alice", "alice", "bob")}
    observed = {account: 0.0 for account in accounts.values()}
    completed = []
    original = InlineProcess.call

    async def timed_call(process, method, *args):
      if method != "execute" or args[0].op != Op.OPTIM_STEP:
        return await original(process, method, *args)
      run = args[0].run
      start = time.monotonic()
      await asyncio.sleep(0.012 if accounts[run] == "alice" else 0.004)
      result = await original(process, method, *args)
      observed[accounts[run]] += time.monotonic() - start
      completed.append(run)
      return result

    self.start_scheduler()
    for run, account in accounts.items():
      self.bus.create_run(Run(run, "tiny", 4, lora=LORA, account=account))
      await self.run_cmd(run, Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 4, "lora": LORA}).encode())
    with mock.patch.object(InlineProcess, "call", timed_call):
      for run in accounts:
        for _ in range(100):
          self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
      async with asyncio.timeout(30):
        while len(completed) < 80:
          await asyncio.sleep(0.005)

    self.assertEqual(len(node.processes), 1)
    self.assertEqual(set(completed), set(accounts))
    self.assertAlmostEqual(observed["alice"] / sum(observed.values()), 0.5, delta=0.2)

  async def test_lora_runs_split_onto_new_node_and_merge_back_with_exact_weights(self):
    opt = json.dumps({"adam_params": {"learning_rate": 1e-2}}).encode()
    await self.add_node("node-a", gpus=1)
    self.start_scheduler()
    r1, r2 = new_id(), new_id()
    for r in (r1, r2):
      await self.create(r, rank=4, lora=LORA)
      await self.run_cmd(r, Op.FORWARD_BACKWARD, proto_body(r, 10))
      await self.run_cmd(r, Op.OPTIM_STEP, opt)
    s0 = self.bus.snapshot_scheduling()["state"]
    self.assertEqual(s0["home"][r1], s0["home"][r2])

    await self.add_node("node-b", gpus=1)
    futs = [
      (await self.submit(r, Op.FORWARD_BACKWARD, proto_body(r, 9)), await self.submit(r, Op.OPTIM_STEP, opt))
      for r in (r1, r2)
    ]
    for a, b in futs:
      self.assertTrue((await self.bus.wait_for_reply(a, 10)).ok)
      self.assertTrue((await self.bus.wait_for_reply(b, 10)).ok)
    s1 = self.bus.snapshot_scheduling()["state"]
    self.assertNotEqual(s1["home"][r1], s1["home"][r2])
    self.assertEqual({s1["procs"][s1["home"][r]]["node"] for r in (r1, r2)}, {"node-a", "node-b"})

    excl = new_id()
    await self.create(excl, exclusive=True)
    await self.run_cmd(excl, Op.FORWARD_BACKWARD, proto_body(excl, 6))

    async def merged() -> bool:
      # Trigger a command on the evicted run so it page-faults onto the surviving peer trainer.
      h = self.bus.snapshot_scheduling()["state"]["home"]
      return h.get(r1) is not None and h.get(r1) == h.get(r2)

    for r in (r1, r2):
      await self.run_cmd(r, Op.FORWARD_BACKWARD, proto_body(r, 11))
      await self.run_cmd(r, Op.OPTIM_STEP, opt)
    await self.until(merged)

    ref = Model(tiny)
    for r in (r1, r2):
      ref.create_run(r, "tiny", 4, LORA)
      for n in (10, 9, 11):
        ref.forward_backward(r, [counting(n, 0), counting(n, 40)], "cross_entropy", {}, False)
        ref.optim_step(r, {"learning_rate": 1e-2})
      torch.testing.assert_close(self.live_params(r), ref.params[r])

  async def test_stateless_sampler_scales_out_across_nodes_recovers_crash_and_scales_in(self):
    served_by: list[int] = []

    class SlowEngine(FakeEngine):
      async def sample(self, job: dict, name: str | None):
        served_by.append(id(self))
        await asyncio.sleep(0.05)
        return await super().sample(job, name)

    await self.add_node("node-a", gpus=1, engine=SlowEngine)
    await self.add_node("node-b", gpus=1, engine=SlowEngine)
    self.start_scheduler()

    sid = new_id()
    self.bus.create_run(Run(sid, "tiny", 4, role=Role.SAMPLER, account="team-s"))
    total_cmds = 12
    payload = json.dumps(JOB).encode()
    with mock.patch("tinyrl.types.MAX_SEQS", 2):
      reqs = [await self.submit(sid, Op.SAMPLE, payload, cost=50) for _ in range(total_cmds)]

      async def scaled_out() -> bool:
        return len(self.bus.workers_of(sid)) >= 2 and len(set(served_by)) >= 2

      await self.until(scaled_out)
      for req in reqs:
        reply = await self.bus.wait_for_reply(req, 10)
        self.assertTrue(reply.ok)
        self.assertEqual(len(pb.SampleResponse.FromString(reply.body).sequences), 1)

    self.assertEqual((self.bus.get_run(sid))["tokens"], total_cmds * 50)
    svc_before = self.bus.service["team-s"]
    self.assertGreaterEqual(svc_before, total_cmds * 20.0)

    async def scaled_in() -> bool:
      return len(self.bus.workers_of(sid)) == 1

    await self.until(scaled_in)

    snap = self.bus.snapshot_scheduling()["state"]
    active_node = snap["procs"][self.bus.workers_of(sid)[0]]["node"]
    req = self.bus.submit_command(sid, Op.SAMPLE, payload, cost=50)
    self.bus.nodes[active_node].lease_until = 0.0
    self.assertTrue((await self.bus.wait_for_reply(req, 5)).ok)
    self.assertEqual((self.bus.get_run(sid))["tokens"], (total_cmds + 1) * 50)
    self.assertGreater(self.bus.service["team-s"], svc_before)

  async def test_controller_enforces_gpu_exclusivity_oversized_rejection_and_single_writer(self):
    node_a = await self.add_node("node-a", gpus=1)
    node_b = await self.add_node("node-b", gpus=1)
    self.start_scheduler()
    r1, r2, gang = new_id(), new_id(), new_id()
    await self.create(r1)
    await self.create(r2)
    await self.create(gang, gpus=2)

    fb, opt = proto_body(r1, 10), json.dumps({"adam_params": {"learning_rate": 1e-2}}).encode()
    futs = []
    for _ in range(3):
      futs.append(await self.submit(r1, Op.FORWARD_BACKWARD, fb))
      futs.append(await self.submit(r1, Op.OPTIM_STEP, opt))

    for req in futs:
      async with asyncio.timeout(10):
        while (await self.bus.wait_for_reply(req, 0)) is None:
          for node in (node_a, node_b):
            flat = [g for h in node.workers.values() for g in h.resident]
            self.assertEqual(len(flat), len(set(flat)))
            self.assertTrue(all(h.spec.gpus <= node.gpus for h in node.workers.values()))
          holders = [f"{n.id}:{p}" for n in (node_a, node_b) for p, h in n.workers.items() if r1 in h.loaded]
          self.assertLessEqual(len(holders), 1)
          await asyncio.sleep(0.005)
      self.assertTrue((await self.bus.wait_for_reply(req, 5)).ok)

    ref, ctrl = Model(tiny), new_id()
    ref.create_run(ctrl, "tiny", 0, {})
    for _ in range(3):
      ref.forward_backward(ctrl, [counting(10, 0), counting(10, 40)], "cross_entropy", {}, False)
      ref.optim_step(ctrl, {"learning_rate": 1e-2})
    torch.testing.assert_close(self.live_params(r1), ref.params[ctrl])

  async def test_local_quantum_shares_one_gpu_and_overtime_hands_off_immediately(self):
    await self.add_node("node-a", gpus=1)
    self.start_scheduler()
    a, b = new_id(), new_id()
    before = time.monotonic()
    await self.create(a, account="alice")
    await self.run_cmd(a, Op.FORWARD_BACKWARD, proto_body(a, 6))
    pa = self.bus.snapshot_scheduling()["state"]["home"][a]
    cw_a = self.bus.nodes["node-a"].workers[pa]
    self.assertGreaterEqual(cw_a.hold_until, before)
    cw_a.hold_until = expired = time.monotonic() - 1.0
    await self.run_cmd(a, Op.FORWARD_BACKWARD, proto_body(a, 6))
    self.assertEqual(cw_a.hold_until, expired)
    await self.create(b, account="bob")
    t0 = time.monotonic()
    fa = [await self.submit(a, Op.FORWARD_BACKWARD, proto_body(a, 6)) for _ in range(2)]
    fb = await self.submit(b, Op.FORWARD_BACKWARD, proto_body(b, 6))
    self.assertTrue((await self.bus.wait_for_reply(fb, 5)).ok)
    self.assertLess(time.monotonic() - t0, 1.5)
    for req in fa:
      self.assertTrue((await self.bus.wait_for_reply(req, 5)).ok)
    self.assertGreater(self.bus.service["alice"], 0.0)
    self.assertGreater(self.bus.service["bob"], 0.0)


if __name__ == "__main__":
  unittest.main()
