"""Stress the control plane: several accounts, runs, samplers and nodes on few GPUs, with rebalancing and optional chaos.

  SEED=3 CHAOS=1 python -m test.stress_fleet

SEED picks the cluster and the workload. CHAOS=1 also cancels workers, kills model processes, loses, drains and adds
nodes, and cuts nodes off from Redis while clients run. It is not part of the suite: run a range of seeds by hand.

Half of each client's training cycles are pipelined, like the SDK's futures. A second Model replays exactly what each
client was told succeeded, and at the end each run's saved state must equal it: the system may fail a command, but it
may not lose or invent an acknowledged one. It also checks that every command gets a reply, that no two processes are
ever on one GPU, and that deleting every run leaves no lease, owner, queue or write intent behind.
"""

import asyncio
import collections
import copy
import itertools
import json
import os
import random
import sys
import tempfile
import time
import unittest
from unittest import mock

import torch

from test.execution import InlineProcess, TestNode
from test.helpers import FakeEngine, RedisTest, new_id, tiny
from test.test_api import proto_body
from test.test_sampler import JOB
from test.test_trainer import LORA, counting
from tinyrl.bus import Bus, Op, Role, Run, connect, key
from tinyrl.control import scheduler
from tinyrl.storage.store import Store
from tinyrl.workers.model import Model
from tinyrl.workers.trainer import Trainer

SEED, CHAOS, STEPS = int(os.getenv("SEED", 0)), os.getenv("CHAOS", "0") == "1", int(os.getenv("STEPS", 8))
BLOCK = float(os.getenv("BLOCK", 0))
WAIT = float(os.getenv("WAIT", 60))  # seconds a client waits for one reply before calling it hung
OPT = json.dumps({"adam_params": {"learning_rate": 1e-2}}).encode()
serial = itertools.count()
if (
  os.getenv("THREADS", "1") != "0"
):  # THREADS=0 leaves torch's default, which starves the machine when seeds run side by side
  torch.set_num_threads(int(os.getenv("THREADS", "1")))


class Unreachable:
  """A Redis that a node can no longer reach."""

  def __getattr__(self, name):
    def call(*args, **kwargs):
      raise ConnectionError(f"redis is unreachable ({name})")

    return call


class Occupancy:
  """Which processes are physically on each of a node's GPUs, as the processes themselves see it."""

  def __init__(self, node: str, violations: list[str]):
    self.node, self.on, self.violations = node, {}, violations

  def enter(self, who: str, gpus: tuple[int, ...]) -> None:
    if clash := {o: g for o, g in self.on.items() if o != who and set(g) & set(gpus)}:
      self.violations.append(f"{self.node}: {who} resumed on {gpus} while {clash} still there")
    self.on[who] = tuple(gpus)

  def leave(self, who: str) -> None:
    self.on.pop(who, None)


class TrackedProcess(InlineProcess):
  def __init__(self, trainer, occ: Occupancy, name: str, gpus: tuple[int, ...], delay):
    super().__init__(trainer)
    self.occ, self.name, self.gpus, self.delay = occ, name, gpus, delay

  async def call(self, method, *args):
    if self.closed:
      raise EOFError("the model process exited")
    if method == "resume":
      self.occ.enter(self.name, (args[0],) if args and args[0] is not None else self.gpus)
    if method == "execute":
      if self.name not in self.occ.on:
        self.occ.violations.append(f"{self.occ.node}: {self.name} executed {args[0].op} off its GPUs")
      await asyncio.sleep(self.delay())  # the GPU time a command takes
      if BLOCK:
        time.sleep(BLOCK * self.delay() * 50)  # a starved machine: the model call stalls the node's whole event loop
    result = await super().call(method, *args)
    if method == "park":
      self.occ.leave(self.name)
    return result

  async def close(self):
    self.occ.leave(self.name)
    await super().close()


def tracked_engine(occ: Occupancy, delay):
  class Engine(FakeEngine):
    @classmethod
    async def start(cls, base, lora, env=None):
      engine = await super().start(base, lora, env)
      engine.name, engine.gpus = f"sampler#{next(serial)}", tuple(map(int, env["CUDA_VISIBLE_DEVICES"].split(",")))
      occ.enter(engine.name, engine.gpus)
      return engine

    async def sleep(self):
      occ.leave(self.name)

    async def wake(self):
      occ.enter(self.name, self.gpus)

    async def close(self):
      occ.leave(self.name)

    async def sample(self, job, name):
      if self.name not in occ.on:
        occ.violations.append(f"{occ.node}: {self.name} sampled off its GPUs")
      await asyncio.sleep(delay())
      return await super().sample(job, name)

  return Engine


class StressNode(TestNode):
  def __init__(self, bus, store, gpus, node, violations, errors, delay):
    super().__init__(bus, store=store, gpus=gpus, node=node)
    self.occ, self.errors, self.delay = Occupancy(node, violations), errors, delay
    self.engine_type = tracked_engine(self.occ, delay)

  def open_trainer(self, proc, width, env):
    gpus = tuple(map(int, env["CUDA_VISIBLE_DEVICES"].split(",")))
    name = f"{proc}#{next(serial)}"
    process = TrackedProcess(Trainer(Model(self.default_model), self.store), self.occ, name, gpus, self.delay)
    self.processes[proc] = process
    return process

  def dispatch(self):
    for proc, task in self.tasks.items():
      if task.done() and not task.cancelled() and (err := task.exception()):
        self.errors.append(f"{self.id}/{proc}: {err!r}")
    super().dispatch()


class Stress(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.rng = random.Random(SEED)
    self.bucket = f"file://{tempfile.mkdtemp()}"
    self.nodes: dict[str, StressNode] = {}
    self.loops: dict[str, asyncio.Task] = {}
    self.violations, self.errors, self.problems, self.injected = [], [], [], []
    self.cut: list[StressNode] = []
    self.calls_made = {"ok": 0, "failed": 0, "hung": 0}
    self.made = 0

  async def asyncTearDown(self):
    for task in self.loops.values():
      task.cancel()
    await asyncio.gather(*self.loops.values(), return_exceptions=True)
    for node in self.nodes.values():
      await node.aclose()
      await node.bus.aclose()
    for node in self.cut:
      await asyncio.to_thread(node.store.close)
    await super().asyncTearDown()

  # ***** the cluster *****

  async def add_node(self, gpus: int) -> StressNode:
    self.made += 1
    name = f"node-{self.made}"
    store = Store(self.bucket, cache=tempfile.mkdtemp())
    delay = lambda: self.rng.uniform(0.004, 0.02)  # noqa: E731
    node = StressNode(Bus(connect(self.url)), store, gpus, name, self.violations, self.errors, delay)
    self.nodes[name] = node
    await node.announce_node()

    async def loop():
      while True:
        await node.tick()
        await asyncio.sleep(0.01)

    self.loops[name] = asyncio.create_task(loop())
    return node

  async def remove_node(self, name: str, graceful: bool) -> None:
    node = self.nodes.pop(name)
    self.loops.pop(name).cancel()
    await node.close(graceful=graceful)
    if task := node.sweeping:
      task.cancel()
    await self.bus.r.hdel(key("nodes"), name)
    await node.bus.aclose()

  async def cut_node(self, name: str) -> None:
    """The node loses Redis for good, mid-work: none of its replies, releases or lease drops arrive. Its leases and
    heartbeat run out by themselves. Its store goes on uploading, so the saves it acknowledged are not lost."""
    node = self.nodes.pop(name)
    self.loops.pop(name).cancel()
    await node.bus.aclose()
    node.bus.r = node.bus.s = Unreachable()
    for task in node.tasks.values():
      task.cancel()
    await asyncio.gather(*node.tasks.values(), return_exceptions=True)
    self.cut.append(node)

  def start_scheduler(self) -> None:
    bus = Bus(connect(self.url))
    self.addAsyncCleanup(bus.aclose)

    async def loop():
      while True:
        await scheduler.tick(bus)
        await asyncio.sleep(0.01)

    self.loops["scheduler"] = asyncio.create_task(loop())

  async def chaos(self) -> None:
    while True:
      await asyncio.sleep(self.rng.uniform(0.3, 1.0))
      nodes = sorted(self.nodes)
      action = self.rng.choice(["cancel", "crash", "crash", "lose", "drain", "cut", "add"])
      if action in ("cancel", "crash"):
        node = self.nodes[self.rng.choice(nodes)]
        if not (live := sorted(p for p, t in node.tasks.items() if not t.done())):
          continue
        proc = self.rng.choice(live)
        self.injected.append(f"{action} {node.id}/{proc}")
        if action == "cancel":
          node.tasks[proc].cancel()
        elif proc in node.processes:
          node.processes[proc].closed = True  # the model process died
      elif action in ("lose", "drain") and len(nodes) > 1:
        name = self.rng.choice(nodes)
        self.injected.append(f"{action} {name}")
        await self.remove_node(name, graceful=action == "drain")
      elif action == "cut" and len(nodes) > 1:
        name = self.rng.choice(nodes)
        self.injected.append(f"cut {name}")
        await self.cut_node(name)
      elif action == "add" and len(nodes) < 3:
        self.injected.append("add")
        await self.add_node(self.rng.choice([1, 2]))

  # ***** clients *****

  async def call(self, run: str, op: Op, body: bytes, cost: int = 100):
    return (await self.calls(run, [(op, body)], cost))[0]

  async def calls(self, run: str, cmds: list[tuple[Op, bytes]], cost: int = 100) -> list:
    """Submit every command in order, then wait for all the replies, like a client that pipelines futures."""
    requests = [await self.bus.submit_command(run, op, body, cost=cost) for op, body in cmds]
    return await asyncio.gather(*(self.reply(run, op, request) for (op, _), request in zip(cmds, requests)))

  async def reply(self, run: str, op: Op, request: str | None):
    if request is None:
      self.problems.append(f"{run}: could not submit {op}")
      return None
    reply = await self.bus.wait_for_reply(request, WAIT)
    if reply is None:
      await self.dump(run, request)
      late = await self.bus.wait_for_reply(request, 2 * WAIT)  # slow, or stuck for good?
      self.problems.append(
        f"{run}: {op} {request} had no reply in {WAIT:.0f} s, and {'one' if late else 'none'} after {3 * WAIT:.0f} s"
      )
    self.calls_made["hung" if reply is None else "ok" if reply.ok else "failed"] += 1
    return reply

  async def dump(self, run: str, request: str) -> None:
    """Where a command that got no reply is stuck."""
    raw = await self.bus.read_state()
    row = {
      k.decode(): v.decode()[:80] for k, v in (await self.bus.r.hgetall(key("run", run))).items() if k != b"record"
    }
    plan, out = raw["plan"], [f"--- {request} at {time.monotonic() - self.started:.1f}s", f"row {row}"]
    out.append(f"queue {await self.bus.queued_and_in_flight(run)}")
    out.append(
      f"home {plan.get('home', {}).get(run)} procs { {n: p['node'] for n, p in plan.get('procs', {}).items()} }"
    )
    out.append(
      f"lag { {a: round(v) for a, v in plan.get('lag', {}).items()} } service { {a: round(v) for a, v in raw['service'].items()} }"
    )
    for node in self.nodes.values():
      out.append(f"{node.id}: gpus={node.gpus} on={node.on} physical={node.occ.on} draining={node.draining}")
      for proc, lf in node.lives.items():
        heads = [(c.run[:4], c.op.value) for c in lf.heads.values()]
        owned = [r[:4] for r in getattr(lf, "owned", ())]
        out.append(
          f"  {proc} {lf.life[-6:]} role={lf.spec.role.value} gpus={lf.spec.gpus} active={lf.active} wanted={lf.wanted()} "
          f"owned={owned} runs={[r[:4] for r in node.state.runs(proc)]} heads={heads} "
          f"task={'done' if node.tasks[proc].done() else 'running'} q={lf.quantum:.2f} granted={time.monotonic() - lf.granted_at:.1f}s ago"
        )
    print("\n".join(out), file=sys.stderr)

  async def train(self, run: str, account: str, rank: int, rng: random.Random) -> None:
    lora = LORA if rank else {}
    await self.bus.create_run(Run(run, "tiny", rank, lora=lora, account=account))
    ref = Model(tiny)  # what the client was told happened
    create = json.dumps({"base_model": "tiny", "rank": rank, "lora": lora}).encode()
    saved = snapshot = None

    async def save(name: str) -> bool:
      nonlocal saved, snapshot
      uri = f"tinker://{run}/weights/{name}"
      reply = await self.call(run, Op.SAVE_WEIGHTS, json.dumps({"uri": uri}).encode())
      if reply is not None and reply.ok:
        tensors, meta = ref.training_state(run)
        saved, snapshot = uri, ({k: v.clone() for k, v in tensors.items()}, copy.deepcopy(meta))
      return bool(reply is not None and reply.ok)

    async def recover() -> bool:
      """After any failure the client reloads its last save, as a real client must."""
      for _ in range(40):
        body = json.dumps({"uri": saved, "optimizer": True}).encode()
        reply = await self.call(run, Op.LOAD_WEIGHTS, body)
        if reply is None:
          return False
        if reply.ok:
          ref.restore_run(run, {k: v.clone() for k, v in snapshot[0].items()}, copy.deepcopy(snapshot[1]), True, False)
          return True
        await asyncio.sleep(0.2)
      self.problems.append(f"{run}: could not reload {saved} in 40 tries")
      return False

    reply = await self.call(run, Op.CREATE_MODEL, create)
    if reply is None or not reply.ok:
      self.abandoned.append(run)
      return
    ref.create_run(run, "tiny", rank, lora)
    if not await save("s0"):
      self.abandoned.append(run)
      return
    step = 0
    while step < STEPS:
      # One training cycle: one or two gradient batches and the step that applies them.
      sizes = [rng.randrange(6, 12) for _ in range(rng.choice([1, 1, 2]))]
      cycle = [(Op.FORWARD_BACKWARD, proto_body(run, n)) for n in sizes] + [(Op.OPTIM_STEP, OPT)]
      if rng.random() < 0.5:
        replies = await self.calls(run, cycle)
      else:
        replies = []
        for op, body in cycle:
          replies.append(await self.call(run, op, body))
          if replies[-1] is None or not replies[-1].ok:
            break
      if any(r is None for r in replies):
        return
      if not all(r.ok for r in replies):
        if not await recover():
          return
        continue
      for n in sizes:
        ref.forward_backward(run, [counting(n, 0), counting(n, 40)], "cross_entropy", {}, False)
      ref.optim_step(run, {"learning_rate": 1e-2})
      step += 1
      if rng.random() < 0.3 and not await save(f"s{step}") and not await recover():
        return
      await asyncio.sleep(rng.uniform(0, 0.02))  # the client thinks
    # The state the system holds for the run must be exactly what the client was told.
    for _ in range(40):
      if await save("final"):
        break
      if not await recover():
        return
    else:
      self.problems.append(f"{run}: could not save its final state")
      return
    await self.compare(run, saved, ref)
    self.finished.append(run)

  async def compare(self, run: str, uri: str, ref: Model) -> None:
    version = await self.bus.checkpoint_version(uri)
    want, _ = ref.training_state(run)
    for _ in range(100):  # the save may still be on its way from the node that made it to the bucket
      for node in list(self.nodes.values()):
        try:
          got, _ = await asyncio.to_thread(node.store.get, version)
          break
        except (ValueError, OSError):
          continue
      else:
        await asyncio.sleep(0.1)
        continue
      break
    else:
      self.problems.append(f"{run}: could not read its final save {version}")
      return
    if got.keys() != want.keys():
      self.problems.append(f"{run}: final save has other tensors: {sorted(got.keys() ^ want.keys())[:4]}")
      return
    wrong = [k for k in want if not torch.allclose(got[k].float(), want[k].float(), rtol=1e-5, atol=1e-6)]
    if wrong:
      self.problems.append(f"{run}: {len(wrong)} of {len(want)} tensors differ from what was acknowledged: {wrong[:3]}")

  async def sample(self, session: str, account: str, rng: random.Random) -> None:
    await self.bus.create_run(Run(session, "tiny", 4, role=Role.SAMPLER, account=account))
    payload = json.dumps(JOB).encode()
    for _ in range(STEPS):
      replies = await asyncio.gather(*(self.call(session, Op.SAMPLE, payload, 50) for _ in range(rng.randrange(1, 6))))
      if any(r is None for r in replies):
        return
      await asyncio.sleep(rng.uniform(0, 0.05))
    self.finished.append(session)

  # ***** the scenario *****

  async def test_stress(self):
    rng, started = self.rng, time.monotonic()
    self.started = started
    self.abandoned, self.finished = [], []
    patches = [
      mock.patch("tinyrl.control.node_daemon.QUANTUM_MS", 100),
      mock.patch("tinyrl.control.scheduler.REBALANCE_MS", 1000),
    ]
    if CHAOS:  # so that a node cut off from Redis is given up on in seconds
      patches += [mock.patch(f"tinyrl.{m}.LEASE_MS", 3000) for m in ("bus", "control.node_daemon")]
      patches.append(mock.patch("tinyrl.bus.NODE_MS", 4000))
    for p in patches:
      p.start()
      self.addCleanup(p.stop)
    gpus = [rng.choice([1, 2]) for _ in range(rng.choice([1, 2, 2]))]
    for g in gpus:
      await self.add_node(g)
    self.start_scheduler()
    clients, trainers, samplers = [], [], []
    for account in ("ana", "ben", "cy")[: rng.choice([2, 3])]:
      for _ in range(rng.randrange(1, 4)):
        trainers.append(run := new_id())
        clients.append(self.train(run, account, 4, random.Random(rng.random())))
      if rng.random() < 0.4:
        trainers.append(run := new_id())
        clients.append(self.train(run, account, 0, random.Random(rng.random())))
      for _ in range(rng.randrange(0, 3)):
        samplers.append(session := new_id())
        clients.append(self.sample(session, account, random.Random(rng.random())))
    chaos = asyncio.create_task(self.chaos()) if CHAOS else None
    results = await asyncio.gather(*clients, return_exceptions=True)
    self.problems += [f"client raised {r!r}" for r in results if isinstance(r, BaseException)]
    if chaos:
      chaos.cancel()
      await asyncio.gather(chaos, return_exceptions=True)
    elapsed, capacity = time.monotonic() - started, sum(n.gpus for n in self.nodes.values())
    service = (await self.bus.read_state())["service"]

    # Delete everything, then nothing may be left behind.
    for run in trainers:
      if await self.bus.r.sismember(key("runs"), run):
        await self.call(run, Op.DELETE_MODEL, b"{}")
    for session in samplers:
      await self.bus.delete_run(session)
    try:
      async with asyncio.timeout(30):
        while (await self.bus.read_state())["plan"].get("procs") or any(
          n.lives or n.tasks for n in self.nodes.values()
        ):
          await asyncio.sleep(0.05)
    except TimeoutError:
      plan = (await self.bus.read_state())["plan"]
      lives = {n.id: sorted(n.lives) for n in self.nodes.values()}
      self.problems.append(
        f"did not settle after deleting every run: procs {sorted(plan.get('procs', {}))} lives {lives}"
      )
    # A node cut off from Redis cannot drop its leases. They must carry an expiry, and then they lapse by themselves.
    cut, leases = {n.id for n in self.cut}, [k.decode() async for k in self.bus.r.scan_iter(match=key("lease", "*"))]
    lapsing = [k for k in leases if k.split("@")[-1].rsplit("-", 1)[0] in cut and await self.bus.r.pttl(k) != -1]
    left = {
      "leases": sorted(set(leases) - set(lapsing)),
      "queues": [k.decode() async for k in self.bus.r.scan_iter(match=key("q", "*"))],
      "writing": sorted(k.decode() for k in await self.bus.r.hkeys(key("writing"))),
      "owners": [r for r in trainers if await self.bus.r.hget(key("run", r), "owner")],
      "on GPUs": {n.id: dict(n.occ.on) for n in self.nodes.values() if n.occ.on},
    }
    self.problems += [f"left behind {what}: {found}" for what, found in left.items() if found]
    self.problems += self.violations
    expected = ("exited", "EOFError", "lost ownership", "lost lease") if CHAOS else ()
    self.problems += [f"worker stopped: {e}" for e in self.errors if not any(x in e for x in expected)]

    shares = {a: round(v / sum(service.values()), 2) for a, v in sorted(service.items())} if service else {}
    print(
      f"\nSEED={SEED} chaos={CHAOS} block={BLOCK} nodes={gpus} trainers={len(trainers)} samplers={len(samplers)} "
      f"calls={self.calls_made} finished={len(self.finished)} abandoned={len(self.abandoned)} injected={dict(collections.Counter(i.split()[0] for i in self.injected))} "
      f"elapsed={elapsed:.1f}s gpu_use={sum(service.values()) / 1000 / max(1e-9, elapsed * capacity):.2f} shares={shares}",
      file=sys.stderr,
    )
    for problem in self.problems:
      print(f"  PROBLEM {problem}", file=sys.stderr)
    self.assertEqual(self.problems, [])


if __name__ == "__main__":
  unittest.main()
