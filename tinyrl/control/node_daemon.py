"""Node daemon: fixed-quantum + EEVDF local GPU scheduler and process reconciler.

Enforces local GPU exclusivity (`on`), single-writer fencing (`run.owner`), and process leases. Processes placed on
this node share its GPUs via `plan_gpus`: each grant runs for at least `quantum` wall seconds before entering overtime
(reset only on owner change), while idle residents stay warm in VRAM and yield immediately when a peer is ready.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass

from tinyrl.bus import LEASE_MS, Bus
from tinyrl.control import scheduler
from tinyrl.control.accounting import Charge, Wave
from tinyrl.control.fairness import Demand, Worker, blocks, plan_gpus, select_runs
from tinyrl.control.process import Process
from tinyrl.helpers import DEBUG, getenv
from tinyrl.storage.store import Store
from tinyrl.trace import span
from tinyrl.types import Cmd, Ev, Op, Reply, Role, unpack_snap
from tinyrl.workers.sampler import Sampler

GPUS, DRAIN_S, TICK, IDLE_MS = getenv("GPUS", 1), getenv("DRAIN_S", 20.0), getenv("TICK", 0.25), getenv("IDLE_MS", 150)
QUANTUM_MS, MAX_QUANTUM_MS = getenv("QUANTUM_MS", 2000), getenv("MAX_QUANTUM_MS", 15000)
CYCLE_GRACE_MS, SWITCH_BUDGET = getenv("CYCLE_GRACE_MS", 100), float(getenv("SWITCH_BUDGET", 0.1))
SAMPLE_RUNS = getenv("SAMPLE_RUNS", 3)
SAMPLE_COMMANDS, SWEEP_S, SWEEP_GRACE_S = getenv("SAMPLE_COMMANDS", 128), getenv("SWEEP_S", 60.0), 600


def gpu_memory_mib() -> int:
  if override := getenv("GPU_MEMORY_MIB", 0):
    return int(override)
  try:
    cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"]
    return int(subprocess.check_output(cmd, timeout=2, stderr=subprocess.DEVNULL))
  except (OSError, subprocess.SubprocessError, ValueError):
    return 0


async def stop_tasks(*tasks: asyncio.Task | None) -> None:
  if live := [t for t in tasks if t]:
    for t in live:
      t.cancel()
    await asyncio.gather(*live, return_exceptions=True)


class NodeDaemon:
  def __init__(
    self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store: Store | None = None
  ):
    self.bus = bus
    self.store = store if store is not None else Store()
    self.store.start()
    self.gpus, self.id = gpus, node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.on: dict[str, tuple[int, ...]] = {}  # life -> occupied GPUs (single source of local GPU truth)
    self.lives: dict[str, Lifetime] = {}
    self.tasks: dict[str, asyncio.Task] = {}
    self.timer: asyncio.TimerHandle | None = None
    self.state, self.changed, self.draining = scheduler.State(), asyncio.Event(), False
    self.heartbeat, self.lag, self.swept, self.sweeping = time.monotonic(), 0.0, 0.0, None

  async def serve(self) -> None:
    await self.announce_node()
    wake_id = "0-0"
    try:
      while True:
        await self.tick()
        before = time.monotonic()
        wake_id = await self.bus.wait_wake(wake_id, int(TICK * 1000), stream="plan_wake")
        self.lag = max(0.0, time.monotonic() - before - TICK)
    finally:
      await self.close()

  async def push_heartbeat(self) -> None:
    on = {life.split("@", 1)[0]: list(gs) for life, gs in self.on.items()}
    pids = {p: m.proc.pid for p, lf in self.lives.items() if (m := lf.model) and m.proc and m.proc.pid is not None}
    leases = tuple(lf.life for lf in self.lives.values() if not lf.closing)
    before = time.monotonic()
    await self.bus.node_heartbeat(
      self.id, self.gpus, self.gpu_mib * 2**20, leases=leases, on=on, pids=pids, lag=self.lag
    )
    self.heartbeat = before

  async def tick(self) -> None:
    if not self.draining and time.monotonic() - self.heartbeat >= LEASE_MS / 1000:
      for lf in list(self.lives.values()):
        if not lf.closing and lf.task and not lf.task.cancelling():
          lf.task.cancel()
    await self.push_heartbeat()
    self.state = await self.read_state()
    self.dispatch()
    if not self.draining:
      for proc, p in sorted(self.state.procs.items()):
        if p.node == self.id and proc not in self.tasks and self.state.runs(proc):
          self.start(proc)
    if time.time() - self.swept > SWEEP_S and (self.sweeping is None or self.sweeping.done()):
      self.swept = time.time()
      self.sweeping = asyncio.create_task(self.bus.sweep_checkpoints(self.store, SWEEP_GRACE_S))
    self.notify()

  def dispatch(self) -> None:
    for proc, task in list(self.tasks.items()):
      if task.done():
        if not task.cancelled() and (err := task.exception()):
          print(f"{proc} stopped: {err!r}", file=sys.stderr, flush=True)
        del self.tasks[proc]
        self.lives.pop(proc, None)
      elif not task.cancelling() and (lf := self.lives.get(proc)) and not lf.closing and lf.exited():
        task.cancel()

  def select_gpus(self, proc: str) -> tuple[int, ...]:
    occupied = set().union(*self.on.values())
    workers = {p: lf.worker() for p, lf in self.lives.items() if lf.wanted()}
    keep, wake_at = plan_gpus(workers, self.gpus, now := time.monotonic(), occupied)
    if self.timer:
      self.timer.cancel()
    self.timer = asyncio.get_running_loop().call_later(max(0.0, wake_at - now), self.notify) if wake_at else None
    return keep.get(proc, ())

  def claim_gpus(self, life: str, gpus: tuple[int, ...], width: int) -> bool:
    if len(gpus) != width or len(set(gpus)) != width or any(g < 0 or g >= self.gpus for g in gpus):
      return False
    if {g for k, gs in self.on.items() if k != life for g in gs} & set(gpus):
      return False
    self.on[life] = gpus
    self.notify()
    return True

  def release_gpus(self, life: str) -> None:
    if self.on.pop(life, None) is not None:
      self.notify()

  def start(self, proc: str) -> asyncio.Task:
    cls = SamplerLife if self.state.procs[proc].spec.role == Role.SAMPLER else TrainerLife
    self.tasks[proc] = task = asyncio.create_task(cls(self, proc).run())
    return task

  async def close(self, graceful: bool = True) -> None:
    self.draining = True
    if self.timer:
      self.timer.cancel()
    self.notify()
    deadline, pending = time.monotonic() + (DRAIN_S if graceful else 0), set(self.tasks.values())
    while pending and time.monotonic() < deadline:
      _, pending = await asyncio.wait(pending, timeout=min(LEASE_MS / 4000, deadline - time.monotonic()))
      with contextlib.suppress(Exception):
        await self.bus.touch_leases(*(lf.life for lf in self.lives.values() if not lf.closing), xx=True)
    await stop_tasks(*pending)
    self.dispatch()
    await asyncio.to_thread(self.store.close)

  async def read_state(self) -> scheduler.State:
    plan, service = await asyncio.gather(self.bus.read_plan(), self.bus.account_service())
    return scheduler.from_plan(plan, service)

  def notify(self) -> None:
    changed, self.changed = self.changed, asyncio.Event()
    changed.set()

  async def announce_node(self) -> None:
    await self.push_heartbeat()
    await self.bus.append_event(Ev.NODE_UP, node=self.id, gpus=self.gpus, memory=self.gpu_mib * 2**20)

  def open_trainer(self, proc: str, width: int, env: dict[str, str]) -> Process:
    dist = ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={width}"] if width > 1 else []
    return Process([sys.executable, *dist, "-m", "tinyrl.workers.trainer"], env)

  def open_sampler(self, proc: str, spec: scheduler.Spec, env: dict[str, str]) -> Sampler:
    return Sampler(spec.base, spec.lora, env)


class Lifetime:
  """One process incarnation: owns local GPU quantum residency and fenced subprocess teardown."""

  gang = False

  def __init__(self, node: NodeDaemon, proc: str):
    self.node, self.bus, self.store, self.proc = node, node.bus, node.store, proc
    self.spec, self.eager, self.pinned = node.state.procs[proc].spec, True, True
    self.life = f"{proc}@{node.id}-{os.urandom(6).hex()}"
    self.task = self.model = self.closing = self.reader = None
    self.reader_streams: list[str] = []
    self.bound_gpus, self.demand = (), Demand.IDLE
    self.hold_until, self.park_s, self.gpu_ms = 0.0, 0.0, 0.0
    self.heads: dict[str, Cmd] = {}
    self.service, self.service_floor = {}, 0.0

  def exited(self) -> bool:
    return bool(self.model and (p := self.model.proc) and p.poll() is not None)

  def streams(self) -> list[str]:
    return self.node.state.runs(self.proc)

  def ready(self) -> tuple[list[Cmd], list[Cmd]]:
    return list(self.heads.values()), []

  def ready_runs(self) -> list[str]:
    return [c.run for c in self.ready()[0]]

  async def sync_runs(self, *, detach_all: bool = False) -> None:
    pass

  def wanted(self) -> bool:
    p = self.node.state.procs.get(self.proc)
    if self.node.draining or not p or p.node != self.node.id or not self.node.state.runs(self.proc):
      return False
    excl = {g for o in self.node.lives.values() if o.spec.exclusive for g in self.node.on.get(o.life, ())}
    if self.life in self.node.on or excl.isdisjoint(self.bound_gpus):
      return True
    # A blocked sampler exits immediately; a blocked multi-GPU trainer detaches once work and another block exist.
    return self.gang and not (
      self.ready_runs() and any(excl.isdisjoint(b) for b in blocks(self.node.gpus, p.spec.gpus))
    )

  def worker(self) -> Worker:
    s, sp, bnd = self.node.state, self.spec, self.bound_gpus
    runs = [r for r in self.ready_runs() if r in s.jobs] or [r for r in self.streams() if r in s.jobs]
    cur, lag = self.node.on.get(self.life, ()), max(map(s.ready_lag, runs), default=-self.gpu_ms)
    return Worker(sp.gpus, sp.exclusive, self.demand, cur, bnd, self.gang, self.hold_until, lag, self.gpu_ms)

  def spawn_model(self, gpus: tuple[int, ...]) -> None:
    env = {**scheduler.settings(self.node.gpus, self.node.gpu_mib * 2**20), **os.environ, **self.store.env()}
    self.model = self.open_model(env | {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))})

  async def run(self) -> None:
    self.task = asyncio.current_task()
    try:
      if self.node.draining:
        return
      opening = asyncio.create_task(self.bus.touch_leases(self.life))
      try:
        await asyncio.shield(opening)
      except BaseException:
        with contextlib.suppress(Exception):
          await opening
        raise
      self.node.lives[self.proc] = self
      if self.eager:
        self.spawn_model(tuple(range(self.node.gpus)))
      while self.wanted():
        changed = self.node.changed
        await self.sync_runs()
        await self.collect()
        ready, failed = self.ready()
        if failed:
          await self.dispatch(failed, failed=True)
          ready, _ = self.ready()
          if not self.wanted():
            break
        demand, wait_s = self.turn(ready)
        await self.sync_gpus(demand)
        if not self.wanted():
          break
        if ready and self.life in self.node.on:
          if await self.dispatch(ready):
            break
        else:
          self.start_reader(*self.reader_args())
          with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(changed.wait(), wait_s)
      self.demand = Demand.IDLE
      await self.stop_reader()
      await self.sync_runs(detach_all=True)
    finally:
      self.demand = Demand.IDLE
      self.closing = asyncio.create_task(self.cleanup())
      self.node.notify()
      while True:
        try:
          await asyncio.shield(self.closing)
          break
        except asyncio.CancelledError:
          if self.closing.cancelled():
            self.closing = asyncio.create_task(self.cleanup())

  def start_reader(self, peek, streams: list[str], delay: float = 0) -> None:
    if self.reader is not None or not streams:
      return
    self.reader_streams = list(streams)

    async def read() -> list[Cmd]:
      if delay:
        await asyncio.sleep(delay)
      return await peek(streams, IDLE_MS)

    self.reader = asyncio.create_task(read())
    self.reader.add_done_callback(lambda _: self.node.notify())

  async def stop_reader(self) -> None:
    await stop_tasks(self.reader)
    self.reader, self.reader_streams = None, []

  async def poll_reader(self, peek, streams: list[str], fetch: bool = True) -> list[Cmd] | None:
    if self.reader and self.reader.done():
      res, self.reader, self.reader_streams = self.reader.result(), None, []
      return res
    if self.reader and self.reader_streams == streams:
      return None
    await self.stop_reader()
    return await peek(streams, 0) if fetch and streams else None

  async def sync_gpus(self, demand: Demand) -> None:
    """Between model calls, release GPUs only after park succeeds. Cleanup retains them until the child closes."""
    if self.exited():
      raise RuntimeError(f"process {self.proc} exited")
    cur = self.node.on.get(self.life)
    if self.demand != demand:
      self.demand = demand
      self.node.notify()
    want = self.node.select_gpus(self.proc)
    if cur is not None and want != cur:
      t0 = time.monotonic()
      with span("node.park"):
        await self.model.park()
      self.park_s = time.monotonic() - t0
      self.node.release_gpus(self.life)
      await self.node.push_heartbeat()
    elif cur is None and want and demand != Demand.IDLE and self.node.claim_gpus(self.life, want, self.spec.gpus):
      try:
        t0 = time.monotonic()
        with span("node.resume"):
          if self.model is None:
            self.spawn_model(want)
          if not self.bound_gpus and self.pinned:
            self.bound_gpus = want
          await self.model.resume(want)
        now = time.monotonic()
        sw = (self.park_s + now - t0) / max(0.01, SWITCH_BUDGET) if self.park_s else 0.0
        self.hold_until = now + min(MAX_QUANTUM_MS / 1000, max(QUANTUM_MS / 1000, sw))
        self.node.select_gpus(self.proc)
      except Exception as e:
        await self.fail_promote(Reply.error(e))
        raise
      await self.node.push_heartbeat()

  def choose_runs(self, candidates: list[Cmd], width: int = 1) -> list[str]:
    for c in candidates:
      self.service[c.run] = max(self.service.get(c.run, self.service_floor), self.service_floor)
    return select_runs(candidates, self.node.state.account, self.node.state.ready_lag, self.service, width)

  def bill(self, cmd: Cmd, acct: str, charge: Charge, billed: bool) -> None:
    if billed and charge.gpu_ms:
      self.service[cmd.run] = self.service.get(cmd.run, self.service_floor) + charge.gpu_ms
      self.gpu_ms += charge.gpu_ms
      if acct:
        self.node.state.service[acct] = self.node.state.service.get(acct, 0.0) + charge.gpu_ms
    if DEBUG:
      print(f"{self.proc} {cmd.request_id} {cmd.op:24s} {charge.busy_ms:9d} ms")

  def advance_floor(self) -> None:
    seen = [self.service.get(r, self.service_floor) for r in (*self.service, *self.ready_runs())]
    self.service_floor = max(self.service_floor, min(seen, default=self.service_floor))

  async def cleanup(self) -> None:
    """Close the child first, then release GPUs (`on`), fail stranded trainer work, and drop the lease."""
    await stop_tasks(self.reader, *getattr(self, "staged", {}).values())
    while self.model:
      try:
        with span("node.close"):
          await self.model.close()
        break
      except (Exception, asyncio.CancelledError) as e:
        print(f"{self.proc} could not close, retrying: {e!r}", file=sys.stderr, flush=True)
        await asyncio.sleep(TICK)
    self.node.release_gpus(self.life)
    try:
      for run, state in getattr(self, "owned", {}).items():
        await self.bus.fail_stranded_commands(run, state.token)
    finally:
      await self.bus.drop_lease(self.life)


@dataclass
class OwnedRun:
  token: str
  pos: str | None
  snap: str | None
  loaded: bool = False
  open_grad: bool = False


class TrainerLife(Lifetime):
  """Single-writer training process: owns runs via epoch tokens, executes one command at a time."""

  def __init__(self, node: NodeDaemon, proc: str):
    super().__init__(node, proc)
    self.gang = self.pinned = self.spec.gpus > 1
    self.eager = self.spec.gpus == 1
    self.owned: dict[str, OwnedRun] = {}
    self.idle_since = 0.0

  def streams(self) -> list[str]:
    return sorted(self.owned)

  def open_model(self, env: dict[str, str]):
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return self.node.open_trainer(self.proc, self.spec.gpus, env)

  def reader_args(self):
    return self.bus.peek_trainer, [r for r in sorted(self.owned) if r not in self.heads], 0

  def turn(self, ready: list[Cmd]) -> tuple[Demand, float | None]:
    # Keep the GPU across client round-trips (`CYCLE_GRACE_MS`) and open training steps (avoid moving gradients).
    grace = self.idle_since + CYCLE_GRACE_MS / 1000 - time.monotonic()
    linger = self.life in self.node.on and grace > 0
    closing = any(c.op == Op.OPTIM_STEP and self.owned[c.run].open_grad for c in ready)
    if closing or (linger and not ready and any(s.open_grad for s in self.owned.values())):
      return Demand.FINISH_STEP, grace if linger else None
    return (Demand.READY if ready or linger else Demand.IDLE), grace if linger else None

  async def fail_promote(self, err: Reply) -> None:
    for run, state in self.owned.items():
      if cmd := await self.bus.claim_trainer(run, state.token):
        await self.bus.commit_reply(cmd, err, state.token)

  async def sync_runs(self, *, detach_all: bool = False) -> None:
    want = set() if detach_all else set(self.node.state.runs(self.proc))
    for r in sorted(self.owned.keys() - want):
      await self.detach(r, drop=not detach_all)
    for r in sorted(want - self.owned.keys()):
      if got := await self.bus.acquire_run(r, self.life):
        self.owned[r] = state = OwnedRun(*got)
        await self.bus.fail_stranded_commands(r, state.token)

  async def detach(self, run: str, drop: bool) -> None:
    """Flush `run` if dirty (0 bytes if clean), release `run.owner` fenced by `(token, pos)`, and drop if requested."""
    state, checkpoint = self.owned[run], None
    if state.pos is not None and state.snap is None and state.loaded:
      version = await self.bus.begin_checkpoint(run, state.token)
      if checkpoint := await self.model.call("save_run", run, version):
        state.snap = f"{checkpoint['uri']}\n{checkpoint['version']}"
    await self.bus.detach_run(run, state.token, state.pos, state.snap, checkpoint)
    for d in (self.owned, self.heads, self.service):
      d.pop(run, None)
    if drop and state.loaded:
      await self.model.call("drop_run", run)

  async def collect(self) -> None:
    missing = [r for r in sorted(self.owned) if r not in self.heads]
    if (cmds := await self.poll_reader(self.bus.peek_trainer, missing)) is not None:
      self.heads |= {c.run: c for c in cmds if c.run in self.owned}
      self.service = {r: self.service.get(r, self.service_floor) for r in self.heads}

  async def dispatch(self, ready: list[Cmd], failed: bool = False) -> bool:
    [run] = self.choose_runs(ready, 1)
    self.heads.pop(run, None)
    state = self.owned[run]
    with span("node.claim"):
      cmd = await self.bus.claim_trainer(run, state.token)
    if cmd is None:
      return False
    charge, acct, effects, version = Charge(), self.node.state.account(run), {}, None
    with span("node.execute"):
      try:
        if cmd.op == Op.LOAD_WEIGHTS:
          version = await self.bus.checkpoint_version(json.loads(cmd.body)["uri"])
        elif cmd.op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER):
          version = await self.bus.begin_checkpoint(run, state.token)
      except (KeyError, ValueError) as exc:
        reply = Reply.error(exc)
      else:
        job = self.node.state.jobs.get(run)
        saved = unpack_snap(state.snap)[1] if state.snap and not state.loaded and cmd.op != Op.LOAD_WEIGHTS else None
        reply, effects = await Wave(self.spec.gpus).run(
          self.model.call("execute", cmd, job.record if job else None, saved, version), charge
        )
        state.loaded = bool(effects.get("loaded"))
    with span("node.publish"):
      mutates = bool(effects.get("mutated"))
      billed = await self.bus.commit_reply(cmd, reply, state.token, acct, mutates, charge, effects.get("saved"))
      if not billed:
        self.owned.pop(run, None)
        raise RuntimeError(f"lost ownership of {run}")
      if cmd.op == Op.DELETE_MODEL and reply.ok:
        self.owned.pop(run, None)
      elif mutates:
        state.pos, state.snap = cmd.entry_id, None
      state.open_grad = cmd.op == Op.FORWARD_BACKWARD and reply.ok
    self.bill(cmd, acct, charge, billed)
    self.advance_floor()
    self.idle_since = time.monotonic()
    return cmd.op == Op.DELETE_MODEL and reply.ok and not self.spec.lora


class SamplerLife(Lifetime):
  """Stateless sampling pool replica: stages checkpoints per version off-GPU and runs bounded concurrent waves."""

  def __init__(self, node: NodeDaemon, proc: str):
    super().__init__(node, proc)
    self.inputs, self.staged = {}, {}

  def open_model(self, env: dict[str, str]):
    return self.node.open_sampler(self.proc, self.spec, env)

  def reader_args(self):
    return self.bus.peek_sampler, sorted(self.streams()), IDLE_MS / 1000 if self.heads else 0

  def turn(self, ready: list[Cmd]) -> tuple[Demand, float | None]:
    return (Demand.READY if ready or self.model.warm_pending else Demand.IDLE), None

  async def fail_promote(self, err: Reply) -> None:
    for cmd in await self.bus.take_commands(self.streams(), self.life, 1):
      await self.bus.commit_sample(cmd, err)

  async def stage(self, version: str):
    with span("node.prepare"):
      try:
        return await asyncio.to_thread(self.store.local_folder, version)
      except Exception as e:
        return e

  def ready(self) -> tuple[list[Cmd], list[Cmd]]:
    live = set(self.streams())
    done = [
      (c, v.result() if isinstance(v, asyncio.Task) else v)
      for r, c in self.heads.items()
      if r in live and r in self.inputs and (not isinstance(v := self.inputs[r], asyncio.Task) or v.done())
    ]
    return [c for c, v in done if not isinstance(v, Exception)], [c for c, v in done if isinstance(v, Exception)]

  async def collect(self) -> None:
    live, jobs = sorted(self.streams()), self.node.state.jobs
    live_set = set(live)
    self.heads = {r: c for r, c in self.heads.items() if r in live_set}
    if (cmds := await self.poll_reader(self.bus.peek_sampler, live, not self.heads)) is None:
      return
    self.heads = {c.run: c for c in cmds if c.run in live_set}
    paths = {r: jobs[r].record.get("model_path") for r in self.heads}
    versions = await self.bus.checkpoint_versions(sorted(set(paths.values()) - {None}))
    wanted = {v for v in versions.values() if isinstance(v, str)}
    self.staged = {v: t for v, t in self.staged.items() if v in wanted or not t.done()}
    for v in wanted - self.staged.keys():
      self.staged[v] = task = asyncio.create_task(self.stage(v))
      task.add_done_callback(lambda _: self.node.notify())
    self.inputs = {}
    for r, path in paths.items():
      version = versions.get(path)
      self.inputs[r] = self.staged.get(version, version)
    self.service = {r: self.service.get(r, self.service_floor) for r in self.heads}

  async def dispatch(self, candidates: list[Cmd], failed: bool = False) -> bool:
    runs = self.choose_runs(candidates, SAMPLE_RUNS if self.spec.lora else 1)
    inputs = {r: (v.result() if isinstance(v := self.inputs[r], asyncio.Task) else v) for r in runs}
    # Resolve consumed versions again next wave: cache eviction and failed transfers are both recoverable.
    consumed = {self.inputs[r] for r in runs}
    self.staged = {v: task for v, task in self.staged.items() if task not in consumed}
    await self.stop_reader()
    with span("node.claim"):
      cmds = await self.bus.take_commands(runs, self.life, 1 if failed else SAMPLE_COMMANDS)
    for r in runs:
      self.heads.pop(r, None)
    if not cmds:
      return False
    wave = Wave(self.spec.gpus)
    uses = Counter(inputs[c.run] for c in cmds)
    loads, errors = {}, {}
    for ckpt in uses:
      if ckpt is None or failed:
        continue
      loads[ckpt] = load = Charge()
      try:
        await wave.run(self.model.load(ckpt), load)
      except Exception as e:
        errors[ckpt] = e

    async def answer(cmd: Cmd) -> None:
      charge, acct, inp = Charge(), self.node.state.account(cmd.run), inputs[cmd.run]
      with span("node.execute"):
        error = errors.get(inp, inp if failed else None)
        reply = Reply.error(error) if error else await wave.run(self.model.execute(cmd, inp, True), charge)
        if load := loads.get(inp):
          charge.started_ms = load.started_ms
          charge.busy_ms += load.busy_ms
          charge.gpu_ms += load.gpu_ms / uses[inp]
      with span("node.publish"):
        billed = await self.bus.commit_sample(cmd, reply, acct, charge=charge)
      self.bill(cmd, acct, charge, billed)

    async with asyncio.TaskGroup() as group:
      for cmd in cmds:
        group.create_task(answer(cmd))
    self.advance_floor()
    return False


if __name__ == "__main__":

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await NodeDaemon(Bus()).serve()

  asyncio.run(main())
