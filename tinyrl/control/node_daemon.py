"""Local node control: `Lease` (Redis liveness), `GpuScheduler` (GPU turns & parking), `NodeDaemon`, and `Worker`."""

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
from dataclasses import dataclass

from tinyrl.bus import LEASE_MS, Bus
from tinyrl.control import scheduler
from tinyrl.control.accounting import Charge, Wave
from tinyrl.control.fairness import Demand, GpuSlot, RunFairness, blocks, plan_gpus
from tinyrl.control.process import Process
from tinyrl.helpers import DEBUG, getenv
from tinyrl.storage.store import Store
from tinyrl.trace import span
from tinyrl.types import Cmd, Ev, Op, Reply, Role, unpack_snapshot
from tinyrl.workers.sampler import Sampler

GPUS, DRAIN_S, TICK = getenv("GPUS", 1), getenv("DRAIN_S", 20.0), getenv("TICK", 0.25)
IDLE_MS, QUANTUM_MS, CYCLE_GRACE_MS = getenv("IDLE_MS", 150), getenv("QUANTUM_MS", 2000), getenv("CYCLE_GRACE_MS", 100)
SAMPLE_RUNS, SAMPLE_COMMANDS = getenv("SAMPLE_RUNS", 3), getenv("SAMPLE_COMMANDS", 128)
SWEEP_S, SWEEP_GRACE_S = getenv("SWEEP_S", 60.0), 600


def gpu_memory_mib() -> int:
  if override := getenv("GPU_MEMORY_MIB", 0):
    return int(override)
  try:
    cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"]
    return int(subprocess.check_output(cmd, timeout=2, stderr=subprocess.DEVNULL))
  except (OSError, subprocess.SubprocessError, ValueError):
    return 0


async def cancel_tasks(*tasks: asyncio.Task | None) -> None:
  for t in (live := [t for t in tasks if t]):
    t.cancel()
  await asyncio.gather(*live, return_exceptions=True)


async def run_shielded(fn) -> None:
  task = asyncio.create_task(fn())
  while not task.done():
    with contextlib.suppress(asyncio.CancelledError):
      await asyncio.shield(task)
    if task.cancelled():
      task = asyncio.create_task(fn())


@dataclass
class OwnedRun:
  """Single-writer state for a training run held by a `Worker`."""

  token: str
  last_cmd_id: str | None = None
  snapshot: str | None = None
  loaded: bool = False
  open_grad: bool = False


class Lease:
  """Redis worker lease with background renewal that cancels its owner task if liveness is lost."""

  def __init__(self, bus: Bus, worker_id: str, is_healthy):
    self.bus, self.worker_id, self.is_healthy = bus, worker_id, is_healthy
    self.closing, self._owner, self._task = False, None, None

  async def open(self) -> bool:
    if not await self.bus.open_lease(self.worker_id):
      return False
    self._owner, self._task = asyncio.current_task(), asyncio.create_task(self._keep_alive())
    return True

  def cancel_owner(self) -> None:
    if not self.closing and self._owner and not self._owner.cancelling():
      self._owner.cancel()

  async def renew(self) -> None:
    await self.bus.renew_lease(self.worker_id, recover=not self.closing)

  async def close(self) -> None:
    self.closing = True
    await cancel_tasks(self._task)
    await self.bus.drop_lease(self.worker_id)

  async def _keep_alive(self) -> None:
    last_renewed = time.monotonic()
    while True:
      await asyncio.sleep(LEASE_MS / 4000)
      if not self.closing and not self.is_healthy():
        break
      try:
        started = time.monotonic()
        await self.renew()
        last_renewed = started
      except Exception:
        if not self.closing and time.monotonic() - last_renewed >= LEASE_MS / 1000:
          break
    self.cancel_owner()


class GpuScheduler:
  """Schedules local GPU turns (`PARKED | WARM | BUSY | PARKING`) and performs all park/resume handoffs."""

  def __init__(self, node: NodeDaemon):
    self.node = node
    self.allocations: dict[str, tuple[int, ...]] = {}  # worker_id -> held GPUs (local source of truth)
    self.busy: set[str] = set()  # worker_ids actively calling their model (cannot be parked)
    self.parking: dict[str, asyncio.Task] = {}  # worker_id -> in-flight park() task
    self.changed = asyncio.Event()
    self._timer: asyncio.TimerHandle | None = None

  def notify(self) -> None:
    prev, self.changed = self.changed, asyncio.Event()
    prev.set()

  def claim(self, worker_id: str, gpus: tuple[int, ...], width: int) -> bool:
    taken = {g for k, gs in self.allocations.items() if k != worker_id for g in gs}
    valid = len(gpus) == width and len(set(gpus)) == width and set(gpus) <= set(range(self.node.gpus))
    if not valid or taken.intersection(gpus):
      return False
    self.allocations[worker_id] = gpus
    self.notify()
    return True

  def release(self, worker_id: str) -> None:
    if self.allocations.pop(worker_id, None) is not None:
      self.notify()

  def plan(self) -> dict[str, tuple[int, ...]]:
    now, occupied = time.monotonic(), set().union(*self.allocations.values())
    active = {p: w for p, w in self.node.workers.items() if w.should_run()}
    slots = {p: w.gpu_slot(now) for p, w in active.items()}
    keep, wake_at = plan_gpus(slots, self.node.gpus, now, occupied)
    contended = any(s.demand != Demand.IDLE and keep.get(p) != s.current for p, s in slots.items())
    graces = [w.grace_until for p, w in active.items() if p in keep and w.demand == Demand.IDLE and w.grace_until > now]
    if self._timer:
      self._timer.cancel()
      self._timer = None
    if deadlines := ([wake_at] if wake_at else []) + (graces if contended else []):
      loop = asyncio.get_running_loop()
      self._timer = loop.call_later(max(0.0, min(deadlines) - now), lambda: (self.reconcile(), self.notify()))
    return keep

  def reconcile(self) -> dict[str, tuple[int, ...]]:
    keep = self.plan()
    for proc, w in list(self.node.workers.items()):
      wid, cur = w.worker_id, self.allocations.get(w.worker_id)
      if cur and keep.get(proc) != cur and wid not in self.busy and wid not in self.parking and w.should_run():
        self.parking[wid] = asyncio.create_task(self._park(w))
    return keep

  async def _park(self, worker: Worker) -> None:
    wid = worker.worker_id
    try:
      worker.check_process_running()
      with span("node.park"):
        await worker.model.park()
      self.release(wid)
      with contextlib.suppress(Exception):
        await self.node.push_heartbeat()
    except BaseException:
      worker.lease.cancel_owner()  # Keep GPUs reserved until worker.cleanup() closes the child process.
    finally:
      self.parking.pop(wid, None)
      self.notify()

  @contextlib.asynccontextmanager
  async def hold_model(self, worker: Worker):
    """Pause background parking while `worker` saves or drops a run on its model."""
    wid, clean = worker.worker_id, False
    if park := self.parking.get(wid):
      await asyncio.gather(park, return_exceptions=True)
    self.busy.add(wid)
    try:
      yield
      clean = True
    finally:
      self.busy.discard(wid)
      if clean:
        self.reconcile()

  @contextlib.asynccontextmanager
  async def turn(self, worker: Worker, demand: Demand = Demand.READY):
    """Acquire `worker`'s GPU turn (parking evicted holders first), mark it BUSY, and leave it WARM on exit."""
    wid, worker.demand, granted, clean = worker.worker_id, demand, False, False
    self.notify()
    try:
      while worker.should_run():
        changed = self.changed
        if park := self.parking.get(wid):
          await asyncio.gather(park, return_exceptions=True)
          continue
        worker.check_process_running()
        worker.collect_watch()
        if worker.ready_heads()[1]:
          break
        want, cur = self.reconcile().get(worker.proc, ()), self.allocations.get(wid)
        if want and (cur == want or (cur is None and self.claim(wid, want, worker.spec.gpus))):
          self.busy.add(wid)
          if cur is None:
            await worker.wake_on_gpus(want)
            worker.hold_until = time.monotonic() + QUANTUM_MS / 1000
            self.reconcile()
            await self.node.push_heartbeat()
          granted = True
          break
        worker.start_watch(IDLE_MS / 1000 if worker.heads else 0)
        await changed.wait()
      yield granted
      clean = True
    finally:
      self.busy.discard(wid)
      worker.demand = Demand.IDLE
      if clean:
        if granted and not worker.is_sampler and not worker.lease.closing:
          worker.grace_until = time.monotonic() + CYCLE_GRACE_MS / 1000
        self.reconcile()
      self.notify()


class NodeDaemon:
  def __init__(self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store=None):
    self.bus, self.store = bus, store if store is not None else Store()
    self.store.start()
    self.gpus, self.id = gpus, node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.gpu_scheduler = GpuScheduler(self)
    self.gpu_allocations = self.gpu_scheduler.allocations
    self.workers: dict[str, Worker] = {}
    self.tasks: dict[str, asyncio.Task] = {}
    self.state, self.draining = scheduler.State(), False
    self.heartbeat, self.lag, self.swept, self.sweeping = time.monotonic(), 0.0, 0.0, None

  async def serve(self) -> None:
    await self.announce_node()
    wake_id = "$"
    try:
      while True:
        await self.tick()
        before = time.monotonic()
        wake_id = await self.bus.wait_wake(wake_id, int(TICK * 1000), stream="plan_wake")
        self.lag = max(0.0, time.monotonic() - before - TICK)
    finally:
      await self.close()

  async def push_heartbeat(self) -> None:
    on = {wid.split("@", 1)[0]: list(gs) for wid, gs in self.gpu_allocations.items()}
    pids = {p: m.proc.pid for p, w in self.workers.items() if (m := w.model) and m.proc and m.proc.pid is not None}
    self.heartbeat = time.monotonic()
    await self.bus.node_heartbeat(self.id, self.gpus, self.gpu_mib * 2**20, gpu_allocations=on, pids=pids, lag=self.lag)

  async def tick(self) -> None:
    await self.push_heartbeat()
    self.state = await self.read_state()
    self.reap_workers()
    if not self.draining:
      for proc, p in sorted(self.state.procs.items()):
        if p.node == self.id and proc not in self.tasks and self.state.runs(proc):
          self.start_worker(proc)
    if time.time() - self.swept > SWEEP_S and (self.sweeping is None or self.sweeping.done()):
      self.swept, self.sweeping = time.time(), asyncio.create_task(self.bus.sweep(self.store, SWEEP_GRACE_S))
    self.gpu_scheduler.reconcile()
    self.gpu_scheduler.notify()

  def reap_workers(self) -> None:
    for proc, task in list(self.tasks.items()):
      if task.done():
        if not task.cancelled() and (err := task.exception()):
          print(f"{proc} stopped: {err!r}", file=sys.stderr, flush=True)
        self.tasks.pop(proc, None)
        self.workers.pop(proc, None)
      elif self.id not in self.state.nodes and not task.cancelling():
        task.cancel()

  def start_worker(self, proc: str) -> asyncio.Task:
    self.workers[proc] = worker = Worker(self, proc)
    self.tasks[proc] = task = asyncio.create_task(worker.run())
    return task

  async def close(self, graceful: bool = True) -> None:
    self.draining = True
    if self.gpu_scheduler._timer:
      self.gpu_scheduler._timer.cancel()
    self.gpu_scheduler.notify()
    if self.tasks:
      _, pending = await asyncio.wait(self.tasks.values(), timeout=DRAIN_S if graceful else 0)
      await cancel_tasks(*pending)
      await asyncio.gather(*self.tasks.values(), return_exceptions=True)
      self.reap_workers()
    await cancel_tasks(*self.gpu_scheduler.parking.values())
    await asyncio.to_thread(self.store.close)

  async def read_state(self) -> scheduler.State:
    return scheduler.from_state(await self.bus.read_state(backlogs=False))

  async def announce_node(self) -> None:
    await self.push_heartbeat()
    await self.bus.append_event(Ev.NODE_UP, node=self.id, gpus=self.gpus, memory=self.gpu_mib * 2**20)

  def open_trainer(self, proc: str, width: int, env: dict[str, str]) -> Process:
    dist = ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={width}"] if width > 1 else []
    return Process([sys.executable, *dist, "-m", "tinyrl.workers.trainer"], env)

  def open_sampler(self, proc: str, spec: scheduler.Spec, env: dict[str, str]) -> Sampler:
    return Sampler(spec.base, spec.lora, env)


class Worker:
  """Drives one trainer process or sampling engine: own runs -> read batch -> run inside a GPU turn -> commit."""

  def __init__(self, node: NodeDaemon, proc: str):
    self.node, self.bus, self.store, self.proc = node, node.bus, node.store, proc
    self.spec = node.state.procs[proc].spec
    self.is_sampler = self.spec.role == Role.SAMPLER
    self.worker_id = f"{proc}@{node.id}-{os.urandom(6).hex()}"
    self.lease = Lease(self.bus, self.worker_id, self._is_healthy)
    self.model = self.watch_task = None
    self.bound_gpus, self.demand, self.hold_until, self.grace_until = (), Demand.IDLE, 0.0, 0.0
    self.owned_runs: dict[str, OwnedRun] = {}
    self.heads: dict[str, Cmd] = {}
    self.downloads: dict[str, asyncio.Task] = {}
    self.fairness = RunFairness()

  @property
  def has_gpus(self) -> bool:
    return self.worker_id in self.node.gpu_allocations

  def _is_healthy(self) -> bool:
    heartbeat_ok = self.node.draining or time.monotonic() - self.node.heartbeat < LEASE_MS / 1000
    return heartbeat_ok and not (self.model and (p := self.model.proc) and p.poll() is not None)

  def check_process_running(self) -> None:
    if self.model and (p := self.model.proc) and p.poll() is not None:
      raise RuntimeError(f"process {self.proc} exited")

  def active_runs(self) -> list[str]:
    return self.node.state.runs(self.proc) if self.is_sampler else sorted(self.owned_runs)

  def should_run(self) -> bool:
    p, s = self.node.state.procs.get(self.proc), self.node.state
    if self.lease.closing or self.node.draining or not p or p.node != self.node.id or not s.runs(self.proc):
      return False
    allocs = self.node.gpu_allocations
    excl = {g for w in self.node.workers.values() if w.spec.exclusive for g in allocs.get(w.worker_id, ())}
    if self.has_gpus or excl.isdisjoint(self.bound_gpus):
      return True
    can_rebind = self.ready_heads()[0] and any(excl.isdisjoint(b) for b in blocks(self.node.gpus, p.spec.gpus))
    return not self.is_sampler and not can_rebind

  def gpu_slot(self, now: float) -> GpuSlot:
    s, sp, ms = self.node.state, self.spec, self.fairness.total_gpu_ms
    runs = [c.run for c in self.ready_heads()[0] if c.run in s.jobs] or [r for r in self.active_runs() if r in s.jobs]
    cur, lag = self.node.gpu_allocations.get(self.worker_id, ()), max(map(s.ready_lag, runs), default=-ms)
    demand = self.demand
    if demand == Demand.IDLE and cur and now < self.grace_until:
      demand = Demand.FINISH_STEP if any(r.open_grad for r in self.owned_runs.values()) else Demand.READY
    return GpuSlot(sp.gpus, sp.exclusive, demand, cur, self.bound_gpus, not self.is_sampler, self.hold_until, lag, ms)

  async def run(self) -> None:
    sched = self.node.gpu_scheduler
    try:
      if self.node.draining or not await self.lease.open():
        return
      self.node.workers.setdefault(self.proc, self)
      if self.is_sampler or self.spec.gpus == 1:
        self.spawn_model(tuple(range(self.node.gpus)))
      self.start_watch()
      if self.model and self.model.warm_pending:
        async with sched.turn(self):
          pass
      while self.should_run():
        await self.update_owned_runs()
        self.collect_watch()
        ready, failed = self.ready_heads()
        if failed:
          await self.run_batch(failed, limit=1)
        elif ready:
          finishing = any(c.op == Op.OPTIM_STEP and r.open_grad for c in ready if (r := self.owned_runs.get(c.run)))
          async with sched.turn(self, Demand.FINISH_STEP if finishing else Demand.READY) as granted:
            if granted and await self.run_batch(self.ready_heads()[0] or ready):
              break
        else:
          changed = sched.changed
          self.check_process_running()
          self.start_watch(IDLE_MS / 1000 if self.heads else 0)
          await changed.wait()
      await self.stop_watch()
      for r in list(self.owned_runs):
        await self.detach_run(r, drop=False)
    finally:
      self.lease.closing = True
      await run_shielded(self.cleanup)

  def spawn_model(self, gpus: tuple[int, ...]) -> None:
    alloc = {} if self.is_sampler else {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    base = {**scheduler.node_env(self.node.gpus, self.node.gpu_mib * 2**20), **alloc, **os.environ, **self.store.env()}
    env = {**base, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
    open_fn, arg = (self.node.open_sampler, self.spec) if self.is_sampler else (self.node.open_trainer, self.spec.gpus)
    self.model = open_fn(self.proc, arg, env)

  async def wake_on_gpus(self, gpus: tuple[int, ...]) -> None:
    try:
      await self.lease.renew()
      with span("node.resume"):
        if self.model is None:
          self.spawn_model(gpus)
        self.bound_gpus = self.bound_gpus or (gpus if self.is_sampler or self.spec.gpus > 1 else ())
        await self.model.resume(gpus)
    except Exception as e:
      err = Reply.error(e)
      for cmd in await self.claim_commands(self.active_runs(), count=1):
        await self.commit_command(cmd, err, "", None, {})
      raise

  async def cleanup(self) -> None:
    """Close the child first, then release GPUs, fail any stranded trainer command, and drop the lease."""
    await cancel_tasks(self.watch_task, *self.downloads.values(), self.node.gpu_scheduler.parking.get(self.worker_id))
    while self.model:
      try:
        with span("node.close"):
          await self.model.close()
        break
      except (Exception, asyncio.CancelledError) as e:
        print(f"{self.proc} could not close, retrying: {e!r}", file=sys.stderr, flush=True)
        await asyncio.sleep(TICK)
    self.node.gpu_scheduler.release(self.worker_id)
    for r, state in list(self.owned_runs.items()):
      await self.bus.fail_stranded_commands(r, state.token)
    await self.lease.close()

  # ***** trainer run ownership (no-op for samplers) *****

  async def update_owned_runs(self) -> None:
    if self.is_sampler:
      return
    assigned, added = set(self.node.state.runs(self.proc)), False
    for r in [r for r in list(self.owned_runs) if r not in assigned]:
      await self.detach_run(r, drop=True)
    for r in sorted(assigned - self.owned_runs.keys()):
      if got := await self.bus.acquire_run(r, self.worker_id):
        self.owned_runs[r], added = OwnedRun(*got), True
        await self.bus.fail_stranded_commands(r, got[0])
    if added:
      await self.stop_watch()
      self.start_watch()

  async def detach_run(self, run: str, drop: bool) -> None:
    """Save `run` if modified (from GPU or host RAM), release ownership fenced by `(token, last_cmd_id)`, and drop."""
    state, ckpt = self.owned_runs[run], None
    async with self.node.gpu_scheduler.hold_model(self):
      if state.last_cmd_id is not None and state.snapshot is None and state.loaded:
        await self.lease.renew()
        if ckpt := await self.model.call("save_run", run, await self.bus.begin_checkpoint(run, state.token)):
          state.snapshot = f"{ckpt['uri']}\n{ckpt['version']}"
      await self.bus.detach_run(run, state.token, state.last_cmd_id, state.snapshot, ckpt)
      self.forget_run(run)
      if state.loaded and drop:
        await self.model.call("drop_run", run)

  def forget_run(self, run: str) -> None:
    self.owned_runs.pop(run, None)
    self.heads.pop(run, None)

  # ***** queue watching and off-GPU checkpoint downloads *****

  def start_watch(self, delay: float = 0) -> None:
    if self.watch_task is None:
      self.watch_task = asyncio.create_task(self.poll_heads(delay))
      self.watch_task.add_done_callback(lambda _: self.node.gpu_scheduler.notify())

  async def stop_watch(self) -> None:
    await cancel_tasks(self.watch_task)
    self.watch_task = None

  async def poll_heads(self, delay: float = 0) -> list[Cmd]:
    await asyncio.sleep(delay)
    peek = self.bus.peek_samples if self.is_sampler else self.bus.peek_trainer
    while not (cmds := await peek(self.active_runs(), IDLE_MS)) and not self.heads:
      pass
    self.heads = {c.run: c for c in cmds if c.run in set(self.active_runs())}
    if self.is_sampler:
      for req in self.downloads.keys() - {c.request_id for c in self.heads.values()}:
        self.downloads.pop(req).cancel()
      for cmd in [c for c in self.heads.values() if c.request_id not in self.downloads]:
        self.downloads[cmd.request_id] = task = asyncio.create_task(self.download_checkpoint(cmd.run))
        task.add_done_callback(lambda _: self.node.gpu_scheduler.notify())
    return list(self.heads.values())

  async def download_checkpoint(self, session: str):
    with span("node.prepare"):
      try:
        if (run := await self.bus.get_run(session)) is None:
          raise ValueError(f"no sampling session {session}")
        uri = run.get("model_path")
        return await asyncio.to_thread(self.store.local_folder, await self.bus.checkpoint_version(uri)) if uri else None
      except Exception as e:
        return e

  def collect_watch(self) -> None:
    if self.watch_task and self.watch_task.done():
      self.watch_task.result()
      self.watch_task = None
      self.fairness.sync_active(self.heads)

  def ready_heads(self) -> tuple[list[Cmd], list[Cmd]]:
    live, ready, failed = set(self.active_runs()), [], []
    for run, cmd in self.heads.items():
      task = self.downloads.get(cmd.request_id)
      if run in live and (task is None or task.done()):
        (failed if task and isinstance(task.result(), Exception) else ready).append(cmd)
    return ready, failed

  # ***** unified batch execution (1x1 for trainer, wave for sampler) *****

  async def claim_commands(self, runs: list[str], count: int) -> list[Cmd]:
    if self.is_sampler:
      return await self.bus.claim_samples(runs, self.worker_id, count)
    return [c for r in runs if (s := self.owned_runs.get(r)) and (c := await self.bus.claim_trainer(r, s.token))]

  async def run_batch(self, candidates: list[Cmd], *, limit: int | None = None) -> bool:
    width = SAMPLE_RUNS if self.is_sampler and self.spec.lora else 1
    runs = self.fairness.pick(candidates, self.node.state.account, self.node.state.ready_lag, width)
    staged = {}
    for r in runs:
      if (head := self.heads.pop(r, None)) and (task := self.downloads.pop(head.request_id, None)):
        staged[r] = task.result()
    await self.stop_watch()
    await self.lease.renew()
    with span("node.claim"):
      cmds = await self.claim_commands(runs, limit or (SAMPLE_COMMANDS if self.is_sampler else 1))
    if self.is_sampler:
      self.start_watch(0 if cmds else 0.05)
    if not cmds:
      return False
    wave, replies = Wave(self.spec.gpus), []
    async with asyncio.TaskGroup() as group:
      for cmd in cmds:
        group.create_task(self.run_command(cmd, staged.get(cmd.run), wave, replies))
    if not self.is_sampler:
      self.heads.clear()  # Re-peek every owned run next turn so active accounts compete on fresh heads.
    self.fairness.finish_batch([c.run for group in (*self.ready_heads(), cmds) for c in group])
    return not self.is_sampler and not self.spec.lora and cmds[0].op == Op.DELETE_MODEL and replies[0].ok

  async def run_trainer_model(self, cmd: Cmd, wave: Wave, charge: Charge) -> tuple[Reply, dict]:
    state, version = self.owned_runs[cmd.run], None
    if cmd.op == Op.LOAD_WEIGHTS:
      try:
        version = await self.bus.checkpoint_version(json.loads(cmd.body)["uri"])
      except (KeyError, ValueError) as exc:
        return Reply.error(exc), {}
    elif cmd.op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER):
      version = await self.bus.begin_checkpoint(cmd.run, state.token)
    fetch = not state.loaded and cmd.op not in (Op.CREATE_MODEL, Op.DELETE_MODEL)
    rec, saved = await self.bus.get_run(cmd.run) if fetch else None, unpack_snapshot(state.snapshot)[1]
    await self.lease.renew()
    reply, effects = await wave.run(self.model.call("execute", cmd, rec, saved, version), charge)
    state.loaded = bool(effects.get("loaded"))
    return reply, effects

  async def run_command(self, cmd: Cmd, staged, wave: Wave, replies: list[Reply]) -> None:
    charge, acct, effects = Charge(), self.node.state.account(cmd.run), {}
    with span("node.execute"):
      if isinstance(staged, Exception):
        reply = Reply.error(staged)
      elif self.is_sampler:
        reply = await wave.run(self.model.execute(cmd, staged), charge)
      else:
        reply, effects = await self.run_trainer_model(cmd, wave, charge)
    with span("node.publish"):
      billed = await self.commit_command(cmd, reply, acct, charge, effects)
    if billed and charge.gpu_ms:
      self.fairness.record(cmd.run, charge.gpu_ms)
      if acct:
        self.node.state.service[acct] = self.node.state.service.get(acct, 0.0) + charge.gpu_ms
    if DEBUG:
      print(f"{self.proc} {cmd.request_id} {cmd.op:24s} {charge.busy_ms:9d} ms")
    replies.append(reply)

  async def commit_command(self, cmd: Cmd, reply: Reply, acct: str, charge: Charge | None, effects: dict) -> bool:
    if self.is_sampler:
      return await self.bus.commit_sample(cmd, reply, acct, charge=charge)
    if (state := self.owned_runs.get(cmd.run)) is None:
      return False
    mutates = bool(effects.get("mutated"))
    if not await self.bus.commit_reply(cmd, reply, state.token, acct, mutates, charge, effects.get("saved")):
      self.forget_run(cmd.run)
      raise RuntimeError(f"lost ownership of {cmd.run}")
    if mutates:
      state.last_cmd_id, state.snapshot = cmd.entry_id, None
    state.open_grad = cmd.op == Op.FORWARD_BACKWARD and reply.ok
    if cmd.op == Op.DELETE_MODEL and reply.ok:
      self.forget_run(cmd.run)
    return True


if __name__ == "__main__":

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await NodeDaemon(Bus()).serve()

  asyncio.run(main())
