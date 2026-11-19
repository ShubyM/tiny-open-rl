"""Node daemon and unified `Worker` loop: heartbeats, exclusive local GPU ownership, leases, and batch execution."""

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
  if live := [t for t in tasks if t]:
    for t in live:
      t.cancel()
    await asyncio.gather(*live, return_exceptions=True)


@dataclass
class OwnedRun:
  """Single-writer state for a training run held by a `Worker`."""

  token: str
  last_cmd_id: str | None = None
  snapshot: str | None = None
  loaded: bool = False
  open_grad: bool = False


class NodeDaemon:
  def __init__(self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store=None):
    self.bus = bus
    self.store = store if store is not None else Store()
    self.store.start()
    self.gpus, self.id = gpus, node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.gpu_allocations: dict[str, tuple[int, ...]] = {}  # worker_id -> held GPUs (local source of truth)
    self.workers: dict[str, Worker] = {}
    self.tasks: dict[str, asyncio.Task] = {}
    self.timer: asyncio.TimerHandle | None = None
    self.state, self.changed, self.draining = scheduler.State(), asyncio.Event(), False
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
    allocs = {wid.split("@", 1)[0]: list(gs) for wid, gs in self.gpu_allocations.items()}
    pids = {p: m.proc.pid for p, w in self.workers.items() if (m := w.model) and m.proc and m.proc.pid is not None}
    self.heartbeat = time.monotonic()
    await self.bus.node_heartbeat(
      self.id, self.gpus, self.gpu_mib * 2**20, gpu_allocations=allocs, pids=pids, lag=self.lag
    )

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
    self.notify()

  def reap_workers(self) -> None:
    for proc, task in list(self.tasks.items()):
      if task.done():
        if not task.cancelled() and (err := task.exception()):
          print(f"{proc} stopped: {err!r}", file=sys.stderr, flush=True)
        del self.tasks[proc]
        self.workers.pop(proc, None)
      elif self.id not in self.state.nodes and not task.cancelling():
        task.cancel()

  def select_gpus(self, proc: str) -> tuple[int, ...]:
    occupied = set().union(*self.gpu_allocations.values())
    slots = {p: w.gpu_slot() for p, w in self.workers.items() if w.should_run()}
    keep, wake_at = plan_gpus(slots, self.gpus, now := time.monotonic(), occupied)
    if self.timer:
      self.timer.cancel()
    self.timer = asyncio.get_running_loop().call_later(max(0.0, wake_at - now), self.notify) if wake_at else None
    return keep.get(proc, ())

  def claim_gpus(self, worker_id: str, gpus: tuple[int, ...], width: int) -> bool:
    if len(gpus) != width or len(set(gpus)) != width or any(g < 0 or g >= self.gpus for g in gpus):
      return False
    if {g for k, gs in self.gpu_allocations.items() if k != worker_id for g in gs} & set(gpus):
      return False
    self.gpu_allocations[worker_id] = gpus
    self.notify()
    return True

  def release_gpus(self, worker_id: str) -> None:
    if self.gpu_allocations.pop(worker_id, None) is not None:
      self.notify()

  def start_worker(self, proc: str) -> asyncio.Task:
    self.tasks[proc] = task = asyncio.create_task(Worker(self, proc).run())
    return task

  async def close(self, graceful: bool = True) -> None:
    self.draining = True
    if self.timer:
      self.timer.cancel()
    self.notify()
    if self.tasks:
      _, pending = await asyncio.wait(self.tasks.values(), timeout=DRAIN_S if graceful else 0)
      for task in pending:
        task.cancel()
      await asyncio.gather(*self.tasks.values(), return_exceptions=True)
      self.reap_workers()
    await asyncio.to_thread(self.store.close)

  async def read_state(self) -> scheduler.State:
    return scheduler.from_state(await self.bus.read_state(backlogs=False))

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


class Worker:
  """Supervises one trainer process or sampling engine: lease, GPU quantum turns, queue watch, and batch execution."""

  def __init__(self, node: NodeDaemon, proc: str):
    self.node, self.bus, self.store, self.proc = node, node.bus, node.store, proc
    self.spec = node.state.procs[proc].spec
    self.is_sampler = self.spec.role == Role.SAMPLER
    self.worker_id = f"{proc}@{node.id}-{os.urandom(6).hex()}"
    self.model = self.lease_task = self.watch_task = None
    self.closing = False
    self.bound_gpus, self.demand, self.hold_until, self.idle_since = (), Demand.IDLE, 0.0, 0.0
    self.owned_runs: dict[str, OwnedRun] = {}
    self.heads: dict[str, Cmd] = {}
    self.downloads: dict[str, asyncio.Task] = {}  # request_id -> checkpoint download task (sampler only)
    self.fairness = RunFairness()

  @property
  def has_gpus(self) -> bool:
    return self.worker_id in self.node.gpu_allocations

  def active_runs(self) -> list[str]:
    return self.node.state.runs(self.proc) if self.is_sampler else sorted(self.owned_runs)

  def should_run(self) -> bool:
    p = self.node.state.procs.get(self.proc)
    if self.node.draining or not p or p.node != self.node.id or not self.node.state.runs(self.proc):
      return False
    excl = {
      g for w in self.node.workers.values() if w.spec.exclusive for g in self.node.gpu_allocations.get(w.worker_id, ())
    }
    if self.has_gpus or excl.isdisjoint(self.bound_gpus):
      return True
    # Exit when our bound GPUs are taken by an exclusive and a replacement worker can bind a free block.
    return not self.is_sampler and not (
      self.ready_heads()[0] and any(excl.isdisjoint(b) for b in blocks(self.node.gpus, p.spec.gpus))
    )

  def gpu_slot(self) -> GpuSlot:
    s, sp, ms = self.node.state, self.spec, self.fairness.total_gpu_ms
    runs = [c.run for c in self.ready_heads()[0] if c.run in s.jobs] or [r for r in self.active_runs() if r in s.jobs]
    cur, lag = self.node.gpu_allocations.get(self.worker_id, ()), max(map(s.ready_lag, runs), default=-ms)
    return GpuSlot(
      sp.gpus, sp.exclusive, self.demand, cur, self.bound_gpus, not self.is_sampler, self.hold_until, lag, ms
    )

  async def run(self) -> None:
    try:
      if not await self.start(asyncio.current_task()):
        return
      while self.should_run():
        changed = self.node.changed
        await self.update_owned_runs()
        self.collect_watch()
        ready, failed = self.ready_heads()
        if failed:
          await self.run_batch(failed, limit=1)
          ready, _ = self.ready_heads()
          if not self.should_run():
            break
        demand, wait_s = self.gpu_demand(ready)
        await self.update_gpu_allocation(demand)
        if not self.should_run():
          break
        if ready and self.has_gpus:
          if await self.run_batch(ready):
            break
        else:
          if self.watch_task is None:
            self.start_watch(IDLE_MS / 1000 if self.heads else 0)
          with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(changed.wait(), wait_s)
      self.demand = Demand.IDLE
      await self.stop_watch()
      for r in list(self.owned_runs):
        await self.detach_run(r, drop=False)
    finally:
      self.demand, self.closing = Demand.IDLE, True
      cleanup = asyncio.create_task(self.cleanup())
      self.node.notify()
      while True:
        try:
          await asyncio.shield(cleanup)
          break
        except asyncio.CancelledError:
          if cleanup.cancelled():
            cleanup = asyncio.create_task(self.cleanup())

  # ***** lease and subprocess lifecycle *****

  def spawn_model(self, gpus: tuple[int, ...]) -> None:
    env = {**scheduler.node_env(self.node.gpus, self.node.gpu_mib * 2**20), **os.environ, **self.store.env()}
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    if self.is_sampler:
      self.model = self.node.open_sampler(self.proc, self.spec, env)
    else:
      env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
      self.model = self.node.open_trainer(self.proc, self.spec.gpus, env)

  async def start(self, owner_task: asyncio.Task) -> bool:
    if self.node.draining or not await self.bus.open_lease(self.worker_id):
      return False
    self.node.workers[self.proc] = self
    if self.is_sampler or self.spec.gpus == 1:
      self.spawn_model(tuple(range(self.node.gpus)))
    self.lease_task = asyncio.create_task(self.keep_lease_alive(owner_task))
    self.start_watch()
    return True

  async def renew_lease(self) -> None:
    await self.bus.renew_lease(self.worker_id, recover=not self.closing)

  async def keep_lease_alive(self, owner_task: asyncio.Task) -> None:
    last_renewed = time.monotonic()
    while True:
      await asyncio.sleep(LEASE_MS / 4000)
      if not self.closing and not self.node.draining and time.monotonic() - self.node.heartbeat >= LEASE_MS / 1000:
        break
      try:
        started = time.monotonic()
        await self.renew_lease()
        last_renewed = started
        if not self.closing and self.model and (p := self.model.proc) and p.poll() is not None:
          break
      except Exception:
        if not self.closing and time.monotonic() - last_renewed >= LEASE_MS / 1000:
          break
    if not owner_task.cancelling():
      owner_task.cancel()

  async def cleanup(self) -> None:
    """Close the child first, then release GPUs, fail any stranded trainer command, and drop the lease."""
    await cancel_tasks(self.watch_task, *self.downloads.values())
    while self.model:
      try:
        with span("node.close"):
          await self.model.close()
        break
      except (Exception, asyncio.CancelledError) as e:
        print(f"{self.proc} could not close, retrying: {e!r}", file=sys.stderr, flush=True)
        await asyncio.sleep(TICK)
    self.node.release_gpus(self.worker_id)
    await cancel_tasks(self.lease_task)
    for r, state in list(self.owned_runs.items()):
      await self.bus.fail_stranded_commands(r, state.token)
    await self.bus.drop_lease(self.worker_id)

  # ***** trainer run ownership (no-op for samplers) *****

  async def update_owned_runs(self) -> None:
    if self.is_sampler:
      return
    assigned, added = set(self.node.state.runs(self.proc)), False
    for r in [r for r in list(self.owned_runs) if r not in assigned]:
      await self.detach_run(r, drop=True)
    for r in sorted(assigned - self.owned_runs.keys()):
      if got := await self.bus.acquire_run(r, self.worker_id):
        token, last_cmd_id, snapshot = got
        self.owned_runs[r], added = OwnedRun(token, last_cmd_id, snapshot), True
        await self.bus.fail_stranded_commands(r, token)
    if added:
      await self.stop_watch()
      self.start_watch()

  async def detach_run(self, run: str, drop: bool) -> None:
    """Save `run` if modified (from GPU or host RAM), release ownership fenced by `(token, last_cmd_id)`, and drop."""
    state, checkpoint = self.owned_runs[run], None
    if state.last_cmd_id is not None and state.snapshot is None and state.loaded:
      await self.renew_lease()
      version = await self.bus.begin_checkpoint(run, state.token)
      if checkpoint := await self.model.call("save_run", run, version):
        state.snapshot = f"{checkpoint['uri']}\n{checkpoint['version']}"
    await self.bus.detach_run(run, state.token, state.last_cmd_id, state.snapshot, checkpoint)
    self.forget_run(run)
    if state.loaded and drop:
      await self.model.call("drop_run", run)

  def forget_run(self, run: str) -> None:
    self.owned_runs.pop(run, None)
    self.heads.pop(run, None)

  # ***** local GPU residency and turns *****

  def gpu_demand(self, ready: list[Cmd]) -> tuple[Demand, float | None]:
    grace = self.idle_since + CYCLE_GRACE_MS / 1000 - time.monotonic()
    linger = not self.is_sampler and self.has_gpus and grace > 0
    open_grads = {r for r, s in self.owned_runs.items() if s.open_grad}
    if open_grads and (any(c.op == Op.OPTIM_STEP and c.run in open_grads for c in ready) or (linger and not ready)):
      return Demand.FINISH_STEP, grace if linger else None
    warm = self.model is not None and self.model.warm_pending
    return (Demand.READY if ready or linger or warm else Demand.IDLE), grace if linger else None

  async def update_gpu_allocation(self, demand: Demand) -> None:
    if self.model and (p := self.model.proc) and p.poll() is not None:
      raise RuntimeError(f"process {self.proc} exited")
    cur = self.node.gpu_allocations.get(self.worker_id)
    if self.demand != demand:
      self.demand = demand
      self.node.notify()
    want = self.node.select_gpus(self.proc)
    if cur is not None and want != cur:
      with span("node.park"):
        await self.model.park()
      self.node.release_gpus(self.worker_id)
      await self.node.push_heartbeat()
    elif cur is None and want and self.node.claim_gpus(self.worker_id, want, self.spec.gpus):
      await self.acquire_gpus(want)
      await self.node.push_heartbeat()

  async def acquire_gpus(self, gpus: tuple[int, ...]) -> None:
    try:
      await self.renew_lease()
      with span("node.resume"):
        if self.model is None:
          self.spawn_model(gpus)
        if not self.bound_gpus and (self.is_sampler or self.spec.gpus > 1):
          self.bound_gpus = gpus
        await self.model.resume(gpus)
      self.hold_until = time.monotonic() + QUANTUM_MS / 1000
      self.node.select_gpus(self.proc)
    except Exception as e:
      err = Reply.error(e)
      for cmd in await self.claim_commands(self.active_runs(), count=1):
        await self.commit_command(cmd, err, "", None, {})
      raise

  # ***** queue watching and off-GPU checkpoint downloads *****

  def start_watch(self, delay: float = 0) -> None:
    self.watch_task = asyncio.create_task(self.poll_heads(delay))
    self.watch_task.add_done_callback(lambda _: self.node.notify())

  async def stop_watch(self) -> None:
    await cancel_tasks(self.watch_task)
    self.watch_task = None

  async def poll_heads(self, delay: float = 0) -> list[Cmd]:
    await asyncio.sleep(delay)
    peek = self.bus.peek_samples if self.is_sampler else self.bus.peek_trainer
    while not (cmds := await peek(self.active_runs(), IDLE_MS)) and not self.heads:
      pass
    live = set(self.active_runs())
    self.heads = {c.run: c for c in cmds if c.run in live}
    if self.is_sampler:
      for req in self.downloads.keys() - {c.request_id for c in self.heads.values()}:
        self.downloads.pop(req).cancel()
      for cmd in self.heads.values():
        if cmd.request_id not in self.downloads:
          self.downloads[cmd.request_id] = task = asyncio.create_task(self.download_checkpoint(cmd.run))
          task.add_done_callback(lambda _: self.node.notify())
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
      if run not in live:
        continue
      task = self.downloads.get(cmd.request_id)
      if task is None:
        ready.append(cmd)
      elif task.done():
        (failed if isinstance(task.result(), Exception) else ready).append(cmd)
    return ready, failed

  # ***** unified batch execution (1x1 for trainer, wave for sampler) *****

  async def claim_commands(self, runs: list[str], count: int) -> list[Cmd]:
    if self.is_sampler:
      return await self.bus.claim_samples(runs, self.worker_id, count)
    cmds = []
    for r in runs:
      if (state := self.owned_runs.get(r)) and (cmd := await self.bus.claim_trainer(r, state.token)):
        cmds.append(cmd)
    return cmds

  async def run_batch(self, candidates: list[Cmd], *, limit: int | None = None) -> bool:
    width = SAMPLE_RUNS if self.is_sampler and self.spec.lora else 1
    runs = self.fairness.pick(candidates, self.node.state.account, self.node.state.ready_lag, width)
    staged = {}
    for r in runs:
      if (head := self.heads.pop(r, None)) and (task := self.downloads.pop(head.request_id, None)):
        staged[r] = task.result() if task.done() else None
        task.cancel()
    await self.stop_watch()
    await self.renew_lease()
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
      self.heads.clear()  # Re-peek every owned run before the next command so active accounts compete fresh.
      if not any(isinstance(v, Exception) for v in staged.values()):
        self.idle_since = time.monotonic()
    r1, r2 = self.ready_heads()
    self.fairness.finish_batch([c.run for c in (*r1, *r2, *cmds)])
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
    need_rec = not state.loaded and cmd.op not in (Op.CREATE_MODEL, Op.DELETE_MODEL)
    rec = await self.bus.get_run(cmd.run) if need_rec else None
    saved = unpack_snapshot(state.snapshot)[1] if state.snapshot else None
    await self.renew_lease()
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
    if cmd.op == Op.DELETE_MODEL and reply.ok:
      self.forget_run(cmd.run)
    else:
      if mutates:
        state.last_cmd_id, state.snapshot = cmd.entry_id, None
      state.open_grad = cmd.op == Op.FORWARD_BACKWARD and reply.ok
    return True


if __name__ == "__main__":

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await NodeDaemon(Bus()).serve()

  asyncio.run(main())
