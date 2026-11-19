"""Run local workers and enforce GPU reservations, process leases and fenced ownership of trainer runs."""

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

import torch

from tinyrl.bus import LEASE_MS, Bus, Cmd, Ev, Op, Reply, Role, unpack_snap
from tinyrl.control import scheduler
from tinyrl.control.accounting import Charge, Wave
from tinyrl.control.fairness import Demand, blocks, plan_gpus, select_runs
from tinyrl.control.process import Process
from tinyrl.helpers import DEBUG, getenv
from tinyrl.storage.registry import Registry
from tinyrl.storage.store import Store
from tinyrl.trace import span
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


class NodeDaemon:
  def __init__(
    self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store: Store | None = None
  ):
    self.bus, self.registry = bus, Registry(bus)
    self.store = store if store is not None else Store()
    self.store.start()
    self.gpus, self.id = gpus, node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.on: dict[str, tuple[int, ...]] = {}  # life -> occupied GPUs (single source of local GPU truth)
    self.drains: dict[str, tuple[int, ...]] = {}  # ordered requests for rank-bound checkpoint/restore
    self.lives: dict[str, Lifetime] = {}
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

  async def push_heartbeat(self, wake: bool = False) -> None:
    on = {life.split("@", 1)[0]: list(gs) for life, gs in self.on.items()}
    pids = {p: m.proc.pid for p, lf in self.lives.items() if (m := lf.model) and m.proc and m.proc.pid is not None}
    self.heartbeat = time.monotonic()
    await self.bus.node_heartbeat(self.id, self.gpus, self.gpu_mib * 2**20, wake, on=on, pids=pids, lag=self.lag)

  async def tick(self) -> None:
    await self.push_heartbeat()
    self.state = await self.read_state()
    self.dispatch()
    if not self.draining:
      for proc, p in sorted(self.state.procs.items()):
        if p.node == self.id and proc not in self.tasks and self.state.runs(proc):
          self.start(proc)
    if time.time() - self.swept > SWEEP_S and (self.sweeping is None or self.sweeping.done()):
      self.swept, self.sweeping = time.time(), asyncio.create_task(self.registry.sweep(self.store, SWEEP_GRACE_S))
    self.notify()

  def dispatch(self) -> None:
    for proc, task in list(self.tasks.items()):
      if task.done():
        if not task.cancelled() and (err := task.exception()):
          print(f"{proc} stopped: {err!r}", file=sys.stderr, flush=True)
        del self.tasks[proc]
        self.lives.pop(proc, None)
      elif self.id not in self.state.nodes and not task.cancelling():
        task.cancel()

  def select_gpus(self, proc: str) -> tuple[int, ...]:
    draining = {g for block in self.drains.values() for g in block}
    lives = {p: lf for p, lf in self.lives.items() if lf.life not in self.drains}
    keep, wake_at = plan_gpus(lives, self.on, self.gpus, now := time.monotonic(), draining)
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

  async def claim_after_drain(self, life: str, gpus: tuple[int, ...]) -> None:
    """Reserve a block as its holders park. Overlapping requests claim in arrival order."""
    devices = set(gpus)
    self.drains[life] = gpus
    self.notify()
    try:
      while True:
        first = next(owner for owner, block in self.drains.items() if devices.intersection(block))
        if first == life and self.claim_gpus(life, gpus, len(gpus)):
          return
        await self.changed.wait()
    finally:
      del self.drains[life]
      self.notify()

  def start(self, proc: str) -> asyncio.Task:
    self.tasks[proc] = task = asyncio.create_task(Lifetime(self, proc).run())
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
      self.dispatch()
    await asyncio.to_thread(self.store.close)

  async def read_state(self) -> scheduler.State:
    return scheduler.from_state(await self.bus.read_state())

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
  """One process incarnation: reconciles local GPU quantum residency and run ownership toward `node.state`."""

  def __init__(self, node: NodeDaemon, proc: str):
    self.node, self.bus, self.registry, self.store, self.proc = node, node.bus, node.registry, node.store, proc
    self.spec, self.sampling = (sp := node.state.procs[proc].spec), sp.role == Role.SAMPLER
    self.life = f"{proc}@{node.id}-{os.urandom(6).hex()}"
    self.task = self.model = self.guard = self.closing = self.reader = None
    self.bound_gpus, self.warm, self.demand = (), False, Demand.IDLE
    self.hold_until, self.park_s, self.gpu_ms = 0.0, 0.0, 0.0
    self.owned: dict[str, tuple[str, str | None, str | None]] = {}  # run -> (token, pos, snap)
    self.loaded, self.open_grads = set(), set()
    self.heads, self.prepared, self.run_service = [], {}, {}
    # A newly ready run joins at the least service earned by its active peers.
    self.renewed, self.service_floor, self.idle_since = time.monotonic(), 0.0, 0.0

  def streams(self) -> list[str]:
    return self.node.state.runs(self.proc) if self.sampling else sorted(self.owned)

  def wanted(self) -> bool:
    p = self.node.state.procs.get(self.proc)
    if self.node.draining or not p or p.node != self.node.id or not self.node.state.runs(self.proc):
      return False
    excl = {g for o in self.node.lives.values() if o.spec.exclusive for g in self.node.on.get(o.life, ())}
    if self.life in self.node.on or excl.isdisjoint(self.bound_gpus):
      return True
    # Retire the blocked rank group through normal fenced detach. Its successor can bind a different block.
    return not (
      self.sampling or self.ready()[0] and any(excl.isdisjoint(b) for b in blocks(self.node.gpus, p.spec.gpus))
    )

  def ready_lag(self) -> float:
    s = self.node.state
    runs = [c.run for c in self.ready()[0] if c.run in s.jobs] or [r for r in self.streams() if r in s.jobs]
    return max(map(s.ready_lag, runs), default=-self.gpu_ms)

  async def run(self) -> None:
    self.task = asyncio.current_task()
    try:
      if not await self.open():
        return
      while self.wanted():
        changed = self.node.changed
        await self.sync_runs()
        await self.collect()
        ready, failed = self.ready()
        if failed:
          await self.execute(failed)
          ready, _ = self.ready()
          if not self.wanted():
            break
        # A trainer keeps its turn for CYCLE_GRACE_MS after a command: its client's next one is a round trip away.
        grace = self.idle_since + CYCLE_GRACE_MS / 1000 - time.monotonic()
        linger = not self.sampling and self.life in self.node.on and grace > 0
        await self.sync_gpus(self.turn_demand(ready, linger))
        if not self.wanted():
          break
        if ready and self.life in self.node.on:
          if (cmd := await self.execute(ready)) and cmd.op == Op.DELETE_MODEL and not self.spec.lora:
            break
        else:
          if self.reader is None:
            self.read(IDLE_MS / 1000 if self.heads else 0)
          with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(changed.wait(), grace if linger else None)
      self.demand = Demand.IDLE
      await self.stop_reader()
      for r in list(self.owned):
        await self.detach(r, drop=False)
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

  # ***** lease and subprocess *****

  def spawn_model(self, gpus: tuple[int, ...]) -> None:
    env = {**scheduler.settings(self.node.gpus, self.node.gpu_mib * 2**20), **os.environ, **self.store.env()}
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    if not self.sampling:
      env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    open_s, open_t = self.node.open_sampler, self.node.open_trainer
    self.model = open_s(self.proc, self.spec, env) if self.sampling else open_t(self.proc, self.spec.gpus, env)
    self.warm = self.sampling and env.get("TINYRL_WARM_SAMPLER", "0") != "0"

  async def open(self) -> bool:
    if self.node.draining or not await self.bus.open_lease(self.life):
      return False
    self.node.lives[self.proc] = self
    if self.sampling or self.spec.gpus == 1:
      self.spawn_model(tuple(range(self.node.gpus)))
    self.guard = asyncio.create_task(self.keepalive())
    self.read()
    return True

  async def keep_lease(self) -> None:
    before = time.monotonic()
    if not (
      await self.bus.renew_lease(self.life) or (not self.closing and await self.bus.renew_lease(self.life, True))
    ):
      raise RuntimeError(f"lost lease {self.life}")
    self.renewed = before

  async def keepalive(self) -> None:
    while True:
      await asyncio.sleep(LEASE_MS / 4000)
      if not self.closing and not self.node.draining and time.monotonic() - self.node.heartbeat >= LEASE_MS / 1000:
        break
      try:
        await self.keep_lease()
        if not self.closing and self.model and (p := self.model.proc) and p.poll() is not None:
          break
      except Exception:
        if not self.closing and time.monotonic() - self.renewed >= LEASE_MS / 1000:
          break
    if self.task and not self.task.cancelling():
      self.task.cancel()

  # ***** acquire & detach (L2/L3 <-> process) *****

  async def sync_runs(self) -> None:
    if self.sampling:
      return
    want, added = set(self.node.state.runs(self.proc)), False
    for r in [r for r in list(self.owned) if r not in want]:
      await self.detach(r, drop=True)
    for r in sorted(want - self.owned.keys()):
      if got := await self.bus.acquire_run(r, self.life):
        self.owned[r], added = got, True
        await self.bus.fail_stranded_commands(r, got[0])
    if added:
      await self.stop_reader()
      self.read()

  async def detach(self, run: str, drop: bool) -> None:
    """Flush `run` if dirty (0 bytes if clean), release `run.owner` fenced by `(token, pos)`, and drop if requested."""
    token, pos, snap = self.owned[run]
    checkpoint = None
    if pos is not None and snap is None and run in self.loaded and self.model is not None:
      if self.spec.gpus > 1 and self.life not in self.node.on and torch.cuda.is_available():
        await self.node.claim_after_drain(self.life, self.bound_gpus)
        await self.model.call("resume", None)
      await self.keep_lease()
      version = await self.bus.begin_checkpoint(run, token)
      if checkpoint := await self.model.call("save_run", run, version):
        snap = f"{checkpoint['uri']}\n{checkpoint['version']}"
    await self.bus.detach_run(run, token, pos, snap, checkpoint)
    self.owned.pop(run, None)
    self.heads = [c for c in self.heads if c.run != run]
    for req in [req for req, (r, _) in self.prepared.items() if r == run]:
      self.prepared.pop(req)[1].cancel()
    if run in self.loaded:
      self.loaded.discard(run)
      if drop and self.model is not None:
        await self.model.call("drop_run", run)

  # ***** promote & demote (L1 <-> L0) *****

  def turn_demand(self, ready: list[Cmd], linger: bool) -> Demand:
    if self.open_grads and (
      any(c.op == Op.OPTIM_STEP and c.run in self.open_grads for c in ready) or (linger and not ready)
    ):
      return Demand.FINISH_STEP
    if ready or linger or (self.warm and self.model is not None and self.model.running is None):
      return Demand.READY
    return Demand.IDLE

  async def sync_gpus(self, demand: Demand) -> None:
    if self.model and (p := self.model.proc) and p.poll() is not None:
      raise RuntimeError(f"process {self.proc} exited")
    cur = self.node.on.get(self.life)
    if self.demand != demand:
      self.demand = demand
      self.node.notify()
    want = self.node.select_gpus(self.proc)
    if cur is not None and want != cur:
      t0 = time.monotonic()
      with span("node.park"):
        await (self.model.park() if self.sampling else self.model.call("park"))
      self.park_s = time.monotonic() - t0
      self.node.release_gpus(self.life)
      if self.spec.exclusive and self.wanted():
        # Lend the original devices for checkpointing, including when the exclusive is idle.
        await self.node.claim_after_drain(self.life, cur)
        await self.promote(cur)
      await self.node.push_heartbeat()
    elif cur is None and want and demand != Demand.IDLE and self.node.claim_gpus(self.life, want, self.spec.gpus):
      await self.promote(want)
      await self.node.push_heartbeat()

  async def promote(self, gpus: tuple[int, ...]) -> None:
    try:
      await self.keep_lease()
      t0 = time.monotonic()
      with span("node.resume"):
        if not self.sampling:
          if self.model is None:
            self.bound_gpus = gpus
            self.spawn_model(gpus)
          await self.model.call("resume", gpus[0] if self.spec.gpus == 1 else None)
        elif self.model.running is None:
          self.bound_gpus = gpus
          self.model.env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
          await self.model.start()
        else:
          await self.model.resume()
      resumed = time.monotonic()
      sw = (self.park_s + resumed - t0) / max(0.01, SWITCH_BUDGET) if self.park_s else 0.0
      self.hold_until = resumed + min(MAX_QUANTUM_MS / 1000, max(QUANTUM_MS / 1000, sw))
      self.node.select_gpus(self.proc)
    except Exception as e:
      err = Reply.error(e)
      for cmd in await self.claim(self.streams(), 1):
        if self.sampling:
          await self.bus.commit_sample(cmd, err)
        elif info := self.owned.get(cmd.run):
          await self.bus.commit_reply(cmd, err, info[0])
      raise

  # ***** peeking, preparing off-GPU, and executing *****

  async def claim(self, runs: list[str], count: int) -> list[Cmd]:
    if self.sampling:
      return await self.bus.take_commands(runs, self.life, count, reclaim=True)
    cmds = []
    for run in runs:
      if cmd := await self.bus.take_trainer_command(run, self.owned[run][0]):
        cmds.append(cmd)
    return cmds

  def read(self, delay: float = 0) -> None:
    self.reader = asyncio.create_task(self.peek(delay))
    self.reader.add_done_callback(lambda _: self.node.notify())

  async def stop_reader(self) -> None:
    if self.reader:
      self.reader.cancel()
      await asyncio.gather(self.reader, return_exceptions=True)
      self.reader = None

  async def peek(self, delay: float) -> list[Cmd]:
    await asyncio.sleep(delay)
    while True:
      cmds = await self.bus.peek_commands(self.streams(), IDLE_MS, reclaim=self.sampling)
      # Refresh known heads even when another replica drained them. Otherwise keep reading during model execution.
      if cmds or self.heads:
        break
    for cmd in cmds:
      if cmd.request_id not in self.prepared:
        task = asyncio.create_task(self.prepare(cmd))
        task.add_done_callback(lambda _: self.node.notify())
        self.prepared[cmd.request_id] = (cmd.run, task)
    return cmds

  async def prepare(self, cmd: Cmd):
    with span("node.prepare"):
      try:
        run = await self.bus.get_run(cmd.run)
        if not self.sampling:
          return run
        if run is None:
          raise ValueError(f"no sampling session {cmd.run}")
        if not (uri := run.get("model_path")):
          return None
        return await asyncio.to_thread(self.store.local_folder, await self.registry.version(uri))
      except Exception as e:
        return e

  async def collect(self) -> None:
    if not self.reader or not self.reader.done():
      return
    live = set(self.streams())
    self.heads, self.reader = [c for c in self.reader.result() if c.run in live], None
    keep = {c.request_id for c in self.heads}
    for req in self.prepared.keys() - keep:
      self.prepared.pop(req)[1].cancel()
    self.run_service = {c.run: self.run_service.get(c.run, self.service_floor) for c in self.heads}

  def ready(self) -> tuple[list[Cmd], list[Cmd]]:
    live = set(self.streams())
    done = [c for c in self.heads if c.run in live and (t := self.prepared.get(c.request_id)) and t[1].done()]
    failed = [c for c in done if isinstance(self.prepared[c.request_id][1].result(), Exception)]
    return [c for c in done if c not in failed], failed

  async def execute(self, candidates: list[Cmd]) -> Cmd | None:
    for c in candidates:
      self.run_service[c.run] = max(self.run_service.get(c.run, self.service_floor), self.service_floor)
    width = SAMPLE_RUNS if self.sampling and self.spec.lora else 1
    runs = select_runs(self.node.state, candidates, self.run_service, width)
    inputs = {c.run: self.prepared[c.request_id][1].result() for c in candidates if c.run in runs}
    bad = any(isinstance(v, Exception) for v in inputs.values())
    await self.stop_reader()
    await self.keep_lease()
    with span("node.claim"):
      cmds = await self.claim(runs, SAMPLE_COMMANDS if self.sampling and not bad else 1)
    self.heads = [c for c in self.heads if c.run not in runs]
    self.read()
    if not cmds:
      return None
    wave = Wave(self.spec.gpus)
    first, result = cmds[0], None
    if self.sampling:
      async with asyncio.TaskGroup() as group:
        for cmd in cmds:
          group.create_task(self.answer(cmd, inputs[cmd.run], wave))
    else:
      if (await self.answer(first, inputs[first.run], wave)).ok:
        result = first
    self.idle_since = time.monotonic() if not bad else self.idle_since
    r1, r2 = self.ready()
    self.service_floor = max(
      self.service_floor, min(self.run_service.get(c.run, self.service_floor) for c in [*r1, *r2, *cmds])
    )
    return result

  async def execute_trainer(self, cmd: Cmd, run: dict, wave: Wave, charge: Charge):
    token, _, snap = self.owned[cmd.run]
    version = None
    if cmd.op == Op.LOAD_WEIGHTS:
      # Resolve at dispatch: a preceding save can still be executing while this command is prefetched.
      try:
        version = await self.registry.version(json.loads(cmd.body)["uri"])
      except (KeyError, ValueError) as exc:
        return Reply.error(exc), {}
    elif cmd.op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER):
      version = await self.bus.begin_checkpoint(cmd.run, token)
    saved = unpack_snap(snap)[1] if snap and cmd.run not in self.loaded and cmd.op != Op.LOAD_WEIGHTS else None
    await self.keep_lease()
    reply, effects = await wave.run(self.model.call("execute", cmd, run, saved, version), charge)
    self.loaded.add(cmd.run) if effects.get("loaded") else self.loaded.discard(cmd.run)
    return reply, effects

  async def answer(self, cmd: Cmd, inputs, wave: Wave) -> Reply:
    charge = Charge()
    acct = self.node.state.account(cmd.run)
    effects = {}
    with span("node.execute"):
      if isinstance(inputs, Exception):
        reply = Reply.error(inputs)
      elif self.sampling:
        reply = await wave.run(self.model.execute(cmd, inputs), charge)
      else:
        reply, effects = await self.execute_trainer(cmd, inputs, wave, charge)
    with span("node.publish"):
      if self.sampling:
        billed = await self.bus.commit_sample(cmd, reply, acct, charge=charge)
      else:
        token = self.owned[cmd.run][0]
        mutates = bool(effects.get("mutated"))
        billed, old = await self.bus.commit_reply(cmd, reply, token, acct, mutates, charge, effects.get("saved"))
        if not billed:
          self.owned.pop(cmd.run, None)
          self.loaded.discard(cmd.run)
          raise RuntimeError(f"lost ownership of {cmd.run}")
        if cmd.op == Op.DELETE_MODEL and reply.ok:
          self.owned.pop(cmd.run, None)
        elif mutates:
          self.owned[cmd.run] = (token, cmd.entry_id, None)
          if old:
            await self.registry.unregister_checkpoints(unpack_snap(old)[0])
        self.open_grads.discard(cmd.run)
        if cmd.op == Op.FORWARD_BACKWARD and reply.ok:
          self.open_grads.add(cmd.run)
    if billed and charge.gpu_ms:
      self.run_service[cmd.run] = self.run_service.get(cmd.run, self.service_floor) + charge.gpu_ms
      self.gpu_ms += charge.gpu_ms
      if acct:
        self.node.state.service[acct] = self.node.state.service.get(acct, 0.0) + charge.gpu_ms
    if DEBUG:
      print(f"{self.proc} {cmd.request_id} {cmd.op:24s} {charge.busy_ms:9d} ms")
    return reply

  async def cleanup(self) -> None:
    """Close the child first, then release GPUs (`on`), fail stranded commands, and drop the lease."""
    futures = [f for f in [self.reader, *(t for _, t in self.prepared.values())] if f]
    for f in futures:
      f.cancel()
    await asyncio.gather(*futures, return_exceptions=True)
    while self.model:
      try:
        with span("node.close"):
          await self.model.close()
        break
      except (Exception, asyncio.CancelledError) as e:
        print(f"{self.proc} could not close, retrying: {e!r}", file=sys.stderr, flush=True)
        await asyncio.sleep(TICK)
    self.node.release_gpus(self.life)
    if self.guard:
      self.guard.cancel()
      await asyncio.gather(self.guard, return_exceptions=True)
    for r, (token, _, _) in list(self.owned.items()):
      await self.bus.fail_stranded_commands(r, token)
    await self.bus.drop_lease(self.life)


if __name__ == "__main__":

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await NodeDaemon(Bus()).serve()

  asyncio.run(main())
