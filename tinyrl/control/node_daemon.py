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

from tinyrl.bus import LEASE_MS, Bus, Cmd, Ev, Op, Reply, Role, unpack_snap
from tinyrl.control import scheduler
from tinyrl.control.accounting import Charge, Wave
from tinyrl.control.process import Process
from tinyrl.helpers import DEBUG, getenv
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


def blocks(total: int, width: int) -> list[tuple[int, ...]]:
  step = width if width > 0 and total % width == 0 else 1
  return [tuple(range(i, i + width)) for i in range(0, total - width + 1, step)] if 1 <= width <= total else []


def plan_gpus(
  lives: dict[str, Lifetime], on: dict[str, tuple[int, ...]], total: int, now: float
) -> tuple[dict[str, tuple[int, ...]], float | None]:
  """Select active GPU blocks: hold `ACTIVE_QUANTUM`, fill rest by `(exclusive, lag, -gpu_ms)`, keep idle warm."""
  cur = {p: on.get(lf.life, ()) for p, lf in lives.items()}
  held = {g for gs in on.values() for g in gs}
  pinned = {g for lf in lives.values() if lf.spec.role == Role.TRAINER and lf.spec.gpus > 1 for g in lf.bound_gpus}
  want_excl = any(lf.spec.exclusive and lf.active and not cur[p] for p, lf in lives.items())
  # 1. Exclusive residents and busy residents inside ACTIVE_QUANTUM stay put without switching.
  residents = {p: lf for p, lf in lives.items() if cur[p] and lf.wanted()}
  ends = {p: lf.granted_at + lf.quantum for p, lf in residents.items() if lf.active and not want_excl}
  holding = {p: end for p, end in ends.items() if end > now}
  keep = {
    p: cur[p]
    for p, lf in residents.items()
    if p in holding or lf.spec.exclusive or (p in ends and getattr(lf, "step_ready", False))
  }
  taken, hold = set().union(*keep.values()), set()
  excl = {g for p, block in keep.items() if lives[p].spec.exclusive for g in block}
  # 2. OVERTIME residents and ready waiters claim open blocks in priority order; prefer cur[p] or truly empty GPUs.
  waiting = False
  for p, lf in sorted(lives.items(), key=lambda kv: kv[1].priority(), reverse=True):
    if p in keep or not lf.active or not lf.wanted():
      continue
    opts = [lf.bound_gpus] if lf.bound_gpus else blocks(total, lf.spec.gpus)
    if free := [b for b in opts if not (taken & set(b)) and (b == cur[p] or not (hold & set(b)))]:
      avoid = pinned if lf.spec.exclusive else set()
      keep[p] = min(free, key=lambda b: (b != cur[p], not avoid.isdisjoint(b), not held.isdisjoint(b), b))
      taken |= set(keep[p])
      excl |= set(keep[p]) if lf.spec.exclusive else set()
    else:
      waiting = True
      if cand := [b for b in opts if not (excl & set(b))]:
        hold |= set(cand[0])
  # 3. Idle residents stay warm in VRAM on uncontested GPUs (zero park/resume when alone).
  keep.update((p, b) for p, b in cur.items() if b and p not in keep and taken.isdisjoint(b) and lives[p].wanted())
  return keep, min(holding.values(), default=None) if waiting else None


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
      elif not task.cancelling() and (
        self.id not in self.state.nodes or ((lf := self.lives.get(proc)) and not lf.closing and lf.exited())
      ):
        task.cancel()

  def select_gpus(self, proc: str) -> tuple[int, ...]:
    keep, wake_at = plan_gpus(self.lives, self.on, self.gpus, now := time.monotonic())
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
    for task in pending:
      task.cancel()
    if self.tasks:
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
  """One process incarnation: owns local GPU quantum residency and fenced subprocess teardown."""

  def __init__(self, node: NodeDaemon, proc: str):
    self.node, self.bus, self.store, self.proc = node, node.bus, node.store, proc
    self.spec = node.state.procs[proc].spec
    self.life = f"{proc}@{node.id}-{os.urandom(6).hex()}"
    self.task = self.model = self.closing = self.reader = None
    self.reader_streams: list[str] = []
    self.bound_gpus, self.active = (), False
    self.granted_at, self.quantum, self.park_s, self.gpu_ms = 0.0, QUANTUM_MS / 1000, 0.0, 0.0
    self.heads: dict[str, Cmd] = {}
    self.service: dict[str, float] = {}

  def exited(self) -> bool:
    return bool(self.model and (p := self.model.proc) and p.poll() is not None)

  def streams(self) -> list[str]:
    return self.node.state.runs(self.proc)

  def wanted(self) -> bool:
    p = self.node.state.procs.get(self.proc)
    return not self.node.draining and bool(p) and p.node == self.node.id and bool(self.node.state.runs(self.proc))

  def priority(self) -> tuple:
    s = self.node.state
    runs = [r for r in self.ready_runs() if r in s.jobs] or [r for r in self.streams() if r in s.jobs]
    top = round(max(map(s.ready_lag, runs), default=-self.gpu_ms), 3)
    return (self.spec.exclusive, top, round(-self.gpu_ms, 3), self.life in self.node.on, self.proc)

  def env(self, gpus: tuple[int, ...]) -> dict[str, str]:
    e = {**scheduler.settings(self.node.gpus, self.node.gpu_mib * 2**20), **os.environ, **self.store.env()}
    return e | {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}

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
      await self.loop()
    finally:
      self.active = False
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
    if self.reader:
      self.reader.cancel()
      await asyncio.gather(self.reader, return_exceptions=True)
      self.reader, self.reader_streams = None, []

  async def poll_reader(self, peek, streams: list[str], fetch: bool = True) -> list[Cmd] | None:
    if self.reader and self.reader.done():
      res, self.reader, self.reader_streams = self.reader.result(), None, []
      return res
    if self.reader and self.reader_streams == streams:
      return None
    await self.stop_reader()
    return await peek(streams, 0) if fetch and streams else None

  async def sync_gpus(self, has_work: bool) -> None:
    if self.exited():
      raise RuntimeError(f"process {self.proc} exited")
    cur = self.node.on.get(self.life)
    if self.active != has_work:
      self.active = has_work
      self.node.notify()
    want = self.node.select_gpus(self.proc)
    if cur is not None and want != cur:
      t0 = time.monotonic()
      with span("node.park"):
        await self.park_model()
      self.park_s = time.monotonic() - t0
      self.node.release_gpus(self.life)
      await self.node.push_heartbeat()
    elif cur is None and want and has_work and self.node.claim_gpus(self.life, want, self.spec.gpus):
      try:
        t0 = time.monotonic()
        with span("node.resume"):
          await self.resume_model(want)
        self.granted_at = time.monotonic()
        sw = (self.park_s + self.granted_at - t0) / max(0.01, SWITCH_BUDGET) if self.park_s else 0.0
        self.quantum = min(MAX_QUANTUM_MS / 1000, max(QUANTUM_MS / 1000, sw))
        self.node.select_gpus(self.proc)
      except Exception as e:
        await self.fail_promote(Reply.error(e))
        raise
      await self.node.push_heartbeat()

  def bill(self, cmd: Cmd, acct: str, charge: Charge, billed: bool, floor: float) -> None:
    if billed and charge.gpu_ms:
      self.service[cmd.run] = self.service.get(cmd.run, floor) + charge.gpu_ms
      self.gpu_ms += charge.gpu_ms
      if acct:
        self.node.state.service[acct] = self.node.state.service.get(acct, 0.0) + charge.gpu_ms
    if DEBUG:
      print(f"{self.proc} {cmd.request_id} {cmd.op:24s} {charge.busy_ms:9d} ms")

  async def cleanup(self) -> None:
    """Close the child first, then release GPUs (`on`), fail stranded trainer work, and drop the lease."""
    tasks = [t for t in (self.reader, *getattr(self, "staged", {}).values()) if t]
    for t in tasks:
      t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
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
      for r, (token, _, _) in list(getattr(self, "owned", {}).items()):
        await self.bus.fail_stranded_commands(r, token)
    finally:
      await self.bus.drop_lease(self.life)


class TrainerLife(Lifetime):
  """Single-writer training process: owns runs via epoch tokens, executes one command at a time."""

  def __init__(self, node: NodeDaemon, proc: str):
    super().__init__(node, proc)
    self.owned, self.loaded, self.open_grads, self.step_ready, self.idle_since = {}, set(), set(), False, 0.0

  def streams(self) -> list[str]:
    return sorted(self.owned)

  def ready_runs(self) -> list[str]:
    return [r for r in self.heads if r in self.owned]

  def wanted(self) -> bool:
    if not super().wanted():
      return False
    excl = {g for o in self.node.lives.values() if o.spec.exclusive for g in self.node.on.get(o.life, ())}
    if self.life in self.node.on or excl.isdisjoint(self.bound_gpus):
      return True
    # Retire the blocked rank group through normal fenced detach. Its successor can bind a different block.
    return not (self.heads and any(excl.isdisjoint(b) for b in blocks(self.node.gpus, self.spec.gpus)))

  def spawn_model(self, gpus: tuple[int, ...]) -> None:
    e = self.env(gpus)
    e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    self.model = self.node.open_trainer(self.proc, self.spec.gpus, e)

  async def park_model(self) -> None:
    await self.model.call("park")

  async def resume_model(self, gpus: tuple[int, ...]) -> None:
    if self.model is None:
      self.bound_gpus = gpus
      self.spawn_model(gpus)
    await self.model.call("resume", gpus[0] if self.spec.gpus == 1 else None)

  async def fail_promote(self, err: Reply) -> None:
    for r, (token, _, _) in list(self.owned.items()):
      if cmd := await self.bus.take_trainer_command(r, token):
        await self.bus.commit_reply(cmd, err, token)

  async def sync_runs(self) -> None:
    want = set(self.node.state.runs(self.proc))
    for r in [r for r in list(self.owned) if r not in want]:
      await self.detach(r, drop=True)
    for r in sorted(want - self.owned.keys()):
      if got := await self.bus.acquire_run(r, self.life):
        self.owned[r] = got
        await self.bus.fail_stranded_commands(r, got[0])

  async def detach(self, run: str, drop: bool) -> None:
    """Flush `run` if dirty (0 bytes if clean), release `run.owner` fenced by `(token, pos)`, and drop if requested."""
    token, pos, snap = self.owned[run]
    checkpoint = None
    if pos is not None and snap is None and run in self.loaded and self.model is not None:
      version = await self.bus.begin_checkpoint(run, token)
      if checkpoint := await self.model.call("save_run", run, version):
        snap = f"{checkpoint['uri']}\n{checkpoint['version']}"
    await self.bus.detach_run(run, token, pos, snap, checkpoint)
    for d in (self.owned, self.heads, self.service):
      d.pop(run, None)
    self.open_grads.discard(run)
    if run in self.loaded:
      self.loaded.discard(run)
      if drop and self.model is not None:
        await self.model.call("drop_run", run)

  async def collect(self) -> None:
    missing = [r for r in sorted(self.owned) if r not in self.heads]
    if cmds := await self.poll_reader(self.bus.peek_trainer, missing):
      self.heads |= {c.run: c for c in cmds if c.run in self.owned}
    self.service = {r: self.service[r] for r in self.heads if r in self.service}

  async def execute_one(self, ready: list[Cmd]) -> Cmd | None:
    floor = min((self.service[c.run] for c in ready if c.run in self.service), default=0.0)
    [run] = scheduler.select_runs(self.node.state, ready, self.service, 1)
    self.heads.pop(run, None)
    token, _, snap = self.owned[run]
    with span("node.claim"):
      cmd = await self.bus.take_trainer_command(run, token)
    if cmd is None:
      return None
    charge, acct, effects, version = Charge(), self.node.state.account(run), {}, None
    with span("node.execute"):
      try:
        if cmd.op == Op.LOAD_WEIGHTS:
          version = await self.bus.checkpoint_version(json.loads(cmd.body)["uri"])
        elif cmd.op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER):
          version = await self.bus.begin_checkpoint(run, token)
      except (KeyError, ValueError) as exc:
        reply = Reply.error(exc)
      else:
        job = self.node.state.jobs.get(run)
        saved = unpack_snap(snap)[1] if snap and run not in self.loaded and cmd.op != Op.LOAD_WEIGHTS else None
        reply, effects = await Wave(self.spec.gpus).run(
          self.model.call("execute", cmd, job.record if job else None, saved, version), charge
        )
        self.loaded.add(run) if effects.get("loaded") else self.loaded.discard(run)
    with span("node.publish"):
      mutates = bool(effects.get("mutated"))
      billed, old = await self.bus.commit_reply(cmd, reply, token, acct, mutates, charge, effects.get("saved"))
      if not billed:
        self.owned.pop(run, None)
        self.loaded.discard(run)
        raise RuntimeError(f"lost ownership of {run}")
      if cmd.op == Op.DELETE_MODEL and reply.ok:
        self.owned.pop(run, None)
      elif mutates:
        self.owned[run] = (token, cmd.entry_id, None)
        if old:
          await self.bus.unregister_checkpoints(unpack_snap(old)[0])
      self.open_grads.discard(run)
      if cmd.op == Op.FORWARD_BACKWARD and reply.ok:
        self.open_grads.add(run)
    self.bill(cmd, acct, charge, billed, floor)
    self.idle_since = time.monotonic()
    return cmd if reply.ok else None

  async def loop(self) -> None:
    if self.spec.gpus == 1:
      self.spawn_model(tuple(range(self.node.gpus)))
    while self.wanted():
      changed = self.node.changed
      await self.sync_runs()
      await self.collect()
      ready = list(self.heads.values())
      # A trainer keeps its turn for CYCLE_GRACE_MS after a command: its client's next one is a round trip away.
      grace = self.idle_since + CYCLE_GRACE_MS / 1000 - time.monotonic()
      linger = self.life in self.node.on and grace > 0
      # It also outstays its quantum while a training step is open, because parking then moves the gradients too.
      closing = any(c.op == Op.OPTIM_STEP and c.run in self.open_grads for c in ready)
      self.step_ready = closing or (linger and not ready and bool(self.open_grads))
      await self.sync_gpus(bool(ready) or linger)
      if not self.wanted():
        break
      if ready and self.life in self.node.on:
        if (cmd := await self.execute_one(ready)) and cmd.op == Op.DELETE_MODEL and not self.spec.lora:
          break
      else:
        self.start_reader(self.bus.peek_trainer, [r for r in sorted(self.owned) if r not in self.heads])
        with contextlib.suppress(TimeoutError):
          await asyncio.wait_for(changed.wait(), grace if linger else None)
    self.active = False
    await self.stop_reader()
    for r in list(self.owned):
      await self.detach(r, drop=False)


class SamplerLife(Lifetime):
  """Stateless sampling pool replica: stages checkpoints per version off-GPU and runs bounded concurrent waves."""

  def __init__(self, node: NodeDaemon, proc: str):
    super().__init__(node, proc)
    self.warm, self.inputs, self.staged = False, {}, {}

  async def park_model(self) -> None:
    await self.model.park()

  async def resume_model(self, gpus: tuple[int, ...]) -> None:
    if self.model.running is None:
      self.bound_gpus = gpus
      self.model.env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
      await self.model.start()
    else:
      await self.model.resume()

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

  def ready_runs(self) -> list[str]:
    return [c.run for c in self.ready()[0]]

  async def collect(self) -> None:
    live, jobs = sorted(self.streams()), self.node.state.jobs
    live_set = set(live)
    self.heads = {r: c for r, c in self.heads.items() if r in live_set}
    if (cmds := await self.poll_reader(self.bus.peek_sampler, live, not self.heads)) is None:
      return
    self.heads = {c.run: c for c in cmds if c.run in live_set}
    uris = sorted({p for r in self.heads if (j := jobs.get(r)) and (p := j.record.get("model_path"))})
    got = await self.bus.checkpoint_versions(uris)
    self.inputs = {}
    for r in self.heads:
      if not (p := jobs[r].record.get("model_path")):
        self.inputs[r] = None
      elif isinstance(v := got[p], str):
        if v not in self.staged:
          (task := asyncio.create_task(self.stage(v))).add_done_callback(lambda _: self.node.notify())
          self.staged[v] = task
        self.inputs[r] = self.staged[v]
      else:
        self.inputs[r] = v
    active = set(self.inputs.values())
    for v in list(self.staged):
      if len(self.staged) > 64 and self.staged[v].done() and self.staged[v] not in active:
        del self.staged[v]
    self.service = {r: self.service[r] for r in self.heads if r in self.service}

  async def execute_wave(self, candidates: list[Cmd], failed: bool = False) -> None:
    floor = min((self.service[c.run] for c in candidates if c.run in self.service), default=0.0)
    runs = scheduler.select_runs(self.node.state, candidates, self.service, SAMPLE_RUNS if self.spec.lora else 1)
    inputs = {r: (v.result() if isinstance(v := self.inputs[r], asyncio.Task) else v) for r in runs}
    if failed:
      dead = {self.inputs.get(r) for r in runs}
      self.staged = {k: t for k, t in self.staged.items() if t not in dead}  # allow retry if upload lands later
    await self.stop_reader()
    with span("node.claim"):
      cmds = await self.bus.take_commands(runs, self.life, 1 if failed else SAMPLE_COMMANDS)
    for r in runs:
      self.heads.pop(r, None)
    if not cmds:
      return
    for ckpt in ({inputs[c.run] for c in cmds} - {None}) if not failed else ():
      try:
        await self.model.load(ckpt)
      except Exception as e:
        inputs = {r: e if v == ckpt else v for r, v in inputs.items()}
    wave = Wave(self.spec.gpus)

    async def answer(cmd: Cmd) -> None:
      charge, acct, inp = Charge(), self.node.state.account(cmd.run), inputs[cmd.run]
      with span("node.execute"):
        reply = (
          Reply.error(inp) if isinstance(inp, Exception) else await wave.run(self.model.execute(cmd, inp, True), charge)
        )
      with span("node.publish"):
        billed = await self.bus.commit_sample(cmd, reply, acct, charge=charge)
      self.bill(cmd, acct, charge, billed, floor)

    async with asyncio.TaskGroup() as group:
      for cmd in cmds:
        group.create_task(answer(cmd))

  async def loop(self) -> None:
    e = self.env(tuple(range(self.node.gpus)))
    self.model, self.warm = self.node.open_sampler(self.proc, self.spec, e), e.get("TINYRL_WARM_SAMPLER", "0") != "0"
    while self.wanted():
      changed = self.node.changed
      await self.collect()
      ready, failed = self.ready()
      if failed:
        await self.execute_wave(failed, failed=True)
        ready, _ = self.ready()
        if not self.wanted():
          break
      await self.sync_gpus(bool(ready) or (self.warm and self.model.running is None))
      if not self.wanted():
        break
      if ready and self.life in self.node.on:
        await self.execute_wave(ready)
      else:
        self.start_reader(self.bus.peek_sampler, sorted(self.streams()), IDLE_MS / 1000 if self.heads else 0)
        await changed.wait()
    self.active = False
    await self.stop_reader()


if __name__ == "__main__":

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await NodeDaemon(Bus()).serve()

  asyncio.run(main())
