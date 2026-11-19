"""Per-process supervisors: `TrainerRunner` (fenced single-writer) and `SamplerRunner` (stateless wave reader)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from typing import TYPE_CHECKING

from tinyrl.bus import LEASE_MS
from tinyrl.control import scheduler
from tinyrl.control.accounting import Charge, Wave
from tinyrl.control.fairness import Demand, Worker, blocks, select_runs
from tinyrl.helpers import DEBUG, getenv
from tinyrl.trace import span
from tinyrl.types import Cmd, Op, Reply, unpack_snap

if TYPE_CHECKING:
  from tinyrl.control.node_daemon import NodeDaemon

IDLE_MS, RETRY_CLOSE_S = getenv("IDLE_MS", 150), getenv("TICK", 0.25)
QUANTUM_MS, MAX_QUANTUM_MS = getenv("QUANTUM_MS", 2000), getenv("MAX_QUANTUM_MS", 15000)
CYCLE_GRACE_MS, SWITCH_BUDGET = getenv("CYCLE_GRACE_MS", 100), float(getenv("SWITCH_BUDGET", 0.1))
SAMPLE_RUNS, SAMPLE_COMMANDS = getenv("SAMPLE_RUNS", 3), getenv("SAMPLE_COMMANDS", 128)


async def stop_tasks(*tasks: asyncio.Task | None) -> None:
  if live := [t for t in tasks if t]:
    for t in live:
      t.cancel()
    await asyncio.gather(*live, return_exceptions=True)


class Runner:
  """Shared worker lifetime: lease keepalive, local GPU turns, background head peeking, and shielded cleanup."""

  gang = False

  def __init__(self, node: NodeDaemon, proc: str):
    self.node, self.bus, self.store, self.proc = node, node.bus, node.store, proc
    self.spec, self.eager, self.pinned = node.state.procs[proc].spec, True, True
    self.life = f"{proc}@{node.id}-{os.urandom(6).hex()}"
    self.task = self.model = self.guard = self.closing = self.reader = None
    self.bound_gpus, self.demand, self.renewed = (), Demand.IDLE, time.monotonic()
    self.hold_until, self.park_s, self.gpu_ms = 0.0, 0.0, 0.0
    self.heads, self.staging, self.records = {}, {}, {}
    self.run_service, self.service_floor = {}, 0.0  # Newly ready runs join at the active service floor.

  def streams(self) -> list[str]:
    return self.node.state.runs(self.proc)

  async def load_record(self, run: str) -> dict | None:
    if (rec := self.records.get(run)) is None and (rec := await self.bus.get_run(run)) is not None:
      self.records[run] = rec
    return rec

  def wanted(self) -> bool:
    p = self.node.state.procs.get(self.proc)
    if self.node.draining or not p or p.node != self.node.id or not self.node.state.runs(self.proc):
      return False
    excl = {g for o in self.node.lives.values() if o.spec.exclusive for g in self.node.on.get(o.life, ())}
    if self.life in self.node.on or excl.isdisjoint(self.bound_gpus):
      return True
    # A blocked sampler exits immediately; a blocked multi-GPU trainer detaches once work and another block exist.
    return self.gang and not (self.ready()[0] and any(excl.isdisjoint(b) for b in blocks(self.node.gpus, p.spec.gpus)))

  def worker(self) -> Worker:
    s, sp, bnd = self.node.state, self.spec, self.bound_gpus
    runs = [c.run for c in self.ready()[0] if c.run in s.jobs] or [r for r in self.streams() if r in s.jobs]
    cur, lag = self.node.on.get(self.life, ()), max(map(s.ready_lag, runs), default=-self.gpu_ms)
    return Worker(sp.gpus, sp.exclusive, self.demand, cur, bnd, self.gang, self.hold_until, lag, self.gpu_ms)

  async def sync_runs(self) -> None:
    pass

  async def finish(self, clean: bool) -> None:
    pass

  async def run(self) -> None:
    self.task = asyncio.current_task()
    try:
      if not await self.open():
        return
      while self.wanted():
        changed = self.node.changed
        await self.sync_runs()
        self.collect_reader()
        ready, failed = self.ready()
        if failed:
          await self.dispatch(failed, count=1)
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
          if self.reader is None:
            self.read(IDLE_MS / 1000 if self.heads else 0)
          with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(changed.wait(), wait_s)
      self.demand = Demand.IDLE
      await self.stop_reader()
      await self.finish(clean=True)
    finally:
      self.demand, self.closing = Demand.IDLE, asyncio.create_task(self.cleanup())
      self.node.notify()
      while True:
        try:
          await asyncio.shield(self.closing)
          break
        except asyncio.CancelledError:
          if self.closing.cancelled():
            self.closing = asyncio.create_task(self.cleanup())

  # ***** lease and local GPU turns *****

  def spawn_model(self, gpus: tuple[int, ...]) -> None:
    env = {**scheduler.settings(self.node.gpus, self.node.gpu_mib * 2**20), **os.environ, **self.store.env()}
    self.model = self.open_model(env | {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))})

  async def open(self) -> bool:
    if self.node.draining or not await self.bus.open_lease(self.life):
      return False
    self.node.lives[self.proc] = self
    if self.eager:
      self.spawn_model(tuple(range(self.node.gpus)))
    self.guard = asyncio.create_task(self.keepalive())
    self.read()
    return True

  async def keep_lease(self) -> None:
    before = time.monotonic()
    await self.bus.renew_lease(self.life, recover=not self.closing)
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
        await self.model.park()
      self.park_s = time.monotonic() - t0
      self.node.release_gpus(self.life)
      await self.node.push_heartbeat()
    elif cur is None and want and self.node.claim_gpus(self.life, want, self.spec.gpus):
      await self.promote(want)
      await self.node.push_heartbeat()

  async def promote(self, gpus: tuple[int, ...]) -> None:
    try:
      await self.keep_lease()
      t0 = time.monotonic()
      with span("node.resume"):
        if self.model is None:
          self.spawn_model(gpus)
        if not self.bound_gpus and self.pinned:
          self.bound_gpus = gpus
        await self.model.resume(gpus)
      resumed = time.monotonic()
      sw = (self.park_s + resumed - t0) / max(0.01, SWITCH_BUDGET) if self.park_s else 0.0
      self.hold_until = resumed + min(MAX_QUANTUM_MS / 1000, max(QUANTUM_MS / 1000, sw))
      self.node.select_gpus(self.proc)
    except Exception as e:
      await self.fail_boot(Reply.error(e))
      raise

  # ***** stream peeking, fairness bookkeeping, and teardown *****

  def read(self, delay: float = 0) -> None:
    self.reader = asyncio.create_task(self.peek(delay))
    self.reader.add_done_callback(lambda _: self.node.notify())

  async def stop_reader(self) -> None:
    await stop_tasks(self.reader)
    self.reader = None

  async def peek(self, delay: float) -> list[Cmd]:
    await asyncio.sleep(delay)
    while not (cmds := await self.poll()) and not self.heads:
      pass
    self.set_heads(cmds)
    return list(self.heads.values())

  def collect_reader(self) -> None:
    if self.reader and self.reader.done():
      self.reader.result()
      self.reader = None
      self.run_service = {r: self.run_service.get(r, self.service_floor) for r in self.heads}

  def choose_runs(self, candidates: list[Cmd], width: int = 1) -> list[str]:
    for c in candidates:
      self.run_service[c.run] = max(self.run_service.get(c.run, self.service_floor), self.service_floor)
    return select_runs(candidates, self.node.state.account, self.node.state.ready_lag, self.run_service, width)

  def record_charge(self, cmd: Cmd, acct: str, charge: Charge, billed: bool) -> None:
    if billed and charge.gpu_ms:
      self.run_service[cmd.run] = self.run_service.get(cmd.run, self.service_floor) + charge.gpu_ms
      self.gpu_ms += charge.gpu_ms
      if acct:
        self.node.state.service[acct] = self.node.state.service.get(acct, 0.0) + charge.gpu_ms
    if DEBUG:
      print(f"{self.proc} {cmd.request_id} {cmd.op:24s} {charge.busy_ms:9d} ms")

  def advance_floor(self, executed: list[Cmd]) -> None:
    ready, failed = self.ready()
    seen = (self.run_service.get(c.run, self.service_floor) for c in (*ready, *failed, *executed))
    self.service_floor = max(self.service_floor, min(seen))

  async def cleanup(self) -> None:
    """Close the child first, then release GPUs (`on`), abort stranded work, and drop the lease."""
    await stop_tasks(self.reader, *self.staging.values())
    while self.model:
      try:
        with span("node.close"):
          await self.model.close()
        break
      except (Exception, asyncio.CancelledError) as e:
        print(f"{self.proc} could not close, retrying: {e!r}", file=sys.stderr, flush=True)
        await asyncio.sleep(RETRY_CLOSE_S)
    self.node.release_gpus(self.life)
    await stop_tasks(self.guard)
    await self.finish(clean=False)
    await self.bus.drop_lease(self.life)


class TrainerRunner(Runner):
  """Stateful single-writer supervisor: fenced epoch ownership, host-memory parking, sequential step execution."""

  gang = True

  def __init__(self, node: NodeDaemon, proc: str):
    super().__init__(node, proc)
    self.eager, self.pinned = self.spec.gpus == 1, self.spec.gpus > 1
    self.owned: dict[str, tuple[str, str | None, str | None]] = {}  # run -> (token, pos, snap)
    self.loaded, self.open_grads, self.in_flight, self.idle_since = set(), set(), {}, 0.0

  def streams(self) -> list[str]:
    return sorted(self.owned)

  def open_model(self, env: dict[str, str]):
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return self.node.open_trainer(self.proc, self.spec.gpus, env)

  def forget(self, run: str) -> None:
    self.owned.pop(run, None)
    self.records.pop(run, None)
    self.heads.pop(run, None)
    self.open_grads.discard(run)

  async def sync_runs(self) -> None:
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
    """Save `run` if dirty (from GPU or host RAM), release `owner` fenced by `(token, pos)`, and drop if asked."""
    token, pos, snap = self.owned[run]
    checkpoint = None
    if pos is not None and snap is None and run in self.loaded:
      await self.keep_lease()
      version = await self.bus.begin_checkpoint(run, token)
      if checkpoint := await self.model.call("save_run", run, version):
        snap = f"{checkpoint['uri']}\n{checkpoint['version']}"
    await self.bus.detach_run(run, token, pos, snap, checkpoint)
    self.forget(run)
    was_loaded = run in self.loaded
    self.loaded.discard(run)
    if was_loaded and drop:
      await self.model.call("drop_run", run)

  async def poll(self) -> list[Cmd]:
    return await self.bus.peek_trainer(self.streams(), IDLE_MS, self.in_flight)

  def set_heads(self, cmds: list[Cmd]) -> None:
    self.heads = {c.run: c for c in cmds if c.run in self.owned}

  def ready(self) -> tuple[list[Cmd], list[Cmd]]:
    return [c for r, c in self.heads.items() if r in self.owned], []

  def turn(self, ready: list[Cmd]) -> tuple[Demand, float | None]:
    # Hold the GPU briefly after each command so a synchronous client's next step arrives before parking.
    grace = self.idle_since + CYCLE_GRACE_MS / 1000 - time.monotonic()
    linger = self.life in self.node.on and grace > 0
    step_open = any(c.op == Op.OPTIM_STEP and c.run in self.open_grads for c in ready) or (linger and not ready)
    demand = Demand.FINISH_STEP if self.open_grads and step_open else (Demand.READY if ready or linger else Demand.IDLE)
    return demand, grace if linger else None

  async def dispatch(self, ready: list[Cmd]) -> bool:
    [run] = self.choose_runs(ready, width=1)
    await self.stop_reader()
    await self.keep_lease()
    with span("node.claim"):
      cmd = await self.bus.claim_trainer(run, self.owned[run][0])
    self.heads.pop(run, None)
    if cmd is None:
      self.read()
      return False
    self.in_flight = {run: cmd.entry_id}
    self.read()
    try:
      reply = await self.execute_one(cmd)
    finally:
      self.in_flight.clear()
    self.idle_since = time.monotonic()
    self.advance_floor([cmd])
    return reply.ok and cmd.op == Op.DELETE_MODEL and not self.spec.lora

  async def run_model(self, cmd: Cmd, token: str, snap: str | None, charge: Charge) -> tuple[Reply, dict]:
    version = None
    if cmd.op == Op.LOAD_WEIGHTS:
      try:
        version = await self.bus.checkpoint_version(json.loads(cmd.body)["uri"])
      except (KeyError, ValueError) as exc:
        return Reply.error(exc), {}
    elif cmd.op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER):
      version = await self.bus.begin_checkpoint(cmd.run, token)
    need_rec = cmd.run not in self.loaded and cmd.op not in (Op.CREATE_MODEL, Op.DELETE_MODEL)
    rec = await self.load_record(cmd.run) if need_rec else None
    await self.keep_lease()
    call = self.model.call("execute", cmd, rec, unpack_snap(snap)[1] if snap else None, version)
    reply, effects = await Wave(self.spec.gpus).run(call, charge)
    self.loaded.add(cmd.run) if effects.get("loaded") else self.loaded.discard(cmd.run)
    return reply, effects

  async def execute_one(self, cmd: Cmd) -> Reply:
    token, _, snap = self.owned[cmd.run]
    acct, charge = self.node.state.account(cmd.run), Charge()
    with span("node.execute"):
      reply, effects = await self.run_model(cmd, token, snap, charge)
    with span("node.publish"):
      mutates = bool(effects.get("mutated"))
      billed = await self.bus.commit_reply(cmd, reply, token, acct, mutates, charge, effects.get("saved"))
      if not billed:
        self.forget(cmd.run)
        self.loaded.discard(cmd.run)
        raise RuntimeError(f"lost ownership of {cmd.run}")
      if cmd.op == Op.DELETE_MODEL and reply.ok:
        self.forget(cmd.run)
      elif mutates:
        self.owned[cmd.run] = (token, cmd.entry_id, None)
      self.open_grads.discard(cmd.run)
      if cmd.op == Op.FORWARD_BACKWARD and reply.ok:
        self.open_grads.add(cmd.run)
    self.record_charge(cmd, acct, charge, billed)
    return reply

  async def fail_boot(self, err: Reply) -> None:
    for r in self.streams():
      if cmd := await self.bus.claim_trainer(r, token := self.owned[r][0]):
        await self.bus.commit_reply(cmd, err, token)

  async def finish(self, clean: bool) -> None:
    for r in list(self.owned):
      await (self.detach(r, drop=False) if clean else self.bus.fail_stranded_commands(r, self.owned[r][0]))


class SamplerRunner(Runner):
  """Stateless sampling supervisor: off-GPU checkpoint staging and single-account concurrent waves."""

  def open_model(self, env: dict[str, str]):
    return self.node.open_sampler(self.proc, self.spec, env)

  async def poll(self) -> list[Cmd]:
    return await self.bus.peek_samples(self.streams(), IDLE_MS)

  def set_heads(self, cmds: list[Cmd]) -> None:
    live = set(self.streams())
    self.heads = {c.run: c for c in cmds if c.run in live}
    for req in self.staging.keys() - {c.request_id for c in self.heads.values()}:
      self.staging.pop(req).cancel()
    for cmd in self.heads.values():
      if cmd.request_id not in self.staging:
        self.staging[cmd.request_id] = task = asyncio.create_task(self.stage_checkpoint(cmd.run))
        task.add_done_callback(lambda _: self.node.notify())

  async def stage_checkpoint(self, session: str):
    with span("node.prepare"):
      try:
        if (run := await self.load_record(session)) is None:
          raise ValueError(f"no sampling session {session}")
        uri = run.get("model_path")
        return await asyncio.to_thread(self.store.local_folder, await self.bus.checkpoint_version(uri)) if uri else None
      except Exception as e:
        return e

  def ready(self) -> tuple[list[Cmd], list[Cmd]]:
    live = set(self.streams())
    tasks = ((c, self.staging.get(c.request_id)) for r, c in self.heads.items() if r in live)
    done = [(c, t.result()) for c, t in tasks if t and t.done()]
    return [c for c, v in done if not isinstance(v, Exception)], [c for c, v in done if isinstance(v, Exception)]

  def turn(self, ready: list[Cmd]) -> tuple[Demand, float | None]:
    return (Demand.READY if ready or (self.model is not None and self.model.warm_pending) else Demand.IDLE), None

  async def dispatch(self, candidates: list[Cmd], *, count: int | None = None) -> bool:
    sessions = self.choose_runs(candidates, width=SAMPLE_RUNS if self.spec.lora else 1)
    staged = {c.run: self.staging[c.request_id].result() for c in candidates if c.run in sessions}
    await self.stop_reader()
    await self.keep_lease()
    with span("node.claim"):
      cmds = await self.bus.claim_samples(sessions, self.life, SAMPLE_COMMANDS if count is None else count)
    for s in sessions:
      if (head := self.heads.pop(s, None)) and (task := self.staging.pop(head.request_id, None)):
        task.cancel()
    self.read(0 if cmds else 0.05)
    if cmds:
      wave = Wave(self.spec.gpus)
      async with asyncio.TaskGroup() as group:
        for cmd in cmds:
          group.create_task(self.answer(cmd, staged[cmd.run], wave))
      self.advance_floor(cmds)
    return False

  async def answer(self, cmd: Cmd, staged, wave: Wave) -> None:
    charge, acct = Charge(), self.node.state.account(cmd.run)
    with span("node.execute"):
      bad = isinstance(staged, Exception)
      reply = Reply.error(staged) if bad else await wave.run(self.model.execute(cmd, staged), charge)
    with span("node.publish"):
      billed = await self.bus.commit_sample(cmd, reply, acct, charge=charge)
    self.record_charge(cmd, acct, charge, billed)

  async def fail_boot(self, err: Reply) -> None:
    for cmd in await self.bus.claim_samples(self.streams(), self.life, 1):
      await self.bus.commit_sample(cmd, err)
