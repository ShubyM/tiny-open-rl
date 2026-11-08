"""One node owns its worker leases, ready commands and GPU reservations. Children only execute model operations."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import signal
import socket
import subprocess
import sys
import time

from tinyrl.bus import LEASE_MS, Bus, Ev, Op, Reply, Role
from tinyrl.control import scheduler
from tinyrl.control.process import Process
from tinyrl.helpers import DEBUG, getenv
from tinyrl.storage.registry import Registry
from tinyrl.storage.store import Store
from tinyrl.trace import span
from tinyrl.workers.sampler import Sampler

GPUS, DRAIN_S, TICK = getenv("GPUS", 1), getenv("DRAIN_S", 20.0), getenv("TICK", 0.25)
IDLE_MS, STRANDED_MS = getenv("IDLE_MS", 150), 5000
RUN_COMMANDS, RUN_QUANTUM = getenv("RUN_COMMANDS", 4), getenv("RUN_QUANTUM", 1.0)
SAMPLE_RUNS, SAMPLE_COMMANDS = getenv("SAMPLE_RUNS", 3), getenv("SAMPLE_COMMANDS", 128)
SAVE_WAIT_S, SAMPLE_WAIT_S = getenv("SAVE_WAIT_S", 0.015), getenv("SAMPLE_WAIT_S", 0.02)
SWEEP_S, SWEEP_GRACE_S = getenv("SWEEP_S", 60.0), 600


def gpu_memory_mib() -> int:
  if override := getenv("GPU_MEMORY_MIB", 0):
    return int(override)
  try:
    cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"]
    return int(subprocess.check_output(cmd, timeout=2, stderr=subprocess.DEVNULL))
  except (OSError, subprocess.SubprocessError, ValueError):
    return 0


def movable(w, gpus: int) -> bool:
  return w.kind.role == Role.TRAINER and w.kind.lora and len(w.gpus) == 1 and gpus > 1


class NodeDaemon:
  def __init__(
    self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store: Store | None = None
  ):
    self.bus, self.registry = bus, Registry(bus)
    self.store = store if store is not None else Store()
    self.store.start()
    self.gpus, self.id = gpus, node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.running: dict[str, scheduler.Worker] = {}
    self.tasks: dict[str, asyncio.Task] = {}
    self.state, self.after, self.serial = scheduler.State(), "0-0", 0
    self.changed, self.draining = asyncio.Event(), False
    self.heartbeat, self.lag = time.monotonic(), 0.0
    self.swept, self.sweeping = 0.0, None
    self.turns: asyncio.Queue | None = None
    self.logging: asyncio.Task | None = None

  def log(self, kind: Ev, worker: str, **body) -> None:
    """Append a turn to the event log, in order and without waiting, for traces and the dashboard. No scheduler folds it."""
    if self.logging is None:
      self.turns, self.logging = asyncio.Queue(), asyncio.create_task(self.append_turns())
    self.turns.put_nowait((kind, {"worker": worker, **body}))

  async def append_turns(self) -> None:
    while True:
      kind, body = await self.turns.get()
      try:
        await self.bus.append_event(kind, **body)
      except Exception as e:  # a trace that misses a turn is better than a worker that stops
        print(f"could not log {kind}: {e!r}", file=sys.stderr, flush=True)

  def notify(self) -> None:
    changed, self.changed = self.changed, asyncio.Event()
    changed.set()

  def reserve(self, worker: str) -> bool:
    """Try admission without blocking observation of newly ready work or preparation errors."""
    r = self.running[worker]
    if r.since is not None:
      return True
    if not r.ticket:
      with span("node.request_gpu"):
        self.serial += 1
        r.ticket = self.serial
      self.log(Ev.WANTS_GPU, worker)
      self.notify()
    if scheduler.can_run(self.state, self.id, self.running, worker):
      with span("node.grant_gpu"):
        r.since, r.ticket = time.time(), 0
      self.log(Ev.GRANTED, worker)
      self.notify()
      return True
    return False

  async def landed(self, uri: str) -> str | None:
    """The registered version of a checkpoint, once this node can read it. It waits while the node that saved it is live, since that node is still uploading it, and fails once that node is gone."""
    gone = set()
    while True:
      info = await self.registry.get_checkpoint(uri) or {}
      if not (version := info.get("version")) or await asyncio.to_thread(self.store.landed, uri, version):
        return version
      if (node := info.get("node")) in gone:
        raise ValueError(f"version {version} of {uri} was lost with node {node} before it uploaded")
      if node not in await self.bus.live_nodes():
        gone.add(node)  # one more look, in case it landed just before the node left
      await asyncio.sleep(0.5)

  async def announce_node(self) -> None:
    await self.bus.node_heartbeat(self.id, self.gpus)
    self.heartbeat = time.monotonic()
    await self.bus.append_event(Ev.NODE_UP, node=self.id, gpus=self.gpus, memory=self.gpu_mib * 2**20)

  def open_trainer(self, worker: str, env: dict[str, str]) -> Process:
    width, cmd = len(self.running[worker].gpus), [sys.executable]
    if width > 1:
      cmd += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={width}"]
    return Process([*cmd, "-m", "tinyrl.workers.trainer"], env)

  def open_sampler(self, worker: str, env: dict[str, str]) -> Sampler:
    kind = self.running[worker].kind
    return Sampler(kind.base, kind.lora, env)

  def start(self, worker: str) -> asyncio.Task:
    task = self.tasks[worker] = asyncio.create_task(self.work(worker))
    return task

  async def work(self, worker: str) -> None:
    r, task = dataclasses.replace(self.state.workers[worker]), asyncio.current_task()
    owner = f"{self.id}-{os.urandom(8).hex()}"
    sampling = r.kind.role == Role.SAMPLER
    heads, prepared, service = [], {}, {}
    reader = guard = model = closing = None
    renewed, swept, virtual = time.monotonic(), 0.0, 0.0
    last_run, turn_started, turn_commands = None, 0.0, 0

    async def owned():
      nonlocal renewed
      before = time.monotonic()
      if not (
        await self.bus.renew_lease(worker, owner)
        or (not closing and await self.bus.take_lease(worker, owner, recover=True))
      ):
        if not closing and asyncio.current_task() is not task and not task.cancelling():
          task.cancel()
        raise RuntimeError(f"lost worker {worker}")
      renewed = before

    async def keepalive():
      while True:
        await asyncio.sleep(LEASE_MS / 4000)
        if not closing and not self.draining and time.monotonic() - self.heartbeat >= LEASE_MS / 1000:
          task.cancel()
          return
        try:
          await owned()
          if not closing and (p := model.proc) and p.poll() is not None:
            task.cancel()
            return
        except Exception:
          if not closing and time.monotonic() - renewed >= LEASE_MS / 1000:
            task.cancel()
            return

    async def prepare(cmd):
      with span("node.prepare"):
        try:
          run, checkpoint = await self.bus.get_run(cmd.run), None
          if sampling:
            if run is None:
              raise ValueError(f"no sampling session {cmd.run}")
            if uri := run.get("model_path"):
              checkpoint = await asyncio.to_thread(self.store.local_folder, uri, await self.landed(uri))
          return run, checkpoint
        except Exception as e:
          return e

    async def read(delay=0):
      await asyncio.sleep(delay)
      commands = await self.bus.peek_commands(scheduler.members(self.state, worker), IDLE_MS)
      for cmd in commands:
        if cmd.request_id not in prepared:
          preparing = prepared[cmd.request_id] = asyncio.create_task(prepare(cmd))
          preparing.add_done_callback(lambda _: self.notify())
      return commands

    def collect():
      nonlocal heads, reader, service
      if reader and reader.done():
        heads, reader = reader.result(), None
        current = {c.request_id for c in heads}
        for request in list(prepared):
          if request not in current:
            prepared.pop(request).cancel()
        # Idle runs do not bank service credit; a new contender joins the active floor.
        service = {c.run: service.get(c.run, virtual) for c in heads}

    def ready():
      return [c for c in heads if c.request_id in prepared and prepared[c.request_id].done()]

    took = {}  # request -> ms it took, for the batch_done trace event

    async def answer(cmd, inputs):
      before = time.time()
      with span("node.execute"):
        if isinstance(inputs, Exception):
          reply = Reply.error(inputs)
        else:
          run, checkpoint = inputs
          if sampling:
            await owned()
            reply = await model.execute(cmd, run, checkpoint)
          else:
            parked, version = await self.bus.parked_checkpoint(cmd.run), None
            if cmd.op == Op.LOAD_WEIGHTS:
              version = await self.landed(json.loads(cmd.body)["uri"])
            saved = (parked, await self.landed(parked)) if parked else None
            await owned()  # metadata awaits may have outlived our ownership
            reply, effects = await model.call("execute", cmd, run, saved, version)
            await owned()
            if checkpoint := effects.get("saved"):
              await self.registry.register_checkpoint(**checkpoint, node=self.id)
            if restored := effects.get("restored"):
              if not await self.bus.set_parked_checkpoint(cmd.run, None, worker, owner):
                raise RuntimeError(f"lost worker {worker}")
              await self.registry.unregister_checkpoints(restored)
      with span("node.publish"):
        if not await self.bus.write_reply(cmd, reply, worker, owner, started=before):
          raise RuntimeError(f"lost worker {worker}")
      took[cmd.request_id] = (time.time() - before) * 1000
      if DEBUG:
        print(f"{worker} {cmd.request_id} {cmd.op:24s} {took[cmd.request_id]:9.2f} ms")
      return reply

    def wanted():
      w = self.state.workers.get(worker)
      return (
        not self.draining
        and w
        and w.node == self.id
        and scheduler.members(self.state, worker)
        and (w.gpus == r.gpus or movable(r, self.gpus))
      )

    async def activate():
      if r.since is None:
        with span("node.wait_gpu"):
          while True:  # only final checkpointing waits here; ordinary admission never blocks the ready loop
            changed = self.changed
            if self.reserve(worker):
              break
            await changed.wait()
      try:
        await owned()
        with span("node.resume"):
          if not sampling:
            await model.call("resume", r.gpus[0] if movable(r, self.gpus) else None)
          else:
            await model.resume()
            if model.running is None:
              await model.start()
              r.pid = model.proc.pid if model.proc else None
              r.since = time.time()  # cold boot is outside the first GPU quantum
              await asyncio.sleep(SAMPLE_WAIT_S)
      except Exception as e:
        # A boot failure still answers waiting clients. Keep its reservation until close acknowledges.
        await owned()
        for cmd in await self.bus.take_commands(scheduler.members(self.state, worker), owner, 1):
          if not await self.bus.write_reply(cmd, Reply.error(e), worker, owner):
            raise RuntimeError(f"lost worker {worker}") from e
        raise

    async def turn(gpus, runnable):
      if (p := model.proc) and p.poll() is not None:
        raise RuntimeError(f"worker {worker} exited")
      waiting = scheduler.has_waiter(self.running, r.gpus)
      if r.since is not None and (
        gpus != r.gpus or (waiting and (not runnable or time.time() >= r.since + self.state.quantum))
      ):
        with span("node.park"):
          await model.park() if sampling else await model.call("park")
        r.since = None
        self.log(Ev.RELEASED, worker)
        self.notify()
      if r.gpus != gpus:
        r.gpus = gpus
        self.notify()
      if not runnable and r.ticket:
        r.ticket = 0
        self.notify()
      if runnable and r.since is None and self.reserve(worker):
        await activate()

    async def execute(candidates):
      nonlocal reader, last_run, turn_started, turn_commands, heads, virtual
      now = time.monotonic()
      for cmd in candidates:
        service[cmd.run] = max(service.get(cmd.run, virtual), virtual)
      prefer = last_run if turn_commands < RUN_COMMANDS and now < turn_started + RUN_QUANTUM else None
      runs = scheduler.select_runs(
        candidates, service, SAMPLE_RUNS if sampling and r.kind.lora else 1, prefer if not sampling else None
      )
      inputs = {c.run: prepared[c.request_id].result() for c in candidates if c.run in runs}
      failed = all(isinstance(value, Exception) for value in inputs.values())
      # Discard a peek that could finish with pre-claim cursors. The next read starts after the claim.
      if reader:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
      await owned()
      with span("node.claim"):
        cmds = await self.bus.take_commands(runs, owner, SAMPLE_COMMANDS if sampling and not failed else 1)
      reader = asyncio.create_task(read())
      reader.add_done_callback(lambda _: self.notify())
      heads = [c for c in heads if c.run not in runs]
      await owned()
      if not cmds:
        return
      before = time.monotonic()
      if sampling:
        async with asyncio.TaskGroup() as group:
          for cmd in cmds:
            group.create_task(answer(cmd, inputs[cmd.run]))
      else:
        cmd = cmds[0]
        if cmd.run != last_run or prefer is None:
          last_run, turn_started, turn_commands = cmd.run, before, 0
        turn_commands += 1
        reply = await answer(cmd, inputs[cmd.run])
      seconds = time.monotonic() - before
      r.busy += seconds
      self.log(
        Ev.BATCH_DONE,
        worker,
        seconds=seconds,
        queued=len(heads),
        ops=[[c.op, took.pop(c.request_id, 0.0)] for c in cmds],  # a wave's overlap, so traces draw requests in flight
      )
      for cmd in cmds:
        service[cmd.run] = service.get(cmd.run, 0.0) + seconds * len(r.gpus) / len(cmds)
      active = {c.run for c in [*ready(), *cmds]}
      virtual = max(virtual, min(service[run] for run in active))
      if not sampling and cmds[-1].op == Op.OPTIM_STEP and r.kind.lora:
        await asyncio.sleep(SAVE_WAIT_S)  # bounded grace for the client's following sampler save
      return cmds[0] if not sampling and reply.ok else None

    async def cleanup():
      for future in [reader, *prepared.values()]:
        if future:
          future.cancel()
      await asyncio.gather(*(f for f in [reader, *prepared.values()] if f), return_exceptions=True)
      if model:
        while True:
          try:
            with span("node.close"):
              await model.close()
            break
          except (Exception, asyncio.CancelledError) as e:
            print(f"{worker} could not close, retrying: {e!r}", file=sys.stderr, flush=True)
            await asyncio.sleep(TICK)
      if r.since is not None:
        self.log(Ev.RELEASED, worker)
      r.since = None  # release only after the child has acknowledged park or close
      self.log(Ev.EXITED, worker)
      self.notify()
      if guard:
        guard.cancel()
        await asyncio.gather(guard, return_exceptions=True)
      if await self.bus.lease_owner(worker) == owner:
        if died:
          await self.bus.take_commands(scheduler.members(self.state, worker), owner, 1)
        await self.bus.fail_stranded_commands(scheduler.members(self.state, worker), worker, owner, 0)
      await self.bus.release_lease(worker, owner)

    try:
      if not await self.bus.take_lease(worker, owner):
        return
      await self.bus.fail_stranded_commands(scheduler.members(self.state, worker), worker, owner, 0)
      if not await self.bus.renew_lease(worker, owner) or self.draining:
        return
      self.running[worker] = r
      env = {**scheduler.settings(r.kind, self.gpus, self.gpu_mib * 2**20), **os.environ}
      env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(self.gpus) if movable(r, self.gpus) else r.gpus))
      env.update(self.store.env())
      if not sampling:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
      model = self.open_sampler(worker, env) if sampling else self.open_trainer(worker, env)
      r.pid = model.proc.pid if model.proc else None
      self.log(Ev.STARTED, worker, node=self.id)
      guard = asyncio.create_task(keepalive())
      reader = asyncio.create_task(read())
      reader.add_done_callback(lambda _: self.notify())

      # Each step observes ready heads, yields when necessary, and executes one bounded batch.
      while wanted():
        changed = self.changed
        collect()
        runnable = ready()
        failed = [c for c in runnable if isinstance(prepared[c.request_id].result(), Exception)]
        if failed:
          await execute(failed)
          runnable = [c for c in ready() if not isinstance(prepared[c.request_id].result(), Exception)]
          if not wanted():
            break
        warm = sampling and env.get("TINYRL_WARM_SAMPLER", "0") != "0" and model.running is None
        await turn(self.state.workers[worker].gpus, bool(runnable) or warm)
        if not wanted():
          break
        if runnable and r.since is not None:
          if (cmd := await execute(runnable)) and cmd.op == Op.DELETE_MODEL and not r.kind.lora:
            break
        else:
          if reader is None:
            reader = asyncio.create_task(read(IDLE_MS / 1000 if heads else 0))
            reader.add_done_callback(lambda _: self.notify())
          await changed.wait()
        if time.monotonic() - swept >= STRANDED_MS / 1000:
          swept = time.monotonic()
          await owned()
          await self.bus.fail_stranded_commands(scheduler.members(self.state, worker), worker, owner, 0)

      # No drain: preparation never claimed work. Only the completed batch belonged to this process.
      if not sampling:
        if len(r.gpus) > 1 and r.since is None:
          await activate()
        await owned()
        for saved in await model.call("park_all"):
          await owned()
          run = saved.pop("run")
          await self.registry.register_checkpoint(**saved)
          if not await self.bus.set_parked_checkpoint(run, saved["uri"], worker, owner):
            raise RuntimeError(f"lost worker {worker}")
    finally:
      r.ticket = 0
      died = model is not None and (p := model.proc) is not None and p.poll() is not None
      closing = asyncio.create_task(cleanup())
      self.notify()
      while True:
        try:
          await asyncio.shield(closing)
          break
        except asyncio.CancelledError:
          if closing.cancelled():
            closing = asyncio.create_task(cleanup())

  def dispatch(self) -> None:
    for worker, task in list(self.tasks.items()):
      if task.done():
        try:
          task.result()
        except BaseException as e:
          if not isinstance(e, asyncio.CancelledError):
            print(f"{worker} stopped: {e!r}", file=sys.stderr, flush=True)
        del self.tasks[worker]
        self.running.pop(worker, None)
      elif self.id not in self.state.nodes and not task.cancelling():
        task.cancel()
    before = {n: w.gpus for n, w in self.state.workers.items()}
    scheduler.place_gpus(self.state, self.id, self.running)
    if before != {n: w.gpus for n, w in self.state.workers.items()}:
      self.notify()

  async def tick(self) -> None:
    after = self.after
    workers = {
      n: dataclasses.asdict(r) | {"asked": r.ticket, "on": [(self.id, g) for g in r.gpus]}
      for n, r in self.running.items()
    }
    await self.bus.node_heartbeat(self.id, self.gpus, workers, self.lag)
    self.heartbeat = time.monotonic()
    if await self.bus.event_log_size() < self.state.n:
      for task in self.tasks.values():
        if not task.cancelling():
          task.cancel()
      self.state, self.after = scheduler.State(), "0-0"
    while events := await self.bus.read_events(self.after):
      for e in events:
        scheduler.fold(self.state, e)
      self.after = events[-1].id
    self.dispatch()
    mine = {n for n, w in self.state.workers.items() if w.node == self.id and scheduler.members(self.state, n)}
    if not self.draining:
      for worker in sorted(mine - self.tasks.keys()):
        self.start(worker)
    if self.id not in self.state.nodes and not self.tasks:
      await self.announce_node()
    if time.time() - self.swept > SWEEP_S and (self.sweeping is None or self.sweeping.done()):
      self.swept = time.time()
      self.sweeping = asyncio.create_task(self.registry.sweep(self.store, SWEEP_GRACE_S))
    if self.after != after:
      self.notify()

  async def close(self, graceful: bool = True) -> None:
    self.draining = True
    self.notify()
    if self.tasks:
      _, pending = await asyncio.wait(self.tasks.values(), timeout=DRAIN_S if graceful else 0)
      for task in pending:
        if not task.cancelling():
          task.cancel()
      await asyncio.gather(*self.tasks.values(), return_exceptions=True)
      self.dispatch()
    if self.logging:
      self.logging.cancel()
    closing = asyncio.create_task(asyncio.to_thread(self.store.close))
    while not closing.done():  # live until its saves land, so other nodes wait for them instead of failing
      with contextlib.suppress(Exception):  # the uploads matter more than the heartbeat
        await self.bus.node_heartbeat(self.id, self.gpus)
      await asyncio.wait({closing}, timeout=TICK)
    await closing

  async def serve(self) -> None:
    await self.announce_node()
    try:
      while True:
        await self.tick()
        before = time.monotonic()
        await asyncio.sleep(TICK)
        # how late the loop woke: every worker on the node shares it, so a stall here delays all their commands
        self.lag = time.monotonic() - before - TICK
    finally:
      await self.close()


if __name__ == "__main__":

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await NodeDaemon(Bus()).serve()

  asyncio.run(main())
