"""One node runs its workers' processes on the GPUs the cluster grants them, and owns the runs they serve. Children only execute model operations."""

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
RUN_COMMANDS = getenv("RUN_COMMANDS", 4)  # a trainer stays with a run this long while its next command is queued
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


def movable(w, gpus: int) -> bool:
  return w.spec.role == Role.TRAINER and w.spec.lora and len(w.gpus) == 1 and gpus > 1


class NodeDaemon:
  def __init__(
    self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store: Store | None = None
  ):
    self.bus, self.registry = bus, Registry(bus)
    self.store = store if store is not None else Store()
    self.store.start()
    self.gpus, self.id = gpus, node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.running: dict[str, scheduler.Worker] = {}  # this node's processes, and the GPUs each is on
    self.tasks: dict[str, asyncio.Task] = {}
    self.state, self.ready = scheduler.Assignment(), {}
    self.returned: dict[str, float] = {}  # worker -> when it gave its GPUs back, until a table read began after that
    self.changed, self.draining = asyncio.Event(), False
    self.heartbeat, self.lag = time.monotonic(), 0.0
    self.swept, self.sweeping = 0.0, None

  def notify(self) -> None:
    changed, self.changed = self.changed, asyncio.Event()
    changed.set()

  def clear(self, worker: str) -> bool:
    """Whether the processes on a worker's GPUs can share them with it. This daemon started every one, so it knows whatever the table says."""
    r = self.running[worker]
    on = {
      g: [n for n, x in self.running.items() if n != worker and x.served is not None and g in x.gpus] for g in r.gpus
    }
    return all(scheduler.can_share(self.state, self.id, [worker, *others]) for others in on.values())

  def someone_waits(self, worker: str) -> bool:
    """Whether another of this node's processes has work ready and is waiting for a GPU this one is on."""
    r = self.running[worker]
    waiting = (x for n, x in self.running.items() if n != worker and n in self.ready and n not in self.state.granted)
    return any(set(x.gpus) & set(r.gpus) for x in waiting)

  async def ask(self, worker: str, accounts: list[str] | None) -> None:
    """Say which accounts a worker has work ready for, which is what asks the scheduler for its GPUs. None for no work."""
    await self.bus.set_ready(worker, accounts)
    if accounts is None:  # its neighbours see it wait, or not, before the next read of the table
      self.ready.pop(worker, None)
    else:
      self.ready[worker] = accounts
    self.notify()

  async def release(self, worker: str) -> None:
    """Give a worker's GPUs back, once its process has left them or is gone. Only the node they are on does."""
    await self.bus.release_gpus(self.id, worker)
    self.returned[worker] = time.monotonic()
    self.state.granted.pop(worker, None)
    self.notify()

  async def read_state(self) -> tuple[scheduler.Assignment, dict[str, list[str]]]:
    began = time.monotonic()
    table = await self.bus.read_table()
    state = scheduler.from_table(table)
    self.returned = {w: at for w, at in self.returned.items() if at >= began}
    for worker in self.returned:  # this read may be from before it gave its GPUs back
      state.granted.pop(worker, None)
    return state, table["ready"]

  async def announce_node(self) -> None:
    await self.bus.node_heartbeat(self.id, self.gpus, self.gpu_mib * 2**20)
    self.heartbeat = time.monotonic()
    await self.bus.append_event(Ev.NODE_UP, node=self.id, gpus=self.gpus, memory=self.gpu_mib * 2**20)  # for traces

  def open_trainer(self, worker: str, env: dict[str, str]) -> Process:
    width, cmd = len(self.running[worker].gpus), [sys.executable]
    if width > 1:
      cmd += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={width}"]
    return Process([*cmd, "-m", "tinyrl.workers.trainer"], env)

  def open_sampler(self, worker: str, env: dict[str, str]) -> Sampler:
    spec = self.running[worker].spec
    return Sampler(spec.base, spec.lora, env)

  def start(self, worker: str) -> asyncio.Task:
    task = self.tasks[worker] = asyncio.create_task(self.work(worker))
    return task

  async def work(self, worker: str) -> None:
    r, task = dataclasses.replace(self.state.workers[worker]), asyncio.current_task()
    owner = f"{self.id}-{os.urandom(8).hex()}"  # this process, never used again
    sampling = r.spec.role == Role.SAMPLER
    heads, prepared, service, owned = [], {}, {}, set()
    reader = guard = model = closing = None
    swept, virtual = 0.0, 0
    last_run, turn_commands = None, 0

    async def keepalive():
      """Renew the lease while the node is heard from and the process lives. When it stops, so does the worker."""
      try:
        while True:
          await asyncio.sleep(LEASE_MS / 4000)
          cut_off = not self.draining and time.monotonic() - self.heartbeat >= LEASE_MS / 1000
          if not closing and (cut_off or ((p := model.proc) and p.poll() is not None)):
            return
          await self.bus.renew_lease(owner)
      finally:
        if not closing:
          task.cancel()

    async def prepare(cmd):
      with span("node.prepare"):
        try:
          run, checkpoint = await self.bus.get_run(cmd.run), None
          if sampling:
            if run is None:
              raise ValueError(f"no sampling session {cmd.run}")
            if uri := run.get("model_path"):
              checkpoint = await asyncio.to_thread(self.store.local_folder, uri, await self.registry.version(uri))
          return run, checkpoint
        except Exception as e:
          return e

    async def read(delay=0):
      await asyncio.sleep(delay)
      commands = await self.bus.peek_commands(scheduler.jobs_of(self.state, worker), IDLE_MS)
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

    async def own():
      """Take the worker's runs that have no owner: new ones, ones handed off, ones freed from a dead process. A run just taken starts from its row: what the trainer held for it is dropped, and what its last owner left in flight is failed."""
      jobs = scheduler.jobs_of(self.state, worker)
      owned.intersection_update(jobs)  # a run leaves its worker only when it is deleted, with its owner
      for run in jobs:
        if run not in owned and await self.bus.take_ownership(run, owner):
          if not sampling:
            await model.call("drop", run)
          await self.bus.fail_stranded_commands([run], owner)
          owned.add(run)

    def ready():
      """The heads this process may answer: prepared, and of runs it owns."""
      return [c for c in heads if c.run in owned and c.request_id in prepared and prepared[c.request_id].done()]

    def account(cmd):
      """Who a command's tokens are charged to: its training run, which its sampling sessions share."""
      inputs = prepared[cmd.request_id].result()
      return (not isinstance(inputs, Exception) and inputs[0] and inputs[0].get("train_run")) or cmd.run

    async def answer(cmd, inputs, account):
      before, effects, last = time.time(), {}, None
      with span("node.execute"):
        if isinstance(inputs, Exception):
          reply = Reply.error(inputs)
        else:
          run, folder = inputs
          if sampling:
            reply = await model.execute(cmd, run, folder)
          else:
            run = await self.bus.get_run(cmd.run)  # read as its owner, so it is as this process left it
            last, version = run.get("checkpoint"), None  # where its state is saved, if nothing changed it since
            if cmd.op == Op.LOAD_WEIGHTS:
              version = await self.registry.version(json.loads(cmd.body)["uri"])
            reply, effects = await model.call("execute", cmd, run, last and (last["uri"], last["version"]), version)
            if made := effects.get("saved"):
              await self.registry.register_checkpoint(**made)
      with span("node.publish"):
        mutated = effects.get("mutated", False)
        checkpoint = effects.get("checkpoint") if reply.ok else None
        if not await self.bus.write_reply(cmd, reply, owner, before, account, mutated, checkpoint):
          owned.discard(cmd.run)  # freed while this process was paused, or deleted. Its next owner fails the command
          return None
      if (mutated or checkpoint) and last and "/parked/" in last["uri"]:  # a handoff checkpoint the run has moved past
        await self.registry.unregister_checkpoints(last["uri"])
      if cmd.op == Op.DELETE_MODEL and reply.ok:
        owned.discard(cmd.run)
      if DEBUG:
        print(f"{worker} {cmd.request_id} {cmd.op:24s} {(time.time() - before) * 1000:9.2f} ms")
      return reply

    def wanted():
      w = self.state.workers.get(worker)
      return (
        not self.draining
        and w
        and w.node == self.id
        and scheduler.jobs_of(self.state, worker)
        and (w.gpus == r.gpus or movable(r, self.gpus))
      )

    def granted():
      return self.state.granted.get(worker) == (self.id, r.gpus)

    async def activate():
      r.served = 0  # on its GPUs from here, before any await or child startup
      self.notify()
      try:
        with span("node.resume"):
          if not sampling:
            await model.call("resume", r.gpus[0] if movable(r, self.gpus) else None)
          else:
            await model.resume()
            if model.running is None:
              await model.start()
              r.pid = model.proc.pid if model.proc else None
      except Exception as e:
        # A boot failure still answers waiting clients. It keeps its GPUs until close acknowledges.
        for cmd in await self.bus.take_commands(sorted(owned), owner, 1):
          await self.bus.write_reply(cmd, Reply.error(e), owner)
        raise

    async def turn(gpus, accounts):
      """Leave the GPUs when it should, and take them when it may. `accounts` are those it has work ready for, None for no work."""
      if (p := model.proc) and p.poll() is not None:
        raise RuntimeError(f"worker {worker} exited")
      if self.ready.get(worker) != accounts:  # as the table has it, so a request that was lost is made again
        await self.ask(worker, accounts)
      waiting = self.someone_waits(worker)
      if r.served is not None and (gpus != r.gpus or (waiting and (accounts is None or r.served >= scheduler.TURN))):
        with span("node.park"):
          await model.park() if sampling else await model.call("park")
        r.served = None
        await self.release(worker)
      elif r.served is None and accounts is None and granted():
        await self.release(worker)  # granted GPUs it no longer has work for
      if r.gpus != gpus:
        r.gpus = gpus
        self.notify()
      if accounts is not None and r.served is None and granted() and self.clear(worker):
        await activate()

    async def execute(candidates):
      nonlocal reader, last_run, turn_commands, heads, virtual
      for cmd in candidates:
        service[cmd.run] = max(service.get(cmd.run, virtual), virtual)
      prefer = last_run if turn_commands < RUN_COMMANDS else None
      runs = scheduler.select_runs(
        candidates, service, SAMPLE_RUNS if sampling and r.spec.lora else 1, prefer if not sampling else None
      )
      inputs = {c.run: (prepared[c.request_id].result(), account(c)) for c in candidates if c.run in runs}
      failed = all(isinstance(value, Exception) for value, _ in inputs.values())
      # Discard a peek that could finish with pre-claim cursors. The next read starts after the claim.
      if reader:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
      with span("node.claim"):
        cmds = await self.bus.take_commands(runs, owner, SAMPLE_COMMANDS if sampling and not failed else 1)
      reader = asyncio.create_task(read())
      reader.add_done_callback(lambda _: self.notify())
      heads = [c for c in heads if c.run not in runs]
      if not cmds:
        return
      if sampling:
        async with asyncio.TaskGroup() as group:
          for cmd in cmds:
            group.create_task(answer(cmd, *inputs[cmd.run]))
      else:
        cmd = cmds[0]
        if cmd.run != last_run or prefer is None:
          last_run, turn_commands = cmd.run, 0
        turn_commands += 1
        reply = await answer(cmd, *inputs[cmd.run])
      for cmd in cmds:
        service[cmd.run] = service.get(cmd.run, 0) + cmd.cost
      if r.served is not None:  # a head that failed to prepare is answered without the GPUs
        r.served += sum(cmd.cost for cmd in cmds)
      active = {c.run for c in [*ready(), *cmds]}
      virtual = max(virtual, min(service[run] for run in active))
      return cmds[0] if not sampling and reply and reply.ok else None

    async def hand_off(run):
      """Give a run to its next owner. A trainer first saves the state it holds, unless the run already has a checkpoint of it or never changed."""
      row, made = await self.bus.get_run(run) or {}, None
      if not sampling and row.get("acked") and not row.get("checkpoint"):
        if made := await model.call("export", run):
          await self.registry.register_checkpoint(**made)
      if not await self.bus.hand_off(run, owner, made and {"uri": made["uri"], "version": made["version"]}) and made:
        await self.registry.unregister_checkpoints(made["uri"])  # it lost the run first, so nobody restores this

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
      r.served = None  # its GPUs go back only after the child has acknowledged park or close
      await self.ask(worker, None)
      await self.release(worker)
      if died:
        await self.bus.take_commands(sorted(owned), owner, 1)
      await self.bus.fail_stranded_commands(sorted(owned), owner)
      if guard:
        guard.cancel()
        await asyncio.gather(guard, return_exceptions=True)
      await self.bus.release_lease(owner)  # the scheduler frees the runs it still owned

    try:
      if self.draining:
        return
      self.running[worker] = r
      env = {**scheduler.settings(r.spec, self.gpus, self.gpu_mib * 2**20), **os.environ}
      env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(self.gpus) if movable(r, self.gpus) else r.gpus))
      env.update(self.store.env())
      if not sampling:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
      model = self.open_sampler(worker, env) if sampling else self.open_trainer(worker, env)
      r.pid = model.proc.pid if model.proc else None
      await self.bus.renew_lease(owner)
      guard = asyncio.create_task(keepalive())
      reader = asyncio.create_task(read())
      reader.add_done_callback(lambda _: self.notify())

      # Each step observes ready heads, yields when necessary, and executes one bounded batch.
      while wanted():
        changed = self.changed
        collect()
        await own()
        runnable = ready()
        failed = [c for c in runnable if isinstance(prepared[c.request_id].result(), Exception)]
        if failed:
          await execute(failed)
          runnable = [c for c in ready() if not isinstance(prepared[c.request_id].result(), Exception)]
          if not wanted():
            break
        warm = sampling and env.get("TINYRL_WARM_SAMPLER", "0") != "0" and model.running is None
        await turn(
          self.state.workers[worker].gpus, sorted({account(c) for c in runnable}) if runnable or warm else None
        )
        if not wanted():
          break
        if runnable and r.served is not None:
          if (cmd := await execute(runnable)) and cmd.op == Op.DELETE_MODEL and not r.spec.lora:
            break
        else:
          if reader is None:
            reader = asyncio.create_task(read(IDLE_MS / 1000 if heads else 0))
            reader.add_done_callback(lambda _: self.notify())
          await changed.wait()
        if time.monotonic() - swept >= STRANDED_MS / 1000:
          swept = time.monotonic()
          await self.bus.fail_stranded_commands(sorted(owned), owner)

      # A worker that stops hands its runs off. Preparation never claimed work, so nothing is in flight.
      if not sampling and len(r.gpus) > 1 and r.served is None:
        # A wide trainer saves through all its ranks, on its GPUs. It waits for the node's own processes to leave
        # them, not for a grant: a stopping node no longer reads the table.
        await self.ask(worker, [])
        while not self.clear(worker):
          await self.changed.wait()
        await activate()
      for run in sorted(owned):
        await hand_off(run)
    finally:
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

  async def tick(self) -> None:
    workers = {n: {"gpus": r.gpus, "pid": r.pid} for n, r in self.running.items()}  # where its processes are
    await self.bus.node_heartbeat(self.id, self.gpus, self.gpu_mib * 2**20, workers, self.lag)
    self.heartbeat = time.monotonic()
    old = (self.state, self.ready)
    self.state, self.ready = await self.read_state()
    self.dispatch()
    mine = {
      n for n, w in self.state.workers.items() if w.node == self.id and w.gpus and scheduler.jobs_of(self.state, n)
    }
    if not self.draining:
      for worker in sorted(mine - self.tasks.keys()):
        self.start(worker)
    for worker, (node, _) in list(self.state.granted.items()):
      if node == self.id and worker not in self.tasks:  # granted to a process this node isn't running
        await self.release(worker)
    if time.time() - self.swept > SWEEP_S and (self.sweeping is None or self.sweeping.done()):
      self.swept = time.time()
      self.sweeping = asyncio.create_task(self.registry.sweep(self.store, SWEEP_GRACE_S))
    if old != (self.state, self.ready):
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
    await asyncio.to_thread(self.store.close)  # other nodes copy its saves from it until they're in the bucket

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
