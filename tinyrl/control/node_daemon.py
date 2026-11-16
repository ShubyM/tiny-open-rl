"""One node owns its worker leases, ready commands and GPU reservations. Children only execute model operations.

The daemon ticks: it heartbeats, reads the cluster's table, and starts a task for each worker placed here. A task is one Lifetime. It takes the worker's lease and starts its model process, and then each step of its loop

  peeks      at each run's next command, without claiming it, and prepares its inputs
  turns      off the GPUs when another worker's turn has come, and onto them when its own has
  executes   one bounded batch: it claims the least-served runs' commands, runs them on the model and replies

When the worker leaves this node a trainer saves each run for the next one, and the GPUs and the lease are given up only after the process has parked or exited.
"""

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

from tinyrl.bus import LEASE_MS, Bus, Cmd, Ev, Op, Reply, Role
from tinyrl.control import scheduler, turns
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


def assignment(s: scheduler.Assignment) -> tuple:
  return {n: w.node for n, w in s.workers.items()}, {r: x.worker for r, x in s.jobs.items()}


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
    self.running: dict[str, scheduler.Worker] = {}  # the workers with a process here, and the GPUs each is on
    self.tasks: dict[str, asyncio.Task] = {}
    self.state, self.serial = scheduler.Assignment(), 0  # serial: the last ticket given to a worker waiting for GPUs
    self.changed, self.draining = asyncio.Event(), False
    self.heartbeat, self.lag = time.monotonic(), 0.0
    self.swept, self.sweeping = 0.0, None

  async def serve(self) -> None:
    """Tick until cancelled, then stop every worker and finish uploading."""
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

  async def tick(self) -> None:
    """Heartbeat, read the cluster's table, and start a task for each worker placed here that has none."""
    workers = {n: {"gpus": r.gpus, "pid": r.pid} for n, r in self.running.items()}  # where its processes are
    await self.bus.node_heartbeat(self.id, self.gpus, self.gpu_mib * 2**20, workers, self.lag)
    self.heartbeat = time.monotonic()
    old, self.state = self.state, await self.read_state()
    self.dispatch()
    mine = {n for n, w in self.state.workers.items() if w.node == self.id and scheduler.jobs_of(self.state, n)}
    if not self.draining:
      for worker in sorted(mine - self.tasks.keys()):
        self.start(worker)
    if time.time() - self.swept > SWEEP_S and (self.sweeping is None or self.sweeping.done()):
      self.swept = time.time()
      self.sweeping = asyncio.create_task(self.registry.sweep(self.store, SWEEP_GRACE_S))
    if assignment(old) != assignment(self.state):
      self.notify()

  def dispatch(self) -> None:
    """Forget the tasks that ended, stop them all if the cluster no longer counts this node as live, and choose the GPUs of the workers placed here."""
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
    turns.place_gpus(self.state, self.id, self.running)
    if before != {n: w.gpus for n, w in self.state.workers.items()}:
      self.notify()

  def start(self, worker: str) -> asyncio.Task:
    task = self.tasks[worker] = asyncio.create_task(self.work(worker))
    return task

  async def work(self, worker: str) -> None:
    await Lifetime(self, worker).run()

  async def close(self, graceful: bool = True) -> None:
    """Stop the workers, giving them DRAIN_S to finish their batch and save their runs, and upload what they saved."""
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

  async def read_state(self) -> scheduler.Assignment:
    """The cluster's table, with the GPUs this node chose for its workers, which the table doesn't hold."""
    return turns.keep_chosen_gpus(scheduler.from_table(await self.bus.read_table()), self.state)

  def notify(self) -> None:
    """Wake every worker's loop: something they decide from has changed."""
    changed, self.changed = self.changed, asyncio.Event()
    changed.set()

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


class Lifetime:
  """One worker on this node, from taking its lease to giving it up: its model process, its ready commands, and its turns on the GPUs."""

  def __init__(self, node: NodeDaemon, worker: str):
    self.node, self.bus, self.registry, self.store, self.worker = node, node.bus, node.registry, node.store, worker
    self.r = dataclasses.replace(node.state.workers[worker])  # this node's record of it: its actual GPUs, turn and pid
    self.task, self.owner = asyncio.current_task(), f"{node.id}-{os.urandom(8).hex()}"
    self.sampling = self.r.spec.role == Role.SAMPLER
    self.model = self.guard = self.closing = None
    self.renewed, self.swept = time.monotonic(), 0.0
    # each run's next unclaimed command, the task preparing each one's inputs, and the peek that looks for them
    self.heads, self.prepared, self.reader = [], {}, None
    # the tokens served to each run with a head, the floor a new head joins at, and the run a trainer is staying with
    self.service, self.virtual, self.last_run, self.turn_commands = {}, 0, None, 0

  async def run(self) -> None:
    node, r = self.node, self.r
    try:
      if not await self.open():
        return
      # Each step observes ready heads, yields when necessary, and executes one bounded batch.
      while self.wanted():
        changed = node.changed
        self.collect()
        ready, failed = self.ready()
        if failed:  # answered at once, without the GPUs
          await self.execute(failed)
          ready, _ = self.ready()
          if not self.wanted():
            break
        await self.turn(bool(ready) or (self.warm and self.model.running is None))
        if not self.wanted():
          break
        if ready and r.served is not None:
          if (cmd := await self.execute(ready)) and cmd.op == Op.DELETE_MODEL and not r.spec.lora:
            break  # a full fine-tune's process ends with its run
        else:
          if self.reader is None:
            self.read(IDLE_MS / 1000 if self.heads else 0)
          await changed.wait()
        await self.fail_stranded()
      await self.save_runs()
    finally:
      r.ticket = 0
      died = self.model is not None and (p := self.model.proc) is not None and p.poll() is not None
      self.closing = asyncio.create_task(self.cleanup(died))
      node.notify()
      while True:  # cleanup finishes, whatever cancels this task meanwhile
        try:
          await asyncio.shield(self.closing)
          break
        except asyncio.CancelledError:
          if self.closing.cancelled():
            self.closing = asyncio.create_task(self.cleanup(died))

  def jobs(self) -> list[str]:
    return scheduler.jobs_of(self.node.state, self.worker)

  def wanted(self) -> bool:
    """Whether this process should go on: the node isn't stopping, the worker is still placed here with jobs, and its GPUs are the ones the process started on, or ones it can move to."""
    w = self.node.state.workers.get(self.worker)
    return (
      not self.node.draining
      and w
      and w.node == self.node.id
      and self.jobs()
      and (w.gpus == self.r.gpus or movable(self.r, self.node.gpus))
    )

  # ***** its lease and its process *****

  async def open(self) -> bool:
    """Take the worker's lease, fail what its last owner left in flight, and start its process. False if another node owns the worker, or this one is stopping."""
    node, r, worker = self.node, self.r, self.worker
    if not await self.bus.take_lease(worker, self.owner):
      return False
    await self.bus.fail_stranded_commands(self.jobs(), worker, self.owner, 0)
    if not await self.bus.renew_lease(worker, self.owner) or node.draining:
      return False
    node.running[worker] = r
    env = {**scheduler.settings(r.spec, node.gpus, node.gpu_mib * 2**20), **os.environ}
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(node.gpus) if movable(r, node.gpus) else r.gpus))
    env.update(self.store.env())
    if not self.sampling:
      env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    self.model = node.open_sampler(worker, env) if self.sampling else node.open_trainer(worker, env)
    r.pid = self.model.proc.pid if self.model.proc else None
    self.warm = self.sampling and env.get("TINYRL_WARM_SAMPLER", "0") != "0"  # it starts its engine before any command
    self.guard = asyncio.create_task(self.keepalive())
    self.read()
    return True

  async def owned(self) -> None:
    """Renew the lease, or take it back if it ran out and no other node served the worker since. Raises if it is lost."""
    before = time.monotonic()
    if not (
      await self.bus.renew_lease(self.worker, self.owner)
      or (not self.closing and await self.bus.take_lease(self.worker, self.owner, recover=True))
    ):
      if not self.closing and asyncio.current_task() is not self.task and not self.task.cancelling():
        self.task.cancel()
      raise RuntimeError(f"lost worker {self.worker}")
    self.renewed = before

  async def keepalive(self) -> None:
    """Renew the lease through long model operations, and stop the worker when it cannot go on: this node lost touch with Redis, the process died, or the lease could not be renewed for as long as it lasts."""
    while True:
      await asyncio.sleep(LEASE_MS / 4000)
      if not self.closing and not self.node.draining and time.monotonic() - self.node.heartbeat >= LEASE_MS / 1000:
        break
      try:
        await self.owned()
        if not self.closing and (p := self.model.proc) and p.poll() is not None:
          break
      except Exception:
        if not self.closing and time.monotonic() - self.renewed >= LEASE_MS / 1000:
          break
    self.task.cancel()

  async def fail_stranded(self) -> None:
    """Every few seconds, fail the commands an earlier owner of the worker left in flight."""
    if time.monotonic() - self.swept >= STRANDED_MS / 1000:
      self.swept = time.monotonic()
      await self.owned()
      await self.bus.fail_stranded_commands(self.jobs(), self.worker, self.owner, 0)

  # ***** its ready commands *****

  def read(self, delay: float = 0) -> None:
    """Peek at the runs' heads in the background. The node wakes when the peek returns."""
    self.reader = asyncio.create_task(self.peek(delay))
    self.reader.add_done_callback(lambda _: self.node.notify())

  async def peek(self, delay: float) -> list[Cmd]:
    await asyncio.sleep(delay)
    commands = await self.bus.peek_commands(self.jobs(), IDLE_MS)
    for cmd in commands:
      if cmd.request_id not in self.prepared:
        preparing = self.prepared[cmd.request_id] = asyncio.create_task(self.prepare(cmd))
        preparing.add_done_callback(lambda _: self.node.notify())
    return commands

  async def prepare(self, cmd: Cmd):
    """A command's inputs, fetched off the GPUs: its run's row, and for a sample its checkpoint's files. An error is returned, to be the command's reply."""
    with span("node.prepare"):
      try:
        run, checkpoint = await self.bus.get_run(cmd.run), None
        if self.sampling:
          if run is None:
            raise ValueError(f"no sampling session {cmd.run}")
          if uri := run.get("model_path"):
            checkpoint = await asyncio.to_thread(self.store.local_folder, uri, await self.registry.version(uri))
        return run, checkpoint
      except Exception as e:
        return e

  def collect(self) -> None:
    """Take in a finished peek: these are the heads now, and a head that is gone needs no inputs."""
    if self.reader and self.reader.done():
      self.heads, self.reader = self.reader.result(), None
      current = {c.request_id for c in self.heads}
      for request in list(self.prepared):
        if request not in current:
          self.prepared.pop(request).cancel()
      # Idle runs do not bank service credit; a new contender joins the active floor.
      self.service = {c.run: self.service.get(c.run, self.virtual) for c in self.heads}

  def ready(self) -> tuple[list[Cmd], list[Cmd]]:
    """The heads whose inputs are prepared, and the heads whose preparation failed."""
    done = [c for c in self.heads if c.request_id in self.prepared and self.prepared[c.request_id].done()]
    failed = [c for c in done if isinstance(self.prepared[c.request_id].result(), Exception)]
    return [c for c in done if c not in failed], failed

  # ***** its turns on the GPUs *****

  def reserve(self) -> bool:
    """Ask for its GPUs, and take them if its turn has come. It never blocks, so the loop keeps observing newly ready work and preparation errors."""
    node, r = self.node, self.r
    if not r.ticket:
      with span("node.request_gpu"):
        node.serial += 1
        r.ticket = node.serial
      node.notify()
    if turns.can_run(node.state, node.id, node.running, self.worker):
      with span("node.grant_gpu"):
        r.served, r.ticket = 0, 0
      node.notify()
      return True
    return False

  async def turn(self, has_work: bool) -> None:
    """Leave the GPUs when it must, follow the node to the GPUs it chose, and take them when there is work and its turn has come."""
    node, r = self.node, self.r
    gpus = node.state.workers[self.worker].gpus
    if (p := self.model.proc) and p.poll() is not None:
      raise RuntimeError(f"worker {self.worker} exited")
    waiting = turns.someone_waits(node.running, r.gpus)
    if r.served is not None and (gpus != r.gpus or (waiting and (not has_work or r.served >= turns.TURN))):
      with span("node.park"):
        await self.model.park() if self.sampling else await self.model.call("park")
      r.served = None
      node.notify()
    if r.gpus != gpus:
      r.gpus = gpus
      node.notify()
    if not has_work and r.ticket:
      r.ticket = 0
      node.notify()
    if has_work and r.served is None and self.reserve():
      await self.resume()

  async def resume(self) -> None:
    """Bring the process onto the GPUs it has just reserved. If it cannot start, a waiting client is told why."""
    r = self.r
    try:
      await self.owned()
      with span("node.resume"):
        if not self.sampling:
          await self.model.call("resume", r.gpus[0] if movable(r, self.node.gpus) else None)
        else:
          await self.model.resume()
          if self.model.running is None:
            await self.model.start()
            r.pid = self.model.proc.pid if self.model.proc else None
    except Exception as e:
      # A boot failure still answers waiting clients. Keep its reservation until close acknowledges.
      await self.owned()
      for cmd in await self.bus.take_commands(self.jobs(), self.owner, 1):
        if not await self.bus.write_reply(cmd, Reply.error(e), self.worker, self.owner):
          raise RuntimeError(f"lost worker {self.worker}") from e
      raise

  # ***** executing *****

  async def execute(self, candidates: list[Cmd]) -> Cmd | None:
    """Serve one bounded batch of the ready commands. Returns a trainer's command, if it succeeded."""
    runs = self.choose(candidates)
    inputs = {c.run: self.prepared[c.request_id].result() for c in candidates if c.run in runs}
    failed = all(isinstance(value, Exception) for value in inputs.values())
    cmds = await self.claim(runs, SAMPLE_COMMANDS if self.sampling and not failed else 1)
    if not cmds:
      return
    if self.sampling:
      async with asyncio.TaskGroup() as group:
        for cmd in cmds:
          group.create_task(self.answer(cmd, inputs[cmd.run]))
    else:
      cmd = cmds[0]
      if cmd.run != self.last_run or self.turn_commands >= RUN_COMMANDS:
        self.last_run, self.turn_commands = cmd.run, 0
      self.turn_commands += 1
      reply = await self.answer(cmd, inputs[cmd.run])
    self.charge(cmds)
    return cmds[0] if not self.sampling and reply.ok else None

  def choose(self, candidates: list[Cmd]) -> list[str]:
    """The runs served next: those served the fewest tokens, after the run a trainer is staying with."""
    for cmd in candidates:
      self.service[cmd.run] = max(self.service.get(cmd.run, self.virtual), self.virtual)
    stay = self.last_run if not self.sampling and self.turn_commands < RUN_COMMANDS else None
    width = SAMPLE_RUNS if self.sampling and self.r.spec.lora else 1
    return turns.select_runs(candidates, self.service, width, stay)

  async def claim(self, runs: list[str], count: int) -> list[Cmd]:
    """Take up to `count` commands of each run from its stream, and look for the heads behind them."""
    # Discard a peek that could finish with pre-claim cursors. The next read starts after the claim.
    if self.reader:
      self.reader.cancel()
      await asyncio.gather(self.reader, return_exceptions=True)
    await self.owned()
    with span("node.claim"):
      cmds = await self.bus.take_commands(runs, self.owner, count)
    self.read()
    self.heads = [c for c in self.heads if c.run not in runs]
    await self.owned()
    return cmds

  async def answer(self, cmd: Cmd, inputs) -> Reply:
    before = time.time()
    with span("node.execute"):
      if isinstance(inputs, Exception):
        reply = Reply.error(inputs)
      elif self.sampling:
        await self.owned()
        reply = await self.model.execute(cmd, *inputs)
      else:
        reply = await self.train(cmd, inputs[0])
    with span("node.publish"):
      if not await self.bus.write_reply(cmd, reply, self.worker, self.owner, started=before):
        raise RuntimeError(f"lost worker {self.worker}")
    if DEBUG:
      print(f"{self.worker} {cmd.request_id} {cmd.op:24s} {(time.time() - before) * 1000:9.2f} ms")
    return reply

  async def train(self, cmd: Cmd, run: dict | None) -> Reply:
    """Run a trainer's command. A run the last trainer parked is restored first, and then its parked checkpoint is let go."""
    parked, version = await self.bus.parked_checkpoint(cmd.run), None
    if cmd.op == Op.LOAD_WEIGHTS:
      version = await self.registry.version(json.loads(cmd.body)["uri"])
    saved = (parked, await self.registry.version(parked)) if parked else None
    await self.owned()  # metadata awaits may have outlived our ownership
    reply, effects = await self.model.call("execute", cmd, run, saved, version)
    await self.owned()
    if checkpoint := effects.get("saved"):
      await self.registry.register_checkpoint(**checkpoint)
    if restored := effects.get("restored"):
      if not await self.bus.set_parked_checkpoint(cmd.run, None, self.worker, self.owner):
        raise RuntimeError(f"lost worker {self.worker}")
      await self.registry.unregister_checkpoints(restored)
    return reply

  def charge(self, cmds: list[Cmd]) -> None:
    """Count a batch's tokens against its runs and this GPU turn, and raise the floor to the least-served run with work."""
    for cmd in cmds:
      self.service[cmd.run] = self.service.get(cmd.run, 0) + cmd.cost
    if self.r.served is not None:  # a head that failed to prepare is answered without the GPUs
      self.r.served += sum(cmd.cost for cmd in cmds)
    ready, failed = self.ready()
    active = {c.run for c in [*ready, *failed, *cmds]}
    self.virtual = max(self.virtual, min(self.service[run] for run in active))

  # ***** stopping *****

  async def save_runs(self) -> None:
    """A trainer that stops saves each run it holds, for the next trainer to restore. Preparation never claimed work, so nothing is in flight."""
    node, r = self.node, self.r
    if self.sampling:
      return
    if len(r.gpus) > 1 and r.served is None:  # a wide trainer saves through all its ranks, so it needs its GPUs
      with span("node.wait_gpu"):
        while True:
          changed = node.changed
          if self.reserve():
            break
          await changed.wait()
      await self.resume()
    await self.owned()
    for saved in await self.model.call("park_all"):
      await self.owned()
      run = saved.pop("run")
      await self.registry.register_checkpoint(**saved)
      if not await self.bus.set_parked_checkpoint(run, saved["uri"], self.worker, self.owner):
        raise RuntimeError(f"lost worker {self.worker}")

  async def cleanup(self, died: bool) -> None:
    """Close the process, and only then give up its GPUs and its lease. A process that `died` took a command with it, which is failed."""
    node, worker = self.node, self.worker
    for future in [self.reader, *self.prepared.values()]:
      if future:
        future.cancel()
    await asyncio.gather(*(f for f in [self.reader, *self.prepared.values()] if f), return_exceptions=True)
    if self.model:
      while True:
        try:
          with span("node.close"):
            await self.model.close()
          break
        except (Exception, asyncio.CancelledError) as e:
          print(f"{worker} could not close, retrying: {e!r}", file=sys.stderr, flush=True)
          await asyncio.sleep(TICK)
    self.r.served = None  # release only after the child has acknowledged park or close
    node.notify()
    if self.guard:
      self.guard.cancel()
      await asyncio.gather(self.guard, return_exceptions=True)
    if await self.bus.lease_owner(worker) == self.owner:
      if died:
        await self.bus.take_commands(self.jobs(), self.owner, 1)
      await self.bus.fail_stranded_commands(self.jobs(), worker, self.owner, 0)
    await self.bus.release_lease(worker, self.owner)


if __name__ == "__main__":

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await NodeDaemon(Bus()).serve()

  asyncio.run(main())
