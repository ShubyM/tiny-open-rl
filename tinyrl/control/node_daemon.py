"""One node owns its worker leases, ready commands and GPU reservations. Children only execute model operations."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import json
import os
import shutil
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
from tinyrl.workers.sampler import Sampler

GPUS, DRAIN_S, TICK = getenv("GPUS", 1), getenv("DRAIN_S", 20.0), getenv("TICK", 0.25)
IDLE_MS, STRANDED_MS = getenv("IDLE_MS", 150), 5000
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


def enqueue(queue, commands, lora: bool) -> None:
  blocked, front, back = {x[0].run for x in queue}, [], []
  for item in commands:
    cmd = item[0]
    (front if lora and cmd.op == Op.SAVE_WEIGHTS_FOR_SAMPLER and cmd.run not in blocked else back).append(item)
    if cmd.op != Op.SAVE_WEIGHTS_FOR_SAMPLER:
      blocked.add(cmd.run)
  queue.extendleft(reversed(front))
  queue.extend(back)


class NodeDaemon:
  def __init__(
    self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store: Store | None = None
  ):
    self.bus, self.registry = bus, Registry(bus)
    self.store = store if store is not None else Store()
    self.gpus, self.id = gpus, node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.running: dict[str, scheduler.Worker] = {}
    self.tasks: dict[str, asyncio.Task] = {}
    self.state, self.after, self.serial = scheduler.State(), "0-0", 0
    self.changed, self.draining = asyncio.Event(), False
    self.heartbeat = time.monotonic()
    self.swept, self.uploading, self.sweeping = 0.0, None, None

  def notify(self) -> None:
    changed, self.changed = self.changed, asyncio.Event()
    changed.set()

  async def acquire(self, worker: str, warm: bool = False) -> bool:
    r = self.running[worker]
    self.serial += 1
    r.ticket = self.serial
    self.notify()
    try:
      while True:
        changed = self.changed
        w = self.state.workers.get(worker)
        if warm and (self.draining or not w or w.node != self.id or not scheduler.members(self.state, worker)):
          return False
        if w and w.node == self.id and w.gpus and movable(r, self.gpus):
          r.gpus = w.gpus
        if scheduler.can_run(self.state, self.id, self.running, worker):
          r.since, r.ticket = time.time(), 0
          return True
        await changed.wait()
    finally:
      r.ticket = 0
      self.notify()

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
    task.add_done_callback(lambda _: self.notify())
    return task

  async def work(self, worker: str) -> None:
    r, task = dataclasses.replace(self.state.workers[worker]), asyncio.current_task()
    owner = f"{self.id}-{os.urandom(8).hex()}"
    sampling = r.kind.role == Role.SAMPLER
    queue, reader, guard, model = collections.deque(), None, None, None
    renewed, closing, draining = time.monotonic(), False, False

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

    def proc():
      return getattr(model.running, "server", None) if sampling else model.proc

    async def keepalive():
      while True:
        await asyncio.sleep(LEASE_MS / 4000)
        if not closing and not self.draining and time.monotonic() - self.heartbeat >= LEASE_MS / 1000:
          task.cancel()
          return
        try:
          await owned()
          if not closing and (p := proc()) and p.poll() is not None:
            task.cancel()
            return
        except Exception:
          if not closing and time.monotonic() - renewed >= LEASE_MS / 1000:
            task.cancel()
            return

    async def read():
      commands = await self.bus.take_commands(
        scheduler.members(self.state, worker), owner, 128 if sampling else 16, IDLE_MS if r.since is not None else 1000
      )
      if not sampling:
        return [(c, None, None) for c in commands]
      runs = {run: await self.bus.get_run(run) for run in dict.fromkeys(c.run for c in commands)}

      async def stage(uri):
        try:
          version = (await self.registry.get_checkpoint(uri) or {}).get("version")
          return await asyncio.to_thread(self.store.local_folder, uri, version)
        except Exception as e:
          return e

      uris = {x["model_path"] for x in runs.values() if x and x.get("model_path")}
      staged = dict(zip(uris, await asyncio.gather(*(stage(uri) for uri in uris))))
      return [(c, runs[c.run], staged.get((runs[c.run] or {}).get("model_path"))) for c in commands]

    async def answer(cmd, run, checkpoint):
      nonlocal draining
      before = time.time()
      if sampling:
        await owned()
        reply = await model.execute(cmd, run, checkpoint)
      else:
        run, parked = await self.bus.get_run(cmd.run), await self.bus.parked_checkpoint(cmd.run)
        version = None
        if cmd.op == Op.LOAD_WEIGHTS:
          version = (await self.registry.get_checkpoint(json.loads(cmd.body)["uri"]) or {}).get("version")
        saved = (parked, (await self.registry.get_checkpoint(parked) or {}).get("version")) if parked else None
        await owned()
        reply, effects = await model.call("execute", cmd, run, saved, version)
        await owned()
        if checkpoint := effects.get("saved"):
          await self.registry.register_checkpoint(**checkpoint)
        if restored := effects.get("restored"):
          if await self.bus.set_parked_checkpoint(cmd.run, None, worker, owner):
            await self.registry.unregister_checkpoints(restored)
      if not await self.bus.write_reply(cmd, reply, worker, owner, started=before):
        raise RuntimeError(f"lost worker {worker}")
      if cmd.op == Op.DELETE_MODEL and reply.ok:
        await self.bus.delete_run(cmd.run, worker)
        await self.registry.unregister_checkpoints(*await self.registry.checkpoints_owned_by_run(cmd.run))
        draining |= not r.kind.lora
      if DEBUG:
        print(f"{worker} {cmd.request_id} {cmd.op:24s} {(time.time() - before) * 1000:9.2f} ms")

    async def cleanup():
      if reader:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
      if model:
        while True:
          try:
            await model.close()
            break
          except (Exception, asyncio.CancelledError) as e:
            print(f"{worker} could not close, retrying: {e!r}", file=sys.stderr, flush=True)
            await asyncio.sleep(TICK)
      r.since = None  # release only after the child has acknowledged park or close
      self.notify()
      if guard:
        guard.cancel()
        await asyncio.gather(guard, return_exceptions=True)
      if await self.bus.lease_owner(worker) == owner:
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
      env["CKPT"], env["CKPT_CACHE"] = self.store.root, self.store.cache
      if not sampling:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
      model = self.open_sampler(worker, env) if sampling else self.open_trainer(worker, env)
      warm = sampling and env.get("TINYRL_WARM_SAMPLER", "0") != "0"
      r.pid = proc().pid if proc() else None
      guard = asyncio.create_task(keepalive())
      while True:
        changed = self.changed
        if (p := proc()) and p.poll() is not None:
          raise RuntimeError(f"worker {worker} exited")
        w = self.state.workers.get(worker)
        draining |= (
          self.draining
          or not w
          or w.node != self.id
          or not scheduler.members(self.state, worker)
          or (w.gpus != r.gpus and not movable(r, self.gpus))
        )
        warm &= not draining
        if reader and reader.done():
          commands, reader = reader.result(), None
          enqueue(queue, commands, r.kind.lora)
          if not commands and not queue:
            await self.bus.fail_stranded_commands(scheduler.members(self.state, worker), worker, owner, STRANDED_MS)
        if not draining and reader is None and len(queue) < (128 if sampling else 16):
          reader = asyncio.create_task(read())
        finishing = draining and not queue and reader is None
        if finishing and (sampling or len(r.gpus) == 1 or r.since is not None):
          if not sampling:
            await owned()
            for saved in await model.call("park_all"):
              await owned()
              run = saved.pop("run")
              await self.registry.register_checkpoint(**saved)
              if not await self.bus.set_parked_checkpoint(run, saved["uri"], worker, owner):
                raise RuntimeError(f"lost worker {worker}")
          return
        moving = not draining and w.gpus != r.gpus
        if r.since is not None and (
          moving
          or (scheduler.has_waiter(self.running, r.gpus) and (not queue or time.time() >= r.since + self.state.quantum))
        ):
          await model.park() if sampling else await model.call("park")
          r.since = None
          self.notify()
          continue
        if moving:
          r.gpus = w.gpus
          self.notify()
        if r.since is None and (queue or warm or finishing):
          if not await self.acquire(worker, warm=warm and not queue and not finishing):
            continue
          await owned()
          warm = False
          if sampling:
            await model.resume()  # an uncertain wake keeps its reservation until close
            try:
              if model.running is None:
                await model.start()
                r.pid = proc().pid if proc() else None
                r.since = time.time()  # cold boot is outside the first quantum
                await asyncio.sleep(0.02)
            except Exception as e:
              if reader and reader.done():
                enqueue(queue, reader.result(), r.kind.lora)
                reader = None
              while queue:
                cmd, _, _ = queue.popleft()
                if not await self.bus.write_reply(cmd, Reply.error(e), worker, owner):
                  raise RuntimeError(f"lost worker {worker}") from e
              r.since = None
              self.notify()
          else:
            await model.call("resume", r.gpus[0] if movable(r, self.gpus) else None)
          continue
        if queue:
          before = time.time()
          if sampling:
            sessions = list(dict.fromkeys(c.run for c, _, _ in queue))[: 3 if r.kind.lora else 1]
            batch = [x for x in queue if x[0].run in sessions]
            queue = collections.deque(x for x in queue if x[0].run not in sessions)
            async with asyncio.TaskGroup() as group:
              for item in batch:
                group.create_task(answer(*item))
          else:
            cmd, run, checkpoint = queue.popleft()
            await answer(cmd, run, checkpoint)
            if cmd.op == Op.OPTIM_STEP and r.kind.lora and not draining:
              await asyncio.sleep(0.015)  # let the client's sampler save join before another run's command
          r.busy += time.time() - before
          continue
        waiting = asyncio.create_task(changed.wait())
        try:
          await asyncio.wait([waiting, *([reader] if reader else [])], return_when=asyncio.FIRST_COMPLETED)
        finally:
          waiting.cancel()
          await asyncio.gather(waiting, return_exceptions=True)
    finally:
      closing, r.ticket = True, 0
      self.notify()
      done = asyncio.create_task(cleanup())
      while True:
        try:
          await asyncio.shield(done)
          break
        except asyncio.CancelledError:
          if done.cancelled():
            done = asyncio.create_task(cleanup())

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
      n: {
        "kind": dataclasses.asdict(r.kind),
        "node": self.id,
        "gpus": r.gpus,
        "on": [(self.id, g) for g in r.gpus],
        "asked": r.ticket,
        "since": r.since,
        "busy": r.busy,
        "pid": r.pid,
      }
      for n, r in self.running.items()
    }
    await self.bus.node_heartbeat(self.id, self.gpus, workers)
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
      self.sweeping = asyncio.create_task(self.sweep_checkpoints())
    if self.uploading is None or self.uploading.done():
      self.uploading = asyncio.create_task(asyncio.to_thread(self.store.upload_pending_saves))
    if self.after != after:
      self.notify()

  async def sweep_checkpoints(self) -> None:
    """Delete the checkpoints in this node's store that the registry doesn't have: deleted, expired, or their run's. Checkpoints older than the registry itself are left alone, so turning it on never deletes what came before. Then the links of full sampler weight chains no registered checkpoint is built from. It also keeps the node's cache of a bucket store under CKPT_CACHE_BYTES."""
    await asyncio.to_thread(self.store.trim_cache)
    if (since := await self.registry.first_registered_at()) is None:
      return
    for uri in await asyncio.to_thread(self.store.list_checkpoints):
      try:
        if await self.registry.get_checkpoint(uri) is None:
          saved = await asyncio.to_thread(self.store.saved_at, uri)
          if since < saved < time.time() - SWEEP_GRACE_S:
            await asyncio.to_thread(self.store.delete, uri)
      except Exception as e:  # one checkpoint it can't read or delete mustn't stop the rest of the sweep
        print(f"sweep could not delete {uri}, trying again next sweep: {e!r}", file=sys.stderr, flush=True)
    await asyncio.to_thread(
      self.store.delete_unused_chain_links,
      await self.registry.registered_versions(),
      since,
      time.time() - SWEEP_GRACE_S,
    )
    if os.path.isdir(self.store.cache):
      for e in list(os.scandir(self.store.cache)):
        if e.name.endswith(".upload") and since < e.stat().st_mtime < time.time() - SWEEP_GRACE_S:
          try:
            with open(e.path) as f:
              uri = json.load(f)["uri"]
          except Exception:
            continue
          if await self.registry.get_checkpoint(uri) is None:
            version = e.name.removesuffix(".upload")
            await asyncio.to_thread(shutil.rmtree, f"{self.store.cache}/{version}", ignore_errors=True)
            with contextlib.suppress(OSError):
              await asyncio.to_thread(os.unlink, e.path)

  async def wait(self, timeout: float) -> None:
    with contextlib.suppress(TimeoutError):
      await asyncio.wait_for(self.changed.wait(), timeout)

  async def close(self, graceful: bool = True) -> None:
    self.draining = True
    self.notify()
    deadline = time.monotonic() + (DRAIN_S if graceful else 0)
    while self.tasks:
      if time.monotonic() >= deadline:
        for task in self.tasks.values():
          if not task.done() and not task.cancelling():
            task.cancel()
      self.dispatch()
      await self.wait(TICK)
    await asyncio.to_thread(self.store.upload_pending_saves)

  async def serve(self) -> None:
    await self.announce_node()
    next_tick = 0.0
    try:
      while True:
        changed = self.changed
        if time.monotonic() >= next_tick:
          await self.tick()
          next_tick = time.monotonic() + TICK
        self.dispatch()
        with contextlib.suppress(TimeoutError):
          await asyncio.wait_for(changed.wait(), max(0.0, next_tick - time.monotonic()))
    finally:
      await self.close()


if __name__ == "__main__":

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await NodeDaemon(Bus()).serve()

  asyncio.run(main())
