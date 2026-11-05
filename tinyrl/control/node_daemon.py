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
from dataclasses import dataclass, field

from tinyrl.bus import LEASE_MS, Bus, Cmd, Ev, Op, Reply, Role
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


@dataclass
class Running:
  owner: str
  kind: scheduler.Kind
  gpus: tuple[int, ...]
  model: Process | None = None
  sampler: Sampler | None = None
  task: asyncio.Task | None = None
  reader: asyncio.Task | None = None
  queue: collections.deque = field(default_factory=collections.deque)  # prepared (command, run, checkpoint) triples
  ticket: int = 0
  since: float | None = None  # reservation begins before wake and ends only after park or close
  busy: float = 0.0
  stop: str = ""  # drain claimed work, abort uncertain work, or close/retry cleanup
  renewed: float = field(default_factory=time.monotonic)

  @property
  def proc(self):
    return self.model.proc if self.model else getattr(self.sampler.running, "server", None)

  def cancel(self) -> None:
    if self.stop == "close":
      return
    self.stop, self.ticket = "abort", 0
    for task in (self.task, self.reader):
      if task and not task.done() and not task.cancelling():
        task.cancel()


class NodeDaemon:
  def __init__(
    self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store: Store | None = None
  ):
    self.bus, self.registry = bus, Registry(bus)
    self.store = store if store is not None else Store()
    self.gpus, self.id = gpus, node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.running: dict[str, Running] = {}
    self.state, self.after, self.serial = scheduler.State(), "0-0", 0
    self.swept, self.uploading, self.sweeping = 0.0, None, None

  def request(self, r: Running) -> None:
    if not r.ticket and r.since is None:
      self.serial += 1
      r.ticket = self.serial

  async def announce_node(self) -> None:
    await self.bus.node_heartbeat(self.id, self.gpus)
    await self.bus.append_event(Ev.NODE_UP, node=self.id, gpus=self.gpus, memory=self.gpu_mib * 2**20)

  async def owned(self, worker: str) -> None:
    r, before = self.running[worker], time.monotonic()
    if r.stop == "abort" or not (
      await self.bus.renew_lease(worker, r.owner)
      or (r.stop != "close" and await self.bus.take_lease(worker, r.owner, recover=True))
    ):
      r.cancel()
      raise RuntimeError(f"lost worker {worker}")
    r.renewed = before

  async def renew(self) -> None:
    for worker, r in list(self.running.items()):
      if time.monotonic() - r.renewed >= LEASE_MS / 4000:
        try:
          await self.owned(worker)
        except Exception:
          if time.monotonic() - r.renewed >= LEASE_MS / 1000:
            r.cancel()

  def open_trainer(self, worker: str, env: dict[str, str]) -> Process:
    width, cmd = len(self.running[worker].gpus), [sys.executable]
    if width > 1:
      cmd += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={width}"]
    return Process([*cmd, "-m", "tinyrl.workers.trainer"], env)

  async def open_worker(self, worker: str, owner: str = "") -> Running | None:
    w, owner = self.state.workers[worker], owner or f"{self.id}-{os.urandom(8).hex()}"
    if not await self.bus.take_lease(worker, owner):
      return None
    try:
      await self.bus.fail_stranded_commands(scheduler.members(self.state, worker), worker, owner, 0)
      if not await self.bus.renew_lease(worker, owner):
        return None
      r = self.running[worker] = Running(owner, w.kind, w.gpus)
      env = {**scheduler.settings(w.kind, self.gpus, self.gpu_mib * 2**20), **os.environ}
      env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(self.gpus) if movable(w, self.gpus) else w.gpus))
      env["CKPT"], env["CKPT_CACHE"] = self.store.root, self.store.cache
      if w.kind.role == Role.TRAINER:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        r.model = self.open_trainer(worker, env)
      else:
        r.sampler = Sampler(w.kind.base, w.kind.lora, env)
        if env.get("TINYRL_WARM_SAMPLER", "0") != "0":
          self.request(r)
      return r
    except BaseException:
      self.running.pop(worker, None)
      await self.bus.release_lease(worker, owner)
      raise

  async def read(self, worker: str):
    r = self.running[worker]
    commands = await self.bus.take_commands(
      scheduler.members(self.state, worker), r.owner, 128 if r.sampler else 16, IDLE_MS if r.since is not None else 1000
    )
    if not commands and not r.task and not r.queue:
      await self.bus.fail_stranded_commands(scheduler.members(self.state, worker), worker, r.owner, STRANDED_MS)
    if not r.sampler:
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

  async def answer_command(self, worker: str, cmd: Cmd, run=None, checkpoint=None) -> Reply:
    r = self.running[worker]
    if r.sampler:
      await self.owned(worker)
      return await r.sampler.execute(cmd, run, checkpoint)
    run, parked = await self.bus.get_run(cmd.run), await self.bus.parked_checkpoint(cmd.run)
    version = None
    if cmd.op == Op.LOAD_WEIGHTS:
      version = (await self.registry.get_checkpoint(json.loads(cmd.body)["uri"]) or {}).get("version")
    saved = (parked, (await self.registry.get_checkpoint(parked) or {}).get("version")) if parked else None
    await self.owned(worker)
    reply, effects = await r.model.call("execute", cmd, run, saved, version)
    await self.owned(worker)
    if checkpoint := effects.get("saved"):
      await self.registry.register_checkpoint(**checkpoint)
    if restored := effects.get("restored"):
      if await self.bus.set_parked_checkpoint(cmd.run, None, worker, r.owner):
        await self.registry.unregister_checkpoints(restored)
    return reply

  async def wake(self, worker: str) -> int:
    r = self.running[worker]
    await self.owned(worker)
    if r.model:
      await r.model.call("resume", r.gpus[0] if movable(r, self.gpus) else None)
    else:
      await r.sampler.resume()  # an uncertain wake keeps its reservation until close
      try:
        if r.sampler.running is None:
          await r.sampler.start()
          r.since = time.time()  # cold boot is outside the first quantum
          await asyncio.sleep(0.02)
      except Exception as e:
        commands, r.queue = r.queue, collections.deque()
        for cmd, _, _ in commands:
          if not await self.bus.write_reply(cmd, Reply.error(e), worker, r.owner):
            raise RuntimeError(f"lost worker {worker}") from e
        r.since = None
        return len(commands)
    return 0

  async def execute(self, worker: str) -> int:
    r, start = self.running[worker], time.time()
    if r.sampler:
      sessions = list(dict.fromkeys(c.run for c, _, _ in r.queue))[: 3 if r.kind.lora else 1]
      batch = [x for x in r.queue if x[0].run in sessions]
      r.queue = collections.deque(x for x in r.queue if x[0].run not in sessions)
    else:
      batch = [r.queue.popleft()]

    async def one(cmd, run, checkpoint):
      before = time.time()
      reply = await self.answer_command(worker, cmd, run, checkpoint)
      if not await self.bus.write_reply(cmd, reply, worker, r.owner, started=before):
        r.cancel()
        raise RuntimeError(f"lost worker {worker}")
      if cmd.op == Op.DELETE_MODEL and reply.ok:
        await self.bus.delete_run(cmd.run, worker)
        await self.registry.unregister_checkpoints(*await self.registry.checkpoints_owned_by_run(cmd.run))
        if not r.kind.lora:
          r.stop = "drain"
      if DEBUG:
        print(f"{worker} {cmd.request_id} {cmd.op:24s} {(time.time() - before) * 1000:9.2f} ms")

    async with asyncio.TaskGroup() as group:
      for item in batch:
        group.create_task(one(*item))
    if r.model and batch[0][0].op == Op.OPTIM_STEP and r.kind.lora and not r.stop:
      await asyncio.sleep(0.015)  # let the client's sampler save join before taking another run's command
    r.busy += time.time() - start
    return len(batch)

  async def park(self, worker: str) -> int:
    r = self.running[worker]
    await r.model.call("park") if r.model else await r.sampler.park()
    r.since = None
    return 0

  async def park_runs(self, worker: str) -> None:
    r = self.running[worker]
    if r.model:
      await self.owned(worker)
      for saved in await r.model.call("park_all"):
        await self.owned(worker)
        run = saved.pop("run")
        await self.registry.register_checkpoint(**saved)
        if not await self.bus.set_parked_checkpoint(run, saved["uri"], worker, r.owner):
          raise RuntimeError(f"lost worker {worker}")

  async def close_worker(self, worker: str, save: bool = False) -> int:
    r = self.running[worker]
    try:
      if save:
        await self.park_runs(worker)
    finally:
      r.stop, r.ticket = "close", 0
      if r.reader:
        r.reader.cancel()
        await asyncio.gather(r.reader, return_exceptions=True)
        r.reader = None
      closing = asyncio.create_task(r.model.close() if r.model else r.sampler.close())
      try:
        await asyncio.shield(closing)
      except asyncio.CancelledError:
        await closing
      r.since = None
      if await self.bus.lease_owner(worker) == r.owner:
        await self.bus.fail_stranded_commands(scheduler.members(self.state, worker), worker, r.owner, 0)
      await self.bus.release_lease(worker, r.owner)
    return 0

  def dispatch(self) -> None:
    for worker, r in list(self.running.items()):
      for attr in ("reader", "task"):
        task = getattr(r, attr)
        if task and task.done():
          setattr(r, attr, None)
          try:
            result = task.result()
            if attr == "reader":
              enqueue(r.queue, result, r.kind.lora)
            elif r.stop == "close":
              del self.running[worker]
          except BaseException as e:
            if not isinstance(e, asyncio.CancelledError):
              print(f"{worker} stopped: {e!r}", file=sys.stderr, flush=True)
            r.cancel()
    scheduler.place_gpus(self.state, self.id, self.running)
    for worker, r in self.running.items():
      w = self.state.workers.get(worker)
      if self.id not in self.state.nodes or (r.proc and r.proc.poll() is not None):
        r.cancel()
      if not r.stop and (
        not w
        or w.node != self.id
        or not scheduler.members(self.state, worker)
        or (w.gpus != r.gpus and not movable(r, self.gpus))
      ):
        r.stop = "drain"
      if not r.stop and r.reader is None and len(r.queue) < (128 if r.sampler else 16):
        r.reader = asyncio.create_task(self.read(worker))
      if r.queue and r.stop not in ("abort", "close"):
        self.request(r)
      elif r.stop == "drain" and r.sampler:
        r.ticket = 0
    # Admit compatible waiters before deciding whether an existing holder has to park.
    for worker, r in sorted(self.running.items(), key=lambda item: item[1].since is not None):
      if r.task:
        continue
      if r.stop in ("abort", "close"):
        r.stop = "close"  # own cleanup before its task starts, including repeated dispatches after node loss
        r.task = asyncio.create_task(self.close_worker(worker))
        continue
      w = self.state.workers.get(worker)
      finishing = r.stop == "drain" and not r.queue and r.reader is None
      if finishing and (not r.model or len(r.gpus) == 1 or r.since is not None):
        r.task = asyncio.create_task(self.close_worker(worker, save=True))
        continue
      moving = not r.stop and w and r.gpus != w.gpus
      if r.since is not None:
        if moving or (
          scheduler.has_waiter(self.running, r.gpus) and (not r.queue or time.time() >= r.since + self.state.quantum)
        ):
          r.task = asyncio.create_task(self.park(worker))
        elif r.queue:
          r.task = asyncio.create_task(self.execute(worker))
      else:
        if moving:
          r.gpus = w.gpus
        if finishing:
          self.request(r)  # FSDP checkpoint gathers need their GPUs together
        if scheduler.can_run(self.state, self.id, self.running, worker):
          r.since, r.ticket = time.time(), 0
          r.task = asyncio.create_task(self.wake(worker))

  async def tick(self) -> None:
    workers = {
      n: {
        "kind": dataclasses.asdict(r.kind),
        "node": self.id,
        "gpus": r.gpus,
        "on": [(self.id, g) for g in r.gpus],
        "asked": r.ticket,
        "since": r.since,
        "busy": r.busy,
        "pid": r.proc.pid if r.proc else None,
      }
      for n, r in self.running.items()
    }
    await self.bus.node_heartbeat(self.id, self.gpus, workers)
    if await self.bus.event_log_size() < self.state.n:
      for r in self.running.values():
        r.cancel()
      self.state, self.after = scheduler.State(), "0-0"
    while events := await self.bus.read_events(self.after):
      for e in events:
        scheduler.fold(self.state, e)
      self.after = events[-1].id
    await self.renew()
    self.dispatch()
    mine = {n for n, w in self.state.workers.items() if w.node == self.id and scheduler.members(self.state, n)}
    for worker in sorted(mine - self.running.keys()):
      await self.open_worker(worker)
    if self.id not in self.state.nodes and not self.running:
      await self.announce_node()
    if time.time() - self.swept > SWEEP_S and (self.sweeping is None or self.sweeping.done()):
      self.swept = time.time()
      self.sweeping = asyncio.create_task(self.sweep_checkpoints())
    if self.uploading is None or self.uploading.done():
      self.uploading = asyncio.create_task(asyncio.to_thread(self.store.upload_pending_saves))

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
    tasks = {t for r in self.running.values() for t in (r.task, r.reader) if t and not t.done()}
    if tasks:
      await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    else:
      await asyncio.sleep(timeout)

  async def serve(self) -> None:
    await self.announce_node()
    next_tick = 0.0
    try:
      while True:
        if time.monotonic() >= next_tick:
          await self.tick()
          next_tick = time.monotonic() + TICK
        self.dispatch()
        await self.wait(max(0.0, next_tick - time.monotonic()))
    finally:
      for r in self.running.values():
        r.stop = r.stop or "drain"
      deadline = time.monotonic() + DRAIN_S
      while self.running:
        if time.monotonic() >= deadline:
          for r in self.running.values():
            r.cancel()
        await self.renew()
        self.dispatch()
        await self.wait(TICK)
      await asyncio.to_thread(self.store.upload_pending_saves)


if __name__ == "__main__":

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await NodeDaemon(Bus()).serve()

  asyncio.run(main())
