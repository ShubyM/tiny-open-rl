"""The node owns leases, queues and GPU turns. Model processes only execute local commands."""

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

from tinyrl.bus import LEASE_MS, Bus, Cmd, Ev, Event, Op, Reply, Role
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


def movable(w: scheduler.Worker, gpus: int) -> bool:
  return w.kind.role == Role.TRAINER and w.kind.lora and len(w.gpus) == 1 and gpus > 1


def enqueue(queue: collections.deque[Cmd], cmds: list[Cmd], lora: bool) -> None:
  blocked, front, back = {x.run for x in queue}, [], []
  for cmd in cmds:
    (front if lora and cmd.op == Op.SAVE_WEIGHTS_FOR_SAMPLER and cmd.run not in blocked else back).append(cmd)
    if cmd.op != Op.SAVE_WEIGHTS_FOR_SAMPLER:
      blocked.add(cmd.run)
  queue.extendleft(reversed(front))
  queue.extend(back)


@dataclass
class Running:
  owner: str
  gpus: tuple[int, ...]
  model: Process | None = None
  sampler: Sampler | None = None
  task: asyncio.Task | None = None
  queue: collections.deque[Cmd] = field(default_factory=collections.deque)
  stopping: bool = False
  lost: bool = False
  awake: bool = False
  closed: bool = False
  renewed: float = field(default_factory=time.monotonic)

  @property
  def proc(self):
    return self.model.proc if self.model else getattr(self.sampler.running, "server", None)

  def cancel(self) -> None:
    self.stopping = self.lost = True
    if self.task and not self.task.cancelling():
      self.task.cancel()


class NodeDaemon:
  def __init__(
    self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store: Store | None = None
  ):
    self.bus, self.registry = bus, Registry(bus)
    self.store = store if store is not None else Store()
    self.gpus, self.id = gpus, node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.running: dict[str, Running] = {}
    self.state, self.after, self.seen = scheduler.State(), "0-0", 0
    self.swept, self.uploading, self.sweeping = 0.0, None, None

  def note(self, kind: Ev, **body) -> None:
    scheduler.fold(self.state, Event("", time.time(), kind, body))

  def schedule(self) -> None:
    for kind, body in scheduler.local_decide(self.state, self.id, time.time()):
      r = self.running.get(body["worker"])
      if body["rule"] == "turn.quantum_expired" and r and r.sampler and r.sampler.running is None:
        continue  # boot time is charged when start returns; it does not consume a sampling turn
      self.note(kind, **body)

  async def announce_node(self) -> None:
    await self.bus.node_heartbeat(self.id, self.gpus)
    await self.bus.append_event(Ev.NODE_UP, node=self.id, gpus=self.gpus, memory=self.gpu_mib * 2**20)

  async def owned(self, worker: str) -> None:
    r, before = self.running[worker], time.monotonic()
    if r.lost or not (
      await self.bus.renew_lease(worker, r.owner)
      or (not r.closed and await self.bus.take_lease(worker, r.owner, recover=True))
    ):
      r.stopping = r.lost = True
      raise RuntimeError(f"lost worker {worker}")
    r.renewed = before

  async def renew(self) -> None:
    for worker, r in list(self.running.items()):
      if r.task and r.task.done():
        continue
      if time.monotonic() - r.renewed >= LEASE_MS / 4000:
        try:
          await self.owned(worker)
        except Exception:
          if r.lost or time.monotonic() - r.renewed >= LEASE_MS / 1000:
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
    r = self.running[worker] = Running(owner, w.gpus)
    env = {**scheduler.settings(w.kind, self.gpus, self.gpu_mib * 2**20), **os.environ}
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, w.gpus))
    env["CKPT"], env["CKPT_CACHE"] = self.store.root, self.store.cache
    try:
      if w.kind.role == Role.TRAINER:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if movable(w, self.gpus):
          env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(self.gpus)))
        r.model = self.open_trainer(worker, env)
      else:
        r.sampler = Sampler(w.kind.base, w.kind.lora, env)
      self.note(Ev.STARTED, worker=worker, node=self.id)
      return r
    except BaseException:
      self.running.pop(worker)
      await self.bus.release_lease(worker, owner)
      raise

  async def acquire(self, worker: str, drain: bool = False) -> bool:
    r = self.running[worker]
    while not r.lost and (not r.stopping or r.queue or drain):
      w = self.state.workers[worker]
      w.closing = r.stopping
      if not r.stopping and not r.awake and w.since is None and movable(w, self.gpus) and r.gpus != w.gpus:
        r.gpus = w.gpus
        self.note(Ev.STARTED, worker=worker, node=self.id)
      self.note(Ev.WANTS_GPU, worker=worker, queued=len(r.queue))
      self.schedule()
      if w.since is not None and not w.revoked:
        await self.owned(worker)
        if not r.awake:
          if r.model:
            await r.model.call("resume", r.gpus[0] if movable(w, self.gpus) else None)
          else:
            await r.sampler.resume()
          r.awake = True
        if not w.revoked:
          return True
      await self.release(worker)
      await asyncio.sleep(TICK)
    return False

  async def release(self, worker: str) -> None:
    r, w = self.running[worker], self.state.workers[worker]
    if w.revoked:
      if r.awake:
        await r.model.call("park") if r.model else await r.sampler.park()
        r.awake = False
      self.note(Ev.RELEASED, worker=worker)
      self.schedule()

  async def answer_command(self, worker: str, cmd: Cmd, runs=None, staged=None) -> Reply:
    r = self.running[worker]
    await self.owned(worker)  # a resumed node checks ownership before sending anything to its child
    if r.sampler:
      return await r.sampler.execute(cmd, runs.get(cmd.run), staged)
    run, parked = await self.bus.get_run(cmd.run), await self.bus.parked_checkpoint(cmd.run)
    version = None
    if cmd.op == Op.LOAD_WEIGHTS:
      version = (await self.registry.get_checkpoint(json.loads(cmd.body)["uri"]) or {}).get("version")
    saved = (parked, (await self.registry.get_checkpoint(parked) or {}).get("version")) if parked else None
    reply, effects = await r.model.call("execute", cmd, run, saved, version)
    await self.owned(worker)
    if checkpoint := effects.get("saved"):
      await self.registry.register_checkpoint(**checkpoint)
    if restored := effects.get("restored"):
      if await self.bus.set_parked_checkpoint(cmd.run, None, worker, r.owner):
        await self.registry.unregister_checkpoints(restored)
    return reply

  async def step(self, worker: str, count: int = 16, block_ms: int = 1000) -> int:
    r, w = self.running[worker], self.state.workers[worker]
    if r.stopping and not r.queue:
      return 0

    async def take(block):
      return await self.bus.take_commands(scheduler.members(self.state, worker), r.owner, count, block)

    if not r.queue:
      enqueue(r.queue, await take(min(block_ms, IDLE_MS) if r.awake else block_ms), w.kind.lora)
    if not r.queue:
      self.note(Ev.IDLE, worker=worker)
      self.schedule()
      await self.release(worker)
      return 0
    runs, staged = {}, {}

    async def stage_new():
      for run in dict.fromkeys(c.run for c in r.queue):
        if run not in runs:
          runs[run] = await self.bus.get_run(run)

      async def stage(uri):
        try:
          version = (await self.registry.get_checkpoint(uri) or {}).get("version")
          staged[uri] = await asyncio.to_thread(self.store.local_folder, uri, version)
        except Exception as e:
          staged[uri] = e

      await asyncio.gather(
        *(stage(uri) for uri in {x["model_path"] for x in runs.values() if x and x.get("model_path")} - staged.keys())
      )

    if r.sampler:
      await stage_new()
    if not await self.acquire(worker):
      return 0
    start, processed = time.time(), 0
    if r.sampler and r.sampler.running is None:
      with contextlib.suppress(Exception):
        boot = await r.sampler.start()
        self.note(Ev.BATCH_DONE, worker=worker, boot=boot)
    if r.sampler and block_ms and not r.stopping:
      await asyncio.sleep(0.02)
      r.queue.extend(await take(0))
      await stage_new()

    async def one(cmd):
      before = time.time()
      reply = await self.answer_command(worker, cmd, runs, staged)
      if not await self.bus.write_reply(cmd, reply, worker, r.owner, started=before):
        r.stopping = r.lost = True
        raise RuntimeError(f"lost worker {worker}")
      if cmd.op == Op.DELETE_MODEL and reply.ok:
        await self.bus.delete_run(cmd.run, worker)
        self.note(Ev.RUN_DELETED, run=cmd.run, worker=worker)
        await self.registry.unregister_checkpoints(*await self.registry.checkpoints_owned_by_run(cmd.run))
        if not w.kind.lora:
          r.stopping = True
      ms = (time.time() - before) * 1000
      if DEBUG:
        print(f"{worker} {cmd.request_id} {cmd.op:24s} {ms:9.2f} ms" + ("" if reply.ok else f"  {reply.body.decode()}"))

    while r.queue and not w.revoked and not r.lost:
      if r.sampler:
        sessions = list(dict.fromkeys(c.run for c in r.queue))[: 3 if w.kind.lora else 1]
        batch = [c for c in r.queue if c.run in sessions]
        r.queue = collections.deque(c for c in r.queue if c.run not in sessions)
        async with asyncio.TaskGroup() as group:
          tasks = [group.create_task(one(c)) for c in batch]
        processed += len(tasks)
      else:
        cmd = r.queue.popleft()
        await one(cmd)
        processed += 1
        if w.kind.lora and r.queue and not r.stopping and (cmd.op == Op.OPTIM_STEP or r.queue[0].run != cmd.run):
          if cmd.op == Op.OPTIM_STEP:
            await asyncio.sleep(0.015)
          enqueue(r.queue, await take(0), True)
      self.schedule()
    self.note(Ev.BATCH_DONE, worker=worker, queued=len(r.queue), seconds=time.time() - start)
    await self.release(worker)
    return processed

  async def park_runs(self, worker: str) -> None:
    r = self.running[worker]
    if r.model:
      if len(r.gpus) > 1 and not r.awake:
        await self.acquire(worker, drain=True)  # FSDP checkpoint gathers need its CUDA/NCCL group
      for saved in await r.model.call("park_all"):
        await self.owned(worker)
        run = saved.pop("run")
        await self.registry.register_checkpoint(**saved)
        if not await self.bus.set_parked_checkpoint(run, saved["uri"], worker, r.owner):
          r.lost = True
          break

  async def close_worker(self, worker: str) -> None:
    r = self.running[worker]
    if r.closed:
      return
    closing = asyncio.create_task(r.model.close() if r.model else r.sampler.close())
    try:
      await asyncio.shield(closing)
    except asyncio.CancelledError:
      await closing
    r.closed = True
    self.note(Ev.EXITED, worker=worker)  # only the completed reap releases the actual GPUs
    if not r.lost:
      await self.bus.fail_stranded_commands(scheduler.members(self.state, worker), worker, r.owner, 0)
    await self.bus.release_lease(worker, r.owner)

  async def serve_worker(self, worker: str) -> None:
    r = self.running[worker]
    try:
      await self.bus.fail_stranded_commands(scheduler.members(self.state, worker), worker, r.owner, 0)
      if r.sampler and r.sampler.env.get("TINYRL_WARM_SAMPLER", "0") != "0" and await self.acquire(worker):
        with contextlib.suppress(Exception):
          boot = await r.sampler.start()
          self.note(Ev.BATCH_DONE, worker=worker, seconds=boot, boot=boot)
      while not r.stopping or r.queue:
        await self.step(worker, count=128 if r.sampler else 16)
        if not r.queue:
          await self.bus.fail_stranded_commands(scheduler.members(self.state, worker), worker, r.owner, STRANDED_MS)
      if not r.lost:
        await self.park_runs(worker)
    finally:
      await self.close_worker(worker)

  async def tick(self) -> None:
    workers = {
      n: dataclasses.asdict(self.state.workers[n]) | {"pid": r.proc.pid if r.proc else None}
      for n, r in self.running.items()
      if n in self.state.workers
    }
    await self.bus.node_heartbeat(self.id, self.gpus, workers)
    if await self.bus.event_log_size() < self.seen:
      for r in self.running.values():
        r.cancel()
      self.state, self.after, self.seen = scheduler.State(), "0-0", 0
    while events := await self.bus.read_events(self.after):
      for e in events:
        scheduler.fold(self.state, e)
      self.after, self.seen = events[-1].id, self.seen + len(events)
    await self.renew()
    self.schedule()
    mine = {n: w for n, w in self.state.workers.items() if w.node == self.id and scheduler.members(self.state, n)}
    for worker, r in list(self.running.items()):
      w = mine.get(worker)
      if self.id not in self.state.nodes:
        r.cancel()
      if not w or (r.gpus != w.gpus and not movable(w, self.gpus)):
        r.stopping = True
      if r.proc and r.proc.poll() is not None and r.task and not r.task.done() and not r.task.cancelling():
        r.task.cancel()
      if r.task and r.task.done():
        if not r.task.cancelled() and (error := r.task.exception()):
          print(f"{worker} stopped: {error!r}", file=sys.stderr, flush=True)
        if r.closed:
          del self.running[worker]
        else:
          r.task = asyncio.create_task(self.close_worker(worker))
    for worker in sorted(set(mine) - set(self.running)):
      if r := await self.open_worker(worker):
        r.task = asyncio.create_task(self.serve_worker(worker))
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

  async def serve(self) -> None:
    await self.announce_node()
    try:
      while True:
        await self.tick()
        await asyncio.sleep(TICK)
    finally:
      for r in self.running.values():
        r.stopping = True
      deadline = time.monotonic() + DRAIN_S
      while any(r.task and not r.task.done() for r in self.running.values()) and time.monotonic() < deadline:
        await self.renew()
        await asyncio.sleep(TICK)
      tasks = [r.task for r in self.running.values() if r.task]
      for task in tasks:
        if not task.done() and not task.cancelling():
          task.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)
      await asyncio.to_thread(self.store.upload_pending_saves)


if __name__ == "__main__":

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await NodeDaemon(Bus()).serve()

  asyncio.run(main())
