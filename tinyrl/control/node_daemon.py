"""The node owns leases, GPU turns and replies. Model processes only execute local commands."""

from __future__ import annotations

import asyncio
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

from tinyrl.bus import LEASE_MS, Bus, Cmd, Ev, Event, Op, Reply, Role
from tinyrl.control import scheduler
from tinyrl.control.process import Process
from tinyrl.helpers import DEBUG, getenv
from tinyrl.storage.registry import Registry
from tinyrl.storage.store import Store
from tinyrl.workers.sampler import Sampler

GPUS, DRAIN_S, TICK = getenv("GPUS", 1), getenv("DRAIN_S", 20.0), getenv("TICK", 0.25)
IDLE_MS, SAVE_WAIT_S, SWEEP_S, SWEEP_GRACE_S = getenv("IDLE_MS", 150), 0.015, getenv("SWEEP_S", 60.0), 600


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


def choose(heads: list[tuple[str, Op]], width: int) -> list[str]:
  return [run for run, op in sorted(heads, key=lambda head: head[1] != Op.SAVE_WEIGHTS_FOR_SAMPLER)][:width]


class Lease:
  def __init__(self, bus: Bus, worker: str, owner: str):
    self.bus, self.worker, self.owner, self.renewed, self.lost = bus, worker, owner, time.monotonic(), False

  @classmethod
  async def take(cls, bus: Bus, worker: str, owner: str) -> Lease | None:
    return cls(bus, worker, owner) if await bus.take_lease(worker, owner) else None

  def lose(self) -> RuntimeError:
    self.lost = True
    return RuntimeError(f"lost worker {self.worker}")

  async def hold(self) -> None:
    before = time.monotonic()
    if self.lost or not (
      await self.bus.renew_lease(self.worker, self.owner)
      or (not self.lost and await self.bus.take_lease(self.worker, self.owner, recover=True))
    ):
      raise self.lose()
    self.renewed = before

  async def keep(self) -> bool:
    if time.monotonic() - self.renewed >= LEASE_MS / 4000:
      with contextlib.suppress(Exception):
        await self.hold()
    return not self.lost and time.monotonic() - self.renewed < LEASE_MS / 1000

  async def reply(self, cmd: Cmd, reply: Reply, started: float) -> None:
    if not await self.bus.write_reply(cmd, reply, self.worker, self.owner, started=started):
      raise self.lose()

  async def park(self, run: str, uri: str | None) -> None:
    if not await self.bus.set_parked_checkpoint(run, uri, self.worker, self.owner):
      raise self.lose()

  async def fail_stranded(self, runs: list[str], died: bool = False) -> None:
    if not self.lost:
      if died:  # a worker that died before claiming anything still answers each run's waiting client
        await self.bus.take_commands(runs, self.owner, 1)
      await self.bus.fail_stranded_commands(runs, self.worker, self.owner)

  async def release(self) -> None:
    self.lost = True
    await self.bus.release_lease(self.worker, self.owner)


class TrainerEngine:
  batch, top_up, warm, started = (1, 1), 0, False, True

  def __init__(self, process: Process, lease: Lease, registry: Registry):
    self.process, self.proc, self.lease, self.registry, self.awake = process, process.proc, lease, registry, False

  async def start(self) -> float:
    return 0.0

  async def resume(self, gpu: int | None) -> None:
    if not self.awake:
      await self.process.call("resume", gpu)
      self.awake = True

  async def park(self) -> None:
    if self.awake:
      await self.process.call("park")
      self.awake = False

  async def execute(self, cmd: Cmd, run: dict | None, staged: dict) -> Reply:
    version = None
    if cmd.op == Op.LOAD_WEIGHTS:
      version = (await self.registry.get_checkpoint(json.loads(cmd.body)["uri"]) or {}).get("version")
    if parked := await self.lease.bus.parked_checkpoint(cmd.run):
      parked = (parked, (await self.registry.get_checkpoint(parked) or {}).get("version"))
    reply, effects = await self.process.call("execute", cmd, run, parked, version)
    await self.lease.hold()
    if checkpoint := effects.get("saved"):
      await self.registry.register_checkpoint(**checkpoint)
    if restored := effects.get("restored"):
      await self.lease.park(cmd.run, None)
      await self.registry.unregister_checkpoints(restored)
    return reply

  async def park_runs(self) -> None:
    await self.lease.hold()
    for saved in await self.process.call("park_all"):
      await self.lease.hold()
      run = saved.pop("run")
      await self.registry.register_checkpoint(**saved)
      await self.lease.park(run, saved["uri"])

  async def close(self) -> None:
    await self.process.close()


class LocalWorker:
  def __init__(self, engine: TrainerEngine | Sampler, lease: Lease, serve):
    self.engine, self.lease, self.name, self.swept = engine, lease, lease.worker, 0.0
    self.task: asyncio.Task = asyncio.create_task(serve(self))

  def cancel(self, lost: bool = False) -> None:
    self.lease.lost |= lost
    if not self.task.cancelling():
      self.task.cancel()


class NodeDaemon:
  def __init__(
    self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store: Store | None = None
  ):
    self.bus, self.registry = bus, Registry(bus)
    self.store = store if store is not None else Store()
    self.gpus, self.id = gpus, node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.local: dict[str, LocalWorker] = {}
    self.state, self.after, self.seen, self.stopping = scheduler.State(), "0-0", 0, False
    self.swept, self.uploading, self.sweeping, self.decided = 0.0, None, None, asyncio.Event()

  def note(self, kind: Ev, **body) -> None:
    scheduler.fold(self.state, Event("", time.time(), kind, body))

  def schedule(self) -> None:
    if decisions := scheduler.local_decide(self.state, self.id, time.time()):
      for kind, body in decisions:
        self.note(kind, **body)
      self.decided.set()
      self.decided = asyncio.Event()

  def wanted(self, worker: str) -> bool:
    w = self.state.workers.get(worker)
    here = not self.stopping and w is not None and w.node == self.id and scheduler.members(self.state, worker)
    return bool(here) and (scheduler.placed(w) or movable(w, self.gpus))

  async def announce_node(self) -> None:
    await self.bus.node_heartbeat(self.id, self.gpus)
    await self.bus.append_event(Ev.NODE_UP, node=self.id, gpus=self.gpus, memory=self.gpu_mib * 2**20)

  async def renew(self) -> None:
    for lw in list(self.local.values()):
      if not lw.task.done() and not await lw.lease.keep():
        lw.cancel(lost=True)

  def engine(self, w: scheduler.Worker, lease: Lease) -> TrainerEngine | Sampler:
    env = {**scheduler.settings(w.kind, self.gpus, self.gpu_mib * 2**20), **os.environ}
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, w.gpus))
    env["CKPT"], env["CKPT_CACHE"] = self.store.root, self.store.cache
    if w.kind.role == Role.SAMPLER:
      return Sampler(w.kind.base, w.kind.lora, env)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if movable(w, self.gpus):
      env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(self.gpus)))
    cmd = [sys.executable]
    if len(w.gpus) > 1:
      cmd += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={len(w.gpus)}"]
    return TrainerEngine(Process([*cmd, "-m", "tinyrl.workers.trainer"], env), lease, self.registry)

  async def open(self, worker: str) -> LocalWorker | None:
    if not (lease := await Lease.take(self.bus, worker, f"{self.id}-{os.urandom(8).hex()}")):
      return None
    try:
      engine = self.engine(self.state.workers[worker], lease)
    except BaseException:
      await lease.release()
      raise
    self.note(Ev.STARTED, worker=worker, node=self.id)
    lw = LocalWorker(engine, lease, self.serve_worker)
    await asyncio.sleep(0)  # its task enters the try whose finally closes it before anyone can cancel it
    return lw

  async def acquire(self, lw: LocalWorker, leaving: bool = False) -> bool:
    while not lw.lease.lost and (leaving or self.wanted(lw.name)):
      decided, w = self.decided, self.state.workers.get(lw.name)
      if not w or not w.on:
        return False
      if not leaving and w.since is None and not scheduler.placed(w):
        self.note(Ev.STARTED, worker=lw.name, node=self.id)
      self.note(Ev.WANTS_GPU, worker=lw.name, booting=not lw.engine.started)
      self.schedule()
      if w.since is not None and not w.revoked:
        await lw.lease.hold()
        if not lw.engine.started:
          with contextlib.suppress(Exception):
            self.note(Ev.BATCH_DONE, worker=lw.name, boot=await lw.engine.start())
        await lw.engine.resume(w.on[0][1] if movable(w, self.gpus) else None)
        if not w.revoked:
          return True
      await self.release(lw)
      with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(decided.wait(), TICK)
    return False

  async def release(self, lw: LocalWorker) -> None:
    if (w := self.state.workers.get(lw.name)) and w.revoked:
      await lw.engine.park()
      self.note(Ev.RELEASED, worker=lw.name)
      self.schedule()

  async def stage(self, runs: list[str], records: dict, staged: dict) -> None:
    new = [r for r in runs if r not in records]
    records.update(zip(new, await asyncio.gather(*map(self.bus.get_run, new))))

    async def one(uri):
      try:
        version = (await self.registry.get_checkpoint(uri) or {}).get("version")
        staged[uri] = await asyncio.to_thread(self.store.local_folder, uri, version)
      except Exception as e:
        staged[uri] = e

    uris = {x["model_path"] for x in records.values() if x and x.get("model_path")} - staged.keys()
    await asyncio.gather(*map(one, uris))

  async def answer(self, lw: LocalWorker, cmd: Cmd, run: dict | None, staged: dict) -> None:
    before = time.time()
    reply = await lw.engine.execute(cmd, run, staged)
    await lw.lease.reply(cmd, reply, before)
    if cmd.op == Op.DELETE_MODEL and reply.ok:
      await self.bus.delete_run(cmd.run, lw.name)
      self.note(Ev.RUN_DELETED, run=cmd.run, worker=lw.name)
      await self.registry.unregister_checkpoints(*await self.registry.checkpoints_owned_by_run(cmd.run))
    ms = (time.time() - before) * 1000
    if DEBUG:
      print(f"{lw.name} {cmd.request_id} {cmd.op:24s} {ms:9.2f} ms" + ("" if reply.ok else f"  {reply.body.decode()}"))

  async def step(self, lw: LocalWorker) -> None:
    w, members, (width, count) = self.state.workers[lw.name], scheduler.members(self.state, lw.name), lw.engine.batch
    heads = await self.bus.peek_commands(members, 1000 if w.since is None else IDLE_MS)
    if not heads:
      self.note(Ev.IDLE, worker=lw.name)
      self.schedule()
      return await self.release(lw)
    records, staged, held = {}, {}, w.since is not None and not w.revoked
    await self.stage(choose(heads, width), records, staged)  # only what it is about to serve: a Redis read per run
    if not await self.acquire(lw):
      return
    start = time.time()
    if lw.engine.top_up or not held:  # a wave gathers for the top-up, and a turn waited for may have aged the heads
      await asyncio.sleep(lw.engine.top_up)
      heads = await self.bus.peek_commands(members)
    runs = choose(heads, width)
    await self.stage(runs, records, staged)
    cmds = await self.bus.take_commands(runs, lw.lease.owner, count)
    await lw.lease.hold()  # a resumed node checks ownership before sending anything to its child
    while cmds:
      if len(cmds) == 1:
        await self.answer(lw, cmds[0], records.get(cmds[0].run), staged)
      else:
        async with asyncio.TaskGroup() as group:
          for cmd in cmds:
            group.create_task(self.answer(lw, cmd, records.get(cmd.run), staged))
      if cmds[-1].op == Op.OPTIM_STEP and len(members) > width:
        await asyncio.sleep(SAVE_WAIT_S)
      if count > 1 or self.stopping or w.revoked or not scheduler.placed(w):
        break
      # a one-command engine serves its run's chain (forward_backward, optim_step, its sampler save) in this turn, each claimed only when it is next
      cmds = await self.bus.take_commands([cmds[0].run], lw.lease.owner, count) if await lw.lease.keep() else []
    self.note(Ev.BATCH_DONE, worker=lw.name, seconds=time.time() - start)
    self.schedule()
    await self.release(lw)

  async def close(self, lw: LocalWorker) -> None:
    died = lw.engine.proc is not None and lw.engine.proc.poll() is not None  # exited on its own, before the reap

    async def retire():
      while True:
        try:
          await lw.engine.close()
          break
        except Exception as e:
          print(f"{lw.name} did not close, trying again: {e!r}", file=sys.stderr, flush=True)
          await asyncio.sleep(TICK)
      self.note(Ev.EXITED, worker=lw.name)  # only the completed reap releases the actual GPUs
      await lw.lease.fail_stranded(scheduler.members(self.state, lw.name), died)
      await lw.lease.release()

    closing = asyncio.ensure_future(retire())
    try:
      await asyncio.shield(closing)
    except asyncio.CancelledError:
      await closing

  async def serve_worker(self, lw: LocalWorker) -> None:
    try:
      await lw.lease.fail_stranded(scheduler.members(self.state, lw.name))
      if lw.engine.warm:
        await self.acquire(lw)
      while self.wanted(lw.name) and not lw.lease.lost:
        await self.step(lw)
        if time.monotonic() - lw.swept > 0.5:  # a dead owner's claims fail within a second, not after every command
          lw.swept = time.monotonic()
          await lw.lease.fail_stranded(scheduler.members(self.state, lw.name))
      if (w := self.state.workers.get(lw.name)) and scheduler.parks_in_place(w):
        await self.acquire(lw, leaving=True)  # FSDP checkpoint gathers need its CUDA/NCCL group, resumed on it
      await lw.engine.park_runs()
    finally:
      await self.close(lw)

  async def tick(self, block_ms: int = 0) -> None:
    workers = {
      n: dataclasses.asdict(self.state.workers[n]) | {"pid": lw.engine.proc.pid if lw.engine.proc else None}
      for n, lw in self.local.items()
      if n in self.state.workers
    }
    await self.bus.node_heartbeat(self.id, self.gpus, workers)
    if await self.bus.event_log_size() < self.seen:
      for lw in self.local.values():
        lw.cancel(lost=True)
      self.state, self.after, self.seen = scheduler.State(), "0-0", 0
    while events := await self.bus.read_events(self.after, block_ms=block_ms):
      for e in events:
        scheduler.fold(self.state, e)
      self.after, self.seen, block_ms = events[-1].id, self.seen + len(events), 0
    await self.renew()
    self.schedule()
    for worker, lw in list(self.local.items()):
      if self.id not in self.state.nodes or (lw.engine.proc and lw.engine.proc.poll() is not None):
        lw.cancel(lost=self.id not in self.state.nodes)
      if lw.task.done():
        if not lw.task.cancelled() and (error := lw.task.exception()):
          print(f"{worker} stopped: {error!r}", file=sys.stderr, flush=True)
        del self.local[worker]
    for worker, w in sorted(self.state.workers.items()):
      if worker not in self.local and w.node == self.id and scheduler.members(self.state, worker):
        if lw := await self.open(worker):
          self.local[worker] = lw
    if self.id not in self.state.nodes and not self.local:
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
        await self.tick(int(TICK * 1000))
    finally:
      self.stopping = True
      deadline = time.monotonic() + DRAIN_S
      while any(not lw.task.done() for lw in self.local.values()) and time.monotonic() < deadline:
        await self.renew()
        await asyncio.sleep(TICK)
      for lw in self.local.values():
        lw.cancel()
      await asyncio.gather(*(lw.task for lw in self.local.values()), return_exceptions=True)
      await asyncio.to_thread(self.store.upload_pending_saves)


if __name__ == "__main__":

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await NodeDaemon(Bus()).serve()

  asyncio.run(main())
