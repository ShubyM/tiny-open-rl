"""Stateless node daemon: executes controller `Task`s over one full-duplex msgpack WebSocket."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
import uuid

import msgspec

from tinyrl.control.accounting import Charge, Wave
from tinyrl.control.process import Process
from tinyrl.helpers import DEBUG, LEASE_MS, SWEEP_S, getenv
from tinyrl.storage.store import Store
from tinyrl.trace import span
from tinyrl.types import ControlMsg, Done, FencedError, NodeMsg, Reply, Role, Task, TaskKind, WorkerInfo, WorkerSpec
from tinyrl.workers.sampler import Sampler

GPUS = getenv("GPUS", 1)
DRAIN_S = getenv("DRAIN_S", 20.0)
TICK = getenv("TICK", 0.25)
SWEEP_GRACE_S = 600
DONE_FRAME_BYTES = getenv("DONE_FRAME_BYTES", 8 * 2**20)
CLOSE_RETRIES = 5

encode_msg = msgspec.msgpack.encode
decode_msg = msgspec.msgpack.decode


def gpu_memory_mib() -> int:
  override = getenv("GPU_MEMORY_MIB", 0)
  if override:
    return int(override)
  with contextlib.suppress(OSError, subprocess.SubprocessError, ValueError):
    cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"]
    return int(subprocess.check_output(cmd, timeout=2, stderr=subprocess.DEVNULL))
  return 0


class WorkerHandle:
  def __init__(
    self,
    worker_id: str,
    spec: WorkerSpec,
    model,
    resident: tuple[int, ...] = (),
    bound: tuple[int, ...] = (),
  ):
    self.id = worker_id
    self.spec = spec
    self.model = model
    self.bound = tuple(bound)
    self.resident = tuple(resident)
    self.loaded: dict[str, bool] = {}
    self.wave = Wave(spec.gpus)
    self.lock = asyncio.Lock()
    self.broken = False

  def exited(self) -> bool:
    proc = getattr(self.model, "proc", None)
    return proc is not None and proc.poll() is not None

  def pid(self) -> int | None:
    proc = getattr(self.model, "proc", None)
    if proc is None or proc.poll() is not None:
      return None
    return proc.pid

  def info(self) -> WorkerInfo:
    return WorkerInfo(
      worker=self.id,
      spec=self.spec,
      gpus=self.resident,
      bound=self.bound,
      loaded=dict(self.loaded),
      pid=self.pid(),
    )

  async def ensure_gpu(self, gpus: tuple[int, ...]) -> float:
    if self.resident == gpus:
      return 0.0
    async with self.lock:
      if self.broken:
        raise RuntimeError("worker broken")
      if self.resident == gpus:
        return 0.0
      started = time.monotonic()
      pinned_trainer = self.spec.role == Role.TRAINER and not self.spec.movable
      await self.model.resume(() if pinned_trainer else gpus)
      self.resident = gpus
      if not self.bound and not self.spec.movable:
        self.bound = gpus
      return time.monotonic() - started

  async def park(self) -> float:
    async with self.lock:
      if self.exited():
        raise RuntimeError("worker exited before park")
      started = time.monotonic()
      await self.model.park()
      self.resident = ()
      return time.monotonic() - started

  async def close(self) -> None:
    try:
      for attempt in range(CLOSE_RETRIES):
        try:
          await self.model.close()
          return
        except Exception as exc:
          print(f"{self.id} could not close ({attempt + 1}/{CLOSE_RETRIES}): {exc!r}", file=sys.stderr, flush=True)
          if attempt + 1 < CLOSE_RETRIES:
            await asyncio.sleep(min(0.05, TICK))
    finally:
      self.resident = ()


class NodeDaemon:
  def __init__(
    self,
    control,
    gpus: int = GPUS,
    node: str = "",
    gpu_mib: int | None = None,
    store: Store | None = None,
  ):
    self.control = control
    self.store = store if store is not None else Store()
    self.store.start()
    self.gpus = gpus
    self.id = node or f"{socket.gethostname()}-{os.getpid()}"
    self.boot_id = uuid.uuid4().hex[:12]
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib

    self.workers: dict[str, WorkerHandle] = {}
    self.active: dict[str, tuple[Task, asyncio.Task]] = {}
    self.done_buf: dict[str, Done] = {}
    self.sent_done: set[str] = set()
    self.staged: dict[str, str] = {}
    self.stage_dirty = False
    self.staging: dict[str, asyncio.Task] = {}
    self.stage_failures: dict[str, int] = {}

    self.changed = asyncio.Event()
    self.draining = False
    self.connected = False
    self.sweep_task: asyncio.Task | None = None
    self.last_ok = time.monotonic()
    self.lag = 0.0
    self.swept = 0.0

  def notify(self) -> None:
    self.changed.set()

  def begin_drain(self) -> None:
    self.draining = True
    self.notify()

  def busy(self) -> bool:
    return bool(self.active or self.done_buf or self.staging or self.sweep_task)

  def lease_expired(self) -> bool:
    return not self.draining and time.monotonic() - self.last_ok >= LEASE_MS / 1000

  def drain_deadline(self, deadline: float | None) -> float | None:
    if not self.draining:
      return None
    return time.monotonic() + DRAIN_S if deadline is None else deadline

  def is_drained(self, deadline: float | None) -> bool:
    if deadline is None:
      return False
    quiet = not self.workers and not self.active and not self.done_buf
    return quiet or time.monotonic() >= deadline

  def open_trainer(self, worker_id: str, width: int, env: dict[str, str]) -> Process:
    launcher = ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={width}"] if width > 1 else []
    return Process([sys.executable, *launcher, "-m", "tinyrl.workers.trainer"], env)

  def open_sampler(self, worker_id: str, spec: WorkerSpec, env: dict[str, str]) -> Sampler:
    return Sampler(spec.base, spec.lora, env, store=self.store)

  def spawn(self, worker_id: str, spec: WorkerSpec, gpus: tuple[int, ...]) -> WorkerHandle:
    visible = range(self.gpus) if spec.movable else gpus
    env = {
      **os.environ,
      **self.store.env(),
      "CUDA_VISIBLE_DEVICES": ",".join(map(str, visible)),
    }
    if spec.role == Role.SAMPLER and self.gpu_mib >= 64 * 1024:
      env.setdefault("TINYRL_KV_BYTES", str(24 * 2**30))
    elif spec.role == Role.TRAINER:
      env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if spec.role == Role.SAMPLER:
      model = self.open_sampler(worker_id, spec, env)
    else:
      model = self.open_trainer(worker_id, spec.gpus, env)
    bound = () if spec.movable else gpus
    return WorkerHandle(worker_id, spec, model, bound=bound)

  async def drop_worker(self, worker_id: str, worker: WorkerHandle) -> None:
    worker.broken = True
    await worker.close()
    if self.workers.get(worker_id) is worker:
      del self.workers[worker_id]

  async def stage_version(self, version: str) -> str:
    cached = self.staged.get(version)
    if cached and os.path.isdir(cached):
      return cached
    folder, _ = await asyncio.to_thread(self.store.local_folder, version)
    self.staged[version] = folder
    self.stage_dirty = True
    return folder

  async def execute_task(self, task: Task) -> Done:
    kind = task.kind
    worker = self.workers.get(task.worker)
    try:
      if worker is None and kind in (TaskKind.TRAIN, TaskKind.SAMPLE):
        worker = self.spawn(task.worker, task.spec, task.gpus)
        self.workers[task.worker] = worker
      if worker is None:
        return Done(task.id, dead=True)

      with span(f"node.{kind}"):
        if kind == TaskKind.PARK:
          return Done(task.id, duration_s=await worker.park(), pid=worker.pid())

        if kind in (TaskKind.STOP, TaskKind.DROP, TaskKind.SAVE):
          async with worker.lock:
            if kind == TaskKind.STOP or (kind == TaskKind.SAVE and worker.exited()):
              await self.drop_worker(task.worker, worker)
              return Done(task.id, ok=(kind == TaskKind.STOP), dead=True)
            if kind == TaskKind.DROP:
              if not worker.exited() and task.run_id in worker.loaded:
                await worker.model.call("drop_run", task.run_id)
              worker.loaded.pop(task.run_id, None)
              return Done(task.id, pid=worker.pid())
            try:
              saved = await worker.model.call("save_run", task.run_id, task.version)
              return Done(task.id, saved=saved, pid=worker.pid())
            except Exception:
              if worker.exited() or getattr(worker.model, "broken", False):
                raise
              return Done(task.id, ok=False, pid=worker.pid())

        for other in self.workers.values():
          if other is not worker and not set(other.resident).isdisjoint(task.gpus):
            raise RuntimeError(f"GPU collision on {task.gpus}")

        resume_s = await worker.ensure_gpu(task.gpus)
        charge = Charge()
        cmd = task.cmd

        if kind == TaskKind.SAMPLE:
          checkpoint = None
          if task.version:
            try:
              folder = await self.stage_version(task.version)
            except Exception:
              self.staged.pop(task.version, None)
              self.stage_dirty = True
              return Done(task.id, ok=False, reply=None, dead=False, pid=worker.pid())
            checkpoint = (folder, task.version.partition("@")[0])
          reply = await worker.model.execute(cmd, checkpoint, wave=worker.wave, charge=charge)
          if not reply.ok and worker.exited():
            await self.drop_worker(task.worker, worker)
            return Done(task.id, ok=False, reply=reply, charge=charge, dead=True)
          return Done(task.id, ok=True, reply=reply, charge=charge, duration_s=resume_s, pid=worker.pid())

        async with worker.lock:
          call = worker.model.call("execute", cmd, task.run, task.snapshot, task.version)
          reply, effect = await worker.wave.run(call, charge)
          worker.loaded.pop(cmd.run, None)
          if effect.loaded:
            worker.loaded[cmd.run] = effect.open_grad
          if DEBUG:
            print(f"{worker.id} {cmd.request_id} {cmd.op:24s} {charge.busy_ms:9d} ms")
          return Done(
            task.id, ok=True, reply=reply, charge=charge, effect=effect, duration_s=resume_s, pid=worker.pid()
          )
    except asyncio.CancelledError:
      if worker is not None and kind in (TaskKind.TRAIN, TaskKind.PARK, TaskKind.SAVE, TaskKind.STOP):
        await asyncio.shield(self.drop_worker(task.worker, worker))
      raise
    except Exception as exc:
      if worker is not None:
        await self.drop_worker(task.worker, worker)
      err_reply = Reply.error(exc) if task.cmd else None
      return Done(task.id, ok=False, reply=err_reply, dead=True)

  def start_background(self, registry: dict, key: str, coro, *, wrap=None) -> None:
    async def runner() -> None:
      try:
        await coro
      finally:
        registry.pop(key, None)
        self.notify()

    bg_task = asyncio.create_task(runner())
    registry[key] = wrap(bg_task) if wrap else bg_task

  async def record_done(self, task: Task) -> None:
    self.done_buf[task.id] = await self.execute_task(task)

  async def prefetch_version(self, version: str) -> None:
    try:
      await self.stage_version(version)
      self.stage_failures.pop(version, None)
    except Exception:
      fails = self.stage_failures[version] = self.stage_failures.get(version, 0) + 1
      await asyncio.sleep(min(10.0, 0.05 * (2 ** min(fails - 1, 8))))
      self.stage_dirty = True

  async def run_sweep(self, keep: set[str]) -> None:
    try:
      present = await asyncio.to_thread(self.store.versions)
      await asyncio.to_thread(self.store.sweep, present - keep, SWEEP_GRACE_S)
    finally:
      self.sweep_task = None
      self.notify()

  async def build_msg(self, hello: bool = False) -> NodeMsg:
    busy_workers = {task.worker for task, _ in self.active.values()}
    exited = tuple(
      wid
      for wid, worker in list(self.workers.items())
      if (worker.exited() or worker.broken) and wid not in busy_workers
    )
    for wid in exited:
      await self.drop_worker(wid, self.workers[wid])

    lost_stage = False
    for version, folder in list(self.staged.items()):
      if not os.path.isdir(folder):
        del self.staged[version]
        lost_stage = True
        self.stage_dirty = True

    want_sweep = not self.draining and self.sweep_task is None and time.time() - self.swept > SWEEP_S
    if want_sweep:
      self.swept = time.time()

    if hello:
      self.sent_done.clear()
    done_items, frame_bytes = [], 0
    for tid, item in self.done_buf.items():
      if tid in self.sent_done:
        continue
      size = (len(item.reply.body) if item.reply else 0) + 512
      if done_items and frame_bytes + size > DONE_FRAME_BYTES:
        self.notify()
        break
      done_items.append(item)
      frame_bytes += size
    done = tuple(done_items)
    self.sent_done.update(d.id for d in done)

    staged = tuple(self.staged) if (hello or self.stage_dirty) else None
    self.stage_dirty = False
    return NodeMsg(
      "hello" if hello else "events",
      self.id,
      self.boot_id,
      self.gpus,
      self.gpu_mib,
      self.draining,
      self.lag,
      want_sweep,
      tuple(w.info() for w in self.workers.values()) if hello else (),
      staged,
      exited,
      tuple(self.active) if (hello or self.draining or lost_stage) else (),
      done,
    )

  def apply(self, msg: ControlMsg) -> None:
    self.last_ok = time.monotonic()
    if msg.reset:
      raise FencedError("controller reset node incarnation")
    for task_id in msg.ack:
      self.done_buf.pop(task_id, None)
      self.sent_done.discard(task_id)
    cancelled = set(msg.cancel)
    for task_id in msg.cancel:
      running = self.active.get(task_id)
      if running and running[0].kind == TaskKind.SAMPLE:
        running[1].cancel()
      self.done_buf.pop(task_id, None)
      self.sent_done.discard(task_id)
    for version in msg.stage:
      if version not in self.staged and version not in self.staging:
        self.start_background(self.staging, version, self.prefetch_version(version))
    if msg.sweep is not None and self.sweep_task is None:
      self.sweep_task = asyncio.create_task(self.run_sweep(set(msg.sweep)))
    for task in msg.tasks:
      if task.id not in cancelled and task.id not in self.active and task.id not in self.done_buf:
        self.start_background(self.active, task.id, self.record_done(task), wrap=lambda bg, t=task: (t, bg))

  async def step_once(self) -> bool:
    if self.lease_expired():
      await self.reset()
    msg = await self.build_msg(hello=not self.connected)
    sent_at = time.monotonic()
    try:
      reply = self.control.handle(msg)
      if asyncio.iscoroutine(reply):
        reply = await reply
      self.lag = max(0.0, time.monotonic() - sent_at)
      self.connected = True
      self.apply(reply)
      return True
    except FencedError:
      await (self.clear() if self.draining else self.reset())
      return False
    except Exception:
      self.connected = False
      if self.lease_expired():
        await self.reset()
      return False

  async def wait_for_change(self, delay: float) -> None:
    with contextlib.suppress(TimeoutError):
      await asyncio.wait_for(self.changed.wait(), max(0.0, delay))

  async def serve_socket(self, ws, heartbeat_s: float, deadline: float | None) -> float | None:
    from aiohttp import WSMsgType

    close_types = (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR)
    sent_at = time.monotonic()
    self.connected = True
    self.changed.clear()
    await ws.send_bytes(encode_msg(await self.build_msg(hello=True)))

    async def reader() -> None:
      nonlocal sent_at
      async for frame in ws:
        if frame.type in close_types:
          return
        if frame.type != WSMsgType.BINARY:
          continue
        if sent_at:
          self.lag = max(0.0, time.monotonic() - sent_at)
          sent_at = 0.0
        self.apply(decode_msg(frame.data, type=ControlMsg))
        if self.draining:
          self.notify()

    async def writer() -> None:
      nonlocal sent_at, deadline
      while not ws.closed:
        deadline = self.drain_deadline(deadline)
        if not self.is_drained(deadline):
          wait_s = heartbeat_s if deadline is None else min(heartbeat_s, deadline - time.monotonic())
          await self.wait_for_change(wait_s)
        deadline = self.drain_deadline(deadline)
        if self.is_drained(deadline):
          await ws.close()
          return
        self.changed.clear()
        msg = await self.build_msg(hello=False)
        sent_at = time.monotonic()
        await ws.send_bytes(encode_msg(msg))

    async def lease_watch() -> None:
      while not self.lease_expired():
        remaining = LEASE_MS / 1000 - (time.monotonic() - self.last_ok)
        await asyncio.sleep(heartbeat_s if self.draining else min(heartbeat_s, max(0.02, remaining)))
      raise FencedError("controller lease expired")

    tasks = {asyncio.create_task(reader()), asyncio.create_task(writer()), asyncio.create_task(lease_watch())}
    try:
      done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
      for task in tasks:
        if not task.done():
          task.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)

    errors = [exc for task in done if not task.cancelled() and (exc := task.exception()) is not None]
    if errors:
      fenced = next((exc for exc in errors if isinstance(exc, FencedError)), None)
      raise fenced or errors[0]
    return deadline

  async def serve(self) -> None:
    deadline: float | None = None
    heartbeat_s = min(TICK, max(0.05, LEASE_MS / 4000))
    try:
      while True:
        deadline = self.drain_deadline(deadline)
        if self.is_drained(deadline):
          break
        if not hasattr(self.control, "connect"):
          self.changed.clear()
          ok = await self.step_once()
          if not self.is_drained(deadline) and not self.changed.is_set():
            delay = min(0.05, TICK) if (not ok or self.busy() or self.draining) else TICK
            await self.wait_for_change(delay)
          continue
        if self.lease_expired():
          await self.reset()
        try:
          async with self.control.connect() as ws:
            deadline = await self.serve_socket(ws, heartbeat_s, deadline)
        except FencedError:
          if self.draining:
            await self.clear()
            break
          await self.reset()
        except Exception:
          self.connected = False
          await asyncio.sleep(min(0.05, TICK))
    finally:
      await self.close()

  async def clear(self) -> None:
    tasks = [bg for _, bg in self.active.values()]
    tasks.extend(self.staging.values())
    if self.sweep_task is not None:
      tasks.append(self.sweep_task)
    self.active.clear()
    self.staging.clear()
    self.stage_failures.clear()
    self.done_buf.clear()
    self.sent_done.clear()
    self.sweep_task = None

    for task in tasks:
      task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    for wid, worker in list(self.workers.items()):
      await self.drop_worker(wid, worker)

  async def reset(self) -> None:
    self.staged.clear()
    await self.clear()
    self.connected = False
    self.boot_id = uuid.uuid4().hex[:12]
    self.last_ok = time.monotonic()
    self.notify()

  async def close(self) -> None:
    self.draining = True
    await self.clear()
    await asyncio.to_thread(self.store.close)


class ControlClient:
  def __init__(self, url: str | None = None):
    raw = getenv("CONTROL_URL", "http://127.0.0.1:8000") if url is None else url
    base = raw.rstrip("/")
    ws_base = "ws" + base[4:] if base.startswith("http") else base
    self.ws_url = f"{ws_base}/internal/ws"

  @contextlib.asynccontextmanager
  async def connect(self):
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=None)
    async with (
      aiohttp.ClientSession(timeout=timeout) as session,
      session.ws_connect(self.ws_url, max_msg_size=0, heartbeat=None) as ws,
    ):
      yield ws

  async def aclose(self) -> None:
    pass


if __name__ == "__main__":
  daemon = NodeDaemon(ControlClient())

  async def main() -> None:
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, daemon.begin_drain)
    with contextlib.suppress(asyncio.CancelledError):
      await daemon.serve()

  asyncio.run(main())
