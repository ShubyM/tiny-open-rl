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

GPUS, DRAIN_S, TICK = getenv("GPUS", 1), getenv("DRAIN_S", 20.0), getenv("TICK", 0.25)
SWEEP_GRACE_S, DONE_FRAME_BYTES, CLOSE_RETRIES = 600, getenv("DONE_FRAME_BYTES", 8 * 2**20), 5
encode_msg, decode_msg = msgspec.msgpack.encode, msgspec.msgpack.decode


def gpu_memory_mib() -> int:
  if override := getenv("GPU_MEMORY_MIB", 0):
    return int(override)
  with contextlib.suppress(OSError, subprocess.SubprocessError, ValueError):
    cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"]
    return int(subprocess.check_output(cmd, timeout=2, stderr=subprocess.DEVNULL))
  return 0


class WorkerHandle:
  def __init__(
    self, worker_id: str, spec: WorkerSpec, model, resident: tuple[int, ...] = (), bound: tuple[int, ...] = ()
  ):
    self.id, self.spec, self.model = worker_id, spec, model
    self.bound, self.resident = tuple(bound), tuple(resident)
    self.loaded: dict[str, bool] = {}
    self.wave, self.lock, self.broken = Wave(spec.gpus), asyncio.Lock(), False

  def exited(self) -> bool:
    return (proc := getattr(self.model, "proc", None)) is not None and proc.poll() is not None

  def pid(self) -> int | None:
    proc = getattr(self.model, "proc", None)
    return None if (proc is None or proc.poll() is not None) else proc.pid

  def info(self) -> WorkerInfo:
    return WorkerInfo(self.id, self.spec, self.resident, self.bound, dict(self.loaded), self.pid())

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
  def __init__(self, control, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store: Store | None = None):
    self.control = control
    self.store = Store() if store is None else store
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
    self.staging: dict[str, asyncio.Task] = {}
    self.stage_failures: dict[str, int] = {}
    self.stage_dirty = self.draining = self.connected = False
    self.changed = asyncio.Event()
    self.sweep_task: asyncio.Task | None = None
    self.last_ok, self.lag, self.swept = time.monotonic(), 0.0, 0.0

  def notify(self) -> None:
    self.changed.set()

  def begin_drain(self) -> None:
    self.draining = True
    self.changed.set()

  def busy(self) -> bool:
    return bool(self.active or self.done_buf or self.staging or self.sweep_task)

  def _lease_expired(self) -> bool:
    return not self.draining and time.monotonic() - self.last_ok >= LEASE_MS / 1000

  def _drained(self, deadline: float | None) -> bool:
    return deadline is not None and (
      (not self.workers and not self.active and not self.done_buf) or time.monotonic() >= deadline
    )

  def open_trainer(self, worker_id: str, width: int, env: dict[str, str]) -> Process:
    launcher = ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={width}"] if width > 1 else []
    return Process([sys.executable, *launcher, "-m", "tinyrl.workers.trainer"], env)

  def open_sampler(self, worker_id: str, spec: WorkerSpec, env: dict[str, str]) -> Sampler:
    return Sampler(spec.base, spec.lora, env, store=self.store)

  def spawn(self, worker_id: str, spec: WorkerSpec, gpus: tuple[int, ...]) -> WorkerHandle:
    visible = range(self.gpus) if spec.movable else gpus
    env = {**os.environ, **self.store.env(), "CUDA_VISIBLE_DEVICES": ",".join(map(str, visible))}
    if spec.role == Role.SAMPLER and self.gpu_mib >= 64 * 1024:
      env.setdefault("TINYRL_KV_BYTES", str(24 * 2**30))
    elif spec.role == Role.TRAINER:
      env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    model = (
      self.open_sampler(worker_id, spec, env)
      if spec.role == Role.SAMPLER
      else self.open_trainer(worker_id, spec.gpus, env)
    )
    return WorkerHandle(worker_id, spec, model, bound=() if spec.movable else gpus)

  async def drop_worker(self, worker_id: str, worker: WorkerHandle) -> None:
    worker.broken = True
    await worker.close()
    if self.workers.get(worker_id) is worker:
      del self.workers[worker_id]

  async def stage_version(self, version: str) -> str:
    if (cached := self.staged.get(version)) and os.path.isdir(cached):
      return cached
    folder, _ = await asyncio.to_thread(self.store.local_folder, version)
    self.staged[version], self.stage_dirty = folder, True
    return folder

  async def execute_task(self, task: Task) -> Done:
    kind, worker = task.kind, self.workers.get(task.worker)
    try:
      if worker is None and kind in (TaskKind.TRAIN, TaskKind.SAMPLE):
        worker = self.workers[task.worker] = self.spawn(task.worker, task.spec, task.gpus)
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
              return Done(
                task.id, saved=await worker.model.call("save_run", task.run_id, task.version), pid=worker.pid()
              )
            except Exception:
              if worker.exited() or getattr(worker.model, "broken", False):
                raise
              return Done(task.id, ok=False, pid=worker.pid())

        if any(o is not worker and not set(o.resident).isdisjoint(task.gpus) for o in self.workers.values()):
          raise RuntimeError(f"GPU collision on {task.gpus}")

        resume_s, charge, cmd = await worker.ensure_gpu(task.gpus), Charge(), task.cmd
        if kind == TaskKind.SAMPLE:
          checkpoint = None
          if task.version:
            try:
              checkpoint = (await self.stage_version(task.version), task.version.partition("@")[0])
            except Exception:
              self.staged.pop(task.version, None)
              self.stage_dirty = True
              return Done(task.id, ok=False, reply=None, dead=False, pid=worker.pid())
          reply = await worker.model.execute(cmd, checkpoint, wave=worker.wave, charge=charge)
          if not reply.ok and worker.exited():
            await self.drop_worker(task.worker, worker)
            return Done(task.id, ok=False, reply=reply, charge=charge, dead=True)
          return Done(task.id, ok=True, reply=reply, charge=charge, duration_s=resume_s, pid=worker.pid())

        async with worker.lock:
          reply, effect = await worker.wave.run(
            worker.model.call("execute", cmd, task.run, task.snapshot, task.version), charge
          )
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
      return Done(task.id, ok=False, reply=Reply.error(exc) if task.cmd else None, dead=True)

  def _bg(self, registry: dict, key: str, coro, *, wrap=None) -> None:
    async def runner() -> None:
      try:
        await coro
      finally:
        registry.pop(key, None)
        self.changed.set()

    t = asyncio.create_task(runner())
    registry[key] = wrap(t) if wrap else t

  async def _prefetch(self, version: str) -> None:
    try:
      await self.stage_version(version)
      self.stage_failures.pop(version, None)
    except Exception:
      fails = self.stage_failures[version] = self.stage_failures.get(version, 0) + 1
      await asyncio.sleep(min(10.0, 0.05 * (2 ** min(fails - 1, 8))))
      self.stage_dirty = True

  async def _sweep(self, keep: set[str]) -> None:
    try:
      present = await asyncio.to_thread(self.store.versions)
      await asyncio.to_thread(self.store.sweep, present - keep, SWEEP_GRACE_S)
    finally:
      self.sweep_task = None
      self.changed.set()

  async def build_msg(self, hello: bool = False) -> NodeMsg:
    busy_w = {task.worker for task, _ in self.active.values()}
    exited = tuple(wid for wid, w in list(self.workers.items()) if (w.exited() or w.broken) and wid not in busy_w)
    for wid in exited:
      await self.drop_worker(wid, self.workers[wid])

    lost_stage = False
    for version, folder in list(self.staged.items()):
      if not os.path.isdir(folder):
        del self.staged[version]
        lost_stage = self.stage_dirty = True

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
        self.changed.set()
        break
      done_items.append(item)
      frame_bytes += size
    done = tuple(done_items)
    self.sent_done.update(d.id for d in done)

    staged, self.stage_dirty = (tuple(self.staged) if (hello or self.stage_dirty) else None), False
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
    for tid in msg.ack:
      self.done_buf.pop(tid, None)
      self.sent_done.discard(tid)
    cancelled = set(msg.cancel)
    for tid in msg.cancel:
      if (running := self.active.get(tid)) and running[0].kind == TaskKind.SAMPLE:
        running[1].cancel()
      self.done_buf.pop(tid, None)
      self.sent_done.discard(tid)
    for version in msg.stage:
      if version not in self.staged and version not in self.staging:
        self._bg(self.staging, version, self._prefetch(version))
    if msg.sweep is not None and self.sweep_task is None:
      self.sweep_task = asyncio.create_task(self._sweep(set(msg.sweep)))
    for task in msg.tasks:
      if task.id not in cancelled and task.id not in self.active and task.id not in self.done_buf:

        async def run_task(t: Task = task) -> None:
          self.done_buf[t.id] = await self.execute_task(t)

        self._bg(self.active, task.id, run_task(), wrap=lambda bg, t=task: (t, bg))

  async def step_once(self) -> bool:
    if self._lease_expired():
      await self.reset()
    msg, sent_at = await self.build_msg(hello=not self.connected), time.monotonic()
    try:
      reply = self.control.handle(msg)
      if asyncio.iscoroutine(reply):
        reply = await reply
      self.lag, self.connected = max(0.0, time.monotonic() - sent_at), True
      self.apply(reply)
      return True
    except FencedError:
      await (self.clear() if self.draining else self.reset())
      return False
    except Exception:
      self.connected = False
      if self._lease_expired():
        await self.reset()
      return False

  async def serve_socket(self, ws, heartbeat_s: float, deadline: float | None) -> float | None:
    from aiohttp import WSMsgType

    close_types = (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR)
    sent_at, self.connected = time.monotonic(), True
    self.changed.clear()
    await ws.send_bytes(encode_msg(await self.build_msg(hello=True)))

    async def reader() -> None:
      nonlocal sent_at
      async for frame in ws:
        if frame.type in close_types:
          return
        if frame.type == WSMsgType.BINARY:
          if sent_at:
            self.lag, sent_at = max(0.0, time.monotonic() - sent_at), 0.0
          self.apply(decode_msg(frame.data, type=ControlMsg))
          if self.draining:
            self.changed.set()

    async def writer() -> None:
      nonlocal sent_at, deadline
      while not ws.closed:
        if self.draining and deadline is None:
          deadline = time.monotonic() + DRAIN_S
        if not self._drained(deadline):
          wait_s = heartbeat_s if deadline is None else min(heartbeat_s, deadline - time.monotonic())
          with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.changed.wait(), max(0.0, wait_s))
        if self.draining and deadline is None:
          deadline = time.monotonic() + DRAIN_S
        if self._drained(deadline):
          await ws.close()
          return
        self.changed.clear()
        msg = await self.build_msg(hello=False)
        sent_at = time.monotonic()
        await ws.send_bytes(encode_msg(msg))

    async def lease_watch() -> None:
      while not self._lease_expired():
        rem = LEASE_MS / 1000 - (time.monotonic() - self.last_ok)
        await asyncio.sleep(heartbeat_s if self.draining else min(heartbeat_s, max(0.02, rem)))
      raise FencedError("controller lease expired")

    tasks = {asyncio.create_task(fn()) for fn in (reader, writer, lease_watch)}
    try:
      done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
      for t in tasks:
        t.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)

    if errors := [exc for t in done if not t.cancelled() and (exc := t.exception()) is not None]:
      raise next((e for e in errors if isinstance(e, FencedError)), errors[0])
    return deadline

  async def serve(self) -> None:
    deadline: float | None = None
    heartbeat_s = min(TICK, max(0.05, LEASE_MS / 4000))
    try:
      while True:
        if self.draining and deadline is None:
          deadline = time.monotonic() + DRAIN_S
        if self._drained(deadline):
          break
        if not hasattr(self.control, "connect"):
          self.changed.clear()
          ok = await self.step_once()
          if not self._drained(deadline) and not self.changed.is_set():
            delay = min(0.05, TICK) if (not ok or self.busy() or self.draining) else TICK
            with contextlib.suppress(TimeoutError):
              await asyncio.wait_for(self.changed.wait(), max(0.0, delay))
          continue
        if self._lease_expired():
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
    tasks = (
      [bg for _, bg in self.active.values()]
      + list(self.staging.values())
      + ([self.sweep_task] if self.sweep_task else [])
    )
    for d in (self.active, self.staging, self.stage_failures, self.done_buf, self.sent_done):
      d.clear()
    self.sweep_task = None
    for t in tasks:
      t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    for wid, worker in list(self.workers.items()):
      await self.drop_worker(wid, worker)

  async def reset(self) -> None:
    self.staged.clear()
    await self.clear()
    self.connected, self.boot_id, self.last_ok = False, uuid.uuid4().hex[:12], time.monotonic()
    self.changed.set()

  async def close(self) -> None:
    self.draining = True
    await self.clear()
    await asyncio.to_thread(self.store.close)


class ControlClient:
  def __init__(self, url: str | None = None):
    base = (getenv("CONTROL_URL", "http://127.0.0.1:8000") if url is None else url).rstrip("/")
    self.ws_url = f"{'ws' + base[4:] if base.startswith('http') else base}/internal/ws"

  @contextlib.asynccontextmanager
  async def connect(self):
    import aiohttp

    async with (
      aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session,
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
