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

from tinyrl import types as T
from tinyrl.control.accounting import Charge, Wave
from tinyrl.control.process import Process
from tinyrl.helpers import DEBUG, LEASE_MS, SWEEP_S, getenv
from tinyrl.storage.store import Store
from tinyrl.trace import span
from tinyrl.workers.sampler import Sampler

GPUS, DRAIN_S, TICK, SWEEP_GRACE_S = getenv("GPUS", 1), getenv("DRAIN_S", 20.0), getenv("TICK", 0.25), 600
ENC, DEC, K = msgspec.msgpack.encode, msgspec.msgpack.decode, T.TaskKind


def gpu_memory_mib() -> int:
  if override := getenv("GPU_MEMORY_MIB", 0):
    return int(override)
  with contextlib.suppress(OSError, subprocess.SubprocessError, ValueError):
    cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"]
    return int(subprocess.check_output(cmd, timeout=2, stderr=subprocess.DEVNULL))
  return 0


class WorkerHandle:
  def __init__(self, wid: str, spec: T.WorkerSpec, model, resident: tuple[int, ...] = (), bound: tuple[int, ...] = ()):
    self.id, self.spec, self.model, self.bound, self.resident = wid, spec, model, tuple(bound), tuple(resident)
    self.loaded, self.wave, self.lock, self.broken = {}, Wave(spec.gpus), asyncio.Lock(), False

  def exited(self) -> bool:
    return bool((p := getattr(self.model, "proc", None)) and p.poll() is not None)

  def pid(self) -> int | None:
    return p.pid if (p := getattr(self.model, "proc", None)) and p.poll() is None else None

  def info(self) -> T.WorkerInfo:
    return T.WorkerInfo(self.id, self.spec, self.resident, self.bound, dict(self.loaded), self.pid())

  async def ensure_gpu(self, gpus: tuple[int, ...]) -> float:
    if self.resident == gpus:
      return 0.0
    async with self.lock:
      if self.broken:
        raise RuntimeError("worker broken")
      if self.resident == gpus:
        return 0.0
      t0 = time.monotonic()
      await self.model.resume(() if self.spec.role == T.Role.TRAINER and not self.spec.movable else gpus)
      self.resident, self.bound = gpus, self.bound or (() if self.spec.movable else gpus)
      return time.monotonic() - t0

  async def park(self) -> float:
    async with self.lock:
      if self.exited():
        raise RuntimeError("worker exited before park")
      t0 = time.monotonic()
      await self.model.park()
      self.resident = ()
      return time.monotonic() - t0

  async def close(self) -> None:
    while True:
      try:
        await self.model.close()
        self.resident = ()
        return
      except Exception as exc:
        print(f"{self.id} could not close, retrying: {exc!r}", file=sys.stderr, flush=True)
        await asyncio.sleep(min(0.05, TICK))


class NodeDaemon:
  def __init__(self, control, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store: Store | None = None):
    self.control, self.store = control, store if store is not None else Store()
    self.store.start()
    self.gpus, self.id, self.boot_id = gpus, node or f"{socket.gethostname()}-{os.getpid()}", uuid.uuid4().hex[:12]
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.workers, self.active, self.done_buf, self.staged, self.staging = {}, {}, {}, {}, {}
    self.changed, self.draining, self.connected, self.sweep_task = asyncio.Event(), False, False, None
    self.last_ok, self.lag, self.swept = time.monotonic(), 0.0, 0.0

  def notify(self) -> None:
    self.changed.set()

  def begin_drain(self) -> None:
    self.draining = True
    self.notify()

  def busy(self) -> bool:
    return bool(self.active or self.done_buf or self.staging or self.sweep_task)

  def _drained(self, end: float | None) -> bool:
    return end is not None and ((not self.workers and not self.active and not self.done_buf) or time.monotonic() >= end)

  def open_trainer(self, worker_id: str, width: int, env: dict[str, str]) -> Process:
    dist = ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={width}"] if width > 1 else []
    return Process([sys.executable, *dist, "-m", "tinyrl.workers.trainer"], env)

  def open_sampler(self, worker_id: str, spec: T.WorkerSpec, env: dict[str, str]) -> Sampler:
    return Sampler(spec.base, spec.lora, env, store=self.store)

  def spawn(self, wid: str, spec: T.WorkerSpec, gpus: tuple[int, ...]) -> WorkerHandle:
    env = {
      **os.environ,
      **self.store.env(),
      "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(self.gpus) if spec.movable else gpus)),
    }
    if spec.role == T.Role.SAMPLER and self.gpu_mib >= 64 * 1024:
      env.setdefault("TINYRL_KV_BYTES", str(24 * 2**30))
    elif spec.role == T.Role.TRAINER:
      env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    model = self.open_sampler(wid, spec, env) if spec.role == T.Role.SAMPLER else self.open_trainer(wid, spec.gpus, env)
    return WorkerHandle(wid, spec, model, bound=() if spec.movable else gpus)

  async def _drop(self, wid: str, w: WorkerHandle) -> None:
    w.broken = True
    await w.close()
    if self.workers.get(wid) is w:
      del self.workers[wid]

  async def _stage(self, ver: str) -> str:
    if not ((p := self.staged.get(ver)) and os.path.isdir(p)):
      self.staged[ver] = p = (await asyncio.to_thread(self.store.local_folder, ver))[0]
    return p

  async def _exec(self, task: T.Task) -> T.Done:
    kind, w = task.kind, self.workers.get(task.worker)
    if w is None and kind in (K.TRAIN, K.SAMPLE):
      self.workers[task.worker] = w = self.spawn(task.worker, task.spec, task.gpus)
    if w is None:
      return T.Done(task.id, dead=True)
    try:
      with span(f"node.{kind}"):
        if kind == K.PARK:
          return T.Done(task.id, duration_s=await w.park(), pid=w.pid())
        if kind in (K.STOP, K.DROP, K.SAVE):
          async with w.lock:
            if kind == K.STOP or (kind == K.SAVE and w.exited()):
              await self._drop(task.worker, w)
              return T.Done(task.id, kind == K.STOP, dead=True)
            if kind == K.DROP:
              if not w.exited() and task.run_id in w.loaded:
                await w.model.call("drop_run", task.run_id)
              w.loaded.pop(task.run_id, None)
              return T.Done(task.id, pid=w.pid())
            try:
              return T.Done(task.id, saved=await w.model.call("save_run", task.run_id, task.version), pid=w.pid())
            except Exception:
              if w.exited() or getattr(w.model, "broken", False):
                raise
              return T.Done(task.id, False, pid=w.pid())
        if any(o is not w and not set(o.resident).isdisjoint(task.gpus) for o in self.workers.values()):
          raise RuntimeError(f"GPU collision on {task.gpus}")
        rs, charge, cmd = await w.ensure_gpu(task.gpus), Charge(), task.cmd
        if kind == K.SAMPLE:
          ckpt = (await self._stage(task.version), task.version.partition("@")[0]) if task.version else None
          rep = await w.model.execute(cmd, ckpt, wave=w.wave, charge=charge)
          if not rep.ok and w.exited():
            await self._drop(task.worker, w)
            return T.Done(task.id, False, rep, charge, dead=True)
          return T.Done(task.id, True, rep, charge, duration_s=rs, pid=w.pid())
        async with w.lock:
          rep, eff = await w.wave.run(w.model.call("execute", cmd, task.run, task.snapshot, task.version), charge)
          w.loaded.pop(cmd.run, None)
          if eff.loaded:
            w.loaded[cmd.run] = eff.open_grad
          if DEBUG:
            print(f"{w.id} {cmd.request_id} {cmd.op:24s} {charge.busy_ms:9d} ms")
          return T.Done(task.id, True, rep, charge, eff, duration_s=rs, pid=w.pid())
    except asyncio.CancelledError:
      if kind in (K.TRAIN, K.PARK, K.SAVE, K.STOP):
        await asyncio.shield(self._drop(task.worker, w))
      raise
    except Exception as exc:
      await self._drop(task.worker, w)
      return T.Done(task.id, False, T.Reply.error(exc) if task.cmd else None, dead=True)

  def _spawn_bg(self, table: dict, key: str, coro, wrap=lambda tk: tk) -> None:
    async def run():
      try:
        await coro
      finally:
        table.pop(key, None)
        self.notify()

    table[key] = wrap(asyncio.create_task(run()))

  async def _put_done(self, task: T.Task) -> None:
    self.done_buf[task.id] = await self._exec(task)

  async def _try_stage(self, ver: str) -> None:
    with contextlib.suppress(Exception):
      await self._stage(ver)

  async def _do_sweep(self, keep: set[str]) -> None:
    try:
      await asyncio.to_thread(self.store.sweep, (await asyncio.to_thread(self.store.versions)) - keep, SWEEP_GRACE_S)
    finally:
      self.sweep_task = None
      self.notify()

  async def build_msg(self, hello: bool = False) -> T.NodeMsg:
    busy_w = {t.worker for t, _ in self.active.values()}
    exited = tuple(k for k, w in list(self.workers.items()) if (w.exited() or w.broken) and k not in busy_w)
    for k in exited:
      await self._drop(k, self.workers[k])
    lost = [self.staged.pop(v) for v, p in list(self.staged.items()) if not os.path.isdir(p)]
    if sweep := not self.draining and not self.sweep_task and time.time() - self.swept > SWEEP_S:
      self.swept = time.time()
    infos = tuple(w.info() for w in self.workers.values()) if hello else ()
    act = tuple(self.active) if (hello or self.draining or lost) else ()
    hdr = (
      "hello" if hello else "events",
      self.id,
      self.boot_id,
      self.gpus,
      self.gpu_mib,
      self.draining,
      self.lag,
      sweep,
    )
    return T.NodeMsg(*hdr, infos, tuple(self.staged), exited, act, tuple(self.done_buf.values()))

  def apply(self, msg: T.ControlMsg) -> None:
    self.last_ok = time.monotonic()
    if msg.reset:
      raise T.FencedError("controller reset node incarnation")
    for tid in msg.ack:
      self.done_buf.pop(tid, None)
    for tid in msg.cancel:
      if (pair := self.active.pop(tid, None)) and pair[0].kind == K.SAMPLE:
        pair[1].cancel()
      self.done_buf.pop(tid, None)
    for v in [v for v in msg.stage if v not in self.staged and v not in self.staging]:
      self._spawn_bg(self.staging, v, self._try_stage(v))
    if msg.sweep and not self.sweep_task:
      self.sweep_task = asyncio.create_task(self._do_sweep(set(msg.sweep)))
    for t in [t for t in msg.tasks if t.id not in self.active and t.id not in self.done_buf]:
      self._spawn_bg(self.active, t.id, self._put_done(t), lambda tk, task=t: (task, tk))

  async def step_once(self, wait_ms: int = 0) -> bool:
    if not self.draining and time.monotonic() - self.last_ok >= LEASE_MS / 1000:
      await self.reset()
    msg, t0 = await self.build_msg(hello=not self.connected), time.monotonic()
    try:
      rep = self.control.handle(msg)
      rep = await rep if asyncio.iscoroutine(rep) else rep
      self.lag, self.connected = max(0.0, time.monotonic() - t0), True
      self.apply(rep)
      return True
    except T.FencedError:
      await (self.clear() if self.draining else self.reset())
      return False
    except Exception:
      self.connected = False
      if not self.draining and time.monotonic() - self.last_ok >= LEASE_MS / 1000:
        await self.reset()
      return False

  async def _wait(self, delay: float) -> None:
    with contextlib.suppress(TimeoutError):
      await asyncio.wait_for(self.changed.wait(), max(0.0, delay))

  async def serve(self) -> None:
    end, beat = None, min(TICK, max(0.05, LEASE_MS / 4000))
    try:
      while not self._drained(end := (end or time.monotonic() + DRAIN_S) if self.draining else None):
        if not hasattr(self.control, "connect"):
          self.changed.clear()
          ok = await self.step_once()
          if not self._drained(end) and not self.changed.is_set():
            await self._wait(min(0.05, TICK) if (not ok or self.busy() or self.draining) else TICK)
          continue
        if not self.draining and time.monotonic() - self.last_ok >= LEASE_MS / 1000:
          await self.reset()
        try:
          async with self.control.connect() as ws:
            t0, self.connected = time.monotonic(), True
            await ws.send_bytes(ENC(await self.build_msg(hello=True)))

            async def reader():
              nonlocal t0
              async for raw in ws:
                if raw.type != 2:
                  if raw.type in (8, 256, 257):
                    return
                  continue
                self.lag, t0 = (max(0.0, time.monotonic() - t0), 0.0) if t0 else (self.lag, 0.0)
                self.apply(DEC(raw.data, type=T.ControlMsg))
                self.notify()

            async def writer():
              nonlocal t0, end
              while not ws.closed:
                self.changed.clear()
                if not self._drained(end := (end or time.monotonic() + DRAIN_S) if self.draining else end):
                  await self._wait(beat if end is None else min(beat, end - time.monotonic()))
                if self._drained(end := (end or time.monotonic() + DRAIN_S) if self.draining else end):
                  return await ws.close()
                msg, t0 = await self.build_msg(hello=False), time.monotonic()
                await ws.send_bytes(ENC(msg))

            done, pend = await asyncio.wait(
              {asyncio.create_task(reader()), asyncio.create_task(writer())}, return_when=asyncio.FIRST_COMPLETED
            )
            for p in pend:
              p.cancel()
            await asyncio.gather(*pend, return_exceptions=True)
            if excs := [e for d in done if (e := d.exception())]:
              raise next((e for e in excs if isinstance(e, T.FencedError)), excs[0])
        except T.FencedError:
          await (self.clear() if self.draining else self.reset())
          if self.draining:
            break
        except Exception:
          self.connected = False
          await asyncio.sleep(min(0.05, TICK))
    finally:
      await self.close()

  async def clear(self) -> None:
    tasks = [
      *(tk for _, tk in self.active.values()),
      *self.staging.values(),
      *((self.sweep_task,) if self.sweep_task else ()),
    ]
    self.active.clear()
    self.staging.clear()
    self.done_buf.clear()
    self.sweep_task = None
    for tk in tasks:
      tk.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    for wid, w in list(self.workers.items()):
      await self._drop(wid, w)

  async def reset(self) -> None:
    self.staged.clear()
    await self.clear()
    self.connected, self.boot_id, self.last_ok = False, uuid.uuid4().hex[:12], time.monotonic()
    self.notify()

  async def close(self) -> None:
    self.draining = True
    await self.clear()
    await asyncio.to_thread(self.store.close)


class ControlClient:
  def __init__(self, url: str | None = None):
    base = (getenv("CONTROL_URL", "http://127.0.0.1:8000") if url is None else url).rstrip("/")
    self.ws_url = ("ws" + base[4:] if base.startswith("http") else base) + "/internal/ws"

  @contextlib.asynccontextmanager
  async def connect(self):
    import aiohttp

    async with (
      aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as s,
      s.ws_connect(self.ws_url, max_msg_size=0, heartbeat=None) as ws,
    ):
      yield ws

  async def aclose(self) -> None:
    pass


if __name__ == "__main__":
  daemon = NodeDaemon(ControlClient())

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, daemon.begin_drain)
    with contextlib.suppress(asyncio.CancelledError):
      await daemon.serve()

  asyncio.run(main())
