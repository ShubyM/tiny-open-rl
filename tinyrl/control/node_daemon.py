"""Node daemon: reports observed capacity, stages checkpoints off-GPU, and executes fleet grants."""

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
from dataclasses import dataclass

import msgspec

from tinyrl.control.accounting import Charge, Wave
from tinyrl.control.process import Process
from tinyrl.helpers import DEBUG, LEASE_MS, SWEEP_S, getenv
from tinyrl.storage.store import Store
from tinyrl.trace import span
from tinyrl.types import (
  RPC_TYPES,
  DetachRun,
  FencedError,
  LocalWorker,
  NodePlan,
  NodeStatus,
  Release,
  Reply,
  Role,
  Work,
  WorkerSpec,
)
from tinyrl.workers.sampler import Sampler

GPUS, DRAIN_S, TICK, SWEEP_GRACE_S = getenv("GPUS", 1), getenv("DRAIN_S", 20.0), getenv("TICK", 0.25), 600


def gpu_memory_mib() -> int:
  if override := getenv("GPU_MEMORY_MIB", 0):
    return int(override)
  with contextlib.suppress(OSError, subprocess.SubprocessError, ValueError):
    cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"]
    return int(subprocess.check_output(cmd, timeout=2, stderr=subprocess.DEVNULL))
  return 0


@dataclass
class LoadedRun:
  dirty: bool = False
  open_grad: bool = False


@dataclass
class SharedLoad:
  task: asyncio.Task
  charge: Charge
  users: int = 0


class WorkerHandle:
  """The process and state this node can observe directly."""

  def __init__(self, wid: str, spec: WorkerSpec, model, resident=(), bound=()):
    self.id, self.spec, self.model = wid, spec, model
    self.resident, self.bound, self.target_gpus = tuple(resident), tuple(bound), ()
    self.resumed_at = self.idle_since = self.switch_s = self.gpu_ms = 0.0
    self.loaded: dict[str, LoadedRun] = {}
    self.running, self.op_task = {}, None
    self.wave, self.loads, self.lock, self.broken = Wave(spec.gpus), {}, asyncio.Lock(), False

  def exited(self) -> bool:
    return bool((p := getattr(self.model, "proc", None)) and p.poll() is not None)

  def pid(self) -> int | None:
    return p.pid if (p := getattr(self.model, "proc", None)) and p.poll() is None else None

  def observe(self, now: float) -> LocalWorker:
    return LocalWorker(
      current=self.resident or self.target_gpus,
      bound=self.bound,
      holding=tuple(self.running),
      busy=bool(self.op_task or self.broken or self.exited()),
      resident_s=now - self.resumed_at if self.resident and self.resumed_at else None,
      switch_s=self.switch_s,
      idle_s=now - self.idle_since if self.idle_since else None,
      gpu_ms=self.gpu_ms,
      loaded={r: state.open_grad for r, state in self.loaded.items()},
      pid=self.pid(),
    )

  async def ensure_gpu(self, gpus: tuple[int, ...]) -> None:
    if not gpus:
      raise ValueError("no gpus assigned")
    async with self.lock:
      if self.broken:
        raise RuntimeError("worker broken")
      if self.resident == gpus:
        return
      t0 = time.monotonic()
      await self.model.resume(() if self.spec.role == Role.TRAINER and not self.spec.movable else gpus)
      now = time.monotonic()
      self.resident, self.bound = gpus, (self.bound or (() if self.spec.movable else gpus))
      self.resumed_at = now
      if self.switch_s:
        self.switch_s += now - t0

  async def park(self) -> None:
    async with self.lock:
      if self.exited():
        raise RuntimeError("worker exited before park")
      t0 = time.monotonic()
      await self.model.park()
      self.resident, self.target_gpus, self.switch_s = (), (), time.monotonic() - t0

  async def close(self) -> None:
    while True:
      try:
        await self.model.close()
        self.resident = self.target_gpus = ()
        return
      except Exception as exc:
        print(f"{self.id} could not close, retrying: {exc!r}", file=sys.stderr, flush=True)
        await asyncio.sleep(min(0.05, TICK))

  async def load_shared(self, ckpt: tuple[str, str], charge: Charge) -> None:
    if (load := self.loads.get(ckpt)) is None:
      cost = Charge()
      task = asyncio.create_task(self.wave.run(self.model.load(ckpt), cost))
      self.loads[ckpt] = load = SharedLoad(task, cost)
    load.users += 1
    try:
      await load.task
    except Exception as exc:
      if not isinstance(exc, ValueError):
        self.broken = True
      raise
    finally:
      if self.loads.get(ckpt) is load:
        del self.loads[ckpt]
      charge.started_ms = load.charge.started_ms
      charge.busy_ms += load.charge.busy_ms
      charge.gpu_ms += load.charge.gpu_ms / load.users

  async def sample(self, work: Work, store: Store, folder: str | None, charge: Charge) -> Reply:
    name = None
    if work.version:
      try:
        if folder and os.path.isdir(folder):
          ckpt = folder, work.version.partition("@")[0]
        else:
          ckpt = await asyncio.to_thread(store.local_folder, work.version)
        await self.load_shared(ckpt, charge)
        name = ckpt[1]
      except Exception as exc:
        return Reply.error(exc)
    return await self.wave.run(self.model.execute(work.cmd, name), charge)

  async def execute(self, work: Work, gpus: tuple[int, ...], store: Store, folder=None):
    cmd, charge = work.cmd, Charge()
    try:
      with span(f"node.{self.spec.role}"):
        await self.ensure_gpu(gpus)
        if self.spec.role == Role.SAMPLER:
          rep, done = await self.sample(work, store, folder, charge), {}
        else:
          async with self.lock:
            rep, eff = await self.wave.run(
              self.model.call("execute", cmd, work.run, work.snapshot, work.version), charge
            )
            if eff.loaded:
              state = self.loaded.setdefault(cmd.run, LoadedRun())
              state.dirty |= eff.mutated
              state.open_grad = eff.open_grad
            else:
              self.loaded.pop(cmd.run, None)
            self.idle_since = time.monotonic()
            done = {"mutated": eff.mutated, "checkpoint": eff.checkpoint}
            if DEBUG:
              print(f"{self.id} {cmd.request_id} {cmd.op:24s} {charge.busy_ms:9d} ms")
        self.gpu_ms += charge.gpu_ms
        return rep, charge, done
    except Exception as exc:
      self.broken = True
      return Reply.error(exc), None, {"dead": True}


class NodeDaemon:
  def __init__(self, control, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store: Store | None = None):
    self.control, self.store = control, store if store is not None else Store()
    self.store.start()
    self.gpus, self.id, self.boot_id = gpus, node or f"{socket.gethostname()}-{os.getpid()}", uuid.uuid4().hex[:12]
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.workers, self.staging, self.staged = {}, {}, {}
    self.changed, self.draining, self.fenced = asyncio.Event(), False, False
    self.last_ok, self.lag, self.swept, self.sweeping, self.next_wake = time.monotonic(), 0.0, 0.0, None, None

  def notify(self) -> None:
    ev, self.changed = self.changed, asyncio.Event()
    ev.set()

  def begin_drain(self) -> None:
    self.draining = True
    self.notify()

  def busy(self) -> bool:
    return bool(self.staging or any(w.running or w.op_task for w in self.workers.values()))

  def open_trainer(self, worker_id: str, width: int, env: dict[str, str]) -> Process:
    dist = ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={width}"] if width > 1 else []
    return Process([sys.executable, *dist, "-m", "tinyrl.workers.trainer"], env)

  def open_sampler(self, worker_id: str, spec: WorkerSpec, env: dict[str, str]) -> Sampler:
    return Sampler(spec.base, spec.lora, env)

  def spawn(self, wid: str, spec: WorkerSpec, gpus: tuple[int, ...]) -> WorkerHandle:
    visible = range(self.gpus) if spec.movable else gpus
    env = {**os.environ, **self.store.env(), "CUDA_VISIBLE_DEVICES": ",".join(map(str, visible))}
    if spec.role == Role.SAMPLER:
      if self.gpu_mib >= 64 * 1024:
        env.setdefault("TINYRL_KV_BYTES", str(24 * 2**30))
      model = self.open_sampler(wid, spec, env)
    else:
      env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
      model = self.open_trainer(wid, spec.gpus, env)
    return WorkerHandle(wid, spec, model, bound=() if spec.movable else gpus)

  def start_task(self, coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    task.add_done_callback(lambda _: self.notify())
    return task

  async def call(self, name: str, *args, retry: bool = False, **kw):
    while True:
      try:
        res = getattr(self.control, name)(self.id, self.boot_id, *args, **kw)
        out = await res if asyncio.iscoroutine(res) else res
        self.last_ok = time.monotonic()
        return out
      except FencedError:
        raise
      except Exception as exc:
        if not self.draining and time.monotonic() - self.last_ok >= LEASE_MS / 1000:
          raise FencedError("lease expired") from exc
        if not retry:
          raise
        await asyncio.sleep(min(0.05, TICK))

  async def run_work(self, w: WorkerHandle, work: Work, gpus: tuple[int, ...], folder: str | None) -> None:
    try:
      rep, ch, done = await w.execute(work, gpus, self.store, folder)
      await self.call("finish", w.id, work.cmd.request_id, rep, ch, retry=True, attempt=work.attempt, **done)
    except FencedError:
      self.fenced = True
    finally:
      if not (w.running.pop(work.cmd.request_id, None) and w.running):
        w.target_gpus = ()

  async def run_op(self, w: WorkerHandle, kind: str, run: str = "") -> None:
    try:
      if kind == "park":
        await w.park()
        return
      if kind in ("detach", "stop"):
        runs = [run] if kind == "detach" else sorted(w.loaded)
        for rid in runs:
          try:
            await self.detach_run(w, rid, drop=kind == "detach")
          except FencedError:
            raise
          except Exception as exc:
            if w.exited() or getattr(w.model, "broken", False):
              raise
            print(f"{w.id} could not detach {rid}, retrying: {exc!r}", file=sys.stderr, flush=True)
        if w.loaded.keys() & runs:
          await asyncio.sleep(TICK)
          return
        if kind == "detach":
          return
      await w.close()
      if self.workers.get(w.id) is w:
        del self.workers[w.id]
    except FencedError:
      self.fenced = True
    except Exception:
      w.broken = True
    finally:
      w.op_task = None

  async def detach_run(self, w: WorkerHandle, run: str, *, drop: bool) -> None:
    async with w.lock:
      state = w.loaded[run]
      live = not w.exited()
      snapshot = await w.model.call("save_run", run, uuid.uuid4().hex) if live and state.dirty else None
      await self.call("detach", w.id, run, snapshot, retry=True)
      state.dirty = False
      if live and drop:
        await w.model.call("drop_run", run)
      del w.loaded[run]

  def stage(self, ver: str) -> None:
    async def fetch():
      try:
        self.staged[ver] = (await asyncio.to_thread(self.store.local_folder, ver))[0]
      except Exception as exc:
        if DEBUG:
          print(f"{self.id} retrying checkpoint {ver}: {exc!r}", file=sys.stderr)
        await asyncio.sleep(TICK)
      finally:
        self.staging.pop(ver, None)

    self.staging[ver] = self.start_task(fetch())

  def apply(self, plan: NodePlan) -> None:
    now = time.monotonic()
    self.next_wake = None if plan.wake_after is None else now + plan.wake_after
    for ver in plan.stage:
      if not self.draining and ver not in self.staging:
        self.stage(ver)

    for wid, w in list(self.workers.items()):
      if w.running or w.op_task:
        continue
      action = Release.STOP if self.draining and not plan.grants else plan.release.get(wid)
      if w.broken or w.exited():
        w.op_task = self.start_task(self.run_op(w, "close"))
      elif action:
        if isinstance(action, DetachRun):
          w.op_task = self.start_task(self.run_op(w, "detach", action.run))
        else:
          w.op_task = self.start_task(self.run_op(w, action))

    for g in plan.grants:
      if (w := self.workers.get(g.worker)) is None:
        self.workers[g.worker] = w = self.spawn(g.worker, g.spec, g.gpus)
      if w.op_task or w.broken or w.exited():
        continue
      w.target_gpus = g.gpus
      for item in g.work:
        rid = item.cmd.request_id
        if rid not in w.running:
          w.running[rid] = self.start_task(self.run_work(w, item, g.gpus, self.staged.get(item.version)))

  async def step_once(self, wait_ms: int = 0) -> bool:
    if self.fenced or (not self.draining and time.monotonic() - self.last_ok >= LEASE_MS / 1000):
      await (self.clear() if self.draining else self.reset())
      if self.draining:
        return False
    try:
      idle = 0 if (self.busy() or self.draining) else wait_ms
      if self.next_wake is not None and idle > 0:
        idle = min(idle, max(0, int((self.next_wake - time.monotonic()) * 1000)))
      self.staged = {v: path for v, path in self.staged.items() if os.path.isdir(path)}
      due = not self.draining and time.time() - self.swept > SWEEP_S and (not self.sweeping or self.sweeping.done())
      t0 = time.monotonic()
      workers = {wid: w.observe(t0) for wid, w in self.workers.items()}
      status = NodeStatus(self.gpus, self.draining, self.lag, workers, tuple(self.staged), due)
      plan = await self.call("take", status, wait_ms=idle)
      self.lag = max(0.0, time.monotonic() - t0 - idle / 1000)
      if due:
        self.swept = time.time()

        def do_sweep() -> None:
          if plan.sweep is None:
            self.store.trim_cache()
          else:
            self.store.sweep(self.store.versions() - set(plan.sweep), SWEEP_GRACE_S)

        self.sweeping = asyncio.create_task(asyncio.to_thread(do_sweep))
      self.apply(plan)
      return True
    except FencedError:
      await (self.clear() if self.draining else self.reset())
      return False
    except Exception:
      return False

  async def serve(self) -> None:
    deadline: float | None = None
    try:
      while True:
        if self.draining:
          deadline = deadline or time.monotonic() + DRAIN_S
          if (not self.workers and not self.busy()) or time.monotonic() >= deadline:
            break
        changed, ok = self.changed, await self.step_once(wait_ms=int(TICK * 1000))
        if self.draining and not self.workers and not self.busy():
          break
        if not ok or self.busy() or self.draining:
          delay = min(0.1, TICK) if not ok else TICK
          delay = delay if self.next_wake is None else min(delay, max(0.0, self.next_wake - time.monotonic()))
          with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(changed.wait(), delay)
    finally:
      await self.close()

  async def clear(self, *extra) -> None:
    tasks = [t for w in self.workers.values() for t in (*w.running.values(), w.op_task) if t]
    tasks += [*self.staging.values(), *(t for t in extra if t)]
    self.staging.clear()
    for t in tasks:
      t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    for w in list(self.workers.values()):
      await w.close()
    self.workers.clear()

  async def reset(self) -> None:
    self.staged.clear()
    await self.clear()
    self.fenced, self.boot_id, self.last_ok = False, uuid.uuid4().hex[:12], time.monotonic()
    self.notify()

  async def close(self) -> None:
    self.draining = True
    await self.clear(self.sweeping)
    await asyncio.to_thread(self.store.close)


class ControlClient:
  """Msgpack client for the controller's fenced node RPCs."""

  def __init__(self, url: str | None = None):
    self.url = (getenv("CONTROL_URL", "http://127.0.0.1:8000") if url is None else url).rstrip("/")
    self.session = None

  async def aclose(self) -> None:
    if self.session and not self.session.closed:
      await self.session.close()

  def __getattr__(self, method: str):
    if method not in RPC_TYPES:
      raise AttributeError(method)

    async def rpc(*args, **kw):
      import aiohttp

      if self.session is None or self.session.closed:
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
      to = aiohttp.ClientTimeout(total=max(10.0, kw.get("wait_ms", 0) / 1000 + 5.0))
      async with self.session.post(
        f"{self.url}/internal/{method}", data=msgspec.msgpack.encode((args, kw)), timeout=to
      ) as r:
        if r.status == 409:
          raise FencedError(await r.text())
        r.raise_for_status()
        return msgspec.msgpack.decode(await r.read(), type=RPC_TYPES[method])

    return rpc


if __name__ == "__main__":

  async def main():
    control = ControlClient()
    try:
      daemon = NodeDaemon(control)
      asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, daemon.begin_drain)
      with contextlib.suppress(asyncio.CancelledError):
        await daemon.serve()
    finally:
      await control.aclose()

  asyncio.run(main())
