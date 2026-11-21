"""Node daemon: executes concrete turns from the controller's single `exchange` RPC.

Owns worker process supervision, physical GPU exclusion guard, off-GPU checkpoint staging (`prefetch`), periodic cache
sweeps, local `LEASE_MS` fencing, and graceful `SIGTERM` drain. Contains zero fairness or scheduling policy.
"""

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

from tinyrl.bus import LEASE_MS, Bus
from tinyrl.control import scheduler
from tinyrl.control.accounting import Charge, Wave
from tinyrl.control.process import Process
from tinyrl.helpers import DEBUG, getenv
from tinyrl.storage.store import Store
from tinyrl.trace import span
from tinyrl.types import NodeReply, NodeReport, Reply, Role, Turn, TurnResult, WorkerSpec, WorkerStatus
from tinyrl.workers.sampler import Sampler

GPUS, DRAIN_S, TICK = getenv("GPUS", 1), getenv("DRAIN_S", 20.0), getenv("TICK", 0.25)
SWEEP_S, SWEEP_GRACE_S = getenv("SWEEP_S", 60.0), 600


def gpu_memory_mib() -> int:
  if override := getenv("GPU_MEMORY_MIB", 0):
    return int(override)
  try:
    cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"]
    return int(subprocess.check_output(cmd, timeout=2, stderr=subprocess.DEVNULL))
  except (OSError, subprocess.SubprocessError, ValueError):
    return 0


def failed(turn: Turn, exc: Exception, charge: Charge | None = None, **times) -> TurnResult:
  """The result of a turn that broke: its command (if any) gets the error."""
  return TurnResult(turn.id, False, Reply.error(exc) if turn.cmd else None, charge, **times)


class WorkerHandle:
  """Thin per-worker executor: translates controller `Turn`s into trainer or sampler calls."""

  def __init__(self, wid: str, spec: WorkerSpec, model, resident=(), bound=()):
    self.id, self.spec, self.model, self.resident, self.bound = wid, spec, model, resident, bound
    self.loaded, self.wave, self.loads, self.lock = set(), Wave(spec.gpus), {}, asyncio.Lock()
    self.claimed: tuple[int, ...] = ()
    self.active, self.park_s = 0, 0.0

  def exited(self) -> bool:
    return bool((p := getattr(self.model, "proc", None)) and p.poll() is not None)

  def pid(self) -> int | None:
    return p.pid if (p := getattr(self.model, "proc", None)) and p.poll() is None else None

  async def ensure_gpu(self, gpus: tuple[int, ...]) -> tuple[float, float]:
    if self.resident == gpus:
      return 0.0, 0.0
    async with self.lock:
      if self.resident == gpus:
        return 0.0, 0.0
      t0, park_s = time.monotonic(), self.park_s
      await self.model.resume(gpus)
      self.resident, self.claimed = gpus, ()
      if not self.bound and (self.spec.role == Role.SAMPLER or self.spec.gpus > 1):
        self.bound = gpus
      return time.monotonic() - t0, park_s

  async def park(self) -> float:
    async with self.lock:
      if self.exited():
        raise RuntimeError("worker exited before park")
      t0 = time.monotonic()
      await self.model.park()
      self.resident, self.claimed, self.park_s = (), (), time.monotonic() - t0
      return self.park_s

  async def load_shared(self, ckpt: tuple[str, str]) -> tuple[Charge, int, Exception | None]:
    """Coalesce concurrent loads of the same checkpoint on a sampler so load GPU time is billed once."""
    if (entry := self.loads.get(ckpt)) is None:
      charge, waiters = Charge(), [0]

      async def do_load() -> Exception | None:
        try:
          await self.wave.run(self.model.load(ckpt), charge)
          return None
        except Exception as exc:
          return exc
        finally:
          self.loads.pop(ckpt, None)

      self.loads[ckpt] = entry = (asyncio.create_task(do_load()), charge, waiters)
    task, charge, waiters = entry
    waiters[0] += 1
    err = await task
    return charge, waiters[0], err

  async def close(self) -> None:
    while True:
      try:
        await self.model.close()
        self.resident = self.claimed = ()
        return
      except Exception as exc:
        print(f"{self.id} could not close, retrying: {exc!r}", file=sys.stderr, flush=True)
        await asyncio.sleep(min(0.05, TICK))

  async def run(self, turn: Turn, store: Store) -> TurnResult:
    if turn.kind == "park":
      return TurnResult(turn.id, park_s=await self.park())
    if turn.kind == "close":
      async with self.lock:
        await self.close()
      return TurnResult(turn.id)
    if turn.kind == "snapshot":
      async with self.lock:
        rid, ver = turn.args["run"], turn.args.get("version")
        saved = await self.model.call("save_run", rid, ver) if ver and rid in self.loaded else None
        if turn.args.get("close"):
          await self.close()
        elif rid in self.loaded:
          await self.model.call("drop_run", rid)
        self.loaded.discard(rid)
        return TurnResult(turn.id, saved=saved)
    if turn.kind == "sample":
      (resume_s, park_s), charge, cmd = await self.ensure_gpu(turn.gpus), Charge(), turn.cmd
      ver, ckpt, err, load, stab = turn.args.get("version"), None, None, None, {}
      if ver is not None:
        try:
          ckpt = await asyncio.to_thread(store.local_folder, ver)
        except Exception as exc:
          err = exc
      if ckpt is not None and err is None:
        load_c, waiters, err = await self.load_shared(ckpt)
        load = (load_c, waiters)
      rep = (
        Reply.error(err) if err is not None else await self.wave.run(self.model.execute(cmd, ckpt, True, stab), charge)
      )
      if load is not None:
        load_c, waiters = load
        charge.started_ms, charge.busy_ms = load_c.started_ms, charge.busy_ms + load_c.busy_ms
        charge.gpu_ms += load_c.gpu_ms / waiters
      return TurnResult(turn.id, True, rep, charge, resume_s=resume_s, park_s=park_s, stability=stab or None)
    if turn.kind == "train":
      (resume_s, park_s), charge, cmd = await self.ensure_gpu(turn.gpus), Charge(), turn.cmd
      async with self.lock:
        args = (turn.args.get("record") or None, turn.args.get("saved"), turn.args.get("version"))
        if (stab := turn.args.get("stability")) is not None:
          args = (*args, stab)
        rep, eff = await Wave(turn.spec.gpus).run(self.model.call("execute", cmd, *args), charge)
        loaded = bool(eff.get("loaded"))
        (self.loaded.add if loaded else self.loaded.discard)(cmd.run)
        if DEBUG:
          print(f"{self.id} {cmd.request_id} {cmd.op:24s} {charge.busy_ms:9d} ms")
        return TurnResult(
          turn.id, True, rep, charge, bool(eff.get("mutated")), loaded, eff.get("saved"), resume_s, park_s
        )
    raise ValueError(f"unknown turn kind {turn.kind}")


class NodeDaemon:
  def __init__(self, control, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store: Store | None = None):
    self.control, self.store = control, store if store is not None else Store()
    self.store.start()
    self.gpus, self.id, self.boot_id = gpus, node or f"{socket.gethostname()}-{os.getpid()}", uuid.uuid4().hex[:12]
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.workers, self.active, self.staging, self.staged_paths, self.stage_errors = {}, {}, {}, {}, {}
    self.gpu_cond, self.changed, self.draining = asyncio.Condition(), asyncio.Event(), False
    self.last_exchange, self.lag, self.swept, self.sweeping = time.monotonic(), 0.0, 0.0, None

  def notify(self) -> None:
    changed, self.changed = self.changed, asyncio.Event()
    changed.set()

  def begin_drain(self) -> None:
    self.draining = True
    self.notify()

  def open_trainer(self, worker_id: str, width: int, env: dict[str, str]) -> Process:
    dist = ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={width}"] if width > 1 else []
    return Process([sys.executable, *dist, "-m", "tinyrl.workers.trainer"], env)

  def open_sampler(self, worker_id: str, spec: WorkerSpec, env: dict[str, str]) -> Sampler:
    return Sampler(spec.base, spec.lora, env)

  def spawn_worker(self, wid: str, spec: WorkerSpec, gpus: tuple[int, ...]) -> WorkerHandle:
    env = {**scheduler.settings(self.gpus, self.gpu_mib * 2**20), **os.environ, **self.store.env()}
    if spec.role == Role.SAMPLER:
      env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
      return WorkerHandle(wid, spec, self.open_sampler(wid, spec, env), bound=gpus)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, tuple(range(self.gpus)) if spec.gpus == 1 else gpus))
    return WorkerHandle(wid, spec, self.open_trainer(wid, spec.gpus, env), bound=() if spec.gpus == 1 else gpus)

  def build_report(self) -> NodeReport:
    done = [(t, tk) for t, tk in list(self.active.values()) if tk.done()]
    finished = [failed(t, RuntimeError("cancelled")) if tk.cancelled() else tk.result() for t, tk in done]
    for ver, tk in list(self.staging.items()):
      if tk.done() and not (self.staging.pop(ver) and tk.cancelled()):
        if isinstance(res := tk.result(), Exception):
          self.stage_errors[ver] = f"{type(res).__name__}: {res}"
        else:
          self.staged_paths[ver] = res
    staged = [v for v, (f, _) in list(self.staged_paths.items()) if os.path.isdir(f) or not self.staged_paths.pop(v)]
    stage_errors, self.stage_errors = self.stage_errors, {}
    statuses = [
      WorkerStatus(h.id, h.resident, h.bound, h.pid(), tuple(sorted(h.loaded)), h.park_s) for h in self.workers.values()
    ]
    sweep = not self.draining and time.time() - self.swept > SWEEP_S and (not self.sweeping or self.sweeping.done())
    args = (self.id, self.boot_id, self.gpus, self.gpu_mib * 2**20, self.draining, self.lag)
    return NodeReport(*args, statuses, finished, staged, stage_errors, sweep, list(self.active))

  def start_prefetch(self, version: str) -> None:
    if version in self.staged_paths or version in self.staging:
      return

    async def stage():
      try:
        return await asyncio.to_thread(self.store.local_folder, version)
      except Exception as exc:
        return exc

    self.staging[version] = task = asyncio.create_task(stage())
    task.add_done_callback(lambda _: self.notify())

  def start_sweep(self, live_versions: list[str]) -> None:
    self.swept = time.time()

    async def sweep() -> None:
      unused = (await asyncio.to_thread(self.store.versions)) - set(live_versions)
      await asyncio.to_thread(self.store.sweep, unused, SWEEP_GRACE_S)

    self.sweeping = asyncio.create_task(sweep())

  async def step_once(self, wait_ms: int = 0) -> NodeReply | None:
    if not self.draining and time.monotonic() - self.last_exchange >= LEASE_MS / 1000:
      await self.reset()
    busy_workers = {t.worker for t, _ in self.active.values()}
    for wid, h in list(self.workers.items()):
      if h.exited() and h.active == 0 and wid not in busy_workers:
        await self.drop_worker(wid, h)
    report, before = self.build_report(), time.monotonic()
    try:
      reply = await self.control.exchange(report, wait_ms=wait_ms)
    except Exception as exc:
      if not self.draining and time.monotonic() - self.last_exchange >= LEASE_MS / 1000:
        await self.reset()
      if DEBUG:
        print(f"node {self.id} exchange failed: {exc!r}", file=sys.stderr, flush=True)
      return None
    self.last_exchange = time.monotonic()
    self.lag = max(0.0, self.last_exchange - before - wait_ms / 1000)
    if reply.reset:
      await self.reset()
      return reply
    for tid in reply.acknowledged:
      self.active.pop(tid, None)
    for ver in reply.prefetch:
      self.start_prefetch(ver)
    if reply.live_versions is not None:
      self.start_sweep(reply.live_versions)
    for turn in reply.turns:
      if turn.id not in self.active:
        task = asyncio.create_task(self.execute_turn(turn))
        task.add_done_callback(lambda _: self.notify())
        self.active[turn.id] = (turn, task)
    return reply

  async def serve(self) -> None:
    deadline: float | None = None
    try:
      while True:
        if self.draining:
          deadline = deadline or time.monotonic() + DRAIN_S
          if (not self.active and not self.workers) or time.monotonic() >= deadline:
            break
        idle, changed = not self.active and not self.staging and not self.draining, self.changed
        reply = await self.step_once(wait_ms=int(TICK * 1000) if idle else 0)
        if self.draining and not self.active and not self.workers:
          break
        any_done = any(t.done() for _, t in self.active.values()) or any(t.done() for t in self.staging.values())
        if reply is None or ((not idle or self.active or self.staging) and not any_done):
          delay = min(0.1, TICK) if reply is None else (min(0.05, TICK) if (self.active or self.draining) else TICK)
          with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(changed.wait(), delay)
    finally:
      await self.close()

  async def drop_worker(self, wid: str, h: WorkerHandle) -> None:
    await h.close()
    async with self.gpu_cond:
      if self.workers.get(wid) is h:
        del self.workers[wid]
      self.gpu_cond.notify_all()

  async def claim_gpus(self, h: WorkerHandle, gpus: tuple[int, ...]) -> None:
    """Defense-in-depth physical GPU exclusion: never resume onto a GPU still occupied locally."""
    while True:
      async with self.gpu_cond:
        overlap = [
          o
          for o in self.workers.values()
          if o is not h and (occ := o.resident or o.claimed) and not set(occ).isdisjoint(gpus)
        ]
        if h.resident and h.resident != gpus:
          overlap.append(h)
        if not overlap:
          h.claimed = gpus
          return
        if any(o.active > (1 if o is h else 0) for o in overlap):
          await self.gpu_cond.wait()
          continue
        for o in overlap:
          o.active += 1
      for o in overlap:
        try:
          await o.park()
        except Exception:
          await self.drop_worker(o.id, o)
        finally:
          async with self.gpu_cond:
            o.active -= 1
            self.gpu_cond.notify_all()

  async def execute_turn(self, turn: Turn) -> TurnResult:
    h = self.workers.get(turn.worker)
    if turn.kind in ("train", "sample") and h is None:
      self.workers[turn.worker] = h = self.spawn_worker(turn.worker, turn.spec, turn.gpus)
    if h is None or (turn.kind == "snapshot" and h.exited()):
      return TurnResult(turn.id)
    async with self.gpu_cond:
      h.active += 1
    try:
      if turn.kind in ("train", "sample"):
        await self.claim_gpus(h, turn.gpus)
      with span(f"node.{turn.kind}"):
        res = await h.run(turn, self.store)
      if turn.kind == "close" or turn.args.get("close"):
        async with self.gpu_cond:
          if self.workers.get(turn.worker) is h:
            del self.workers[turn.worker]
      return res
    except asyncio.CancelledError:
      await asyncio.shield(self.drop_worker(turn.worker, h))
      raise
    except Exception as exc:
      await self.drop_worker(turn.worker, h)
      return failed(turn, exc)
    finally:
      async with self.gpu_cond:
        h.active -= 1
        if h.active == 0:
          h.claimed = ()
        self.gpu_cond.notify_all()

  async def clear_local(self, *extra: asyncio.Task | None) -> None:
    tasks = [t for t in (*(t for _, t in self.active.values()), *self.staging.values(), *extra) if t]
    self.active.clear()
    self.staging.clear()
    for t in tasks:
      t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    for wid, h in list(self.workers.items()):
      await self.drop_worker(wid, h)

  async def reset(self) -> None:
    self.staged_paths.clear()
    self.stage_errors.clear()
    await self.clear_local()
    self.boot_id, self.last_exchange = uuid.uuid4().hex[:12], time.monotonic()
    self.notify()

  async def close(self) -> None:
    self.draining = True
    await self.clear_local(self.sweeping)
    await asyncio.to_thread(self.store.close)


class ControlClient:
  """Async HTTP client for the controller's single `POST /internal/exchange` RPC."""

  def __init__(self, url: str | None = None):
    self.url = (getenv("CONTROL_URL", "http://127.0.0.1:8000") if url is None else url).rstrip("/")
    self.session = None

  async def aclose(self) -> None:
    if self.session and not self.session.closed:
      await self.session.close()
      self.session = None

  async def exchange(self, report: NodeReport, wait_ms: int = 0) -> NodeReply:
    import aiohttp

    if self.session is None or self.session.closed:
      self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
    t = aiohttp.ClientTimeout(total=max(5.0, wait_ms / 1000 + 5.0))
    url, params = f"{self.url}/internal/exchange", {"wait_ms": str(wait_ms)}
    async with self.session.post(url, params=params, data=msgspec.msgpack.encode(report), timeout=t) as r:
      if r.status >= 400:
        raise aiohttp.ClientError(f"control server returned {r.status}")
      return msgspec.msgpack.decode(await r.read(), type=NodeReply)


def connect(target: str | None = None) -> Bus | ControlClient:
  url = target if target is not None else os.getenv("CONTROL_URL", "")
  return ControlClient(url) if url.startswith(("http://", "https://")) else Bus(target)


if __name__ == "__main__":

  async def main():
    control = connect()
    daemon = NodeDaemon(control)
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, daemon.begin_drain)
    try:
      with contextlib.suppress(asyncio.CancelledError):
        await daemon.serve()
    finally:
      await control.aclose()

  asyncio.run(main())
