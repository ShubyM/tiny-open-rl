"""Node daemon: reconciles assigned workers, schedules local GPUs, stages checkpoints off-GPU, and runs commands."""

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

from tinyrl.control import scheduler
from tinyrl.control.accounting import Charge, Wave
from tinyrl.control.fairness import Demand, Worker, blocks, plan_gpus
from tinyrl.control.ledger import LEASE_MS, SWEEP_S, Ledger
from tinyrl.control.process import Process
from tinyrl.helpers import DEBUG, getenv
from tinyrl.storage.store import Store
from tinyrl.trace import span
from tinyrl.types import MAX_SEQS, Assignment, FencedError, Op, Reply, Role, Work, WorkerSpec
from tinyrl.workers.sampler import Sampler

GPUS, DRAIN_S, TICK, SWEEP_GRACE_S = getenv("GPUS", 1), getenv("DRAIN_S", 20.0), getenv("TICK", 0.25), 600
QUANTUM_MS, MAX_QUANTUM_MS = getenv("QUANTUM_MS", 2000), getenv("MAX_QUANTUM_MS", 15000)
SWITCH_BUDGET, CYCLE_GRACE_MS = float(getenv("SWITCH_BUDGET", 0.1)), getenv("CYCLE_GRACE_MS", 100)
_RPC = {"heartbeat": tuple[list[Assignment], list[str] | None], "take": list[Work], "finish": bool, "detach": bool}


def gpu_memory_mib() -> int:
  if override := getenv("GPU_MEMORY_MIB", 0):
    return int(override)
  with contextlib.suppress(OSError, subprocess.SubprocessError, ValueError):
    cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"]
    return int(subprocess.check_output(cmd, timeout=2, stderr=subprocess.DEVNULL))
  return 0


class WorkerHandle:
  """One local worker process, its GPU residency, and its per-run service history."""

  def __init__(self, wid: str, spec: WorkerSpec, model, resident=(), bound=()):
    self.id, self.spec, self.model = wid, spec, model
    self.resident, self.bound, self.target_gpus = tuple(resident), tuple(bound), ()
    self.hold_until = self.idle_since = self.park_s = self.gpu_ms = 0.0
    self.run_service, self.loaded, self.dirty, self.open_grad = {}, set(), set(), set()
    self.running, self.op_task = {}, None
    self.wave, self.loads, self.lock, self.broken = Wave(spec.gpus), {}, asyncio.Lock(), False

  def exited(self) -> bool:
    return bool((p := getattr(self.model, "proc", None)) and p.poll() is not None)

  def pid(self) -> int | None:
    return p.pid if (p := getattr(self.model, "proc", None)) and p.poll() is None else None

  def seed(self, runs) -> None:
    floor = min(self.run_service.values(), default=0.0)
    for r in runs:
      self.run_service.setdefault(r, floor)

  def bill(self, run: str, charge: Charge | None) -> None:
    if charge and charge.gpu_ms:
      self.gpu_ms += charge.gpu_ms
      self.run_service[run] = self.run_service.get(run, min(self.run_service.values(), default=0.0)) + charge.gpu_ms

  async def ensure_gpu(self, gpus: tuple[int, ...]) -> None:
    async with self.lock:
      if self.resident == gpus:
        return
      t0, prev = time.monotonic(), self.park_s
      await self.model.resume(gpus)
      now = time.monotonic()
      self.resident = gpus
      self.bound = self.bound or (gpus if self.spec.role == Role.SAMPLER or self.spec.gpus > 1 else ())
      sw = (prev + (now - t0)) / max(0.01, SWITCH_BUDGET) if prev else 0.0
      self.hold_until = now + min(MAX_QUANTUM_MS / 1000, max(QUANTUM_MS / 1000, sw))

  async def park(self) -> None:
    async with self.lock:
      if self.exited():
        raise RuntimeError("worker exited before park")
      t0 = time.monotonic()
      await self.model.park()
      self.resident, self.target_gpus, self.park_s = (), (), time.monotonic() - t0

  async def detach_run(self, run: str, *, keep_proc: bool = True) -> tuple[dict | None, bool]:
    async with self.lock:
      live = run in self.loaded and not self.exited()
      saved = await self.model.call("save_run", run, uuid.uuid4().hex) if live and run in self.dirty else None
      if live and keep_proc:
        await self.model.call("drop_run", run)
      lost = run in self.dirty and saved is None
      for s in (self.loaded, self.dirty, self.open_grad):
        s.discard(run)
      return saved, lost

  async def close(self) -> None:
    while True:
      try:
        await self.model.close()
        self.resident = self.target_gpus = ()
        return
      except Exception as exc:
        print(f"{self.id} could not close, retrying: {exc!r}", file=sys.stderr, flush=True)
        await asyncio.sleep(min(0.05, TICK))

  async def _load_shared(self, ckpt: tuple[str, str]) -> tuple[Charge, int, Exception | None]:
    if (entry := self.loads.get(ckpt)) is None:
      charge = Charge()

      async def do_load():
        try:
          await self.wave.run(self.model.load(ckpt), charge)
        except Exception as exc:
          return exc
        finally:
          self.loads.pop(ckpt, None)

      self.loads[ckpt] = entry = [asyncio.create_task(do_load()), charge, 0]
    entry[2] += 1
    err = await entry[0]
    return entry[1], entry[2], err

  async def execute(self, work: Work, gpus: tuple[int, ...], store: Store) -> tuple[Reply, Charge | None, dict]:
    cmd, charge = work.cmd, Charge()
    try:
      with span(f"node.{self.spec.role}"):
        await self.ensure_gpu(gpus)
        if self.spec.role == Role.SAMPLER:
          ckpt = load = err = None
          if work.version is not None:
            try:
              ckpt = await asyncio.to_thread(store.local_folder, work.version)
              load = await self._load_shared(ckpt)
              err = load[2]
            except Exception as exc:
              err = exc
          rep = Reply.error(err) if err else await self.wave.run(self.model.execute(cmd, ckpt, True, store), charge)
          if load is not None:
            charge.started_ms, charge.busy_ms = load[0].started_ms, charge.busy_ms + load[0].busy_ms
            charge.gpu_ms += load[0].gpu_ms / load[1]
          eff_out = {}
        else:
          async with self.lock:
            rep, eff = await self.wave.run(self.model.call("execute", cmd, work.run, work.snap, work.version), charge)
            loaded, mutated = bool(eff.get("loaded")), bool(eff.get("mutated"))
            (self.loaded.add if loaded else self.loaded.discard)(cmd.run)
            self.dirty.add(cmd.run) if mutated else (self.dirty.discard(cmd.run) if not loaded else None)
            (self.open_grad.add if cmd.op == Op.FORWARD_BACKWARD and rep.ok else self.open_grad.discard)(cmd.run)
            self.idle_since, eff_out = time.monotonic(), {"mutated": mutated, "saved": eff.get("saved")}
            if DEBUG:
              print(f"{self.id} {cmd.request_id} {cmd.op:24s} {charge.busy_ms:9d} ms")
        self.bill(cmd.run, charge)
        return rep, charge, eff_out
    except asyncio.CancelledError:
      raise
    except Exception as exc:
      self.broken = True
      return Reply.error(exc), None, {"dead": True}


class NodeDaemon:
  def __init__(self, control: Ledger | ControlClient, gpus: int = GPUS, node: str = "", gpu_mib=None, store=None):
    self.control, self.store = control, store if store is not None else Store()
    self.store.start()
    self.gpus, self.id, self.boot_id = gpus, node or f"{socket.gethostname()}-{os.getpid()}", uuid.uuid4().hex[:12]
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.workers, self.plan, self.staging, self.staged, self.stage_errors = {}, [], {}, {}, {}
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

  def _spawn(self, wid: str, spec: WorkerSpec, gpus: tuple[int, ...]) -> WorkerHandle:
    env = {**scheduler.settings(self.gpus, self.gpu_mib * 2**20), **os.environ, **self.store.env()}
    if spec.role == Role.SAMPLER:
      env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
      return WorkerHandle(wid, spec, self.open_sampler(wid, spec, env), bound=gpus)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(self.gpus) if spec.gpus == 1 else gpus))
    return WorkerHandle(wid, spec, self.open_trainer(wid, spec.gpus, env), bound=() if spec.gpus == 1 else gpus)

  def _bg(self, coro) -> asyncio.Task:
    t = asyncio.create_task(coro)
    t.add_done_callback(lambda _: self.notify())
    return t

  async def _call(self, name: str, *args, retry: bool = False, **kw):
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

  async def _run_work(
    self, w: WorkerHandle | None, wid: str, work: Work, gpus: tuple[int, ...], err: str | None
  ) -> None:
    try:
      bad = err or w is None
      rep, ch, eff = (Reply.error(RuntimeError(err)), None, {}) if bad else await w.execute(work, gpus, self.store)
      await self._call("finish", wid, work.cmd.request_id, rep, ch, retry=True, **eff)
    except FencedError:
      self.fenced = True
    finally:
      if w is not None and not (w.running.pop(work.cmd.request_id, None) and w.running):
        w.target_gpus = ()

  async def _op(self, w: WorkerHandle, kind: str, run: str = "") -> None:
    try:
      if kind == "park":
        await w.park()
      elif kind == "detach":
        await self._call("detach", w.id, run, *(await w.detach_run(run)), retry=True)
      else:
        for r in sorted(list(w.loaded if kind == "evict" else w.dirty)):
          res = (await w.detach_run(r, keep_proc=False)) if kind == "evict" else (None, True)
          await self._call("detach", w.id, r, *res, retry=True)
        await w.close()
        if self.workers.get(w.id) is w:
          del self.workers[w.id]
    except FencedError:
      self.fenced = True
    except Exception:
      if kind != "drop":
        await self._op(w, "drop")
    finally:
      w.op_task = None

  def _stage(self, ver: str) -> None:
    async def fetch():
      try:
        self.staged[ver] = (await asyncio.to_thread(self.store.local_folder, ver))[0]
      except Exception as exc:
        self.stage_errors[ver] = f"{type(exc).__name__}: {exc}"
      finally:
        self.staging.pop(ver, None)

    self.staging[ver] = self._bg(fetch())

  def _demand(self, a: Assignment, w: WorkerHandle | None, now: float, linger: list[float]) -> Worker:
    cur, active = ((w.resident or w.target_gpus) if w else ()), bool(w and w.running)
    if a.spec.role == Role.SAMPLER:
      demand = Demand.READY if (active or a.ready or any(v in self.staged for v in a.versions)) else Demand.IDLE
      hold = w.hold_until if w else 0.0
    else:
      grace = (w.idle_since + CYCLE_GRACE_MS / 1000) if (w and w.resident) else 0.0
      if ling := grace > now:
        linger.append(grace)
      step = active or bool(w and (w.open_grad & set(a.closing))) or (ling and not a.ready and bool(w and w.open_grad))
      demand = Demand.FINISH_STEP if step else (Demand.READY if (a.ready or ling) else Demand.IDLE)
      hold = float("inf") if active else (w.hold_until if w else 0.0)
    gpu_ms, gang = (w.gpu_ms if w else 0.0), a.spec.role == Role.TRAINER and a.spec.gpus > 1
    lag = a.lag if (a.ready or a.versions or active) else -gpu_ms
    return Worker(a.spec.gpus, a.spec.exclusive, demand, cur, w.bound if w else (), gang, hold, lag, gpu_ms)

  async def _reconcile(self) -> None:
    self.staged = {v: f for v, f in self.staged.items() if os.path.isdir(f)}
    now, desired = time.monotonic(), ({} if self.draining else {a.worker: a for a in self.plan})
    excl = {g for w in self.workers.values() if w.spec.exclusive for g in w.resident}
    for wid, w in list(self.workers.items()):
      if w.running or w.op_task:
        continue
      if w.broken or w.exited():
        w.op_task = self._bg(self._op(w, "drop"))
        continue
      blocked = not w.resident and bool(w.bound) and bool(excl & set(w.bound))
      can_move = blocked and w.spec.gpus > 1 and any(excl.isdisjoint(b) for b in blocks(self.gpus, w.spec.gpus))
      rebind = can_move and w.spec.role == Role.TRAINER and wid in desired and desired[wid].ready
      if wid not in desired or (blocked and w.spec.role == Role.SAMPLER) or rebind:
        w.op_task = self._bg(self._op(w, "evict"))
      elif w.spec.role == Role.TRAINER and (extra := sorted(w.loaded - set(desired[wid].runs))):
        w.op_task = self._bg(self._op(w, "detach", extra[0]))
    if self.draining:
      self.next_wake = None
      return

    cands, linger = {}, []
    for wid, a in sorted(desired.items()):
      for v in set(a.versions) - self.staged.keys() - self.stage_errors.keys() - self.staging.keys():
        self._stage(v)
      if not ((w := self.workers.get(wid)) and (w.broken or w.op_task)):
        cands[wid] = self._demand(a, w, now, linger)
    occupied = {
      g for k, w in self.workers.items() if k not in cands or w.running for g in (w.resident or w.target_gpus)
    }
    keep, wake_at = plan_gpus(cands, self.gpus, now, occupied)
    self.next_wake = min((d for d in (wake_at, *linger) if d is not None and d > now), default=None)
    for wid, w in list(self.workers.items()):
      if w.resident and not w.running and not w.op_task and keep.get(wid) != w.resident:
        w.op_task = self._bg(self._op(w, "park"))

    for wid, a in sorted(desired.items()):
      gpus, w = keep.get(wid, ()), self.workers.get(wid)
      ready_w = not (w and (w.op_task or w.broken or (w.resident and w.resident != gpus)))
      busy_gpus = {g for o in self.workers.values() if o is not w for g in (o.resident or o.target_gpus)}
      free = bool(gpus) and ready_w and busy_gpus.isdisjoint(gpus)
      if w and a.runs:
        w.seed(a.runs)
      svc, hold = (dict(w.run_service), tuple(w.running)) if w else ({}, ())
      if a.spec.role == Role.TRAINER:
        if not (free and a.ready and not hold):
          continue
        items = await self._call("take", wid, service=svc, loaded=tuple(w.loaded) if w else (), holding=hold)
      else:
        fl = [v for v in a.versions if v in self.stage_errors]
        st = tuple([*(v for v in a.versions if v in self.staged and free), *fl])
        slots = max(len(fl), MAX_SEQS - len(hold) if free else 0)
        if not ((free and a.ready and slots) or st):
          continue
        act = {"active_account": next((wk.account for wk, _ in w.running.values()), "") if w else ""}
        act["active_versions"] = tuple({wk.version for wk, _ in w.running.values()}) if w else ()
        items = await self._call("take", wid, slots=slots, service=svc, staged=st, holding=hold, **act)
      for item in items:
        err = self.stage_errors.pop(item.version, None) if item.version else None
        if err and w is None:
          self._bg(self._run_work(None, wid, item, (), err))
          continue
        if w is None:
          self.workers[wid] = w = self._spawn(wid, a.spec, gpus)
          w.seed(a.runs)
        if item.cmd.request_id not in w.running:
          if not err:
            w.target_gpus, w.hold_until = gpus, (w.hold_until if w.resident else time.monotonic() + QUANTUM_MS / 1000)
          w.running[item.cmd.request_id] = (item, self._bg(self._run_work(w, wid, item, gpus, err)))

  async def step_once(self, wait_ms: int = 0) -> bool:
    if self.fenced or (not self.draining and time.monotonic() - self.last_ok >= LEASE_MS / 1000):
      await self.reset()
    try:
      idle = 0 if (self.busy() or self.draining) else wait_ms
      if self.next_wake is not None and idle > 0:
        idle = min(idle, max(0, int((self.next_wake - time.monotonic()) * 1000)))
      pids = {k: p for k, w in self.workers.items() if (p := w.pid()) is not None}
      on = {k: w.resident for k, w in self.workers.items() if w.resident}
      due = not self.draining and time.time() - self.swept > SWEEP_S and (not self.sweeping or self.sweeping.done())
      info = {"memory": self.gpu_mib * 2**20, "draining": self.draining, "lag": self.lag, "pids": pids, "on": on}
      info |= {"wait_ms": idle, "live_workers": tuple(self.workers), "want_sweep": due}
      t0 = time.monotonic()
      self.plan, sweep = await self._call("heartbeat", self.gpus, **info)
      self.lag = max(0.0, time.monotonic() - t0 - idle / 1000)
      if due:
        self.swept = time.time()
        run_sweep = lambda: self.store.sweep(self.store.versions() - set(sweep), SWEEP_GRACE_S)  # noqa: E731
        self.sweeping = asyncio.create_task(asyncio.to_thread(self.store.trim_cache if sweep is None else run_sweep))
      await self._reconcile()
      return True
    except FencedError:
      await self.reset()
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
          delay = min(0.1 if not ok else 0.05, TICK)
          delay = delay if self.next_wake is None else min(delay, max(0.0, self.next_wake - time.monotonic()))
          with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(changed.wait(), delay)
    finally:
      await self.close()

  async def _clear(self, *extra) -> None:
    tasks = [t for w in self.workers.values() for t in (*(tk for _, tk in w.running.values()), w.op_task) if t]
    tasks += [*self.staging.values(), *(t for t in extra if t)]
    self.staging.clear()
    self.plan.clear()
    for t in tasks:
      t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    for w in list(self.workers.values()):
      await w.close()
    self.workers.clear()

  async def reset(self) -> None:
    self.staged, self.stage_errors = {}, {}
    await self._clear()
    self.fenced, self.boot_id, self.last_ok = False, uuid.uuid4().hex[:12], time.monotonic()
    self.notify()

  async def close(self) -> None:
    self.draining = True
    await self._clear(self.sweeping)
    await asyncio.to_thread(self.store.close)


class ControlClient:
  """Thin msgpack HTTP client for the controller's 4 node RPCs (`heartbeat`, `take`, `finish`, `detach`)."""

  def __init__(self, url: str | None = None):
    self.url = (getenv("CONTROL_URL", "http://127.0.0.1:8000") if url is None else url).rstrip("/")
    self.session = None

  async def aclose(self) -> None:
    if self.session and not self.session.closed:
      await self.session.close()

  def __getattr__(self, method: str):
    if method not in _RPC:
      raise AttributeError(method)

    async def rpc(*args, **kw):
      import aiohttp

      if self.session is None or self.session.closed:
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
      to = aiohttp.ClientTimeout(total=max(10.0, kw.get("wait_ms", 0) / 1000 + 5.0))
      body = msgspec.msgpack.encode((args, kw))
      async with self.session.post(f"{self.url}/internal/{method}", data=body, timeout=to) as r:
        if r.status == 409:
          raise FencedError(await r.text())
        r.raise_for_status()
        return msgspec.msgpack.decode(await r.read(), type=_RPC[method])

    return rpc


def connect(target: str | None = None) -> Ledger | ControlClient:
  url = target if target is not None else os.getenv("CONTROL_URL", "")
  return ControlClient(url) if url.startswith(("http://", "https://")) else Ledger(target)


if __name__ == "__main__":

  async def main():
    control, daemon = connect(), None
    try:
      daemon = NodeDaemon(control)
      asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, daemon.begin_drain)
      with contextlib.suppress(asyncio.CancelledError):
        await daemon.serve()
    finally:
      await control.aclose()

  asyncio.run(main())
