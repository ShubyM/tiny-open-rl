"""Node daemon: executes concrete turns from the controller's single `exchange` RPC.

Owns process supervision, physical GPU handoff (`prepare`), off-GPU checkpoint staging (`prefetch`), periodic cache
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
from collections import Counter

from tinyrl.bus import LEASE_MS, connect
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


class WorkerHandle:
  def __init__(self, wid: str, spec: WorkerSpec, model, resident=(), bound=()):
    self.id, self.spec, self.model, self.resident, self.bound = wid, spec, model, resident, bound
    self.loaded, self.busy, self.parking, self.park_s = set(), False, False, 0.0

  def exited(self) -> bool:
    return bool((p := getattr(self.model, "proc", None)) and p.poll() is not None)

  def pid(self) -> int | None:
    return p.pid if (p := getattr(self.model, "proc", None)) and p.poll() is None else None


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
    finished = [
      TurnResult(tid, t.kind, t.worker, False, Reply.error(RuntimeError("cancelled")))
      if tk.cancelled()
      else tk.result()
      for tid, (t, tk) in list(self.active.items())
      if tk.done()
    ]
    for ver, tk in list(self.staging.items()):
      if tk.done() and not (self.staging.pop(ver) and tk.cancelled()):
        if isinstance(res := tk.result(), Exception):
          self.stage_errors[ver] = f"{type(res).__name__}: {res}"
        else:
          self.staged_paths[ver] = res
    staged = [v for v, (f, _) in list(self.staged_paths.items()) if os.path.isdir(f) or not self.staged_paths.pop(v)]
    stage_errors, self.stage_errors, statuses = self.stage_errors, {}, []
    for h in list(self.workers.values()):
      if h.exited() and not h.busy and not h.parking:
        h.parking = True
        asyncio.create_task(self.drop_handle(h))
        continue
      statuses.append(WorkerStatus(h.id, h.resident, h.bound, h.pid(), tuple(sorted(h.loaded)), h.park_s))
    sweep = not self.draining and time.time() - self.swept > SWEEP_S and (not self.sweeping or self.sweeping.done())
    args = (self.id, self.boot_id, self.gpus, self.gpu_mib * 2**20, self.draining, self.lag)
    return NodeReport(*args, statuses, finished, staged, stage_errors, sweep, list(self.active))

  def start_prefetch(self, version: str) -> None:
    if version in self.staged_paths or version in self.staging:
      return

    async def stage():
      with span("node.prepare"):
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

  # ***** physical GPU handoff and turn execution *****

  async def release(self, h: WorkerHandle, *, drop: bool = False, park: bool = False) -> None:
    async with self.gpu_cond:
      if drop:
        self.workers.pop(h.id, None)
        h.resident, h.busy, h.parking = (), False, False
      elif park:
        h.resident, h.parking = (), False
      else:
        h.busy = False
      self.gpu_cond.notify_all()

  async def drop_handle(self, handle: WorkerHandle) -> None:
    while True:
      try:
        with span("node.close"):
          await handle.model.close()
        break
      except Exception as exc:
        print(f"{handle.id} could not close, retrying: {exc!r}", file=sys.stderr, flush=True)
        await asyncio.sleep(min(0.05, TICK))
    await self.release(handle, drop=True)

  async def park_handle(self, handle: WorkerHandle) -> None:
    if handle.exited():
      return await self.drop_handle(handle)
    t0 = time.monotonic()
    try:
      with span("node.park"):
        await handle.model.park()
      handle.park_s = time.monotonic() - t0
      await self.release(handle, park=True)
    except Exception:
      await self.drop_handle(handle)

  async def prepare(self, wid: str, spec: WorkerSpec, gpus: tuple[int, ...]) -> tuple[WorkerHandle, float, float]:
    """Ensure `wid` is resident and busy on `gpus`, parking overlapping residents first."""
    while True:
      async with self.gpu_cond:
        h = self.workers.get(wid)
        overlap = [
          o for o in self.workers.values() if o.id != wid and o.resident and not set(o.resident).isdisjoint(gpus)
        ]
        if (h and (h.busy or h.parking)) or any(o.busy or o.parking for o in overlap):
          await self.gpu_cond.wait()
          continue
        need_park = h and (h.exited() or (h.resident and h.resident != gpus))
        if to_park := overlap + ([h] if need_park else []):
          for o in to_park:
            o.parking = True
        else:
          if h is None:
            self.workers[wid] = h = self.spawn_worker(wid, spec, gpus)
          warm = h.resident == gpus and not getattr(h.model, "warm_pending", False)
          h.busy, h.resident, park_s = True, gpus, h.park_s
          if not h.bound and (spec.role == Role.SAMPLER or spec.gpus > 1):
            h.bound = gpus
          break
      for o in to_park:
        await self.park_handle(o)
    resume_s = 0.0
    if not warm:
      t0 = time.monotonic()
      try:
        with span("node.resume"):
          await h.model.resume(gpus)
        resume_s = time.monotonic() - t0
      except Exception:
        await self.drop_handle(h)
        raise
    return h, resume_s, park_s

  async def execute_turn(self, turn: Turn) -> TurnResult:
    try:
      if turn.kind in ("train", "sample"):
        return await self.run_gpu_turn(turn)
      if turn.kind == "snapshot":
        if not (h := self.workers.get(turn.worker)) or h.exited():
          return TurnResult(turn.id, "snapshot", turn.worker)
        rid, ver = turn.args["run"], turn.args.get("version")
        saved = await h.model.call("save_run", rid, ver) if ver and rid in h.loaded else None
        if turn.args.get("drop") and rid in h.loaded:
          await h.model.call("drop_run", rid)
        h.loaded.discard(rid)
        if turn.args.get("close"):
          await self.drop_handle(h)
        return TurnResult(turn.id, "snapshot", turn.worker, saved=saved)
      if turn.kind == "close":
        async with self.gpu_cond:
          while (h := self.workers.get(turn.worker)) and (h.busy or h.parking):
            await self.gpu_cond.wait()
          if h is None:
            return TurnResult(turn.id, "close", turn.worker)
          h.busy = True
        await self.drop_handle(h)
        return TurnResult(turn.id, "close", turn.worker)
      raise ValueError(f"unknown turn kind {turn.kind}")
    except asyncio.CancelledError:
      raise
    except Exception as exc:
      return TurnResult(turn.id, turn.kind, turn.worker, False, Reply.error(exc))

  async def run_gpu_turn(self, turn: Turn) -> TurnResult:
    h, resume_s, park_s = await self.prepare(turn.worker, turn.spec, turn.gpus)
    charge = Charge()
    try:
      if turn.kind == "train":
        with span("node.execute"):
          call = h.model.call(
            "execute", turn.cmd, turn.args.get("record") or None, turn.args.get("saved"), turn.args.get("version")
          )
          rep, eff = await Wave(turn.spec.gpus).run(call, charge)
        loaded = bool(eff.get("loaded"))
        (h.loaded.add if loaded else h.loaded.discard)(turn.cmd.run)
        if DEBUG:
          print(f"{turn.worker} {turn.cmd.request_id} {turn.cmd.op:24s} {charge.busy_ms:9d} ms")
        mut, saved = bool(eff.get("mutated")), eff.get("saved")
        return TurnResult(
          turn.id, "train", turn.worker, True, rep, charge, mut, loaded, saved, resume_s=resume_s, park_s=park_s
        )
      if not turn.cmds:
        return TurnResult(turn.id, "sample", turn.worker, resume_s=resume_s, park_s=park_s)
      inputs, errors = {}, {}
      for rid, ver in turn.args.get("versions", {}).items():
        if ver is None:
          inputs[rid] = None
        else:
          try:
            inputs[rid] = await asyncio.to_thread(self.store.local_folder, ver)
          except Exception as exc:
            inputs[rid], errors[ver] = ver, exc
      wave, uses, loads, results = Wave(turn.spec.gpus), Counter(inputs[c.run] for c in turn.cmds), {}, []
      for ckpt in uses:
        if ckpt is not None and ckpt not in errors:
          loads[ckpt] = load = Charge()
          try:
            await wave.run(h.model.load(ckpt), load)
          except Exception as exc:
            errors[ckpt] = exc

      async def answer(cmd) -> None:
        c, inp = Charge(), inputs[cmd.run]
        with span("node.execute"):
          rep = Reply.error(err) if (err := errors.get(inp)) else await wave.run(h.model.execute(cmd, inp, True), c)
          if load := loads.get(inp):
            c.started_ms, c.busy_ms = load.started_ms, c.busy_ms + load.busy_ms
            c.gpu_ms += load.gpu_ms / uses[inp]
        results.append((cmd.request_id, rep, c))

      async with asyncio.TaskGroup() as group:
        for cmd in turn.cmds:
          group.create_task(answer(cmd))
      return TurnResult(turn.id, "sample", turn.worker, samples=results, resume_s=resume_s, park_s=park_s)
    except asyncio.CancelledError:
      await asyncio.shield(self.drop_handle(h))
      raise
    except Exception as exc:
      await self.drop_handle(h)
      c = charge if turn.kind == "train" else None
      return TurnResult(turn.id, turn.kind, turn.worker, False, Reply.error(exc), c, resume_s=resume_s, park_s=park_s)
    finally:
      await self.release(h)

  async def clear_local(self, *extra: asyncio.Task | None) -> None:
    tasks = [t for t in (*(t for _, t in self.active.values()), *self.staging.values(), *extra) if t]
    self.active.clear()
    self.staging.clear()
    for t in tasks:
      t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    for h in list(self.workers.values()):
      await self.drop_handle(h)

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
