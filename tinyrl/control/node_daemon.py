"""Node daemon: executes fleet assignments, manages local processes, storage preparation, and physical GPU release.

The fleet scheduler decides node/GPU placement, intent (`run` | `park` | `drain`), bounded budgets, job generations,
and restore/prefetch inputs. This daemon enforces those contracts locally, never overlaps incompatible GPU holders,
confirms GPU release via observations, restores jobs explicitly before executing commands, exposes rollbacks past the
last durable checkpoint, waits for explicit save durability off-GPU, and handles per-job drains in shared workers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field

from tinyrl.bus import LEASE_MS, Bus, Cmd, Ev, Op, Reply, Role, id_ge
from tinyrl.control import scheduler
from tinyrl.control.process import Process
from tinyrl.helpers import DEBUG, getenv
from tinyrl.storage.registry import Registry
from tinyrl.storage.store import Store, parse_version
from tinyrl.trace import span
from tinyrl.workers.sampler import Sampler, SamplingAbortedError

GPUS, DRAIN_S, TICK = getenv("GPUS", 1), getenv("DRAIN_S", 20.0), getenv("TICK", 0.25)
IDLE_MS, RUN_COMMANDS = getenv("IDLE_MS", 150), getenv("RUN_COMMANDS", 4)
SAMPLE_RUNS, SAMPLE_COMMANDS = getenv("SAMPLE_RUNS", 3), getenv("SAMPLE_COMMANDS", 128)
SWEEP_S, SWEEP_GRACE_S, PARK_TIMEOUT_S, READ_REF_TTL = (
  getenv("SWEEP_S", 60.0),
  600,
  float(getenv("PARK_TIMEOUT_S", 60.0)),
  300.0,
)


def gpu_memory_mib() -> int:
  if override := getenv("GPU_MEMORY_MIB", 0):
    return int(override)
  try:
    cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"]
    return int(subprocess.check_output(cmd, timeout=2, stderr=subprocess.DEVNULL))
  except (OSError, subprocess.SubprocessError, ValueError):
    return 0


def host_ram_bytes() -> int:
  if override := getenv("HOST_RAM_BYTES", 0):
    return int(override)
  with contextlib.suppress(OSError, ValueError):
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
  return 0


def disk_free_bytes(path: str) -> int:
  with contextlib.suppress(OSError):
    return shutil.disk_usage(path).free
  return 0


def _empty_obs(rev: int, phase: str) -> dict:
  return {
    "rev": rev,
    "owner": "",
    "pid": None,
    "gpus": [],
    "gpus_held": [],
    "gpu_released": True,
    "phase": phase,
    "host_ram_bytes": 0,
    "jobs": {},
  }


@contextlib.asynccontextmanager
async def _read_ref(registry: Registry, node_id: str, ver_id: str):
  tok = f"read:{node_id}:{uuid.uuid4().hex[:8]}"
  acq = await registry.acquire_ref(ver_id, tok, ttl=READ_REF_TTL) if ver_id else False
  try:
    yield acq
  finally:
    if acq:
      with contextlib.suppress(Exception):
        await registry.release_ref(ver_id, tok)


@dataclass
class Runtime:
  spec: scheduler.WorkerSpec
  owner: str = ""
  rev: int = 0
  pid: int | None = None
  gpus: tuple[int, ...] = ()
  gpus_held: tuple[int, ...] = ()
  gpu_released: bool = True
  phase: str = "idle"
  host_ram_bytes: int = 0
  jobs: dict[str, dict] = field(default_factory=dict)

  def obs(self) -> dict:
    return {
      "rev": self.rev,
      "owner": self.owner,
      "pid": self.pid,
      "gpus": list(self.gpus),
      "gpus_held": list(self.gpus_held),
      "gpu_released": self.gpu_released,
      "phase": self.phase,
      "host_ram_bytes": self.host_ram_bytes,
      "jobs": {k: dict(v) for k, v in self.jobs.items()},
    }


class Prefetcher:
  """Assignment-driven storage prefetcher with deduplication, cancellation, and active-reader refs."""

  def __init__(self, node_id: str, store: Store, registry: Registry, on_ready):
    self.node_id, self.store, self.registry, self.on_ready = node_id, store, registry, on_ready
    self.tasks: dict[str, tuple[asyncio.Task, threading.Event, int]] = {}

  def reconcile(self, state: scheduler.Assignment) -> None:
    desired: dict[str, tuple[int, str, list[str]]] = {}
    for w in state.workers.values():
      if w.node != self.node_id or w.intent == "drain":
        continue
      if w.intent == "run":
        for ref in w.restore_refs.values():
          if (ver := parse_version(ref.get("version"))[0]) and ref.get("uri"):
            desired[ver] = (0, ref["uri"], list(ref.get("peers") or ()))
      for item in w.prefetch:
        if (ver := parse_version(item.get("version"))[0]) and item.get("uri") and ver not in desired:
          desired[ver] = (1, item["uri"], list(item.get("peers") or ()))

    for ver in list(self.tasks):
      if ver not in desired:
        task, cancel, _ = self.tasks.pop(ver)
        cancel.set()
        task.cancel()

    ready_local = {v for v, info in self.store.transfer_status().items() if info.get("ready")}
    for ver, (prio, uri, peers) in sorted(desired.items(), key=lambda kv: (kv[1][0], kv[0])):
      if ver not in ready_local and ver not in self.tasks:
        cancel = threading.Event()
        self.tasks[ver] = (asyncio.create_task(self._fetch(uri, ver, peers, cancel)), cancel, prio)

  async def _fetch(self, uri: str, ver: str, peers: list[str], cancel: threading.Event) -> None:
    try:
      async with _read_ref(self.registry, self.node_id, ver):
        await asyncio.to_thread(self.store.local_folder, uri, ver, peers, cancel)
        await self.registry.add_location(ver, self.node_id, self.store.peer)
        self.on_ready()
    except (asyncio.CancelledError, InterruptedError, Exception):
      pass
    finally:
      if (cur := self.tasks.get(ver)) and cur[0] is asyncio.current_task():
        self.tasks.pop(ver, None)

  async def close(self) -> None:
    for task, cancel, _ in list(self.tasks.values()):
      cancel.set()
      task.cancel()
    if self.tasks:
      await asyncio.gather(*(t for t, _, _ in self.tasks.values()), return_exceptions=True)
    self.tasks.clear()


class NodeDaemon:
  def __init__(
    self,
    bus: Bus,
    gpus: int = GPUS,
    node: str = "",
    gpu_mib: int | None = None,
    store: Store | None = None,
    host_ram: int | None = None,
  ):
    self.bus, self.registry = bus, Registry(bus)
    self.store = store if store is not None else Store()
    self.store.on_durable = self._on_durable_sync
    self.store.start()
    self.gpus, self.id = gpus, node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.host_ram = host_ram_bytes() if host_ram is None else host_ram
    self.gpu_class = os.environ.get("GPU_CLASS", "")
    self.running: dict[str, Runtime] = {}
    self.tombstones: dict[str, dict] = {}
    self.tasks: dict[str, asyncio.Task] = {}
    self.bg_tasks: set[asyncio.Task] = set()
    self.pending_durable: dict[str, tuple[str, str, str]] = {}
    self.state = scheduler.Assignment()
    self.changed, self.draining = asyncio.Event(), False
    self.heartbeat, self.lag, self.swept, self.sweeping = time.monotonic(), 0.0, 0.0, None
    self._loop: asyncio.AbstractEventLoop | None = None
    self.prefetcher = Prefetcher(self.id, self.store, self.registry, self._schedule_wake_heartbeat)

  def notify(self) -> None:
    changed, self.changed = self.changed, asyncio.Event()
    changed.set()

  def _spawn_bg(self, coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    self.bg_tasks.add(task)
    task.add_done_callback(self.bg_tasks.discard)
    return task

  def _schedule_wake_heartbeat(self) -> None:
    if self._loop and not self._loop.is_closed():
      self._spawn_bg(self.push_heartbeat(wake_scheduler=True))
    self.notify()

  def _on_durable_sync(self, uri: str, ver: str) -> None:
    if self._loop and not self._loop.is_closed():
      with contextlib.suppress(RuntimeError):
        self._loop.call_soon_threadsafe(self._spawn_bg, self._handle_durable(uri, ver))

  async def _handle_durable(self, uri: str, ver: str) -> None:
    if not (ver_id := parse_version(ver)[0]):
      return
    with contextlib.suppress(Exception):
      await self.registry.mark_durable(ver_id)
      if info := self.pending_durable.pop(ver_id, None):
        await self.bus.promote_durable_state(info[0], info[1], ver_id, info[2])

  async def _register_saved(self, saved: dict, durable: bool | None = None) -> None:
    await self.registry.register_checkpoint(
      uri=saved["uri"],
      size=saved["size"],
      ttl=saved["ttl"],
      owned=saved["owned"],
      version=saved["version"],
      durable=saved["durable"] if durable is None else durable,
      node=self.id,
      meta=saved.get("meta"),
    )

  async def read_state(self) -> scheduler.Assignment:
    return scheduler.from_table(await self.bus.read_table())

  def _heartbeat_workers(self) -> dict[str, dict]:
    out = {n: r.obs() for n, r in self.running.items()}
    for n, tomb in self.tombstones.items():
      out.setdefault(n, dict(tomb))
    for n, w in self.state.workers.items():
      if w.node == self.id and n not in out and w.intent in ("drain", "park"):
        out[n] = _empty_obs(w.rev, "drained" if w.intent == "drain" else "idle")
    return out

  async def push_heartbeat(self, wake_scheduler: bool = False) -> None:
    workers = self._heartbeat_workers()
    cache_path = self.store.cache if self.store.remote else self.store.base
    await self.bus.node_heartbeat(
      self.id,
      self.gpus,
      self.gpu_mib * 2**20,
      workers,
      self.lag,
      wake_scheduler=wake_scheduler,
      host_ram=self.host_ram,
      host_ram_used=sum(w.get("host_ram_bytes", 0) for w in workers.values()),
      disk_free=disk_free_bytes(cache_path),
      gpu_class=self.gpu_class,
      peer=self.store.peer,
      versions=self.store.transfer_status(),
    )
    self.heartbeat = time.monotonic()

  async def announce_node(self) -> None:
    self._loop = asyncio.get_running_loop()
    await self.push_heartbeat(wake_scheduler=True)
    await self.bus.append_event(Ev.NODE_UP, node=self.id, gpus=self.gpus, memory=self.gpu_mib * 2**20)

  def _can_activate_locally(self, worker: str, gpus: tuple[int, ...]) -> bool:
    """Defense-in-depth physical GPU overlap guard before activating a worker on physical GPUs."""
    r, gpu_bytes = self.running[worker], self.gpu_mib * 2**20
    for g in gpus:
      holders = [o for wn, o in self.running.items() if wn != worker and g in o.gpus_held]
      if not holders:
        continue
      if r.spec.exclusive or any(h.spec.exclusive for h in holders):
        return False
      if not scheduler.fits(
        [scheduler.memory_budget(self.state, h.spec, self.id) for h in holders]
        + [scheduler.memory_budget(self.state, r.spec, self.id)],
        gpu_bytes,
      ):
        return False
    return True

  def open_trainer(self, worker: str, env: dict[str, str]) -> Process:
    width, cmd = max(1, self.running[worker].spec.gpus), [sys.executable]
    if width > 1:
      cmd += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={width}"]
    return Process([*cmd, "-m", "tinyrl.workers.trainer"], env)

  def open_sampler(self, worker: str, env: dict[str, str]) -> Sampler:
    spec = self.running[worker].spec
    return Sampler(spec.base, spec.lora, env)

  def start(self, worker: str) -> asyncio.Task:
    task = self.tasks[worker] = asyncio.create_task(self.work(worker))
    return task

  async def work(self, worker: str) -> None:
    w_init = self.state.workers[worker]
    r = Runtime(
      w_init.spec, owner=f"{self.id}-{os.urandom(8).hex()}", rev=w_init.rev, gpus=w_init.gpus, phase="starting"
    )
    task, owner, sampling = asyncio.current_task(), r.owner, r.spec.role == Role.SAMPLER
    heads, pending_replay, prepared, loaded_runs, last_completed, durable_saves = [], [], {}, set(), {}, set()
    reader = guard = model = closing = last_run = failed_exc = None
    renewed, turn_commands, grant_rev_seen, grant_usage, grant_cmds, local_svc, started = (
      time.monotonic(),
      0,
      -1,
      {},
      0,
      {},
      False,
    )

    def cur_w() -> scheduler.Worker | None:
      return self.state.workers.get(worker)

    def job_gen(run_id: str) -> int:
      w = cur_w()
      if w and run_id in w.generations:
        return w.generations[run_id]
      return self.state.jobs[run_id].gen if run_id in self.state.jobs else 0

    def active_runs() -> list[str]:
      draining = w.drain_jobs if (w := cur_w()) else {}
      return [j for j in scheduler.jobs_of(self.state, worker) if j not in draining]

    def job_entry(run_id: str) -> dict:
      return r.jobs.setdefault(
        run_id, {"gen": job_gen(run_id), "usage": 0.0, "tokens": 0, "completed_id": "", "state": "active"}
      )

    def record_usage(run_id: str, gpu_sec: float, cost: int) -> None:
      je = job_entry(run_id)
      je["usage"] += gpu_sec
      je["tokens"] += cost
      je["gen"] = job_gen(run_id)
      grant_usage[run_id] = grant_usage.get(run_id, 0.0) + gpu_sec
      local_svc[run_id] = local_svc.get(run_id, 0.0) + cost

    async def owned():
      nonlocal renewed
      before = time.monotonic()
      if not (
        await self.bus.renew_lease(worker, owner)
        or (not closing and await self.bus.take_lease(worker, owner, recover=True))
      ):
        if not closing and asyncio.current_task() is not task and not task.cancelling():
          task.cancel()
        raise RuntimeError(f"lost worker {worker}")
      renewed = before

    async def keepalive():
      while True:
        await asyncio.sleep(LEASE_MS / 4000)
        if not closing and not self.draining and time.monotonic() - self.heartbeat >= LEASE_MS / 1000:
          task.cancel()
          return
        try:
          await owned()
          if not closing and model and (p := model.proc) and p.poll() is not None:
            task.cancel()
            return
        except Exception:
          if not closing and time.monotonic() - renewed >= LEASE_MS / 1000:
            task.cancel()
            return

    async def prepare(cmd: Cmd):
      with span("node.prepare"):
        try:
          run, checkpoint = await self.bus.get_run(cmd.run), None
          if sampling:
            if run is None:
              raise ValueError(f"no sampling session {cmd.run}")
            if uri := run.get("model_path"):
              ref = (w.restore_refs.get(cmd.run) if (w := cur_w()) else None) or {}
              ver_hint = ref.get("version") or await self.registry.version(uri)
              ver_id, peers, _ = await self.registry.resolve_version(uri, ver_hint)
              async with _read_ref(self.registry, self.id, ver_id):
                checkpoint = await asyncio.to_thread(self.store.local_folder, uri, ver_id or ver_hint, peers)
          return run, checkpoint
        except Exception as e:
          return e

    async def read(delay: float = 0):
      await asyncio.sleep(delay)
      commands = await self.bus.peek_commands(active_runs(), IDLE_MS)
      for cmd in commands:
        if cmd.request_id not in prepared:
          (prepared.setdefault(cmd.request_id, asyncio.create_task(prepare(cmd)))).add_done_callback(
            lambda _: self.notify()
          )
      return commands

    def collect():
      nonlocal heads, reader
      if reader and reader.done():
        heads, reader = reader.result(), None
      allowed = set(active_runs())
      heads = [c for c in heads if c.run in allowed]
      current = {c.request_id for c in heads}
      for request in list(prepared):
        if request not in current:
          prepared.pop(request).cancel()

    def ready() -> list[Cmd]:
      return [c for c in heads if c.request_id in prepared and prepared[c.request_id].done()]

    async def commit_parked(run_id: str, saved: dict) -> None:
      await self._register_saved(saved)
      boundary = saved.get("completed_id") or last_completed.get(run_id, "")
      ok = await self.bus.commit_job_state(
        run_id,
        saved["uri"],
        saved["version"],
        boundary,
        durable=saved["durable"],
        worker=worker,
        owner=owner,
        generation=job_gen(run_id),
        set_parked=True,
      )
      if not ok:
        raise RuntimeError(f"lost generation for {run_id}")
      if not saved["durable"] and (vid := parse_version(saved["version"])[0]):
        self.pending_durable[vid] = (run_id, saved["uri"], boundary)

    async def ensure_restored(cmd: Cmd, run: dict | None, gen: int) -> Reply | None:
      """Explicitly restore a training job from its committed version/boundary, or expose rollback past durable state."""
      if sampling or cmd.op in (Op.CREATE_MODEL, Op.DELETE_MODEL) or cmd.run in loaded_runs:
        return None
      if await model.call("has_run", cmd.run):
        loaded_runs.add(cmd.run)
        return None
      if run is None:
        return Reply.error(KeyError(f"no training run {cmd.run}"))
      if cmd.op == Op.LOAD_WEIGHTS:
        return None
      rollback_msg = (
        f"model {cmd.run} lost its unsaved steps past durable checkpoint; call load_state to restore explicitly"
      )
      if run.get("rollback_required"):
        return Reply.error(KeyError(rollback_msg))

      last_mut, candidates = run.get("last_mutating_id") or "", []
      for u_k, v_k, b_k in (
        ("restore_uri", "restore_version", "restore_boundary"),
        ("durable_uri", "durable_version", "durable_boundary"),
      ):
        if run.get(u_k) and run.get(v_k):
          candidates.append((run[u_k], run[v_k], run.get(b_k) or ""))
      if not candidates and (parked := await self.bus.parked_checkpoint(cmd.run)):
        candidates.append((parked, (await self.registry.version(parked)) or "", last_mut))

      for uri, ver_ref, boundary in candidates:
        if last_mut and not id_ge(boundary, last_mut):
          continue
        ver_id, peers, _ = await self.registry.resolve_version(uri, ver_ref)
        try:
          async with _read_ref(self.registry, self.id, ver_id):
            await owned()
            await model.call("restore_job", cmd.run, run, uri, ver_id or ver_ref or None, peers, True, True)
            loaded_runs.add(cmd.run)
            last_completed[cmd.run] = boundary or last_mut
            if await self.bus.parked_checkpoint(cmd.run) == uri:
              await self.bus.set_parked_checkpoint(cmd.run, None, worker, owner, generation=gen)
              await self.registry.unregister_checkpoints(uri)
            return None
        except Exception:
          continue

      if last_mut:
        await self.bus.mark_rollback_required(cmd.run, worker, owner, generation=gen)
      return Reply.error(KeyError(rollback_msg))

    async def finish_durable_save(cmd: Cmd, reply: Reply, saved: dict, before: float, gen: int) -> None:
      """Complete explicit save_state only after durable storage commit, outside GPU ownership."""
      ver_id, _ = parse_version(saved["version"])
      try:
        if not saved["durable"] and not await asyncio.to_thread(
          self.store.wait_durable, saved["uri"], saved["version"]
        ):
          raise RuntimeError(f"durable upload did not complete for {saved['uri']}@{ver_id}")
        await self.registry.mark_durable(saved["version"])
        await self._register_saved(saved, durable=True)
        await self.bus.commit_job_state(
          cmd.run,
          saved["uri"],
          saved["version"],
          saved.get("completed_id") or cmd.entry_id,
          durable=True,
          worker=worker,
          owner=owner,
          generation=gen,
        )
        await self.bus.write_reply(cmd, reply, worker, owner, started=before, generation=gen)
      except Exception as exc:
        with contextlib.suppress(Exception):
          await self.bus.write_reply(cmd, Reply.error(exc), worker, owner, started=before, generation=gen)

    async def answer(cmd: Cmd, inputs) -> Reply | None:
      before, gen = time.time(), job_gen(cmd.run)
      with span("node.execute"):
        if isinstance(inputs, Exception):
          reply = Reply.error(inputs)
        else:
          run, checkpoint = inputs
          if sampling:
            await owned()
            try:
              reply = await model.execute(cmd, run, checkpoint)
            except SamplingAbortedError:
              return None
          elif (restore_err := await ensure_restored(cmd, run, gen)) is not None:
            reply = restore_err
          else:
            version, peers = None, []
            if cmd.op == Op.LOAD_WEIGHTS:
              uri = json.loads(cmd.body)["uri"]
              ver_id, peers, _ = await self.registry.resolve_version(uri)
              version = ver_id or await self.registry.version(uri)
            source_ver = (run or {}).get("restore_version") or (run or {}).get("durable_version") or None
            await owned()
            reply, effects = await model.call("execute", cmd, run, None, version, peers, source_ver)
            await owned()
            if reply.ok:
              if cmd.op in (Op.CREATE_MODEL, Op.LOAD_WEIGHTS):
                loaded_runs.add(cmd.run)
              elif cmd.op == Op.DELETE_MODEL:
                loaded_runs.discard(cmd.run)
              last_completed[cmd.run] = job_entry(cmd.run)["completed_id"] = cmd.entry_id
            if saved := effects.get("saved"):
              await self._register_saved(saved)
              if cmd.op == Op.SAVE_WEIGHTS and reply.ok:
                if saved["durable"]:
                  await self.bus.commit_job_state(
                    cmd.run,
                    saved["uri"],
                    saved["version"],
                    saved.get("completed_id") or cmd.entry_id,
                    durable=True,
                    worker=worker,
                    owner=owner,
                    generation=gen,
                  )
                else:
                  dt = asyncio.create_task(finish_durable_save(cmd, reply, saved, before, gen))
                  durable_saves.add(dt)
                  dt.add_done_callback(durable_saves.discard)
                  return reply
      with span("node.publish"):
        if not await self.bus.write_reply(cmd, reply, worker, owner, started=before, generation=gen):
          if job_gen(cmd.run) == gen:
            raise RuntimeError(f"lost worker {worker}")
          return None
      if DEBUG:
        print(f"{worker} {cmd.request_id} {cmd.op:24s} {(time.time() - before) * 1000:9.2f} ms")
      return reply

    async def process_drain_jobs() -> None:
      if not (w := cur_w()) or not w.drain_jobs:
        return
      for jid in list(w.drain_jobs):
        je = job_entry(jid)
        if je["state"] == "drained":
          continue
        je["state"] = "draining"
        if durable_saves:
          await asyncio.gather(*list(durable_saves), return_exceptions=True)
        if not sampling and jid in loaded_runs:
          await owned()
          saved = await model.call("park_run", jid, last_completed.get(jid, ""))
          loaded_runs.discard(jid)
          if saved:
            saved.pop("run", None)
            await commit_parked(jid, saved)
        je["state"] = "drained"
        r.rev = max(r.rev, w.rev)
        await self.push_heartbeat(wake_scheduler=True)

    async def park_gpus() -> None:
      stalled = bool(heads) and not ready() and not pending_replay and any(not t.done() for t in prepared.values())
      if not r.gpus_held:
        r.gpu_released = True
        if r.phase not in ("parked", "draining", "drained", "failed", "preparing"):
          r.phase = "preparing" if stalled else "parked"
        return
      r.phase = "parking"
      with span("node.park"):
        if sampling:
          await asyncio.wait_for(model.pause("wait"), timeout=PARK_TIMEOUT_S)
          await asyncio.wait_for(model.park(), timeout=PARK_TIMEOUT_S)
        else:
          await asyncio.wait_for(model.call("park"), timeout=PARK_TIMEOUT_S)
          r.host_ram_bytes = int(await model.call("host_ram_bytes"))
      r.gpus_held, r.gpu_released, r.phase = (), True, ("preparing" if stalled else "parked")
      if w := cur_w():
        r.rev = max(r.rev, w.rev)
      self.notify()
      await self.push_heartbeat(wake_scheduler=True)

    async def activate_gpus(target_gpus: tuple[int, ...]) -> bool:
      nonlocal grant_rev_seen, grant_usage, grant_cmds
      if not target_gpus:
        return False
      if r.gpus_held == target_gpus:
        return True
      if r.gpus_held:
        if not scheduler.is_movable(r.spec, self.gpus):
          return False
        await park_gpus()
      if not self._can_activate_locally(worker, target_gpus):
        return False
      await owned()
      r.gpus, r.gpus_held, r.gpu_released, r.phase = target_gpus, target_gpus, False, "running"
      if w := cur_w():
        r.rev = max(r.rev, w.rev)
        if w.grant_rev != grant_rev_seen:
          grant_rev_seen, grant_usage, grant_cmds = w.grant_rev, {}, 0
      try:
        with span("node.resume"):
          if sampling:
            if model.running is None:
              await model.start()
              r.pid = model.proc.pid if model.proc else None
            else:
              await model.resume()
              await model.resume_engine()
          else:
            await model.call("resume", target_gpus[0] if scheduler.is_movable(r.spec, self.gpus) else None)
            r.host_ram_bytes = 0
      except Exception as e:
        r.gpus_held, r.gpu_released, r.phase = (), True, "failed"
        await owned()
        gens = {j: job_gen(j) for j in active_runs()}
        for cmd in await self.bus.take_commands(list(gens), owner, 1, generations=gens):
          await self.bus.write_reply(cmd, Reply.error(e), worker, owner, generation=gens[cmd.run])
        raise
      self.notify()
      await self.push_heartbeat(wake_scheduler=False)
      return True

    def wanted() -> bool:
      w = cur_w()
      if (
        self.draining
        or w is None
        or w.node != self.id
        or w.intent == "drain"
        or not scheduler.jobs_of(self.state, worker)
      ):
        return False
      return not (
        w.intent == "run"
        and w.gpus
        and r.gpus
        and w.gpus != r.gpus
        and r.pid is not None
        and not scheduler.is_movable(r.spec, self.gpus)
      )

    def budget_exhausted() -> bool:
      w = cur_w()
      return bool(
        w
        and w.budgets
        and grant_cmds > 0
        and w.grant_rev == grant_rev_seen
        and sum(grant_usage.values()) >= sum(w.budgets.values())
      )

    def pick_runs(candidates: list[Cmd], width: int) -> list[str]:
      ready_map: dict[str, Cmd] = {}
      for cmd in candidates:
        ready_map.setdefault(cmd.run, cmd)
      prefer = last_run if (not sampling and turn_commands < RUN_COMMANDS) else None
      floor = min((local_svc[j] for j in ready_map if j in local_svc), default=0.0)
      ordered = sorted(
        ready_map.values(),
        key=lambda c: (
          c.run != prefer,
          c.op != Op.SAVE_WEIGHTS_FOR_SAMPLER,
          local_svc.get(c.run, floor) / max(1e-6, self.state.jobs[c.run].weight if c.run in self.state.jobs else 1.0),
          tuple(map(int, c.entry_id.split("-"))),
          c.run,
        ),
      )
      return [c.run for c in ordered[:width]]

    async def execute_batch(candidates: list[Cmd], replayed_cmds: list[Cmd] | None = None) -> Cmd | None:
      nonlocal reader, last_run, turn_commands, heads, grant_cmds
      if replayed_cmds is not None:
        cmds, inputs = replayed_cmds, {c.run: await prepare(c) for c in replayed_cmds}
      else:
        runs = pick_runs(candidates, SAMPLE_RUNS if sampling and r.spec.lora else 1)
        inputs = {c.run: prepared[c.request_id].result() for c in candidates if c.run in runs}
        failed = all(isinstance(v, Exception) for v in inputs.values())
        if reader:
          reader.cancel()
          await asyncio.gather(reader, return_exceptions=True)
          reader = None
        await owned()
        gens = {j: job_gen(j) for j in runs}
        with span("node.claim"):
          cmds = await self.bus.take_commands(
            runs, owner, SAMPLE_COMMANDS if sampling and not failed else 1, generations=gens
          )
        reader = asyncio.create_task(read())
        reader.add_done_callback(lambda _: self.notify())
        heads = [c for c in heads if c.run not in runs]
      if not cmds:
        return None
      await owned()
      t0 = time.monotonic()
      if sampling:
        wave_done = asyncio.Event()

        async def watch_drain():
          while not wave_done.is_set():
            w = cur_w()
            if self.draining or w is None or w.node != self.id or w.intent != "run":
              with contextlib.suppress(Exception):
                await model.pause("abort" if (self.draining or w is None or w.intent == "drain") else "wait")
              return
            wait_t, done_t = asyncio.create_task(self.changed.wait()), asyncio.create_task(wave_done.wait())
            _, pending = await asyncio.wait({wait_t, done_t}, return_when=asyncio.FIRST_COMPLETED)
            for p in pending:
              p.cancel()

        watcher = asyncio.create_task(watch_drain())
        try:
          replies = await asyncio.gather(*(answer(cmd, inputs[cmd.run]) for cmd in cmds))
        finally:
          wave_done.set()
          await asyncio.gather(watcher, return_exceptions=True)
        gpu_sec, total_cost, active_set = (
          max(1e-4, time.monotonic() - t0) * max(1, len(r.gpus_held)),
          max(1, sum(c.cost for c in cmds)),
          set(active_runs()),
        )
        for cmd, rep in zip(cmds, replies):
          if rep is None:
            if (
              not self.draining
              and cmd.run in active_set
              and cmd.request_id not in {c.request_id for c in pending_replay}
            ):
              pending_replay.append(cmd)
          else:
            record_usage(cmd.run, gpu_sec * (cmd.cost / total_cost), cmd.cost)
        grant_cmds += len(cmds)
        return None

      cmd = cmds[0]
      if cmd.run != last_run or turn_commands >= RUN_COMMANDS:
        last_run, turn_commands = cmd.run, 0
      turn_commands += 1
      reply = await answer(cmd, inputs[cmd.run])
      record_usage(cmd.run, max(1e-4, time.monotonic() - t0) * max(1, len(r.gpus_held)), cmd.cost)
      grant_cmds += 1
      return cmd if (reply and reply.ok) else None

    async def cleanup(failed: bool) -> None:
      for fut in [reader, *prepared.values()]:
        if fut:
          fut.cancel()
      await asyncio.gather(*(f for f in [reader, *prepared.values()] if f), return_exceptions=True)
      if durable_saves:
        await asyncio.gather(*list(durable_saves), return_exceptions=True)
      if model:
        while True:
          try:
            with span("node.close"):
              await model.close()
            break
          except (Exception, asyncio.CancelledError) as e:
            print(f"{worker} could not close, retrying: {e!r}", file=sys.stderr, flush=True)
            await asyncio.sleep(TICK)
      w = cur_w()
      r.gpus_held, r.gpu_released, r.pid, r.host_ram_bytes = (), True, None, 0
      r.phase, r.rev = ("failed" if failed else "drained"), max(r.rev, w.rev if w else 0)
      if w is not None and w.node == self.id:
        self.tombstones[worker] = r.obs()
      self.notify()
      if guard:
        guard.cancel()
        await asyncio.gather(guard, return_exceptions=True)
      if await self.bus.lease_owner(worker) == owner:
        runs = scheduler.jobs_of(self.state, worker)
        gens = {j: job_gen(j) for j in runs}
        if died:
          await self.bus.take_commands(runs, owner, 1, generations=gens)
        if not sampling or died:
          await self.bus.fail_stranded_commands(runs, worker, owner, 0, generations=gens, replay_sampler=False)
      await self.bus.release_lease(worker, owner)
      with contextlib.suppress(Exception):
        await self.push_heartbeat(wake_scheduler=True)

    try:
      if not await self.bus.take_lease(worker, owner):
        return
      started = True
      init_runs = scheduler.jobs_of(self.state, worker)
      pending_replay = await self.bus.fail_stranded_commands(
        init_runs, worker, owner, 0, generations={j: job_gen(j) for j in init_runs}, replay_sampler=sampling
      )
      if not await self.bus.renew_lease(worker, owner) or self.draining:
        return
      self.tombstones.pop(worker, None)
      self.running[worker] = r
      env = {**scheduler.settings(r.spec, self.gpus, self.gpu_mib * 2**20), **os.environ, **self.store.env()}
      visible = range(self.gpus) if scheduler.is_movable(r.spec, self.gpus) else (r.gpus or tuple(range(r.spec.gpus)))
      env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, visible))
      if not sampling:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
      model = self.open_sampler(worker, env) if sampling else self.open_trainer(worker, env)
      r.pid, r.phase, r.gpu_released = (model.proc.pid if model.proc else None), "parked", True
      guard = asyncio.create_task(keepalive())
      reader = asyncio.create_task(read())
      reader.add_done_callback(lambda _: self.notify())

      while wanted():
        changed = self.changed
        if (p := model.proc) and p.poll() is not None:
          raise RuntimeError(f"worker {worker} exited")
        await process_drain_jobs()
        if not wanted():
          break
        collect()
        runnable = ready()
        if failed_prep := [c for c in runnable if isinstance(prepared[c.request_id].result(), Exception)]:
          await execute_batch(failed_prep)
          runnable = [c for c in ready() if not isinstance(prepared[c.request_id].result(), Exception)]
          if not wanted():
            break

        stalled_prep = (
          bool(heads) and not runnable and not pending_replay and any(not t.done() for t in prepared.values())
        )
        if stalled_prep and r.phase != "preparing":
          r.phase = "preparing"
          await self.push_heartbeat(wake_scheduler=True)
        elif not stalled_prep and r.phase == "preparing":
          r.phase = "running" if r.gpus_held else "parked"
          await self.push_heartbeat(wake_scheduler=True)

        w = cur_w()
        if w is None or w.intent != "run":
          if r.gpus_held:
            await park_gpus()
          elif w and r.rev < w.rev:
            r.rev = w.rev
            await self.push_heartbeat(wake_scheduler=True)
        else:
          if w.grant_rev != grant_rev_seen:
            grant_rev_seen, grant_usage, grant_cmds = w.grant_rev, {}, 0
          if budget_exhausted():
            await self.push_heartbeat(wake_scheduler=True)
            await changed.wait()
            continue
          if (stalled_prep and not r.gpus_held) or not await activate_gpus(w.gpus):
            await changed.wait()
            continue
          if pending_replay:
            batch, pending_replay = pending_replay, []
            await execute_batch([], replayed_cmds=batch)
            continue
          if runnable:
            if (cmd := await execute_batch(runnable)) and cmd.op == Op.DELETE_MODEL and not r.spec.lora:
              break
            if budget_exhausted():
              await self.push_heartbeat(wake_scheduler=True)
            continue

        if reader is None:
          reader = asyncio.create_task(read(IDLE_MS / 1000 if heads else 0))
          reader.add_done_callback(lambda _: self.notify())
        await changed.wait()

      if durable_saves:
        await asyncio.gather(*list(durable_saves), return_exceptions=True)
      if not sampling and loaded_runs:
        r.phase = "draining"
        await owned()
        for saved in await model.call("park_all", last_completed):
          await owned()
          await commit_parked(saved.pop("run"), saved)
    except BaseException as exc:
      failed_exc = exc
      raise
    finally:
      if started:
        died = model is not None and (p := model.proc) is not None and p.poll() is not None
        w_end = cur_w()
        unplanned = (
          isinstance(failed_exc, asyncio.CancelledError)
          and not self.draining
          and w_end is not None
          and w_end.node == self.id
          and w_end.intent != "drain"
        )
        is_failed = died or unplanned or (failed_exc is not None and not isinstance(failed_exc, asyncio.CancelledError))
        closing = asyncio.create_task(cleanup(is_failed))
        self.notify()
        while True:
          try:
            await asyncio.shield(closing)
            break
          except asyncio.CancelledError:
            if closing.cancelled():
              closing = asyncio.create_task(cleanup(is_failed))

  def dispatch(self) -> None:
    for worker, task in list(self.tasks.items()):
      if task.done():
        try:
          task.result()
        except BaseException as e:
          if not isinstance(e, asyncio.CancelledError):
            print(f"{worker} stopped: {e!r}", file=sys.stderr, flush=True)
        del self.tasks[worker]
        self.running.pop(worker, None)
      elif self.id not in self.state.nodes and not task.cancelling():
        task.cancel()
    for worker, tomb in list(self.tombstones.items()):
      if (w := self.state.workers.get(worker)) is None or w.node != self.id or w.rev > tomb.get("rev", 0):
        del self.tombstones[worker]
    self.prefetcher.reconcile(self.state)

  async def tick(self) -> None:
    if self._loop is None:
      self._loop = asyncio.get_running_loop()
    await self.push_heartbeat(wake_scheduler=False)
    old, self.state = self.state, await self.read_state()
    self.dispatch()
    if not self.draining:
      for worker, w in sorted(self.state.workers.items()):
        if w.node != self.id or w.intent == "drain" or worker in self.tasks or worker in self.tombstones:
          continue
        if (jobs := scheduler.jobs_of(self.state, worker)) and (
          w.intent == "run"
          or w.drain_jobs
          or (w.spec.role == Role.SAMPLER and any(self.state.jobs[j].ready for j in jobs))
        ):
          self.start(worker)
    if time.time() - self.swept > SWEEP_S and (self.sweeping is None or self.sweeping.done()):
      self.swept, self.sweeping = time.time(), asyncio.create_task(self.registry.sweep(self.store, SWEEP_GRACE_S))
    if old.workers != self.state.workers or old.jobs != self.state.jobs:
      self.notify()

  async def close(self, graceful: bool = True) -> None:
    self.draining = True
    self.notify()
    await self.prefetcher.close()
    if self.tasks:
      _, pending = await asyncio.wait(self.tasks.values(), timeout=DRAIN_S if graceful else 0)
      for task in pending:
        if not task.cancelling():
          task.cancel()
      await asyncio.gather(*self.tasks.values(), return_exceptions=True)
      self.dispatch()
    if self.bg_tasks:
      await asyncio.gather(*list(self.bg_tasks), return_exceptions=True)
    await asyncio.to_thread(self.store.close)

  async def serve(self) -> None:
    await self.announce_node()
    cursor = await self.bus.wake_cursor(f"node:{self.id}")
    try:
      while True:
        await self.tick()
        before = time.monotonic()
        cursor = await self.bus.wait_wake(f"node:{self.id}", cursor, block_ms=int(TICK * 1000))
        self.lag = max(0.0, time.monotonic() - before - TICK)
    finally:
      await self.close()


if __name__ == "__main__":

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await NodeDaemon(Bus()).serve()

  asyncio.run(main())
