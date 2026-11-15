"""Node execution agent: runs scheduler-issued assignments, prepares checkpoint versions, enforces generation-fenced command execution and acknowledged suspension, and reports verified physical GPU release."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

from tinyrl.bus import LEASE_MS, Bus, Cmd, Ev, Op, Reply, Role
from tinyrl.control import estimator, scheduler
from tinyrl.control.process import Process, gpu_pids_clear, rss_bytes
from tinyrl.helpers import DEBUG, getenv
from tinyrl.storage.registry import READ_REF_TTL, Registry
from tinyrl.storage.store import Store, TransferCancelled, split_version
from tinyrl.trace import span
from tinyrl.workers.sampler import Sampler

GPUS, DRAIN_S, TICK = getenv("GPUS", 1), getenv("DRAIN_S", 20.0), getenv("TICK", 0.25)
IDLE_MS, RUN_COMMANDS = getenv("IDLE_MS", 100), getenv("RUN_COMMANDS", 4)
SAMPLE_RUNS, SAMPLE_COMMANDS = getenv("SAMPLE_RUNS", 3), getenv("SAMPLE_COMMANDS", 128)
SWEEP_S, SWEEP_GRACE_S = getenv("SWEEP_S", 60.0), 600


def gpu_memory_mib() -> int:
  if override := getenv("GPU_MEMORY_MIB", 0):
    return int(override)
  with contextlib.suppress(OSError, subprocess.SubprocessError, ValueError):
    cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"]
    return int(subprocess.check_output(cmd, timeout=2, stderr=subprocess.DEVNULL))
  return 0


def movable(spec: scheduler.WorkerSpec, gpus: int) -> bool:
  return spec.role == Role.TRAINER and spec.lora and spec.gpus == 1 and gpus > 1


class NodeDaemon:
  def __init__(
    self,
    bus: Bus,
    gpus: int = GPUS,
    node: str = "",
    gpu_mib: int | None = None,
    store: Store | None = None,
    *,
    host_mib: int = 0,
    gpu_class: str = "",
    rate: float = 0.0,
    bundles: list[list[int]] | None = None,
  ):
    self.bus, self.registry, self.store = bus, Registry(bus), (store if store is not None else Store())
    self.store.start()
    self.gpus, self.gpu_ids, self.id = gpus, tuple(range(gpus)), node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.host_mib, self.gpu_class = host_mib or getenv("HOST_MEMORY_MIB", 0), gpu_class or getenv("GPU_CLASS", "")
    self.rate, self.bundles = rate or float(os.getenv("GPU_RATE", "0")), list(bundles or [])
    self.tasks: dict[str, asyncio.Task] = {}
    self.running: dict[str, dict] = {}
    self.prefetches: dict[str, tuple[threading.Event, asyncio.Task]] = {}
    self.durable_tasks: set[asyncio.Task] = set()
    self.state, self.changed, self.draining = scheduler.Assignment(), asyncio.Event(), False
    self.heartbeat, self.lag, self.swept, self.sweeping, self._loop = time.monotonic(), 0.0, 0.0, None, None
    prev_cb = self.store.on_durable

    def on_durable(uri: str, version: str) -> None:
      if prev_cb:
        prev_cb(uri, version)
      if self._loop and not self._loop.is_closed():
        self._loop.call_soon_threadsafe(self._schedule_durable, uri, version)

    self.store.on_durable = on_durable

  def _attach_loop(self) -> None:
    if self._loop is None:
      self._loop = asyncio.get_running_loop()

  def _schedule_durable(self, uri: str, version: str) -> None:
    run = uri.removeprefix("tinker://").split("/", 1)[0]

    async def promote():
      with contextlib.suppress(Exception):
        await self.bus.complete_durable_version(run, version, self.store.peer)
        await self.registry.mark_durable(version, self.store.peer)

    self.durable_tasks.add(t := asyncio.create_task(promote()))
    t.add_done_callback(self.durable_tasks.discard)

  async def flush_durable_saves(self) -> None:
    await asyncio.sleep(0)
    while self.durable_tasks:
      await asyncio.gather(*list(self.durable_tasks), return_exceptions=True)
      await asyncio.sleep(0)

  def notify(self) -> None:
    changed, self.changed = self.changed, asyncio.Event()
    changed.set()

  def node_info(self) -> scheduler.NodeInfo:
    return self.state.nodes.get(self.id) or scheduler.NodeInfo(
      self.gpus, self.gpu_mib * 2**20, self.host_mib * 2**20, self.gpu_ids, self.gpu_class, self.rate, self.bundles
    )

  def gpus_physically_safe(self, worker: str, gpus: tuple[int, ...], spec: scheduler.WorkerSpec) -> bool:
    if not gpus:
      return False
    info, my_mem = self.node_info(), scheduler.memory_budget(self.node_info(), spec)
    for g in gpus:
      if not (
        others := [
          v
          for n, v in self.running.items()
          if n != worker and g in v["gpus"] and v["gpu_state"] in ("active", "paused")
        ]
      ):
        continue
      if spec.exclusive or any(v["spec"].exclusive for v in others):
        return False
      off = sum(
        max(v.get("gpu_bytes", 0), estimator.OVERHEAD if info.gpu_bytes else 0) // max(1, len(v["gpus"]))
        for n, v in self.running.items()
        if n != worker and g in v["gpus"] and v["gpu_state"] == "offloaded"
      )
      if not scheduler.fits([*(scheduler.memory_budget(info, v["spec"]) for v in others), my_mem], info.gpu_bytes, off):
        return False
    return True

  async def version_reachable(
    self, uri: str, version: str | None, live_peers: set[str]
  ) -> tuple[bool, bool, tuple[str, ...]]:
    if not (ver := split_version(version)[0]):
      return False, False, ()
    if await asyncio.to_thread(self.store.has_local, uri, ver):
      return True, await asyncio.to_thread(self.store.is_durable, uri, ver), ()
    _, peers, dur, _ = await self.registry.resolve(uri, ver)
    dur = dur or await asyncio.to_thread(self.store.is_durable, uri, ver)
    good = tuple(p for p in peers if p and p != self.store.peer and (not live_peers or p in live_peers))
    return bool(dur or good), dur, good

  async def stage_version(
    self, uri: str, version: str | None = None, *, lane: str = "required", cancel: threading.Event | None = None
  ) -> tuple[str, str, tuple[str, ...]]:
    """Unified staging path for both trainer restores and sampler checkpoints, protected by a cluster reader ref."""
    async with self.registry.read_ref(uri, version, ttl=READ_REF_TTL) as (ver, peers, _):
      folder, resolved = await asyncio.to_thread(self.store.local_folder, uri, ver or version, peers, cancel, lane)
      if self.store.remote and self.store.peer:
        await self.registry.set_location(resolved, self.store.peer, True)
      return folder, resolved, peers

  def sync_prefetches(self) -> None:
    needed = {
      v
      for w in self.state.workers.values()
      if not self.draining and w.node == self.id and not (w.intent == "drain" and w.drain_mode in ("export", "close"))
      for v in w.prepare
    }
    for ver, (ev, task) in list(self.prefetches.items()):
      self.prefetches.pop(ver, None) if task.done() else (ev.set() if ver not in needed else None)
    for ver in sorted(needed - self.prefetches.keys()):
      if vinfo := self.state.versions.get(ver):
        self.prefetches[ver] = (ev := threading.Event(), asyncio.create_task(self._run_prefetch(vinfo["uri"], ver, ev)))

  async def _run_prefetch(self, uri: str, ver: str, cancel: threading.Event) -> None:
    holder = f"stage:{self.id}:{ver}"
    if not await self.registry.acquire_ref(ver, holder, ttl=READ_REF_TTL):
      return
    try:
      _, peers, _, _ = await self.registry.resolve(uri, ver)
      await asyncio.to_thread(self.store.local_folder, uri, ver, peers, cancel, "prefetch")
      if self.store.remote and self.store.peer:
        await self.registry.set_location(ver, self.store.peer, True)
      self.notify()
    except (TransferCancelled, Exception):
      pass
    finally:
      await self.registry.release_ref(ver, holder)

  async def read_state(self) -> scheduler.Assignment:
    return scheduler.from_table(await self.bus.read_table())

  def heartbeat_extra(self) -> dict:
    return {
      "host_memory": self.host_mib * 2**20,
      "gpu_ids": list(self.gpu_ids),
      "gpu_class": self.gpu_class,
      "rate": self.rate,
      "bundles": self.bundles,
      "cache": self.store.cache_status(),
    }

  async def announce_node(self) -> None:
    self._attach_loop()
    await self.bus.node_heartbeat(self.id, self.gpus, self.gpu_mib * 2**20, **self.heartbeat_extra())
    self.heartbeat = time.monotonic()
    await self.bus.append_event(
      Ev.NODE_UP, node=self.id, gpus=self.gpus, memory=self.gpu_mib * 2**20, gpu_class=self.gpu_class
    )

  def open_trainer(self, worker: str, gpus: tuple[int, ...], env: dict[str, str]) -> Process:
    dist_args = (
      ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={len(gpus)}"] if len(gpus) > 1 else []
    )
    return Process([sys.executable, *dist_args, "-m", "tinyrl.workers.trainer"], env)

  def open_sampler(self, worker: str, spec: scheduler.WorkerSpec, env: dict[str, str]) -> Sampler:
    return Sampler(spec.base, spec.lora, env)

  def start(self, worker: str) -> asyncio.Task:
    self._attach_loop()
    self.tasks[worker] = task = asyncio.create_task(self.work(worker))
    return task

  async def work(self, worker: str) -> None:
    spec = self.state.workers[worker].spec
    sampling, owner, task = spec.role == Role.SAMPLER, f"{self.id}-{os.urandom(8).hex()}", asyncio.current_task()
    model = guard = reader = closing = last_run = busy_since = None
    bound_gpus = active_gpus = ()
    gpu_state, phase = "released", "preparing"
    gpu_bytes = host_bytes = seen_rev = reported_rev = turn_cmds = 0
    virtual, renewed, failed = 0.0, time.monotonic(), False
    usage, served, busy_weights, local_svc = {}, {}, {}, {}
    sample_tasks: dict[asyncio.Task, Cmd] = {}
    boundaries, loaded_gens, rolled_back, replays, prepared = {}, {}, {}, {}, {}
    exported_jobs, heads = [], []

    def forget_run(r: str) -> None:
      for d in (loaded_gens, boundaries, rolled_back, replays):
        d.pop(r, None)

    def clear_now() -> bool:
      return (
        True
        if not model or (p := model.proc) is None
        else (p.poll() is not None and gpu_pids_clear(getattr(p, "pid", None)))
      )

    async def report(rev: int | None = None, ph: str | None = None) -> None:
      nonlocal phase, host_bytes, reported_rev
      phase = phase if ph is None else ph
      reported_rev = seen_rev if rev is None else rev
      w, pid = self.state.workers.get(worker), (getattr(model.proc, "pid", None) if model and model.proc else None)
      if gpu_state == "offloaded" and not host_bytes and pid:
        host_bytes = rss_bytes(pid)
      gens = dict(w.generations) if w else dict(loaded_gens)
      p_clear = clear_now() if gpu_state == "released" else False
      obs = {"node": self.id, "revision": reported_rev, "generations": gens, "gpus": list(active_gpus), "pid": pid}
      obs |= {
        "phase": phase,
        "gpu_state": gpu_state,
        "pids_clear": p_clear,
        "gpu_bytes": gpu_bytes,
        "host_bytes": host_bytes,
      }
      obs |= {"usage": dict(usage), "served": dict(served), "boundaries": dict(boundaries)}
      obs |= {"exported_jobs": list(exported_jobs), "local_state": sorted(loaded_gens)}
      if sampling and model and (sinfo := model.startup_info()):
        obs["startup"] = sinfo
      self.running[worker] = obs | {"spec": spec, "gpus": active_gpus}
      await self.bus.write_observation(worker, obs)
      self.notify()

    async def owned() -> None:
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

    async def keepalive() -> None:
      while True:
        await asyncio.sleep(LEASE_MS / 4000)
        if not closing and not self.draining and time.monotonic() - self.heartbeat >= LEASE_MS / 1000:
          return bool(task.cancel()) or None
        try:
          await owned()
          if not closing and model and (p := model.proc) and p.poll() is not None:
            return bool(task.cancel()) or None
        except Exception:
          if not closing and time.monotonic() - renewed >= LEASE_MS / 1000:
            return bool(task.cancel()) or None

    def charge(weights: dict[str, float], elapsed: float) -> None:
      if elapsed <= 0 or (total_w := sum(w for w in weights.values() if w > 0)) <= 0:
        return
      info = self.node_info()
      cost = elapsed * estimator.service_rate(max(1, len(active_gpus) or spec.gpus), info.gpu_class, info.rate)
      for acct, w in weights.items():
        if w > 0:
          usage[acct] = round(usage.get(acct, 0.0) + cost * (w / total_w), 6)

    async def activate(gpus: tuple[int, ...], rev: int) -> bool:
      nonlocal model, active_gpus, bound_gpus, gpu_state, gpu_bytes, host_bytes
      if not gpus or not self.gpus_physically_safe(worker, gpus, spec):
        return False
      if model is not None and bound_gpus and gpus != bound_gpus and not movable(spec, self.gpus):
        return False
      if model is None:
        info = self.node_info()
        env = {
          **scheduler.settings(spec, self.gpus, self.gpu_mib * 2**20, info.gpu_class),
          **os.environ,
          **self.store.env(),
        }
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids if movable(spec, self.gpus) else gpus))
        if not sampling:
          env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        bound_gpus, model = (
          gpus,
          (self.open_sampler(worker, spec, env) if sampling else self.open_trainer(worker, gpus, env)),
        )
      if (p := model.proc) and p.poll() is not None:
        raise RuntimeError(f"worker {worker} exited with {p.returncode}")
      await owned()
      if gpu_state != "active" or active_gpus != gpus:
        if gpu_state == "active" and active_gpus and active_gpus != gpus and not sampling:
          await asyncio.wait_for(model.call("park"), timeout=DRAIN_S)
        active_gpus = bound_gpus = gpus
        with span("node.resume"):
          if not sampling:
            mem = await model.call("resume", gpus[0] if movable(spec, self.gpus) else None)
            gpu_bytes, host_bytes = int(mem.get("gpu_bytes", 0)), int(mem.get("host_bytes", 0))
          else:
            t0 = time.monotonic()
            await model.resume()
            if model.running is None:
              try:
                await model.start()
              except Exception as boot_err:
                if not isinstance(boot_err, asyncio.CancelledError) and await self.bus.lease_owner(worker) == owner:
                  gmap = dict(w_cur.generations) if (w_cur := self.state.workers.get(worker)) else dict(loaded_gens)
                  if waiting := await self.bus.peek_commands(list(gmap), 0):
                    err = Reply.error(boot_err)
                    for c in await self.bus.take_commands({c.run: gmap[c.run] for c in waiting}, owner, 1):
                      await self.bus.write_reply(c, err, worker, owner, generation=gmap[c.run])
                raise
            if (
              accts := {
                self.state.jobs[r].account: 1.0 for r in scheduler.jobs_of(self.state, worker) if r in self.state.jobs
              }
            ) and time.monotonic() - t0 > 0.01:
              charge(accts, time.monotonic() - t0)
        gpu_state = "active"
        await report(rev, "running")
      return True

    async def park_local(rev: int) -> None:
      nonlocal gpu_state, gpu_bytes, host_bytes
      if model is None or gpu_state in ("offloaded", "released"):
        gpu_state = "offloaded" if model is not None else "released"
        stalled = (
          gpu_state == "offloaded"
          and any(not f.done() for f in prepared.values())
          and not ready_heads(set(loaded_gens))
        )
        return await report(rev, "preparing" if stalled else ("parked" if gpu_state == "offloaded" else "released"))
      await report(rev, "draining")
      with span("node.park"):
        if sampling:
          await asyncio.wait_for(model.park("wait"), timeout=DRAIN_S)
          if sample_tasks:
            await asyncio.gather(*list(sample_tasks), return_exceptions=True)
            await reap_samples(report_now=False)
          gpu_bytes = 0
        else:
          mem = await asyncio.wait_for(model.call("park"), timeout=DRAIN_S)
          gpu_bytes, host_bytes = int(mem.get("gpu_bytes", 0)), int(mem.get("host_bytes", 0))
      gpu_state = "offloaded"
      await report(
        rev,
        "preparing" if any(not f.done() for f in prepared.values()) and not ready_heads(set(loaded_gens)) else "parked",
      )

    async def export_run(run_id: str, gen: int, *, priority: bool = False) -> bool:
      if (
        sampling
        or not model
        or run_id in rolled_back
        or run_id not in loaded_gens
        or await self.bus.get_generation(run_id) != gen
      ):
        return False
      if not (boundary := boundaries.get(run_id, "")):
        st = ((await self.bus.get_run(run_id)) or {}).get("state") or {}
        boundary = st.get("acked_mutation") or st.get("acked_progress") or ""
      if not boundary:
        return False
      if spec.gpus > 1 and gpu_state != "active" and active_gpus:
        await activate(active_gpus, seen_rev)
      await owned()
      saved = await asyncio.wait_for(
        model.call("snapshot_job", run_id, f"tinker://{run_id}/recovery/{time.time_ns()}", boundary, priority),
        timeout=DRAIN_S,
      )
      await owned()
      return await self.bus.commit_job_state(
        run_id,
        gen,
        saved["uri"],
        saved["version"],
        boundary,
        saved["size"],
        saved["meta"],
        durable=saved["durable"],
        location=saved["location"] or self.store.peer,
      )

    async def sync_membership(w: scheduler.Worker) -> None:
      for r in [r for r in list(w.draining_jobs) if r not in exported_jobs]:
        if not sampling and model and r in loaded_gens and (gen := loaded_gens.get(r) or w.generations.get(r)):
          await report(w.revision, "draining")
          await export_run(r, gen, priority=False)
          await model.call("drop_run", r)
          forget_run(r)
        exported_jobs.append(r)
        await report(w.revision)
      exported_jobs[:] = [r for r in exported_jobs if r in w.draining_jobs]
      for r, old_gen in list(loaded_gens.items()):
        if w.generations.get(r) != old_gen:
          if model:
            with contextlib.suppress(Exception):
              await (model.cancel_run(r) if sampling else model.call("drop_run", r))
          forget_run(r)

    async def reconcile_jobs(w: scheduler.Worker) -> None:
      live_peers = {n.peer for n in self.state.nodes.values() if n.peer}
      durable_vers = {v for v, info in self.state.versions.items() if info.get("durable")}
      for r, gen in sorted(w.generations.items()):
        if r in w.draining_jobs or loaded_gens.get(r) == gen or await self.bus.get_generation(r) != gen:
          continue
        await owned()
        if sampling:
          if model:
            model.unfence_run(r)
          if claimed := await self.bus.reclaim_commands(r, worker, owner, gen, replay_sample=True):
            replays[r] = claimed
          loaded_gens[r] = gen
          continue
        await self.bus.reclaim_commands(r, worker, owner, gen, replay_sample=False)
        if not (run := await self.bus.get_run(r)):
          continue
        st = run.get("state") or {}
        wk, dur = st.get("working"), st.get("durable")
        wk_ok, wk_dur, _ = (
          await self.version_reachable(wk["uri"], wk["version"], live_peers) if wk else (False, False, ())
        )
        dur_ok, _, _ = (
          await self.version_reachable(dur["uri"], dur["version"], live_peers) if dur else (False, False, ())
        )
        durable_vers |= ({wk["version"]} if wk and wk_dur else set()) | ({dur["version"]} if dur and dur_ok else set())
        action, info = await self.bus.reconcile_recovery(r, gen, wk_ok, dur_ok, durable_vers)
        if action in ("uncreated", "rollback"):
          loaded_gens[r] = gen
          rolled_back[r] = info if action == "rollback" else rolled_back.pop(r, None)
          if action == "uncreated":
            rolled_back.pop(r, None)
        elif action in ("restore", "recreate"):
          staged_peers: tuple[str, ...] = ()
          if action == "restore":
            await report(w.revision, "preparing")
            _, resolved, staged_peers = await self.stage_version(info["uri"], info["version"], lane="required")
            info = info | {"version": resolved}
          if w.intent != "run" or not w.gpus or not await activate(w.gpus, w.revision):
            continue
          await report(w.revision, "restoring")
          t0 = time.monotonic()
          if action == "restore":
            with self.store.pin(info["version"]):
              await model.call(
                "restore_job",
                r,
                run["base"],
                run["rank"],
                run.get("lora", {}),
                info["uri"],
                info["version"],
                staged_peers,
                True,
                True,
              )
            boundaries[r] = info.get("boundary", "")
          else:
            await model.call("ensure_run", r, run["base"], run["rank"], run.get("lora", {}))
            boundaries[r] = st.get("created_entry", "")
          charge({st.get("account", r): 1.0}, time.monotonic() - t0)
          loaded_gens[r] = gen
          rolled_back.pop(r, None)
          await report(w.revision, "running")

    async def prepare_cmd(cmd: Cmd):
      with span("node.prepare"):
        try:
          run = await self.bus.get_run(cmd.run)
          if sampling:
            if run is None:
              raise ValueError(f"no sampling session {cmd.run}")
            if uri := run.get("model_path"):
              pinned = (run.get("state") or {}).get("weights_version") or (await self.registry.resolve(uri))[0]
              folder, ver, _ = await self.stage_version(uri, pinned, lane="required")
              if not (run.get("state") or {}).get("weights_version"):
                await self.bus.pin_sampler_weights(cmd.run, ver)
              return run, (folder, ver)
          return run, None
        except Exception as e:
          return e

    def eligible_runs(w: scheduler.Worker, *, for_gpu: bool = False) -> dict[str, int]:
      return {
        r: gen
        for r, gen in w.generations.items()
        if r not in w.draining_jobs
        and loaded_gens.get(r) == gen
        and (
          (w.intent == "run" and served.get(r, 0) < w.budgets.get(r, 0))
          or (not for_gpu and (r in rolled_back or w.intent == "idle"))
        )
      }

    async def read_heads(runs: list[str], delay: float = 0.0) -> list[Cmd]:
      if delay:
        await asyncio.sleep(delay)
      out = [cmds[0] for r, cmds in replays.items() if r in runs and cmds]
      if peek_for := [r for r in runs if not replays.get(r)]:
        out += await self.bus.peek_commands(peek_for, IDLE_MS if not out else 0)
      for cmd in out:
        if cmd.request_id not in prepared:
          prepared[cmd.request_id] = t = asyncio.create_task(prepare_cmd(cmd))
          t.add_done_callback(lambda _: self.notify())
      return out

    def collect() -> None:
      nonlocal heads, reader, local_svc
      if reader and reader.done():
        heads, reader = ([] if reader.exception() else reader.result()), None
        for rid in [k for k in prepared if k not in {c.request_id for c in heads}]:
          prepared.pop(rid).cancel()
        local_svc = {c.run: local_svc.get(c.run, virtual) for c in heads}

    def ready_heads(allowed: set[str]) -> list[Cmd]:
      return [c for c in heads if c.run in allowed and (f := prepared.get(c.request_id)) is not None and f.done()]

    async def answer_sampler(cmd: Cmd, gen: int, inputs) -> tuple[str, int]:
      before = time.time()
      if isinstance(inputs, Exception):
        await self.bus.write_reply(cmd, Reply.error(inputs), worker, owner, started=before, generation=gen)
        return cmd.run, 0
      run, checkpoint = inputs
      acct = (run.get("state") or {}).get("account", cmd.run)
      await owned()
      if await self.bus.get_generation(cmd.run) != gen:
        return acct, 0
      try:
        with self.store.pin(checkpoint[1] if checkpoint else None):
          reply, tokens = await model.execute(cmd, run, checkpoint)
      except asyncio.CancelledError:
        return acct, 0
      await self.bus.write_reply(cmd, reply, worker, owner, started=before, generation=gen)
      return acct, max(1, tokens) if reply.ok else 0

    async def answer_trainer(cmd: Cmd, gen: int, inputs) -> tuple[Reply | None, str, float]:
      before = time.time()
      if isinstance(inputs, Exception):
        await self.bus.write_reply(cmd, Reply.error(inputs), worker, owner, started=before, generation=gen)
        return None, cmd.run, 0.0
      run, _ = inputs
      st = (run or {}).get("state") or {}
      acct = st.get("account", cmd.run)
      if rb := rolled_back.get(cmd.run):
        if cmd.op not in (Op.LOAD_WEIGHTS, Op.DELETE_MODEL):
          msg = f"job {cmd.run} rolled back after acknowledged command {rb['acked']} (last durable checkpoint: {rb.get('durable_uri') or rb.get('durable') or 'none'}); restore explicitly with load_weights before further mutations"
          await self.bus.write_reply(cmd, Reply.error(RuntimeError(msg)), worker, owner, started=before, generation=gen)
          return None, acct, 0.0
        if cmd.op == Op.DELETE_MODEL:
          rep = Reply(True, "application/json", json.dumps({"type": cmd.op, "model_id": cmd.run}).encode())
          if await self.bus.write_reply(cmd, rep, worker, owner, started=before, generation=gen):
            forget_run(cmd.run)
          return rep, acct, 0.0
      version, peers = None, ()
      if cmd.op == Op.LOAD_WEIGHTS:
        body = json.loads(cmd.body)
        try:
          _, version, peers = await self.stage_version(body["uri"], body.get("version"), lane="required")
        except Exception as e:
          await self.bus.write_reply(cmd, Reply.error(e), worker, owner, started=before, generation=gen)
          return None, acct, 0.0
      cur_boundary = boundaries.get(cmd.run, "") or cmd.entry_id
      await owned()
      if await self.bus.get_generation(cmd.run) != gen:
        return None, acct, 0.0
      t0 = time.monotonic()
      with self.store.pin(version):
        reply, effects = await model.call(
          "execute", cmd, run, version, peers, cmd.entry_id, (st.get("working") or {}).get("version")
        )
      dt = time.monotonic() - t0
      await owned()

      defer_save = None
      if s := effects.get("saved"):
        loc = s.get("location") or self.store.peer
        rkw = {"meta": s.get("meta"), "location": loc}
        if s.get("optimizer"):
          ckw = {"durable": s["durable"], "location": loc}
          if not await self.bus.commit_job_state(
            cmd.run, gen, s["uri"], s["version"], cmd.entry_id, s["size"], s.get("meta"), **ckw
          ):
            return None, acct, dt
          if s["durable"]:
            await self.registry.register_checkpoint(
              s["uri"], s["size"], s["ttl"], s["owned"], s["version"], durable=True, **rkw
            )
          else:
            defer_save = s
        else:
          if await self.bus.get_generation(cmd.run) != gen:
            return None, acct, dt
          await self.registry.register_checkpoint(
            s["uri"], s["size"], s["ttl"], s["owned"], s["version"], durable=s["durable"], **rkw
          )
          if sid := s.get("sampling_session_id"):
            await self.bus.pin_sampler_weights(sid, s["version"])
      clear_rb = bool(ld := effects.get("loaded"))
      if ld and ld.get("optimizer"):
        ckw = {"durable": ld["durable"], "location": self.store.peer, "profile": ld.get("profile")}
        if not await self.bus.commit_job_state(cmd.run, gen, ld["uri"], ld["version"], cmd.entry_id, **ckw):
          return None, acct, dt
      wkw = {"started": before, "generation": gen, "mutates": bool(effects.get("mutated")), "clear_rollback": clear_rb}
      wkw |= {"profile": effects.get("profile"), "lora_config": effects.get("lora_config"), "defer_save": defer_save}
      if not await self.bus.write_reply(cmd, reply, worker, owner, **wkw):
        if await self.bus.lease_owner(worker) != owner:
          raise RuntimeError(f"lost worker {worker}")
        return None, acct, dt
      if defer_save and await asyncio.to_thread(self.store.is_durable, defer_save["uri"], defer_save["version"]):
        await self.bus.complete_durable_version(cmd.run, defer_save["version"], self.store.peer)
      if clear_rb:
        rolled_back.pop(cmd.run, None)
      if cmd.op == Op.DELETE_MODEL and reply.ok:
        forget_run(cmd.run)
      elif reply.ok or defer_save:
        boundaries[cmd.run] = cmd.entry_id if cmd.op != Op.SAVE_WEIGHTS_FOR_SAMPLER else (cur_boundary or cmd.entry_id)
      if DEBUG:
        print(f"{worker} {cmd.request_id} {cmd.op:24s} {(time.time() - before) * 1000:9.2f} ms")
      return reply, acct, dt

    async def run_batch(w: scheduler.Worker, candidates: list[Cmd], gens: dict[str, int]) -> Cmd | None:
      nonlocal reader, heads, last_run, turn_cmds, virtual, busy_since, busy_weights
      for c in candidates:
        local_svc[c.run] = max(local_svc.get(c.run, virtual), virtual)
      prefer = last_run if (not sampling and turn_cmds < RUN_COMMANDS) else None
      chosen = scheduler.select_runs(candidates, local_svc, SAMPLE_RUNS if sampling and spec.lora else 1, prefer)
      inputs = {c.run: prepared[c.request_id].result() for c in candidates if c.run in chosen}
      all_err = all(isinstance(v, Exception) for v in inputs.values())
      take_n = SAMPLE_COMMANDS if sampling and not all_err else 1
      if reader:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        reader = None
      await owned()
      cmds, claim_map = [], {}
      for r in chosen:
        if replays.get(r):
          cmds, replays[r] = [*cmds, *replays[r][:take_n]], replays[r][take_n:]
          if not replays[r]:
            replays.pop(r, None)
        elif r in gens:
          claim_map[r] = gens[r]
      if claim_map:
        with span("node.claim"):
          cmds += await self.bus.take_commands(claim_map, owner, take_n)
      heads = [c for c in heads if c.run not in chosen]
      if not cmds:
        return None
      if sampling:
        if all_err:
          await asyncio.gather(*(answer_sampler(c, gens[c.run], inputs[c.run]) for c in cmds))
          return None
        if not sample_tasks:
          busy_since, busy_weights = time.monotonic(), {}
        for c in cmds:
          local_svc[c.run], served[c.run] = local_svc.get(c.run, 0.0) + c.cost, served.get(c.run, 0) + c.cost
          sample_tasks[st := asyncio.create_task(answer_sampler(c, gens[c.run], inputs[c.run]))] = c
          st.add_done_callback(lambda _: self.notify())
        return None
      cmd = cmds[0]
      last_run, turn_cmds = cmd.run, (1 if cmd.run != last_run or prefer is None else turn_cmds + 1)
      reply, acct, dt = await answer_trainer(cmd, gens[cmd.run], inputs[cmd.run])
      charge({acct: 1.0}, dt)
      local_svc[cmd.run] = local_svc.get(cmd.run, 0.0) + cmd.cost
      if gpu_state == "active" and not all_err and cmd.run not in rolled_back:
        served[cmd.run] = served.get(cmd.run, 0) + cmd.cost
      if active_runs := {c.run for c in [*ready_heads(set(gens)), cmd] if c.run in local_svc}:
        virtual = max(virtual, min(local_svc[r] for r in active_runs))
      await report(w.revision, "running" if gpu_state == "active" else phase)
      return cmd if reply and reply.ok else None

    async def reap_samples(*, report_now: bool = True) -> None:
      nonlocal busy_since, busy_weights
      if not (done := [t for t in sample_tasks if t.done()]):
        return
      for t in done:
        c = sample_tasks.pop(t)
        acct, toks = (c.run, 0) if (t.cancelled() or t.exception()) else t.result()
        if toks > 0:
          busy_weights[acct] = busy_weights.get(acct, 0.0) + toks
          local_svc[c.run] = local_svc.get(c.run, 0.0) + max(0, toks - c.cost)
        else:
          served[c.run] = max(0, served.get(c.run, 0) - c.cost)
      if not sample_tasks and busy_since is not None:
        charge(busy_weights, time.monotonic() - busy_since)
        busy_since, busy_weights = None, {}
      if report_now:
        await report(seen_rev, "running" if gpu_state == "active" else phase)

    async def cleanup() -> None:
      nonlocal gpu_state, active_gpus, gpu_bytes, host_bytes
      futs = [f for f in (reader, *prepared.values(), *sample_tasks) if f]
      for f in futs:
        f.cancel()
      await asyncio.gather(*futs, return_exceptions=True)
      sample_tasks.clear()
      while model:
        try:
          with span("node.close"):
            await (model.close() if sampling else model.close(graceful=not failed))
          break
        except (Exception, asyncio.CancelledError) as e:
          print(f"{worker} could not close, retrying: {e!r}", file=sys.stderr, flush=True)
          await asyncio.sleep(TICK)
      gpu_state, active_gpus, gpu_bytes, host_bytes = "released", (), 0, 0
      if guard:
        guard.cancel()
        await asyncio.gather(guard, return_exceptions=True)
      if await self.bus.lease_owner(worker) == owner and not sampling:
        await self.bus.fail_stranded_commands(dict(loaded_gens), worker, owner, 0)
      await report(seen_rev, "failed" if failed else "released")
      self.running.pop(worker, None)
      await self.bus.release_lease(worker, owner)

    try:
      if not await self.bus.take_lease(worker, owner) or not await self.bus.renew_lease(worker, owner) or self.draining:
        return
      guard = asyncio.create_task(keepalive())
      if w0 := self.state.workers.get(worker):
        seen_rev = w0.revision
        await report(seen_rev, "preparing")

      while not self.draining:
        await reap_samples()
        w = self.state.workers.get(worker)
        if not w or w.node != self.id or not scheduler.jobs_of(self.state, worker):
          break
        if w.revision != seen_rev:
          if reader:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            reader = None
          seen_rev, served = w.revision, {}
        if model and (p := model.proc) and p.poll() is not None:
          failed = True
          raise RuntimeError(f"worker {worker} exited with {p.returncode}")

        await sync_membership(w)
        if w.intent == "drain":
          if w.drain_mode in ("export", "close"):
            break
          await park_local(w.revision)
          ch = self.changed
          if (w2 := self.state.workers.get(worker)) and w2.revision == w.revision and not self.draining:
            await ch.wait()
          continue
        if w.gpus and model is not None and bound_gpus and w.gpus != bound_gpus and not movable(spec, self.gpus):
          break

        await reconcile_jobs(w)
        peek_gens, gpu_gens = eligible_runs(w, for_gpu=False), eligible_runs(w, for_gpu=True)
        collect()
        if no_gpu := [
          c
          for c in ready_heads(set(peek_gens))
          if isinstance(prepared[c.request_id].result(), Exception)
          or (
            not sampling
            and (
              (c.run in rolled_back and c.op != Op.LOAD_WEIGHTS)
              or (c.op == Op.DELETE_MODEL and c.run not in loaded_gens)
            )
          )
        ]:
          await run_batch(w, no_gpu, peek_gens)
          continue

        runnable = [] if (sampling and not spec.lora and sample_tasks) else ready_heads(set(gpu_gens))
        warm = (
          sampling
          and w.intent in ("run", "idle")
          and bool(w.gpus)
          and self.gpus >= 2
          and (model is None or model.running is None)
        )
        if w.intent == "run" and w.gpus and (runnable or warm):
          if not await activate(w.gpus, w.revision):
            await self.changed.wait()
            continue
          if runnable:
            done_cmd = await run_batch(w, runnable, gpu_gens)
            if done_cmd and done_cmd.op == Op.DELETE_MODEL and not spec.lora:
              break
            continue
        if warm:
          await activate(w.gpus, w.revision)
        prep_ready = ready_heads(set(peek_gens))
        stalled = any(not f.done() for f in prepared.values()) and not prep_ready and not sample_tasks
        if stalled and (phase != "preparing" or reported_rev != w.revision):
          await report(w.revision, "preparing")
        elif not stalled and phase == "preparing" and (prep_ready or sample_tasks):
          await report(w.revision, "running" if gpu_state == "active" else "ready")
        elif (
          w.intent == "idle"
          and gpu_state == "active"
          and not sample_tasks
          and (phase != "idle" or reported_rev != w.revision)
        ):
          await report(w.revision, "idle")

        ch = self.changed
        if peek_gens and reader is None:
          (
            reader := asyncio.create_task(read_heads(list(peek_gens), IDLE_MS / 1000 if heads else 0.0))
          ).add_done_callback(lambda _: self.notify())
        await ch.wait()

      w_end = self.state.workers.get(worker)
      close_only = bool(w_end and w_end.drain_mode == "close") or not scheduler.jobs_of(self.state, worker)
      if model and not close_only:
        await report(seen_rev, "draining")
        if not sampling:
          for r, gen in list(loaded_gens.items()):
            await export_run(r, gen, priority=self.draining)
        else:
          await asyncio.wait_for(model.pause("wait"), timeout=DRAIN_S)
          if sample_tasks:
            await asyncio.gather(*list(sample_tasks), return_exceptions=True)
            await reap_samples(report_now=False)
    except (Exception, asyncio.CancelledError) as e:
      if not self.draining or isinstance(e, (TimeoutError, RuntimeError)):
        failed = True
      raise
    finally:
      if model and (p := model.proc) and p.poll() is not None:
        failed = True
      closing = asyncio.create_task(cleanup())
      self.notify()
      while True:
        try:
          await asyncio.shield(closing)
          break
        except asyncio.CancelledError:
          if closing.cancelled():
            closing = asyncio.create_task(cleanup())

  def dispatch(self) -> None:
    for worker, task in list(self.tasks.items()):
      if task.done():
        with contextlib.suppress(asyncio.CancelledError):
          if exc := task.exception():
            print(f"{worker} stopped: {exc!r}", file=sys.stderr, flush=True)
        del self.tasks[worker]
        self.running.pop(worker, None)
      elif self.id not in self.state.nodes and not task.cancelling():
        task.cancel()
    self.sync_prefetches()

  async def tick(self) -> None:
    self._attach_loop()
    local_w = {
      n: {"gpus": list(v["gpus"]), "pid": v.get("pid"), "gpu_state": v.get("gpu_state"), "phase": v.get("phase")}
      for n, v in self.running.items()
    }
    await self.bus.node_heartbeat(self.id, self.gpus, self.gpu_mib * 2**20, local_w, self.lag, **self.heartbeat_extra())
    self.heartbeat = time.monotonic()
    old, self.state = self.state, await self.read_state()
    self.dispatch()
    if not self.draining:
      mine = {
        n
        for n, w in self.state.workers.items()
        if w.node == self.id
        and scheduler.jobs_of(self.state, n)
        and not (w.intent == "drain" and w.drain_mode == "close")
        and not (
          (o := self.state.observations.get(n))
          and o.node == self.id
          and o.phase == "failed"
          and o.revision >= w.revision
        )
      }
      for worker in sorted(mine - self.tasks.keys()):
        self.start(worker)
    if time.time() - self.swept > SWEEP_S and (self.sweeping is None or self.sweeping.done()):
      self.swept = time.time()
      self.sweeping = asyncio.create_task(
        self.registry.sweep(self.store, SWEEP_GRACE_S, {n.peer for n in self.state.nodes.values() if n.peer} or None)
      )

    def sig(st: scheduler.Assignment) -> tuple:
      return {n: (w.node, w.revision, w.gpus, w.intent, w.generations) for n, w in st.workers.items()}, {
        r: j.queued for r, j in st.jobs.items()
      }

    if sig(old) != sig(self.state):
      self.notify()

  async def close(self, graceful: bool = True) -> None:
    self.draining = True
    for ev, _ in self.prefetches.values():
      ev.set()
    self.notify()
    if self.tasks:
      _, pending = await asyncio.wait(self.tasks.values(), timeout=DRAIN_S if graceful else 0)
      for t in pending:
        if not t.cancelling():
          t.cancel()
      await asyncio.gather(*self.tasks.values(), return_exceptions=True)
      self.dispatch()
    if self.prefetches:
      await asyncio.gather(*(t for _, t in self.prefetches.values()), return_exceptions=True)
      self.prefetches.clear()
    if self.sweeping:
      self.sweeping.cancel()
      await asyncio.gather(self.sweeping, return_exceptions=True)
    await asyncio.to_thread(self.store.close, graceful, DRAIN_S if graceful else 0.0)
    if self.durable_tasks:
      await asyncio.gather(*list(self.durable_tasks), return_exceptions=True)

  async def serve(self) -> None:
    await self.announce_node()
    try:
      while True:
        await self.tick()
        before = time.monotonic()
        await asyncio.sleep(TICK)
        self.lag = time.monotonic() - before - TICK
    finally:
      await self.close()


if __name__ == "__main__":

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await NodeDaemon(Bus()).serve()

  asyncio.run(main())
