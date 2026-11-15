"""Keep job owners warm; execute the fleet's finite turns and acknowledge physical release."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import socket
import sys
import time

from tinyrl.bus import LEASE_MS, Bus, Op, Reply, Role
from tinyrl.control.process import Process
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry
from tinyrl.storage.store import Store, manifest
from tinyrl.trace import span
from tinyrl.workers.sampler import Sampler

GPUS, DRAIN_S, TICK = getenv("GPUS", 1), getenv("DRAIN_S", 20.0), getenv("TICK", 0.25)
IDLE_MS, SAMPLE_RUNS, SAMPLE_COMMANDS = 150, getenv("SAMPLE_RUNS", 3), getenv("SAMPLE_COMMANDS", 128)
SWEEP_S, SWEEP_GRACE_S = 60, 600


class NodeDaemon:
  def __init__(
    self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store: Store | None = None
  ):
    self.bus, self.registry, self.store = bus, Registry(bus), store if store is not None else Store()
    self.store.start()
    self.gpus, self.id = gpus, node or getenv("NODE_NAME", socket.gethostname())
    self.gpu_mib = getenv("GPU_MEMORY_MIB", 0) if gpu_mib is None else gpu_mib
    self.tasks, self.held = {}, {}
    self.state, self.changed, self.draining = {}, asyncio.Event(), False
    self.heartbeat, self.maintenance = time.monotonic(), None

  def notify(self):
    event, self.changed = self.changed, asyncio.Event()
    event.set()

  async def read_state(self):
    return await self.bus.read_table()

  def jobs(self, engine):
    return {r: c for r, c in self.state.get("claims", {}).items() if c["engine"] == engine}

  def open_trainer(self, engine, run, env):
    cmd = [sys.executable]
    if run["gpus"] > 1:
      cmd += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={run['gpus']}"]
    return Process([*cmd, "-m", "tinyrl.workers.trainer"], env)

  def open_sampler(self, engine, run, env):
    return Sampler(run["base"], run["rank"] > 0, env)

  async def prepare(self, r, export=False):
    """Stage exact inputs while the engine has no GPU turn. Live tensors are the trainer's concern."""
    cmd, restore = None, None
    while not export and not self.draining and r not in self.state.get("moves", {}):
      if cmd := await self.bus.head(r, IDLE_MS):
        break
      if r not in self.state.get("claims", {}):
        return None, ValueError(f"model {r} was deleted"), None
    try:
      if not (run := await self.bus.get_run(r)) or not run.get("state"):
        raise ValueError(f"model {r} was deleted")
      if run["role"] == Role.SAMPLER:
        if uri := run.get("model_path"):
          restore = await asyncio.to_thread(self.store.local_folder, uri, run["version"])
      elif cmd is None or cmd.op not in (Op.CREATE_MODEL, Op.LOAD_WEIGHTS, Op.DELETE_MODEL):
        for cp in (run["state"].get("working"), run["state"].get("durable")):
          if cp:
            try:
              await asyncio.to_thread(self.store.local_folder, cp["uri"], cp["version"])
              restore = (cp["uri"], cp["version"])
              break
            except (OSError, ValueError):
              continue
      if cmd and cmd.op == Op.LOAD_WEIGHTS:
        args = json.loads(cmd.body)
        await asyncio.to_thread(self.store.local_folder, args["uri"], args["version"])
      return cmd, run, restore
    except Exception as exc:
      return cmd, exc, None

  async def answer(self, model, cmd, run, restore, engine):
    started, checkpoint, mutated, defer = time.time(), None, False, False
    if isinstance(run, Exception):
      reply = Reply.error(run)
    elif run["role"] == Role.SAMPLER:
      with self.store.hold(run["model_path"], run["version"]) if run.get("model_path") else contextlib.nullcontext():
        reply = await model.execute(cmd, run, restore)
    else:
      with span("node.execute"):
        reply, effects = await model.call("execute", cmd, run, restore)
      mutated, checkpoint = effects.get("mutated", False), effects.get("checkpoint")
      if checkpoint:
        checkpoint["durable"] = False
        with contextlib.suppress(OSError):
          checkpoint["durable"] = await asyncio.to_thread(
            self.store.is_durable, checkpoint["uri"], checkpoint["version"]
          )
        defer = cmd.op == Op.SAVE_WEIGHTS and not checkpoint["durable"]
    if not await self.bus.write_reply(cmd, reply, engine, started, checkpoint=checkpoint, mutated=mutated, defer=defer):
      raise RuntimeError("lost ownership while committing a command")

  async def maintain(self):
    """SAVE completion belongs to storage, independently of any trainer or GPU turn."""
    swept = 0.0
    while True:
      for request, cp in (await self.bus.pending_saves()).items():
        with contextlib.suppress(OSError):
          if manifest(self.store.folder(cp["version"])):
            await asyncio.to_thread(self.store.upload_version, cp["uri"], cp["version"])
          if await asyncio.to_thread(self.store.is_durable, cp["uri"], cp["version"]):
            await self.bus.finish_save(request, cp)
      if time.time() - swept > SWEEP_S:
        await self.registry.sweep(self.store, SWEEP_GRACE_S)
        swept = time.time()
      await asyncio.sleep(TICK)

  async def work(self, engine):
    if not await self.bus.start_engine(engine, self.id):
      return
    run = self.state["runs"][next(iter(self.jobs(engine)))]
    sampling, model, bound = run["role"] == Role.SAMPLER, None, ()
    pending, active, weights, started = {}, None, {}, 0.0
    task = asyncio.current_task()

    async def report(*, turn=None, closed=False):
      obs = {
        "node": self.id,
        "gpus": list(bound),
        "closed": closed,
        "draining": self.draining,
        "ready": sorted(r for r, f in pending.items() if f.done() and not f.cancelled()),
        "pid": model.proc.pid if model and model.proc else None,
      }
      return await self.bus.write_observation(engine, obs, turn=turn)

    async def guard():
      while True:
        await asyncio.sleep(TICK)
        if time.monotonic() - self.heartbeat >= LEASE_MS / 1000 or not await self.bus.renew_lease(engine):
          task.cancel()
          return
        if model and model.proc and model.proc.poll() is not None:
          task.cancel()
          return
        if sampling and model:
          for r in (set(model.adapters) | {r for r, _ in model.inflight.values()}) - self.jobs(engine).keys():
            await model.cancel_run(r)

    async def release():
      nonlocal active
      if active:
        total = sum(weights.values())
        elapsed = (time.monotonic() - started) * len(bound) * float(os.getenv("GPU_RATE", "1"))
        usage = {a: elapsed * n / total for a, n in weights.items()} if total else {}
        await self.bus.release_turn(active, engine, usage)
        active = None
      self.held.pop(engine, None)

    def stage(r):
      pending[r] = future = asyncio.create_task(self.prepare(r))
      future.add_done_callback(lambda _: self.notify())

    keepalive = asyncio.create_task(guard())
    keepalive.add_done_callback(lambda f: task.cancel() if not f.cancelled() and f.exception() else None)
    try:
      while self.jobs(engine):
        for r in pending.keys() - self.jobs(engine).keys():
          pending.pop(r).cancel()
        for r in self.jobs(engine).keys() - pending.keys():
          stage(r)
        if not await report():
          raise RuntimeError("engine lease expired")
        granted = next(((t, v) for t, v in self.state.get("turns", {}).items() if v["engine"] == engine), None)
        if granted is None:
          await self.changed.wait()
          continue
        turn, ticket = granted
        # The ticket reserves capacity until release, even while a slow process is starting or stopping.
        if any(e != engine and set(ticket["gpus"]) & set(gs) for e, gs in self.held.items()):
          raise RuntimeError("overlapping GPU turns")
        if bound and bound != tuple(ticket["gpus"]):
          raise RuntimeError("a live engine cannot change physical GPUs")
        bound = tuple(ticket["gpus"])
        self.held[engine] = bound
        if not await report(turn=turn):
          self.held.pop(engine, None)
          await self.changed.wait()
          continue
        active, weights, started = turn, {}, time.monotonic()
        if model is None:
          env = {**os.environ, **self.store.env(), "CUDA_VISIBLE_DEVICES": ",".join(map(str, bound))}
          model = self.open_sampler(engine, run, env) if sampling else self.open_trainer(engine, run, env)
        with span("node.resume"):
          await (model.resume() if sampling else model.call("resume"))
        if not sampling:
          await model.call("retain", list(self.jobs(engine)))
        remaining = {r: limit for r, (_, limit) in ticket["work"].items()}
        for r in [r for r, limit in remaining.items() if limit == 0]:
          del remaining[r]
          if f := pending.pop(r, None):
            f.cancel()
            await asyncio.gather(f, return_exceptions=True)
          _, record, restore = await self.prepare(r, export=True)
          if isinstance(record, Exception):
            raise record
          weights[record.get("train_run") or r] = 1
          try:
            saved = None if sampling else await model.call("export", r, record, restore)
          except (OSError, RuntimeError) as exc:
            if getattr(model, "broken", False):
              raise
            print(f"export {r} failed, keeping its live state: {exc}", file=sys.stderr, flush=True)
            continue
          if saved:
            saved["durable"] = False
            with contextlib.suppress(OSError):
              saved["durable"] = await asyncio.to_thread(self.store.is_durable, saved["uri"], saved["version"])
          if not await self.bus.export(turn, r, saved, engine):
            raise RuntimeError("lost ownership while exporting a job")
          await (model.cancel_run(r) if sampling else model.call("drop", r))
        while remaining:
          waiting = [pending[r] for r in remaining if r in pending]
          if not waiting:
            break
          await asyncio.wait(waiting, timeout=IDLE_MS / 1000, return_when=asyncio.FIRST_COMPLETED)
          ready = [r for r in remaining if r in pending and pending[r].done()]
          ready = ready[: SAMPLE_RUNS if sampling and run["rank"] else 1]
          if not ready:
            break
          commands = []
          for r in ready:
            head, record, restore = pending.pop(r).result()
            if head is None:
              remaining[r] = 0
              continue
            batch = await self.bus.take_commands(
              turn,
              r,
              count=SAMPLE_COMMANDS if sampling and not isinstance(record, Exception) else 1,
              limit=remaining[r],
            )
            for cmd in batch:
              account = record.get("train_run") or r if isinstance(record, dict) else r
              weights[account] = weights.get(account, 0) + cmd.cost
              remaining[r] -= cmd.cost
              commands.append((cmd, record, restore))
            if isinstance(record, Exception):
              remaining[r] = 0
          async with asyncio.TaskGroup() as batch:
            for cmd, record, restore in commands:
              batch.create_task(self.answer(model, cmd, record, restore, engine))
          for r in ready:
            if (
              remaining[r] > 0 and r in self.jobs(engine) and r not in self.state.get("moves", {}) and not self.draining
            ):
              stage(r)
            else:
              remaining.pop(r)
        with span("node.park"):
          await (model.park() if sampling else model.call("park"))
        await release()
        self.state.get("turns", {}).pop(turn, None)
    finally:

      async def cleanup():
        for f in [keepalive, *pending.values()]:
          f.cancel()
        await asyncio.gather(keepalive, *pending.values(), return_exceptions=True)
        if model:
          await model.close()
        await report(closed=True)  # fence new tickets before collecting any that never reached activation
        for t, grant in (await self.bus.read_table())["turns"].items():
          if grant["engine"] == engine and t != active:
            await self.bus.release_turn(t, engine, {})
        await release()  # failed native cleanup leaves both the observation and physical turn occupied
        await self.bus.release_lease(engine)

      closing = asyncio.create_task(cleanup())
      while not closing.done():
        with contextlib.suppress(asyncio.CancelledError):
          await asyncio.shield(closing)
      closing.result()

  async def tick(self):
    await self.bus.node_heartbeat(
      self.id, self.gpus, self.gpu_mib * 2**20, draining=self.draining, rate=float(os.getenv("GPU_RATE", "1"))
    )
    self.heartbeat = time.monotonic()
    old, self.state = self.state, await self.read_state()
    for engine, task in list(self.tasks.items()):
      if task.done():
        if not task.cancelled() and task.exception():
          print(f"{engine} stopped: {task.exception()!r}", file=sys.stderr, flush=True)
        del self.tasks[engine]
      elif not self.jobs(engine):
        task.cancel()
    if not self.draining:
      for claim in self.state["claims"].values():
        engine = claim["engine"]
        if claim["node"] == self.id and engine not in self.tasks and engine not in self.state["observations"]:
          self.tasks[engine] = task = asyncio.create_task(self.work(engine))
          task.add_done_callback(lambda _: self.notify())
    if self.maintenance is None:
      self.maintenance = asyncio.create_task(self.maintain())
    if old != self.state:
      self.notify()

  async def close(self, graceful=True):
    self.draining = True
    await self.bus.node_heartbeat(self.id, self.gpus, draining=True)
    self.notify()
    if graceful:
      with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(DRAIN_S):
          while any(not t.done() for t in self.tasks.values()):
            await self.tick()
            await asyncio.sleep(TICK)
    tasks = [*self.tasks.values(), *([self.maintenance] if self.maintenance else [])]
    for t in tasks:
      t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.to_thread(self.store.close)

  async def serve(self):
    try:
      while True:
        await self.tick()
        await asyncio.sleep(TICK)
    finally:
      await self.close()


if __name__ == "__main__":

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await NodeDaemon(Bus()).serve()

  asyncio.run(main())
