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
from collections import Counter

from tinyrl.bus import LEASE_MS, Bus, Op, Reply, Role
from tinyrl.control.process import Process
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry
from tinyrl.storage.store import Store
from tinyrl.trace import span
from tinyrl.workers.sampler import Sampler

GPUS, DRAIN_S, TICK = getenv("GPUS", 1), getenv("DRAIN_S", 20.0), getenv("TICK", 0.25)
IDLE_MS, SAMPLE_RUNS, SAMPLE_COMMANDS = 150, getenv("SAMPLE_RUNS", 3), getenv("SAMPLE_COMMANDS", 128)


async def cancel(*tasks):
  for task in tasks:
    task.cancel()
  await asyncio.gather(*tasks, return_exceptions=True)


async def finish(coro):
  """Complete physical cleanup before allowing cancellation to leave its owner."""
  task = asyncio.create_task(coro)
  try:
    return await asyncio.shield(task)
  except asyncio.CancelledError:
    while not task.done():
      with contextlib.suppress(asyncio.CancelledError):
        await asyncio.shield(task)
    task.result()
    raise


class NodeDaemon:
  def __init__(
    self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store: Store | None = None
  ):
    self.bus, self.store = bus, store if store is not None else Store()
    self.store.start()
    self.gpus, self.id = gpus, node or getenv("NODE_NAME", socket.gethostname())
    self.gpu_mib = getenv("GPU_MEMORY_MIB", 0) if gpu_mib is None else gpu_mib
    self.tasks = {}
    self.state, self.changed, self.draining = {}, asyncio.Event(), False
    self.heartbeat, self.maintenance = time.monotonic(), None

  async def serve(self):
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    try:
      while True:
        await self.tick()
        await asyncio.sleep(TICK)
    finally:
      await self.close()

  async def tick(self):
    await self.bus.node_heartbeat(
      self.id, self.gpus, self.gpu_mib * 2**20, draining=self.draining, rate=float(os.getenv("GPU_RATE", "1"))
    )
    self.heartbeat = time.monotonic()
    old, self.state = self.state, await self.bus.read_table()
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
          self.tasks[engine] = task = asyncio.create_task(self.serve_engine(engine))
          task.add_done_callback(lambda _: self.notify())
    if self.maintenance is None:
      self.maintenance = asyncio.create_task(Registry(self.bus).maintain(self.store, TICK))
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
    await cancel(*tasks)
    await asyncio.to_thread(self.store.close)

  async def serve_engine(self, engine):
    """One process keeps its tensors between the fleet's turns."""
    run = next((self.state["runs"][job] for job in self.jobs(engine)), None)
    if run is None or not await self.bus.start_engine(engine, self.id):
      return
    model, gpus, pending = None, (), {}
    owner = asyncio.current_task()

    async def guard():
      while True:
        await asyncio.sleep(TICK)
        if time.monotonic() - self.heartbeat >= LEASE_MS / 1000 or not await self.bus.renew_lease(engine):
          owner.cancel()
          return
        if model and model.proc and model.proc.poll() is not None:
          owner.cancel()
          return
        if isinstance(model, Sampler):
          await model.retain(self.jobs(engine))

    keepalive = asyncio.create_task(guard())
    keepalive.add_done_callback(lambda task: owner.cancel() if not task.cancelled() and task.exception() else None)
    try:
      while jobs := self.jobs(engine):
        changed = self.changed
        for job in pending.keys() - jobs:
          await cancel(pending.pop(job))
        for job in jobs - pending.keys():
          self.stage(pending, job)
        if not await self.observe(engine, gpus, model, pending):
          raise RuntimeError("engine lease expired")
        granted = next(
          ((t, grant) for t, grant in self.state.get("turns", {}).items() if grant["engine"] == engine), None
        )
        if granted is None:
          await changed.wait()
          continue
        turn, grant = granted
        if gpus and gpus != tuple(grant["gpus"]):
          raise RuntimeError("a live engine cannot change physical GPUs")
        gpus = tuple(grant["gpus"])
        if not await self.observe(engine, gpus, model, pending, turn=turn):
          await changed.wait()
          continue
        if model is None:
          env = {**os.environ, **self.store.env(), "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
          model = (
            self.open_sampler(engine, run, env) if run["role"] == Role.SAMPLER else self.open_trainer(engine, run, env)
          )
        await self.run_turn(turn, grant, model, pending, keepalive)
        self.state.get("turns", {}).pop(turn, None)
    finally:
      await finish(self.close_engine(engine, gpus, model, [keepalive, *pending.values()]))

  async def run_turn(self, turn, grant, model, pending, keepalive):
    """Own one grant through execution and confirmed park or exit."""
    engine, usage, started, parked = grant["engine"], Counter(), time.monotonic(), False
    try:
      with span("node.resume"):
        await model.call("resume")
      await model.call("retain", list(self.jobs(engine)))
      for job, (_, allowance) in grant["work"].items():
        if allowance == 0:
          record = self.state["runs"].get(job, {})
          usage[record.get("train_run") or job] += 1
          await self.export_job(engine, turn, job, model, pending)
      await self.run_commands(turn, grant, model, pending, usage)
      with span("node.park"):
        await model.call("park")
      parked = True
    finally:

      async def release():
        if not parked:
          await cancel(keepalive)
          await model.close()
        elapsed = (time.monotonic() - started) * len(grant["gpus"]) * float(os.getenv("GPU_RATE", "1"))
        total = sum(usage.values())
        await self.bus.release_turn(
          turn, engine, {account: elapsed * n / total for account, n in usage.items()} if total else {}
        )

      await finish(release())

  async def run_commands(self, turn, grant, model, pending, usage):
    """Consume the grant in bounded waves, staging each next queue head."""
    engine = grant["engine"]
    sampling = isinstance(model, Sampler)
    limit = SAMPLE_RUNS if sampling and model.lora else 1
    remaining = {job: allowance for job, (_, allowance) in grant["work"].items() if allowance > 0}
    while waiting := [pending[job] for job in remaining if job in pending]:
      await asyncio.wait(waiting, timeout=IDLE_MS / 1000, return_when=asyncio.FIRST_COMPLETED)
      ready = [job for job in remaining if job in pending and pending[job].done()][:limit]
      if not ready:
        break
      async with asyncio.TaskGroup() as batch:
        for job in ready:
          prepared = pending.pop(job)
          failed = prepared.exception() is not None
          if not failed and prepared.result() is None:
            remaining[job] = 0
            continue
          commands = await self.bus.take_commands(
            turn, job, count=SAMPLE_COMMANDS if sampling and not failed else 1, limit=remaining[job]
          )
          record = self.state["runs"].get(job, {})
          for cmd in commands:
            usage[record.get("train_run") or job] += cmd.cost
            remaining[job] -= cmd.cost
            batch.create_task(self.answer(model, cmd, prepared, engine))
          if failed:
            remaining[job] = 0
      for job in ready:
        if remaining[job] > 0 and job in self.jobs(engine):
          self.stage(pending, job)
        else:
          remaining.pop(job)

  async def export_job(self, engine, turn, job, model, pending):
    if task := pending.pop(job, None):
      await cancel(task)
    run = await self.bus.get_run(job)
    restore = await self.inputs(run)
    try:
      saved = await model.call("export", job, run, restore)
    except (OSError, RuntimeError) as exc:
      if getattr(model, "broken", False):
        raise
      print(f"export {job} failed, keeping its live state: {exc}", file=sys.stderr, flush=True)
      return
    if not await self.bus.export(turn, job, await self.checkpoint(saved), engine):
      raise RuntimeError("lost ownership while exporting a job")
    await model.call("drop", job)

  async def answer(self, model, cmd, prepared, engine):
    started, effects = time.time(), {}
    if error := prepared.exception():
      reply = Reply.error(error)
    else:
      run, restore = prepared.result()
      hold = run["role"] == Role.SAMPLER and run.get("model_path")
      with (
        self.store.hold(run["model_path"], run["version"]) if hold else contextlib.nullcontext(),
        span("node.execute"),
      ):
        reply, effects = await model.call("execute", cmd, run, restore)
    saved = await self.checkpoint(effects.get("checkpoint"))
    defer = bool(saved and cmd.op == Op.SAVE_WEIGHTS and not saved["durable"])
    if not await self.bus.write_reply(
      cmd, reply, engine, started, checkpoint=saved, mutated=effects.get("mutated", False), defer=defer
    ):
      raise RuntimeError("lost ownership while committing a command")

  async def prepare(self, job):
    """Wait for a queue head and stage its exact inputs before asking for GPU time."""
    while job in self.state.get("claims", {}) and not self.draining and job not in self.state.get("moves", {}):
      if cmd := await self.bus.head(job, IDLE_MS):
        run = await self.bus.get_run(job)
        return run, await self.inputs(run, cmd)

  async def inputs(self, run, cmd=None):
    if not run or not run.get("state"):
      raise ValueError("model was deleted")
    if run["role"] == Role.SAMPLER:
      if uri := run.get("model_path"):
        return await asyncio.to_thread(self.store.local_folder, uri, run["version"])
    elif cmd and cmd.op == Op.LOAD_WEIGHTS:
      args = json.loads(cmd.body)
      await asyncio.to_thread(self.store.local_folder, args["uri"], args["version"])
    elif cmd is None or cmd.op not in (Op.CREATE_MODEL, Op.DELETE_MODEL):
      for cp in (run["state"].get("working"), run["state"].get("durable")):
        if cp:
          try:
            await asyncio.to_thread(self.store.local_folder, cp["uri"], cp["version"])
            return cp["uri"], cp["version"]
          except (OSError, ValueError):
            continue

  async def checkpoint(self, saved):
    if saved:
      saved["durable"] = False
      with contextlib.suppress(OSError):
        saved["durable"] = await asyncio.to_thread(self.store.is_durable, saved["uri"], saved["version"])
    return saved

  async def close_engine(self, engine, gpus, model, tasks):
    """Confirm process death before retiring any remaining tickets."""
    await cancel(*tasks)
    if model:
      await model.close()
    await self.observe(engine, gpus, model, {}, closed=True)
    # Closing fences new grants before we collect tickets this engine never observed.
    for turn, grant in (await self.bus.read_table())["turns"].items():
      if grant["engine"] == engine:
        await self.bus.release_turn(turn, engine, {})
    await self.bus.release_lease(engine)

  async def observe(self, engine, gpus, model, pending, *, turn=None, closed=False):
    return await self.bus.write_observation(
      engine,
      {
        "node": self.id,
        "gpus": list(gpus),
        "closed": closed,
        "draining": self.draining,
        "ready": sorted(job for job, task in pending.items() if task.done() and not task.cancelled()),
        "pid": model.proc.pid if model and model.proc else None,
      },
      turn=turn,
    )

  def open_trainer(self, engine, run, env):
    cmd = [sys.executable]
    if run["gpus"] > 1:
      cmd += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={run['gpus']}"]
    return Process([*cmd, "-m", "tinyrl.workers.trainer"], env)

  def open_sampler(self, engine, run, env):
    return Sampler(run["base"], run["rank"] > 0, env)

  def jobs(self, engine):
    return {r for r, c in self.state.get("claims", {}).items() if c["engine"] == engine}

  def stage(self, pending, job):
    pending[job] = task = asyncio.create_task(self.prepare(job))
    task.add_done_callback(lambda _: self.notify())

  def notify(self):
    event, self.changed = self.changed, asyncio.Event()
    event.set()


if __name__ == "__main__":
  with contextlib.suppress(asyncio.CancelledError):
    asyncio.run(NodeDaemon(Bus()).serve())
