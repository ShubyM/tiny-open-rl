"""Execute fleet grants. A process is a cache of job states; only committed versions survive its lifetime."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import socket
import sys
import time

from tinyrl.bus import LEASE_MS, Bus, Cmd, Op, Reply, Role, entry_key, key
from tinyrl.control import scheduler
from tinyrl.control.process import Process
from tinyrl.helpers import getenv
from tinyrl.storage.registry import Registry
from tinyrl.storage.store import Store
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
    self.gpus, self.id = gpus, node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = getenv("GPU_MEMORY_MIB", 0) if gpu_mib is None else gpu_mib
    self.tasks, self.running = {}, {}
    self.state, self.changed, self.draining = scheduler.Assignment(), asyncio.Event(), False
    self.heartbeat, self.swept, self.sweeping = time.monotonic(), 0.0, None

  def notify(self):
    event, self.changed = self.changed, asyncio.Event()
    event.set()

  async def read_state(self):
    return scheduler.from_table(await self.bus.read_table())

  def open_trainer(self, worker, env):
    width, cmd = self.state.workers[worker].spec.gpus, [sys.executable]
    if width > 1:
      cmd += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={width}"]
    return Process([*cmd, "-m", "tinyrl.workers.trainer"], env)

  def open_sampler(self, worker, env):
    spec = self.state.workers[worker].spec
    return Sampler(spec.base, spec.lora, env)

  def safe(self, worker, gpus):
    return not any(n != worker and not o["released"] and set(gpus) & set(o["gpus"]) for n, o in self.running.items())

  async def work(self, worker):
    spec, owner = self.state.workers[worker].spec, f"{self.id}-{os.urandom(8).hex()}"
    sampling, model, held, bound = spec.role == Role.SAMPLER, None, False, ()
    known, resident, prepared, replays, uploads = {}, {}, {}, {}, {}
    usage, served, weights, executing = {}, {}, {}, set()
    guard, closing, last_charge = None, False, time.monotonic()

    def charge():
      nonlocal last_charge
      now, total = time.monotonic(), sum(weights.values())
      if held and total:
        cost = (now - last_charge) * len(bound) * float(os.getenv("GPU_RATE", "1"))
        for account, weight in weights.items():
          usage[account] = usage.get(account, 0.0) + cost * weight / total
      last_charge = now

    async def report(*, starting=False, closed=False, revision=None):
      charge()
      w = self.state.workers.get(worker)
      obs = {
        "node": self.id,
        "revision": revision if revision is not None else w.revision if w else 0,
        "gpus": list(bound),
        "released": not held,
        "closed": closed,
        "pid": model.proc.pid if model and model.proc else None,
        "resident": {r: g for r, (g, _) in resident.items()},
        "served": dict(served),
        "usage": dict(usage),
        "ready": sorted(executing | {r for r, f in prepared.items() if f.done()}),
        "export": list(resident) if self.draining else [],
      }
      self.running[worker] = obs
      return await self.bus.write_observation(worker, owner, obs, starting=starting)

    async def keepalive():
      while True:
        await asyncio.sleep(TICK)
        if time.monotonic() - self.heartbeat >= LEASE_MS / 1000 or not await self.bus.renew_lease(worker, owner):
          self.tasks[worker].cancel()
          return
        if model and model.proc and model.proc.poll() is not None:
          self.tasks[worker].cancel()
          return
        if sampling and model and (w := self.state.workers.get(worker)):
          for r in list(resident):
            if self.draining or w.members.get(r) != resident[r][0]:
              await model.cancel_run(r)
              resident.pop(r, None)
        if not closing:
          await report()

    async def prepare(r, replay=None):
      cmd = replay
      while cmd is None:
        cmd = await self.bus.head(r, IDLE_MS)
      try:
        run, restore = await self.bus.get_run(r), None
        if not run or not run.get("state"):
          raise ValueError(f"model {r} was deleted")
        if sampling:
          if uri := run.get("model_path"):
            version = run.get("version") or await self.registry.version(uri)
            await self.bus.pin_weights(r, version)
            run["version"] = version
            restore = await asyncio.to_thread(self.store.local_folder, uri, version)
        elif r not in resident and cmd.op not in (Op.CREATE_MODEL, Op.LOAD_WEIGHTS, Op.DELETE_MODEL):
          st = run["state"]
          for cp in (st.get("working"), st.get("durable")):
            if cp:
              try:
                await asyncio.to_thread(self.store.local_folder, cp["uri"], cp["version"])
                restore = (cp["uri"], cp["version"])
                break
              except (OSError, ValueError):
                continue
          boundary = cp["boundary"] if restore else st.get("created", "")
          if entry_key(st.get("mutation", "")) > entry_key(boundary):
            raise RuntimeError(
              f"model {r} rolled back to {boundary or 'creation'}; explicitly load a checkpoint before continuing"
            )
        if cmd.op == Op.LOAD_WEIGHTS:
          args = json.loads(cmd.body)
          await asyncio.to_thread(self.store.local_folder, args["uri"], args["version"])
        return cmd, run, restore
      except Exception as exc:
        return cmd, exc, None

    async def finish_save(cmd, saved):
      try:
        await asyncio.to_thread(self.store.local_folder, saved["uri"], saved["version"])
        await asyncio.to_thread(self.store.upload_version, saved["uri"], saved["version"])
        saved = saved | {"durable": await asyncio.to_thread(self.store.is_durable, saved["uri"], saved["version"])}
        if not saved["durable"]:
          raise RuntimeError("checkpoint upload did not finish")
        reply = Reply(
          True, "application/json", json.dumps({"type": cmd.op, "model_id": cmd.run, "path": saved["uri"]}).encode()
        )
        await self.bus.write_reply(cmd, reply, worker, owner, checkpoint=saved)
      except Exception as exc:
        await self.bus.write_reply(cmd, Reply.error(exc), worker, owner)

    def upload(cmd, saved):
      f = uploads[cmd] = asyncio.create_task(finish_save(cmd, saved))
      f.add_done_callback(lambda _: (uploads.pop(cmd, None), self.notify()))

    async def answer(cmd, run, restore):
      before = time.time()
      checkpoint, mutated, defer = None, False, False
      try:
        if isinstance(run, Exception):
          reply = Reply.error(run)
        elif sampling:
          resident[cmd.run] = (cmd.generation, "")
          with (
            self.store.hold(run["model_path"], run["version"]) if run.get("model_path") else contextlib.nullcontext()
          ):
            reply = await model.execute(cmd, run, restore)
        else:
          with span("node.execute"):
            reply, effects = await model.call(
              "execute", cmd, run, restore, json.loads(cmd.body).get("version") if cmd.op == Op.LOAD_WEIGHTS else None
            )
          mutated, checkpoint = effects.get("mutated", False), effects.get("saved")
          if (loaded := effects.get("loaded")) and loaded["optimizer"]:
            checkpoint = loaded | {"boundary": cmd.entry_id}
          if checkpoint:
            checkpoint["durable"] = await asyncio.to_thread(
              self.store.is_durable, checkpoint["uri"], checkpoint["version"]
            )
            defer = cmd.op == Op.SAVE_WEIGHTS and not checkpoint["durable"]
          if effects.get("resident", reply.ok and cmd.op != Op.DELETE_MODEL):
            st = run["state"]
            boundary = next(
              (
                cp["boundary"]
                for cp in (st.get("working"), st.get("durable"))
                if cp and restore and cp["version"] == restore[1]
              ),
              st.get("created", ""),
            )
            resident[cmd.run] = (cmd.generation, cmd.entry_id if reply.ok else resident.get(cmd.run, (0, boundary))[1])
          else:
            resident.pop(cmd.run, None)
      except Exception as exc:
        if mutated or (model and (getattr(model, "broken", False) or (model.proc and model.proc.poll() is not None))):
          raise
        reply, checkpoint, defer = Reply.error(exc), None, False
      if not await self.bus.write_reply(
        cmd, reply, worker, owner, before, checkpoint=checkpoint, mutated=mutated, defer=defer
      ):
        raise RuntimeError("lost ownership while committing a command")
      if defer:
        upload(cmd, checkpoint)
      if checkpoint and (sid := json.loads(cmd.body).get("sampling_session_id")):
        await self.bus.pin_weights(sid, checkpoint["version"])

    async def park():
      nonlocal held
      if held and model:
        with span("node.park"):
          await (model.park() if sampling else model.call("park"))
        charge()
        held = False
        await report()

    async def activate(w):
      nonlocal model, held, bound
      if held:
        return self.state.workers.get(worker) == w
      if not self.safe(worker, w.gpus):
        return False
      bound, held = w.gpus, True  # reserve locally before any await or child startup
      if not await report(starting=True, revision=w.revision):
        held = False
        return False
      if model is None:
        env = {
          **os.environ,
          **self.store.env(),
          "CUDA_VISIBLE_DEVICES": ",".join(map(str, bound)),
        }
        model = self.open_sampler(worker, env) if sampling else self.open_trainer(worker, env)
      with span("node.resume"):
        if sampling:
          await model.resume()
        else:
          await model.call("resume")
      return self.state.workers.get(worker) == w

    try:
      if not await self.bus.take_lease(worker, owner):
        return
      await report()
      guard = asyncio.create_task(keepalive())
      guard.add_done_callback(
        lambda f: self.tasks[worker].cancel() if not closing and not f.cancelled() and f.exception() else None
      )
      while (w := self.state.workers.get(worker)) and w.node == self.id:
        if w.owner != owner:
          await self.changed.wait()
          continue
        for r, gen in w.members.items():
          if known.get(r) != gen:
            if f := prepared.pop(r, None):
              f.cancel()
            replays.pop(r, None)
            if r in resident and model:
              await model.cancel_run(r) if sampling else await model.call("drop", r)
              resident.pop(r, None)
            for c, saved in await self.bus.reclaim_commands({r: gen}, worker, owner):
              upload(c, saved) if saved else replays.setdefault(r, []).append(c)
            known[r] = gen
        outgoing = [r for r in resident if self.draining or w.members.get(r) != resident[r][0]]
        for r in outgoing:
          if sampling:
            await model.cancel_run(r)
          else:
            if not w.gpus or r not in w.budgets:
              continue
            charge()
            weights = {self.state.jobs[r].account: 1}
            if not await activate(w):
              continue
            gen, boundary = resident[r]
            saved = await model.call("save", r, f"tinker://{r}/parked", False, None, True, boundary)
            saved["durable"] = await asyncio.to_thread(self.store.is_durable, saved["uri"], saved["version"])
            if not await self.bus.write_reply(
              Cmd(r, boundary, Op.SAVE_WEIGHTS, b"{}", generation=gen), None, worker, owner, checkpoint=saved
            ):
              raise RuntimeError("lost job while exporting its state")
            await model.call("drop", r)
            charge()
            weights = {}
          resident.pop(r, None)
          await report()
        if (self.draining or not w.members) and not resident:
          break
        wanted = {} if self.draining else w.members
        for r in list(prepared):
          if r not in wanted:
            prepared.pop(r).cancel()
        for r in wanted:
          if r not in prepared:
            replay = replays.get(r, [])
            f = prepared[r] = asyncio.create_task(prepare(r, replay.pop(0) if replay else None))
            f.add_done_callback(lambda _: self.notify())
        ready = {r: f.result() for r, f in prepared.items() if f.done() and not f.cancelled()}
        await report()
        candidates = [
          r for r, (_, run, _) in ready.items() if isinstance(run, Exception) or served.get(r, 0) < w.budgets.get(r, 0)
        ]
        candidates = candidates[: SAMPLE_RUNS if sampling and spec.lora else 1]
        if candidates:
          charge()
          weights = {self.state.jobs[r].account: 1 for r in candidates}
        if candidates and (
          all(isinstance(ready[r][1], Exception) for r in candidates) or (w.gpus and await activate(w))
        ):
          limits = {r: max(1, w.budgets.get(r, 0) - served.get(r, 0)) for r in candidates}
          replayed = [ready[r][0] for r in candidates if ready[r][0].generation is not None]
          claimed = await self.bus.take_commands(
            {r: w.members[r] for r in candidates if ready[r][0].generation is None},
            owner,
            SAMPLE_COMMANDS if sampling else 1,
            limits=limits,
          )
          cmds = replayed + claimed
          charge()
          weights = {}
          for c in cmds:
            served[c.run] = served.get(c.run, 0) + c.cost
            account = self.state.jobs[c.run].account
            weights[account] = weights.get(account, 0) + c.cost
          executing = {c.run for c in cmds}
          async with asyncio.TaskGroup() as batch:
            for c in cmds:
              batch.create_task(answer(c, *ready[c.run][1:]))
          charge()
          executing.clear()
          for r in candidates:
            prepared.pop(r, None)
          continue
        await park()
        await self.changed.wait()
    except Exception as exc:
      for c in await self.bus.take_commands({r: known[r] for r in prepared if r in known}, owner, count=1):
        await self.bus.write_reply(c, Reply.error(exc), worker, owner)
      raise
    finally:
      closing = True

      async def cleanup():
        nonlocal held
        futures = [f for f in [guard, *prepared.values(), *uploads.values()] if f]
        for f in futures:
          f.cancel()
        await asyncio.gather(*futures, return_exceptions=True)
        if model:
          await model.close()
        charge()
        held = False
        await report(closed=True)
        self.running.pop(worker, None)
        await self.bus.release_lease(worker, owner)

      cleanup_task = asyncio.create_task(cleanup())
      while not cleanup_task.done():
        with contextlib.suppress(asyncio.CancelledError):
          await asyncio.shield(cleanup_task)
      cleanup_task.result()

  async def tick(self):
    await self.bus.node_heartbeat(self.id, self.gpus, self.gpu_mib * 2**20, rate=float(os.getenv("GPU_RATE", "1")))
    self.heartbeat = time.monotonic()
    old, self.state = self.state, await self.read_state()
    for n, t in list(self.tasks.items()):
      if t.done():
        if not t.cancelled() and t.exception():
          print(f"{n} stopped: {t.exception()!r}", file=sys.stderr, flush=True)
        del self.tasks[n]
      elif n not in self.state.workers or self.state.workers[n].node != self.id:
        t.cancel()
    if not self.draining:
      for n, w in self.state.workers.items():
        if w.node == self.id and n not in self.tasks and n not in self.running and w.members:
          self.tasks[n] = task = asyncio.create_task(self.work(n))
          task.add_done_callback(lambda _: self.notify())
    if time.time() - self.swept > SWEEP_S and (self.sweeping is None or self.sweeping.done()):
      self.swept = time.time()
      self.sweeping = asyncio.create_task(self.registry.sweep(self.store, SWEEP_GRACE_S))
    if old != self.state:
      self.notify()

  async def close(self, graceful=True):
    self.draining = True
    self.notify()
    if graceful:
      with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(DRAIN_S):
          while any(not t.done() for t in self.tasks.values()):
            await self.tick()
            await asyncio.sleep(TICK)
    for t in self.tasks.values():
      t.cancel()
    await asyncio.gather(*self.tasks.values(), return_exceptions=True)
    if self.sweeping:
      self.sweeping.cancel()
      await asyncio.gather(self.sweeping, return_exceptions=True)
    await asyncio.to_thread(self.store.close)
    await self.bus.r.hdel(key("nodes"), self.id)

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
