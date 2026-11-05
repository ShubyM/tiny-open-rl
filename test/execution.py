"""Real node control with a synchronous CPU model in place of the local socket process."""

import asyncio
import time

from test.helpers import FakeEngine, tiny
from tinyrl.bus import Ev, Role
from tinyrl.control import scheduler
from tinyrl.control.node_daemon import LocalWorker, NodeDaemon, TrainerEngine
from tinyrl.workers.model import Model
from tinyrl.workers.sampler import Sampler
from tinyrl.workers.trainer import Trainer


class InlineProcess:
  def __init__(self, trainer: Trainer):
    self.trainer, self.proc, self.closed = trainer, self, False
    self.pid = None

  async def call(self, method, *args):
    if self.closed:
      raise EOFError("the model process exited")
    result = getattr(self.trainer, method)(*args)
    await asyncio.sleep(0)  # an actual socket call lets other requests and the node loop run
    return result

  def poll(self):
    return 0 if self.closed else None

  def kill(self):
    self.closed = True

  async def close(self):
    self.closed = True


def watch(process: InlineProcess, delays: dict[str, float] | None = None) -> list[tuple[float, str, tuple]]:
  log, call = [], process.call

  async def recorded(method, *args):
    await asyncio.sleep((delays or {}).get(method, 0))
    log.append((time.time(), method, args))
    return await call(method, *args)

  process.call = recorded
  return log


def gate(process: InlineProcess, method: str, after: bool = False) -> tuple[asyncio.Event, asyncio.Event]:
  entered, go, call = asyncio.Event(), asyncio.Event(), process.call

  async def held(name, *args):
    first = name == method and not entered.is_set()
    if first and not after:
      entered.set()
      await go.wait()
    result = await call(name, *args)
    if first and after:
      entered.set()
      await go.wait()
    return result

  process.call = held
  return entered, go


class TestNode(NodeDaemon):
  def __init__(self, bus, store=None, gpus=8, node="test-node", model=tiny, engine=FakeEngine):
    super().__init__(bus, gpus=gpus, node=node, gpu_mib=0, store=store)
    self.factories, self.default_model, self.engine_type = {}, model, engine
    self.note(Ev.NODE_UP, node=self.id, gpus=gpus, memory=0)

  def engine(self, w, lease):
    if w.kind.role == Role.SAMPLER:
      return Sampler(w.kind.base, w.kind.lora, scheduler.settings(w.kind, self.gpus, 0), engine=self.engine_type)
    trainer = Trainer(Model(self.factories.get(lease.worker, self.default_model)), self.store)
    return TrainerEngine(InlineProcess(trainer), lease, self.registry)

  async def sync(self):
    while events := await self.bus.read_events(self.after):
      for event in events:
        self.state = scheduler.fold(self.state, event)
      self.after = events[-1].id
      self.seen += len(events)

  async def add(self, worker, gpus=None) -> LocalWorker:
    await self.sync()
    if (lw := self.local.get(worker)) and not lw.task.done():
      self.note(Ev.PLACED, worker=worker, node=self.id, gpus=list(self.state.workers[worker].gpus))
      return lw
    self.local.pop(worker, None)
    self.note(Ev.PLACED, worker=worker, node=self.id, **({"gpus": gpus} if gpus is not None else {}))
    self.schedule()
    if (lw := await self.open(worker)) is None:
      raise AssertionError(f"could not acquire {worker}")
    self.local[worker] = lw
    return lw

  async def worker_for(self, run):
    await self.sync()
    for kind, body in scheduler.decide(self.state, time.time()):
      if kind == Ev.ASSIGNED:
        await self.bus.assign_run_to_worker(body["run"], body["worker"])
    await self.sync()
    return self.state.runs[run].worker

  async def drain(self, run, model=None, timeout=30) -> LocalWorker | None:
    await self.sync()
    if run not in self.state.runs:
      return None
    worker = await self.worker_for(run)
    if model is not None:
      self.factories[worker] = model
    lw = await self.add(worker)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      await self.tick()
      queue = await self.bus.queued_and_in_flight(run)
      empty = not queue["live"] or (queue["queued"] == 0 and not queue["in_flight"])
      if empty and (lw.task.done() or (w := self.state.workers.get(worker)) is None or w.idle):
        return self.local.get(worker)
      await asyncio.sleep(0.02)
    raise AssertionError(f"{run} never drained: {queue}")

  async def aclose(self):
    for lw in self.local.values():
      lw.cancel(lost=True)
    await asyncio.gather(*(lw.task for lw in self.local.values()), return_exceptions=True)
    self.local.clear()
    for task in (self.uploading, self.sweeping):
      if task:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
