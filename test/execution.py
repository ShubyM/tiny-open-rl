"""Real node control with a synchronous CPU model in place of the local socket process."""

import asyncio
import time

from test.helpers import FakeEngine, tiny
from tinyrl.bus import Ev
from tinyrl.control import scheduler
from tinyrl.control.node_daemon import NodeDaemon
from tinyrl.workers.model import Model
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


class TestNode(NodeDaemon):
  def __init__(self, bus, store=None, gpus=8, node="test-node", model=tiny, engine=FakeEngine):
    super().__init__(bus, gpus=gpus, node=node, gpu_mib=0, store=store)
    self.factories, self.default_model, self.engine_type = {}, model, engine
    self.note(Ev.NODE_UP, node=self.id, gpus=gpus, memory=0)

  def open_trainer(self, worker, env):
    return InlineProcess(Trainer(Model(self.factories.get(worker, self.default_model)), self.store))

  async def sync(self):
    while events := await self.bus.read_events(self.after):
      for event in events:
        self.state = scheduler.fold(self.state, event)
      self.after = events[-1].id
      self.seen += len(events)

  async def add(self, worker, owner="", gpus=None):
    await self.sync()
    if worker in self.running:
      running = self.running[worker]
      if running.model and running.model.proc.poll() is not None:
        self.running.pop(worker)  # the production tick reaps completed worker tasks
      else:
        self.note(Ev.PLACED, worker=worker, node=self.id, gpus=list(running.gpus))
        return running
    self.note(Ev.PLACED, worker=worker, node=self.id, **({"gpus": gpus} if gpus is not None else {}))
    self.schedule()
    running = await self.open_worker(worker, owner)
    if running is None:
      raise AssertionError(f"could not acquire {worker}")
    if running.sampler is not None:
      running.sampler.engine_type = self.engine_type
    return running

  async def worker_for(self, run):
    await self.sync()
    for kind, body in scheduler.decide(self.state, time.time()):
      if kind == Ev.ASSIGNED:
        await self.bus.assign_run_to_worker(body["run"], body["worker"])
    await self.sync()
    return self.state.runs[run].worker

  async def drain(self, run, model=None):
    worker = await self.worker_for(run)
    if model is not None:
      self.factories[worker] = model
    await self.add(worker)
    while await self.step(worker, block_ms=10):
      await self.sync()
      await self.renew()
    return self.running.get(worker)

  async def aclose(self):
    for running in list(self.running.values()):
      running.stopping = running.lost = True
      if running.task and not running.task.done():
        running.task.cancel()
        await asyncio.gather(running.task, return_exceptions=True)
    for worker in list(self.running):
      await self.close_worker(worker)
      self.running.pop(worker, None)
