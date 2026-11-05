"""Real node control with a synchronous CPU model in place of the local socket process."""

import asyncio
import collections
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
  def __init__(self, bus, store=None, gpus=8, node="test-node", model=tiny, engine=FakeEngine, gpu_mib=0):
    super().__init__(bus, gpus=gpus, node=node, gpu_mib=gpu_mib, store=store)
    self.factories, self.default_model, self.engine_type = {}, model, engine
    self.finished = collections.defaultdict(collections.deque)
    self.state.nodes[self.id] = (gpus, gpu_mib * 2**20)

  def open_trainer(self, worker, env):
    return InlineProcess(Trainer(Model(self.factories.get(worker, self.default_model)), self.store))

  async def open_worker(self, worker, owner=""):
    running = await super().open_worker(worker, owner)
    if running is not None and running.sampler is not None:
      running.sampler.engine_type = self.engine_type
    return running

  async def sync(self):
    while events := await self.bus.read_events(self.after):
      for event in events:
        self.state = scheduler.fold(self.state, event)
      self.after = events[-1].id

  async def add(self, worker, owner="", gpus=None):
    await self.sync()
    desired = self.state.workers[worker]
    desired.node = self.id
    if gpus is not None:
      desired.gpus = tuple(gpus)
    scheduler.place_gpus(self.state, self.id, self.running)
    if worker in self.running:
      running = self.running[worker]
      if running.stop == "close" and running.task is None:
        self.running.pop(worker)  # an explicitly awaited close has already reaped the child
      else:
        return running
    running = await self.open_worker(worker, owner)
    if running is None:
      raise AssertionError(f"could not acquire {worker}")
    return running

  def dispatch(self):
    for worker, running in self.running.items():
      for kind in ("reader", "task"):
        task = getattr(running, kind)
        if task is not None and task.done():
          self.finished[worker].append((kind, task))
    super().dispatch()

  async def step(self, worker, block_ms=10):
    """Drive real dispatch until ready work drains or yields; block_ms is only the test driver's wait interval."""
    processed = 0
    async with asyncio.timeout(15):
      while True:
        self.dispatch()
        empty = False
        while self.finished[worker]:
          kind, task = self.finished[worker].popleft()
          result = task.result()
          if kind == "task":
            processed += result
          else:
            empty |= not result
        running = self.running.get(worker)
        if running is None or (running.task is None and processed and (not running.queue or running.since is None)):
          return processed
        if empty and running.task is None and not running.queue:
          queues = await asyncio.gather(
            *(self.bus.queued_and_in_flight(run) for run in scheduler.members(self.state, worker))
          )
          if not any(q.get("queued") or q.get("in_flight") for q in queues):
            return processed
        await self.wait(max(0.001, min(block_ms / 1000, 0.01)))

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
    tasks = []
    for running in self.running.values():
      running.stop = "abort"
      for task in (running.reader, running.task):
        if task:
          task.cancel()
          tasks.append(task)
    await asyncio.gather(*tasks, return_exceptions=True)
    for worker in list(self.running):
      await self.close_worker(worker)
      self.running.pop(worker, None)
    for task in (self.uploading, self.sweeping):
      if task:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
