"""Real node control with a synchronous CPU model in place of the local socket process."""

import asyncio

from test.helpers import FakeEngine, tiny
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
    self.default_model, self.engine_type = model, engine
    self.processes, self.samplers = {}, {}
    self.ticking = asyncio.Lock()
    self.driver = asyncio.create_task(self.drive())

  async def drive(self):
    while True:
      await self.tick()
      await asyncio.sleep(0.001)

  async def tick(self):
    async with self.ticking:
      await super().tick()

  def open_trainer(self, worker, run, env):
    process = InlineProcess(Trainer(Model(self.default_model), self.store))
    self.processes[worker] = process
    return process

  def open_sampler(self, worker, run, env):
    sampler = super().open_sampler(worker, run, env)
    sampler.engine_type = self.engine_type
    self.samplers[worker] = sampler
    return sampler

  async def read_state(self):
    old = await self.bus.read_table()
    await scheduler.commit(self.bus, old, scheduler.decide(old))
    return await self.bus.read_table()

  async def sync(self):
    await self.tick()

  async def add(self, worker, gpus=None):
    async with asyncio.timeout(10):
      while True:
        await self.tick()
        if observation := self.state["observations"].get(worker):
          return observation
        await asyncio.sleep(0.001)

  async def worker_for(self, run):
    await self.tick()
    return self.state["claims"].get(run, {}).get("engine")

  async def drain(self, run):
    async with asyncio.timeout(30):
      while True:
        await self.tick()
        queue = await self.bus.queued_and_in_flight(run)
        saving = any(r.startswith(f"{run}/") for r in await self.bus.pending_saves())
        if not queue.get("queued") and not queue.get("in_flight") and not saving:
          return self.state["observations"].get(await self.worker_for(run))
        await asyncio.sleep(0.001)

  async def close(self, graceful=True):
    self.driver.cancel()
    await asyncio.gather(self.driver, return_exceptions=True)
    await super().close(graceful)

  async def aclose(self):
    await self.close(graceful=False)
