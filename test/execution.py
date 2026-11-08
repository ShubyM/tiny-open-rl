"""Real node control with a synchronous CPU model in place of the local socket process."""

import asyncio

from test.helpers import FakeEngine, assign_new_runs, tiny
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
    self.processes, self.samplers, self.pinned = {}, {}, set()
    self.state.nodes[self.id] = (gpus, gpu_mib * 2**20)

  def open_trainer(self, worker, env):
    process = InlineProcess(Trainer(Model(self.default_model), self.store))
    self.processes[worker] = process
    return process

  def open_sampler(self, worker, env):
    sampler = super().open_sampler(worker, env)
    sampler.engine_type = self.engine_type
    self.samplers[worker] = sampler
    return sampler

  async def read_state(self):
    """The table, with the workers a test placed here itself, since no scheduler places them."""
    state = scheduler.from_table(await self.bus.read_table())
    state.nodes.setdefault(self.id, (self.gpus, self.gpu_mib * 2**20))  # live before its first heartbeat
    for worker in self.pinned & state.workers.keys():
      state.workers[worker].node = self.id
    return scheduler.keep_chosen_gpus(state, self.state)

  async def sync(self):
    self.state = await self.read_state()

  async def add(self, worker, gpus=None):
    self.pinned.add(worker)
    await self.sync()
    self.dispatch()
    desired = self.state.workers[worker]
    if gpus is not None:
      desired.gpus = tuple(gpus)
    scheduler.place_gpus(self.state, self.id, self.running)
    if worker in self.running:
      return self.running[worker]
    task = self.tasks.get(worker) or self.start(worker)
    async with asyncio.timeout(5):
      while worker not in self.running:
        if task.done():
          task.result()
          raise AssertionError(f"could not acquire {worker}")
        await asyncio.sleep(0.001)
    return self.running[worker]

  async def worker_for(self, run):
    await assign_new_runs(self.bus)
    await self.sync()
    return r.worker if (r := self.state.runs.get(run)) else None

  async def drain(self, run):
    """Let the real worker coroutines finish the commands visible in their Redis streams."""
    if (worker := await self.worker_for(run)) is None:
      return None  # a live worker may already have completed the run's DELETE_MODEL
    await self.add(worker)
    async with asyncio.timeout(15):
      while True:
        await self.tick()
        queues = await asyncio.gather(
          *(self.bus.queued_and_in_flight(r) for r in scheduler.runs_of(self.state, worker))
        )
        if not any(q.get("queued") or q.get("in_flight") for q in queues):
          break
        await asyncio.sleep(0.001)
    return self.running.get(worker)

  async def aclose(self):
    await self.close(graceful=False)
    for task in (self.uploading, self.sweeping):
      if task:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
