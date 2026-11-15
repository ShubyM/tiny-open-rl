"""Real node control and the real scheduler, with a synchronous CPU model in place of the local socket process."""

import asyncio
import dataclasses

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
  """The cluster's only node. Every tick is also one look by the scheduler, so workers are placed and granted GPUs as in a cluster."""

  def __init__(self, bus, store=None, gpus=8, node="test-node", model=tiny, engine=FakeEngine, gpu_mib=0):
    super().__init__(bus, gpus=gpus, node=node, gpu_mib=gpu_mib, store=store)
    self.default_model, self.engine_type = model, engine
    self.processes, self.samplers, self.pumping, self.stopping = {}, {}, None, set()

  def open_trainer(self, worker, env):
    process = InlineProcess(Trainer(Model(self.default_model), self.store))
    self.processes[worker] = process
    return process

  def open_sampler(self, worker, env):
    sampler = super().open_sampler(worker, env)
    sampler.engine_type = self.engine_type
    self.samplers[worker] = sampler
    return sampler

  async def tick(self):
    await super().tick()
    await scheduler.step(self.bus, self.registry)

  async def read_state(self):
    state, ready = await super().read_state()
    for worker in self.stopping & state.workers.keys():
      state.workers[worker].node = None  # as this node sees a worker the cluster moved away
    return state, ready

  async def restart(self, worker):
    """Stop a worker's process the planned way, so it hands its runs off, and let the next tick start another."""
    self.stopping.add(worker)
    await asyncio.wait_for(asyncio.gather(self.tasks[worker], return_exceptions=True), 10)
    self.stopping.discard(worker)

  async def kill(self, worker):
    """Lose a worker's process with no notice. The next tick starts another."""
    self.tasks[worker].cancel()
    await asyncio.wait_for(asyncio.gather(self.tasks[worker], return_exceptions=True), 10)

  def pump(self):
    async def run():
      while True:
        await self.tick()
        await asyncio.sleep(0.005)

    self.pumping = self.pumping or asyncio.create_task(run())

  async def add(self, worker, gpus=None):
    """Run a worker here, on the GPUs the scheduler chooses or the test names, and keep the cluster ticking."""
    if gpus is not None:
      s = scheduler.from_table(await self.bus.read_table())
      placed = {"spec": dataclasses.asdict(s.workers[worker].spec), "node": self.id, "gpus": tuple(gpus)}
      await self.bus.write_table({}, {worker: placed}, [], s.workers_made, [])
    self.pump()
    async with asyncio.timeout(5):
      while worker not in self.running:
        if (task := self.tasks.get(worker)) and task.done():
          task.result()
          raise AssertionError(f"could not start {worker}")
        await asyncio.sleep(0.001)
    return self.running[worker]

  async def worker_for(self, run):
    await scheduler.step(self.bus, self.registry)
    jobs = scheduler.from_table(await self.bus.read_table()).jobs
    return jobs[run].worker if run in jobs else None

  async def drain(self, run):
    """Let the real worker coroutines finish the commands visible in their Redis streams."""
    if (worker := await self.worker_for(run)) is None:
      return None  # a live worker may already have completed the run's DELETE_MODEL
    await self.add(worker)
    async with asyncio.timeout(15):
      while True:
        await self.tick()
        queues = await asyncio.gather(
          *(self.bus.queued_and_in_flight(r) for r in scheduler.jobs_of(self.state, worker))
        )
        if not any(q.get("queued") or q.get("in_flight") for q in queues):
          break
        await asyncio.sleep(0.001)
    return self.running.get(worker)

  async def aclose(self, graceful=False):
    if self.pumping:
      self.pumping.cancel()
      await asyncio.gather(self.pumping, return_exceptions=True)
    await self.close(graceful)
    if task := self.sweeping:
      task.cancel()
      await asyncio.gather(task, return_exceptions=True)
