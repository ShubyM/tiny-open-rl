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
    self.processes, self.samplers, self.pinned, self.pinned_gpus = {}, {}, set(), {}
    self.state.nodes[self.id] = scheduler.NodeInfo(gpus=gpus, memory=gpu_mib * 2**20)

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
    state.nodes.setdefault(self.id, scheduler.NodeInfo(gpus=self.gpus, memory=self.gpu_mib * 2**20))
    pinned = sorted(self.pinned & state.workers.keys())
    ready_gpus: set[int] = set()
    for worker in pinned:
      w = state.workers[worker]
      w.node = self.id
      w.gpus = self.pinned_gpus.get(worker) or w.gpus or tuple(range(min(self.gpus, w.spec.gpus)))
      w.generations = {j: state.jobs[j].gen for j in scheduler.jobs_of(state, worker)}
      if any(state.jobs[j].ready for j in scheduler.jobs_of(state, worker)):
        ready_gpus.update(w.gpus)
    for worker in pinned:
      w = state.workers[worker]
      is_ready = any(state.jobs[j].ready for j in scheduler.jobs_of(state, worker))
      w.intent = "park" if (not is_ready and bool(ready_gpus & set(w.gpus))) else "run"
    return state

  async def sync(self):
    self.state = await self.read_state()

  async def add(self, worker, gpus=None):
    self.pinned.add(worker)
    await self.sync()
    if gpus is not None:
      self.pinned_gpus[worker] = tuple(gpus)
    elif worker not in self.pinned_gpus:
      w = self.state.workers.get(worker)
      width = w.spec.gpus if w else 1
      used = {g for r in self.running.values() for g in r.gpus_held}
      free = [g for g in range(self.gpus) if g not in used]
      self.pinned_gpus[worker] = tuple(free[:width]) if len(free) >= width else tuple(range(min(self.gpus, width)))
    await self.sync()
    self.dispatch()
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
    return r.worker if (r := self.state.jobs.get(run)) else None

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

  async def aclose(self):
    await self.close(graceful=False)
    if task := self.sweeping:
      task.cancel()
      await asyncio.gather(task, return_exceptions=True)
