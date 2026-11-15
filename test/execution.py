"""Real node control with a synchronous CPU model in place of the local socket process, driven by the real fleet scheduler."""

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
    await asyncio.sleep(0)
    return result

  def poll(self):
    return 0 if self.closed else None

  def kill(self):
    self.closed = True

  async def close(self, graceful: bool = False):
    self.closed = True
    return True


class TestNode(NodeDaemon):
  def __init__(
    self,
    bus,
    store=None,
    gpus=8,
    node="test-node",
    model=tiny,
    engine=FakeEngine,
    gpu_mib=0,
    *,
    auto_schedule=True,
    **kwargs,
  ):
    super().__init__(bus, gpus=gpus, node=node, gpu_mib=gpu_mib, store=store, **kwargs)
    self.default_model, self.engine_type = model, engine
    self.processes, self.samplers = {}, {}
    self.auto_schedule = auto_schedule
    self._announced = False

  def open_trainer(self, worker, gpus, env):
    process = InlineProcess(Trainer(Model(self.default_model), self.store))
    self.processes[worker] = process
    return process

  def open_sampler(self, worker, spec, env):
    sampler = super().open_sampler(worker, spec, env)
    sampler.engine_type = self.engine_type
    self.samplers[worker] = sampler
    return sampler

  async def _ensure_announced(self):
    if not self._announced:
      self._announced = True
      await self.announce_node()

  async def sync(self):
    await self._ensure_announced()
    if self.auto_schedule:
      await scheduler.step(self.bus)
    self.state = await self.read_state()
    self.notify()

  async def add(self, worker, gpus=None):
    await self._ensure_announced()
    if gpus is not None:
      s = scheduler.from_table(await self.bus.read_table())
      if w := s.workers.get(worker):
        old = scheduler.from_table(await self.bus.read_table())
        w.node = self.id
        w.gpus = tuple(gpus)
        scheduler.assign_budgets(s, worker, self.id, asyncio.get_running_loop().time())
        w.revision += 1
        w.intent, w.drain_mode = "run", ""
        await scheduler.commit(self.bus, old, s, [], {})
    elif self.auto_schedule:
      s = scheduler.from_table(await self.bus.read_table())
      if (w := s.workers.get(worker)) and w.node is None:
        w.node = self.id
        await self.bus.write_table({}, {worker: scheduler.worker_to_dict(w)}, [], s.workers_made, [])
      await scheduler.step(self.bus)
    self.state = await self.read_state()
    self.notify()
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
    await self.sync()
    return r.worker if (r := self.state.jobs.get(run)) else None

  async def tick(self) -> None:
    self._announced = True
    if self.auto_schedule:
      local_w = {
        n: {"gpus": list(v["gpus"]), "pid": v.get("pid"), "gpu_state": v.get("gpu_state"), "phase": v.get("phase")}
        for n, v in self.running.items()
      }
      await self.bus.node_heartbeat(
        self.id, self.gpus, self.gpu_mib * 2**20, local_w, self.lag, **self.heartbeat_extra()
      )
      self.heartbeat = asyncio.get_running_loop().time()
      await scheduler.step(self.bus)
    await super().tick()

  async def drain(self, run):
    """Drive real scheduler + node ticks until `run`'s worker finishes all queued and in-flight work."""
    if (worker := await self.worker_for(run)) is None:
      return None
    await self.add(worker)
    async with asyncio.timeout(15):
      while True:
        await self.tick()
        worker = r.worker if (r := self.state.jobs.get(run)) else worker
        runs = (
          scheduler.jobs_of(self.state, worker)
          if worker in self.state.workers
          else ([run] if run in self.state.jobs else [])
        )
        queues = await asyncio.gather(*(self.bus.queued_and_in_flight(r) for r in runs))
        if not any(q.get("queued") or q.get("in_flight") for q in queues):
          break
        await asyncio.sleep(0.002)
    return self.running.get(worker)

  async def aclose(self):
    await self.close(graceful=False)
