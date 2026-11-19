"""Real node control with a synchronous CPU model in place of the local socket process."""

import asyncio

from test.helpers import FakeEngine, assign_new_runs, tiny
from tinyrl.bus import Role
from tinyrl.control import scheduler
from tinyrl.control.node_daemon import NodeDaemon
from tinyrl.workers.model import Model
from tinyrl.workers.trainer import Trainer


class InlineProcess:
  warm_pending = False

  def __init__(self, trainer: Trainer):
    self.trainer, self.proc, self.closed, self.pid = trainer, self, False, None

  async def call(self, method, *args):
    if self.closed:
      raise EOFError("the model process exited")
    result = getattr(self.trainer, method)(*args)
    await asyncio.sleep(0)  # an actual socket call lets other requests and the node loop run
    return result

  async def resume(self, gpus):
    await self.call("resume", gpus[0] if len(gpus) == 1 else None)

  async def park(self):
    await self.call("park")

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
    self.processes, self.samplers, self.pinned, self.unpinned = {}, {}, set(), set()
    self.announced = False
    self.state.nodes[self.id] = (gpus, gpu_mib * 2**20)

  async def announce_node(self):
    self.announced = True
    await super().announce_node()

  def open_trainer(self, proc, width, env):
    process = InlineProcess(Trainer(Model(self.default_model), self.store))
    self.processes[proc] = process
    return process

  def open_sampler(self, proc, spec, env):
    sampler = super().open_sampler(proc, spec, env)
    sampler.engine_type = self.engine_type
    self.samplers[proc] = sampler
    return sampler

  async def read_state(self):
    if self.announced:
      state = await super().read_state()
    else:
      state = scheduler.from_state(await self.bus.read_state())
      state.nodes.setdefault(self.id, (self.gpus, self.gpu_mib * 2**20))
    for proc in list(self.pinned):
      if proc in state.procs and proc not in self.unpinned:
        state.procs[proc].node = self.id
    return state

  async def sync(self):
    self.state = await self.read_state()

  async def bind(self, proc: str, *runs: str):
    """Create or update `proc` in `tinyrl:plan` hosting `runs` on this node, then start its `Lifetime`."""
    state = scheduler.from_state(await self.bus.read_state())
    for r in runs:
      state.procs.setdefault(proc, scheduler.Proc(state.jobs[r].spec))
      if state.jobs[r].spec.role == Role.TRAINER:
        state.home[r] = proc
    await self.bus.write_plan(scheduler.to_plan(state))
    return await self.add(proc)

  async def add(self, proc: str):
    self.unpinned.discard(proc)
    self.pinned.add(proc)
    await self.sync()
    self.dispatch()
    if proc in self.lives:
      self.notify()
      return self.lives[proc]
    task = self.tasks.get(proc) or self.start(proc)
    async with asyncio.timeout(5):
      while proc not in self.lives:
        if task.done():
          task.result()
          raise AssertionError(f"could not start {proc}")
        await asyncio.sleep(0.001)
    return self.lives[proc]

  def drop_local(self, proc: str):
    """Unplace a locally pinned process so its `Lifetime` detaches its runs and exits."""
    self.pinned.discard(proc)
    self.unpinned.add(proc)
    if p := self.state.procs.get(proc):
      p.node = None
    self.notify()

  async def worker_for(self, run: str) -> str | None:
    await assign_new_runs(self.bus)
    await self.sync()
    ps = self.state.procs_of(run)
    return ps[0] if ps else None

  async def drain(self, run: str):
    """Let real `Lifetime` coroutines execute every queued command for `run`'s process."""
    if (proc := await self.worker_for(run)) is None:
      return None
    await self.add(proc)
    async with asyncio.timeout(15):
      while True:
        await self.tick()
        runs = self.state.runs(proc)
        queues = await asyncio.gather(*(self.bus.queued_and_in_flight(r) for r in runs))
        if not any(q.get("queued") or q.get("in_flight") for q in queues):
          break
        await asyncio.sleep(0.002)
    return self.lives.get(proc)

  async def aclose(self):
    await self.close(graceful=False)
    if task := self.sweeping:
      task.cancel()
      await asyncio.gather(task, return_exceptions=True)
