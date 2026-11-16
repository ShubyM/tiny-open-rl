"""Real node control with a synchronous CPU model in place of the local socket process."""

import asyncio

from test.helpers import FakeEngine, assign_new_runs, tiny
from tinyrl.control import scheduler
from tinyrl.control.node_daemon import NodeDaemon
from tinyrl.workers.model import Model
from tinyrl.workers.trainer import Trainer


class InlineProcess:
  def __init__(self, trainer: Trainer):
    self.trainer, self.proc, self.closed, self.pid = trainer, self, False, None

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
    self.processes, self.samplers, self.pinned, self.unpinned = {}, {}, {}, set()
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
    state = scheduler.from_state(await self.bus.read_state())
    if not self.announced:
      state.nodes.setdefault(self.id, (self.gpus, self.gpu_mib * 2**20))
    for proc, gpus in list(self.pinned.items()):
      if proc in state.procs and proc not in self.unpinned:
        state.procs[proc].node = self.id
        if gpus is not None:
          state.procs[proc].gpus = gpus
        elif not state.procs[proc].gpus and (fb := state.free(self.id, state.procs[proc].spec.gpus)):
          state.procs[proc].gpus = fb[0]
    return state

  async def sync(self):
    self.state = await self.read_state()

  async def bind(self, proc: str, *runs: str, gpus=None):
    """Create or update `proc` in `tinyrl:plan` hosting `runs` on this node, then start its `Lifetime`."""
    raw = await self.bus.read_state()
    state = scheduler.from_state(raw)
    for r in runs:
      state.procs.setdefault(proc, scheduler.Proc(state.jobs[r].spec))
      state.home[r] = [proc]
    plan = scheduler.to_plan(state)
    await self.bus.write_plan(plan)
    return await self.add(proc, gpus=gpus)

  async def add(self, proc: str, gpus=None):
    self.unpinned.discard(proc)
    self.pinned[proc] = None if gpus is None else tuple(gpus)
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
    self.pinned.pop(proc, None)
    self.unpinned.add(proc)
    if p := self.state.procs.get(proc):
      p.node, p.gpus = None, ()
    self.notify()

  async def worker_for(self, run: str) -> str | None:
    await assign_new_runs(self.bus)
    await self.sync()
    ps = self.state.home.get(run)
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
