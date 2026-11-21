"""Real `NodeDaemon` with a synchronous CPU model in place of the local socket process."""

import asyncio

from test.helpers import FakeEngine, tiny
from tinyrl.control import scheduler
from tinyrl.control.node_daemon import NodeDaemon
from tinyrl.types import Role
from tinyrl.workers.model import Model
from tinyrl.workers.trainer import Trainer


class InlineProcess:
  def __init__(self, trainer: Trainer):
    self.trainer, self.proc, self.closed, self.pid = trainer, self, False, None

  async def call(self, method, *args):
    if self.closed:
      raise EOFError("the model process exited")
    result = getattr(self.trainer, method)(*args)
    await asyncio.sleep(0)  # let other coroutines run between model calls
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
  def __init__(self, control, store=None, gpus=8, node="test-node", model=tiny, engine=FakeEngine, gpu_mib=0):
    super().__init__(control, gpus=gpus, node=node, gpu_mib=gpu_mib, store=store)
    self.default_model, self.engine_type = model, engine
    self.processes, self.samplers = {}, {}

  def open_trainer(self, worker_id, width, env):
    process = InlineProcess(Trainer(Model(self.default_model), self.store))
    self.processes[worker_id] = process
    return process

  def open_sampler(self, worker_id, spec, env):
    sampler = super().open_sampler(worker_id, spec, env)
    sampler.engine_type = self.engine_type
    self.samplers[worker_id] = sampler
    return sampler

  async def tick(self):
    return await self.step_once(wait_ms=0)

  async def bind(self, proc: str, *runs: str):
    """Pin `proc` hosting `runs` onto this node in the controller's plan and announce this node."""
    state = self.control.state()
    for r in runs:
      spec = state.jobs[r].spec
      if spec.role == Role.SAMPLER:
        for old in [n for n, p in list(state.procs.items()) if n != proc and p.spec == spec]:
          state.procs.pop(old, None)
      state.procs[proc] = scheduler.Proc(spec, self.id)
      if spec.role == Role.TRAINER:
        state.home[r] = proc
    self.control.plan_state, self.control.plan_due = state, 0.0
    await self.tick()
    return self.workers.get(proc)

  def drop_local(self, proc: str):
    """Remove `proc` from this node's plan so the next exchange closes/detaches it."""
    if p := self.control.plan_state.procs.get(proc):
      p.node, self.control.plan_due = None, 0.0
    self.notify()

  async def worker_for(self, run: str) -> str | None:
    await self.tick()
    state = self.control.state()
    ps = state.procs_of(run)
    return ps[0] if ps else None

  async def drain(self, run: str):
    """Run real `exchange` ticks until all queued/in-flight commands for `run` finish."""
    async with asyncio.timeout(15):
      while True:
        await self.tick()
        q = await self.control.queued_and_in_flight(run)
        if not q.get("queued") and not q.get("in_flight") and not self.active and not self.staging:
          break
        await asyncio.sleep(0.002)
    state = self.control.state()
    ps = state.procs_of(run)
    return self.workers.get(ps[0]) if ps else None

  async def aclose(self):
    await self.close()
