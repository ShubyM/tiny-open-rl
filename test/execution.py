"""Real `NodeDaemon` with a synchronous CPU model in place of the local socket process."""

import asyncio

from test.helpers import FakeEngine, tiny
from tinyrl.control.controller import LiveWorker
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
    result = getattr(self.trainer, method)(*((None,) if method == "resume" else args))
    await asyncio.sleep(0)
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
    if self.store.server:  # Test nodes share a host and use separate caches.
      self.store.peer = f"127.0.0.1:{self.store.server.server_port}"
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
    return await self.step_once()

  def tasks(self) -> list[asyncio.Task]:
    return [
      *(tk for _, tk in self.active.values()),
      *self.staging.values(),
      *((self.sweep_task,) if self.sweep_task else ()),
    ]

  async def bind(self, proc: str, *runs: str):
    """Ensure this node is registered and optionally pin `proc` hosting `runs` onto it."""
    if self.id not in self.control.nodes:
      await self.tick()
    ns = self.control.nodes[self.id]
    for r in runs:
      rs = self.control.runs[r]
      spec = rs.run.spec
      if spec.role == Role.SAMPLER:
        for wid in [k for k, w in list(ns.workers.items()) if k != proc and w.spec == spec]:
          ns.workers.pop(wid, None)
      ns.workers.setdefault(proc, LiveWorker(proc, spec))
      if spec.role == Role.TRAINER and rs.owner is None:
        rs.owner = (proc, self.id, self.boot_id)
    self.control.schedule()
    await self.tick()
    return self.workers.get(proc)

  def drop_local(self, proc: str):
    if (ns := self.control.nodes.get(self.id)) and (w := ns.workers.get(proc)):
      w.stopping = True
    self.control.schedule()
    self.notify()

  async def worker_for(self, run: str) -> str | None:
    await self.tick()
    ws = self.control.workers_of(run)
    return ws[0] if ws else None

  async def drain(self, run: str):
    async with asyncio.timeout(15):
      while True:
        await self.tick()
        q = self.control.queued_and_in_flight(run)
        if not q.get("queued") and not q.get("in_flight") and not self.busy():
          break
        await asyncio.sleep(0.002)
    ws = self.control.workers_of(run)
    return self.workers.get(ws[0]) if ws else None

  async def aclose(self):
    await self.close()
