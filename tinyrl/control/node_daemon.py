"""Node daemon: heartbeats, enforces exclusive local GPU table `on`, and supervises worker runners."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import subprocess
import sys
import time

from tinyrl.bus import Bus
from tinyrl.control import scheduler
from tinyrl.control.fairness import plan_gpus
from tinyrl.control.process import Process
from tinyrl.control.workers import Runner, SamplerRunner, TrainerRunner
from tinyrl.helpers import getenv
from tinyrl.storage.store import Store
from tinyrl.types import Ev, Role
from tinyrl.workers.sampler import Sampler

GPUS, DRAIN_S, TICK = getenv("GPUS", 1), getenv("DRAIN_S", 20.0), getenv("TICK", 0.25)
SWEEP_S, SWEEP_GRACE_S = getenv("SWEEP_S", 60.0), 600


def gpu_memory_mib() -> int:
  if override := getenv("GPU_MEMORY_MIB", 0):
    return int(override)
  try:
    cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"]
    return int(subprocess.check_output(cmd, timeout=2, stderr=subprocess.DEVNULL))
  except (OSError, subprocess.SubprocessError, ValueError):
    return 0


class NodeDaemon:
  def __init__(self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None, store=None):
    self.bus = bus
    self.store = store if store is not None else Store()
    self.store.start()
    self.gpus, self.id = gpus, node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.on: dict[str, tuple[int, ...]] = {}  # life -> occupied GPUs (single source of local GPU truth)
    self.lives: dict[str, Runner] = {}
    self.tasks: dict[str, asyncio.Task] = {}
    self.timer: asyncio.TimerHandle | None = None
    self.state, self.changed, self.draining = scheduler.State(), asyncio.Event(), False
    self.heartbeat, self.lag, self.swept, self.sweeping = time.monotonic(), 0.0, 0.0, None

  async def serve(self) -> None:
    await self.announce_node()
    wake_id = "$"
    try:
      while True:
        await self.tick()
        before = time.monotonic()
        wake_id = await self.bus.wait_wake(wake_id, int(TICK * 1000), stream="plan_wake")
        self.lag = max(0.0, time.monotonic() - before - TICK)
    finally:
      await self.close()

  async def push_heartbeat(self) -> None:
    on = {life.split("@", 1)[0]: list(gs) for life, gs in self.on.items()}
    pids = {p: m.proc.pid for p, lf in self.lives.items() if (m := lf.model) and m.proc and m.proc.pid is not None}
    self.heartbeat = time.monotonic()
    await self.bus.node_heartbeat(self.id, self.gpus, self.gpu_mib * 2**20, on=on, pids=pids, lag=self.lag)

  async def tick(self) -> None:
    await self.push_heartbeat()
    self.state = await self.read_state()
    self.dispatch()
    if not self.draining:
      for proc, p in sorted(self.state.procs.items()):
        if p.node == self.id and proc not in self.tasks and self.state.runs(proc):
          self.start(proc)
    if time.time() - self.swept > SWEEP_S and (self.sweeping is None or self.sweeping.done()):
      self.swept, self.sweeping = time.time(), asyncio.create_task(self.bus.sweep(self.store, SWEEP_GRACE_S))
    self.notify()

  def dispatch(self) -> None:
    for proc, task in list(self.tasks.items()):
      if task.done():
        if not task.cancelled() and (err := task.exception()):
          print(f"{proc} stopped: {err!r}", file=sys.stderr, flush=True)
        del self.tasks[proc]
        self.lives.pop(proc, None)
      elif self.id not in self.state.nodes and not task.cancelling():
        task.cancel()

  def select_gpus(self, proc: str) -> tuple[int, ...]:
    occupied = set().union(*self.on.values())
    workers = {p: lf.worker() for p, lf in self.lives.items() if lf.wanted()}
    keep, wake_at = plan_gpus(workers, self.gpus, now := time.monotonic(), occupied)
    if self.timer:
      self.timer.cancel()
    self.timer = asyncio.get_running_loop().call_later(max(0.0, wake_at - now), self.notify) if wake_at else None
    return keep.get(proc, ())

  def claim_gpus(self, life: str, gpus: tuple[int, ...], width: int) -> bool:
    if len(gpus) != width or len(set(gpus)) != width or any(g < 0 or g >= self.gpus for g in gpus):
      return False
    if {g for k, gs in self.on.items() if k != life for g in gs} & set(gpus):
      return False
    self.on[life] = gpus
    self.notify()
    return True

  def release_gpus(self, life: str) -> None:
    if self.on.pop(life, None) is not None:
      self.notify()

  def start(self, proc: str) -> asyncio.Task:
    cls = SamplerRunner if self.state.procs[proc].spec.role == Role.SAMPLER else TrainerRunner
    self.tasks[proc] = task = asyncio.create_task(cls(self, proc).run())
    return task

  async def close(self, graceful: bool = True) -> None:
    self.draining = True
    if self.timer:
      self.timer.cancel()
    self.notify()
    if self.tasks:
      _, pending = await asyncio.wait(self.tasks.values(), timeout=DRAIN_S if graceful else 0)
      for task in pending:
        task.cancel()
      await asyncio.gather(*self.tasks.values(), return_exceptions=True)
      self.dispatch()
    await asyncio.to_thread(self.store.close)

  async def read_state(self) -> scheduler.State:
    return scheduler.from_state(await self.bus.read_state(backlogs=False))

  def notify(self) -> None:
    changed, self.changed = self.changed, asyncio.Event()
    changed.set()

  async def announce_node(self) -> None:
    await self.push_heartbeat()
    await self.bus.append_event(Ev.NODE_UP, node=self.id, gpus=self.gpus, memory=self.gpu_mib * 2**20)

  def open_trainer(self, proc: str, width: int, env: dict[str, str]) -> Process:
    dist = ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={width}"] if width > 1 else []
    return Process([sys.executable, *dist, "-m", "tinyrl.workers.trainer"], env)

  def open_sampler(self, proc: str, spec: scheduler.Spec, env: dict[str, str]) -> Sampler:
    return Sampler(spec.base, spec.lora, env)


if __name__ == "__main__":

  async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await NodeDaemon(Bus()).serve()

  asyncio.run(main())
