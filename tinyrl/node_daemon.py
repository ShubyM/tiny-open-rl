"""Runs the trainers and samplers the scheduler placed on this node, on the GPUs it chose.

python -m tinyrl.node_daemon replaces starting `python -m tinyrl.trainer <worker>` by hand. Every node daemon heartbeats
into Redis and appends node_up when it starts. It folds tinyrl:events like the scheduler does, starts a process for each
worker placed on it, starts it again if it dies, and stops the ones placed elsewhere or emptied. It decides nothing. A
worker's lease still decides who serves it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import subprocess
import sys
import time

from tinyrl import scheduler
from tinyrl.bus import Bus
from tinyrl.helpers import getenv

GPUS = getenv("GPUS", 1)
DRAIN_S = getenv(
  "DRAIN_S", 20.0
)  # how long a stopping node daemon waits for its processes, so trainers save their runs
TICK = getenv("TICK", 0.25)


def gpu_memory_mib() -> int:
  if override := getenv("GPU_MEMORY_MIB", 0):
    return int(override)
  try:
    res = subprocess.run(
      ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"],
      capture_output=True,
      text=True,
      timeout=2,
      check=False,
    )
    if res.returncode == 0:
      return int(res.stdout.strip())
  except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
    pass
  return 0


def reap(proc: subprocess.Popen) -> None:
  """Kill what is left of a process's group, like the engine a sampler started, which outlives its parent."""
  if isinstance(proc.pid, int) and proc.pid > 1:
    with contextlib.suppress(ProcessLookupError):
      os.killpg(proc.pid, signal.SIGKILL)


class NodeDaemon:
  def __init__(self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None):
    self.bus = bus
    self.gpus = gpus
    self.procs: dict[str, tuple[list[int], subprocess.Popen]] = {}  # worker -> (its GPUs, its process)
    self.stopping: dict[str, subprocess.Popen] = {}  # told to stop, and maybe still on its GPUs
    self.id = node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib
    self.state, self.after = scheduler.State(), "0-0"

  async def up(self) -> None:
    await self.bus.heartbeat(self.id, self.gpus)  # first, so the scheduler never finds it up without a heartbeat
    await self.bus.emit("node_up", node=self.id, gpus=self.gpus, memory=self.gpu_mib * 2**20)

  async def tick(self) -> None:
    await self.bus.heartbeat(self.id, self.gpus)
    if await self.bus.log_size() < self.state.n:  # the log shrank, so Redis was flushed
      self.state, self.after = scheduler.State(), "0-0"
    while events := await self.bus.events(self.after):
      for e in events:
        self.state = scheduler.fold(self.state, e)
      self.after = events[-1].id
    mine = {n: p for n, p in self.state.workers.items() if p.node == self.id and scheduler.members(self.state, n)}
    for worker, (gpus, proc) in list(self.procs.items()):
      if proc.poll() is not None or worker not in mine or list(mine[worker].gpus) != gpus:
        proc.terminate()
        self.stopping[worker] = self.procs.pop(worker)[1]
    for worker, proc in list(self.stopping.items()):
      if proc.poll() is not None:  # only now is the GPU free, so only now does its turn end
        reap(proc)
        del self.stopping[worker]
        await self.bus.emit("exited", worker=worker)
    for worker in sorted(set(mine) - set(self.procs) - set(self.stopping)):
      self.procs[worker] = self.start(worker, mine[worker])
      await self.bus.emit("started", worker=worker, node=self.id)

  def start(self, worker: str, p: scheduler.Worker) -> tuple[list[int], subprocess.Popen]:
    """Start a worker's process on its GPUs, with the settings for its kind. This node's environment wins."""
    gpus, k = list(p.gpus), p.kind
    env = {**scheduler.settings(k, self.gpus, self.gpu_mib * 2**20), **os.environ}
    env["CUDA_VISIBLE_DEVICES"], env["GPU"] = (
      ",".join(map(str, gpus)),
      ",".join(f"{socket.gethostname()}:{g}" for g in gpus),
    )
    env["TINYRL_LORA"], env["TINYRL_BASE"] = str(int(k.lora)), k.base
    mod = f"tinyrl.{k.role}"
    if mod == "tinyrl.trainer":
      env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return gpus, subprocess.Popen([sys.executable, "-m", mod, worker], env=env, start_new_session=True)

  async def serve(self) -> None:
    await self.up()
    try:
      while True:
        await self.tick()
        await asyncio.sleep(TICK)
    finally:
      for _, proc in self.procs.values():
        proc.terminate()  # a trainer saves its runs, then exits
      deadline = time.monotonic() + DRAIN_S
      for proc in [*(p for _, p in self.procs.values()), *self.stopping.values()]:
        with contextlib.suppress(subprocess.TimeoutExpired):
          proc.wait(max(0.0, deadline - time.monotonic()))
        reap(proc)


if __name__ == "__main__":
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # so serve's finally stops its processes
  asyncio.run(NodeDaemon(Bus()).serve())
