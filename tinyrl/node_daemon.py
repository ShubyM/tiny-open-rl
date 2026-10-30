"""Runs the trainers and samplers the scheduler placed on this node, on the GPUs it chose.

python -m tinyrl.node_daemon replaces starting `python -m tinyrl.trainer <pool>` by hand. Every node daemon heartbeats
into Redis and appends node_up when it starts. It folds tinyrl:events like the scheduler does, starts a process for each
pool placed on it, starts it again if it dies, and stops the ones placed elsewhere or emptied. It decides nothing. A
pool's lease still decides who serves it.
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
DRAIN_S = getenv("DRAIN_S", 20.0)  # how long a stopping node daemon waits for its trainers to save their runs
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


class NodeDaemon:
  def __init__(self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None):
    self.bus = bus
    self.gpus = gpus
    self.trainers: dict[str, tuple[list[int], subprocess.Popen]] = {}  # pool -> (its GPUs, process)
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
    mine = {name: p for name, p in self.state.pools.items() if p.node == self.id and p.runs}
    for pool, (gpus, proc) in list(self.trainers.items()):
      if proc.poll() is not None or pool not in mine or list(mine[pool].gpus) != gpus:
        proc.terminate()
        self.stopping[pool] = self.trainers.pop(pool)[1]
    for pool, proc in list(self.stopping.items()):
      if proc.poll() is not None:  # only now is the GPU free, so only now does its turn end
        del self.stopping[pool]
        await self.bus.emit("exited", pool=pool)
    for pool in sorted(set(mine) - set(self.trainers) - set(self.stopping)):
      self.trainers[pool] = self.start(pool, mine[pool])
      await self.bus.emit("started", pool=pool, node=self.id)

  def start(self, pool: str, p: scheduler.Pool) -> tuple[list[int], subprocess.Popen]:
    """Start a pool's process on the GPUs and with the settings it was placed with. This node's environment wins."""
    gpus = list(p.gpus)
    env = {**p.env, **os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
    env["GPU"] = ",".join(f"{socket.gethostname()}:{gpu}" for gpu in gpus)
    env["TINYRL_LORA"], env["TINYRL_BASE"] = str(int(p.lora)), p.base
    mod = f"tinyrl.{p.role}"
    if mod == "tinyrl.trainer":
      env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return gpus, subprocess.Popen([sys.executable, "-m", mod, pool], env=env)

  async def serve(self) -> None:
    await self.up()
    try:
      while True:
        await self.tick()
        await asyncio.sleep(TICK)
    finally:
      for _, proc in self.trainers.values():
        proc.terminate()  # a trainer saves its runs, then exits
      deadline = time.monotonic() + DRAIN_S
      for _, proc in self.trainers.values():
        with contextlib.suppress(subprocess.TimeoutExpired):
          proc.wait(max(0.0, deadline - time.monotonic()))


if __name__ == "__main__":
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # so serve's finally stops the trainers
  asyncio.run(NodeDaemon(Bus()).serve())
