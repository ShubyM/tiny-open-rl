"""Starts a trainer for every pool that has runs and no trainer, on this node's GPUs.

python -m tinyrl.node_daemon replaces starting `python -m tinyrl.trainer <pool>` by hand. Each trainer gets the GPU with
the fewest trainers on it. A pool's lease still decides who serves it, so a second node daemon only adds standbys.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys

from tinyrl.bus import Bus
from tinyrl.helpers import getenv

GPUS = getenv("GPUS", 1)
TICK = 1.0


class NodeDaemon:
  def __init__(self, bus: Bus, gpus: int = GPUS):
    self.bus = bus
    self.gpus = gpus
    self.trainers: dict[str, tuple[int, subprocess.Popen]] = {}  # pool -> (gpu, process)
    self.id = f"{socket.gethostname()}-{os.getpid()}"

  async def tick(self) -> None:
    await self.bus.heartbeat(self.id, self.gpus)  # how many shards a LoRA base can spread over
    for pool, (_, proc) in list(self.trainers.items()):
      if proc.poll() is not None:  # it exited: its pool emptied, it lost the lease, or it crashed
        del self.trainers[pool]
    for pool in await self.bus.pools():
      if pool not in self.trainers and await self.bus.runs(pool) and not await self.bus.owner(pool):
        self.trainers[pool] = self.start(pool)

  def start(self, pool: str) -> tuple[int, subprocess.Popen]:
    on = [gpu for gpu, _ in self.trainers.values()]
    gpu = min(range(self.gpus), key=on.count)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    return gpu, subprocess.Popen([sys.executable, "-m", "tinyrl.trainer", pool], env=env)

  async def serve(self) -> None:
    try:
      while True:
        await self.tick()
        await asyncio.sleep(TICK)
    finally:
      for _, proc in self.trainers.values():
        proc.terminate()


if __name__ == "__main__":
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # so serve's finally stops the trainers
  asyncio.run(NodeDaemon(Bus()).serve())
