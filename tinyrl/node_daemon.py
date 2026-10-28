"""Runs the trainers and samplers placed on this node, on its GPUs.

python -m tinyrl.node_daemon replaces starting `python -m tinyrl.trainer <pool>` by hand. Every node daemon heartbeats
into Redis. The one holding the scheduler lease places each pool that has runs on the live node daemon with the fewest
pools per GPU, and moves a dead node daemon's pools to live ones. Each node daemon runs the pools placed on it, each on
its least used GPU, and stops the ones placed elsewhere. A pool's lease still decides who serves it.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import os
import signal
import socket
import subprocess
import sys
import time

from tinyrl.bus import Bus
from tinyrl.helpers import getenv

GPUS = getenv("GPUS", 1)
DRAIN_S = getenv("DRAIN_S", 20.0)  # how long a stopping node daemon waits for its trainers to save their runs
TICK = 1.0
SCHEDULER = "scheduler"  # the lease whose holder places pools


class NodeDaemon:
  def __init__(self, bus: Bus, gpus: int = GPUS, node: str = ""):
    self.bus = bus
    self.gpus = gpus
    self.trainers: dict[str, tuple[int, subprocess.Popen]] = {}  # pool -> (gpu, process)
    self.id = node or f"{socket.gethostname()}-{os.getpid()}"

  async def tick(self) -> None:
    await self.bus.heartbeat(self.id, self.gpus)
    if await self.bus.renew(SCHEDULER, self.id) or await self.bus.own(SCHEDULER, self.id):
      await self.schedule()
    placed = await self.bus.placements()
    for pool, (_, proc) in list(self.trainers.items()):
      if proc.poll() is not None or placed.get(pool) != self.id:  # it exited, or its pool moved or emptied
        proc.terminate()
        del self.trainers[pool]
    for pool, node in placed.items():
      if node == self.id and pool not in self.trainers:
        self.trainers[pool] = self.start(pool)

  async def schedule(self) -> None:
    nodes, placed = await self.bus.nodes(), await self.bus.placements()
    pools = {pool for pool in await self.bus.pools() if await self.bus.runs(pool)}
    for pool in set(placed) - pools:
      await self.bus.place(pool, None)
      del placed[pool]
    for pool in sorted(pools):
      if placed.get(pool) not in nodes and nodes:
        load = collections.Counter(placed.values())
        placed[pool] = min(nodes, key=lambda node: load[node] / nodes[node])
        await self.bus.place(pool, placed[pool])

  def start(self, pool: str) -> tuple[int, subprocess.Popen]:
    on = [gpu for gpu, _ in self.trainers.values()]
    gpu = min(range(self.gpus), key=on.count)
    # trainers on one GPU take turns on it, named by host so trainers a restarted node daemon left running count too
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "GPU": f"{socket.gethostname()}:{gpu}"}
    mod = "tinyrl.sampler" if pool.startswith("sampler:") else "tinyrl.trainer"
    return gpu, subprocess.Popen([sys.executable, "-m", mod, pool], env=env)

  async def serve(self) -> None:
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
      await self.bus.release(SCHEDULER, self.id)


if __name__ == "__main__":
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # so serve's finally stops the trainers
  asyncio.run(NodeDaemon(Bus()).serve())
