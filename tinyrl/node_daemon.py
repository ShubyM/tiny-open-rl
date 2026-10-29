"""Runs the trainers and samplers placed on this node, on its GPUs.

python -m tinyrl.node_daemon replaces starting `python -m tinyrl.trainer <pool>` by hand. Every node daemon heartbeats
into Redis. The one holding the scheduler lease places each pool that has runs on the live node daemon with the fewest
pools per GPU, and moves a dead node daemon's pools to live ones. Each node daemon runs the pools placed on it, each on
its least used GPU, and stops the ones placed elsewhere. A pool's lease still decides who serves it.
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

from tinyrl import schedule
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
    self.trainers: dict[str, tuple[list[int], subprocess.Popen]] = {}  # pool -> (its GPUs, process)
    self.decisions = {}
    self.id = node or f"{socket.gethostname()}-{os.getpid()}"

  async def tick(self) -> None:
    await self.bus.heartbeat(self.id, self.gpus)
    if await self.bus.renew(SCHEDULER, self.id):
      await self.schedule()
    else:  # a new scheduler waits a tick, so after a Redis restart every node daemon heartbeats before it judges them
      await self.bus.own(SCHEDULER, self.id)
    placed = await self.bus.placements()
    for pool, (_, proc) in list(self.trainers.items()):
      if proc.poll() is not None or placed.get(pool) != self.id:  # it exited, or its pool moved or emptied
        proc.terminate()
        del self.trainers[pool]
        if self.bus:
          await self.bus.forget_decision("gpu", pool)
    excl = await self.bus.exclusive() if self.bus else set()
    to_start = [p for p, n in placed.items() if n == self.id and p not in self.trainers]
    for pool in schedule.start_order(to_start, excl):
      width = await self.bus.width(pool) if self.bus else 1
      self.trainers[pool] = self.start(pool, width, excl)
      if self.bus:
        await self.bus.decide("gpu", pool, self.decisions[pool].as_dict())

  async def schedule(self) -> None:
    nodes, placed = await self.bus.nodes(), await self.bus.placements()
    pools = {pool for pool in await self.bus.pools() if await self.bus.runs(pool)}
    for pool in set(placed) - pools:
      await self.bus.place(pool, None)

    next_placed, decisions = schedule.place_pools(pools, nodes, placed)
    for pool, dec in decisions.items():
      await self.bus.place(pool, next_placed[pool])
      await self.bus.decide("node", pool, dec.as_dict())

  def start(self, pool: str, width: int = 1, exclusive: set[str] = frozenset()) -> tuple[list[int], subprocess.Popen]:
    """Start a pool's process on `width` GPUs: the least used block of them, aligned so blocks never straddle."""
    running = [schedule.RunningPool(p, gpus) for p, (gpus, _) in self.trainers.items()]
    dec = schedule.choose_gpus(pool, width, self.gpus, running, exclusive)
    self.decisions[pool] = dec
    gpus = dec.choice
    leases = ",".join(f"{socket.gethostname()}:{gpu}" for gpu in gpus)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "GPU": leases}
    mod = "tinyrl.sampler" if schedule.Pool.parse(pool).role == "sampler" else "tinyrl.trainer"
    return gpus, subprocess.Popen([sys.executable, "-m", mod, pool], env=env)

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
