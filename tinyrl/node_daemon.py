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
TICK = getenv("TICK", 0.25)
SCHEDULER = "scheduler"  # the lease whose holder places pools


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
    self.slots: dict[str, int] = {}
    self.decisions = {}
    self.id = node or f"{socket.gethostname()}-{os.getpid()}"
    self.gpu_mib = gpu_memory_mib() if gpu_mib is None else gpu_mib

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
        self.slots.pop(pool, None)
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
    pools = {pool for pool in await self.bus.pools() if await self.bus.runs(pool)} | set(await self.bus.warm_samplers())
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
    parsed = schedule.Pool.parse(pool)
    leases = ",".join(f"{socket.gethostname()}:{gpu}" for gpu in gpus)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
    if parsed.role == "sampler" and schedule.gpu_zones(self.gpus, width).partitioned:
      env.setdefault("TINYRL_WARM_SAMPLER", "1")
    if self.gpu_mib >= 65536 and parsed.role == "sampler" and width == 1:
      if pool.startswith("sampler:fft:") and pool not in exclusive:
        slot = min(
          (0, 1),
          key=lambda s: sum(1 for p, (g, _) in self.trainers.items() if g == gpus and self.slots.get(p) == s),
        )
        self.slots[pool] = slot
        leases = f"{socket.gethostname()}:{gpus[0]}:{slot}"
        env.setdefault("TINYRL_GPU_MEM", "0.42")
        env.setdefault("TINYRL_KV_BYTES", str(16 * 1024**3))
      else:
        env.setdefault("TINYRL_KV_BYTES", str(24 * 1024**3))
    env["GPU"] = leases
    mod = "tinyrl.sampler" if parsed.role == "sampler" else "tinyrl.trainer"
    if mod == "tinyrl.trainer":
      env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
