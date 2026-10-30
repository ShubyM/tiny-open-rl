"""Run the worker processes placed on this node, once per process ID.

The scheduler owns placement and turns. A stopped child retains its reservation until it exits; a replacement gets a
fresh ID. Losing contact with Redis stops local children, with a bounded grace period for saving training state.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
import uuid

from tinyrl import scheduler
from tinyrl.bus import LEASE_MS, Bus
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


def group_alive(group: int) -> bool:
  """Ignore orphan zombies, which have released their GPU resources but may await init's reaper."""
  if os.path.isdir("/proc"):
    try:
      for path in pathlib.Path("/proc").glob("[0-9]*/stat"):
        try:
          fields = path.read_text().rsplit(") ", 1)[1].split()
        except FileNotFoundError:
          continue
        if int(fields[2]) == group and fields[0] not in ("Z", "X"):
          return True
      return False
    except OSError:
      pass  # without readable /proc, require the entire group to disappear
  try:
    os.killpg(group, 0)
    return True
  except ProcessLookupError:
    return False


class NodeDaemon:
  def __init__(self, bus: Bus, gpus: int = GPUS, node: str = "", gpu_mib: int | None = None):
    self.bus = bus
    self.gpus = gpus
    self.trainers: dict[str, tuple[list[int], subprocess.Popen]] = {}  # worker -> (its GPUs, process)
    self.stopping: dict[str, tuple[subprocess.Popen, float]] = {}
    self.finished: set[str] = set()  # a process ID is never started twice
    self.id = node or f"{socket.gethostname()}-{uuid.uuid4().hex}"
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
    mine = {name: w for name, w in self.state.workers.items() if w.node == self.id and not w.stopping and not w.exited}
    for worker, (_, proc) in list(self.trainers.items()):
      if proc.poll() is not None or worker not in mine:
        if proc.poll() is None:
          proc.terminate()
        self.stopping[worker] = proc, time.monotonic() + DRAIN_S
        del self.trainers[worker]
    for worker, (proc, deadline) in list(self.stopping.items()):
      if proc.poll() is None and time.monotonic() >= deadline:
        self.kill(proc)
      if proc.poll() is not None:
        self.kill(proc)  # a killed sampler can leave its engine alive after the direct parent exits
      if proc.poll() is not None and not group_alive(proc.pid):
        await self.bus.emit("exited", worker=worker, node=self.id)
        del self.stopping[worker]
        self.finished.add(worker)
    for worker in sorted(set(mine) - set(self.trainers) - set(self.stopping) - self.finished):
      self.trainers[worker] = self.start(worker, mine[worker])
      await self.bus.emit("started", worker=worker, node=self.id)

  def start(self, worker: str, w: scheduler.Worker) -> tuple[list[int], subprocess.Popen]:
    """A fixed placement and one process incarnation; the scheduler's resource settings win."""
    gpus = list(w.gpus)
    env = {
      **os.environ,
      **w.env,
      "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
      "TINYRL_WORKER_ID": worker,
      "TINYRL_ROLE": w.role,
      "TINYRL_LORA": str(int(w.lora)),
    }
    env["GPU"] = ",".join(f"{socket.gethostname()}:{gpu}" for gpu in gpus)
    if w.role == "trainer":
      env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return gpus, subprocess.Popen([sys.executable, "-m", f"tinyrl.{w.role}", worker], env=env, start_new_session=True)

  @staticmethod
  def kill(proc) -> None:
    with contextlib.suppress(ProcessLookupError):
      os.killpg(proc.pid, signal.SIGKILL)  # include the sampler's engine subprocesses

  async def serve(self) -> None:
    await self.up()
    try:
      while True:
        # A disconnected agent cannot keep authorizing local GPU processes forever.
        await asyncio.wait_for(self.tick(), LEASE_MS / 1000)
        if self.id not in self.state.nodes:
          return  # lost registration: drain, then let the supervisor restart with a fresh identity
        await asyncio.sleep(TICK)
    finally:
      processes = {w: p for w, (_, p) in self.trainers.items()}
      processes.update({w: p for w, (p, _) in self.stopping.items()})
      for proc in processes.values():
        if proc.poll() is None:
          proc.terminate()
      deadline = time.monotonic() + DRAIN_S
      for worker, proc in processes.items():
        try:
          await asyncio.to_thread(proc.wait, max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
          self.kill(proc)
          await asyncio.to_thread(proc.wait)
        self.kill(proc)
        group_deadline = time.monotonic() + DRAIN_S
        while group_alive(proc.pid) and time.monotonic() < group_deadline:
          await asyncio.sleep(TICK)
        if not group_alive(proc.pid):
          with contextlib.suppress(Exception):
            await asyncio.wait_for(self.bus.emit("exited", worker=worker, node=self.id), LEASE_MS / 1000)


if __name__ == "__main__":
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # so serve's finally stops the trainers
  asyncio.run(NodeDaemon(Bus()).serve())
