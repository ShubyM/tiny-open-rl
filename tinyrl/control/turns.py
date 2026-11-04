"""A worker's turn on its GPUs, when it shares them with other workers.

The worker appends facts to tinyrl:events: wants_gpu, batch_done, idle, released. It reads the scheduler's decisions about its worker, granted and revoked, from the same log. See scheduler.py for the rules.
"""

from __future__ import annotations

from tinyrl.bus import Bus, Ev
from tinyrl.helpers import getenv

# how long a worker holding its GPU waits for commands before it reports idle, so waiters don't wait
IDLE_MS = getenv("IDLE_MS", 150)
# set when its process sees all of its node's GPUs, so a move to another is a park and a resume
NODE = getenv("TINYRL_NODE", "")


class GpuTurn:
  def __init__(self, bus: Bus, worker: str, shared: bool):
    self.bus, self.worker, self.shared = bus, worker, shared
    self.held = not shared  # with no GPU to share there is nothing to take turns on
    self.idle = False
    self.after: str | None = None  # the last log entry read. None until its first ask, so it never reads older ones
    self.gpu: int | None = None  # the GPU the scheduler last moved it to, which it starts on once it holds nothing
    self.moved = False  # moved, and not yet told the scheduler it is on its new GPU

  async def wait_for_turn(self, queued: int) -> bool:
    """Wait for a grant. True if this call took the turn, so the worker brings its tensors back to the GPU."""
    if self.held:
      return False
    while True:
      asked = await self.bus.append_event(Ev.WANTS_GPU, worker=self.worker, queued=queued)
      self.after = self.after or asked
      while (decision := await self.next_decision(5000)) != Ev.GRANTED and not self.moved:
        pass
      if decision == Ev.GRANTED:
        break
      await self.start_on_new_gpu()
    self.held = True
    return True

  async def was_revoked(self) -> bool:
    if not self.shared or not (self.held or NODE and self.after):  # one that holds nothing reads only to follow a move
      return False
    decision = await self.next_decision(0)
    if decision == Ev.REVOKED and not self.held:
      await self.start_on_new_gpu()
    return self.held and decision == Ev.REVOKED

  async def start_on_new_gpu(self) -> None:
    """With nothing of its on its old GPU, tell the scheduler its process is on its new one."""
    if self.moved:
      self.moved = False
      await self.bus.append_event(Ev.STARTED, worker=self.worker, node=NODE)

  async def report_batch_done(self, **report) -> None:
    self.idle = False
    if self.shared:
      await self.bus.append_event(Ev.BATCH_DONE, worker=self.worker, **report)

  async def report_idle(self) -> None:
    if self.held and self.shared and not self.idle:
      self.idle = True
      await self.bus.append_event(Ev.IDLE, worker=self.worker)

  async def give_turn_back(self) -> None:
    if self.held and self.shared:
      self.held = False
      await self.bus.append_event(Ev.RELEASED, worker=self.worker)
      await self.start_on_new_gpu()

  async def next_decision(self, block_ms: int) -> str | None:
    """The next decision about this worker in the log, reading past everything else. None once it has read it all, after waiting up to block_ms for a new entry."""
    while events := await self.bus.read_events(self.after, count=100, block_ms=block_ms):
      block_ms = 0
      for e in events:
        self.after = e.id
        if e.kind in (Ev.GRANTED, Ev.REVOKED) and e.body["worker"] == self.worker:
          return e.kind
        if NODE and e.kind == Ev.PLACED and e.body["worker"] == self.worker and e.body["node"] == NODE:
          # moved on its node: it gives its GPU back as if revoked, then starts on the new one
          self.gpu, self.moved = e.body["gpus"][0], True
          return Ev.REVOKED
    return None
