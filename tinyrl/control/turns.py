"""A worker's turn on its GPUs, when it shares them with other workers.

The worker appends facts to tinyrl:events: wants_gpu, batch_done, idle, released. It reads the scheduler's decisions
about its worker, granted and revoked, from the same log. See scheduler.py for the rules.
"""

from __future__ import annotations

from tinyrl.bus import Bus
from tinyrl.helpers import getenv

# how long a worker holding its GPU waits for commands before it reports idle, so waiters don't wait
IDLE_MS = getenv("IDLE_MS", 150)


class GpuTurn:
  def __init__(self, bus: Bus, worker: str, shared: bool):
    self.bus, self.worker, self.shared = bus, worker, shared
    self.held = not shared  # with no GPU to share there is nothing to take turns on
    self.idle = False
    self.after: str | None = None  # the last log entry read. None until its first ask, so it never reads older ones

  async def wait_for_turn(self, queued: int) -> bool:
    """Wait for a grant. True if this call took the turn, so the worker brings its tensors back to the GPU."""
    if self.held:
      return False
    asked = await self.bus.append_event("wants_gpu", worker=self.worker, queued=queued)
    self.after = self.after or asked
    while await self.next_decision(5000) != "granted":
      pass
    self.held = True
    return True

  async def was_revoked(self) -> bool:
    return self.held and self.shared and await self.next_decision(0) == "revoked"

  async def report_batch_done(self, **report) -> None:
    self.idle = False
    if self.shared:
      await self.bus.append_event("batch_done", worker=self.worker, **report)

  async def report_idle(self) -> None:
    if self.held and self.shared and not self.idle:
      self.idle = True
      await self.bus.append_event("idle", worker=self.worker)

  async def give_turn_back(self) -> None:
    if self.held and self.shared:
      self.held = False
      await self.bus.append_event("released", worker=self.worker)

  async def next_decision(self, block_ms: int) -> str | None:
    """The next decision about this worker in the log, reading past everything else. None once it has read it all, after
    waiting up to block_ms for a new entry."""
    while events := await self.bus.read_events(self.after, count=100, block_ms=block_ms):
      block_ms = 0
      for e in events:
        self.after = e.id
        if e.kind in ("granted", "revoked") and e.body["worker"] == self.worker:
          return e.kind
    return None
