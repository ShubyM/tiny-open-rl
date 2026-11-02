"""A worker's turn on its GPUs, when it shares them with other workers.

The worker appends facts to tinyrl:events: wants_gpu, batch_done, idle, released. It reads the scheduler's decisions about its worker, granted and revoked, from the same log. See scheduler.py for the rules.
"""

from __future__ import annotations

import json

from tinyrl.bus import Bus, Ev, key
from tinyrl.helpers import getenv

# how long a worker holding its GPU waits for commands before it reports idle, so waiters don't wait
IDLE_MS = getenv("IDLE_MS", 150)


class GpuTurn:
  def __init__(self, bus: Bus, worker: str, shared: bool):
    self.bus, self.worker, self.shared = bus, worker, shared
    self.instance = getenv("TINYRL_PROCESS", "")
    self.held = not shared  # with no GPU to share there is nothing to take turns on
    self.idle = False
    self.after: str | None = None  # the last log entry read. None until its first ask, so it never reads older ones

  async def wait_for_turn(self, queued: int) -> bool:
    """Wait for a grant. True if this call took the turn, so the worker brings its tensors back to the GPU."""
    if self.held:
      return False
    self.after = await self.bus.append_event(Ev.WANTS_GPU, worker=self.worker, instance=self.instance, queued=queued)
    while await self.next_decision(5000) != Ev.GRANTED:
      pass
    self.held = True
    return True

  async def was_revoked(self) -> bool:
    return self.held and self.shared and await self.next_decision(0) == Ev.REVOKED

  async def report_batch_done(self, **report) -> None:
    self.idle = False
    if self.shared:
      await self.bus.append_event(Ev.BATCH_DONE, worker=self.worker, instance=self.instance, **report)

  async def report_idle(self) -> None:
    if self.held and self.shared and not self.idle:
      self.idle = True
      await self.bus.append_event(Ev.IDLE, worker=self.worker, instance=self.instance)

  async def give_turn_back(self) -> None:
    if self.held and self.shared:
      self.held = False
      await self.bus.append_event(Ev.RELEASED, worker=self.worker, instance=self.instance)

  async def next_decision(self, block_ms: int) -> str | None:
    """The next decision about this worker in the log, reading past everything else. None once it has read it all, after waiting up to block_ms for a new entry."""
    saved = json.loads(await self.bus.r.get(key("state")) or "{}")
    if saved.get("after") and tuple(map(int, saved["after"].split("-"))) >= tuple(
      map(int, (self.after or "0-0").split("-"))
    ):
      self.after = saved["after"]  # recover before newer unrelated events advance past a trimmed grant
      w = saved.get("workers", {}).get(self.worker, {})
      if w.get("instance", "") != self.instance:
        raise RuntimeError("this worker process was replaced")
      if self.held and w.get("revoked"):
        return Ev.REVOKED
      if not self.held and w.get("since") is not None:
        return Ev.GRANTED
    while events := await self.bus.read_events(self.after, count=100, block_ms=block_ms):
      block_ms = 0
      for e in events:
        self.after = e.id
        if (
          e.kind in (Ev.GRANTED, Ev.REVOKED)
          and e.body["worker"] == self.worker
          and e.body.get("instance", "") == self.instance
        ):
          return e.kind
    return None
