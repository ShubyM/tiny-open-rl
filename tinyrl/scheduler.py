"""The scheduler: one process that decides GPU turns on every node daemon's GPUs.

python -m tinyrl.scheduler. It folds tinyrl:events into a control.State and appends what control.decide says. Run
exactly one. A restarted scheduler folds the whole log again and carries on, since its earlier decisions are in it.
"""

from __future__ import annotations

import asyncio
import time

from tinyrl import control
from tinyrl.bus import Bus, key

TICK_MS = 100  # how long it waits for an entry before it checks the quanta again
BATCH = 10_000  # entries read at once. It decides only once a read comes back short, so never from half the log


async def run(bus: Bus, policy: control.Policy = control.Policy()) -> None:
  state, after = control.State(policy=policy), "0-0"
  while True:
    events = await bus.events(after, count=BATCH, block_ms=TICK_MS)
    for e in events:
      state = control.fold(state, e)
    after = events[-1].id if events else after
    if len(events) == BATCH:
      continue
    for kind, body in control.decide(state, time.time()):
      await bus.emit(kind, **body)  # folded when it is read back on the next pass
    if events:
      await bus.r.set(key("state"), control.to_json(state))


if __name__ == "__main__":
  asyncio.run(run(Bus()))
