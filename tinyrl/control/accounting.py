"""GPU service shared equally by the model calls currently executing in a wave."""

from dataclasses import dataclass, field
from time import monotonic_ns, time_ns


@dataclass
class Charge:
  started_ms: int = field(default_factory=lambda: time_ns() // 1_000_000)  # entry ids start with Unix time
  busy_ms: int = 0
  gpu_ms: float = 0.0


@dataclass
class Wave:
  gpus: int
  active: int = 0
  last_ns: int = 0
  service_ms: float = 0.0  # cumulative share of a call active since the start of the wave

  def _change_active(self, delta: int) -> tuple[int, float]:
    now = monotonic_ns()
    if self.active:
      self.service_ms += (now - self.last_ns) * self.gpus / (1_000_000 * self.active)
    self.last_ns, self.active = now, self.active + delta
    return now, self.service_ms

  async def run(self, call, charge: Charge):
    start, initial_ms = self._change_active(1)
    try:
      return await call
    finally:
      end, final_ms = self._change_active(-1)
      charge.busy_ms += (end - start) // 1_000_000
      charge.gpu_ms += final_ms - initial_ms
