"""Turns on a node's GPUs: whose turn it is on shared GPUs, and which of a worker's runs is served next.

The cluster scheduler chooses each worker's `(node, gpus)`; a node only arbitrates turns among the processes on its GPUs.
"""

from tinyrl.bus import Cmd, Op
from tinyrl.control.scheduler import Assignment
from tinyrl.helpers import getenv

TURN = getenv("TURN", 200_000)  # tokens a busy holder serves on its GPUs before yielding to a less-served waiter


def peers(running, name: str) -> list:
  return [(n, x) for n, x in running.items() if n != name and {*x.gpus} & {*running[name].gpus}]


def someone_waits(running, name: str) -> bool:
  """Whether a waiting worker on overlapping GPUs has smaller (vtime, name), or `name` is idle and anyone waits."""
  wait = [(x.vtime, n) for n, x in peers(running, name) if x.served is None and x.vtime is not None]
  return bool(wait) and (running[name].vtime is None or min(wait) < (running[name].vtime, name))


def can_run(s: Assignment, node: str, running, name: str) -> bool:
  """Whether a waiting worker may take its GPUs, all together: no process is on any of them, and no smaller (vtime, name) wants one."""
  r, overlap = running[name], peers(running, name)
  if r.vtime is None or r.served is not None or not r.gpus or node not in s.nodes or (r.spec.exclusive and overlap):
    return False
  return not any(x.served is not None or (x.vtime is not None and (x.vtime, n) < (r.vtime, name)) for n, x in overlap)


def select_runs(heads: list[Cmd], service: dict[str, float], width: int, prefer: str | None = None) -> list[str]:
  """Choose ready heads by their accounts' virtual time, after any preferred continuation or sampler save."""
  if width < 1:
    raise ValueError("batch width must be positive")
  ready: dict[str, Cmd] = {}
  for cmd in heads:
    ready.setdefault(cmd.run, cmd)
  floor, save = min((service[r] for r in ready if r in service), default=0.0), Op.SAVE_WEIGHTS_FOR_SAMPLER

  def skey(c: Cmd) -> tuple:
    return (c.run != prefer, c.op != save, service.get(c.run, floor), (*map(int, c.entry_id.split("-")),), c.run)

  return [cmd.run for cmd in sorted(ready.values(), key=skey)[:width]]
