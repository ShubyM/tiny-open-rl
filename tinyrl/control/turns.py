"""Turns on a node's GPUs: which GPUs each worker placed there takes its turns on, whose turn it is, and which of a worker's runs is served next. The cluster's scheduler only places workers on nodes. A node decides everything here alone, from the processes it is running."""

from __future__ import annotations

from tinyrl.bus import Cmd, Op, Role
from tinyrl.control.scheduler import Assignment, Worker, jobs_of
from tinyrl.helpers import getenv

TURN = getenv("TURN", 200_000)  # tokens a busy holder serves on its GPUs while others wait


# ***** which GPUs a worker uses *****


def keep_chosen_gpus(new: Assignment, old: Assignment) -> Assignment:
  """A node keeps the GPUs it chose for each worker that stays where it was."""
  for name, w in new.workers.items():
    if (was := old.workers.get(name)) and was.node == w.node:
      w.gpus = was.gpus
  return new


def place_gpus(s: Assignment, node: str, running) -> None:
  """Choose desired GPUs in place: for the workers that have none, exclusive ones first, and then for trainers crowded beside a GPU that emptied. Actual process placements and reservations stay in the node's running table."""
  if node not in s.nodes:
    return
  for name, w in sorted(s.workers.items(), key=lambda item: (not item[1].spec.exclusive, item[0])):
    if w.node != node or not jobs_of(s, name):
      continue
    while not w.gpus:
      gpus, aside = choose_gpus(s, node, name)
      if aside:
        s.workers[aside[0]].gpus = choose_gpus(s, node, aside[0], avoid=gpus)[0]
      else:
        w.gpus = gpus
  spread_trainers(s, node, running)


def spread_trainers(s: Assignment, node: str, running) -> None:
  """Spread trainers after departures: one that takes turns on a crowded GPU moves to an empty one. A GPU is empty only after its old process has left it."""
  gpus = s.nodes[node][0]
  occupied = {g for r in running.values() for g in r.gpus}
  while True:
    there = {g: [n for n, w in s.workers.items() if w.node == node and g in w.gpus] for g in range(gpus)}
    empty = [g for g in range(gpus) if not there[g] and g not in occupied]
    movable = {
      g: [n for n in names if takes_turns(s.workers[n]) and len(s.workers[n].gpus) == 1]
      for g, names in there.items()
      if len(names) > 1
    }
    if not empty or not (crowded := [g for g, names in movable.items() if names]):
      return
    g = max(crowded, key=lambda g: (len(there[g]), -g))
    name = min(
      movable[g], key=lambda n: (not s.workers[n].spec.lora, n in running and running[n].served is not None, n)
    )
    s.workers[name].gpus = (empty[0],)


def takes_turns(w: Worker) -> bool:
  """Whether a worker may share a GPU it doesn't fit on. A trainer parks and resumes in about a second. A sampler wakes its engine slowly, and an exclusive worker is alone by definition."""
  return w.spec.role == Role.TRAINER and not w.spec.exclusive


def choose_gpus(s: Assignment, node: str, name: str, avoid: tuple[int, ...] = ()) -> tuple[tuple[int, ...], list[str]]:
  """Choose GPUs and trainers to move aside. Take empty GPUs first, keep exclusive workers alone, and otherwise take turns on the GPUs with the fewest workers. A sampler moves trainers rather than sharing with them when there is room."""
  gpus = s.nodes[node][0]
  me, width = s.workers[name], s.workers[name].spec.gpus
  there = {
    g: [(n, w) for n, w in s.workers.items() if n != name and w.node == node and g in w.gpus] for g in range(gpus)
  }

  def clear(g: int) -> bool:
    """Whether it could have the GPU to itself: it is empty, or holds only trainers with somewhere else to go."""
    return all(takes_turns(w) and w.spec.gpus <= gpus - width for _, w in there[g])

  options = [g for g in range(gpus) if g not in avoid]
  if len(empty := [g for g in options if not there[g]]) >= width:
    return tuple(empty[:width]), []
  if takes_turns(me) or len(options) <= width:  # with no spare GPUs there is nowhere to move to
    return tuple(sorted(options, key=lambda g: (not clear(g), len(there[g]), g))[:width]), []
  # as few moves as it can, LoRA trainers first, since a full fine-tune saves its weights and AdamW state far slower
  lora = {g: all(w.spec.lora for _, w in there[g]) for g in options}
  moves = {g: sum(w.spec.gpus for _, w in there[g]) for g in options}
  chosen = tuple(sorted(options, key=lambda g: (not clear(g), moves[g], not lora[g], g))[:width])
  aside = sorted({n for g in chosen for n, _ in there[g]})
  return chosen, aside if all(map(clear, chosen)) else []


# ***** whose turn it is on them *****


def someone_waits(running, name: str) -> bool:
  """Whether a waiting worker on overlapping GPUs has smaller (vtime, name), or `name` is idle and anyone waits."""
  me, gpus = running[name], set(running[name].gpus)
  waiters = [
    (x.vtime, n)
    for n, x in running.items()
    if n != name and x.served is None and x.vtime is not None and gpus & set(x.gpus)
  ]
  return bool(waiters) and (me.vtime is None or min(waiters) < (me.vtime, name))


def can_run(s: Assignment, node: str, running, name: str) -> bool:
  """Whether a waiting worker may take its GPUs, all together: no process is on any of them, and no smaller (vtime, name) wants one."""
  r = running[name]
  if r.vtime is None or r.served is not None or not r.gpus or node not in s.nodes:
    return False
  overlap = [(n, x) for n, x in running.items() if n != name and set(x.gpus) & set(r.gpus)]
  return not any(x.served is not None or (x.vtime is not None and (x.vtime, n) < (r.vtime, name)) for n, x in overlap)


# ***** which of a worker's runs is served next *****


def select_runs(heads: list[Cmd], service: dict[str, float], width: int, prefer: str | None = None) -> list[str]:
  """Choose ready heads by their accounts' virtual time, after any preferred continuation or sampler save."""
  if width < 1:
    raise ValueError("batch width must be positive")
  ready = {}
  for cmd in heads:
    ready.setdefault(cmd.run, cmd)
  floor = min((service[run] for run in ready if run in service), default=0.0)
  ordered = sorted(
    ready.values(),
    key=lambda cmd: (
      cmd.run != prefer,
      cmd.op != Op.SAVE_WEIGHTS_FOR_SAMPLER,  # first, since a sampler waits on it, and it arrives after its run's turn
      service.get(cmd.run, floor),
      tuple(map(int, cmd.entry_id.split("-"))),
      cmd.run,
    ),
  )
  return [cmd.run for cmd in ordered[:width]]
