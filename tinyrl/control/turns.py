"""Turns on a node's GPUs: which GPUs each worker placed there takes its turns on, whose turn it is, and which of a worker's runs is served next. The cluster's scheduler only places workers on nodes. A node decides everything here alone, from the processes it is running."""

from __future__ import annotations

from tinyrl.bus import Cmd, Op, Role
from tinyrl.control.scheduler import Assignment, Worker, fits, jobs_of, memory_budget
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
  gpus, gpu_bytes = s.nodes[node]
  occupied = {g for r in running.values() for g in r.gpus}
  while True:
    there = {g: [n for n, w in s.workers.items() if w.node == node and g in w.gpus] for g in range(gpus)}
    empty = [g for g in range(gpus) if not there[g] and g not in occupied]
    movable = {
      g: [n for n in names if takes_turns(s.workers[n]) and len(s.workers[n].gpus) == 1]
      for g, names in there.items()
      if len(names) > 1 and not fits([memory_budget(s, s.workers[n].spec, node) for n in names], gpu_bytes)
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
  """Choose GPUs and trainers to move aside. Pack known memory budgets, keep exclusive workers alone, and otherwise take turns on the GPUs with the fewest workers. A sampler moves trainers rather than sharing with them when there is room."""
  gpus, gpu_bytes = s.nodes[node]
  me, width, memory = s.workers[name], s.workers[name].spec.gpus, memory_budget(s, s.workers[name].spec, node)
  there = {
    g: [(n, w) for n, w in s.workers.items() if n != name and w.node == node and g in w.gpus] for g in range(gpus)
  }

  def fits_on(g: int) -> bool:
    others = [w for _, w in there[g]]
    ok = not me.spec.exclusive and not any(w.spec.exclusive for w in others)
    return not others or (ok and fits([memory_budget(s, w.spec, node) for w in others] + [memory], gpu_bytes))

  def trainers_only(g: int) -> bool:
    return bool(there[g]) and all(takes_turns(w) and w.spec.gpus <= gpus - width for _, w in there[g])

  def clear(g: int) -> bool:
    return fits_on(g) or trainers_only(g)

  options = [g for g in range(gpus) if g not in avoid]
  if len(fitting := [g for g in options if fits_on(g)]) >= width:
    return tuple(sorted(fitting, key=lambda g: (bool(there[g]) != bool(memory), g))[:width]), []
  if takes_turns(me) or len(options) <= width:  # with no spare GPUs there is nowhere to move to
    return tuple(sorted(options, key=lambda g: (not clear(g), len(there[g]), g))[:width]), []
  # as few moves as it can, LoRA trainers first, since a full fine-tune saves its weights and AdamW state far slower
  lora = {g: all(w.spec.lora for _, w in there[g]) for g in options}
  moves = {g: 0 if fits_on(g) else sum(w.spec.gpus for _, w in there[g]) for g in options}
  chosen = tuple(sorted(options, key=lambda g: (not clear(g), moves[g], not lora[g], g))[:width])
  aside = sorted({n for g in chosen if not fits_on(g) for n, _ in there[g]})
  return chosen, aside if all(map(clear, chosen)) else []


# ***** whose turn it is on them *****


def someone_waits(running, gpus: tuple[int, ...]) -> bool:
  return any(r.ticket > 0 and any(g in r.gpus for g in gpus) for r in running.values())


def can_run(s: Assignment, node: str, running, name: str) -> bool:
  """Reserve all actual GPUs together, in ticket order on each GPU, including processes still draining after unplacement."""
  r = running[name]
  if r.ticket <= 0 or r.served is not None or not r.gpus or node not in s.nodes:
    return False
  for gpu in r.gpus:
    if any(x.ticket > 0 and (x.ticket, n) < (r.ticket, name) and gpu in x.gpus for n, x in running.items()):
      return False
    memory = [memory_budget(s, x.spec, node) for x in running.values() if x.served is not None and gpu in x.gpus]
    if not fits([*memory, memory_budget(s, r.spec, node)], s.nodes[node][1]):
      return False
  return True


# ***** which of a worker's runs is served next *****


def select_runs(heads: list[Cmd], service: dict[str, float], width: int, prefer: str | None = None) -> list[str]:
  """Choose ready heads by the tokens each run has been served, not a global tenant share.
  The caller bounds preferred continuations. New runs join the active service floor.
  """
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
