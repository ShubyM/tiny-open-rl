"""Controller-side live-node registry, lease expiry, and durable boot_id fencing."""

from __future__ import annotations

import time
from dataclasses import dataclass

from tinyrl.helpers import LEASE_MS
from tinyrl.types import FencedError, NodeStatus, Owner

MAX_EXPIRED_BOOTS = 1024


@dataclass
class Node:
  boot_id: str
  until: float
  status: NodeStatus


class NodeTable(dict[str, Node]):
  def __init__(self, expired_boots: list[str] | tuple[str, ...] = (), *, held: bool = False, now: float | None = None):
    super().__init__()
    mono = time.monotonic() if now is None else now
    self.expired_boots: list[str] = list(expired_boots)
    self.grace_until = mono + LEASE_MS / 1000 if held else 0.0

  def expired(self, active_owners: list[Owner | None]) -> list[tuple[str, str]]:
    now = time.monotonic()
    stale = [(nid, n.boot_id) for nid, n in self.items() if n.until <= now]
    if self.grace_until and now >= self.grace_until:
      self.grace_until = 0.0
      live = {(nid, n.boot_id) for nid, n in self.items()} - set(stale)
      stale.extend(sorted({o[1:] for o in active_owners if o} - live))
    return stale

  def remove(self, node_id: str, boot_id: str) -> list[str]:
    if (cur := self.get(node_id)) and cur.boot_id == boot_id:
      del self[node_id]
    if boot_id not in self.expired_boots:
      self.expired_boots = [*self.expired_boots[-(MAX_EXPIRED_BOOTS - 1) :], boot_id]
    return list(self.expired_boots)

  def fence(self, node_id: str, boot_id: str) -> Node | None:
    cur = self.get(node_id)
    if boot_id in self.expired_boots or (cur is not None and cur.boot_id != boot_id):
      raise FencedError(f"fenced boot {node_id}/{boot_id}")
    if cur is not None:
      cur.until = time.monotonic() + LEASE_MS / 1000
    return cur

  def record(self, node_id: str, boot_id: str, cur: Node | None, status: NodeStatus, now: float) -> tuple[bool, bool]:
    changed = cur is not None and (cur.status.gpus, cur.status.draining) != (status.gpus, status.draining)
    self[node_id] = Node(boot_id, now + LEASE_MS / 1000, status)
    return cur is None, changed

  def active_gpus(self) -> dict[str, int]:
    return {nid: n.status.gpus for nid, n in self.items() if not n.status.draining}
