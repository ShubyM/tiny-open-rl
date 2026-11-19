"""Domain types shared by the gateway, control plane, and workers (stdlib only)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum


class Op(StrEnum):
  CREATE_MODEL = "create_model"
  FORWARD_BACKWARD = "forward_backward"
  OPTIM_STEP = "optim_step"
  SAVE_WEIGHTS = "save_weights"
  SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"
  LOAD_WEIGHTS = "load_weights"
  DELETE_MODEL = "delete_model"
  SAMPLE = "sample"


class Ev(StrEnum):
  """Cluster trace events: runs and nodes coming and going, and planner decisions."""

  RUN_CREATED = "run_created"
  ASSIGNED = "assigned"
  RUN_DELETED = "run_deleted"
  NODE_UP = "node_up"
  NODE_LOST = "node_lost"
  PLACED = "placed"
  UNPLACED = "unplaced"


class Role(StrEnum):
  TRAINER = "trainer"
  SAMPLER = "sampler"


@dataclass(frozen=True)
class Run:
  id: str
  base: str
  rank: int
  session: str = ""
  model_path: str | None = None
  lora: dict = field(default_factory=dict)
  exclusive: bool = False
  role: Role = Role.TRAINER
  train_run: str | None = None
  gpus: int = 1
  account: str = ""


@dataclass(frozen=True)
class Cmd:
  run: str
  entry_id: str
  op: Op
  body: bytes
  cost: int = 1

  @property
  def request_id(self) -> str:
    return f"{self.run}/{self.entry_id}"


@dataclass(frozen=True)
class Reply:
  ok: bool
  content_type: str
  body: bytes

  @classmethod
  def error(cls, exc: Exception) -> Reply:
    cat = "user" if isinstance(exc, (ValueError, KeyError)) else "server"
    msg = json.dumps({"error": f"{type(exc).__name__}: {exc}", "category": cat})
    return cls(False, "application/json", msg.encode())


@dataclass(frozen=True)
class Event:
  id: str
  t: float
  kind: str
  body: dict


def unpack_snapshot(snapshot: str | None) -> tuple[str, str | None]:
  return tuple(snapshot.split("\n", 1)) if snapshot else ("", None)


def parse_uri(uri: str) -> tuple[str, str]:
  """Split a validated `tinker://<run>/<checkpoint_id>` URI into `(run, checkpoint_id)`."""
  run, sep, cid = uri.removeprefix("tinker://").partition("/") if uri.startswith("tinker://") else ("", "", "")
  if not (sep and run and cid) or cid.startswith("/") or ".." in f"{run}/{cid}".split("/"):
    raise ValueError(f"bad checkpoint URI {uri}")
  return run, cid
