"""Domain types shared by the gateway, control plane, and workers (stdlib only)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

from tinyrl.control.accounting import Charge

MAX_LORAS, MAX_SEQS = 16, 256  # vLLM engine limits: --max-loras and max captured CUDA-graph batch size


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
class WorkerSpec:
  role: Role = Role.TRAINER
  base: str = ""
  lora: bool = False
  exclusive: bool = False
  train_run: str | None = None
  gpus: int = 1

  @property
  def shareable(self) -> bool:
    return self.role == Role.SAMPLER or (self.lora and not self.exclusive)

  @property
  def elastic(self) -> bool:
    return self.role == Role.SAMPLER and not self.exclusive


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
    return cls(
      False, "application/json", json.dumps({"error": f"{type(exc).__name__}: {exc}", "category": cat}).encode()
    )


@dataclass(frozen=True)
class Event:
  id: str
  t: float
  kind: str
  body: dict


@dataclass(frozen=True)
class WorkerStatus:
  worker: str
  gpus: tuple[int, ...] = ()
  bound: tuple[int, ...] = ()
  pid: int | None = None
  loaded: tuple[str, ...] = ()
  park_s: float = 0.0


@dataclass(frozen=True)
class Turn:
  id: str
  kind: str  # "train" | "sample" | "park" | "snapshot" | "close"
  worker: str
  spec: WorkerSpec = WorkerSpec()
  gpus: tuple[int, ...] = ()
  cmd: Cmd | None = None
  args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TurnResult:
  turn_id: str
  ok: bool = True
  reply: Reply | None = None
  charge: Charge | None = None
  mutated: bool = False
  loaded: bool = False
  saved: dict | None = None
  resume_s: float = 0.0
  park_s: float = 0.0


@dataclass(frozen=True)
class NodeReport:
  node: str
  boot_id: str
  gpus: int
  memory: int = 0
  draining: bool = False
  lag: float = 0.0
  workers: list[WorkerStatus] = field(default_factory=list)
  finished: list[TurnResult] = field(default_factory=list)
  staged: list[str] = field(default_factory=list)
  stage_errors: dict[str, str] = field(default_factory=dict)
  want_sweep: bool = False
  active: list[str] = field(default_factory=list)  # turns this node already holds


@dataclass(frozen=True)
class NodeReply:
  acknowledged: list[str] = field(default_factory=list)
  turns: list[Turn] = field(default_factory=list)
  prefetch: list[str] = field(default_factory=list)
  live_versions: list[str] | None = None
  reset: bool = False


def entry_sort_key(entry_id: str) -> tuple[int, int]:
  """An entry id `<ms>-<seq>` as numbers, which sort in the order the ids were made."""
  ms, seq = entry_id.split("-", 1)
  return int(ms), int(seq)


def parse_uri(uri: str) -> tuple[str, str]:
  """Split a validated `tinker://<run>/<checkpoint_id>` URI into `(run, checkpoint_id)`."""
  run, sep, cid = uri.removeprefix("tinker://").partition("/") if uri.startswith("tinker://") else ("", "", "")
  if not (sep and run and cid) or cid.startswith("/") or ".." in f"{run}/{cid}".split("/"):
    raise ValueError(f"bad checkpoint URI {uri}")
  return run, cid
