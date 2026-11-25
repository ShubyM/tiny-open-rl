"""Domain and node-wire types shared by gateway, controller, node daemon, and workers (stdlib only)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

from tinyrl.control.accounting import Charge

MAX_LORAS, MAX_SEQS = 16, 256  # vLLM engine limits: --max-loras and max captured CUDA-graph batch size
Role = StrEnum("Role", "TRAINER SAMPLER")
Ev = StrEnum("Ev", "RUN_CREATED ASSIGNED RUN_DELETED NODE_UP NODE_LOST PLACED UNPLACED")
RunStatus = StrEnum("RunStatus", "FRESH LIVE SAVED LOST DELETED")
Op = StrEnum(
  "Op",
  "CREATE_MODEL FORWARD_BACKWARD OPTIM_STEP SAVE_WEIGHTS SAVE_WEIGHTS_FOR_SAMPLER LOAD_WEIGHTS DELETE_MODEL SAMPLE",
)
TaskKind = StrEnum("TaskKind", "TRAIN SAMPLE PARK SAVE DROP STOP")
Owner = tuple[str, str, str]  # (worker, node, boot_id)
StabSlice = tuple[str, int, int]  # (store_reference, byte_offset, byte_length)


class FencedError(RuntimeError):
  pass


class RequestGoneError(LookupError):
  pass


class QueueFullError(RuntimeError):
  pass


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
  def movable(self) -> bool:
    return self.role == Role.TRAINER and self.gpus == 1 and self.lora

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

  @property
  def spec(self) -> WorkerSpec:
    return WorkerSpec(
      self.role, self.base, self.rank > 0, self.exclusive, self.train_run if self.exclusive else None, self.gpus
    )


@dataclass(frozen=True)
class Cmd:
  run: str
  entry_id: str
  op: Op
  body: bytes
  cost: int = 1
  stability: dict[str, StabSlice] | None = None

  @property
  def request_id(self) -> str:
    return f"{self.run}/{self.entry_id}"


@dataclass(frozen=True)
class Reply:
  ok: bool
  content_type: str
  body: bytes
  stability: dict[str, StabSlice] | None = None

  @classmethod
  def error(cls, exc: Exception) -> Reply:
    cat = "user" if isinstance(exc, (ValueError, KeyError)) else "server"
    return cls(
      False, "application/json", json.dumps({"error": f"{type(exc).__name__}: {exc}", "category": cat}).encode()
    )


@dataclass(frozen=True)
class Checkpoint:
  uri: str
  version: str
  size: int
  time: float = 0.0
  expires_at: float | None = None
  owned: bool = False
  meta: dict = field(default_factory=dict)

  @staticmethod
  def model_info(meta: dict) -> dict:
    info = {k: meta[k] for k in ("base", "rank", "lora") if k in meta}
    if info.get("rank"):
      targets = meta.get("targets", ())
      info["flags"] = {
        "train_unembed": "lm_head" in targets,
        "train_mlp": "gate_proj" in targets,
        "train_attn": "q_proj" in targets,
      }
    return info


@dataclass(frozen=True)
class Effect:
  loaded: bool = False
  mutated: bool = False
  open_grad: bool = False
  checkpoint: Checkpoint | None = None


@dataclass(frozen=True)
class Event:
  id: str
  t: float
  kind: str
  body: dict


@dataclass(frozen=True)
class Task:
  id: str
  kind: TaskKind
  worker: str = ""
  spec: WorkerSpec = WorkerSpec()
  gpus: tuple[int, ...] = ()
  cmd: Cmd | None = None
  run: Run | None = None
  run_id: str = ""
  snapshot: str | None = None
  version: str | None = None
  attempt: int = 0


@dataclass(frozen=True)
class Done:
  id: str
  ok: bool = True
  reply: Reply | None = None
  charge: Charge | None = None
  effect: Effect | None = None
  saved: str | None = None
  duration_s: float = 0.0
  dead: bool = False
  pid: int | None = None


@dataclass(frozen=True)
class WorkerInfo:
  worker: str
  spec: WorkerSpec
  gpus: tuple[int, ...] = ()
  bound: tuple[int, ...] = ()
  loaded: dict[str, bool] = field(default_factory=dict)
  pid: int | None = None


@dataclass(frozen=True)
class NodeMsg:
  kind: str  # "hello" | "events"
  node: str
  boot_id: str
  gpus: int = 0
  gpu_mib: int = 0
  draining: bool = False
  lag: float = 0.0
  want_sweep: bool = False
  workers: tuple[WorkerInfo, ...] = ()
  staged: tuple[str, ...] | None = None
  exited: tuple[str, ...] = ()
  active: tuple[str, ...] = ()
  done: tuple[Done, ...] = ()


@dataclass(frozen=True)
class ControlMsg:
  ack: tuple[str, ...] = ()
  tasks: tuple[Task, ...] = ()
  cancel: tuple[str, ...] = ()
  stage: tuple[str, ...] = ()
  sweep: tuple[str, ...] | None = None
  reset: bool = False


def entry_sort_key(entry_id: str) -> tuple[int, int]:
  ms, seq = entry_id.split("-", 1)
  return int(ms), int(seq)


def parse_uri(uri: str) -> tuple[str, str]:
  if not uri.startswith("tinker://"):
    raise ValueError(f"bad checkpoint URI {uri}")
  run, sep, checkpoint_id = uri.removeprefix("tinker://").partition("/")
  parts = f"{run}/{checkpoint_id}".split("/")
  if not sep or "" in parts or ".." in parts:
    raise ValueError(f"bad checkpoint URI {uri}")
  return run, checkpoint_id
