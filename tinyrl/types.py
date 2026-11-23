"""Domain and RPC contract types shared by gateway, controller, node daemon, and workers (stdlib only)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

MAX_LORAS, MAX_SEQS = 16, 256  # vLLM engine limits: --max-loras and max captured CUDA-graph batch size

Role = StrEnum("Role", "TRAINER SAMPLER")
Ev = StrEnum("Ev", "RUN_CREATED ASSIGNED RUN_DELETED NODE_UP NODE_LOST PLACED UNPLACED")
RunStatus = StrEnum("RunStatus", "FRESH LIVE SAVED LOST DELETED")  # fresh -> live <-> saved, live -> lost, * -> deleted
OPS = "CREATE_MODEL FORWARD_BACKWARD OPTIM_STEP SAVE_WEIGHTS SAVE_WEIGHTS_FOR_SAMPLER LOAD_WEIGHTS DELETE_MODEL SAMPLE"
Op = StrEnum("Op", OPS)

Owner = tuple[str, str, str]  # (worker, node, boot_id)


class FencedError(RuntimeError):
  """Raised when a stale or expired node incarnation calls the controller (maps to HTTP 409)."""


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
    tr = self.train_run if self.exclusive else None
    return WorkerSpec(self.role, self.base, self.rank > 0, self.exclusive, tr, self.gpus)


@dataclass(frozen=True)
class Cmd:
  run: str
  entry_id: str
  op: Op
  body: bytes
  cost: int = 1
  stability: dict[str, bytes] | None = None

  @property
  def request_id(self) -> str:
    return f"{self.run}/{self.entry_id}"


@dataclass(frozen=True)
class Reply:
  ok: bool
  content_type: str
  body: bytes
  stability: dict[str, bytes] | None = None

  @classmethod
  def error(cls, exc: Exception) -> Reply:
    cat = "user" if isinstance(exc, (ValueError, KeyError)) else "server"
    msg = json.dumps({"error": f"{type(exc).__name__}: {exc}", "category": cat}).encode()
    return cls(False, "application/json", msg)


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
  """Local post-command state reported by a Trainer subprocess to its NodeDaemon."""

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
class LocalWorker:
  """One observation from a node. Durations are relative to its observation time."""

  current: tuple[int, ...] = ()
  bound: tuple[int, ...] = ()
  holding: tuple[str, ...] = ()
  busy: bool = False
  resident_s: float | None = None
  switch_s: float = 0.0
  idle_s: float | None = None
  gpu_ms: float = 0.0
  loaded: dict[str, bool] = field(default_factory=dict)  # whether each loaded run has an open gradient step
  pid: int | None = None


@dataclass(frozen=True)
class NodeStatus:
  gpus: int
  draining: bool = False
  lag: float = 0.0
  workers: dict[str, LocalWorker] = field(default_factory=dict)
  staged: tuple[str, ...] = ()
  want_sweep: bool = False


@dataclass(frozen=True)
class Assignment:
  worker: str
  spec: WorkerSpec
  runs: tuple[str, ...] = ()
  versions: tuple[str, ...] = ()
  ready: bool = False
  closing: tuple[str, ...] = ()
  lag: float = 0.0


@dataclass(frozen=True)
class Work:
  cmd: Cmd
  run: Run | None = None
  snapshot: str | None = None
  version: str | None = None
  attempt: int = 0


@dataclass(frozen=True)
class Grant:
  worker: str
  spec: WorkerSpec
  gpus: tuple[int, ...]
  work: tuple[Work, ...]


class Release(StrEnum):
  PARK = "park"
  STOP = "stop"


@dataclass(frozen=True)
class DetachRun:
  run: str


@dataclass(frozen=True)
class NodePlan:
  grants: list[Grant] = field(default_factory=list)
  release: dict[str, Release | DetachRun] = field(default_factory=dict)
  stage: tuple[str, ...] = ()
  sweep: list[str] | None = None
  wake_after: float | None = None


RPC_TYPES = {"take": NodePlan, "finish": bool, "detach": bool}


def entry_sort_key(entry_id: str) -> tuple[int, int]:
  ms, seq = entry_id.split("-", 1)
  return int(ms), int(seq)


def parse_uri(uri: str) -> tuple[str, str]:
  run, sep, cid = uri.removeprefix("tinker://").partition("/") if uri.startswith("tinker://") else ("", "", "")
  parts = f"{run}/{cid}".split("/")
  if not sep or "" in parts or ".." in parts:
    raise ValueError(f"bad checkpoint URI {uri}")
  return run, cid
