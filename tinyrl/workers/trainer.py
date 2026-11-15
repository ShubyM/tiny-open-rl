"""A synchronous model process. The node sends commands and publishes the returned replies and checkpoint metadata.

Model tensors, optimizer state and checkpoint I/O stay here. Only request bytes, reply bytes and checkpoint references cross the local socket.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import sys

import torch
import torch.distributed as dist

from tinyrl.api import wire
from tinyrl.bus import Cmd, Op, Reply
from tinyrl.helpers import getenv
from tinyrl.storage.store import Store, nbytes, split_version
from tinyrl.trace import span
from tinyrl.workers.model import Model

COLL_TIMEOUT_S = getenv("COLL_TIMEOUT_S", 180)
CONTROL = None  # gloo group for idle RPC dispatch so parked followers neither spin GPUs nor hit NCCL's timeout


def ok(body: dict) -> Reply:
  return Reply(True, wire.JSON, json.dumps(body).encode())


def serve_follower(model: Model) -> None:
  while True:
    dist.broadcast_object_list(msg := [None], src=0, group=CONTROL)
    method, args = msg[0]
    if method == "__stop__":
      dist.barrier()
      return
    with contextlib.suppress(ValueError, KeyError):  # rejected identically on every rank before any collective
      getattr(model, method)(*args)


class Trainer:
  def __init__(self, model: Model | None = None, store: Store | None = None):
    self.model = model if model is not None else Model()
    self.store = store if store is not None else Store()
    self.world_size = int(os.environ.get("WORLD_SIZE", 1))

  def call_model(self, method: str, *args):
    try:
      if self.world_size > 1:
        dist.broadcast_object_list([(method, args)], src=0, group=CONTROL)
      with span(f"model.{method}"):
        return getattr(self.model, method)(*args)
    except Exception as e:
      if self.world_size > 1 and not isinstance(e, (ValueError, KeyError)):
        os._exit(1)
      raise

  def ensure_run(self, run: str, base: str, rank: int, lora: dict | None = None) -> dict:
    if run not in self.model.params:
      self.call_model("create_run", run, base, rank, lora or {})
    return {"profile": self.model.profile(), "lora_config": dict(self.model.configs[run])}

  def restore_job(
    self,
    run: str,
    base: str,
    rank: int,
    lora: dict | None,
    uri: str,
    version: str | None = None,
    peers: tuple[str, ...] = (),
    optimizer: bool = True,
    gradients: bool = True,
  ) -> dict:
    created = run not in self.model.params
    if created:
      self.call_model("create_run", run, base, rank, lora or {})
    try:
      tensors, meta = self.store.get(uri, version, peers)
      info = self.call_model("restore_run", run, tensors, meta, optimizer, gradients)
      return info | {"profile": self.model.profile(), "lora_config": dict(self.model.configs[run])}
    except Exception:
      if created and run in self.model.params:
        self.call_model("delete_run", run)
      raise

  def drop_run(self, run: str) -> dict:
    if run in self.model.params:
      self.call_model("delete_run", run)
    return self.model.memory_state()

  def save(
    self,
    run: str,
    uri: str,
    *,
    sampler: bool = False,
    ttl_seconds: int | None = None,
    owned: bool = False,
    boundary: str = "",
    source_version: str | None = None,
    priority: bool = False,
  ) -> dict:
    if sampler:
      tensors, meta = self.call_model("sampler_weights", run, source_version, boundary)
    else:
      tensors, meta = self.call_model("training_state", run, boundary)
    raw = self.store.put(uri, tensors, meta, priority=priority)
    ver, loc = split_version(raw)
    size = self.store.version_size(uri, ver) or sum(map(nbytes, tensors.values()))
    head = {"uri": uri, "size": size, "ttl": ttl_seconds, "owned": owned, "version": ver, "location": loc}
    return head | {
      "durable": self.store.is_durable(uri, ver),
      "boundary": boundary,
      "optimizer": not sampler,
      "meta": meta,
    }

  def snapshot_job(self, run: str, uri: str, boundary: str, priority: bool = False) -> dict:
    return self.save(run, uri, owned=True, boundary=boundary, priority=priority)

  def execute(
    self,
    cmd: Cmd,
    run: dict | None = None,
    version: str | None = None,
    peers: tuple[str, ...] = (),
    boundary: str = "",
    source_version: str | None = None,
  ) -> tuple[Reply, dict]:
    with span("trainer.execute"):
      effects, created = {}, False
      try:
        if cmd.op == Op.DELETE_MODEL:
          if cmd.run in self.model.params:
            self.call_model("delete_run", cmd.run)
          return ok({"type": cmd.op, "model_id": cmd.run}), effects

        if cmd.op == Op.FORWARD_BACKWARD:
          if cmd.run not in self.model.params:
            raise KeyError(f"model {cmd.run} is not restored on this trainer; call load_weights")
          req = wire.parse(cmd.body)
          fb = (cmd.run, wire.datums(req), req.loss_fn or "cross_entropy", wire.loss_config(req), req.forward_only)
          logprobs, metrics = self.call_model("forward_backward", *fb)
          effects["mutated"] = not req.forward_only
          return Reply(True, wire.PROTO, wire.output(logprobs, metrics)), effects

        args = json.loads(cmd.body)
        if cmd.op not in (Op.CREATE_MODEL, Op.LOAD_WEIGHTS) and cmd.run not in self.model.params:
          raise KeyError(f"model {cmd.run} is not restored on this trainer; call load_weights")

        match cmd.op:
          case Op.CREATE_MODEL:
            self.call_model("create_run", cmd.run, args["base_model"], args["rank"], args["lora"])
            effects["profile"] = self.model.profile()
            effects["lora_config"] = dict(self.model.configs[cmd.run])
          case Op.OPTIM_STEP:
            metrics = self.call_model("optim_step", cmd.run, args["adam_params"])
            effects["mutated"] = True
            return ok({"type": cmd.op, "metrics": metrics}), effects
          case Op.SAVE_WEIGHTS:
            effects["saved"] = self.save(
              cmd.run,
              args["uri"],
              ttl_seconds=args.get("ttl_seconds"),
              owned=args.get("owned", False),
              boundary=boundary or cmd.entry_id,
              priority=True,
            )
          case Op.SAVE_WEIGHTS_FOR_SAMPLER:
            saved = self.save(
              cmd.run,
              args["uri"],
              sampler=True,
              ttl_seconds=args.get("ttl_seconds"),
              owned=args.get("owned", False),
              boundary=boundary,
              source_version=source_version,
            )
            if sid := args.get("sampling_session_id"):
              saved["sampling_session_id"] = sid
            effects["saved"] = saved
          case Op.LOAD_WEIGHTS:
            if cmd.run not in self.model.params:
              if run is None:
                raise KeyError(f"no training run {cmd.run}")
              self.call_model("create_run", cmd.run, run["base"], run["rank"], run.get("lora", {}))
              created = True
            with self.store.hold(args["uri"], version, peers) as (_, resolved):
              tensors, meta = self.store.get(args["uri"], resolved, peers)
            info = self.call_model("restore_run", cmd.run, tensors, meta, args["optimizer"], False)
            created = False
            effects["mutated"] = True
            ld = {
              "uri": args["uri"],
              "version": resolved,
              "optimizer": bool(args["optimizer"]),
              "boundary": cmd.entry_id,
            }
            effects["loaded"] = ld | {
              "durable": self.store.is_durable(args["uri"], resolved),
              "profile": self.model.profile(),
              "rng_remapped": info["rng_remapped"],
            }
          case _:
            raise ValueError(f"unknown op {cmd.op}")

        reply = {"type": cmd.op, "model_id": cmd.run, **({"path": args["uri"]} if "uri" in args else {})}
        if "sampling_session_id" in args:
          reply["sampling_session_id"] = args["sampling_session_id"]
        return ok(reply), effects
      except Exception as e:
        if created and cmd.run in self.model.params:
          self.call_model("delete_run", cmd.run)
        return Reply.error(e), effects

  def park(self) -> dict:
    return self.call_model("park_on_host")

  def resume(self, gpu: int | None = None) -> dict:
    if gpu is not None and torch.cuda.is_available():
      torch.cuda.set_device(gpu)
    return self.call_model("resume_on_gpu")

  def memory(self) -> dict:
    return self.model.memory_state()

  def shutdown(self) -> bool:
    if self.world_size > 1 and dist.is_initialized():
      dist.broadcast_object_list([("__stop__", ())], src=0, group=CONTROL)
      dist.barrier()
    return True


def main(path: str) -> None:
  global CONTROL
  from tinyrl.control.process import die_with_parent, serve

  die_with_parent()
  if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
  if int(os.environ.get("WORLD_SIZE", 1)) > 1:
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
    timeout = datetime.timedelta(seconds=COLL_TIMEOUT_S)
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", device_id=dev, timeout=timeout)
    CONTROL = dist.new_group(backend="gloo", timeout=datetime.timedelta(days=365))
  try:
    if dist.is_initialized() and dist.get_rank() > 0:
      serve_follower(Model())
    else:
      trainer = Trainer()
      serve(path, lambda method, *args: getattr(trainer, method)(*args))
  finally:
    if dist.is_initialized():
      with contextlib.suppress(Exception):
        dist.destroy_process_group()


if __name__ == "__main__":
  main(sys.argv[1])
