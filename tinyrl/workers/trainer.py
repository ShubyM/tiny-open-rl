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
from tinyrl.bus import Cmd, Op, Reply, entry_key
from tinyrl.storage.store import Store, nbytes
from tinyrl.trace import span
from tinyrl.workers.model import Model

CONTROL = None  # parked ranks wait on CPU, not inside a long-lived NCCL collective


def ok(body: dict) -> Reply:
  return Reply(True, wire.JSON, json.dumps(body).encode())


def serve_follower(model: Model) -> None:
  while True:
    dist.broadcast_object_list(msg := [None], src=0, group=CONTROL)
    with contextlib.suppress(ValueError, KeyError):  # rejected the same way on every rank, before any collective
      getattr(model, msg[0][0])(*msg[0][1])


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
      # The other ranks may be waiting in a collective this rank left; torchrun must restart them all.
      if self.world_size > 1 and not isinstance(e, (ValueError, KeyError)):
        os._exit(1)
      raise

  def save(
    self,
    run: str,
    uri: str,
    sampler: bool = False,
    ttl_seconds: int | None = None,
    owned: bool = False,
    boundary: str = "",
  ) -> dict:
    tensors, meta = self.call_model("sampler_weights" if sampler else "training_state", run)
    meta = meta | {"boundary": boundary}
    version = self.store.put(uri, tensors, meta)
    return {
      "uri": uri,
      "size": sum(map(nbytes, tensors.values())),
      "ttl": ttl_seconds,
      "owned": owned,
      "version": version,
      "boundary": boundary,
      "meta": meta,
    }

  def load_run(self, run: str, record: dict | None, uri: str, version: str, optimizer: bool) -> None:
    created = run not in self.model.params
    try:
      if created:
        if record is None:
          raise KeyError(f"no training run {run}")
        self.call_model("create_run", run, record["base"], record["rank"], record.get("lora", {}))
      self.call_model("restore_run", run, *self.store.get(uri, version), optimizer)
    except Exception:
      if created:
        self.drop(run)
      raise

  def ensure(self, run: str, record: dict | None, parked: tuple[str, str] | None = None) -> None:
    if run in self.model.params:
      return
    if record is None:
      raise KeyError(f"no training run {run}")
    state = record.get("state", {})
    boundary = state.get("created", "")
    if parked:
      boundary = next(
        (
          cp["boundary"]
          for cp in (state.get("working"), state.get("durable"))
          if cp and (cp["uri"], cp["version"]) == parked
        ),
        "",
      )
    if not boundary or entry_key(state.get("mutation", "")) > entry_key(boundary):
      raise RuntimeError(
        f"model {run} rolled back to {boundary or 'creation'}; explicitly load a checkpoint before continuing"
      )
    if parked:
      self.load_run(run, record, *parked, optimizer=True)
    else:
      self.call_model("create_run", run, record["base"], record["rank"], record.get("lora", {}))

  def export(self, run: str, record: dict, parked: tuple[str, str] | None = None) -> dict | None:
    state = record["state"]
    if run not in self.model.params and not (state.get("created") or state.get("mutation")):
      return None
    self.ensure(run, record, parked)
    return self.save(run, f"tinker://{run}/parked", owned=True, boundary=state["mutation"])

  def execute(self, cmd: Cmd, run: dict | None = None, parked: tuple[str, str] | None = None) -> tuple[Reply, dict]:
    with span("trainer.execute"):
      effects = {}
      try:
        if cmd.op == Op.DELETE_MODEL and cmd.run not in self.model.params:
          return ok({"type": cmd.op, "model_id": cmd.run}), effects
        if cmd.op not in (Op.CREATE_MODEL, Op.LOAD_WEIGHTS):
          self.ensure(cmd.run, run, parked)

        if cmd.op == Op.FORWARD_BACKWARD:
          req = wire.parse(cmd.body)
          fb = (cmd.run, wire.datums(req), req.loss_fn or "cross_entropy", wire.loss_config(req), req.forward_only)
          logprobs, metrics = self.call_model("forward_backward", *fb)
          effects["mutated"] = not req.forward_only
          return Reply(True, wire.PROTO, wire.output(logprobs, metrics)), effects

        args = json.loads(cmd.body)
        match cmd.op:
          case Op.CREATE_MODEL:
            self.call_model("create_run", cmd.run, args["base_model"], args["rank"], args["lora"])
            effects["mutated"] = True
          case Op.OPTIM_STEP:
            metrics = self.call_model("optim_step", cmd.run, args["adam_params"])
            effects["mutated"] = True
            return ok({"type": cmd.op, "metrics": metrics}), effects
          case Op.SAVE_WEIGHTS | Op.SAVE_WEIGHTS_FOR_SAMPLER:
            effects["checkpoint"] = self.save(
              cmd.run,
              args["uri"],
              cmd.op == Op.SAVE_WEIGHTS_FOR_SAMPLER,
              args.get("ttl_seconds"),
              args.get("owned", False),
              cmd.entry_id,
            )
          case Op.LOAD_WEIGHTS:
            self.load_run(cmd.run, run, args["uri"], args["version"], args["optimizer"])
            effects["mutated"] = True
            if args["optimizer"]:
              effects["checkpoint"] = {
                "uri": args["uri"],
                "version": args["version"],
                "optimizer": True,
                "boundary": cmd.entry_id,
              }
          case Op.DELETE_MODEL:
            self.call_model("delete_run", cmd.run)
          case _:
            raise ValueError(f"unknown op {cmd.op}")

        reply = {"type": cmd.op, "model_id": cmd.run, **({"path": args["uri"]} if "uri" in args else {})}
        if "sampling_session_id" in args:
          reply["sampling_session_id"] = args["sampling_session_id"]
        return ok(reply), effects
      except Exception as e:
        return Reply.error(e), effects

  def drop(self, run: str) -> None:
    if run in self.model.params:
      self.call_model("delete_run", run)

  def retain(self, runs) -> None:
    for run in self.model.params.keys() - set(runs):
      self.drop(run)

  def park(self) -> None:
    self.call_model("park_on_host")

  def resume(self) -> None:
    self.call_model("resume_on_gpu")


def main(path: str) -> None:
  global CONTROL
  from tinyrl.control.process import die_with_parent, serve

  die_with_parent()  # also for torchrun ranks, whose sessions are outside their launcher's process group
  if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
  if int(os.environ.get("WORLD_SIZE", 1)) > 1:
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", device_id=dev)
    CONTROL = dist.new_group(backend="gloo", timeout=datetime.timedelta(days=365))
  try:
    if dist.is_initialized() and dist.get_rank() > 0:
      serve_follower(Model())
    else:
      trainer = Trainer()
      serve(path, lambda method, *args: getattr(trainer, method)(*args))
  finally:
    if dist.is_initialized():
      dist.destroy_process_group()


if __name__ == "__main__":
  main(sys.argv[1])
