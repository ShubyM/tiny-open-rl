"""A synchronous model process. The node sends commands and publishes the returned replies and checkpoint metadata.

Model tensors, optimizer state and checkpoint I/O stay here. Only request bytes, reply bytes and checkpoint references cross the local socket.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time

import torch
import torch.distributed as dist

from tinyrl.api import wire
from tinyrl.bus import Cmd, Op, Reply
from tinyrl.storage.store import Store, nbytes
from tinyrl.trace import trace
from tinyrl.workers.model import Model


def ok(body: dict) -> Reply:
  return Reply(True, wire.JSON, json.dumps(body).encode())


def serve_follower(model: Model) -> None:
  while True:
    dist.broadcast_object_list(msg := [None], src=0)
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
        dist.broadcast_object_list([(method, args)], src=0)
      return getattr(self.model, method)(*args)
    except Exception as e:
      # The other ranks may be waiting in a collective this rank left; torchrun must restart them all.
      if self.world_size > 1 and not isinstance(e, (ValueError, KeyError)):
        os._exit(1)
      raise

  def save(
    self, run: str, uri: str, sampler: bool = False, ttl_seconds: int | None = None, owned: bool = False
  ) -> dict:
    tensors, meta = self.call_model("sampler_weights" if sampler else "training_state", run)
    self.store.put(uri, tensors, meta)
    return {
      "uri": uri,
      "size": sum(map(nbytes, tensors.values())),
      "ttl": ttl_seconds,
      "owned": owned,
      "version": self.store.versions.get(uri),
    }

  def load_run(self, run: str, uri: str, version: str | None, optimizer: bool, gradients: bool) -> None:
    self.call_model("restore_run", run, *self.store.get(uri, version), optimizer, gradients)

  def execute(
    self, cmd: Cmd, run: dict | None = None, parked: tuple[str, str | None] | None = None, version: str | None = None
  ) -> tuple[Reply, dict]:
    with trace("trainer.execute", cmd.request_id):
      effects, created = {}, False
      try:
        if cmd.op != Op.CREATE_MODEL and cmd.run not in self.model.params:
          if cmd.op == Op.DELETE_MODEL:
            return ok({"type": cmd.op, "model_id": cmd.run}), effects
          if not parked and cmd.op != Op.LOAD_WEIGHTS:
            raise KeyError(f"model {cmd.run} lost its unsaved steps when its trainer stopped; call load_state")
          if run is None:
            raise KeyError(f"no training run {cmd.run}")
          self.call_model("create_run", cmd.run, run["base"], run["rank"], run.get("lora", {}))
          created = True
          if parked:
            self.load_run(cmd.run, *parked, optimizer=True, gradients=True)
            effects["restored"] = parked[0]
            created = False  # recovery succeeded, even if the command that follows is rejected

        if cmd.op == Op.FORWARD_BACKWARD:
          req = wire.parse(cmd.body)
          fb = (cmd.run, wire.datums(req), req.loss_fn or "cross_entropy", wire.loss_config(req), req.forward_only)
          logprobs, metrics = self.call_model("forward_backward", *fb)
          return Reply(True, wire.PROTO, wire.output(logprobs, metrics)), effects

        args = json.loads(cmd.body)
        match cmd.op:
          case Op.CREATE_MODEL:
            self.call_model("create_run", cmd.run, args["base_model"], args["rank"], args["lora"])
          case Op.OPTIM_STEP:
            metrics = self.call_model("optim_step", cmd.run, args["adam_params"])
            return ok({"type": cmd.op, "metrics": metrics}), effects
          case Op.SAVE_WEIGHTS | Op.SAVE_WEIGHTS_FOR_SAMPLER:
            effects["saved"] = self.save(
              cmd.run,
              args["uri"],
              cmd.op == Op.SAVE_WEIGHTS_FOR_SAMPLER,
              args.get("ttl_seconds"),
              args.get("owned", False),
            )
          case Op.LOAD_WEIGHTS:
            self.load_run(cmd.run, args["uri"], version, optimizer=args["optimizer"], gradients=False)
          case Op.DELETE_MODEL:
            self.call_model("delete_run", cmd.run)
          case _:
            raise ValueError(f"unknown op {cmd.op}")

        reply = {"type": cmd.op, "model_id": cmd.run, **({"path": args["uri"]} if "uri" in args else {})}
        if "sampling_session_id" in args:
          reply["sampling_session_id"] = args["sampling_session_id"]
        return ok(reply), effects
      except Exception as e:
        if created:  # a failed restore must not leave a fresh model serving requests as recovered state
          self.call_model("delete_run", cmd.run)
        return Reply.error(e), effects

  def park_all(self) -> list[dict]:
    return [
      {"run": run, **self.save(run, f"tinker://{run}/parked/{time.time_ns()}", owned=True)}
      for run in list(self.model.params)
    ]

  def park(self) -> None:
    self.call_model("park_on_host")

  def resume(self, gpu: int | None = None) -> None:
    if gpu is not None and torch.cuda.is_available():
      torch.cuda.set_device(gpu)
    self.call_model("resume_on_gpu")


def main(path: str) -> None:
  from tinyrl.control.process import die_with_parent, serve

  die_with_parent()  # also for torchrun ranks, whose sessions are outside their launcher's process group
  if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
  if int(os.environ.get("WORLD_SIZE", 1)) > 1:
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", device_id=dev)
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
