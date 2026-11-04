"""A trainer serves one worker: the LoRA adapters that share a base model, or a single full fine-tune.

It takes commands from its worker's streams on the bus, runs each one on the model, and writes the reply back.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import contextlib
import enum
import json
import os
import signal
import socket
import sys
import time

import torch
import torch.distributed as dist

from tinyrl.api import wire
from tinyrl.bus import LEASE_MS, Bus, Cmd, Op, Reply
from tinyrl.control.turns import IDLE_MS, GpuTurn
from tinyrl.helpers import DEBUG, getenv
from tinyrl.storage.registry import Registry
from tinyrl.storage.store import Store, nbytes
from tinyrl.workers.model import Model

STRANDED_MS = 5000  # a command in flight this long while its worker's owner has nothing in flight was stranded
GPU = getenv("GPU", "")  # the GPU this trainer takes turns on, set by the node daemon. Unset, it never takes turns


# ***** trainer *****


class Stop(enum.Enum):
  EXIT = "exit"  # leave at once: another trainer has the worker, or a full fine-tune's run was deleted
  PARK = "park"  # save every run for the worker's next trainer, then leave. What SIGTERM asks for


def ok(body: dict) -> Reply:
  return Reply(True, wire.JSON, json.dumps(body).encode())


def enqueue_cmds(queue: collections.deque[Cmd], cmds: list[Cmd], lora: bool) -> None:
  if not lora:
    queue.extend(cmds)
    return
  blocked, front, back = {x.run for x in queue}, [], []
  for c in cmds:
    (front if c.op == Op.SAVE_WEIGHTS_FOR_SAMPLER and c.run not in blocked else back).append(c)
    if c.op != Op.SAVE_WEIGHTS_FOR_SAMPLER:
      blocked.add(c.run)
  queue.extendleft(reversed(front))
  queue.extend(back)


def serve_follower(model: Model) -> None:
  while True:
    dist.broadcast_object_list(msg := [None], src=0)
    if msg[0][0] == "stop":
      return
    with contextlib.suppress(ValueError, KeyError):  # rejected the same way on every rank, before any collective
      getattr(model, msg[0][0])(*msg[0][1])


class Trainer:
  def __init__(self, bus: Bus, worker: str, model: Model | None = None, owner: str = "", store: Store | None = None):
    self.bus, self.registry, self.worker = bus, Registry(bus), worker
    self.model = model if model is not None else Model()
    self.store = store if store is not None else Store()
    self.owner = owner or f"{socket.gethostname()}-{os.getpid()}-{os.urandom(3).hex()}"
    self.stopping: Stop | None = None
    self.turn = GpuTurn(bus, worker, bool(GPU))
    self.queue: collections.deque[Cmd] = collections.deque()
    self.revoked_turn, self.world_size = False, int(os.environ.get("WORLD_SIZE", 1))

  def call_model(self, method: str, *args):
    if self.world_size > 1:
      dist.broadcast_object_list([(method, args)], src=0)
    try:
      return None if method == "stop" else getattr(self.model, method)(*args)
    except Exception as e:
      # the other ranks may be waiting in a collective this rank left, so torchrun restarts them all
      if self.world_size > 1 and not isinstance(e, (ValueError, KeyError)):
        os._exit(1)
      raise

  async def serve(self) -> None:
    await self.wait_for_lease()
    renewing = asyncio.create_task(self.keep_renewing_lease())
    try:
      # anything in flight now was taken by a trainer that died or lost the worker
      await self.bus.fail_stranded_commands(self.worker, self.owner, 0)
      while self.stopping is None:
        await self.step()
        if not self.queue:
          await self.bus.fail_stranded_commands(self.worker, self.owner, STRANDED_MS)
      if self.stopping == Stop.PARK:
        await self.park_every_run()
    finally:
      renewing.cancel()
      try:
        await self.release_gpu_and_worker()
      finally:
        await asyncio.to_thread(self.call_model, "stop")

  async def release_gpu_and_worker(self) -> None:
    """Hand the GPU back with nothing of ours left on it, then the worker to its next trainer."""
    if self.turn.held and self.turn.shared:
      await asyncio.to_thread(self.call_model, "park_on_host")
    await self.turn.give_turn_back()
    await self.bus.release_lease(self.worker, self.owner)

  async def wait_for_lease(self) -> None:
    """Wait until this trainer owns its worker. Only the owner's replies land, see Bus.write_reply."""
    while not await self.bus.take_lease(self.worker, self.owner):
      await asyncio.sleep(LEASE_MS / 4000)

  async def keep_renewing_lease(self) -> None:
    """Renew the worker's lease until another trainer has it. A lease that ran out with nobody serving the worker since, as while Redis restarts, is taken again."""
    while True:  # until serve() cancels it, so the lease holds while a stopping trainer saves its runs
      await asyncio.sleep(LEASE_MS / 4000)
      if not await self.bus.renew_lease(self.worker, self.owner) and not await self.bus.take_lease(
        self.worker, self.owner, recover=True
      ):
        self.stopping = Stop.EXIT
        return

  def stop(self) -> None:
    """Finish the batch in hand, save every run for the worker's next trainer, and exit. SIGTERM calls it."""
    self.stopping = self.stopping or Stop.PARK

  async def park_every_run(self) -> None:
    for run in list(self.model.params):
      uri = f"tinker://{run}/parked/{time.time_ns()}"
      tensors, meta = await asyncio.to_thread(self.call_model, "training_state", run)
      await asyncio.to_thread(self.store.put, uri, tensors, meta)
      ver = self.store.versions.get(uri)
      await self.registry.register_checkpoint(uri, sum(map(nbytes, tensors.values())), owned=True, version=ver)
      await self.bus.set_parked_checkpoint(run, uri, self.worker, self.owner)

  async def load_run(self, run_id: str, uri: str, optimizer: bool, gradients: bool) -> None:
    version = (await self.registry.get_checkpoint(uri) or {}).get("version")
    tensors, meta = await asyncio.to_thread(self.store.get, uri, version)
    await asyncio.to_thread(self.call_model, "restore_run", run_id, tensors, meta, optimizer, gradients)

  async def answer_command(self, cmd: Cmd) -> Reply:
    try:
      if cmd.op != Op.CREATE_MODEL and cmd.run not in self.model.params:
        if cmd.op == Op.DELETE_MODEL:
          return ok({"type": cmd.op, "model_id": cmd.run})
        run, parked = await self.bus.get_run(cmd.run), await self.bus.parked_checkpoint(cmd.run)
        if not parked and cmd.op != Op.LOAD_WEIGHTS:
          raise KeyError(f"model {cmd.run} lost its unsaved steps when its trainer stopped; call load_state")
        await asyncio.to_thread(self.call_model, "create_run", cmd.run, run["base"], run["rank"], run.get("lora", {}))
        if parked:
          try:
            await self.load_run(cmd.run, parked, optimizer=True, gradients=True)
          except Exception:
            await asyncio.to_thread(self.call_model, "delete_run", cmd.run)
            raise
          if await self.bus.set_parked_checkpoint(cmd.run, None, self.worker, self.owner):
            await self.registry.unregister_checkpoints(parked)  # the node daemons' sweeps delete it

      if cmd.op == Op.FORWARD_BACKWARD:
        req = await asyncio.to_thread(wire.parse, cmd.body)
        data, config = await asyncio.to_thread(wire.datums, req), wire.loss_config(req)
        fb = (cmd.run, data, req.loss_fn or "cross_entropy", config, req.forward_only)
        logprobs, metrics = await asyncio.to_thread(self.call_model, "forward_backward", *fb)
        return Reply(True, wire.PROTO, await asyncio.to_thread(wire.output, logprobs, metrics))

      args = json.loads(cmd.body)
      match cmd.op:
        case Op.CREATE_MODEL:
          create = (cmd.run, args["base_model"], args["rank"], args["lora"])
          await asyncio.to_thread(self.call_model, "create_run", *create)
        case Op.OPTIM_STEP:
          metrics = await asyncio.to_thread(self.call_model, "optim_step", cmd.run, args["adam_params"])
          return ok({"type": cmd.op, "metrics": metrics})
        case Op.SAVE_WEIGHTS | Op.SAVE_WEIGHTS_FOR_SAMPLER:
          fn = "training_state" if cmd.op == Op.SAVE_WEIGHTS else "sampler_weights"
          tensors, meta = await asyncio.to_thread(self.call_model, fn, cmd.run)
          await asyncio.to_thread(self.store.put, args["uri"], tensors, meta)
          size, ver = sum(map(nbytes, tensors.values())), self.store.versions.get(args["uri"])
          await self.registry.register_checkpoint(
            args["uri"], size, args.get("ttl_seconds"), args.get("owned", False), ver
          )
        case Op.LOAD_WEIGHTS:
          await self.load_run(cmd.run, args["uri"], optimizer=args["optimizer"], gradients=False)
        case Op.DELETE_MODEL:
          if cmd.run in self.model.params:
            await asyncio.to_thread(self.call_model, "delete_run", cmd.run)
        case _:
          raise ValueError(f"unknown op {cmd.op}")

      reply = {"type": cmd.op, "model_id": cmd.run, **({"path": args["uri"]} if "uri" in args else {})}
      if "sampling_session_id" in args:
        reply["sampling_session_id"] = args["sampling_session_id"]
      return ok(reply)
    except Exception as e:
      return Reply.error(e)

  async def step(self, count: int = 16, block_ms: int = 1000) -> int:
    """Take the next commands from the worker and run them. Returns how many there were."""
    if self.stopping is not None and not self.queue:
      return 0
    if not self.queue:
      take_block = min(block_ms, IDLE_MS) if self.turn.held and GPU else block_ms
      cmds = await self.bus.take_commands(self.worker, self.owner, count, take_block)
      if not cmds:
        await self.turn.report_idle()
        await self.yield_gpu()
        return 0
      enqueue_cmds(self.queue, cmds, self.model.lora)
    processed = 0
    async with self.hold_gpu_for_batch(len(self.queue)):
      batch_start = time.time()
      ops = []
      while self.queue:
        self.revoked_turn = self.revoked_turn or await self.turn.was_revoked()
        if self.revoked_turn:
          break
        cmd = self.queue.popleft()
        start = time.time()
        # no lock needed, the worker's owner is the only writer of its seq keys
        first = await self.bus.reply_to_repeated_command(cmd)
        reply = first or await self.answer_command(cmd)
        ms = (time.time() - start) * 1000
        ops.append([cmd.op, ms])
        if not await self.bus.write_reply(cmd, reply, self.worker, self.owner, repeat=first is not None, started=start):
          self.stopping = Stop.EXIT  # another trainer owns the worker now
          break

        if cmd.op == Op.DELETE_MODEL and reply.ok:
          await self.bus.delete_run(cmd.run, self.worker)
          await self.registry.unregister_checkpoints(*await self.registry.checkpoints_owned_by_run(cmd.run))
          if not self.model.lora:  # a full fine-tune's trainer exits with its run
            self.stopping = Stop.EXIT

        if DEBUG:
          error = "" if reply.ok else f"  {reply.body.decode()}"
          print(f"{self.worker} {cmd.request_id} {cmd.op:24s} {ms:9.2f} ms{error}")

        processed += 1
        if self.model.lora and self.queue and (cmd.op == Op.OPTIM_STEP or self.queue[0].run != cmd.run):
          if self.stopping is None and not self.revoked_turn:
            if cmd.op == Op.OPTIM_STEP:
              await asyncio.sleep(0.015)
            enqueue_cmds(self.queue, await self.bus.take_commands(self.worker, self.owner, count, 0), True)
      await self.turn.report_batch_done(queued=len(self.queue), seconds=time.time() - batch_start, ops=ops)
    return processed

  @contextlib.asynccontextmanager
  async def hold_gpu_for_batch(self, queued: int):
    """Hold the GPU for one batch. Waits for this trainer's turn and brings its tensors back, then gives the turn back after the batch if the scheduler revoked it."""
    if await self.turn.wait_for_turn(queued):
      if self.turn.gpu is not None:
        await asyncio.to_thread(torch.cuda.set_device, self.turn.gpu)
      await asyncio.to_thread(self.call_model, "resume_on_gpu")
    try:
      yield
    finally:
      await self.yield_gpu()

  async def yield_gpu(self) -> None:
    revoked = self.revoked_turn or await self.turn.was_revoked()
    self.revoked_turn = False
    if revoked:
      await asyncio.to_thread(self.call_model, "park_on_host")
      await self.turn.give_turn_back()


async def main(worker: str) -> None:
  if torch.cuda.is_available() and ("LOCAL_RANK" in os.environ or "TINYRL_NODE" in os.environ):
    gpu = int(os.environ.get("LOCAL_RANK") or GPU.rsplit(":", 1)[1])
    torch.cuda.set_device(gpu)
    pool = concurrent.futures.ThreadPoolExecutor(1, initializer=torch.cuda.set_device, initargs=(gpu,))
    asyncio.get_running_loop().set_default_executor(pool)
  if int(os.environ.get("WORLD_SIZE", 1)) > 1:
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", device_id=dev)
  try:
    if dist.is_initialized() and dist.get_rank() > 0:
      serve_follower(Model())
      return
    trainer = Trainer(Bus(), worker)
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, trainer.stop)
    await trainer.serve()
  finally:
    if dist.is_initialized():
      dist.destroy_process_group()


if __name__ == "__main__":
  asyncio.run(main(sys.argv[1]))
