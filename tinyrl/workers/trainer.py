"""A trainer serves one worker: the LoRA adapters that share a base model, or a single full fine-tune.

It takes commands from its worker's streams on the bus, runs each one on the model, and writes the reply back.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import enum
import json
import os
import signal
import socket
import sys
import time

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


def fail(error: Exception) -> Reply:
  # a request the trainer can't take, like bad input or an unknown model, is the user's error. Anything else is ours.
  category = "user" if isinstance(error, (ValueError, KeyError)) else "server"
  body = {"error": f"{type(error).__name__}: {error}", "category": category}
  return Reply(False, wire.JSON, json.dumps(body).encode())


def enqueue_cmds(queue: collections.deque[Cmd], cmds: list[Cmd], lora: bool) -> None:
  if not lora:
    queue.extend(cmds)
    return
  blocked = {x.run for x in queue}
  front, back = [], []
  for c in cmds:
    if c.op == Op.SAVE_WEIGHTS_FOR_SAMPLER and c.run not in blocked:
      front.append(c)
    else:
      blocked.add(c.run)
      back.append(c)
  queue.extendleft(reversed(front))
  queue.extend(back)


class Trainer:
  def __init__(self, bus: Bus, worker: str, model: Model | None = None, owner: str = "", store: Store | None = None):
    self.bus = bus
    self.registry = Registry(bus)
    self.worker = worker
    self.model = model if model is not None else Model()
    self.store = store if store is not None else Store()
    self.owner = owner or f"{socket.gethostname()}-{os.getpid()}-{os.urandom(3).hex()}"
    self.stopping: Stop | None = None
    self.turn = GpuTurn(bus, worker, bool(GPU))
    self.queue: collections.deque[Cmd] = collections.deque()

  async def serve(self) -> None:
    await self.wait_for_lease()
    renewing = asyncio.create_task(self.keep_renewing_lease())
    try:
      await self.fail_stranded_commands(0)  # anything in flight now was taken by a trainer that died or lost the worker
      while self.stopping is None:
        await self.step()
        if not self.queue:
          await self.fail_stranded_commands(STRANDED_MS)
      if self.stopping == Stop.PARK:
        await self.park_every_run()
        await self.bus.return_commands(self.worker, self.owner, list(self.queue))
        self.queue.clear()
    finally:
      renewing.cancel()
      await self.release_gpu_and_worker()

  async def release_gpu_and_worker(self) -> None:
    """Hand the GPU back with nothing of ours left on it, then the worker to its next trainer."""
    if self.turn.held and self.turn.shared:
      await asyncio.to_thread(self.model.park_on_host)
    await self.turn.give_turn_back()
    await self.bus.release_lease(self.worker, self.owner)

  async def wait_for_lease(self) -> None:
    """Wait until this trainer owns its worker. Only the owner's replies land, see Bus.write_reply."""
    while not await self.bus.take_lease(self.worker, self.owner):
      await asyncio.sleep(LEASE_MS / 4000)

  async def keep_renewing_lease(self) -> None:
    """Renew the worker's lease until another trainer has it."""
    while True:  # until serve() cancels it, so the lease holds while a stopping trainer saves its runs
      await asyncio.sleep(LEASE_MS / 4000)
      if not await self.bus.renew_lease(self.worker, self.owner) and not await self.retake_lost_lease():
        self.stopping = Stop.EXIT
        return

  async def retake_lost_lease(self) -> bool:
    """Keep live state only if nobody else has owned this worker since our lease expired."""
    return await self.bus.take_lease(self.worker, self.owner, recover=True)

  def stop(self) -> None:
    """Finish the current command, save every run, and return unstarted commands. SIGTERM calls it."""
    self.stopping = self.stopping or Stop.PARK

  async def park_every_run(self) -> None:
    for run in list(self.model.params):
      uri, (tensors, meta) = f"tinker://{run}/parked/{time.time_ns()}", self.model.training_state(run)
      version = await asyncio.to_thread(self.store.put, uri, tensors, meta)
      await self.bus.set_parked_checkpoint(
        run,
        uri,
        self.worker,
        self.owner,
        checkpoint={"uri": uri, "size": sum(map(nbytes, tensors.values())), "owned": True, "version": version},
      )

  async def load_run(self, cmd: Cmd) -> None:
    """Fetch a whole checkpoint before installing a missing run. Explicit loads reset accumulation; handoffs preserve it."""
    run, parked = await self.bus.get_run(cmd.run), await self.bus.parked_checkpoint(cmd.run)
    args = json.loads(cmd.body) if cmd.op == Op.LOAD_WEIGHTS else {"uri": parked, "optimizer": True}
    if not args["uri"]:
      raise KeyError(
        f"model {cmd.run} lost its unsaved steps when its trainer stopped. load_state to go on from a checkpoint"
      )
    version = (await self.registry.get_checkpoint(args["uri"]) or {}).get("version")
    tensors, meta = await asyncio.to_thread(self.store.get, args["uri"], version)
    created = cmd.run not in self.model.params
    if created:
      await asyncio.to_thread(self.model.create_run, cmd.run, run["base"], run["rank"], run.get("lora", {}))
    try:
      await asyncio.to_thread(
        self.model.restore_run, cmd.run, tensors, meta, args["optimizer"], cmd.op != Op.LOAD_WEIGHTS
      )
    except Exception:
      if created and cmd.run in self.model.params:
        await asyncio.to_thread(self.model.delete_run, cmd.run)
      raise
    if parked:
      if await self.bus.set_parked_checkpoint(cmd.run, None, self.worker, self.owner):
        await self.registry.unregister_checkpoints(parked)  # the node daemons' sweeps delete it

  async def answer_command(self, cmd: Cmd) -> tuple[Reply, dict | None]:
    """Execute one accepted command; publish a save's version with its lease-fenced reply."""
    model, saved = self.model, None
    try:
      if cmd.op not in (Op.CREATE_MODEL, Op.LOAD_WEIGHTS, Op.DELETE_MODEL) and cmd.run not in model.params:
        await self.load_run(cmd)
      if cmd.op == Op.FORWARD_BACKWARD:
        req = wire.parse(cmd.body)
        logprobs, metrics = await asyncio.to_thread(
          model.forward_backward,
          cmd.run,
          wire.datums(req),
          req.loss_fn or "cross_entropy",
          wire.loss_config(req),
          req.forward_only,
        )
        return Reply(True, wire.PROTO, wire.output(logprobs, metrics)), None
      args = json.loads(cmd.body)
      reply = {"type": cmd.op, "model_id": cmd.run}
      match cmd.op:
        case Op.CREATE_MODEL:
          await asyncio.to_thread(model.create_run, cmd.run, args["base_model"], args["rank"], args["lora"])
        case Op.OPTIM_STEP:
          reply["metrics"] = await asyncio.to_thread(model.optim_step, cmd.run, args["adam_params"])
        case Op.SAVE_WEIGHTS | Op.SAVE_WEIGHTS_FOR_SAMPLER:
          tensors, meta = await asyncio.to_thread(
            model.training_state if cmd.op == Op.SAVE_WEIGHTS else model.sampler_weights, cmd.run
          )
          version = await asyncio.to_thread(self.store.put, args["uri"], tensors, meta)
          saved = {
            "uri": args["uri"],
            "size": sum(map(nbytes, tensors.values())),
            "ttl": args.get("ttl_seconds"),
            "owned": args.get("owned", False),
            "version": version,
          }
        case Op.LOAD_WEIGHTS:
          await self.load_run(cmd)
        case Op.DELETE_MODEL:
          if cmd.run in model.params:
            await asyncio.to_thread(model.delete_run, cmd.run)
        case _:
          raise ValueError(f"unknown op {cmd.op}")
      if "uri" in args:
        reply["path"] = args["uri"]
      if "sampling_session_id" in args:
        reply["sampling_session_id"] = args["sampling_session_id"]
      return ok(reply), saved
    except Exception as e:
      return fail(e), None

  async def fail_stranded_commands(self, min_idle_ms: int) -> None:
    for cmd in await self.bus.reclaim_stranded_commands(self.worker, self.owner, min_idle_ms):
      reply = fail(RuntimeError(f"its trainer died, or lost the worker, during {cmd.op}"))
      await self.bus.write_reply(cmd, reply, self.worker, self.owner)

  async def step(self, count: int = 16, block_ms: int = 1000) -> int:
    """Take the next commands from the worker and run them. Returns how many there were."""
    if self.stopping is not None:
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
    waiting = asyncio.create_task(self.turn.wait_for_turn(len(self.queue)))
    try:
      while self.stopping is None and not waiting.done():
        await asyncio.wait([waiting], timeout=0.1)
      if self.stopping is not None:
        return 0
      if await waiting:
        await asyncio.to_thread(self.model.resume_on_gpu)
      batch_start = time.time()
      ops = []
      while self.queue and self.stopping is None:
        if await self.yield_gpu():
          break
        cmd = self.queue.popleft()
        start = time.time()
        reply, checkpoint = await self.answer_command(cmd)
        ms = (time.time() - start) * 1000
        ops.append([cmd.op, ms])
        if not await self.bus.write_reply(cmd, reply, self.worker, self.owner, started=start, checkpoint=checkpoint):
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
        if self.stopping is not None:
          break
        if self.model.lora and self.queue and (cmd.op == Op.OPTIM_STEP or self.queue[0].run != cmd.run):
          if cmd.op == Op.OPTIM_STEP:
            await asyncio.sleep(0.015)
          more = await self.bus.take_commands(self.worker, self.owner, count, 0)
          enqueue_cmds(self.queue, more, True)
      await self.turn.report_batch_done(queued=len(self.queue), seconds=time.time() - batch_start, ops=ops)
    finally:
      waiting.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await waiting
      await self.yield_gpu()
    return processed

  async def yield_gpu(self) -> bool:
    if await self.turn.was_revoked():
      await asyncio.to_thread(self.model.park_on_host)
      await self.turn.give_turn_back()
      return True
    return False


async def main(worker: str) -> None:
  trainer = Trainer(Bus(), worker)
  asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, trainer.stop)
  await trainer.serve()


if __name__ == "__main__":
  asyncio.run(main(sys.argv[1]))
