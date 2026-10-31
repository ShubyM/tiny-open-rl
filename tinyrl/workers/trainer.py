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

  async def serve(self) -> None:
    await self.wait_for_lease()
    renewing = asyncio.create_task(self.keep_renewing_lease())
    try:
      await self.fail_stranded_commands(0)  # anything in flight now was taken by a trainer that died or lost the worker
      while self.stopping is None:
        await self.step()
        await self.fail_stranded_commands(STRANDED_MS)
      if self.stopping == Stop.PARK:
        await self.park_every_run()
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
    """Take the worker's lease again if it ran out with nobody taking it, as it does while Redis restarts. Not if
    another trainer has parked one of this trainer's runs since, which it only does after serving the worker."""
    if not await self.bus.take_lease(self.worker, self.owner):
      return False
    if any([await self.bus.parked_checkpoint(run) for run in self.model.params]):
      await self.bus.release_lease(self.worker, self.owner)
      return False
    return True

  def stop(self) -> None:
    """Finish the batch in hand, save every run for the worker's next trainer, and exit. SIGTERM calls it."""
    self.stopping = self.stopping or Stop.PARK

  async def park_every_run(self) -> None:
    for run in list(self.model.params):
      uri, (tensors, meta) = f"tinker://{run}/parked/{time.time_ns()}", self.model.training_state(run)
      await asyncio.to_thread(self.store.put, uri, tensors, meta)
      await self.registry.register_checkpoint(
        uri, sum(map(nbytes, tensors.values())), owned=True, version=self.store.versions.get(uri)
      )
      await self.bus.set_parked_checkpoint(run, uri, self.worker, self.owner)

  async def restore_missing_run(self, cmd: Cmd) -> None:
    """Bring back a run this trainer doesn't hold: from the state its last trainer saved when it stopped, or empty, for
    load_weights to fill. A run whose trainer died without saving it can't go on from where it was."""
    run, uri = await self.bus.get_run(cmd.run), await self.bus.parked_checkpoint(cmd.run)
    if not uri and cmd.op not in (Op.LOAD_WEIGHTS, Op.DELETE_MODEL):
      raise KeyError(
        f"model {cmd.run} lost its unsaved steps when its trainer stopped. load_state to go on from a checkpoint"
      )
    await asyncio.to_thread(self.model.create_run, cmd.run, run["base"], run["rank"], run.get("lora", {}))
    if uri:
      version = (await self.registry.get_checkpoint(uri) or {}).get("version")
      tensors, meta = await asyncio.to_thread(self.store.get, uri, version)
      await asyncio.to_thread(self.model.restore_run, cmd.run, tensors, meta, True)
      if await self.bus.set_parked_checkpoint(cmd.run, None, self.worker, self.owner):
        await self.registry.unregister_checkpoints(uri)  # the node daemons' sweeps delete it

  async def answer_command(self, cmd: Cmd) -> Reply:
    if cmd.op == Op.DELETE_MODEL and cmd.run not in self.model.params:  # nothing here to free, and its state not needed
      return ok({"type": cmd.op, "model_id": cmd.run})
    try:
      if cmd.op != Op.CREATE_MODEL and cmd.run not in self.model.params:
        await self.restore_missing_run(cmd)
    except Exception as e:
      return fail(e)
    version = None
    if cmd.op == Op.LOAD_WEIGHTS:  # the version saved last, which may be another node's, still uploading
      version = (await self.registry.get_checkpoint(json.loads(cmd.body)["uri"]) or {}).get("version")
    # off the event loop, so redis stays responsive
    reply, size = await asyncio.to_thread(self.run_command, cmd, version)
    if reply.ok and cmd.op in (Op.SAVE_WEIGHTS, Op.SAVE_WEIGHTS_FOR_SAMPLER):  # registered before the reply lands
      args = json.loads(cmd.body)
      saved = self.store.versions.get(args["uri"])
      await self.registry.register_checkpoint(
        args["uri"], size, args.get("ttl_seconds"), args.get("owned", False), saved
      )
    return reply

  async def fail_stranded_commands(self, min_idle_ms: int) -> None:
    for cmd in await self.bus.reclaim_stranded_commands(self.worker, self.owner, min_idle_ms):
      reply = fail(RuntimeError(f"its trainer died, or lost the worker, during {cmd.op}"))
      await self.bus.write_reply(cmd, reply, self.worker, self.owner)

  async def step(self, count: int = 16, block_ms: int = 1000) -> int:
    """Take the next commands from the worker and run them. Returns how many there were."""
    if self.stopping is not None:
      return 0
    take_block = min(block_ms, IDLE_MS) if self.turn.held and GPU else block_ms
    cmds = await self.bus.take_commands(self.worker, self.owner, count, take_block)
    if not cmds:
      await self.turn.report_idle()
      await self.yield_gpu()
      return 0
    queue: collections.deque[Cmd] = collections.deque()
    enqueue_cmds(queue, cmds, self.model.lora)
    processed = 0
    async with self.hold_gpu_for_batch(len(cmds)):
      batch_start = time.time()
      ops = []
      while queue:
        cmd = queue.popleft()
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
        if self.model.lora and queue and (cmd.op == Op.OPTIM_STEP or queue[0].run != cmd.run):
          if cmd.op == Op.OPTIM_STEP:
            await asyncio.sleep(0.015)
          more = await self.bus.take_commands(self.worker, self.owner, count, 0)
          enqueue_cmds(queue, more, True)
      await self.turn.report_batch_done(queued=len(queue), seconds=time.time() - batch_start, ops=ops)
    return processed

  @contextlib.asynccontextmanager
  async def hold_gpu_for_batch(self, queued: int):
    """Hold the GPU for one batch. Waits for this trainer's turn and brings its tensors back, then gives the turn back
    after the batch if the scheduler revoked it."""
    if await self.turn.wait_for_turn(queued):
      await asyncio.to_thread(self.model.resume_on_gpu)
    try:
      yield
    finally:
      await self.yield_gpu()

  async def yield_gpu(self) -> None:
    if await self.turn.was_revoked():
      await asyncio.to_thread(self.model.park_on_host)
      await self.turn.give_turn_back()

  def run_command(self, cmd: Cmd, version: str | None = None) -> tuple[Reply, int]:
    """Run one command on the model, loading `version` of a checkpoint. Returns the reply, which a failure becomes instead of an exception, and the bytes a save wrote."""
    model, size = self.model, 0
    try:
      if cmd.op == Op.FORWARD_BACKWARD:
        req = wire.parse(cmd.body)
        data, loss_fn, config = wire.datums(req), req.loss_fn or "cross_entropy", wire.loss_config(req)
        logprobs, metrics = model.forward_backward(cmd.run, data, loss_fn, config, req.forward_only)
        return Reply(True, wire.PROTO, wire.output(logprobs, metrics)), 0

      args = json.loads(cmd.body)
      match cmd.op:
        case Op.CREATE_MODEL:
          model.create_run(cmd.run, args["base_model"], args["rank"], args["lora"])
        case Op.OPTIM_STEP:
          return ok({"type": cmd.op, "metrics": model.optim_step(cmd.run, args["adam_params"])}), 0
        case Op.SAVE_WEIGHTS | Op.SAVE_WEIGHTS_FOR_SAMPLER:
          tensors, meta = model.training_state(cmd.run) if cmd.op == Op.SAVE_WEIGHTS else model.sampler_weights(cmd.run)
          self.store.put(args["uri"], tensors, meta)
          size = sum(map(nbytes, tensors.values()))
        case Op.LOAD_WEIGHTS:
          model.restore_run(cmd.run, *self.store.get(args["uri"], version), args["optimizer"])
        case Op.DELETE_MODEL:
          model.delete_run(cmd.run)
        case _:
          raise ValueError(f"unknown op {cmd.op}")

      reply = {"type": cmd.op, "model_id": cmd.run}
      if "uri" in args:
        reply["path"] = args["uri"]
      if "sampling_session_id" in args:
        reply["sampling_session_id"] = args["sampling_session_id"]
      return ok(reply), size
    except Exception as e:
      return fail(e), 0


async def main(worker: str) -> None:
  trainer = Trainer(Bus(), worker)
  asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, trainer.stop)
  await trainer.serve()


if __name__ == "__main__":
  asyncio.run(main(sys.argv[1]))
