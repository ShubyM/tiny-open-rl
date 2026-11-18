import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from test.helpers import RedisTest
from tinyrl.bus import Op, Reply, Role, Run
from tinyrl.control import scheduler
from tinyrl.control.node_daemon import Lifetime


class TestAccounting(RedisTest):
  async def lifetime(self, role, *runs):
    for run in runs:
      await self.bus.create_run(Run(run, "base", 4, role=role, account=f"{run}-account", gpus=2))
    state = scheduler.from_state(await self.bus.read_state())
    state.procs["worker"] = scheduler.Proc(state.jobs[runs[0]].spec)
    state.home = dict.fromkeys(runs, ["worker"])
    node = SimpleNamespace(id="node", bus=self.bus, registry=self.registry, store=None, state=state, notify=Mock())
    life = Lifetime(node, "worker")
    await self.bus.open_lease(life.life)
    if role == Role.TRAINER:
      for run in runs:
        life.owned[run] = await self.bus.acquire_run(run, life.life)
    self.addAsyncCleanup(life.stop_reader)
    return life

  async def execute(self, life):
    life.heads = await life.peek(0)
    await asyncio.gather(*(task for _, task in life.prepared.values()))
    return await life.execute(life.heads)

  async def test_sampler_shares_busy_time_until_the_short_request_fails(self):
    life = await self.lifetime(Role.SAMPLER, "short", "long")
    finished = asyncio.Event()

    async def sample(cmd, *_):
      if cmd.run == "short":
        await asyncio.sleep(0)  # both calls have started
        finished.set()
        return Reply.error(ValueError("sampling failed"))
      await finished.wait()
      return Reply(True, "application/json", b"{}")

    life.model = SimpleNamespace(execute=sample)
    for run in ("short", "long"):
      await self.bus.submit_command(run, Op.SAMPLE, b"{}", cost=999)
    with patch("tinyrl.control.accounting.monotonic_ns", side_effect=[0, 0, 10_000_000, 30_000_000]):
      await self.execute(life)
    self.assertEqual(life.service, {"short": 10, "long": 50})
    self.assertEqual(life.gpu_ms, 60)
    expected = {"short-account": 10, "long-account": 50}
    self.assertEqual(life.node.state.service, expected)
    self.assertEqual((await self.bus.read_state())["service"], expected)
    short, long = await self.bus.get_run("short"), await self.bus.get_run("long")
    self.assertEqual((short["busy_ms"], short["failed"], short["tokens"]), (10, 1, 0))
    self.assertEqual((long["busy_ms"], long["failed"], long["tokens"]), (30, 0, 999))

  async def test_trainer_bills_only_the_failure_that_entered_the_model(self):
    life = await self.lifetime(Role.TRAINER, "run")
    failure = ValueError("bad input")
    life.prepare = AsyncMock(return_value=failure)
    life.model = SimpleNamespace(call=AsyncMock(return_value=(Reply.error(failure), {})))
    await self.bus.submit_command("run", Op.OPTIM_STEP, b"{}")
    await self.execute(life)
    life.model.call.assert_not_awaited()
    self.assertEqual(life.gpu_ms, 0)
    self.assertEqual((await self.bus.read_state())["service"], {})
    life.prepare.return_value = ({}, None)
    await self.bus.submit_command("run", Op.OPTIM_STEP, b"{}")
    with patch("tinyrl.control.accounting.monotonic_ns", side_effect=[0, 250_000]):
      await self.execute(life)
    self.assertEqual(life.service, {"run": 0.5})
    self.assertEqual(life.gpu_ms, 0.5)
    self.assertEqual(life.node.state.service, {"run-account": 0.5})
    self.assertEqual((await self.bus.read_state())["service"], life.node.state.service)
    run = await self.bus.get_run("run")
    self.assertEqual((run["cmds"], run["failed"], run["steps"], run["tokens"]), (2, 2, 0, 0))
