import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from test.helpers import RedisTest
from tinyrl.bus import Op, Reply, Role, Run
from tinyrl.control import scheduler
from tinyrl.control.node_daemon import SamplerLife, TrainerLife


class TestAccounting(RedisTest):
  async def lifetime(self, role, **accounts):
    runs = list(accounts)
    for run, account in accounts.items():
      await self.bus.create_run(Run(run, "base", 4, role=role, account=account, gpus=2))
    state = scheduler.from_state(await self.bus.read_state())
    state.procs["worker"] = scheduler.Proc(state.jobs[runs[0]].spec)
    if role == Role.TRAINER:
      state.home = dict.fromkeys(runs, "worker")
    node = SimpleNamespace(id="node", bus=self.bus, store=None, state=state, notify=Mock())
    life = (SamplerLife if role == Role.SAMPLER else TrainerLife)(node, "worker")
    await self.bus.touch_leases(life.life)
    if role == Role.TRAINER:
      for run in runs:
        life.owned[run] = await self.bus.acquire_run(run, life.life)
    return life

  async def execute(self, life):
    await life.collect()
    if isinstance(life, SamplerLife):
      await asyncio.gather(*life.staged.values())
    ready, _ = life.ready()
    return await life.dispatch(ready)

  async def test_sampler_shares_busy_time_until_the_short_request_fails(self):
    life = await self.lifetime(Role.SAMPLER, short="alice", long="alice")
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
    expected = {"alice": 60}
    self.assertEqual(life.node.state.service, expected)
    self.assertEqual((await self.bus.read_state())["service"], expected)
    short, long = await self.bus.get_run("short"), await self.bus.get_run("long")
    self.assertEqual((short["busy_ms"], short["failed"], short["tokens"]), (10, 1, 0))
    self.assertEqual((long["busy_ms"], long["failed"], long["tokens"]), (30, 0, 999))

  async def test_three_sampler_sessions_get_the_same_gpu_time_as_one(self):
    accounts = {"a1": "alice", "a2": "alice", "a3": "alice", "b": "bob"}
    life = await self.lifetime(Role.SAMPLER, **accounts)
    for run in accounts:
      for _ in range(6):
        await self.bus.submit_command(run, Op.SAMPLE, b"{}")

    now = 0

    async def sample(cmd, *_):
      nonlocal now
      await asyncio.sleep(0)  # All calls in this wave overlap for 12 ms.
      now = end
      return Reply(True, "application/json", b"{}")

    life.model = SimpleNamespace(execute=sample)
    with (
      patch("tinyrl.control.node_daemon.SAMPLE_COMMANDS", 1),
      patch("tinyrl.control.accounting.monotonic_ns", lambda: now),
    ):
      for _ in range(6):
        end = now + 12_000_000
        await self.execute(life)

    self.assertEqual((await self.bus.read_state())["service"], {"alice": 72, "bob": 72})
    self.assertEqual(life.service, {"a1": 24, "a2": 24, "a3": 24, "b": 72})

  async def test_trainer_bills_only_the_failure_that_entered_the_model(self):
    life = await self.lifetime(Role.TRAINER, run="run-account")
    failure = ValueError("bad input")
    life.model = SimpleNamespace(call=AsyncMock(return_value=(Reply.error(failure), {})))
    await self.bus.submit_command("run", Op.LOAD_WEIGHTS, b'{"uri":"tinker://run/weights/missing"}')
    await self.execute(life)
    life.model.call.assert_not_awaited()
    self.assertEqual(life.gpu_ms, 0)
    self.assertEqual((await self.bus.read_state())["service"], {})
    await self.bus.submit_command("run", Op.OPTIM_STEP, b"{}")
    with patch("tinyrl.control.accounting.monotonic_ns", side_effect=[0, 250_000]):
      await self.execute(life)
    self.assertEqual(life.service, {"run": 0.5})
    self.assertEqual(life.gpu_ms, 0.5)
    self.assertEqual(life.node.state.service, {"run-account": 0.5})
    self.assertEqual((await self.bus.read_state())["service"], life.node.state.service)
    run = await self.bus.get_run("run")
    self.assertEqual((run["cmds"], run["failed"], run["steps"], run["tokens"]), (2, 2, 0, 0))
