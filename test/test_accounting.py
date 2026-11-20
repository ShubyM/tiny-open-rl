import asyncio
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from test.execution import TestNode
from test.helpers import BusTest
from tinyrl.bus import Op, Reply, Role, Run
from tinyrl.control.node_daemon import WorkerHandle
from tinyrl.storage.store import Store


class TestAccounting(BusTest):
  async def setup_node(self, role, **accounts):
    folder = tempfile.TemporaryDirectory()
    self.addCleanup(folder.cleanup)
    store = Store(root=folder.name)
    runs = list(accounts)
    for run, account in accounts.items():
      await self.bus.create_run(Run(run, "base", 4, role=role, account=account, gpus=2))
    node = TestNode(self.bus, store=store, gpus=2, node="node")
    self.addAsyncCleanup(node.aclose)
    await node.bind("worker", *runs)
    if node.active:
      await asyncio.gather(*(t for _, t in node.active.values()))
      await node.tick()
    spec = self.bus.plan_state.procs["worker"].spec
    return node, spec

  async def step_turn(self, node: TestNode):
    reply = await node.tick()
    if node.active:
      await asyncio.gather(*(t for _, t in node.active.values()))
    await node.tick()
    return reply

  async def test_sampler_shares_busy_time_until_the_short_request_fails(self):
    node, spec = await self.setup_node(Role.SAMPLER, short="alice", long="alice")
    finished = asyncio.Event()

    async def sample(cmd, *_):
      if cmd.run == "short":
        await asyncio.sleep(0)  # both calls have started
        finished.set()
        return Reply.error(ValueError("sampling failed"))
      await finished.wait()
      return Reply(True, "application/json", b"{}")

    model = SimpleNamespace(execute=sample, resume=AsyncMock(), park=AsyncMock(), close=AsyncMock(), proc=None)
    node.workers["worker"] = WorkerHandle("worker", spec, model, resident=(0, 1), bound=(0, 1))
    for run in ("short", "long"):
      await self.bus.submit_command(run, Op.SAMPLE, b"{}", cost=999)
    with patch("tinyrl.control.accounting.monotonic_ns", side_effect=[0, 0, 10_000_000, 30_000_000]):
      await self.step_turn(node)
    w = self.bus.nodes["node"].workers["worker"]
    self.assertEqual(w.run_service, {"short": 10, "long": 50})
    self.assertEqual(w.gpu_ms, 60)
    self.assertEqual((await self.bus.read_state())["service"], {"alice": 60})
    short, long = await self.bus.get_run("short"), await self.bus.get_run("long")
    self.assertEqual((short["busy_ms"], short["failed"], short["tokens"]), (10, 1, 0))
    self.assertEqual((long["busy_ms"], long["failed"], long["tokens"]), (30, 0, 999))

  async def test_checkpoint_load_is_charged_once_across_requests_even_if_it_fails(self):
    node, spec = await self.setup_node(Role.SAMPLER, a1="alice", a2="alice", b="bob")
    now, uri = 0, "tinker://trainer/sampler_weights/shared"
    await self.seed_checkpoint(uri, 100, version="version")
    for run in ("a1", "a2"):
      self.bus.runs[run]["record"]["model_path"] = uri
    cache_dir = tempfile.TemporaryDirectory()
    self.addCleanup(cache_dir.cleanup)
    node.store.local_folder = Mock(return_value=(cache_dir.name, "version"))

    async def load(_):
      nonlocal now
      now += 10_000_000
      if model.load.await_count == 2:
        raise ValueError("bad weights")

    async def sample(*_):
      return Reply(True, "application/json", b"{}")

    model = SimpleNamespace(
      load=AsyncMock(side_effect=load),
      execute=AsyncMock(side_effect=sample),
      resume=AsyncMock(),
      park=AsyncMock(),
      close=AsyncMock(),
      proc=None,
    )
    node.workers["worker"] = WorkerHandle("worker", spec, model, resident=(0, 1), bound=(0, 1))
    with patch("tinyrl.control.accounting.monotonic_ns", lambda: now):
      for _ in range(2):
        for run in ("a1", "a1", "a2", "b"):
          await self.bus.submit_command(run, Op.SAMPLE, b"{}")
        # First tick prefetches version off-GPU; wait for prefetch then run waves for alice & bob.
        await node.tick()
        if node.staging:
          await asyncio.gather(*node.staging.values())
        await self.step_turn(node)
        await self.step_turn(node)
    self.assertEqual(model.load.await_count, 2)
    self.assertEqual(model.execute.await_count, 5)
    w = self.bus.nodes["node"].workers["worker"]
    self.assertAlmostEqual(w.run_service["a1"], 80 / 3)
    self.assertAlmostEqual(w.run_service["a2"], 40 / 3)
    self.assertAlmostEqual((await self.bus.read_state())["service"]["alice"], 40)
    self.assertEqual((await self.bus.get_run("a1"))["failed"], 2)

  async def test_three_sampler_sessions_get_the_same_gpu_time_as_one(self):
    accounts = {"a1": "alice", "a2": "alice", "a3": "alice", "b": "bob"}
    node, spec = await self.setup_node(Role.SAMPLER, **accounts)
    for run in accounts:
      for _ in range(6):
        await self.bus.submit_command(run, Op.SAMPLE, b"{}")

    now = 0

    async def sample(cmd, *_):
      nonlocal now
      await asyncio.sleep(0)  # All calls in this wave overlap for 12 ms.
      now = end
      return Reply(True, "application/json", b"{}")

    model = SimpleNamespace(execute=sample, resume=AsyncMock(), park=AsyncMock(), close=AsyncMock(), proc=None)
    node.workers["worker"] = WorkerHandle("worker", spec, model, resident=(0, 1), bound=(0, 1))
    with (
      patch("tinyrl.bus.SAMPLE_COMMANDS", 1),
      patch("tinyrl.control.accounting.monotonic_ns", lambda: now),
    ):
      for _ in range(6):
        end = now + 12_000_000
        await self.step_turn(node)

    self.assertEqual((await self.bus.read_state())["service"], {"alice": 72, "bob": 72})
    w = self.bus.nodes["node"].workers["worker"]
    self.assertEqual(w.run_service, {"a1": 24, "a2": 24, "a3": 24, "b": 72})

  async def test_backlogged_training_runs_of_one_account_alternate_equal_cost_steps(self):
    node, spec = await self.setup_node(Role.TRAINER, first="account", second="account")
    order = []

    async def execute(method, cmd, *args):
      order.append(cmd.run)
      return Reply(True, "application/json", b"{}"), {}

    model = SimpleNamespace(call=execute, resume=AsyncMock(), park=AsyncMock(), close=AsyncMock(), proc=None)
    node.workers["worker"] = WorkerHandle("worker", spec, model, resident=(0, 1), bound=(0, 1))
    for run in ("first", "second"):  # All "first" heads arrive before all "second" heads.
      for _ in range(4):
        await self.bus.submit_command(run, Op.OPTIM_STEP, b"{}")
      await asyncio.sleep(0.002)
    with patch("tinyrl.control.accounting.monotonic_ns", side_effect=range(0, 16_000_000, 1_000_000)):
      for _ in range(8):
        await self.step_turn(node)
    self.assertEqual(order, ["first", "second"] * 4)

  async def test_trainer_bills_only_the_failure_that_entered_the_model(self):
    node, spec = await self.setup_node(Role.TRAINER, run="run-account")
    failure = ValueError("bad input")
    model = SimpleNamespace(
      call=AsyncMock(return_value=(Reply.error(failure), {})),
      resume=AsyncMock(),
      park=AsyncMock(),
      close=AsyncMock(),
      proc=None,
    )
    node.workers["worker"] = WorkerHandle("worker", spec, model, resident=(0, 1), bound=(0, 1))
    await self.bus.submit_command("run", Op.LOAD_WEIGHTS, b'{"uri":"tinker://run/weights/missing"}')
    await self.step_turn(node)
    model.call.assert_not_awaited()
    w = self.bus.nodes["node"].workers["worker"]
    self.assertEqual(w.gpu_ms, 0)
    self.assertEqual((await self.bus.read_state())["service"], {})
    await self.bus.submit_command("run", Op.OPTIM_STEP, b"{}")
    with patch("tinyrl.control.accounting.monotonic_ns", side_effect=[0, 250_000]):
      await self.step_turn(node)
    self.assertEqual(w.run_service, {"run": 0.5})
    self.assertEqual(w.gpu_ms, 0.5)
    self.assertEqual((await self.bus.read_state())["service"], {"run-account": 0.5})
    run = await self.bus.get_run("run")
    self.assertEqual((run["cmds"], run["failed"], run["steps"], run["tokens"]), (1, 2, 0, 0))
