import asyncio
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from test.execution import TestNode
from test.helpers import BusTest
from tinyrl.control.node_daemon import WorkerHandle
from tinyrl.storage.store import Store
from tinyrl.types import Effect, Op, Reply, Role, Run


class TestAccounting(BusTest):
  async def setup_node(self, role, **accounts):
    self.receipts = self.enterContext(patch.object(self.bus, "bill", wraps=self.bus.bill))
    folder = tempfile.TemporaryDirectory()
    self.addCleanup(folder.cleanup)
    store = Store(root=folder.name)
    runs = list(accounts)
    for run, account in accounts.items():
      self.bus.create_run(Run(run, "base", 4, role=role, account=account, gpus=2))
    node = TestNode(self.bus, store=store, gpus=2, node="node")
    self.addAsyncCleanup(node.aclose)
    await node.bind("worker", *runs)
    if ts := node.tasks():
      await asyncio.gather(*ts)
    spec = self.bus.plan_state.procs["worker"].spec
    return node, spec

  def gpu_ms(self, run):
    return sum(c.args[3].gpu_ms for c in self.receipts.call_args_list if c.args[1].run == run and c.args[3])

  async def step_turn(self, node: TestNode):
    await node.tick()
    if ts := node.tasks():
      await asyncio.gather(*ts)

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
      self.bus.submit_command(run, Op.SAMPLE, b"{}", cost=999)
    with patch("tinyrl.control.accounting.monotonic_ns", side_effect=[0, 0, 10_000_000, 30_000_000]):
      await self.step_turn(node)
    w = node.workers["worker"]
    self.assertEqual({r: self.gpu_ms(r) for r in ("short", "long")}, {"short": 10, "long": 50})
    self.assertEqual(w.gpu_ms, 60)
    self.assertEqual(self.bus.service, {"alice": 60})
    short, long = self.bus.get_run("short"), self.bus.get_run("long")
    self.assertEqual((short["busy_ms"], short["failed"], short["tokens"]), (10, 1, 0))
    self.assertEqual((long["busy_ms"], long["failed"], long["tokens"]), (30, 0, 999))

  async def test_checkpoint_load_is_charged_once_across_requests_even_if_it_fails(self):
    node, spec = await self.setup_node(Role.SAMPLER, a1="alice", a2="alice", b="bob")
    now, uri = 0, "tinker://trainer/sampler_weights/shared"
    self.seed_checkpoint(uri, 100, version="version")
    from dataclasses import replace

    for run in ("a1", "a2"):
      self.bus.runs[run].run = replace(self.bus.runs[run].run, model_path=uri)
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
          self.bus.submit_command(run, Op.SAMPLE, b"{}")
        # First tick prefetches version off-GPU; wait for prefetch then run waves for alice & bob.
        await node.tick()
        if node.staging:
          await asyncio.gather(*node.staging.values())
        await self.step_turn(node)
        await self.step_turn(node)
    self.assertEqual(model.load.await_count, 2)
    self.assertEqual(model.execute.await_count, 5)
    self.assertAlmostEqual(self.gpu_ms("a1"), 80 / 3)
    self.assertAlmostEqual(self.gpu_ms("a2"), 40 / 3)
    self.assertAlmostEqual(self.bus.service["alice"], 40)
    self.assertEqual((self.bus.get_run("a1"))["failed"], 2)

  async def test_three_sampler_sessions_get_the_same_gpu_time_as_one(self):
    accounts = {"a1": "alice", "a2": "alice", "a3": "alice", "b": "bob"}
    node, spec = await self.setup_node(Role.SAMPLER, **accounts)
    for run in accounts:
      for _ in range(6):
        self.bus.submit_command(run, Op.SAMPLE, b"{}")

    now = 0

    async def sample(cmd, *_):
      nonlocal now
      await asyncio.sleep(0)  # All calls in this wave overlap for 12 ms.
      now = end
      return Reply(True, "application/json", b"{}")

    model = SimpleNamespace(execute=sample, resume=AsyncMock(), park=AsyncMock(), close=AsyncMock(), proc=None)
    node.workers["worker"] = WorkerHandle("worker", spec, model, resident=(0, 1), bound=(0, 1))
    with (
      patch("tinyrl.types.MAX_SEQS", 2),
      patch("tinyrl.control.accounting.monotonic_ns", lambda: now),
    ):
      for _ in range(6):
        end = now + 12_000_000
        await self.step_turn(node)

    self.assertEqual(self.bus.service, {"alice": 72, "bob": 72})
    self.assertEqual({r: self.gpu_ms(r) for r in accounts}, {"a1": 24, "a2": 24, "a3": 24, "b": 72})

  async def test_backlogged_training_runs_of_one_account_alternate_equal_cost_steps(self):
    node, spec = await self.setup_node(Role.TRAINER, first="account", second="account")
    order = []

    async def execute(method, cmd, *args):
      order.append(cmd.run)
      return Reply(True, "application/json", b"{}"), Effect()

    model = SimpleNamespace(call=execute, resume=AsyncMock(), park=AsyncMock(), close=AsyncMock(), proc=None)
    node.workers["worker"] = WorkerHandle("worker", spec, model, resident=(0, 1), bound=(0, 1))
    for run in ("first", "second"):  # All "first" heads arrive before all "second" heads.
      for _ in range(4):
        self.bus.submit_command(run, Op.OPTIM_STEP, b"{}")
      await asyncio.sleep(0.002)
    with patch("tinyrl.control.accounting.monotonic_ns", side_effect=range(0, 16_000_000, 1_000_000)):
      for _ in range(8):
        await self.step_turn(node)
    self.assertEqual(order, ["first", "second"] * 4)

  async def test_trainer_bills_only_the_failure_that_entered_the_model(self):
    node, spec = await self.setup_node(Role.TRAINER, run="run-account")
    failure = ValueError("bad input")
    model = SimpleNamespace(
      call=AsyncMock(return_value=(Reply.error(failure), Effect())),
      resume=AsyncMock(),
      park=AsyncMock(),
      close=AsyncMock(),
      proc=None,
    )
    node.workers["worker"] = WorkerHandle("worker", spec, model, resident=(0, 1), bound=(0, 1))
    self.bus.submit_command(
      "run", Op.LOAD_WEIGHTS, b'{"uri":"tinker://run/weights/missing"}', uri="tinker://run/weights/missing"
    )
    await self.step_turn(node)
    model.call.assert_not_awaited()
    w = node.workers["worker"]
    self.assertEqual(w.gpu_ms, 0)
    self.assertEqual(self.bus.service, {})
    self.bus.submit_command("run", Op.OPTIM_STEP, b"{}")
    with patch("tinyrl.control.accounting.monotonic_ns", side_effect=[0, 250_000]):
      await self.step_turn(node)
    self.assertEqual(self.gpu_ms("run"), 0.5)
    self.assertEqual(w.gpu_ms, 0.5)
    self.assertEqual(self.bus.service, {"run-account": 0.5})
    run = self.bus.get_run("run")
    self.assertEqual((run["cmds"], run["failed"], run["steps"], run["tokens"]), (1, 2, 0, 0))
