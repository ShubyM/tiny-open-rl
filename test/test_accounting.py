import asyncio
import tempfile
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from test.execution import TestNode
from test.helpers import BusTest
from tinyrl.control.node_daemon import WorkerHandle
from tinyrl.storage.store import Store
from tinyrl.types import Effect, Op, Reply, Role, Run
from tinyrl.workers.sampler import Sampler


class TestAccounting(BusTest):
  async def setup_node(self, role, **accounts):
    self.receipts = self.enterContext(patch.object(self.bus, "_bill", wraps=self.bus._bill))
    folder = tempfile.TemporaryDirectory()
    self.addCleanup(folder.cleanup)
    store = Store(root=folder.name)
    runs = list(accounts)
    for run, account in accounts.items():
      self.bus.create_run(Run(run, "base", 4, role=role, account=account, gpus=2))
    node = TestNode(self.bus, store=store, gpus=2, node="node")
    self.addAsyncCleanup(node.aclose)
    await node.bind("worker", *runs)
    spec = self.bus.runs[runs[0]].run.spec
    return node, spec

  def gpu_ms(self, run):
    return sum(c.args[3].gpu_ms for c in self.receipts.call_args_list if c.args[1].run == run and c.args[3])

  def worker_gpu_ms(self):
    return sum(w.gpu_ms for w in self.bus.nodes["node"].workers.values())

  async def step_turn(self, node: TestNode):
    await node.tick()
    if ts := node.tasks():
      await asyncio.gather(*ts)
    await node.tick()

  async def test_sampler_shares_busy_time_until_the_short_request_fails(self):
    node, spec = await self.setup_node(Role.SAMPLER, short="alice", long="alice")
    finished = asyncio.Event()

    async def sample(cmd, ckpt=None, *, wave=None, charge=None):
      async def inner():
        if cmd.run == "short":
          await asyncio.sleep(0)
          finished.set()
          return Reply.error(ValueError("sampling failed"))
        await finished.wait()
        return Reply(True, "application/json", b"{}")

      return await wave.run(inner(), charge)

    model = SimpleNamespace(execute=sample, resume=AsyncMock(), park=AsyncMock(), close=AsyncMock(), proc=None)
    node.workers["worker"] = WorkerHandle("worker", spec, model, resident=(0, 1), bound=(0, 1))
    for run in ("short", "long"):
      self.bus.submit_command(run, Op.SAMPLE, b"{}", cost=999)
    with patch("tinyrl.control.accounting.monotonic_ns", side_effect=[0, 0, 10_000_000, 30_000_000]):
      await self.step_turn(node)
    self.assertEqual({r: self.gpu_ms(r) for r in ("short", "long")}, {"short": 10, "long": 50})
    self.assertEqual(self.worker_gpu_ms(), 60)
    self.assertEqual(self.bus.service, {"alice": 60})
    short, long = self.bus.get_run("short"), self.bus.get_run("long")
    self.assertEqual((short["busy_ms"], short["failed"], short["tokens"]), (10, 1, 0))
    self.assertEqual((long["busy_ms"], long["failed"], long["tokens"]), (30, 0, 999))

  async def test_checkpoint_load_is_charged_once_across_requests_even_if_it_fails(self):
    node, spec = await self.setup_node(Role.SAMPLER, a1="alice", a2="alice", b="bob")
    now, uri = 0, "tinker://trainer/sampler_weights/shared"
    self.seed_checkpoint(uri, 100, version="version")
    for run in ("a1", "a2"):
      self.bus.runs[run].run = replace(self.bus.runs[run].run, model_path=uri)
    cache_dir = tempfile.TemporaryDirectory()
    self.addCleanup(cache_dir.cleanup)
    node.store.local_folder = Mock(return_value=(cache_dir.name, "version"))

    class DummyEngine:
      server = None
      sample_count = 0

      @classmethod
      async def start(cls, *_):
        return cls()

      async def sample(self, job, name):
        DummyEngine.sample_count += 1
        from tinyrl.workers.sampler import Out, Seq

        return Out([Seq([1], [-0.1], [], True)], [None])

      async def wake(self):
        pass

      async def sleep(self):
        pass

      async def close(self):
        pass

    sampler = Sampler("base", True, engine=DummyEngine, store=node.store)
    sampler.running = DummyEngine()

    async def load(_):
      nonlocal now
      now += 10_000_000
      if sampler.load.await_count == 2:
        raise ValueError("bad weights")

    sampler.load = AsyncMock(side_effect=load)  # type: ignore[method-assign]
    job = b'{"prompt":[1],"num_samples":1,"max_tokens":1,"prompt_logprobs":false,"prompt_logprobs_last_n":null,"topk_prompt":0,"topk_sample":0}'
    node.workers["worker"] = WorkerHandle("worker", spec, sampler, resident=(0, 1), bound=(0, 1))
    with patch("tinyrl.control.accounting.monotonic_ns", lambda: now):
      for _ in range(2):
        for run in ("a1", "a1", "a2", "b"):
          self.bus.submit_command(run, Op.SAMPLE, job)
        await self.step_turn(node)  # stages version + runs bob (or alice once staged)
        await self.step_turn(node)
        await self.step_turn(node)
    self.assertEqual(sampler.load.await_count, 2)
    self.assertEqual(DummyEngine.sample_count, 5)
    self.assertAlmostEqual(self.gpu_ms("a1"), 80 / 3)
    self.assertAlmostEqual(self.gpu_ms("a2"), 40 / 3)
    self.assertAlmostEqual(self.bus.service["alice"], 40)
    self.assertEqual((self.bus.get_run("a1"))["failed"], 2)

  async def test_three_sampler_sessions_get_the_same_gpu_time_as_one(self):
    accounts = {"a1": "alice", "a2": "alice", "a3": "alice", "b": "bob"}
    node, spec = await self.setup_node(Role.SAMPLER, **accounts)
    now = 0

    async def sample(cmd, ckpt=None, *, wave=None, charge=None):
      async def inner():
        nonlocal now
        await asyncio.sleep(0)
        now = end
        return Reply(True, "application/json", b"{}")

      return await wave.run(inner(), charge)

    model = SimpleNamespace(execute=sample, resume=AsyncMock(), park=AsyncMock(), close=AsyncMock(), proc=None)
    node.workers["worker"] = WorkerHandle("worker", spec, model, resident=(0, 1), bound=(0, 1))
    with (
      patch("tinyrl.types.MAX_SEQS", 2),
      patch("tinyrl.control.accounting.monotonic_ns", lambda: now),
    ):
      for run in accounts:
        for _ in range(6):
          self.bus.submit_command(run, Op.SAMPLE, b"{}")
      await node.tick()
      for _ in range(6):
        end = now + 12_000_000
        if ts := node.tasks():
          await asyncio.gather(*ts)
        await node.tick()

    self.assertAlmostEqual(self.bus.service["alice"], 72)
    self.assertAlmostEqual(self.bus.service["bob"], 72)
    for r, want in (("a1", 24), ("a2", 24), ("a3", 24), ("b", 72)):
      self.assertAlmostEqual(self.gpu_ms(r), want)

  async def test_backlogged_training_runs_of_one_account_alternate_equal_cost_steps(self):
    node, spec = await self.setup_node(Role.TRAINER, first="account", second="account")
    order = []

    async def execute(method, cmd, *args):
      order.append(cmd.run)
      return Reply(True, "application/json", b"{}"), Effect(loaded=True, mutated=True)

    model = SimpleNamespace(call=execute, resume=AsyncMock(), park=AsyncMock(), close=AsyncMock(), proc=None)
    node.workers["worker"] = WorkerHandle("worker", spec, model, resident=(0, 1), bound=(0, 1))
    for run in ("first", "second"):
      for _ in range(4):
        self.bus.submit_command(run, Op.OPTIM_STEP, b"{}")
      await asyncio.sleep(0.002)
    with patch("tinyrl.control.accounting.monotonic_ns", side_effect=range(0, 16_000_000, 1_000_000)):
      await node.tick()
      for _ in range(8):
        if ts := node.tasks():
          await asyncio.gather(*ts)
        await node.tick()
    self.assertEqual(order, ["first", "second"] * 4)

  async def test_trainer_bills_only_the_failure_that_entered_the_model(self):
    node, spec = await self.setup_node(Role.TRAINER, run="run-account")
    failure = ValueError("bad input")
    model = SimpleNamespace(
      call=AsyncMock(return_value=(Reply.error(failure), Effect(loaded=True, mutated=True))),
      resume=AsyncMock(),
      park=AsyncMock(),
      close=AsyncMock(),
      proc=None,
    )
    node.open_trainer = lambda wid, width, env: model  # type: ignore[assignment]
    self.bus.submit_command(
      "run", Op.LOAD_WEIGHTS, b'{"uri":"tinker://run/weights/missing"}', uri="tinker://run/weights/missing"
    )
    await self.step_turn(node)
    model.call.assert_not_awaited()
    self.assertEqual(self.worker_gpu_ms(), 0)
    self.assertEqual(self.bus.service, {})
    self.bus.submit_command("run", Op.OPTIM_STEP, b"{}")
    with patch("tinyrl.control.accounting.monotonic_ns", side_effect=[0, 250_000]):
      await self.step_turn(node)
    self.assertEqual(self.gpu_ms("run"), 0.5)
    self.assertEqual(self.worker_gpu_ms(), 0.5)
    self.assertEqual(self.bus.service, {"run-account": 0.5})
    run = self.bus.get_run("run")
    self.assertEqual((run["cmds"], run["failed"], run["steps"], run["tokens"]), (1, 2, 0, 0))
