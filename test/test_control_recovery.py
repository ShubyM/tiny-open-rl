import copy
import dataclasses
import json
import os
from unittest.mock import patch

from test.helpers import RedisTest
from tinyrl.bus import Event, key
from tinyrl.control import scheduler
from tinyrl.control.node_daemon import NodeDaemon
from tinyrl.control.turns import GpuTurn


class TestControlRecovery(RedisTest):
  async def test_snapshot_and_tail_recover_a_reader_after_prefix_trimming(self):
    await self.running({"a": [0], "b": [0]})
    await self.bus.append_event("wants_gpu", worker="a")
    state, after, _ = await scheduler.read_log(self.bus, scheduler.State(), "0-0")
    before = copy.deepcopy(state)
    decisions = scheduler.decide(state, 1)
    self.assertEqual(state, before)
    for kind, body in decisions:
      await self.bus.append_event(kind, **body)
    state, after, _ = await scheduler.read_log(self.bus, state, after)
    await self.bus.r.set(key("state"), json.dumps(dataclasses.asdict(state) | {"after": after}))
    await self.bus.r.xtrim(key("events"), minid=after, approximate=False)
    await self.bus.append_event("wants_gpu", worker="b")
    resumed, cursor, _ = await scheduler.read_log(self.bus, scheduler.State(), "0-0")
    expected, expected_cursor, _ = await scheduler.read_log(self.bus, copy.deepcopy(state), after)
    self.assertEqual((resumed, cursor), (expected, expected_cursor))
    self.assertEqual(scheduler.holders(resumed, ("n", 0)), ["a"])
    self.assertFalse(any(kind == "granted" for kind, _ in scheduler.decide(resumed, 1)))

  async def test_stale_worker_events_cannot_release_a_replacement(self):
    await self.running({"a": [0], "b": [0]})
    state, _, _ = await scheduler.read_log(self.bus, scheduler.State(), "0-0")
    for kind, body in [
      ("started", {"worker": "a", "node": "n", "instance": "old"}),
      ("started", {"worker": "a", "node": "n", "instance": "new"}),
      ("wants_gpu", {"worker": "a", "instance": "new"}),
    ]:
      scheduler.fold(state, Event("", 1, kind, body))
    for kind, body in scheduler.decide(state, 1):
      scheduler.fold(state, Event("", 1, kind, body))
    scheduler.fold(state, Event("", 1, "wants_gpu", {"worker": "b"}))
    for kind in ("released", "exited", "idle", "batch_done"):
      scheduler.fold(state, Event("", 2, kind, {"worker": "a", "instance": "old", "seconds": 100}))
    self.assertEqual(scheduler.holders(state, ("n", 0)), ["a"])
    self.assertEqual(state.workers["a"].busy, 0)
    self.assertEqual(scheduler.decide(state, 2), [])

  @patch("tinyrl.control.turns.getenv", return_value="current")
  async def test_worker_recovers_its_grant_from_snapshot_after_log_trimming(self, _):
    await self.running({"a": [0]})
    await self.bus.append_event("started", worker="a", node="n", instance="current")
    asked = await self.bus.append_event("wants_gpu", worker="a", instance="current")
    state, after, _ = await scheduler.read_log(self.bus, scheduler.State(), "0-0")
    for kind, body in scheduler.decide(state, 1):
      await self.bus.append_event(kind, **body)
    marker = (await self.bus.r.xadd(key("events"), {"kind": "unrelated", "body": "{}"})).decode()
    state, after, _ = await scheduler.read_log(self.bus, state, after)
    await self.bus.r.set(key("state"), json.dumps(dataclasses.asdict(state) | {"after": after}))
    await self.bus.r.xtrim(key("events"), minid=marker, approximate=False)
    turn = GpuTurn(self.bus, "a", True)
    turn.after = asked
    self.assertEqual(await turn.next_decision(0), "granted")
    # A newer tail must not move the reader beyond the snapshot before it recovers the missing grant.
    newer = (await self.bus.r.xadd(key("events"), {"kind": "unrelated", "body": "{}"})).decode()
    await self.bus.r.xtrim(key("events"), minid=newer, approximate=False)
    turn.after = asked
    self.assertEqual(await turn.next_decision(0), "granted")

  async def test_a_flushed_log_does_not_keep_old_placements(self):
    await self.running({"a": [0]})
    state, after, _ = await scheduler.read_log(self.bus, scheduler.State(), "0-0")
    await self.bus.r.flushall()
    state, after, _ = await scheduler.read_log(self.bus, state, after)
    self.assertEqual(state, scheduler.State())
    self.assertEqual(after, "0-0")

  async def test_explicit_kv_cache_cannot_override_a_shared_gpu_budget(self):
    from tinyrl.workers.vllm import VLLM

    for kv, extra in (("1024", ""), ("", "--kv-cache-memory-bytes=1024"), ("", "--kv-cache-memory-bytes 1024")):
      with (
        patch("tinyrl.workers.vllm.GPU_MEM", "0.42"),
        patch("tinyrl.workers.vllm.KV_BYTES", kv),
        patch("tinyrl.workers.vllm.VLLM_ARGS", extra),
      ):
        with self.assertRaisesRegex(ValueError, "overrides"):
          await VLLM.start("unused", False)

  async def test_exclusive_worker_waits_when_no_gpu_is_available(self):
    state = scheduler.State(nodes={"n": (1, 80 * 2**30)})
    for run in ("a", "b"):
      scheduler.fold(state, Event("", 0, "run_created", {"run": run, "worker": run, "exclusive": True}))
    for kind, body in scheduler.decide(state, 0):
      scheduler.fold(state, Event("", 0, kind, body))
    self.assertEqual(state.workers["a"].gpus, (0,))
    self.assertIsNone(state.workers["b"].node)
    self.assertFalse(scheduler.fit([81 * 2**30], 80 * 2**30))

  @patch("tinyrl.control.node_daemon.subprocess.Popen")
  async def test_node_environment_cannot_increase_a_placed_memory_budget(self, popen):
    node = NodeDaemon(self.bus, gpus=2, node="n", gpu_mib=80 * 1024)
    worker = scheduler.Worker(scheduler.Kind(role="sampler"), node="n", gpus=(0,))
    with patch.dict(os.environ, {"TINYRL_GPU_MEM": "0.99"}):
      node.start_worker_process("w", worker)
    self.assertEqual(popen.call_args.kwargs["env"]["TINYRL_GPU_MEM"], "0.42")
