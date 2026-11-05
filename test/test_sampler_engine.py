import asyncio
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from tinyrl.api import wire
from tinyrl.bus import Cmd, Op
from tinyrl.workers import snapshot
from tinyrl.workers.sampler import Out, Sampler, Seq
from tinyrl.workers.vllm import CUDAGRAPH_CONFIG, KERNEL_CONFIG, VLLM

JOB = {
  "prompt": [1, 2],
  "num_samples": 1,
  "max_tokens": 1,
  "temperature": 0.0,
  "top_k": -1,
  "top_p": 1.0,
  "seed": None,
  "stop": None,
  "prompt_logprobs": False,
  "prompt_logprobs_last_n": None,
  "topk_prompt": 0,
  "topk_sample": 0,
}


class RecordingEngine:
  def __init__(self, base, lora, env):
    self.calls, self.loaded = [("start", base, lora, env)], set()
    self.lora = lora
    self.active = self.peak = 0

  @classmethod
  async def start(cls, base, lora, env):
    await asyncio.sleep(0)
    return cls(base, lora, dict(env))

  async def load_checkpoint(self, name, path):
    if name not in self.loaded:
      self.calls.append(("load", name, path))
      self.loaded = self.loaded | {name} if self.lora else {name}

  async def sample(self, job, name):
    self.calls.append(("sample", name))
    self.active += 1
    self.peak = max(self.active, self.peak)
    await asyncio.sleep(0.01)
    self.active -= 1
    return Out([Seq([3], [-1.0], [{3: -1.0}], False)], [])

  async def sleep(self):
    self.calls.append(("sleep",))

  async def wake(self):
    self.calls.append(("wake",))

  async def close(self):
    self.calls.append(("close",))


class TestSamplerDriver(unittest.IsolatedAsyncioTestCase):
  def cmd(self, run="s", n=0):
    return Cmd(run, f"{n}-0", Op.SAMPLE, json.dumps(JOB).encode())

  async def test_start_is_shared_by_concurrent_samples_and_env_is_copied(self):
    env = {"CUDA_VISIBLE_DEVICES": "2"}
    sampler = Sampler("base", True, env, engine=RecordingEngine)
    env["CUDA_VISIBLE_DEVICES"] = "5"
    replies = await asyncio.gather(*(sampler.execute(self.cmd(n=n), {}, None) for n in range(8)))
    engine = sampler.running
    self.assertTrue(all(r.ok for r in replies))
    self.assertEqual(wire.sample_json(replies[0].body)["sequences"][0]["tokens"], [3])
    self.assertEqual(engine.calls[0], ("start", "base", True, {"CUDA_VISIBLE_DEVICES": "2"}))
    self.assertEqual(engine.peak, 8)
    self.assertEqual(await sampler.start(), 0)
    await sampler.park()
    await sampler.resume()
    await sampler.close()
    self.assertEqual(engine.calls[-3:], [("sleep",), ("wake",), ("close",)])
    self.assertIsNone(sampler.running)

  async def test_checkpoint_identity_follows_uri_and_version_and_engine_owns_loaded_state(self):
    sampler = Sampler("base", False, {}, engine=RecordingEngine)
    a, b = "tinker://r/sampler_weights/a", "tinker://r/sampler_weights/b"
    for uri, version in [(a, "1"), (a, "1"), (b, "1"), (a, "1"), (a, "2")]:
      self.assertTrue((await sampler.execute(self.cmd(), {"model_path": uri}, ("/weights", version))).ok)
    engine = sampler.running
    names = [c[1] for c in engine.calls if c[0] == "load"]
    self.assertEqual(len(names), 4)
    self.assertEqual(names[0], names[2])
    self.assertEqual(len(set(names)), 3)
    await sampler.close()

    # Recreating a controller doesn't assign a new name to the same immutable checkpoint.
    other = Sampler("base", True, {}, engine=RecordingEngine)
    await other.execute(self.cmd(), {"model_path": a}, ("/another-node-cache", "1"))
    self.assertEqual(other.running.calls[1][1], names[0])
    await other.close()

  async def test_failed_staging_replies_without_starting_engine_and_next_command_works(self):
    sampler = Sampler("base", True, {}, engine=RecordingEngine)
    uri = "tinker://r/sampler_weights/a"
    reply = await sampler.execute(self.cmd(), {"model_path": uri}, RuntimeError("bucket error"))
    self.assertFalse(reply.ok)
    self.assertIn("bucket error", json.loads(reply.body)["error"])
    self.assertIsNone(sampler.running)
    self.assertTrue((await sampler.execute(self.cmd(), {}, None)).ok)
    await sampler.close()


class TestEngineLaunch(unittest.IsolatedAsyncioTestCase):
  async def test_each_engine_has_its_own_devices_memory_and_flags(self):
    with (
      patch("tinyrl.workers.vllm.models.local", return_value="/model"),
      patch("tinyrl.workers.vllm.launch") as launch,
      patch("tinyrl.workers.vllm.reap"),
      patch("tinyrl.workers.vllm.httpx.AsyncClient.get", new_callable=AsyncMock) as get,
    ):
      launch.side_effect = lambda *args, **kwargs: MagicMock(
        poll=MagicMock(return_value=None), wait=MagicMock(return_value=0)
      )
      get.return_value.is_success = True
      first_env = {
        "CUDA_VISIBLE_DEVICES": "1",
        "TINYRL_GPU_MEM": "0.42",
        "TINYRL_KV_BYTES": "16000",
        "TINYRL_WARM_SAMPLER": "1",
        "VLLM_ARGS": "--compilation-config '{}' --kernel-config '{}' --gpu-memory-utilization 0.8",
      }
      second_env = {
        "CUDA_VISIBLE_DEVICES": "2,3",
        "TINYRL_KV_BYTES": "9000",
        "VLLM_ARGS": "--kv-cache-memory-bytes=123",
      }
      ambient = dict(os.environ)
      engines = await asyncio.gather(VLLM.start("base", True, first_env), VLLM.start("base", False, second_env))
      calls = {c.kwargs["env"]["CUDA_VISIBLE_DEVICES"]: c for c in launch.call_args_list}
      a, b = calls["1"].args[0], calls["2,3"].args[0]
      self.assertNotIn("--tensor-parallel-size", a)
      self.assertEqual(b[b.index("--tensor-parallel-size") + 1], "2")
      self.assertEqual(a[-4:], ["--gpu-memory-utilization", "0.42", "--kv-cache-memory-bytes", "16000"])
      self.assertNotIn("--kv-cache-memory-bytes", b)
      self.assertIn("--kv-cache-memory-bytes=123", b)
      self.assertEqual(a[a.index("--compilation-config") + 1], "{}")
      self.assertEqual(a[a.index("--kernel-config") + 1], "{}")
      self.assertIn(CUDAGRAPH_CONFIG, b)
      self.assertIn(KERNEL_CONFIG, b)
      self.assertEqual(dict(os.environ), ambient)
      self.assertEqual(first_env["TINYRL_GPU_MEM"], "0.42")
      self.assertFalse({"TINYRL_GPU_MEM", "TINYRL_KV_BYTES", "TINYRL_WARM_SAMPLER"} & calls["1"].kwargs["env"].keys())
      await asyncio.gather(*(e.close() for e in engines))

  async def test_failed_or_cancelled_start_reaps_the_server(self):
    for cancelled in (False, True):
      with self.subTest(cancelled=cancelled):
        entered = asyncio.Event()

        async def health(*args, entered=entered, **kwargs):
          entered.set()
          await asyncio.Event().wait()

        server = MagicMock()
        server.poll.return_value = None if cancelled else 1
        server.returncode = 1
        with (
          patch("tinyrl.workers.vllm.models.local", return_value="/model"),
          patch("tinyrl.workers.vllm.launch", return_value=server),
          patch("tinyrl.workers.vllm.reap") as reap,
          patch("tinyrl.workers.vllm.httpx.AsyncClient.get", side_effect=health),
        ):
          task = asyncio.create_task(VLLM.start("base", False, {}))
          if cancelled:
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
          with self.assertRaises(asyncio.CancelledError if cancelled else RuntimeError):
            await task
          server.terminate.assert_not_called()
          reap.assert_called_once_with(server)
          server.wait.assert_called()

  async def test_close_reaps_the_owned_group_and_waits(self):
    server = MagicMock()
    server.wait.return_value = 0
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    engine = VLLM("http://vllm", "base", True, server, client)
    with patch("tinyrl.workers.vllm.reap") as reap:
      await engine.close()
      reap.assert_called_once_with(server)
      server.wait.assert_called_once_with()
      server.terminate.assert_not_called()
    self.assertTrue(client.is_closed)
    self.assertIsNone(engine.server)

  async def test_cancelled_close_retains_handles_until_exit_is_confirmed(self):
    entered, exited = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def wait():
      loop.call_soon_threadsafe(entered.set)
      exited.wait(5)
      return 0

    server = MagicMock(wait=MagicMock(side_effect=wait))
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    engine = VLLM("http://vllm", "base", True, server, client)
    sampler = Sampler("base", True, {}, engine=RecordingEngine)
    sampler.running = engine
    with patch("tinyrl.workers.vllm.reap"):
      task = asyncio.create_task(sampler.close())
      try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
          await task
        self.assertIs(sampler.running, engine)
        self.assertIs(engine.server, server)
      finally:
        exited.set()
      await sampler.close()
    self.assertIsNone(sampler.running)
    self.assertIsNone(engine.server)

  async def test_vllm_checkpoint_loading_is_idempotent_for_the_engine_current_state(self):
    calls = []
    client = httpx.AsyncClient(
      base_url="http://vllm",
      transport=httpx.MockTransport(lambda r: (calls.append(r.url.path), httpx.Response(200))[1]),
    )
    engine = VLLM("http://vllm", "base", False, client=client)
    for name in ("a", "a", "b", "b", "a"):
      await engine.load_checkpoint(name, "/weights")
    self.assertEqual(calls, ["/collective_rpc", "/reset_prefix_cache"] * 3)
    await engine.close()


class TestSnapshotOwnership(unittest.IsolatedAsyncioTestCase):
  async def test_concurrent_saves_from_one_daemon_publish_a_complete_snapshot(self):
    with tempfile.TemporaryDirectory() as folder:
      destination = f"{folder}/image"
      for pid in (123, 456):
        Path(f"{folder}/{pid}.log").write_text("server log")
      started = threading.Barrier(2)

      def pids(_):
        started.wait(2)
        return []

      with patch("tinyrl.workers.snapshot.cuda_pids", side_effect=pids), patch("tinyrl.workers.snapshot.run") as run:
        await asyncio.gather(
          *(asyncio.to_thread(snapshot.save, pid, destination, f"{folder}/{pid}", "GPU-target") for pid in (123, 456))
        )
      with open(f"{destination}/meta.json") as f:
        self.assertEqual(json.load(f)["gpu"], "GPU-target")
      self.assertEqual(len(list(Path(f"{destination}/keep").iterdir())), 1)
      self.assertFalse(Path(f"{destination}.123").exists())
      self.assertFalse(Path(f"{destination}.456").exists())
      dumps = [c.args for c in run.call_args_list if c.args[:2] == ("criu", "dump")]
      self.assertEqual({args[args.index("-D") + 1] for args in dumps}, {f"{destination}.123", f"{destination}.456"})

  async def test_cancelled_restore_reaps_namespace_while_readiness_is_blocked(self):
    entered, unblock = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def read():
      loop.call_soon_threadsafe(entered.set)
      unblock.wait(5)
      return ""

    ns = MagicMock()
    ns.stdout.readline.side_effect = read
    with (
      patch("tinyrl.workers.snapshot.gpu", return_value="GPU-target"),
      patch("tinyrl.workers.snapshot.launch", return_value=ns) as launch,
      patch("tinyrl.workers.snapshot.reap", side_effect=lambda _: unblock.set()) as reap,
    ):
      task = asyncio.create_task(snapshot.restore("/snapshot", "2", {"CUDA_VISIBLE_DEVICES": "2"}))
      try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
          await task
        self.assertEqual(launch.call_args.args[0][-1], "GPU-target")
        reap.assert_called_once_with(ns)
        ns.wait.assert_called_once()
        ns.stdout.close.assert_called_once()
      finally:
        unblock.set()

  def test_snapshot_key_uses_the_selected_gpu_not_the_daemons_first_device(self):
    with (
      patch("tinyrl.workers.snapshot.subprocess.run", return_value=MagicMock(stdout="driver, model")) as run,
      patch("tinyrl.workers.snapshot.importlib.metadata.version", return_value="1"),
    ):
      snapshot.key(["vllm", "serve"], "GPU-target")
      self.assertEqual(run.call_args.args[0][-2:], ["-i", "GPU-target"])
