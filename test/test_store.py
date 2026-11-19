import asyncio
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid

import torch

from test.helpers import RedisTest, new_id
from tinyrl.bus import Op, Reply, Run, key
from tinyrl.storage.store import SHARD_BYTES, Store


class TestStore(unittest.TestCase):
  def test_put_and_get_roundtrip_tensors_and_metadata(self):
    store = Store(root=tempfile.mkdtemp())
    want = {
      "w": torch.arange(64, dtype=torch.float32).reshape(8, 8),
      "b": torch.randn(8, dtype=torch.bfloat16),
      "w.step": torch.tensor(3.0),
    }
    version = store.put(want, {"optimizer": True})
    folder, _ = store.local_folder(version)
    self.assertEqual(os.listdir(folder), ["state.safetensors"])

    got, meta = store.get(version)
    self.assertEqual(meta, {"optimizer": True})
    self.assertTrue(all(torch.equal(got[k], want[k]) for k in want))

  def test_missing_checkpoints_fail_cleanly(self):
    store = Store(root=tempfile.mkdtemp())
    with self.assertRaises(ValueError):
      store.get("tinker://run-1/weights/missing")

  def test_a_remote_checkpoint_saved_on_one_node_loads_on_another(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    saved = saver.put({"w": torch.ones(4)}, {"lora": True, "config": {"r": 4}})
    saver.upload_pending_saves()  # what start() does in the background

    folder, version = loader.local_folder(saved)
    self.assertEqual(sorted(os.listdir(folder)), ["adapter_config.json", "adapter_model.safetensors"])
    self.assertTrue(torch.equal(loader.get(saved)[0]["w"], torch.ones(4)))
    self.assertEqual(loader.local_folder(saved), (folder, version))  # downloaded once

    newer = saver.put({"w": torch.zeros(4)}, {"lora": True, "config": {"r": 4}})  # the same name, saved again
    saver.upload_pending_saves()
    self.assertNotEqual(loader.local_folder(newer)[1], version)
    self.assertTrue(torch.equal(loader.get(newer)[0]["w"], torch.zeros(4)))
    with self.assertRaises(ValueError):
      loader.get("tinker://run-1/weights/missing")

  def test_roundtrips_contiguous_and_non_contiguous_tensors(self):
    store = Store(root=tempfile.mkdtemp())
    base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    want = {
      "contiguous": base.clone(),
      "transposed": base.T,
    }
    version = store.put(want)
    got, _ = store.get(version)
    self.assertTrue(torch.equal(got["contiguous"], want["contiguous"]), "roundtrips contiguous tensor")
    self.assertTrue(torch.equal(got["transposed"], want["transposed"]), "roundtrips non-contiguous tensor")

  def test_shards_large_tensors(self):
    if SHARD_BYTES != 64:
      subprocess.run(
        [sys.executable, "-m", "unittest", "test.test_store.TestStore.test_shards_large_tensors"],
        env=os.environ | {"SHARD_BYTES": "64"},
        check=True,
      )
      return
    store = Store(root=tempfile.mkdtemp())
    want = {
      "w1": torch.arange(64, dtype=torch.float32).reshape(8, 8),
      "w2": torch.randn(8, dtype=torch.bfloat16),
      "w3": torch.tensor(3.0),
    }
    version = store.put(want, {"optimizer": True, "lora": False})
    folder, _ = store.local_folder(version)
    files = sorted(os.listdir(folder))
    self.assertTrue(any("state-00001-of-" in f for f in files))

    got, meta = store.get(version)
    self.assertEqual(meta, {"optimizer": True, "lora": False})
    for k in want:
      self.assertTrue(torch.equal(got[k], want[k]))

    _, v1 = store.local_folder(version)
    version = store.put(want, {"optimizer": True, "lora": False})
    _, v2 = store.local_folder(version)
    self.assertNotEqual(v1, v2)

    version = store.put(want, {"lora": False, "config": {"test": True}})
    sampler_folder, _ = store.local_folder(version)
    sampler_files = sorted(os.listdir(sampler_folder))
    self.assertIn("config.json", sampler_files)
    self.assertIn("model.safetensors.index.json", sampler_files)
    self.assertTrue(any("model-00001-of-" in f for f in sampler_files))
    want_huge = {
      "t1": torch.ones(512, dtype=torch.uint8),
      "t2": torch.ones(512, dtype=torch.uint8),
      "t3": torch.ones(8, dtype=torch.uint8),
      "t4": torch.ones(8, dtype=torch.uint8),
    }
    version = store.put(want_huge, {"lora": False, "config": {"test": True}})
    huge_folder, _ = store.local_folder(version)
    st_file = next(f for f in sorted(os.listdir(huge_folder)) if f.endswith(".safetensors"))
    import safetensors

    with safetensors.safe_open(f"{huge_folder}/{st_file}", framework="pt", device="cpu") as f:
      keys = f.keys()
      self.assertNotEqual("t1" in keys, "t2" in keys)

  def test_writes_the_same_bytes_as_safetensors(self):
    import safetensors.torch

    tensors = {
      "scalar": torch.tensor(3.0, dtype=torch.float32),
      "empty": torch.zeros(0, 4, dtype=torch.float16),
      "f64": torch.arange(5, dtype=torch.float64),
      "i64": torch.arange(20, dtype=torch.int64).view(4, 5).t(),
      "i8": torch.arange(-3, 3, dtype=torch.int8),
      "u8": torch.arange(7, dtype=torch.uint8),
      "bool": torch.tensor([True, False, True]),
      "bf16": torch.arange(30, dtype=torch.bfloat16),
    }
    folder = tempfile.mkdtemp()
    Store(root=folder).write_safetensors(0, f"{folder}/mine.safetensors", tensors, {"meta": "{}"})
    got = safetensors.torch.load_file(f"{folder}/mine.safetensors")
    self.assertEqual(set(got), set(tensors))
    for k, v in tensors.items():
      self.assertEqual((got[k].dtype, got[k].shape), (v.dtype, v.shape))
      self.assertTrue(torch.equal(got[k], v))
    with safetensors.safe_open(f"{folder}/mine.safetensors", framework="pt") as f:
      self.assertEqual(f.metadata(), {"meta": "{}"})

  def test_concurrent_saves_leave_the_opened_version_intact(self):
    store = Store(tempfile.mkdtemp())
    original = store.put({"a": torch.ones(8), "b": torch.ones(8) * 2})
    errors = []
    stop = threading.Event()

    def reader():
      try:
        while not stop.is_set():
          tensors, _ = store.get(original)
          torch.testing.assert_close(tensors, {"a": torch.ones(8), "b": torch.ones(8) * 2})
      except Exception as exc:
        errors.append(exc)

    thread = threading.Thread(target=reader)
    thread.start()
    try:
      for step in range(2, 7):
        self.assertNotEqual(store.put({"a": torch.full((8,), step)}), original)
    finally:
      stop.set()
      thread.join()
    self.assertEqual(errors, [])
    self.assertEqual(len(store.versions()), 6)

  def test_a_remote_save_returns_before_its_upload_and_other_nodes_read_it_once_it_lands(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    version = saver.put({"w": torch.ones(4)})  # returns before anything uploads it
    self.assertTrue(torch.equal(saver.get(version)[0]["w"], torch.ones(4)))  # this node reads it at once
    with self.assertRaisesRegex(ValueError, "has not landed"):
      loader.get(version)
    saver.upload_pending_saves()
    self.assertTrue(torch.equal(loader.get(version)[0]["w"], torch.ones(4)))

  def test_another_node_copies_a_save_from_the_node_that_made_it_before_it_uploads(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    saver.start()
    saver.stopping.set()  # no uploads, so the copy can only come from the saver
    saver.uploader.join()
    meta, w = {"lora": False, "config": {"model_type": "x"}}, torch.randn(64, 64).to(torch.bfloat16)
    versions = [saver.put({"w": w * step}, meta) for step in range(2)]
    self.assertFalse(saver.fs.exists(f"{saver.base}/objects"))
    folder, version = loader.local_folder(versions[1])
    self.assertEqual(version, versions[1].partition("@")[0])
    for name in os.listdir(f"{saver.cache}/{version}"):
      with open(f"{saver.cache}/{version}/{name}", "rb") as a, open(f"{folder}/{name}", "rb") as b:
        self.assertEqual(a.read(), b.read())
    self.assertTrue(torch.equal(loader.get(versions[0])[0]["w"], w * 0))
    saver.close()  # its saves upload, then it stops serving them
    later = Store(root, cache=tempfile.mkdtemp())
    self.assertTrue(torch.equal(later.get(versions[1])[0]["w"], w))

  def test_a_save_whose_node_is_gone_comes_from_the_bucket_or_fails(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    saver.peer = "127.0.0.1:9"  # a node that no longer answers
    version = saver.put({"w": torch.ones(4)})
    with self.assertRaisesRegex(ValueError, "has not landed, and 127.0.0.1:9 failed to copy"):
      loader.get(version)
    saver.upload_pending_saves()
    self.assertTrue(torch.equal(loader.get(version)[0]["w"], torch.ones(4)))

  def test_trim_cache_keeps_unuploaded_saves_and_evicts_oldest_uploaded(self):
    from unittest.mock import patch

    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    store = Store(root, cache=tempfile.mkdtemp())
    old = store.put({"w": torch.ones(64)})
    store.upload_pending_saves()
    pending = store.put({"w": torch.ones(64)})
    with patch("tinyrl.storage.store.CACHE_BYTES", 1):
      store.trim_cache()
    self.assertFalse(os.path.exists(f"{store.cache}/{old}"))
    self.assertTrue(os.path.isdir(f"{store.cache}/{pending}"))


class TestRegistry(RedisTest):
  async def test_deletion_wins_over_concurrent_ttl_updates(self):
    run = new_id()
    await self.bus.create_run(Run(id=run, base="base", rank=8))
    await self.bus.touch_leases("owner")
    token, _, _ = await self.bus.acquire_run(run, "owner")
    uri = f"tinker://{run}/sampler_weights/session"
    await self.seed_checkpoint(uri, 1, ttl=60, owned=True)
    await self.bus.submit_command(run, Op.DELETE_MODEL, b"{}")
    [cmd] = await self.bus.take_commands([run], "owner", count=1)
    await asyncio.gather(
      *(self.bus.set_checkpoint_ttl(uri, 600) for _ in range(20)),
      self.bus.commit_reply(cmd, Reply(True, "application/json", b"{}"), token),
    )
    self.assertFalse(await self.bus.set_checkpoint_ttl(uri, 600))
    self.assertIsNone(await self.bus.get_checkpoint(uri))
    for index in ("ckpts", "expiry"):
      self.assertIsNone(await self.bus.r.zscore(key(index), uri))

  async def test_stale_writer_cannot_replace_published_bytes_on_disk_or_in_bucket(self):
    for root in (tempfile.mkdtemp(), f"file://{tempfile.mkdtemp()}"):
      with self.subTest(root=root):
        old, new = (Store(root, cache=tempfile.mkdtemp()) for _ in range(2))
        run = new_id()
        uri = f"tinker://{run}/weights/final"
        await self.bus.create_run(Run(run, "base", 4))
        await self.bus.touch_leases("previous")
        stale, _, _ = await self.bus.acquire_run(run, "previous")
        await self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
        first = await self.bus.take_trainer_command(run, stale)
        abandoned = await self.bus.begin_checkpoint(run, stale)
        await self.bus.drop_lease("previous")
        await self.bus.touch_leases("current")
        owner, _, _ = await self.bus.acquire_run(run, "current")
        await self.bus.fail_stranded_commands(run, owner)
        await self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
        cmd = await self.bus.take_trainer_command(run, owner)
        version = await self.bus.begin_checkpoint(run, owner)
        winner = new.put({"w": torch.ones(2)}, version=version)
        await self.bus.sweep_checkpoints(new, grace=0)  # protected even before the reply publishes it
        good = {"uri": uri, "size": 8, "version": winner}
        reply = Reply(True, "application/json", b"{}")
        self.assertTrue((await self.bus.commit_reply(cmd, reply, owner, checkpoint=good))[0])
        new.upload_pending_saves()
        late = old.put({"w": torch.zeros(2)}, version=abandoned)
        old.upload_pending_saves()
        self.assertFalse((await self.bus.commit_reply(first, reply, stale, checkpoint=good | {"version": late}))[0])
        reader = Store(root, cache=tempfile.mkdtemp())
        published = await self.bus.checkpoint_version(uri)
        self.assertEqual(published, winner)
        torch.testing.assert_close(reader.get(published)[0]["w"], torch.ones(2))
        await self.bus.sweep_checkpoints(reader, grace=600)
        self.assertEqual(reader.versions(), {abandoned, version})
        await self.bus.sweep_checkpoints(reader, grace=0)
        self.assertEqual(reader.versions(), {version})
        await self.bus.unregister_checkpoints(uri)
        await self.bus.sweep_checkpoints(reader, grace=0)
        self.assertEqual(reader.versions(), set())
        await self.bus.drop_lease("current")

  async def test_sweep_removes_only_unregistered_staged_uploads(self):
    store = Store(f"memory://tinyrl-test/{uuid.uuid4()}", cache=tempfile.mkdtemp())
    kept, dropped = (store.put({"w": torch.ones(1)}) for _ in range(2))
    await self.seed_checkpoint("tinker://r/weights/kept", 1, version=kept)
    await self.bus.sweep_checkpoints(store, 0)
    self.assertTrue(os.path.isdir(f"{store.cache}/{kept}"))
    self.assertTrue(os.path.exists(f"{store.cache}/{kept}.upload"))
    self.assertFalse(os.path.exists(f"{store.cache}/{dropped}"))
    self.assertFalse(os.path.exists(f"{store.cache}/{dropped}.upload"))


if __name__ == "__main__":
  unittest.main()
