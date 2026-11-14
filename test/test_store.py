import asyncio
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid

import torch

from test.helpers import RedisTest, new_id
from tinyrl.bus import Op, Reply, Run, key
from tinyrl.storage.store import SHARD_BYTES, Store


class TestStore(unittest.TestCase):
  def test_put_and_get_roundtrip_tensors_and_metadata(self):
    store = Store(root=tempfile.mkdtemp())
    uri = "tinker://run-1/weights/step-3"
    want = {
      "w": torch.arange(64, dtype=torch.float32).reshape(8, 8),
      "b": torch.randn(8, dtype=torch.bfloat16),
      "w.step": torch.tensor(3.0),
    }
    store.put(uri, want, {"optimizer": True})
    folder, _ = store.local_folder(uri)
    self.assertEqual(sorted(os.listdir(folder)), ["manifest.json", "state.safetensors"])

    got, meta = store.get(uri)
    self.assertEqual(meta, {"optimizer": True})
    self.assertTrue(all(torch.equal(got[k], want[k]) for k in want))

  def test_missing_checkpoints_fail_cleanly(self):
    store = Store(root=tempfile.mkdtemp())
    with self.assertRaises(ValueError):
      store.get("tinker://run-1/weights/missing")

  def test_a_remote_checkpoint_saved_on_one_node_loads_on_another(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    uri = "tinker://run-1/sampler_weights/step-1"
    saved = saver.put(uri, {"w": torch.ones(4)}, {"lora": True, "config": {"r": 4}})
    saver.upload_pending_saves()  # what start() does in the background

    folder, version = loader.local_folder(uri, saved)
    self.assertEqual(sorted(os.listdir(folder)), ["adapter_config.json", "adapter_model.safetensors", "manifest.json"])
    self.assertTrue(torch.equal(loader.get(uri)[0]["w"], torch.ones(4)))
    self.assertEqual(loader.local_folder(uri), (folder, version))  # downloaded once

    v2 = saver.put(uri, {"w": torch.zeros(4)}, {"lora": True, "config": {"r": 4}})  # the same name, saved again
    saver.upload_pending_saves()
    self.assertNotEqual(loader.local_folder(uri)[1], version)
    self.assertTrue(torch.equal(loader.get(uri)[0]["w"], torch.zeros(4)))
    # Explicit older version A remains independently readable after B becomes _latest.
    self.assertTrue(torch.equal(loader.get(uri, saved)[0]["w"], torch.ones(4)))
    self.assertTrue(torch.equal(loader.get(uri, v2)[0]["w"], torch.zeros(4)))
    with self.assertRaises(ValueError):
      loader.get("tinker://run-1/weights/missing")

  def test_roundtrips_contiguous_and_non_contiguous_tensors(self):
    store = Store(root=tempfile.mkdtemp())
    uri = "tinker://run-1/weights/shape-test"
    base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    want = {
      "contiguous": base.clone(),
      "transposed": base.T,
    }
    store.put(uri, want)
    got, _ = store.get(uri)
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
    uri = "tinker://run-1/weights/step-3"
    want = {
      "w1": torch.arange(64, dtype=torch.float32).reshape(8, 8),
      "w2": torch.randn(8, dtype=torch.bfloat16),
      "w3": torch.tensor(3.0),
    }
    store.put(uri, want, {"optimizer": True, "lora": False})
    folder, _ = store.local_folder(uri)
    files = sorted(os.listdir(folder))
    self.assertTrue(any("state-00001-of-" in f for f in files))

    got, meta = store.get(uri)
    self.assertEqual(meta, {"optimizer": True, "lora": False})
    for k in want:
      self.assertTrue(torch.equal(got[k], want[k]))

    _, v1 = store.local_folder(uri)
    store.put(uri, want, {"optimizer": True, "lora": False})
    _, v2 = store.local_folder(uri)
    self.assertNotEqual(v1, v2)

    store.put(uri + "-sampler", want, {"lora": False, "config": {"test": True}})
    sampler_folder, _ = store.local_folder(uri + "-sampler")
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
    store.put(uri + "-huge", want_huge, {"lora": False, "config": {"test": True}})
    huge_folder, _ = store.local_folder(uri + "-huge")
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

  def test_local_overwrite_is_atomic_and_keeps_prior_reader_files(self):
    store = Store(tempfile.mkdtemp())
    uri = "tinker://run-local/weights/atomic"
    ones = torch.ones(8, dtype=torch.float32)

    store.put(uri + ".bak", {"a": ones * 7, "b": ones * 7, "c": ones * 7})
    store.put(uri, {"a": ones, "b": ones * 2, "c": ones * 3})
    p1 = os.path.realpath(store.local_folder(uri)[0])

    errors = []
    running = True

    def reader():
      while running:
        try:
          got, _ = store.get(uri)
          val_a = got["a"][0].item()
          self.assertTrue(torch.equal(got["a"], ones * val_a))
          self.assertTrue(torch.equal(got["b"], ones * (val_a * 2 if val_a == 1 else val_a)))
          self.assertTrue(torch.equal(got["c"], ones * (val_a * 3 if val_a == 1 else val_a)))
        except Exception as e:
          errors.append(e)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
      for step in range(2, 7):
        store.put(uri, {"a": ones * step, "b": ones * step, "c": ones * step})
        self.assertTrue(os.path.exists(p1), "prior immutable version remains readable until reference-counted sweep")
    finally:
      running = False
      t.join()
    self.assertEqual(errors, [])
    self.assertTrue(
      torch.equal(store.get(uri + ".bak")[0]["a"], ones * 7), "sibling checkpoint with prefix stays intact"
    )

  def test_delete_takes_every_version_and_leaves_the_neighbours(self):
    store = Store(root=tempfile.mkdtemp())
    for uri in ("tinker://r/sampler_weights/step-1", "tinker://r/sampler_weights/step-1.5", "tinker://q/weights/final"):
      store.put(uri, {"w": torch.ones(4)})
    store.put("tinker://r/sampler_weights/step-1", {"w": torch.zeros(4)})  # a second version, so there's a .prev
    self.assertEqual(
      sorted(store.list_checkpoints()),
      ["tinker://q/weights/final", "tinker://r/sampler_weights/step-1", "tinker://r/sampler_weights/step-1.5"],
    )
    store.delete("tinker://r/sampler_weights/step-1")
    self.assertEqual(
      sorted(store.list_checkpoints()), ["tinker://q/weights/final", "tinker://r/sampler_weights/step-1.5"]
    )
    self.assertTrue(all("step-1.5" in n for n in os.listdir(f"{store.base}/r/sampler_weights")))  # no orphan versions
    self.assertTrue(torch.equal(store.get("tinker://r/sampler_weights/step-1.5")[0]["w"], torch.ones(4)))

  def test_a_remote_checkpoint_is_listed_and_deleted(self):
    store = Store(f"memory://tinyrl-test/{uuid.uuid4()}", cache=tempfile.mkdtemp())
    store.put("tinker://r/weights/final", {"w": torch.ones(4)})
    store.upload_pending_saves()
    self.assertEqual(store.list_checkpoints(), ["tinker://r/weights/final"])
    self.assertGreater(store.saved_at("tinker://r/weights/final"), 0)
    store.delete("tinker://r/weights/final")
    self.assertEqual(store.list_checkpoints(), [])

  def test_a_remote_save_returns_before_its_upload_and_other_nodes_read_it_once_it_lands(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    uri = "tinker://r/weights/final"
    version = saver.put(uri, {"w": torch.ones(4)})  # returns before anything uploads it
    self.assertTrue(torch.equal(saver.get(uri, version)[0]["w"], torch.ones(4)))  # this node reads it at once
    with self.assertRaisesRegex(ValueError, "has not landed"):
      loader.get(uri, version)
    saver.upload_pending_saves()
    self.assertTrue(torch.equal(loader.get(uri, version)[0]["w"], torch.ones(4)))

  def test_another_node_copies_a_save_from_the_node_that_made_it_before_it_uploads(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    saver.start()
    saver.stopping.set()  # no uploads, so the copy can only come from the saver
    saver.uploader.join()
    meta, w = {"lora": False, "config": {"model_type": "x"}}, torch.randn(64, 64).to(torch.bfloat16)
    versions = [saver.put(f"tinker://r/sampler_weights/{step}", {"w": w * step}, meta) for step in range(2)]
    self.assertFalse(saver.fs.exists(f"{saver.base}/r"))
    folder, version = loader.local_folder("tinker://r/sampler_weights/1", versions[1])
    self.assertEqual(version, versions[1].partition("@")[0])
    for name in os.listdir(f"{saver.cache}/{version}"):
      with open(f"{saver.cache}/{version}/{name}", "rb") as a, open(f"{folder}/{name}", "rb") as b:
        self.assertEqual(a.read(), b.read())
    self.assertTrue(torch.equal(loader.get("tinker://r/sampler_weights/0", versions[0])[0]["w"], w * 0))
    saver.close()  # its saves upload, then it stops serving them
    later = Store(root, cache=tempfile.mkdtemp())
    self.assertTrue(torch.equal(later.get("tinker://r/sampler_weights/1", versions[1])[0]["w"], w))

  def test_a_save_whose_node_is_gone_comes_from_the_bucket_or_fails(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    saver.peer = "127.0.0.1:9"  # a node that no longer answers
    uri = "tinker://r/weights/final"
    version = saver.put(uri, {"w": torch.ones(4)})
    with self.assertRaisesRegex(ValueError, "has not landed, and 127.0.0.1:9 failed to copy"):
      loader.get(uri, version)
    saver.upload_pending_saves()
    self.assertTrue(torch.equal(loader.get(uri, version)[0]["w"], torch.ones(4)))

  def test_all_published_versions_remain_readable_until_explicitly_deleted(self):
    root = f"memory://tinyrl-test-d4-count-{uuid.uuid4()}"
    store = Store(root=root, cache=tempfile.mkdtemp())
    uri = "tinker://r/weights/d4_count"

    v1 = store.put(uri, {"w": torch.ones(1)})
    store.upload_pending_saves()
    v2 = store.put(uri, {"w": torch.ones(2)})
    store.upload_pending_saves()
    v3 = store.put(uri, {"w": torch.ones(3)})
    store.upload_pending_saves()

    remote_dir = f"{store.base}/r/weights/d4_count"
    versions = [e for e in store.fs.ls(remote_dir, detail=False) if os.path.basename(e) != "_latest"]
    self.assertEqual(len(versions), 3)
    self.assertEqual([store.get(uri, v)[0]["w"].numel() for v in (v1, v2, v3)], [1, 2, 3])
    self.assertTrue(store.delete_version(uri, v1))
    versions_after = [e for e in store.fs.ls(remote_dir, detail=False) if os.path.basename(e) != "_latest"]
    self.assertEqual(len(versions_after), 2)


class TestRegistry(RedisTest):
  async def test_deletion_wins_over_concurrent_ttl_updates(self):
    run = new_id()
    await self.bus.create_run(Run(id=run, worker="p", base="base", rank=8))
    await self.bus.take_lease("p", "owner")
    uri = f"tinker://{run}/sampler_weights/session"
    await self.registry.register_checkpoint(uri, 1, ttl=60, owned=True)
    await self.bus.submit_command(run, Op.DELETE_MODEL, b"{}")
    [cmd] = await self.bus.take_commands([run], "owner", count=1)
    await asyncio.gather(
      *(self.registry.set_checkpoint_ttl(uri, 600) for _ in range(20)),
      self.bus.write_reply(cmd, Reply(True, "application/json", b"{}"), "p", "owner"),
    )
    self.assertFalse(await self.registry.set_checkpoint_ttl(uri, 600))
    self.assertIsNone(await self.registry.get_checkpoint(uri))
    for index in ("ckpts", "expiry"):
      self.assertIsNone(await self.bus.r.zscore(key(index), uri))

  async def test_sweep_keeps_registered_and_recent_checkpoints(self):
    store = Store(tempfile.mkdtemp())
    before, kept, deleted, young = [f"tinker://r/weights/{n}" for n in ("before", "kept", "deleted", "young")]
    for uri in (before, kept, deleted, young):
      store.put(uri, {"w": torch.ones(1)})
      age = 1800 if uri == before else 900 if uri != young else 0
      os.utime(store.path(uri), (time.time() - age, time.time() - age), follow_symlinks=False)
    await self.registry.register_checkpoint(kept, 1)
    await self.registry.register_checkpoint(deleted, 1)
    await self.registry.unregister_checkpoints(deleted)
    await self.bus.r.set(key("ckpt_since"), time.time() - 1200)
    await self.registry.sweep(store, 600)
    self.assertEqual(set(store.list_checkpoints()), {before, kept, young})

  async def test_sweep_deletes_an_unregistered_save_still_waiting_to_upload(self):
    store = Store(f"memory://tinyrl-test/{uuid.uuid4()}", cache=tempfile.mkdtemp())
    kept, dropped = "tinker://r/weights/kept", "tinker://r/weights/dropped"
    versions = [store.put(uri, {"w": torch.ones(1)}) for uri in (kept, dropped)]
    for version in versions:
      os.utime(f"{store.cache}/{version}.upload", (time.time() - 900, time.time() - 900))
    await self.registry.register_checkpoint(kept, 1, version=versions[0])
    await self.bus.r.set(key("ckpt_since"), time.time() - 1200)
    await self.registry.sweep(store, 600)
    self.assertTrue(os.path.isdir(f"{store.cache}/{versions[0]}"))
    self.assertTrue(os.path.exists(f"{store.cache}/{versions[0]}.upload"))
    self.assertFalse(os.path.isdir(f"{store.cache}/{versions[1]}"))
    self.assertFalse(os.path.exists(f"{store.cache}/{versions[1]}.upload"))


if __name__ == "__main__":
  unittest.main()
