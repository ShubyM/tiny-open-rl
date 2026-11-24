import os
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid

import torch

from test.helpers import BusTest, new_id
from tinyrl.storage.store import SHARD_BYTES, Store
from tinyrl.types import Checkpoint, Done, Effect, NodeMsg, Op, Reply, Run, TaskKind


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

    self.assertEqual(loader.read_meta(saved), {"lora": True, "config": {"r": 4}})
    self.assertFalse(os.path.exists(f"{loader.cache}/{saved}"))
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
    saver.peer = f"127.0.0.1:{saver.server.server_port}"  # Both test nodes share this host.
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


class TestRegistry(BusTest):
  async def test_preexisting_checkpoint_indexes_support_reads_and_expiry(self):
    # Populate the persisted checkpoint table directly, without using the writer under test.
    saved = Checkpoint("tinker://r/weights/saved", "saved", 1, time=1.0)
    other = Checkpoint("tinker://r/weights/keep", "keep", 1, time=1.0)
    with self.bus.db:
      for row in (saved, other):
        self.bus.put_checkpoint(row)
    self.assertEqual(self.bus.checkpoint_version(saved.uri), "saved")
    self.assertEqual(self.bus.checkpoint_version(other.uri), "keep")
    self.assertTrue(self.bus.set_checkpoint_ttl(saved.uri, -1))
    self.bus.maintain()
    self.assertIsNone(self.bus.get_checkpoint(saved.uri))
    self.assertEqual(self.bus.get_checkpoint(other.uri), other)

  async def test_deletion_wins_over_concurrent_ttl_updates(self):
    run = new_id()
    self.bus.create_run(Run(id=run, base="base", rank=8))
    uri = f"tinker://{run}/sampler_weights/session"
    self.seed_checkpoint(uri, 1, ttl=60, owned=True)
    self.bus.submit_command(run, Op.DELETE_MODEL, b"{}")
    [t] = [x for x in self.bus.handle(NodeMsg("hello", "n1", "b1", gpus=1)).tasks if x.kind == TaskKind.TRAIN]
    for _ in range(20):
      self.bus.set_checkpoint_ttl(uri, 600)
    self.bus.handle(NodeMsg("events", "n1", "b1", done=(Done(t.id, True, Reply(True, "application/json", b"{}")),)))
    self.assertFalse(self.bus.set_checkpoint_ttl(uri, 600))
    self.assertIsNone(self.bus.get_checkpoint(uri))

  async def test_stale_writer_cannot_replace_published_bytes_on_disk_or_in_bucket(self):

    def sweep(store: Store, grace: float) -> None:
      unused = store.versions()
      unused.difference_update(self.bus.live_versions())
      store.sweep(unused, grace)

    for root in (tempfile.mkdtemp(), f"file://{tempfile.mkdtemp()}"):
      with self.subTest(root=root):
        old, new = (Store(root, cache=tempfile.mkdtemp()) for _ in range(2))
        run, b1, b2 = new_id(), uuid.uuid4().hex[:8], uuid.uuid4().hex[:8]
        uri = f"tinker://{run}/weights/final"
        self.bus.create_run(Run(run, "base", 4))
        self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
        [t1] = [x for x in self.bus.handle(NodeMsg("hello", "n1", b1, gpus=1)).tasks if x.kind == TaskKind.TRAIN]
        abandoned = t1.version
        self.bus.nodes["n1"].until = 0.0
        self.bus.maintain()

        self.bus.submit_command(run, Op.SAVE_WEIGHTS, b"{}")
        [t2] = [x for x in self.bus.handle(NodeMsg("hello", "n2", b2, gpus=1)).tasks if x.kind == TaskKind.TRAIN]
        version = t2.version
        winner = new.put({"w": torch.ones(2)}, version=version)
        sweep(new, grace=0)  # protected even before the reply publishes it
        reply = Reply(True, "application/json", b"{}")
        eff_win = Effect(loaded=True, checkpoint=Checkpoint(uri, winner, 8, 1.0))
        self.bus.handle(NodeMsg("events", "n2", b2, done=(Done(t2.id, True, reply, effect=eff_win),)))
        new.upload_pending_saves()
        late = old.put({"w": torch.zeros(2)}, version=abandoned)
        old.upload_pending_saves()
        eff_late = Effect(loaded=True, checkpoint=Checkpoint(uri, late, 8, 1.0))
        res = self.bus.handle(NodeMsg("events", "n1", b1, done=(Done(t1.id, True, reply, effect=eff_late),)))
        self.assertTrue(res.reset)
        reader = Store(root, cache=tempfile.mkdtemp())
        published = self.bus.checkpoint_version(uri)
        self.assertEqual(published, winner)
        torch.testing.assert_close(reader.get(published)[0]["w"], torch.ones(2))
        sweep(reader, grace=600)
        self.assertEqual(reader.versions(), {abandoned, version})
        sweep(reader, grace=0)
        self.assertEqual(reader.versions(), {version})
        self.bus.delete_checkpoint(uri)
        sweep(reader, grace=0)
        self.assertEqual(reader.versions(), set())
        self.bus.expire_node("n2", b2)

  async def test_sweep_removes_only_unregistered_staged_uploads(self):
    store = Store(f"memory://tinyrl-test/{uuid.uuid4()}", cache=tempfile.mkdtemp())
    kept, dropped = (store.put({"w": torch.ones(1)}) for _ in range(2))
    self.seed_checkpoint("tinker://r/weights/kept", 1, version=kept)
    unused = store.versions()
    unused.difference_update(self.bus.live_versions())
    store.sweep(unused, 0)
    self.assertTrue(os.path.isdir(f"{store.cache}/{kept}"))
    self.assertTrue(os.path.exists(f"{store.cache}/{kept}.upload"))
    self.assertFalse(os.path.exists(f"{store.cache}/{dropped}"))
    self.assertFalse(os.path.exists(f"{store.cache}/{dropped}.upload"))


if __name__ == "__main__":
  unittest.main()
