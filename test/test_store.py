import json
import os
import tempfile
import threading
import unittest
import unittest.mock
import uuid

import torch

import tinyrl.store
from tinyrl.store import Store


class TestStore(unittest.TestCase):
  def test_put_and_get_roundtrip_tensors_and_metadata(self):
    store = Store(root=tempfile.mkdtemp())
    uri = "tinker://run-1/weights/step-3"
    want = {
      "w": torch.arange(64, dtype=torch.float32).reshape(8, 8),
      "b": torch.randn(8, dtype=torch.bfloat16),
      "w.step": torch.tensor(3.0),
    }
    folder = store.put(uri, want, {"optimizer": True})
    self.assertEqual(os.listdir(folder), ["state.safetensors"])

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
    saver.put(uri, {"w": torch.ones(4)}, {"lora": True, "config": {"r": 4}})

    folder, version = loader.local(uri)
    self.assertEqual(sorted(os.listdir(folder)), ["adapter_config.json", "adapter_model.safetensors"])
    self.assertTrue(torch.equal(loader.get(uri)[0]["w"], torch.ones(4)))
    self.assertEqual(loader.local(uri), (folder, version))  # downloaded once

    saver.put(uri, {"w": torch.zeros(4)}, {"lora": True, "config": {"r": 4}})  # the same name, saved again
    self.assertNotEqual(loader.local(uri)[1], version)
    self.assertTrue(torch.equal(loader.get(uri)[0]["w"], torch.zeros(4)))
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

  @unittest.mock.patch("tinyrl.store.SHARD_BYTES", 64)
  def test_shards_large_tensors(self):
    store = Store(root=tempfile.mkdtemp())
    uri = "tinker://run-1/weights/step-3"
    want = {
      "w1": torch.arange(64, dtype=torch.float32).reshape(8, 8),
      "w2": torch.randn(8, dtype=torch.bfloat16),
      "w3": torch.tensor(3.0),
    }
    folder = store.put(uri, want, {"optimizer": True, "lora": False})
    files = sorted(os.listdir(folder))
    self.assertTrue(any("state-00001-of-" in f for f in files))

    got, meta = store.get(uri)
    self.assertEqual(meta, {"optimizer": True, "lora": False})
    for k in want:
      self.assertTrue(torch.equal(got[k], want[k]))

    _, v1 = store.local(uri)
    store.put(uri, want, {"optimizer": True, "lora": False})
    _, v2 = store.local(uri)
    self.assertNotEqual(v1, v2)

    sampler_folder = store.put(uri + "-sampler", want, {"lora": False, "config": {"test": True}})
    sampler_files = sorted(os.listdir(sampler_folder))
    self.assertIn("config.json", sampler_files)
    self.assertIn("model.safetensors.index.json", sampler_files)
    self.assertTrue(any("model-00001-of-" in f for f in sampler_files))

  def test_stage_below_and_above_max_pin_bytes(self):
    old_max = tinyrl.store.MAX_PIN_BYTES
    tinyrl.store.MAX_PIN_BYTES = 64
    try:
      store = Store(root=tempfile.mkdtemp())
      tensors = {
        "scalar": torch.tensor(3.0, dtype=torch.float32),
        "f32": torch.arange(10, dtype=torch.float32),
        "i64": torch.arange(20, dtype=torch.int64),
        "bf16": torch.arange(30, dtype=torch.bfloat16),
      }
      cpu = store.stage(tensors, is_cuda=lambda _: True)
      for k, v in tensors.items():
        self.assertEqual(cpu[k].shape, v.shape)
        self.assertTrue(torch.equal(cpu[k], v))
      store.put("tinker://r/weights/chunked", cpu, {"optimizer": True})
      got, _ = store.get("tinker://r/weights/chunked")
      for k, v in tensors.items():
        self.assertEqual(got[k].shape, v.shape)
        self.assertTrue(torch.equal(got[k], v))

      tinyrl.store.MAX_PIN_BYTES = 1000000
      cpu = store.stage(tensors, is_cuda=lambda _: True)
      for k, v in tensors.items():
        self.assertEqual(cpu[k].shape, v.shape)
        self.assertTrue(torch.equal(cpu[k], v))
    finally:
      tinyrl.store.MAX_PIN_BYTES = old_max

  @unittest.mock.patch("tinyrl.store.SHARD_BYTES", 64)
  def test_remote_sharded_upload_is_atomic_and_cleans_stale_shards(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    uri = "tinker://run-1/sampler_weights/step-remote-atomic"
    ones = torch.ones(8, dtype=torch.float32)
    original_put_file = saver.fs.put_file

    def on_first_upload(*args, **kwargs):
      res = original_put_file(*args, **kwargs)
      if args[0].endswith(".safetensors") and not getattr(on_first_upload, "called", False):
        on_first_upload.called = True
        with self.assertRaises(ValueError):
          loader.local(uri)
        with self.assertRaises(ValueError):
          loader.get(uri)
      return res

    with unittest.mock.patch.object(saver.fs, "put_file", side_effect=on_first_upload):
      saver.put(uri, {"a": ones, "b": ones * 2, "c": ones * 3}, {"lora": False, "config": {"v": 1}})

    self.assertTrue(saver.fs.exists(f"{saver.path(uri)}/model-00001-of-00003.safetensors"))
    got, meta = loader.get(uri)
    self.assertEqual(meta["config"], {"v": 1})
    self.assertTrue(torch.equal(got["a"], ones))
    self.assertTrue(torch.equal(got["b"], ones * 2))
    self.assertTrue(torch.equal(got["c"], ones * 3))

    saver.put(uri, {"only": ones * 5}, {"lora": False, "config": {"v": 2}})
    got2, meta2 = loader.get(uri)
    self.assertEqual(meta2["config"], {"v": 2})
    self.assertEqual(list(got2), ["only"])
    self.assertTrue(torch.equal(got2["only"], ones * 5))
    self.assertFalse(saver.fs.exists(f"{saver.path(uri)}/model-00001-of-00003.safetensors"))
    self.assertTrue(saver.fs.exists(f"{saver.path(uri)}/model.safetensors"))

  @unittest.mock.patch("tinyrl.store.SHARD_BYTES", 64)
  def test_local_overwrite_is_atomic_and_keeps_prior_reader_files(self):
    store = Store(tempfile.mkdtemp())
    uri = "tinker://run-local/weights/atomic"
    ones = torch.ones(8, dtype=torch.float32)

    store.put(uri + ".bak", {"a": ones * 7, "b": ones * 7, "c": ones * 7})
    store.put(uri, {"a": ones, "b": ones * 2, "c": ones * 3})
    p1 = os.path.realpath(store.local(uri)[0])

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
        if step == 2:
          self.assertTrue(os.path.exists(p1), "p1 shard files stay intact for in-flight reader")
        if step == 3:
          self.assertFalse(os.path.exists(p1), "p1 is garbage-collected after two newer saves")
          parent = os.path.dirname(store.path(uri))
          atomic_entries = [
            n
            for n in os.listdir(parent)
            if (n == "atomic" or n.startswith(".atomic.")) and not n.startswith(".atomic.bak.")
          ]
          self.assertLessEqual(len(atomic_entries), 4)
    finally:
      running = False
      t.join()
    self.assertEqual(errors, [])
    self.assertTrue(
      torch.equal(store.get(uri + ".bak")[0]["a"], ones * 7), "sibling checkpoint with prefix stays intact"
    )

  def test_failed_transfers_leave_no_temp_folders(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver = Store(root, cache=tempfile.mkdtemp())
    loader = Store(root, cache=tempfile.mkdtemp())
    uri = "tinker://run/fail"

    with unittest.mock.patch.object(saver.fs, "put_file", side_effect=RuntimeError("boom")):
      with self.assertRaises(RuntimeError):
        saver.put(uri, {"w": torch.ones(4)})
    self.assertEqual(os.listdir(saver.cache), [])

    saver.put(uri, {"w": torch.ones(4)})

    with unittest.mock.patch.object(loader.fs, "get_file", side_effect=RuntimeError("boom")):
      with self.assertRaises(RuntimeError):
        loader.get(uri)
    self.assertEqual(os.listdir(loader.cache), [])

  def test_config_only_overwrite_and_empty_or_non_cpu_tensors(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    loader = Store(root, cache=tempfile.mkdtemp())
    uri = "tinker://run/config"
    loader.put(uri, {"w": torch.ones(4)}, {"lora": True, "config": {"r": 4}})
    _, v1 = loader.local(uri)
    loader.put(uri, {"w": torch.ones(4)}, {"lora": True, "config": {"r": 8}})
    f2, v2 = loader.local(uri)
    self.assertNotEqual(v1, v2)
    with open(f"{f2}/adapter_config.json") as f:
      self.assertEqual(json.load(f)["r"], 8)

    store = Store(tempfile.mkdtemp())
    store.put(uri, {"w": torch.ones(4)}, {"lora": True, "config": {"r": 4}})
    _, v1 = store.local(uri)
    store.put(uri, {"w": torch.ones(4)}, {"lora": True, "config": {"r": 8}})
    f2, v2 = store.local(uri)
    self.assertNotEqual(v1, v2)
    with open(f"{f2}/adapter_config.json") as f:
      self.assertEqual(json.load(f)["r"], 8)

    with unittest.mock.patch("tinyrl.store.SHARD_BYTES", 0):
      store.put(uri, {})
      got, _ = store.get(uri)
      self.assertEqual(got, {})

      tensors = {
        "empty": torch.empty((0, 8), dtype=torch.bfloat16),
        "scalar": torch.tensor(42.0),
        "bool": torch.tensor([True, False]),
        "sliced": torch.ones(4, 4)[::2, ::2],
      }
      store.put(uri, tensors)
      got, _ = store.get(uri)
      self.assertEqual(got["empty"].shape, (0, 8))
      self.assertEqual(got["empty"].dtype, torch.bfloat16)
      self.assertEqual(got["scalar"].item(), 42.0)
      self.assertTrue(torch.equal(got["bool"], torch.tensor([True, False])))
      self.assertTrue(torch.equal(got["sliced"], torch.ones(4, 4)[::2, ::2]))


if __name__ == "__main__":
  unittest.main()

  def test_delete_takes_every_version_and_leaves_the_neighbours(self):
    store = Store(root=tempfile.mkdtemp())
    for uri in ("tinker://r/sampler_weights/step-1", "tinker://r/sampler_weights/step-1.5", "tinker://q/weights/final"):
      store.put(uri, {"w": torch.ones(4)})
    store.put("tinker://r/sampler_weights/step-1", {"w": torch.zeros(4)})  # a second version, so there's a .prev
    self.assertEqual(
      sorted(store.stored()),
      ["tinker://q/weights/final", "tinker://r/sampler_weights/step-1", "tinker://r/sampler_weights/step-1.5"],
    )
    store.delete("tinker://r/sampler_weights/step-1")
    self.assertEqual(sorted(store.stored()), ["tinker://q/weights/final", "tinker://r/sampler_weights/step-1.5"])
    self.assertTrue(all("step-1.5" in n for n in os.listdir(f"{store.base}/r/sampler_weights")))  # no orphan versions
    self.assertTrue(torch.equal(store.get("tinker://r/sampler_weights/step-1.5")[0]["w"], torch.ones(4)))

  def test_a_remote_checkpoint_is_listed_and_deleted(self):
    store = Store(f"memory://tinyrl-test/{uuid.uuid4()}", cache=tempfile.mkdtemp())
    store.put("tinker://r/weights/final", {"w": torch.ones(4)})
    self.assertEqual(store.stored(), ["tinker://r/weights/final"])
    self.assertGreater(store.saved_at("tinker://r/weights/final"), 0)
    store.delete("tinker://r/weights/final")
    self.assertEqual(store.stored(), [])
