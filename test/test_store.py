import os
import tempfile
import unittest
import unittest.mock
import uuid

import torch

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
    import tinyrl.store

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
      cpu = store._stage(tensors, is_cuda=lambda _: True, sync=lambda: None)
      for k, v in tensors.items():
        self.assertEqual(cpu[k].shape, v.shape)
        self.assertTrue(torch.equal(cpu[k], v))
      store.put("tinker://r/weights/chunked", cpu, {"optimizer": True})
      got, _ = store.get("tinker://r/weights/chunked")
      for k, v in tensors.items():
        self.assertEqual(got[k].shape, v.shape)
        self.assertTrue(torch.equal(got[k], v))

      tinyrl.store.MAX_PIN_BYTES = 1000000
      cpu = store._stage(tensors, is_cuda=lambda _: True, sync=lambda: None)
      for k, v in tensors.items():
        self.assertEqual(cpu[k].shape, v.shape)
        self.assertTrue(torch.equal(cpu[k], v))
    finally:
      tinyrl.store.MAX_PIN_BYTES = old_max


if __name__ == "__main__":
  unittest.main()
