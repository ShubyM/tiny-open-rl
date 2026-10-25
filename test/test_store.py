import os
import tempfile
import unittest

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

  def test_shm_caches_local_reads_and_never_serves_stale_overwrites(self):
    shared, shm_a, shm_b = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
    node_a = Store(root=shared, shm=shm_a)
    node_b = Store(root=shared, shm=shm_b)
    uri = "tinker://run-1/weights/latest"

    node_a.put(uri, {"w": torch.ones(4)}, {"step": 1})
    self.assertEqual(len(os.listdir(shm_a)), 1)
    got_a, meta_a = node_a.get(uri)
    self.assertEqual(meta_a, {"step": 1})
    self.assertTrue(torch.equal(got_a["w"], torch.ones(4)))

    # node_b overwrites the same URI; node_a must see step 2 instead of its old shm_a blob
    node_b.put(uri, {"w": torch.full((4,), 2.0)}, {"step": 2})
    got_after, meta_after = node_a.get(uri)
    self.assertEqual(meta_after, {"step": 2})
    self.assertTrue(torch.equal(got_after["w"], torch.full((4,), 2.0)))

  def test_bad_uris_and_missing_checkpoints_fail_cleanly(self):
    store = Store(root=tempfile.mkdtemp())
    for bad in ("tinker://only-two/parts", "tinker://a/../b/c", "/abs/path/x"):
      with self.assertRaises(ValueError):
        store.put(bad, {"w": torch.zeros(1)})
    with self.assertRaises(ValueError):
      store.get("tinker://run-1/weights/missing")


if __name__ == "__main__":
  unittest.main()
