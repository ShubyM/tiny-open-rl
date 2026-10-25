import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import torch

from tinyrl.store import Store


class TestStore(unittest.TestCase):
  def test_immutable_roundtrip_and_durable_acknowledgement(self):
    store = Store(root=tempfile.mkdtemp())
    want = {"w": torch.arange(64, dtype=torch.float32).reshape(8, 8), "step": torch.tensor(3.0)}
    with patch("tinyrl.store.os.fsync", wraps=os.fsync) as fsync:
      ref = store.put(want, {"optimizer": True})
    self.assertGreaterEqual(fsync.call_count, 2)  # blob contents and the directory entry
    got, meta = store.get(ref)
    self.assertEqual(meta, {"optimizer": True})
    self.assertTrue(all(torch.equal(got[k], want[k]) for k in want))
    with self.assertRaises(FileExistsError):
      store.put({"w": torch.zeros(1)}, ref=ref)
    self.assertTrue(torch.equal(store.get(ref)[0]["w"], want["w"]))
    self.assertFalse(any(name.startswith(".artifact-") for name in os.listdir(store.root)))

  def test_cache_uses_immutable_refs_and_new_reader_falls_back_to_disk(self):
    shared, cache = tempfile.mkdtemp(), tempfile.mkdtemp()
    source, other = Store(root=shared, shm=cache), Store(root=shared)
    first = source.put({"w": torch.ones(4)}, {"step": 1})
    second = other.put({"w": torch.full((4,), 2.0)}, {"step": 2})
    self.assertNotEqual(first, second)
    self.assertEqual(len(os.listdir(cache)), 1)
    self.assertTrue(torch.equal(source.get(first)[0]["w"], torch.ones(4)))
    self.assertTrue(torch.equal(source.get(second)[0]["w"], torch.full((4,), 2.0)))
    self.assertTrue(torch.equal(other.get(first)[0]["w"], torch.ones(4)))

  def test_references_cannot_name_public_paths(self):
    store = Store(root=tempfile.mkdtemp())
    for bad in ("tinker://run/weights/latest", "../secret", "/abs/path", "x" * 32):
      with self.assertRaises(ValueError):
        store.put({"w": torch.zeros(1)}, ref=bad)
    with self.assertRaises(FileNotFoundError):
      store.get("0" * 32)

  def test_cache_failure_and_corruption_fall_back_to_durable_artifact(self):
    store = Store(root=tempfile.mkdtemp(), shm=tempfile.mkdtemp())
    with patch("tinyrl.store.shutil.copyfile", side_effect=OSError("cache full")):
      ref = store.put({"w": torch.ones(2)})
    self.assertTrue(torch.equal(store.get(ref)[0]["w"], torch.ones(2)))
    pathlib.Path(store.shm, f"{ref}.safetensors").write_bytes(b"corrupt")
    self.assertTrue(torch.equal(store.get(ref)[0]["w"], torch.ones(2)))

  def test_exports_are_immutable_and_training_state_is_not_a_sampler(self):
    store = Store(root=tempfile.mkdtemp())
    meta = {"lora": True, "optimizer": False, "config": {"r": 4}}
    ref = store.put({"w": torch.ones(2)}, meta)
    folder = store.export(ref)
    self.assertEqual(store.export(ref), folder)
    self.assertTrue(os.path.isfile(f"{folder}/adapter_model.safetensors"))
    other = store.put({"w": torch.zeros(2)}, meta)
    with self.assertRaises(FileExistsError):
      store.export(other, folder)
    state = store.put({"w": torch.ones(2)}, {"optimizer": True})
    with self.assertRaises(ValueError):
      store.export(state)
