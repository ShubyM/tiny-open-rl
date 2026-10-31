import json
import os
import tempfile
import threading
import time
import unittest
import unittest.mock
import uuid
from unittest.mock import patch

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
    saver.flush()  # the node daemon's job

    folder, version = loader.local(uri, saver.versions[uri])
    self.assertEqual(sorted(os.listdir(folder)), ["adapter_config.json", "adapter_model.safetensors"])
    self.assertTrue(torch.equal(loader.get(uri)[0]["w"], torch.ones(4)))
    self.assertEqual(loader.local(uri), (folder, version))  # downloaded once

    saver.put(uri, {"w": torch.zeros(4)}, {"lora": True, "config": {"r": 4}})  # the same name, saved again
    saver.flush()
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
    uri = "tinker://run-1/weights/step-remote-atomic"  # full sampler weights upload as chains instead
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
      saver.put(uri, {"a": ones, "b": ones * 2, "c": ones * 3}, {"v": 1})
      saver.flush()
    v1 = saver.versions[uri]

    self.assertTrue(saver.fs.exists(f"{saver.path(uri)}/{v1}/state-00001-of-00003.safetensors"))
    got, meta = loader.get(uri)
    self.assertEqual(meta, {"v": 1})
    self.assertTrue(torch.equal(got["a"], ones))
    self.assertTrue(torch.equal(got["b"], ones * 2))
    self.assertTrue(torch.equal(got["c"], ones * 3))

    saver.put(uri, {"only": ones * 5}, {"v": 2})
    saver.flush()
    got2, meta2 = loader.get(uri)
    self.assertEqual(meta2, {"v": 2})
    self.assertEqual(list(got2), ["only"])
    self.assertTrue(torch.equal(got2["only"], ones * 5))
    self.assertFalse(saver.fs.exists(f"{saver.path(uri)}/{v1}"))  # the version before is gone from the bucket
    self.assertTrue(saver.fs.exists(f"{saver.path(uri)}/{saver.versions[uri]}/state.safetensors"))

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
      saver.put(uri, {"w": torch.ones(4)})  # saved here, so it returns: the upload fails behind it
      with self.assertRaises(RuntimeError):
        saver.flush()
    v = saver.versions[uri]
    self.assertEqual(sorted(os.listdir(saver.cache)), sorted([v, f"{v}.upload", "latest"]))  # no temp folders

    saver.put(uri, {"w": torch.ones(4)})
    saver.flush()  # the one that failed, then this one
    self.assertFalse([n for n in os.listdir(saver.cache) if n.endswith(".upload")])

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
    store.flush()
    self.assertEqual(store.stored(), ["tinker://r/weights/final"])
    self.assertGreater(store.saved_at("tinker://r/weights/final"), 0)
    store.delete("tinker://r/weights/final")
    self.assertEqual(store.stored(), [])

  def test_a_remote_save_returns_before_its_upload_and_other_nodes_wait_for_it(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    uri = "tinker://r/weights/final"
    saver.put(uri, {"w": torch.ones(4)})  # returns before anything uploads it
    self.assertTrue(torch.equal(saver.get(uri)[0]["w"], torch.ones(4)))  # this node reads it at once
    got = {}
    reader = threading.Thread(target=lambda: got.update(w=loader.get(uri, saver.versions[uri])[0]["w"]))
    reader.start()
    reader.join(0.5)
    self.assertTrue(reader.is_alive())  # another node waits for the version
    saver.flush()
    reader.join(10)
    self.assertTrue(torch.equal(got["w"], torch.ones(4)))

  def test_trim_drops_the_oldest_uploaded_versions_only(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    store = Store(root, cache=tempfile.mkdtemp())
    for i in range(3):
      store.put(f"tinker://r/weights/v{i}", {"w": torch.ones(1024)})
      store.flush()
      time.sleep(0.01)
    store.put("tinker://r/weights/uploading", {"w": torch.ones(1024)})
    with unittest.mock.patch("tinyrl.store.CACHE_BYTES", 1):
      store.trim()
    v = store.versions["tinker://r/weights/uploading"]
    self.assertEqual(sorted(n for n in os.listdir(store.cache) if n != "latest"), [v, f"{v}.upload"])
    self.assertTrue(torch.equal(store.get("tinker://r/weights/v0")[0]["w"], torch.ones(1024)))  # from the bucket again

  def test_full_sampler_weights_upload_as_xors_another_node_rebuilds_exactly(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    w, meta, uris = torch.randn(128, 128).to(torch.bfloat16), {"lora": False, "config": {"model_type": "x"}}, []
    with patch("tinyrl.store.SHARD_BYTES", 1024), patch("tinyrl.store.XOR_CHUNK", 4096):  # two files, many chunks
      for step in range(3):
        w.view(-1)[step::101] *= 2  # a step changes a few values
        uris.append(f"tinker://r/sampler_weights/{step}")
        saver.put(uris[-1], {"w": w.clone(), "b": torch.ones(8)}, meta)
      saver.flush()
    versions = [saver.versions[u] for u in uris]
    self.assertEqual([v.rpartition(".")[2] for v in versions], ["0", "1", "2"])
    self.assertEqual(len({v.rpartition(".")[0] for v in versions}), 1)
    links = f"{saver.base}/r/_versions"
    whole, xor = (saver.fs.du(f"{links}/{v}") for v in versions[:2])
    self.assertLess(xor, whole / 4)
    self.assertTrue(all(not n.endswith(".safetensors") for n in saver.fs.ls(f"{links}/{versions[1]}", detail=False)))

    folder = loader.local(uris[1], versions[1])[0]  # link 0 whole, then link 1's XOR
    saver.fs.rm(f"{links}/{versions[0]}", recursive=True)
    folder = loader.local(uris[2], versions[2])[0]  # from link 1 here, so link 0 is never read again
    for name in os.listdir(f"{saver.cache}/{versions[2]}"):
      with open(f"{saver.cache}/{versions[2]}/{name}", "rb") as a, open(f"{folder}/{name}", "rb") as b:
        self.assertEqual(a.read(), b.read())
    self.assertTrue(torch.equal(loader.get(uris[2])[0]["w"], w))

  def test_chains_restart_at_keyframe_every_and_a_link_whose_parent_never_landed_goes_whole(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    meta, v = {"config": {"model_type": "x"}}, []
    with patch("tinyrl.store.KEYFRAME_EVERY", 2):
      for step in range(4):
        saver.put(f"tinker://r/sampler_weights/{step}", {"w": torch.full((64,), float(step))}, meta)
        v.append(saver.versions[f"tinker://r/sampler_weights/{step}"])
        saver.flush()
        if step == 2:
          os.unlink(f"{saver.cache}/{v[2]}.uploaded")  # as if its upload failed
    self.assertEqual([x.rpartition(".")[2] for x in v], ["0", "1", "0", "1"])
    self.assertNotEqual(v[0].rpartition(".")[0], v[2].rpartition(".")[0])
    names = saver.fs.ls(f"{saver.base}/r/_versions/{v[3]}", detail=False)
    self.assertTrue(any(n.endswith(".safetensors") for n in names))
    self.assertEqual(json.loads(saver.fs.cat_file(f"{saver.base}/r/sampler_weights/3/_latest"))["from"], v[3])
    self.assertTrue(torch.equal(loader.get("tinker://r/sampler_weights/3")[0]["w"], torch.full((64,), 3.0)))
    self.assertTrue(torch.equal(loader.get("tinker://r/sampler_weights/1")[0]["w"], torch.full((64,), 1.0)))


if __name__ == "__main__":
  unittest.main()
