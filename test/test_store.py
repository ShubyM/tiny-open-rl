import asyncio
import glob
import json
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
from tinyrl.storage.store import KEYFRAME_EVERY, SHARD_BYTES, Store


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
    saver.upload_pending_saves()  # the node daemon's job

    folder, version = loader.local_folder(uri, saver.versions[uri])
    self.assertEqual(sorted(os.listdir(folder)), ["adapter_config.json", "adapter_model.safetensors"])
    self.assertTrue(torch.equal(loader.get(uri)[0]["w"], torch.ones(4)))
    self.assertEqual(loader.local_folder(uri), (folder, version))  # downloaded once

    saver.put(uri, {"w": torch.zeros(4)}, {"lora": True, "config": {"r": 4}})  # the same name, saved again
    saver.upload_pending_saves()
    self.assertNotEqual(loader.local_folder(uri)[1], version)
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
    folder = store.put(uri, want, {"optimizer": True, "lora": False})
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

    sampler_folder = store.put(uri + "-sampler", want, {"lora": False, "config": {"test": True}})
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
    huge_folder = store.put(uri + "-huge", want_huge, {"lora": False, "config": {"test": True}})
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
    saver.put(uri, {"w": torch.ones(4)})  # returns before anything uploads it
    version = saver.versions[uri]
    self.assertTrue(saver.landed(uri, version))
    self.assertTrue(torch.equal(saver.get(uri)[0]["w"], torch.ones(4)))  # this node reads it at once
    self.assertFalse(loader.landed(uri, version))
    with self.assertRaisesRegex(ValueError, "has not landed"):
      loader.get(uri, version)
    saver.upload_pending_saves()
    self.assertTrue(loader.landed(uri, version))
    self.assertTrue(torch.equal(loader.get(uri, version)[0]["w"], torch.ones(4)))

  def test_one_runs_upload_does_not_hold_up_anothers(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    store, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    big, small = "tinker://a/weights/big", "tinker://b/weights/small"
    store.put(big, {"w": torch.ones(4)})
    store.put(small, {"w": torch.ones(4)})
    gate, upload = threading.Event(), store.upload_version
    store.upload_version = lambda dest, *args: (gate.wait(60) if "/a/" in dest else None, upload(dest, *args))
    first = threading.Thread(target=store.upload_pending_saves)
    first.start()
    for _ in range(200):  # b lands while a's upload is held
      if loader.landed(small, store.versions[small]):
        break
      time.sleep(0.05)
    self.assertEqual([loader.landed(u, store.versions[u]) for u in (small, big)], [True, False])
    store.upload_pending_saves()  # a is still uploading, so this flush leaves it to the first
    gate.set()
    first.join(60)
    self.assertTrue(loader.landed(big, store.versions[big]))

  def test_full_sampler_weights_upload_as_xors_another_node_rebuilds_exactly(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    w, meta, uris = torch.randn(128, 128).to(torch.bfloat16), {"lora": False, "config": {"model_type": "x"}}, []
    # two files, many chunks
    for step in range(3):
      w.view(-1)[step::101] *= 2  # a step changes a few values
      uris.append(f"tinker://r/sampler_weights/{step}")
      saver.put(uris[-1], {"w": w.clone(), "b": torch.ones(8)}, meta)
    saver.upload_pending_saves()
    versions = [saver.versions[u] for u in uris]
    self.assertEqual([v.rpartition(".")[2] for v in versions], ["0", "1", "2"])
    self.assertEqual(len({v.rpartition(".")[0] for v in versions}), 1)
    links = f"{saver.base}/r/_versions"
    whole, xor = (saver.fs.du(f"{links}/{v}") for v in versions[:2])
    self.assertLess(xor, whole / 4)
    self.assertTrue(all(not n.endswith(".safetensors") for n in saver.fs.ls(f"{links}/{versions[1]}", detail=False)))

    folder = loader.local_folder(uris[1], versions[1])[0]  # link 0 whole, then link 1's XOR
    saver.fs.rm(f"{links}/{versions[0]}", recursive=True)
    folder = loader.local_folder(uris[2], versions[2])[0]  # from link 1 here, so link 0 is never read again
    for name in os.listdir(f"{saver.cache}/{versions[2]}"):
      with open(f"{saver.cache}/{versions[2]}/{name}", "rb") as a, open(f"{folder}/{name}", "rb") as b:
        self.assertEqual(a.read(), b.read())
    self.assertTrue(torch.equal(loader.get(uris[2])[0]["w"], w))

  def test_chains_restart_at_keyframe_every_and_a_link_whose_parent_never_landed_goes_whole(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    meta, v = {"config": {"model_type": "x"}}, []
    for step in range(KEYFRAME_EVERY + 2):
      saver.put(f"tinker://r/sampler_weights/{step}", {"w": torch.full((64,), float(step))}, meta)
      v.append(saver.versions[f"tinker://r/sampler_weights/{step}"])
      saver.upload_pending_saves()
      if step == KEYFRAME_EVERY:
        os.unlink(f"{saver.cache}/{v[-1]}.uploaded")  # as if its upload failed
    self.assertEqual([x.rpartition(".")[2] for x in v[-2:]], ["0", "1"])
    self.assertNotEqual(v[0].rpartition(".")[0], v[-2].rpartition(".")[0])
    names = saver.fs.ls(f"{saver.base}/r/_versions/{v[-1]}", detail=False)
    self.assertTrue(any(n.endswith(".safetensors") for n in names))
    self.assertEqual(
      json.loads(saver.fs.cat_file(f"{saver.base}/r/sampler_weights/{KEYFRAME_EVERY + 1}/_latest"))["from"], v[-1]
    )
    self.assertTrue(
      torch.equal(
        loader.get(f"tinker://r/sampler_weights/{KEYFRAME_EVERY + 1}")[0]["w"],
        torch.full((64,), float(KEYFRAME_EVERY + 1)),
      )
    )
    self.assertTrue(torch.equal(loader.get("tinker://r/sampler_weights/1")[0]["w"], torch.full((64,), 1.0)))

  def test_link_0_goes_up_as_its_changes_from_the_base_model_and_another_node_rebuilds_exactly(self):
    import safetensors.torch

    root, base = f"memory://tinyrl-test/{uuid.uuid4()}", tempfile.mkdtemp()
    w = torch.randn(128, 128).to(torch.bfloat16)
    safetensors.torch.save_file({"w": w.clone(), "b": torch.ones(8)}, f"{base}/model.safetensors")
    saver = Store(root, cache=tempfile.mkdtemp())
    meta, uris = {"lora": False, "config": {"model_type": "x"}, "base": f"file://{base}"}, []
    for step in range(2):
      w.view(-1)[step::101] *= 2  # a step changes a few values
      uris.append(f"tinker://r/sampler_weights/{step}")
      # "new" isn't in the base, so it goes up as all its bytes
      saver.put(uris[-1], {"w": w.clone(), "b": torch.ones(8), "new": torch.randn(8)}, meta)
    saver.upload_pending_saves()
    versions = [saver.versions[u] for u in uris]
    self.assertEqual(json.loads(saver.fs.cat_file(f"{saver.base}/r/sampler_weights/0/_latest"))["from"], "base")
    self.assertEqual(json.loads(saver.fs.cat_file(f"{saver.base}/r/sampler_weights/1/_latest"))["from"], "base")
    whole = sum(os.path.getsize(f) for f in glob.glob(f"{saver.cache}/{versions[0]}/*.safetensors"))
    self.assertLess(saver.fs.du(f"{saver.base}/r/_versions/{versions[0]}"), whole / 4)

    for uri, version in zip(uris, versions):  # each on a node that has nothing cached
      folder = Store(root, cache=tempfile.mkdtemp()).local_folder(uri, version)[0]
      for name in os.listdir(f"{saver.cache}/{version}"):
        with open(f"{saver.cache}/{version}/{name}", "rb") as a, open(f"{folder}/{name}", "rb") as b:
          self.assertEqual(a.read(), b.read())
      self.assertEqual(glob.glob(f"{os.path.dirname(folder)}/*.base"), [])

  def test_link_0_goes_up_whole_without_the_base_model_on_the_node(self):
    root = f"memory://tinyrl-test/{uuid.uuid4()}"
    saver, loader = Store(root, cache=tempfile.mkdtemp()), Store(root, cache=tempfile.mkdtemp())
    meta = {"lora": False, "config": {"model_type": "x"}, "base": f"file://{tempfile.mkdtemp()}/missing"}
    saver.put("tinker://r/sampler_weights/0", {"w": torch.randn(64)}, meta)
    saver.upload_pending_saves()
    version = saver.versions["tinker://r/sampler_weights/0"]
    self.assertEqual(json.loads(saver.fs.cat_file(f"{saver.base}/r/sampler_weights/0/_latest"))["from"], version)
    got = loader.get("tinker://r/sampler_weights/0")[0]["w"]
    self.assertTrue(torch.equal(got, saver.get("tinker://r/sampler_weights/0")[0]["w"]))

  def test_d5_delta_cleanup_breaks_future_save(self):
    root = f"memory://tinyrl-test-d5-{uuid.uuid4()}"
    base_model_dir = tempfile.mkdtemp()
    import safetensors.torch

    safetensors.torch.save_file({"w": torch.randn(8)}, f"{base_model_dir}/model.safetensors")
    saver = Store(root=root, cache=tempfile.mkdtemp())
    uri_0, uri_1 = "tinker://r/sampler_weights/0", "tinker://r/sampler_weights/1"
    meta = {"lora": False, "config": {"model_type": "x"}, "base": f"file://{base_model_dir}"}
    saver.put(uri_0, {"w": torch.ones(8)}, meta)
    saver.upload_pending_saves()
    v0 = saver.versions[uri_0]
    saver.fs.rm(f"{saver.base}/r/_versions/{v0}", recursive=True)
    saver.put(uri_1, {"w": torch.ones(8) * 2}, meta)
    saver.upload_pending_saves()
    reader = Store(root=root, cache=tempfile.mkdtemp())
    got, _ = reader.get(uri_1)
    self.assertTrue(torch.equal(got["w"], torch.ones(8) * 2))

  def test_three_saves_leave_exactly_two_versions(self):
    root = f"memory://tinyrl-test-d4-count-{uuid.uuid4()}"
    store = Store(root=root, cache=tempfile.mkdtemp())
    uri = "tinker://r/weights/d4_count"

    store.put(uri, {"w": torch.ones(1)})
    store.upload_pending_saves()

    store.put(uri, {"w": torch.ones(2)})
    store.upload_pending_saves()

    store.put(uri, {"w": torch.ones(3)})
    store.upload_pending_saves()

    # Check that exactly two versions exist in the bucket
    remote_dir = f"{store.base}/r/weights/d4_count"
    versions = [e for e in store.fs.ls(remote_dir, detail=False) if os.path.basename(e) != "_latest"]
    self.assertEqual(len(versions), 2)


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


if __name__ == "__main__":
  unittest.main()
