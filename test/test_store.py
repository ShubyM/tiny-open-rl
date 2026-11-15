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
    folder = store.path(uri)
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
    saved = saver.put(uri, {"w": torch.ones(4)}, {"lora": True, "config": {"r": 4}})
    saver.upload_pending_saves()  # what start() does in the background

    folder, version = loader.local_folder(uri, saved)
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
    store.put(uri, want, {"optimizer": True, "lora": False})
    folder = store.path(uri)
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
    sampler_folder = store.path(uri + "-sampler")
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
    huge_folder = store.path(uri + "-huge")
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
    v1 = store.put(uri, {"a": ones, "b": ones * 2, "c": ones * 3})
    p1 = os.path.realpath(store.local_folder(uri, v1)[0])

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
      # Explicit version v1 stays independently readable across later saves until explicitly deleted.
      self.assertTrue(os.path.exists(p1))
      self.assertTrue(torch.equal(store.get(uri, v1)[0]["c"], ones * 3))
    finally:
      running = False
      t.join()
    self.assertEqual(errors, [])
    self.assertTrue(store.delete_version(uri, v1))
    self.assertFalse(os.path.exists(p1))
    with self.assertRaises(ValueError):
      store.get(uri, v1)
    self.assertTrue(torch.equal(store.get(uri)[0]["a"], ones * 6))
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

  def test_explicit_versions_stay_readable_after_later_saves_until_deleted(self):
    root = f"memory://tinyrl-test-versions-{uuid.uuid4()}"
    writer, reader = Store(root=root, cache=tempfile.mkdtemp()), Store(root=root, cache=tempfile.mkdtemp())
    uri = "tinker://r/weights/series"
    versions = []
    for i in (1, 2, 3):
      versions.append(writer.put(uri, {"w": torch.full((4,), float(i))}, {"step": i}))
      writer.upload_pending_saves()
    # Version 1 is still readable after versions 2 and 3 moved _latest forward, and read_meta avoids shard downloads.
    self.assertEqual(reader.read_meta(uri, versions[0]), {"step": 1})
    self.assertEqual(reader.cached_versions(), [])
    self.assertTrue(torch.equal(reader.get(uri, versions[0])[0]["w"], torch.full((4,), 1.0)))
    self.assertTrue(torch.equal(reader.get(uri)[0]["w"], torch.full((4,), 3.0)))
    self.assertTrue(writer.delete_version(uri, versions[0]))
    fresh = Store(root=root, cache=tempfile.mkdtemp())
    with self.assertRaisesRegex(ValueError, "has not landed"):
      fresh.get(uri, versions[0])
    self.assertTrue(torch.equal(fresh.get(uri, versions[1])[0]["w"], torch.full((4,), 2.0)))

  def test_legacy_layouts_still_read_and_fail_explicitly_on_unknown_version(self):
    import json

    import safetensors.torch

    # Legacy remote layout: files in <uri>/<ver>/ + _latest, with no <ver>.json manifest.
    root = f"memory://tinyrl-legacy-{uuid.uuid4()}"
    store = Store(root=root, cache=tempfile.mkdtemp())
    uri, dest = "tinker://r/weights/legacy", store.path("tinker://r/weights/legacy")
    tmp = tempfile.mkdtemp()
    safetensors.torch.save_file(
      {"w": torch.tensor([7.0])}, f"{tmp}/state.safetensors", metadata={"meta": '{"base":"m"}'}
    )
    store.fs.put_file(f"{tmp}/state.safetensors", f"{dest}/oldver/state.safetensors")
    store.fs.pipe_file(
      f"{dest}/_latest",
      json.dumps({"version": "oldver", "dir": "r/weights/legacy/oldver", "files": ["state.safetensors"]}).encode(),
    )
    self.assertEqual(store.read_meta(uri), {"base": "m"})
    self.assertTrue(torch.equal(Store(root, cache=tempfile.mkdtemp()).get(uri, "oldver")[0]["w"], torch.tensor([7.0])))
    with self.assertRaisesRegex(ValueError, "has not landed"):
      Store(root, cache=tempfile.mkdtemp()).get(uri, "otherver")

    # Legacy local layout: plain folder holding state.safetensors, no manifest.
    local = Store(root=tempfile.mkdtemp())
    ldir = local.path(uri)
    os.makedirs(ldir)
    safetensors.torch.save_file(
      {"w": torch.tensor([9.0])}, f"{ldir}/state.safetensors", metadata={"meta": '{"ok":true}'}
    )
    got, meta = local.get(uri)
    self.assertEqual(meta, {"ok": True})
    self.assertTrue(torch.equal(got["w"], torch.tensor([9.0])))
    with self.assertRaisesRegex(ValueError, "not found"):
      local.get(uri, "missing-legacy")

  def test_priority_uploads_outrun_ordinary_saves_without_moving_latest_backward(self):
    root = f"memory://tinyrl-prio-{uuid.uuid4()}"
    store = Store(root=root, cache=tempfile.mkdtemp())
    uri = "tinker://r/weights/p"
    ordinary = store.put(uri, {"w": torch.tensor([1.0])}, priority=False)
    time.sleep(0.01)
    urgent = store.put(uri, {"w": torch.tensor([2.0])}, priority=True)
    order = []
    real_upload = store.upload_version

    def record(u, v):
      order.append(v)
      return real_upload(u, v)

    store.upload_version = record
    store.upload_pending_saves()
    self.assertEqual(order, [urgent, ordinary])
    # Even though `ordinary` finished uploading after `urgent`, _latest stays at the newer `urgent`.
    self.assertEqual(Store(root, cache=tempfile.mkdtemp()).get(uri)[0]["w"].item(), 2.0)


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


SOURCE_SUBPROCESS = """
import json, sys, torch
from tinyrl.storage.store import Store
root, cache = sys.argv[1], sys.argv[2]
store = Store(root, cache=cache)
store.start()
store.stopping.set()
store.wake.set()
store.uploader.join()
print(json.dumps({"peer": store.peer}), flush=True)
for line in sys.stdin:
  cmd = json.loads(line)
  if cmd["op"] == "put":
    ref = store.put(cmd["uri"], {"w": torch.full((cmd.get("n", 256),), float(cmd["val"]))}, {"tag": cmd["tag"]})
    print(json.dumps({"ref": ref, "durable": store.is_durable(cmd["uri"], ref)}), flush=True)
  elif cmd["op"] == "upload":
    ok = store.upload_version(cmd["uri"], cmd["version"])
    print(json.dumps({"ok": ok, "durable": store.is_durable(cmd["uri"], cmd["version"])}), flush=True)
"""

READER_SUBPROCESS = """
import json, sys, torch
from tinyrl.storage.store import Store
root, cache, uri, version, peer = sys.argv[1:6]
store = Store(root, cache=cache)
with store.hold(uri, version, (peer,) if peer else ()) as (folder, ver):
  print(json.dumps({"locked": folder, "version": ver}), flush=True)
  sys.stdin.readline()
  got, meta = store.get(uri, ver)
  print(json.dumps({"val": got["w"][0].item(), "len": got["w"].numel(), "meta": meta}), flush=True)
"""


class TestStorageLifecycle(RedisTest):
  """Scenario 1: real subprocesses, peer HTTP transfer, file:// durable backend, reader protection and sweep."""

  def spawn(self, code: str, *args: str, env: dict | None = None) -> subprocess.Popen:
    proc = subprocess.Popen(
      [sys.executable, "-c", code, *args],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      text=True,
      env={**os.environ, **(env or {})},
    )

    def stop():
      if proc.poll() is None:
        proc.kill()
        proc.wait()
      for pipe in (proc.stdin, proc.stdout):
        if pipe:
          pipe.close()

    self.addCleanup(stop)
    return proc

  async def test_versions_durability_reader_protection_and_incomplete_publication(self):
    import json

    with tempfile.TemporaryDirectory() as tmp:
      root = f"file://{tmp}/bucket"
      src_cache, dst_cache = f"{tmp}/src", f"{tmp}/dst"
      uri = "tinker://job-1/weights/state"

      # 1. Incomplete local/peer copy held before manifest publication is never read or served.
      dst = Store(root, cache=dst_cache)
      src = self.spawn(SOURCE_SUBPROCESS, root, src_cache)
      peer = json.loads(await asyncio.to_thread(src.stdout.readline))["peer"]
      os.makedirs(f"{src_cache}/halfver", exist_ok=True)
      with open(f"{src_cache}/halfver/state.safetensors", "wb") as f:
        f.write(b"partial")
      with self.assertRaisesRegex(ValueError, "has not landed"):
        dst.get(uri, f"halfver@{peer}")

      # 2. Publish A and B under the same URI; A is readable from peer before upload and is NOT claimed durable.
      src.stdin.write(json.dumps({"op": "put", "uri": uri, "val": 11.0, "tag": "A", "n": 4096}) + "\n")
      src.stdin.flush()
      pub_a = json.loads(await asyncio.to_thread(src.stdout.readline))
      self.assertFalse(pub_a["durable"])
      ver_a = await self.registry.register_checkpoint(
        uri, 16384, version=pub_a["ref"], meta={"tag": "A"}, durable=False
      )
      self.assertFalse((await self.registry.get_version(ver_a))["durable"])

      # Peer-transferred copy gives availability on dst, still NOT durable and NOT evictable by trim_cache.
      reader = self.spawn(READER_SUBPROCESS, root, dst_cache, uri, ver_a, peer)
      held = json.loads(await asyncio.to_thread(reader.stdout.readline))
      self.assertEqual(held["version"], ver_a)
      self.assertFalse(dst.is_durable(uri, ver_a))
      self.assertEqual(dst.trim_cache(limit=1), [])
      await self.registry.set_location(ver_a, "dst-peer", True)

      # Publish B on the same URI, overwriting the logical checkpoint name (moving _latest to B once uploaded).
      src.stdin.write(json.dumps({"op": "put", "uri": uri, "val": 22.0, "tag": "B", "n": 4096}) + "\n")
      src.stdin.flush()
      pub_b = json.loads(await asyncio.to_thread(src.stdout.readline))
      ver_b = await self.registry.register_checkpoint(
        uri, 16384, version=pub_b["ref"], meta={"tag": "B"}, durable=False
      )
      self.assertNotEqual(ver_a, ver_b)

      # Commit A and B durably, plus C (kept strictly non-durable to verify honesty on source death).
      for v in (ver_a, ver_b):
        src.stdin.write(json.dumps({"op": "upload", "uri": uri, "version": v}) + "\n")
        src.stdin.flush()
        up = json.loads(await asyncio.to_thread(src.stdout.readline))
        self.assertTrue(up["ok"] and up["durable"])
        await self.registry.mark_durable(v, peer)
      self.assertTrue(dst.is_durable(uri, ver_a))  # marks local copy durable now that bucket has A.json

      ephem_uri = "tinker://job-1/sampler_weights/ephem"
      src.stdin.write(json.dumps({"op": "put", "uri": ephem_uri, "val": 33.0, "tag": "C"}) + "\n")
      src.stdin.flush()
      pub_c = json.loads(await asyncio.to_thread(src.stdout.readline))
      ver_c = await self.registry.register_checkpoint(ephem_uri, 1024, version=pub_c["ref"], durable=False)
      self.assertFalse((await self.registry.get_version(ver_c))["durable"])

      # 3. Terminate source process abruptly: A and B survive from durable backend (A still reads despite B being _latest);
      # non-durable C never advertised durability and fails cleanly.
      src.kill()
      await asyncio.to_thread(src.wait)
      fresh = Store(root, cache=f"{tmp}/fresh")
      got_a, meta_a = fresh.get(uri, ver_a)
      self.assertEqual((got_a["w"][0].item(), meta_a), (11.0, {"tag": "A"}))
      got_latest, meta_latest = fresh.get(uri)
      self.assertEqual((got_latest["w"][0].item(), meta_latest), (22.0, {"tag": "B"}))
      with self.assertRaisesRegex(ValueError, "has not landed"):
        fresh.get(ephem_uri, f"{ver_c}@{peer}")

      # 4. Active reader protection (both cluster ref and cross-process flock) defeats concurrent sweep + cache eviction.
      # ver_a already lost its ckpt:<uri> ref when B overwrote it; hold a cluster reader ref and sweep.
      self.assertIsNotNone(await self.registry.acquire_ref(ver_a, "read:active", ttl=60))
      await self.registry.sweep(dst, grace=0, live_peers=set())
      self.assertEqual((await self.registry.get_version(ver_a))["state"], "live")
      # Even with cluster ref dropped, the reader subprocess still holds OS flock(LOCK_SH) on dst_cache:
      # neither tight trim_cache(limit=1) nor sweep may delete the files out from under it.
      await self.registry.release_ref(ver_a, "read:active")
      self.assertEqual(dst.trim_cache(limit=1), [])
      await self.registry.sweep(dst, grace=0, live_peers=set())
      # Deletion has been claimed in Redis, so new reference acquisition must fail immediately.
      self.assertIsNone(await self.registry.acquire_ref(ver_a, "read:late", ttl=60))
      # Yet the already-running reader subprocess finishes reading A cleanly!
      reader.stdin.write("go\n")
      reader.stdin.flush()
      done = json.loads(await asyncio.to_thread(reader.stdout.readline))
      self.assertEqual((done["val"], done["len"], done["meta"]), (11.0, 4096, {"tag": "A"}))
      self.assertEqual(await asyncio.to_thread(reader.wait, 10), 0)

      # Once the reader exits, sweep deletes A's local and remote files while leaving B completely intact.
      await self.registry.sweep(dst, grace=0, live_peers=set())
      self.assertIsNone(await self.registry.get_version(ver_a))
      self.assertFalse(os.path.exists(f"{dst_cache}/{ver_a}"))
      with self.assertRaisesRegex(ValueError, "has not landed"):
        Store(root, cache=f"{tmp}/after").get(uri, ver_a)
      self.assertEqual(Store(root, cache=f"{tmp}/after").get(uri, ver_b)[0]["w"][0].item(), 22.0)

      # 5. Previous durable recovery point stays protected while its replacement is only local, and releases once durable.
      await self.registry.acquire_ref(ver_b, "job:j:durable")
      await self.registry.unregister_checkpoints(uri)  # only recovery ref keeps ver_b now
      ver_d = dst.put(uri, {"w": torch.full((16,), 44.0)}, {"tag": "D"})
      ver_d_id = await self.registry.publish_version(uri, ver_d, 64, durable=False, refs={"job:j:work": 0.0})
      await self.registry.sweep(dst, grace=0, live_peers=set())
      self.assertEqual(Store(root, cache=f"{tmp}/rec1").get(uri, ver_b)[0]["w"][0].item(), 22.0)
      dst.upload_version(uri, ver_d_id)
      await self.registry.mark_durable(ver_d_id)
      await self.registry.acquire_ref(ver_d_id, "job:j:durable")
      await self.registry.release_ref(ver_b, "job:j:durable")
      await self.registry.sweep(dst, grace=0, live_peers=set())
      self.assertIsNone(await self.registry.get_version(ver_b))
      self.assertEqual(Store(root, cache=f"{tmp}/rec2").get(uri, ver_d_id)[0]["w"][0].item(), 44.0)

  async def test_single_flight_lane_priority_and_prompt_prefetch_cancellation(self):
    import threading

    from tinyrl.storage.store import TransferCancelled, split_version

    with tempfile.TemporaryDirectory() as tmp:
      root = f"file://{tmp}/bucket"
      writer = Store(root, cache=f"{tmp}/writer")
      writer.start()
      try:
        # 1. Single-flight deduplication: 4 concurrent readers of the same remote version copy from peer once.
        uri_sf = "tinker://job/weights/shared"
        v_sf, peer = split_version(writer.put(uri_sf, {"w": torch.arange(65536, dtype=torch.float32)}, {"tag": "sf"}))
        reader = Store(root, cache=f"{tmp}/reader")
        calls, real_copy = 0, reader._copy_from_peer

        def counted_copy(*args, **kwargs):
          nonlocal calls
          calls += 1
          time.sleep(0.05)
          return real_copy(*args, **kwargs)

        reader._copy_from_peer = counted_copy
        folders = await asyncio.gather(
          *(asyncio.to_thread(reader.local_folder, uri_sf, v_sf, (peer,), lane="required") for _ in range(4))
        )
        self.assertEqual(len(set(folders)), 1)
        self.assertEqual(calls, 1)
        self.assertGreater(reader.cache_status()["transferred_bytes"], 0)

        # 2. Prompt chunk-level prefetch cancellation stops mid-transfer, leaves no partial files, and records cancelled_bytes;
        # while the prefetch lane is completely saturated, the `required` lane proceeds immediately without blocking.
        uri_pre = "tinker://job/weights/prefetch"
        uri_req = "tinker://job/weights/required"
        v_pre, _ = split_version(writer.put(uri_pre, {"w": torch.ones(1_048_576, dtype=torch.float32)}, {"tag": "pre"}))
        v_req, _ = split_version(writer.put(uri_req, {"w": torch.full((1024,), 9.0)}, {"tag": "req"}))

        dst = Store(root, cache=f"{tmp}/dst")
        dst.lanes["prefetch"] = threading.BoundedSemaphore(1)
        started, allow_next, cancel = threading.Event(), threading.Event(), threading.Event()
        orig_chunk = dst._on_chunk

        def gated_chunk(tr, ev, n):
          orig_chunk(tr, ev, n)
          if ev is cancel and n > 0 and not started.is_set():
            started.set()
            allow_next.wait(timeout=5)
            orig_chunk(tr, ev, 0)

        dst._on_chunk = gated_chunk
        prefetch_task = asyncio.create_task(
          asyncio.to_thread(dst.local_folder, uri_pre, v_pre, (peer,), cancel, "prefetch")
        )
        self.assertTrue(await asyncio.to_thread(started.wait, 5))
        # Required lane completes cleanly while prefetch lane is completely saturated.
        req_folder, req_ver = await asyncio.to_thread(dst.local_folder, uri_req, v_req, (peer,), None, "required")
        self.assertEqual(req_ver, v_req)
        self.assertTrue(os.path.isdir(req_folder))
        self.assertEqual(dst.get(uri_req, v_req)[0]["w"][0].item(), 9.0)

        # Cancel the obsolete prefetch mid-stream: raises TransferCancelled promptly and leaves zero partial artifacts.
        cancel.set()
        allow_next.set()
        with self.assertRaises(TransferCancelled):
          await prefetch_task
        self.assertGreater(dst.cache_status()["cancelled_bytes"], 0)
        self.assertFalse(os.path.exists(f"{tmp}/dst/{v_pre}"))
        self.assertEqual([f for f in os.listdir(f"{tmp}/dst") if v_pre in f and ".tmp" in f], [])
      finally:
        writer.close(False)


if __name__ == "__main__":
  unittest.main()
