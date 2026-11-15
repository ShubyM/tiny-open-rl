import asyncio
import concurrent.futures
import errno
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid

import torch

from test.helpers import RedisTest
from tinyrl.bus import Op, Run, key
from tinyrl.storage.store import SHARD_BYTES, Store, manifest, split_version


class TestStore(unittest.TestCase):
  def test_exact_versions_roundtrip_tensor_layouts(self):
    tensors = {
      "scalar": torch.tensor(3.0),
      "empty": torch.zeros(0, 4),
      "bf16": torch.arange(8, dtype=torch.bfloat16),
      "i64": torch.arange(20).view(4, 5).t(),
      "bool": torch.tensor([True, False]),
    }
    for remote in (False, True):
      with self.subTest(remote=remote), tempfile.TemporaryDirectory() as tmp:
        root = f"memory://{uuid.uuid4().hex}" if remote else tmp
        store = Store(root, cache=f"{tmp}/cache")
        uri = "tinker://r/weights/test"
        first = store.put(uri, tensors, {"optimizer": True})
        self.assertEqual(store.is_durable(uri, first), not remote)
        store.upload_pending_saves()
        second = store.put(uri, {"w": torch.ones(1)})
        store.upload_pending_saves()
        self.assertNotEqual(first, second)
        reader = Store(root, cache=f"{tmp}/reader")
        self.assertEqual(reader.read_meta(uri, first), {"optimizer": True})
        self.assertEqual(bool(manifest(reader.folder(first))), not remote)
        got, meta = reader.get(uri, first)
        self.assertEqual(meta, {"optimizer": True})
        for name, want in tensors.items():
          self.assertEqual((got[name].shape, got[name].dtype), (want.shape, want.dtype))
          self.assertTrue(torch.equal(got[name], want))
        self.assertTrue(reader.is_durable(uri, first))
        self.assertEqual(reader.get(uri, second)[0]["w"].item(), 1.0)
        with self.assertRaises(ValueError):
          reader.get(uri, None)

  def test_shards_keep_engine_layout(self):
    if SHARD_BYTES != 64:
      subprocess.run(
        [sys.executable, "-m", "unittest", f"{__name__}.TestStore.test_shards_keep_engine_layout"],
        env=os.environ | {"SHARD_BYTES": "64"},
        check=True,
      )
      return
    with tempfile.TemporaryDirectory() as tmp:
      store, uri = Store(tmp), "tinker://r/sampler_weights/test"
      want = {"a": torch.ones(128), "b": torch.zeros(128), "c": torch.arange(8.0)}
      version = store.put(uri, want, {"lora": False, "config": {"model_type": "test"}})
      with store.hold(uri, version) as folder:
        names = os.listdir(folder)
        self.assertIn("config.json", names)
        self.assertIn("model.safetensors.index.json", names)
        self.assertTrue(any(n.startswith("model-00001-of-") for n in names))
      got, _ = store.get(uri, version)
      self.assertTrue(all(torch.equal(got[k], v) for k, v in want.items()))

  def test_concurrent_cache_publication_and_locked_reader(self):
    with tempfile.TemporaryDirectory() as tmp:
      uri, root = "tinker://r/weights/test", f"file://{tmp}/bucket"
      source = Store(root, cache=f"{tmp}/source")
      want = torch.arange(256 * 1024)
      version = source.put(uri, {"w": want})
      with concurrent.futures.ThreadPoolExecutor(2) as pool:
        self.assertEqual(list(pool.map(lambda _: source.upload_version(uri, version), range(2))), [True, True])
      readers = [Store(root, cache=f"{tmp}/readers") for _ in range(8)]
      ready = threading.Barrier(len(readers))

      def read(store):
        ready.wait(timeout=5)
        return store.get(uri, version)[0]["w"]

      with concurrent.futures.ThreadPoolExecutor(len(readers)) as pool:
        self.assertTrue(all(torch.equal(v, want) for v in pool.map(read, readers)))
      with readers[0].hold(uri, version):
        self.assertFalse(readers[1].delete_version(uri, version))
        self.assertFalse(readers[1]._evict(version, durable=True))
      self.assertTrue(readers[1].delete_version(uri, version))
      with self.assertRaisesRegex(ValueError, "has not landed"):
        Store(root, cache=f"{tmp}/fresh").get(uri, version)

  def test_peer_copy_is_available_before_it_is_durable(self):
    with tempfile.TemporaryDirectory() as tmp:
      uri, root = "tinker://r/weights/test", f"memory://{uuid.uuid4().hex}"
      source, reader = Store(root, cache=f"{tmp}/source"), Store(root, cache=f"{tmp}/reader")
      try:
        source.start()
      except OSError as exc:
        if exc.errno in (errno.EPERM, errno.EACCES):
          self.skipTest(f"socket creation is unavailable: {exc}")
        raise
      source.stopping.set()
      source.uploader.join()
      source.peer = f"127.0.0.1:{source.server.server_port}"
      try:
        version = source.put(uri, {"w": torch.arange(32)})
        self.assertTrue(torch.equal(reader.get(uri, version)[0]["w"], torch.arange(32)))
        self.assertFalse(reader.is_durable(uri, version))
        self.assertFalse(reader._evict(version, durable=True))
      finally:
        source.close()
      self.assertTrue(reader.is_durable(uri, version))
      canonical = split_version(version)[0]
      self.assertTrue(torch.equal(Store(root, cache=f"{tmp}/later").get(uri, canonical)[0]["w"], torch.arange(32)))

  def test_incomplete_copies_are_not_published(self):
    with tempfile.TemporaryDirectory() as tmp:
      store = Store(f"memory://{uuid.uuid4().hex}", cache=tmp)
      version, uri = "abcdef1234", "tinker://r/weights/test"
      os.mkdir(store.folder(version))
      with open(f"{store.folder(version)}/state.safetensors", "wb") as f:
        f.write(b"partial")
      with self.assertRaisesRegex(ValueError, "has not landed"):
        store.get(uri, version)
      version = store.put(uri, {"w": torch.ones(1)})
      os.unlink(f"{store.folder(version)}/state.safetensors")
      self.assertFalse(store.upload_version(uri, version))
      self.assertTrue(os.path.exists(f"{store.folder(version)}.upload"))
      self.assertFalse(store.is_durable(uri, version))


class TestRegistry(RedisTest):
  async def test_owners_and_reader_lock_protect_old_versions(self):
    with tempfile.TemporaryDirectory() as tmp:
      store, uri = Store(tmp), "tinker://r/weights/saved"
      a, b = [store.put(uri, {"w": torch.tensor(i)}) for i in (1, 2)]
      await self.seed_checkpoint(uri, 1, version=a)
      await self.bus.create_run(Run("consumer", "worker", "base", 0))
      await self.bus.submit_command("consumer", Op.LOAD_WEIGHTS, json.dumps({"uri": uri, "version": a}).encode())
      await self.seed_checkpoint(uri, 1, version=b)
      await self.registry.sweep(store, 0)
      self.assertTrue(bool(manifest(store.folder(a))))
      await self.bus.delete_run("consumer", "worker")
      await self.bus.create_run(Run("next", "worker", "base", 0))
      with store.hold(uri, a):
        await self.registry.sweep(store, 0)
        with self.assertRaisesRegex(ValueError, "being deleted"):
          await self.bus.submit_command("next", Op.LOAD_WEIGHTS, json.dumps({"uri": uri, "version": a}).encode())
      await self.registry.sweep(store, 0)
      self.assertFalse(bool(manifest(store.folder(a))))
      self.assertEqual(store.get(uri, await self.registry.version(uri))[0]["w"].item(), 2)

  async def test_deleted_alias_cannot_be_resurrected_by_ttl_update(self):
    uri, version = "tinker://r/weights/test", uuid.uuid4().hex
    await self.seed_checkpoint(uri, 1, ttl=60, version=version)
    await asyncio.gather(
      *(self.registry.set_checkpoint_ttl(uri, 600) for _ in range(20)), self.registry.unregister_checkpoints(uri)
    )
    self.assertIsNone(await self.registry.get_checkpoint(uri))
    self.assertIsNone(await self.bus.r.hget(key("refs"), f"ckpt:{uri}"))
    for index in ("ckpts", "expiry"):
      self.assertIsNone(await self.bus.r.zscore(key(index), uri))


if __name__ == "__main__":
  unittest.main()
