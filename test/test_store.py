import os
import tempfile

import redis
import torch

from test.helpers import RedisTest, new_id, tiny
from test.test_trainer import LORA, same, train
from tinyrl.store import Store
from tinyrl.trainer import Model


class TestStore(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.r = redis.from_url(self.url, protocol=2, decode_responses=True)

  async def asyncTearDown(self):
    self.r.close()
    await super().asyncTearDown()

  async def test_local_shm_roundtrip_without_touching_disk(self):
    store = Store(r=self.r)
    key = f"{tempfile.mkdtemp()}/ckpt"
    tensors = {"w": torch.arange(6, dtype=torch.float32).reshape(2, 3), "step": torch.tensor(4.0)}

    store.put(key, tensors, {"optimizer": "1"})
    self.assertFalse(os.path.exists(key))

    got, meta = store.get(key)
    self.assertEqual(meta, {"optimizer": "1"})
    self.assertTrue(all(torch.equal(got[k], tensors[k]) for k in tensors))

    store.drop(key)
    self.assertEqual(store.get(key), ({}, {}))

  async def test_peer_streams_from_another_agents_shm_without_touching_disk(self):
    source, target = Store(r=self.r), Store(r=self.r)
    source.serve()
    self.addCleanup(source.close)

    key = f"{tempfile.mkdtemp()}/park"
    tensors = {"lora_A": torch.randn(4, 16), "lora_A.exp_avg": torch.randn(4, 16)}
    source.put(key, tensors, {"optimizer": "1"})
    self.assertFalse(os.path.exists(key))

    got, meta = target.get(key)
    self.assertEqual(meta, {"optimizer": "1"})
    self.assertTrue(all(torch.equal(got[k], tensors[k]) for k in tensors))
    self.assertTrue(os.path.exists(target.file(key)[1]))

  async def test_falls_back_to_durable_disk_when_peer_is_gone(self):
    source, target = Store(r=self.r), Store(r=self.r)
    source.serve()

    key = f"{tempfile.mkdtemp()}/ckpt"
    tensors = {"w": torch.randn(8, 8)}
    source.put(key, tensors, {"optimizer": "1"}, disk=f"{key}/model.safetensors")
    source.close()  # the source node died; its URL is still in Redis, but its HTTP server is gone

    got, meta = target.get(key)
    self.assertEqual(meta, {"optimizer": "1"})
    self.assertTrue(torch.equal(got["w"], tensors["w"]))

  async def test_a_run_moved_between_trainers_over_peer_http_trains_on_as_before(self):
    source, target = Store(r=self.r), Store(r=self.r)
    source.serve()
    self.addCleanup(source.close)

    for rank in (0, 4):
      run, cfg = new_id(), LORA if rank else {"seed": 0}
      m1, m2 = Model(tiny, source), Model(tiny, target)
      m1.create(run, "tiny", rank, cfg)
      train(m1, run, steps=3)

      key = f"{tempfile.mkdtemp()}/park-{rank}"
      m1.save(run, key, optimizer=True, durable=False)  # staged in source's /dev/shm only
      self.assertFalse(os.path.exists(key))

      m2.create(run, "tiny", rank, cfg)
      m2.load_weights(run, key, optimizer=True)  # streamed over HTTP from source's /dev/shm into target's

      self.assertEqual(train(m1, run, steps=3), train(m2, run, steps=3))
      self.assertTrue(same(m1.params[run], m2.params[run]))
