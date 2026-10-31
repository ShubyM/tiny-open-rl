import os
import tempfile
import unittest
import uuid
from unittest.mock import patch

import fsspec

from tinyrl.storage import models


def publish(url: str, files: dict[str, bytes]) -> None:
  fs, path = fsspec.core.url_to_fs(url)
  for name, data in files.items():
    fs.pipe_file(f"{path}/{name}", data)


class TestModels(unittest.TestCase):
  def setUp(self):
    self.cache = tempfile.mkdtemp()
    self.patch = patch.multiple(models, MODEL_CACHE=self.cache, MODEL_MIRROR="")
    self.patch.start()
    self.addCleanup(self.patch.stop)

  def test_hub_ids_and_local_folders_pass_through(self):
    self.assertEqual(models.local("Qwen/Qwen3-0.6B"), "Qwen/Qwen3-0.6B")
    self.assertEqual(models.local("hf://Qwen/Qwen3-0.6B"), "Qwen/Qwen3-0.6B")
    self.assertEqual(models.local("/mnt/models/q"), "/mnt/models/q")
    self.assertEqual(models.local("file:///mnt/models/q"), "/mnt/models/q")
    self.assertEqual(os.listdir(self.cache), [])

  def test_a_bucket_model_is_fetched_once_per_node(self):
    url = f"memory://models/{uuid.uuid4()}/q"
    publish(url, {"config.json": b"{}", "model.safetensors": b"weights"})
    folder = models.local(url)
    self.assertEqual(sorted(os.listdir(folder)), ["config.json", "model.safetensors"])
    fsspec.filesystem("memory").rm(url.removeprefix("memory://"), recursive=True)
    self.assertEqual(models.local(url), folder)  # from the cache, the bucket isn't asked again

  def test_a_hub_model_comes_from_the_mirror_when_it_is_there(self):
    mirror = f"memory://mirror/{uuid.uuid4()}"
    publish(f"{mirror}/org/m", {"config.json": b"{}", "model.safetensors": b"weights"})
    with patch.object(models, "MODEL_MIRROR", mirror):
      self.assertTrue(models.local("org/m").startswith(self.cache))
      self.assertEqual(models.local("org/elsewhere"), "org/elsewhere")  # not mirrored, so the hub

  def test_a_folder_without_a_config_is_no_model_and_leaves_nothing(self):
    url = f"memory://models/{uuid.uuid4()}/broken"
    publish(url, {"model.safetensors": b"weights"})
    with self.assertRaises(ValueError):
      models.local(url)
    self.assertEqual(os.listdir(self.cache), [])


if __name__ == "__main__":
  unittest.main()
