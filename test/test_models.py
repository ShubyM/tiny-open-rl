import os
import subprocess
import sys
import tempfile
import unittest


class TestModels(unittest.TestCase):
  def test_mirror_bytes_are_cached_and_survive_source_removal(self):
    script = """
import os
from pathlib import Path
import fsspec
from tinyrl.storage.models import local

fs = fsspec.filesystem("memory")
files = {"config.json": b"{}", "model.safetensors": b"weights"}
for name, data in files.items():
  fs.pipe_file(f"/models/org/m/{name}", data)
folder = Path(local("org/m"))
assert folder.parent == Path(os.environ["MODEL_CACHE"])
assert {p.name: p.read_bytes() for p in folder.iterdir()} == files
fs.rm("/models/org/m", recursive=True)
assert Path(local("memory://models/org/m")) == folder
assert local("hf://org/missing") == "org/missing"
assert local("file:///mnt/models/m") == "/mnt/models/m"
"""
    with tempfile.TemporaryDirectory() as folder:
      out = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "MODEL_CACHE": f"{folder}/cache", "MODEL_MIRROR": "memory://models"},
        capture_output=True,
        text=True,
        timeout=10,
      )
      self.assertEqual(out.returncode, 0, out.stderr)


if __name__ == "__main__":
  unittest.main()
