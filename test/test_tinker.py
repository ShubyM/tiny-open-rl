import asyncio
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

from test.helpers import tiny
from tinyrl.bus import connect

EXAMPLE = pathlib.Path(__file__).parent.parent / "examples" / "tinker_sdk.py"


def free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def wait_until_up(url: str) -> None:
  for _ in range(300):
    try:
      urllib.request.urlopen(url)
      return
    except OSError:
      time.sleep(0.1)


@unittest.skipUnless(os.getenv("TINKER_PYTHON"), "set TINKER_PYTHON to a python that has the tinker SDK")
class TestTinkerSDK(unittest.TestCase):
  """The real Tinker SDK trains through the gateway (control plane) and a node daemon, each its own process."""

  def test_the_tinker_sdk_trains_through_tinyrl(self):
    tmp = tempfile.mkdtemp()
    base = f"{tmp}/tiny"
    tiny().save_pretrained(base)  # a base model on disk, loaded the way any other is

    port, internal_port = free_port(), free_port()
    env = {
      **os.environ,
      "TINYRL_DB": f"{tmp}/tinyrl.db",
      "PORT": str(port),
      "INTERNAL_PORT": str(internal_port),
      "CONTROL_URL": f"http://127.0.0.1:{internal_port}",
      "CKPT": f"file://{tmp}/ckpt",  # a URL, so checkpoints go through upload and download as they do on GCS
      "CKPT_CACHE": f"{tmp}/cache",
      "ENGINE": "test.helpers:FakeEngine",  # the sampler's engine, since CI has no GPU for vLLM
    }
    procs = []
    try:
      procs.append(subprocess.Popen([sys.executable, "-m", "tinyrl.api.gateway"], env=env))
      wait_until_up(f"http://127.0.0.1:{port}/api/v1/healthz")
      procs.append(subprocess.Popen([sys.executable, "-m", "tinyrl.control.node_daemon"], env=env))

      sdk_env = {**env, "BASE_URL": f"http://127.0.0.1:{port}", "BASE_MODEL": base}
      out = subprocess.run(
        [os.environ["TINKER_PYTHON"], EXAMPLE], env=sdk_env, capture_output=True, text=True, timeout=300
      )
      print(out.stdout)
      self.assertEqual(out.returncode, 0, out.stderr[-3000:])

      model_id = out.stdout.split("model ")[1].split()[0]

      async def published() -> dict:
        db = connect(env["CONTROL_URL"])
        try:
          return await db.get_checkpoint(f"tinker://{model_id}/sampler_weights/final")
        finally:
          await db.aclose()

      record = asyncio.run(published())
      version = record["version"].partition("@")[0]
      final = pathlib.Path(tmp, "ckpt", "objects", version)
      for _ in range(300):
        if (final / "manifest.json").exists():
          break
        time.sleep(0.1)
      self.assertTrue((final / "adapter_model.safetensors").is_file())
    finally:
      for proc in procs:
        proc.terminate()
        try:
          proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
          proc.kill()


if __name__ == "__main__":
  unittest.main()
