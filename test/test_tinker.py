import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

import redis

from test.helpers import start_redis, tiny
from test.test_process import require_unix

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
  """The real Tinker SDK trains through redis, the gateway and a trainer, each its own process."""

  def test_the_tinker_sdk_trains_through_tinyrl(self):
    require_unix()
    tmp = tempfile.mkdtemp()
    base = f"{tmp}/tiny"
    tiny().save_pretrained(base)  # a base model on disk, loaded the way any other is

    port = free_port()
    env = {
      **os.environ,
      "REDIS_URL": f"unix://{tmp}/redis.sock",
      "PORT": str(port),
      "CKPT": f"file://{tmp}/ckpt",  # a URL, so checkpoints go through upload and download as they do on GCS
      "CKPT_CACHE": f"{tmp}/cache",
      "ENGINE": "test.helpers:FakeEngine",  # the sampler's engine, since CI has no GPU for vLLM
    }
    procs = [start_redis(f"{tmp}/redis.sock")]
    try:
      procs.append(subprocess.Popen([sys.executable, "-m", "tinyrl.api.gateway"], env=env))
      procs.append(subprocess.Popen([sys.executable, "-m", "tinyrl.control.scheduler"], env=env))
      # starts the trainer
      procs.append(subprocess.Popen([sys.executable, "-m", "tinyrl.control.node_daemon"], env=env))
      wait_until_up(f"http://127.0.0.1:{port}/api/v1/healthz")

      sdk_env = {**env, "BASE_URL": f"http://127.0.0.1:{port}", "BASE_MODEL": base}
      out = subprocess.run(
        [os.environ["TINKER_PYTHON"], EXAMPLE], env=sdk_env, capture_output=True, text=True, timeout=300
      )
      print(out.stdout)
      self.assertEqual(out.returncode, 0, out.stderr[-3000:])

      model_id = out.stdout.split("model ")[1].split()[0]
      with redis.Redis(unix_socket_path=f"{tmp}/redis.sock") as r:
        saved = json.loads(r.hget(f"tinyrl:ckpt:{model_id}", "sampler_weights/final"))
      version = saved["version"].split("@", 1)[0]
      for _ in range(300):
        if os.path.exists(f"{tmp}/ckpt/{version}.json"):
          break
        time.sleep(0.1)
      self.assertTrue(os.path.exists(f"{tmp}/ckpt/{version}/adapter_model.safetensors"))
    finally:
      for proc in procs:
        proc.terminate()
        try:
          proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
          proc.kill()


if __name__ == "__main__":
  unittest.main()
