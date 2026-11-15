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

from test.helpers import start_redis, tiny

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
      # the node daemon uploads the node's saves to CKPT behind, under a version that _latest points at last, so wait for it
      final = f"{tmp}/ckpt/{model_id}/sampler_weights/final"
      for _ in range(300):
        if os.path.exists(f"{final}/_latest"):
          break
        time.sleep(0.1)
      with open(f"{final}/_latest") as f:
        version = json.load(f)["version"]
      self.assertTrue(os.path.exists(f"{final}/{version}/adapter_model.safetensors"))

      # A fresh Tinker SDK session queries weights_info and resumes from the saved durable state with optimizer.
      state_uri = out.stdout.split("state")[1].split()[0]
      resume_code = """
import os, tinker
from tinker import types
sc = tinker.ServiceClient(base_url=os.environ["BASE_URL"], api_key="tml-local")
rest = sc.create_rest_client()
info = rest.get_weights_info_by_tinker_path(os.environ["STATE_URI"]).result()
assert info.is_lora is True and info.lora_rank == 8, info
tc = sc.create_training_client_from_state_with_optimizer(os.environ["STATE_URI"]).result()
datum = types.Datum(
  model_input=types.ModelInput.from_ints([1, 2, 3, 4]),
  loss_fn_inputs={"target_tokens": [2, 3, 4, 5], "weights": [1.0, 1.0, 1.0, 1.0]},
)
fb = tc.forward_backward([datum], "cross_entropy").result()
op = tc.optim_step(types.AdamParams(learning_rate=1e-3)).result()
print("resumed_loss", fb.metrics["loss:sum"])
"""
      resumed = subprocess.run(
        [os.environ["TINKER_PYTHON"], "-c", resume_code],
        env={**sdk_env, "STATE_URI": state_uri},
        capture_output=True,
        text=True,
        timeout=180,
      )
      self.assertEqual(resumed.returncode, 0, resumed.stderr[-3000:])
      self.assertIn("resumed_loss", resumed.stdout)
    finally:
      for proc in procs:
        proc.terminate()
        try:
          proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
          proc.kill()
          proc.wait(timeout=5)


if __name__ == "__main__":
  unittest.main()
