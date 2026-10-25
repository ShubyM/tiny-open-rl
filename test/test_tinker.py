import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

from test.helpers import RedisServer, tiny
from tinyrl.store import Store

EXAMPLE = pathlib.Path(__file__).parent.parent / "examples" / "tinker_sdk.py"


def free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def wait_until_up(url: str) -> None:
  for _ in range(300):
    try:
      with urllib.request.urlopen(url, timeout=1):
        return
    except OSError as error:
      if isinstance(error, urllib.error.HTTPError):
        error.close()
      time.sleep(0.1)
  raise TimeoutError(f"coordinator did not become ready: {url}")


@unittest.skipUnless(os.getenv("TINKER_PYTHON"), "set TINKER_PYTHON to a python that has the tinker SDK")
class TestTinkerSDK(unittest.TestCase):
  """The real SDK and HTTP worker retain a training run across a coordinator restart."""

  def test_the_tinker_sdk_trains_across_a_coordinator_restart(self):
    with tempfile.TemporaryDirectory() as tmp, RedisServer() as redis_server:
      base = f"{tmp}/tiny"
      tiny().save_pretrained(base)
      url = f"http://127.0.0.1:{free_port()}"
      env = {
        **os.environ,
        "PORT": url.rsplit(":", 1)[1],
        "REDIS_URL": redis_server.url,
        "REDIS_NAMESPACE": "sdk-restart-test",
        "COORDINATOR_TTL": "2",
        "CKPT": f"{tmp}/ckpt",
        "PAYLOAD_DIR": f"{tmp}/payloads",
        "COORDINATOR_URL": url,
        "COORDINATOR_TOKEN": "integration-worker-token",
        "BASE_URL": url,
        "BASE_MODEL": base,
      }
      ready, resume = pathlib.Path(tmp, "ready"), pathlib.Path(tmp, "resume")
      script = """
import pathlib, runpy, sys, time
from tinker import types
context = runpy.run_path(sys.argv[1])
trainer, datum = context['trainer'], context['datum']
pathlib.Path(sys.argv[2]).write_text(trainer.model_id)
while not pathlib.Path(sys.argv[3]).exists():
    time.sleep(0.05)
forward = trainer.forward_backward([datum], 'cross_entropy')
step = trainer.optim_step(types.AdamParams(learning_rate=1e-2))
assert forward.result().metrics['loss:sum'] >= 0
assert step.result().metrics['grad_norm:mean'] > 0
print('resumed_sampler ' + trainer.save_weights_for_sampler('after-restart').result().path, flush=True)
"""
      procs = []
      with open(f"{tmp}/services.log", "w+") as logs:
        try:

          def coordinator():
            proc = subprocess.Popen([sys.executable, "-m", "tinyrl.gateway"], env=env, stdout=logs, stderr=logs)
            procs.append(proc)
            try:
              wait_until_up(f"{url}/api/v1/healthz")
            except TimeoutError as error:
              logs.seek(0)
              raise RuntimeError(f"{error}\n{logs.read()[-6000:]}") from error
            return proc

          server = coordinator()
          worker = subprocess.Popen(
            [sys.executable, "-m", "tinyrl.trainer", f"lora:{base}"], env=env, stdout=logs, stderr=logs
          )
          procs.append(worker)
          sdk = subprocess.Popen(
            [os.environ["TINKER_PYTHON"], "-c", script, str(EXAMPLE), str(ready), str(resume)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
          )
          procs.append(sdk)
          deadline = time.monotonic() + 180
          while not ready.exists() and sdk.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
          if not ready.exists():
            if sdk.poll() is None:
              sdk.terminate()
            stdout, stderr = sdk.communicate(timeout=15)
            logs.seek(0)
            self.fail(f"SDK did not reach restart boundary:\n{stdout}\n{stderr[-4000:]}\n{logs.read()[-4000:]}")

          run_id = ready.read_text()
          server.kill()  # retain Redis and the worker while the coordinator loses its lease
          server.wait(timeout=15)
          self.assertIsNone(worker.poll(), "worker exited with the coordinator")
          time.sleep(2.1)  # the dead coordinator cannot renew; replacement waits for its lease to expire
          coordinator()
          resume.touch()
          stdout, stderr = sdk.communicate(timeout=180)
          logs.seek(0)
          self.assertEqual(sdk.returncode, 0, f"{stderr[-4000:]}\n{logs.read()[-4000:]}")
          self.assertIsNone(worker.poll(), "worker did not survive coordinator restart")

          uri = stdout.split("resumed_sampler ")[1].splitlines()[0]
          request = urllib.request.Request(
            f"{url}/api/v1/resolve_checkpoint",
            data=json.dumps({"path": uri}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
          )
          with urllib.request.urlopen(request, timeout=10) as response:
            ref = json.load(response)["ref"]
          folder = Store(root=env["CKPT"]).export(ref)
          self.assertTrue(pathlib.Path(folder, "adapter_model.safetensors").is_file())
          with urllib.request.urlopen(f"{url}/api/v1/runs/{run_id}", timeout=10) as response:
            run = json.load(response)
          self.assertEqual((run["steps"], run["failed"]), (11, 0))
        finally:
          for proc in reversed(procs):
            if proc.poll() is None:
              proc.terminate()
          for proc in reversed(procs):
            try:
              proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
              proc.kill()
              proc.wait()


if __name__ == "__main__":
  unittest.main()
