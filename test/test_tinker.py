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
      "WAIT": "0.1",
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
      script = "\n".join(
        [
          "import os, runpy, sys, json, httpx, redis",
          "from unittest.mock import patch",
          "scope = runpy.run_path(sys.argv[1])",
          "second = scope['client'].create_lora_training_client(base_model=os.environ['BASE_MODEL'], rank=8)",
          "assert 'loss:sum' in second.forward_backward([scope['datum']], 'cross_entropy').result().metrics",
          "second.optim_step(scope['types'].AdamParams(learning_rate=1e-3)).result()",
          # Expire a completed result before the SDK receives it. HTTP 410 would tell it to retry the operation.
          "r = redis.from_url(os.environ['REDIS_URL'])",
          "send, submissions = httpx.AsyncClient.send, []",
          "async def expire(self, request, *args, **kwargs):",
          "  if request.url.path.endswith('/optim_step'): submissions.append(request)",
          "  response = await send(self, request, *args, **kwargs)",
          "  if request.url.path.endswith('/retrieve_future') and response.status_code == 200:",
          "    await response.aread()",
          "    if response.json().get('type') != 'try_again':",
          "      r.delete('tinyrl:f:' + json.loads(request.content)['request_id'])",
          "      response = await send(self, request, *args, **kwargs)",
          "  return response",
          "with patch.object(httpx.AsyncClient, 'send', expire):",
          "  try: second.optim_step(scope['types'].AdamParams(learning_rate=1e-3)).result(timeout=15)",
          "  except ValueError as e: assert 'status 404' in str(e), e",
          "  else: raise AssertionError('expired future must fail')",
          "assert len(submissions) == 1, 'expired future must not resubmit its operation'",
        ]
      )
      out = subprocess.run(
        [os.environ["TINKER_PYTHON"], "-c", script, EXAMPLE], env=sdk_env, capture_output=True, text=True, timeout=300
      )
      print(out.stdout)
      self.assertEqual(out.returncode, 0, out.stderr[-3000:])

      model_id = out.stdout.split("model ")[1].split()[0]
      # The node daemon publishes each immutable version's manifest after its files finish uploading.
      final = f"{tmp}/ckpt/{model_id}/sampler_weights/final"
      for _ in range(300):
        if manifests := sorted(pathlib.Path(final).glob("*.json")):
          break
        time.sleep(0.1)
      self.assertTrue(manifests, "sampler checkpoint never finished uploading")
      with manifests[-1].open() as f:
        manifest = json.load(f)
      self.assertTrue(os.path.exists(f"{tmp}/ckpt/{manifest['dir']}/adapter_model.safetensors"))
    finally:
      for proc in reversed(procs):
        proc.terminate()
        try:
          proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
          proc.kill()
          proc.wait()


if __name__ == "__main__":
  unittest.main()
