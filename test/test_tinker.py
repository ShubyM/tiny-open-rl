import os, pathlib, socket, subprocess, sys, tempfile, time, unittest, urllib.request
from test.helpers import tiny

def port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]

@unittest.skipUnless(os.getenv("TINKER_PYTHON"), "set TINKER_PYTHON to a python that has the tinker sdk")
class TestTinkerSDK(unittest.TestCase):
  # redis, the gateway and a trainer as their own processes, and the real sdk training through them
  def test_the_tinker_sdk_trains_through_tinyrl(self):
    d = tempfile.mkdtemp()
    tiny().save_pretrained(base:=f"{d}/tiny")  # a base model on disk, loaded the way any other is
    redis = [os.getenv("REDIS_SERVER", "redis-server"), "--port", "0", "--unixsocket", f"{d}/redis.sock", "--save", ""]
    procs = [subprocess.Popen(redis, stdout=subprocess.DEVNULL)]
    env = {**os.environ, "REDIS_URL": f"unix://{d}/redis.sock", "PORT": str(p:=port()), "CKPT": f"{d}/ckpt"}
    try:
      while not os.path.exists(f"{d}/redis.sock"): time.sleep(0.01)
      cmds = [[sys.executable, "-m", "tinyrl.gateway"], [sys.executable, "-m", "tinyrl.trainer", f"lora:{base}"]]
      procs += [subprocess.Popen(cmd, env=env) for cmd in cmds]
      for _ in range(300):
        try: urllib.request.urlopen(f"http://127.0.0.1:{p}/api/v1/healthz")
        except OSError: time.sleep(0.1)
        else: break
      example = pathlib.Path(__file__).parent.parent/"examples"/"tinker_sdk.py"
      out = subprocess.run([os.environ["TINKER_PYTHON"], example], env={**env, "BASE_URL": f"http://127.0.0.1:{p}", "BASE_MODEL": base},
                           capture_output=True, text=True, timeout=300)
      print(out.stdout)
      self.assertEqual(out.returncode, 0, out.stderr[-3000:])
      self.assertTrue(os.path.exists(f"{d}/ckpt/{out.stdout.split('model ')[1].split()[0]}/sampler_weights/final/adapter_model.safetensors"))
    finally:
      for proc in procs: proc.terminate()

if __name__ == "__main__":
  unittest.main()
