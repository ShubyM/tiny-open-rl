import os, subprocess, tempfile, time, unittest, uuid
import redis.asyncio as redis, torch
from transformers import LlamaConfig, LlamaForCausalLM
from tinyrl.bus import Bus

def tiny(_name:str="") -> LlamaForCausalLM:
  # a two layer llama with random weights: the same every time, and nothing to download
  torch.manual_seed(0)
  cfg = LlamaConfig(vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
  return LlamaForCausalLM(cfg)

def rid() -> str: return str(uuid.uuid4())  # run ids are uuids, peft finds an adapter's weights by its name

class RedisTest(unittest.IsolatedAsyncioTestCase):
  # a throwaway redis-server on a unix socket per test class, or REDIS_URL
  proc:subprocess.Popen|None = None
  @classmethod
  def setUpClass(cls):
    if (url:=os.getenv("REDIS_URL")): cls.url = url
    else:
      sock = f"{tempfile.mkdtemp()}/redis.sock"
      args = ["--port", "0", "--unixsocket", sock, "--save", "", "--appendonly", "no"]
      cls.proc = subprocess.Popen([os.getenv("REDIS_SERVER", "redis-server"), *args], stdout=subprocess.DEVNULL)
      for _ in range(500):
        if os.path.exists(sock): break
        time.sleep(0.01)
      cls.url = f"unix://{sock}"
  @classmethod
  def tearDownClass(cls):
    if cls.proc: cls.proc.terminate()
  async def asyncSetUp(self):
    self.bus = Bus(redis.from_url(self.url))
    await self.bus.r.flushall()
  async def asyncTearDown(self): await self.bus.r.aclose()
