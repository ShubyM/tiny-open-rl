with open("test/test_trainer.py") as f:
  text = f.read()

# I had already patched test_trainer with test_a3 once, let me reset it
import subprocess

subprocess.run(["git", "checkout", "test/test_trainer.py"])

with open("test/test_trainer.py") as f:
  text = f.read()

a3_test = """
  async def test_a3_restore_missing_run_skip(self):
    run_id = new_id()
    await self.bus.create_run(Run(run_id, "p", "tiny", 0))
    t = Trainer(self.bus, "p", Model(tiny))
    await self.own(t)
    
    cmd = Cmd(run=run_id, entry_id="r1", request_id="req1", op=Op.LOAD_WEIGHTS, body=json.dumps({"uri": "tinker://p/bad", "optimizer": False}).encode())
    reply = await t.answer_command(cmd)
    self.assertFalse(reply.ok)
    
    self.assertNotIn(run_id, t.model.params)
"""

text = text.replace("class TestTrainer(RedisTest):", "class TestTrainer(RedisTest):\n" + a3_test)

model_tests = """
  def test_a1_and_a2_gradients(self):
    model, run_id = Model(tiny), new_id()
    model.create_run(run_id, "tiny", 0, {})
    model.activate_run(run_id)
    for p in model.params[run_id].values():
      p.grad = torch.ones_like(p.data) * 0.5
      
    tensors, meta = model.training_state(run_id)
    self.assertTrue(any("grad" in name for name in tensors.keys()))
    
    model.restore_run(run_id, tensors, meta, True)
    self.assertTrue(any(p.grad is not None for p in model.params[run_id].values()))
    
    meta["internal_handoff"] = False
    model.restore_run(run_id, tensors, meta, True)
    self.assertFalse(any(p.grad is not None for p in model.params[run_id].values()))

  def test_a4_mixed_state(self):
    model, run_id = Model(tiny), new_id()
    model.create_run(run_id, "tiny", 0, {})
    model.activate_run(run_id)
    opt = model.adam_for_run(run_id)
    for p in model.params[run_id].values():
      opt.state[p] = {"exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p), "step": torch.tensor(1.0)}
        
    tensors, meta = model.training_state(run_id)
    bad_tensors = tensors.copy()
    for k in list(bad_tensors.keys()):
      if k.endswith(".step"):
        del bad_tensors[k]
        break
            
    with self.assertRaises(ValueError):
      model.restore_run(run_id, bad_tensors, meta, True)

  def test_a5_partial_gradients(self):
    model, run_id = Model(tiny), new_id()
    model.create_run(run_id, "tiny", 0, {})
    
    import tinyrl.workers.model
    old_budget = tinyrl.workers.model.TOKEN_BUDGET
    tinyrl.workers.model.TOKEN_BUDGET = 1
    
    calls = [0]
    original = tinyrl.workers.model.target_logprobs
    def fake_target_logprobs(*args, **kwargs):
      calls[0] += 1
      if calls[0] == 2:
        raise RuntimeError("OOM")
      return original(*args, **kwargs)
    
    tinyrl.workers.model.target_logprobs = fake_target_logprobs
    
    data = [
      wire.Datum(tokens=np.array([1, 2], dtype=np.int32), inputs={"target_tokens": np.array([1, 2], dtype=np.int64)}),
      wire.Datum(tokens=np.array([3, 4], dtype=np.int32), inputs={"target_tokens": np.array([3, 4], dtype=np.int64)}),
    ]
    
    with self.assertRaises(RuntimeError):
      model.forward_backward(run_id, data, "cross_entropy", {}, False)
        
    has_grad = any(p.grad is not None and torch.any(p.grad != 0) for p in model.params[run_id].values())
    self.assertTrue(has_grad)

    with self.assertRaisesRegex(RuntimeError, "recovery"):
      model.optim_step(run_id, {"learning_rate": 1e-2})
        
    tinyrl.workers.model.target_logprobs = original
    
    model.forward_backward(run_id, data, "cross_entropy", {}, False)
    model.optim_step(run_id, {"learning_rate": 1e-2})
    
    tinyrl.workers.model.TOKEN_BUDGET = old_budget
"""

text = text.replace("class TestModel(unittest.TestCase):", "class TestModel(unittest.TestCase):\n" + model_tests)

with open("test/test_trainer.py", "w") as f:
  f.write(text)
