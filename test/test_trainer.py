import asyncio, copy, json, tempfile, unittest
from unittest.mock import patch
import numpy as np, torch
from peft import PeftModel
from tinyrl import trainer, wire
from tinyrl.bus import Run
from tinyrl.trainer import LOSSES, Model, Trainer, batches, logprobs
from test.helpers import RedisTest, rid, tiny

LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}

def datum(n:int, shift:int=0, **inputs) -> wire.Datum:  # count up from shift: each token's target is the next one
  t = np.arange(shift, shift+n+1) % 96
  return wire.Datum(t[:-1].astype(np.int32), {"target_tokens": t[1:].astype(np.int64), **inputs})

def train(m:Model, run:str, steps:int=8, lr:float=1e-2) -> list[float]:
  losses = []
  for _ in range(steps):
    losses.append(m.forward_backward(run, [datum(12), datum(9, 40)], "cross_entropy", {}, False)[1]["loss:sum"])
    m.optim_step(run, {"learning_rate": lr})
  return losses

def lora(n:int=1) -> tuple[Model, list[str]]:
  m, runs = Model(tiny), [rid() for _ in range(n)]
  for r in runs: m.create(r, "tiny", 4, LORA)
  return m, runs

class TestForward(unittest.TestCase):
  def test_batches(self):
    self.assertEqual(batches([5, 5, 5], 100), [[0, 1, 2]])
    self.assertEqual(batches([5, 5, 5], 10), [[0, 1], [2]])
    self.assertEqual(batches([10]*6+[50], 1000), [[0, 1, 2, 3, 4, 5], [6]])  # a long sequence gets its own pass instead of padding short ones
    self.assertEqual(batches([3, 1, 2], 0), [[1], [2], [0]])  # no budget, one sequence per pass
    every = batches(lens:=[int(x) for x in np.random.default_rng(0).integers(1, 64, 200)], 256)
    self.assertEqual(sorted(i for b in every for i in b), list(range(200)))
    self.assertTrue(all(len(b) == 1 or max(lens[i] for i in b)*len(b) <= 256 for b in every))

  def test_chunked_logprobs_are_the_full_softmax(self):
    lm, ids, tgt = tiny().eval(), torch.randint(0, 96, (3, 11)), torch.randint(0, 96, (3, 11))
    mask = torch.ones_like(ids)
    mask[2, 7:] = 0
    with patch.object(trainer, "CHUNK", 4), torch.no_grad(): got = logprobs(lm, ids, mask, tgt)
    with torch.no_grad(): want = lm(ids, attention_mask=mask).logits.log_softmax(-1).gather(-1, tgt[..., None])[..., 0]
    torch.testing.assert_close(got[mask.bool()], want[mask.bool()])

  def test_losses_are_their_definitions(self):
    g = torch.Generator().manual_seed(0)
    lp, old, adv, w = -torch.rand(4, 9, generator=g), -torch.rand(4, 9, generator=g), torch.randn(4, 9, generator=g), torch.rand(4, 9, generator=g)
    r = torch.exp(lp-old)
    torch.testing.assert_close(LOSSES["cross_entropy"](lp, w, old, adv, {}), -lp*w)
    torch.testing.assert_close(LOSSES["importance_sampling"](lp, w, old, adv, {}), -r*adv*w)
    torch.testing.assert_close(LOSSES["ppo"](lp, w, old, adv, {"clip_range": 0.1}), -torch.min(r*adv, r.clamp(0.9, 1.1)*adv)*w)
    torch.testing.assert_close(LOSSES["ppo"](lp, w, old, adv, {"kl_coeff": 0.5}) - LOSSES["ppo"](lp, w, old, adv, {}), 0.5*((r-1)-(lp-old))*w)

class TestModel(unittest.TestCase):
  def test_lora_runs_share_a_frozen_base_and_train_apart(self):
    m, (a, b) = lora(2)
    base, before = {n:p.detach().clone() for n,p in m.lm.named_parameters() if "lora" not in n}, [p.detach().clone() for p in m.params[b]]
    self.assertLess((losses:=train(m, a))[-1], losses[0]*0.7)
    self.assertTrue(all(torch.equal(x, p) for x,p in zip(before, m.params[b])))
    self.assertTrue(all(torch.equal(base[n], p) for n,p in m.lm.named_parameters() if "lora" not in n))

  def test_interleaved_runs_keep_their_own_gradients(self):
    m, (a, b) = lora(2)
    m.forward_backward(a, [datum(8)], "cross_entropy", {}, False)
    m.forward_backward(b, [datum(8, 5)], "cross_entropy", {}, False)
    grads = [p.grad.clone() for p in m.params[b]]
    m.optim_step(a, {"learning_rate": 1e-2})
    self.assertTrue(all(p.grad is None for p in m.params[a]))
    self.assertTrue(all(torch.equal(g, p.grad) for g,p in zip(grads, m.params[b])))

  def test_forward_only_leaves_no_gradient(self):
    m, (a,) = lora()
    lps, metrics = m.forward_backward(a, [datum(8), datum(5)], "cross_entropy", {}, True)
    self.assertTrue(all(p.grad is None for p in m.params[a]))
    self.assertEqual([x.shape for x in lps], [(8,), (5,)])
    self.assertAlmostEqual(metrics["loss:sum"], -sum(float(x.sum()) for x in lps), places=4)

  def test_rl_losses_move_logprobs_with_the_advantage(self):
    for loss in ("importance_sampling", "ppo"):
      m, (a,) = lora()
      d = datum(10)
      old = m.forward_backward(a, [d], "cross_entropy", {}, True)[0][0]
      for _ in range(5):
        m.forward_backward(a, [wire.Datum(d.tokens, {**d.inputs, "logprobs": old, "advantages": np.ones(10, np.float32)})], loss, {}, False)
        m.optim_step(a, {"learning_rate": 1e-2})
      self.assertGreater(m.forward_backward(a, [d], "cross_entropy", {}, True)[0][0].sum(), old.sum(), loss)

  def test_full_fine_tune(self):
    m = Model(tiny)
    m.create(r:=rid(), "tiny", 0, {})
    self.assertLess((losses:=train(m, r))[-1], losses[0]*0.7)
    with self.assertRaises(ValueError): m.create(rid(), "tiny", 0, {})
    with self.assertRaises(ValueError): m.create(rid(), "tiny", 4, LORA)

  def test_save_and_load_bring_back_weights_and_optimizer(self):
    m, (a, b) = lora(2)
    train(m, a, steps=3)
    m.save(a, d:=tempfile.mkdtemp(), optimizer=True)
    saved, opt = [p.detach().clone() for p in m.params[a]], copy.deepcopy(m.opts[a].state_dict())
    train(m, a, steps=3)
    m.load_weights(a, d, optimizer=True)
    self.assertTrue(all(torch.equal(x, p) for x,p in zip(saved, m.params[a])))
    torch.testing.assert_close(m.opts[a].state_dict()["state"], opt["state"])
    m.load_weights(b, d, optimizer=False)  # a run can start from another run's checkpoint
    self.assertTrue(all(torch.equal(x, p) for x,p in zip(saved, m.params[b])))
    m.create(c:=rid(), "tiny", 4, {**LORA, "train_mlp": False})
    with self.assertRaises(ValueError): m.load_weights(c, d, optimizer=False)
    with self.assertRaises(ValueError): m.load_weights(a, tempfile.mkdtemp(), optimizer=False)

  def test_sampler_weights_are_a_peft_adapter(self):
    m, (a,) = lora()
    train(m, a, steps=3)
    m.save(a, d:=tempfile.mkdtemp(), optimizer=False)
    ids = torch.randint(0, 96, (2, 8))
    with torch.no_grad(): torch.testing.assert_close(PeftModel.from_pretrained(tiny(), d).eval()(ids).logits, m.use(a).eval()(ids).logits)

  def test_mistakes_are_the_users(self):
    m, (a,) = lora()
    with self.assertRaises(KeyError): m.forward_backward(rid(), [datum(4)], "cross_entropy", {}, False)
    with self.assertRaises(ValueError): m.forward_backward(a, [datum(4)], "dpo", {}, False)
    with self.assertRaises(ValueError): m.create(rid(), "other", 4, LORA)
    m.delete(a)
    with self.assertRaises(KeyError): m.optim_step(a, {})

class TestTrainer(RedisTest):
  async def submit(self, run:str, *ops:tuple[str, dict|bytes]) -> list[str]:
    return [await self.bus.submit(run, op, b if isinstance(b, bytes) else json.dumps(b).encode()) for op,b in ops]

  async def test_commands_flow_through_and_delete_drops_the_run(self):
    await self.bus.create(Run(r:=rid(), "p", "tiny", 4))
    rids = await self.submit(r, ("create_model", {"base_model": "tiny", "rank": 4, "lora": LORA}), ("forward_backward", b"junk"),
                             ("optim_step", {"adam_params": {}}), ("delete_model", {}))
    t = Trainer(self.bus, "p", Model(tiny))
    while await t.step(block=10): pass
    reps = [await self.bus.result(x, 0) for x in rids]
    self.assertEqual([x.ok for x in reps], [True, False, True, True])
    self.assertEqual([json.loads(x.body).get("type") for x in reps], ["create_model", None, "optim_step", "delete_model"])
    self.assertEqual(json.loads(reps[1].body)["category"], "user")
    self.assertEqual(((await self.bus.run(r))["steps"], await self.bus.submit(r, "optim_step", b"{}"), t.done), (1, None, False))

  async def test_a_new_trainer_fails_what_a_dead_one_left_in_flight(self):
    await self.bus.create(Run(r:=rid(), "p", "tiny", 4))
    [rq] = await self.submit(r, ("optim_step", {"adam_params": {}}))
    await self.bus.take("p", "dead", block=10)
    t = Trainer(self.bus, "p", Model(tiny), worker="new")
    t.done = True  # serve reclaims, then returns
    await t.serve()
    rep = await self.bus.result(rq, 1)
    self.assertEqual((rep.ok, json.loads(rep.body)["category"], (await self.bus.queue(r))["in_flight"]), (False, "server", []))

  async def test_a_full_fine_tune_trainer_exits_with_its_run(self):
    await self.bus.create(Run(r:=rid(), r, "tiny", 0))
    await self.submit(r, ("create_model", {"base_model": "tiny", "rank": 0, "lora": {}}), ("delete_model", {}))
    await asyncio.wait_for((t:=Trainer(self.bus, r, Model(tiny))).serve(), 60)
    self.assertEqual((t.done, await self.bus.runs(r)), (True, []))

if __name__ == "__main__":
  unittest.main()
