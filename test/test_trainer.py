import asyncio
import copy
import json
import os
import tempfile
import unittest

import numpy as np
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM

import tinyrl.workers.model
from test.execution import TestNode
from test.helpers import RedisTest, new_id, tiny
from test.test_api import proto_body
from tinyrl.api import wire
from tinyrl.bus import Cmd, Op, Run
from tinyrl.storage.store import Store
from tinyrl.workers.model import LOSSES, Model, batches, causal_attention, target_logprobs
from tinyrl.workers.trainer import Trainer

LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


def counting(n: int, start: int = 0, **inputs) -> wire.Datum:
  """A sequence that counts up from `start`: each token's target is the next one."""
  tokens = np.arange(start, start + n + 1) % 96
  return wire.Datum(tokens[:-1].astype(np.int32), {"target_tokens": tokens[1:].astype(np.int64), **inputs})


def train(model: Model, run: str, steps: int = 8, lr: float = 1e-2) -> list[float]:
  losses = []
  for _ in range(steps):
    _, metrics = model.forward_backward(run, [counting(12), counting(9, start=40)], "cross_entropy", {}, False)
    model.optim_step(run, {"learning_rate": lr})
    losses.append(metrics["loss:sum"])
  return losses


def lora_runs(n: int = 1) -> tuple[Model, list[str]]:
  model, runs = Model(tiny), [new_id() for _ in range(n)]
  for run in runs:
    model.create_run(run, "tiny", 4, LORA)
  return model, runs


def same(a, b) -> bool:
  av = a.values() if isinstance(a, dict) else a
  bv = b.values() if isinstance(b, dict) else b
  return all(torch.equal(x, y) for x, y in zip(av, bv))


class TestForwardPass(unittest.TestCase):
  def test_batches(self):
    self.assertEqual(batches([5, 5, 5], 100), [[0, 1, 2]])
    self.assertEqual(batches([5, 5, 5], 10), [[0, 1], [2]])
    self.assertEqual(batches([3, 1, 2], 0), [[1], [2], [0]])  # no budget means one sequence per pass
    # a long sequence gets a pass of its own instead of padding the short ones
    self.assertEqual(batches([10] * 6 + [50], 1000), [[0, 1, 2, 3, 4, 5], [6]])

    lengths = [int(n) for n in np.random.default_rng(0).integers(1, 64, 200)]
    passes = batches(lengths, 256)
    self.assertEqual(sorted(i for p in passes for i in p), list(range(200)))
    self.assertTrue(all(len(p) == 1 or max(lengths[i] for i in p) * len(p) <= 256 for p in passes))

  def test_chunked_logprobs_are_the_full_softmax(self):
    lm = tiny().eval()
    shape = (3, tinyrl.workers.model.CHUNK // 2 + 1)  # enough positions to cross a chunk boundary
    ids, targets = torch.randint(0, 96, shape), torch.randint(0, 96, shape)
    mask = torch.ones_like(ids)
    mask[2, 7:] = 0

    with torch.no_grad():
      got = target_logprobs(lm, ids, mask, targets)
      want = lm(ids, attention_mask=mask).logits.log_softmax(-1).gather(-1, targets[..., None])[..., 0]
    torch.testing.assert_close(got[mask.bool()], want[mask.bool()])

  def test_attention_without_a_mask_is_attention_under_the_causal_and_the_sliding_mask(self):
    torch.manual_seed(0)
    n, heads, kv_heads = 70, 4, 2
    query, key, value = torch.randn(2, heads, n, 8), torch.randn(2, kv_heads, n, 8), torch.randn(2, kv_heads, n, 8)
    module = type("Layer", (), {"num_key_value_groups": heads // kv_heads})()
    q, k = torch.arange(n)[:, None], torch.arange(n)[None]
    wide = lambda t: t.repeat_interleave(heads // kv_heads, dim=1)  # noqa: E731
    # 16 and 31 put block edges inside the row, 70 and 200 reach every earlier position
    for window in (None, 16, 31, 70, 200):
      mask = (k <= q) & (k > q - window if window else True)
      want = torch.nn.functional.scaled_dot_product_attention(query, wide(key), wide(value), attn_mask=mask, scale=0.3)
      got, _ = causal_attention(module, query, key, value, None, scaling=0.3, sliding_window=window)
      self.assertTrue(torch.allclose(got, want.transpose(1, 2), atol=1e-5), window)

  def test_losses_are_their_definitions(self):
    g = torch.Generator().manual_seed(0)
    logprobs, sampler_logprobs = -torch.rand(4, 9, generator=g), -torch.rand(4, 9, generator=g)
    advantages, weights = torch.randn(4, 9, generator=g), torch.rand(4, 9, generator=g)
    ratio = torch.exp(logprobs - sampler_logprobs)

    def loss(name: str, **config) -> torch.Tensor:
      return LOSSES[name](logprobs, weights, sampler_logprobs, advantages, config)

    torch.testing.assert_close(loss("cross_entropy"), -logprobs * weights)
    torch.testing.assert_close(loss("importance_sampling"), -ratio * advantages * weights)
    clipped = torch.min(ratio * advantages, ratio.clamp(0.9, 1.1) * advantages)
    torch.testing.assert_close(loss("ppo", clip_range=0.1), -clipped * weights)
    torch.testing.assert_close(loss("ppo", clip_low_threshold=0.9, clip_high_threshold=1.1), -clipped * weights)
    kl = (ratio - 1) - (logprobs - sampler_logprobs)
    torch.testing.assert_close(loss("ppo", kl_coeff=0.5) - loss("ppo"), 0.5 * kl * weights)
    with self.assertRaises(ValueError):
      loss("ppo", unknown=1.0)


class TestModel(unittest.TestCase):
  def test_a_corrupt_checkpoint_cannot_partially_overwrite_the_model(self):
    model, run_id = Model(tiny), new_id()
    model.create_run(run_id, "tiny", 0, {})
    train(model, run_id, steps=1)
    tensors, meta = copy.deepcopy(model.training_state(run_id))
    tensors.pop(next(k for k in tensors if k.endswith(".step")))
    train(model, run_id, steps=1)
    original_params = {k: p.clone() for k, p in model.params[run_id].items()}
    with self.assertRaisesRegex(ValueError, "bad:"):
      model.restore_run(run_id, tensors, meta, True)
    torch.testing.assert_close(model.params[run_id], original_params, atol=0, rtol=0)

  def test_a_failed_pass_discards_partial_gradients(self):
    model, run_id = Model(tiny), new_id()
    model.create_run(run_id, "tiny", 0, {})
    good, bad = counting(2), counting(8)  # length disparity gives each datum its own pass
    bad.inputs["target_tokens"][0] = 96  # outside the model's vocabulary; the first pass already accumulated
    with self.assertRaises((IndexError, RuntimeError)):
      model.forward_backward(run_id, [good, bad], "cross_entropy", {}, False)
    self.assertTrue(all(p.grad is None for p in model.params[run_id].values()))
    model.forward_backward(run_id, [good], "cross_entropy", {}, False)
    with self.assertRaisesRegex(RuntimeError, "a forward_backward failed after it began"):
      model.optim_step(run_id, {"learning_rate": 1e-2})
    model.forward_backward(run_id, [good], "cross_entropy", {}, False)
    model.optim_step(run_id, {"learning_rate": 1e-2})

  def test_lora_runs_share_a_frozen_base_and_train_apart(self):
    model, (a, b) = lora_runs(2)
    base = {name: p.detach().clone() for name, p in model.lm.named_parameters() if "lora" not in name}
    b_before = [p.detach().clone() for p in model.params[b].values()]

    losses = train(model, a)
    self.assertLess(losses[-1], losses[0] * 0.7)
    self.assertTrue(same(b_before, model.params[b]))
    self.assertTrue(all(torch.equal(base[name], p) for name, p in model.lm.named_parameters() if "lora" not in name))

  def test_interleaved_runs_keep_their_own_gradients(self):
    model, (a, b) = lora_runs(2)
    model.forward_backward(a, [counting(8)], "cross_entropy", {}, False)
    model.forward_backward(b, [counting(8, start=5)], "cross_entropy", {}, False)
    b_grads = [p.grad.clone() for p in model.params[b].values()]

    model.optim_step(a, {"learning_rate": 1e-2})
    self.assertTrue(all(p.grad is None for p in model.params[a].values()))
    self.assertTrue(same(b_grads, [p.grad for p in model.params[b].values()]))

  def test_forward_only_leaves_no_gradient(self):
    model, (a,) = lora_runs()
    logprobs, metrics = model.forward_backward(a, [counting(8), counting(5)], "cross_entropy", {}, True)
    self.assertTrue(all(p.grad is None for p in model.params[a].values()))
    self.assertEqual([lp.shape for lp in logprobs], [(8,), (5,)])
    self.assertAlmostEqual(metrics["loss:sum"], -sum(float(lp.sum()) for lp in logprobs), places=4)

  def test_rl_losses_move_logprobs_with_the_advantage(self):
    for loss in ("importance_sampling", "ppo"):
      model, (a,) = lora_runs()
      datum = counting(10)
      before = model.forward_backward(a, [datum], "cross_entropy", {}, True)[0][0]

      inputs = {**datum.inputs, "logprobs": before, "advantages": np.ones(10, np.float32)}
      for _ in range(5):
        model.forward_backward(a, [wire.Datum(datum.tokens, inputs)], loss, {}, False)
        model.optim_step(a, {"learning_rate": 1e-2})

      after = model.forward_backward(a, [datum], "cross_entropy", {}, True)[0][0]
      self.assertGreater(after.sum(), before.sum(), loss)

  def test_full_fine_tune(self):
    model, run = Model(tiny), new_id()
    model.create_run(run, "tiny", 0, {})
    losses = train(model, run)
    self.assertLess(losses[-1], losses[0] * 0.7)

    with self.assertRaises(ValueError):  # a full fine-tune has its trainer to itself
      model.create_run(new_id(), "tiny", 0, {})
    with self.assertRaises(ValueError):
      model.create_run(new_id(), "tiny", 4, LORA)

  def test_the_optimizer_updates_fp32_master_weights_and_casts_sampler_weights_to_bf16(self):
    def bf16(name: str) -> LlamaForCausalLM:  # how a base model loads on the gpu
      return tiny(name).to(torch.bfloat16)

    for rank in (0, 4):
      model, run = Model(bf16), new_id()
      model.create_run(run, "tiny", rank, {"seed": 0})
      model.forward_backward(run, [counting(12)], "cross_entropy", {}, False)
      grads = [p.grad for p in model.params[run].values()]
      model.optim_step(run, {"learning_rate": 1e-2})
      moments = [m for state in model.optimizers[run].state.values() for m in (state["exp_avg"], state["exp_avg_sq"])]
      self.assertEqual(
        {t.dtype for t in [*model.params[run].values(), *grads, *moments]}, {torch.float32}, f"rank {rank}"
      )
      self.assertEqual({t.dtype for t in model.sampler_weights(run)[0].values()}, {torch.bfloat16}, f"rank {rank}")
      if rank:  # a LoRA run's base model is frozen, so it stays bf16
        self.assertEqual(model.lm.get_input_embeddings().weight.dtype, torch.bfloat16)

  def test_save_and_load_bring_back_weights_and_optimizer(self):
    store = Store(root=tempfile.mkdtemp())
    model, (a, b) = lora_runs(2)
    train(model, a, steps=3)
    uri = f"tinker://{a}/weights/step-3"
    store.put(uri, *model.training_state(a))
    saved_weights = [p.detach().clone() for p in model.params[a].values()]
    saved_optimizer = copy.deepcopy(model.optimizers[a].state_dict())

    train(model, a, steps=3)
    model.restore_run(a, *store.get(uri), optimizer=True)
    self.assertTrue(same(saved_weights, model.params[a]))
    torch.testing.assert_close(model.optimizers[a].state_dict()["state"], saved_optimizer["state"])

    model.restore_run(b, *store.get(uri), optimizer=False)  # a run can start from another run's checkpoint
    self.assertTrue(same(saved_weights, model.params[b]))

  def test_loading_a_checkpoint_that_does_not_fit_fails(self):
    store = Store(root=tempfile.mkdtemp())
    model, (a,) = lora_runs()
    uri = f"tinker://{a}/sampler_weights/step-0"
    store.put(uri, *model.sampler_weights(a))

    no_mlp = new_id()
    model.create_run(no_mlp, "tiny", 4, {**LORA, "train_mlp": False})
    with self.assertRaises(ValueError):
      model.restore_run(no_mlp, *store.get(uri), optimizer=False)
    with self.assertRaises(ValueError):
      model.restore_run(a, *store.get(uri), optimizer=True)  # it was saved without the optimizer
    with self.assertRaises(ValueError):
      store.get(f"tinker://{a}/weights/missing")

  def test_explicit_load_uses_the_saved_version_and_discards_pending_gradients(self):
    with tempfile.TemporaryDirectory() as folder:
      store = Store(root=f"file://{folder}/remote", cache=f"{folder}/cache")
      model, (run,) = lora_runs()
      original, loaded = Trainer(model, store), Trainer(Model(tiny), store)
      train(model, run, steps=1)
      model.forward_backward(run, [counting(12)], "cross_entropy", {}, False)
      uri = f"tinker://{run}/weights/saved"
      first = original.save(run, uri)
      expected = {name: p.detach().clone() for name, p in model.params[run].items()}
      train(model, run, steps=1)
      second = original.save(run, uri)
      self.assertNotEqual(first["version"], second["version"])
      reply, _ = loaded.execute(
        Cmd(run, "1-0", Op.LOAD_WEIGHTS, json.dumps({"uri": uri, "optimizer": True}).encode()),
        run={"base": "tiny", "rank": 4, "lora": LORA},
        version=first["version"],
      )
      self.assertTrue(reply.ok, reply.body)
      torch.testing.assert_close(loaded.model.params[run], expected, atol=0, rtol=0)
      self.assertTrue(all(p.grad is None for p in loaded.model.params[run].values()))
      self.assertTrue(loaded.model.optimizers[run].state)

  def test_sampler_weights_load_in_peft_and_transformers(self):
    store = Store(root=tempfile.mkdtemp())
    ids = torch.randint(0, 96, (2, 8))

    model, (lora_run,) = lora_runs()
    train(model, lora_run, steps=3)
    store.put(f"tinker://{lora_run}/sampler_weights/final", *model.sampler_weights(lora_run))
    lora_dir = store.path(f"tinker://{lora_run}/sampler_weights/final")
    with torch.no_grad():
      torch.testing.assert_close(
        PeftModel.from_pretrained(tiny(), lora_dir).eval()(ids).logits,
        model.activate_run(lora_run).eval()(ids).logits,
      )

    fft, fft_run = Model(tiny), new_id()
    fft.create_run(fft_run, "tiny", 0, {})
    train(fft, fft_run, steps=3)
    store.put(f"tinker://{fft_run}/sampler_weights/final", *fft.sampler_weights(fft_run))
    fft_dir = store.path(f"tinker://{fft_run}/sampler_weights/final")
    with torch.no_grad():
      torch.testing.assert_close(
        LlamaForCausalLM.from_pretrained(fft_dir).eval()(ids).logits,
        fft.activate_run(fft_run).eval()(ids).logits,
      )


class TestTrainer(RedisTest):
  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.store = Store(root=tempfile.mkdtemp())
    self.node = TestNode(self.bus, store=self.store)

  async def asyncTearDown(self):
    await self.node.aclose()
    await super().asyncTearDown()

  async def submit(self, run: str, *cmds: tuple[Op, dict | bytes]) -> list[str]:
    return [
      await self.bus.submit_command(run, op, body if isinstance(body, bytes) else json.dumps(body).encode())
      for op, body in cmds
    ]

  def model(self, worker):
    return self.node.processes[worker].trainer.model

  async def until(self, predicate):
    async with asyncio.timeout(10):
      while not predicate():
        self.node.dispatch()
        await asyncio.sleep(0.01)

  async def finish_worker(self, worker):
    await self.until(lambda: worker not in self.node.running)

  async def test_a_full_fine_tune_process_closes_with_its_run(self):
    run = new_id()
    await self.bus.create_run(Run(run, run, "tiny", 0))
    await self.submit(run, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 0, "lora": {}}), (Op.DELETE_MODEL, {}))
    await self.node.add(run)
    child = self.node.processes[run]
    await self.finish_worker(run)
    self.assertTrue(child.closed)
    self.assertIsNone(await self.bus.lease_owner(run))

  async def test_graceful_replacement_registers_and_restores_weights_optimizer_and_pending_gradients(self):
    self.node.gpus = 2
    self.node.state.nodes[self.node.id] = (2, 0)
    for rank, lora, width in ((4, LORA, 2), (0, {}, 1)):
      ctrl, run = new_id(), new_id()
      for r, worker in ((run, f"p-{rank}"), (ctrl, f"ctrl-{rank}")):
        await self.bus.create_run(Run(r, worker, "tiny", rank, lora=lora, gpus=width))
        await self.submit(r, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": rank, "lora": lora}))
        for _ in range(2):
          await self.submit(
            r, (Op.FORWARD_BACKWARD, proto_body(r)), (Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}})
          )
        await self.submit(r, (Op.FORWARD_BACKWARD, proto_body(r)))  # pause between accumulation and optimizer step
        await self.node.drain(r)
      desired = self.node.state.workers[f"p-{rank}"]
      desired.node, desired.gpus = None, ()  # the parked wide worker must regain its actual GPUs to checkpoint
      self.node.notify()
      await self.finish_worker(f"p-{rank}")
      parked = await self.bus.parked_checkpoint(run)
      self.assertIsNotNone(parked)
      self.assertIsNotNone(await self.registry.get_checkpoint(parked))
      [invalid] = await self.submit(run, (Op.FORWARD_BACKWARD, b"not protobuf"))
      await self.node.drain(run)  # recovery succeeds even when the following command is rejected
      self.assertFalse((await self.bus.wait_for_reply(invalid, 1)).ok)
      for r in (ctrl, run):
        [request] = await self.submit(r, (Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}}))
        await self.node.drain(r)
        self.assertTrue((await self.bus.wait_for_reply(request, 1)).ok)
      self.assertTrue(same(self.model(f"ctrl-{rank}").params[ctrl], self.model(f"p-{rank}").params[run]))
      self.assertIsNone(await self.bus.parked_checkpoint(run))
      self.assertIsNone(await self.registry.get_checkpoint(parked))  # recovery cannot replay old state after a crash

  async def test_a_run_whose_process_died_unsaved_fails_until_it_loads_a_checkpoint(self):
    run, uri = new_id(), "tinker://saved/weights/step-1"
    await self.bus.create_run(Run(run, "p", "tiny", 4, lora=LORA))
    await self.submit(
      run, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA}), (Op.SAVE_WEIGHTS, {"uri": uri})
    )
    await self.node.drain(run)
    self.node.tasks["p"].cancel()  # a lost process has no park checkpoint
    await self.finish_worker("p")
    ids = await self.submit(
      run,
      (Op.OPTIM_STEP, {"adam_params": {}}),
      (Op.LOAD_WEIGHTS, {"uri": "tinker://gone/weights/missing", "optimizer": True}),
      (Op.OPTIM_STEP, {"adam_params": {}}),
      (Op.LOAD_WEIGHTS, {"uri": uri, "optimizer": True}),
      (Op.OPTIM_STEP, {"adam_params": {}}),
    )
    await self.node.drain(run)
    replies = [await self.bus.wait_for_reply(request_id, 1) for request_id in ids]
    self.assertEqual([reply.ok for reply in replies], [False, False, False, True, True])
    self.assertIn("lost its unsaved steps", json.loads(replies[0].body)["error"])


class TestDistributedTrainer(unittest.IsolatedAsyncioTestCase):
  async def test_two_rank_fsdp_matches_a_single_model_process(self):
    import sys

    from test.test_process import require_unix
    from tinyrl.control.process import Process

    require_unix()
    base = tempfile.mkdtemp()
    tiny().save_pretrained(base)
    for rank, lora in ((0, {"seed": 0}), (4, LORA)):
      ckpt, r1, r2 = tempfile.mkdtemp(), new_id(), new_id()
      store = Store(root=ckpt)
      single = Trainer(store=store)
      env = {**os.environ, "CKPT": ckpt, "CUDA_VISIBLE_DEVICES": ""}
      args = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]
      multi = Process([*args, "-m", "tinyrl.workers.trainer"], env)
      try:
        for run in (r1, r2):
          invalid = wire.parse(proto_body(run))
          invalid.loss_fn = "unknown"  # a rejected operation must leave every rank ready for the next collective
          commands = [
            (Op.CREATE_MODEL, {"base_model": base, "rank": rank, "lora": lora}),
            (Op.FORWARD_BACKWARD, invalid.SerializeToString()),
            (Op.FORWARD_BACKWARD, proto_body(run)),
            (Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}}),
            (Op.FORWARD_BACKWARD, proto_body(run)),
            (Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}}),
            (Op.SAVE_WEIGHTS_FOR_SAMPLER, {"uri": f"tinker://{run}/sampler_weights/s1"}),
          ]
          for i, (op, body) in enumerate(commands):
            cmd = Cmd(run, str(i), op, body if isinstance(body, bytes) else json.dumps(body).encode())
            reply, _ = single.execute(cmd) if run == r1 else await asyncio.wait_for(multi.call("execute", cmd), 90)
            self.assertEqual(reply.ok, i != 1, reply.body)
        expected, actual = (store.get(f"tinker://{run}/sampler_weights/s1")[0] for run in (r1, r2))
        self.assertEqual(expected.keys(), actual.keys(), f"rank {rank}")
        for name in expected:
          torch.testing.assert_close(actual[name], expected[name], atol=1e-5, rtol=1e-5, msg=f"rank {rank} {name}")
      finally:
        await multi.close()


class TestCheckpointNames(unittest.TestCase):
  def test_a_multimodal_checkpoint_trains_under_its_own_parameter_names(self):
    """Qwen3.5's checkpoint and vLLM name its layers model.language_model.*, its causal LM class model.layers.*. The trainer loads the class the checkpoint names, so a saved adapter's weights carry names vLLM maps."""
    from safetensors import safe_open
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    text = {
      "hidden_size": 32,
      "intermediate_size": 64,
      "num_hidden_layers": 2,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 16,
      "linear_num_key_heads": 1,
      "linear_num_value_heads": 2,
      "linear_key_head_dim": 16,
      "linear_value_head_dim": 16,
      "vocab_size": 128,
      "layer_types": ["linear_attention", "full_attention"],
    }
    vision = {"hidden_size": 16, "intermediate_size": 32, "depth": 1, "num_heads": 2, "out_hidden_size": 32}
    path = tempfile.mkdtemp()
    Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vision)).save_pretrained(path)
    with safe_open(f"{path}/model.safetensors", "pt") as f:
      names = set(f.keys())

    model, run = Model(tinyrl.workers.model.load), new_id()
    model.create_run(run, path, 4, LORA)
    train(model, run, steps=1)
    tensors, meta = model.sampler_weights(run)
    for key in tensors:  # base_model.model.<module>.lora_A.<...>: the module is one the checkpoint has
      module = key.removeprefix("base_model.model.").split(".lora_")[0]
      self.assertIn(f"{module}.weight", names)
    self.assertTrue(any(".language_model.layers." in key for key in tensors))

  def test_gemma4_lora_skips_clippable_vision_and_audio_wrappers(self):
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    cfg = Gemma4Config(
      text_config={
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "vocab_size": 128,
        "num_kv_shared_layers": 1,
        "layer_types": ["sliding_attention", "full_attention"],
        "hidden_size_per_layer_input": 0,
      },
      vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2},
      audio_config={"hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2, "conf_conv_kernel_size": 3},
    )
    path = tempfile.mkdtemp()
    Gemma4ForConditionalGeneration(cfg).save_pretrained(path)

    model, a, b = Model(tinyrl.workers.model.load), new_id(), new_id()
    model.create_run(a, path, 4, LORA)
    model.create_run(b, path, 4, LORA)
    tensors, _ = model.sampler_weights(b)
    self.assertTrue(tensors)
    self.assertTrue(all(".language_model." in k for k in tensors))


if __name__ == "__main__":
  unittest.main()
