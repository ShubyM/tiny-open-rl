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
from tinyrl.control import scheduler
from tinyrl.storage.store import Store
from tinyrl.workers.model import LOSSES, Model, batches, target_logprobs
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
    await self.bus.create_run(Run(run, "", "tiny", 0))
    await self.submit(run, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 0, "lora": {}}), (Op.DELETE_MODEL, {}))
    worker = await self.node.worker_for(run)
    await self.node.add(worker)
    await self.node.drain(run)
    child = self.node.processes[worker]
    await self.finish_worker(worker)
    self.assertTrue(child.closed)
    self.assertIsNone(await self.bus.lease_owner(worker))

  async def test_graceful_replacement_registers_and_restores_weights_optimizer_and_pending_gradients(self):
    self.node.gpus = 4
    self.node.gpu_ids = (0, 1, 2, 3)
    await self.node.announce_node()
    for rank, lora, width in ((4, LORA, 2), (0, {}, 1)):
      ctrl, run = new_id(), new_id()
      for r, excl in ((run, True), (ctrl, False)):
        await self.bus.create_run(Run(r, "", "tiny", rank, lora=lora, exclusive=excl, gpus=width))
        await self.submit(r, (Op.CREATE_MODEL, {"base_model": "tiny", "rank": rank, "lora": lora}))
        for _ in range(2):
          await self.submit(
            r, (Op.FORWARD_BACKWARD, proto_body(r)), (Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}})
          )
        await self.submit(r, (Op.FORWARD_BACKWARD, proto_body(r)))  # pause between accumulation and optimizer step
        await self.node.drain(r)
      p_worker = await self.node.worker_for(run)
      ctrl_worker = await self.node.worker_for(ctrl)
      first_child = self.node.processes[p_worker]
      # Issue an orderly fleet export drain so the worker snapshots weights, optimizer, RNG, and pending gradients.
      w = self.node.state.workers[p_worker]
      w.intent, w.drain_mode, w.revision = "drain", "export", w.revision + 1
      await self.bus.write_table({}, {p_worker: scheduler.worker_to_dict(w)}, [], self.node.state.workers_made, [])
      self.node.notify()
      await self.finish_worker(p_worker)
      self.assertTrue(first_child.closed)
      working = (await self.bus.get_run(run))["state"]["working"]
      self.assertIsNotNone(working)
      self.assertIsNotNone(await self.registry.get_version(working["version"]))
      [invalid] = await self.submit(run, (Op.FORWARD_BACKWARD, b"not protobuf"))
      await self.node.drain(run)  # recovery succeeds under the new generation even when the next command is invalid
      self.assertFalse((await self.bus.wait_for_reply(invalid, 1)).ok)
      for r in (ctrl, run):
        [request] = await self.submit(r, (Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}}))
        await self.node.drain(r)
        self.assertTrue((await self.bus.wait_for_reply(request, 1)).ok)
      p_worker = await self.node.worker_for(run)
      self.assertIsNot(self.node.processes[p_worker], first_child)
      self.assertTrue(same(self.model(ctrl_worker).params[ctrl], self.model(p_worker).params[run]))
      await self.submit(
        run,
        (Op.DELETE_MODEL, {}),
      )
      await self.submit(
        ctrl,
        (Op.DELETE_MODEL, {}),
      )
      await self.node.drain(run)
      await self.node.drain(ctrl)

  async def test_a_run_whose_process_died_unsaved_fails_until_it_loads_a_checkpoint(self):
    run, uri = new_id(), "tinker://saved/weights/step-1"
    await self.bus.create_run(Run(run, "", "tiny", 4, lora=LORA))
    await self.submit(
      run,
      (Op.CREATE_MODEL, {"base_model": "tiny", "rank": 4, "lora": LORA}),
      (Op.SAVE_WEIGHTS, {"uri": uri}),
      (Op.OPTIM_STEP, {"adam_params": {}}),  # acknowledged mutation past the last durable boundary
    )
    await self.node.drain(run)
    worker = await self.node.worker_for(run)
    self.node.tasks[worker].cancel()  # unexpected crash leaves no handoff snapshot covering the extra step
    await self.finish_worker(worker)
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
    self.assertIn("rolled back", json.loads(replies[0].body)["error"])


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


class TestTrainingContinuation(RedisTest):
  """Scenario 2: exact training/optimizer/gradient/RNG continuation, fresh-session reconstruction, width change and durable save semantics."""

  def test_shared_lora_trainer_isolates_per_job_rng_streams(self):
    lora_a = {"rank": 4, "lora_alpha": 16, "seed": 11, "lora_dropout": 0.25}
    lora_b = {"rank": 4, "lora_alpha": 16, "seed": 29, "lora_dropout": 0.25}
    data_a, data_b = [counting(12, 1), counting(12, 7)], [counting(12, 3), counting(12, 13)]

    solo_a, solo_b, shared = Model(tiny), Model(tiny), Model(tiny)
    solo_a.create_run("a", "tiny", 4, lora_a)
    solo_b.create_run("b", "tiny", 4, lora_b)
    shared.create_run("a", "tiny", 4, lora_a)
    shared.create_run("b", "tiny", 4, lora_b)

    for _ in range(3):
      _, ma_solo = solo_a.forward_backward("a", data_a, "cross_entropy", {}, False)
      _, ma_sh = shared.forward_backward("a", data_a, "cross_entropy", {}, False)
      _, mb_sh = shared.forward_backward("b", data_b, "cross_entropy", {}, False)
      _, mb_solo = solo_b.forward_backward("b", data_b, "cross_entropy", {}, False)
      self.assertEqual(ma_sh["loss:sum"], ma_solo["loss:sum"])
      self.assertEqual(mb_sh["loss:sum"], mb_solo["loss:sum"])
      shared.optim_step("b", {"learning_rate": 1e-2})
      shared.optim_step("a", {"learning_rate": 1e-2})
      solo_a.optim_step("a", {"learning_rate": 1e-2})
      solo_b.optim_step("b", {"learning_rate": 1e-2})

    self.assertTrue(same(solo_a.params["a"], shared.params["a"]))
    self.assertTrue(same(solo_b.params["b"], shared.params["b"]))

  async def test_nondefault_lora_with_dropout_and_full_ft_reconstruct_in_fresh_process_and_session(self):
    import sys

    import httpx

    from test.test_process import require_unix
    from tinyrl.api import gateway
    from tinyrl.control.process import Process

    require_unix()
    tmp = tempfile.mkdtemp()
    base = f"{tmp}/base"
    tiny().save_pretrained(base)
    store = Store(root=f"{tmp}/ckpt")

    # 1. Nondefault LoRA + nonzero dropout + accumulated gradients -> fresh trainer subprocess matches bit-for-bit.
    lora = {
      "rank": 6,
      "lora_alpha": 32,
      "seed": 42,
      "lora_dropout": 0.2,
      "train_attn": True,
      "train_mlp": False,
      "train_unembed": False,
    }
    ctrl = Trainer(store=store)
    ctrl.ensure_run("ctrl", base, 6, lora)
    for i in range(2):
      for op, body in (
        (Op.FORWARD_BACKWARD, proto_body("ctrl")),
        (Op.OPTIM_STEP, json.dumps({"adam_params": {"learning_rate": 1e-2}}).encode()),
      ):
        r, _ = ctrl.execute(Cmd("ctrl", f"{i}-0", op, body))
        self.assertTrue(r.ok, r.body)
    # Accumulate a pending gradient before snapshot (tests mid-accumulation recovery continuation).
    r, _ = ctrl.execute(Cmd("ctrl", "2-0", Op.FORWARD_BACKWARD, proto_body("ctrl")))
    self.assertTrue(r.ok)
    snap = ctrl.snapshot_job("ctrl", "tinker://ctrl/weights/handoff", "2-0")
    await self.registry.register_checkpoint(
      snap["uri"], snap["size"], version=snap["version"], meta=snap["meta"], durable=True
    )

    # Fresh session inspects /api/v1/weights_info with original job completely absent from Redis.
    async with httpx.AsyncClient(
      transport=httpx.ASGITransport(app=gateway.app(self.bus, store)), base_url="http://tinyrl"
    ) as http:
      wi = (await http.post("/api/v1/weights_info", json={"tinker_path": snap["uri"]})).json()
      self.assertEqual((wi["base_model"], wi["is_lora"], wi["lora_rank"]), (base, True, 6))
      self.assertEqual((wi["lora_alpha"], wi["lora_dropout"]), (32, 0.2))
      self.assertEqual((wi["train_attn"], wi["train_mlp"], wi["train_unembed"]), (True, False, False))

      # Fresh trainer subprocess reconstructs from weights_info + exact checkpoint version and continues identically.
      reconstructed = {
        "rank": wi["lora_rank"],
        "lora_alpha": wi["lora_alpha"],
        "lora_dropout": wi["lora_dropout"],
        "train_attn": wi["train_attn"],
        "train_mlp": wi["train_mlp"],
        "train_unembed": wi["train_unembed"],
      }
      fresh = Process([sys.executable, "-m", "tinyrl.workers.trainer"], {**os.environ, "CKPT": f"{tmp}/ckpt"})
      try:
        info = await fresh.call(
          "restore_job",
          "resumed",
          wi["base_model"],
          wi["lora_rank"],
          reconstructed,
          snap["uri"],
          snap["version"],
          (),
          True,
          True,
        )
        self.assertFalse(info["rng_remapped"])
        # Flush the restored pending gradient + run another dropout forward_backward + optim_step on both.
        r_c, _ = ctrl.execute(Cmd("ctrl", "2-1", Op.OPTIM_STEP, b'{"adam_params":{"learning_rate":0.01}}'))
        r_f, _ = await fresh.call(
          "execute", Cmd("resumed", "2-1", Op.OPTIM_STEP, b'{"adam_params":{"learning_rate":0.01}}')
        )
        self.assertTrue(r_c.ok and r_f.ok)
        fb_c, _ = ctrl.execute(Cmd("ctrl", "3-0", Op.FORWARD_BACKWARD, proto_body("ctrl")))
        fb_f, _ = await fresh.call("execute", Cmd("resumed", "3-0", Op.FORWARD_BACKWARD, proto_body("resumed")))
        self.assertEqual(
          wire.output_json(fb_f.body)["metrics"]["loss:sum"], wire.output_json(fb_c.body)["metrics"]["loss:sum"]
        )
        ctrl.execute(Cmd("ctrl", "3-1", Op.OPTIM_STEP, b'{"adam_params":{"learning_rate":0.01}}'))
        await fresh.call("execute", Cmd("resumed", "3-1", Op.OPTIM_STEP, b'{"adam_params":{"learning_rate":0.01}}'))
        s_c = ctrl.snapshot_job("ctrl", "tinker://ctrl/weights/end", "3-1")
        s_f = await fresh.call("snapshot_job", "resumed", "tinker://resumed/weights/end", "3-1")
        tc, _ = store.get(s_c["uri"], s_c["version"])
        tf, _ = store.get(s_f["uri"], s_f["version"])
        self.assertEqual(tc.keys(), tf.keys())
        self.assertTrue(all(torch.equal(tc[k], tf[k]) for k in tc))
      finally:
        await fresh.close()

      # 2. Fresh-session create-via-load for BOTH nondefault LoRA AND full fine-tuning after deleting the source jobs.
      node = TestNode(self.bus, store=store, model=tinyrl.workers.model.load)
      try:
        for mode, create_body in (
          ("lora", {"base_model": base, "lora_config": lora}),
          (
            "full",
            {"base_model": base, "user_metadata": {"openrl.fine_tuning_type": "full"}, "lora_config": {"seed": 7}},
          ),
        ):
          c_res = (await http.post("/api/v1/create_model", json=create_body)).json()
          src_id = c_res["model_id"]
          await node.drain(src_id)
          await http.post("/api/v1/forward_backward", content=proto_body(src_id), headers={"content-type": wire.PROTO})
          await http.post("/api/v1/optim_step", json={"model_id": src_id, "adam_params": {"learning_rate": 1e-2}})
          sv = (await http.post("/api/v1/save_weights", json={"model_id": src_id, "path": f"saved-{mode}"})).json()
          await node.drain(src_id)
          saved_path = (await http.post("/api/v1/retrieve_future", json={"request_id": sv["request_id"]})).json()[
            "path"
          ]
          # Compute control next-step loss before deleting source job.
          nxt = (
            await http.post(
              "/api/v1/forward_backward", content=proto_body(src_id), headers={"content-type": wire.PROTO}
            )
          ).json()
          await node.drain(src_id)
          ctrl_loss = (await http.post("/api/v1/retrieve_future", json={"request_id": nxt["request_id"]})).json()[
            "metrics"
          ]["loss:sum"]
          await http.post("/api/v1/delete_model", json={"model_id": src_id})
          await node.drain(src_id)

          wi2 = (await http.post("/api/v1/weights_info", json={"tinker_path": saved_path})).json()
          self.assertEqual(wi2["is_lora"], mode == "lora")
          self.assertEqual(wi2["lora_rank"], 6 if mode == "lora" else None)

          # Brand-new client session reconstructs via Tinker create-via-load load_weights.
          sid = (await http.post("/api/v1/create_session")).json()["session_id"]
          ld = (
            await http.post(
              "/api/v1/load_weights", json={"session_id": sid, "model_seq_id": 0, "path": saved_path, "optimizer": True}
            )
          ).json()
          dst_id = ld["model_id"]
          await node.drain(dst_id)
          ld_out = (await http.post("/api/v1/retrieve_future", json={"request_id": ld["request_id"]})).json()
          self.assertEqual(ld_out["type"], "load_weights")
          gi = (await http.post("/api/v1/get_info", json={"model_id": dst_id})).json()
          self.assertEqual((gi["is_lora"], gi["lora_rank"]), (mode == "lora", 6 if mode == "lora" else None))
          nxt2 = (
            await http.post(
              "/api/v1/forward_backward", content=proto_body(dst_id), headers={"content-type": wire.PROTO}
            )
          ).json()
          await node.drain(dst_id)
          dst_loss = (await http.post("/api/v1/retrieve_future", json={"request_id": nxt2["request_id"]})).json()[
            "metrics"
          ]["loss:sum"]
          self.assertAlmostEqual(dst_loss, ctrl_loss, places=6)
      finally:
        await node.aclose()

  async def test_rank_width_change_remaps_rng_and_preserves_state(self):
    import sys

    from test.test_process import require_unix
    from tinyrl.control.process import Process

    require_unix()
    tmp = tempfile.mkdtemp()
    base, ckpt = f"{tmp}/base", f"{tmp}/ckpt"
    tiny().save_pretrained(base)
    store = Store(root=ckpt)
    lora = {"rank": 4, "seed": 5, "lora_dropout": 0.15}
    env = {**os.environ, "CKPT": ckpt, "CUDA_VISIBLE_DEVICES": ""}
    args = [
      sys.executable,
      "-m",
      "torch.distributed.run",
      "--standalone",
      "--nproc-per-node=2",
      "-m",
      "tinyrl.workers.trainer",
    ]
    wide = Process(args, env)
    try:
      await asyncio.wait_for(wide.call("ensure_run", "w", base, 4, lora), 90)
      await asyncio.wait_for(wide.call("execute", Cmd("w", "1-0", Op.FORWARD_BACKWARD, proto_body("w"))), 90)
      await asyncio.wait_for(
        wide.call("execute", Cmd("w", "1-1", Op.OPTIM_STEP, b'{"adam_params":{"learning_rate":0.01}}')), 90
      )
      snap2 = await asyncio.wait_for(wide.call("snapshot_job", "w", "tinker://w/weights/from2", "1-1"), 90)

      # Restore 2-rank checkpoint onto two independent 1-rank trainers: rng_remapped=True and deterministic continuation.
      losses = []
      snap1 = None
      for tag in ("n1", "n2"):
        narrow = Trainer(store=store)
        info = narrow.restore_job(tag, base, 4, lora, snap2["uri"], snap2["version"], (), True, True)
        self.assertTrue(info["rng_remapped"])
        fb, _ = narrow.execute(Cmd(tag, "2-0", Op.FORWARD_BACKWARD, proto_body(tag)))
        t_saved, _ = store.get(snap2["uri"], snap2["version"])
        self.assertTrue(any(k.endswith(".exp_avg") and t.dtype == torch.float32 for k, t in t_saved.items()))
        losses.append(fb.body)
        narrow.execute(Cmd(tag, "2-1", Op.OPTIM_STEP, b'{"adam_params":{"learning_rate":0.01}}'))
        if tag == "n1":
          snap1 = narrow.snapshot_job(tag, "tinker://w/weights/from1", "2-1")
      self.assertEqual(losses[0], losses[1])

      # Restore 1-rank checkpoint back onto the 2-rank FSDP trainer: rng_remapped=True and clean step execution.
      info_w = await asyncio.wait_for(
        wide.call("restore_job", "w2", base, 4, lora, snap1["uri"], snap1["version"], (), True, True), 90
      )
      self.assertTrue(info_w["rng_remapped"])
      rep, _ = await asyncio.wait_for(wide.call("execute", Cmd("w2", "3-0", Op.FORWARD_BACKWARD, proto_body("w2"))), 90)
      self.assertTrue(rep.ok, rep.body)
    finally:
      await wide.close()

  async def test_sampler_export_and_tainted_state_are_rejected_for_training_recovery(self):
    store = Store(tempfile.mkdtemp())
    trainer = Trainer(Model(tiny), store)
    trainer.ensure_run("r", "tiny", 4, LORA)
    exp = trainer.save("r", "tinker://r/sampler_weights/s0", sampler=True, owned=True)
    with self.assertRaisesRegex(ValueError, "sampler export"):
      trainer.restore_job("r2", "tiny", 4, LORA, exp["uri"], exp["version"], (), optimizer=True)
    await self.bus.create_run(Run("r", "", "tiny", 4, lora=LORA))
    gen = await self.bus.get_generation("r")
    with self.assertRaisesRegex(ValueError, "sampler"):
      await self.bus.commit_job_state("r", gen, exp["uri"], exp["version"], "1-0", exp["size"], optimizer=False)

    # A mid-step failure leaves the run tainted and blocks checkpoint publication until clean restore.
    trainer.model.needs_recovery.add("r")
    with self.assertRaisesRegex(RuntimeError, "partially failed"):
      trainer.snapshot_job("r", "tinker://r/weights/bad", "1-1")
    with self.assertRaisesRegex(RuntimeError, "partially failed"):
      trainer.save("r", "tinker://r/sampler_weights/bad", sampler=True)

  async def test_explicit_save_waits_for_durable_commit_outside_gpu_survives_handoff_and_fails_on_loss(self):
    import shutil

    with tempfile.TemporaryDirectory() as tmp:
      store = Store(f"file://{tmp}/bucket", cache=f"{tmp}/cache")
      node = TestNode(self.bus, store=store, gpus=1)
      # Pause background auto-uploader so we control exact durable commit timing.
      store.stopping.set()
      store.wake.set()
      if store.uploader:
        store.uploader.join(timeout=2)
      store.stopping.clear()
      try:
        run, other = new_id(), new_id()
        await self.bus.create_run(Run(run, "", "tiny", 0))
        await self.bus.create_run(Run(other, "", "tiny-b", 0))
        c1 = await self.bus.submit_command(
          run, Op.CREATE_MODEL, json.dumps({"base_model": "tiny", "rank": 0, "lora": {}}).encode()
        )
        s0 = await self.bus.submit_command(
          run, Op.SAVE_WEIGHTS, json.dumps({"uri": f"tinker://{run}/weights/s0"}).encode()
        )
        await node.drain(run)
        self.assertTrue((await self.bus.wait_for_reply(c1, 1)).ok)
        # s0 is NOT acknowledged yet because its copy is only in local cache, not durable storage.
        self.assertIsNone(await self.bus.wait_for_reply(s0, 0))
        v0 = (await self.bus.get_run(run))["state"]["working"]["version"]
        self.assertTrue(store.upload_version(f"tinker://{run}/weights/s0", v0))
        await node.flush_durable_saves()
        self.assertTrue((await self.bus.wait_for_reply(s0, 2)).ok)
        self.assertEqual((await self.bus.get_run(run))["state"]["durable"]["version"], v0)

        # Submit s1 (un-uploaded) + subsequent work on BOTH `run` AND contending worker `other`:
        # GPU work completes without waiting on s1's durable upload, and v0 remains retained while s1 is non-durable!
        s1_uri = f"tinker://{run}/weights/s1"
        s1 = await self.bus.submit_command(run, Op.SAVE_WEIGHTS, json.dumps({"uri": s1_uri}).encode())
        nxt = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{"learning_rate":0.01}}')
        c_other = await self.bus.submit_command(
          other, Op.CREATE_MODEL, json.dumps({"base_model": "tiny-b", "rank": 0, "lora": {}}).encode()
        )
        await node.drain(run)
        await node.drain(other)
        self.assertTrue((await self.bus.wait_for_reply(nxt, 1)).ok)
        self.assertTrue((await self.bus.wait_for_reply(c_other, 1)).ok)
        self.assertIsNone(await self.bus.wait_for_reply(s1, 0))
        st = (await self.bus.get_run(run))["state"]
        v1 = st["working"]["version"]
        self.assertEqual(st["durable"]["version"], v0)
        self.assertIn(f"job:{run}:durable", await self.registry.version_refs(v0))

        # Advance job generation (planned handoff g -> g+1) BEFORE s1 finishes uploading -> durable commit still completes s1!
        gen_before = await self.bus.get_generation(run)
        await self.bus.bump_generation(run)
        self.assertEqual(await self.bus.get_generation(run), gen_before + 1)
        self.assertTrue(store.upload_version(s1_uri, v1))
        await node.flush_durable_saves()
        s1_rep = await self.bus.wait_for_reply(s1, 2)
        self.assertIsNotNone(s1_rep)
        self.assertTrue(s1_rep.ok, s1_rep.body)
        self.assertEqual((await self.bus.get_run(run))["state"]["durable"]["version"], v1)
        self.assertNotIn(f"job:{run}:durable", await self.registry.version_refs(v0))

        # Sync generation for s2: first commit s2 from the step after s1 by running load_weights(s1) + s2,
        # then lose s2's un-uploaded local copy -> s2 fails cleanly and v1 remains the committed durable checkpoint!
        s_tbl = await node.read_state()
        rw = s_tbl.jobs[run].worker
        s_tbl.workers[rw].generations[run] = await self.bus.get_generation(run)
        s_tbl.workers[rw].revision += 1
        await self.bus.write_table({}, {rw: scheduler.worker_to_dict(s_tbl.workers[rw])}, [], s_tbl.workers_made, [])
        lw = await self.bus.submit_command(
          run, Op.LOAD_WEIGHTS, json.dumps({"uri": s1_uri, "optimizer": True}).encode()
        )
        s2_uri = f"tinker://{run}/weights/s2"
        s2 = await self.bus.submit_command(run, Op.SAVE_WEIGHTS, json.dumps({"uri": s2_uri}).encode())
        await node.drain(run)
        self.assertTrue((await self.bus.wait_for_reply(lw, 2)).ok)
        self.assertIsNone(await self.bus.wait_for_reply(s2, 0))
        v2 = (await self.bus.get_run(run))["state"]["working"]["version"]
        shutil.rmtree(f"{tmp}/cache/{v2}", ignore_errors=True)
        for ext in (".manifest.json", ".upload"):
          if os.path.exists(f"{tmp}/cache/{v2}{ext}"):
            os.remove(f"{tmp}/cache/{v2}{ext}")
        self.assertTrue(await self.bus.fail_pending_save(run, s2, "source node lost before durable upload"))
        s2_rep = await self.bus.wait_for_reply(s2, 2)
        self.assertIsNotNone(s2_rep)
        self.assertFalse(s2_rep.ok)
        self.assertIn("source node lost", s2_rep.body.decode())
        self.assertEqual((await self.bus.get_run(run))["state"]["durable"]["version"], v1)
      finally:
        await node.aclose()


if __name__ == "__main__":
  unittest.main()
