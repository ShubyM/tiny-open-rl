import json
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import torch

from test.helpers import new_id, tiny
from tinyrl.api import wire
from tinyrl.bus import Cmd, Op
from tinyrl.storage.store import Store, nbytes
from tinyrl.workers.model import Model
from tinyrl.workers.trainer import Trainer, serve_follower

LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


def forward_body(run: str) -> bytes:
  return wire.from_json(
    {
      "model_id": run,
      "forward_backward_input": {
        "data": [
          {
            "model_input": {"chunks": [{"tokens": list(range(start, start + 8))}]},
            "loss_fn_inputs": {"target_tokens": list(range(start + 1, start + 9))},
          }
          for start in (0, 40)
        ]
      },
    }
  )


def command(run: str, op: Op, body: dict | bytes) -> Cmd:
  return Cmd(run, "0-1", op, body if isinstance(body, bytes) else json.dumps(body).encode())


class TestTrainerChild(unittest.TestCase):
  def setUp(self):
    folder = tempfile.TemporaryDirectory()
    self.addCleanup(folder.cleanup)
    self.store = Store(root=folder.name)

  def trainer(self) -> Trainer:
    return Trainer(Model(tiny), self.store)

  def call(self, trainer: Trainer, run_id: str, op: Op, body: dict | bytes, **metadata):
    reply, effects = trainer.execute(command(run_id, op, body), **metadata)
    self.assertTrue(reply.ok, reply.body)
    json.dumps(effects)  # checkpoint metadata can travel to the node without any model tensors
    return reply, effects

  def create(self, trainer: Trainer, rank: int = 4, run: str | None = None) -> str:
    run = run or new_id()
    self.call(trainer, run, Op.CREATE_MODEL, {"base_model": "tiny", "rank": rank, "lora": LORA})
    return run

  def step(self, trainer: Trainer, run: str):
    reply, _ = self.call(trainer, run, Op.FORWARD_BACKWARD, forward_body(run))
    self.call(trainer, run, Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}})
    return wire.output_json(reply.body)["metrics"]["loss:sum"]

  def test_save_returns_registration_metadata_and_the_existing_reply(self):
    trainer = self.trainer()
    run = self.create(trainer)
    self.step(trainer, run)
    for op, kind in ((Op.SAVE_WEIGHTS, "weights"), (Op.SAVE_WEIGHTS_FOR_SAMPLER, "sampler_weights")):
      with self.subTest(op=op):
        uri = f"tinker://{run}/{kind}/test"
        reply, effects = self.call(
          trainer,
          run,
          op,
          {
            "uri": uri,
            "ttl_seconds": 60,
            "owned": True,
            "sampling_session_id": "sample-session",
          },
        )
        tensors, _ = self.store.get(uri, effects["saved"]["version"])
        self.assertEqual(
          effects,
          {
            "saved": {
              "uri": uri,
              "size": sum(map(nbytes, tensors.values())),
              "ttl": 60,
              "owned": True,
              "version": self.store.versions.get(uri),
            }
          },
        )
        self.assertEqual(
          json.loads(reply.body),
          {
            "type": op,
            "model_id": run,
            "path": uri,
            "sampling_session_id": "sample-session",
          },
        )

  def test_load_uses_the_version_supplied_by_the_node_after_the_name_is_saved_again(self):
    with tempfile.TemporaryDirectory() as folder:
      store = Store(root=f"file://{folder}/remote", cache=f"{folder}/cache")
      original, loaded = Trainer(Model(tiny), store), Trainer(Model(tiny), store)
      run = self.create(original)
      self.step(original, run)
      uri = f"tinker://{run}/weights/overwrite"
      _, first = self.call(original, run, Op.SAVE_WEIGHTS, {"uri": uri})
      expected = {name: p.detach().clone() for name, p in original.model.params[run].items()}
      self.step(original, run)
      _, second = self.call(original, run, Op.SAVE_WEIGHTS, {"uri": uri})
      self.assertIsNotNone(first["saved"]["version"])
      self.assertNotEqual(first["saved"]["version"], second["saved"]["version"])
      self.call(
        loaded,
        run,
        Op.LOAD_WEIGHTS,
        {"uri": uri, "optimizer": True},
        run={"base": "tiny", "rank": 4, "lora": LORA},
        version=first["saved"]["version"],
      )
      torch.testing.assert_close(loaded.model.params[run], expected, atol=0, rtol=0)

  def test_a_parked_run_preserves_pending_gradients_and_optimizer_state(self):
    for rank in (0, 4):
      with self.subTest(rank=rank):
        original, restored = self.trainer(), self.trainer()
        run = self.create(original, rank)
        self.step(original, run)
        self.call(original, run, Op.FORWARD_BACKWARD, forward_body(run))
        [saved] = original.park_all()
        self.assertEqual(saved["run"], run)
        self.assertTrue(saved["owned"])
        json.dumps(saved)
        tensors, _ = self.store.get(saved["uri"], saved["version"])
        self.assertTrue(any(k.endswith(".grad") for k in tensors))
        self.assertTrue(any(k.endswith(".exp_avg") for k in tensors))
        metadata = {"base": "tiny", "rank": rank, "lora": LORA}
        _, effects = self.call(
          restored,
          run,
          Op.OPTIM_STEP,
          {"adam_params": {"learning_rate": 1e-2}},
          run=metadata,
          parked=(saved["uri"], saved["version"]),
        )
        self.assertEqual(effects, {"restored": saved["uri"]})
        self.call(original, run, Op.OPTIM_STEP, {"adam_params": {"learning_rate": 1e-2}})
        torch.testing.assert_close(
          restored.model.training_state(run), original.model.training_state(run), atol=0, rtol=0
        )
        self.assertEqual(self.step(restored, run), self.step(original, run))

  def test_explicit_load_discards_saved_gradients(self):
    original, loaded = self.trainer(), self.trainer()
    run = self.create(original, rank=0)
    self.step(original, run)
    self.call(original, run, Op.FORWARD_BACKWARD, forward_body(run))
    uri = f"tinker://{run}/weights/mid-step"
    _, effects = self.call(original, run, Op.SAVE_WEIGHTS, {"uri": uri})
    self.call(
      loaded,
      run,
      Op.LOAD_WEIGHTS,
      {"uri": uri, "optimizer": True},
      run={"base": "tiny", "rank": 0},
      version=effects["saved"]["version"],
    )
    self.assertTrue(all(p.grad is None for p in loaded.model.params[run].values()))
    self.assertTrue(loaded.model.optimizers[run].state)
    for name, param in original.model.params[run].items():
      torch.testing.assert_close(loaded.model.params[run][name], param, atol=0, rtol=0)

  def test_a_crashed_run_fails_until_it_successfully_loads(self):
    original, replacement = self.trainer(), self.trainer()
    run = self.create(original)
    uri = f"tinker://{run}/weights/saved"
    _, saved = self.call(original, run, Op.SAVE_WEIGHTS, {"uri": uri})
    metadata = {"base": "tiny", "rank": 4, "lora": LORA}
    for body in ({"uri": f"tinker://{run}/weights/missing", "optimizer": True}, None):
      if body is not None:
        reply, _ = replacement.execute(command(run, Op.LOAD_WEIGHTS, body), run=metadata)
        self.assertFalse(reply.ok)
        self.assertNotIn(run, replacement.model.params)
      reply, effects = replacement.execute(command(run, Op.OPTIM_STEP, {"adam_params": {}}), run=metadata)
      self.assertFalse(reply.ok)
      self.assertIn("lost its unsaved steps", json.loads(reply.body)["error"])
      self.assertEqual(effects, {})
    self.call(
      replacement,
      run,
      Op.LOAD_WEIGHTS,
      {"uri": uri, "optimizer": True},
      run=metadata,
      version=saved["saved"]["version"],
    )
    self.call(replacement, run, Op.OPTIM_STEP, {"adam_params": {}})

  def test_failed_park_restore_removes_the_partially_created_run(self):
    for rank in (0, 4):
      with self.subTest(rank=rank):
        trainer, run = self.trainer(), new_id()
        metadata = {"base": "tiny", "rank": rank, "lora": LORA}
        reply, effects = trainer.execute(
          command(run, Op.OPTIM_STEP, {"adam_params": {}}),
          run=metadata,
          parked=(f"tinker://{run}/parked/missing", None),
        )
        self.assertFalse(reply.ok)
        self.assertEqual(effects, {})
        self.assertNotIn(run, trainer.model.params)

  def test_a_rejected_command_still_reports_successful_recovery(self):
    original, restored = self.trainer(), self.trainer()
    run = self.create(original)
    [saved] = original.park_all()
    reply, effects = restored.execute(
      command(run, Op.FORWARD_BACKWARD, b"not protobuf"),
      run={"base": "tiny", "rank": 4, "lora": LORA},
      parked=(saved["uri"], saved["version"]),
    )
    self.assertFalse(reply.ok)
    self.assertEqual(json.loads(reply.body)["category"], "user")
    self.assertEqual(effects, {"restored": saved["uri"]})
    self.assertIn(run, restored.model.params)

  def test_deleting_a_missing_run_does_not_try_to_restore_it(self):
    trainer, run = self.trainer(), new_id()
    with patch.object(self.store, "get", side_effect=AssertionError("must not restore")):
      reply, effects = self.call(trainer, run, Op.DELETE_MODEL, {}, parked=("tinker://missing/parked/1", None))
    self.assertEqual(json.loads(reply.body), {"type": "delete_model", "model_id": run})
    self.assertEqual(effects, {})

  def test_resume_sets_the_device_on_the_model_thread_before_restoring_tensors(self):
    seen = []
    model = Mock()
    model.resume_on_gpu.side_effect = lambda: seen.append(("resume", threading.get_ident()))
    trainer = Trainer(model, self.store)
    with (
      patch("torch.cuda.is_available", return_value=True),
      patch("torch.cuda.set_device", side_effect=lambda gpu: seen.append((gpu, threading.get_ident()))),
    ):
      trainer.resume(2)
    self.assertEqual(seen, [(2, threading.get_ident()), ("resume", threading.get_ident())])

  def test_rank_zero_broadcasts_model_operations(self):
    trainer = self.trainer()
    trainer.world_size = 2
    with (
      patch("torch.distributed.broadcast_object_list") as broadcast,
      patch.object(trainer.model, "park_on_host") as park,
    ):
      trainer.park()
      trainer.resume()
    self.assertEqual(
      [args.args for args in broadcast.call_args_list], [([("park_on_host", ())],), ([("resume_on_gpu", ())],)]
    )
    park.assert_called_once_with()

  def test_a_multi_rank_trainer_exits_on_server_errors_but_replies_to_user_errors(self):
    trainer = self.trainer()
    trainer.world_size = 2
    with patch("torch.distributed.broadcast_object_list"), patch("os._exit", side_effect=SystemExit(1)) as exit_:
      with self.assertRaises(KeyError):
        trainer.call_model("optim_step", new_id(), {})
      exit_.assert_not_called()
      with patch.object(trainer.model, "optim_step", side_effect=RuntimeError("CUDA out of memory")):
        with self.assertRaises(SystemExit):
          trainer.call_model("optim_step", new_id(), {})
      exit_.assert_called_once_with(1)

  def test_followers_continue_after_user_errors(self):
    model = Mock()
    model.optim_step.side_effect = KeyError("unknown run")
    commands = iter([("optim_step", ("missing", {})), ("park_on_host", ())])

    def receive(message, src):
      self.assertEqual(src, 0)
      message[0] = next(commands)

    with patch("torch.distributed.broadcast_object_list", side_effect=receive), self.assertRaises(StopIteration):
      serve_follower(model)
    model.optim_step.assert_called_once_with("missing", {})
    model.park_on_host.assert_called_once_with()
    model.stop.assert_not_called()
