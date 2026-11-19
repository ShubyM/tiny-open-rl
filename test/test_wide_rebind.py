import asyncio
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from test.execution import TestNode
from test.helpers import RedisTest, new_id
from test.test_api import proto_body
from tinyrl.bus import Op, Run
from tinyrl.control.fairness import Demand, Worker, plan_gpus
from tinyrl.storage.store import Store


class TestWideSelection(unittest.TestCase):
  def test_exclusive_prefers_to_leave_a_parked_gangs_block_available(self):
    workers = {"gang": Worker(gpus=2, bound=(0, 1), gang=True), "other": Worker(exclusive=True)}
    self.assertEqual(plan_gpus(workers, 4, 1)[0], {"other": (2,), "gang": (0, 1)})
    self.assertEqual(plan_gpus(workers, 2, 1)[0], {"other": (0,)})


class TestWideRebind(RedisTest):
  async def test_trainer_waits_on_its_original_block_when_no_other_block_fits(self):
    with tempfile.TemporaryDirectory() as folder:
      node = TestNode(self.bus, gpus=2, store=Store(root=folder))
      try:
        run, exclusive = new_id(), new_id()
        for name, width in ((run, 2), (exclusive, 1)):
          await self.bus.create_run(Run(name, "tiny", 0, gpus=width, exclusive=name == exclusive))
          await node.bind(name, name)
          request = await self.bus.submit_command(name, Op.CREATE_MODEL, b'{"base_model":"tiny","rank":0,"lora":{}}')
          self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
        gang = node.lives[run]
        self.assertNotIn(gang.life, node.on)
        request = await self.bus.submit_command(run, Op.OPTIM_STEP, b'{"adam_params":{}}')
        async with asyncio.timeout(5):
          while gang.demand == Demand.IDLE:
            await node.changed.wait()
        self.assertTrue(gang.wanted())
        self.assertEqual(node.select_gpus(run), ())
        node.drop_local(exclusive)
        self.assertTrue((await self.bus.wait_for_reply(request, 5)).ok)
        self.assertEqual(node.on, {gang.life: (0, 1)})
      finally:
        await node.aclose()

  async def test_parked_trainer_restores_dirty_state_on_the_other_block(self):
    with tempfile.TemporaryDirectory() as folder:
      node = TestNode(self.bus, gpus=4, store=Store(root=folder))
      try:
        run, neighbor, exclusive = new_id(), new_id(), new_id()
        for name, width in ((run, 2), (neighbor, 2), (exclusive, 1)):
          await self.bus.create_run(Run(name, "tiny", 0, gpus=width, exclusive=name == exclusive))

        async def command(name, op, body):
          request = await self.bus.submit_command(
            name, op, body if isinstance(body, bytes) else json.dumps(body).encode()
          )
          async with asyncio.timeout(10):
            while (reply := await self.bus.wait_for_reply(request, 0)) is None:
              await node.tick()
              await asyncio.sleep(0.005)
          self.assertTrue(reply.ok, reply.body)

        create = {"base_model": "tiny", "rank": 0, "lora": {}}
        gang = await node.bind("wide", run)
        await command(run, Op.CREATE_MODEL, create)
        await command(run, Op.FORWARD_BACKWARD, proto_body(run))
        await command(run, Op.OPTIM_STEP, {"adam_params": {}})
        await command(run, Op.FORWARD_BACKWARD, proto_body(run))
        self.assertEqual(node.on[gang.life], (0, 1))
        child = gang.model
        expected, _ = child.trainer.model.training_state(run)

        sibling = await node.bind("neighbor", neighbor)
        await command(neighbor, Op.CREATE_MODEL, create)
        self.assertEqual(node.on[sibling.life], (2, 3))
        pin = await node.bind("exclusive", exclusive)
        await command(exclusive, Op.CREATE_MODEL, create)
        self.assertEqual(node.on[pin.life], (0,))
        self.assertNotIn(gang.life, node.on)

        # Exercise the CUDA checkpoint admission path with the inline CPU trainer.
        old_call, queued = child.call, []

        async def check_admission(method, *args):
          self.assertEqual(node.on.get(gang.life), (0, 1))
          self.assertNotIn(pin.life, node.on)
          if method == "save_run":
            self.assertEqual(node.drains.get(pin.life), (0,))
            queued.append(await self.bus.submit_command(exclusive, Op.OPTIM_STEP, b'{"adam_params":{}}'))
            body = json.dumps({"uri": f"tinker://{run}/weights/queued"}).encode()
            queued.append(await self.bus.submit_command(run, Op.SAVE_WEIGHTS, body))
          return await old_call(method, *args)

        with patch("tinyrl.control.node_daemon.torch") as daemon_torch, patch.object(child, "call", check_admission):
          daemon_torch.cuda.is_available.return_value = True
          await command(run, Op.SAVE_WEIGHTS, {"uri": f"tinker://{run}/weights/after"})

        self.assertEqual(len(queued), 2)
        for request in queued:
          reply = await self.bus.wait_for_reply(request, 10)
          self.assertIsNotNone(reply)
          self.assertTrue(reply.ok, reply.body)
        self.assertTrue(child.closed)
        replacement = node.lives["wide"]
        self.assertEqual(node.on, {replacement.life: (2, 3), pin.life: (0,)})
        actual, _ = replacement.model.trainer.model.training_state(run)
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
          torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
      finally:
        await node.aclose()
