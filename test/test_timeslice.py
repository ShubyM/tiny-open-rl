import unittest

import torch

from test.helpers import tiny
from test.test_api import proto_body
from tinyrl.api import wire
from tinyrl.workers.model import Model

LORA = {"rank": 4, "seed": 0, "lora_dropout": 0.0}


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestParking(unittest.TestCase):
  def test_parking_frees_the_gpu_and_training_continues_exactly(self):
    def tensors(model):
      yield from model.lm.parameters()
      yield from model.lm.buffers()
      yield from (p.grad for p in model.lm.parameters() if p.grad is not None)
      for optimizer in model.optimizers.values():
        yield from (v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))

    parked, resident = Model(lambda _: tiny().cuda()), Model(lambda _: tiny().cuda())
    data = wire.datums(wire.parse(proto_body("x")))
    losses = {id(parked): [], id(resident): []}
    for step in range(3):
      for model in (parked, resident):
        if step == 0:
          model.create_run("x", "tiny", 4, LORA)
        _, metrics = model.forward_backward("x", data, "cross_entropy", {}, False)
        losses[id(model)].append(metrics["loss:sum"])
        if model is parked:  # between forward_backward and optim_step, so gradients move too
          params = [id(p) for p in model.lm.parameters()]
          before = torch.cuda.memory_allocated()
          model.park_on_host()
          self.assertTrue(all(t.device.type == "cpu" for t in tensors(model)))
          self.assertLess(torch.cuda.memory_allocated(), before)
          model.resume_on_gpu()
          self.assertTrue(all(t.is_cuda for t in tensors(model)))
          self.assertEqual(params, [id(p) for p in model.lm.parameters()])
        model.optim_step("x", {"learning_rate": 1e-2})
    self.assertEqual(losses[id(parked)], losses[id(resident)])
    torch.testing.assert_close(parked.training_state("x")[0], resident.training_state("x")[0], atol=0, rtol=0)


if __name__ == "__main__":
  unittest.main()
