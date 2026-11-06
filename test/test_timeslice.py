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
          # what the allocator holds for them: each tensor's bytes, rounded up to its 512 byte blocks
          on_gpu = sum(-(-t.numel() * t.element_size() // 512) * 512 for t in model.gpu_tensors() if t.is_cuda)
          before = torch.cuda.memory_allocated()
          model.park_on_host()
          self.assertFalse(any(t.is_cuda for t in model.gpu_tensors()))
          self.assertEqual(before - torch.cuda.memory_allocated(), on_gpu)  # all of it freed, nothing else
          model.resume_on_gpu()
          self.assertTrue(all(t.is_cuda or not t.is_pinned() for t in model.gpu_tensors()))
        model.optim_step("x", {"learning_rate": 1e-2})
    self.assertEqual(losses[id(parked)], losses[id(resident)])


if __name__ == "__main__":
  unittest.main()
