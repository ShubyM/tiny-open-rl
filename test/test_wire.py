import unittest

import numpy as np

from tinyrl.api import wire
from tinyrl.api.proto import tinker_public_pb2 as pb


class TestRequests(unittest.TestCase):
  def test_what_the_sdk_sends(self):
    # int32 tensors, tokens split across chunks, and only the legacy config map
    req = pb.ForwardBackwardRequest(model_id="m", loss_fn="ppo", loss_fn_config={"clip_range": 0.1})
    datum = req.data.add()
    datum.model_input.add().encoded_text.tokens = np.array([7, 8], np.int32).tobytes()
    datum.model_input.add().encoded_text.tokens = np.array([9], np.int32).tobytes()
    targets = np.array([8, 9, 10], np.int32).tobytes()
    datum.loss_fn_inputs["target_tokens"].CopyFrom(pb.Tensor(dense=targets, dtype=pb.DTYPE_INT32, shape=[3]))

    req = wire.parse(req.SerializeToString())
    np.testing.assert_equal(wire.datums(req)[0].tokens, [7, 8, 9])
    np.testing.assert_equal(wire.datums(req)[0].inputs["target_tokens"], [8, 9, 10])
    self.assertAlmostEqual(wire.loss_config(req)["clip_range"], 0.1, places=6)

  def test_v2_loss_config_carries_strings_and_numbers(self):
    req = pb.ForwardBackwardRequest(model_id="m", loss_fn="stable_reinforce")
    req.loss_fn_config_v2["clip_range"].CopyFrom(pb.LossConfigValue(number=0.3))
    req.loss_fn_config_v2["mode"].CopyFrom(pb.LossConfigValue(text="x"))
    self.assertEqual(wire.loss_config(wire.parse(req.SerializeToString())), {"clip_range": 0.3, "mode": "x"})

  def test_rejects_what_it_cannot_train(self):
    def with_tensor(**tensor) -> pb.ForwardBackwardRequest:
      req = pb.ForwardBackwardRequest(model_id="m")
      datum = req.data.add()
      datum.model_input.add().encoded_text.tokens = b"\0\0\0\0"
      datum.loss_fn_inputs["x"].CopyFrom(pb.Tensor(**tensor))
      return wire.parse(req.SerializeToString())

    with self.assertRaises(ValueError):
      wire.datums(with_tensor(dtype=pb.DTYPE_BFLOAT16, dense=b"\0\0"))
    with self.assertRaises(ValueError):
      wire.datums(with_tensor(sparse_csr=pb.SparseCsr(), dtype=pb.DTYPE_FLOAT32))

    image = pb.ForwardBackwardRequest(model_id="m")
    image.data.add().model_input.add().image.data = b"png"
    with self.assertRaises(ValueError):
      wire.datums(image)

    with self.assertRaises(ValueError):
      wire.parse(b"\xff\xff\xff")


class TestOutputs(unittest.TestCase):
  def test_round_trip(self):
    logprobs = [np.array([-1.5, -0.25], np.float32), np.zeros(0, np.float32), np.array([-3.0], np.float32)]
    out = pb.ForwardBackwardOutput.FromString(wire.output(logprobs, {"loss:sum": 4.75}))
    self.assertEqual(out.loss_fn_output_type, "ArrayRecord")
    self.assertEqual(dict(out.metrics), {"loss:sum": 4.75})
    field = out.loss_fn_outputs[0].fields["logprobs"]
    offsets = np.frombuffer(field.offsets, np.int64)
    got = [np.frombuffer(field.data[a:b], np.float32).tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    self.assertEqual(got, [[-1.5, -0.25], [], [-3.0]])

  def test_no_datums(self):
    out = pb.ForwardBackwardOutput.FromString(wire.output([], {}))
    self.assertEqual(len(out.loss_fn_outputs), 0)


class TestG2Wire(unittest.TestCase):
  def test_2d_reject(self):
    array_2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    tensor = pb.Tensor(dense=array_2d.tobytes(), dtype=pb.DTYPE_INT64, shape=array_2d.shape)
    with self.assertRaisesRegex(ValueError, "shape"):
      wire.loss_input(tensor)


if __name__ == "__main__":
  unittest.main()
