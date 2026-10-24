import unittest

import numpy as np

from tinyrl import wire
from tinyrl.proto import tinker_public_pb2 as pb


def json_body(n: int = 3, loss_fn: str = "cross_entropy", **config) -> dict:
  """A forward_backward with two datums, the way SDKs before 0.25 send it."""
  data = []
  for start in range(2):
    data.append(
      {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": list(range(start, start + n))}]},
        "loss_fn_inputs": {
          "target_tokens": {"data": list(range(start + 1, start + n + 1)), "dtype": "int64"},
          "weights": {"data": [0.5] * n, "dtype": "float32"},
        },
      }
    )
  return {"model_id": "m", "forward_backward_input": {"loss_fn": loss_fn, "loss_fn_config": config, "data": data}}


class TestRequests(unittest.TestCase):
  def test_json_becomes_the_same_protobuf(self):
    req = wire.parse(wire.from_json(json_body(clip_range=0.3, mode="x")))
    self.assertEqual(req.model_id, "m")
    self.assertEqual(req.loss_fn, "cross_entropy")
    self.assertFalse(req.forward_only)
    self.assertEqual(wire.loss_config(req), {"clip_range": 0.3, "mode": "x"})

    datum = wire.datums(req)[1]
    np.testing.assert_equal(datum.tokens, [1, 2, 3])
    np.testing.assert_equal(datum.inputs["target_tokens"], [2, 3, 4])
    np.testing.assert_equal(datum.inputs["weights"], [0.5, 0.5, 0.5])
    self.assertEqual(datum.tokens.dtype, np.int32)
    self.assertEqual(datum.inputs["target_tokens"].dtype, np.int64)
    self.assertEqual(datum.inputs["weights"].dtype, np.float32)

  def test_bare_lists_forward_input_and_forward_only(self):
    datum = {
      "model_input": {"chunks": [{"tokens": [5, 6]}]},
      "loss_fn_inputs": {"target_tokens": [6, 7], "advantages": [1, -1]},
    }
    req = wire.parse(wire.from_json({"model_id": "m", "forward_input": {"data": [datum]}}, forward_only=True))
    self.assertTrue(req.forward_only)

    inputs = wire.datums(req)[0].inputs
    self.assertEqual(inputs["target_tokens"].dtype, np.int64)
    self.assertEqual(inputs["advantages"].dtype, np.float32)

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
    out = wire.output_json(wire.output(logprobs, {"loss:sum": 4.75}))

    self.assertEqual(out["type"], "forward_backward")
    self.assertEqual(out["loss_fn_output_type"], "ArrayRecord")
    self.assertEqual(out["metrics"], {"loss:sum": 4.75})
    self.assertEqual(
      [row["logprobs"] for row in out["loss_fn_outputs"]],
      [
        {"data": [-1.5, -0.25], "dtype": "float32", "shape": [2]},
        {"data": [], "dtype": "float32", "shape": [0]},
        {"data": [-3.0], "dtype": "float32", "shape": [1]},
      ],
    )

  def test_no_datums(self):
    self.assertEqual(wire.output_json(wire.output([], {}))["loss_fn_outputs"], [])


if __name__ == "__main__":
  unittest.main()
