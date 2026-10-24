import unittest
import numpy as np
from tinyrl import wire
from tinyrl.proto import tinker_public_pb2 as pb

def body(n=3, loss_fn="cross_entropy", **cfg):  # forward_backward as sdks before 0.25 send it
  return {"model_id": "m", "forward_backward_input": {"loss_fn": loss_fn, "loss_fn_config": cfg, "data": [
    {"model_input": {"chunks": [{"type": "encoded_text", "tokens": list(range(i, i+n))}]},
     "loss_fn_inputs": {"target_tokens": {"data": list(range(i+1, i+n+1)), "dtype": "int64"}, "weights": {"data": [0.5]*n, "dtype": "float32"}}}
    for i in range(2)]}}

class TestWire(unittest.TestCase):
  def test_json_becomes_the_same_protobuf(self):
    req = wire.parse(wire.from_json(body(clip_range=0.3, mode="x")))
    ds = wire.datums(req)
    self.assertEqual((req.model_id, req.loss_fn, req.forward_only, wire.config(req)), ("m", "cross_entropy", False, {"clip_range": 0.3, "mode": "x"}))
    np.testing.assert_equal(ds[1].tokens, [1, 2, 3])
    np.testing.assert_equal(ds[1].inputs["target_tokens"], [2, 3, 4])
    np.testing.assert_equal(ds[1].inputs["weights"], [0.5]*3)
    self.assertEqual([x.dtype for x in (ds[0].tokens, ds[0].inputs["target_tokens"], ds[0].inputs["weights"])], [np.int32, np.int64, np.float32])

  def test_bare_lists_forward_input_and_forward_only(self):
    datum = {"model_input": {"chunks": [{"tokens": [5, 6]}]}, "loss_fn_inputs": {"target_tokens": [6, 7], "advantages": [1, -1]}}
    req = wire.parse(wire.from_json({"model_id": "m", "forward_input": {"data": [datum]}}, forward_only=True))
    d = wire.datums(req)[0]
    self.assertTrue(req.forward_only)
    self.assertEqual((d.inputs["target_tokens"].dtype, d.inputs["advantages"].dtype), (np.int64, np.float32))

  def test_what_the_sdk_sends(self):
    # int32 tensors, tokens across chunks, and only the legacy config map
    req = pb.ForwardBackwardRequest(model_id="m", loss_fn="ppo", loss_fn_config={"clip_range": 0.1})
    d = req.data.add()
    d.model_input.add().encoded_text.tokens = np.array([7, 8], np.int32).tobytes()
    d.model_input.add().encoded_text.tokens = np.array([9], np.int32).tobytes()
    d.loss_fn_inputs["target_tokens"].CopyFrom(pb.Tensor(dense=np.array([8, 9, 10], np.int32).tobytes(), dtype=pb.DTYPE_INT32, shape=[3]))
    req = wire.parse(req.SerializeToString())
    np.testing.assert_equal(wire.datums(req)[0].tokens, [7, 8, 9])
    np.testing.assert_equal(wire.datums(req)[0].inputs["target_tokens"], [8, 9, 10])
    self.assertAlmostEqual(wire.config(req)["clip_range"], 0.1, places=6)

  def test_output_round_trip(self):
    lps = [np.array([-1.5, -0.25], np.float32), np.zeros(0, np.float32), np.array([-3.0], np.float32)]
    j = wire.output_json(wire.output(lps, {"loss:sum": 4.75}))
    self.assertEqual((j["type"], j["loss_fn_output_type"], j["metrics"]), ("forward_backward", "ArrayRecord", {"loss:sum": 4.75}))
    self.assertEqual([o["logprobs"] for o in j["loss_fn_outputs"]], [{"data": [-1.5, -0.25], "dtype": "float32", "shape": [2]},
                                                                    {"data": [], "dtype": "float32", "shape": [0]},
                                                                    {"data": [-3.0], "dtype": "float32", "shape": [1]}])
    self.assertEqual(wire.output_json(wire.output([], {}))["loss_fn_outputs"], [])

  def test_rejects_what_it_cannot_train(self):
    def req(**kw):
      r = pb.ForwardBackwardRequest(model_id="m")
      d = r.data.add()
      d.model_input.add().encoded_text.tokens = b"\0\0\0\0"
      d.loss_fn_inputs["x"].CopyFrom(pb.Tensor(**kw))
      return wire.parse(r.SerializeToString())
    for bad in [req(dtype=pb.DTYPE_BFLOAT16, dense=b"\0\0"), req(sparse_csr=pb.SparseCsr(), dtype=pb.DTYPE_FLOAT32)]:
      with self.assertRaises(ValueError): wire.datums(bad)
    image = pb.ForwardBackwardRequest(model_id="m")
    image.data.add().model_input.add().image.data = b"png"
    with self.assertRaises(ValueError): wire.datums(image)
    with self.assertRaises(ValueError): wire.parse(b"\xff\xff\xff")

if __name__ == "__main__":
  unittest.main()
