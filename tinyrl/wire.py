# the tinker wire format. forward_backward is protobuf end to end: the gateway never unpacks it, and the trainer reads its
# tensors straight out of the bytes. every other call is json.
import sys
from dataclasses import dataclass
import numpy as np
from google.protobuf.message import DecodeError
from tinyrl.proto import tinker_public_pb2 as pb
assert sys.byteorder == "little", "the wire is little endian"

PROTO, JSON = "application/x-protobuf", "application/json"
NP = {pb.DTYPE_FLOAT32: np.float32, pb.DTYPE_INT64: np.int64, pb.DTYPE_INT32: np.int32}

@dataclass(frozen=True)
class Datum: tokens:np.ndarray; inputs:dict[str, np.ndarray] # noqa: E702

def parse(raw:bytes) -> pb.ForwardBackwardRequest:
  try: return pb.ForwardBackwardRequest.FromString(raw)
  except DecodeError as e: raise ValueError(f"bad ForwardBackwardRequest: {e}") from e

def array(t:pb.Tensor|pb.BatchedTensor, raw:bytes) -> np.ndarray:
  if t.dtype not in NP: raise ValueError(f"tensors of {pb.DType.Name(t.dtype)} are not supported")
  return np.frombuffer(raw, NP[t.dtype])

def tensor(t:pb.Tensor) -> np.ndarray:
  if t.WhichOneof("encoding") == "sparse_csr": raise ValueError("sparse tensors are not supported")
  return array(t, t.dense)

def tokens(c:pb.Chunk) -> np.ndarray:
  if (kind:=c.WhichOneof("chunk")) != "encoded_text": raise ValueError(f"{kind} chunks are not supported, this trainer is text only")
  return np.frombuffer(c.encoded_text.tokens, np.int32)

def datums(req:pb.ForwardBackwardRequest) -> list[Datum]:
  return [Datum(np.concatenate([np.zeros(0, np.int32), *map(tokens, d.model_input)]), {k:tensor(v) for k,v in d.loss_fn_inputs.items()})
          for d in req.data]

def config(req:pb.ForwardBackwardRequest) -> dict[str, float|str]:
  # v2 carries strings too, the sdk dual writes its numbers into the legacy map for older servers
  return {k:getattr(v, v.WhichOneof("value") or "number") for k,v in req.loss_fn_config_v2.items()} or dict(req.loss_fn_config)

def from_json(b:dict, forward_only:bool=False) -> bytes:
  # sdks before 0.25 send forward_backward as json. it becomes the same protobuf, so the trainer only ever reads one format
  fb, fwd = b.get("forward_backward_input") or b.get("forward_input") or {}, forward_only or bool(b.get("forward_only"))
  req = pb.ForwardBackwardRequest(model_id=b["model_id"], loss_fn=fb.get("loss_fn") or "cross_entropy", forward_only=fwd)
  for k,v in (fb.get("loss_fn_config") or {}).items():
    req.loss_fn_config_v2[k].CopyFrom(pb.LossConfigValue(text=v) if isinstance(v, str) else pb.LossConfigValue(number=v))
  for d in fb.get("data", []):
    pd = req.data.add()
    pd.model_input.add().encoded_text.tokens = np.array([t for c in d["model_input"]["chunks"] for t in c["tokens"]], np.int32).tobytes()
    for k,v in d.get("loss_fn_inputs", {}).items():
      v = v if isinstance(v, dict) else {"data": v}
      a = np.array(v["data"], np.int64 if v.get("dtype", "int64" if k == "target_tokens" else "float32") == "int64" else np.float32)
      pd.loss_fn_inputs[k].CopyFrom(pb.Tensor(dense=a.tobytes(), dtype=pb.DTYPE_INT64 if a.dtype == np.int64 else pb.DTYPE_FLOAT32, shape=a.shape))
  return req.SerializeToString()

def output(logprobs:list[np.ndarray], metrics:dict[str, float]) -> bytes:
  # one ArrayRecord: every datum's values back to back, sliced by byte offsets
  out = pb.ForwardBackwardOutput(loss_fn_output_type="ArrayRecord", metrics=metrics)
  if logprobs:
    f = out.loss_fn_outputs.add(num_datums=len(logprobs)).fields["logprobs"]
    f.data, f.dtype = np.concatenate(logprobs).astype(np.float32).tobytes(), pb.DTYPE_FLOAT32
    f.offsets = np.cumsum([0]+[4*len(x) for x in logprobs], dtype=np.int64).tobytes()
  return out.SerializeToString()

def unbatch(t:pb.BatchedTensor) -> list[np.ndarray]:
  o = np.frombuffer(t.offsets, np.int64)
  return [array(t, t.data[a:b]) for a,b in zip(o[:-1], o[1:])]

def tensor_json(x:np.ndarray) -> dict: return {"data": x.tolist(), "dtype": "int64" if x.dtype.kind == "i" else "float32", "shape": list(x.shape)}

def output_json(raw:bytes) -> dict:
  # for clients that did not ask for protobuf
  out, rows = pb.ForwardBackwardOutput.FromString(raw), []
  for r in out.loss_fn_outputs:
    cols = {k:unbatch(v) for k,v in r.fields.items()}
    rows += [{k:tensor_json(v[i]) for k,v in cols.items()} for i in range(r.num_datums)]
  return {"type": "forward_backward", "loss_fn_output_type": out.loss_fn_output_type, "metrics": dict(out.metrics), "loss_fn_outputs": rows}
