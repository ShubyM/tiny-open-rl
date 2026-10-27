"""The Tinker wire format.

forward_backward and SampleResponse travel as protobuf end to end: the gateway forwards the bytes untouched, and the
workers read or write tensors straight to them. Every other call is JSON.
"""

import sys
from dataclasses import dataclass

import numpy as np
from google.protobuf.message import DecodeError

from tinyrl.proto import tinker_public_pb2 as pb

assert sys.byteorder == "little", "the wire format is little endian"

PROTO = "application/x-protobuf"
JSON = "application/json"

NUMPY_DTYPES = {
  pb.DTYPE_FLOAT32: np.float32,
  pb.DTYPE_INT64: np.int64,
  pb.DTYPE_INT32: np.int32,
}


@dataclass(frozen=True)
class Datum:
  tokens: np.ndarray
  inputs: dict[str, np.ndarray]


# ***** requests *****


def parse(raw: bytes) -> pb.ForwardBackwardRequest:
  try:
    return pb.ForwardBackwardRequest.FromString(raw)
  except DecodeError as e:
    raise ValueError(f"bad ForwardBackwardRequest: {e}") from e


def datums(req: pb.ForwardBackwardRequest) -> list[Datum]:
  out = []
  for d in req.data:
    tokens = [chunk_tokens(chunk) for chunk in d.model_input]
    inputs = {name: loss_input(tensor) for name, tensor in d.loss_fn_inputs.items()}
    out.append(Datum(np.concatenate(tokens) if tokens else np.zeros(0, np.int32), inputs))
  return out


def chunk_tokens(chunk: pb.Chunk) -> np.ndarray:
  kind = chunk.WhichOneof("chunk")
  if kind != "encoded_text":
    raise ValueError(f"{kind} chunks are not supported, this trainer is text only")
  return np.frombuffer(chunk.encoded_text.tokens, np.int32)


def loss_input(tensor: pb.Tensor) -> np.ndarray:
  if tensor.WhichOneof("encoding") == "sparse_csr":
    raise ValueError("sparse tensors are not supported")
  return to_numpy(tensor.dtype, tensor.dense)


def to_numpy(dtype: int, raw: bytes) -> np.ndarray:
  if dtype not in NUMPY_DTYPES:
    raise ValueError(f"tensors of {pb.DType.Name(dtype)} are not supported")
  return np.frombuffer(raw, NUMPY_DTYPES[dtype])


def loss_config(req: pb.ForwardBackwardRequest) -> dict[str, float | str]:
  # v2 holds strings as well as numbers. The SDK also writes its numbers into the legacy map, for older servers.
  if req.loss_fn_config_v2:
    return {k: v.text if v.WhichOneof("value") == "text" else v.number for k, v in req.loss_fn_config_v2.items()}
  return dict(req.loss_fn_config)


def from_json(body: dict, forward_only: bool = False) -> bytes:
  """Turn the JSON forward_backward that SDKs before 0.25 send into the same protobuf, so the trainer reads one
  format."""
  fb = body.get("forward_backward_input") or body.get("forward_input") or {}
  req = pb.ForwardBackwardRequest(
    model_id=body["model_id"],
    loss_fn=fb.get("loss_fn") or "cross_entropy",
    forward_only=forward_only or bool(body.get("forward_only")),
  )

  for k, v in (fb.get("loss_fn_config") or {}).items():
    value = pb.LossConfigValue(text=v) if isinstance(v, str) else pb.LossConfigValue(number=v)
    req.loss_fn_config_v2[k].CopyFrom(value)

  for d in fb.get("data", []):
    datum = req.data.add()
    tokens = [token for chunk in d["model_input"]["chunks"] for token in chunk["tokens"]]
    datum.model_input.add().encoded_text.tokens = np.array(tokens, np.int32).tobytes()

    for name, value in d.get("loss_fn_inputs", {}).items():
      if not isinstance(value, dict):  # older clients send bare lists
        value = {"data": value}
      is_int = value.get("dtype", "int64" if name == "target_tokens" else "float32") == "int64"
      array = np.array(value["data"], np.int64 if is_int else np.float32)
      tensor = pb.Tensor(dense=array.tobytes(), dtype=pb.DTYPE_INT64 if is_int else pb.DTYPE_FLOAT32, shape=array.shape)
      datum.loss_fn_inputs[name].CopyFrom(tensor)

  return req.SerializeToString()


# ***** outputs *****


def output(logprobs: list[np.ndarray], metrics: dict[str, float]) -> bytes:
  """Pack each datum's logprobs into a ForwardBackwardOutput: one ArrayRecord holding every datum's values back to back,
  sliced apart by byte offsets."""
  out = pb.ForwardBackwardOutput(loss_fn_output_type="ArrayRecord", metrics=metrics)
  if logprobs:
    field = out.loss_fn_outputs.add(num_datums=len(logprobs)).fields["logprobs"]
    field.dtype = pb.DTYPE_FLOAT32
    field.data = np.concatenate(logprobs).astype(np.float32).tobytes()
    field.offsets = np.cumsum([0] + [4 * len(x) for x in logprobs], dtype=np.int64).tobytes()
  return out.SerializeToString()


def output_json(raw: bytes) -> dict:
  """Decode a ForwardBackwardOutput, for clients that did not ask for protobuf."""
  out = pb.ForwardBackwardOutput.FromString(raw)

  rows = []
  for record in out.loss_fn_outputs:
    columns = {name: unbatch(tensor) for name, tensor in record.fields.items()}
    for i in range(record.num_datums):
      rows.append({name: tensor_json(values[i]) for name, values in columns.items()})

  return {
    "type": "forward_backward",
    "loss_fn_output_type": out.loss_fn_output_type,
    "metrics": dict(out.metrics),
    "loss_fn_outputs": rows,
  }


def unbatch(tensor: pb.BatchedTensor) -> list[np.ndarray]:
  offsets = np.frombuffer(tensor.offsets, np.int64)
  return [to_numpy(tensor.dtype, tensor.data[start:end]) for start, end in zip(offsets[:-1], offsets[1:])]


def tensor_json(x: np.ndarray) -> dict:
  return {"data": x.tolist(), "dtype": "int64" if x.dtype.kind == "i" else "float32", "shape": list(x.shape)}


@dataclass(frozen=True)
class SampleOut:
  stop: bool
  tokens: np.ndarray
  logprobs: np.ndarray
  topk: tuple[np.ndarray, np.ndarray] | None = None


def sample_output(
  sequences: list[SampleOut],
  prompt_logprobs: np.ndarray | None = None,
  topk_prompt: tuple[np.ndarray, np.ndarray] | None = None,
) -> bytes:
  out = pb.SampleResponse()
  for s in sequences:
    seq = out.sequences.add()
    seq.stop_reason = pb.STOP_REASON_STOP if s.stop else pb.STOP_REASON_LENGTH
    seq.tokens = s.tokens.astype(np.int32).tobytes()
    seq.logprobs = s.logprobs.astype(np.float32).tobytes()
    if s.topk:
      tk_ids, tk_lps = s.topk
      seq.topk_sampled_logprobs.token_ids = tk_ids.astype(np.int32).tobytes()
      seq.topk_sampled_logprobs.logprobs = tk_lps.astype(np.float32).tobytes()
      seq.topk_sampled_logprobs.k = tk_ids.shape[-1]
      seq.topk_sampled_logprobs.length = tk_ids.shape[0]

  if prompt_logprobs is not None:
    out.prompt_logprobs = prompt_logprobs.astype(np.float32).tobytes()
  if topk_prompt is not None:
    tk_ids, tk_lps = topk_prompt
    out.topk_prompt_logprobs.token_ids = tk_ids.astype(np.int32).tobytes()
    out.topk_prompt_logprobs.logprobs = tk_lps.astype(np.float32).tobytes()
    out.topk_prompt_logprobs.k = tk_ids.shape[-1]
    out.topk_prompt_logprobs.length = tk_ids.shape[0]

  return out.SerializeToString()


def _format_topk(tk_pb):
  k, length = tk_pb.k, tk_pb.length
  ids = np.frombuffer(tk_pb.token_ids, np.int32).reshape(length, k)
  lps = np.frombuffer(tk_pb.logprobs, np.float32).reshape(length, k)
  res = []
  for i in range(length):
    if ids[i, 0] == 0 and lps[i, 0] == -99999.0:
      res.append(None)
    else:
      res.append([[int(tid), float(lp)] for tid, lp in zip(ids[i], lps[i]) if not (tid == 0 and lp == -99999.0)])
  return res


def sample_json(raw: bytes) -> dict:
  out = pb.SampleResponse.FromString(raw)
  sequences = []
  for seq in out.sequences:
    item = {
      "stop_reason": "stop" if seq.stop_reason == pb.STOP_REASON_STOP else "length",
      "tokens": np.frombuffer(seq.tokens, np.int32).tolist(),
      "logprobs": np.frombuffer(seq.logprobs, np.float32).tolist(),
    }
    if seq.HasField("topk_sampled_logprobs"):
      item["topk_logprobs"] = _format_topk(seq.topk_sampled_logprobs)
    sequences.append(item)

  res = {"type": "sample", "sequences": sequences}
  if out.prompt_logprobs:
    res["prompt_logprobs"] = [None if np.isnan(x) else float(x) for x in np.frombuffer(out.prompt_logprobs, np.float32)]
  if out.HasField("topk_prompt_logprobs"):
    res["topk_prompt_logprobs"] = _format_topk(out.topk_prompt_logprobs)
  return res
