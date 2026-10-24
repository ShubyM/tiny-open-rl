"""Zero-copy Protobuf <-> PyTorch tensor codec for Tiny-Open-RL.

Unlike the legacy pipeline (Protobuf -> Python list[float] -> JSON string in Redis
-> Python list[float] -> torch.Tensor, and the reverse on output), this codec:
1. Peeks at `(model_id, seq_id)` on the gateway while keeping the raw Protobuf bytes intact in Redis.
2. Uses `torch.frombuffer` on the trainer to construct `torch.Tensor`s directly from Protobuf byte slices.
3. Packs output `torch.Tensor`s directly into `pb.ForwardBackwardOutput` `ArrayRecord` buffers via `.numpy().tobytes()`.
"""

from __future__ import annotations

import array
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from google.protobuf.message import DecodeError

from tiny_open_rl.proto import tinker_public_pb2 as pb

PROTO_CONTENT_TYPE = "application/x-protobuf"

_DTYPE_TO_TORCH: dict[int, torch.dtype] = {
  pb.DTYPE_FLOAT32: torch.float32,
  pb.DTYPE_INT64: torch.int64,
  pb.DTYPE_INT32: torch.int32,
}


@dataclass
class TensorDatum:
  """One training sequence with its input tokens and loss-function tensors already in PyTorch."""

  model_input: torch.Tensor  # 1-D int64 token IDs
  loss_fn_inputs: dict[str, torch.Tensor]


@dataclass
class ForwardBackwardBatch:
  """Decoded `ForwardBackwardRequest` ready for GPU execution."""

  model_id: str
  seq_id: int
  forward_only: bool
  loss_fn: str
  loss_config: dict[str, Any]
  data: list[TensorDatum]


def peek_forward_backward_header(raw_proto: bytes) -> tuple[str, int]:
  """Extract `(model_id, seq_id)` on the gateway without unpacking tensor buffers into Python lists."""
  msg = pb.ForwardBackwardRequest()
  try:
    msg.ParseFromString(raw_proto)
  except DecodeError as exc:
    raise ValueError(f"Malformed ForwardBackwardRequest protobuf: {exc}") from exc
  return msg.model_id, int(msg.seq_id)


def _tensor_from_proto(name: str, tensor_pb: pb.Tensor) -> torch.Tensor:
  if tensor_pb.WhichOneof("encoding") == "sparse_csr":
    raise ValueError(f"loss_fn_inputs[{name!r}] sparse_csr is not supported")
  torch_dtype = _DTYPE_TO_TORCH.get(tensor_pb.dtype)
  if torch_dtype is None:
    raise ValueError(f"Unsupported tensor dtype {tensor_pb.dtype} for {name!r}")
  if not tensor_pb.dense:
    return torch.empty(0, dtype=torch.int64 if torch_dtype == torch.int32 else torch_dtype)
  t = torch.frombuffer(bytearray(tensor_pb.dense), dtype=torch_dtype)
  if torch_dtype == torch.int32:
    t = t.to(torch.int64)
  if tensor_pb.shape:
    t = t.reshape(tuple(int(d) for d in tensor_pb.shape))
  return t


def decode_forward_backward_proto(raw_proto: bytes) -> ForwardBackwardBatch:
  """Zero-copy decode of `pb.ForwardBackwardRequest` bytes into PyTorch `TensorDatum`s."""
  msg = pb.ForwardBackwardRequest()
  try:
    msg.ParseFromString(raw_proto)
  except DecodeError as exc:
    raise ValueError(f"Malformed ForwardBackwardRequest protobuf: {exc}") from exc

  loss_cfg: dict[str, Any] = {}
  if msg.loss_fn_config_v2:
    for k, v in msg.loss_fn_config_v2.items():
      loss_cfg[k] = v.text if v.WhichOneof("value") == "text" else v.number
  else:
    loss_cfg = dict(msg.loss_fn_config)

  datums: list[TensorDatum] = []
  for datum_pb in msg.data:
    chunks: list[torch.Tensor] = []
    for chunk_pb in datum_pb.model_input:
      if chunk_pb.WhichOneof("chunk") != "encoded_text":
        raise ValueError("Only encoded_text chunks are supported")
      raw_tokens = chunk_pb.encoded_text.tokens
      if raw_tokens:
        chunks.append(torch.frombuffer(bytearray(raw_tokens), dtype=torch.int32).to(torch.int64))
    model_input = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.int64)
    loss_inputs = {name: _tensor_from_proto(name, t_pb) for name, t_pb in datum_pb.loss_fn_inputs.items()}
    datums.append(TensorDatum(model_input=model_input, loss_fn_inputs=loss_inputs))

  return ForwardBackwardBatch(
    model_id=msg.model_id,
    seq_id=int(msg.seq_id),
    forward_only=bool(msg.forward_only),
    loss_fn=msg.loss_fn or "cross_entropy",
    loss_config=loss_cfg,
    data=datums,
  )


def encode_forward_backward_proto(
  per_datum_outputs: list[dict[str, torch.Tensor]],
  metrics: dict[str, float],
) -> bytes:
  """Pack per-datum output `torch.Tensor`s directly into `pb.ForwardBackwardOutput` bytes."""
  msg = pb.ForwardBackwardOutput()
  msg.loss_fn_output_type = "ArrayRecord"
  for key, val in metrics.items():
    msg.metrics[key] = float(val)

  if not per_datum_outputs:
    return msg.SerializeToString()

  record = msg.loss_fn_outputs.add()
  record.num_datums = len(per_datum_outputs)
  field_names = list(per_datum_outputs[0].keys())

  for name in field_names:
    tensors = [out[name].detach().cpu().contiguous() for out in per_datum_outputs]
    is_int = tensors[0].dtype in (torch.int32, torch.int64)
    target_dtype = torch.int64 if is_int else torch.float32
    casted = [t.to(target_dtype).view(-1) for t in tensors]

    flat_bytes = torch.cat(casted).numpy().tobytes() if casted else b""
    elem_bytes = 8 if is_int else 4
    offsets = np.zeros(len(casted) + 1, dtype=np.int64)
    np.cumsum([t.numel() * elem_bytes for t in casted], out=offsets[1:])

    field = record.fields[name]
    field.data = flat_bytes
    field.offsets = offsets.tobytes()
    field.dtype = pb.DTYPE_INT64 if is_int else pb.DTYPE_FLOAT32
    if tensors[0].ndim > 1:
      field.trailing_shape.extend(int(s) for s in tensors[0].shape[1:])

  return msg.SerializeToString()


def decode_forward_backward_output_proto(raw_proto: bytes) -> dict[str, Any]:
  """Unpack `pb.ForwardBackwardOutput` bytes into a Python dict (for JSON clients or test assertions)."""
  msg = pb.ForwardBackwardOutput()
  msg.ParseFromString(raw_proto)
  outputs: list[dict[str, list[float | int]]] = []
  if msg.loss_fn_outputs:
    rec = msg.loss_fn_outputs[0]
    outputs = [{} for _ in range(rec.num_datums)]
    for name, field in rec.fields.items():
      offsets = np.frombuffer(field.offsets, dtype=np.int64)
      np_dtype = np.int64 if field.dtype == pb.DTYPE_INT64 else np.float32
      for i in range(rec.num_datums):
        slice_bytes = field.data[offsets[i] : offsets[i + 1]]
        outputs[i][name] = np.frombuffer(slice_bytes, dtype=np_dtype).tolist()
  return {
    "type": "forward_backward_completed",
    "loss_fn_output_type": msg.loss_fn_output_type,
    "metrics": dict(msg.metrics),
    "loss_fn_outputs": outputs,
  }


def build_forward_backward_proto(
  model_id: str,
  sequences: list[tuple[list[int], list[int], list[float]]],
  *,
  loss_fn: str = "cross_entropy",
  seq_id: int = 1,
  forward_only: bool = False,
  extra_inputs: list[dict[str, list[float]]] | None = None,
) -> bytes:
  """Helper to construct a binary `pb.ForwardBackwardRequest` from `(input_tokens, target_tokens, weights)` tuples."""
  msg = pb.ForwardBackwardRequest(model_id=model_id, seq_id=seq_id, loss_fn=loss_fn, forward_only=forward_only)
  for idx, (inp, tgt, wts) in enumerate(sequences):
    datum = msg.data.add()
    chunk = datum.model_input.add()
    chunk.encoded_text.tokens = array.array("i", inp).tobytes()

    t_tgt = datum.loss_fn_inputs["target_tokens"]
    t_tgt.dtype = pb.DTYPE_INT64
    t_tgt.shape.append(len(tgt))
    t_tgt.dense = array.array("q", tgt).tobytes()

    t_wts = datum.loss_fn_inputs["weights"]
    t_wts.dtype = pb.DTYPE_FLOAT32
    t_wts.shape.append(len(wts))
    t_wts.dense = array.array("f", wts).tobytes()

    if extra_inputs and idx < len(extra_inputs):
      for name, vals in extra_inputs[idx].items():
        t_extra = datum.loss_fn_inputs[name]
        t_extra.dtype = pb.DTYPE_FLOAT32
        t_extra.shape.append(len(vals))
        t_extra.dense = array.array("f", vals).tobytes()

  return msg.SerializeToString()
