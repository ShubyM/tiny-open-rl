"""The Tinker wire format.

forward_backward and SampleResponse travel as protobuf end to end: the gateway forwards the bytes untouched, and the workers read or write tensors straight to them. Every other call is JSON.
"""

import sys
from dataclasses import dataclass

import numpy as np
from google.protobuf.message import DecodeError

from tinyrl.api.proto import tinker_public_pb2 as pb

assert sys.byteorder == "little", "the wire format is little endian"

PROTO = "application/x-protobuf"
JSON = "application/json"
MASK_LOGPROB = -99999.0  # the SDK's logprob for a top-k slot that wasn't scored

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


def cost(req: pb.ForwardBackwardRequest) -> int:
  """The tokens a request trains on."""
  return sum(len(chunk.encoded_text.tokens) // 4 for d in req.data for chunk in d.model_input)


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
  if len(tensor.shape) > 1:
    raise ValueError(f"expected 1-D tensor, got shape {list(tensor.shape)}")
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


# ***** outputs *****


def output(logprobs: list[np.ndarray], metrics: dict[str, float]) -> bytes:
  """Pack each datum's logprobs into a ForwardBackwardOutput: one ArrayRecord holding every datum's values back to back, sliced apart by byte offsets."""
  out = pb.ForwardBackwardOutput(loss_fn_output_type="ArrayRecord", metrics=metrics)
  if logprobs:
    field = out.loss_fn_outputs.add(num_datums=len(logprobs)).fields["logprobs"]
    field.dtype = pb.DTYPE_FLOAT32
    field.data = np.concatenate(logprobs).astype(np.float32).tobytes()
    field.offsets = np.cumsum([0] + [4 * len(x) for x in logprobs], dtype=np.int64).tobytes()
  return out.SerializeToString()


@dataclass(frozen=True)
class SampleOut:
  stop: bool
  tokens: np.ndarray
  logprobs: np.ndarray
  topk: tuple[np.ndarray, np.ndarray] | None = None


def pack_topk(dst, topk: tuple[np.ndarray, np.ndarray]) -> None:
  tk_ids, tk_lps = topk
  dst.token_ids, dst.logprobs = tk_ids.astype(np.int32).tobytes(), tk_lps.astype(np.float32).tobytes()
  dst.k, dst.length = tk_ids.shape[-1], tk_ids.shape[0]


def sample_output(
  sequences: list[SampleOut],
  prompt_logprobs: np.ndarray | None = None,
  topk_prompt: tuple[np.ndarray, np.ndarray] | None = None,
) -> bytes:
  out = pb.SampleResponse()
  for s in sequences:
    seq = out.sequences.add()
    seq.stop_reason = pb.STOP_REASON_STOP if s.stop else pb.STOP_REASON_LENGTH
    seq.tokens, seq.logprobs = s.tokens.astype(np.int32).tobytes(), s.logprobs.astype(np.float32).tobytes()
    if s.topk:
      pack_topk(seq.topk_sampled_logprobs, s.topk)
  if prompt_logprobs is not None:
    out.prompt_logprobs = prompt_logprobs.astype(np.float32).tobytes()
  if topk_prompt is not None:
    pack_topk(out.topk_prompt_logprobs, topk_prompt)
  return out.SerializeToString()
