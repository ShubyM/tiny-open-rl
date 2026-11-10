"""Checkpoint storage behind put(uri) and get(uri).

Tensors are stored in safetensors with JSON metadata in the header, never torch.save pickle. Sampler checkpoints write the layout PEFT and vLLM load directly (adapter_model.safetensors + adapter_config.json, or model.safetensors + config.json, sharded with a model.safetensors.index.json when large).

CKPT is a local directory, or a URL like gs://bucket/prefix that every node can reach. With a URL, the node is a cache in front of it: put writes a checkpoint into CKPT_CACHE under a new version and returns, so this node reads it at once. After start(), a background thread uploads the node's saves behind, in order, to <uri>/<version>/ with <uri>/_latest pointing at it last. The node daemon starts it, so a trainer exits, and gives its GPUs back, right after its last save. Another node copies the version it wants from the node that saved it, which serves its cache until its saves upload, else from the bucket, once, into its own CKPT_CACHE, since engines load from a local path. A node lost before its uploads finish loses those versions.

A full fine-tune's sampler weights change in few bf16 values a step, so its saves form a chain per run, named <keyframe>.<n>, kept under <run>/_versions/ while any checkpoint is built from them. Link 0 goes up as the bytes that changed from the base model if this node has it, else whole, and every link whose parent didn't upload goes up whole. The rest go up as the bytes that changed from the link before, zlib compressed, with a CRC per chunk, and a node rebuilds a link from the newest one it has. sweep() deletes the links nothing is built from.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import ctypes
import glob
import hashlib
import http.server
import json
import mmap
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
import weakref
import zlib
from typing import TYPE_CHECKING

import fsspec
import numpy as np

if TYPE_CHECKING:
  import torch

from tinyrl.helpers import getenv
from tinyrl.trace import span

CKPT = getenv("CKPT", "/tmp/tinyrl")
CKPT_CACHE = getenv("CKPT_CACHE", "/tmp/tinyrl-cache")
SHARD_BYTES = getenv("SHARD_BYTES", 256 * 1024 * 1024)  # a full checkpoint this big is written as parallel shards
MAX_SHARDS = 16
# a save copies GPU tensors off in pieces this big, through two pinned buffers per file
STREAM_CHUNK = 256 * 1024 * 1024
# trim_cache() keeps CKPT_CACHE under this, oldest first. 0 is half its disk
CACHE_BYTES = getenv("CKPT_CACHE_BYTES", 0)
KEYFRAME_EVERY = getenv("KEYFRAME_EVERY", 8)  # the most links in a chain of full sampler weights
XOR_CHUNK = 16 * 1024 * 1024


def pinned_bytes(size: int) -> torch.Tensor:
  import torch

  cudart = None
  if size > 0 and sys.platform == "linux" and torch.cuda.is_available():
    with contextlib.suppress(AssertionError, RuntimeError):
      cudart = torch.cuda.cudart()
  if cudart is None:
    return torch.empty(size, dtype=torch.uint8, pin_memory=torch.cuda.is_available())
  alloc = (size + 4095) & ~4095
  m = mmap.mmap(-1, alloc, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
  buf = torch.frombuffer(m, dtype=torch.uint8, count=size)
  ptr = buf.data_ptr()
  step = (alloc // min(32, max(1, os.cpu_count() or 1), max(1, alloc >> 22)) + 4095) & ~4095
  parallel(lambda a: ctypes.memset(ptr + a, 0, min(step, alloc - a)), list(range(0, alloc, step)))
  if cudart.cudaHostRegister(ptr, alloc, 0) != 0:
    m.close()
    return torch.empty(size, dtype=torch.uint8, pin_memory=True)
  weakref.finalize(buf.untyped_storage(), lambda mm, p: (cudart.cudaHostUnregister(p), mm.close()), m, ptr)
  return buf


def nbytes(t: torch.Tensor) -> int:
  return t.numel() * t.element_size()


def aligned(n: int) -> int:
  return (n + 7) & ~7  # so a view of the pinned buffer can start any dtype


def as_bytes(t: torch.Tensor) -> torch.Tensor:
  import torch

  return t.detach().contiguous().reshape(-1).view(torch.uint8)


def header(path: str) -> dict:
  import safetensors

  with safetensors.safe_open(path, framework="pt", device="cpu") as f:
    return json.loads((f.metadata() or {}).get("meta", "{}"))


def parallel(fn, *xs):
  if len(xs[0]) <= 1:
    return [fn(*x) for x in zip(*xs)]
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(xs[0])) as pool:
    return list(pool.map(fn, *xs))


def raw(path: str) -> tuple[bytes, np.ndarray]:
  """A safetensors file's header, and the bytes after it, mapped rather than read."""
  with open(path, "rb") as f:
    head = f.read(int.from_bytes(f.read(8), "little"))
  empty = os.path.getsize(path) == 8 + len(head)
  return head, np.zeros(0, np.uint8) if empty else np.memmap(path, np.uint8, "r", offset=8 + len(head))


def layout(head: bytes) -> dict:
  return {k: v for k, v in json.loads(head).items() if k != "__metadata__"}


def chunks(n: int, size: int) -> list[slice]:
  return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def threads(fn, *xs) -> list:
  with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:
    return list(pool.map(fn, *xs))  # zlib and numpy let go of the GIL


def sparse(old: np.ndarray, new: np.ndarray) -> tuple[bytes, int]:
  """The bytes of new that differ from old, as the gaps between them then their XOR with old, zlib compressed, and new's CRC. Few bytes differ, so this is small, and quicker to compress than the whole XOR."""
  x = np.bitwise_xor(old, new)
  at = np.flatnonzero(x)
  return zlib.compress(np.diff(at, prepend=-1).astype(np.uint32).tobytes() + x[at].tobytes(), 1), zlib.crc32(new)


def unsparse(data: np.ndarray, z: np.ndarray) -> int:
  """Apply sparse()'s bytes to data in place, and return its CRC."""
  raw = zlib.decompress(z)
  at = np.cumsum(np.frombuffer(raw, np.uint32, len(raw) // 5), dtype=np.int64) - 1
  data[at] ^= np.frombuffer(raw, np.uint8, offset=len(raw) // 5 * 4)
  return zlib.crc32(data)


def base_files(ref: str) -> list[str]:
  """The base model's safetensors files on this node, never fetching a hub model. Empty if they aren't here."""
  from tinyrl.storage import models

  path = models.local(ref) if ref else ""
  if path and not os.path.isdir(path):
    from huggingface_hub import snapshot_download

    try:
      path = snapshot_download(path, local_files_only=True)
    except Exception:  # not in this node's hub cache
      return []
  return sorted(glob.glob(f"{path}/*.safetensors")) if path else []


def lay_out_base(heads: dict[str, bytes], out: str) -> bool:
  """Write the run's base model into files laid out like these weights files' headers: each tensor from the base tensor of the same name, dtype and shape, else zeros. A link then goes up as what changed from the base, so the bucket holds the base once. False if the base isn't on this node."""
  meta = json.loads(json.loads(next(iter(heads.values()))).get("__metadata__", {}).get("meta", "{}"))
  found = {
    k: (data, v) for f in base_files(meta.get("base", "")) for head, data in [raw(f)] for k, v in layout(head).items()
  }
  if not found:
    return False
  os.makedirs(out, exist_ok=True)
  for name, head in heads.items():
    tensors, size = layout(head), 8 + len(head)
    with open(f"{out}/{name}", "wb") as f:
      f.write(len(head).to_bytes(8, "little") + head)
      f.truncate(size + max((v["data_offsets"][1] for v in tensors.values()), default=0))
    if os.path.getsize(f"{out}/{name}") > size:
      data = np.memmap(f"{out}/{name}", np.uint8, "r+", offset=size)
      for k, v in tensors.items():
        if (b := found.get(k)) and (b[1]["dtype"], b[1]["shape"]) == (v["dtype"], v["shape"]):
          (s, e), (bs, be) = v["data_offsets"], b[1]["data_offsets"]
          data[s:e] = b[0][bs:be]
      data.flush()
  return True


def fetch(url: str, path: str) -> None:
  with urllib.request.urlopen(url, timeout=10) as r, open(path, "wb") as f:
    shutil.copyfileobj(r, f, 16 * 1024 * 1024)
    if r.length:  # the connection closed early
      raise OSError(f"{url} ended {r.length} bytes short")


class CacheFiles(http.server.SimpleHTTPRequestHandler):
  """Serves a node's cache to other nodes: a version's files, and a version folder as the JSON list of its files."""

  def list_directory(self, path: str) -> None:
    self.send_response(200)
    self.end_headers()  # HTTP/1.0: the body ends when the connection closes
    self.wfile.write(json.dumps(sorted(os.listdir(path))).encode())


class Store:
  def __init__(self, root: str | None = None, cache: str | None = None):
    self.root = (CKPT if root is None else root).rstrip("/")
    self.remote = "://" in self.root
    # a file:// URL makes folders as it writes. No listings cache: other nodes write the bucket, so exists() and glob() must look
    self.fs, self.base = fsspec.core.url_to_fs(self.root, auto_mkdir=True, use_listings_cache=False)
    if not self.remote:
      self.base = os.path.realpath(self.base)
    self.cache = CKPT_CACHE if cache is None else cache
    self.buffers: dict[int, list[torch.Tensor]] = {}  # shard -> pinned buffers its writer keeps, since pinning is slow
    self.lock = threading.RLock()  # one save at a time, since saves share self.buffers
    self.uploading, self.stopping, self.uploader, self.server = threading.Lock(), threading.Event(), None, None
    self.peer = os.environ.get("CKPT_PEER", "")  # host:port of this node's cache server, set by start()
    self.chains: dict[str, str] = {}  # run -> the last link of its chain this process saved

  def path(self, uri: str) -> str:
    return f"{self.base}/{uri.removeprefix('tinker://')}"

  def env(self) -> dict[str, str]:
    """What a worker process on this node needs to open the same store."""
    return {"CKPT": self.root, "CKPT_CACHE": self.cache, "CKPT_PEER": self.peer}

  def start(self) -> None:
    """Make this node's folders, serve its cache to other nodes, and upload its saves behind from now on. The node daemon calls it once."""
    # a sampler restored from a snapshot sees only the folders under /tmp and /dev/shm that exist when it restores
    for folder in [self.cache] + ([] if self.remote else [self.base]):
      os.makedirs(folder, exist_ok=True)
    if self.remote:  # other nodes copy a new save from here, rather than wait for its upload
      self.server = http.server.ThreadingHTTPServer(("", 0), lambda *a: CacheFiles(*a, directory=self.cache))
      threading.Thread(target=self.server.serve_forever, daemon=True).start()
      self.peer = f"{socket.gethostbyname(socket.gethostname())}:{self.server.server_port}"
    self.uploader = threading.Thread(target=self.keep_uploading, daemon=True)
    self.uploader.start()

  def keep_uploading(self) -> None:
    while not self.stopping.wait(0.25):  # seconds between looks for new saves
      with contextlib.suppress(Exception):  # it logs a failed upload. On any failure the next pass retries
        self.upload_pending_saves()

  def close(self) -> None:
    """Return once every save this node made has uploaded."""
    self.stopping.set()
    self.uploader.join()
    self.upload_pending_saves()
    if self.server:  # the bucket has every save now
      self.server.shutdown()

  # ***** put *****

  def write_safetensors(self, i: int, path: str, tensors: dict[str, torch.Tensor], meta: dict) -> None:
    """safetensors.torch.save_file, but GPU tensors stream through two pinned buffers, so each piece is written while the next copies off the GPU, and the save never holds a host copy of itself."""
    import torch

    def dtype(d: torch.dtype) -> str:
      kind = "BF" if d == torch.bfloat16 else "F" if d.is_floating_point else "U" if d == torch.uint8 else "I"
      return "BOOL" if d == torch.bool else f"{kind}{d.itemsize * 8}"

    head, at = {"__metadata__": meta}, 0
    for k, v in tensors.items():
      head[k] = {"dtype": dtype(v.dtype), "shape": list(v.shape), "data_offsets": [at, at + nbytes(v)]}
      at += nbytes(v)
    h = json.dumps(head).encode()
    h += b" " * (-len(h) % 8)
    gpu = next((v.device for v in tensors.values() if v.is_cuda), None)
    if gpu is not None and i not in self.buffers:
      self.buffers[i] = [pinned_bytes(STREAM_CHUNK) for _ in range(2)]
    bufs, pending, j = self.buffers.get(i), [], 0
    with open(path, "wb") as f, torch.cuda.device(-1 if gpu is None else gpu):
      f.write(len(h).to_bytes(8, "little") + h)
      stream = torch.cuda.Stream() if gpu is not None else None

      def flush(keep: int) -> None:
        while len(pending) > keep:
          ev, n, buf = pending.pop(0)
          ev.synchronize()
          f.write(buf[:n].numpy())

      for v in tensors.values():
        src = as_bytes(v)
        if not v.is_cuda:
          flush(0)
          f.write(src.numpy())
          continue
        stream.wait_stream(torch.cuda.current_stream())
        src.record_stream(stream)
        for piece in chunks(len(src), STREAM_CHUNK):
          flush(1)  # write out the buffer this piece is about to reuse
          buf, j = bufs[j % 2], j + 1
          with torch.cuda.stream(stream):
            buf[: piece.stop - piece.start].copy_(src[piece], non_blocking=True)
            pending.append((torch.cuda.Event(), piece.stop - piece.start, buf))
            pending[-1][0].record(stream)
      flush(0)

  def write_checkpoint_files(self, folder: str, tensors: dict[str, torch.Tensor], info: dict) -> list[str]:
    """Write a checkpoint into folder, and return its file names, with the weights last."""
    lora, cfg, total = info.get("lora"), info.get("config"), sum(map(nbytes, tensors.values()))
    n = max(1, min(MAX_SHARDS, len(tensors))) if not lora and total >= SHARD_BYTES else 1
    stem = ("adapter_model" if lora else "model") if cfg else "state"
    shards, sizes = [{} for _ in range(n)], [0] * n
    for k, v in sorted(tensors.items(), key=lambda kv: nbytes(kv[1]), reverse=True):
      i = sizes.index(min(sizes))
      shards[i][k], sizes[i] = v, sizes[i] + nbytes(v)
    shards = [s for s in shards if s] or [{}]
    n = len(shards)
    weights = [f"{stem}.safetensors"] if n == 1 else [f"{stem}-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]
    files = {}
    if cfg:
      files["adapter_config.json" if lora else "config.json"] = cfg
    if cfg and n > 1:
      weight_map = {k: w for w, shard in zip(weights, shards) for k in shard}
      files["model.safetensors.index.json"] = {"metadata": {"total_size": total}, "weight_map": weight_map}
    for name, obj in files.items():
      with open(f"{folder}/{name}", "w") as f:
        json.dump(obj, f)
    meta = {"meta": json.dumps(info)}
    parallel(
      lambda i, name, shard: self.write_safetensors(i, f"{folder}/{name}", shard, meta), range(n), weights, shards
    )
    return [*files, *weights]

  @span("store.put")
  def put(self, uri: str, tensors: dict[str, torch.Tensor], meta: dict | None = None) -> str | None:
    """Save a checkpoint, and return the version to read it by on another node, which names this node so others can copy it from here. A local store has no versions: None."""
    dest = self.path(uri)
    if self.remote:
      os.makedirs(self.cache, exist_ok=True)
      folder, version = tempfile.mkdtemp(dir=self.cache), uuid.uuid4().hex[:16]
      if (meta or {}).get("config") and not (meta or {}).get("lora"):  # a full fine-tune's sampler weights
        run = uri.removeprefix("tinker://").split("/")[0]
        k, _, n = self.chains.get(run, "").rpartition(".")
        version = self.chains[run] = f"{k}.{int(n) + 1}" if k and int(n) + 1 < KEYFRAME_EVERY else f"{version}.0"
      try:
        with self.lock:
          names = self.write_checkpoint_files(folder, tensors, dict(meta or {}))
        os.rename(folder, f"{self.cache}/{version}")
      finally:
        shutil.rmtree(folder, ignore_errors=True)
      self.record_latest_version(uri, version)
      with open(f"{self.cache}/{version}.upload.tmp", "w") as f:
        json.dump({"uri": uri, "names": names}, f)
      # upload_pending_saves() takes it from here
      os.replace(f"{self.cache}/{version}.upload.tmp", f"{self.cache}/{version}.upload")
      return f"{version}@{self.peer}" if self.peer else version
    parent, base = os.path.dirname(dest), os.path.basename(dest)
    os.makedirs(parent, exist_ok=True)
    folder = tempfile.mkdtemp(dir=parent, prefix=f".{base}.")
    link, prev = f"{folder}.link", f"{parent}/.{base}.prev"
    try:
      with self.lock:
        self.write_checkpoint_files(folder, tensors, dict(meta or {}))
        cur = os.readlink(dest) if os.path.islink(dest) else None
        stale = os.readlink(prev) if os.path.islink(prev) else None
        os.symlink(os.path.basename(folder), link)
        if os.path.isdir(dest) and not os.path.islink(dest):
          os.rename(dest, f"{folder}.old")
          os.replace(link, dest)
          shutil.rmtree(f"{folder}.old", ignore_errors=True)
        else:
          os.replace(link, dest)
        if cur:
          os.symlink(cur, link)
          os.replace(link, prev)
        folder = ""
      if stale and stale != cur:
        shutil.rmtree(f"{parent}/{stale}", ignore_errors=True)
    finally:
      if folder:
        shutil.rmtree(folder, ignore_errors=True)
      if os.path.lexists(link):  # left only if the swap failed
        os.unlink(link)

  def record_latest_version(self, uri: str, version: str) -> None:
    """Record on this node which version of a checkpoint it saved last, for this node's other processes."""
    os.makedirs(f"{self.cache}/latest", exist_ok=True)
    tmp = f"{self.cache}/latest/.{uuid.uuid4().hex}"
    with open(tmp, "w") as f:
      f.write(version)
    os.replace(tmp, f"{self.cache}/latest/{hashlib.sha256(uri.encode()).hexdigest()[:32]}")

  def latest_version_here(self, uri: str) -> str | None:
    pointer = f"{self.cache}/latest/{hashlib.sha256(uri.encode()).hexdigest()[:32]}"
    if not os.path.exists(pointer):  # nothing saved on this node yet
      return None
    with open(pointer) as f:
      return f.read()

  def upload_version(self, dest: str, version: str, names: list[str]) -> None:
    folder, rel = f"{self.cache}/{version}", dest[len(self.base) + 1 :]
    info, files = {"version": version, "dir": f"{rel}/{version}", "files": names}, [f"{folder}/{n}" for n in names]
    k, _, n = version.rpartition(".")
    if k:  # a link of the run's chain
      info["dir"], parent = f"{rel.split('/')[0]}/_versions/{version}", f"{self.cache}/{k}.{int(n) - 1}"
      first = ""  # the newest link before it that went up whole, if the one before it went up
      if os.path.exists(f"{parent}.uploaded") and self.fs.exists(
        f"{self.base}/{rel.split('/')[0]}/_versions/{k}.{int(n) - 1}"
      ):
        with open(f"{parent}.uploaded") as f:
          first = f.read()
      xor = int(n) and first and self.write_changed_bytes(parent, folder, names, f"{folder}.xor")
      heads = {name: raw(f"{folder}/{name}")[0] for name in names if name.endswith(".safetensors")}
      if not int(n) and lay_out_base(heads, f"{folder}.base"):
        xor, first = self.write_changed_bytes(f"{folder}.base", folder, names, f"{folder}.xor"), "base"
      if xor:
        files = [f"{folder}.xor/{x}" for x in xor] + [f for f in files if not f.endswith(".safetensors")]
      info["from"] = first if xor else version
    try:
      parallel(lambda f: self.fs.put_file(f, f"{self.base}/{info['dir']}/{os.path.basename(f)}"), files)
    finally:
      shutil.rmtree(f"{folder}.xor", ignore_errors=True)
      shutil.rmtree(f"{folder}.base", ignore_errors=True)
    old = self.latest(dest)
    self.fs.pipe_file(f"{dest}/_latest", json.dumps(info).encode())
    if "." not in version:
      keep = {"_latest", version, old.get("version")}
      if not old.get("dir"):
        keep.update(old.get("files", []))
      for e in self.fs.ls(dest, detail=False):
        if os.path.basename(e) not in keep:
          self.fs.rm(e, recursive=True)
    with open(f"{folder}.uploaded", "w") as f:  # beside the folder, so engines that list it never see it
      f.write(info.get("from", ""))

  def write_changed_bytes(self, parent: str, folder: str, names: list[str], out: str) -> list[str] | None:
    """Write each weights file in folder as the bytes that differ from parent's, by chunk, with the file's header and a CRC per chunk. None if the files don't line up."""
    import safetensors.numpy

    shards = [n for n in names if n.endswith(".safetensors")]
    if not all(os.path.exists(f"{parent}/{n}") for n in shards):  # trimmed from this node's cache
      return None
    files = [(raw(f"{parent}/{n}"), raw(f"{folder}/{n}")) for n in shards]
    if any(layout(was) != layout(head) or len(old) != len(new) for (was, old), (head, new) in files):
      return None
    jobs = [(i, s) for i, (_, (_, new)) in enumerate(files) for s in chunks(len(new), XOR_CHUNK)]
    done = threads(
      lambda i, s: sparse(files[i][0][1][s], files[i][1][1][s]), [i for i, _ in jobs], [s for _, s in jobs]
    )
    os.makedirs(out, exist_ok=True)
    for i, (name, (_, (head, _))) in enumerate(zip(shards, files)):
      mine = [d for (j, _), d in zip(jobs, done) if j == i]
      meta = {"head": head.decode(), "chunk": str(XOR_CHUNK), "crc": json.dumps([c for _, c in mine])}
      safetensors.numpy.save_file(
        {str(k): np.frombuffer(z, np.uint8) for k, (z, _) in enumerate(mine)}, f"{out}/{name}.xor", metadata=meta
      )
    return [f"{n}.xor" for n in shards]

  def upload_pending_saves(self) -> None:
    """Upload this node's saves in the order they were made. A failed upload skips the rest for that URI or chain, so _latest never goes backward, and the next flush tries again."""
    with self.uploading:  # the node daemon's last call at shutdown waits for the one running behind
      found = [e for e in os.scandir(self.cache) if e.name.endswith(".upload")] if os.path.isdir(self.cache) else []
      failed = set()
      for e in sorted(found, key=lambda e: (e.stat().st_mtime_ns, e.name)):
        version = e.name.removesuffix(".upload")
        k = version.rpartition(".")[0] if "." in version else None
        with open(e.path) as f:
          job = json.load(f)
        if (k and k in failed) or job["uri"] in failed:
          continue
        if os.path.isdir(f"{self.cache}/{version}"):
          try:
            self.upload_version(self.path(job["uri"]), version, job["names"])
          except Exception as exc:
            if k:
              failed.add(k)
            failed.add(job["uri"])
            print(f"upload failed for {version}, trying again: {exc!r}", file=sys.stderr, flush=True)
            continue
        os.unlink(e.path)

  # ***** get *****

  def latest(self, dest: str) -> dict:
    return json.loads(self.fs.cat_file(f"{dest}/_latest")) if self.fs.exists(f"{dest}/_latest") else {}

  def local_folder(self, uri: str, version: str | None = None) -> tuple[str, str]:
    """A local directory holding the checkpoint, and its version, which changes when the checkpoint is saved again. A version put() returned comes from this node's cache, else the node that saved it, else the bucket."""
    dest = self.path(uri)
    if not self.remote:
      real = os.path.join(os.path.dirname(dest), os.readlink(dest)) if os.path.islink(dest) else dest
      if not os.path.isdir(real):
        raise ValueError(f"no checkpoint at {uri}")
      with os.scandir(real) as it:
        stats = sorted((e.name, s.st_size, s.st_mtime_ns) for e in it for s in [e.stat()])
      if not any(n.endswith(".safetensors") for n, _, _ in stats):
        raise ValueError(f"no checkpoint at {uri}")
      return dest, hashlib.blake2b(repr((real, stats)).encode(), digest_size=8).hexdigest()
    version, _, peer = (version or "").partition("@")
    # without a version: the one this node saved last, else the bucket's
    want = version or self.latest_version_here(uri)
    if want and os.path.isdir(f"{self.cache}/{want}"):
      os.utime(f"{self.cache}/{want}")  # so trim_cache() takes the ones nobody read for longest
      return f"{self.cache}/{want}", want
    if peer and (folder := self.copy_from(peer, version)):
      return folder, version
    info = self.latest(dest)
    if version and info.get("version") != version:
      raise ValueError(f"version {version} of {uri} has not landed" + (f", and {peer} failed to copy" if peer else ""))
    if not info:
      raise ValueError(f"no checkpoint at {uri}")
    version, names = info["version"], info["files"]
    if not any(n.endswith(".safetensors") for n in names):
      raise ValueError(f"no checkpoint at {uri}")
    src = f"{self.base}/{info['dir']}" if info.get("dir") else dest
    if info.get("from", version) == version:
      return self.cached_folder(version, lambda tmp: self.download_files(src, names, tmp)), version
    return self.cached_folder(
      version, lambda tmp: self.rebuild_chain_link(os.path.dirname(src), version, info["from"], names, tmp)
    ), version

  def copy_from(self, peer: str, version: str) -> str | None:
    """A version copied from the node that saved it, which keeps it until it has uploaded. None if the copy fails: that node is gone, or no longer has it."""
    url = f"http://{peer}/{version}"
    try:
      names = json.load(urllib.request.urlopen(f"{url}/", timeout=10))
      return self.cached_folder(version, lambda tmp: parallel(lambda n: fetch(f"{url}/{n}", f"{tmp}/{n}"), names))
    except Exception:  # the bucket has it, or it's lost
      return None

  def cached_folder(self, version: str, fill) -> str:
    """This node's folder for a version, which fill(folder) makes the first time."""
    folder = f"{self.cache}/{version}"
    if not os.path.isdir(folder):
      os.makedirs(self.cache, exist_ok=True)
      tmp = tempfile.mkdtemp(dir=self.cache)
      try:
        fill(tmp)
        try:
          os.rename(tmp, folder)
        except OSError:
          if not os.path.isdir(folder):
            raise
        open(f"{folder}.uploaded", "w").close()  # the bucket has it, so trim_cache() may drop it here
      finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return folder

  def download_files(self, src: str, names: list[str], folder: str) -> None:
    parallel(lambda n: self.fs.get_file(f"{src}/{n}", f"{folder}/{n}"), names)

  def rebuild_chain_link(self, links: str, version: str, whole: str, names: list[str], folder: str) -> None:
    """Make a link of a chain in folder, from the newest link before it this node has, or else the newest one uploaded whole, or the base model, by XORing in each link after that. The CRCs check the result is the exact bytes saved."""
    import safetensors

    k, _, n = version.rpartition(".")
    chain = [f"{k}.{i}" for i in range(0 if whole == "base" else int(whole.rpartition(".")[2]), int(n) + 1)]
    cached = [i for i in range(len(chain) - 1) if os.path.isdir(f"{self.cache}/{chain[i]}")]
    from_base = whole == "base" and not cached  # a chain whose link 0 went up as its changes from the base model
    have = -1 if from_base else (cached or [0])[-1]
    shards, after = [n for n in names if n.endswith(".safetensors")], chain[have + 1 :]
    self.download_files(f"{links}/{version}", [n for n in names if n not in shards], folder)
    pairs = [(f"{link}/{name}.xor", f"{folder}/{link}.{name}.xor") for link in after for name in shards]
    parallel(lambda a, b: self.fs.get_file(f"{links}/{a}", b), *zip(*pairs))
    if from_base:
      heads = {}
      for name in shards:
        with safetensors.safe_open(f"{folder}/{chain[0]}.{name}.xor", framework="np") as f:
          heads[name] = f.metadata()["head"].encode()
      if not lay_out_base(heads, start := f"{folder}.base"):
        raise ValueError(f"{version} is built on its base model, which isn't on this node")
    else:
      start = self.cached_folder(chain[have], lambda tmp: self.download_files(f"{links}/{chain[have]}", names, tmp))
    data = dict(zip(shards, parallel(lambda name: np.array(raw(f"{start}/{name}")[1]), shards)))
    shutil.rmtree(f"{folder}.base", ignore_errors=True)
    for link in after:
      metas, jobs = {}, []
      for name in shards:
        with safetensors.safe_open(f"{folder}/{link}.{name}.xor", framework="np") as f:
          metas[name] = f.metadata()
          size = int(metas[name]["chunk"])
          jobs += [(name, s, f.get_tensor(str(k))) for k, s in enumerate(chunks(len(data[name]), size))]
        os.unlink(f"{folder}/{link}.{name}.xor")
      crcs = threads(lambda name, s, z: (name, unsparse(data[name][s], z)), *zip(*jobs)) if jobs else []
      if any([c for n, c in crcs if n == name] != json.loads(metas[name]["crc"]) for name in shards):
        raise ValueError(f"{link} rebuilt wrong")

    def write(name: str) -> None:
      head = metas[name]["head"].encode()
      with open(f"{folder}/{name}", "wb") as f:
        f.write(len(head).to_bytes(8, "little") + head)
        f.write(data[name])

    parallel(write, shards)

  def weight_files(self, uri: str, version: str | None = None) -> list[str]:
    folder = self.local_folder(uri, version)[0]
    if os.path.islink(folder):
      folder = os.path.join(os.path.dirname(folder), os.readlink(folder))
    return sorted(glob.glob(f"{folder}/*.safetensors"))

  @span("store.get")
  def get(self, uri: str, version: str | None = None) -> tuple[dict[str, torch.Tensor], dict]:
    import safetensors

    def load(path: str) -> dict[str, torch.Tensor]:
      with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}

    paths = self.weight_files(uri, version)
    return {k: v for s in parallel(load, paths) for k, v in s.items()}, header(paths[0])

  def read_meta(self, uri: str) -> dict:
    return header(self.weight_files(uri)[0])

  # ***** delete *****

  def delete(self, uri: str) -> None:
    """Delete a checkpoint and every version of it this store kept."""
    dest = self.path(uri)
    if self.remote:
      if self.fs.exists(dest):
        self.fs.rm(dest, recursive=True)
      return
    parent, base = os.path.split(dest)
    # the random suffix of a version folder has no dot, so step-1 never takes step-1.5's
    tail = len(parent) + len(base) + 3
    versions = [p for p in glob.glob(f"{glob.escape(parent)}/.{glob.escape(base)}.*") if "." not in p[tail:]]
    for p in [dest, *versions]:  # the link to the live version, the .prev link, and the folders they name
      if os.path.islink(p):
        os.unlink(p)
      else:
        shutil.rmtree(p, ignore_errors=True)

  def list_checkpoints(self) -> list[str]:
    """Every checkpoint in the store, including one whose upload never wrote its _latest."""
    if self.remote:
      return [
        f"tinker://{p[len(self.base) + 1 :]}" for p in self.fs.glob(f"{self.base}/*/*/*") if "/_versions/" not in p
      ]
    found = glob.glob(f"{glob.escape(self.base)}/*/*/*")  # glob skips the dot folders a save writes into
    return [f"tinker://{os.path.relpath(p, self.base)}" for p in found]

  def saved_at(self, uri: str) -> float:
    """When a checkpoint was last saved."""
    if self.remote and self.fs.exists(latest := f"{self.path(uri)}/_latest"):
      return self.fs.modified(latest).timestamp()
    if self.remote:  # an upload that never finished: when its newest file landed
      return max((self.fs.modified(f).timestamp() for f in self.fs.find(self.path(uri))), default=0.0)
    return os.lstat(self.path(uri)).st_mtime

  def sweep(self, registered: set[str], versions: set[str], since: float | None, grace: float) -> None:
    """Delete what the registry doesn't have, saved after since, when it began, and more than grace seconds ago: checkpoints deleted, expired or their run's, the links of chains none of the versions is built from, and saves of those checkpoints still waiting to upload. Then keep the cache under CKPT_CACHE_BYTES."""
    self.trim_cache()
    if since is None:  # so turning the registry on never deletes what came before
      return
    before, versions = time.time() - grace, {v.partition("@")[0] for v in versions}
    for uri in set(self.list_checkpoints()) - registered:
      try:
        if since < self.saved_at(uri) < before:
          self.delete(uri)
      except Exception as e:  # one checkpoint it can't read or delete mustn't stop the rest of the sweep
        print(f"sweep could not delete {uri}, trying again next sweep: {e!r}", file=sys.stderr, flush=True)
    keep = {f"{k}.{i}" for v in versions if "." in v for k, _, n in [v.rpartition(".")] for i in range(int(n) + 1)}
    for links in self.fs.glob(f"{self.base}/*/_versions/*") if self.remote else []:
      if links.rsplit("/", 1)[1] not in keep and (files := self.fs.ls(links, detail=False)):
        if since < self.fs.modified(files[0]).timestamp() < before:
          self.fs.rm(links, recursive=True)
    for e in list(os.scandir(self.cache)) if os.path.isdir(self.cache) else []:
      if e.name.endswith(".upload") and since < e.stat().st_mtime < before:
        with contextlib.suppress(OSError, ValueError), open(e.path) as f:  # it may have uploaded meanwhile
          if json.load(f)["uri"] not in registered:
            shutil.rmtree(f"{self.cache}/{e.name.removesuffix('.upload')}", ignore_errors=True)
            os.unlink(e.path)

  def trim_cache(self) -> None:
    """Delete the versions in CKPT_CACHE nobody read for longest, until it is under CKPT_CACHE_BYTES. Only ones the bucket has: a version still uploading stays."""
    if not os.path.isdir(self.cache):
      return
    limit = CACHE_BYTES or shutil.disk_usage(self.cache).total // 2
    found = [e for e in os.scandir(self.cache) if e.is_dir() and os.path.exists(f"{e.path}.uploaded")]
    size = {e.path: sum(f.stat().st_size for f in os.scandir(e.path)) for e in found}
    total = sum(size.values())
    for e in sorted(found, key=lambda e: e.stat().st_mtime):
      if total <= limit:
        break
      shutil.rmtree(e.path, ignore_errors=True)
      os.unlink(f"{e.path}.uploaded")
      total -= size[e.path]
