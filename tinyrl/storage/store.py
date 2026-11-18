"""Immutable safetensors checkpoints. Redis binds each logical name to a version after the trainer commits.

Local saves rename a completed directory into objects/<id>. Bucket saves stage the same files in the node cache,
serve them to peers, and upload a per-version manifest last so readers see only complete versions.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import ctypes
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
from pathlib import Path
from typing import TYPE_CHECKING

import fsspec

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


def object_id(name: str) -> bool:
  return len(name) == 32 and all(c in "0123456789abcdef" for c in name)


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


def chunks(n: int, size: int) -> list[slice]:
  return [slice(i, min(i + size, n)) for i in range(0, n, size)]


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
    self.cache_root = CKPT_CACHE if cache is None else cache
    namespace = hashlib.sha256(self.root.encode()).hexdigest()[:16]
    self.cache = os.path.join(self.cache_root, namespace) if self.remote else self.cache_root
    self.unreferenced: dict[str, float] = {}
    self.buffers: dict[int, list[torch.Tensor]] = {}  # shard -> pinned buffers its writer keeps, since pinning is slow
    self.lock = threading.RLock()  # one save at a time, since saves share self.buffers
    self.uploading, self.stopping, self.uploader, self.server = threading.Lock(), threading.Event(), None, None
    self.peer = os.environ.get("CKPT_PEER", "")  # host:port of this node's cache server, set by start()

  def env(self) -> dict[str, str]:
    """What a worker process on this node needs to open the same store."""
    return {"CKPT": self.root, "CKPT_CACHE": self.cache_root, "CKPT_PEER": self.peer}

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
    if self.uploader:
      self.uploader.join()
      self.upload_pending_saves()
    if self.server:  # the bucket has every save now
      self.server.shutdown()
      self.server.server_close()

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
  def put(self, tensors: dict[str, torch.Tensor], meta: dict | None = None, *, version: str | None = None) -> str:
    version = version or uuid.uuid4().hex
    if not object_id(version):
      raise ValueError(f"invalid checkpoint version {version!r}")
    parent = Path(self.cache if self.remote else f"{self.base}/objects")
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / f"{version}.writing"
    folder.mkdir()
    try:
      with self.lock:
        names = self.write_checkpoint_files(str(folder), tensors, meta or {})
      folder.rename(parent / version)
      if self.remote:
        job = parent / f"{version}.uploading"
        job.write_text(json.dumps(names))
        job.rename(parent / f"{version}.upload")
    finally:
      shutil.rmtree(folder, ignore_errors=True)
    return f"{version}@{self.peer}" if self.remote and self.peer else version

  def upload_pending_saves(self) -> None:
    """An independent, retryable upload per immutable version. Publish its manifest after its files."""
    if not self.remote:
      return
    with self.uploading:
      for job in Path(self.cache).glob("*.upload"):
        version = job.stem
        folder = Path(self.cache, version)
        dest = f"{self.base}/objects/{version}"
        try:
          names = json.loads(job.read_text())
          parallel(self.fs.put_file, [str(folder / n) for n in names], [f"{dest}/{n}" for n in names])
          self.fs.pipe_file(f"{dest}/manifest.json", json.dumps(names).encode())
          Path(f"{folder}.uploaded").touch()
          job.unlink()
        except Exception as exc:
          print(f"upload failed for {version}, trying again: {exc!r}", file=sys.stderr, flush=True)

  def local_folder(self, reference: str) -> tuple[str, str]:
    """Resolve exactly the published version through shared disk, the local cache, a peer, or the bucket."""
    version, _, peer = (reference or "").partition("@")
    if not object_id(version):
      raise ValueError(f"no checkpoint for version {version!r}")
    dest = f"{self.base}/objects/{version}"
    if not self.remote:
      if not os.path.isdir(dest):
        raise ValueError(f"no checkpoint for version {version}")
      return dest, version
    folder = f"{self.cache}/{version}"
    if os.path.isdir(folder):
      os.utime(folder)
      return folder, version
    if peer and (copied := self.copy_from(peer, version)):
      return copied, version
    if not self.fs.exists(f"{dest}/manifest.json"):
      raise ValueError(f"version {version} has not landed" + (f", and {peer} failed to copy" if peer else ""))
    names = json.loads(self.fs.cat_file(f"{dest}/manifest.json"))
    return self.cached_folder(version, lambda tmp: self.download_files(dest, names, tmp)), version

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

  def weight_files(self, version: str) -> list[Path]:
    folder, _ = self.local_folder(version)
    return sorted(Path(folder).glob("*.safetensors"))

  @span("store.get")
  def get(self, version: str) -> tuple[dict[str, torch.Tensor], dict]:
    from safetensors.torch import load_file

    paths = self.weight_files(version)
    return {k: v for shard in parallel(load_file, paths) for k, v in shard.items()}, header(paths[0])

  def read_meta(self, version: str) -> dict:
    return header(self.weight_files(version)[0])

  def versions(self) -> set[str]:
    found = {os.path.basename(p).removesuffix(".writing") for p in self.fs.glob(f"{self.base}/objects/*")}
    if self.remote and os.path.isdir(self.cache):
      found.update(entry.name.split(".", 1)[0] for entry in os.scandir(self.cache))
    return {name for name in found if object_id(name)}

  def sweep(self, unused: set[str], grace: float) -> None:
    """Allow in-flight reads a grace period after a version first becomes unreachable."""
    now = time.monotonic()
    self.unreferenced = {v: self.unreferenced.get(v, now) for v in unused}
    self.discard_versions({v for v, since in self.unreferenced.items() if now - since >= grace})
    self.trim_cache()

  def discard_versions(self, versions: set[str]) -> None:
    with self.uploading:
      for version in versions:
        path = f"{self.base}/objects/{version}"
        with contextlib.suppress(FileNotFoundError):
          self.fs.rm(path, recursive=True)
        if self.remote:
          for suffix in (".upload", ".uploading", ".uploaded"):
            Path(self.cache, version + suffix).unlink(missing_ok=True)
          for suffix in ("", ".writing"):
            shutil.rmtree(Path(self.cache, version + suffix), ignore_errors=True)
        else:
          shutil.rmtree(path + ".writing", ignore_errors=True)

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
