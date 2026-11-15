"""Immutable checkpoint versions: local files first, peers immediately, durable upload behind.

The manifest publishes a complete copy. Readers hold a shared file lock; publication
and eviction take its exclusive lock. Logical checkpoint names live only in Registry.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import ctypes
import fcntl
import glob
import http.server
import json
import mmap
import os
import shutil
import socket
import sys
import tempfile
import threading
import urllib.parse
import urllib.request
import uuid
import weakref
from typing import TYPE_CHECKING

import fsspec

from tinyrl.helpers import getenv
from tinyrl.trace import span

if TYPE_CHECKING:
  import torch

CKPT, CKPT_CACHE = getenv("CKPT", "/tmp/tinyrl"), getenv("CKPT_CACHE", "/tmp/tinyrl-cache")
SHARD_BYTES, MAX_SHARDS, STREAM_CHUNK = getenv("SHARD_BYTES", 256 * 2**20), 16, 256 * 2**20
CACHE_BYTES = getenv("CKPT_CACHE_BYTES", 0)


def split_version(ref: str | None) -> tuple[str | None, str]:
  version, _, peer = (ref or "").partition("@")
  return version or None, peer


def write_json(path: str, value: dict) -> None:
  tmp = f"{path}.{uuid.uuid4().hex}.tmp"
  try:
    with open(tmp, "w") as f:
      json.dump(value, f)
    os.replace(tmp, path)
  finally:
    with contextlib.suppress(FileNotFoundError):
      os.unlink(tmp)


def manifest(folder: str) -> dict | None:
  with contextlib.suppress(OSError, ValueError, KeyError):
    with open(f"{folder}.json") as f:
      m = json.load(f)
    if m["files"] and all(
      os.path.basename(n) == n and os.path.getsize(f"{folder}/{n}") == s for n, s in m["files"].items()
    ):
      return m
  return None


@contextlib.contextmanager
def file_lock(folder: str, mode: int = fcntl.LOCK_SH):
  os.makedirs(os.path.dirname(folder), exist_ok=True)
  # Keep the inode: unlinking a lock can split waiters across two different locks.
  with open(f"{folder}.lock", "a") as lock:
    fcntl.flock(lock, mode)
    yield lock


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


def as_bytes(t: torch.Tensor) -> torch.Tensor:
  import torch

  return t.detach().contiguous().reshape(-1).view(torch.uint8)


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


class CacheFiles(http.server.BaseHTTPRequestHandler):
  def __init__(self, *args, directory: str):
    self.directory = directory
    super().__init__(*args)

  def log_message(self, *args):
    pass

  def do_GET(self):
    parts = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path).strip("/").split("/")
    version, name = (parts[0][:-5], None) if len(parts) == 1 and parts[0].endswith(".json") else (parts[0], parts[-1])
    if len(parts) > 2 or not version.isalnum():
      return self.send_error(404)
    folder = f"{self.directory}/{version}"
    with file_lock(folder):
      if not (m := manifest(folder)) or (name is not None and name not in m["files"]):
        return self.send_error(404)
      path = f"{folder}/{name}" if name else f"{folder}.json"
      self.send_response(200)
      self.send_header("Content-Length", str(os.path.getsize(path)))
      self.end_headers()
      with open(path, "rb") as f, contextlib.suppress(BrokenPipeError, ConnectionResetError):
        shutil.copyfileobj(f, self.wfile, 2**20)


class Store:
  def __init__(self, root: str | None = None, cache: str | None = None):
    self.root = (CKPT if root is None else root).rstrip("/")
    self.remote = "://" in self.root
    self.fs, self.base = fsspec.core.url_to_fs(self.root, auto_mkdir=True, use_listings_cache=False)
    self.cache = (CKPT_CACHE if cache is None else cache) if self.remote else os.path.realpath(self.base)
    self.buffers: dict[int, list[torch.Tensor]] = {}
    self.lock, self.uploading, self.stopping = threading.RLock(), threading.Lock(), threading.Event()
    self.uploader = self.server = None
    self.peer = os.environ.get("CKPT_PEER", "")

  def folder(self, version: str | None) -> str:
    if not (v := split_version(version)[0]) or not v.isalnum():
      raise ValueError("an exact checkpoint version is required")
    return f"{self.cache}/{v}"

  def env(self) -> dict[str, str]:
    return {"CKPT": self.root, "CKPT_CACHE": self.cache, "CKPT_PEER": self.peer}

  def start(self) -> None:
    os.makedirs(self.cache, exist_ok=True)
    if self.remote and self.server is None:
      self.server = http.server.ThreadingHTTPServer(("", 0), lambda *a: CacheFiles(*a, directory=self.cache))
      threading.Thread(target=self.server.serve_forever, daemon=True).start()
      self.peer = f"{socket.gethostbyname(socket.gethostname())}:{self.server.server_port}"
    if self.uploader is None:
      self.uploader = threading.Thread(target=self.keep_uploading, daemon=True)
      self.uploader.start()

  def keep_uploading(self) -> None:
    while not self.stopping.wait(0.25):
      with contextlib.suppress(OSError):
        self.upload_pending_saves()

  def close(self) -> None:
    self.stopping.set()
    if self.uploader:
      self.uploader.join()
    self.upload_pending_saves()
    if self.server:
      self.server.shutdown()
      self.server.server_close()

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
  def put(self, uri: str, tensors: dict[str, torch.Tensor], meta: dict | None = None) -> str:
    version, info = uuid.uuid4().hex[:16], dict(meta or {})
    folder = self.folder(version)
    os.makedirs(self.cache, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=self.cache, prefix=".put.")
    try:
      with self.lock:
        files = self.write_checkpoint_files(tmp, tensors, info)
      m = {"uri": uri, "version": version, "meta": info, "files": {n: os.path.getsize(f"{tmp}/{n}") for n in files}}
      with file_lock(folder, fcntl.LOCK_EX):
        os.rename(tmp, folder)
        write_json(f"{folder}.json", m)
        open(f"{folder}.upload" if self.remote else f"{folder}.uploaded", "w").close()
      return f"{version}@{self.peer}" if self.remote and self.peer else version
    finally:
      shutil.rmtree(tmp, ignore_errors=True)

  def is_durable(self, uri: str, version: str) -> bool:
    v, folder = split_version(version)[0], self.folder(version)
    if os.path.exists(f"{folder}.uploaded") and manifest(folder):
      return True
    if self.remote and self.fs.exists(f"{self.base}/{v}.json"):
      with file_lock(folder):
        if manifest(folder):
          open(f"{folder}.uploaded", "w").close()
      return True
    return False

  def upload_version(self, uri: str, version: str) -> bool:
    if not self.remote:
      return manifest(self.folder(version)) is not None
    v, folder = split_version(version)[0], self.folder(version)
    with file_lock(f"{folder}.upload", fcntl.LOCK_EX), file_lock(folder):
      if not (m := manifest(folder)):
        return False
      if not os.path.exists(f"{folder}.uploaded"):
        parallel(lambda n: self.fs.put_file(f"{folder}/{n}", f"{self.base}/{v}/{n}"), list(m["files"]))
        self.fs.pipe_file(f"{self.base}/{v}.json", json.dumps(m).encode())
        open(f"{folder}.uploaded", "w").close()
      with contextlib.suppress(FileNotFoundError):
        os.unlink(f"{folder}.upload")
      return True

  def upload_pending_saves(self) -> None:
    if not self.remote:
      return
    with self.uploading:
      pending = []
      for path in glob.glob(f"{self.cache}/*.upload"):
        with contextlib.suppress(FileNotFoundError):
          pending.append((os.path.getmtime(path), path))
      for _, path in sorted(pending):
        try:
          if m := manifest(path[:-7]):
            self.upload_version(m["uri"], m["version"])
        except Exception as exc:
          print(f"upload failed for {path}, trying again: {exc!r}", file=sys.stderr, flush=True)

  @contextlib.contextmanager
  def hold(self, uri: str, version: str):
    """Fill a missing version under EX, then hold SH through the caller's read."""
    v, peer = split_version(version)
    folder = self.folder(version)
    with file_lock(folder) as lock:
      if not manifest(folder):
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not manifest(folder):
          self._fetch(uri, v, peer, folder)
        fcntl.flock(lock, fcntl.LOCK_SH)
      os.utime(folder)
      yield folder

  def _manifest(self, version: str, peer: str = "") -> dict:
    if peer:
      with urllib.request.urlopen(f"http://{peer}/{version}.json", timeout=10) as r:
        return json.load(r)
    return json.loads(self.fs.cat_file(f"{self.base}/{version}.json"))

  def _fetch(self, uri: str, version: str, peer: str, folder: str) -> None:
    if not self.remote:
      raise ValueError(f"version {version} of {uri} has not landed")
    for source in ([peer] if peer and peer != self.peer else []) + [""]:
      tmp = tempfile.mkdtemp(dir=self.cache, prefix=".get.")
      try:
        m = self._manifest(version, source)
        if m["version"] != version or not m["files"] or any(os.path.basename(n) != n for n in m["files"]):
          raise ValueError("invalid checkpoint manifest")
        parallel(
          lambda n, source=source, tmp=tmp: (
            fetch(f"http://{source}/{version}/{n}", f"{tmp}/{n}")
            if source
            else self.fs.get_file(f"{self.base}/{version}/{n}", f"{tmp}/{n}")
          ),
          list(m["files"]),
        )
        if any(os.path.getsize(f"{tmp}/{n}") != s for n, s in m["files"].items()):
          raise ValueError("incomplete checkpoint copy")
        shutil.rmtree(folder, ignore_errors=True)
        os.rename(tmp, folder)
        write_json(f"{folder}.json", m)
        if not source:
          open(f"{folder}.uploaded", "w").close()
        return
      except (OSError, ValueError, KeyError):
        if not source:
          raise ValueError(
            f"version {version} of {uri} has not landed" + (f", and {peer} failed to copy" if peer else "")
          ) from None
      finally:
        shutil.rmtree(tmp, ignore_errors=True)

  def local_folder(self, uri: str, version: str) -> tuple[str, str]:
    with self.hold(uri, version) as folder:
      return folder, split_version(version)[0]

  @span("store.get")
  def get(self, uri: str, version: str) -> tuple[dict[str, torch.Tensor], dict]:
    import safetensors

    def load(path):
      with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}

    with self.hold(uri, version) as folder:
      paths = sorted(glob.glob(f"{folder}/*.safetensors"))
      return {k: v for s in parallel(load, paths) for k, v in s.items()}, manifest(folder)["meta"]

  def read_meta(self, uri: str, version: str) -> dict:
    if m := manifest(self.folder(version)):
      return m["meta"]
    v, peer = split_version(version)
    if peer:
      with contextlib.suppress(OSError, ValueError):
        return self._manifest(v, peer)["meta"]
    return self._manifest(v)["meta"]

  def _evict(self, version: str, durable: bool = False) -> bool:
    folder = self.folder(version)
    try:
      with file_lock(folder, fcntl.LOCK_EX | fcntl.LOCK_NB):
        if durable and (not os.path.exists(f"{folder}.uploaded") or os.path.exists(f"{folder}.upload")):
          return False
        shutil.rmtree(folder, ignore_errors=True)
        for suffix in (".json", ".upload", ".uploaded"):
          with contextlib.suppress(FileNotFoundError):
            os.unlink(folder + suffix)
      return True
    except BlockingIOError:
      return False

  def delete_version(self, uri: str, version: str) -> bool:
    if not self._evict(version):
      return False
    if self.remote:
      v = split_version(version)[0]
      for path in (f"{self.base}/{v}.json", f"{self.base}/{v}"):
        with contextlib.suppress(FileNotFoundError):
          self.fs.rm(path, recursive=True)
    return True

  def trim_cache(self) -> None:
    if not self.remote or not os.path.isdir(self.cache):
      return
    limit = CACHE_BYTES or shutil.disk_usage(self.cache).total // 2
    entries = [(p[:-5], m) for p in glob.glob(f"{self.cache}/*.json") if (m := manifest(p[:-5]))]
    total = sum(sum(m["files"].values()) for _, m in entries)
    for _, m in sorted(entries, key=lambda e: os.stat(e[0]).st_mtime):
      if total <= limit:
        break
      if self._evict(m["version"], durable=True):
        total -= sum(m["files"].values())
