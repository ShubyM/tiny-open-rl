"""Checkpoint storage behind put(uri) and get(uri).

Tensors are stored in safetensors with JSON metadata in the header, never torch.save pickle. Sampler checkpoints write the layout PEFT and vLLM load directly (adapter_model.safetensors + adapter_config.json, or model.safetensors + config.json, sharded with model.safetensors.index.json when large). Each published copy carries a sidecar manifest beside its folder so engines see only model files, explicit versions resolve directly without _latest, and local or peer availability is distinct from confirmed durable upload.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import ctypes
import fcntl
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
import urllib.parse
import urllib.request
import uuid
import weakref
from typing import TYPE_CHECKING

import fsspec

if TYPE_CHECKING:
  import torch

from tinyrl.helpers import getenv
from tinyrl.trace import span

CKPT, CKPT_CACHE = getenv("CKPT", "/tmp/tinyrl"), getenv("CKPT_CACHE", "/tmp/tinyrl-cache")
SHARD_BYTES, MAX_SHARDS, STREAM_CHUNK = getenv("SHARD_BYTES", 256 * 2**20), 16, 256 * 2**20
COPY_CHUNK, CACHE_BYTES = getenv("COPY_CHUNK", 2**20), getenv("CKPT_CACHE_BYTES", 0)
REQ_TRANSFERS, PREFETCH_TRANSFERS, MANIFEST_FORMAT = getenv("REQ_TRANSFERS", 4), getenv("PREFETCH_TRANSFERS", 2), 1


class TransferCancelled(RuntimeError):
  pass


def split_version(ref: str | None) -> tuple[str | None, str]:
  """Separate a checkpoint version identity from an optional legacy version@peer location hint."""
  version, _, peer = (ref or "").partition("@")
  return version or None, peer


def peer_list(hint: str, peers: tuple[str, ...], skip: str = "") -> tuple[str, ...]:
  return tuple(p for p in dict.fromkeys([*([hint] if hint else []), *peers]) if p and p != skip)


def write_json(path: str, data: dict) -> None:
  tmp = f"{path}.{uuid.uuid4().hex}.tmp"
  try:
    with open(tmp, "w") as f:
      json.dump(data, f)
    os.replace(tmp, path)
  finally:
    with contextlib.suppress(OSError):
      os.unlink(tmp)


def read_json(path: str) -> dict | None:
  with contextlib.suppress(OSError, ValueError):
    with open(path) as f:
      return data if isinstance(data := json.load(f), dict) else None
  return None


def complete_copy(folder: str, manifest: dict | None) -> bool:
  files = manifest.get("files") if manifest and manifest.get("format") == MANIFEST_FORMAT else None
  if not isinstance(files, dict) or not files or not os.path.isdir(folder):
    return False
  with contextlib.suppress(OSError):
    return all(os.path.getsize(f"{folder}/{name}") == size for name, size in files.items())
  return False


def make_manifest(uri: str, version: str, folder: str, names: list[str], meta: dict) -> dict:
  files = {name: os.path.getsize(f"{folder}/{name}") for name in names}
  head = {"format": MANIFEST_FORMAT, "version": version, "uri": uri, "files": files}
  return head | {"size_bytes": sum(files.values()), "meta": meta, "created_at": time.time()}


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
  ptr, step = buf.data_ptr(), (alloc // min(32, max(1, os.cpu_count() or 1), max(1, alloc >> 22)) + 4095) & ~4095
  parallel(lambda a: ctypes.memset(ptr + a, 0, min(step, alloc - a)), list(range(0, alloc, step)))
  if cudart.cudaHostRegister(ptr, alloc, 0) != 0:
    m.close()
    return torch.empty(size, dtype=torch.uint8, pin_memory=True)
  weakref.finalize(buf.untyped_storage(), lambda mm, p: (cudart.cudaHostUnregister(p), mm.close()), m, ptr)
  return buf


def nbytes(t: torch.Tensor) -> int:
  return t.numel() * t.element_size()


def aligned(n: int) -> int:
  return (n + 7) & ~7


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


def copy_stream(src, path: str, expect: int | None = None, on_chunk=None) -> int:
  total = 0
  with open(path, "wb") as out:
    while True:
      if on_chunk:
        on_chunk(0)
      if not (buf := src.read(COPY_CHUNK)):
        break
      out.write(buf)
      total += len(buf)
      if on_chunk:
        on_chunk(len(buf))
  if expect is not None and total != expect:
    raise OSError(f"{path} got {total} bytes, expected {expect}")
  return total


def fetch(url: str, path: str, expect: int | None = None, on_chunk=None) -> int:
  with urllib.request.urlopen(url, timeout=10) as r:
    total = copy_stream(r, path, expect, on_chunk)
    if r.length:
      raise OSError(f"{url} ended {r.length} bytes short")
    return total


@contextlib.contextmanager
def shared_file_lock(lock_path: str):
  os.makedirs(os.path.dirname(lock_path), exist_ok=True)
  while True:
    with open(lock_path, "a+") as f:
      fcntl.flock(f.fileno(), fcntl.LOCK_SH)
      with contextlib.suppress(FileNotFoundError):
        if os.stat(lock_path).st_ino == os.fstat(f.fileno()).st_ino:
          yield
          return


def evict_dir(folder: str, manifest_path: str, lock_path: str, extras: tuple[str, ...] = (), guard=None) -> bool:
  """Remove one version copy only when no process holds its shared reader lock."""
  os.makedirs(os.path.dirname(lock_path), exist_ok=True)
  while True:
    moved = ""
    with open(lock_path, "a+") as f:
      try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
      except BlockingIOError:
        return False
      with contextlib.suppress(FileNotFoundError):
        if os.stat(lock_path).st_ino != os.fstat(f.fileno()).st_ino:
          continue
        if guard and not guard():
          return False
        for p in (manifest_path, *extras):
          with contextlib.suppress(FileNotFoundError):
            os.unlink(p)
        if os.path.isdir(folder):
          os.rename(folder, moved := f"{folder}.del.{uuid.uuid4().hex}")
        os.unlink(lock_path)
        break
  if moved:
    shutil.rmtree(moved, ignore_errors=True)
  return True


class Transfer:
  def __init__(self, version: str, required: bool):
    self.version, self.source, self.bytes_done, self.bytes_total = version, "", 0, 0
    self.waiters, self.required, self.error = 1, int(required), None
    self.done, self.lock = threading.Event(), threading.Lock()


class CacheFiles(http.server.BaseHTTPRequestHandler):
  """Serves only published version manifests and files, holding a shared reader lock across each transfer."""

  def __init__(self, *args, directory: str, **kwargs):
    self.directory = directory
    super().__init__(*args, **kwargs)

  def log_message(self, format: str, *args) -> None:
    pass

  def do_GET(self) -> None:
    parts = [p for p in urllib.parse.unquote(self.path.split("?", 1)[0]).split("/") if p and p not in (".", "..")]
    if len(parts) == 1 and parts[0].endswith(".manifest.json"):
      version, target = parts[0][:-14], None
    elif len(parts) == 2:
      version, target = parts
    else:
      return self.send_error(404)
    folder, mpath, lpath = (
      f"{self.directory}/{version}",
      f"{self.directory}/{version}.manifest.json",
      f"{self.directory}/.{version}.lock",
    )
    if version.startswith(".") or not os.path.exists(mpath):
      return self.send_error(404)
    with shared_file_lock(lpath):
      if not complete_copy(folder, m := read_json(mpath)) or (target is not None and target not in m["files"]):
        return self.send_error(404)
      src = mpath if target is None else f"{folder}/{target}"
      self.send_response(200)
      self.send_header("Content-Length", str(os.path.getsize(src)))
      self.end_headers()
      with open(src, "rb") as f, contextlib.suppress(BrokenPipeError, ConnectionResetError):
        shutil.copyfileobj(f, self.wfile, COPY_CHUNK)


class Store:
  def __init__(self, root: str | None = None, cache: str | None = None):
    self.root = (CKPT if root is None else root).rstrip("/")
    self.remote = "://" in self.root
    self.fs, self.base = fsspec.core.url_to_fs(self.root, auto_mkdir=True, use_listings_cache=False)
    self.base = self.base if self.remote else os.path.realpath(self.base)
    self.cache, self.peer = (CKPT_CACHE if cache is None else cache), os.environ.get("CKPT_PEER", "")
    self.buffers: dict[int, list[torch.Tensor]] = {}
    self.lock, self.uploading, self.tlock = threading.RLock(), threading.Lock(), threading.Lock()
    self.stopping, self.wake = threading.Event(), threading.Event()
    self.uploader = self.server = self.on_durable = None
    self.transfers: dict[str, Transfer] = {}
    self.pins: dict[str, int] = {}
    self.lanes = {
      "required": threading.BoundedSemaphore(max(1, REQ_TRANSFERS)),
      "prefetch": threading.BoundedSemaphore(max(1, PREFETCH_TRANSFERS)),
    }
    self.bandwidth = {"peer": 512.0 * 2**20, "bucket": 192.0 * 2**20}
    self.cancelled_bytes = self.transferred_bytes = 0

  def path(self, uri: str) -> str:
    return f"{self.base}/{uri.removeprefix('tinker://')}"

  def local_paths(self, uri: str, ver: str) -> tuple[str, str, str]:
    parent, base = os.path.split(self.path(uri))
    vroot = f"{parent}/.{base}@v"
    return f"{vroot}/{ver}", f"{vroot}/{ver}.manifest.json", f"{vroot}/.{ver}.lock"

  def cache_paths(self, ver: str) -> tuple[str, str, str, str, str]:
    c = self.cache
    return f"{c}/{ver}", f"{c}/{ver}.manifest.json", f"{c}/.{ver}.lock", f"{c}/{ver}.durable", f"{c}/{ver}.upload"

  def env(self) -> dict[str, str]:
    return {"CKPT": self.root, "CKPT_CACHE": self.cache, "CKPT_PEER": self.peer}

  def start(self) -> None:
    from tinyrl.storage.models import MODEL_CACHE

    for folder in [self.cache, MODEL_CACHE, *([] if self.remote else [self.base])]:
      os.makedirs(folder, exist_ok=True)
    if self.remote and self.server is None:
      self.server = http.server.ThreadingHTTPServer(("", 0), lambda *a: CacheFiles(*a, directory=self.cache))
      threading.Thread(target=self.server.serve_forever, daemon=True).start()
      self.peer = f"{socket.gethostbyname(socket.gethostname())}:{self.server.server_port}"
    if self.uploader is None:
      self.stopping.clear()
      self.uploader = threading.Thread(target=self.keep_uploading, daemon=True)
      self.uploader.start()

  def keep_uploading(self) -> None:
    while not self.stopping.is_set():
      self.wake.wait(0.25)
      self.wake.clear()
      if not self.stopping.is_set():
        with contextlib.suppress(Exception):
          self.upload_pending_saves()

  def close(self, flush: bool = True, timeout: float | None = None) -> bool:
    self.stopping.set()
    self.wake.set()
    if self.uploader:
      self.uploader.join(timeout=timeout)
    flushed = (
      self.upload_pending_saves(deadline=None if timeout is None else time.monotonic() + timeout) if flush else True
    )
    if self.server:
      self.server.shutdown()
      self.server.server_close()
      self.server = None
    return flushed

  # ***** local pins & inspection *****

  @contextlib.contextmanager
  def pin(self, version: str | None):
    ver, _ = split_version(version)
    if ver:
      with self.tlock:
        self.pins[ver] = self.pins.get(ver, 0) + 1
    try:
      yield ver
    finally:
      if ver:
        with self.tlock:
          if (left := self.pins.get(ver, 0) - 1) > 0:
            self.pins[ver] = left
          else:
            self.pins.pop(ver, None)

  def has_local(self, uri: str, version: str) -> bool:
    if not (ver := split_version(version)[0]):
      return False
    folder, mpath, *_ = self.cache_paths(ver) if self.remote else self.local_paths(uri, ver)
    return complete_copy(folder, read_json(mpath))

  def is_durable(self, uri: str, version: str) -> bool:
    """Whether `version` is confirmed committed in durable storage, distinct from local or peer availability."""
    if not (ver := split_version(version)[0]):
      return False
    if not self.remote:
      return self.has_local(uri, ver)
    folder, mpath, _, dpath, _ = self.cache_paths(ver)
    local_ok = complete_copy(folder, read_json(mpath))
    if os.path.exists(dpath) and local_ok:
      return True
    dest = self.path(uri)
    if self.fs.exists(f"{dest}/{ver}.json"):
      if local_ok:
        with contextlib.suppress(OSError):
          open(dpath, "w").close()
      return True
    info = self.latest(dest)
    return bool(info.get("version") == ver and not info.get("manifest"))

  def version_size(self, uri: str, version: str) -> int:
    if not (ver := split_version(version)[0]):
      return 0
    return int(
      (read_json(self.cache_paths(ver)[1] if self.remote else self.local_paths(uri, ver)[1]) or {}).get("size_bytes", 0)
    )

  def _cache_entries(self) -> list[tuple[str, int, float]]:
    if not self.remote or not os.path.isdir(self.cache):
      return []
    out = []
    for e in os.scandir(self.cache):
      if e.name.endswith(".manifest.json") and complete_copy(
        folder := f"{self.cache}/{e.name[:-14]}", m := read_json(e.path)
      ):
        with contextlib.suppress(OSError):
          out.append((e.name[:-14], int(m.get("size_bytes", 0)), os.stat(folder).st_mtime))
    return out

  def cached_versions(self) -> list[str]:
    return sorted(v for v, _, _ in self._cache_entries())

  def cache_status(self) -> dict:
    entries = self._cache_entries()
    limit = (
      (CACHE_BYTES or shutil.disk_usage(self.cache).total // 2) if self.remote and os.path.isdir(self.cache) else 0
    )
    with self.tlock:
      tx = {
        v: {"done": t.bytes_done, "total": t.bytes_total, "source": t.source, "required": bool(t.required)}
        for v, t in self.transfers.items()
      }
    head = {
      "peer": self.peer if self.remote else "",
      "versions": sorted(v for v, _, _ in entries),
      "used": sum(s for _, s, _ in entries),
      "limit": limit,
    }
    bw = {
      "peer_bps": self.bandwidth["peer"],
      "bucket_bps": self.bandwidth["bucket"],
      "transferred_bytes": self.transferred_bytes,
    }
    return head | bw | {"cancelled_bytes": self.cancelled_bytes, "transfers": tx}

  # ***** put *****

  def write_safetensors(self, i: int, path: str, tensors: dict[str, torch.Tensor], meta: dict) -> None:
    """safetensors.torch.save_file streaming GPU tensors through two pinned buffers without holding a full host copy."""
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
          flush(1)
          buf, j, plen = bufs[j % 2], j + 1, piece.stop - piece.start
          with torch.cuda.stream(stream):
            buf[:plen].copy_(src[piece], non_blocking=True)
            pending.append((ev := torch.cuda.Event(), plen, buf))
            ev.record(stream)
      flush(0)

  def write_checkpoint_files(self, folder: str, tensors: dict[str, torch.Tensor], info: dict) -> list[str]:
    lora, cfg, total = info.get("lora"), info.get("config"), sum(map(nbytes, tensors.values()))
    n = max(1, min(MAX_SHARDS, len(tensors))) if not lora and total >= SHARD_BYTES else 1
    stem, shards, sizes = (("adapter_model" if lora else "model") if cfg else "state"), [{} for _ in range(n)], [0] * n
    for k, v in sorted(tensors.items(), key=lambda kv: nbytes(kv[1]), reverse=True):
      i = sizes.index(min(sizes))
      shards[i][k], sizes[i] = v, sizes[i] + nbytes(v)
    shards = [s for s in shards if s] or [{}]
    n = len(shards)
    weights = [f"{stem}.safetensors"] if n == 1 else [f"{stem}-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]
    files = {("adapter_config.json" if lora else "config.json"): cfg} if cfg else {}
    if cfg and n > 1:
      files["model.safetensors.index.json"] = {
        "metadata": {"total_size": total},
        "weight_map": {k: w for w, s in zip(weights, shards) for k in s},
      }
    for name, obj in files.items():
      with open(f"{folder}/{name}", "w") as f:
        json.dump(obj, f)
    meta = {"meta": json.dumps(info)}
    parallel(
      lambda i, name, shard: self.write_safetensors(i, f"{folder}/{name}", shard, meta), range(n), weights, shards
    )
    return [*files, *weights]

  @span("store.put")
  def put(self, uri: str, tensors: dict[str, torch.Tensor], meta: dict | None = None, *, priority: bool = False) -> str:
    """Save an immutable checkpoint version, publishing its manifest only after every file is complete."""
    dest, version, info = self.path(uri), uuid.uuid4().hex[:16], dict(meta or {})
    if self.remote:
      os.makedirs(self.cache, exist_ok=True)
      tmp = tempfile.mkdtemp(dir=self.cache, prefix=".put.")
      folder, mpath, _, _, upath = self.cache_paths(version)
      try:
        with self.lock:
          manifest = make_manifest(uri, version, tmp, self.write_checkpoint_files(tmp, tensors, info), info)
        os.rename(tmp, folder)
        tmp = ""
        write_json(mpath, manifest)  # locally available; durable only after upload_version writes <version>.json
      finally:
        if tmp:
          shutil.rmtree(tmp, ignore_errors=True)
      self.record_latest_version(uri, version)
      write_json(upath, {"uri": uri, "priority": int(priority), "created": manifest["created_at"]})
      self.wake.set()
      return f"{version}@{self.peer}" if self.peer else version

    vdir, mpath, _ = self.local_paths(uri, version)
    os.makedirs(vroot := os.path.dirname(vdir), exist_ok=True)
    tmp = tempfile.mkdtemp(dir=vroot, prefix=".put.")
    link = f"{tmp}.link"
    try:
      with self.lock:
        manifest = make_manifest(uri, version, tmp, self.write_checkpoint_files(tmp, tensors, info), info)
        os.rename(tmp, vdir)
        tmp = ""
        write_json(mpath, manifest)
        os.symlink(f"{os.path.basename(vroot)}/{version}", link)
        if os.path.isdir(dest) and not os.path.islink(dest):
          os.rename(dest, old := f"{vroot}/.legacy.{uuid.uuid4().hex}")
          os.replace(link, dest)
          shutil.rmtree(old, ignore_errors=True)
        else:
          os.replace(link, dest)
      return version
    finally:
      if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
      if os.path.lexists(link):
        os.unlink(link)

  def record_latest_version(self, uri: str, version: str) -> None:
    os.makedirs(f"{self.cache}/latest", exist_ok=True)
    with open(tmp := f"{self.cache}/latest/.{uuid.uuid4().hex}", "w") as f:
      f.write(version)
    os.replace(tmp, f"{self.cache}/latest/{hashlib.sha256(uri.encode()).hexdigest()[:32]}")

  def latest_version_here(self, uri: str) -> str | None:
    with contextlib.suppress(OSError):
      with open(f"{self.cache}/latest/{hashlib.sha256(uri.encode()).hexdigest()[:32]}") as f:
        return f.read().strip() or None
    return None

  def upload_version(self, uri: str, version: str) -> bool:
    """Commit one local version to durable storage: files first, then <version>.json, then _latest, then .durable."""
    dest = self.path(uri)
    folder, mpath, lpath, dpath, _ = self.cache_paths(version)
    with shared_file_lock(lpath):
      if not complete_copy(folder, manifest := read_json(mpath)):
        return False
      rel, names = dest[len(self.base) + 1 :], list(manifest["files"])
      parallel(lambda n: self.fs.put_file(f"{folder}/{n}", f"{dest}/{version}/{n}"), names)
      self.fs.pipe_file(f"{dest}/{version}.json", json.dumps(manifest).encode())
      if manifest["created_at"] >= float(self.latest(dest).get("created_at", 0.0)):
        latest = {
          "version": version,
          "dir": f"{rel}/{version}",
          "files": names,
          "manifest": f"{rel}/{version}.json",
          "created_at": manifest["created_at"],
        }
        self.fs.pipe_file(f"{dest}/_latest", json.dumps(latest).encode())
      open(dpath, "w").close()
    if self.on_durable:
      with contextlib.suppress(Exception):
        self.on_durable(uri, version)
    return True

  def upload_pending_saves(self, priority_only: bool = False, deadline: float | None = None) -> bool:
    """Upload pending saves, priority recovery saves first so explicit commits never queue behind temporary exports."""
    if not self.remote:
      return True
    with self.uploading:
      entries = os.scandir(self.cache) if os.path.isdir(self.cache) else ()
      jobs = [
        (e.path, e.name[:-7], j)
        for e in entries
        if e.name.endswith(".upload") and (j := read_json(e.path)) and (not priority_only or j.get("priority"))
      ]
      all_ok = True
      for upath, version, job in sorted(
        jobs, key=lambda x: (-int(x[2].get("priority", 0)), float(x[2].get("created", 0)), x[1])
      ):
        if deadline is not None and time.monotonic() >= deadline:
          return False
        folder, mpath, *_ = self.cache_paths(version)
        if os.path.isdir(folder) and os.path.exists(mpath):
          try:
            self.upload_version(job["uri"], version)
          except Exception as exc:
            all_ok = False
            print(f"upload failed for {version}, trying again: {exc!r}", file=sys.stderr, flush=True)
            continue
        with contextlib.suppress(FileNotFoundError):
          os.unlink(upath)
      return all_ok

  # ***** get *****

  def latest(self, dest: str) -> dict:
    with contextlib.suppress(OSError, ValueError):
      if self.fs.exists(f"{dest}/_latest"):
        return json.loads(self.fs.cat_file(f"{dest}/_latest"))
    return {}

  def remote_manifest(self, dest: str, version: str) -> dict | None:
    with contextlib.suppress(OSError, ValueError):
      if self.fs.exists(p := f"{dest}/{version}.json"):
        data = json.loads(self.fs.cat_file(p))
        return (
          data
          if isinstance(data, dict) and data.get("format") == MANIFEST_FORMAT and data.get("version") == version
          else None
        )
    return None

  def _resolve_local_store(self, uri: str, version: str | None) -> tuple[str, str]:
    dest = self.path(uri)
    if version:
      vdir, mpath, _ = self.local_paths(uri, version)
      if complete_copy(vdir, read_json(mpath)):
        return vdir, version
      with contextlib.suppress(ValueError):
        ldir, lver = self._legacy_local(uri, dest)
        if lver == version:
          return ldir, lver
      raise ValueError(f"version {version} of {uri} not found")
    if os.path.islink(dest):
      vdir, mpath, _ = self.local_paths(uri, ver := os.path.basename(os.readlink(dest)))
      if complete_copy(vdir, read_json(mpath)):
        return vdir, ver
    return self._legacy_local(uri, dest)

  def _legacy_local(self, uri: str, dest: str) -> tuple[str, str]:
    real = os.path.join(os.path.dirname(dest), os.readlink(dest)) if os.path.islink(dest) else dest
    if not os.path.isdir(real):
      raise ValueError(f"no checkpoint at {uri}")
    with os.scandir(real) as it:
      stats = sorted((e.name, s.st_size, s.st_mtime_ns) for e in it if e.is_file() for s in [e.stat()])
    if not any(n.endswith(".safetensors") for n, _, _ in stats):
      raise ValueError(f"no checkpoint at {uri}")
    return real, hashlib.blake2b(repr((real, stats)).encode(), digest_size=8).hexdigest()

  def _should_abort(self, tr: Transfer, cancel: threading.Event | None) -> bool:
    with tr.lock:
      return False if tr.required > 0 else (tr.waiters <= 0 or bool(cancel and cancel.is_set() and tr.waiters <= 1))

  def _on_chunk(self, tr: Transfer, cancel: threading.Event | None, n: int) -> None:
    if self._should_abort(tr, cancel):
      raise TransferCancelled(f"transfer of {tr.version} cancelled")
    if n:
      with tr.lock:
        tr.bytes_done += n

  def _stage_copy(self, version: str, source: str, total: int, durable: bool, tr: Transfer, build) -> str:
    with tr.lock:
      tr.source, tr.bytes_done, tr.bytes_total = source, 0, total
    os.makedirs(self.cache, exist_ok=True)
    tmp, t0 = tempfile.mkdtemp(dir=self.cache, prefix=".get."), time.monotonic()
    try:
      manifest = build(tmp)
      folder, mpath, lpath, dpath, _ = self.cache_paths(version)
      with shared_file_lock(lpath):
        if not complete_copy(folder, read_json(mpath)):
          shutil.rmtree(folder, ignore_errors=True)
          os.rename(tmp, folder)
          tmp = ""
          write_json(mpath, manifest)
        if durable and not os.path.exists(dpath):
          open(dpath, "w").close()
      with contextlib.suppress(OSError):
        os.utime(folder)
      size, dt = int(manifest.get("size_bytes", 0)), time.monotonic() - t0
      self.transferred_bytes += size
      if source in self.bandwidth and size >= 64 * 1024 and dt > 1e-4:
        self.bandwidth[source] = 0.7 * self.bandwidth[source] + 0.3 * (size / dt)
      return folder
    finally:
      if tmp:
        shutil.rmtree(tmp, ignore_errors=True)

  def _peer_manifest(self, peer: str, version: str) -> dict | None:
    with urllib.request.urlopen(f"http://{peer}/{version}.manifest.json", timeout=5) as r:
      m = json.loads(r.read())
    return m if isinstance(m, dict) and m.get("format") == MANIFEST_FORMAT and m.get("version") == version else None

  def _copy_from_peer(self, peer: str, version: str, tr: Transfer, cancel: threading.Event | None) -> str | None:
    try:
      if not (manifest := self._peer_manifest(peer, version)):
        return None

      def build(tmp: str) -> dict:
        for name, size in manifest["files"].items():
          fetch(
            f"http://{peer}/{version}/{urllib.parse.quote(name)}",
            f"{tmp}/{name}",
            size,
            lambda n: self._on_chunk(tr, cancel, n),
          )
        return manifest

      return self._stage_copy(version, "peer", int(manifest["size_bytes"]), False, tr, build)
    except TransferCancelled:
      raise
    except Exception:
      return None

  def _copy_remote_files(
    self, uri: str, src_dir: str, files: dict[str, int | None], tmp: str, tr: Transfer, cancel: threading.Event | None
  ) -> None:
    if not any(n.endswith(".safetensors") for n in files):
      raise ValueError(f"no checkpoint at {uri}")
    for name, size in files.items():
      with self.fs.open(f"{src_dir}/{name}", "rb") as src:
        copy_stream(src, f"{tmp}/{name}", size, lambda n: self._on_chunk(tr, cancel, n))

  def ensure_version(
    self,
    uri: str,
    version: str,
    peers: tuple[str, ...] = (),
    cancel: threading.Event | None = None,
    lane: str = "required",
    legacy: dict | None = None,
  ) -> tuple[str, str]:
    """Fetch an explicit version once (single-flight), preferring local cache, then peers, then durable storage."""
    folder, mpath, *_ = self.cache_paths(version)
    if cancel and cancel.is_set() and not complete_copy(folder, read_json(mpath)):
      raise TransferCancelled(f"transfer of {version} cancelled")
    required = lane != "prefetch"
    with self.tlock:
      if complete_copy(folder, read_json(mpath)):
        with contextlib.suppress(OSError):
          os.utime(folder)
        return folder, version
      if leader := version not in self.transfers:
        tr = self.transfers[version] = Transfer(version, required)
      else:
        tr = self.transfers[version]
        with tr.lock:
          tr.waiters += 1
          tr.required += int(required)

    if not leader:
      try:
        while not tr.done.wait(0.02):
          if cancel and cancel.is_set():
            raise TransferCancelled(f"transfer of {version} cancelled")
      finally:
        with tr.lock:
          tr.waiters -= 1
          tr.required -= int(required)
      if complete_copy(folder, read_json(mpath)):
        return folder, version
      if tr.error and not isinstance(tr.error, TransferCancelled):
        raise tr.error
      return self.ensure_version(uri, version, peers, cancel, lane, legacy)

    sem, acquired = self.lanes["prefetch" if lane == "prefetch" else "required"], False
    dest, sources = self.path(uri), peer_list("", peers, self.peer)
    try:
      while not (acquired := sem.acquire(timeout=0.02)):
        if self._should_abort(tr, cancel):
          raise TransferCancelled(f"transfer of {version} cancelled")
      if complete_copy(folder, read_json(mpath)):
        return folder, version
      for peer in sources:
        if got := self._copy_from_peer(peer, version, tr, cancel):
          return got, version
      if m := self.remote_manifest(dest, version):
        build = lambda tmp: (self._copy_remote_files(uri, f"{dest}/{version}", m["files"], tmp, tr, cancel), m)[1]  # noqa: E731
        return self._stage_copy(version, "bucket", int(m["size_bytes"]), True, tr, build), version
      info = legacy if legacy is not None else self.latest(dest)
      if info.get("version") == version and not info.get("manifest"):
        names, src_dir = info.get("files") or [], (f"{self.base}/{info['dir']}" if info.get("dir") else dest)

        def build_legacy(tmp: str) -> dict:
          self._copy_remote_files(uri, src_dir, dict.fromkeys(names), tmp, tr, cancel)
          return make_manifest(
            uri, version, tmp, names, header(next(f"{tmp}/{n}" for n in names if n.endswith(".safetensors")))
          )

        return self._stage_copy(version, "bucket", 0, True, tr, build_legacy), version
      raise ValueError(
        f"version {version} of {uri} has not landed" + (f", and {sources[0]} failed to copy" if sources else "")
      )
    except TransferCancelled as e:
      self.cancelled_bytes += tr.bytes_done
      tr.error = e
      raise
    except BaseException as e:
      tr.error = e
      raise
    finally:
      if acquired:
        sem.release()
      with self.tlock:
        self.transfers.pop(version, None)
      with tr.lock:
        tr.waiters -= 1
        tr.required -= int(required)
      tr.done.set()

  def discover_version(self, uri: str) -> tuple[str, dict | None]:
    """Use _latest (or local latest) only when a caller asked for no specific version."""
    if not self.remote:
      return self._resolve_local_store(uri, None)[1], None
    if (local := self.latest_version_here(uri)) and self.has_local(uri, local):
      return local, None
    if not (info := self.latest(self.path(uri))) or not info.get("version"):
      raise ValueError(f"no checkpoint at {uri}")
    return info["version"], info

  def local_folder(
    self,
    uri: str,
    version: str | None = None,
    peers: tuple[str, ...] = (),
    cancel: threading.Event | None = None,
    lane: str = "required",
  ) -> tuple[str, str]:
    ver, hint = split_version(version)
    if not self.remote:
      return self._resolve_local_store(uri, ver)
    ver, legacy = self.discover_version(uri) if ver is None else (ver, None)
    return self.ensure_version(uri, ver, peer_list(hint, peers), cancel=cancel, lane=lane, legacy=legacy)

  @contextlib.contextmanager
  def hold(
    self,
    uri: str,
    version: str | None = None,
    peers: tuple[str, ...] = (),
    cancel: threading.Event | None = None,
    lane: str = "required",
  ):
    """Resolve a version once, then hold a cross-process shared lock across reading or engine loading."""
    ver, hint = split_version(version)
    sources, (ver, legacy) = peer_list(hint, peers), (self.discover_version(uri) if ver is None else (ver, None))
    while True:
      folder, ver = (
        self.ensure_version(uri, ver, sources, cancel, lane, legacy)
        if self.remote
        else self._resolve_local_store(uri, ver)
      )
      legacy = None  # retry the chosen version, never re-read _latest
      vdir, mpath, lpath, *_ = self.cache_paths(ver) if self.remote else self.local_paths(uri, ver)
      if not self.remote and folder != vdir:
        yield folder, ver
        return
      with shared_file_lock(lpath):
        if complete_copy(folder, read_json(mpath)):
          with contextlib.suppress(OSError):
            os.utime(folder)
          yield folder, ver
          return

  @span("store.get")
  def get(
    self, uri: str, version: str | None = None, peers: tuple[str, ...] = ()
  ) -> tuple[dict[str, torch.Tensor], dict]:
    import safetensors

    def load(path: str) -> dict[str, torch.Tensor]:
      with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}

    with self.hold(uri, version, peers) as (folder, ver):
      m = read_json(self.cache_paths(ver)[1] if self.remote else self.local_paths(uri, ver)[1])
      paths = sorted(glob.glob(f"{folder}/*.safetensors"))
      return {k: v for s in parallel(load, paths) for k, v in s.items()}, (
        m["meta"] if m and "meta" in m else header(paths[0])
      )

  def read_meta(self, uri: str, version: str | None = None, peers: tuple[str, ...] = ()) -> dict:
    """Read a version's metadata from its manifest without fetching shards; fall back to shard header for legacy copies."""
    ver, hint = split_version(version)
    sources = peer_list(hint, peers, self.peer)
    if not self.remote:
      folder, ver = self._resolve_local_store(uri, ver)
      m = read_json(self.local_paths(uri, ver)[1])
      return dict(m["meta"]) if m and "meta" in m else header(sorted(glob.glob(f"{folder}/*.safetensors"))[0])
    ver, legacy = self.discover_version(uri) if ver is None else (ver, None)
    if (m := read_json(self.cache_paths(ver)[1])) and "meta" in m:
      return dict(m["meta"])
    for peer in sources:
      with contextlib.suppress(Exception):
        if m := self._peer_manifest(peer, ver):
          return dict(m.get("meta", {}))
    if m := self.remote_manifest(self.path(uri), ver):
      return dict(m.get("meta", {}))
    folder, _ = self.ensure_version(uri, ver, sources, legacy=legacy)
    return header(sorted(glob.glob(f"{folder}/*.safetensors"))[0])

  # ***** delete & eviction *****

  def evict_local(self, version: str, *, require_durable: bool = True) -> bool:
    """Drop a cached version only if unlocked, unpinned, and (unless deleting globally) confirmed durable."""
    if not (ver := split_version(version)[0]) or not self.remote:
      return False
    folder, mpath, lpath, dpath, upath = self.cache_paths(ver)
    if not any(os.path.exists(p) for p in (folder, mpath, upath)):
      return True

    def guard() -> bool:
      with self.tlock:
        busy = ver in self.pins or ver in self.transfers
      return not busy and not (require_durable and (os.path.exists(upath) or not os.path.exists(dpath)))

    return evict_dir(folder, mpath, lpath, (dpath,) if require_durable else (dpath, upath), guard=guard)

  def delete_version(self, uri: str, version: str) -> bool:
    """Delete one claimed-unreferenced version without touching sibling versions or active locked readers."""
    if not (ver := split_version(version)[0]):
      return True
    dest = self.path(uri)
    if self.remote:
      if not self.evict_local(ver, require_durable=False):
        return False
      with contextlib.suppress(OSError):
        for p in (f"{dest}/{ver}.json", f"{dest}/{ver}"):
          if self.fs.exists(p):
            self.fs.rm(p, recursive=True)
        if self.latest(dest).get("version") == ver:
          self.fs.rm(f"{dest}/_latest")
        if self.fs.exists(dest) and not self.fs.ls(dest, detail=False):
          self.fs.rm(dest, recursive=True)
      return True
    vdir, mpath, lpath = self.local_paths(uri, ver)
    if (os.path.isdir(vdir) or os.path.exists(mpath)) and not evict_dir(
      vdir, mpath, lpath, guard=lambda: ver not in self.pins
    ):
      return False
    if os.path.islink(dest) and os.path.basename(os.readlink(dest)) == ver:
      with contextlib.suppress(FileNotFoundError):
        os.unlink(dest)
    with contextlib.suppress(OSError):
      os.rmdir(os.path.dirname(vdir))
    return True

  def delete(self, uri: str) -> None:
    dest = self.path(uri)
    if self.remote:
      if self.fs.exists(dest):
        self.fs.rm(dest, recursive=True)
      return
    parent, base = os.path.split(dest)
    if os.path.isdir(vroot := f"{parent}/.{base}@v"):
      for e in list(os.scandir(vroot)):
        if e.name.endswith(".manifest.json"):
          self.delete_version(uri, e.name[:-14])
      if os.path.isdir(vroot) and not any(n.endswith((".manifest.json", ".lock")) for n in os.listdir(vroot)):
        shutil.rmtree(vroot, ignore_errors=True)
    tail = len(parent) + len(base) + 3
    legacy = [
      p
      for p in glob.glob(f"{glob.escape(parent)}/.{glob.escape(base)}.*")
      if "." not in p[tail:] and not p.endswith("@v")
    ]
    for p in (dest, *legacy):
      with contextlib.suppress(FileNotFoundError):
        os.unlink(p) if os.path.islink(p) else shutil.rmtree(p, ignore_errors=True)

  def list_checkpoints(self) -> list[str]:
    if self.remote:
      return sorted(f"tinker://{p[len(self.base) + 1 :]}" for p in self.fs.glob(f"{self.base}/*/*/*"))
    return sorted(f"tinker://{os.path.relpath(p, self.base)}" for p in glob.glob(f"{glob.escape(self.base)}/*/*/*"))

  def saved_at(self, uri: str) -> float:
    if self.remote and self.fs.exists(latest := f"{self.path(uri)}/_latest"):
      return self.fs.modified(latest).timestamp()
    if self.remote:
      return max((self.fs.modified(f).timestamp() for f in self.fs.find(self.path(uri))), default=0.0)
    return os.lstat(self.path(uri)).st_mtime

  def sweep(
    self,
    registered: set[str],
    since: float | None,
    grace: float,
    pinned_cache: set[str] | None = None,
    live_versions: set[str] | None = None,
  ) -> list[str]:
    live, evicted = live_versions or set(), self.trim_cache(pinned_cache)
    if since is None:
      return evicted
    before = time.time() - grace
    for uri in set(self.list_checkpoints()) - registered:
      try:
        if since < self.saved_at(uri) < before:
          self.delete(uri)
      except Exception as e:
        print(f"sweep could not delete {uri}, trying again next sweep: {e!r}", file=sys.stderr, flush=True)
    for e in list(os.scandir(self.cache)) if self.remote and os.path.isdir(self.cache) else ():
      if e.name.endswith(".upload") and (ver := e.name[:-7]) not in live:
        with contextlib.suppress(OSError, ValueError):
          if (
            since < e.stat().st_mtime < before
            and (job := read_json(e.path))
            and job["uri"] not in registered
            and self.evict_local(ver, require_durable=False)
          ):
            evicted.append(ver)
    return evicted

  def trim_cache(self, protected: set[str] | None = None, limit: int | None = None) -> list[str]:
    """Keep CKPT_CACHE under CKPT_CACHE_BYTES by evicting oldest durable, unreferenced, unlocked local copies."""
    if not self.remote or not os.path.isdir(self.cache):
      return []
    keep, entries = protected or set(), self._cache_entries()
    cap = (CACHE_BYTES or shutil.disk_usage(self.cache).total // 2) if limit is None else limit
    total, evicted = sum(s for _, s, _ in entries), []
    for ver, size, _ in sorted(entries, key=lambda x: (x[2], x[0])):
      if total <= cap:
        break
      if (
        ver not in keep
        and os.path.exists(f"{self.cache}/{ver}.durable")
        and not os.path.exists(f"{self.cache}/{ver}.upload")
        and self.evict_local(ver)
      ):
        total -= size
        evicted.append(ver)
    return evicted
