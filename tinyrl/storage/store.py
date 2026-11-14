"""Checkpoint storage behind put(uri) and get(uri).

Each saved version is an immutable directory containing safetensors/config files and a manifest.json written last.
Tensors are stored in safetensors with JSON metadata in the header, never torch.save pickle. Sampler checkpoints write
the layout PEFT and vLLM load directly.

CKPT is a local directory or a URL like gs://bucket/prefix that every node can reach. With a URL, the node stages
saves into CKPT_CACHE/<version>/ and a background thread uploads them to <uri>/<version>/ (writing manifest.json last),
updating <uri>/_latest monotonically. Complete local/peer copies (manifest.json) are distinct from confirmed durable
uploads (.durable + remote manifest.json). Cross-process flock pins protect active readers from eviction and deletion.
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
import urllib.request
import uuid
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

import fsspec

if TYPE_CHECKING:
  import torch

from tinyrl.helpers import getenv
from tinyrl.trace import span

CKPT, CKPT_CACHE = getenv("CKPT", "/tmp/tinyrl"), getenv("CKPT_CACHE", "/tmp/tinyrl-cache")
SHARD_BYTES, MAX_SHARDS, STREAM_CHUNK = getenv("SHARD_BYTES", 256 * 1024 * 1024), 16, 256 * 1024 * 1024
CACHE_BYTES, MANIFEST, FORMAT_VERSION = getenv("CKPT_CACHE_BYTES", 0), "manifest.json", 1


def parse_version(ref: str | None) -> tuple[str, str]:
  """Split a version reference (optionally legacy 'version@peer') into (version, peer)."""
  if not ref:
    return "", ""
  ver, _, peer = ref.partition("@")
  return ver, peer


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


def lock_file(folder: str) -> str:
  return f"{folder.rstrip('/')}.lock"


@contextlib.contextmanager
def shared_lock(folder: str):
  """Hold a cross-process shared lock beside a version folder while reading or serving it."""
  if parent := os.path.dirname(folder.rstrip("/")):
    os.makedirs(parent, exist_ok=True)
  f = open(lock_file(folder), "a+")
  try:
    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
    yield f
  finally:
    with contextlib.suppress(OSError):
      fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    f.close()


def try_exclusive_lock(folder: str):
  """Try a non-blocking cross-process exclusive lock before evicting or deleting a version folder."""
  if not os.path.isdir(os.path.dirname(folder.rstrip("/"))):
    return None
  f = open(lock_file(folder), "a+")
  try:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return f
  except BlockingIOError:
    f.close()
    return None


def release_lock(f) -> None:
  if f is None:
    return
  try:
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
  finally:
    f.close()


def _valid_manifest(data) -> dict | None:
  return (
    data
    if isinstance(data, dict) and data.get("format") == FORMAT_VERSION and isinstance(data.get("files"), dict)
    else None
  )


def read_manifest(folder: str) -> dict | None:
  path = os.path.join(folder, MANIFEST)
  if not os.path.isfile(path):
    return None
  with contextlib.suppress(OSError, ValueError), open(path) as f:
    return _valid_manifest(json.load(f))
  return None


def verify_manifest(folder: str, manifest: dict) -> bool:
  return all(
    os.path.isfile(p := os.path.join(folder, n)) and os.path.getsize(p) == sz
    for n, sz in manifest.get("files", {}).items()
  )


def _is_ready_folder(folder: str) -> bool:
  return (m := read_manifest(folder)) is not None and verify_manifest(folder, m)


def fetch(
  url: str, path: str, on_bytes: Callable[[int], None] | None = None, cancel: threading.Event | None = None
) -> int:
  written = 0
  with urllib.request.urlopen(url, timeout=10) as r, open(path, "wb") as f:
    while chunk := r.read(1024 * 1024):
      if cancel and cancel.is_set():
        raise InterruptedError(f"cancelled fetch {url}")
      f.write(chunk)
      written += len(chunk)
      if on_bytes:
        on_bytes(len(chunk))
    if r.length:
      raise OSError(f"{url} ended {r.length} bytes short")
  return written


class CacheFiles(http.server.SimpleHTTPRequestHandler):
  """Serves only complete manifest-published versions from a node's cache, holding a shared flock while serving."""

  def _version_folder(self) -> str | None:
    parts = [p for p in self.path.split("?", 1)[0].split("/") if p]
    return os.path.join(self.directory, parts[0]) if parts and parts[0] not in ("..", ".", "latest") else None

  def do_GET(self) -> None:
    if not (folder := self._version_folder()):
      self.send_error(404)
      return
    with shared_lock(folder):
      if read_manifest(folder) is None:
        self.send_error(404)
        return
      super().do_GET()

  def list_directory(self, path: str) -> None:
    if (manifest := read_manifest(path)) is None:
      self.send_error(404)
      return
    body = json.dumps(sorted([*manifest["files"], MANIFEST])).encode()
    self.send_response(200)
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def log_message(self, format: str, *args) -> None:
    pass


class Store:
  def __init__(
    self, root: str | None = None, cache: str | None = None, on_durable: Callable[[str, str], None] | None = None
  ):
    self.root = (CKPT if root is None else root).rstrip("/")
    self.remote = "://" in self.root
    self.fs, self.base = fsspec.core.url_to_fs(self.root, auto_mkdir=True, use_listings_cache=False)
    if not self.remote:
      self.base = os.path.realpath(self.base)
    self.cache, self.on_durable = CKPT_CACHE if cache is None else cache, on_durable
    self.buffers: dict[int, list[torch.Tensor]] = {}
    self.lock, self.uploading, self.stopping = threading.RLock(), threading.Lock(), threading.Event()
    self.uploader, self.server = None, None
    self._upload_wake, self._durable_cond, self._transfer_lock = (
      threading.Event(),
      threading.Condition(),
      threading.Lock(),
    )
    self.transfers: dict[str, dict] = {}
    self.peer = os.environ.get("CKPT_PEER", "")

  def path(self, uri: str) -> str:
    return f"{self.base}/{uri.removeprefix('tinker://')}"

  def env(self) -> dict[str, str]:
    """What a worker process on this node needs to open the same store."""
    return {"CKPT": self.root, "CKPT_CACHE": self.cache, "CKPT_PEER": self.peer}

  def start(self) -> None:
    """Make this node's folders, serve its cache to other nodes, and upload its saves behind from now on."""
    for folder in [self.cache] + ([] if self.remote else [self.base]):
      os.makedirs(folder, exist_ok=True)
    if self.remote:
      self.server = http.server.ThreadingHTTPServer(("", 0), lambda *a: CacheFiles(*a, directory=self.cache))
      threading.Thread(target=self.server.serve_forever, daemon=True).start()
      self.peer = f"{socket.gethostbyname(socket.gethostname())}:{self.server.server_port}"
    self.uploader = threading.Thread(target=self.keep_uploading, daemon=True)
    self.uploader.start()

  def keep_uploading(self) -> None:
    while not self.stopping.is_set():
      self._upload_wake.wait(0.25)
      self._upload_wake.clear()
      if not self.stopping.is_set():
        with contextlib.suppress(Exception):
          self.upload_pending_saves()

  def close(self) -> None:
    """Return once every save this node made has uploaded."""
    self.stopping.set()
    self._upload_wake.set()
    if self.uploader:
      self.uploader.join()
    self.upload_pending_saves()
    with self._durable_cond:
      self._durable_cond.notify_all()
    if self.server:
      self.server.shutdown()
      self.server.server_close()

  # ***** put *****

  def write_safetensors(self, i: int, path: str, tensors: dict[str, torch.Tensor], meta: dict) -> None:
    """safetensors.torch.save_file streaming GPU tensors through two pinned buffers."""
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

  def _publish_local_latest(self, dest: str, version: str, names: list[str], created: float) -> None:
    rel, latest_path, cur = dest[len(self.base) + 1 :], f"{dest}/_latest", {}
    if os.path.isfile(latest_path):
      with contextlib.suppress(OSError, ValueError), open(latest_path) as f:
        cur = json.load(f)
    if created < cur.get("created", 0.0):
      return
    tmp = f"{dest}/._latest.{uuid.uuid4().hex}"
    with open(tmp, "w") as f:
      json.dump(
        {"format": FORMAT_VERSION, "version": version, "dir": f"{rel}/{version}", "files": names, "created": created}, f
      )
    os.replace(tmp, latest_path)

  def _stage_version(
    self, uri: str, writer: Callable[[str], list[str]], meta: dict, priority: int = 1
  ) -> tuple[str, dict]:
    dest, version, now = self.path(uri), uuid.uuid4().hex[:16], time.time()
    if not self.remote and os.path.islink(dest):  # migrate legacy symlink in place cleanly
      legacy_target = os.path.realpath(dest)
      os.unlink(dest)
      os.makedirs(dest, exist_ok=True)
      if os.path.isdir(legacy_target):
        shutil.move(legacy_target, f"{dest}/legacy-{uuid.uuid4().hex[:8]}")
    parent = self.cache if self.remote else dest
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=parent, prefix=f".{version}.")
    try:
      names = writer(tmp)
      manifest = {
        "format": FORMAT_VERSION,
        "version": version,
        "uri": uri,
        "created": now,
        "files": {n: os.path.getsize(f"{tmp}/{n}") for n in names},
        "meta": meta,
      }
      with open(f"{tmp}/{MANIFEST}", "w") as f:
        json.dump(manifest, f)
      os.rename(tmp, f"{parent}/{version}")
      tmp = ""
    finally:
      if tmp:
        shutil.rmtree(tmp, ignore_errors=True)

    if self.remote:
      self.record_latest_version(uri, version)
      with open(f"{self.cache}/{version}.upload.tmp", "w") as f:
        json.dump({"uri": uri, "version": version, "names": names, "priority": priority, "created": now}, f)
      os.replace(f"{self.cache}/{version}.upload.tmp", f"{self.cache}/{version}.upload")
      self._upload_wake.set()
      return (f"{version}@{self.peer}" if self.peer else version), manifest
    self._publish_local_latest(dest, version, names, now)
    if self.on_durable:
      with contextlib.suppress(Exception):
        self.on_durable(uri, version)
    return version, manifest

  @span("store.put")
  def put(self, uri: str, tensors: dict[str, torch.Tensor], meta: dict | None = None, priority: int = 1) -> str:
    """Save a checkpoint as an immutable version with manifest.json, returning its version reference."""
    info = dict(meta or {})

    def write(folder: str) -> list[str]:
      with self.lock:
        return self.write_checkpoint_files(folder, tensors, info)

    return self._stage_version(uri, write, info, priority=priority)[0]

  def put_files(
    self, uri: str, files: dict[str, bytes] | str, meta: dict | None = None, priority: int = 1
  ) -> tuple[str, int]:
    """Publish arbitrary immutable files (e.g., compiled artifacts or mirrored bases) as a versioned manifest folder."""

    def write(folder: str) -> list[str]:
      names = []
      items = (
        [
          (os.path.relpath(os.path.join(r, fn), files), os.path.join(r, fn))
          for r, _, fns in os.walk(files)
          for fn in sorted(fns)
        ]
        if isinstance(files, str)
        else sorted(files.items())
      )
      for rel, payload in items:
        dst = os.path.join(folder, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if isinstance(files, str):
          shutil.copy2(payload, dst)
        else:
          with open(dst, "wb") as f:
            f.write(payload)
        names.append(rel)
      return names

    ref, manifest = self._stage_version(uri, write, dict(meta or {}), priority=priority)
    return ref, sum(manifest["files"].values())

  def record_latest_version(self, uri: str, version: str) -> None:
    """Record on this node which version of a checkpoint it saved last, for discovery when no version is given."""
    os.makedirs(f"{self.cache}/latest", exist_ok=True)
    tmp = f"{self.cache}/latest/.{uuid.uuid4().hex}"
    with open(tmp, "w") as f:
      f.write(version)
    os.replace(tmp, f"{self.cache}/latest/{hashlib.sha256(uri.encode()).hexdigest()[:32]}")

  def latest_version_here(self, uri: str) -> str | None:
    pointer = f"{self.cache}/latest/{hashlib.sha256(uri.encode()).hexdigest()[:32]}"
    if not os.path.exists(pointer):
      return None
    with open(pointer) as f:
      return f.read().strip() or None

  def upload_version(self, uri: str, dest: str, version: str, names: list[str], created: float = 0.0) -> None:
    folder, rel = f"{self.cache}/{version}", dest[len(self.base) + 1 :]
    with shared_lock(folder):
      if (manifest := read_manifest(folder)) is None:
        raise FileNotFoundError(f"incomplete local version {version}")
      remote_dir = f"{self.base}/{rel}/{version}"
      if files := [f"{folder}/{n}" for n in names]:
        parallel(lambda f: self.fs.put_file(f, f"{remote_dir}/{os.path.relpath(f, folder)}"), files)
      self.fs.put_file(f"{folder}/{MANIFEST}", f"{remote_dir}/{MANIFEST}")
      ts = created or manifest.get("created", time.time())
      if ts >= self.latest(dest).get("created", 0.0):
        info = {"format": FORMAT_VERSION, "version": version, "dir": f"{rel}/{version}", "files": names, "created": ts}
        self.fs.pipe_file(f"{dest}/_latest", json.dumps(info).encode())
      open(f"{folder}.durable", "w").close()
    if self.on_durable:
      with contextlib.suppress(Exception):
        self.on_durable(uri, version)

  def upload_pending_saves(self) -> None:
    """Upload this node's saves ordered by priority then creation time. Never deletes older versions automatically."""
    with self.uploading:
      found = [e for e in os.scandir(self.cache) if e.name.endswith(".upload")] if os.path.isdir(self.cache) else []
      jobs = []
      for e in found:
        with contextlib.suppress(OSError, ValueError), open(e.path) as f:
          job = json.load(f)
          jobs.append((job.get("priority", 1), job.get("created", e.stat().st_mtime), e.name, e.path, job))
      failed: set[str] = set()
      for _, created, name, path, job in sorted(jobs):
        version = name.removesuffix(".upload")
        if job["uri"] in failed:
          continue
        if os.path.isdir(f"{self.cache}/{version}"):
          try:
            self.upload_version(job["uri"], self.path(job["uri"]), version, job["names"], created)
          except Exception as exc:
            failed.add(job["uri"])
            print(f"upload failed for {version}, trying again: {exc!r}", file=sys.stderr, flush=True)
            continue
        with contextlib.suppress(FileNotFoundError):
          os.unlink(path)
      with self._durable_cond:
        self._durable_cond.notify_all()

  def is_durable(self, uri: str, version: str) -> bool:
    if not (ver := parse_version(version)[0]):
      return False
    if not self.remote:
      return read_manifest(f"{self.path(uri)}/{ver}") is not None
    if os.path.exists(f"{self.cache}/{ver}.durable"):
      return True
    if self.fs.exists(f"{self.path(uri)}/{ver}/{MANIFEST}"):
      if os.path.isdir(f"{self.cache}/{ver}"):
        with contextlib.suppress(OSError):
          open(f"{self.cache}/{ver}.durable", "w").close()
      return True
    return False

  def wait_durable(self, uri: str, version: str, timeout: float | None = None) -> bool:
    """Wait until an immutable version is durably committed in storage."""
    ver, _ = parse_version(version)
    if self.is_durable(uri, ver):
      return True
    if not self.remote:
      return False
    deadline = None if timeout is None else time.monotonic() + timeout
    while not self.is_durable(uri, ver):
      if self.uploader is None or not self.uploader.is_alive():
        self.upload_pending_saves()
        if self.is_durable(uri, ver):
          return True
        if not os.path.exists(f"{self.cache}/{ver}.upload"):
          return False
      self._upload_wake.set()
      with self._durable_cond:
        if self.is_durable(uri, ver):
          return True
        if self.stopping.is_set() and not os.path.exists(f"{self.cache}/{ver}.upload"):
          return False
        rem = None if deadline is None else deadline - time.monotonic()
        if rem is not None and rem <= 0:
          raise TimeoutError(f"timed out waiting for durable commit of {uri}@{ver}")
        self._durable_cond.wait(min(0.2, rem) if rem is not None else 0.2)
    return True

  # ***** get *****

  def latest(self, dest: str) -> dict:
    if self.remote:
      return json.loads(self.fs.cat_file(f"{dest}/_latest")) if self.fs.exists(f"{dest}/_latest") else {}
    if not os.path.isfile(p := f"{dest}/_latest"):
      return {}
    with open(p) as f:
      return json.load(f)

  def _remote_manifest(self, dest: str, version: str) -> dict | None:
    if not self.fs.exists(path := f"{dest}/{version}/{MANIFEST}"):
      return None
    with contextlib.suppress(Exception):
      return _valid_manifest(json.loads(self.fs.cat_file(path)))
    return None

  def _peer_manifest(self, peer: str, version: str) -> dict | None:
    with contextlib.suppress(Exception):
      with urllib.request.urlopen(f"http://{peer}/{version}/{MANIFEST}", timeout=10) as r:
        return _valid_manifest(json.loads(r.read()))
    return None

  def _is_legacy_local(self, dest: str) -> str | None:
    real = os.path.join(os.path.dirname(dest), os.readlink(dest)) if os.path.islink(dest) else dest
    if (
      os.path.isdir(real)
      and not os.path.isfile(f"{real}/_latest")
      and any(n.endswith(".safetensors") for n in os.listdir(real))
    ):
      return real
    return None

  def transfer_status(self) -> dict[str, dict]:
    """Snapshot of active in-flight transfers and complete local versions for node observations."""
    out: dict[str, dict] = {}
    if self.remote and os.path.isdir(self.cache):
      for e in os.scandir(self.cache):
        if e.is_dir() and not e.name.startswith(".") and e.name != "latest" and (m := read_manifest(e.path)):
          out[e.name] = {
            "ready": True,
            "durable": os.path.exists(f"{e.path}.durable"),
            "bytes": sum(m["files"].values()),
            "uri": m.get("uri", ""),
          }
    with self._transfer_lock:
      for ver, info in self.transfers.items():
        out[ver] = {
          "ready": False,
          "durable": False,
          "progress": min(1.0, info["bytes_done"] / max(1, info["bytes_total"])),
          "bytes": info["bytes_total"],
          "uri": info["uri"],
        }
    return out

  def local_folder(
    self,
    uri: str,
    version: str | None = None,
    peers: list[str] | tuple[str, ...] = (),
    cancel: threading.Event | None = None,
  ) -> tuple[str, str]:
    """Resolve an explicit version directly (never re-consulting _latest) or discover _latest when version is None."""
    dest, (ver, inline_peer) = self.path(uri), parse_version(version)
    peer_list = ([inline_peer] if inline_peer else []) + [p for p in peers if p and p != inline_peer and p != self.peer]

    if not self.remote:
      if ver:
        if _is_ready_folder(vdir := f"{dest}/{ver}"):
          return vdir, ver
        if self._is_legacy_local(dest) is not None:
          raise ValueError(f"legacy checkpoint at {uri} cannot resolve explicit version {ver}")
        raise ValueError(f"version {ver} of {uri} not found")
      if (latest_ver := self.latest(dest).get("version")) and _is_ready_folder(vdir := f"{dest}/{latest_ver}"):
        return vdir, latest_ver
      if legacy := self._is_legacy_local(dest):
        with os.scandir(legacy) as it:
          stats = sorted((e.name, s.st_size, s.st_mtime_ns) for e in it for s in [e.stat()])
        return legacy, hashlib.blake2b(repr((legacy, stats)).encode(), digest_size=8).hexdigest()
      raise ValueError(f"no checkpoint at {uri}")

    if not ver:
      if (local_ver := self.latest_version_here(uri)) and read_manifest(f"{self.cache}/{local_ver}") is not None:
        os.utime(f"{self.cache}/{local_ver}")
        return f"{self.cache}/{local_ver}", local_ver
      info = self.latest(dest)
      if not (ver := info.get("version")):
        raise ValueError(f"no checkpoint at {uri}")
      if not info.get("format") and not self.fs.exists(f"{dest}/{ver}/{MANIFEST}"):
        names = info.get("files", [])
        if not any(n.endswith(".safetensors") for n in names):
          raise ValueError(f"no checkpoint at {uri}")
        src = f"{self.base}/{info['dir']}" if info.get("dir") else dest
        return self._fetch_cached(uri, ver, lambda tmp, _: self.download_files(src, names, tmp), durable=True), ver

    folder = f"{self.cache}/{ver}"
    if _is_ready_folder(folder):
      with contextlib.suppress(OSError):
        os.utime(folder)
      return folder, ver
    return self._fetch_version(uri, dest, ver, peer_list, cancel), ver

  def _fetch_version(self, uri: str, dest: str, ver: str, peers: list[str], cancel: threading.Event | None) -> str:
    while True:
      with self._transfer_lock:
        if _is_ready_folder(f"{self.cache}/{ver}"):
          return f"{self.cache}/{ver}"
        existing = self.transfers.get(ver)
        if owner := existing is None:
          state = self.transfers[ver] = {
            "uri": uri,
            "bytes_done": 0,
            "bytes_total": 0,
            "done": threading.Event(),
            "error": None,
          }
      if not owner:
        while not existing["done"].wait(0.05):
          if cancel and cancel.is_set():
            raise InterruptedError(f"cancelled wait for {ver}")
        if _is_ready_folder(f"{self.cache}/{ver}"):
          return f"{self.cache}/{ver}"
        if existing["error"] is not None:
          if isinstance(existing["error"], InterruptedError):
            continue
          raise existing["error"]
        continue
      try:
        return self._download_from_peers_or_remote(uri, dest, ver, peers, state, cancel)
      except BaseException as exc:
        state["error"] = exc
        raise
      finally:
        with self._transfer_lock:
          self.transfers.pop(ver, None)
        state["done"].set()

  def _download_from_peers_or_remote(
    self, uri: str, dest: str, ver: str, peers: list[str], state: dict, cancel: threading.Event | None
  ) -> str:
    def add_bytes(n: int) -> None:
      state["bytes_done"] += n

    for peer in peers:
      if cancel and cancel.is_set():
        raise InterruptedError(f"cancelled fetch {ver}")
      if folder := self.copy_from(peer, ver, state, add_bytes, cancel):
        return folder
    if cancel and cancel.is_set():
      raise InterruptedError(f"cancelled fetch {ver}")
    if (manifest := self._remote_manifest(dest, ver)) is not None:
      state["bytes_total"], state["bytes_done"] = sum(manifest["files"].values()), 0
      names, src = list(manifest["files"]), f"{dest}/{ver}"

      def fill(tmp: str, _) -> None:
        self.download_files(src, names, tmp, on_bytes=add_bytes, cancel=cancel)
        if not verify_manifest(tmp, manifest):
          raise OSError(f"remote version {ver} failed manifest size verification")
        with open(f"{tmp}/{MANIFEST}", "w") as f:
          json.dump(manifest, f)

      return self._fetch_cached(uri, ver, fill, durable=True)

    info = self.latest(dest)
    if info and not info.get("format") and info.get("version") == ver:
      src = f"{self.base}/{info['dir']}" if info.get("dir") else dest
      return self._fetch_cached(
        uri, ver, lambda tmp, _: self.download_files(src, info.get("files", []), tmp), durable=True
      )
    raise ValueError(f"version {ver} of {uri} has not landed{f', and {peers[0]} failed to copy' if peers else ''}")

  def copy_from(
    self,
    peer: str,
    version: str,
    state: dict | None = None,
    on_bytes: Callable[[int], None] | None = None,
    cancel: threading.Event | None = None,
  ) -> str | None:
    """Copy a complete manifest-verified version from a peer node. Never marks the copy as durable."""
    if (manifest := self._peer_manifest(peer, version)) is None:
      return None
    if state is not None:
      state["bytes_total"], state["bytes_done"] = sum(manifest["files"].values()), 0
    url, names = f"http://{peer}/{version}", list(manifest["files"])
    try:

      def fill(tmp: str, _) -> None:
        if names:
          parallel(
            lambda n: (
              os.makedirs(os.path.dirname(f"{tmp}/{n}"), exist_ok=True),
              fetch(f"{url}/{n}", f"{tmp}/{n}", on_bytes, cancel),
            ),
            names,
          )
        if not verify_manifest(tmp, manifest):
          raise OSError(f"peer copy {version} from {peer} failed manifest size verification")
        with open(f"{tmp}/{MANIFEST}", "w") as f:
          json.dump(manifest, f)

      return self._fetch_cached(manifest.get("uri", ""), version, fill, durable=False)
    except InterruptedError:
      raise
    except Exception:
      return None

  def _fetch_cached(self, uri: str, version: str, fill, durable: bool) -> str:
    folder = f"{self.cache}/{version}"
    if read_manifest(folder) is None:
      os.makedirs(self.cache, exist_ok=True)
      tmp = tempfile.mkdtemp(dir=self.cache, prefix=f".{version}.")
      try:
        fill(tmp, uri)
        try:
          os.rename(tmp, folder)
          tmp = ""
        except OSError:
          if not os.path.isdir(folder):
            raise
      finally:
        if tmp:
          shutil.rmtree(tmp, ignore_errors=True)
    if durable:
      open(f"{folder}.durable", "w").close()
    return folder

  def download_files(
    self,
    src: str,
    names: list[str],
    folder: str,
    on_bytes: Callable[[int], None] | None = None,
    cancel: threading.Event | None = None,
  ) -> None:
    def one(n: str) -> None:
      if cancel and cancel.is_set():
        raise InterruptedError(f"cancelled download {src}/{n}")
      dst = f"{folder}/{n}"
      os.makedirs(os.path.dirname(dst), exist_ok=True)
      self.fs.get_file(f"{src}/{n}", dst)
      if on_bytes:
        on_bytes(os.path.getsize(dst))

    if names:
      parallel(one, names)

  @contextlib.contextmanager
  def pin_local(self, uri: str, version: str | None = None, peers: list[str] | tuple[str, ...] = ()):
    """Resolve and cross-process pin a local version directory for the duration of a read."""
    for attempt in range(2):
      folder, ver = self.local_folder(uri, version, peers=peers)
      with shared_lock(folder):
        if os.path.isdir(folder) and (read_manifest(folder) is not None or self._is_legacy_local(folder) is not None):
          yield folder, ver
          return
        if attempt == 1:
          raise FileNotFoundError(f"version {ver} of {uri} disappeared before read pin")

  def weight_files(self, uri: str, version: str | None = None, peers: list[str] | tuple[str, ...] = ()) -> list[str]:
    folder = self.local_folder(uri, version, peers=peers)[0]
    if os.path.islink(folder):
      folder = os.path.join(os.path.dirname(folder), os.readlink(folder))
    return sorted(glob.glob(f"{folder}/*.safetensors"))

  @span("store.get")
  def get(
    self, uri: str, version: str | None = None, peers: list[str] | tuple[str, ...] = ()
  ) -> tuple[dict[str, torch.Tensor], dict]:
    import safetensors

    def load(path: str) -> dict[str, torch.Tensor]:
      with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}

    with self.pin_local(uri, version, peers=peers) as (folder, _):
      paths = sorted(glob.glob(f"{folder}/*.safetensors"))
      if not paths:
        raise ValueError(f"no safetensors files in {uri}")
      tensors = {k: v for s in parallel(load, paths) for k, v in s.items()}
      return tensors, manifest["meta"] if (manifest := read_manifest(folder)) is not None else header(paths[0])

  def read_meta(self, uri: str, version: str | None = None, peers: list[str] | tuple[str, ...] = ()) -> dict:
    """Read checkpoint metadata from manifest.json directly without downloading safetensors shards."""
    dest, (ver, inline_peer) = self.path(uri), parse_version(version)
    peer_list = ([inline_peer] if inline_peer else []) + [p for p in peers if p and p != inline_peer and p != self.peer]
    if not self.remote:
      ver = ver or self.latest(dest).get("version", "")
      if ver and (m := read_manifest(f"{dest}/{ver}")) is not None:
        return dict(m.get("meta", {}))
      if (
        not ver and (legacy := self._is_legacy_local(dest)) and (paths := sorted(glob.glob(f"{legacy}/*.safetensors")))
      ):
        return header(paths[0])
      raise ValueError(f"no checkpoint metadata at {uri}" + (f" version {ver}" if ver else ""))

    if not ver:
      if (local_ver := self.latest_version_here(uri)) and (m := read_manifest(f"{self.cache}/{local_ver}")) is not None:
        return dict(m.get("meta", {}))
      ver = self.latest(dest).get("version", "")
    if ver:
      if (m := read_manifest(f"{self.cache}/{ver}")) is not None:
        return dict(m.get("meta", {}))
      for peer in peer_list:
        if (m := self._peer_manifest(peer, ver)) is not None:
          return dict(m.get("meta", {}))
      if (m := self._remote_manifest(dest, ver)) is not None:
        return dict(m.get("meta", {}))
    return header(self.weight_files(uri, version, peers=peers)[0])

  # ***** delete & sweep *****

  def _rm_local_dir(self, folder: str, remove_markers: bool = False) -> bool:
    if not os.path.isdir(folder):
      return True
    if (lk := try_exclusive_lock(folder)) is None:
      return False
    try:
      shutil.rmtree(folder, ignore_errors=True)
      if remove_markers:
        for ext in (".durable", ".upload", ".upload.tmp"):
          with contextlib.suppress(FileNotFoundError):
            os.unlink(f"{folder}{ext}")
      return True
    finally:
      release_lock(lk)
      with contextlib.suppress(FileNotFoundError):
        os.unlink(lock_file(folder))

  def delete_version(self, uri: str, version: str) -> bool:
    """Delete one version from durable storage and local cache, honoring cross-process read locks."""
    if not (ver := parse_version(version)[0]):
      self.delete(uri)
      return True
    deleted_cache = self._rm_local_dir(f"{self.cache}/{ver}", remove_markers=True) if self.remote else True
    dest = self.path(uri)
    if self.remote:
      if self.fs.exists(vpath := f"{dest}/{ver}"):
        self.fs.rm(vpath, recursive=True)
      if self.fs.exists(dest) and [os.path.basename(e) for e in self.fs.ls(dest, detail=False)] in ([], ["_latest"]):
        with contextlib.suppress(Exception):
          self.fs.rm(dest, recursive=True)
      return deleted_cache
    deleted = self._rm_local_dir(f"{dest}/{ver}")
    if (
      deleted
      and os.path.isdir(dest)
      and [n for n in os.listdir(dest) if not n.startswith(".") and not n.endswith(".lock")] in ([], ["_latest"])
    ):
      shutil.rmtree(dest, ignore_errors=True)
    return deleted

  def delete(self, uri: str) -> None:
    """Delete a checkpoint URI and all unpinned versions under it."""
    dest = self.path(uri)
    if self.remote:
      if self.fs.exists(dest):
        self.fs.rm(dest, recursive=True)
      return
    if os.path.islink(dest):
      os.unlink(dest)
    elif os.path.isdir(dest):
      for e in list(os.scandir(dest)):
        if e.is_dir() and not e.name.startswith("."):
          self._rm_local_dir(e.path)
      if not [n for n in os.listdir(dest) if not n.startswith(".") and not n.endswith(".lock") and n != "_latest"]:
        shutil.rmtree(dest, ignore_errors=True)

  def list_checkpoints(self) -> list[str]:
    """Every checkpoint URI present in the store."""
    if self.remote:
      return [f"tinker://{p[len(self.base) + 1 :]}" for p in self.fs.glob(f"{self.base}/*/*/*")]
    return [f"tinker://{os.path.relpath(p, self.base)}" for p in glob.glob(f"{glob.escape(self.base)}/*/*/*")]

  def list_versions(self) -> list[tuple[str, str, float]]:
    """Every (uri, version, created_ts) stored in durable storage (legacy unversioned URIs have version='')."""
    out: list[tuple[str, str, float]] = []
    for uri in self.list_checkpoints():
      dest = self.path(uri)
      if self.remote:
        entries = self.fs.ls(dest, detail=False) if self.fs.exists(dest) else []
        vers = [os.path.basename(e) for e in entries if os.path.basename(e) != "_latest"]
        if not vers:
          out.append((uri, "", self.saved_at(uri)))
        for v in vers:
          m = self._remote_manifest(dest, v)
          out.append((uri, v, m["created"] if m and "created" in m else self.saved_at(uri)))
      elif self._is_legacy_local(dest) is not None:
        out.append((uri, "", os.lstat(dest).st_mtime))
      elif os.path.isdir(dest):
        dest_mtime = os.lstat(dest).st_mtime
        for e in (x for x in os.scandir(dest) if x.is_dir() and not x.name.startswith(".")):
          m = read_manifest(e.path)
          out.append(
            (
              uri,
              e.name,
              min(m["created"], dest_mtime, e.stat().st_mtime)
              if m and "created" in m
              else min(dest_mtime, e.stat().st_mtime),
            )
          )
    return out

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
    protected_versions: set[str] | None = None,
    versionless_uris: set[str] | None = None,
  ) -> None:
    """Delete unreferenced checkpoints/versions saved after since and older than grace, then trim cache."""
    protected = set(protected_versions or ())
    legacy_uris = (
      set(versionless_uris or ())
      if versionless_uris is not None
      else (registered if protected_versions is None else set())
    )
    self.trim_cache(protected_versions=protected)
    if since is None:
      return
    before = time.time() - grace
    for uri, ver, created in self.list_versions():
      if (ver and (ver in protected or uri in legacy_uris)) or (not ver and uri in registered):
        continue
      try:
        if since < created < before:
          self.delete_version(uri, ver)
      except Exception as e:
        print(f"sweep could not delete {uri}@{ver}, trying again next sweep: {e!r}", file=sys.stderr, flush=True)
    for e in list(os.scandir(self.cache)) if os.path.isdir(self.cache) else []:
      if e.name.endswith(".upload") and since < e.stat().st_mtime < before:
        if (ver := e.name.removesuffix(".upload")) in protected:
          continue
        with contextlib.suppress(OSError, ValueError), open(e.path) as f:
          if json.load(f)["uri"] not in legacy_uris and self._rm_local_dir(f"{self.cache}/{ver}", remove_markers=True):
            with contextlib.suppress(FileNotFoundError):
              os.unlink(e.path)

  def trim_cache(self, protected_versions: set[str] = frozenset()) -> None:
    """Evict least-recently-used durable cached versions until under CKPT_CACHE_BYTES, honoring pins and active reads."""
    if not os.path.isdir(self.cache):
      return
    limit = CACHE_BYTES or shutil.disk_usage(self.cache).total // 2
    found = [
      e
      for e in os.scandir(self.cache)
      if e.is_dir() and not e.name.startswith(".") and e.name != "latest" and os.path.exists(f"{e.path}.durable")
    ]
    sizes = {}
    for e in found:
      with contextlib.suppress(OSError):
        sizes[e.path] = sum(os.path.getsize(os.path.join(r, fn)) for r, _, fns in os.walk(e.path) for fn in fns)
    total = sum(sizes.values())
    for e in sorted(found, key=lambda e: e.stat().st_mtime):
      if total <= limit:
        break
      if e.name not in protected_versions and self._rm_local_dir(e.path, remove_markers=True):
        total -= sizes.get(e.path, 0)
