"""Checkpoint storage behind put(uri) and get(uri).

Tensors are stored in safetensors with JSON metadata in the header, never torch.save pickle. Sampler checkpoints write
the layout PEFT and vLLM load directly (adapter_model.safetensors + adapter_config.json, or model.safetensors +
config.json, sharded with a model.safetensors.index.json when large).

CKPT is a local directory, or a URL like gs://bucket/prefix that every node can reach. With a URL, the node is a cache
in front of it: put writes a checkpoint into CKPT_CACHE under a new version and returns, so this node reads it at once.
flush() uploads the node's saves behind, in order, to <uri>/<version>/ with <uri>/_latest pointing at it last. The node
daemon runs it, so a trainer exits, and gives its GPUs back, right after its last save. Another node waits for the
version it wants to land, then downloads it once into its own CKPT_CACHE, since engines load from a local path. A node
lost before its uploads finish loses those versions.

A full fine-tune's sampler weights change in few bf16 values a step, so its saves form a chain per run, named
<keyframe>.<n>, kept under <run>/_versions/ while any checkpoint is built from them. Link 0, and every link whose parent
didn't upload, goes up whole. The rest go up as the bytes that changed from the link before, zlib compressed, with a CRC
per chunk, and a node rebuilds a link from the newest one it has. collect() deletes the links nothing is built from.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import glob
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
import zlib
from typing import TYPE_CHECKING

import fsspec
import numpy as np

if TYPE_CHECKING:
  import torch

from tinyrl.helpers import getenv

CKPT = getenv("CKPT", "/tmp/tinyrl")
CKPT_CACHE = getenv("CKPT_CACHE", "/tmp/tinyrl-cache")
SHARD_BYTES = getenv("SHARD_BYTES", 256 * 1024 * 1024)  # a full checkpoint this big is written as parallel shards
MAX_SHARDS = 16
MAX_PIN_BYTES = getenv("MAX_PIN_BYTES", 10 * 1024 * 1024 * 1024)  # the most pinned host memory a save holds
UPLOAD_WAIT_S = getenv("UPLOAD_WAIT_S", 900.0)  # how long a node waits for a version another node is uploading
CACHE_BYTES = getenv("CKPT_CACHE_BYTES", 0)  # trim() keeps CKPT_CACHE under this, oldest first. 0 is half its disk
KEYFRAME_EVERY = getenv("KEYFRAME_EVERY", 8)  # the most links in a chain of full sampler weights
XOR_CHUNK = 16 * 1024 * 1024


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
  """The bytes of new that differ from old, as the gaps between them then their XOR with old, zlib compressed, and
  new's CRC. Few bytes differ, so this is small, and quicker to compress than the whole XOR."""
  x = np.bitwise_xor(old, new)
  at = np.flatnonzero(x)
  return zlib.compress(np.diff(at, prepend=-1).astype(np.uint32).tobytes() + x[at].tobytes(), 1), zlib.crc32(new)


def unsparse(data: np.ndarray, z: np.ndarray) -> int:
  """Apply sparse()'s bytes to data in place, and return its CRC."""
  raw = zlib.decompress(z)
  at = np.cumsum(np.frombuffer(raw, np.uint32, len(raw) // 5), dtype=np.int64) - 1
  data[at] ^= np.frombuffer(raw, np.uint8, offset=len(raw) // 5 * 4)
  return zlib.crc32(data)


class Store:
  def __init__(self, root: str | None = None, cache: str | None = None):
    self.root = (CKPT if root is None else root).rstrip("/")
    self.remote = "://" in self.root
    self.fs, self.base = fsspec.core.url_to_fs(self.root, auto_mkdir=True)  # a file:// URL makes folders as it writes
    if not self.remote:
      self.base = os.path.realpath(self.base)
    self.cache = CKPT_CACHE if cache is None else cache
    self.pinned: torch.Tensor | None = None  # reused by every save, since pinning host memory is slow
    self.lock = threading.RLock()  # a save's tensors can be views of self.pinned until it is written
    self.versions: dict[str, str] = {}  # uri -> the version this process last saved
    self.chains: dict[str, str] = {}  # run -> the last link of its chain this process saved

  def path(self, uri: str) -> str:
    return f"{self.base}/{uri.removeprefix('tinker://')}"

  # ***** put *****

  def stage(self, tensors: dict[str, torch.Tensor], is_cuda=lambda t: t.is_cuda) -> dict[str, torch.Tensor]:
    """The tensors on the CPU. GPU tensors go through one pinned buffer, so the copy runs at DMA speed. When they all
    fit, they are views of it, else each streams through it in pieces into its own host tensor."""
    import torch

    with self.lock:
      gpu = {k for k, v in tensors.items() if is_cuda(v)}
      out = {k: v.detach().cpu().contiguous() for k, v in tensors.items() if k not in gpu}
      total = sum(aligned(nbytes(tensors[k])) for k in gpu)
      if not total:
        return out
      size = min(total, MAX_PIN_BYTES)
      if self.pinned is None or self.pinned.numel() < size:
        self.pinned = None  # free the old buffer before pinning a bigger one
        self.pinned = torch.empty(size, dtype=torch.uint8, pin_memory=torch.cuda.is_available())
      sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
      at = 0
      for k in gpu:
        src, v = as_bytes(tensors[k]), tensors[k]
        if total <= MAX_PIN_BYTES:
          self.pinned[at : at + len(src)].copy_(src, non_blocking=True)
          out[k] = self.pinned[at : at + len(src)].view(v.dtype).view(v.shape)
          at += aligned(len(src))
          continue
        dst = torch.empty(len(src), dtype=torch.uint8)
        for i in range(0, len(src), size):
          piece = src[i : i + size]
          self.pinned[: len(piece)].copy_(piece, non_blocking=True)
          sync()
          dst[i : i + len(piece)].copy_(self.pinned[: len(piece)])
        out[k] = dst.view(v.dtype).view(v.shape)
      sync()
      return {k: out[k] for k in tensors}

  def write(self, folder: str, tensors: dict[str, torch.Tensor], info: dict) -> list[str]:
    """Write a checkpoint into folder, and return its file names, with the weights last."""
    import safetensors.torch

    lora, cfg, total = info.get("lora"), info.get("config"), sum(map(nbytes, tensors.values()))
    n = max(1, min(MAX_SHARDS, len(tensors))) if not lora and total >= SHARD_BYTES else 1
    stem = ("adapter_model" if lora else "model") if cfg else "state"
    weights = [f"{stem}.safetensors"] if n == 1 else [f"{stem}-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]
    shards = [dict(list(tensors.items())[i::n]) for i in range(n)]
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
    parallel(lambda name, shard: safetensors.torch.save_file(shard, f"{folder}/{name}", metadata=meta), weights, shards)
    return [*files, *weights]

  def put(self, uri: str, tensors: dict[str, torch.Tensor], meta: dict | None = None) -> str:
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
          names = self.write(folder, self.stage(tensors), dict(meta or {}))
        os.rename(folder, f"{self.cache}/{version}")
      finally:
        shutil.rmtree(folder, ignore_errors=True)
      self.point(uri, version)
      self.versions[uri] = version
      with open(f"{self.cache}/{version}.upload.tmp", "w") as f:
        json.dump({"uri": uri, "names": names}, f)
      os.replace(f"{self.cache}/{version}.upload.tmp", f"{self.cache}/{version}.upload")  # flush() takes it from here
      return dest
    parent, base = os.path.dirname(dest), os.path.basename(dest)
    os.makedirs(parent, exist_ok=True)
    folder = tempfile.mkdtemp(dir=parent, prefix=f".{base}.")
    link, prev = f"{folder}.link", f"{parent}/.{base}.prev"
    try:
      with self.lock:
        self.write(folder, self.stage(tensors), dict(meta or {}))
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
      with contextlib.suppress(FileNotFoundError):
        os.unlink(link)
    return dest

  def point(self, uri: str, version: str) -> None:
    """Record on this node which version of a checkpoint it saved last, for this node's other processes."""
    os.makedirs(f"{self.cache}/latest", exist_ok=True)
    tmp = f"{self.cache}/latest/.{uuid.uuid4().hex}"
    with open(tmp, "w") as f:
      f.write(version)
    os.replace(tmp, f"{self.cache}/latest/{hashlib.sha256(uri.encode()).hexdigest()[:32]}")

  def pointed(self, uri: str) -> str | None:
    with (
      contextlib.suppress(FileNotFoundError),
      open(f"{self.cache}/latest/{hashlib.sha256(uri.encode()).hexdigest()[:32]}") as f,
    ):
      return f.read()
    return None

  def upload(self, dest: str, version: str, names: list[str]) -> None:
    folder, rel = f"{self.cache}/{version}", dest[len(self.base) + 1 :]
    info, files = {"version": version, "dir": f"{rel}/{version}", "files": names}, [f"{folder}/{n}" for n in names]
    k, _, n = version.rpartition(".")
    if k:  # a link of the run's chain
      info["dir"], parent = f"{rel.split('/')[0]}/_versions/{version}", f"{self.cache}/{k}.{int(n) - 1}"
      first = ""  # the newest link before it that went up whole, if the one before it went up
      with contextlib.suppress(FileNotFoundError), open(f"{parent}.uploaded") as f:
        first = f.read()
      xor = int(n) and first and self.diff(parent, folder, names, f"{folder}.xor")
      if xor:
        files = [f"{folder}.xor/{x}" for x in xor] + [f for f in files if not f.endswith(".safetensors")]
      info["from"] = first if xor else version
    try:
      parallel(lambda f: self.fs.put_file(f, f"{self.base}/{info['dir']}/{os.path.basename(f)}"), files)
    finally:
      shutil.rmtree(f"{folder}.xor", ignore_errors=True)
    old = {}
    with contextlib.suppress(FileNotFoundError):
      old = json.loads(self.fs.cat_file(f"{dest}/_latest"))
    self.fs.pipe_file(f"{dest}/_latest", json.dumps(info).encode())
    if "." not in old.get("version", "."):  # a link stays until collect(), since other checkpoints may build on it
      stale = [f"{self.base}/{old['dir']}"] if old.get("dir") else [f"{dest}/{n}" for n in old.get("files", [])]
      for path in stale:  # the version before, which readers that started on it may still be downloading
        with contextlib.suppress(FileNotFoundError):
          self.fs.rm(path, recursive=True)
    with open(f"{folder}.uploaded", "w") as f:  # beside the folder, so engines that list it never see it
      f.write(info.get("from", ""))

  def diff(self, parent: str, folder: str, names: list[str], out: str) -> list[str] | None:
    """Write each weights file in folder as the bytes that differ from parent's, by chunk, with the file's header and a
    CRC per chunk. None if the files don't line up."""
    import safetensors.numpy

    shards = [n for n in names if n.endswith(".safetensors")]
    try:
      files = [(raw(f"{parent}/{n}"), raw(f"{folder}/{n}")) for n in shards]
    except FileNotFoundError:
      return None
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

  def flush(self) -> None:
    """Upload this node's saves in the order they were made. The first that fails stops the rest, so a checkpoint's
    _latest never goes back to an older version, and the next flush tries it again."""
    found = [e for e in os.scandir(self.cache) if e.name.endswith(".upload")] if os.path.isdir(self.cache) else []
    for e in sorted(found, key=lambda e: (e.stat().st_mtime_ns, e.name)):
      version = e.name.removesuffix(".upload")
      try:
        with open(e.path) as f:
          job = json.load(f)
      except FileNotFoundError:  # another flush took it
        continue
      if os.path.isdir(f"{self.cache}/{version}"):
        self.upload(self.path(job["uri"]), version, job["names"])
      with contextlib.suppress(FileNotFoundError):
        os.unlink(e.path)

  # ***** get *****

  def local(self, uri: str, version: str | None = None) -> tuple[str, str]:
    """A local directory holding the checkpoint, and its version, which changes when the checkpoint is saved again.
    With a version, from a store on another node, it waits for that version to land."""
    dest = self.path(uri)
    if not self.remote:
      real = dest
      with contextlib.suppress(OSError):
        real = os.path.join(os.path.dirname(dest), os.readlink(dest))
      if not os.path.isdir(real):
        raise ValueError(f"no checkpoint at {uri}")
      with os.scandir(real) as it:
        stats = sorted((e.name, s.st_size, s.st_mtime_ns) for e in it for s in [e.stat()])
      if not any(n.endswith(".safetensors") for n, _, _ in stats):
        raise ValueError(f"no checkpoint at {uri}")
      return dest, hashlib.blake2b(repr((real, stats)).encode(), digest_size=8).hexdigest()
    want = version or self.pointed(uri)  # without a version: the one this node saved last, else the bucket's
    if want and os.path.isdir(f"{self.cache}/{want}"):
      os.utime(f"{self.cache}/{want}")  # so trim() takes the ones nobody read for longest
      return f"{self.cache}/{want}", want
    deadline = time.time() + UPLOAD_WAIT_S
    while True:
      try:
        info = json.loads(self.fs.cat_file(f"{dest}/_latest"))
      except FileNotFoundError:
        info = {}
      if not version or info.get("version") == version:
        break
      if time.time() > deadline:
        raise ValueError(f"version {version} of {uri} never landed")
      time.sleep(0.5)  # another node is still uploading it
    if not info:
      raise ValueError(f"no checkpoint at {uri}")
    version, names = info["version"], info["files"]
    if not any(n.endswith(".safetensors") for n in names):
      raise ValueError(f"no checkpoint at {uri}")
    src = f"{self.base}/{info['dir']}" if info.get("dir") else dest
    if info.get("from", version) == version:
      return self.fetch(version, lambda tmp: self.download(src, names, tmp)), version
    return self.fetch(
      version, lambda tmp: self.rebuild(os.path.dirname(src), version, info["from"], names, tmp)
    ), version

  def fetch(self, version: str, fill) -> str:
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
        open(f"{folder}.uploaded", "w").close()  # the bucket has it, so trim() may drop it here
      finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return folder

  def download(self, src: str, names: list[str], folder: str) -> None:
    parallel(lambda n: self.fs.get_file(f"{src}/{n}", f"{folder}/{n}"), names)

  def rebuild(self, links: str, version: str, whole: str, names: list[str], folder: str) -> None:
    """Make a link of a chain in folder, from the newest link before it this node has, or else the newest one uploaded
    whole, by XORing in each link after that. The CRCs check the result is the exact bytes saved."""
    import safetensors

    k, _, n = version.rpartition(".")
    chain = [f"{k}.{i}" for i in range(int(whole.rpartition(".")[2]), int(n) + 1)]
    have = next(i for i in reversed(range(len(chain) - 1)) if i == 0 or os.path.isdir(f"{self.cache}/{chain[i]}"))
    start = self.fetch(chain[have], lambda tmp: self.download(f"{links}/{chain[have]}", names, tmp))
    shards, after = [n for n in names if n.endswith(".safetensors")], chain[have + 1 :]
    self.download(f"{links}/{version}", [n for n in names if n not in shards], folder)
    pairs = [(f"{link}/{name}.xor", f"{folder}/{link}.{name}.xor") for link in after for name in shards]
    parallel(lambda a, b: self.fs.get_file(f"{links}/{a}", b), *zip(*pairs))
    data = dict(zip(shards, parallel(lambda name: np.array(raw(f"{start}/{name}")[1]), shards)))
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

  def weights(self, uri: str, version: str | None = None) -> list[str]:
    folder = self.local(uri, version)[0]
    with contextlib.suppress(OSError):
      folder = os.path.join(os.path.dirname(folder), os.readlink(folder))
    return sorted(glob.glob(f"{folder}/*.safetensors"))

  def get(self, uri: str, version: str | None = None) -> tuple[dict[str, torch.Tensor], dict]:
    import safetensors

    def load(path: str) -> dict[str, torch.Tensor]:
      with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}

    paths = self.weights(uri, version)
    return {k: v for s in parallel(load, paths) for k, v in s.items()}, header(paths[0])

  def meta(self, uri: str) -> dict:
    return header(self.weights(uri)[0])

  # ***** delete *****

  def delete(self, uri: str) -> None:
    """Delete a checkpoint and every version of it this store kept."""
    dest = self.path(uri)
    if self.remote:
      with contextlib.suppress(FileNotFoundError):
        self.fs.rm(dest, recursive=True)
      return
    parent, base = os.path.split(dest)
    tail = (
      len(parent) + len(base) + 3
    )  # the random suffix of a version folder has no dot, so step-1 never takes step-1.5's
    versions = [p for p in glob.glob(f"{glob.escape(parent)}/.{glob.escape(base)}.*") if "." not in p[tail:]]
    for p in [dest, *versions]:  # the link to the live version, the .prev link, and the folders they name
      if os.path.islink(p):
        os.unlink(p)
      else:
        shutil.rmtree(p, ignore_errors=True)

  def stored(self) -> list[str]:
    """Every checkpoint in the store."""
    if self.remote:
      return [
        f"tinker://{p[len(self.base) + 1 : -len('/_latest')]}" for p in self.fs.glob(f"{self.base}/*/*/*/_latest")
      ]
    found = glob.glob(f"{glob.escape(self.base)}/*/*/*")  # glob skips the dot folders a save writes into
    return [f"tinker://{os.path.relpath(p, self.base)}" for p in found]

  def saved_at(self, uri: str) -> float:
    """When a checkpoint was last saved."""
    if self.remote:
      return self.fs.modified(f"{self.path(uri)}/_latest").timestamp()
    return os.lstat(self.path(uri)).st_mtime

  def collect(self, live: set[str], since: float, before: float) -> None:
    """Delete the links of chains that none of the live versions is built from, uploaded between since and before."""
    keep = {f"{k}.{i}" for v in live if "." in v for k, _, n in [v.rpartition(".")] for i in range(int(n) + 1)}
    for links in self.fs.glob(f"{self.base}/*/_versions/*") if self.remote else []:
      if links.rsplit("/", 1)[1] not in keep and (files := self.fs.ls(links, detail=False)):
        if since < self.fs.modified(files[0]).timestamp() < before:
          self.fs.rm(links, recursive=True)

  def trim(self) -> None:
    """Delete the versions in CKPT_CACHE nobody read for longest, until it is under CKPT_CACHE_BYTES. Only ones the
    bucket has: a version still uploading stays."""
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
