"""Checkpoint storage behind put(uri) and get(uri).

Tensors are stored in safetensors with JSON metadata in the header, never torch.save pickle. Sampler checkpoints write the layout PEFT and vLLM load directly (adapter_model.safetensors + adapter_config.json, or model.safetensors + config.json, sharded with a model.safetensors.index.json when large).

CKPT is a local directory, or a URL like gs://bucket/prefix. put writes an immutable version and returns its id; the caller publishes that id in the registry under its worker lease. A remote store first writes into CKPT_CACHE. The node daemon uploads the files and then <uri>/<version>.json, so readers wait for exactly the version they requested. An upload cannot overwrite another version's manifest. A node lost before its uploads finish loses those versions.

Historical versions remain readable until their logical checkpoint is deleted. Sweeps retain the ancestors of every remaining manifest and the chains of active writers. Cache readers hold a shared file lock while loading, so a concurrent trim cannot remove their files. Old _latest manifests and local checkpoint directories are readable for upgrades; new saves never write mutable pointers.

A full fine-tune's sampler weights change in few bf16 values a step, so its saves form a chain per run, named <keyframe>.<n>, kept under <run>/_versions/ while any checkpoint is built from them. Link 0 goes up as the bytes that changed from the base model if this node has it, else whole, and every link whose parent didn't upload goes up whole. The rest go up as the bytes that changed from the link before, zlib compressed, with a CRC per chunk, and a node rebuilds a link from the newest one it has. delete_unused_chain_links() deletes the links nothing is built from.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import fcntl
import glob
import json
import os
import shutil
import sys
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
# trim_cache() keeps CKPT_CACHE under this, oldest first. 0 is half its disk
CACHE_BYTES = getenv("CKPT_CACHE_BYTES", 0)
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
  with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1, len(xs[0]))) as pool:
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


class Store:
  def __init__(self, root: str | None = None, cache: str | None = None):
    self.root = (CKPT if root is None else root).rstrip("/")
    self.remote = "://" in self.root
    # a file:// URL makes folders as it writes. No listings cache: other nodes write the bucket, so exists() and glob() must look
    self.fs, self.base = fsspec.core.url_to_fs(self.root, auto_mkdir=True, use_listings_cache=False)
    if not self.remote:
      self.base = os.path.realpath(self.base)
    self.cache = (CKPT_CACHE if cache is None else cache) if self.remote else f"{self.base}/_versions"
    self.pinned: torch.Tensor | None = None  # reused by every save, since pinning host memory is slow
    self.lock = threading.RLock()  # a save's tensors can be views of self.pinned until it is written
    self.uploading = threading.Lock()
    self.chains: dict[str, str] = {}  # run -> the last link of its chain this process saved

  def path(self, uri: str) -> str:
    path = uri.removeprefix("tinker://")
    if not uri.startswith("tinker://") or any(p in ("", ".", "..") for p in path.split("/")):
      raise ValueError(f"invalid checkpoint URI {uri}")
    return f"{self.base}/{path}"

  # ***** put *****

  def copy_to_host(self, tensors: dict[str, torch.Tensor], is_cuda=lambda t: t.is_cuda) -> dict[str, torch.Tensor]:
    """The tensors on the CPU. GPU tensors go through one pinned buffer, so the copy runs at DMA speed. When they all fit, they are views of it, else each streams through it in pieces into its own host tensor."""
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

  def write_checkpoint_files(self, folder: str, tensors: dict[str, torch.Tensor], info: dict) -> list[str]:
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
    """Write an immutable version locally. Its returned id is published by the caller under its worker lease."""
    dest = self.path(uri)
    os.makedirs(self.cache, exist_ok=True)
    folder, version = tempfile.mkdtemp(dir=self.cache), f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
    if self.remote and (meta or {}).get("config") and not (meta or {}).get("lora"):
      run = uri.removeprefix("tinker://").split("/")[0]
      k, _, n = self.chains.get(run, "").rpartition(".")
      version = self.chains[run] = f"{k}.{int(n) + 1}" if k and int(n) + 1 < KEYFRAME_EVERY else f"{version}.0"
    try:
      with self.lock:
        names = self.write_checkpoint_files(folder, self.copy_to_host(tensors), dict(meta or {}))
        with open(f"{folder}/.uri", "w") as f:
          f.write(uri)
        names.append(".uri")
        os.rename(folder, f"{self.cache}/{version}")
      if self.remote:
        with open(f"{self.cache}/{version}.upload.tmp", "w") as f:
          json.dump({"uri": uri, "names": names}, f)
        os.replace(f"{self.cache}/{version}.upload.tmp", f"{self.cache}/{version}.upload")
      else:
        os.makedirs(dest, exist_ok=True)
        with open(f"{self.cache}/{version}.json.tmp", "w") as f:
          json.dump({"version": version, "dir": f"_versions/{version}", "files": names}, f)
        os.replace(f"{self.cache}/{version}.json.tmp", f"{dest}/{version}.json")
    finally:
      shutil.rmtree(folder, ignore_errors=True)
    return version

  def upload_version(self, dest: str, version: str, names: list[str]) -> None:
    folder, rel = f"{self.cache}/{version}", dest[len(self.base) + 1 :]
    info, files = (
      {"version": version, "dir": f"{rel.split('/')[0]}/_versions/{version}", "files": names},
      [f"{folder}/{n}" for n in names],
    )
    k, _, n = version.rpartition(".")
    if k:  # a link of the run's chain
      parent = f"{self.cache}/{k}.{int(n) - 1}"
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
    self.fs.pipe_file(f"{dest}/{version}.json", json.dumps(info).encode())
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
    done = parallel(
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
    """Upload each immutable version independently. A missing delta parent makes that version a whole checkpoint."""
    with self.uploading, self.cache_lock():
      for version, job in self.pending_saves():
        try:
          if os.path.isdir(f"{self.cache}/{version}"):
            self.upload_version(self.path(job["uri"]), version, job["names"])
          os.unlink(f"{self.cache}/{version}.upload")
        except Exception as exc:
          print(f"upload failed for {version}, trying again: {exc!r}", file=sys.stderr, flush=True)

  def pending_saves(self) -> list[tuple[str, dict]]:
    found = []
    for p in sorted(glob.glob(f"{self.cache}/*.upload")):
      with open(p) as f:
        found.append((p.rsplit("/", 1)[-1].removesuffix(".upload"), json.load(f)))
    return found

  # ***** get *****

  def local_folder(self, uri: str, version: str | None = None) -> tuple[str, str]:
    """Resolve an exact immutable version; a missing version waits for its own manifest, never for a latest pointer."""
    dest = self.path(uri)
    if version is not None and (version in ("", ".", "..") or "/" in version or "\\" in version):
      raise ValueError("invalid checkpoint version")
    info = None
    if version is None:
      versions = [p.rsplit("/", 1)[-1][:-5] for p in self.fs.glob(f"{dest}/[0-9]*.json")]
      versions += [v for v, job in self.pending_saves() if job["uri"] == uri]
      if not versions:
        if not self.remote and glob.glob(f"{dest}/*.safetensors"):
          return os.path.realpath(dest), os.path.basename(os.path.realpath(dest))
        if not self.fs.exists(f"{dest}/_latest"):
          raise ValueError(f"no checkpoint at {uri}")
        info = json.loads(self.fs.cat_file(f"{dest}/_latest"))  # read checkpoints written before immutable manifests
      version = (
        max(versions, key=lambda v: (v.partition(".")[0], int(v.partition(".")[2] or -1)))
        if versions
        else info["version"]
      )
    folder, manifest = f"{self.cache}/{version}", f"{dest}/{version}.json"
    if os.path.isfile(f"{folder}/.uri"):
      with open(f"{folder}/.uri") as f:
        if f.read() != uri:
          raise ValueError("checkpoint version belongs to another URI")
      os.utime(folder)
      return folder, version
    deadline = time.monotonic() + UPLOAD_WAIT_S
    while info is None and not self.fs.exists(manifest):
      if self.fs.exists(f"{dest}/_latest"):
        old = json.loads(self.fs.cat_file(f"{dest}/_latest"))
        if old.get("version") == version:
          info = old
          break
      if not self.remote or time.monotonic() > deadline:
        raise ValueError(f"version {version} of {uri} never landed")
      time.sleep(0.05)
    info = info or json.loads(self.fs.cat_file(manifest))
    names, src = info["files"], f"{self.base}/{info['dir']}"
    if not any(n.endswith(".safetensors") for n in names):
      raise ValueError(f"no checkpoint at {uri}")
    if info.get("from", version) == version:
      return self.cached_folder(version, lambda tmp: self.download_files(src, names, tmp), uri), version
    return self.cached_folder(
      version, lambda tmp: self.rebuild_chain_link(os.path.dirname(src), version, info["from"], names, tmp), uri
    ), version

  @contextlib.contextmanager
  def cache_lock(self, exclusive: bool = False):
    os.makedirs(self.cache, exist_ok=True)
    with open(f"{self.cache}/.readers", "a") as lock:
      try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB if exclusive else fcntl.LOCK_SH)
      except BlockingIOError:
        yield False
      else:
        yield True

  @contextlib.contextmanager
  def hold(self, uri: str, version: str | None = None):
    """Pin local files against another process trimming this cache while an engine or trainer reads them."""
    with self.cache_lock():
      yield self.local_folder(uri, version)

  def cached_folder(self, version: str, fill, uri: str | None = None) -> str:
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
    if uri and not os.path.isfile(f"{folder}/.uri"):
      with open(f"{folder}/.uri", "w") as f:
        f.write(uri)
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
      crcs = parallel(lambda name, s, z: (name, unsparse(data[name][s], z)), *zip(*jobs)) if jobs else []
      if any([c for n, c in crcs if n == name] != json.loads(metas[name]["crc"]) for name in shards):
        raise ValueError(f"{link} rebuilt wrong")

    def write(name: str) -> None:
      head = metas[name]["head"].encode()
      with open(f"{folder}/{name}", "wb") as f:
        f.write(len(head).to_bytes(8, "little") + head)
        f.write(data[name])

    parallel(write, shards)

  def get(self, uri: str, version: str | None = None) -> tuple[dict[str, torch.Tensor], dict]:
    import safetensors

    def load(path: str) -> dict[str, torch.Tensor]:
      with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}

    with self.hold(uri, version) as (folder, _):
      paths = sorted(glob.glob(f"{folder}/*.safetensors"))
      return {k: v for s in parallel(load, paths) for k, v in s.items()}, header(paths[0])

  def read_meta(self, uri: str, version: str | None = None) -> dict:
    with self.hold(uri, version) as (folder, _):
      return header(sorted(glob.glob(f"{folder}/*.safetensors"))[0])

  # ***** delete *****

  def delete(self, uri: str) -> None:
    """Remove manifests and pending uploads. Version files are collected after their run no longer needs them."""
    with self.uploading, self.cache_lock(exclusive=True) as available:
      if not available:
        return
      for version, job in self.pending_saves():
        if job["uri"] == uri:
          os.unlink(f"{self.cache}/{version}.upload")
          shutil.rmtree(f"{self.cache}/{version}", ignore_errors=True)
      if self.fs.exists(self.path(uri)):
        self.fs.rm(self.path(uri), recursive=True)

  def list_checkpoints(self) -> list[str]:
    found = {
      f"tinker://{p[len(self.base) + 1 :].rsplit('/', 1)[0]}"
      for p in self.fs.glob(f"{self.base}/*/*/*/*.json")
      if "/_versions/" not in p
    }
    found |= {f"tinker://{p[len(self.base) + 1 : -8]}" for p in self.fs.glob(f"{self.base}/*/*/*/_latest")}
    return sorted(found | {job["uri"] for _, job in self.pending_saves()})

  def saved_at(self, uri: str) -> float:
    times = [self.fs.modified(p).timestamp() for p in self.fs.glob(f"{self.path(uri)}/*.json")]
    if not self.remote and os.path.exists(self.path(uri)):
      times = [os.stat(self.path(uri)).st_mtime]
    elif not times and self.fs.exists(f"{self.path(uri)}/_latest"):
      times = [self.fs.modified(f"{self.path(uri)}/_latest").timestamp()]
    times += [os.stat(f"{self.cache}/{v}.upload").st_mtime for v, job in self.pending_saves() if job["uri"] == uri]
    return max(times, default=0)

  def delete_unused_chain_links(
    self, live: set[str], since: float, before: float, active_runs: set[str] = frozenset()
  ) -> None:
    """Keep a running writer's chain and every retained checkpoint's ancestors; collect only retired runs' unreferenced links."""
    with self.cache_lock(exclusive=True) as available:
      if not available:
        return
      keep = {f"{k}.{i}" for v in live if "." in v for k, _, n in [v.rpartition(".")] for i in range(int(n) + 1)} | live
      for manifest in self.fs.glob(f"{self.base}/*/*/*/*.json"):
        if "/_versions/" in manifest:
          continue
        version = manifest.rsplit("/", 1)[-1][:-5]
        keep |= (
          {version}
          if "." not in version
          else {f"{version.rpartition('.')[0]}.{i}" for i in range(int(version.rpartition(".")[2]) + 1)}
        )
      for links in self.fs.glob(f"{self.base}/*/_versions/*") if self.remote else self.fs.glob(f"{self.cache}/*"):
        version = links.rsplit("/", 1)[-1]
        run = links[len(self.base) + 1 :].split("/")[0]
        if not self.remote and os.path.isfile(f"{links}/.uri"):
          with open(f"{links}/.uri") as f:
            run = f.read().removeprefix("tinker://").split("/")[0]
        if (
          self.fs.isdir(links)
          and version not in keep
          and run not in active_runs
          and (files := self.fs.ls(links, detail=False))
        ):
          if since < self.fs.modified(files[0]).timestamp() < before:
            self.fs.rm(links, recursive=True)

  def trim_cache(self) -> None:
    """Trim uploaded versions while no process is reading this cache; pending saves always stay."""
    if not self.remote or not os.path.isdir(self.cache):
      return
    with self.cache_lock(exclusive=True) as available:
      if not available:
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
