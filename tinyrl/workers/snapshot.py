"""Snapshots of an asleep vLLM server, so a sampler starts in seconds instead of booting vLLM. The first start of a configuration boots vLLM, puts it to sleep at level 2 (weights and KV cache dropped), and saves its processes with CRIU and cuda-checkpoint. Later starts restore that image onto their own GPU, in their own pid, network and mount namespaces, and reload the weights."""

from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time

from tinyrl.helpers import getenv

# where to keep snapshots, like /dev/shm/tinyrl-snapshots; empty turns them off
SNAPSHOTS = getenv("TINYRL_SNAPSHOTS", "")
RUN = "/run/tinyrl-vllm"  # each server's socket and log, private to every restored copy
# CRIU can't save an io_uring, and uvloop's bundled libuv opens one for every loop: run vLLM's server on asyncio instead
SITECUSTOMIZE = "import asyncio, uvloop\nuvloop.run, uvloop.new_event_loop, uvloop.install = asyncio.run, asyncio.new_event_loop, lambda: None\n"
ENV = {"VLLM_HOST_IP": "127.0.0.1", "GLOO_SOCKET_IFNAME": "lo"}  # a restored copy has only loopback
# --shell-job: the server stays in its sampler's process group, which the node daemon kills to stop the sampler
CRIU = ["--shell-job", "--file-locks", "--link-remap", "--tcp-established", "-W", "/tmp", "-v1"]
# a restored copy's own pids, network and mounts, all ended by a kill of unshare
UNSHARE = ["unshare", "--kill-child", "--pid", "--net", "--mount", "--mount-proc", "--propagation", "private"]


def site() -> str:
  """A folder to put first on a vLLM server's PYTHONPATH."""
  os.makedirs(f"{RUN}/site", exist_ok=True)
  with open(f"{RUN}/site/sitecustomize.py", "w") as f:
    f.write(SITECUSTOMIZE)
  return f"{RUN}/site"


def run(*args: str, log: str = "") -> None:
  """Run a command; a failure carries the errors from its log, which is where CRIU writes them."""
  if subprocess.run(args, stdout=subprocess.DEVNULL).returncode:
    errors = [line for line in open(log, errors="replace") if "Error" in line][-3:] if os.path.exists(log) else []
    raise RuntimeError(f"{args[0]} {args[1]} failed: {''.join(errors).strip()}")


def gpus() -> list[str]:
  out = subprocess.run(
    ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], check=True, capture_output=True, text=True
  )
  return out.stdout.split()


def gpu() -> str:
  """This process's GPU, from CUDA_VISIBLE_DEVICES as an index or a UUID."""
  visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
  return visible if visible.startswith("GPU-") else gpus()[int(visible)]


def key(args: list[str]) -> str:
  """What a snapshot depends on: the server's command line, vLLM's version, the driver and the GPU model."""
  smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader", "-i", "0"],
    capture_output=True,
    text=True,
  )
  return hashlib.sha256(
    json.dumps([args, importlib.metadata.version("vllm"), smi.stdout.strip()]).encode()
  ).hexdigest()[:16]


def cuda_pids(pid: int) -> list[int]:
  """The process and its descendants that hold a GPU."""
  tree, i = [pid], 0
  while i < len(tree):
    with open(f"/proc/{tree[i]}/task/{tree[i]}/children") as f:
      tree += [int(c) for c in f.read().split()]
    i += 1
  return [
    p
    for p in tree
    if any(os.readlink(fd).startswith("/dev/nvidia") for fd in glob.glob(f"/proc/{p}/fd/*") if os.path.islink(fd))
  ]


def save(pid: int, folder: str, name: str) -> None:
  """Save the asleep server whose root process is `pid`, and whose socket and log are `name`.sock and `name`.log, to `folder`. This ends the server."""
  pids = cuda_pids(pid)
  for p in pids:
    run("cuda-checkpoint", "--action", "lock", "--pid", str(p), "--timeout", "10000")
  for p in pids:
    run("cuda-checkpoint", "--action", "checkpoint", "--pid", str(p))
  tmp = f"{folder}.{os.getpid()}"
  try:
    os.makedirs(f"{tmp}/keep")
    log = f"/tmp/criu-dump-{os.getpid()}.log"
    before = set(glob.glob("/dev/shm/link_remap.*"))
    run("criu", "dump", "-t", str(pid), "-D", tmp, *CRIU, "--log-file", log, log=log)
    # files the image refers to that a restore consumes or checks: vLLM's unlinked semaphores, which CRIU links back as link_remap.N (moved, since the next dump reuses those names), and the server's log
    for f in set(glob.glob("/dev/shm/link_remap.*")) - before:
      shutil.move(f, f"{tmp}/keep/")
    shutil.copy2(f"{name}.log", f"{tmp}/keep/")
    for f in glob.glob(f"{tmp}/pages-*.img"):
      run("fallocate", "--dig-holes", f)  # about a third of the image is zero pages
    with open(f"{tmp}/meta.json", "w") as f:
      json.dump({"pids": pids, "gpu": gpu(), "sock": f"{os.path.basename(name)}.sock"}, f)
    if not os.path.isdir(folder):  # another sampler on this node may have saved the same configuration first
      os.rename(tmp, folder)
  finally:
    shutil.rmtree(tmp, ignore_errors=True)


def restore(folder: str) -> tuple[subprocess.Popen, str]:
  """Restore a snapshot onto this process's GPU. Returns the namespaces' root process, which ends them all when it ends, and the server's socket."""
  ns = subprocess.Popen([*UNSHARE, sys.executable, "-m", __name__, folder, gpu()], stdout=subprocess.PIPE, text=True)
  if ns.stdout.readline().strip() != "restored":
    ns.kill()
    raise RuntimeError(f"could not restore the snapshot in {folder}")
  with open(f"/proc/{ns.pid}/task/{ns.pid}/children") as f:
    init = int(f.read().split()[0])
  with open(f"{folder}/meta.json") as f:
    return ns, f"/proc/{init}/root{RUN}/{json.load(f)['sock']}"


def restore_here(folder: str, target: str) -> None:
  """Runs as the init of fresh namespaces: give the copy its own loopback and private /tmp, /dev/shm and RUN, restore, move it onto `target`, then reap."""
  run("ip", "link", "set", "lo", "up")
  # the snapshot may sit under /dev/shm, which the copy gets a private one of: reach it from a mount of its own first
  os.makedirs("/mnt/tinyrl-snapshot", exist_ok=True)
  run("mount", "--bind", folder, "/mnt/tinyrl-snapshot")
  folder = "/mnt/tinyrl-snapshot"
  for d in ("/tmp", "/dev/shm", RUN):
    os.makedirs(d, exist_ok=True)
    run("mount", "-t", "tmpfs", "tmpfs", d)
  for f in glob.glob(f"{folder}/keep/*"):
    shutil.copy2(f, "/dev/shm" if os.path.basename(f).startswith("link_remap.") else RUN)
  with open(f"{folder}/meta.json") as f:
    meta = json.load(f)
  log = "/tmp/criu-restore.log"
  run("criu", "restore", "-D", folder, *CRIU, "--tcp-close", "-d", "--log-file", log, log=log)
  swap = {meta["gpu"]: target, target: meta["gpu"]}
  device_map = ",".join(f"{u}={swap.get(u, u)}" for u in gpus())  # cuda-checkpoint wants every GPU listed
  for p in meta["pids"]:
    run("cuda-checkpoint", "--action", "restore", "--pid", str(p), "--device-map", device_map)
    run("cuda-checkpoint", "--action", "unlock", "--pid", str(p))
  print("restored", flush=True)
  while True:
    try:
      os.wait()
    except ChildProcessError:
      time.sleep(1)


if __name__ == "__main__":
  restore_here(sys.argv[1], sys.argv[2])
