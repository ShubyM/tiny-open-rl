"""Local model commands and owned process groups. The private socket carries trusted request/reply objects, never model tensors."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import os
import pickle
import signal
import socket
import subprocess
import sys
import tempfile
import time


def die_with_parent(sig: int = signal.SIGKILL) -> None:
  parent = os.getppid()
  expected = int(os.environ.get("TINYRL_PARENT_PID", parent))
  if expected == os.getpid():
    expected = parent
  if ctypes.CDLL(None).prctl(1, sig) != 0:  # PR_SET_PDEATHSIG
    raise OSError("could not arrange process cleanup on parent death")
  if parent != expected or os.getppid() != parent or (parent == 1 and os.environ.get("TINYRL_PARENT_PID") != "1"):
    os.kill(os.getpid(), sig)


def launch(cmd: list[str], env: dict[str, str], *, checkpointable: bool = False, **kwargs) -> subprocess.Popen:
  """Own the command's group until it exits. A checkpointable target omits the parent-death signal CRIU would save."""
  return subprocess.Popen(
    [sys.executable, "-m", __name__, *(["--checkpointable"] if checkpointable else []), *cmd],
    env={**env, "TINYRL_PARENT_PID": str(os.getpid())},
    start_new_session=True,
    **kwargs,
  )


def reap(proc: subprocess.Popen) -> None:
  with contextlib.suppress(ProcessLookupError):
    os.killpg(proc.pid, signal.SIGKILL)


def group_pids(pid: int | None) -> list[int]:
  """Every live non-zombie process belonging to process group `pid` or descended from `pid`."""
  if not pid or pid <= 0 or not os.path.isdir("/proc"):
    return []
  live, children = [], {}
  for name in os.listdir("/proc"):
    if not name.isdigit():
      continue
    with contextlib.suppress(OSError, ValueError):
      with open(f"/proc/{name}/stat") as f:
        tail = f.read().rsplit(")", 1)[1].split()
      state, ppid, pgrp = tail[0], int(tail[1]), int(tail[2])
      if state == "Z":
        continue
      p = int(name)
      children.setdefault(ppid, []).append(p)
      if p == pid or pgrp == pid:
        live.append(p)
  seen, stack = set(live), list(live)
  while stack:
    for c in children.get(stack.pop(), ()):
      if c not in seen:
        seen.add(c)
        stack.append(c)
  return sorted(seen)


def gpu_pids_clear(pid: int | None) -> bool:
  """Verify the worker process group and all descendants have exited."""
  return not group_pids(pid)


def rss_bytes(pid: int | None) -> int:
  total = 0
  for p in group_pids(pid):
    with contextlib.suppress(OSError, ValueError):
      with open(f"/proc/{p}/statm") as f:
        total += int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
  return total


def target_pid(proc: subprocess.Popen) -> int:
  """The actual command PID, for CRIU. Its supervisor is not part of the saved tree."""
  while proc.poll() is None:
    with contextlib.suppress(FileNotFoundError):
      with open(f"/proc/{proc.pid}/task/{proc.pid}/children") as f:
        if children := f.read().split():
          return int(children[0])
    time.sleep(0.01)
  raise RuntimeError(f"process exited with {proc.returncode}")


def supervise(cmd: list[str], checkpointable: bool = False) -> None:
  signal.signal(signal.SIGUSR1, lambda *_: os.killpg(0, signal.SIGKILL))
  die_with_parent(signal.SIGUSR1)
  os.environ["TINYRL_PARENT_PID"] = str(os.getpid())
  child = os.fork()
  if child == 0:
    if not checkpointable:
      die_with_parent()
    os.environ["TINYRL_PARENT_PID"] = str(os.getpid())
    os.execvp(cmd[0], cmd)

  def forward(sig, _):
    signal.signal(sig, signal.SIG_IGN)
    os.killpg(0, sig)

  signal.signal(signal.SIGTERM, forward)
  signal.signal(signal.SIGINT, forward)
  _, status = os.waitpid(child, 0)
  sys.exit(os.waitstatus_to_exitcode(status))


def packet(value) -> bytes:
  data = pickle.dumps(value, protocol=5)
  return len(data).to_bytes(4, "big") + data


def serve(path: str, handler) -> None:
  """A synchronous model loop; the node closes the private socket when its child is no longer owned."""
  with socket.socket(socket.AF_UNIX) as listener:
    listener.bind(path)
    listener.listen(1)
    conn, _ = listener.accept()
    with conn, conn.makefile("rwb") as stream:
      while head := stream.read(4):
        method, args = pickle.loads(stream.read(int.from_bytes(head, "big")))
        try:
          result = True, handler(method, *args)
        except Exception as e:
          result = False, f"{type(e).__name__}: {e}"
        stream.write(packet(result))
        stream.flush()


class Process:
  def __init__(self, cmd: list[str], env: dict[str, str]):
    self.folder = tempfile.TemporaryDirectory(prefix="tinyrl-")
    self.path = f"{self.folder.name}/model.sock"
    self.reader = self.writer = None
    self.broken = self.closed = False
    try:
      self.proc = launch([*cmd, self.path], env)
    except BaseException:
      self.folder.cleanup()
      raise

  @property
  def pid(self) -> int | None:
    return None if self.closed else self.proc.pid

  def released(self) -> bool:
    return self.proc.poll() is not None and gpu_pids_clear(self.proc.pid)

  async def call(self, method: str, *args):
    if self.closed or self.broken:
      raise RuntimeError("model connection is closed")
    while self.writer is None:
      if self.proc.poll() is not None:
        raise RuntimeError(f"model process exited with {self.proc.returncode}")
      try:
        self.reader, self.writer = await asyncio.open_unix_connection(self.path)
      except (FileNotFoundError, ConnectionRefusedError):
        await asyncio.sleep(0.01)
    try:
      self.writer.write(packet((method, args)))
      await self.writer.drain()
      size = int.from_bytes(await self.reader.readexactly(4), "big")
      ok, result = pickle.loads(await self.reader.readexactly(size))
    except BaseException:
      self.broken = True
      raise
    if not ok:
      raise RuntimeError(result)
    return result

  async def close(self, graceful: bool = False) -> bool:
    if self.closed:
      return True
    if graceful and not self.broken and self.writer is not None and self.proc.poll() is None:
      with contextlib.suppress(Exception):
        await asyncio.wait_for(self.call("shutdown"), timeout=5.0)
    if self.writer:
      self.writer.close()
    reap(self.proc)
    while self.proc.poll() is None or not gpu_pids_clear(self.proc.pid):
      reap(self.proc)
      await asyncio.sleep(0.01)
    self.folder.cleanup()
    self.closed = True
    return True


if __name__ == "__main__":
  checkpointable = sys.argv[1:2] == ["--checkpointable"]
  supervise(sys.argv[2:] if checkpointable else sys.argv[1:], checkpointable)
