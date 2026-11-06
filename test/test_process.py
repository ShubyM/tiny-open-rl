import asyncio
import contextlib
import errno
import os
import select
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tinyrl.control.process import Process, die_with_parent, launch, reap, target_pid

SERVER = """
import os, socket, sys, time
from types import SimpleNamespace
from tinyrl.control import process

# Exercise the same framing and synchronous serve loop over TCP on runtimes that prohibit Unix sockets.
if port := int(sys.argv[1]):
  class Listener:
    def __init__(self): self.sock = socket.socket()
    def __enter__(self): return self
    def __exit__(self, *args): self.sock.close()
    def bind(self, path): self.sock.bind(('127.0.0.1', port))
    def listen(self, n): self.sock.listen(n)
    def accept(self): return self.sock.accept()
  process.socket = SimpleNamespace(AF_UNIX=socket.AF_UNIX, socket=lambda *_: Listener())

def handle(method, *args):
  if method == 'echo': return args
  if method == 'fail': raise ValueError('rejected command')
  if method == 'exit': os._exit(17)
  if method == 'wait':
    with open(args[0], 'w') as f: f.write('waiting')
    time.sleep(60)

process.serve(sys.argv[2], handle)
"""


def require_unix():
  with tempfile.TemporaryDirectory() as folder:
    try:
      with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(f"{folder}/probe")
    except OSError as e:
      if e.errno == errno.EPERM:
        raise unittest.SkipTest("this runtime prohibits Unix sockets; TCP tests exercise the same IPC framing") from e
      raise


async def wait_for_file(path: str):
  async def wait():
    while not os.path.exists(path):
      await asyncio.sleep(0.01)

  await asyncio.wait_for(wait(), 10)


def read_line(proc: subprocess.Popen) -> bytes:
  if not select.select([proc.stdout], [], [], 10)[0]:
    raise AssertionError("child did not become ready")
  return proc.stdout.readline()


class TestProcess(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    with socket.socket() as sock:
      sock.bind(("127.0.0.1", 0))
      self.port = sock.getsockname()[1]

    async def connect(_path):
      return await asyncio.open_connection("127.0.0.1", self.port)

    self.connection = patch("tinyrl.control.process.asyncio.open_unix_connection", side_effect=connect)
    self.connection.start()
    self.addCleanup(self.connection.stop)
    self.child = Process([sys.executable, "-c", SERVER, str(self.port)], dict(os.environ))

  async def asyncTearDown(self):
    await asyncio.wait_for(self.child.close(), 10)

  async def test_framing_round_trips_bytes_and_metadata_in_a_private_directory(self):
    self.assertEqual(stat.S_IMODE(os.stat(self.child.folder.name).st_mode), 0o700)
    body = bytes(range(256)) * 8192
    self.assertEqual(
      await self.child.call("echo", body, {"version": "v1", "size": 42}), (body, {"version": "v1", "size": 42})
    )
    self.assertEqual(await self.child.call("echo", b"next"), (b"next",))

  async def test_a_handler_error_keeps_the_process_and_connection_usable(self):
    with self.assertRaisesRegex(RuntimeError, "ValueError: rejected command"):
      await self.child.call("fail")
    self.assertIsNone(self.child.proc.poll())
    self.assertEqual(await self.child.call("echo", "still here"), ("still here",))

  async def test_child_exit_is_eof_and_a_broken_connection_is_not_reused(self):
    await self.child.call("echo", "ready")
    with self.assertRaises(EOFError):
      await self.child.call("exit")
    with self.assertRaisesRegex(RuntimeError, "closed"):
      await self.child.call("echo", "too late")
    await self.child.close()
    self.assertIsNotNone(self.child.proc.poll())
    self.assertFalse(os.path.exists(self.child.folder.name))

  async def test_cancelling_an_inflight_command_kills_the_child_without_another_rpc(self):
    marker = f"{self.child.folder.name}/waiting"
    call = asyncio.create_task(self.child.call("wait", marker))
    await wait_for_file(marker)
    call.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await call
    await asyncio.wait_for(self.child.close(), 5)
    self.assertIsNotNone(self.child.proc.poll())
    self.assertFalse(os.path.exists(self.child.folder.name))

  async def test_cancelled_close_can_be_retried_until_the_process_is_reaped(self):
    await self.child.call("echo", "ready")
    reaped, poll = asyncio.Event(), self.child.proc.poll
    with patch.object(self.child.proc, "poll", side_effect=lambda: poll() if reaped.is_set() else None):
      closing = asyncio.create_task(self.child.close())
      await asyncio.sleep(0.02)
      closing.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await closing
      self.assertFalse(self.child.closed)
      reaped.set()
      await self.child.close()
    self.assertIsNotNone(self.child.proc.poll())
    self.assertFalse(os.path.exists(self.child.folder.name))
    await self.child.close()

  async def test_a_process_that_never_listened_reports_its_exit(self):
    child = Process([sys.executable, "-c", "raise SystemExit(23)"], dict(os.environ))
    try:
      with self.assertRaisesRegex(RuntimeError, "process exited with 23"):
        # This child did not bind either transport; use a path that cannot connect to the other fixture.
        with patch("tinyrl.control.process.asyncio.open_unix_connection", side_effect=FileNotFoundError):
          await asyncio.wait_for(child.call("echo"), 10)
    finally:
      await child.close()

  async def test_close_before_the_first_call_does_not_leave_a_process(self):
    await self.child.close()
    self.assertIsNotNone(self.child.proc.poll())
    self.assertFalse(os.path.exists(self.child.folder.name))
    with self.assertRaisesRegex(RuntimeError, "closed"):
      await self.child.call("echo", "already stopped")


class TestUnixProcess(unittest.IsolatedAsyncioTestCase):
  async def test_real_private_unix_socket(self):
    require_unix()
    child = Process([sys.executable, "-c", SERVER, "0"], dict(os.environ))
    try:
      self.assertEqual(
        await asyncio.wait_for(child.call("echo", b"request", {"version": "v1"}), 10), (b"request", {"version": "v1"})
      )
    finally:
      await asyncio.wait_for(child.close(), 10)
    self.assertIsNotNone(child.proc.poll())


class TestProcessOwnership(unittest.TestCase):
  def test_only_a_checkpointable_target_omits_its_parent_death_signal(self):
    cmd = """
import ctypes, time
sig = ctypes.c_int()
if ctypes.CDLL(None).prctl(2, ctypes.byref(sig)) != 0: raise OSError('PR_GET_PDEATHSIG failed')
print(sig.value, flush=True)
time.sleep(60)
"""
    for checkpointable in (False, True):
      with self.subTest(checkpointable=checkpointable):
        proc = launch(
          [sys.executable, "-c", cmd],
          dict(os.environ),
          checkpointable=checkpointable,
          stdout=subprocess.PIPE,
          bufsize=0,
        )
        try:
          self.assertEqual(int(read_line(proc)), 0 if checkpointable else signal.SIGKILL)
        finally:
          reap(proc)
          proc.wait()
          proc.stdout.close()

  def test_a_container_init_can_be_the_explicit_parent(self):
    with (
      patch.dict(os.environ, {"TINYRL_PARENT_PID": "1"}, clear=True),
      patch("os.getppid", return_value=1),
      patch("os.getpid", return_value=42),
      patch("ctypes.CDLL") as libc,
      patch("os.kill") as kill,
    ):
      libc.return_value.prctl.return_value = 0
      die_with_parent(signal.SIGUSR1)
    libc.return_value.prctl.assert_called_once_with(1, signal.SIGUSR1)
    kill.assert_not_called()

  def test_orphaned_and_reparented_children_still_die(self):
    # No marker, a dead expected parent, an exec child's own marker, or loss while installing the handler.
    for marker, parents in ((None, [1, 1]), ("9", [1, 1]), ("42", [1, 1]), ("9", [9, 1])):
      with (
        self.subTest(marker=marker, parents=parents),
        patch.dict(os.environ, {} if marker is None else {"TINYRL_PARENT_PID": marker}, clear=True),
        patch("os.getppid", side_effect=parents),
        patch("os.getpid", return_value=42),
        patch("ctypes.CDLL") as libc,
        patch("os.kill") as kill,
      ):
        libc.return_value.prctl.return_value = 0
        die_with_parent()
      kill.assert_called_once_with(42, signal.SIGKILL)

  def test_termination_is_forwarded_and_the_handle_waits_for_the_command(self):
    cmd = """
import signal, sys, time
signal.signal(signal.SIGTERM, lambda *_: (print('stopped', flush=True), sys.exit(0)))
print('ready', flush=True)
time.sleep(60)
"""
    proc = launch([sys.executable, "-c", cmd], dict(os.environ), stdout=subprocess.PIPE, bufsize=0)
    try:
      self.assertEqual(read_line(proc), b"ready\n")
      self.assertIsNone(proc.poll())
      proc.terminate()
      self.assertEqual(proc.wait(10), 0)
      self.assertEqual(proc.stdout.read(), b"stopped\n")
    finally:
      reap(proc)
      proc.wait()
      proc.stdout.close()

  def test_target_pid_is_the_command_not_its_supervisor(self):
    proc = launch(
      [sys.executable, "-c", "import os,time; print(os.getpid(),flush=True); time.sleep(60)"],
      dict(os.environ),
      stdout=subprocess.PIPE,
      bufsize=0,
    )
    try:
      command_pid = int(read_line(proc))
      if not os.path.exists(f"/proc/{proc.pid}/task/{proc.pid}/children"):
        self.skipTest("this runtime's virtual PIDs are absent from /proc")
      self.assertEqual(target_pid(proc), command_pid)
      self.assertNotEqual(command_pid, proc.pid)
    finally:
      reap(proc)
      proc.wait()
      proc.stdout.close()

  def test_parent_death_cleans_the_group_and_detached_ranks(self):
    cases = [
      (checkpointable, detached, delayed)
      for checkpointable in (False, True)
      for detached, delayed in ((False, False), (True, False), (True, True))
    ]
    for checkpointable, detached, delayed in cases:
      with self.subTest(checkpointable=checkpointable, detached=detached, delayed=delayed):
        rank_code = f"""
import os, time
if {delayed!r}:
  print('rank',os.getpid(),flush=True)
  time.sleep(0.3)
from tinyrl.control.process import die_with_parent
die_with_parent()
if not {delayed!r}: print('rank',os.getpid(),flush=True)
time.sleep(60)
"""
        target_code = f"""
import os, subprocess, sys, time
subprocess.Popen([sys.executable,'-c',{rank_code!r}], start_new_session={detached!r})
print('target',os.getpid(),flush=True)
time.sleep(60)
"""
        owner_code = f"""
import os, sys, time
from tinyrl.control.process import launch
proc = launch([sys.executable,'-c',{target_code!r}], dict(os.environ), checkpointable={checkpointable!r})
print('owner',proc.pid,flush=True)
time.sleep(60)
"""
        owner = subprocess.Popen(
          [sys.executable, "-c", owner_code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0
        )
        pids = {}
        try:
          for _ in range(3):
            line = read_line(owner).split()
            self.assertEqual(len(line), 2, line)
            pids[line[0].decode()] = int(line[1])
          self.assertEqual(set(pids), {"owner", "target", "rank"})
          owner.kill()
          owner.wait(10)
          self.assertTrue(select.select([owner.stdout], [], [], 10)[0], "a child survived the owner")
          self.assertEqual(owner.stdout.read(), b"")  # every child inherited this pipe, so EOF means all exited
        finally:
          if owner.poll() is None:
            owner.kill()
          owner.wait()
          if "owner" in pids:
            reap(SimpleNamespace(pid=pids["owner"]))
          for pid in pids.values():
            with contextlib.suppress(ProcessLookupError):
              os.kill(pid, signal.SIGKILL)
          owner.stdout.close()
