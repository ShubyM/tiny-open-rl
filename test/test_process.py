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

from tinyrl.control.process import Process, launch, reap, target_pid

SERVER = """
import os, sys, time
from tinyrl.control.process import serve

def handle(method, *args):
  if method == 'echo': return args
  if method == 'fail': raise ValueError('rejected command')
  if method == 'exit': os._exit(17)
  if method == 'wait':
    with open(args[0], 'w') as f: f.write('waiting')
    time.sleep(60)

serve(sys.argv[1], handle)
"""


def require_unix():
  with tempfile.TemporaryDirectory() as folder:
    try:
      with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(f"{folder}/probe")
    except OSError as e:
      if e.errno == errno.EPERM:
        raise unittest.SkipTest("this runtime prohibits Unix sockets") from e
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
    require_unix()
    self.child = Process([sys.executable, "-c", SERVER], dict(os.environ))

  async def asyncTearDown(self):
    await asyncio.wait_for(self.child.close(), 10)

  async def test_framing_round_trips_bytes_and_metadata_in_a_private_directory(self):
    self.assertEqual(stat.S_IMODE(os.stat(self.child.folder.name).st_mode), 0o700)
    body = bytes(range(256)) * 8192
    self.assertEqual(
      await self.child.call("echo", body, {"version": "v1", "size": 42}), (body, {"version": "v1", "size": 42})
    )
    with self.assertRaisesRegex(RuntimeError, "ValueError: rejected command"):
      await self.child.call("fail")
    self.assertEqual(await self.child.call("echo", b"next"), (b"next",))

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
    with self.assertRaisesRegex(RuntimeError, "closed"):
      await self.child.call("echo", "after cancellation")
    await asyncio.wait_for(self.child.close(), 5)
    self.assertIsNotNone(self.child.proc.poll())
    self.assertFalse(os.path.exists(self.child.folder.name))


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
