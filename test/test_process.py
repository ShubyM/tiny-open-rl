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

from tinyrl.control.process import Process, reap

SERVER = """
import os, subprocess, sys, threading, time
from tinyrl.control.process import serve

def handle(method, *args):
  if method == 'echo': return args
  if method == 'fail': raise ValueError('rejected command')
  if method == 'exit': os._exit(17)
  if method == 'wait':
    def launch():
      child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
      with open(args[0], 'w') as f: f.write(str(child.pid))
      time.sleep(60)
    threading.Thread(target=launch).start()
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
    with open(marker) as f:
      descendant = os.pidfd_open(int(f.read()))
    self.addCleanup(os.close, descendant)
    call.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await call
    with self.assertRaisesRegex(RuntimeError, "closed"):
      await self.child.call("echo", "after cancellation")
    await asyncio.wait_for(self.child.close(), 5)
    self.assertTrue(select.select([descendant], [], [], 0)[0], "detached descendant survived close")
    self.assertIsNotNone(self.child.proc.poll())
    self.assertFalse(os.path.exists(self.child.folder.name))


class TestProcessOwnership(unittest.TestCase):
  def test_parent_death_cleans_the_group_and_detached_ranks(self):
    for detached, delayed in ((False, False), (True, False), (True, True)):
      with self.subTest(detached=detached, delayed=delayed):
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
proc = launch([sys.executable,'-c',{target_code!r}], dict(os.environ))
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
