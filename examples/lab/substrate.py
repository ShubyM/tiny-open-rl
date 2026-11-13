"""An episode's sandbox as an Agent Substrate environment: LAB's sandbox image, run as it is, with the task's files sent in and LAB's
tools run against it. The environment is suspended once no tool has run for a while, so it holds no worker while the model
generates, and the next tool call resumes it.

Substrate keeps an environment until it is deleted and has no call that lists them, so a driver that is stopped deletes its own.
"""

import asyncio
import contextlib
import functools
import io
import json
import os
import re
import shlex
import signal
import tarfile
import uuid
from pathlib import Path

from ate_env import Client, EnvError, NotFoundError
from harness.tools import ToolExecutor
from sandbox.sandbox import DOCUMENTS_PATH, OUTPUT_PATH, WORKSPACE_PATH, ExecResult
from sandbox.sandbox import Sandbox as Podman

SPACE = "ate-env"
LIVE: set[str] = set()  # environments not yet deleted
# glob and grep as LAB's tools do them, for python3 in the sandbox: files under the root that no link leads out of, the newest
# 100 for a glob, 250 lines for a grep
SEARCH = r"""
import json, re, sys
from pathlib import Path
a = json.loads(sys.argv[1])
root = Path(a["root"])
real = root.resolve()
def inside(f):
  try:
    return bool(f.resolve().relative_to(real))
  except ValueError:
    return False
files = [f for f in root.glob(a["glob"]) if f.is_file() and inside(f)]
if "regex" not in a:
  files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
  print(json.dumps([str(f.relative_to(root)) for f in files[:100]]))
  sys.exit()
regex, out = re.compile(a["regex"]), []
for f in files:
  try:
    text = f.read_text(encoding="utf-8", errors="replace")
  except Exception:
    continue
  hits, name = len(list(regex.finditer(text))), str(f.relative_to(root))
  if hits and a["mode"] == "files_with_matches":
    out.append(name)
  elif hits and a["mode"] == "count":
    out.append(f"{name}: {hits}")
  elif hits and a["mode"] == "content":
    out += [f"{name}:{i + 1}: {line}" for i, line in enumerate(text.split("\n")) if regex.search(line)]
print(json.dumps(out[:250]))
"""


@functools.cache
def client(endpoint: str) -> Client:
  loop = asyncio.get_running_loop()

  async def leave():
    await asyncio.gather(*(delete(endpoint, name) for name in list(LIVE)), return_exceptions=True)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTERM)

  loop.add_signal_handler(signal.SIGTERM, lambda: loop.create_task(leave()))
  return Client(endpoint, auth_token=os.environ.get("SUBSTRATE_ENV_AUTH_TOKEN"))


async def delete(endpoint: str, name: str) -> None:
  with contextlib.suppress(NotFoundError):
    await asyncio.wait_for(client(endpoint).delete(name, atespace=SPACE), 60)
  LIVE.discard(name)


class Files:
  """What LAB's tools ask of a sandbox, answered by the environment. The tools run in a thread, and each call here runs on the event loop."""

  def __init__(
    self, env, loop: asyncio.AbstractEventLoop, documents: Path, workspace: Path, output: Path, timeout: int
  ):
    self.env, self.loop, self.timeout = env, loop, timeout
    self.documents_dir, self.workspace_dir, self.output_dir = (
      documents,
      workspace,
      output,
    )  # the host's, which LAB counts documents in

  def wait(self, call, seconds: float):
    return asyncio.run_coroutine_threadsafe(asyncio.wait_for(call, seconds), self.loop).result()

  def exec(
    self, command: str, *, cwd: str = WORKSPACE_PATH, timeout: int | None = None, env: dict | None = None
  ) -> ExecResult:
    """A command as LAB's podman sandbox runs one: a login shell under coreutils timeout, with the folders' names in its environment."""
    Podman.assert_sandbox_path(cwd)
    timeout = timeout or self.timeout
    places = {"DOCUMENTS_DIR": DOCUMENTS_PATH, "OUTPUT_DIR": OUTPUT_PATH, "WORKSPACE_DIR": WORKSPACE_PATH}
    wrapped = f"timeout --kill-after=2 {timeout} bash -lc {shlex.quote(command)}"
    try:
      # past the command's own timeout there is a resume from a snapshot and the calls' round trips to wait for
      done = self.wait(self.env.shell(wrapped, cwd=cwd, env=places | (env or {})), timeout + 180)
    except Exception as e:
      return ExecResult(stdout="", stderr=f"sandbox exec failed: {type(e).__name__}: {e}", returncode=1)
    if done.exit_code in (124, 137):  # what timeout exits with, having ended or killed the command
      return ExecResult(stdout=done.stdout, stderr=done.stderr, returncode=None, timed_out=True)
    return ExecResult(stdout=done.stdout, stderr=done.stderr, returncode=done.exit_code)

  def read_file(self, path: str) -> bytes:
    Podman.assert_sandbox_path(path)
    try:
      return self.wait(self.env.read_file_bytes(path), 240)
    except NotFoundError as e:
      raise FileNotFoundError(path) from e
    except EnvError as e:
      raise OSError(f"{path}: {e}") from e

  def write_file(self, path: str, content: bytes | str) -> None:
    Podman.assert_sandbox_path(path)
    if not Podman.is_writable(path):
      raise PermissionError(f"write denied: {path} is not under a writable mount")
    self.wait(self.env.write_file(path, content.encode() if isinstance(content, str) else content), 240)

  def exists(self, path: str) -> bool:
    try:
      Podman.assert_sandbox_path(path)
    except ValueError:
      return False
    return self.exec(f"test -e {shlex.quote(path)}", timeout=30).returncode == 0

  def search(self, **what) -> list[str]:
    done = self.exec(f"python3 -c {shlex.quote(SEARCH)} {shlex.quote(json.dumps(what))}", timeout=60)
    if done.returncode != 0:
      raise OSError(f"search failed: {done.stderr.strip()[-500:]}")
    return json.loads(done.stdout)


class Tools(ToolExecutor):
  """LAB's tools, with glob and grep run inside the sandbox. LAB's own walk the host's folders, which podman mounts into its container and a remote sandbox does not share."""

  def root(self, search_path: str | None) -> str | None:
    path = self._resolve_search_path(search_path)
    return (DOCUMENTS_PATH if path == "/" else path) if self.sandbox.exists(path) else None

  def _glob(self, pattern: str, search_path: str | None) -> str:
    if not pattern:
      return "Error: pattern is required"
    self.glob_count += 1
    if (root := self.root(search_path)) is None:
      return f"Error: path does not exist: {search_path}"
    found = self.sandbox.search(root=root, glob=pattern)
    return "\n".join(found) or f"No files matching '{pattern}' in {self._resolve_search_path(search_path)}"

  def _grep(self, pattern_str: str, search_path: str | None, file_glob: str | None, output_mode: str) -> str:
    if not pattern_str:
      return "Error: pattern is required"
    self.grep_count += 1
    if (root := self.root(search_path)) is None:
      return f"Error: path does not exist: {search_path}"
    try:
      re.compile(pattern_str)
    except re.error as e:
      return f"Error: invalid regex: {e}"
    found = self.sandbox.search(root=root, glob=file_glob or "**/*", regex=pattern_str, mode=output_mode)
    return "\n".join(found) or f"No matches for '{pattern_str}'"


class Sandbox:
  """One episode's environment and LAB's tools on it, suspended `idle` seconds after its last tool call ends; 0 never suspends it."""

  def __init__(self, endpoint: str, env, tools: Tools, idle: float):
    self.endpoint, self.env, self.tools, self.idle = endpoint, env, tools, idle
    self.calls = self.asked = self.suspends = 0
    self.resting = asyncio.Lock()  # held while the environment is being suspended, which a call waits out
    self.timers: set[asyncio.Task] = set()

  @contextlib.asynccontextmanager
  async def working(self):
    self.asked += 1
    async with self.resting:
      self.calls += 1
    try:
      yield
    finally:
      self.calls -= 1
      if not self.calls and self.idle:
        self.timers.add(timer := asyncio.create_task(self.rest(self.asked)))
        timer.add_done_callback(self.timers.discard)

  async def rest(self, asked: int) -> None:
    await asyncio.sleep(self.idle)
    async with self.resting:
      if self.calls or self.asked != asked:
        return
      try:
        await asyncio.wait_for(self.env.suspend(), 120)
        self.suspends += 1
      except Exception as e:  # it stays running, and the next call goes on as usual
        print(f"could not suspend {self.env.id}: {type(e).__name__}: {e}", flush=True)

  async def execute(self, name: str, arguments: str | dict) -> str:
    async with self.working():
      return await asyncio.to_thread(self.tools.execute, name, arguments)

  def metrics(self) -> dict:
    return self.tools.get_metrics() | {"substrate_suspends": self.suspends}

  async def close(self, output: Path) -> None:
    """Bring what the episode wrote home, and delete the environment."""
    self.idle = 0
    try:
      async with self.working(), asyncio.timeout(300):
        # packed outside the output folder, so the archive does not hold itself
        if (done := await self.env.shell(f"tar -C {OUTPUT_PATH} -cf /tmp/out.tar .")).exit_code:
          raise OSError(f"could not pack the outputs of {self.env.id}: {done.stderr.strip()[-500:]}")
        with tarfile.open(fileobj=io.BytesIO(await self.env.read_file_bytes("/tmp/out.tar"))) as tar:
          await asyncio.to_thread(tar.extractall, output, filter="data")  # which refuses members that leave the folder
    finally:
      for timer in self.timers:
        timer.cancel()
      await delete(self.endpoint, self.env.id)


def packed(folders: list[tuple[Path, str]]) -> bytes:
  out = io.BytesIO()
  with tarfile.open(fileobj=out, mode="w") as tar:
    for folder, name in folders:
      tar.add(folder, arcname=name)
  return out.getvalue()


async def open_sandbox(
  endpoint: str, image: str, idle: float, documents: Path, workspace: Path, output: Path
) -> Sandbox:
  """A new environment on `image`, a digest-pinned LAB sandbox image, holding the task's documents and the episode's workspace."""
  name = f"lab-{uuid.uuid4().hex[:16]}"
  LIVE.add(name)
  try:
    env = await asyncio.wait_for(client(endpoint).create(name, rootfs_image=image, atespace=SPACE), 300)
    files = await asyncio.to_thread(packed, [(workspace, "."), (documents, "documents")])
    async with asyncio.timeout(300):
      while (
        True
      ):  # a call to an environment that is still booting waits for it, or fails, and the client does not say which
        with contextlib.suppress(EnvError):
          await env.write_file("/tmp/in.tar", files)
          break
        await asyncio.sleep(1)
      unpack = f"mkdir -p {OUTPUT_PATH} && tar --no-same-owner -C {WORKSPACE_PATH} -xf /tmp/in.tar && rm /tmp/in.tar"
      if (done := await env.shell(unpack)).exit_code:
        raise RuntimeError(f"could not unpack the task's files in {name}: {done.stderr.strip()[-500:]}")
  except BaseException:
    with contextlib.suppress(Exception):
      await delete(endpoint, name)
    raise
  tools = Tools(
    sandbox=Files(env, asyncio.get_running_loop(), documents, workspace, output, timeout=60), shell_timeout=60
  )
  return Sandbox(endpoint, env, tools, idle)
