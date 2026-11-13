"""An episode's sandbox as an Agent Substrate environment: LAB's sandbox image, run as it is, with the task's files sent in and LAB's
tools run against it. The environment is suspended once nothing has been asked of it for a while, so it holds no worker while the
model generates, and the next call resumes it.

`Substrate` is a tinker-cookbook sandbox (its SandboxInterface); `Files` and `Tools` put LAB's tools on one.

Substrate keeps an environment until it is deleted and has no call that lists them, so a driver that is stopped deletes its own.
"""

import asyncio
import contextlib
import functools
import io
import os
import shlex
import shutil
import signal
import tarfile
import uuid
from pathlib import Path

from ate_env import Client, EnvError, NotFoundError
from ate_env.types import EnvironmentStatus
from harness.tools import ToolExecutor
from sandbox.sandbox import DOCUMENTS_PATH, OUTPUT_PATH, WORKSPACE_PATH, ExecResult
from sandbox.sandbox import Sandbox as Paths  # LAB's rules on sandbox paths, static methods of its local sandbox
from tinker_cookbook.sandbox import SandboxResult, SandboxTerminatedError

SPACE = "ate-env"
LIVE: set[str] = set()  # environments not yet deleted


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


class Substrate:
  """One environment as a tinker-cookbook sandbox, suspended `idle` seconds after its last call ends; 0 never suspends it."""

  def __init__(self, endpoint: str, env, idle: float):
    self.endpoint, self.env, self.idle = endpoint, env, idle
    self.calls = self.asked = self.suspends = 0
    self.gone: SandboxTerminatedError | None = None  # set once the environment has crashed
    self.resting = asyncio.Lock()  # held while the environment is being suspended, which a call waits out
    self.timers: set[asyncio.Task] = set()

  @property
  def sandbox_id(self) -> str:
    return self.env.id

  async def send_heartbeat(self, timeout: int = 30) -> None:
    pass  # an environment lives until it is deleted

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

  async def ask(self, call, seconds: float):
    """One call to the environment, which wakes it. Substrate does not start a crashed environment again, so its episode cannot go
    on: a call that fails on one raises the cookbook's SandboxTerminatedError, and `gone` keeps it for callers that LAB's tools,
    which turn every error into text for the model, stand between."""
    self.asked += 1
    async with self.resting:
      self.calls += 1
    try:
      return await asyncio.wait_for(call, seconds)
    except Exception as e:
      with contextlib.suppress(EnvError, TimeoutError):
        if (await asyncio.wait_for(self.env.info(), 30)).status == EnvironmentStatus.CRASHED:
          self.gone = SandboxTerminatedError(f"{self.env.id} crashed")
          raise self.gone from e
      raise
    finally:
      self.calls -= 1
      if not self.calls and self.idle:
        self.timers.add(timer := asyncio.create_task(self.rest(self.asked)))
        timer.add_done_callback(self.timers.discard)

  async def run_command(self, command: str, workdir=None, timeout=60, max_output_bytes=None) -> SandboxResult:
    # past the command's own time there is a resume from a snapshot and the call's round trips to wait for
    done = await self.ask(self.env.shell(command, cwd=workdir or ""), timeout + 180)
    out, err = done.stdout[:max_output_bytes], done.stderr[:max_output_bytes]
    return SandboxResult(stdout=out, stderr=err, exit_code=done.exit_code)

  async def read_bytes(self, path: str, timeout: int = 240) -> bytes:
    return await self.ask(self.env.read_file_bytes(path), timeout)

  async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult:
    try:
      text = (await self.read_bytes(path, timeout))[:max_bytes].decode(errors="replace")
    except EnvError as e:
      return SandboxResult(stdout="", stderr=str(e), exit_code=1)
    return SandboxResult(stdout=text, stderr="", exit_code=0)

  async def write_file(self, path: str, content: str | bytes, executable=False, timeout=60) -> SandboxResult:
    data = content.encode() if isinstance(content, str) else content
    await self.ask(self.env.write_file(path, data, mode=0o755 if executable else 0o644), timeout)
    return SandboxResult(stdout="", stderr="", exit_code=0)

  async def cleanup(self) -> None:
    self.idle = 0
    for timer in self.timers:
      timer.cancel()
    await delete(self.endpoint, self.env.id)


class Files:
  """What LAB's tools ask of a sandbox, answered by a Substrate. The tools run in a thread, and each call here runs on the event loop."""

  def __init__(self, box: Substrate, loop, documents: Path, workspace: Path, output: Path, timeout: int):
    self.box, self.loop, self.timeout = box, loop, timeout
    # the host's folders, which LAB counts documents in
    self.documents_dir, self.workspace_dir, self.output_dir = documents, workspace, output

  def wait(self, call):
    return asyncio.run_coroutine_threadsafe(call, self.loop).result()

  def exec(self, command: str, *, cwd=WORKSPACE_PATH, timeout=None, env=None) -> ExecResult:
    """A command as LAB's podman sandbox runs one: a login shell under coreutils timeout, with the folders' names in its environment."""
    Paths.assert_sandbox_path(cwd)
    timeout = timeout or self.timeout
    places = {"DOCUMENTS_DIR": DOCUMENTS_PATH, "OUTPUT_DIR": OUTPUT_PATH, "WORKSPACE_DIR": WORKSPACE_PATH} | (env or {})
    named = " ".join(f"{name}={shlex.quote(value)}" for name, value in places.items())
    wrapped = f"env {named} timeout --kill-after=2 {timeout} bash -lc {shlex.quote(command)}"
    try:
      done = self.wait(self.box.run_command(wrapped, workdir=cwd, timeout=timeout))
    except (EnvError, TimeoutError) as e:
      return ExecResult(stdout="", stderr=f"sandbox exec failed: {type(e).__name__}: {e}", returncode=1)
    if done.exit_code in (124, 137):  # what timeout exits with, having ended or killed the command
      return ExecResult(stdout=done.stdout, stderr=done.stderr, returncode=None, timed_out=True)
    return ExecResult(stdout=done.stdout, stderr=done.stderr, returncode=done.exit_code)

  def read_file(self, path: str) -> bytes:
    Paths.assert_sandbox_path(path)
    try:
      return self.wait(self.box.read_bytes(path))
    except NotFoundError as e:
      raise FileNotFoundError(path) from e
    except EnvError as e:
      raise OSError(f"{path}: {e}") from e

  def write_file(self, path: str, content: bytes | str) -> None:
    Paths.assert_sandbox_path(path)
    if not Paths.is_writable(path):
      raise PermissionError(f"write denied: {path} is not under a writable mount")
    self.wait(self.box.write_file(path, content, timeout=240))

  def exists(self, path: str) -> bool:
    try:
      Paths.assert_sandbox_path(path)
    except ValueError:
      return False
    return self.exec(f"test -e {shlex.quote(path)}", timeout=30).returncode == 0

  def bring_home(self) -> None:
    self.wait(bring_home(self.box, self.workspace_dir, self.output_dir))


class Tools(ToolExecutor):
  """LAB's tools. Its glob and grep walk the host's folders, which podman mounts into its container. A remote sandbox shares no folder with the host, so what the episode has written is brought home before they look."""

  def _glob(self, *args) -> str:
    self.sandbox.bring_home()
    return super()._glob(*args)

  def _grep(self, *args) -> str:
    self.sandbox.bring_home()
    return super()._grep(*args)


def unpack(archive: bytes, folder: Path) -> None:
  """An archive from the sandbox as a folder that then holds exactly that. A member that would leave the folder, as a link out of it would, is left behind."""

  def inside(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
    try:
      return tarfile.data_filter(member, path)
    except tarfile.FilterError:
      return None

  shutil.rmtree(folder)
  folder.mkdir()
  with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    tar.extractall(folder, filter=inside)


async def bring_home(box: Substrate, workspace: Path, output: Path) -> None:
  """What the episode has written, as the host's two folders: LAB keeps the workspace and the output folder apart there. The task's documents are not brought, as tools do not change them."""
  # packed outside the folders, so no archive holds itself
  pack = f"tar -C {WORKSPACE_PATH} --exclude=./documents --exclude=./output -cf /tmp/workspace.tar ."
  if (done := await box.run_command(f"{pack} && tar -C {OUTPUT_PATH} -cf /tmp/output.tar .", timeout=120)).exit_code:
    raise OSError(f"could not pack what {box.sandbox_id} wrote: {done.stderr.strip()[-500:]}")
  for name, folder in (("workspace", workspace), ("output", output)):
    await asyncio.to_thread(unpack, await box.read_bytes(f"/tmp/{name}.tar"), folder)


def packed(folders: list[tuple[Path, str]]) -> bytes:
  out = io.BytesIO()
  with tarfile.open(fileobj=out, mode="w") as tar:
    for folder, name in folders:
      tar.add(folder, arcname=name)
  return out.getvalue()


async def open_sandbox(endpoint: str, image: str, idle: float, documents: Path, workspace: Path) -> Substrate:
  """A new environment on `image`, a digest-pinned LAB sandbox image, holding the task's documents and the episode's workspace."""
  name = f"lab-{uuid.uuid4().hex[:16]}"
  LIVE.add(name)
  try:
    env = await asyncio.wait_for(client(endpoint).create(name, rootfs_image=image, atespace=SPACE), 300)
    box = Substrate(endpoint, env, idle)
    files = await asyncio.to_thread(packed, [(workspace, "."), (documents, "documents")])
    async with asyncio.timeout(300):
      # a call to an environment that is still booting waits for it, or fails, and the client does not say which
      while True:
        with contextlib.suppress(EnvError):
          await box.write_file("/tmp/in.tar", files, timeout=240)
          break
        await asyncio.sleep(1)
      unpack = f"mkdir -p {OUTPUT_PATH} && tar --no-same-owner -C {WORKSPACE_PATH} -xf /tmp/in.tar && rm /tmp/in.tar"
      if (done := await box.run_command(unpack, timeout=120)).exit_code:
        raise RuntimeError(f"could not unpack the task's files in {name}: {done.stderr.strip()[-500:]}")
  except BaseException:
    with contextlib.suppress(Exception):
      await delete(endpoint, name)
    raise
  return box
