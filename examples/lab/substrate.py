"""An episode's sandbox as an Agent Substrate environment: LAB's sandbox image, run as it is, with the task's files sent in and LAB's
tools run against it. The environment is suspended once nothing has been asked of it for a while, so it holds no worker while the
model generates, and the next call resumes it.

`SubstrateSandbox` is a tinker-cookbook sandbox (its SandboxInterface); `create_tools` gives LAB's tools on a new one.

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

ATESPACE = "ate-env"
OPEN_ENVS: set[str] = set()  # environments not yet deleted


@functools.cache
def client(endpoint: str) -> Client:
  loop = asyncio.get_running_loop()

  async def delete_all_and_exit():
    await asyncio.gather(*(delete(endpoint, name) for name in list(OPEN_ENVS)), return_exceptions=True)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTERM)

  loop.add_signal_handler(signal.SIGTERM, lambda: loop.create_task(delete_all_and_exit()))
  return Client(endpoint, auth_token=os.environ.get("SUBSTRATE_ENV_AUTH_TOKEN"))


async def delete(endpoint: str, name: str) -> None:
  with contextlib.suppress(NotFoundError):
    await asyncio.wait_for(client(endpoint).delete(name, atespace=ATESPACE), 60)
  OPEN_ENVS.discard(name)


class SubstrateSandbox:
  """One environment as a tinker-cookbook sandbox, suspended `idle` seconds after its last call ends; 0 never suspends it."""

  def __init__(self, endpoint: str, env, idle: float):
    self.endpoint, self.env, self.idle = endpoint, env, idle
    self.in_flight = self.call_seq = self.suspends = 0
    self.crashed: SandboxTerminatedError | None = None  # set once the environment has crashed
    self.suspend_lock = asyncio.Lock()  # held while the environment is being suspended, which a call waits out
    self.timers: set[asyncio.Task] = set()

  @property
  def sandbox_id(self) -> str:
    return self.env.id

  async def send_heartbeat(self, timeout: int = 30) -> None:
    pass  # an environment lives until it is deleted

  async def suspend_when_idle(self, call_seq: int) -> None:
    await asyncio.sleep(self.idle)
    async with self.suspend_lock:
      if self.in_flight or self.call_seq != call_seq:
        return
      try:
        await asyncio.wait_for(self.env.suspend(), 120)
        self.suspends += 1
      except Exception as e:  # it stays running, and the next call goes on as usual
        print(f"could not suspend {self.env.id}: {type(e).__name__}: {e}", flush=True)

  async def guest_call(self, coro, seconds: float):
    """One call to the environment, which wakes it. Substrate does not start a crashed environment again, so its episode cannot go
    on: a call that fails on one raises the cookbook's SandboxTerminatedError, and `crashed` keeps it for LabTools, whose base
    class turns every error into text for the model."""
    self.call_seq += 1
    async with self.suspend_lock:
      self.in_flight += 1
    try:
      return await asyncio.wait_for(coro, seconds)
    except Exception as e:
      with contextlib.suppress(EnvError, TimeoutError):
        if (await asyncio.wait_for(self.env.info(), 30)).status == EnvironmentStatus.CRASHED:
          self.crashed = SandboxTerminatedError(f"{self.env.id} crashed")
          raise self.crashed from e
      raise
    finally:
      self.in_flight -= 1
      if not self.in_flight and self.idle:
        self.timers.add(timer := asyncio.create_task(self.suspend_when_idle(self.call_seq)))
        timer.add_done_callback(self.timers.discard)

  async def run_command(self, command: str, workdir=None, timeout=60, max_output_bytes=None) -> SandboxResult:
    # past the command's own time there is a resume from a snapshot and the call's round trips to wait for
    done = await self.guest_call(self.env.shell(command, cwd=workdir or ""), timeout + 180)
    out, err = done.stdout[:max_output_bytes], done.stderr[:max_output_bytes]
    return SandboxResult(stdout=out, stderr=err, exit_code=done.exit_code)

  async def read_bytes(self, path: str, timeout: int = 240) -> bytes:
    return await self.guest_call(self.env.read_file_bytes(path), timeout)

  async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult:
    try:
      text = (await self.read_bytes(path, timeout))[:max_bytes].decode(errors="replace")
    except EnvError as e:
      return SandboxResult(stdout="", stderr=str(e), exit_code=1)
    return SandboxResult(stdout=text, stderr="", exit_code=0)

  async def write_file(self, path: str, content: str | bytes, executable=False, timeout=60) -> SandboxResult:
    data = content.encode() if isinstance(content, str) else content
    await self.guest_call(self.env.write_file(path, data, mode=0o755 if executable else 0o644), timeout)
    return SandboxResult(stdout="", stderr="", exit_code=0)

  async def cleanup(self) -> None:
    self.idle = 0
    for timer in self.timers:
      timer.cancel()
    await delete(self.endpoint, self.env.id)


class LabSandbox:
  """What LAB's tools ask of a sandbox, answered by a SubstrateSandbox. The tools run in a thread, and each call here runs on the event loop."""

  def __init__(self, substrate: SubstrateSandbox, loop, documents: Path, workspace: Path, output: Path, timeout: int):
    self.substrate, self.loop, self.timeout = substrate, loop, timeout
    # the host's folders, which LAB counts documents in
    self.documents_dir, self.workspace_dir, self.output_dir = documents, workspace, output

  def run_on_loop(self, coro):
    return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

  def exec(self, command: str, *, cwd=WORKSPACE_PATH, timeout=None, env=None) -> ExecResult:
    """A command as LAB's podman sandbox runs one: a login shell under coreutils timeout, with the folders' names in its environment."""
    Paths.assert_sandbox_path(cwd)
    timeout = timeout or self.timeout
    env_vars = {"DOCUMENTS_DIR": DOCUMENTS_PATH, "OUTPUT_DIR": OUTPUT_PATH, "WORKSPACE_DIR": WORKSPACE_PATH}
    env_vars |= env or {}
    exports = " ".join(f"{name}={shlex.quote(value)}" for name, value in env_vars.items())
    shell_line = f"env {exports} timeout --kill-after=2 {timeout} bash -lc {shlex.quote(command)}"
    try:
      done = self.run_on_loop(self.substrate.run_command(shell_line, workdir=cwd, timeout=timeout))
    except (EnvError, TimeoutError) as e:
      return ExecResult(stdout="", stderr=f"sandbox exec failed: {type(e).__name__}: {e}", returncode=1)
    if done.exit_code in (124, 137):  # what timeout exits with, having ended or killed the command
      return ExecResult(stdout=done.stdout, stderr=done.stderr, returncode=None, timed_out=True)
    return ExecResult(stdout=done.stdout, stderr=done.stderr, returncode=done.exit_code)

  def read_file(self, path: str) -> bytes:
    Paths.assert_sandbox_path(path)
    try:
      return self.run_on_loop(self.substrate.read_bytes(path))
    except NotFoundError as e:
      raise FileNotFoundError(path) from e
    except EnvError as e:
      raise OSError(f"{path}: {e}") from e

  def write_file(self, path: str, content: bytes | str) -> None:
    Paths.assert_sandbox_path(path)
    if not Paths.is_writable(path):
      raise PermissionError(f"write denied: {path} is not under a writable mount")
    self.run_on_loop(self.substrate.write_file(path, content, timeout=240))

  def exists(self, path: str) -> bool:
    try:
      Paths.assert_sandbox_path(path)
    except ValueError:
      return False
    return self.exec(f"test -e {shlex.quote(path)}", timeout=30).returncode == 0

  async def download(self) -> None:
    await copy_to_host(self.substrate, self.workspace_dir, self.output_dir)


class LabTools(ToolExecutor):
  """LAB's tools. Its glob and grep walk the host's folders, which podman mounts into its container. A remote sandbox shares no folder with the host, so what the episode has written is copied to the host before they look."""

  def execute(self, name: str, arguments: dict) -> str:
    result = super().execute(name, arguments)
    # LAB's tools turn every error into text for the model, a sandbox that has crashed among them
    if crashed := self.sandbox.substrate.crashed:
      raise crashed
    return result

  def _glob(self, *args) -> str:
    self.sandbox.run_on_loop(self.sandbox.download())
    return super()._glob(*args)

  def _grep(self, *args) -> str:
    self.sandbox.run_on_loop(self.sandbox.download())
    return super()._grep(*args)


def extract_tar(archive: bytes, folder: Path) -> None:
  """An archive from the sandbox as a folder that then holds exactly that. A member that would leave the folder, as a link out of it would, is left behind."""

  def safe_member(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
    try:
      return tarfile.data_filter(member, path)
    except tarfile.FilterError:
      return None

  shutil.rmtree(folder)
  folder.mkdir()
  with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    tar.extractall(folder, filter=safe_member)


async def copy_to_host(sandbox: SubstrateSandbox, workspace: Path, output: Path) -> None:
  """What the episode has written, as the host's two folders: LAB keeps the workspace and the output folder apart there. The task's documents are not copied, as tools do not change them."""
  # packed outside the folders, so no archive holds itself
  tar_cmd = f"tar -C {WORKSPACE_PATH} --exclude=./documents --exclude=./output -cf /tmp/workspace.tar ."
  tar_cmd += f" && tar -C {OUTPUT_PATH} -cf /tmp/output.tar ."
  if (done := await sandbox.run_command(tar_cmd, timeout=120)).exit_code:
    raise OSError(f"could not pack what {sandbox.sandbox_id} wrote: {done.stderr.strip()[-500:]}")
  for name, folder in (("workspace", workspace), ("output", output)):
    await asyncio.to_thread(extract_tar, await sandbox.read_bytes(f"/tmp/{name}.tar"), folder)


def make_tar(folders: list[tuple[Path, str]]) -> bytes:
  out = io.BytesIO()
  with tarfile.open(fileobj=out, mode="w") as tar:
    for folder, name in folders:
      tar.add(folder, arcname=name)
  return out.getvalue()


async def create_sandbox(endpoint: str, image: str, idle: float, documents: Path, workspace: Path) -> SubstrateSandbox:
  """A new environment on `image`, a digest-pinned LAB sandbox image, holding the task's documents and the episode's workspace."""
  name = f"lab-{uuid.uuid4().hex[:16]}"
  OPEN_ENVS.add(name)
  try:
    env = await asyncio.wait_for(client(endpoint).create(name, rootfs_image=image, atespace=ATESPACE), 300)
    sandbox = SubstrateSandbox(endpoint, env, idle)
    archive = await asyncio.to_thread(make_tar, [(workspace, "."), (documents, "documents")])
    async with asyncio.timeout(300):
      # a call to an environment that is still booting waits for it, or fails, and the client does not say which
      while True:
        with contextlib.suppress(EnvError):
          await sandbox.write_file("/tmp/in.tar", archive, timeout=240)
          break
        await asyncio.sleep(1)
      untar = f"mkdir -p {OUTPUT_PATH} && tar --no-same-owner -C {WORKSPACE_PATH} -xf /tmp/in.tar && rm /tmp/in.tar"
      if (done := await sandbox.run_command(untar, timeout=120)).exit_code:
        raise RuntimeError(f"could not unpack the task's files in {name}: {done.stderr.strip()[-500:]}")
  except BaseException:
    with contextlib.suppress(Exception):
      await delete(endpoint, name)
    raise
  return sandbox


async def create_tools(endpoint: str, image: str, idle: float, documents: Path, run: Path) -> LabTools:
  """LAB's tools on a new sandbox for one episode, whose host folders are run/workspace and run/output."""
  workspace, output = run / "workspace", run / "output"
  substrate = await create_sandbox(endpoint, image, idle, documents, workspace)
  files = LabSandbox(substrate, asyncio.get_running_loop(), documents, workspace, output, timeout=60)
  return LabTools(sandbox=files, shell_timeout=60)
