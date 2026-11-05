"""Break tinyrl on GKE on purpose and check it survives. Rerun after any change to know it still does.

Needs kubectl pointed at a cluster running k8s/ on at least two GPU nodes, and the tinker SDK. It port-forwards the gateway, trains two LoRA runs through the SDK, then injects one failure at a time and checks what the user sees:

  trainer_killed   a trainer's supervisor dies: the node daemon starts another, calls fail cleanly, load_state recovers
  gateway_deleted  every gateway pod is deleted: the SDK's retries land and no call runs twice
  redis_restarted  Redis is down past the worker lease: runs survive on its append-only file and training goes on
  node_drained     a node is drained, as on a Spot notice: its trainer saves the run, which goes on on another node
  node_died        a node is reset with no notice: its workers move, calls fail cleanly, load_state recovers
  no_leaks         afterwards, one owned process group per trainer or live engine, and no stray processes or leases

python k8s/e2e.py prints one line per check and writes the report to e2e.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import tinker

NS = os.getenv("NAMESPACE", "tinyrl")
MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B")
PORT = 18000
PROJECT, ZONE = os.getenv("PROJECT", "open-rl-2-sandbox-102433"), os.getenv("ZONE", "us-central1-a")


def kubectl(*args: str, check: bool = True) -> str:
  out = subprocess.run(["kubectl", "-n", NS, *args], capture_output=True, text=True)
  if check and out.returncode:
    raise RuntimeError(f"kubectl {' '.join(args)}: {out.stderr}")
  return out.stdout


def redis(*args: str) -> str:
  return kubectl("exec", "redis-0", "--", "redis-cli", "--raw", *args).strip()


def port_forward() -> None:
  """Keep the gateway forwarded to localhost, starting kubectl again whenever the pod it picked goes away."""

  def loop():
    while True:
      subprocess.run(["kubectl", "-n", NS, "port-forward", "svc/gateway", f"{PORT}:8000"], capture_output=True)
      time.sleep(0.5)

  threading.Thread(target=loop, daemon=True).start()


def daemons() -> dict[str, str]:
  """Node daemon pod -> its node."""
  pods = json.loads(kubectl("get", "pods", "-l", "app=node-daemon", "-o", "json"))["items"]
  return {p["metadata"]["name"]: p["spec"].get("nodeName", "") for p in pods if p["status"].get("phase") == "Running"}


def placements() -> dict[str, str]:
  """Worker -> the node daemon pod the scheduler placed it on. A node daemon's id is its pod name and pid."""
  workers = json.loads(redis("GET", "tinyrl:state") or "{}").get("workers", {})
  return {worker: p["node"].rsplit("-", 1)[0] for worker, p in workers.items() if p["node"]}


def serving(worker: str) -> str | None:
  return placements().get(worker)


def live_workers() -> dict[tuple[str, str], dict]:
  """Physical (pod, worker) -> live state, including its owned supervisor PID. Placement may already name another node."""
  records = redis("HGETALL", "tinyrl:nodes").splitlines()
  workers = {}
  for node, raw in zip(records[::2], records[1::2]):
    info = json.loads(raw)
    if info["until"] <= time.time():
      continue
    for worker, state in info.get("workers", {}).items():
      key = node.rsplit("-", 1)[0], worker
      assert key not in workers, f"two live node daemons report {key}"
      workers[key] = state
  return workers


def processes(pod: str) -> dict[int, tuple[int, int, str]]:
  ps = kubectl("exec", pod, "--", "ps", "-ww", "-eo", "pid=,ppid=,pgid=,args=")
  return {
    int(pid): (int(parent), int(group), args)
    for pid, parent, group, args in (s.split(None, 3) for s in ps.splitlines())
  }


def trainer_process(worker: str) -> tuple[str, int]:
  pod = serving(worker)
  state = live_workers()[(pod, worker)]
  pid = state["pid"]
  assert state["kind"]["role"] == "trainer" and isinstance(pid, int) and pid > 1, state
  args = processes(pod)[pid][2]
  assert "-m tinyrl.control.process " in args and "-m tinyrl.workers.trainer " in args, args
  return pod, pid


def check_processes(placed: dict, workers: dict, procs: dict, owners: list[str]) -> None:
  assert set(workers) == {(pod, worker) for worker, pod in placed.items()}, (placed, workers)
  for pod, ps in procs.items():
    local = {worker: state for (at, worker), state in workers.items() if at == pod}
    missing = [worker for worker, s in local.items() if s["kind"]["role"] == "trainer" and s["pid"] is None]
    assert not missing, (pod, "trainers without a process", missing)
    expected = [s["pid"] for s in local.values() if s["pid"] is not None]
    supervisors = {pid for pid, (_, _, args) in ps.items() if "-m tinyrl.control.process " in args}
    assert len(expected) == len(set(expected)) and set(expected) == supervisors, (pod, expected, supervisors)
    owned, groups = set(expected), {ps[pid][1] for pid in expected}
    while children := {pid for pid, (parent, _, _) in ps.items() if parent in owned} - owned:
      owned.update(children)
    # torchrun ranks may have separate process groups; restored vLLM copies have their own PID namespace.
    names = ("tinyrl.workers.trainer", "tinyrl.workers.snapshot", "vllm serve", "VLLM::")
    extra = {
      pid: args
      for pid, (_, group, args) in ps.items()
      if any(name in args for name in names) and pid not in owned and group not in groups
    }
    assert not extra, (pod, "unowned model processes", extra)
  assert not (missing_pods := {pod for pod, _ in workers} - procs.keys()), missing_pods
  assert not (leaked := set(owners) - placed.keys()), ("leases without a worker", leaked)


def wait(what: str, ok, timeout: float) -> float:
  start = time.time()
  while time.time() - start < timeout:
    try:
      if ok():
        return time.time() - start
    except Exception:
      pass
    time.sleep(1)
  raise TimeoutError(f"{what} not within {timeout:.0f} s")


class Run:
  def __init__(self, client: tinker.ServiceClient, tok, name: str):
    from tinker import types

    self.name = name
    self.tc = client.create_lora_training_client(MODEL, rank=8, seed=0, user_metadata={"openrl.exclusive": "true"})
    self.run_id = self.tc.model_id
    ids = tok.encode("tinyrl keeps training when its pods, nodes and Redis go away.", add_special_tokens=False)
    self.datum = types.Datum(
      model_input=types.ModelInput.from_ints(ids[:-1]),
      loss_fn_inputs={"target_tokens": ids[1:], "weights": [1.0] * (len(ids) - 1)},
    )
    self.steps = 0

  @property
  def worker(self):
    w = redis("HGET", f"tinyrl:run:{self.run_id}", "worker")
    return w if w else self.run_id

  def step(self) -> float:
    from tinker import types

    fb = self.tc.forward_backward([self.datum], "cross_entropy")
    self.tc.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    self.steps += 1
    return fb.result().metrics["loss:sum"]

  def server_steps(self) -> int:
    return int(redis("HGET", f"tinyrl:run:{self.run_id}", "steps"))

  def fails_cleanly(self) -> str:
    try:
      self.step()
    except Exception as e:
      return str(e)
    raise AssertionError("expected the call to fail, since the run's unsaved state is gone")


def main() -> None:
  import tinker
  from tinker import types

  port_forward()
  wait(
    "gateway", lambda: subprocess.run(["curl", "-sf", f"http://127.0.0.1:{PORT}/api/v1/healthz"]).returncode == 0, 120
  )
  client = tinker.ServiceClient(api_key="tml-e2e", base_url=f"http://127.0.0.1:{PORT}")
  tok = __import__("transformers").AutoTokenizer.from_pretrained(MODEL)
  control, run = Run(client, tok, "control"), Run(client, tok, "run")
  report, losses = [], {"control": [], "run": []}

  def check(name: str, fn) -> None:
    start = time.time()
    try:
      details, ok = fn(), True
    except Exception as e:
      details, ok = f"{type(e).__name__}: {e}", False
    report.append({"check": name, "pass": ok, "seconds": round(time.time() - start, 1), "details": details})
    print(f"{'PASS' if ok else 'FAIL'} {name} ({time.time() - start:.0f} s): {details}", flush=True)

  def both_step() -> tuple[float, float]:
    c, r = control.step(), run.step()
    losses["control"].append(c)
    losses["run"].append(r)
    return c, r

  for _ in range(2):
    both_step()
  sampler = run.tc.save_weights_and_get_sampling_client()
  sampled = sampler.sample(
    types.ModelInput.from_ints(run.datum.model_input.to_ints()[:8]), 2, types.SamplingParams(max_tokens=4)
  )
  print(f"warmup losses {losses}, sampled {[list(s.tokens) for s in sampled.result().sequences]}", flush=True)
  checkpoint = run.tc.save_state("e2e").result().path

  def trainer_killed():
    pod, pid = trainer_process(run.worker)
    kubectl("exec", pod, "--", "kill", "-9", str(pid))
    restart = wait("a new trainer supervisor", lambda: trainer_process(run.worker) != (pod, pid), 15)
    error = run.fails_cleanly()
    assert "lost its unsaved steps" in error, error
    run.tc.load_state_with_optimizer(checkpoint).result()
    return f"restarted in {restart:.0f} s, then the call failed with the expected error, and load_state recovered"

  def gateway_deleted():
    before = run.server_steps()
    kubectl("delete", "pods", "-l", "app=gateway", "--wait=false")
    _, loss = both_step()  # the control steps too, so the two stay comparable
    assert run.server_steps() == before + 1, (before, run.server_steps())
    return f"a step during the restart landed once (steps {before} -> {before + 1}), loss {loss:.3f}"

  def redis_restarted():
    kubectl("scale", "statefulset/redis", "--replicas=0")
    kubectl("wait", "--for=delete", "pod/redis-0", "--timeout=120s", check=False)
    time.sleep(15)  # down longer than the 10 s worker lease, so every lease runs out
    kubectl("scale", "statefulset/redis", "--replicas=1")
    c, r = both_step()
    assert redis("EXISTS", f"tinyrl:run:{run.run_id}") == "1"
    return f"runs survived, and both stepped after the restart (losses {c:.3f}, {r:.3f})"

  def node_drained():
    pod = serving(run.worker)
    node = daemons()[pod]
    kubectl("taint", "node", node, "tinyrl-e2e-drain=true:NoExecute")  # evicts the node daemon, as a Spot notice does
    try:
      wait("the run parked", lambda: redis("EXISTS", f"tinyrl:parked:{run.run_id}") == "1", 40)
      moved = wait("the worker moved", lambda: serving(run.worker) not in (None, pod), 90)
      c, r = both_step()
      gap = abs(r - c) / c
      assert gap < 0.05, f"after the move the run's loss {r:.4f} is {gap:.1%} off the control's {c:.4f}"
      return f"parked, moved in {moved:.0f} s, next loss {r:.4f} vs control {c:.4f} ({gap:.2%} apart)"
    finally:
      kubectl("taint", "node", node, "tinyrl-e2e-drain-", check=False)

  def node_died():
    pod = serving(run.worker)
    node = daemons()[pod]
    token = subprocess.run(
      ["gcloud", "auth", "application-default", "print-access-token"], capture_output=True, text=True
    )
    url = f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}/instances/{node}/reset"
    subprocess.run(
      ["curl", "-sSf", "-X", "POST", "-H", f"Authorization: Bearer {token.stdout.strip()}", url],
      check=True,
      capture_output=True,
    )
    moved = wait("the worker moved off the dead node", lambda: serving(run.worker) not in (None, pod), 120)
    error = run.fails_cleanly()
    assert "lost its unsaved steps" in error, error
    run.tc.load_state_with_optimizer(checkpoint).result()
    run.step()
    return f"moved in {moved:.0f} s, the call failed with the expected error, and load_state recovered"

  def no_leaks():
    time.sleep(10)
    placed = placements()
    workers = live_workers()
    procs = {pod: processes(pod) for pod in daemons()}
    owners = [k.removeprefix("tinyrl:owner:") for k in redis("KEYS", "tinyrl:owner:*").split()]
    check_processes(placed, workers, procs, owners)
    engines = sum(s["pid"] is not None for s in workers.values())
    return f"{len(placed)} workers, {engines} owned process groups, and no stray processes or leases"

  for name, fn in [
    ("trainer_killed", trainer_killed),
    ("gateway_deleted", gateway_deleted),
    ("redis_restarted", redis_restarted),
    ("node_drained", node_drained),
    ("node_died", node_died),
    ("no_leaks", no_leaks),
  ]:
    check(name, fn)

  json.dump({"report": report, "losses": losses}, open("e2e.json", "w"), indent=1)
  sys.exit(0 if all(r["pass"] for r in report) else 1)


if __name__ == "__main__":
  main()
