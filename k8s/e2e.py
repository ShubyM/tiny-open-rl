"""Break tinyrl on GKE on purpose and check it survives. Rerun after any change to know it still does.

Needs kubectl pointed at a cluster running k8s/ on at least two GPU nodes, and the tinker SDK. It port-forwards the
gateway, trains two LoRA runs through the SDK, then injects one failure at a time and checks what the user sees:

  trainer_killed   a trainer process dies: the node daemon starts another, calls fail cleanly, load_state recovers
  gateway_deleted  every gateway pod is deleted: the SDK's retries land and no call runs twice
  redis_restarted  Redis is down past the pool lease: runs survive on its append-only file and training goes on
  node_drained     a node is drained, as on a Spot notice: its trainer saves the run, which goes on on another node
  node_died        a node is reset with no notice: its pools move, calls fail cleanly, load_state recovers
  no_leaks         afterwards, one trainer process per placed pool and no lease without a pool

python k8s/e2e.py prints one line per check and writes the report to e2e.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import tinker
from tinker import types

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


def serving(pool: str) -> str | None:
  """The node daemon pod a pool is placed on. A node daemon's id is its pod name and pid."""
  node = redis("HGET", "tinyrl:place", pool)
  return node.rsplit("-", 1)[0] if node else None


def trainers(pod: str) -> list[str]:
  ps = kubectl("exec", pod, "--", "ps", "-eo", "args", check=False)
  return [line for line in ps.splitlines() if "tinyrl.trainer" in line or "tinyrl.sampler" in line]


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
    self.name = name
    self.tc = client.create_lora_training_client(MODEL, rank=8, seed=0, user_metadata={"openrl.exclusive": "true"})
    self.pool = self.tc.model_id  # an exclusive run has a pool of its own, named by its id
    ids = tok.encode("tinyrl keeps training when its pods, nodes and Redis go away.", add_special_tokens=False)
    self.datum = types.Datum(
      model_input=types.ModelInput.from_ints(ids[:-1]),
      loss_fn_inputs={"target_tokens": ids[1:], "weights": [1.0] * (len(ids) - 1)},
    )
    self.steps = 0

  def step(self) -> float:
    fb = self.tc.forward_backward([self.datum], "cross_entropy")
    self.tc.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    self.steps += 1
    return fb.result().metrics["loss:sum"]

  def server_steps(self) -> int:
    return int(redis("HGET", f"tinyrl:run:{self.pool}", "steps"))

  def fails_cleanly(self) -> str:
    try:
      self.step()
    except Exception as e:
      return str(e)
    raise AssertionError("expected the call to fail, since the run's unsaved state is gone")


def main() -> None:
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
    pod = serving(run.pool)
    kubectl("exec", pod, "--", "pkill", "-9", "-f", f"tinyrl.trainer {run.pool}")
    restart = wait("a new trainer", lambda: any(run.pool in t for t in trainers(pod)), 15)
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
    time.sleep(15)  # down longer than the 10 s pool lease, so every lease runs out
    kubectl("scale", "statefulset/redis", "--replicas=1")
    c, r = both_step()
    assert redis("EXISTS", f"tinyrl:run:{run.pool}") == "1"
    return f"runs survived, and both stepped after the restart (losses {c:.3f}, {r:.3f})"

  def node_drained():
    pod = serving(run.pool)
    node = daemons()[pod]
    kubectl("taint", "node", node, "tinyrl-e2e-drain=true:NoExecute")  # evicts the node daemon, as a Spot notice does
    try:
      wait("the run parked", lambda: redis("EXISTS", f"tinyrl:parked:{run.pool}") == "1", 40)
      moved = wait("the pool moved", lambda: serving(run.pool) not in (None, pod), 90)
      c, r = both_step()
      gap = abs(r - c) / c
      assert gap < 0.05, f"after the move the run's loss {r:.4f} is {gap:.1%} off the control's {c:.4f}"
      return f"parked, moved in {moved:.0f} s, next loss {r:.4f} vs control {c:.4f} ({gap:.2%} apart)"
    finally:
      kubectl("taint", "node", node, "tinyrl-e2e-drain-", check=False)

  def node_died():
    pod = serving(run.pool)
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
    moved = wait("the pool moved off the dead node", lambda: serving(run.pool) not in (None, pod), 120)
    error = run.fails_cleanly()
    assert "lost its unsaved steps" in error, error
    run.tc.load_state_with_optimizer(checkpoint).result()
    run.step()
    return f"moved in {moved:.0f} s, the call failed with the expected error, and load_state recovered"

  def no_leaks():
    time.sleep(10)
    placed = {pool: node.rsplit("-", 1)[0] for pool, node in zip(*[iter(redis("HGETALL", "tinyrl:place").split())] * 2)}
    procs = {pod: trainers(pod) for pod in daemons()}
    extra = {
      pod: [p for p in ps if not any(pool in p for pool, at in placed.items() if at == pod)]
      for pod, ps in procs.items()
    }
    missing = [pool for pool, pod in placed.items() if sum(pool in p for p in procs.get(pod, [])) != 1]
    owners = [k.removeprefix("tinyrl:owner:") for k in redis("KEYS", "tinyrl:owner:*").split()]
    leaked = [o for o in owners if o not in placed and o != "scheduler" and not o.startswith("gpu:")]
    assert not any(extra.values()) and not missing and not leaked, (extra, missing, leaked)
    return f"{len(placed)} pools, each with one process, and no stray processes or leases"

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
