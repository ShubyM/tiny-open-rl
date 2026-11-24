"""Run targeted control-plane mutations against the production code in a disposable copy."""

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = "tinyrl/control/controller.py"
POLICY = "tinyrl/control/policy.py"
ACCOUNTING = "tinyrl/control/accounting.py"
ORDER = "test.test_scheduler.TestLocalQuantum"
BILLING = "test.test_accounting.TestAccounting"
CLAIM = "test.test_ledger.TestLedger.test_reaped_boot_id_is_fenced_and_gets_reset_without_overwriting_checkpoint"
DEDUP = "test.test_ledger.TestLedger.test_competing_sampler_replicas_answer_and_charge_exactly_once"
RECOVERY = "test.test_trainer.TestTrainer.test_parked_gradients_and_the_recovery_warning_cannot_replay_after_a_failure"

# (source, function, original expression, faulty replacement, tests that should detect it)
CASES = [
  (POLICY, "rank_runs", "-lag_of(cmd.run)", "lag_of(cmd.run)", ORDER),
  (
    POLICY,
    "admit_samples",
    "while len(admitted) < slots and run_queues:",
    "while len(admitted) < 1 and run_queues:",
    ORDER + ".test_sampler_wave_obeys_account_and_version_limits_and_rotates_within_them",
  ),
  (POLICY, "rank_runs", "entry_sort_key(cmd.entry_id)", "cmd.entry_id", ORDER),
  (POLICY, "ready_lag", "service.get(acct, 0.0)", "seen.get(acct, 0.0)", ORDER),
  (ACCOUNTING, "_change_active", "1_000_000 * self.active", "1_000_000", BILLING),
  (ACCOUNTING, "_change_active", "* self.gpus", "* 1", BILLING),
  (CONTROLLER, "apply_charge", "if charge.gpu_ms:", "if reply.ok and charge.gpu_ms:", BILLING),
  (CONTROLLER, "handle", "if self.db.is_fenced(msg.boot_id):", "if False:", CLAIM),
  (CONTROLLER, "finish_sample", "item.owner != owner or item.attempt != attempt", "item.attempt != attempt", DEDUP),
  (CONTROLLER, "finish_train", "if mutated:", "if reply.ok and mutated:", RECOVERY),
]


def mutate(source, function, original, replacement):
  functions = [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
  [target] = [n for n in functions if n.name == function]
  lines = source.splitlines(keepends=True)
  body = "".join(lines[target.lineno - 1 : target.end_lineno])
  if body.count(original) != 1:
    raise ValueError(f"update mutation for {function}: expected one occurrence of {original!r}")
  lines[target.lineno - 1 : target.end_lineno] = [body.replace(original, replacement)]
  return "".join(lines)


def run_tests(folder, tests):
  env = os.environ.copy()
  env.pop("TINYRL_DB", None)
  env.pop("CONTROL_URL", None)
  env.pop("PYTHONPATH", None)
  env["PYTHONPATH"] = str(folder)
  env["TMPDIR"] = str(folder)
  return subprocess.run(
    [sys.executable, "-B", "-m", "test.mutation", *tests],
    cwd=folder,
    env=env,
    capture_output=True,
    text=True,
    timeout=120,
  )


def measure():
  started = time.monotonic()
  killed = 0
  with tempfile.TemporaryDirectory(prefix="tinyrl-mutation-") as directory:
    folder = Path(directory)
    for name in ("tinyrl", "test"):
      shutil.copytree(ROOT / name, folder / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    baseline = run_tests(folder, sorted({case[-1] for case in CASES}))
    if baseline.returncode:
      raise SystemExit(f"Clean baseline failed:\n{baseline.stderr}")
    for path, function, old, new, test in CASES:
      source = (folder / path).read_text()
      (folder / path).write_text(mutate(source, function, old, new))
      result = run_tests(folder, [test])
      (folder / path).write_text(source)
      if result.returncode not in (0, 10):
        raise SystemExit(f"Inconclusive mutation in {function}:\n{result.stderr}")
      detected = result.returncode == 10
      killed += detected
      print(f"{'KILLED' if detected else 'SURVIVED'} {function}: {old} -> {new}", flush=True)
  print(f"{killed}/{len(CASES)} targeted mutants killed in {time.monotonic() - started:.1f}s")
  return int(killed != len(CASES))


if __name__ == "__main__":
  if len(sys.argv) == 1:
    sys.exit(measure())
  suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
  result = unittest.TextTestRunner().run(suite)
  if not result.testsRun or result.errors or result.skipped or result.unexpectedSuccesses:
    sys.exit(2)
  sys.exit(10 if result.failures else 0)
