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
SCHEDULER = "tinyrl/control/scheduler.py"
LEDGER = "tinyrl/control/ledger.py"
NODES = "tinyrl/control/nodes.py"
SAMPLES = "tinyrl/control/samples.py"
FAIRNESS = "tinyrl/control/fairness.py"
ACCOUNTING = "tinyrl/control/accounting.py"
ORDER = "test.test_scheduler.TestLocalQuantum"
BILLING = "test.test_accounting.TestAccounting"
CLAIM = "test.test_ledger.TestLedger.test_reaped_boot_id_is_fenced_and_gets_reset_without_overwriting_checkpoint"
DEDUP = "test.test_ledger.TestLedger.test_competing_sampler_replicas_answer_and_charge_exactly_once"
RECOVERY = "test.test_trainer.TestTrainer.test_parked_gradients_and_the_recovery_warning_cannot_replay_after_a_failure"

# (source, function, original expression, faulty replacement, tests that should detect it)
CASES = [
  (FAIRNESS, "rank_runs", "-ready_lag(cmd.run)", "ready_lag(cmd.run)", ORDER),
  (FAIRNESS, "admit_samples", "while len(out) < slots and runs:", "while len(out) < 1 and runs:", BILLING),
  (FAIRNESS, "rank_runs", "entry_sort_key(cmd.entry_id)", "cmd.entry_id", ORDER),
  (SCHEDULER, "ready_lag", "self.service.get(account, 0.0)", "self.seen.get(account, 0.0)", ORDER),
  (ACCOUNTING, "_change_active", "1_000_000 * self.active", "1_000_000", BILLING),
  (ACCOUNTING, "_change_active", "* self.gpus", "* 1", BILLING),
  (LEDGER, "bill", "if charge and charge.gpu_ms", "if reply.ok and charge and charge.gpu_ms", BILLING),
  (NODES, "fence", "if boot_id in self.expired_boots", "if False", CLAIM),
  (SAMPLES, "complete", "entry.owner != owner", "False", DEDUP),
  (LEDGER, "finish_train", "if mutated:", "if reply.ok and mutated:", RECOVERY),
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
      source = (ROOT / path).read_text()
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
