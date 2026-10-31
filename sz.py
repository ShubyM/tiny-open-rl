#!/usr/bin/env python3
"""Count the lines of code in each tinyrl file, the way tinygrad counts them: lines with a token on them, not counting
blank lines, comments or docstrings. MAX_LINE_COUNT=n fails if the total is over n."""

import os
import sys
import token
import tokenize

from tabulate import tabulate

CODE_TOKENS = (token.OP, token.NAME, token.NUMBER, token.STRING)


def is_docstring(t: tokenize.TokenInfo) -> bool:
  return t.type == token.STRING and t.string.startswith('"""') and t.line.strip().startswith('"""')


def count(path: str) -> tuple[int, int]:
  """The lines of code in a file, and its tokens."""
  with tokenize.open(path) as f:
    tokens = [t for t in tokenize.generate_tokens(f.readline) if t.type in CODE_TOKENS and not is_docstring(t)]
  lines = {line for t in tokens for line in range(t.start[0], t.end[0] + 1)}
  return len(lines), len(tokens)


def stats(base: str = ".") -> list[list]:
  table = []
  for directory, _, files in os.walk(os.path.join(base, "tinyrl")):
    if "tinyrl/api/proto" in directory.replace("\\", "/"):  # generated
      continue
    for name in files:
      if name.endswith(".py"):
        path = os.path.join(directory, name)
        lines, tokens = count(path)
        if lines:
          table.append([os.path.relpath(path, base), lines, tokens / lines])
  return sorted(table, key=lambda row: -row[1])


if __name__ == "__main__":
  table = stats(sys.argv[1] if len(sys.argv) > 1 else ".")
  print(tabulate(table, headers=["Name", "Lines", "Tokens/Line"], floatfmt=".1f") + "\n")

  total = sum(row[1] for row in table)
  print(f"total lines: {total}")
  max_lines = int(os.getenv("MAX_LINE_COUNT", "-1"))
  assert max_lines == -1 or total <= max_lines, f"OVER {max_lines} LINES"
