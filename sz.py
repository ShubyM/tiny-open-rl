#!/usr/bin/env python3
# lines and tokens per file, counted the way tinygrad counts them. MAX_LINE_COUNT=n fails past n lines
import os, sys, token, tokenize
from tabulate import tabulate

TOKEN_WHITELIST = [token.OP, token.NAME, token.NUMBER, token.STRING]
def is_docstring(t): return t.type == token.STRING and t.string.startswith('"""') and t.line.strip().startswith('"""')

def gen_stats(base_path="."):
  table = []
  for path, _, files in os.walk(os.path.join(base_path, "tinyrl")):
    for name in files:
      if not name.endswith(".py") or "tinyrl/proto" in path.replace("\\", "/"): continue
      with tokenize.open(filepath:=os.path.join(path, name)) as f:
        tokens = [t for t in tokenize.generate_tokens(f.readline) if t.type in TOKEN_WHITELIST and not is_docstring(t)]
      if line_count:=len(set(x for t in tokens for x in range(t.start[0], t.end[0]+1))):
        table.append([os.path.relpath(filepath, base_path), line_count, len(tokens)/line_count])
  return table

if __name__ == "__main__":
  table = gen_stats(sys.argv[1] if len(sys.argv) > 1 else ".")
  print(tabulate([["Name", "Lines", "Tokens/Line"]] + sorted(table, key=lambda x: -x[1]), headers="firstrow", floatfmt=".1f")+"\n")
  print(f"total lines: {(total:=sum(x[1] for x in table))}")
  assert (max_lines:=int(os.getenv("MAX_LINE_COUNT", "-1"))) == -1 or total <= max_lines, f"OVER {max_lines} LINES"
