"""scripts/ci_matrix.py — emit a GitHub Actions matrix for the parallel sweep.

Turns space-separated saturations × confounds × seeds into one matrix entry per (sat, cf, seed) cell,
written as a `matrix=<json>` line for $GITHUB_OUTPUT. Each cell runs as its own runner, so an N-cell
sweep finishes in ~one-cell wall-clock instead of N×.

Usage: python scripts/ci_matrix.py "0.5 1.0 2.0" "0.6 0.3" "11 22 33 44 55"
"""
import json
import sys

sats = sys.argv[1].split() if len(sys.argv) > 1 else ["1.0"]
cfs = sys.argv[2].split() if len(sys.argv) > 2 else ["0.6"]
seeds = sys.argv[3].split() if len(sys.argv) > 3 else ["11"]

cells = [{"sat": s, "cf": c, "seed": d, "tag": f"sat{s}_cf{c}_seed{d}"}
         for s in sats for c in cfs for d in seeds]
print("matrix=" + json.dumps({"cell": cells}))
print(f"count={len(cells)}", file=sys.stderr)
