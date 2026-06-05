#!/usr/bin/env bash
# scripts/run_one_cell.sh — run ONE graded dataset (a single conditional/robustness cell):
# fresh data at (seed, saturation, confound) -> the fast engine set at best config -> snapshot JSON.
#
# Shared by the local sweep (scripts/run_conditional_sweep.sh) and the CI matrix
# (.github/workflows/sweep.yml), so a cell is defined in exactly one place. The caller is responsible
# for the Python environment (a venv locally, the runner's pip env in CI) — python must be on PATH.
#
# Usage: bash scripts/run_one_cell.sh SEED SATURATION CONFOUND OUT_JSON
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SEED="$1"; SAT="$2"; CF="$3"; OUT="$4"
export RETICULATE_PYTHON="$(command -v python)"
mkdir -p artifacts "$(dirname "$OUT")"
rm -f artifacts/*.json artifacts/*.nc artifacts/*.pkl artifacts/*.rds   # no stale engines leak in
run() { if timeout "${CELL_TIMEOUT:-1800}" "$@"; then :; else echo "  FAIL: $* (rc=$?)" >&2; fi; }

run python -m draftzone_mmm.datagen --seed "$SEED" --hetero-geos --spend-ladder --geo-panel \
    --saturation-scale "$SAT" --confound "$CF"
run python -m draftzone_mmm.fit_bayes --out artifacts/idata.nc
run python -m draftzone_mmm.experiment --all-channels --out artifacts/anchors.json
run python -m draftzone_mmm.fit_bayes --anchors artifacts/anchors.json --out artifacts/idata_anchored.nc
run python -m draftzone_mmm.spend_ladder
run python scripts/fit_robyn_style.py
run python scripts/fit_meridian.py --mode national --seasonality fourier
run python scripts/fit_meridian.py --mode national --seasonality aks
python scripts/snapshot_results.py --out "$OUT"
