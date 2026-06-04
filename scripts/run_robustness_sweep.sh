#!/usr/bin/env bash
# scripts/run_robustness_sweep.sh — multi-seed robustness sweep over the FAST national engines.
#
# For each seed: regenerate everything, fit the engines that are fast enough to run at their best
# config across many datasets (naive, frequentist, PyMC obs + curve-aware anchored, spend ladder,
# Robyn-style, Meridian national Fourier + AKS), snapshot the graded results, commit. The slow
# deep-dives (real Meta Robyn 2000x5, the geo-panel control spectrum) stay single-seed documented
# results — Robyn's METHOD is represented in the sweep by the Robyn-style reimplementation.
#
# Each seed is an independent world, so artifacts are WIPED first to guarantee the snapshot grades
# only this seed's fits (no stale results from a previous seed leaking in).
#
# Usage: bash scripts/run_robustness_sweep.sh 101 202 303 404 505
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
BRANCH="claude/draftzone-mmm-skeptics-guide-NrBPj"
say() { echo "[$(date +%H:%M:%S)] $*"; }

for SEED in "$@"; do
  LOG="artifacts/sweep_${SEED}.log"; mkdir -p artifacts
  say ">>> SEED $SEED"
  rm -f artifacts/*.json artifacts/*.nc artifacts/*.pkl artifacts/*.rds
  run() { if timeout 1200 "$@" >>"$LOG" 2>&1; then :; else say "  FAIL: $* (rc=$?)"; fi; }
  run python -m draftzone_mmm.datagen --seed "$SEED" --hetero-geos --spend-ladder --geo-panel
  run python -m draftzone_mmm.fit_bayes --out artifacts/idata.nc
  run python -m draftzone_mmm.experiment --all-channels --out artifacts/anchors.json
  run python -m draftzone_mmm.fit_bayes --anchors artifacts/anchors.json --out artifacts/idata_anchored.nc
  run python -m draftzone_mmm.spend_ladder
  run python scripts/fit_robyn_style.py
  run python scripts/fit_meridian.py --mode national --seasonality fourier
  run python scripts/fit_meridian.py --mode national --seasonality aks
  python scripts/snapshot_results.py 2>&1 | tee -a "$LOG" | grep -E "Snapshot|MAE=" | head -12
  git add docs/robustness/run_${SEED}.json >/dev/null 2>&1
  git commit -q -m "robustness: snapshot seed ${SEED}" >/dev/null 2>&1 && git push -q origin "$BRANCH" >/dev/null 2>&1
  say "OK SEED $SEED snapshotted + pushed"
done

say ">>> aggregating"
python scripts/robustness.py
git add docs/robustness >/dev/null 2>&1
git commit -q -m "robustness: aggregate across seeds" >/dev/null 2>&1 && git push -q origin "$BRANCH" >/dev/null 2>&1
say "=== SWEEP COMPLETE ==="
