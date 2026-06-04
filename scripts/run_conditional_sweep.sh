#!/usr/bin/env bash
# scripts/run_conditional_sweep.sh — factorial sweep over a DATA CHARACTERISTIC (saturation), to
# answer "which engine works best for which kind of data?".
#
# For each (saturation level x seed): regenerate the world at that saturation, fit the fast engine
# set at best config, and snapshot the graded results tagged with the regime into
# docs/robustness/conditional/. scripts/conditional.py then aggregates per regime into a
# "which engine when" decision guide. Saturation is set with datagen --saturation-scale:
#   0.5 = MORE saturated (low headroom) · 1.0 = baseline · 2.0 = LESS saturated (headroom).
#
# Usage: bash scripts/run_conditional_sweep.sh "0.5 1.0 2.0" "11 22 33 44"
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
BRANCH="claude/draftzone-mmm-skeptics-guide-NrBPj"
SATS="$1"; SEEDS="$2"
say() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p docs/robustness/conditional artifacts

for SAT in $SATS; do
  for SEED in $SEEDS; do
    LOG="artifacts/cond_sat${SAT}_seed${SEED}.log"
    say ">>> sat=$SAT seed=$SEED"
    rm -f artifacts/*.json artifacts/*.nc artifacts/*.pkl artifacts/*.rds
    run() { if timeout 1200 "$@" >>"$LOG" 2>&1; then :; else say "  FAIL: $* (rc=$?)"; fi; }
    run python -m draftzone_mmm.datagen --seed "$SEED" --hetero-geos --spend-ladder --geo-panel --saturation-scale "$SAT"
    run python -m draftzone_mmm.fit_bayes --out artifacts/idata.nc
    run python -m draftzone_mmm.experiment --all-channels --out artifacts/anchors.json
    run python -m draftzone_mmm.fit_bayes --anchors artifacts/anchors.json --out artifacts/idata_anchored.nc
    run python -m draftzone_mmm.spend_ladder
    run python scripts/fit_robyn_style.py
    run python scripts/fit_meridian.py --mode national --seasonality fourier
    run python scripts/fit_meridian.py --mode national --seasonality aks
    python scripts/snapshot_results.py --out "docs/robustness/conditional/run_sat${SAT}_seed${SEED}.json" \
      2>&1 | tee -a "$LOG" | grep -E "Snapshot|MAE=" | head -10
    git add "docs/robustness/conditional/run_sat${SAT}_seed${SEED}.json" >/dev/null 2>&1
    git commit -q -m "conditional: snapshot sat=$SAT seed=$SEED" >/dev/null 2>&1 && git push -q origin "$BRANCH" >/dev/null 2>&1
    say "OK sat=$SAT seed=$SEED"
  done
done

say ">>> aggregating conditional decision guide"
python scripts/conditional.py
git add docs/conditional docs/robustness/conditional >/dev/null 2>&1
git commit -q -m "conditional: aggregate which-engine-when decision guide" >/dev/null 2>&1 && git push -q origin "$BRANCH" >/dev/null 2>&1
say "=== CONDITIONAL SWEEP COMPLETE ==="
