#!/usr/bin/env bash
# scripts/run_all_engines.sh — the full optimized pipeline, end to end, for ONE dataset.
#
# Generates a (fresh) synthetic dataset, then fits EVERY engine at its best-learned configuration
# and grades all of them against the same sealed truth. Each engine's lesson is baked in:
#   - PyMC: observational + curve-aware experiment anchor
#   - Spend ladder: bracketed multi-cell curve fit (cracks saturated channels)
#   - Meridian: national Fourier AND its idiomatic AKS; geo panel (hardened confounder) with no
#     control, an imperfect demand proxy, and a near-perfect one (the control-quality spectrum)
#   - Meta Robyn (real R package): converged 2000x5, plain AND experiment-calibrated to the ladder
#   - Robyn-style Python reimplementation
# Robust for unattended runs: every engine is isolated (a failure logs and the batch continues),
# and results are committed/pushed at checkpoints so partial progress survives a container reclaim.
#
# Usage: bash scripts/run_all_engines.sh [SEED] [ROBYN_ITERS] [ROBYN_TRIALS]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SEED="${1:-2025}"; RITERS="${2:-2000}"; RTRIALS="${3:-5}"
LABEL="seed${SEED}"
LOG="artifacts/run_${SEED}.log"; mkdir -p artifacts
source .venv/bin/activate
export RETICULATE_PYTHON="$(which python)"
BRANCH="claude/draftzone-mmm-skeptics-guide-NrBPj"

say()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
step() { local name="$1"; shift; say ">>> $name"; if timeout "${TIMEOUT:-2400}" "$@" >>"$LOG" 2>&1; then say "OK  $name"; else say "FAIL $name (rc=$?) — continuing"; fi; }
ckpt() { git add -A >/dev/null 2>&1; git commit -q -m "run($SEED): checkpoint — $1" >/dev/null 2>&1 && git push -q -u origin "$BRANCH" >/dev/null 2>&1; say "checkpoint committed: $1"; }

say "=== FULL PIPELINE RUN seed=$SEED robyn=${RITERS}x${RTRIALS} ==="

# 1) fresh data (national + hetero geos + spend ladder + hardened geo panel)
step "datagen" python -m draftzone_mmm.datagen --seed "$SEED" --hetero-geos --spend-ladder --geo-panel

# 2) fast / core engines
step "pymc_obs"      python -m draftzone_mmm.fit_bayes --out artifacts/idata.nc
step "experiments"   python -m draftzone_mmm.experiment --all-channels --out artifacts/anchors.json
step "pymc_anchored" python -m draftzone_mmm.fit_bayes --anchors artifacts/anchors.json --out artifacts/idata_anchored.nc
step "spend_ladder"  python -m draftzone_mmm.spend_ladder
step "robyn_style"   python scripts/fit_robyn_style.py
step "evaluate"      python -m draftzone_mmm.evaluate --out docs/data/scorecard.json
step "figures"       python -m draftzone_mmm.export_dashboard_data --out docs/data/
step "leaderboard_1" python scripts/engine_leaderboard.py
ckpt "core engines (pymc, ladder, robyn-style)"

# 3) Meridian — national (Fourier + idiomatic AKS)
step "meridian_fourier" python scripts/fit_meridian.py --mode national --seasonality fourier
step "meridian_aks"     python scripts/fit_meridian.py --mode national --seasonality aks
step "leaderboard_2"    python scripts/engine_leaderboard.py
ckpt "meridian national"

# 4) Meridian — geo panel, control-quality spectrum
TIMEOUT=3000 step "meridian_geo"       python scripts/fit_meridian.py --mode geo
TIMEOUT=3000 step "meridian_geo_ctrl"  python scripts/fit_meridian.py --mode geo --demand-control
TIMEOUT=3000 step "meridian_geo_ctrlhi" python scripts/fit_meridian.py --mode geo --demand-control demand_proxy_hi
step "leaderboard_3" python scripts/engine_leaderboard.py
ckpt "meridian geo spectrum"

# 5) Meta Robyn — real package, converged, plain + experiment-calibrated
TIMEOUT=3600 step "meta_robyn"            Rscript scripts/fit_meta_robyn.R --iterations "$RITERS" --trials "$RTRIALS" --cores 4
TIMEOUT=3600 step "meta_robyn_calibrated" Rscript scripts/fit_meta_robyn.R --calibrate --iterations "$RITERS" --trials "$RTRIALS" --cores 4
step "leaderboard_4" python scripts/engine_leaderboard.py
ckpt "meta robyn (plain + calibrated)"

# 6) reports
step "ladder_report" python scripts/spend_ladder_report.py
step "run_report"    python scripts/make_report.py --label "$LABEL"
ckpt "reports"

say "=== ALL DONE seed=$SEED ==="
