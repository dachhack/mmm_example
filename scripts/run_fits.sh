#!/usr/bin/env bash
# Heavy fits — run on a VM (multi-core PyMC). Produces docs/data/* for the Pages build.
set -euo pipefail
pip install -e ".[fit]"
mkdir -p artifacts docs/data
python -m draftzone_mmm.datagen
python -m draftzone_mmm.fit_bayes --out artifacts/idata.nc
python -m draftzone_mmm.experiment --all-channels --out artifacts/anchors.json
python -m draftzone_mmm.fit_bayes --anchors artifacts/anchors.json --out artifacts/idata_anchored.nc
python -m draftzone_mmm.evaluate --out docs/data/scorecard.json          # only reader of data_sealed/
python -m draftzone_mmm.export_dashboard_data --out docs/data/
echo "Fits complete. Commit/upload docs/data/* so the Pages workflow can publish."
