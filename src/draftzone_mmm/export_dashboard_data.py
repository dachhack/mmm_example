"""draftzone_mmm.export_dashboard_data — write the dashboard data contracts.

Produces (per docs/DASHBOARD_SPEC.md):
  docs/data/decomposition.json  per-week, per-channel posterior-mean contribution + bands
  docs/data/repair.json         per-channel before/after-anchor contribution distributions
  docs/data/roi.json            avg & marginal ROI with intervals
  docs/data/optim_draws.json    per-draw optimal allocations (interactive optimizer)
  docs/data/timeseries.csv      the public national dataset for the charts

scorecard.json is produced separately by evaluate.py (the only truth-reading step).

CONTRACT: part of the modeling pipeline; MUST NOT read the sealed answer key.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import numpy as np

from .model import ARTIFACTS, CHANNELS, COLORS, DATA_DIR, decompose, load_idata, load_national, stacked_draws
from .optimize import optimize_budget
from .revenue import compute_roi


def _decomposition(df, idata):
    draws, idx = stacked_draws(idata, max_draws=600)
    dec = decompose(df, draws, idx)
    weeks = [d.strftime("%Y-%m-%d") for d in df["week"]]
    mu = dec["mu"]
    nonmedia = dec["baseline"][:, None] + dec["controls"]
    out = dict(
        weeks=weeks,
        observed=[round(v, 1) for v in df["conversions"].to_numpy(float)],
        mu_mean=[round(v, 1) for v in mu.mean(0)],
        mu_lo=[round(v, 1) for v in np.percentile(mu, 5.5, 0)],
        mu_hi=[round(v, 1) for v in np.percentile(mu, 94.5, 0)],
        nonmedia_mean=[round(v, 1) for v in nonmedia.mean(0)],
        colors=COLORS,
        channels={c: {"mean": [round(v, 1) for v in dec["channel"][c].mean(0)]} for c in CHANNELS},
    )
    return out


def _avg_contrib_draws(df, idata, channels=CHANNELS, max_draws=400):
    draws, idx = stacked_draws(idata, max_draws=max_draws)
    dec = decompose(df, draws, idx, channels=channels)
    return {c: dec["channel"][c].mean(1) for c in channels}  # per-draw avg weekly contribution


def _repair(df, idata_before, idata_after, anchors):
    before = _avg_contrib_draws(df, idata_before)
    after = _avg_contrib_draws(df, idata_after)

    def summ(arr):
        return dict(mean=round(float(arr.mean()), 1),
                    lo=round(float(np.percentile(arr, 5.5)), 1),
                    hi=round(float(np.percentile(arr, 94.5)), 1),
                    samples=[round(float(v), 1) for v in arr[:200]])

    channels = []
    for c in CHANNELS:
        anc = anchors.get(c, {})
        channels.append(dict(
            channel=c, color=COLORS[c],
            before=summ(before[c]), after=summ(after[c]),
            anchor_mu=round(float(anc.get("prior_mu", 0)), 1) if anc else None,
            anchor_sd=round(float(anc.get("prior_sd", 0)), 1) if anc else None,
        ))
    return dict(channels=channels,
                note="Each channel's geo-experiment pulls its estimate toward a confound-immune "
                     "anchor; the whole decomposition tightens as experiments are fed in.")


def main():
    ap = argparse.ArgumentParser(description="Export dashboard data contracts.")
    ap.add_argument("--out", default="docs/data", help="output directory")
    ap.add_argument("--idata", default=str(ARTIFACTS / "idata.nc"))
    ap.add_argument("--idata-anchored", default=str(ARTIFACTS / "idata_anchored.nc"))
    ap.add_argument("--anchors", default=str(ARTIFACTS / "anchors.json"))
    args = ap.parse_args()

    def resolve(p):
        p = pathlib.Path(p)
        if not p.exists() and p.with_suffix(".pkl").exists():
            return p.with_suffix(".pkl")
        return p

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_national()

    idata_before = load_idata(resolve(args.idata))
    anchored_path = resolve(args.idata_anchored)
    idata_after = load_idata(anchored_path) if anchored_path.exists() else idata_before

    anchors = {}
    if pathlib.Path(args.anchors).exists():
        raw = json.load(open(args.anchors))
        anchors = raw.get("anchors", raw)

    # decomposition (from the anchored model — the final decomposition)
    with open(out / "decomposition.json", "w") as f:
        json.dump(_decomposition(df, idata_after), f, indent=2)

    # repair: before vs after anchoring
    with open(out / "repair.json", "w") as f:
        json.dump(_repair(df, idata_before, idata_after, anchors), f, indent=2)

    # roi
    with open(out / "roi.json", "w") as f:
        json.dump(compute_roi(anchored_path), f, indent=2)

    # optimizer draws
    with open(out / "optim_draws.json", "w") as f:
        json.dump(optimize_budget(anchored_path), f, indent=2)

    # public timeseries for the charts
    shutil.copyfile(DATA_DIR / "national_weekly.csv", out / "timeseries.csv")

    print(f"Wrote dashboard contracts to {out}/: "
          "decomposition.json, repair.json, roi.json, optim_draws.json, timeseries.csv")


if __name__ == "__main__":
    main()
