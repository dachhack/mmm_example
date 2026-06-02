"""draftzone_mmm.revenue — conversions -> revenue valuation and ROI (anchored model).

Blended LTV with uncertainty. Per channel, with posterior + LTV uncertainty propagated:
  - revenue contribution = conversions_contrib * LTV
  - average ROI = revenue / spend
  - MARGINAL ROI = d(revenue)/d(spend) at current spend (the number that drives decisions)

CONTRACT: part of the modeling pipeline; MUST NOT read the sealed answer key.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from .model import (
    ARTIFACTS,
    CHANNELS,
    channel_contribution,
    draw_params,
    load_idata,
    load_national,
    media_inputs,
    stacked_draws,
)

# Blended LTV (single figure with a range — deliberately not per-channel; see limitations).
LTV_MU, LTV_LO, LTV_HI = 220.0, 180.0, 260.0
LTV_SD = (LTV_HI - LTV_LO) / (2 * 1.6)  # ~89% mass within [180, 260]


def compute_roi(idata_path, max_draws=600, seed=6):
    df = load_national()
    _, _, imp_s = media_inputs(df)
    spend = {c: df[f"{c}_spend"].to_numpy(float) for c in CHANNELS}
    total_spend = {c: float(spend[c].sum()) for c in CHANNELS}
    n_weeks = len(df)

    idata = load_idata(idata_path)
    draws, idx = stacked_draws(idata, max_draws=max_draws)
    rng = np.random.default_rng(seed)

    acc = {c: dict(conv=[], roi=[], mroi=[]) for c in CHANNELS}
    for i in idx:
        ltv = rng.normal(LTV_MU, LTV_SD)
        for c in CHANNELS:
            th, sl, hs, be = draw_params(draws, i, c)
            contrib = channel_contribution(imp_s[c], th, hs, sl, be)
            conv_total = contrib.sum()
            roi = (conv_total * ltv) / total_spend[c] if total_spend[c] else np.nan
            # marginal ROI: +1% impressions (proxy for +1% spend), finite difference
            contrib2 = channel_contribution(imp_s[c] * 1.01, th, hs, sl, be)
            dconv = (contrib2 - contrib).sum()
            dspend = total_spend[c] * 0.01
            mroi = (dconv * ltv) / dspend if dspend else np.nan
            acc[c]["conv"].append(contrib.mean())
            acc[c]["roi"].append(roi)
            acc[c]["mroi"].append(mroi)

    def stat(a):
        a = np.array(a)
        return [float(a.mean()), float(np.percentile(a, 5.5)), float(np.percentile(a, 94.5))]

    channels = []
    blended_rev = 0.0
    for c in CHANNELS:
        conv_avg = float(np.mean(acc[c]["conv"]))
        blended_rev += conv_avg * n_weeks * LTV_MU
        channels.append(dict(
            channel=c, avg_conv=round(conv_avg, 1), total_spend=round(total_spend[c], 0),
            roi=stat(acc[c]["roi"]), mroi=stat(acc[c]["mroi"]),
        ))
    blended_roi = float(blended_rev / sum(total_spend.values()))

    return dict(
        ltv=dict(mu=LTV_MU, lo=LTV_LO, hi=LTV_HI, sd=round(LTV_SD, 1)),
        blended_roi=round(blended_roi, 2),
        channels=channels,
        note="ROI = total revenue / total spend (average). mROI = revenue on the NEXT dollar. "
             "Decisions use mROI: shift toward high mROI, away from mROI<1.",
    )


def main():
    ap = argparse.ArgumentParser(description="Revenue / ROI from the anchored model.")
    ap.add_argument("--idata", default=str(ARTIFACTS / "idata_anchored.nc"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    idata_path = pathlib.Path(args.idata)
    if not idata_path.exists():
        alt = idata_path.with_suffix(".pkl")
        idata_path = alt if alt.exists() else idata_path

    roi = compute_roi(idata_path)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(roi, f, indent=2)

    print(f"Blended LTV ${LTV_MU:.0f} (89% ${LTV_LO:.0f}-${LTV_HI:.0f})  "
          f"blended media ROI {roi['blended_roi']:.2f}")
    for ch in roi["channels"]:
        print(f"  {ch['channel']:10s} ROI {ch['roi'][0]:.2f} [{ch['roi'][1]:.2f},{ch['roi'][2]:.2f}]  "
              f"mROI {ch['mroi'][0]:.2f} [{ch['mroi'][1]:.2f},{ch['mroi'][2]:.2f}]")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
