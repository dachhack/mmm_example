"""draftzone_mmm.optimize — budget optimization under uncertainty (anchored model).

1) Point-estimate optimal reallocation at fixed total budget (the tempting, overconfident view).
2) Robust view: optimize per posterior draw -> a DISTRIBUTION of recommended allocations,
   separating CONFIDENT moves (CI clears 0) from TEST-FIRST moves (CI straddles 0).

Writes the per-draw recommendations so the dashboard's interactive optimizer can recompute
verdicts at a user-chosen confidence threshold.

CONTRACT: part of the modeling pipeline; MUST NOT read the sealed answer key.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.optimize import minimize

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


def _total_conv(alloc, params, imp, imp_mean, cur_spend):
    """Total model conversions for a spend allocation (scales each channel's impr pattern)."""
    tot = 0.0
    for c in CHANNELS:
        scale = alloc[c] / cur_spend[c] if cur_spend[c] > 0 else 0.0
        s_series = (imp[c] * scale) / imp_mean[c]
        th, sl, hs, be = params[c]
        tot += channel_contribution(s_series, th, hs, sl, be).sum()
    return tot


def _optimize(params, budget, cur_spend, imp, imp_mean, bounds_frac=(0.2, 3.0)):
    x0 = np.array([cur_spend[c] for c in CHANNELS])

    def negconv(x):
        alloc = {c: max(x[i], 1.0) for i, c in enumerate(CHANNELS)}
        return -_total_conv(alloc, params, imp, imp_mean, cur_spend)

    cons = [{"type": "eq", "fun": lambda x: x.sum() - budget}]
    bnds = [(cur_spend[c] * bounds_frac[0], cur_spend[c] * bounds_frac[1]) for c in CHANNELS]
    res = minimize(negconv, x0, method="SLSQP", bounds=bnds, constraints=cons,
                   options=dict(maxiter=300, ftol=1e-6))
    return {c: float(res.x[i]) for i, c in enumerate(CHANNELS)}


def verdict(lo, hi, thr=5.0):
    if lo > thr:
        return "INCREASE"
    if hi < -thr:
        return "DECREASE"
    return "TEST FIRST"


def optimize_budget(idata_path, n_draws=120):
    df = load_national()
    imp, imp_mean, _ = media_inputs(df)
    cur_spend = {c: float(df[f"{c}_spend"].sum()) for c in CHANNELS}
    budget = sum(cur_spend.values())

    idata = load_idata(idata_path)
    draws, idx = stacked_draws(idata, max_draws=n_draws)

    pm_params = {c: (float(draws[f"theta_{c}"].mean()), float(draws[f"slope_{c}"].mean()),
                     float(draws[f"hs_{c}"].mean()), float(draws[f"beta_{c}"].mean()))
                 for c in CHANNELS}
    cur_conv = _total_conv(cur_spend, pm_params, imp, imp_mean, cur_spend)
    opt = _optimize(pm_params, budget, cur_spend, imp, imp_mean)
    opt_conv = _total_conv(opt, pm_params, imp, imp_mean, cur_spend)

    # robust: per-draw recommended fractional change
    per_draw = {c: [] for c in CHANNELS}
    for i in idx:
        pr = {c: draw_params(draws, i, c) for c in CHANNELS}
        # draw_params returns (theta, slope, hs, beta) — same order _total_conv expects
        a = _optimize(pr, budget, cur_spend, imp, imp_mean)
        for c in CHANNELS:
            per_draw[c].append(a[c] / cur_spend[c] - 1.0)

    channels = []
    for c in CHANNELS:
        arr = np.array(per_draw[c]) * 100
        md, lo, hi = float(np.median(arr)), float(np.percentile(arr, 5.5)), float(np.percentile(arr, 94.5))
        channels.append(dict(
            channel=c, cur_spend=round(cur_spend[c], 0), opt_spend=round(opt[c], 0),
            median_change=round(md, 1), ci=[round(lo, 1), round(hi, 1)], verdict=verdict(lo, hi),
            draws=[round(v, 4) for v in (np.array(per_draw[c]) * 100)],
        ))

    return dict(
        total_budget=round(budget, 0),
        point_estimate=dict(cur_conv=round(cur_conv, 0), opt_conv=round(opt_conv, 0),
                            lift_pct=round(100 * (opt_conv / cur_conv - 1), 1)),
        channels=channels,
        note="Point estimate looks confident; only moves whose 89% CI clears 0 are robust. "
             "Everything else is TEST FIRST.",
    )


def main():
    ap = argparse.ArgumentParser(description="Robust budget optimization under uncertainty.")
    ap.add_argument("--idata", default=str(ARTIFACTS / "idata_anchored.nc"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=120)
    args = ap.parse_args()

    idata_path = pathlib.Path(args.idata)
    if not idata_path.exists():
        alt = idata_path.with_suffix(".pkl")
        idata_path = alt if alt.exists() else idata_path

    result = optimize_budget(idata_path, n_draws=args.draws)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    pe = result["point_estimate"]
    print(f"Point-estimate optimum: +{pe['lift_pct']}% conversions (same budget) — but across "
          "uncertainty:")
    for ch in result["channels"]:
        print(f"  {ch['channel']:10s} {ch['median_change']:+6.0f}% "
              f"[{ch['ci'][0]:+.0f}%,{ch['ci'][1]:+.0f}%]  {ch['verdict']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
