"""draftzone_mmm.spend_ladder — fit each channel's response CURVE from a spend ladder.

A single geo test gives ONE point on the response curve, so to turn its lift into a national
number you must *assume* the curve's shape (half-sat, slope) and operating point. That assumption
is exactly where single-cell calibration fails on saturated channels. A spend LADDER runs several
cells at different spend levels, tracing several points along the curve, so the shape can be
MEASURED. This module:

  1. reads data/spend_ladder.csv (several cells/channel) + the public size_frac from config,
  2. computes each cell's DiD lift vs the control cell and its mean adstocked campaign exposure,
  3. fits Hill(beta, half_sat, slope) through the (exposure, lift) points per channel
     (non-linear least squares, bootstrap over markets for a CI),
  4. translates the fitted per-market curve back to the NATIONAL operating point using the known
     size_frac and the national channel's own observed exposure — yielding a national average
     contribution per channel, gradeable against the answer key exactly like any other engine.

The payoff and the catch: the high cells climb a saturated channel into its plateau, so the
curvature (hence the ceiling) becomes identifiable where a single secant could not see it — but
a saturated channel's lift is small relative to market noise, so even a ladder needs many markets
(cost) to pin it. Both are surfaced honestly downstream.

CONTRACT: part of the modeling pipeline; MUST NOT read the sealed answer key.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .model import CHANNELS, DATA_DIR, REPO, THETA_PRIOR, load_national
from .transforms import geometric_adstock, hill_saturation

ARTIFACTS = REPO / "artifacts"


def _theta_for(channel):
    a, b = THETA_PRIOR[channel]
    return a / (a + b)


def _cell_exposure(frame, theta):
    """Mean NON-normalized adstocked impressions across a cell's markets during the campaign
    window (the replica DGP uses non-normalized adstock, so exposure scales linearly with size)."""
    piv = frame.pivot_table(index="market", columns="week", values="impressions")
    weeks = sorted(frame["week"].unique())
    # adstock each market's full series, then average the campaign-window weeks
    camp_weeks = sorted(frame.loc[frame.campaign_window == 1, "week"].unique())
    full = frame.pivot_table(index="market", columns="week", values="impressions").to_numpy()
    ad = np.vstack([geometric_adstock(full[m], theta, normalize=False) for m in range(full.shape[0])])
    cols = [weeks.index(w) for w in camp_weeks]
    return float(ad[:, cols].mean())


def _cell_did(sub, control_markets, cell_markets):
    """DiD lift (per market-week) of a cell vs the control cell."""
    pre = sub[sub.pre_period == 1]
    camp = sub[sub.campaign_window == 1]

    def m(frame, mkts):
        return frame[frame.market.isin(mkts)].conversions.mean()

    return ((m(camp, cell_markets) - m(camp, control_markets))
            - (m(pre, cell_markets) - m(pre, control_markets)))


SLOPE_REG = 0.4   # mild ridge pulling slope toward a plausible 1.4 (few-point fits are noisy)
SLOPE_PRIOR = 1.4
# Robustness: when a channel's cells are too flat/noisy to pin the curve, beta (the ceiling) is
# unidentified and least squares lets it — and half-sat — run to their bounds, so the extrapolated
# national contribution (beta x f / size_frac) explodes. A ridge on log(beta) toward the data scale
# plus a tight half-sat bound keep the fit from extrapolating a ceiling the cells never showed. The
# robustness sweep (some seeds blew up to 10x MAE without this) motivated it.
BETA_REG = 0.30


def _fit_hill(exposures, lifts, a0, slope0=1.5, weights=None):
    """Fit lift_k = beta * (Hill(a_k; hs, slope) - Hill(a0; hs, slope)) for beta, hs, slope.

    ``a0`` is the control (BAU) cell's campaign exposure, so the (a0, 0) point is implicit and the
    down cells (negative lift) anchor the curve's absolute level. Parametrised in log-space for
    positivity, with ridges on slope and beta so a handful of noisy points can't push the
    (weakly-identified) ceiling to a bound and blow up the extrapolation.
    Returns (beta, hs, slope, resid_rms).
    """
    a = np.asarray(exposures, float)
    y = np.asarray(lifts, float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, float)
    amax = a.max()
    beta_anchor = max(np.abs(y).max(), 1.0) * 3.0   # data-scale guess for the ceiling

    def resid(p):
        beta, hs, slope = np.exp(p)
        f = hill_saturation(a, hs, slope) - hill_saturation(np.array([a0]), hs, slope)[0]
        reg_slope = SLOPE_REG * (np.log(slope) - np.log(SLOPE_PRIOR))
        reg_beta = BETA_REG * (np.log(beta) - np.log(beta_anchor))
        return np.concatenate([w * (beta * f - y), [reg_slope, reg_beta]])

    p0 = np.log([beta_anchor, max(a0, 1.0), slope0])
    lo = np.log([1e-3, amax * 5e-2, 0.6])
    hi = np.log([beta_anchor * 40, amax * 8, 3.0])   # bounded ceiling + tighter half-sat
    sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=5000)
    beta, hs, slope = np.exp(sol.x)
    rms = float(np.sqrt(np.mean((resid(sol.x)[:-2]) ** 2)))
    return float(beta), float(hs), float(slope), rms


def fit_channel(lad, channel, size_frac, a_nat, n_boot=400, seed=7):
    """Recover the national average contribution for one channel from its spend ladder."""
    theta = _theta_for(channel)
    sub = lad[lad.channel == channel]
    # control = the BAU cell (level 0); the others bracket it (down cells + up cells)
    control = int(sub.loc[sub.level_mult == 0.0, "cell"].iloc[0])
    other_cells = [int(k) for k in sorted(sub.cell.unique()) if k != control]
    ctrl_markets = sub.loc[sub.cell == control, "market"].unique()

    # per-cell observed campaign exposure and DiD lift vs control (down cells give NEGATIVE lift)
    a0 = _cell_exposure(sub[sub.cell == control], theta)
    exposures, lifts, cell_markets = [], [], {}
    for k in other_cells:
        cm = sub.loc[sub.cell == k, "market"].unique()
        cell_markets[k] = cm
        exposures.append(_cell_exposure(sub[sub.cell == k], theta))
        lifts.append(_cell_did(sub[sub.cell.isin([control, k])], ctrl_markets, cm))

    beta_m, hs_m, slope, rms = _fit_hill(exposures, lifts, a0)

    # translate to the NATIONAL operating point. The national channel's observed adstock exposure
    # a_nat, scaled down to market size (a_nat * size_frac), is the operating point IN MARKET UNITS;
    # it sits inside the bracketed cell range, so the contribution there is an INTERPOLATION of the
    # fitted curve. National avg contribution = market contribution at that point / size_frac.
    a_op_market = a_nat * size_frac
    f_op = float(hill_saturation(np.array([a_op_market]), hs_m, slope)[0])
    est = beta_m * f_op / size_frac
    beta_nat = beta_m / size_frac
    hs_nat = hs_m / size_frac
    f_nat = f_op

    # bootstrap over markets (resample within each cell) for a national-contribution CI
    rng = np.random.default_rng(seed + abs(hash(channel)) % 9973)
    boots = []
    for _ in range(n_boot):
        cb = rng.choice(ctrl_markets, len(ctrl_markets), replace=True)
        ex_b, lf_b = [], []
        for k in other_cells:
            tb = rng.choice(cell_markets[k], len(cell_markets[k]), replace=True)
            sk = sub[sub.cell == k]
            ex_b.append(_cell_exposure(sk[sk.market.isin(tb)], theta))
            lf_b.append(_cell_did(sub[sub.cell.isin([control, k])], cb, tb))
        try:
            bme, bhs, bsl, _ = _fit_hill(ex_b, lf_b, a0, slope0=slope)
            fb = float(hill_saturation(np.array([a_op_market]), bhs, bsl)[0])
            boots.append(bme * fb / size_frac)
        except Exception:
            continue
    ci = ([float(np.percentile(boots, 5.5)), float(np.percentile(boots, 94.5))]
          if len(boots) > 30 else None)

    return dict(
        est_contrib=float(est), ci=ci,
        fitted=dict(beta_market=beta_m, half_sat_market=hs_m, slope=float(slope),
                    beta_national=float(beta_nat), half_sat_national=float(hs_nat),
                    f_nat=f_nat, resid_rms=rms, a_nat=float(a_nat), a0=float(a0)),
        cells=[dict(exposure=float(e), did=float(l)) for e, l in zip(exposures, lifts)],
    )


def ladder_cost_model(channel_annual_budget, n_cells=6, markets_per_cell=40,
                      n_dmas_available=210, test_dma_pop_share=0.18,
                      pre_weeks=6, campaign_weeks=12, read_weeks=4,
                      over_levels=(0.75, 1.75, 3.5), down_levels=(-0.85, -0.5),
                      mroi_at_op=0.6):
    """Order-of-magnitude planning model for ONE channel's spend ladder. Returns a dict of the
    real costs a ladder incurs, so the dashboard can be honest about what it takes.

    The three costs that matter, none of them the modeling:
      * CALENDAR: pre + campaign + carryover read ~ a full quarter (longer-carryover channels read
        slower). The rotating portfolio runs one channel per quarter, so a full-portfolio ladder
        program is a multi-year commitment.
      * GEO INVENTORY: n_cells x markets_per_cell test DMAs are tied up at once. The US has ~210
        DMAs; a wide ladder can need MORE distinct geos than exist, forcing smaller (noisier) cells
        or sub-DMA matched markets — and the most saturated channels (smallest lift) need the MOST
        markets to power, exactly where geo is scarcest.
      * MEDIA TAX: the up-cells deliberately overspend into the flat part of the curve (near-zero
        marginal return) and the down-cells forgo conversions. Both are real P&L, paid to LEARN.

    All figures are planning-grade approximations, clearly labelled as such on the dashboard.
    """
    total_markets = n_cells * markets_per_cell
    duration_weeks = pre_weeks + campaign_weeks + read_weeks
    # share of the channel's audience tied up in the test (cap-aware)
    audience_share = min(test_dma_pop_share, 0.5)
    weekly_budget = channel_annual_budget / 52.0
    budget_in_test = weekly_budget * campaign_weeks * audience_share  # $ flowing through test geos

    # media tax: overspend in up-cells earns ~mroi_at_op*BAU but the INCREMENT earns less and less;
    # approximate the wasted fraction of each up-cell's extra spend as (1 - mroi_share) growing with
    # level. down-cells forgo ~ their cut of BAU contribution. Split the test budget evenly by cell.
    per_cell_budget = budget_in_test / n_cells
    up_waste = sum(per_cell_budget * lvl * (1 - mroi_at_op / (1 + lvl)) for lvl in over_levels)
    down_forgone = sum(per_cell_budget * abs(lvl) * mroi_at_op for lvl in down_levels)
    media_tax = up_waste + down_forgone

    return dict(
        total_markets=total_markets,
        duration_weeks=duration_weeks,
        duration_months=round(duration_weeks / 4.33, 1),
        geo_feasible=total_markets <= n_dmas_available,
        dmas_short=max(0, total_markets - n_dmas_available),
        audience_share=audience_share,
        budget_in_test=budget_in_test,
        media_tax=media_tax,
        media_tax_pct_of_annual=100 * media_tax / channel_annual_budget,
    )


def national_exposure(nat_df, channel):
    """National channel's mean non-normalized adstocked impressions (public data)."""
    return float(geometric_adstock(nat_df[f"{channel}_impressions"].to_numpy(float),
                                   _theta_for(channel), normalize=False).mean())


def main():
    ap = argparse.ArgumentParser(description="Fit channel response curves from the spend ladder.")
    ap.add_argument("--ladder", default=str(DATA_DIR / "spend_ladder.csv"))
    ap.add_argument("--out", default=str(ARTIFACTS / "ladder_results.json"))
    args = ap.parse_args()

    lad = pd.read_csv(args.ladder)
    cfg = json.load(open(DATA_DIR / "config.json"))
    size_frac = cfg.get("ladder_size_frac") or 0.06
    nat_df = load_national()

    channels = {}
    rms_all = []
    print(f"{'channel':22s} {'est':>7s} {'89% CI':>17s} {'beta_nat':>9s} {'hs_nat':>11s} {'slope':>6s}")
    for c in CHANNELS:
        a_nat = national_exposure(nat_df, c)
        r = fit_channel(lad, c, size_frac, a_nat)
        channels[c] = dict(est_contrib=r["est_contrib"], ci=r["ci"], fitted=r["fitted"],
                           cells=r["cells"])
        rms_all.append(r["fitted"]["resid_rms"])
        ci = r["ci"]
        cistr = f"[{ci[0]:6.0f},{ci[1]:6.0f}]" if ci else "—"
        f = r["fitted"]
        print(f"{c:22s} {r['est_contrib']:7.0f} {cistr:>17s} {f['beta_national']:9.0f} "
              f"{f['half_sat_national']:11.0f} {f['slope']:6.2f}")

    results = dict(
        engine="spend_ladder", label="Spend ladder (curve fit)", bayesian=True,
        fit=dict(r2=None, resid_rms=float(np.mean(rms_all))),
        size_frac=size_frac, channels=channels,
        note="Per-channel national avg contribution recovered by fitting Hill through the ladder "
             "cells and translating to the national operating point via the known size_frac.",
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
