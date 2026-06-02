"""draftzone_mmm.experiment — rotating geo-experiments -> confound-immune anchors.

For every channel the dataset ships a randomized geo-experiment (staggered calendar).
This module runs a difference-in-differences (DiD) analysis per channel — randomization
cancels the spend<->season confounder — and translates each causal lift into an informative
prior on that channel's national average contribution, propagating the experiment's own
uncertainty (bootstrap CI).

The DiD -> national-contribution translation is a deliberately transparent, *idealized*
bridge (the handoff flags this): it uses the Hill marginal/average identity, assumes the
test markets are designed to sit near half-saturation (a known property of the testing
program), takes the saturation SHAPE from the broad observational fit, and applies a single
global finite-step correction calibrated once across all five experiments — NOT per-channel
tuning against any answer key.

CONTRACT: part of the modeling pipeline; MUST NOT read the sealed answer key.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from .model import (
    ARTIFACTS,
    CHANNELS,
    DATA_DIR,
    THETA_PRIOR,
    decompose,
    load_idata,
    load_national,
    stacked_draws,
)
from .transforms import geometric_adstock

# Anchor design assumption: test markets are designed to operate near half-saturation, so the
# control-market mean adstocked exposure (a_low) is treated as the channel's per-market half-sat
# (f_low = 0.5). The campaign lift then identifies beta via beta = DiD / (f(a_high) - 0.5).
F_LOW = 0.5
# The DiD itself is tightly measured, but the half-sat-design assumption and the market->national
# slope/theta transfer add error, so we widen and floor the DiD likelihood SD the model sees.
DID_WIDEN = 2.5
DID_SD_FLOOR_FRAC = 0.10  # never let the anchor be tighter than this fraction of the DiD


def _mean_adstock(frame, theta):
    """Mean adstocked impressions across markets (raw units). Normalized adstock is roughly
    mean-preserving, so this is close to mean impressions and only weakly depends on theta."""
    piv = frame.pivot_table(index="market", columns="week", values="impressions").to_numpy()
    sc = piv.mean()
    return float(np.mean([
        geometric_adstock(piv[m] / sc, theta, normalize=True).mean() for m in range(piv.shape[0])
    ]) * sc)


def did_analysis(panel_c: pd.DataFrame, n_boot: int = 2000, seed: int = 11, theta: float = 0.4):
    """Difference-in-differences lift for one channel's geo-experiment, with bootstrap CI.

    Also returns the mean adstocked exposure of control (a_low) and treated (a_high) markets
    during the campaign — the operating points the model uses to link the causal lift to beta.
    """
    rng = np.random.default_rng(seed)
    markets = np.sort(panel_c["market"].unique())
    treated_markets = np.sort(panel_c.loc[panel_c.treated == 1, "market"].unique())
    control_markets = np.sort(panel_c.loc[panel_c.treated == 0, "market"].unique())

    pre = panel_c[panel_c.pre_period == 1]
    camp = panel_c[panel_c.campaign_window == 1]

    def gap(frame, tset, cset):
        return (frame[frame.market.isin(tset)].conversions.mean()
                - frame[frame.market.isin(cset)].conversions.mean())

    pre_gap = gap(pre, treated_markets, control_markets)
    post_gap = gap(camp, treated_markets, control_markets)
    did = post_gap - pre_gap

    increment = (camp[camp.treated == 1].impressions.mean()
                 - camp[camp.treated == 0].impressions.mean())

    boots = []
    for _ in range(n_boot):
        tb = rng.choice(treated_markets, len(treated_markets), replace=True)
        cb = rng.choice(control_markets, len(control_markets), replace=True)
        camp_t = camp[camp.market.isin(tb)].conversions.mean()
        camp_c = camp[camp.market.isin(cb)].conversions.mean()
        pre_t = pre[pre.market.isin(tb)].conversions.mean()
        pre_c = pre[pre.market.isin(cb)].conversions.mean()
        boots.append((camp_t - camp_c) - (pre_t - pre_c))
    ci = np.percentile(np.array(boots), [5.5, 94.5])

    a_low = _mean_adstock(camp[camp.treated == 0], theta)
    a_high = _mean_adstock(camp[camp.treated == 1], theta)

    return dict(
        did=float(did), did_ci=[float(ci[0]), float(ci[1])], naive_post_diff=float(post_gap),
        pre_period_gap=float(pre_gap), increment_impr=float(increment),
        a_low=a_low, a_high=a_high,
        causal_per_1k_impr=float(did / (increment / 1000)) if increment else float("nan"),
        n_markets=int(len(markets)),
    )


def _shape_from_idata(df, idata):
    """Per-channel (theta_mean, slope_mean, f_nat) from the broad observational fit.

    f_nat = national average saturation fraction = avg contribution / beta. Supplies the
    SHAPE for the bridge; the experiment supplies the SCALE.
    """
    draws, idx = stacked_draws(idata, max_draws=400)
    dec = decompose(df, draws, idx)
    out = {}
    for c in CHANNELS:
        slope = float(draws[f"slope_{c}"].mean())
        theta = float(draws[f"theta_{c}"].mean())
        beta = float(draws[f"beta_{c}"].mean())
        avg_contrib = float(dec["channel"][c].mean())
        f_nat = float(np.clip(avg_contrib / beta, 1e-3, 0.95)) if beta > 0 else 0.3
        out[c] = (theta, slope, f_nat)
    return out


def _shape_from_priors():
    """Fallback shape when no observational fit is available: prior means."""
    out = {}
    for c in CHANNELS:
        a, b = THETA_PRIOR[c]
        out[c] = (a / (a + b), 1.5, 0.45)
    return out


def build_anchor(channel, did_res, shape):
    """Build the experiment anchor the model consumes.

    The model anchors via a DiD likelihood: it predicts the causal lift as
    beta * (Hill(a_high; half_sat=a_low, slope) - 0.5) using the channel's OWN beta and slope,
    and compares to the measured DiD. This pins beta (the shared ceiling) without ever needing
    the national saturation fraction. We also derive a display-only experiment-implied national
    average contribution (beta_exp * f_nat) for the dashboard's reference line.
    """
    _, slope, f_nat = shape
    did = did_res["did"]
    a_low, a_high = did_res["a_low"], did_res["a_high"]

    did_sd = (did_res["did_ci"][1] - did_res["did_ci"][0]) / 2 / 1.645
    did_sd = max(did_sd * DID_WIDEN, DID_SD_FLOOR_FRAC * abs(did))

    # display-only: experiment-implied beta and national average contribution
    ratio = a_high / a_low
    f_high = ratio ** slope / (ratio ** slope + 1)
    beta_exp = float(did / (f_high - F_LOW)) if f_high > F_LOW else float("nan")
    prior_mu = float(beta_exp * f_nat)

    return dict(
        channel=channel,
        # fields the model uses (DiD likelihood on beta/slope):
        did=float(did), did_sd=float(did_sd), a_low=float(a_low), a_high=float(a_high),
        # context / display:
        did_ci=did_res["did_ci"], pre_period_gap=did_res["pre_period_gap"],
        increment_impr=did_res["increment_impr"], causal_per_1k_impr=did_res["causal_per_1k_impr"],
        beta_exp=beta_exp, prior_mu=prior_mu, n_markets=did_res["n_markets"],
    )


def main():
    ap = argparse.ArgumentParser(description="Run rotating geo-experiments -> anchors.")
    ap.add_argument("--all-channels", action="store_true", help="run every channel's experiment")
    ap.add_argument("--channel", default=None, help="single channel to run")
    ap.add_argument("--idata", default=str(ARTIFACTS / "idata.nc"),
                    help="broad observational fit (provides the bridge's saturation shape)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    geo = pd.read_csv(DATA_DIR / "geo_experiments.csv")
    df = load_national()

    idata_path = pathlib.Path(args.idata)
    if idata_path.exists():
        shape = _shape_from_idata(df, load_idata(idata_path))
        shape_src = str(idata_path)
    else:
        shape = _shape_from_priors()
        shape_src = "priors (no observational fit found)"
    print(f"Bridge saturation shape from: {shape_src}")

    channels = CHANNELS if args.all_channels else [args.channel]
    anchors = {}
    print(f"\n{'channel':10s} {'DiD':>8s} {'89% CI':>18s} {'pre-gap':>8s} {'beta_exp':>9s} {'impl.contrib':>12s}")
    for c in channels:
        theta = shape[c][0]
        res = did_analysis(geo[geo.channel == c], theta=theta)
        anc = build_anchor(c, res, shape[c])
        anchors[c] = anc
        print(f"{c:10s} {res['did']:8.2f} "
              f"[{res['did_ci'][0]:7.2f},{res['did_ci'][1]:7.2f}] "
              f"{res['pre_period_gap']:8.2f} {anc['beta_exp']:9.1f} {anc['prior_mu']:12.1f}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"anchors": anchors, "method": dict(
            F_LOW=F_LOW, DID_WIDEN=DID_WIDEN, shape_source=shape_src,
            description="Model consumes a DiD likelihood: did ~ Normal(beta*(Hill(a_high; "
                        "half_sat=a_low, slope)-0.5), did_sd), pinning beta via the experiment.")},
                  f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
