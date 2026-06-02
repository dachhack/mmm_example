"""draftzone_mmm.evaluate — grade recovery against the sealed ground truth.

This is the ONLY module permitted to read data_sealed/ground_truth.json. It compares the
pipeline's recovered parameters/contributions to truth and writes a scorecard (with credible
intervals, HIT/MISS flags, and interval-coverage / calibration metrics) for the dashboard.

Outputs docs/data/scorecard.json (the data contract the dashboard's recovery section reads).
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from .model import CHANNELS, REPO, decompose, load_idata, load_national, stacked_draws

SEALED = REPO / "data_sealed" / "ground_truth.json"


def _ci(arr, lo=5.5, hi=94.5):
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def _naive_vs_truth(df, gtd):
    """Raw-spend OLS (no adstock/saturation/season) implied contribution vs truth.

    The teaching foil for "good R^2 != good attribution". Truth is read here (this is the
    only truth-reading module), so the naive comparison lives in the scorecard.
    """
    y = df["conversions"].to_numpy(float)
    X = np.column_stack([df[f"{c}_spend"].to_numpy(float) for c in CHANNELS]
                        + [np.arange(len(df)), np.ones(len(df))])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    r2 = float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))
    rows = []
    for i, c in enumerate(CHANNELS):
        implied = float(coef[i] * df[f"{c}_spend"].mean())
        rows.append(dict(channel=c, true_contrib=float(gtd[f"media_{c}"]),
                         naive_contrib=round(implied, 1)))
    return dict(r2=round(r2, 3), intercept=round(float(coef[-1]), 0), channels=rows)


def grade(idata_path, truth_path=SEALED):
    truth = json.load(open(truth_path))  # the ONLY read of the sealed key
    gtd = truth["avg_contribution_decomposition"]
    df = load_national()
    y = df["conversions"].to_numpy(float)

    idata = load_idata(idata_path)
    draws, idx = stacked_draws(idata, max_draws=800)
    dec = decompose(df, draws, idx)

    mu = dec["mu"]
    mu_mean = mu.mean(0)
    mu_lo, mu_hi = np.percentile(mu, 5.5, 0), np.percentile(mu, 94.5, 0)
    r2 = float(1 - np.sum((y - mu_mean) ** 2) / np.sum((y - y.mean()) ** 2))
    mape = float(np.mean(np.abs((y - mu_mean) / y)) * 100)
    pp_cov = float(np.mean((y >= mu_lo) & (y <= mu_hi)))

    channels = []
    contrib_hits = theta_hits = 0
    media_true = media_est = 0.0
    for c in CHANNELS:
        contrib_draws = dec["channel"][c].mean(1)  # avg weekly contribution per draw
        est = float(contrib_draws.mean())
        clo, chi = _ci(contrib_draws)
        tru = float(gtd[f"media_{c}"])
        hit = clo <= tru <= chi
        contrib_hits += hit

        theta_arr = draws[f"theta_{c}"].to_numpy()
        test = float(theta_arr.mean())
        tlo, thi = _ci(theta_arr)
        ttru = float(truth["channels"][c]["theta"])
        thit = tlo <= ttru <= thi
        theta_hits += thit

        media_true += tru
        media_est += est
        channels.append(dict(
            channel=c, true_contrib=tru, est_contrib=est, ci=[clo, chi], hit=bool(hit),
            true_theta=ttru, est_theta=test, theta_ci=[tlo, thi], theta_hit=bool(thit),
        ))

    truth_nonmedia = (gtd["baseline"] + gtd["trend"] + gtd["seasonality"] + gtd["promo"]
                      + gtd["price"] + gtd["competitor"] + gtd["holiday"])
    est_nonmedia = float(dec["baseline"].mean() + dec["controls"].mean())

    scorecard = dict(
        fit=dict(r2=r2, mape=mape, pp_interval_nominal=89.0,
                 pp_interval_coverage=round(100 * pp_cov, 1)),
        channels=channels,
        summary=dict(
            n_channels=len(CHANNELS),
            contrib_ci_hits=int(contrib_hits),
            theta_hits=int(theta_hits),
            media_total_true=round(media_true, 1),
            media_total_est=round(media_est, 1),
            media_under_credit_pct=round(100 * (1 - media_est / media_true), 1),
            overconfident=bool(100 * pp_cov < 89 - 5),
        ),
        nonmedia=dict(true=round(truth_nonmedia, 1), est=round(est_nonmedia, 1)),
        naive=_naive_vs_truth(df, gtd),
        note="89% credible/predictive intervals throughout. 'overconfident' = empirical "
             "coverage well below the nominal 89%.",
    )
    return scorecard


def main():
    ap = argparse.ArgumentParser(description="Grade the MMM against the sealed truth.")
    ap.add_argument("--idata", default=str(REPO / "artifacts" / "idata_anchored.nc"),
                    help="InferenceData to grade (default: the anchored fit)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    idata_path = pathlib.Path(args.idata)
    if not idata_path.exists():
        alt = idata_path.with_suffix(".pkl")
        idata_path = alt if alt.exists() else idata_path
    sc = grade(idata_path)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(sc, f, indent=2)

    s = sc["summary"]
    print(f"Fit: R²={sc['fit']['r2']:.3f}  MAPE={sc['fit']['mape']:.1f}%  "
          f"89% PP coverage={sc['fit']['pp_interval_coverage']:.0f}%")
    print(f"Contribution CIs containing truth: {s['contrib_ci_hits']}/{s['n_channels']}  |  "
          f"theta hits: {s['theta_hits']}/{s['n_channels']}")
    print(f"Media total: true={s['media_total_true']}  est={s['media_total_est']}  "
          f"(under-credit {s['media_under_credit_pct']}%)")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
