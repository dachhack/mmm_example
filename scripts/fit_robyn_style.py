"""scripts/fit_robyn_style.py — a faithful Python reimplementation of Meta's Robyn methodology.

The real Robyn is an R package; this environment has no R and CRAN is blocked, so — exactly as the
project already does for its own PyMC engine — we reimplement Robyn's DISTINCTIVE method in Python
and grade it on the same sealed-truth harness. This is NOT the Robyn package; it captures what makes
Robyn *Robyn*, so the leaderboard comparison is about method, not implementation:

  * Prophet-style baseline: linear trend + Fourier seasonality (3 harmonics) + observable controls.
  * Geometric adstock + Hill saturation per channel (Robyn's transforms).
  * NON-NEGATIVE ridge regression (glmnet with sign constraints): media coefficients >= 0,
    baseline/controls free; L2 penalty lambda.
  * Nevergrad MULTI-OBJECTIVE hyperparameter search (the actual optimiser Robyn uses) over per-
    channel (theta, alpha, gamma) + lambda, minimising two losses simultaneously:
      - NRMSE (fit error), and
      - DECOMP.RSSD — Robyn's signature business regulariser: the root-sum-squared distance between
        each channel's EFFECT share and its SPEND share. It encodes the prior "a channel's share of
        effect should resemble its share of spend" (roughly, ROI is comparable across channels).
    We then take the Pareto front and pick the balanced knee.

Applying what we learned elsewhere in this project:
  * We feed the SAME Fourier seasonality our other engines use, because the spend<->season confound
    otherwise inflates attribution ~2x (the Meridian lesson). Robyn's Prophet decomposition is the
    package's equivalent of this control.
  * DECOMP.RSSD is itself a PRIOR. Like every prior in this project it can help or hurt depending on
    whether the truth matches it — here channel ROIs are fairly similar, so share-matching is a
    decent prior. Where one channel had very different ROI, it would mislead. We grade to find out.

CONTRACT: a modeling engine — MUST NOT read data_sealed/ground_truth.json. Grading is downstream.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import nevergrad as ng
from scipy.optimize import lsq_linear

from draftzone_mmm.transforms import geometric_adstock

REPO = pathlib.Path(__file__).resolve().parents[1]
CHANNELS = ["paid_social", "paid_search", "programmatic_display", "influencer", "dooh", "tv_ctv"]


def hill(x, alpha, gamma):
    """Robyn saturation: x^alpha / (x^alpha + inflexion^alpha), inflexion = (1-gamma)*min+gamma*max."""
    inflex = (1 - gamma) * x.min() + gamma * x.max()
    return x ** alpha / (x ** alpha + inflex ** alpha + 1e-12)


def base_features(df):
    """Prophet-style baseline design: intercept, linear trend, 3 Fourier harmonics, controls.
    Returns (B, names) where columns are sign-free (fit with no non-negativity constraint)."""
    T = len(df)
    t = np.arange(T)
    cols, names = [np.ones(T), t / T], ["intercept", "trend"]
    for k in range(1, 4):
        cols += [np.sin(2 * np.pi * k * t / 52), np.cos(2 * np.pi * k * t / 52)]
        names += [f"sin{k}", f"cos{k}"]
    for c in ["promo_flag", "price_index", "competitor_pressure", "holiday_flag"]:
        cols.append(df[c].to_numpy(float))
        names.append(c)
    B = np.column_stack(cols)
    return B, names


def media_features(df, theta, alpha, gamma):
    """Per-channel saturated-adstock feature scaled to [0,1] (>=0 so its coefficient stays >=0)."""
    feats, spend_share = [], []
    spends = np.array([df[f"{c}_spend"].sum() for c in CHANNELS])
    for i, c in enumerate(CHANNELS):
        imp = df[f"{c}_impressions"].to_numpy(float)
        ad = geometric_adstock(imp, float(theta[i]))
        sat = hill(ad, float(alpha[i]), float(gamma[i]))
        mx = sat.max() or 1.0
        feats.append(sat / mx)
    return np.column_stack(feats), spends / spends.sum()


def _standardize(B):
    mu = B.mean(0)
    sd = B.std(0)
    sd[sd == 0] = 1.0
    mu[0] = 0.0  # leave intercept column as ones
    sd[0] = 1.0
    return (B - mu) / sd, mu, sd


def fit_ridge_nonneg(B, M, y, lam):
    """Non-negative ridge: media (M) coefs >= 0, baseline (B) coefs free. L2 via row augmentation.
    Returns (beta_base, beta_media, yhat)."""
    Bs, _, _ = _standardize(B)
    nb, nm = Bs.shape[1], M.shape[1]
    X = np.column_stack([Bs, M])
    p = X.shape[1]
    aug = np.sqrt(lam) * np.eye(p)
    aug[0, 0] = 0.0  # do not penalise intercept
    Xa = np.vstack([X, aug])
    ya = np.concatenate([y, np.zeros(p)])
    lo = np.concatenate([np.full(nb, -np.inf), np.zeros(nm)])
    hi = np.full(p, np.inf)
    sol = lsq_linear(Xa, ya, bounds=(lo, hi), max_iter=200)
    coef = sol.x
    yhat = X @ coef
    return coef[:nb], coef[nb:], yhat


def evaluate(df, y, params):
    theta, alpha, gamma, lam = params["theta"], params["alpha"], params["gamma"], params["lam"]
    B, _ = base_features(df)
    M, spend_share = media_features(df, theta, alpha, gamma)
    beta_b, beta_m, yhat = fit_ridge_nonneg(B, M, y, lam)
    nrmse = float(np.sqrt(np.mean((y - yhat) ** 2)) / (y.max() - y.min()))
    contrib = M * beta_m[None, :]                       # T x C, >= 0
    eff = contrib.sum(0)
    eff_share = eff / (eff.sum() + 1e-9)
    rssd = float(np.sqrt(np.sum((eff_share - spend_share) ** 2)))
    return nrmse, rssd, beta_m, contrib


def main():
    ap = argparse.ArgumentParser(description="Fit a Robyn-style MMM and write the engine contract.")
    ap.add_argument("--out", default=str(REPO / "artifacts" / "robyn_style_results.json"))
    ap.add_argument("--budget", type=int, default=2500, help="nevergrad evaluations")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(REPO / "data" / "national_weekly.csv")
    y = df["conversions"].to_numpy(float)
    T = len(df)

    param = ng.p.Dict(
        theta=ng.p.Array(init=np.full(6, 0.3)).set_bounds(0.0, 0.8),
        alpha=ng.p.Array(init=np.full(6, 1.5)).set_bounds(0.5, 3.0),
        gamma=ng.p.Array(init=np.full(6, 0.6)).set_bounds(0.3, 1.0),
        lam=ng.p.Log(lower=1e-2, upper=1e6),
    )
    param.random_state.seed(args.seed)
    opt = ng.optimizers.NGOpt(parametrization=param, budget=args.budget)

    records = []
    for _ in range(args.budget):
        cand = opt.ask()
        nrmse, rssd, _, _ = evaluate(df, y, cand.value)
        opt.tell(cand, [nrmse, rssd])
        records.append((cand.value, nrmse, rssd))

    # Pareto front, then the balanced knee (min sum of min-max-normalised objectives).
    front = opt.pareto_front()
    fvals = []
    for c in front:
        nrmse, rssd, _, _ = evaluate(df, y, c.value)
        fvals.append((c.value, nrmse, rssd))
    nr = np.array([v[1] for v in fvals]); rs = np.array([v[2] for v in fvals])
    def _nz(a):
        rng = a.max() - a.min()
        return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)
    knee = int(np.argmin(_nz(nr) + _nz(rs)))
    best = fvals[knee][0]

    nrmse, rssd, beta_m, contrib = evaluate(df, y, best)
    channels = {c: dict(est_contrib=float(contrib[:, i].mean()), ci=None)
                for i, c in enumerate(CHANNELS)}
    B, _ = base_features(df)
    M, _ = media_features(df, best["theta"], best["alpha"], best["gamma"])
    _, _, yhat = fit_ridge_nonneg(B, M, y, best["lam"])
    r2 = float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))

    results = dict(
        engine="robyn_style", label="Robyn-style (Python reimpl.)", bayesian=False,
        fit=dict(r2=r2, nrmse=nrmse, decomp_rssd=rssd),
        hyperparameters=dict(theta=[round(x, 3) for x in best["theta"]],
                             alpha=[round(x, 3) for x in best["alpha"]],
                             gamma=[round(x, 3) for x in best["gamma"]],
                             lam=float(best["lam"])),
        channels=channels,
        note="Point-estimate engine (ridge): no credible intervals. Pareto-knee model from a "
             f"{args.budget}-eval Nevergrad search over NRMSE + DECOMP.RSSD. Faithful reimplementation "
             "of Robyn's method (not the R package — CRAN unavailable here).",
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Pareto knee: NRMSE={nrmse:.4f}  DECOMP.RSSD={rssd:.3f}  R²={r2:.3f}  (front={len(front)})")
    for i, c in enumerate(CHANNELS):
        print(f"  {c:22s} est={channels[c]['est_contrib']:6.1f}  "
              f"theta={best['theta'][i]:.2f} alpha={best['alpha'][i]:.2f} gamma={best['gamma'][i]:.2f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
