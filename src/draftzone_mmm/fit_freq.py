"""draftzone_mmm.fit_freq — frequentist MMM (multi-start nonlinear least squares).

Fits conversions ~ baseline + controls + sum_c beta_c * Hill(adstock(impr_c)) by NLS.
The transform params (theta, hs, slope) are nonlinear; betas + control coefs are linear
given the transforms, so we optimise the nonlinear params and least-squares the rest.

This is the "fast, confident, and easily degenerate" foil to the Bayesian fit (see the
notebooks): good R^2 can hide a catastrophic decomposition. The analyst uses domain-prior
BOUNDS, never the answer key.

CONTRACT: part of the modeling pipeline; MUST NOT read the sealed answer key.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.optimize import lsq_linear, minimize

from .model import ADSTOCK_L, CHANNELS, build_controls, load_national, media_inputs
from .transforms import geometric_adstock, hill_saturation

# Analyst's domain-prior bounds on carryover (NOT truth).
BOUNDS_THETA = {
    "paid_social": (0.1, 0.8), "paid_search": (0.0, 0.4),
    "programmatic_display": (0.1, 0.7), "influencer": (0.0, 0.6),
    "dooh": (0.2, 0.9), "tv_ctv": (0.3, 0.95),
}
# Robustness: the media betas were solved by UNCONSTRAINED lstsq, so collinear Hill features let one
# channel's beta explode (a +3000% degenerate fit on some seeds — the robustness sweep caught it).
# We now solve them with NON-NEGATIVITY (media effects >= 0) plus a small ridge in CONTRIBUTION space
# (penalising beta_c * mean(feature_c), i.e. each channel's average contribution magnitude — scale-
# free, so it tames a runaway without biasing legitimate channels). Same cure as the spend-ladder fit.
CONTRIB_RIDGE = 0.02


def fit(df, n_starts=8, seed=0):
    y = df["conversions"].to_numpy(float)
    T = len(df)
    Xc, _ = build_controls(df)
    _, _, imp_s = media_inputs(df)

    def unpack(params):
        return {c: dict(theta=params[3 * i], hs=np.exp(params[3 * i + 1]), slope=params[3 * i + 2])
                for i, c in enumerate(CHANNELS)}

    def build_media(p):
        cols = []
        for c in CHANNELS:
            ad = geometric_adstock(imp_s[c], p[c]["theta"], normalize=True, L=ADSTOCK_L)
            cols.append(hill_saturation(ad, p[c]["hs"], p[c]["slope"]))
        return np.column_stack(cols)

    nch = len(CHANNELS)

    def fit_linear(M):
        """Non-negative (media betas >= 0) ridge-regularised linear solve. The ridge penalises each
        channel's average contribution magnitude, so a collinear feature can't be handed a huge beta."""
        A = np.column_stack([np.ones(T), M, Xc])
        p = A.shape[1]
        ridge = np.sqrt(CONTRIB_RIDGE) * M.mean(0)              # one penalty row per media channel
        aug = np.zeros((nch, p))
        aug[np.arange(nch), 1 + np.arange(nch)] = ridge
        Aa = np.vstack([A, aug])
        ya = np.concatenate([y, np.zeros(nch)])
        lo = np.full(p, -np.inf); hi = np.full(p, np.inf)
        lo[1:1 + nch] = 0.0                                     # media effects non-negative
        sol = lsq_linear(Aa, ya, bounds=(lo, hi), max_iter=200)
        return A, sol.x

    def objective(params):
        M = build_media(unpack(params))
        A, coef = fit_linear(M)
        return float(np.sum((y - A @ coef) ** 2))

    x0 = []
    bnds = []
    for c in CHANNELS:
        lo, hi = BOUNDS_THETA[c]
        x0 += [(lo + hi) / 2, np.log(1.0), 1.5]
        bnds += [BOUNDS_THETA[c], (np.log(0.05), np.log(20)), (0.5, 3.5)]

    rng = np.random.default_rng(seed)
    best = None
    for s in range(n_starts):
        if s == 0:
            start = np.array(x0)
        else:
            start = []
            for c in CHANNELS:
                lo, hi = BOUNDS_THETA[c]
                start += [rng.uniform(lo, hi), np.log(rng.uniform(0.1, 10)), rng.uniform(0.7, 3.0)]
            start = np.array(start)
        res = minimize(objective, start, method="L-BFGS-B", bounds=bnds, options=dict(maxiter=500))
        if best is None or res.fun < best.fun:
            best = res

    p = unpack(best.x)
    M = build_media(p)
    A, coef = fit_linear(M)
    betas = coef[1:1 + len(CHANNELS)]
    intercept = float(coef[0])
    yhat = A @ coef
    r2 = float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))
    contrib = {c: float(betas[i] * M[:, i].mean()) for i, c in enumerate(CHANNELS)}

    return dict(
        method="frequentist_nls_multistart", r2=r2, intercept=intercept,
        params={c: dict(theta=float(p[c]["theta"]), half_sat_scaled=float(p[c]["hs"]),
                        slope=float(p[c]["slope"]), beta=float(betas[i]), avg_contrib=contrib[c])
                for i, c in enumerate(CHANNELS)},
    )


def main():
    ap = argparse.ArgumentParser(description="Frequentist MMM fit.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--starts", type=int, default=8)
    args = ap.parse_args()

    df = load_national()
    res = fit(df, n_starts=args.starts)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"Frequentist R²={res['r2']:.3f}   intercept/baseline={res['intercept']:.0f}")
    for c in CHANNELS:
        pc = res["params"][c]
        print(f"  {c:10s} theta={pc['theta']:.2f} slope={pc['slope']:.2f} "
              f"contrib={pc['avg_contrib']:.1f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
