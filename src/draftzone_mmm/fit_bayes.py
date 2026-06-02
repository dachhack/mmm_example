"""draftzone_mmm.fit_bayes — Bayesian MMM (PyMC).

conversions ~ baseline + controls + sum_c beta_c * Hill(adstock(impr_c; theta_c); hs_c, slope_c)

Priors encode domain knowledge (TV carries over a lot, search barely) but NOT the answer
key. With ``--anchors`` the model adds a soft observation per experiment-anchored channel
that pulls its average contribution toward the geo-experiment's confound-immune estimate —
this is the "experiment repair" mechanism.

CONTRACT: part of the modeling pipeline; MUST NOT read the sealed answer key.

Heavy step: run multi-core on a VM. In CI/sandbox use ``--cores 1`` (default), short chains.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from .model import ADSTOCK_L, CHANNELS, THETA_PRIOR, build_controls, load_national, media_inputs


def _adstock_pt(pt, imp_padded, lag_idx, wexp, theta):
    """Truncated, normalized geometric adstock in pytensor.

    weights put the most mass on the CURRENT week (theta**0) decaying into the past,
    matching transforms.geometric_adstock(..., normalize=True, L=L).
    """
    w = theta ** pt.as_tensor_variable(wexp)  # [theta^0, theta^1, ... theta^(L-1)]
    w = w / pt.sum(w)
    w_rev = w[::-1]  # so the LAST column (current week) gets theta^0
    window = imp_padded[lag_idx]  # T x L; column L-1 is the current week
    return pt.sum(window * w_rev[None, :], axis=1)


def build_model(df, anchors=None, L=ADSTOCK_L):
    """Construct the PyMC model. ``anchors`` maps channel -> {prior_mu, prior_sd}."""
    import pymc as pm
    import pytensor.tensor as pt

    anchors = anchors or {}
    y = df["conversions"].to_numpy(float)
    T = len(df)
    Xc, _ = build_controls(df)
    _, _, imp_s = media_inputs(df)

    # Padded impressions + lag index for the truncated adstock.
    lag_idx = np.arange(T)[:, None] + np.arange(L)[None, :]
    imp_padded = {c: np.concatenate([np.zeros(L - 1), imp_s[c]]) for c in CHANNELS}
    wexp = np.arange(L).astype(float)

    with pm.Model() as model:
        baseline = pm.Normal("baseline", 1300, 300)
        ctrl_coef = pm.Normal("ctrl_coef", 0, 50, shape=Xc.shape[1])
        sigma = pm.HalfNormal("sigma", 100)
        Xc_t = pt.as_tensor_variable(Xc)

        media = pt.zeros(T)
        slopes = {}
        betas = {}
        for c in CHANNELS:
            a, b = THETA_PRIOR[c]
            theta = pm.Beta(f"theta_{c}", a, b)
            slope = pm.Gamma(f"slope_{c}", mu=1.5, sigma=0.6)
            hs = pm.Gamma(f"hs_{c}", mu=1.0, sigma=0.6)
            beta = pm.HalfNormal(f"beta_{c}", sigma=300)
            slopes[c] = slope
            betas[c] = beta
            ad = _adstock_pt(pt, pt.as_tensor_variable(imp_padded[c]), lag_idx, wexp, theta)
            sat = ad ** slope / (ad ** slope + hs ** slope + 1e-9)
            media = media + beta * sat

        # Experiment calibration via a DiD LIKELIHOOD. The geo-experiment measured a causal
        # lift (did) when exposure rose from a_low to a_high. With markets designed to sit at
        # half-saturation (half_sat = a_low), the model predicts that lift as
        #   did_pred = beta_c * (Hill(a_high; half_sat=a_low, slope_c) - 0.5),
        # using the channel's OWN beta and slope. This pins beta (the shared ceiling) with a
        # confound-immune measurement and breaks the beta<->half_sat degeneracy.
        for c, anc in anchors.items():
            ratio = float(anc["a_high"]) / float(anc["a_low"])
            f_high = ratio ** slopes[c] / (ratio ** slopes[c] + 1.0)
            did_pred = betas[c] * (f_high - 0.5)
            pm.Normal(f"{c}_anchor", mu=did_pred, sigma=float(anc["did_sd"]),
                      observed=float(anc["did"]))

        mu = baseline + pt.dot(Xc_t, ctrl_coef) + media
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return model


def main():
    ap = argparse.ArgumentParser(description="Fit the Bayesian MMM.")
    ap.add_argument("--out", required=True, help="output InferenceData path (.nc or .pkl)")
    ap.add_argument("--anchors", default=None, help="anchors.json to add experiment priors")
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--tune", type=int, default=600)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pymc as pm

    df = load_national()
    anchors = None
    if args.anchors:
        anchors = json.load(open(args.anchors))
        # normalise: file may be {channel: {...}} possibly nested under "anchors"
        anchors = anchors.get("anchors", anchors)
        anchors = {c: v for c, v in anchors.items() if c in CHANNELS}
        print(f"Loaded {len(anchors)} experiment anchor(s): {sorted(anchors)}")

    model = build_model(df, anchors=anchors)
    with model:
        idata = pm.sample(
            args.draws, tune=args.tune, chains=args.chains, cores=args.cores,
            target_accept=0.92, random_seed=args.seed, progressbar=False,
        )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        idata.to_netcdf(str(out))
    except Exception as e:  # pragma: no cover - fallback if h5netcdf missing
        import pickle
        alt = out.with_suffix(".pkl")
        pickle.dump(idata, open(alt, "wb"))
        print(f"NetCDF unavailable ({e}); wrote {alt} instead")
        return
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
