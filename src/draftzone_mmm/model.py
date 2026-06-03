"""draftzone_mmm.model — shared featurization & reconstruction.

Used by the fitting, evaluation, revenue and optimisation modules so the design
matrix and the media reconstruction are defined in EXACTLY one place (a frequent
source of "the fit and the grader disagree" bugs).

CONTRACT: this module is part of the modeling pipeline and MUST NOT read the sealed
answer key (see CLAUDE.md). It only touches the public data/ directory.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from .transforms import geometric_adstock, hill_saturation

CHANNELS = ["paid_social", "paid_search", "programmatic_display", "influencer", "dooh", "tv_ctv"]

# Truncated-adstock window (lags). Must match the PyMC model's lag matrix so the
# fit and the numpy reconstruction agree exactly.
ADSTOCK_L = 12

COLORS = {
    "paid_social": "#4267B2",
    "paid_search": "#1f77b4",
    "programmatic_display": "#9467bd",
    "influencer": "#e377c2",
    "dooh": "#ff7f0e",
    "tv_ctv": "#d62728",
}

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
ARTIFACTS = REPO / "artifacts"

# Analyst's prior on carryover, expressed as Beta(a, b). NOT the answer key — these encode
# domain knowledge (TV/DOOH carry over a lot, search barely, influencer/social in between).
THETA_PRIOR = {
    "paid_social": (3, 4),            # ~0.43
    "paid_search": (1.2, 7),          # ~0.15
    "programmatic_display": (2.5, 4.5),  # ~0.36
    "influencer": (2, 5),             # ~0.29
    "dooh": (4, 3),                   # ~0.57
    "tv_ctv": (6, 2),                 # ~0.75
}


def load_national(data_dir: pathlib.Path | str = DATA_DIR) -> pd.DataFrame:
    """Load the public national weekly modeling dataset."""
    path = pathlib.Path(data_dir) / "national_weekly.csv"
    return pd.read_csv(path, parse_dates=["week"])


def build_controls(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Construct the control design matrix the analyst would build.

    Deliberately uses Fourier seasonality (the analyst does NOT know the true
    NFL-season shape), plus the observable controls. Intercept handled separately.
    """
    t = np.arange(len(df))
    trend = t / len(df)
    fourier_cols, fourier_names = [], []
    for k in range(1, 4):
        fourier_cols.append(np.sin(2 * np.pi * k * t / 52))
        fourier_cols.append(np.cos(2 * np.pi * k * t / 52))
        fourier_names += [f"sin{k}", f"cos{k}"]
    fourier = np.column_stack(fourier_cols)
    promo = df["promo_flag"].to_numpy(float)
    price = (df["price_index"].to_numpy(float) - 100) / 10
    comp = df["competitor_pressure"].to_numpy(float)
    comp = (comp - comp.mean()) / comp.std()
    holiday = df["holiday_flag"].to_numpy(float)
    Xc = np.column_stack([trend, fourier, promo, price, comp, holiday]).astype(float)
    names = ["trend", *fourier_names, "promo", "price", "comp", "holiday"]
    return Xc, names


def media_inputs(df: pd.DataFrame, channels=CHANNELS):
    """Return raw impressions, their per-channel scale, and scaled (~O(1)) impressions."""
    imp = {c: df[f"{c}_impressions"].to_numpy(float) for c in channels}
    imp_scale = {c: float(imp[c].mean()) for c in channels}
    imp_s = {c: imp[c] / imp_scale[c] for c in channels}
    return imp, imp_scale, imp_s


def channel_contribution(imp_s_c, theta, half_sat, slope, beta, L=ADSTOCK_L):
    """Reconstruct one channel's weekly conversion contribution from scaled impressions.

    Uses the same truncated, normalized geometric adstock (window L) as the PyMC model,
    so reconstruction matches the fit exactly.
    """
    ad = geometric_adstock(imp_s_c, theta, normalize=True, L=L)
    sat = hill_saturation(ad, half_sat, slope)
    return beta * sat


def draw_params(draws, i, channel):
    """Pull a single posterior draw's (theta, slope, half_sat, beta) for a channel.

    `draws` is an arviz posterior stacked over (chain, draw) -> dim 's'.
    """
    return (
        float(draws[f"theta_{channel}"][i]),
        float(draws[f"slope_{channel}"][i]),
        float(draws[f"hs_{channel}"][i]),
        float(draws[f"beta_{channel}"][i]),
    )


def load_idata(path):
    """Load InferenceData from NetCDF (.nc) or pickle (.pkl)."""
    path = pathlib.Path(path)
    if path.suffix == ".pkl":
        import pickle
        return pickle.load(open(path, "rb"))
    import arviz as az
    return az.from_netcdf(str(path))


def stacked_draws(idata, max_draws=800, seed=0):
    """Return (draws, idx) where draws is the posterior stacked over (chain, draw)->'s'
    and idx selects up to ``max_draws`` evenly-spaced draws."""
    post = idata.posterior
    if hasattr(post, "to_dataset") and not hasattr(post, "data_vars"):
        post = post.to_dataset()  # older arviz returned a DataTree-like object
    draws = post.stack(s=("chain", "draw"))
    S = draws.sizes["s"]
    idx = np.linspace(0, S - 1, min(max_draws, S)).astype(int)
    return draws, idx


def decompose(df, draws, idx, channels=CHANNELS):
    """Reconstruct per-draw decomposition.

    Returns dict with:
      mu        (n_draws, T) total predicted conversions
      channel   {c: (n_draws, T) contribution}
      baseline  (n_draws,) intercept per draw
      controls  (n_draws, T) control contribution
    """
    Xc, _ = build_controls(df)
    _, _, imp_s = media_inputs(df, channels)
    T = len(df)
    n = len(idx)
    chan = {c: np.zeros((n, T)) for c in channels}
    base = np.zeros(n)
    ctrl = np.zeros((n, T))
    mu = np.zeros((n, T))
    for j, i in enumerate(idx):
        b = float(draws["baseline"][i])
        cc = draws["ctrl_coef"][:, i].to_numpy()
        ctrl_series = Xc @ cc
        base[j] = b
        ctrl[j] = ctrl_series
        m = b + ctrl_series
        for c in channels:
            th, sl, hs, be = draw_params(draws, i, c)
            contrib = channel_contribution(imp_s[c], th, hs, sl, be)
            chan[c][j] = contrib
            m = m + contrib
        mu[j] = m
    return dict(mu=mu, channel=chan, baseline=base, controls=ctrl)
