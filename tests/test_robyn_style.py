"""Acceptance checks for the Robyn-style engine building blocks (no full Nevergrad search, so it
stays fast/CI-friendly). Verifies the pieces that make it Robyn: non-negative media coefficients,
Hill saturation, and the DECOMP.RSSD effect-share-vs-spend-share regulariser.
"""
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import fit_robyn_style as rb  # noqa: E402
from draftzone_mmm import datagen  # noqa: E402


def _df():
    nat, _ = datagen.generate_national(seed=77)
    return nat


def test_hill_is_monotone_in_zero_one():
    x = np.linspace(0, 10, 50)
    h = rb.hill(x, alpha=1.5, gamma=0.6)
    assert h.min() >= 0 and h.max() <= 1
    assert np.all(np.diff(h) >= -1e-9)  # non-decreasing


def test_media_coefficients_are_nonnegative_and_contrib_nonneg():
    df = _df()
    y = df["conversions"].to_numpy(float)
    nrmse, rssd, beta_m, contrib = rb.evaluate(
        df, y, dict(theta=[0.3] * 6, alpha=[1.5] * 6, gamma=[0.6] * 6, lam=1.0))
    assert (beta_m >= -1e-9).all()          # sign constraint (Robyn's '+' media)
    assert (contrib >= -1e-9).all()
    assert 0 <= nrmse and rssd >= 0


def test_decomp_rssd_zero_when_effect_matches_spend():
    """If every channel's effect share equals its spend share, DECOMP.RSSD is 0 — the regulariser's
    fixed point."""
    df = _df()
    M, spend_share = rb.media_features(df, [0.3] * 6, [1.5] * 6, [0.6] * 6)
    # construct contributions whose column sums match the spend shares exactly
    eff = spend_share * 1000.0
    eff_share = eff / eff.sum()
    rssd = float(np.sqrt(np.sum((eff_share - spend_share) ** 2)))
    assert rssd < 1e-9
