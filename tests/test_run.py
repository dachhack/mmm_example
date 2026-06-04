"""Acceptance checks for the productised runner (draftzone_mmm.run). The column/feature helpers and
the decision logic are tested fast; one tiny end-to-end smoke fit validates the API shape.
"""
import numpy as np
import pandas as pd
import pytest

from draftzone_mmm import run as R


def _toy(n=80, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    season = 50 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 52))
    df = pd.DataFrame({"week": pd.date_range("2022-01-02", periods=n, freq="W")})
    y = 200 + season + rng.normal(0, 10, n)
    for c, eff in [("a", 0.004), ("b", 0.002)]:
        spend = np.clip(2000 + 1500 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 500, n), 0, None)
        df[f"{c}_spend"] = spend
        y = y + eff * spend
    df["conversions"] = y
    return df


def test_resolve_cols_and_seasonality():
    df = _toy()
    sp, ex = R._resolve_cols(df, ["a", "b"], None, "_spend", None, None)
    assert sp == {"a": "a_spend", "b": "b_spend"} and ex == sp   # exposure defaults to spend
    cols, names = R._seasonality(52, 52, 3)
    assert names[0] == "trend" and "sin1" in names and len(names) == 1 + 6


def test_resolve_cols_raises_on_missing():
    df = _toy()
    with pytest.raises(KeyError):
        R._resolve_cols(df, ["a", "missing"], None, "_spend", None, None)


def test_smoke_fit_returns_decision_shaped_result():
    df = _toy()
    res = R.run_mmm(df, kpi="conversions", channels=["a", "b"], date="week",
                    draws=40, tune=40, chains=1, cores=1, seed=1)
    # uncertainty everywhere: every channel has a credible interval
    assert set(res.contributions.columns) >= {"channel", "mean", "lo", "hi", "share"}
    assert (res.contributions["lo"] <= res.contributions["hi"]).all()
    # marginal ROI reported with CI; ROI table covers both channels
    assert set(res.roi["channel"]) == {"a", "b"}
    assert {"mroi_mean", "mroi_lo", "mroi_hi"} <= set(res.roi.columns)
    # diagnostics include the confound + sampling quality
    d = res.diagnostics
    assert {"r2", "confound", "ess_min", "rhat_max", "n_weeks"} <= set(d)
    # recommendation splits into a verdict per channel
    rec = res.recommend()
    assert set(rec["channel"]) == {"a", "b"}
    assert rec["verdict"].str.contains("confident|test first").all()
    assert isinstance(res.summary(), str) and "marginal roi" in res.summary().lower()
