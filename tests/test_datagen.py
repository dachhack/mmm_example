"""Acceptance checks for the DraftZone v2 data generator (docs/DATA_SPEC.md).

These run on the lightweight stack (numpy/pandas/scipy) — no PyMC — so they pass in CI.
"""
import pandas as pd

from draftzone_mmm import datagen
from draftzone_mmm.experiment import did_analysis


def _gen():
    nat, truth = datagen.generate_national(seed=2024)
    geo, exp_truth, _ = datagen.generate_geo_experiments(seed=2024 // 2 + 808)
    return nat, truth, geo, exp_truth


def test_confound_within_tolerance():
    _, truth, _, _ = _gen()
    corr = truth["meta"]["realized_corr_totalspend_season"]
    assert abs(corr - datagen.TARGET_CONFOUND) <= 0.05, corr


def test_baseline_share_realistic():
    nat, _, _, _ = _gen()
    share = datagen.BASELINE / nat["conversions"].mean()
    assert 0.35 <= share <= 0.50, share


def test_theta_distinct_and_span_low_to_high():
    _, truth, _, _ = _gen()
    thetas = sorted(truth["channels"][c]["theta"] for c in datagen.CHANNELS)
    assert thetas[0] <= 0.2, thetas       # at least one low-carryover channel
    assert thetas[-1] >= 0.7, thetas      # at least one high-carryover channel
    assert len(set(thetas)) == len(thetas)  # all distinct


def test_every_geo_experiment_recovers_true_increment():
    """DiD must recover each channel's true incremental lift to within ~15%."""
    _, _, geo, exp_truth = _gen()
    for c in datagen.CHANNELS:
        res = did_analysis(geo[geo.channel == c], n_boot=200)
        true_inc = exp_truth[c]["true_increment_per_market_week"]
        recovered = res["did"] / true_inc
        assert 0.85 <= recovered <= 1.15, (c, recovered, res["did"], true_inc)


def test_markets_sit_on_responsive_hill_region():
    _, _, _, exp_truth = _gen()
    for c in datagen.CHANNELS:
        ratio = exp_truth[c]["market_adstock_over_halfsat"]
        assert 0.3 <= ratio <= 2.0, (c, ratio)


def test_national_shape():
    nat, _, _, _ = _gen()
    assert len(nat) == datagen.T_NATIONAL
    for c in datagen.CHANNELS:
        assert f"{c}_spend" in nat.columns and f"{c}_impressions" in nat.columns
    assert pd.api.types.is_datetime64_any_dtype(nat["week"])
