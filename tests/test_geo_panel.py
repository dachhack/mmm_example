"""Acceptance checks for the multi-geo panel used by geo-level engines (Meridian).

The panel must (a) sum back to the national impressions series each week, so grading stays on the
same answer key, and (b) carry cross-sectional, season-decorrelated spend variation — the signal a
geo MMM exploits that a national time series cannot. Lightweight stack only.
"""
import numpy as np

from draftzone_mmm import datagen


def _panel(seed=77, n_geos=40):
    nat, truth = datagen.generate_national(seed=seed)
    panel, geo_truth = datagen.generate_geo_panel(nat, truth, seed=seed // 3 + 606, n_geos=n_geos)
    return nat, truth, panel, geo_truth


def test_geo_impressions_sum_to_national():
    nat, _, panel, _ = _panel()
    for c in datagen.CHANNELS:
        per_week = panel.groupby("week")[f"{c}_impressions"].sum().to_numpy()
        nat_imp = nat[f"{c}_impressions"].to_numpy()
        assert np.allclose(per_week, nat_imp, rtol=1e-6), c


def test_geo_truth_close_to_national_with_small_jensen_gap():
    _, truth, _, geo_truth = _panel()
    nat_media = truth["avg_contribution_decomposition"]
    for c in datagen.CHANNELS:
        nat_v = nat_media[f"media_{c}"]
        geo_v = geo_truth["avg_contribution_decomposition"][c]
        # concave Hill => summed-geo contribution is at/below national, but within a few percent
        assert geo_v <= nat_v * 1.01, (c, geo_v, nat_v)
        assert geo_v >= nat_v * 0.90, (c, geo_v, nat_v)


def test_geo_panel_has_cross_sectional_spend_variation():
    _, _, panel, _ = _panel()
    wk = sorted(panel["week"].unique())[40]
    sub = panel[panel.week == wk].copy()
    pc = sub["paid_social_spend"] / sub["population"]
    cv = pc.std() / pc.mean()
    assert cv > 0.1, cv  # geos differ within a week (the identifying signal), not scaled copies
