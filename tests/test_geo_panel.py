"""Acceptance checks for the multi-geo panel used by geo-level engines (Meridian).

The panel must (a) sum back to the national impressions series each week, so grading stays on the
same answer key, and (b) carry cross-sectional, season-decorrelated spend variation — the signal a
geo MMM exploits that a national time series cannot. Lightweight stack only.
"""
import numpy as np

from draftzone_mmm import datagen


def _panel(seed=77, n_geos=40, confound=0.0, noise_frac=0.06):
    nat, truth = datagen.generate_national(seed=seed)
    panel, geo_truth = datagen.generate_geo_panel(
        nat, truth, seed=seed // 3 + 606, n_geos=n_geos, confound=confound, noise_frac=noise_frac)
    return nat, truth, panel, geo_truth


def test_geo_impressions_sum_to_national():
    # impressions must sum to national in BOTH the clean and confounded panels (share normalisation)
    for conf in (0.0, 1.0):
        nat, _, panel, _ = _panel(confound=conf)
        for c in datagen.CHANNELS:
            per_week = panel.groupby("week")[f"{c}_impressions"].sum().to_numpy()
            assert np.allclose(per_week, nat[f"{c}_impressions"].to_numpy(), rtol=1e-6), (c, conf)


def test_clean_panel_small_jensen_gap():
    _, truth, _, geo_truth = _panel(confound=0.0)
    nat_media = truth["avg_contribution_decomposition"]
    for c in datagen.CHANNELS:
        nat_v, geo_v = nat_media[f"media_{c}"], geo_truth["avg_contribution_decomposition"][c]
        # concave Hill => summed-geo contribution is at/below national, but within a few percent
        assert geo_v <= nat_v * 1.01 and geo_v >= nat_v * 0.90, (c, geo_v, nat_v)


def test_confounder_targets_spend_and_lowers_geo_media():
    """With confound>0 the latent demand factor draws targeted spend (positive realized corr) and,
    by concentrating spend into Hill saturation, lowers the geo world's true media total."""
    _, truth, _, geo_truth = _panel(confound=1.0, noise_frac=0.18)
    assert geo_truth["realized_corr_spend_demand"] > 0.3, geo_truth["realized_corr_spend_demand"]
    nat_tot = sum(truth["avg_contribution_decomposition"][f"media_{c}"] for c in datagen.CHANNELS)
    geo_tot = sum(geo_truth["avg_contribution_decomposition"].values())
    assert geo_tot < nat_tot, (geo_tot, nat_tot)  # targeted concentration costs efficiency


def test_geo_panel_has_cross_sectional_spend_variation():
    _, _, panel, _ = _panel(confound=1.0, noise_frac=0.18)
    wk = sorted(panel["week"].unique())[40]
    sub = panel[panel.week == wk].copy()
    pc = sub["paid_social_spend"] / sub["population"]
    cv = pc.std() / pc.mean()
    assert cv > 0.1, cv  # geos differ within a week (the identifying signal), not scaled copies
