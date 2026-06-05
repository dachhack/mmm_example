"""Acceptance checks for the spend-ladder capability (multi-cell experiments that MEASURE the
response curve instead of assuming it). Lightweight stack only (numpy/pandas/scipy) — no PyMC.

The headline claim under test: a bracketed spend ladder right-sizes the SATURATED channels that a
single-cell anchor mis-credits, because the cells below/above the operating point let the fit see
the curvature. Plus the honest cost model returns sane planning numbers.
"""
import numpy as np

from draftzone_mmm import datagen
from draftzone_mmm.spend_ladder import fit_channel, ladder_cost_model, national_exposure


def _gen_ladder(seed=77):
    nat, truth = datagen.generate_national(seed=seed)
    d = truth["avg_contribution_decomposition"]
    nonmedia = sum(d[k] for k in ["baseline", "trend", "seasonality", "promo",
                                   "price", "competitor", "holiday"])
    ctx = {c: dict(imp_mean=float(nat[f"{c}_impressions"].mean()),
                   hs=truth["channels"][c]["half_sat"], beta=truth["channels"][c]["beta"],
                   nonmedia=float(nonmedia)) for c in datagen.CHANNELS}
    lad, lad_truth, size_frac = datagen.generate_spend_ladder(
        seed=seed // 2 + 909, national_ctx=ctx, size_frac=0.06)
    return nat, truth, lad, size_frac


def test_ladder_structure_brackets_operating_point():
    _, _, lad, _ = _gen_ladder()
    levels = sorted(lad["level_mult"].unique())
    assert min(levels) < 0, levels        # has pull-DOWN cells (the bracketing insight)
    assert max(levels) > 1.0, levels      # has push-UP cells into the plateau
    assert 0.0 in levels                  # a BAU control cell
    # control is the only untreated cell
    assert set(lad.loc[lad.level_mult == 0.0, "treated"]) == {0}
    assert set(lad.loc[lad.level_mult != 0.0, "treated"]) == {1}


def test_ladder_cracks_saturated_channels():
    nat, truth, lad, size_frac = _gen_ladder()
    media = truth["avg_contribution_decomposition"]
    errs = {}
    for c in datagen.CHANNELS:
        a_nat = national_exposure(nat, c)
        est = fit_channel(lad, c, size_frac, a_nat, n_boot=0)["est_contrib"]
        errs[c] = abs(est - media[f"media_{c}"])
    # the two most-saturated channels — where single-cell calibration fails — within ~30%
    for c in ["paid_social", "paid_search"]:
        rel = errs[c] / media[f"media_{c}"]
        assert rel < 0.30, (c, rel, errs[c])
    # whole-portfolio recovery is competitive (single-cell anchored PyMC was MAE ~79)
    assert np.mean(list(errs.values())) < 65, errs


def test_cost_model_is_honest():
    cm = ladder_cost_model(50e6)
    assert cm["total_markets"] == 240
    assert not cm["geo_feasible"] and cm["dmas_short"] > 0   # needs more DMAs than exist
    assert 3 <= cm["duration_months"] <= 7                   # ~a quarter+read
    assert cm["media_tax"] > 0 and cm["media_tax_pct_of_annual"] > 0
