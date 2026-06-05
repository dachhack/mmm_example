"""draftzone_mmm.datagen — DraftZone v2 synthetic data generator.

Same DFS-subscription premise as the prototype, redesigned so the *full
experimentation path* is possible: a randomized geo-experiment for **every** channel,
staggered over time (a realistic always-on testing program).

Writes:
  data/national_weekly.csv   public modeling dataset
  data/geo_experiments.csv   rotating per-channel experiment panel
  data/config.json           public knobs (confound level, seeds, calendar) — NO truth
  data_sealed/ground_truth.json   SEALED answer key (every true parameter + effect)

CONTRACT: this module WRITES the sealed truth but the rest of the pipeline must never
READ it (enforced by tests/test_no_truth_leak.py). Only evaluate.py may read it back.

Causal chain (per channel):
    spend --(noisy, mildly concave)--> impressions
    impressions --geometric adstock(theta)--> --Hill(half_sat, slope)--> x beta --> contribution
    conversions = baseline + trend + seasonality + controls + sum(contributions) + noise
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from .transforms import geometric_adstock, hill_saturation

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
SEALED_DIR = REPO / "data_sealed"

CHANNELS = ["paid_social", "paid_search", "programmatic_display", "influencer", "dooh", "tv_ctv"]

# ----------------------------------------------------------------------
# TRUE per-channel parameters (the sealed truth) — a realistic DFS marketing mix.
# Deliberately DISTINCT so recovery is a genuine test: theta spans 0.10 -> 0.75.
# hs/slope live on the IMPRESSION-adstock scale. beta = max conversions/wk.
# Saturation is set via hs RELATIVE to each channel's impression level:
#   high-saturation (low headroom): paid_social, paid_search, tv_ctv  -> small hs
#   headroom (low saturation):      influencer, dooh                  -> large hs
# ----------------------------------------------------------------------
CH = {
    # largest budget, mature -> high saturation, medium carryover
    "paid_social": dict(base_spend=45000, season_coef=11000, flight=False,
                        imp_per_dollar=30.0, noise=0.08, theta=0.45, hs=900000, slope=1.5, beta=400),
    # responsive -> short carryover, fast saturation
    "paid_search": dict(base_spend=22000, season_coef=6000, flight=False,
                        imp_per_dollar=22.0, noise=0.06, theta=0.10, hs=240000, slope=2.2, beta=270),
    # medium carryover / medium saturation
    "programmatic_display": dict(base_spend=16000, season_coef=4000, flight=False,
                                imp_per_dollar=25.0, noise=0.10, theta=0.35, hs=430000, slope=1.6, beta=250),
    # CPA creator + promo codes -> spiky spend, lower saturation (headroom)
    "influencer": dict(base_spend=12000, season_coef=3500, flight=True,
                      imp_per_dollar=8.0, noise=0.14, theta=0.25, hs=175000, slope=2.0, beta=300),
    # billboards/CTV screens -> longer carryover, lower saturation (headroom)
    "dooh": dict(base_spend=9000, season_coef=2500, flight=False,
                imp_per_dollar=40.0, noise=0.10, theta=0.60, hs=1250000, slope=1.3, beta=200),
    # linear + CTV -> long carryover, high saturation
    "tv_ctv": dict(base_spend=38000, season_coef=10000, flight=True,
                  imp_per_dollar=9.0, noise=0.10, theta=0.75, hs=380000, slope=1.6, beta=320),
}

# Controls / baseline truth
BASELINE = 1200.0
TREND_PER_WK = 4.0
PROMO_EFFECT = 180.0
PRICE_COEF = -6.0
COMP_COEF = -2.2
HOLIDAY_EFFECT = 90.0

# Rotating geo-experiment calendar: one randomized test per channel, staggered.
GEO_CALENDAR = {
    "paid_social": "2024-Q1",
    "paid_search": "2024-Q2",
    "programmatic_display": "2024-Q3",
    "influencer": "2024-Q4",
    "dooh": "2025-Q1",
    "tv_ctv": "2025-Q2",
}

# Per-channel geo-experiment design. BAU impressions and per-market half-sat are
# chosen so each market sits on the RESPONSIVE part of the Hill curve (verified by
# an assertion below) — NOT saturated. INCREMENT is the campaign bump (treatment only).
GEO_DESIGN = {
    # channel: BAU impr/mkt/wk, season sensitivity (~bau/1000, a modest confounding ripple),
    # increment impr (campaign bump, treatment only), per-market half_sat (~1.25*bau -> responsive).
    "paid_social": dict(bau=45000, season_sens=45, increment=31000, hs_mkt=56000),
    "paid_search": dict(bau=38000, season_sens=38, increment=27000, hs_mkt=48000),
    "programmatic_display": dict(bau=34000, season_sens=34, increment=24000, hs_mkt=42000),
    "influencer": dict(bau=22000, season_sens=22, increment=16000, hs_mkt=28000),
    "dooh": dict(bau=30000, season_sens=30, increment=21000, hs_mkt=38000),
    "tv_ctv": dict(bau=50000, season_sens=50, increment=35000, hs_mkt=62000),
}

TARGET_CONFOUND = 0.60
N_MARKETS = 80
T_NATIONAL = 156

# Optional modifier: time-varying saturation. Spend is paced to the sports calendar, so during
# the NFL peak every channel is flooded and sits HIGHER on its Hill curve (effective half-sat
# drops). At the seasonal peak half_sat is reduced by SAT_SEASONAL_AMP. The MMM assumes a CONSTANT
# half-sat, so turning this on makes the model misspecified — a stress test for the experiments.
SAT_SEASONAL_AMP = 0.35

# Spend-ladder design: instead of ONE test cell per channel, run several cells at different
# spend levels so the response *curve* can be measured (not assumed). Each level is an additive
# multiple of the cell's BAU impressions applied during the campaign window (impr = bau*(1+level)),
# so NEGATIVE levels pull spend DOWN. The cells deliberately BRACKET the current operating point:
# the down cells (toward dark) pin the curve's absolute level (the channel's total contribution),
# while the up cells climb a saturated channel into its plateau to expose the diminishing returns.
# Bracketing turns the read into an INTERPOLATION at the operating point — the thing a single
# always-UP secant cannot do, and the reason single-cell calibration mis-sizes saturated channels.
LADDER_LEVELS = (-0.85, -0.5, 0.0, 0.75, 1.75, 3.5)


# ----------------------------------------------------------------------
# National series
# ----------------------------------------------------------------------
def _national_controls(week_idx):
    """Build the deterministic non-media truth (baseline, trend, season, controls)."""
    T = len(week_idx)
    trend = TREND_PER_WK * week_idx

    # NFL-season seasonality: annual cycle peaking ~Sep-Jan + a sharper playoff bump.
    phase = 2 * np.pi * (week_idx - 35) / 52.0
    seasonal_smooth = 350.0 * (0.5 + 0.5 * np.cos(phase))
    playoff = np.zeros(T)
    for yr_start in range(0, T, 52):
        for w in range(2, 7):  # Jan playoff weeks
            if yr_start + w < T:
                playoff[yr_start + w] += 120.0
    seasonality = seasonal_smooth + playoff

    promo_flag = np.zeros(T)
    for w in [10, 11, 33, 60, 61, 85, 110, 111, 138, 139]:
        if w < T:
            promo_flag[w] = 1
    promo_contrib = PROMO_EFFECT * promo_flag

    price_index = np.ones(T) * 100.0
    price_index[45:] = 108.0
    price_index[100:] = 115.0
    price_contrib = PRICE_COEF * (price_index - 100.0)

    return dict(
        trend=trend, seasonality=seasonality, promo_flag=promo_flag,
        promo_contrib=promo_contrib, price_index=price_index, price_contrib=price_contrib,
    )


def _channel_spend(p, expected_demand, season_scale, rng, T):
    """One channel's weekly spend: base + season ramp (confound) + indep noise + flighting."""
    season_part = season_scale * p["season_coef"] * expected_demand
    indep = rng.normal(0, p["base_spend"] * 0.35, T)
    spend = p["base_spend"] + season_part + indep
    if p["flight"]:
        dark = rng.random(T) < 0.25
        spend = np.where(dark, spend * 0.1, spend)
    return np.clip(spend, 0, None)


def _realized_confound(season_scale, expected_demand, seasonality, seeds, T):
    """Realized corr(total spend, seasonality) at a given season_scale (deterministic in seed)."""
    rng = np.random.default_rng(seeds)
    total = np.zeros(T)
    for _, p in CH.items():
        total += _channel_spend(p, expected_demand, season_scale, rng, T)
    return float(np.corrcoef(total, seasonality)[0, 1])


def _tune_confound(expected_demand, seasonality, seeds, T, target=TARGET_CONFOUND):
    """Bisection on a global season_scale so realized confound matches the target."""
    lo, hi = 0.0, 4.0
    best = 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        c = _realized_confound(mid, expected_demand, seasonality, seeds, T)
        best = mid
        if abs(c - target) < 0.005:
            break
        if c < target:
            lo = mid
        else:
            hi = mid
    return best


def generate_national(seed=2024, seasonal_saturation=False, saturation_scale=1.0,
                      confound_target=TARGET_CONFOUND):
    """Generate the national weekly dataset and the national portion of the truth.

    ``saturation_scale`` multiplies every channel's half-sat: >1 pushes channels LOWER on their
    Hill curves (less saturated, more headroom — a fast-growing advertiser); <1 saturates them.
    Beta (the shared ceiling) is unchanged, so the geo experiments stay consistent.

    If ``seasonal_saturation`` is True, half-sat also shrinks at the seasonal peak (channels more
    saturated when everyone floods them) — a time-varying response the constant-half-sat MMM
    cannot represent.
    """
    rng = np.random.default_rng(seed)
    week_idx = np.arange(T_NATIONAL)
    dates = pd.date_range("2023-01-01", periods=T_NATIONAL, freq="W-SUN")
    ctrl = _national_controls(week_idx)
    seasonality = ctrl["seasonality"]
    # season in [0,1], 1 at the NFL peak — drives the time-varying half-sat
    season_norm = (seasonality - seasonality.min()) / (seasonality.max() - seasonality.min())

    competitor_pressure = np.clip(
        50 + 30 * np.sin(2 * np.pi * week_idx / 26 + 1.0) + rng.normal(0, 8, T_NATIONAL),
        0, None,
    )
    comp_contrib = COMP_COEF * (competitor_pressure - competitor_pressure.mean())

    holiday_flag = np.zeros(T_NATIONAL)
    for w in [5, 35, 47, 57, 87, 99, 109, 139, 151]:
        if w < T_NATIONAL:
            holiday_flag[w] = 1
    holiday_contrib = HOLIDAY_EFFECT * holiday_flag

    expected_demand = (seasonality - seasonality.mean()) / seasonality.std()

    # Tune the confound to the target by scaling all season_coefs jointly.
    spend_seed = seed + 7
    season_scale = _tune_confound(expected_demand, seasonality, spend_seed, T_NATIONAL,
                                  target=confound_target)
    rng_spend = np.random.default_rng(spend_seed)

    data = {"week": dates}
    channel_contribs = {}
    channels_truth = {}
    spends = []
    for name in CHANNELS:
        p = CH[name]
        spend = _channel_spend(p, expected_demand, season_scale, rng_spend, T_NATIONAL)
        imp_mean = p["imp_per_dollar"] * spend ** 0.97
        impressions = np.clip(imp_mean * rng.normal(1.0, p["noise"], T_NATIONAL), 0, None)

        ad = geometric_adstock(impressions, p["theta"])
        hs_base = p["hs"] * saturation_scale
        hs_t = hs_base * (1 - SAT_SEASONAL_AMP * season_norm) if seasonal_saturation else hs_base
        sat = hill_saturation(ad, hs_t, p["slope"])
        contrib = p["beta"] * sat

        data[f"{name}_spend"] = spend
        data[f"{name}_impressions"] = impressions
        channel_contribs[name] = contrib
        spends.append(spend)
        channels_truth[name] = dict(
            theta=p["theta"], half_sat=float(hs_base), slope=p["slope"], beta=float(p["beta"]),
            imp_per_dollar=p["imp_per_dollar"], mean_spend=float(spend.mean()),
            mean_contrib=float(contrib.mean()),
        )

    media_total = sum(channel_contribs.values())
    expected_conversions = (
        BASELINE + ctrl["trend"] + seasonality + ctrl["promo_contrib"]
        + ctrl["price_contrib"] + comp_contrib + holiday_contrib + media_total
    )
    conversions = np.clip(
        expected_conversions * rng.normal(1.0, 0.05, T_NATIONAL) + rng.normal(0, 35, T_NATIONAL),
        0, None,
    )

    data["conversions"] = conversions
    data["promo_flag"] = ctrl["promo_flag"]
    data["price_index"] = ctrl["price_index"]
    data["competitor_pressure"] = competitor_pressure
    data["holiday_flag"] = holiday_flag
    df = pd.DataFrame(data)

    total_spend = np.sum(spends, axis=0)
    realized_corr = float(np.corrcoef(total_spend, seasonality)[0, 1])

    decomp = {
        "baseline": float(BASELINE),
        "trend": float(ctrl["trend"].mean()),
        "seasonality": float(seasonality.mean()),
        "promo": float(ctrl["promo_contrib"].mean()),
        "price": float(ctrl["price_contrib"].mean()),
        "competitor": float(comp_contrib.mean()),
        "holiday": float(holiday_contrib.mean()),
    }
    for name in CHANNELS:
        decomp[f"media_{name}"] = float(channel_contribs[name].mean())

    truth = dict(
        meta=dict(
            T=T_NATIONAL, baseline=BASELINE, trend_per_wk=TREND_PER_WK,
            promo_effect=PROMO_EFFECT, price_coef=PRICE_COEF, comp_coef=COMP_COEF,
            holiday_effect=HOLIDAY_EFFECT, target_corr_spend_season=confound_target,
            realized_corr_totalspend_season=realized_corr, season_scale=float(season_scale),
            seasonal_saturation=bool(seasonal_saturation),
            sat_seasonal_amp=float(SAT_SEASONAL_AMP) if seasonal_saturation else 0.0,
            season_norm=season_norm.tolist() if seasonal_saturation else None,
        ),
        channels=channels_truth,
        avg_contribution_decomposition=decomp,
    )
    return df, truth


# ----------------------------------------------------------------------
# Rotating geo-experiment panel
# ----------------------------------------------------------------------
def generate_geo_experiments(seed=808, T_exp=20, pre_period=6, camp_len=12,
                             national_ctx=None, hetero_sigma=0.0):
    """One randomized geo-experiment per channel, staggered across the calendar.

    Returns (tidy_dataframe, truth_dict, assignment_dict).

    Default (hetero_sigma=0): every market sits at ~half-saturation (powered, responsive) using
    the per-channel GEO_DESIGN — NOT a scaled replica of the national channel, so the experiment's
    marginal-per-dollar does not equal the national marginal (a known limitation).

    Replica mode (hetero_sigma>0 + national_ctx): each market is a population-weighted SCALED
    REPLICA of the national channel — market impressions = national x size x per-capita exposure,
    market half-sat = national x size, market beta = national x size. Then market contribution =
    size x national, and crucially the market's marginal-ROI-per-dollar EQUALS the national one
    (the size cancels in the Hill ratio). Per-capita exposure e ~ Normal(1, hetero_sigma) scatters
    each market's operating point normally around national (big metros saturated, small DMAs with
    headroom). So the pooled lift measures the *national* marginal — the honest normalization.
    Tradeoff: saturated channels are genuinely hard to lift-test (small, noisy lift) — realistic.
    """
    rows = []
    exp_truth = {}
    assignments = {}
    SIZE_FRAC = 0.06  # each test market ~6% of national scale (synthetic; cancels in the marginal)

    for ci, channel in enumerate(CHANNELS):
        p = CH[channel]
        g = GEO_DESIGN[channel]
        slope, theta = p["slope"], p["theta"]
        rng = np.random.default_rng(seed + 101 * (ci + 1))
        week = np.arange(T_exp)
        camp_end = min(pre_period + camp_len, T_exp)
        in_campaign = (week >= pre_period) & (week < camp_end)
        season_norm = 0.5 + 0.5 * np.sin(2 * np.pi * (week - 3) / T_exp)  # 0..1 shared confounder
        perm = rng.permutation(N_MARKETS)
        treat_mask = np.zeros(N_MARKETS, bool)
        treat_mask[perm[: N_MARKETS // 2]] = True

        hetero = hetero_sigma > 0 and national_ctx is not None
        adnorm = not hetero  # replica markets use national's NON-normalized adstock; homogeneous keeps normalized
        if hetero:
            ctx = national_ctx[channel]
            size = SIZE_FRAC * np.exp(rng.normal(0, 0.3, N_MARKETS))         # DMA size (lognormal)
            expo = np.clip(rng.normal(1.0, hetero_sigma, N_MARKETS), 0.35, 1.9)  # per-capita exposure
            hs_arr = ctx["hs"] * size                                        # replica half-sat
            beta_arr = ctx["beta"] * size                                    # replica ceiling
            mean_impr = ctx["imp_mean"] * size * expo                        # per-market BAU level
            base_mean = (ctx["imp_mean"] * size * expo)[:, None] * (1 + 0.08 * season_norm[None, :])
            increment = 0.7 * mean_impr
            market_base = ctx["nonmedia"] * size * np.exp(rng.normal(0, 0.1, N_MARKETS))
            season_amp = 0.4 * ctx["nonmedia"] * size
            noise_sd = 0.12 * np.maximum(market_base, 1.0)
        else:
            hs_arr = np.full(N_MARKETS, g["hs_mkt"], float)
            beta_arr = np.full(N_MARKETS, p["beta"], float)
            mean_impr = np.full(N_MARKETS, g["bau"], float)
            base_mean = (g["bau"] + g["season_sens"] * 250 * season_norm)[None, :].repeat(N_MARKETS, 0)
            increment = np.full(N_MARKETS, g["increment"], float)
            market_base = rng.normal(320, 55, N_MARKETS)
            season_amp = np.full(N_MARKETS, 250.0)
            noise_sd = np.full(N_MARKETS, 15.0)

        true_incs, ratios, fracs = [], [], []
        for m in range(N_MARKETS):
            treated = bool(treat_mask[m])
            hs_m, beta_m = hs_arr[m], beta_arr[m]

            def contrib(impr, hs_m=hs_m, beta_m=beta_m):
                return beta_m * hill_saturation(geometric_adstock(impr, theta, normalize=adnorm), hs_m, slope)

            mean_bau = base_mean[m]
            bau = np.clip(mean_bau + rng.normal(0, mean_impr[m] * 0.12, T_exp), 0, None)
            extra = np.where(in_campaign, increment[m], 0.0) if treated else np.zeros(T_exp)
            impr = bau + extra
            spend = impr / p["imp_per_dollar"]
            conv = (market_base[m] + season_amp[m] * season_norm + contrib(impr)
                    + rng.normal(0, noise_sd[m], T_exp))
            for w in range(T_exp):
                rows.append(dict(
                    channel=channel, market=m, week=int(w), treated=int(treated),
                    spend=float(spend[w]), impressions=float(impr[w]), conversions=float(conv[w]),
                    campaign_window=int(bool(in_campaign[w])), pre_period=int(w < pre_period),
                ))
            inc_with = contrib(np.where(in_campaign, mean_bau + increment[m], mean_bau))
            true_incs.append(float((inc_with - contrib(mean_bau))[in_campaign].mean()))
            ad_m = geometric_adstock(mean_bau, theta, normalize=adnorm).mean()
            ratios.append(ad_m / hs_m)
            fracs.append(float(hill_saturation(np.array([ad_m]), hs_m, slope)[0]))

        ratio = float(np.mean(ratios))
        # At the default saturation this should sit on the responsive band; at extreme
        # --saturation-scale the test markets can be far up/down the curve (a real phenomenon —
        # you can't cleanly lift-test a deeply saturated channel), so warn rather than halt.
        if not (0.2 <= ratio <= 3.0):
            print(f"  [warn] {channel}: mean market adstock/half_sat {ratio:.2f} off the responsive "
                  "band (expected at extreme saturation-scale)")
        exp_truth[channel] = dict(
            quarter=GEO_CALENDAR[channel],
            true_increment_per_market_week=float(np.mean(true_incs)),
            increment_impr=float(np.mean(increment)),
            n_markets=N_MARKETS, T_exp=T_exp, pre_period=pre_period, camp_end=int(camp_end),
            market_adstock_over_halfsat=ratio,
            market_saturation_mean=float(np.mean(fracs)),
            market_saturation_sd=float(np.std(fracs)),
        )
        assignments[channel] = dict(seed=int(seed + 101 * (ci + 1)),
                                    treated_markets=sorted(int(x) for x in np.where(treat_mask)[0]))

    geo_df = pd.DataFrame(rows)
    return geo_df, exp_truth, assignments


# ----------------------------------------------------------------------
# Spend ladder: multi-cell experiments that MEASURE the response curve
# ----------------------------------------------------------------------
def generate_spend_ladder(seed=909, T_exp=20, pre_period=6, camp_len=12,
                          national_ctx=None, size_frac=0.06, hetero_sigma=0.12,
                          levels=LADDER_LEVELS, per_cell=40):
    """One spend LADDER per channel: several geo cells, each at a different campaign spend level.

    Returns (tidy_dataframe, ladder_truth, size_frac).

    Each channel's markets are split into ``len(levels)`` equal cells. Every market is a
    scale-consistent SCALED REPLICA of the national channel (impr = national x size x exposure,
    half-sat = national x size, beta = national x size), using the national NON-normalized adstock
    so exposure scales linearly. Cell k applies an additive campaign bump of ``levels[k] x BAU``
    impressions during the campaign window (level 0 = pure control). Reading the DiD lift of each
    cell vs the control cell traces several points along the channel's response curve; fitting Hill
    through them recovers (half-sat, slope, beta) directly instead of assuming a shape. Crucially,
    the high cells climb a SATURATED channel into its plateau, where the curvature — and therefore
    the ceiling — finally becomes identifiable. ``size_frac`` (the test markets' known share of
    national scale) is the only quantity the downstream fit needs to translate per-market curves
    back to the national operating point; it is a public design knob, not truth.
    """
    if national_ctx is None:
        raise ValueError("generate_spend_ladder requires national_ctx (replica scaling)")
    rows = []
    ladder_truth = {}
    n_cells = len(levels)
    n_markets = n_cells * per_cell  # a ladder needs MORE inventory than a single test — part of its cost

    for ci, channel in enumerate(CHANNELS):
        p = CH[channel]
        ctx = national_ctx[channel]
        slope, theta = p["slope"], p["theta"]
        rng = np.random.default_rng(seed + 137 * (ci + 1))
        week = np.arange(T_exp)
        camp_end = min(pre_period + camp_len, T_exp)
        in_campaign = (week >= pre_period) & (week < camp_end)
        season_norm = 0.5 + 0.5 * np.sin(2 * np.pi * (week - 3) / T_exp)

        # market-level replica params (fixed known size, only per-capita exposure jitters)
        size = size_frac
        hs_m = ctx["hs"] * size
        beta_m = ctx["beta"] * size
        cells_truth = []
        for k, mult in enumerate(levels):
            for j in range(per_cell):
                m = k * per_cell + j
                expo = float(np.clip(rng.normal(1.0, hetero_sigma), 0.4, 1.8))
                bau_mean = ctx["imp_mean"] * size * expo
                mkt_base = ctx["nonmedia"] * size * float(np.exp(rng.normal(0, 0.1)))
                season_amp = 0.4 * ctx["nonmedia"] * size
                noise_sd = 0.12 * max(mkt_base, 1.0)
                bump = mult * bau_mean

                def contrib(impr):
                    return beta_m * hill_saturation(
                        geometric_adstock(impr, theta, normalize=False), hs_m, slope)

                bau = np.clip(bau_mean * (1 + 0.08 * season_norm)
                              + rng.normal(0, bau_mean * 0.12, T_exp), 0, None)
                extra = np.where(in_campaign, bump, 0.0)
                impr = np.clip(bau + extra, 0, None)
                spend = impr / p["imp_per_dollar"]
                conv = (mkt_base + season_amp * season_norm + contrib(impr)
                        + rng.normal(0, noise_sd, T_exp))
                for w in range(T_exp):
                    rows.append(dict(
                        channel=channel, cell=k, level_mult=float(mult), market=m,
                        week=int(w), treated=int(mult != 0), spend=float(spend[w]),
                        impressions=float(impr[w]), conversions=float(conv[w]),
                        campaign_window=int(bool(in_campaign[w])), pre_period=int(w < pre_period),
                    ))
            # sealed truth: this cell's true per-market-week incremental contribution vs its own BAU
            bm = ctx["imp_mean"] * size  # cell-representative BAU level (expo ~ 1)
            bm_series = np.full(T_exp, bm)
            true_lift = float((contrib(np.where(in_campaign, bm * (1 + mult), bm))
                               - contrib(bm_series))[in_campaign].mean())
            cells_truth.append(dict(level_mult=float(mult), true_lift_per_market_week=true_lift))

        ladder_truth[channel] = dict(
            quarter=GEO_CALENDAR[channel], n_cells=n_cells, per_cell=per_cell,
            levels=list(map(float, levels)), size_frac=float(size_frac),
            T_exp=T_exp, pre_period=pre_period, camp_end=int(camp_end),
            cells=cells_truth,
        )

    return pd.DataFrame(rows), ladder_truth, float(size_frac)


# ----------------------------------------------------------------------
# National geo PANEL (for geo-level engines like Meridian)
# ----------------------------------------------------------------------
def generate_geo_panel(nat_df, truth, seed=606, n_geos=50, idio_sigma=0.30,
                       confound=0.0, noise_frac=0.06, demand_rel=0.20, proxy_noise=0.8):
    """Decompose the national series into a multi-geo panel for geo-level MMM engines.

    Meridian (and geo MMMs generally) are built for geo×time data: spend that varies ACROSS geos
    within a week supplies cross-sectional identification the national time series cannot. We split
    each channel's national weekly impressions across ``n_geos`` geos so they SUM BACK to the
    national series (grading stays on the same answer key), each geo a population-weighted scaled
    replica (per-capita exposure = national × idiosyncratic multiplier; contribution =
    size_g × beta × Hill(adstock(per-capita))). Because Hill is concave, the summed geo contribution
    is a hair below national (an honest Jensen gap, reported in the truth).

    HARDENING (``confound`` > 0): a latent geo×time DEMAND factor d[g,t] (local economy/sports
    swings, AR over time, demeaned across geos so national totals are preserved) does two things the
    model never sees — it (1) raises that geo-week's NON-media conversions and (2) makes the marketer
    TARGET it with more spend. That is the geo analogue of the national spend↔season confound: within
    a geo, weeks with more spend also have more demand-driven conversions, so an MMM with no proxy for
    d over-credits media. Meridian's per-geo intercept and shared seasonal baseline absorb the
    time-invariant and national-seasonal parts, but NOT geo-specific time-varying demand — exactly the
    residual that bites real geo studies. ``noise_frac`` sets realistic per-geo-week noise.
    Returns (panel_df, geo_truth).
    """
    rng = np.random.default_rng(seed)
    T = len(nat_df)
    week = nat_df["week"].to_numpy()
    week_idx = np.arange(T)
    ctrl = _national_controls(week_idx)
    seasonality = ctrl["seasonality"]
    comp = nat_df["competitor_pressure"].to_numpy()
    nonmedia = (BASELINE + ctrl["trend"] + seasonality + ctrl["promo_contrib"]
                + ctrl["price_contrib"] + COMP_COEF * (comp - comp.mean())
                + HOLIDAY_EFFECT * nat_df["holiday_flag"].to_numpy())

    size = np.exp(rng.normal(0, 0.45, n_geos))
    size = size / size.sum()                       # population shares, sum to 1

    # latent geo×time demand confounder: AR(1) over time, demeaned across geos each week so it
    # neither changes national totals nor is captured by a national seasonal baseline.
    d = np.zeros((n_geos, T))
    phi = 0.6
    eps = rng.normal(0, 1, (n_geos, T))
    d[:, 0] = eps[:, 0]
    for t in range(1, T):
        d[:, t] = phi * d[:, t - 1] + np.sqrt(1 - phi ** 2) * eps[:, t]
    d -= d.mean(axis=0, keepdims=True)             # zero-mean across geos each week
    spend_tilt = np.exp(0.9 * confound * d)        # marketer targets high-demand geo-weeks
    # an observable but IMPERFECT proxy for the latent demand (e.g. regional search interest / app
    # rank): correlated with d but noisy. A real analyst might have this; feeding it as a control is
    # how you fight an otherwise-unobserved geo confounder. Drawn from a DEDICATED rng so adding the
    # proxy does not perturb the rest of the panel's random stream.
    demand_proxy = d + np.random.default_rng(seed + 4242).normal(0, proxy_noise, (n_geos, T))
    # a near-perfect proxy too, to trace the other end of the control-quality spectrum.
    demand_proxy_hi = d + np.random.default_rng(seed + 4343).normal(0, 0.22, (n_geos, T))

    rows = []
    summed_contrib = {c: np.zeros(T) for c in CHANNELS}
    for c in CHANNELS:
        p = CH[c]
        I_c = nat_df[f"{c}_impressions"].to_numpy(float)         # national impressions
        # idiosyncratic, time-varying geo multipliers, tilted toward demand, normalised so geo
        # impressions still sum to the national series each week.
        m_raw = np.exp(rng.normal(0, idio_sigma, (n_geos, T))) * spend_tilt
        w = size[:, None] * m_raw
        share = w / w.sum(axis=0, keepdims=True)                 # geo share of national impr each week
        I_g = share * I_c[None, :]                               # geo impressions, sum_g = I_c
        hs_used = truth["channels"][c]["half_sat"]               # effective half-sat (matches national)
        for g in range(n_geos):
            percap = np.divide(I_g[g], size[g])                  # geo per-capita exposure
            ad = geometric_adstock(percap, p["theta"])           # non-normalized (matches national)
            contrib = size[g] * p["beta"] * hill_saturation(ad, hs_used, p["slope"])
            summed_contrib[c] += contrib
            for t in range(T):
                rows.append((g, t, c, float(I_g[g, t]), float(I_g[g, t] / p["imp_per_dollar"]),
                             float(contrib[t])))

    # assemble wide geo panel
    import collections
    by_gt = collections.defaultdict(dict)
    contrib_gt = collections.defaultdict(float)
    spend_gt = collections.defaultdict(float)
    for g, t, c, imp, spend, contrib in rows:
        by_gt[(g, t)][f"{c}_impressions"] = imp
        by_gt[(g, t)][f"{c}_spend"] = spend
        contrib_gt[(g, t)] += contrib
        spend_gt[(g, t)] += spend

    panel = []
    demand_series, spend_series = [], []           # for the realized-confound diagnostic
    for g in range(n_geos):
        for t in range(T):
            demand_mult = 1.0 + demand_rel * confound * d[g, t]
            base = nonmedia[t] * size[g] * demand_mult           # demand lifts NON-media conversions
            conv = base + contrib_gt[(g, t)] + rng.normal(0, noise_frac * max(base, 1.0))
            row = dict(geo=f"geo_{g:03d}", week=week[t], population=float(size[g]),
                       conversions=float(max(conv, 0.0)),
                       promo_flag=float(ctrl["promo_flag"][t]),
                       price_index=float(ctrl["price_index"][t]),
                       competitor_pressure=float(comp[t]),
                       holiday_flag=float(nat_df["holiday_flag"].to_numpy()[t]),
                       demand_proxy=float(demand_proxy[g, t]),
                       demand_proxy_hi=float(demand_proxy_hi[g, t]))
            row.update(by_gt[(g, t)])
            panel.append(row)
            demand_series.append(d[g, t])
            spend_series.append(spend_gt[(g, t)] / size[g])      # per-capita total spend
    panel_df = pd.DataFrame(panel)

    realized_confound = float(np.corrcoef(np.array(spend_series), np.array(demand_series))[0, 1])
    proxy_fidelity = float(np.corrcoef(demand_proxy.ravel(), d.ravel())[0, 1])
    geo_truth = dict(
        n_geos=n_geos, idio_sigma=float(idio_sigma),
        confound=float(confound), noise_frac=float(noise_frac),
        realized_corr_spend_demand=realized_confound,
        demand_proxy_fidelity=proxy_fidelity,
        demand_proxy_hi_fidelity=float(np.corrcoef(demand_proxy_hi.ravel(), d.ravel())[0, 1]),
        avg_contribution_decomposition={c: float(summed_contrib[c].mean()) for c in CHANNELS},
        note="Geo-world truth: summed-across-geos avg weekly contribution per channel. Slightly "
             "below the national truth by the Hill aggregation (Jensen) gap.",
    )
    return panel_df, geo_truth


def main():
    ap = argparse.ArgumentParser(description="Generate DraftZone v2 synthetic data.")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--sealed-dir", default=str(SEALED_DIR))
    ap.add_argument("--seasonal-saturation", action="store_true",
                    help="time-varying saturation (channels more saturated at the NFL peak)")
    ap.add_argument("--saturation-scale", type=float, default=1.0,
                    help=">1 = less saturated / more headroom; <1 = more saturated")
    ap.add_argument("--confound", type=float, default=TARGET_CONFOUND,
                    help="target corr(total spend, season): lower = weaker spend↔demand confound")
    ap.add_argument("--hetero-geos", action="store_true",
                    help="test markets scatter (normal) around each channel's national saturation")
    ap.add_argument("--hetero-sigma", type=float, default=0.12,
                    help="std of per-market saturation around national (with --hetero-geos)")
    ap.add_argument("--spend-ladder", action="store_true",
                    help="also emit data/spend_ladder.csv: multi-cell experiments that MEASURE "
                         "each channel's response curve (several spend levels) instead of assuming it")
    ap.add_argument("--ladder-size-frac", type=float, default=0.06,
                    help="aggregate share of national scale each ladder test market represents")
    ap.add_argument("--geo-panel", action="store_true",
                    help="also emit data/geo_panel.csv: a multi-geo panel (geos sum to national) "
                         "for geo-level engines like Meridian")
    ap.add_argument("--n-geos", type=int, default=50)
    ap.add_argument("--geo-confound", type=float, default=1.0,
                    help="strength of the latent geo×time demand confounder (targeted spend + "
                         "demand-driven baseline). 0 = clean idealized panel; ~1 = realistic")
    ap.add_argument("--geo-noise-frac", type=float, default=0.18,
                    help="per-geo-week conversion noise as a fraction of the geo baseline")
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    sealed_dir = pathlib.Path(args.sealed_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)

    nat_df, truth = generate_national(seed=args.seed, seasonal_saturation=args.seasonal_saturation,
                                      confound_target=args.confound,
                                      saturation_scale=args.saturation_scale)
    d = truth["avg_contribution_decomposition"]
    nonmedia = (d["baseline"] + d["trend"] + d["seasonality"] + d["promo"]
                + d["price"] + d["competitor"] + d["holiday"])
    national_ctx = {c: dict(imp_mean=float(nat_df[f"{c}_impressions"].mean()),
                            hs=truth["channels"][c]["half_sat"], beta=truth["channels"][c]["beta"],
                            nonmedia=float(nonmedia)) for c in CHANNELS}
    geo_df, exp_truth, assignments = generate_geo_experiments(
        seed=args.seed // 2 + 808, national_ctx=national_ctx,
        hetero_sigma=(args.hetero_sigma if args.hetero_geos else 0.0))
    truth["experiments"] = exp_truth

    nat_df.to_csv(data_dir / "national_weekly.csv", index=False)
    geo_df.to_csv(data_dir / "geo_experiments.csv", index=False)

    ladder_size_frac = None
    if args.spend_ladder:
        ladder_df, ladder_truth, ladder_size_frac = generate_spend_ladder(
            seed=args.seed // 2 + 909, national_ctx=national_ctx,
            size_frac=args.ladder_size_frac,
            hetero_sigma=(args.hetero_sigma if args.hetero_geos else 0.12))
        for c in CHANNELS:  # stamp the national target into the sealed ladder truth (datagen may)
            ladder_truth[c]["true_national_avg_contrib"] = truth["avg_contribution_decomposition"][f"media_{c}"]
        truth["spend_ladder"] = ladder_truth
        ladder_df.to_csv(data_dir / "spend_ladder.csv", index=False)

    if args.geo_panel:
        panel_df, geo_truth = generate_geo_panel(nat_df, truth, seed=args.seed // 3 + 606,
                                                 n_geos=args.n_geos, confound=args.geo_confound,
                                                 noise_frac=args.geo_noise_frac)
        truth["geo_panel"] = geo_truth
        panel_df.to_csv(data_dir / "geo_panel.csv", index=False)

    config = dict(
        seed=args.seed,
        n_weeks=T_NATIONAL,
        channels=CHANNELS,
        target_confound=float(args.confound),
        realized_confound=truth["meta"]["realized_corr_totalspend_season"],
        geo_calendar=GEO_CALENDAR,
        n_markets=N_MARKETS,
        geo_assignments=assignments,
        seasonal_saturation=bool(args.seasonal_saturation),
        saturation_scale=float(args.saturation_scale),
        hetero_geos=bool(args.hetero_geos),
        hetero_sigma=float(args.hetero_sigma) if args.hetero_geos else 0.0,
        spend_ladder=bool(args.spend_ladder),
        ladder_levels=list(map(float, LADDER_LEVELS)) if args.spend_ladder else None,
        ladder_size_frac=float(ladder_size_frac) if ladder_size_frac is not None else None,
        geo_panel=bool(args.geo_panel),
        n_geos=int(args.n_geos) if args.geo_panel else None,
        geo_confound=float(args.geo_confound) if args.geo_panel else None,
        geo_noise_frac=float(args.geo_noise_frac) if args.geo_panel else None,
        note="Public config — contains NO true model parameters (see data_sealed/ for truth).",
    )
    with open(data_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(sealed_dir / "ground_truth.json", "w") as f:
        json.dump(truth, f, indent=2)

    corr = truth["meta"]["realized_corr_totalspend_season"]
    base_share = BASELINE / nat_df["conversions"].mean()
    print(f"Wrote {data_dir/'national_weekly.csv'} ({len(nat_df)} weeks)")
    print(f"Wrote {data_dir/'geo_experiments.csv'} ({geo_df['channel'].nunique()} channel experiments)")
    if args.spend_ladder:
        print(f"Wrote {data_dir/'spend_ladder.csv'} "
              f"({len(LADDER_LEVELS)}-cell ladder/channel, size_frac={ladder_size_frac})")
    if args.geo_panel:
        gtp = truth["geo_panel"]
        gp = gtp["avg_contribution_decomposition"]
        nat_med = sum(truth['avg_contribution_decomposition'][f'media_{c}'] for c in CHANNELS)
        print(f"Wrote {data_dir/'geo_panel.csv'} ({args.n_geos} geos; "
              f"geo media total {sum(gp.values()):.0f} vs national {nat_med:.0f} conv/wk; "
              f"confound={args.geo_confound}, noise={args.geo_noise_frac:.0%}, "
              f"realized corr(spend,demand)={gtp['realized_corr_spend_demand']:+.2f})")
    print(f"Realized confound corr(total spend, season) = {corr:.3f}  (target {args.confound})")
    print(f"Baseline share of conversions ~ {base_share:.0%}")
    print("Sealed truth written to data_sealed/ground_truth.json (pipeline must not read it).")


if __name__ == "__main__":
    main()
