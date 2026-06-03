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


def generate_national(seed=2024, seasonal_saturation=False, saturation_scale=1.0):
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
    season_scale = _tune_confound(expected_demand, seasonality, spend_seed, T_NATIONAL)
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
            holiday_effect=HOLIDAY_EFFECT, target_corr_spend_season=TARGET_CONFOUND,
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
                             national_sat=None, hetero_sigma=0.0):
    """One randomized geo-experiment per channel, staggered across the calendar.

    Returns (tidy_dataframe, truth_dict, assignment_dict).

    Default (hetero_sigma=0): every market sits at ~half-saturation (powered, responsive).
    Heterogeneous mode (hetero_sigma>0 + national_sat): each market's operating saturation is
    drawn ~ Normal(channel's national saturation, hetero_sigma), so the test markets SCATTER around
    the national operating point (big metros saturated, small DMAs with headroom). Centering the
    geos on national means the pooled lift measures ~the national marginal — the normalization that
    makes the experiment->model translation honest.
    """
    rows = []
    exp_truth = {}
    assignments = {}

    for ci, channel in enumerate(CHANNELS):
        p = CH[channel]
        g = GEO_DESIGN[channel]
        hs, slope, theta, beta = g["hs_mkt"], p["slope"], p["theta"], p["beta"]
        rng = np.random.default_rng(seed + 101 * (ci + 1))
        week = np.arange(T_exp)
        camp_end = min(pre_period + camp_len, T_exp)
        in_campaign = (week >= pre_period) & (week < camp_end)
        season = 250 * (0.5 + 0.5 * np.sin(2 * np.pi * (week - 3) / T_exp))  # shared confounder

        market_base = rng.normal(320, 55, N_MARKETS)
        perm = rng.permutation(N_MARKETS)
        treat_idx = perm[: N_MARKETS // 2]
        treat_mask = np.zeros(N_MARKETS, bool)
        treat_mask[treat_idx] = True

        hetero = hetero_sigma > 0 and national_sat is not None
        if hetero:
            # per-market mean adstocked exposure that lands each market at target saturation f_m
            f = np.clip(rng.normal(national_sat[channel], hetero_sigma, N_MARKETS), 0.03, 0.92)
            a_target = hs * (f / (1 - f)) ** (1.0 / slope)
            increment = 0.7 * a_target                      # per-market campaign bump (impr)
            season_ripple = 0.10 * a_target[:, None] * (season / season.max())[None, :]
            base = a_target[:, None] + season_ripple        # N_MARKETS x T_exp mean BAU
        else:
            a_target = np.full(N_MARKETS, g["bau"], float)
            increment = np.full(N_MARKETS, g["increment"], float)
            base = (g["bau"] + g["season_sens"] * season)[None, :].repeat(N_MARKETS, 0)

        def contrib(impr):
            return beta * hill_saturation(geometric_adstock(impr, theta, normalize=True), hs, slope)

        true_incs, ratios, fracs = [], [], []
        for m in range(N_MARKETS):
            treated = bool(treat_mask[m])
            mean_bau = base[m]
            bau = np.clip(mean_bau + rng.normal(0, a_target[m] * 0.12, T_exp), 0, None)
            extra = np.where(in_campaign, increment[m], 0.0) if treated else np.zeros(T_exp)
            impr = bau + extra
            spend = impr / p["imp_per_dollar"]
            conv = market_base[m] + season + contrib(impr) + rng.normal(0, 15, T_exp)
            for w in range(T_exp):
                rows.append(dict(
                    channel=channel, market=m, week=int(w), treated=int(treated),
                    spend=float(spend[w]), impressions=float(impr[w]), conversions=float(conv[w]),
                    campaign_window=int(bool(in_campaign[w])), pre_period=int(w < pre_period),
                ))
            inc_with = contrib(np.where(in_campaign, mean_bau + increment[m], mean_bau))
            inc_without = contrib(mean_bau)
            true_incs.append(float((inc_with - inc_without)[in_campaign].mean()))
            ad_m = geometric_adstock(mean_bau, theta, normalize=True).mean()
            ratios.append(ad_m / hs)
            fracs.append(float(hill_saturation(np.array([ad_m]), hs, slope)[0]))

        ratio = float(np.mean(ratios))
        assert 0.2 <= ratio <= 3.0, (f"{channel}: mean market adstock/half_sat {ratio:.2f} off-band")
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
                                    treated_markets=sorted(int(x) for x in treat_idx))

    geo_df = pd.DataFrame(rows)
    return geo_df, exp_truth, assignments


def main():
    ap = argparse.ArgumentParser(description="Generate DraftZone v2 synthetic data.")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--sealed-dir", default=str(SEALED_DIR))
    ap.add_argument("--seasonal-saturation", action="store_true",
                    help="time-varying saturation (channels more saturated at the NFL peak)")
    ap.add_argument("--saturation-scale", type=float, default=1.0,
                    help=">1 = less saturated / more headroom; <1 = more saturated")
    ap.add_argument("--hetero-geos", action="store_true",
                    help="test markets scatter (normal) around each channel's national saturation")
    ap.add_argument("--hetero-sigma", type=float, default=0.12,
                    help="std of per-market saturation around national (with --hetero-geos)")
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    sealed_dir = pathlib.Path(args.sealed_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)

    nat_df, truth = generate_national(seed=args.seed, seasonal_saturation=args.seasonal_saturation,
                                      saturation_scale=args.saturation_scale)
    national_sat = {c: truth["channels"][c]["mean_contrib"] / truth["channels"][c]["beta"]
                    for c in CHANNELS}
    geo_df, exp_truth, assignments = generate_geo_experiments(
        seed=args.seed // 2 + 808, national_sat=national_sat,
        hetero_sigma=(args.hetero_sigma if args.hetero_geos else 0.0))
    truth["experiments"] = exp_truth

    nat_df.to_csv(data_dir / "national_weekly.csv", index=False)
    geo_df.to_csv(data_dir / "geo_experiments.csv", index=False)

    config = dict(
        seed=args.seed,
        n_weeks=T_NATIONAL,
        channels=CHANNELS,
        target_confound=TARGET_CONFOUND,
        realized_confound=truth["meta"]["realized_corr_totalspend_season"],
        geo_calendar=GEO_CALENDAR,
        n_markets=N_MARKETS,
        geo_assignments=assignments,
        seasonal_saturation=bool(args.seasonal_saturation),
        saturation_scale=float(args.saturation_scale),
        hetero_geos=bool(args.hetero_geos),
        hetero_sigma=float(args.hetero_sigma) if args.hetero_geos else 0.0,
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
    print(f"Realized confound corr(total spend, season) = {corr:.3f}  (target {TARGET_CONFOUND})")
    print(f"Baseline share of conversions ~ {base_share:.0%}")
    print("Sealed truth written to data_sealed/ground_truth.json (pipeline must not read it).")


if __name__ == "__main__":
    main()
