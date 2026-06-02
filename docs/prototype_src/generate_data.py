"""
DraftZone MMM — Stage 1: Synthetic Data Generator
==================================================
A fictional DFS / fantasy-sports subscription app.

Causal chain encoded in the truth:
    spend  --(noisy, mildly nonlinear)-->  impressions
    impressions --adstock(theta)--> --Hill(half_sat, slope)--> x beta --> channel contribution
    conversions = baseline + trend + seasonality + promo + price + competitor + holiday
                  + sum(channel contributions) + noise

Everything true is saved to ground_truth.json so later stages can be graded.
Design choices reflecting earlier discussion:
 - 5 channels with deliberately DISTINCT adstock + saturation (recovery is a real test)
 - spend is partially driven by expected seasonal demand -> ~0.6 confound (realistic)
 - conversions driven by IMPRESSIONS (not spend) -> lets us show spend-vs-impressions later
"""
import numpy as np
import pandas as pd
import json

rng = np.random.default_rng(2024)
T = 140
week_idx = np.arange(T)
dates = pd.date_range("2023-01-01", periods=T, freq="W-SUN")

# ----------------------------------------------------------------------
# 1. CONTROLS / BASELINE DEMAND  (the non-media truth)
# ----------------------------------------------------------------------
BASELINE = 1200.0                              # intercept: organic conversions/wk
TREND_PER_WK = 4.0                             # slow organic growth
trend = TREND_PER_WK * week_idx

# NFL-season seasonality: annual cycle peaking ~Sep-Jan (weeks ~35-52 / 0-4)
# Build a smooth seasonal curve + a sharper playoff/Super Bowl bump.
phase = 2*np.pi*(week_idx - 35)/52.0
seasonal_smooth = 350.0 * (0.5 + 0.5*np.cos(phase))      # 0..350, peak in fall/winter
# playoff bump: weeks near Jan (Super Bowl ~ week 5-6 of year)
playoff = np.zeros(T)
for yr_start in [0, 52, 104]:
    for w in range(2, 7):                      # Jan playoff weeks
        if yr_start + w < T:
            playoff[yr_start + w] += 120.0
seasonality = seasonal_smooth + playoff

# Promo weeks (signup discounts): a handful of bursts
promo_flag = np.zeros(T)
for w in [10, 11, 33, 60, 61, 85, 110, 111]:
    if w < T: promo_flag[w] = 1
PROMO_EFFECT = 180.0
promo_contrib = PROMO_EFFECT * promo_flag

# Price index: subscription price with two increases (step ups). Mild negative elasticity.
price_index = np.ones(T) * 100.0
price_index[45:] = 108.0
price_index[100:] = 115.0
PRICE_COEF = -6.0                              # conversions per index point
price_contrib = PRICE_COEF * (price_index - 100.0)

# Competitor pressure proxy (exogenous, suppresses conversions)
competitor_pressure = np.clip(50 + 30*np.sin(2*np.pi*week_idx/26 + 1.0)
                              + rng.normal(0, 8, T), 0, None)
COMP_COEF = -2.2
comp_contrib = COMP_COEF * (competitor_pressure - competitor_pressure.mean())

# Holiday flag (Super Bowl, Thanksgiving-ish, season kickoff)
holiday_flag = np.zeros(T)
for w in [5, 35, 47, 57, 87, 99, 109]:
    if w < T: holiday_flag[w] = 1
HOLIDAY_EFFECT = 90.0
holiday_contrib = HOLIDAY_EFFECT * holiday_flag

# "expected demand" signal media planners react to (drives the confound)
expected_demand = (seasonality - seasonality.mean()) / seasonality.std()

# ----------------------------------------------------------------------
# 2. CHANNELS: spend -> impressions -> adstock -> saturation -> contribution
# ----------------------------------------------------------------------
def adstock(x, theta):
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + theta * out[t-1]
    return out

def hill(x, half_sat, slope):
    x = np.maximum(x, 0)
    return x**slope / (x**slope + half_sat**slope + 1e-9)

# Per-channel TRUE parameters.
# base_spend = typical weekly $; season_coef = how much planners ramp w/ demand (confound);
# cpm-ish = impressions per $ ; theta = adstock ; hs/slope on IMPRESSION-adstock scale ; beta = max conversions
# season_coef reduced and independent noise raised so realized corr lands ~0.6.
CH = {
    "tv": dict(base_spend=42000, season_coef=9000, flight=True,
               imp_per_dollar=9.0,  noise=0.10, theta=0.75, hs=160000, slope=1.6, beta=420),
    "search": dict(base_spend=16000, season_coef=5000, flight=False,
               imp_per_dollar=22.0, noise=0.06, theta=0.10, hs=120000, slope=2.0, beta=300),
    "social": dict(base_spend=24000, season_coef=5000, flight=False,
               imp_per_dollar=30.0, noise=0.08, theta=0.40, hs=260000, slope=1.5, beta=350),
    "affiliate": dict(base_spend=9000, season_coef=3000, flight=True,
               imp_per_dollar=15.0, noise=0.12, theta=0.30, hs=45000, slope=2.4, beta=180),
    "brand": dict(base_spend=6000, season_coef=1500, flight=False,
               imp_per_dollar=12.0, noise=0.10, theta=0.60, hs=70000, slope=1.3, beta=120),
}

data = {"week": dates}
ground_truth = {
    "meta": dict(T=T, baseline=BASELINE, trend_per_wk=TREND_PER_WK,
                 promo_effect=PROMO_EFFECT, price_coef=PRICE_COEF,
                 comp_coef=COMP_COEF, holiday_effect=HOLIDAY_EFFECT,
                 target_corr_spend_season="~0.6 (realistic)"),
    "channels": {}
}

channel_contribs = {}
spend_for_corr = []
for name, p in CH.items():
    # ----- spend: base + seasonal ramp (the confound) + independent noise + optional flighting
    season_part = p["season_coef"] * expected_demand
    indep = rng.normal(0, p["base_spend"]*0.35, T)        # independent variation (identifiability!)
    spend = p["base_spend"] + season_part + indep
    if p["flight"]:
        # flighting: zero out some weeks to create bursts (TV/affiliate go dark sometimes)
        dark = rng.random(T) < 0.25
        spend = np.where(dark, spend*0.1, spend)
    spend = np.clip(spend, 0, None)

    # ----- impressions: noisy, mildly concave function of spend
    imp_mean = p["imp_per_dollar"] * spend**0.97          # slight concavity in buying efficiency
    impressions = imp_mean * rng.normal(1.0, p["noise"], T)
    impressions = np.clip(impressions, 0, None)

    # ----- adstock then saturation on impressions
    ad = adstock(impressions, p["theta"])
    sat = hill(ad, p["hs"], p["slope"])                   # 0..1 fraction of ceiling
    contrib = p["beta"] * sat

    data[f"{name}_spend"] = spend
    data[f"{name}_impressions"] = impressions
    channel_contribs[name] = contrib
    spend_for_corr.append(spend)

    ground_truth["channels"][name] = dict(
        theta=p["theta"], half_sat=p["hs"], slope=p["slope"], beta=p["beta"],
        imp_per_dollar=p["imp_per_dollar"], mean_spend=float(spend.mean()),
        mean_contrib=float(contrib.mean())
    )

# ----------------------------------------------------------------------
# 3. ASSEMBLE CONVERSIONS
# ----------------------------------------------------------------------
media_total = sum(channel_contribs.values())
expected_conversions = (BASELINE + trend + seasonality + promo_contrib
                        + price_contrib + comp_contrib + holiday_contrib + media_total)
# multiplicative + additive noise
noise_mult = rng.normal(1.0, 0.05, T)
noise_add = rng.normal(0, 35, T)
conversions = np.clip(expected_conversions * noise_mult + noise_add, 0, None)

# store control series
data["conversions"] = conversions
data["seasonality_true"] = seasonality
data["trend_true"] = trend
data["promo_flag"] = promo_flag
data["price_index"] = price_index
data["competitor_pressure"] = competitor_pressure
data["holiday_flag"] = holiday_flag

df = pd.DataFrame(data)

# ----------------------------------------------------------------------
# 4. DIAGNOSTICS + SAVE
# ----------------------------------------------------------------------
total_spend_series = np.sum(spend_for_corr, axis=0)
corr = np.corrcoef(total_spend_series, seasonality)[0,1]
ground_truth["meta"]["realized_corr_totalspend_season"] = float(corr)

# true contribution decomposition (avg share) -- the thing the model must recover
contrib_table = {
    "baseline": float(BASELINE),
    "trend": float(trend.mean()),
    "seasonality": float(seasonality.mean()),
    "promo": float(promo_contrib.mean()),
    "price": float(price_contrib.mean()),
    "competitor": float(comp_contrib.mean()),
    "holiday": float(holiday_contrib.mean()),
}
for name in CH: contrib_table[f"media_{name}"] = float(channel_contribs[name].mean())
ground_truth["avg_contribution_decomposition"] = contrib_table

df.to_csv("draftzone_mmm_data.csv", index=False)
with open("ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=2)

# ----- console report -----
print("="*60)
print(f"Generated {T} weeks: {dates[0].date()} -> {dates[-1].date()}")
print(f"Columns: {list(df.columns)}")
print(f"\nRealized corr(total spend, seasonality) = {corr:.3f}  (target ~0.6)")
print(f"\nConversions: mean={conversions.mean():.0f}  min={conversions.min():.0f}  max={conversions.max():.0f}")
print("\nPer-channel correlation of spend with seasonality:")
for name, s in zip(CH, spend_for_corr):
    print(f"  {name:10s}: {np.corrcoef(s, seasonality)[0,1]:+.2f}   mean spend ${s.mean():,.0f}")
print("\nTrue avg contribution decomposition (conversions/wk):")
tot = sum(contrib_table.values())
for k,v in contrib_table.items():
    print(f"  {k:16s}: {v:8.1f}  ({100*v/tot:4.1f}%)")
print(f"  {'TOTAL':16s}: {tot:8.1f}")
print("\nSaved: draftzone_mmm_data.csv, ground_truth.json")
