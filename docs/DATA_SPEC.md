# DATA_SPEC.md — New Synthetic Dataset ("DraftZone v2")

Goal: same DFS subscription premise as the prototype, but designed so the **full
experimentation path** is possible — a randomized geo-experiment available for **every**
channel, enabling channel-by-channel causal anchoring and therefore **confident
recommendations across the board**.

## National time series (same shape as prototype)
- ~156 weekly rows (3 years), week-ending dates.
- Target: `conversions` (weekly new paid signups).
- 5 channels: tv, search, social, affiliate, brand. For each: `{c}_spend`, `{c}_impressions`.
- Causal chain: spend → impressions (noisy, mildly concave) → geometric adstock(θ) →
  Hill(half_sat, slope) → ×β → channel contribution.
- Controls: baseline intercept, slow trend, NFL-season seasonality (smooth + playoff bump),
  promo_flag, price_index (step changes), competitor_pressure, holiday_flag.
- Noise: multiplicative (~5%) + additive.
- **Confound knob:** each channel's spend = base + `season_coef`·expected_demand + indep_noise.
  Tune `season_coef`/indep so realized corr(total_spend, season) ≈ **0.6** (configurable).
- Keep per-channel θ/half_sat/slope **distinct** (TV high θ/late sat; search low θ/early sat;
  etc.) so recovery is a genuine test.

## NEW: rotating geo-experiment calendar
Generate a **market-level panel** that mirrors the national truth and supports one clean
randomized experiment **per channel**, staggered over time (a realistic "always-on testing"
program).

For each channel `c` in a defined calendar (e.g. social Q1, search Q2, affiliate Q3, brand Q4,
tv Q1-next):
- `N_markets` (e.g. 80) comparable geos; per-market heterogeneous baseline.
- Shared seasonality across markets (the confounder must persist at market level — BAU spend is
  season-correlated, just like nationally).
- **Random assignment** to treatment/control (record the seed and assignment).
- Treatment markets receive an **incremental** spend/impression bump for that channel during a
  defined window (with a pre-period for diff-in-differences).
- **Units must be coherent:** per-market half-sat must place markets on the *responsive* part
  of the Hill curve, NOT saturated. (Prototype bug: dividing national half_sat by N put markets
  at ~99.8% saturation, so the campaign drove ~0 lift. Verify mean adstocked market impressions
  are within ~0.3–2× of per-market half_sat.)
- Output one tidy table: `data/geo_experiments.csv` with columns
  `channel, market, week, treated, spend, impressions, conversions, campaign_window, pre_period`.

## Files the generator writes
- `data/national_weekly.csv` — the public modeling dataset.
- `data/geo_experiments.csv` — the rotating experiment panel.
- `data/config.json` — confound level, seeds, calendar, market counts (public, no truth).
- `data_sealed/ground_truth.json` — **SEALED.** Every true θ, half_sat, slope, β, baseline,
  control coefs, per-channel true avg contribution, and each experiment's true incremental
  effect. The pipeline must never read this; only `evaluate.py` may.

## Acceptance checks (write as tests)
- realized corr(total spend, season) within ±0.05 of target.
- every channel's geo-experiment recovers its true incremental lift to within ~15% via DiD
  on the generated panel (proves the experiment path is wired correctly and units are sane).
- channel θ values are distinct and span low→high.
- baseline share of conversions is realistic (~35–50%).
