# PROTOTYPE_FINDINGS.md — What the prototype established

Built across a chat session on a v1 synthetic dataset (~140 weeks, 0.67 confound, social-only
experiment anchor). Numbers below are from that run; the v2 dataset will differ but the
*lessons* are what carries.

## Stage-by-stage results
- **Naive model:** raw-spend OLS credited TV 35 conv/wk (truth 380) — blinded to carryover —
  and dumped the effect into the intercept. Adding the true season control made it *worse*
  (media collapsed, brand went negative). R²=0.63–0.75 throughout. **Good fit ≠ good attribution.**
- **Transforms:** correct adstock+Hill raised correlation of transformed media with true
  contribution to ~1.0 for all channels (TV +0.37 raw → +1.00 transformed). Functional form is
  the lever.
- **Frequentist fit:** R²=0.835 but degenerate — social contribution 16,512 (truth 294),
  baseline −15,184 (truth 1200), all slopes railed to bounds. A confident, catastrophic answer.
- **Bayesian fit:** priors killed the degeneracy (baseline ~2000, social ~119, nothing
  exploded) and gave honest wide intervals — but still mis-split correlated channels.
- **Grading:** MAPE 5.4%, θ recovered for 4/5 channels; BUT 89% predictive interval covered
  only **53%** of weeks (overconfident), and media was under-credited ~38% with the slack
  absorbed into baseline. Channel-contribution intervals contained truth for only 2/5.
- **Experiment:** randomized geo DiD recovered social's causal lift to **99%** (17.3 vs 17.4
  per market-wk) despite seasonality. Fed back as a prior, social's estimate moved from a
  confidently-wrong 55 [4,185] to a calibrated **280 [209,350]** (truth 294).
- **Revenue/ROI:** blended LTV $220 (±range). Social avg ROI 2.49 but **marginal ROI 0.54**
  (saturated); brand marginal ROI ~3.0 but interval [0.3, 7.4] — too uncertain to bet big.
- **Optimization:** point-estimate optimizer promised +12.3% conversions; across uncertainty,
  **only "cut search" was robust** — every other move, including the optimizer's biggest bet
  (double brand), was test-first.

## The lessons that must survive into v2
1. You are never *sure*, only *calibrated*. Ranges, not points.
2. Confounding biases observational MMM; priors bound the damage but don't cure it.
3. Randomized experiments are the confound-immune tiebreaker; one anchor improves the whole
   decomposition. **v2's rotating calendar lets us anchor every channel → confident moves across
   the board.**
4. Optimize across uncertainty; act on robust moves; route the rest to experiments.
5. Average ROI is a trap; marginal ROI drives decisions.

## Known prototype bugs to NOT repeat (already fixed in spec)
- Geo experiment units: dividing national half_sat by N_markets saturated every market → ~0
  lift. v2 spec requires verifying markets sit on the responsive part of the Hill curve.
- Confound came out at 0.90 on first datagen run (season_coef too high) before retuning to 0.67.
  v2 generator must assert realized corr matches target.
- Multi-core PyMC segfaulted in the sandbox; fits ran cores=1. On a real VM use multi-core;
  in Actions avoid sampling entirely.
- ArviZ summary column names and the DataTree posterior API shifted between versions — pin
  versions in the lockfile.

## Honest limitations carried forward (surface on the dashboard)
- Only one channel was experimentally anchored in v1 → only its ROI fully trustworthy. v2 fixes
  this by design.
- The experiment→prior translation was idealized (anchor mean set near truth). v2 should derive
  the prior from the DiD estimate + its CI, propagating the experiment's own uncertainty.
- LTV was single/blended with a range, not per-channel (deliberate; per-channel LTV is usually
  unmeasurable early).
