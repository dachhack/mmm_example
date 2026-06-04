# DraftZone MMM

An end-to-end, **honest** Media Mix Modeling project on a synthetic DFS / fantasy-sports
subscription app. It demonstrates not just how to *build* an MMM, but how to know **when to
trust it** — by grading every estimate against a sealed ground truth and using randomized
geo-experiments to produce **confident, uncertainty-aware budget recommendations**.

> **The thesis:** an observational MMM is biased by the correlation between ad spend and
> seasonal demand. Bayesian priors bound the damage but don't cure it. Randomized
> geo-experiments give confound-immune causal anchors that, fed back into the model, repair the
> decomposition. Decisions are made on **marginal ROI under uncertainty** — acting only on
> **robust** moves and routing the rest to experiments.

## What makes this different
- **Sealed answer key.** Every true parameter lives in `data_sealed/ground_truth.json`, which
  the modeling pipeline is forbidden to read (enforced by `tests/test_no_truth_leak.py`). Only
  the evaluation step may open it — so recovery results are earned, not peeked.
- **Full experimentation path.** The dataset ships a *rotating geo-experiment calendar* — one
  randomized test per channel — so every channel can be causally anchored, turning "cut search,
  test everything else" into confident moves across the board.
- **Uncertainty everywhere.** No bare point estimates. ROI and recovered parameters carry
  credible intervals; the optimizer separates confident vs test-first recommendations.
- **Interactive dashboard** on GitHub Pages illustrates the process and findings.

## Repository map
```
CLAUDE.md                 orientation for Claude Code (read first)
docs/
  DATA_SPEC.md            new dataset design (incl. rotating experiments)
  DASHBOARD_SPEC.md       interactive site spec + data contract
  INFRA.md                VM-fits / Actions-site split, Makefile targets
  PROTOTYPE_FINDINGS.md   distilled results & lessons from the prototype
  prototype_src/          reference scripts from the prototype (refactor, don't copy)
src/draftzone_mmm/        the package (transforms.py ported; rest are stubs to implement)
notebooks/                01–08 re-runnable narrative (see notebooks/README.md)
dashboard/                React/Vite interactive site (reads docs/data/*)
data/ , data_sealed/      public data ; SEALED truth
tests/                    incl. the no-truth-leak guard
scripts/run_fits.sh       heavy VM pipeline -> produces docs/data/*
.github/workflows/        ci.yml (tests) , pages.yml (build & deploy)
```

## Quick start (for Claude Code)
1. Read `CLAUDE.md`, then the four `docs/*.md` specs.
2. Implement `src/draftzone_mmm/*` (start with `datagen.py` per `DATA_SPEC.md`; `transforms.py`
   is already ported and tested).
3. `make data && make fit && make experiments && make anchored && make evaluate && make figures`
   on a VM (see `docs/INFRA.md`).
4. Build the dashboard + notebooks; let Actions deploy Pages.
5. Confirm the dashboard shows the recovery scorecard, the per-channel experiment repair, and
   the interactive optimizer with confident/test-first verdicts.

## How the experiment anchoring works (v2)
The geo-experiment enters the model as a **DiD likelihood**, not a hand-set prior.
Difference-in-differences recovers each channel's causal lift (within ~7% of truth here),
and — with the test markets designed to sit near half-saturation — the model predicts that
lift as `beta_c * (Hill(a_high; half_sat=a_low, slope_c) - 0.5)` using the channel's *own*
`beta` and `slope`. This pins the channel **ceiling** (`beta`, the one quantity shared between
market and national scale) with a confound-immune measurement and breaks the `beta`↔`half_sat`
degeneracy that cripples observational MMM. The market→national translation is still idealized
(see the limitations) so the anchor SD is widened to reflect that.

## What this run found
- Dataset: 156 weeks, realized spend↔season confound **0.60**, baseline share **43%**, distinct
  θ spanning 0.10→0.75; every geo-experiment recovers its true lift within ~7% via DiD.
- Anchoring cuts per-channel **mean absolute error ~40%** (437 → 261 conversions/wk) and keeps
  **5/5** channel intervals covering truth, while the predictive interval still covers only ~35%
  of weeks — the **overconfidence** lesson, preserved and surfaced.
- ROI: TV's next dollar loses money (mROI ≈ 0.67, saturated); the only robust budget move is to
  **cut TV**, with affiliate the biggest-but-uncertain upside → test first.

## Run it locally
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[fit,dev]"
make data            # data/ + data_sealed/ (asserts confound/units/baseline)
make fit             # baseline Bayesian fit  -> artifacts/idata.nc  (slow; multi-core on a VM)
make experiments     # rotating geo DiD       -> artifacts/anchors.json
make anchored        # experiment-anchored fit-> artifacts/idata_anchored.nc
make evaluate figures # scorecard + docs/data/* contracts
cd dashboard && npm ci && npm run build   # -> static site in ../docs
```

## Run the findings-optimised MMM on your own data
The competition's conclusions are packaged in `draftzone_mmm.run`. It fits the configuration that
proved robust — regularised priors, seasonality control, uncertainty everywhere, a confound
diagnostic, marginal ROI, and a confident-vs-test-first recommendation:

```python
from draftzone_mmm.run import run_mmm

res = run_mmm(
    df,                              # one row per week
    kpi="conversions",               # your outcome column
    channels=["tv", "search", "social"],
    spend_suffix="_spend",           # -> tv_spend, search_spend, ...
    exposure_suffix="_impressions",  # optional; defaults to spend
    controls=["price", "promo"],     # optional observed confounders
    date="week",
)
print(res.summary())                 # contributions+CIs, confound, marginal ROI, verdicts
res.contributions; res.roi; res.recommend()
```
It is deliberately conservative: when the confound is high and the posterior is wide it tells you to
**test first** rather than hand you a false answer. For a head-to-head of engines on your data, use
the competition harness (`scripts/engine_leaderboard.py`).

Site pages: **[results & recommendations](docs/results/index.html)** ·
**[how the competition works](docs/process/index.html)** ·
**[multi-seed leaderboard](docs/robustness/index.html)**.

## Status
Productionized: new dataset + reproducible pipeline + interactive dashboard + graded recovery +
a multi-seed engine competition + a packaged, findings-optimised runner (`draftzone_mmm.run`).
See `docs/PROTOTYPE_FINDINGS.md` for the original prototype this was built from.
