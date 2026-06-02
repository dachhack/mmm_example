"""Generate the notebooks/01-08 narrative as .ipynb files.

These are the re-runnable record of the full path. They LOAD cached artifacts produced by
scripts/run_fits.sh (they do not re-sample the heavy Bayesian chains), per notebooks/README.md.

Run:  python scripts/make_notebooks.py
"""
import json
import pathlib

NB_DIR = pathlib.Path(__file__).resolve().parents[1] / "notebooks"


def nb(*cells):
    return {
        "cells": list(cells),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": _src(lines)}


def _src(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


NOTEBOOKS = {
    "01_data.ipynb": nb(
        md("# 01 · The data",
           "",
           "Generate/load the DraftZone v2 dataset and look at the national series, the spend,",
           "and the **confound** (spend rises with seasonal demand). No truth leak — we read",
           "`data/config.json`, never the sealed key."),
        code("import json, pandas as pd, matplotlib.pyplot as plt",
             "from draftzone_mmm import datagen  # run `python -m draftzone_mmm.datagen` first",
             "df = pd.read_csv('data/national_weekly.csv', parse_dates=['week'])",
             "cfg = json.load(open('data/config.json'))",
             "print('weeks:', len(df), '| realized confound corr(spend, season):', round(cfg['realized_confound'], 3))"),
        code("fig, ax = plt.subplots(figsize=(12, 4))",
             "ax.plot(df.week, df.conversions, lw=1)",
             "ax.set_title('Weekly conversions'); ax.set_ylabel('conversions/wk'); plt.tight_layout()"),
        code("# total spend vs (proxy) season: the ~0.6 confound that makes attribution hard",
             "chans = cfg['channels']",
             "tot = sum(df[f'{c}_spend'] for c in chans)",
             "ax = df.set_index('week')[[f'{c}_spend' for c in chans]].plot(figsize=(12,4), lw=1)",
             "ax.set_title('Per-channel spend'); plt.tight_layout()"),
        md("The causal chain is `spend -> impressions -> adstock(theta) -> Hill(half_sat, slope)",
           "-> x beta -> contribution`, summed with baseline + controls. ~43% of conversions are",
           "organic baseline. Each channel has a **distinct** theta/half_sat/slope so recovery is a",
           "genuine test."),
    ),
    "02_naive.ipynb": nb(
        md("# 02 · The naive model — good fit ≠ good attribution",
           "",
           "A raw-spend OLS (no adstock/saturation/season). It can post a respectable R² and still",
           "attribute wildly, dumping carryover into the intercept."),
        code("import json, numpy as np, pandas as pd",
             "df = pd.read_csv('data/national_weekly.csv', parse_dates=['week'])",
             "chans = ['tv','search','social','affiliate','brand']",
             "X = np.column_stack([df[f'{c}_spend'] for c in chans] + [np.arange(len(df)), np.ones(len(df))])",
             "coef, *_ = np.linalg.lstsq(X, df.conversions.values, rcond=None)",
             "yhat = X @ coef; r2 = 1 - ((df.conversions-yhat)**2).sum()/((df.conversions-df.conversions.mean())**2).sum()",
             "print('naive R^2 =', round(r2,3), '| intercept =', round(coef[-1]))"),
        code("# The eval scorecard (produced later by evaluate.py) carries the truth for contrast.",
             "sc = json.load(open('docs/data/scorecard.json'))['naive']",
             "for c in sc['channels']:",
             "    print(f\"{c['channel']:10s} true={c['true_contrib']:7.1f}  naive={c['naive_contrib']:8.1f}\")"),
        md("Adding the *true* season as a control makes it **worse**, not better — functional form",
           "(adstock + saturation) matters more than piling on controls."),
    ),
    "03_transforms.ipynb": nb(
        md("# 03 · The transforms — adstock & Hill",
           "",
           "Carryover (adstock, θ) and diminishing returns (Hill: half-sat, slope). Order matters:",
           "adstock first (accumulate exposure), then saturation (model the response)."),
        code("import numpy as np, matplotlib.pyplot as plt",
             "from draftzone_mmm.transforms import geometric_adstock, hill_saturation",
             "x = np.zeros(40); x[[5,6,18,19,20,30]] = 100",
             "fig, ax = plt.subplots(1,2, figsize=(12,3.5))",
             "for th in [0.0,0.4,0.7,0.9]:",
             "    ax[0].plot(geometric_adstock(x, th, normalize=True), label=f'θ={th}')",
             "ax[0].set_title('adstock / carryover'); ax[0].legend()",
             "xs = np.linspace(0,400,300)",
             "for hs,sl in [(80,1.0),(120,1.6),(160,2.6)]:",
             "    ax[1].plot(xs, hill_saturation(xs, hs, sl), label=f'hs={hs}, slope={sl}')",
             "ax[1].set_title('Hill / diminishing returns'); ax[1].legend(); plt.tight_layout()"),
        md("Correctly transformed media tracks the true contribution shape almost perfectly — that",
           "is the lever the naive model gives up."),
    ),
    "04_fit.ipynb": nb(
        md("# 04 · Fitting — frequentist vs Bayesian",
           "",
           "The frequentist NLS is fast but **degenerate-yet-confident**. The Bayesian fit's priors",
           "kill the degeneracy and report honest uncertainty. We *load* the cached Bayesian",
           "InferenceData (produced on the VM) rather than re-sampling here."),
        code("from draftzone_mmm.fit_freq import fit",
             "from draftzone_mmm.model import load_national",
             "res = fit(load_national(), n_starts=8)",
             "print('frequentist R^2 =', round(res['r2'],3), '| intercept =', round(res['intercept']))",
             "for c,p in res['params'].items(): print(f\"  {c:10s} contrib={p['avg_contrib']:8.1f}\")"),
        code("import arviz as az",
             "from draftzone_mmm.model import load_idata",
             "idata = load_idata('artifacts/idata_anchored.nc')  # run scripts/run_fits.sh first",
             "az.summary(idata, var_names=[f'beta_{c}' for c in ['tv','search','social','affiliate','brand']])[['mean','sd','r_hat']]"),
    ),
    "05_evaluate.ipynb": nb(
        md("# 05 · Grading against the sealed truth",
           "",
           "`evaluate.py` is the **only** module allowed to open the answer key. Here we render its",
           "scorecard: recovered contribution & θ with 89% intervals, HIT/MISS, and the",
           "**overconfidence** check (interval coverage vs the nominal 89%)."),
        code("import json",
             "sc = json.load(open('docs/data/scorecard.json'))",
             "f = sc['fit']",
             "print(f\"R^2={f['r2']:.3f}  MAPE={f['mape']:.1f}%  89% PP coverage={f['pp_interval_coverage']}% (want 89%)\")",
             "print(f\"contribution CIs containing truth: {sc['summary']['contrib_ci_hits']}/{sc['summary']['n_channels']}\")"),
        code("for c in sc['channels']:",
             "    flag = 'HIT' if c['hit'] else 'MISS'",
             "    print(f\"{c['channel']:10s} true={c['true_contrib']:7.1f} est={c['est_contrib']:7.1f} \"",
             "          f\"[{c['ci'][0]:6.1f},{c['ci'][1]:6.1f}]  theta {c['true_theta']:.2f}->{c['est_theta']:.2f}  {flag}\")"),
        md("Strong aggregate fit can hide an overconfident decomposition — calibration, not R²,",
           "is the honest test."),
    ),
    "06_experiments.ipynb": nb(
        md("# 06 · The rotating geo-experiments (the repair)",
           "",
           "One randomized geo-experiment per channel. Difference-in-differences cancels the",
           "season confounder and recovers each channel's causal lift; fed back as an anchor it",
           "slides the estimate toward truth."),
        code("import pandas as pd",
             "from draftzone_mmm.experiment import did_analysis",
             "geo = pd.read_csv('data/geo_experiments.csv')",
             "for c in ['tv','search','social','affiliate','brand']:",
             "    r = did_analysis(geo[geo.channel==c], n_boot=500)",
             "    print(f\"{c:10s} DiD={r['did']:6.2f} 89% CI [{r['did_ci'][0]:.2f},{r['did_ci'][1]:.2f}]  pre-gap={r['pre_period_gap']:+.2f}\")"),
        code("import json",
             "rep = json.load(open('docs/data/repair.json'))",
             "sc = {c['channel']: c['true_contrib'] for c in json.load(open('docs/data/scorecard.json'))['channels']}",
             "for c in rep['channels']:",
             "    print(f\"{c['channel']:10s} before={c['before']['mean']:7.1f}  after={c['after']['mean']:7.1f}  truth={sc[c['channel']]:7.1f}\")"),
        md("The rotating calendar lets us anchor **every** channel — the payoff that turns 'test",
           "everything' into confident moves across the board."),
    ),
    "07_revenue.ipynb": nb(
        md("# 07 · Revenue & ROI — average is a trap",
           "",
           "Blended LTV with uncertainty. Average ROI and **marginal** ROI disagree; decisions ride",
           "on the next dollar."),
        code("import json",
             "roi = json.load(open('docs/data/roi.json'))",
             "print('blended LTV $%d  blended media ROI %.2f' % (roi['ltv']['mu'], roi['blended_roi']))",
             "for c in roi['channels']:",
             "    print(f\"{c['channel']:10s} avgROI={c['roi'][0]:.2f}  mROI={c['mroi'][0]:.2f} \"",
             "          f\"[{c['mroi'][1]:.2f},{c['mroi'][2]:.2f}]\")"),
        md("A channel can be great on average yet lose money on its next dollar (mROI < 1) once it",
           "saturates — that's the number that should move budget."),
    ),
    "08_optimize.ipynb": nb(
        md("# 08 · Optimization under uncertainty",
           "",
           "A point-estimate optimizer confidently reallocates the whole budget. Across the",
           "posterior, only **robust** moves survive; the rest are *test-first*."),
        code("import json, numpy as np",
             "opt = json.load(open('docs/data/optim_draws.json'))",
             "print('point-estimate optimum: +%.1f%% conversions at same budget' % opt['point_estimate']['lift_pct'])",
             "for c in opt['channels']:",
             "    print(f\"{c['channel']:10s} median {c['median_change']:+5.0f}%  \"",
             "          f\"89% CI [{c['ci'][0]:+.0f}%,{c['ci'][1]:+.0f}%]  {c['verdict']}\")"),
        md("The credible deliverable to a CMO: *cut the saturated channel now — we're confident;",
           "for everything else, run a geo-test before moving money, starting with the highest-upside,",
           "highest-uncertainty channel.*"),
    ),
}


def main():
    NB_DIR.mkdir(exist_ok=True)
    for name, content in NOTEBOOKS.items():
        with open(NB_DIR / name, "w") as f:
            json.dump(content, f, indent=1)
        print("wrote", NB_DIR / name)


if __name__ == "__main__":
    main()
