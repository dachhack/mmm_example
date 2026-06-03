"""scripts/engine_leaderboard.py — grade multiple MMM engines against the sealed truth.

Produces a head-to-head leaderboard (docs/engines/index.html) comparing every engine's
recovered per-channel contribution decomposition to the answer key, on the SAME dataset:
  naive OLS · frequentist NLS · our PyMC (observational + experiment-anchored) · Google Meridian.

This is a grading/reporting tool (like evaluate.py / make_report.py) — it legitimately reads
data_sealed/ to score. It lives in scripts/, outside the pipeline the no-truth-leak guard checks.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_report as mr  # noqa: E402  (reuse CSS + table helper; sibling in scripts/)
from draftzone_mmm.fit_freq import fit as freq_fit  # noqa: E402
from draftzone_mmm.model import (  # noqa: E402
    ARTIFACTS,
    CHANNELS,
    REPO,
    decompose,
    load_idata,
    load_national,
    stacked_draws,
)

SEALED = REPO / "data_sealed" / "ground_truth.json"
OUT = REPO / "docs" / "engines"
ENGINE_COLOR = {
    "naive_ols": "#888888",
    "frequentist_nls": "#b5651d",
    "pymc_obs": "#7fb3ff",
    "pymc_anchored": "#1f77b4",
    "google_meridian": "#2ca02c",
    "google_meridian_calibrated": "#17a589",
    "spend_ladder": "#d62728",
}


def naive_results(df):
    y = df["conversions"].to_numpy(float)
    X = np.column_stack([df[f"{c}_spend"] for c in CHANNELS] + [np.arange(len(df)), np.ones(len(df))])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    ch = {c: dict(est_contrib=float(coef[i] * df[f"{c}_spend"].mean()), ci=None)
          for i, c in enumerate(CHANNELS)}
    return dict(engine="naive_ols", label="Naive OLS", bayesian=False, fit=dict(r2=float(r2)), channels=ch)


def freq_results(df):
    r = freq_fit(df)
    ch = {c: dict(est_contrib=r["params"][c]["avg_contrib"], ci=None) for c in CHANNELS}
    return dict(engine="frequentist_nls", label="Frequentist NLS", bayesian=False,
                fit=dict(r2=r["r2"]), channels=ch)


def pymc_results(df, idata_path, label, engine):
    draws, idx = stacked_draws(load_idata(idata_path), 800)
    dec = decompose(df, draws, idx)
    y = df["conversions"].to_numpy(float)
    mm = dec["mu"].mean(0)
    r2 = 1 - ((y - mm) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    ch = {}
    for c in CHANNELS:
        arr = dec["channel"][c].mean(1)
        ch[c] = dict(est_contrib=float(arr.mean()),
                     ci=[float(np.percentile(arr, 5.5)), float(np.percentile(arr, 94.5))])
    return dict(engine=engine, label=label, bayesian=True, fit=dict(r2=float(r2)), channels=ch)


def grade(res, gtd):
    mae = me = mt = 0.0
    hits = n_ci = 0
    for c in CHANNELS:
        est = res["channels"][c]["est_contrib"]
        tru = gtd[f"media_{c}"]
        mae += abs(est - tru)
        me += est
        mt += tru
        ci = res["channels"][c]["ci"]
        if ci:
            n_ci += 1
            hits += int(ci[0] <= tru <= ci[1])
    return dict(mae=mae / len(CHANNELS), media_bias=100 * (me / mt - 1),
                hits=hits, n_ci=n_ci, r2=res["fit"].get("r2"))


def fig_leaderboard(path, engines, gtd):
    x = np.arange(len(CHANNELS))
    n = len(engines)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, e in enumerate(engines):
        vals = [e["channels"][c]["est_contrib"] for c in CHANNELS]
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=e["label"],
               color=ENGINE_COLOR.get(e["engine"], "#999"))
    ax.plot(x, [gtd[f"media_{c}"] for c in CHANNELS], "k_", ms=22, mew=3, label="TRUTH")
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS, rotation=20, ha="right")
    ax.set_ylabel("avg contribution (conv/wk)")
    ax.set_title("Recovered channel contribution vs sealed truth, by engine")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_national()
    gtd = json.load(open(SEALED))["avg_contribution_decomposition"]
    engines = [naive_results(df), freq_results(df),
               pymc_results(df, ARTIFACTS / "idata.nc", "DraftZone PyMC (obs)", "pymc_obs"),
               pymc_results(df, ARTIFACTS / "idata_anchored.nc", "DraftZone PyMC (anchored)", "pymc_anchored")]
    for path, label in [(ARTIFACTS / "meridian_results.json", "Google Meridian"),
                        (ARTIFACTS / "meridian_calibrated_results.json", "Meridian (naive lift→prior)"),
                        (ARTIFACTS / "ladder_results.json", "Spend ladder (curve fit)")]:
        if path.exists():
            m = json.load(open(path))
            m["label"] = label
            engines.append(m)

    OUT.mkdir(parents=True, exist_ok=True)
    fig_leaderboard(OUT / "leaderboard.png", engines, gtd)

    rows = []
    for e in sorted(engines, key=lambda e: grade(e, gtd)["mae"]):
        g = grade(e, gtd)
        cov = f"{g['hits']}/{g['n_ci']}" if g["n_ci"] else "—"
        rows.append([e["label"], "Bayesian" if e.get("bayesian") else "point",
                     f"{g['r2']:.3f}" if g["r2"] is not None else "—",
                     f"{g['mae']:.0f}", f"{g['media_bias']:+.0f}%", cov])
    table = mr._tbl(["engine", "type", "R²", "MAE/ch ↓", "media bias", "CIs hit"], rows)

    calib = next((e for e in engines if e["engine"] == "google_meridian_calibrated"), None)
    calib_note = ""
    if calib:
        g = grade(calib, gtd)
        calib_note = (
            '<div class="callout warn"><b>Naive lift-calibration mis-sets channels '
            f'(media bias {g["media_bias"]:+.0f}%, MAE {g["mae"]:.0f}/ch).</b> Feeding one geo-lift '
            "number into a model's <i>marginal</i> prior is fragile for two reasons. "
            "<b>(1) Operating point:</b> if the test markets aren't scale-consistent replicas of the "
            "national channel, the lift is measured at the wrong point and grossly over-credits "
            "(+113% on half-sat geos here; ~+20% once the geos are centered on national). "
            "<b>(2) Secant bias:</b> a detectable lift needs a big spend bump, so it measures the "
            "<i>average</i> response over that jump, not the <i>marginal</i> — understating saturated "
            "channels (flat region) and overstating headroom ones (convex region), shifting credit "
            "from saturated to unsaturated channels. The fix isn't a scalar prior: feed the lift at "
            "its measured <b>exposure levels</b> and let the model's curve translate it (our DiD-"
            "likelihood anchor; Meridian's <code>roi_calibration_period</code>), or run a "
            "<b>spend ladder</b> to trace the curve. You can't just hand a lift test to an MMM.</div>")

    ladder = next((e for e in engines if e["engine"] == "spend_ladder"), None)
    ladder_note = ""
    if ladder:
        g = grade(ladder, gtd)
        ladder_note = (
            '<div class="callout"><b>The spend ladder is the cure for the saturated channels '
            f'(MAE {g["mae"]:.0f}/ch, {g["hits"]}/{g["n_ci"]} CIs).</b> Instead of feeding ONE lift '
            "into a prior, it runs several cells per channel at different spend levels — including "
            "cells <i>below</i> current spend — and fits the response curve through them. The "
            "bracketing turns the read into an interpolation at the operating point, so it "
            "right-sizes the saturated channels a single secant mis-credits (paid_social, paid_search). "
            'It is also the most expensive option. <a href="../ladder/index.html">See the curves '
            "and the honest cost/time analysis →</a></div>")

    gt_total = sum(gtd[f"media_{c}"] for c in CHANNELS)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MMM engine leaderboard</title><style>{mr.CSS}</style></head><body>
<header class="hero"><div class="wrap">
<div class="kicker">A Skeptic's Guide to Marketing Mix Modeling</div>
<h1>Engine leaderboard</h1>
<p class="sub">Every engine we've tried, one sealed answer key — graded on the same dataset.
<a href="../runs/index.html">← run tracker</a></p>
<p>Every engine recovers a per-channel contribution decomposition; we score each against the
true values (media total {gt_total:.0f} conv/wk). Lower mean absolute error is better. The two
point-estimate engines (naive, frequentist) have no credible intervals; the Bayesian engines
(our PyMC, Meridian) report whether their 89% interval contains the truth.</p>
</div></header>
<main class="wrap"><section>
<div class="card"><img src="leaderboard.png" alt="engine comparison">
{table}
{ladder_note}
{calib_note}
<p class="small">All engines share the same Fourier-seasonality control set and public data — no
engine reads the answer key. "media bias" is total recovered media vs truth (+ = over-credit).
Configuration (prior scale, seasonality handling) dominates engine choice: see the run reports
for how the same gremlins — confound, prior scale, marginal-vs-average — drive every engine.</p>
</div></section></main>
<footer class="wrap">Generated by <code>scripts/engine_leaderboard.py</code>.</footer>
</body></html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("Leaderboard (best→worst by MAE/channel):")
    for r in rows:
        print(f"  {r[0]:30s} R²={r[2]} MAE={r[3]} bias={r[4]} CIs={r[5]}")
    print(f"Wrote {OUT/'index.html'}")


if __name__ == "__main__":
    main()
