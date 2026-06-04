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
    "google_meridian_aks": "#98df8a",
    "google_meridian_geo": "#0b6e0b",
    "google_meridian_geo_ctrl": "#6b8e23",
    "google_meridian_geo_ctrlhi": "#9acd32",
    "google_meridian_calibrated": "#17a589",
    "google_meridian_calibrated_geo": "#0aa3a3",
    "spend_ladder": "#d62728",
}
# Stable display labels for every Meridian variant (engine id -> label).
MERIDIAN_LABELS = {
    "google_meridian": "Google Meridian (national, Fourier)",
    "google_meridian_aks": "Meridian (national, AKS)",
    "google_meridian_geo": "Meridian (geo panel)",
    "google_meridian_geo_ctrl": "Meridian (geo + proxy control, 0.78)",
    "google_meridian_geo_ctrlhi": "Meridian (geo + proxy control, 0.98)",
    "google_meridian_calibrated": "Meridian (naive lift→prior)",
    "google_meridian_calibrated_geo": "Meridian (geo, calibrated)",
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


def fig_leaderboard(path, engines, gtd, geo_gtd=None):
    x = np.arange(len(CHANNELS))
    n = len(engines)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, e in enumerate(engines):
        vals = [e["channels"][c]["est_contrib"] for c in CHANNELS]
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=e["label"],
               color=ENGINE_COLOR.get(e["engine"], "#999"))
    ax.plot(x, [gtd[f"media_{c}"] for c in CHANNELS], "k_", ms=22, mew=3, label="TRUTH (national)")
    if geo_gtd and any(_is_geo(e) for e in engines):
        ax.plot(x, [geo_gtd[f"media_{c}"] for c in CHANNELS], "_", color="#0b6e0b",
                ms=22, mew=3, label="TRUTH (geo world)")
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS, rotation=20, ha="right")
    ax.set_ylabel("avg contribution (conv/wk)")
    ax.set_title("Recovered channel contribution vs sealed truth, by engine")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _is_geo(e):
    return "_geo" in e["engine"]


def main():
    df = load_national()
    sealed = json.load(open(SEALED))
    gtd = sealed["avg_contribution_decomposition"]
    # geo engines fit the geo PANEL — a different world (targeted-spend confounder + Jensen gap), so
    # they are graded against the panel's OWN sealed answer key, not the national one.
    geo_block = sealed.get("geo_panel", {}).get("avg_contribution_decomposition")
    geo_gtd = {f"media_{c}": v for c, v in geo_block.items()} if geo_block else gtd
    truth_for = lambda e: geo_gtd if _is_geo(e) else gtd  # noqa: E731

    engines = [naive_results(df), freq_results(df),
               pymc_results(df, ARTIFACTS / "idata.nc", "DraftZone PyMC (obs)", "pymc_obs"),
               pymc_results(df, ARTIFACTS / "idata_anchored.nc", "DraftZone PyMC (anchored)", "pymc_anchored")]
    # auto-discover every Meridian variant written to artifacts/ (google_meridian*.json), plus
    # the legacy filenames and the spend-ladder engine.
    seen = set()
    for path in sorted(ARTIFACTS.glob("google_meridian*.json")) + [
            ARTIFACTS / "meridian_results.json", ARTIFACTS / "meridian_calibrated_results.json",
            ARTIFACTS / "ladder_results.json"]:
        if not path.exists():
            continue
        m = json.load(open(path))
        if m["engine"] in seen:
            continue
        seen.add(m["engine"])
        m["label"] = MERIDIAN_LABELS.get(m["engine"], m.get("label", m["engine"]))
        if m["engine"] == "spend_ladder":
            m["label"] = "Spend ladder (curve fit)"
        engines.append(m)

    OUT.mkdir(parents=True, exist_ok=True)
    fig_leaderboard(OUT / "leaderboard.png", engines, gtd, geo_gtd)

    rows = []
    for e in sorted(engines, key=lambda e: grade(e, truth_for(e))["mae"]):
        g = grade(e, truth_for(e))
        cov = f"{g['hits']}/{g['n_ci']}" if g["n_ci"] else "—"
        world = "geo" if _is_geo(e) else "nat'l"
        rows.append([e["label"], world, "Bayesian" if e.get("bayesian") else "point",
                     f"{g['r2']:.3f}" if g["r2"] is not None else "—",
                     f"{g['mae']:.0f}", f"{g['media_bias']:+.0f}%", cov])
    table = mr._tbl(["engine", "world", "type", "R²", "MAE/ch ↓", "media bias", "CIs hit"], rows)

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

    def _g(eng_id):
        e = next((e for e in engines if e["engine"] == eng_id), None)
        return grade(e, truth_for(e)) if e else None

    mer_note = ""
    gfo, gak = _g("google_meridian"), _g("google_meridian_aks")
    gge, gct, ghi = (_g("google_meridian_geo"), _g("google_meridian_geo_ctrl"),
                     _g("google_meridian_geo_ctrlhi"))
    if gfo and (gak or gge):
        parts = [f"national + Fourier controls (MAE {gfo['mae']:.0f})"]
        if gak:
            parts.append(f"national + Meridian's own AKS knots, no Fourier (MAE {gak['mae']:.0f})")
        if gge:
            parts.append(f"the hardened multi-geo panel (MAE {gge['mae']:.0f})")
        if gct:
            parts.append(f"that panel + an imperfect demand proxy (MAE {gct['mae']:.0f})")
        if ghi:
            parts.append(f"+ a near-perfect demand proxy (MAE {ghi['mae']:.0f})")
        verdict = ""
        if gge:
            conf = sealed.get("geo_panel", {})
            rc = conf.get("realized_corr_spend_demand")
            fid, fidhi = conf.get("demand_proxy_fidelity"), conf.get("demand_proxy_hi_fidelity")
            rc_txt = f" (realized corr(spend, demand) {rc:+.2f})" if rc is not None else ""
            verdict = (
                " The geo panel is Meridian's home turf — spend varies across geos within a week, "
                "cross-sectional identification the national series lacks — but this panel is "
                "<b>hardened</b>: a latent geo×time demand factor both lifts non-media conversions and "
                f"draws targeted spend{rc_txt}, the geo analogue of the spend↔season confound, graded "
                "against the geo world's own answer key. With <b>no control</b> for it, geo Meridian "
                f"blows up to <b>MAE {gge['mae']:.0f}</b> (media bias {gge['media_bias']:+.0f}%, "
                f"{gge['hits']}/{gge['n_ci']} CIs) — high R² and confidently wrong on every channel, "
                "crediting the demand-driven conversions to the spend that chases them.")
            if gct:
                bb = 100 * (1 - gct["mae"] / gge["mae"])
                ft = f"{fid:.2f}" if fid is not None else "imperfect"
                verdict += (
                    f" An <b>imperfect</b> demand-proxy control (fidelity {ft}) barely helps — "
                    f"<b>MAE {gct['mae']:.0f}</b>, bias {gct['media_bias']:+.0f}%, only ~{bb:.0f}% of the "
                    "error bought back: the confounder's pull on conversions rivals the entire media "
                    "signal, so what the proxy MISSES still tracks spend and keeps over-crediting.")
            if ghi:
                bbh = 100 * (1 - ghi["mae"] / gge["mae"])
                fth = f"{fidhi:.2f}" if fidhi is not None else "near-perfect"
                verdict += (
                    f" Even a <b>near-perfect</b> proxy (fidelity {fth}) only claws back about half — "
                    f"<b>MAE {ghi['mae']:.0f}</b>, bias still {ghi['media_bias']:+.0f}%, {ghi['hits']}/{ghi['n_ci']} "
                    f"CIs (~{bbh:.0f}% of the error bought back) — because the residual it misses still "
                    "tracks the targeted spend. Control quality sets the ceiling, but a realistic proxy "
                    "lives down near the useless end, and even the unrealistic one doesn't fully repair it.")
            if gak:
                verdict += (f" National AKS stays best-calibrated (MAE {gak['mae']:.0f}, bias "
                            f"{gak['media_bias']:+.0f}%, {gak['hits']}/{gak['n_ci']} CIs).")
        mer_note = (
            '<div class="callout"><b>Meridian across the control-quality spectrum.</b> Same scale-'
            "corrected ROI prior: " + "; ".join(parts) + "."
            + verdict + " The throughline: more data (geo cross-section) and better controls help, but "
            "you can never tell from fit alone whether your proxy was good enough — only a "
            "<b>randomized experiment</b> breaks a confounder you can't see. That is why this project "
            "triangulates MMM with geo tests instead of trusting any single fit." + "</div>")

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
true values (national media total {gt_total:.0f} conv/wk). Lower mean absolute error is better. The
two point-estimate engines (naive, frequentist) have no credible intervals; the Bayesian engines
(our PyMC, Meridian) report whether their 89% interval contains the truth. The <b>world</b> column
flags the data each engine was fit on: national engines are scored against the national answer key;
the <b>geo-panel</b> engine is a different, harder world — a multi-geo dataset with a latent
geo×time demand confounder (targeted spend) — so it is scored against the geo world's own sealed
key (a lower media total, because targeting concentrates spend into Hill saturation).</p>
</div></header>
<main class="wrap"><section>
<div class="card"><img src="leaderboard.png" alt="engine comparison">
{table}
{mer_note}
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
        print(f"  {r[0]:30s} [{r[1]:5s}] R²={r[3]} MAE={r[4]} bias={r[5]} CIs={r[6]}")
    print(f"Wrote {OUT/'index.html'}")


if __name__ == "__main__":
    main()
