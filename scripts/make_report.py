"""scripts/make_report.py — publish a per-run report + rebuild the runs index for GitHub Pages.

After running the pipeline (datagen -> fit -> experiments -> anchored -> ...), call:

    python scripts/make_report.py --label seed77

It reads the current pipeline artifacts, grades everything against the sealed truth,
simulates the idealized long-run test-and-learn trajectory, renders a self-contained HTML
report into docs/runs/<id>/, and regenerates docs/runs/index.html (the run tracker).

This is a REPORTING/eval tool — like evaluate.py it legitimately reads data_sealed/ to grade.
It lives in scripts/ (outside src/draftzone_mmm/), so the no-truth-leak guard does not apply.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from draftzone_mmm import evaluate, optimize, revenue  # noqa: E402
from draftzone_mmm.fit_freq import fit as freq_fit  # noqa: E402
from draftzone_mmm.model import (  # noqa: E402
    ARTIFACTS,
    CHANNELS,
    COLORS,
    REPO,
    load_national,
)
from draftzone_mmm.transforms import geometric_adstock, hill_saturation  # noqa: E402

RUNS_DIR = REPO / "docs" / "runs"
SEALED = REPO / "data_sealed" / "ground_truth.json"


def _resolve(p):
    p = pathlib.Path(p)
    return p if p.exists() else p.with_suffix(".pkl")


# ---------------------------------------------------------------- truth-graded ROI + sim
def true_roi(df, gt):
    out = {}
    for c in CHANNELS:
        p = gt["channels"][c]
        imp = df[f"{c}_impressions"].to_numpy(float)
        spend = df[f"{c}_spend"].sum()
        contrib = p["beta"] * hill_saturation(geometric_adstock(imp, p["theta"]), p["half_sat"], p["slope"])
        contrib2 = p["beta"] * hill_saturation(geometric_adstock(imp * 1.01, p["theta"]), p["half_sat"], p["slope"])
        avg = contrib.sum() * revenue.LTV_MU / spend
        mar = (contrib2 - contrib).sum() * revenue.LTV_MU / (spend * 0.01)
        out[c] = (float(avg), float(mar))
    return out


def simulate_test_and_learn(df, gt, rounds=30, eta=0.5):
    imp = {c: df[f"{c}_impressions"].to_numpy(float) for c in CHANNELS}
    cur = {c: df[f"{c}_spend"].sum() for c in CHANNELS}
    budget = sum(cur.values())
    P = {c: gt["channels"][c] for c in CHANNELS}

    def out_c(c, spend):
        k = spend / cur[c]
        p = P[c]
        return float((p["beta"] * hill_saturation(geometric_adstock(imp[c] * k, p["theta"]),
                                                   p["half_sat"], p["slope"])).sum())

    def total(a):
        return sum(out_c(c, a[c]) for c in CHANNELS)

    def mroi(c, spend):
        d = spend * 0.01
        return (out_c(c, spend + d) - out_c(c, spend)) * revenue.LTV_MU / d

    from scipy.optimize import minimize
    lo, hi = 0.2, 5.0
    x0 = np.array([cur[c] for c in CHANNELS])
    cons = [{"type": "eq", "fun": lambda x: x.sum() - budget}]
    bnds = [(cur[c] * lo, cur[c] * hi) for c in CHANNELS]
    best = None
    for s in range(6):
        st = x0 if s == 0 else x0 * np.random.default_rng(s).uniform(0.5, 2, len(CHANNELS))
        st = st / st.sum() * budget
        r = minimize(lambda x: -total({c: max(x[i], 1) for i, c in enumerate(CHANNELS)}),
                     st, method="SLSQP", bounds=bnds, constraints=cons,
                     options=dict(maxiter=400, ftol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    opt = {c: float(best.x[i]) for i, c in enumerate(CHANNELS)}

    def run(sigma, seed=1):
        rng = np.random.default_rng(seed)
        a = dict(cur)
        outs, allocs, mrois = [total(a)], [dict(a)], [{c: mroi(c, a[c]) for c in CHANNELS}]
        for _ in range(rounds):
            meas = {c: mroi(c, a[c]) * (1 + rng.normal(0, sigma)) for c in CHANNELS}
            mbar = np.mean(list(meas.values()))
            raw = {c: a[c] * np.exp(eta * (meas[c] - mbar)) for c in CHANNELS}
            ssum = sum(raw.values())
            a = {c: float(np.clip(raw[c] / ssum * budget, cur[c] * lo, cur[c] * hi)) for c in CHANNELS}
            ssum = sum(a.values())
            a = {c: a[c] / ssum * budget for c in CHANNELS}
            outs.append(total(a))
            allocs.append(dict(a))
            mrois.append({c: mroi(c, a[c]) for c in CHANNELS})
        return outs, allocs, mrois

    return dict(cur=cur, opt=opt, budget=budget, cur_out=total(cur), opt_out=total(opt),
                opt_mroi={c: mroi(c, opt[c]) for c in CHANNELS},
                low=run(0.10), high=run(0.40))


def rounds_to(outs, cur_out, opt_out, thresh=0.90):
    """First round at which the trajectory reaches `thresh` of the attainable gain (or None)."""
    gap = opt_out - cur_out
    if gap <= 1e-9:
        return 0
    for i, o in enumerate(outs):
        if (o - cur_out) / gap >= thresh:
            return i
    return None


def convergence_estimate(sim, weeks_per_cycle=13, thresh=0.90):
    """Translate rounds-to-convergence into a calendar estimate. One round = one experiment
    cycle (measure every channel's marginal, reallocate, re-measure) ~= a quarter if tests run
    in parallel across channels."""
    r_low = rounds_to(sim["low"][0], sim["cur_out"], sim["opt_out"], thresh)
    r_high = rounds_to(sim["high"][0], sim["cur_out"], sim["opt_out"], thresh)
    def wks(r):
        return None if r is None else int(round(r * weeks_per_cycle))
    return dict(thresh=thresh, weeks_per_cycle=weeks_per_cycle,
                rounds_low=r_low, rounds_high=r_high,
                weeks_low=wks(r_low), weeks_high=wks(r_high))


# ---------------------------------------------------------------- figures
def fig_ladder(path, naive, freq, before, after, truth):
    tiers = ["naive\n(OLS)", "frequentist\n(NLS)", "Bayesian\n(observational)", "anchored\n(+experiments)"]
    maes = []
    for src in (naive, freq, before, after):
        maes.append(np.mean([abs(src[c] - truth[c]) for c in CHANNELS]))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(tiers, maes, color=["#888", "#b5651d", "#1f77b4", "#2ca02c"])
    for b, m in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("mean abs error per channel\n(conversions/wk vs truth)")
    ax.set_title("Attribution error by modeling tier (lower = better)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_repair(path, before, after, truth):
    x = np.arange(len(CHANNELS))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, c in enumerate(CHANNELS):
        bm, bl, bh = before[c]
        am, al, ah = after[c]
        ax.plot([i - 0.16, i - 0.16], [bl, bh], color="#888", lw=4, solid_capstyle="round")
        ax.plot(i - 0.16, bm, "o", color="#888", ms=5)
        ax.plot([i + 0.16, i + 0.16], [al, ah], color=COLORS[c], lw=4, solid_capstyle="round")
        ax.plot(i + 0.16, am, "o", color=COLORS[c], ms=5)
        ax.plot([i - 0.34, i + 0.34], [truth[c], truth[c]], color="#d9a200", lw=2, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS)
    ax.set_ylabel("avg contribution (conv/wk)")
    ax.set_title("Experiment repair: before (grey) → after (color) · gold = truth")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_roi(path, roi, troi):
    estd = {ch["channel"]: ch for ch in roi["channels"]}
    x = np.arange(len(CHANNELS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.8))
    av = [estd[c]["roi"][0] for c in CHANNELS]
    mr = [estd[c]["mroi"][0] for c in CHANNELS]
    ax.bar(x - w / 2, av, w, color=[COLORS[c] for c in CHANNELS], alpha=0.45, label="avg ROI (est)")
    ax.bar(x + w / 2, mr, w, color=[COLORS[c] for c in CHANNELS], label="marginal ROI (est)")
    ax.plot(x + w / 2, [troi[c][1] for c in CHANNELS], "k_", ms=14, mew=2, label="marginal ROI (TRUE)")
    ax.axhline(1.0, color="#d62728", ls="--", lw=1, label="break-even")
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS)
    ax.set_ylabel("ROI")
    ax.set_title("Average vs marginal ROI (faint=avg, solid=marginal)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_tnl(path, sim, conv=None):
    cur_out, opt_out = sim["cur_out"], sim["opt_out"]

    def gain(o):
        return 100 * (np.array(o) - cur_out) / (opt_out - cur_out + 1e-9)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(gain(sim["low"][0]), color="#2ca02c", lw=2, label="high volume (low noise)")
    ax[0].plot(gain(sim["high"][0]), color="#d62728", lw=1.6, alpha=0.8, label="low volume (noisy)")
    ax[0].axhline(100, ls="--", color="#888")
    if conv and conv.get("rounds_low") is not None:
        thr = 100 * conv["thresh"]
        r = conv["rounds_low"]
        ax[0].axhline(thr, ls=":", color="#bbb")
        ax[0].axvline(r, ls=":", color="#2ca02c")
        ax[0].annotate(f"≈{r} rounds → ~{conv['weeks_low']} wks", (r, thr),
                       textcoords="offset points", xytext=(6, -14), fontsize=8, color="#2ca02c")
    ax[0].set_title("Convergence to true optimum")
    ax[0].set_xlabel("round  (≈ one experiment cycle ≈ a quarter)")
    ax[0].set_ylabel("% of attainable gain")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    for c in CHANNELS:
        ax[1].plot([100 * a[c] / sim["budget"] for a in sim["low"][1]], color=COLORS[c], lw=1.8, label=c)
        ax[1].axhline(100 * sim["opt"][c] / sim["budget"], color=COLORS[c], ls=":", lw=1.1, alpha=0.7)
    ax[1].set_title("Allocation converging (dotted = optimum)")
    ax[1].set_xlabel("round")
    ax[1].set_ylabel("% of budget")
    ax[1].legend(fontsize=7, ncol=2)
    ax[1].grid(alpha=0.3)
    for c in CHANNELS:
        ax[2].plot([h[c] for h in sim["low"][2]], color=COLORS[c], lw=1.8, label=c)
    ax[2].axhline(1.0, ls="--", color="#d62728")
    ax[2].set_title("Marginal ROIs equalizing")
    ax[2].set_xlabel("round")
    ax[2].set_ylabel("true marginal ROI")
    ax[2].legend(fontsize=7, ncol=2)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- html
CSS = """
:root{--bg:#0e1116;--panel:#161b22;--ink:#e6edf3;--muted:#9aa7b4;--line:#2a313c;--accent:#4da3ff;--good:#2ca02c;--bad:#d62728;--warn:#d9a200}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:0 20px}a{color:var(--accent)}
header.hero{padding:54px 0 30px;border-bottom:1px solid var(--line);background:radial-gradient(1100px 360px at 50% -90px,#1b2b40 0,transparent 70%)}
h1{font-size:2rem;margin:0 0 6px}.sub{color:var(--muted);margin:0 0 14px}
.kicker{text-transform:uppercase;letter-spacing:2px;font-size:.72rem;color:var(--accent);font-weight:700;margin-bottom:8px}
section{padding:34px 0;border-bottom:1px solid var(--line)}h2{font-size:1.4rem;margin:0 0 4px}.lead{color:var(--muted);margin:0 0 16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0}
table{width:100%;border-collapse:collapse;font-size:.9rem}th,td{text-align:right;padding:7px 9px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
.hit{color:var(--good);font-weight:700}.miss{color:var(--bad);font-weight:700}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:.74rem;font-weight:700}
.pill.inc{background:rgba(44,160,44,.18);color:#69d36a}.pill.dec{background:rgba(214,39,40,.18);color:#ff6b6b}.pill.test{background:rgba(217,162,0,.18);color:#f0c64b}
.callout{border-left:3px solid var(--accent);background:#1d2530;padding:11px 15px;border-radius:0 8px 8px 0;margin:12px 0}
.callout.warn{border-left-color:var(--warn)}.callout.good{border-left-color:var(--good)}
img{max-width:100%;border-radius:8px;border:1px solid var(--line);margin:8px 0}
.metric{display:flex;gap:24px;flex-wrap:wrap;margin:8px 0}.metric .m b{font-size:1.4rem;display:block}.metric .m span{color:var(--muted);font-size:.78rem}
.small{font-size:.82rem;color:var(--muted)}footer{padding:36px 0 70px;color:var(--muted);font-size:.85rem}
.tag{display:inline-block;background:#1d2530;border:1px solid var(--line);border-radius:6px;padding:2px 8px;margin:2px 4px 2px 0;font-size:.8rem}
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:4px 0}table{min-width:520px}
@media(max-width:640px){
 .wrap{padding:0 14px}header.hero{padding:34px 0 22px}h1{font-size:1.5rem}h2{font-size:1.2rem}
 .sub{font-size:.95rem}section{padding:24px 0}.card{padding:13px 14px}
 th,td{padding:6px 7px;font-size:.84rem}.metric{gap:16px}.metric .m b{font-size:1.2rem}
 body{font-size:15px}.kicker{font-size:.66rem}
}
"""


def _tbl(headers, rows):
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def _verdict_pill(v):
    cls = {"INCREASE": "inc", "DECREASE": "dec", "TEST FIRST": "test"}[v]
    return f'<span class="pill {cls}">{v}</span>'


def build_report_html(meta, before, after, truth, naive, freq, roi, troi, optim, sim, conv):
    gtd = truth
    chans = CHANNELS
    mae_b = np.mean([abs(before[c][0] - gtd[c]) for c in chans])
    mae_a = np.mean([abs(after[c][0] - gtd[c]) for c in chans])
    media_true = sum(gtd[c] for c in chans)
    uc_b = 100 * (1 - sum(before[c][0] for c in chans) / media_true)
    uc_a = 100 * (1 - sum(after[c][0] for c in chans) / media_true)
    attain = 100 * (sim["opt_out"] / sim["cur_out"] - 1)
    robust = [f"{c} {optim_ch['verdict'].lower()}" for c in chans
              for optim_ch in [next(o for o in optim["channels"] if o["channel"] == c)]
              if optim_ch["verdict"] != "TEST FIRST"]
    all_mroi_below1 = all(sim["opt_mroi"][c] < 1 for c in chans)
    mroi_note = ("— and every channel's marginal ROI at the optimum was below break-even, "
                 "implying the budget itself is too large (or a deliberate growth bet)."
                 if all_mroi_below1 else ".")
    cw, ch_ = conv.get("weeks_low"), conv.get("weeks_high")
    conv_metric = f"~{cw} wks" if cw is not None else "—"
    conv_years = f"≈{cw / 52:.1f} yrs" if cw is not None else ""
    pct_thr = int(conv["thresh"] * 100)
    r_low, r_high, wpc = conv["rounds_low"], conv["rounds_high"], conv["weeks_per_cycle"]
    if cw is not None:
        noisy_rounds = "" if ch_ is None else f" / ≈{r_high} (noisy)"
        noisy_weeks = "" if ch_ is None else f" — or ~{ch_} wks if tests are noisy"
        conv_sentence = (
            f"Reaching <b>{pct_thr}%</b> of the attainable gain took ≈<b>{r_low}</b> rounds "
            f"(high-volume testing){noisy_rounds}. At one experiment cycle per quarter "
            f"(~{wpc} wks, run in parallel across channels) that's ≈<b>{cw} weeks ({conv_years})</b>"
            f"{noisy_weeks}; testing one channel at a time instead of in parallel would be roughly "
            "5× longer. This is the <i>no-drift</i> ideal — with drift the optimum keeps moving, "
            "so you never fully arrive, you perpetually pursue."
        )
    else:
        conv_sentence = ("The high-volume trajectory did not reach the threshold within the "
                         "simulated horizon.")

    # ladder table rows
    ladder_rows = []
    for c in chans:
        ladder_rows.append([c, f"{gtd[c]:.0f}", f"{naive[c]:.0f}", f"{freq[c]:.0f}",
                            f"{before[c][0]:.0f}", f"{after[c][0]:.0f}"])

    sc_rows = []
    for c in chans:
        am, al, ah = after[c]
        hit = al <= gtd[c] <= ah
        sc_rows.append([c, f"{gtd[c]:.0f}", f"{am:.0f}", f"[{al:.0f}, {ah:.0f}]",
                        f'<span class="{"hit" if hit else "miss"}">{"HIT" if hit else "MISS"}</span>'])

    roi_rows = []
    estd = {ch["channel"]: ch for ch in roi["channels"]}
    for c in chans:
        e = estd[c]
        roi_rows.append([c, f"{e['roi'][0]:.2f} <span class='small'>[{e['roi'][1]:.1f},{e['roi'][2]:.1f}]</span>",
                        f"{e['mroi'][0]:.2f} <span class='small'>[{e['mroi'][1]:.1f},{e['mroi'][2]:.1f}]</span>",
                        f"{troi[c][1]:.2f}"])

    opt_rows = []
    for o in optim["channels"]:
        opt_rows.append([o["channel"], f"{o['median_change']:+.0f}%",
                        f"[{o['ci'][0]:+.0f}%, {o['ci'][1]:+.0f}%]", _verdict_pill(o["verdict"])])

    asm = meta["assumptions"]
    tags = "".join(f'<span class="tag">{k}: {v}</span>' for k, v in asm.items())

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MMM run — {meta['id']}</title><style>{CSS}</style></head><body>
<header class="hero"><div class="wrap">
<div class="kicker">DraftZone MMM · run report</div>
<h1>{meta['label']}</h1>
<p class="sub">seed {meta['seed']} · generated {meta['timestamp']} · <a href="../index.html">← all runs</a></p>
<div>{tags}</div>
</div></header>
<main class="wrap">

<section><h2>TL;DR</h2>
<div class="card"><div class="metric">
<div class="m"><b>{uc_b:.0f}% → {uc_a:.0f}%</b><span>media under-credit (obs → anchored)</span></div>
<div class="m"><b>{mae_b:.0f} → {mae_a:.0f}</b><span>mean abs error / channel</span></div>
<div class="m"><b>{meta['pp_coverage']:.0f}%</b><span>89% interval coverage (overconfidence)</span></div>
<div class="m"><b>+{attain:.1f}%</b><span>max attainable fixed-budget lift</span></div>
<div class="m"><b>{conv_metric}</b><span>est. time to {pct_thr}% of optimum ({conv_years}, no drift)</span></div>
</div>
<p>On this dataset (confound ρ≈{asm['confound']}, baseline {asm['baseline_share']}), the observational
Bayesian model under-credited media and was overconfident. One randomized geo-experiment per channel
cut mean per-channel attribution error from <b>{mae_b:.0f}</b> to <b>{mae_a:.0f}</b> conv/wk. The robust
optimizer's only confident move(s): <b>{', '.join(robust) if robust else 'none — everything test-first'}</b>.
Under idealized always-on testing the maximum attainable fixed-budget lift was <b>+{attain:.1f}%</b> {mroi_note}</p>
</div></section>

<section><h2>1 · The modeling ladder</h2>
<p class="lead">Same data, four tiers. Fit improves; attribution does not — until experiments enter.</p>
<div class="card"><img src="ladder.png" alt="error by tier">
{_tbl(["channel", "true", "naive", "freq", "Bayesian", "anchored"], ladder_rows)}
<p class="small">Per-channel avg contribution (conv/wk) vs the sealed truth, across tiers.</p></div></section>

<section><h2>2 · Experiment repair</h2>
<p class="lead">Confound-immune geo anchors slide the estimate toward truth and tighten the intervals.</p>
<div class="card"><img src="repair.png" alt="repair before/after">
{_tbl(["channel", "true", "anchored est", "89% CI", "covers truth?"], sc_rows)}
</div></section>

<section><h2>3 · ROI — average vs marginal</h2>
<p class="lead">Average flatters; the next dollar (marginal) decides. Graded against true marginal ROI.</p>
<div class="card"><img src="roi.png" alt="roi">
{_tbl(["channel", "avg ROI (est)", "marginal ROI (est)", "marginal TRUE"], roi_rows)}
<p class="small">Blended LTV ${revenue.LTV_MU:.0f}. Estimated marginal ROI tends to run optimistic vs truth.</p></div></section>

<section><h2>4 · Robust budget verdicts</h2>
<p class="lead">Point estimate promised +{optim['point_estimate']['lift_pct']:.1f}%; across uncertainty only moves whose interval clears the dead-band are confident.</p>
<div class="card">{_tbl(["channel", "median Δ", "89% CI", "verdict"], opt_rows)}</div></section>

<section><h2>5 · Idealized long-run test-and-learn</h2>
<p class="lead">Unbiased experiments, no drift: iterate small steps, re-measure, converge.</p>
<div class="card"><img src="tnl.png" alt="test and learn">
<p>{conv_sentence}</p>
<p class="small">Max attainable fixed-budget lift <b>+{attain:.1f}%</b>; high-volume testing captures essentially
all of it, noisy testing leaves some on the table. Marginal ROIs compress toward equalization — the optimality condition.</p></div></section>

<section><h2>What we learned</h2><div class="card">
<ol>
<li>Good fit ≠ good attribution. R² rose across tiers while the decomposition stayed wrong until experiments entered.</li>
<li>The confound biases observational MMM; priors bound the damage but don't cure it (see the channel the model still misses).</li>
<li>Experiments repair the worst of it — but a <b>biased anchor injects its own error</b>, and robustness-across-uncertainty does not catch anchor bias.</li>
<li>Average ROI is a trap; marginal ROI drives decisions — and estimated marginal ROI runs optimistic.</li>
<li>The attainable fixed-budget lift is small ({attain:+.1f}%); the real program is steady small moves + continuous re-testing, not a one-time jump.</li>
</ol></div></section>

</main>
<footer class="wrap">Synthetic data, graded against a sealed answer key. Reports are precomputed by
<code>scripts/make_report.py</code>. <a href="../index.html">All runs →</a></footer>
</body></html>"""


def build_index_html(manifest):
    rows = []
    for r in manifest:
        a = r["assumptions"]
        rows.append([
            f'<a href="{r["id"]}/report.html">{r["label"]}</a>',
            r["seed"], a.get("confound", "–"), a.get("baseline_share", "–"),
            f'{r["mae_before"]:.0f} → {r["mae_after"]:.0f}',
            f'{r["uc_before"]:.0f}% → {r["uc_after"]:.0f}%',
            f'{r["pp_coverage"]:.0f}%',
            f'+{r["attainable_lift"]:.1f}%',
            (f'~{r["weeks_to_converge"]} wk' if r.get("weeks_to_converge") is not None else "—"),
            r.get("robust_moves") or "—",
            r["timestamp"][:16].replace("T", " "),
        ])
    table = _tbl(["run", "seed", "confound", "baseline", "MAE obs→anch",
                  "media under-credit", "interval cov.", "attainable lift",
                  "≈time to 90%", "robust move(s)", "generated"], rows)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DraftZone MMM — run tracker</title><style>{CSS}</style></head><body>
<header class="hero"><div class="wrap">
<div class="kicker">A Skeptic's Guide to Marketing Mix Modeling</div>
<h1>DraftZone MMM — run tracker</h1>
<p class="sub">Each row is one synthetic dataset, modeled end-to-end and graded against a sealed answer key.</p>
<p>We regenerate the data under different seeds and assumptions to see what survives. Every run measures
the same things: how badly observational MMM is fooled by the spend↔season confound, how much a per-channel
geo-experiment repairs it, whether marginal ROI is trustworthy, and what an idealized test-and-learn program
can actually attain. Open a run for the full report. <a href="../index.html">Interactive dashboard →</a></p>
</div></header>
<main class="wrap"><section>
<div class="card">{table}</div>
<p class="small">"MAE obs→anch" = mean absolute per-channel attribution error, observational vs experiment-anchored
(lower is better). "interval cov." is what % of weeks the nominal 89% predictive band actually covers
(≪89% = overconfident). "attainable lift" is the max fixed-budget conversion gain under perfect test-and-learn.
"≈time to 90%" estimates the calendar weeks to capture 90% of that gain (one experiment cycle ≈ a quarter, run
in parallel; no-drift idealization — with drift you never fully arrive).</p>
</section></main>
<footer class="wrap">Generated by <code>scripts/make_report.py</code>.</footer>
</body></html>"""


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None, help="run label (default seed{N}); same label overwrites")
    ap.add_argument("--idata", default=str(ARTIFACTS / "idata.nc"))
    ap.add_argument("--idata-anchored", default=str(ARTIFACTS / "idata_anchored.nc"))
    ap.add_argument("--weeks-per-cycle", type=int, default=13,
                    help="calendar weeks per test-and-learn round (one experiment cycle ~ a quarter)")
    args = ap.parse_args()

    df = load_national()
    cfg = json.load(open(REPO / "data" / "config.json"))
    gt = json.load(open(SEALED))
    gtd = {c: gt["avg_contribution_decomposition"][f"media_{c}"] for c in CHANNELS}
    seed = cfg["seed"]
    label = args.label or f"seed{seed}"
    run_id = re.sub(r"[^a-zA-Z0-9_-]", "-", label)

    base_path, anch_path = _resolve(args.idata), _resolve(args.idata_anchored)
    print("Grading baseline + anchored fits...")
    sc_b = evaluate.grade(base_path)
    sc_a = evaluate.grade(anch_path)
    before = {ch["channel"]: (ch["est_contrib"], ch["ci"][0], ch["ci"][1]) for ch in sc_b["channels"]}
    after = {ch["channel"]: (ch["est_contrib"], ch["ci"][0], ch["ci"][1]) for ch in sc_a["channels"]}
    naive = {r["channel"]: r["naive_contrib"] for r in sc_b["naive"]["channels"]}

    print("Frequentist fit...")
    fr = freq_fit(df)
    freq = {c: fr["params"][c]["avg_contrib"] for c in CHANNELS}

    print("ROI + optimizer...")
    roi = revenue.compute_roi(anch_path)
    troi = true_roi(df, gt)
    optim = optimize.optimize_budget(anch_path, n_draws=120)

    print("Test-and-learn simulation...")
    sim = simulate_test_and_learn(df, gt)
    conv = convergence_estimate(sim, weeks_per_cycle=args.weeks_per_cycle)

    rundir = RUNS_DIR / run_id
    rundir.mkdir(parents=True, exist_ok=True)
    fig_ladder(rundir / "ladder.png", naive, freq, {c: before[c][0] for c in CHANNELS},
               {c: after[c][0] for c in CHANNELS}, gtd)
    fig_repair(rundir / "repair.png", before, after, gtd)
    fig_roi(rundir / "roi.png", roi, troi)
    fig_tnl(rundir / "tnl.png", sim, conv)

    meta = dict(
        id=run_id, label=label, seed=seed,
        timestamp=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        pp_coverage=sc_a["fit"]["pp_interval_coverage"],
        assumptions=dict(confound=round(cfg["realized_confound"], 2),
                         baseline_share=f"{gt['avg_contribution_decomposition']['baseline']/df['conversions'].mean():.0%}",
                         weeks=cfg["n_weeks"], markets=cfg["n_markets"]),
    )
    html = build_report_html(meta, before, after, gtd, naive, freq, roi, troi, optim, sim, conv)
    (rundir / "report.html").write_text(html, encoding="utf-8")

    # manifest upsert
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    mpath = RUNS_DIR / "manifest.json"
    manifest = json.load(open(mpath)) if mpath.exists() else []
    manifest = [r for r in manifest if r["id"] != run_id]
    mae_b = float(np.mean([abs(before[c][0] - gtd[c]) for c in CHANNELS]))
    mae_a = float(np.mean([abs(after[c][0] - gtd[c]) for c in CHANNELS]))
    mt = sum(gtd[c] for c in CHANNELS)
    robust = ", ".join(f"{o['channel']} {o['verdict'].lower()}" for o in optim["channels"]
                       if o["verdict"] != "TEST FIRST")
    manifest.append(dict(
        id=run_id, label=label, seed=seed, timestamp=meta["timestamp"],
        assumptions=meta["assumptions"],
        mae_before=mae_b, mae_after=mae_a,
        uc_before=100 * (1 - sum(before[c][0] for c in CHANNELS) / mt),
        uc_after=100 * (1 - sum(after[c][0] for c in CHANNELS) / mt),
        pp_coverage=sc_a["fit"]["pp_interval_coverage"],
        attainable_lift=100 * (sim["opt_out"] / sim["cur_out"] - 1),
        weeks_to_converge=conv["weeks_low"],
        rounds_to_converge=conv["rounds_low"],
        weeks_per_cycle=conv["weeks_per_cycle"],
        robust_moves=robust,
    ))
    manifest.sort(key=lambda r: r["timestamp"], reverse=True)
    json.dump(manifest, open(mpath, "w"), indent=2)
    (RUNS_DIR / "index.html").write_text(build_index_html(manifest), encoding="utf-8")

    print(f"\nWrote {rundir/'report.html'}")
    print(f"Updated {RUNS_DIR/'index.html'} ({len(manifest)} run(s))")


if __name__ == "__main__":
    main()
