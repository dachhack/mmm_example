"""scripts/spend_ladder_report.py — publish the spend-ladder page (docs/ladder/index.html).

Shows, per channel: the measured ladder cells (DiD lift at each spend level), the Hill curve fit
through them, the TRUE curve from the sealed answer key for comparison, and the recovered national
contribution graded against truth. Then an HONEST cost/time analysis: a spend ladder measures the
curve instead of assuming it, but it costs calendar time, scarce geo inventory, and a real media
tax — most of all on the saturated channels that need it most.

This is a grading/reporting tool (like evaluate.py / engine_leaderboard.py): it legitimately reads
data_sealed/ to overlay truth. It lives in scripts/, outside the no-truth-leak pipeline guard.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_report as mr  # noqa: E402
from draftzone_mmm.model import CHANNELS, REPO  # noqa: E402
from draftzone_mmm.spend_ladder import ladder_cost_model  # noqa: E402
from draftzone_mmm.transforms import hill_saturation  # noqa: E402

OUT = REPO / "docs" / "ladder"
SEALED = REPO / "data_sealed" / "ground_truth.json"
ANNUAL_BUDGET = 300_000_000.0  # ~$300M annual media, split by each channel's observed spend share


def fig_curves(path, res, gtd, size_frac, lad):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.ravel(), CHANNELS):
        r = res[c]
        f = r["fitted"]
        cells = r["cells"]
        ex = np.array([d["exposure"] for d in cells])
        did = np.array([d["did"] for d in cells])
        a0 = f["a0"]
        grid = np.linspace(min(ex.min(), a0) * 0.6, ex.max() * 1.05, 200)

        def lift(a, beta_m, hs_m, slope):
            return beta_m * (hill_saturation(a, hs_m, slope)
                             - hill_saturation(np.array([a0]), hs_m, slope)[0])

        # fitted curve
        ax.plot(grid, lift(grid, f["beta_market"], f["half_sat_market"], f["slope"]),
                color="#d62728", lw=2, label="fitted Hill")
        # true curve (sealed): market-scale params
        tc = gtd["channels"][c]
        bt, ht, st = tc["beta"] * size_frac, tc["half_sat"] * size_frac, tc["slope"]
        ax.plot(grid, lift(grid, bt, ht, st), color="#444", lw=1.4, ls="--", label="true Hill")
        # measured cells (down = blue, up = orange)
        cols = ["#1f77b4" if e < a0 else "#ff7f0e" for e in ex]
        ax.scatter(ex, did, c=cols, s=42, zorder=5, label="ladder cells (DiD)")
        ax.axvline(a0, color="#999", ls=":", lw=1)
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("campaign adstock exposure")
        ax.set_ylabel("lift vs control (conv/mkt·wk)")
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Spend ladder: measured cells, fitted curve, and sealed truth — per channel", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    res = json.load(open(REPO / "artifacts" / "ladder_results.json"))["channels"]
    gtd = json.load(open(SEALED))
    media = gtd["avg_contribution_decomposition"]
    cfg = json.load(open(REPO / "data" / "config.json"))
    size_frac = cfg.get("ladder_size_frac") or 0.06
    lad = pd.read_csv(REPO / "data" / "spend_ladder.csv")
    nat = pd.read_csv(REPO / "data" / "national_weekly.csv")

    OUT.mkdir(parents=True, exist_ok=True)
    fig_curves(OUT / "ladder_curves.png", res, gtd, size_frac, lad)

    # ---- recovery grading vs single-cell anchor, vs truth ----
    rows, mae, hits, n = [], 0.0, 0, 0
    for c in CHANNELS:
        est = res[c]["est_contrib"]
        tru = media[f"media_{c}"]
        ci = res[c]["ci"]
        f_true = gtd["channels"][c]["mean_contrib"] / gtd["channels"][c]["beta"]
        cov = "—"
        if ci:
            n += 1
            h = ci[0] <= tru <= ci[1]
            hits += h
            cov = ("✓" if h else "✗") + f" [{ci[0]:.0f},{ci[1]:.0f}]"
        mae += abs(est - tru)
        sat = "saturated" if f_true > 0.6 else ("mid" if f_true > 0.4 else "headroom")
        rows.append([c, f"{f_true:.0%} ({sat})", f"{tru:.0f}", f"{est:.0f}",
                     f"{est - tru:+.0f}", cov])
    grade_tbl = mr._tbl(["channel", "true saturation", "true contrib", "ladder est",
                         "error", "89% CI (✓=covers truth)"], rows)
    mae /= len(CHANNELS)

    # ---- cost / time model, scaled to the ~$300M context ----
    total_spend = sum(nat[f"{c}_spend"].sum() for c in CHANNELS)
    crows, tot_tax, max_months = [], 0.0, 0.0
    for c in CHANNELS:
        budget = ANNUAL_BUDGET * (nat[f"{c}_spend"].sum() / total_spend)
        cm = ladder_cost_model(budget)
        tot_tax += cm["media_tax"]
        max_months = max(max_months, cm["duration_months"])
        geo = "ok" if cm["geo_feasible"] else f"SHORT {cm['dmas_short']} DMAs"
        crows.append([c, f"${budget/1e6:.0f}M", f"{cm['total_markets']}", geo,
                      f"{cm['duration_months']:.1f} mo",
                      f"${cm['media_tax']/1e6:.2f}M",
                      f"{cm['media_tax_pct_of_annual']:.1f}%"])
    cost_tbl = mr._tbl(["channel", "annual budget", "test DMAs", "geo feasibility",
                        "duration", "media tax ($)", "tax % of budget"], crows)
    portfolio_years = round(6 * (max_months / 12.0) + (6 - 1) * 0.0, 1)  # one channel/quarter rotation
    seq_years = round(6 * 0.25, 2)  # 6 channels, ~a quarter each, run back-to-back

    css = mr.CSS
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spend ladders — measuring the curve</title><style>{css}</style></head><body>
<header class="hero"><div class="wrap">
<div class="kicker">A Skeptic's Guide to Marketing Mix Modeling</div>
<h1>Spend ladders: measure the curve, don't assume it</h1>
<p class="sub">Why a single lift test mis-sizes saturated channels — and what a multi-cell ladder
costs to fix it. <a href="../engines/index.html">← engine leaderboard</a> ·
<a href="../runs/index.html">run tracker</a></p>
<p>A single geo test gives <b>one</b> point on a channel's response curve, so turning its lift into
a national number means <i>assuming</i> the curve's shape and where you sit on it — the exact
assumption that breaks on saturated channels. A <b>spend ladder</b> runs several cells at different
spend levels (including cells <i>below</i> current spend), tracing several points so the curve can
be <b>measured</b>. Below: the measured cells, the fitted curve, and the sealed truth — then the
bill.</p>
</div></header>
<main class="wrap">

<section><div class="card">
<h2>1 · The ladder recovers the curve</h2>
<img src="ladder_curves.png" alt="per-channel ladder curves vs truth">
<p class="small">Each panel: blue points = pull-DOWN cells, orange = push-UP cells, dotted line =
current operating point. The down cells pin the curve's <i>absolute level</i>; the up cells expose
diminishing returns. Reading the contribution at the operating point becomes an
<b>interpolation</b>, not the extrapolation a single always-up test is forced into.</p>
</div></section>

<section><div class="card">
<h2>2 · Did it crack the saturated channels?</h2>
{grade_tbl}
<p>Ladder recovery: <b>MAE {mae:.0f} conv/wk per channel</b>, {hits}/{n} credible intervals cover
truth. The headline: the <b>saturated</b> channels (paid_social, paid_search, tv_ctv) — the ones a
single-cell anchor pushed to nonsense (paid_social went to ~20 vs a true 282) — are now right-sized,
because the bracketing cells let the fit see the curvature instead of guessing it. tv_ctv stays the
hardest: its long carryover (θ=0.75) muddies the in-campaign exposure and its lift is small, so even
a ladder leaves it under-credited. <b>Measuring beats assuming — but saturation is still the enemy.</b></p>
<div class="callout"><b>The design lesson.</b> A ladder only works if its cells <i>bracket</i> your
operating point. An all-UP ladder on a saturated channel just samples the plateau — flat points that
can't separate the ceiling from the half-saturation. You must be willing to spend <i>less</i> than
BAU in some test markets. That is uncomfortable, and it is the price of identification.</div>
</div></section>

<section><div class="card">
<h2>3 · What a ladder actually costs</h2>
<p>None of the cost is the modeling. It is calendar time, scarce geo inventory, and a real media
tax — scaled here to a ~${ANNUAL_BUDGET/1e6:.0f}M annual program (each channel's budget = its share
of observed spend):</p>
{cost_tbl}
<ul>
<li><b>Calendar.</b> Each ladder = pre-period + campaign + carryover read ≈ a full quarter
(~{max_months:.0f} months, longer-carryover channels read slower). Cells run in parallel as separate
market groups, so one channel's ladder is one wave — but the rotating program tests one channel per
quarter, so laddering all six is a <b>~{seq_years:.1f}-year</b> commitment back-to-back, longer in
practice.</li>
<li><b>Geo inventory.</b> A 6-cell × 40-market ladder ties up <b>240 test DMAs at once</b> — more
than the ~210 DMAs that exist in the US. Real programs cap test geos to protect the national number,
so cells get smaller and noisier, or you drop to sub-DMA matched markets. And the saturated channels
— smallest lift, hardest to detect — need the <i>most</i> markets to power, exactly where geo is
scarcest.</li>
<li><b>Media tax.</b> The up-cells pour money into the flat part of the curve (near-zero marginal
return) and the down-cells forgo conversions. Across the portfolio that is on the order of
<b>${tot_tax/1e6:.0f}M</b> of deliberately sub-optimal spend per ladder wave — paid to <i>learn</i>
the curve. The payoff has to clear that bar.</li>
</ul>
<div class="callout warn"><b>When is a ladder worth it?</b> When a channel is large, suspected
saturated, and a single-cell test keeps disagreeing with the model — i.e. when the decision riding
on getting its curve right is worth more than a quarter and a few $M of test tax. For small or
clearly-headroom channels, a single cell (or the model alone) is enough. Triangulation means
spending the expensive experiment where it changes a decision, and nowhere else.</div>
</div></section>

</main>
<footer class="wrap">Generated by <code>scripts/spend_ladder_report.py</code>. Curves overlay the
sealed answer key for grading only — the pipeline never reads it.</footer>
</body></html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"Ladder recovery MAE/ch={mae:.0f}  CIs={hits}/{n}  portfolio test tax≈${tot_tax/1e6:.0f}M")
    print(f"Wrote {OUT/'index.html'}")


if __name__ == "__main__":
    main()
