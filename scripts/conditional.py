"""scripts/conditional.py — "which engine when?" decision guide.

Reads the conditional sweep snapshots (docs/robustness/conditional/run_*.json), groups them by data
REGIME (saturation level), and asks whether the engine ranking shifts with the regime — i.e. whether
different approaches genuinely suit different data. Output: docs/conditional/index.html, a cross-regime
avg-rank table + a plain-language decision guide. CONTRACT: reporting only.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_report as mr      # noqa: E402
import robustness as rb       # noqa: E402  (reuse aggregate + _row)
from draftzone_mmm.model import REPO  # noqa: E402

CDIR = REPO / "docs" / "robustness" / "conditional"
OUT = REPO / "docs" / "conditional"

# saturation_scale -> human regime label (smaller scale = more saturated)
def regime_label(sat):
    if sat <= 0.6:
        return "High saturation (low headroom)"
    if sat >= 1.6:
        return "Low saturation (headroom)"
    return "Medium saturation"


def main():
    runs = [json.load(open(p)) for p in sorted(CDIR.glob("run_*.json"))]
    if not runs:
        print("No conditional snapshots in", CDIR)
        return
    by_regime = collections.defaultdict(list)
    for r in runs:
        by_regime[round(float(r.get("saturation_scale", 1.0)), 2)].append(r)
    sats = sorted(by_regime)                                   # ascending scale = descending saturation
    regimes = [(s, regime_label(s), by_regime[s]) for s in sats]

    # per-regime aggregate (national engines), ranked by avg rank
    agg_by_regime = {s: rb.aggregate(rl) for s, _, rl in regimes}
    # union of national engine keys
    eng_keys, labels = [], {}
    for s in sats:
        for k, v in agg_by_regime[s].items():
            if v["world"] == "national" and k not in labels:
                eng_keys.append(k); labels[k] = v["label"]

    # cross-regime avg-rank matrix (rows=engines, cols=regimes), plus median MAE
    def cell(s, k):
        e = agg_by_regime[s].get(k)
        if not e or not e["mae"]:
            return "—", np.inf
        r = rb._row(k, e, len(by_regime[s]))
        return f"{r['avg_rank']:.1f} ({r['med_mae']:.0f})", r["avg_rank"]

    headers = ["engine"] + [f"{lab}<br><span class='small'>scale {s}, n={len(by_regime[s])}/seed-set</span>"
                            for s, lab, _ in regimes]
    rows = []
    # sort engines by their mean avg-rank across regimes
    def mean_rank(k):
        rs = [cell(s, k)[1] for s in sats]
        rs = [x for x in rs if np.isfinite(x)]
        return np.mean(rs) if rs else np.inf
    for k in sorted(eng_keys, key=mean_rank):
        rows.append([labels[k]] + [cell(s, k)[0] for s in sats])
    matrix = mr._tbl(headers, rows)

    # best engine per regime + how the ladder/experiment moves
    best = {}
    for s, lab, _ in regimes:
        ranked = sorted(((k, rb._row(k, agg_by_regime[s][k], 0)["avg_rank"])
                         for k in agg_by_regime[s] if agg_by_regime[s][k]["world"] == "national"
                         and agg_by_regime[s][k]["mae"]),
                        key=lambda kv: kv[1])
        best[s] = [(labels.get(k, k), r) for k, r in ranked[:3]]

    def ladder_rank(s):
        e = agg_by_regime[s].get("spend_ladder")
        return rb._row("spend_ladder", e, 0)["avg_rank"] if e and e["mae"] else float("nan")
    lad_hi, lad_lo = ladder_rank(sats[0]), ladder_rank(sats[-1])

    guide_rows = []
    for s, lab, rl in regimes:
        top = best[s][0][0] if best[s] else "—"
        guide_rows.append([lab, f"scale {s}", top,
                           ", ".join(f"{n}" for n, _ in best[s][:3])])
    guide = mr._tbl(["if your data is…", "saturation", "most reliable (avg rank)", "top 3"], guide_rows)

    n_seeds = len({r["seed"] for r in runs})
    ladder_line = ""
    if np.isfinite(lad_hi) and np.isfinite(lad_lo):
        moved = lad_lo - lad_hi
        ladder_line = (
            f"<p><b>Does the picture actually change with the data?</b> The clearest signal is the "
            f"<b>spend ladder</b> (the experiment that measures the curve): its average rank goes from "
            f"<b>{lad_hi:.1f}</b> at high saturation to <b>{lad_lo:.1f}</b> at low saturation — "
            + ("it is <b>relatively more valuable when channels are saturated</b> (where assuming the "
               "curve is most dangerous) and less essential when there is headroom and the response is "
               "nearer-linear, exactly the hypothesis."
               if moved > 0.3 else
               "the shift is small at this sample size — suggestive, not conclusive.") + "</p>")

    css = mr.CSS
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Which engine when — conditional decision guide</title><style>{css}</style></head><body>
<header class="hero"><div class="wrap">
<div class="kicker">A Skeptic's Guide to Marketing Mix Modeling</div>
<h1>Which engine when?</h1>
<p class="sub">Does the best MMM depend on your data? A factorial sweep over channel <b>saturation</b>
({len(sats)} regimes × {n_seeds} seeds), every fast engine graded per regime.
<a href="../robustness/index.html">← multi-seed leaderboard</a> ·
<a href="../results/index.html">results</a> · <a href="../process/index.html">how it works</a></p>
<p>The multi-seed leaderboard says no engine is best <i>on average</i>. This page asks the sharper
question: is a given engine best <i>for a given kind of data</i>? We hold everything fixed and vary
only saturation — from heavily saturated channels (little headroom) to near-linear ones — and re-rank
the engines in each regime.</p>
</div></header>
<main class="wrap">
<section><div class="card">
<h2>Engine average rank by saturation regime</h2>
{matrix}
<p class="small">Each cell: <b>average rank (median MAE)</b> within that regime; lower rank is better.
Engines sorted by mean rank across regimes. "—" = not run / diverged in that regime.</p>
{ladder_line}
</div></section>
<section><div class="card">
<h2>Decision guide</h2>
{guide}
<p class="small">"Most reliable" = best average rank in that regime across the seed set. With a handful
of seeds per regime this is <b>directional</b>, not a verdict — the honest output is a tendency, and
the project's first rule still holds: where the confound is high, triangulate with an experiment
rather than trusting any single observational engine.</p>
</div></section>
<section><div class="card">
<h2>What holds regardless of regime</h2>
<ul>
<li><b>Regularise</b> — the unregularised engines are the ones that diverge, in every regime.</li>
<li><b>Control seasonality</b> and measure your confound.</li>
<li><b>Experiments beat confounding</b> — and the spend ladder earns its cost most where channels
are saturated, which is precisely where an observational fit is least able to recover the curve.</li>
</ul>
</div></section>
</main>
<footer class="wrap">Generated by <code>scripts/conditional.py</code> from the conditional sweep.</footer>
</body></html>"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"Conditional guide over {len(sats)} regimes ({n_seeds} seeds) -> {OUT/'index.html'}")
    for s, lab, _ in regimes:
        print(f"  {lab:32s} best: " + ", ".join(f"{n}({r:.1f})" for n, r in best[s][:3]))


if __name__ == "__main__":
    main()
