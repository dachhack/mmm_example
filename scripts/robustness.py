"""scripts/robustness.py — aggregate per-seed snapshots into a ROBUSTNESS leaderboard.

Reads docs/robustness/run_*.json (one per seed) and answers the question a single leaderboard
cannot: which engines are reliably good across data realisations, not just lucky on one. For each
national engine it reports, across seeds: average rank, win-rate, MAE distribution (median + spread
= stability), mean bias, CI coverage, divergences, and Bayesian sampling quality (ESS / R-hat).
Geo engines (a different, harder world) are summarised separately — they are about the confound
finding, not the horse race.

Run after several `make`/orchestrator runs on different seeds. CONTRACT: reporting only.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_report as mr  # noqa: E402  (reuse CSS + table helper)
from draftzone_mmm.model import REPO  # noqa: E402

DIR = REPO / "docs" / "robustness"


def load_runs():
    runs = [json.load(open(p)) for p in sorted(DIR.glob("run_*.json"))]
    return runs


def aggregate(runs):
    """Per-engine stats across runs. Ranks are computed among NON-diverged national engines per run."""
    # collect per-engine series
    eng = {}
    for run in runs:
        nat = {k: v for k, v in run["engines"].items() if v["world"] == "national" and not v["diverged"]}
        order = sorted(nat, key=lambda k: nat[k]["mae"])
        rank = {k: i + 1 for i, k in enumerate(order)}
        for k, v in run["engines"].items():
            e = eng.setdefault(k, dict(label=v["label"], world=v["world"], mae=[], bias=[],
                                       cov=[], rank=[], wins=0, diverged=0, ess=[], rhat=[]))
            if v["diverged"]:
                e["diverged"] += 1
                continue
            e["mae"].append(v["mae"]); e["bias"].append(v["bias"])
            if v["n_ci"]:
                e["cov"].append(v["hits"] / v["n_ci"])
            if k in rank:
                e["rank"].append(rank[k]); e["wins"] += int(rank[k] == 1)
            s = v.get("sampling")
            if s and "ess_bulk_min" in s:
                e["ess"].append(s["ess_bulk_min"]); e["rhat"].append(s["rhat_max"])
    return eng


def _row(k, e, n_runs):
    med = np.median(e["mae"]) if e["mae"] else float("nan")
    return dict(
        key=k, label=e["label"], n=len(e["mae"]),
        avg_rank=np.mean(e["rank"]) if e["rank"] else float("inf"),
        wins=e["wins"], med_mae=med, mae_std=np.std(e["mae"]) if len(e["mae"]) > 1 else 0.0,
        mean_bias=np.mean(e["bias"]) if e["bias"] else float("nan"),
        cov=np.mean(e["cov"]) if e["cov"] else None,
        diverged=e["diverged"],
        ess=np.median(e["ess"]) if e["ess"] else None,
        rhat=np.max(e["rhat"]) if e["rhat"] else None)


def main():
    runs = load_runs()
    if not runs:
        print("No runs found in docs/robustness/. Run snapshot_results.py after some pipeline runs.")
        return
    seeds = [r["seed"] for r in runs]
    eng = aggregate(runs)

    nat = {k: v for k, v in eng.items() if v["world"] == "national"}
    rows = sorted((_row(k, v, len(runs)) for k, v in nat.items()), key=lambda r: r["avg_rank"])
    tbl_rows = []
    for r in rows:
        cov = f"{100*r['cov']:.0f}%" if r["cov"] is not None else "—"
        samp = f"{r['ess']:.0f} / {r['rhat']:.3f}" if r["ess"] is not None else "—"
        rank = f"{r['avg_rank']:.1f}" if np.isfinite(r["avg_rank"]) else "—"
        div = f"{r['diverged']}" if r["diverged"] else "—"
        tbl_rows.append([r["label"], str(r["n"]), rank, str(r["wins"]),
                         f"{r['med_mae']:.0f}", f"±{r['mae_std']:.0f}", f"{r['mean_bias']:+.0f}%",
                         cov, div, samp])
    nat_tbl = mr._tbl(["engine", "runs", "avg rank ↓", "wins", "median MAE", "MAE spread",
                       "mean bias", "CI cov", "diverged", "ESS / R̂"], tbl_rows)

    geo = {k: v for k, v in eng.items() if v["world"] == "geo"}
    geo_rows = []
    for k, v in sorted(geo.items(), key=lambda kv: np.mean(kv[1]["mae"]) if kv[1]["mae"] else 9e9):
        if not v["mae"]:
            continue
        geo_rows.append([v["label"], str(len(v["mae"])), f"{np.median(v['mae']):.0f}",
                         f"{np.mean(v['bias']):+.0f}%"])
    geo_tbl = mr._tbl(["geo engine", "runs", "median MAE", "mean bias"], geo_rows)

    DIR.mkdir(parents=True, exist_ok=True)
    css = mr.CSS
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MMM multi-seed leaderboard</title><style>{css}</style></head><body>
<header class="hero"><div class="wrap">
<div class="kicker">A Skeptic's Guide to Marketing Mix Modeling</div>
<h1>Multi-seed leaderboard</h1>
<p class="sub">{len(runs)} synthetic worlds (seeds {", ".join(map(str, seeds))}) — which engines are
reliably good, not just lucky on one. This is the canonical ranking; the
<a href="../engines/index.html">single-dataset leaderboard</a> is a per-seed deep dive with the
full engine set and figures. <a href="../runs/index.html">run tracker</a></p>
<p>A single leaderboard answers "who won this dataset?"; that ranking is unstable. This page answers
"who wins on average, and how often?" — the only fair way to compare methods. <b>Average rank</b> and
<b>win-rate</b> are computed among non-diverged national engines within each run; <b>MAE spread</b>
(std across seeds) is the stability we actually care about; <b>ESS / R̂</b> is sampling quality for
the Bayesian engines (a recovery you couldn't sample isn't a recovery).</p>
</div></header>
<main class="wrap">
<section><div class="card">
<h2>National engines — ranked by average rank across {len(runs)} seeds</h2>
{nat_tbl}
<p class="small">Each seed regenerates everything (data, experiments, sealed truth), so results are
recomputed every run — nothing is reused. Lower average rank is better; "wins" counts datasets where
the engine was #1; "MAE spread" is the across-seed std (lower = more stable). With only {len(runs)}
run(s) this is indicative, not final — more seeds tighten it.</p>
</div></section>
<section><div class="card">
<h2>Geo engines — the confounder, separately</h2>
<p>Geo engines are graded against the geo world's own (harder) answer key and are dominated by the
unobserved demand confounder, so they are not part of the national horse race. Summarised here to
confirm the confound result reproduces across seeds.</p>
{geo_tbl}
</div></section>
</main>
<footer class="wrap">Generated by <code>scripts/robustness.py</code> from per-seed snapshots.</footer>
</body></html>"""
    (DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"Robustness across {len(runs)} seed(s): {seeds}")
    print(f"{'engine':34s} {'n':>2s} {'avgRank':>7s} {'wins':>4s} {'medMAE':>6s} {'±std':>5s} {'bias':>6s}")
    for r in rows:
        rank = f"{r['avg_rank']:.1f}" if np.isfinite(r["avg_rank"]) else "—"
        print(f"  {r['label']:32s} {r['n']:>2d} {rank:>7s} {r['wins']:>4d} "
              f"{r['med_mae']:>6.0f} {r['mae_std']:>5.0f} {r['mean_bias']:>+5.0f}%")
    print(f"Wrote {DIR/'index.html'}")


if __name__ == "__main__":
    main()
