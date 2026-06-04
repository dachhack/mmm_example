"""scripts/snapshot_results.py — persist ONE run's graded results for multi-seed robustness.

Each seed is an independent synthetic world (new data, new experiments, new sealed truth), so its
engine results can't be reused — they're recomputed every run. To study ROBUSTNESS we snapshot each
run's grades into docs/robustness/run_<seed>.json, then scripts/robustness.py aggregates across
seeds (average rank, win-rate, MAE distribution, stability).

For the Bayesian (PyMC) engines we also record sampling quality — min bulk ESS and max R-hat over
the channel betas — so we can tell a robust recovery from one that merely didn't mix.

CONTRACT: a grading/reporting tool (like the leaderboard); it legitimately reads data_sealed/.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import engine_leaderboard as lb  # noqa: E402
from draftzone_mmm.model import ARTIFACTS, CHANNELS, REPO, load_idata, load_national  # noqa: E402

OUT = REPO / "docs" / "robustness"


def pymc_diagnostics(idata_path):
    """min bulk-ESS and max R-hat over the channel beta parameters (what drives recovery)."""
    if not idata_path.exists():
        return None
    try:
        import arviz as az
        idata = load_idata(idata_path)
        betas = [f"beta_{c}" for c in CHANNELS]
        ess = az.ess(idata, var_names=betas, method="bulk")
        rhat = az.rhat(idata, var_names=betas)
        ess_min = float(min(float(ess[v].values) for v in betas))
        rhat_max = float(max(float(rhat[v].values) for v in betas))
        return dict(ess_bulk_min=ess_min, rhat_max=rhat_max)
    except Exception as e:  # pragma: no cover
        return dict(error=str(e))


def main():
    cfg = json.load(open(REPO / "data" / "config.json"))
    seed = cfg.get("seed", 0)
    df = load_national()
    sealed, gtd, geo_gtd = lb.load_truths()
    truth_for = lambda e: geo_gtd if lb._is_geo(e) else gtd  # noqa: E731
    engines = lb.discover_engines(df)

    diag = {"pymc_obs": pymc_diagnostics(ARTIFACTS / "idata.nc"),
            "pymc_anchored": pymc_diagnostics(ARTIFACTS / "idata_anchored.nc")}

    out = dict(seed=seed,
               national_media_total=float(sum(gtd[f"media_{c}"] for c in CHANNELS)),
               confound=cfg.get("realized_confound"), engines={})
    for e in engines:
        g = lb.grade(e, truth_for(e))
        out["engines"][e["engine"]] = dict(
            label=e["label"], world=("geo" if lb._is_geo(e) else "national"),
            mae=g["mae"], bias=g["media_bias"], hits=g["hits"], n_ci=g["n_ci"],
            r2=g["r2"], diverged=bool(lb.diverged(e, truth_for(e))),
            sampling=diag.get(e["engine"]))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"run_{seed}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Snapshot seed={seed}: {len(out['engines'])} engines -> {path}")
    for eng, r in sorted(out["engines"].items(), key=lambda kv: kv[1]["mae"]):
        s = r["sampling"]
        sd = f"  ESS_min={s['ess_bulk_min']:.0f} Rhat_max={s['rhat_max']:.3f}" if s and "ess_bulk_min" in s else ""
        flag = " DIVERGED" if r["diverged"] else ""
        print(f"  {r['label']:34s} MAE={r['mae']:7.0f} bias={r['bias']:+6.0f}%{sd}{flag}")


if __name__ == "__main__":
    main()
