"""scripts/fit_meridian.py — Google Meridian as an alternative MMM engine.

Fits Google Meridian on the public national dataset and writes a canonical engine-results
contract (artifacts/meridian_results.json) that the harness can grade against the sealed truth
identically to our own PyMC engine. This is an OPTIONAL engine (heavy TensorFlow dep): install
with `pip install -e ".[meridian]"` and run on the VM, not in CI.

CONTRACT: this is a modeling engine — it MUST NOT read data_sealed/ground_truth.json. It only
reads the public data/. Grading happens downstream.

Fair-configuration notes (learned the hard way benchmarking Meridian against known truth):
  * Meridian's DEFAULT ROI prior (~1.2) assumes a revenue KPI; for a conversions KPI at our
    scale (~0.01 conv/$) it over-credits media ~100x. We set a scale-correct ROI prior from the
    data (no truth): mean ~ 0.5 * total_conversions / total_spend.
  * Meridian's time-varying baseline does not, by itself, absorb a strong seasonal confound, so
    we pass the SAME Fourier seasonality basis (3 harmonics) our PyMC model uses. Without this,
    the spend<->season confound leaks into media and inflates attribution ~2x.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = pathlib.Path(__file__).resolve().parents[1]
CHANNELS = ["paid_social", "paid_search", "programmatic_display", "influencer", "dooh", "tv_ctv"]


def build_input_data(df):
    from meridian.data import data_frame_input_data_builder as bld
    df = df.copy()
    df["geo"] = "national"
    df["population"] = 1.0
    t = np.arange(len(df))
    fourier = []
    for k in range(1, 4):  # match draftzone_mmm.model.build_controls
        df[f"sin{k}"] = np.sin(2 * np.pi * k * t / 52)
        df[f"cos{k}"] = np.cos(2 * np.pi * k * t / 52)
        fourier += [f"sin{k}", f"cos{k}"]
    controls = ["promo_flag", "price_index", "competitor_pressure", "holiday_flag"] + fourier
    spend = [f"{c}_spend" for c in CHANNELS]
    imp = [f"{c}_impressions" for c in CHANNELS]
    b = bld.DataFrameInputDataBuilder(
        kpi_type="non_revenue", default_geo_column="geo", default_time_column="week",
        default_media_time_column="week", default_population_column="population")
    b = b.with_kpi(df, kpi_col="conversions").with_population(df)
    b = b.with_controls(df, control_cols=controls)
    b = b.with_media(df, media_cols=imp, media_spend_cols=spend, media_channels=CHANNELS)
    return b.build()


def scale_correct_roi_prior(df, sigma=0.7):
    """Data-derived ROI prior (no truth): assume media ~half of conversions / total spend."""
    from tensorflow_probability import distributions as tfd
    total_conv = df["conversions"].sum()
    total_spend = sum(df[f"{c}_spend"].sum() for c in CHANNELS)
    mean = 0.5 * total_conv / total_spend
    loc = float(np.log(mean) - 0.5 * sigma ** 2)
    return tfd.LogNormal(loc=np.full(len(CHANNELS), loc, "float32"),
                         scale=np.full(len(CHANNELS), sigma, "float32"))


def experiment_mroi_prior(sigma=0.4):
    """Per-channel MARGINAL-ROI prior from the geo experiments. A lift test measures the
    incremental KPI from incremental spend — i.e. mROI, not average ROI — so this is the prior
    Meridian's lift-calibration should target (the naive average-ROI calibration over-credited).
    mROI_c = causal DiD conversions / incremental campaign spend (both per market-week)."""
    import pandas as pd
    from tensorflow_probability import distributions as tfd
    geo = pd.read_csv(REPO / "data" / "geo_experiments.csv")
    anchors = json.load(open(REPO / "artifacts" / "anchors.json"))["anchors"]
    means = []
    for c in CHANNELS:
        camp = geo[(geo.channel == c) & (geo.campaign_window == 1)]
        inc_spend = camp[camp.treated == 1].spend.mean() - camp[camp.treated == 0].spend.mean()
        means.append(max(anchors[c]["did"] / inc_spend, 1e-6))
    loc = (np.log(np.asarray(means)) - 0.5 * sigma ** 2).astype("float32")
    return tfd.LogNormal(loc=loc, scale=np.full(len(CHANNELS), sigma, "float32")), means


def main():
    ap = argparse.ArgumentParser(description="Fit Google Meridian and write the engine contract.")
    ap.add_argument("--out", default=str(REPO / "artifacts" / "meridian_results.json"))
    ap.add_argument("--calibrate", action="store_true",
                    help="experiment-calibrate via mROI prior from the geo anchors")
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--adapt", type=int, default=500)
    ap.add_argument("--burnin", type=int, default=500)
    ap.add_argument("--keep", type=int, default=500)
    ap.add_argument("--max-lag", type=int, default=12)
    args = ap.parse_args()

    from meridian.analysis import analyzer
    from meridian.model import model, prior_distribution as pdist, spec

    df = pd.read_csv(REPO / "data" / "national_weekly.csv")
    T = len(df)
    data = build_input_data(df)
    if args.calibrate:
        mroi_prior, mroi_means = experiment_mroi_prior()
        print("Experiment mROI prior (conv/$):", {c: round(m, 5) for c, m in zip(CHANNELS, mroi_means)})
        ms = spec.ModelSpec(max_lag=args.max_lag, paid_media_prior_type="mroi",
                            prior=pdist.PriorDistribution(mroi_m=mroi_prior))
        engine = "google_meridian_calibrated"
    else:
        ms = spec.ModelSpec(max_lag=args.max_lag, paid_media_prior_type="roi",
                            prior=pdist.PriorDistribution(roi_m=scale_correct_roi_prior(df)))
        engine = "google_meridian"
    mmm = model.Meridian(input_data=data, model_spec=ms)
    mmm.sample_prior(100, seed=1)
    print("Sampling Meridian posterior (TFP NUTS)...", flush=True)
    mmm.sample_posterior(n_chains=args.chains, n_adapt=args.adapt,
                         n_burnin=args.burnin, n_keep=args.keep, seed=1)

    az = analyzer.Analyzer(mmm)
    io = az.summary_metrics(confidence_level=0.89)["incremental_outcome"].sel(distribution="posterior")
    pa = az.predictive_accuracy()

    channels = {}
    for c in CHANNELS:
        channels[c] = dict(
            est_contrib=float(io.sel(channel=c, metric="mean")) / T,
            ci=[float(io.sel(channel=c, metric="ci_lo")) / T,
                float(io.sel(channel=c, metric="ci_hi")) / T],
        )
    results = dict(
        engine=engine,
        bayesian=True,
        fit=dict(r2=float(pa["value"].sel(metric="R_Squared").values.item()),
                 mape=float(pa["value"].sel(metric="MAPE").values.item())),
        channels=channels,
        note="Avg weekly contribution per channel (conversions/wk) with 89% CI. Fair config: "
             "scale-corrected ROI prior + Fourier seasonality control.",
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"R²={results['fit']['r2']:.3f}  MAPE={results['fit']['mape']:.3f}")
    for c in CHANNELS:
        ch = channels[c]
        print(f"  {c:22s} est={ch['est_contrib']:6.1f} [{ch['ci'][0]:6.1f},{ch['ci'][1]:6.1f}]")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
