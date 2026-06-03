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
    data (no truth): mean ~ 0.5 * total_conversions / total_spend. (Confirmed against Google's
    getting-started demo, whose roi_m = LogNormal(0.2, 0.9) -> median ROI ~ 1.2, a revenue-KPI
    assumption that does not transfer to our conversions KPI.)
  * SEASONALITY can be handled two ways: (a) FOURIER controls (3 harmonics, the basis our PyMC
    model uses) or (b) Meridian's idiomatic AKS — Automatic Knot Selection on the time-varying
    baseline (``enable_aks=True``, no Fourier). ``--seasonality`` picks; we bake them off.
  * GEO mode (``--mode geo``) fits the multi-geo panel (data/geo_panel.csv). This is what Meridian
    is built for: spend varying ACROSS geos within a week gives cross-sectional identification that
    the national time series cannot, the principled way to break the spend<->season confound.
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


def _add_fourier(df, t):
    cols = []
    for k in range(1, 4):  # match draftzone_mmm.model.build_controls
        df[f"sin{k}"] = np.sin(2 * np.pi * k * t / 52)
        df[f"cos{k}"] = np.cos(2 * np.pi * k * t / 52)
        cols += [f"sin{k}", f"cos{k}"]
    return cols


def build_input_data(df, fourier=True):
    """National (single-geo) InputData."""
    from meridian.data import data_frame_input_data_builder as bld
    df = df.copy()
    df["geo"] = "national"
    df["population"] = 1.0
    controls = ["promo_flag", "price_index", "competitor_pressure", "holiday_flag"]
    if fourier:
        controls += _add_fourier(df, np.arange(len(df)))
    spend = [f"{c}_spend" for c in CHANNELS]
    imp = [f"{c}_impressions" for c in CHANNELS]
    b = bld.DataFrameInputDataBuilder(
        kpi_type="non_revenue", default_geo_column="geo", default_time_column="week",
        default_media_time_column="week", default_population_column="population")
    b = b.with_kpi(df, kpi_col="conversions").with_population(df)
    b = b.with_controls(df, control_cols=controls)
    b = b.with_media(df, media_cols=imp, media_spend_cols=spend, media_channels=CHANNELS)
    return b.build()


def build_geo_input_data(df, fourier=False):
    """Multi-geo panel InputData (data/geo_panel.csv)."""
    from meridian.data import data_frame_input_data_builder as bld
    df = df.sort_values(["geo", "week"]).copy()
    controls = ["promo_flag", "price_index", "competitor_pressure", "holiday_flag"]
    if fourier:
        df["_t"] = df.groupby("geo").cumcount()
        controls += _add_fourier(df, df["_t"].to_numpy())
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


def _pa_value(pa, metric):
    """Overall predictive-accuracy scalar (geo fits may return per-granularity rows)."""
    v = pa["value"].sel(metric=metric)
    try:
        return float(v.sel(geo_granularity="national").values.item())
    except Exception:
        return float(np.asarray(v.values).mean())


def main():
    ap = argparse.ArgumentParser(description="Fit Google Meridian and write the engine contract.")
    ap.add_argument("--out", default=None, help="defaults to artifacts/<engine>.json")
    ap.add_argument("--mode", choices=["national", "geo"], default="national",
                    help="national single-series fit, or the multi-geo panel (Meridian's home turf)")
    ap.add_argument("--seasonality", choices=["fourier", "aks"], default=None,
                    help="fourier controls (default national) or Meridian AKS knots (default geo)")
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

    seasonality = args.seasonality or ("aks" if args.mode == "geo" else "fourier")
    use_fourier = seasonality == "fourier"

    if args.mode == "geo":
        df = pd.read_csv(REPO / "data" / "geo_panel.csv")
        T = df["week"].nunique()
        data = build_geo_input_data(df, fourier=use_fourier)
    else:
        df = pd.read_csv(REPO / "data" / "national_weekly.csv")
        T = len(df)
        data = build_input_data(df, fourier=use_fourier)

    # engine id: keep the canonical national+fourier as "google_meridian" for the leaderboard
    engine = "google_meridian"
    if args.mode == "geo":
        engine += "_geo"
    if seasonality == "aks" and args.mode != "geo":
        engine += "_aks"

    spec_kw = dict(max_lag=args.max_lag, enable_aks=(seasonality == "aks"))
    if args.calibrate:
        mroi_prior, mroi_means = experiment_mroi_prior()
        print("Experiment mROI prior (conv/$):", {c: round(m, 5) for c, m in zip(CHANNELS, mroi_means)})
        ms = spec.ModelSpec(paid_media_prior_type="mroi",
                            prior=pdist.PriorDistribution(mroi_m=mroi_prior), **spec_kw)
        engine = "google_meridian_calibrated" + ("_geo" if args.mode == "geo" else "")
    else:
        ms = spec.ModelSpec(paid_media_prior_type="roi",
                            prior=pdist.PriorDistribution(roi_m=scale_correct_roi_prior(df)), **spec_kw)
    print(f"Mode={args.mode}  seasonality={seasonality}  engine={engine}", flush=True)

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
        engine=engine, bayesian=True,
        fit=dict(r2=_pa_value(pa, "R_Squared"), mape=_pa_value(pa, "MAPE")),
        channels=channels,
        note=f"Avg weekly contribution per channel (conversions/wk) with 89% CI. mode={args.mode}, "
             f"seasonality={seasonality}, scale-corrected ROI prior.",
    )
    out = pathlib.Path(args.out) if args.out else (REPO / "artifacts" / f"{engine}.json")
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
