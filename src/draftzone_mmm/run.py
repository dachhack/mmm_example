"""draftzone_mmm.run — the productised, findings-optimised MMM.

`run_mmm(df, ...)` takes your own weekly data and fits the configuration this project's competition
found to be the robust, trustworthy default — then reports the things that actually matter for a
decision. It encodes the hard-won lessons, not just a fit:

  * REGULARISE. Every parameter has a proper prior (Bayesian), because the competition showed that
    un-regularised least squares (frequentist NLS, the raw spend-ladder curve fit) blows up on a
    meaningful fraction of datasets. Priors are what keep a thin-data, many-parameter MMM sane.
  * CONTROL SEASONALITY. A Fourier seasonal basis + your observed controls go in by default, because
    the spend<->season confound otherwise inflates attribution ~2x.
  * REPORT UNCERTAINTY EVERYWHERE. No bare point estimates — every channel number carries a credible
    interval, and we report sampling quality (ESS / R-hat) so you can tell a recovery from a fit that
    merely didn't mix.
  * MARGINAL != AVERAGE ROI. Decisions use marginal ROI under uncertainty.
  * SEPARATE CONFIDENT MOVES FROM TEST-FIRST. The recommendation splits channels into "act now"
    (the posterior is sure) vs "run an experiment first" (it isn't).
  * DON'T NAIVELY ANCHOR. The competition showed that bolting a single lift number onto the model as
    a prior usually HURTS. Instead we measure the spend<->season confound and tell you when an
    observational fit cannot be trusted and a randomized experiment (geo lift / spend ladder) is the
    only fix.

This is a single-engine convenience wrapper (a well-configured Bayesian MMM). For a head-to-head of
engines on YOUR data, use the competition harness (scripts/engine_leaderboard.py).
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd

from .transforms import geometric_adstock, hill_saturation

ADSTOCK_L = 12


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------
@dataclasses.dataclass
class MMMResult:
    channels: list
    contributions: pd.DataFrame      # channel, mean, lo, hi, share  (avg weekly contribution + 89% CI)
    roi: pd.DataFrame                # channel, avg_roi, mroi_mean, mroi_lo, mroi_hi  (per spend unit)
    diagnostics: dict                # r2, confound, ess_min, rhat_max, n_weeks, kpi_per_period
    idata: object                    # arviz InferenceData (for power users)
    _draws: dict
    _meta: dict

    # --- decisions -----------------------------------------------------
    def recommend(self, confidence: float = 0.89) -> pd.DataFrame:
        """Per-channel verdict: scale up / hold / cut (confident) vs test-first.

        A move is 'confident' only if the channel's marginal ROI is, across the posterior, robustly
        above (scale up) or below (cut) the portfolio's spend-weighted average marginal ROI. If the
        sign is uncertain, the honest answer is to run an experiment before reallocating."""
        m = self._draws["mroi"]                       # draws x channel marginal ROI
        w = self._meta["spend"] / self._meta["spend"].sum()
        port = (m * w[None, :]).sum(1, keepdims=True)  # portfolio avg marginal ROI per draw
        rel = m - port                                 # how much better/worse than portfolio
        lo, hi = np.percentile(rel, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100], axis=0)
        rows = []
        for i, c in enumerate(self.channels):
            if lo[i] > 0:
                v = "scale up (confident)"
            elif hi[i] < 0:
                v = "cut (confident)"
            else:
                v = "test first (uncertain)"
            rows.append(dict(channel=c, spend=float(self._meta["spend"][i]),
                             mroi=float(m[:, i].mean()),
                             mroi_vs_portfolio=float(rel[:, i].mean()), verdict=v))
        return pd.DataFrame(rows).sort_values("mroi_vs_portfolio", ascending=False).reset_index(drop=True)

    def summary(self) -> str:
        d = self.diagnostics
        lines = [
            "DraftZone MMM — findings-optimised fit",
            f"  weeks={d['n_weeks']}  fit R²={d['r2']:.3f}  sampling: min ESS={d['ess_min']:.0f}, "
            f"max R-hat={d['rhat_max']:.3f}",
            f"  spend↔seasonality confound = {d['confound']:+.2f}  "
            + ("(LOW — observational attribution is more trustworthy)"
               if abs(d["confound"]) < 0.3 else
               "(HIGH — spend tracks demand; observational attribution is fragile, "
               "triangulate with a geo experiment)"),
            "",
            "Contribution (avg/wk, 89% CI) and marginal ROI:",
        ]
        roi = self.roi.set_index("channel")
        for _, r in self.contributions.iterrows():
            c = r["channel"]
            lines.append(f"  {c:20s} {r['mean']:8.1f}  [{r['lo']:7.1f},{r['hi']:7.1f}]  "
                         f"({r['share']:4.0%})   mROI={roi.loc[c, 'mroi_mean']:.3f}/unit")
        if d["ess_min"] < 400 or d["rhat_max"] > 1.02:
            lines.append("  ⚠ sampling quality is marginal (low ESS / high R-hat) — increase "
                         "draws/tune before trusting the intervals.")
        lines += [
            "",
            "Recommendation (confident moves vs test-first):",
        ]
        for _, r in self.recommend().iterrows():
            lines.append(f"  {r['channel']:20s} {r['verdict']}")
        lines += [
            "",
            "Reminders from the competition:",
            "  • These are correlational unless your confound is low or you've run experiments.",
            "  • Decisions use MARGINAL ROI under uncertainty (above), not average ROI.",
            "  • Act on the confident moves; route the 'test first' channels to a geo lift / spend",
            "    ladder — that is the only confound-immune evidence.",
        ]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------
def _resolve_cols(df, channels, spend_cols, spend_suffix, exposure_cols, exposure_suffix):
    sp = spend_cols or {c: f"{c}{spend_suffix}" for c in channels}
    if exposure_cols:
        ex = exposure_cols
    elif exposure_suffix:
        ex = {c: f"{c}{exposure_suffix}" for c in channels}
    else:
        ex = dict(sp)  # no exposure series -> model on spend itself
    for c in channels:
        if sp[c] not in df:
            raise KeyError(f"spend column '{sp[c]}' for channel '{c}' not in DataFrame")
        if ex[c] not in df:
            raise KeyError(f"exposure column '{ex[c]}' for channel '{c}' not in DataFrame")
    return sp, ex


def _seasonality(n, period, harmonics):
    t = np.arange(n)
    cols, names = [t / max(n, 1)], ["trend"]
    for k in range(1, harmonics + 1):
        cols += [np.sin(2 * np.pi * k * t / period), np.cos(2 * np.pi * k * t / period)]
        names += [f"sin{k}", f"cos{k}"]
    return cols, names


def run_mmm(df, kpi, channels, *, spend_cols=None, spend_suffix="_spend",
            exposure_cols=None, exposure_suffix=None, controls=None, date=None,
            seasonality_period=52, fourier_harmonics=3,
            draws=600, tune=600, chains=2, cores=1, seed=42, progressbar=False):
    """Fit the findings-optimised Bayesian MMM on a tidy weekly DataFrame.

    df       : one row per period, sorted by time.
    kpi      : outcome column (e.g. "conversions" or "revenue").
    channels : list of channel names.
    spend_cols / spend_suffix      : map channel -> spend column (default "<channel>_spend").
    exposure_cols / exposure_suffix: optional impressions/exposure to model on (default: spend).
    controls : optional list of observed control columns (price, promo, competitor, ...).
    date     : optional date column (unused by the model; for your reference).
    Returns an MMMResult. Heavy step: PyMC sampling — raise draws/chains for production.
    """
    import pymc as pm
    import pytensor.tensor as pt

    df = df.reset_index(drop=True)
    n = len(df)
    y = df[kpi].to_numpy(float)
    sp, ex = _resolve_cols(df, channels, spend_cols, spend_suffix, exposure_cols, exposure_suffix)

    # scaled (~O(1)) exposure per channel + padded windows for truncated, normalized adstock
    L = ADSTOCK_L
    exp_raw = {c: df[ex[c]].to_numpy(float) for c in channels}
    exp_scale = {c: max(exp_raw[c].mean(), 1e-9) for c in channels}
    exp_s = {c: exp_raw[c] / exp_scale[c] for c in channels}
    lag_idx = np.arange(n)[:, None] + np.arange(L)[None, :]
    padded = {c: np.concatenate([np.zeros(L - 1), exp_s[c]]) for c in channels}
    wexp = np.arange(L).astype(float)

    # control design: Fourier seasonality + standardized user controls
    ccols, cnames = _seasonality(n, seasonality_period, fourier_harmonics)
    for col in (controls or []):
        v = df[col].to_numpy(float)
        s = v.std() or 1.0
        ccols.append((v - v.mean()) / s)
        cnames.append(col)
    Xc = np.column_stack(ccols).astype(float)

    y_mean, y_sd = float(y.mean()), float(y.std() or 1.0)

    with pm.Model() as model:
        baseline = pm.Normal("baseline", y_mean, y_sd)
        ctrl = pm.Normal("ctrl", 0, y_sd, shape=Xc.shape[1])
        sigma = pm.HalfNormal("sigma", y_sd)
        media = pt.zeros(n)
        thetas, slopes, hss, betas = {}, {}, {}, {}
        for c in channels:
            theta = pm.Beta(f"theta_{c}", 2.0, 3.0)                 # mean ~0.4 carryover (generic)
            slope = pm.Gamma(f"slope_{c}", mu=1.5, sigma=0.6)
            hs = pm.Gamma(f"hs_{c}", mu=1.0, sigma=0.6)             # on the scaled exposure
            beta = pm.HalfNormal(f"beta_{c}", sigma=y_sd)           # KPI-scale ceiling, regularised
            thetas[c], slopes[c], hss[c], betas[c] = theta, slope, hs, beta
            w = theta ** pt.as_tensor_variable(wexp)
            w = (w / pt.sum(w))[::-1]
            ad = pt.sum(pt.as_tensor_variable(padded[c])[lag_idx] * w[None, :], axis=1)
            sat = ad ** slope / (ad ** slope + hs ** slope + 1e-9)
            media = media + beta * sat
        mu = baseline + pt.dot(pt.as_tensor_variable(Xc), ctrl) + media
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(draws, tune=tune, chains=chains, cores=cores, target_accept=0.92,
                              random_seed=seed, progressbar=progressbar)

    return _summarise(idata, df, kpi, channels, sp, exp_s, exp_scale, exp_raw, Xc, cnames, L)


def _summarise(idata, df, kpi, channels, sp, exp_s, exp_scale, exp_raw, Xc, cnames, L):
    import arviz as az
    post = idata.posterior.stack(s=("chain", "draw"))
    S = post.sizes["s"]
    y = df[kpi].to_numpy(float)
    n = len(y)

    def col(name):
        return post[name].values  # shape (s,) or (dim, s)

    base = col("baseline"); ctrl = col("ctrl");
    # per-channel contribution time series per draw, and marginal ROI per draw
    contrib_draws = {c: np.zeros((S, n)) for c in channels}
    mroi = np.zeros((S, len(channels)))
    avg_roi = np.zeros(len(channels))
    spend_mean = np.array([df[sp[c]].to_numpy(float).mean() for c in channels])

    def hill_ad(es, theta, hs, slope):
        ad = geometric_adstock(es, theta, normalize=True, L=L)
        return hill_saturation(ad, hs, slope)

    for j, c in enumerate(channels):
        th = col(f"theta_{c}"); sl = col(f"slope_{c}"); hsv = col(f"hs_{c}"); be = col(f"beta_{c}")
        es = exp_s[c]
        # exposure responds ~ proportionally to spend; bump spend 1% -> bump exposure 1%
        for i in range(S):
            f0 = hill_ad(es, th[i], hsv[i], sl[i])
            contrib_draws[c][i] = be[i] * f0
        # marginal ROI: dContribution/dSpend at current spend (finite diff on the posterior mean curve)
        c_now = contrib_draws[c].mean(1)                       # avg weekly contribution per draw
        for i in range(S):
            f1 = hill_ad(es * 1.01, th[i], hsv[i], sl[i])
            d_contrib = (be[i] * f1).mean() - c_now[i]
            mroi[i, j] = d_contrib / (0.01 * spend_mean[j]) if spend_mean[j] > 0 else 0.0
        avg_roi[j] = c_now.mean() / spend_mean[j] if spend_mean[j] > 0 else 0.0

    # contributions table
    crows, total = [], np.zeros(S)
    means = {c: contrib_draws[c].mean(1) for c in channels}
    tot_mean = sum(means[c].mean() for c in channels)
    for c in channels:
        a = means[c]
        crows.append(dict(channel=c, mean=float(a.mean()),
                          lo=float(np.percentile(a, 5.5)), hi=float(np.percentile(a, 94.5)),
                          share=float(a.mean() / tot_mean) if tot_mean else 0.0))
    contributions = pd.DataFrame(crows)

    roi = pd.DataFrame([dict(channel=c, avg_roi=float(avg_roi[j]),
                             mroi_mean=float(mroi[:, j].mean()),
                             mroi_lo=float(np.percentile(mroi[:, j], 5.5)),
                             mroi_hi=float(np.percentile(mroi[:, j], 94.5)))
                        for j, c in enumerate(channels)])

    # fit + diagnostics
    mu_hat = (base.mean() + Xc @ ctrl.mean(1)
              + sum(contrib_draws[c].mean(0) for c in channels))
    r2 = float(1 - np.sum((y - mu_hat) ** 2) / np.sum((y - y.mean()) ** 2))
    betas_names = [f"beta_{c}" for c in channels]
    try:
        ess_min = float(min(float(az.ess(idata, var_names=[b])[b].values) for b in betas_names))
        rhat_max = float(max(float(az.rhat(idata, var_names=[b])[b].values) for b in betas_names))
    except Exception:  # e.g. a single-chain fit — diagnostics need >=2 chains
        ess_min, rhat_max = float("nan"), float("nan")
    # confound: corr(total spend, seasonal part of the fitted controls)
    total_spend = sum(df[sp[c]].to_numpy(float) for c in channels)
    season_idx = [i for i, nm in enumerate(cnames) if nm.startswith(("sin", "cos"))]
    season = (Xc[:, season_idx] @ ctrl.mean(1)[season_idx]) if season_idx else np.zeros(n)
    confound = float(np.corrcoef(total_spend, season)[0, 1]) if season.std() > 0 else 0.0

    diagnostics = dict(r2=r2, confound=confound, ess_min=ess_min, rhat_max=rhat_max,
                       n_weeks=n, kpi_per_period=float(y.mean()))
    return MMMResult(channels=list(channels), contributions=contributions, roi=roi,
                     diagnostics=diagnostics, idata=idata,
                     _draws=dict(mroi=mroi), _meta=dict(spend=spend_mean))
