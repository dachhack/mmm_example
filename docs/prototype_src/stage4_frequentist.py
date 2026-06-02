"""
Stage 4a: FREQUENTIST MMM.
Fit conversions ~ baseline + trend + season(Fourier) + controls
                  + sum_c beta_c * Hill( adstock(impressions_c; theta_c); hs_c, slope_c )
via nonlinear least squares. The transform params (theta, hs, slope) are nonlinear;
betas and control coefs are linear given the transforms. We optimize all jointly.

KEY HONESTY: the analyst does NOT know the true params. We give the optimizer
reasonable BOUNDS and starting points reflecting domain priors (TV slow, search fast),
not the answer key.
"""
import pandas as pd, numpy as np, json
from scipy.optimize import minimize
from transforms import geometric_adstock, hill_saturation

df = pd.read_csv("draftzone_mmm_data.csv", parse_dates=["week"])
chans = ["tv","search","social","affiliate","brand"]
T = len(df)
y = df.conversions.values

# ---- build control design matrix (what the analyst would construct) ----
t = np.arange(T)
trend = t / T
# Fourier seasonality: annual cycle (period 52 wks), 3 harmonics — analyst doesn't know true shape
fourier = []
for k in range(1,4):
    fourier.append(np.sin(2*np.pi*k*t/52))
    fourier.append(np.cos(2*np.pi*k*t/52))
fourier = np.column_stack(fourier)
promo = df.promo_flag.values
price = (df.price_index.values - 100)/10
comp = (df.competitor_pressure.values - df.competitor_pressure.mean())/df.competitor_pressure.std()
holiday = df.holiday_flag.values
# control matrix (linear part), plus intercept handled separately
Xc = np.column_stack([trend, fourier, promo, price, comp, holiday])
n_ctrl = Xc.shape[1]

# scale impressions to ~O(1) for numerical stability; remember scales
imp = {c: df[f"{c}_impressions"].values for c in chans}
imp_scale = {c: imp[c].mean() for c in chans}
imp_s = {c: imp[c]/imp_scale[c] for c in chans}

# ---- parameter packing: per channel [theta, log_hs, slope]; then linear solve for betas+controls
# Domain-prior BOUNDS (NOT the answer key):
bounds_theta = {"tv":(0.3,0.95),"search":(0.0,0.5),"social":(0.1,0.8),
                "affiliate":(0.0,0.7),"brand":(0.2,0.9)}   # analyst's priors on carryover
def unpack(params):
    out = {}
    for i,c in enumerate(chans):
        out[c] = dict(theta=params[3*i], hs=np.exp(params[3*i+1]), slope=params[3*i+2])
    return out

def build_media(p):
    cols = []
    for c in chans:
        ad = geometric_adstock(imp_s[c], p[c]["theta"], normalize=True)
        sat = hill_saturation(ad, p[c]["hs"], p[c]["slope"])
        cols.append(sat)
    return np.column_stack(cols)

def fit_linear(M):
    """Given transformed media M (Tx5) and controls Xc, solve least squares for
    [intercept, betas(5), control coefs]. Betas constrained >=0 via NNLS-ish clip refit."""
    A = np.column_stack([np.ones(T), M, Xc])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A, coef

def objective(params):
    p = unpack(params)
    M = build_media(p)
    A, coef = fit_linear(M)
    resid = y - A@coef
    # penalize negative media betas (advertising shouldn't reduce conversions)
    betas = coef[1:6]
    pen = np.sum(np.clip(-betas,0,None)**2) * 1e6
    return np.sum(resid**2) + pen

# starting point: middle of bounds, hs near median adstock scale, slope ~1.5
x0 = []
for c in chans:
    lo,hi = bounds_theta[c]; x0 += [(lo+hi)/2, np.log(1.0), 1.5]
bnds = []
for c in chans:
    bnds += [bounds_theta[c], (np.log(0.05), np.log(20)), (0.5, 3.5)]

best = None
# multi-start to fight local minima (a real pain point in frequentist MMM)
rng = np.random.default_rng(0)
for s in range(8):
    if s==0: start = np.array(x0)
    else:
        start = []
        for c in chans:
            lo,hi = bounds_theta[c]
            start += [rng.uniform(lo,hi), np.log(rng.uniform(0.1,10)), rng.uniform(0.7,3.0)]
        start = np.array(start)
    res = minimize(objective, start, method="L-BFGS-B", bounds=bnds,
                   options=dict(maxiter=500))
    if best is None or res.fun < best.fun:
        best = res

p = unpack(best.x)
M = build_media(p); A, coef = fit_linear(M)
betas = coef[1:6]; intercept = coef[0]
yhat = A@coef
r2 = 1 - np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)

# recover implied avg contribution per channel (beta * mean(sat))
contrib = {c: betas[i]*M[:,i].mean() for i,c in enumerate(chans)}

# save results
out = dict(
    method="frequentist_nls_multistart",
    r2=float(r2), intercept=float(intercept),
    params={c: dict(theta=float(p[c]["theta"]), half_sat_scaled=float(p[c]["hs"]),
                    slope=float(p[c]["slope"]), beta=float(betas[i]),
                    avg_contrib=float(contrib[c])) for i,c in enumerate(chans)},
    imp_scale={c: float(imp_scale[c]) for c in chans},
)
json.dump(out, open("fit_frequentist.json","w"), indent=2)
np.save("freq_yhat.npy", yhat)

gt = json.load(open("ground_truth.json"))
print(f"Frequentist fit  R²={r2:.3f}   (multi-start best of 8)")
print(f"Recovered intercept/baseline: {intercept:.0f}   (true 1200)")
print(f"\n{'chan':9s} {'theta_hat':>9s} {'theta_tru':>9s} | {'slope_hat':>9s} {'slope_tru':>9s} | {'contrib_hat':>11s} {'contrib_tru':>11s}")
for i,c in enumerate(chans):
    g = gt["channels"][c]
    print(f"{c:9s} {p[c]['theta']:9.2f} {g['theta']:9.2f} | {p[c]['slope']:9.2f} {g['slope']:9.2f} | "
          f"{contrib[c]:11.1f} {gt['avg_contribution_decomposition']['media_'+c]:11.1f}")
print("\nSaved fit_frequentist.json")
