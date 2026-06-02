"""Geo experiment intuition: randomization defeats the confound that
broke the regression. Same brutal seasonality everywhere; randomly split
markets into treatment/control; recover a KNOWN true lift by subtraction."""
import numpy as np, matplotlib.pyplot as plt

rng = np.random.default_rng(11)
n_markets = 100
T = 16  # 16-week test
week = np.arange(T)

# Strong shared seasonality (the confounder) hits ALL markets
season = 300 * (0.5 + 0.5*np.sin(2*np.pi*(week-2)/16))   # big swing during test

# Each market has its own baseline size (heterogeneity), but assignment is RANDOM
market_baseline = rng.normal(1000, 150, n_markets)

# Random assignment -- the key step
assign = rng.permutation(n_markets)
treat_idx = assign[:50]
ctrl_idx  = assign[50:]

TRUE_LIFT = 120.0  # campaign truly adds 120 conversions/week/market in treatment

def market_series(base, treated):
    noise = rng.normal(0, 50, T)
    lift = TRUE_LIFT if treated else 0.0
    # campaign runs weeks 4..15 (a ramp), 0 before
    lift_vec = np.where(week >= 4, lift, 0.0)
    return base + season + lift_vec + noise

treat = np.array([market_series(market_baseline[i], True)  for i in treat_idx])
ctrl  = np.array([market_series(market_baseline[i], False) for i in ctrl_idx])

treat_mean = treat.mean(axis=0)
ctrl_mean  = ctrl.mean(axis=0)
diff = treat_mean - ctrl_mean

# estimate lift = avg difference during campaign weeks (4..15)
est_lift = diff[4:].mean()
pre_diff = diff[:4].mean()  # should be ~0 (validates random assignment / parallel pre-period)

print(f"TRUE lift = {TRUE_LIFT}")
print(f"Pre-campaign diff (wks 0-3, should be ~0): {pre_diff:.1f}")
print(f"Estimated lift (wks 4-15 treat-ctrl): {est_lift:.1f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
ax.plot(week, treat_mean, color="#d62728", marker="o", ms=3, label="Treatment markets (avg)")
ax.plot(week, ctrl_mean,  color="#1f77b4", marker="o", ms=3, label="Control markets (avg)")
ax.axvline(4, color="#999", ls=":"); ax.text(4.1, ax.get_ylim()[0]+20, "campaign on", fontsize=8)
ax.set_title("Both groups ride the SAME seasonality...\n(randomization makes them twins)")
ax.set_xlabel("week"); ax.set_ylabel("avg conversions/market"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(week, diff, color="#2ca02c", marker="o", ms=4, label="Treatment − Control")
ax.axhline(TRUE_LIFT, color="#2ca02c", ls="--", lw=1, label=f"true lift = {TRUE_LIFT}")
ax.axhline(0, color="#999", lw=0.8)
ax.axvspan(0, 4, color="#eee", label="pre-period (diff≈0)")
ax.set_title(f"The DIFFERENCE cancels the season.\nRecovered lift = {est_lift:.0f} (truth {TRUE_LIFT:.0f})")
ax.set_xlabel("week"); ax.set_ylabel("conversions lift/market"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("experiment.png", dpi=120, bbox_inches="tight")
print("saved experiment.png")
