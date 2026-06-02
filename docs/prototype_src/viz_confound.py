"""Demonstrate Confound #2: spend correlated with seasonal demand.
Show that OMITTING the seasonal control inflates channel effect,
and INCLUDING it largely fixes it. Deliberately simple: one channel,
linear (no adstock/saturation) so the mechanism is unmistakable."""
import numpy as np, matplotlib.pyplot as plt
import statsmodels.api as sm

rng = np.random.default_rng(42)
T = 140
week = np.arange(T)

# ---- GROUND TRUTH ----
# Seasonal demand: annual sine peaking around football season (wks ~35-52)
season = 600 * (0.5 + 0.5*np.sin(2*np.pi*(week-9)/52))   # 0..600, peak in fall
baseline = 1000.0

# Media planner spends MORE when season is high  -> the confound
# spend is largely a function of season + a little independent noise
spend = 20 + 0.05*season + rng.normal(0, 2.0, T)
spend = np.clip(spend, 0, None)

TRUE_BETA = 8.0   # each unit of spend truly drives 8 conversions (linear, no sat)
true_media_contrib = TRUE_BETA * spend
true_season_contrib = season

conversions = baseline + true_season_contrib + true_media_contrib + rng.normal(0, 40, T)

# correlation between spend and season
corr = np.corrcoef(spend, season)[0,1]

# ---- MODEL A: OMIT seasonal control (naive) ----
Xa = sm.add_constant(spend)
mA = sm.OLS(conversions, Xa).fit()
beta_a = mA.params[1]; ci_a = mA.conf_int()[1]

# ---- MODEL B: INCLUDE seasonal control ----
Xb = sm.add_constant(np.column_stack([spend, season]))
mB = sm.OLS(conversions, Xb).fit()
beta_b = mB.params[1]; ci_b = mB.conf_int()[1]

print(f"corr(spend, season) = {corr:.3f}")
print(f"TRUE beta (conversions per unit spend) = {TRUE_BETA}")
print(f"MODEL A (no season control): beta = {beta_a:.2f}  95% CI [{ci_a[0]:.1f}, {ci_a[1]:.1f}]")
print(f"MODEL B (with season control): beta = {beta_b:.2f}  95% CI [{ci_b[0]:.1f}, {ci_b[1]:.1f}]")
print(f"=> Model A over-states spend effect by {beta_a/TRUE_BETA:.1f}x")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
ax.plot(week, conversions, color="#333", lw=1, label="conversions (observed)")
ax.plot(week, baseline+true_season_contrib, color="#2ca02c", ls="--", label="true baseline+season")
ax2 = ax.twinx()
ax2.plot(week, spend, color="#d62728", alpha=0.6, label="spend (right axis)")
ax.set_title(f"Spend tracks seasonal demand (corr={corr:.2f})")
ax.set_xlabel("week"); ax.set_ylabel("conversions"); ax2.set_ylabel("spend", color="#d62728")
ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
labels = ["TRUE", "Model A\n(no season\ncontrol)", "Model B\n(with season\ncontrol)"]
vals = [TRUE_BETA, beta_a, beta_b]
errs = [[0,0],
        [beta_a-ci_a[0], ci_a[1]-beta_a],
        [beta_b-ci_b[0], ci_b[1]-beta_b]]
errs = np.array(errs).T
colors = ["#2ca02c", "#d62728", "#1f77b4"]
ax.bar(labels, vals, color=colors, alpha=0.85)
ax.errorbar(labels, vals, yerr=errs, fmt="none", ecolor="black", capsize=6, lw=1.5)
ax.axhline(TRUE_BETA, color="#2ca02c", ls="--", lw=1)
ax.set_ylabel("estimated conversions per unit spend (β)")
ax.set_title("Omitting the seasonal control INFLATES channel effect")
for i,v in enumerate(vals):
    ax.text(i, v+0.4, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig("confound.png", dpi=120, bbox_inches="tight")
print("saved confound.png")
