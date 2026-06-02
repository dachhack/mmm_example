"""Stage 2b: the NAIVE model — raw linear OLS, no adstock/saturation, weak controls.
Grade it against the hidden answer key to see how wrong it is."""
import pandas as pd, numpy as np, json, matplotlib.pyplot as plt
import statsmodels.api as sm
df = pd.read_csv("draftzone_mmm_data.csv", parse_dates=["week"])
gt = json.load(open("ground_truth.json"))
chans = ["tv","search","social","affiliate","brand"]

# ---- NAIVE MODEL: conversions ~ raw spend (5 channels) + trend only. No season, no adstock, no saturation.
X_naive = df[[f"{c}_spend" for c in chans]].copy()
X_naive["trend"] = np.arange(len(df))
X_naive = sm.add_constant(X_naive)
m_naive = sm.OLS(df.conversions, X_naive).fit()

# ---- A "less naive" model that ADDS proper controls but still no adstock/saturation
X_ctrl = df[[f"{c}_spend" for c in chans]].copy()
X_ctrl["trend"] = np.arange(len(df))
X_ctrl["season"] = df.seasonality_true        # give it the real season as a control
X_ctrl["promo"] = df.promo_flag
X_ctrl["price"] = df.price_index
X_ctrl["comp"] = df.competitor_pressure
X_ctrl["holiday"] = df.holiday_flag
X_ctrl = sm.add_constant(X_ctrl)
m_ctrl = sm.OLS(df.conversions, X_ctrl).fit()

# ---- Translate spend coefficients into implied avg contribution per channel, compare to truth
print("="*78)
print(f"{'channel':10s} {'TRUE contrib':>13s} {'NAIVE implied':>14s} {'+controls implied':>18s}")
print("-"*78)
results = {}
for c in chans:
    true_contrib = gt["avg_contribution_decomposition"][f"media_{c}"]
    naive_implied = m_naive.params[f"{c}_spend"] * df[f"{c}_spend"].mean()
    ctrl_implied  = m_ctrl.params[f"{c}_spend"] * df[f"{c}_spend"].mean()
    results[c] = (true_contrib, naive_implied, ctrl_implied)
    print(f"{c:10s} {true_contrib:13.1f} {naive_implied:14.1f} {ctrl_implied:18.1f}")
print("-"*78)
true_total = sum(gt["avg_contribution_decomposition"][f"media_{c}"] for c in chans)
naive_total = sum(m_naive.params[f"{c}_spend"]*df[f"{c}_spend"].mean() for c in chans)
ctrl_total  = sum(m_ctrl.params[f"{c}_spend"]*df[f"{c}_spend"].mean() for c in chans)
print(f"{'TOTAL media':10s} {true_total:13.1f} {naive_total:14.1f} {ctrl_total:18.1f}")
print(f"\nNaive total media contribution overstated by {naive_total/true_total:.2f}x")
print(f"With controls (still no adstock/sat) by {ctrl_total/true_total:.2f}x")
print(f"\nNaive implied BASELINE (intercept): {m_naive.params['const']:.0f}  (true 1200)")
print(f"+controls implied baseline:         {m_ctrl.params['const']:.0f}  (true 1200)")
print(f"\nNaive R²={m_naive.rsquared:.3f}   +controls R²={m_ctrl.rsquared:.3f}")

# ---- chart: true vs naive vs controls contribution
fig, ax = plt.subplots(figsize=(11,5))
x = np.arange(len(chans)); w=0.27
true_v = [results[c][0] for c in chans]
naive_v= [results[c][1] for c in chans]
ctrl_v = [results[c][2] for c in chans]
ax.bar(x-w, true_v, w, label="TRUE", color="#2ca02c")
ax.bar(x,   naive_v, w, label="Naive (raw spend, no season)", color="#d62728")
ax.bar(x+w, ctrl_v, w, label="+controls (still no adstock/sat)", color="#1f77b4")
ax.set_xticks(x); ax.set_xticklabels(chans)
ax.set_ylabel("avg conversions/wk attributed"); ax.axhline(0, color="#999", lw=0.8)
ax.set_title("Naive attribution vs truth — what skipping adstock/saturation/season does")
ax.legend(); ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig("naive_vs_truth.png", dpi=110, bbox_inches="tight")
print("\nsaved naive_vs_truth.png")
