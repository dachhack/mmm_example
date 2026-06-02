"""Stage 3 demo: show transforms convert raw spend (poor predictor) into
transformed media (strong predictor of true channel contribution)."""
import pandas as pd, numpy as np, json, matplotlib.pyplot as plt
from transforms import geometric_adstock, hill_saturation
df = pd.read_csv("draftzone_mmm_data.csv", parse_dates=["week"])
gt = json.load(open("ground_truth.json"))
chans = ["tv","search","social","affiliate","brand"]
colors = {"tv":"#d62728","search":"#1f77b4","social":"#2ca02c","affiliate":"#ff7f0e","brand":"#9467bd"}

# Reconstruct each channel's TRUE contribution using the answer-key params, on impressions.
print("Correlation with TRUE channel contribution:")
print(f"{'channel':10s} {'raw spend':>10s} {'raw impr':>10s} {'adstock impr':>13s} {'adstock+Hill':>13s}")
rows = {}
for c in chans:
    p = gt["channels"][c]
    imp = df[f"{c}_impressions"].values
    spend = df[f"{c}_spend"].values
    ad = geometric_adstock(imp, p["theta"])
    sat = hill_saturation(ad, p["half_sat"], p["slope"])
    true_contrib = p["beta"] * sat   # this IS the true contribution series

    r_spend = np.corrcoef(spend, true_contrib)[0,1]
    r_imp   = np.corrcoef(imp, true_contrib)[0,1]
    r_ad    = np.corrcoef(ad, true_contrib)[0,1]
    r_sat   = np.corrcoef(sat, true_contrib)[0,1]
    rows[c] = (r_spend, r_imp, r_ad, r_sat, spend, imp, ad, sat, true_contrib)
    print(f"{c:10s} {r_spend:10.3f} {r_imp:10.3f} {r_ad:13.3f} {r_sat:13.3f}")

# Visualize TV (hardest: high adstock + flighting) through the pipeline
c = "tv"; r = rows[c]
spend, imp, ad, sat, true_contrib = r[4], r[5], r[6], r[7], r[8]
fig, axes = plt.subplots(2,2, figsize=(13,7))
w = df.week
axes[0,0].plot(w, spend, color="#bbb"); axes[0,0].set_title(f"{c}: 1) raw spend  (r with truth={r[0]:+.2f})")
axes[0,1].plot(w, imp/1000, color="#888"); axes[0,1].set_title(f"2) impressions (000s)  (r={r[1]:+.2f})")
axes[1,0].plot(w, ad/1000, color=colors[c], alpha=0.7); axes[1,0].set_title(f"3) + adstock θ={gt['channels'][c]['theta']}  (r={r[2]:+.2f})")
ax = axes[1,1]
ax.plot(w, sat, color=colors[c], label="adstock+Hill (scaled 0-1)")
ax2 = ax.twinx(); ax2.plot(w, true_contrib, color="black", ls="--", alpha=0.6, label="true contribution")
ax.set_title(f"4) + Hill saturation  (r={r[3]:+.2f})")
ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
for a in axes.flat: a.grid(alpha=0.3)
plt.suptitle("TV through the transform pipeline: raw spend is a poor proxy; adstock+Hill nails the true contribution", y=1.02)
plt.tight_layout(); plt.savefig("stage3_pipeline.png", dpi=110, bbox_inches="tight")
print("\nsaved stage3_pipeline.png")

# Summary bar: correlation improvement raw-spend -> transformed
fig2, ax = plt.subplots(figsize=(10,4.5))
x = np.arange(len(chans)); wbar=0.35
ax.bar(x-wbar/2, [rows[c][0] for c in chans], wbar, label="raw spend", color="#bbb")
ax.bar(x+wbar/2, [rows[c][3] for c in chans], wbar, label="adstock+Hill (correct form)",
       color=[colors[c] for c in chans])
ax.set_xticks(x); ax.set_xticklabels(chans); ax.set_ylabel("corr with TRUE contribution")
ax.set_title("Why functional form matters: raw spend vs correctly-transformed media")
ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0,1.05)
plt.tight_layout(); plt.savefig("stage3_corr_gain.png", dpi=110, bbox_inches="tight")
print("saved stage3_corr_gain.png")
