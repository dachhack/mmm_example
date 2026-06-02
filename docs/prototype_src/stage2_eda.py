"""Stage 2a: EDA — explore as an analyst with NO answer key."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
df = pd.read_csv("draftzone_mmm_data.csv", parse_dates=["week"])
chans = ["tv","search","social","affiliate","brand"]
colors = {"tv":"#d62728","search":"#1f77b4","social":"#2ca02c","affiliate":"#ff7f0e","brand":"#9467bd"}

# --- correlation of each spend with conversions (the naive "which channel works?" view) ---
print("Naive correlation of SPEND with conversions (what a spreadsheet jockey sees):")
for c in chans:
    r = np.corrcoef(df[f"{c}_spend"], df.conversions)[0,1]
    print(f"  {c:10s}: {r:+.3f}")

print("\nNaive 'cost per conversion' if you credited ALL conversions to one channel:")
for c in chans:
    print(f"  {c:10s}: ${df[f'{c}_spend'].sum()/df.conversions.sum():.2f}  (meaningless but commonly computed)")

# --- visuals: spend vs conversions scatter (raw, no transforms) ---
fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
for ax, c in zip(axes, chans):
    ax.scatter(df[f"{c}_spend"], df.conversions, s=14, alpha=0.5, color=colors[c])
    z = np.polyfit(df[f"{c}_spend"], df.conversions, 1)
    xs = np.linspace(df[f"{c}_spend"].min(), df[f"{c}_spend"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), color="black", lw=1)
    r = np.corrcoef(df[f"{c}_spend"], df.conversions)[0,1]
    ax.set_title(f"{c}\nr={r:+.2f}"); ax.set_xlabel("spend"); ax.grid(alpha=0.3)
axes[0].set_ylabel("conversions")
plt.suptitle("Raw spend vs conversions — upward slopes everywhere (but how much is real?)", y=1.06)
plt.tight_layout(); plt.savefig("eda_scatter.png", dpi=110, bbox_inches="tight")
print("\nsaved eda_scatter.png")
