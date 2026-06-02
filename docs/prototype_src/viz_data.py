import pandas as pd, numpy as np, matplotlib.pyplot as plt
df = pd.read_csv("draftzone_mmm_data.csv", parse_dates=["week"])
chans = ["tv","search","social","affiliate","brand"]
colors = {"tv":"#d62728","search":"#1f77b4","social":"#2ca02c","affiliate":"#ff7f0e","brand":"#9467bd"}

fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

# (1) conversions + true seasonality
ax = axes[0]
ax.plot(df.week, df.conversions, color="#222", lw=1.3, label="conversions (target)")
ax.plot(df.week, 1200 + df.trend_true + df.seasonality_true, color="#2ca02c", ls="--",
        lw=1.2, label="true baseline+trend+season (hidden)")
ax.fill_between(df.week, 1200, 1200+df.trend_true, color="#2ca02c", alpha=0.06)
ax.set_ylabel("conversions/wk"); ax.legend(fontsize=8, loc="upper left")
ax.set_title("DraftZone — weekly conversions, with hidden organic floor")
ax.grid(alpha=0.3)

# (2) spend by channel
ax = axes[1]
for c in chans:
    ax.plot(df.week, df[f"{c}_spend"], color=colors[c], lw=1, label=c)
ax.set_ylabel("spend ($/wk)"); ax.legend(fontsize=8, ncol=5, loc="upper left")
ax.set_title("Media spend by channel (note TV & affiliate flighting / dark weeks)")
ax.grid(alpha=0.3)

# (3) impressions by channel (what actually drives conversions)
ax = axes[2]
for c in chans:
    ax.plot(df.week, df[f"{c}_impressions"]/1000, color=colors[c], lw=1, label=c)
ax.set_ylabel("impressions (000s/wk)"); ax.legend(fontsize=8, ncol=5, loc="upper left")
ax.set_title("Impressions by channel (the true causal driver)")
ax.set_xlabel("week"); ax.grid(alpha=0.3)

plt.tight_layout(); plt.savefig("data_overview.png", dpi=110, bbox_inches="tight")
print("saved data_overview.png")
