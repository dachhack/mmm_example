"""Throwaway visual intuition for adstock (decay) and saturation.
Not part of the model pipeline -- just to cement the two core transforms."""
import numpy as np
import matplotlib.pyplot as plt

# ---------- ADSTOCK ----------
def adstock(x, theta):
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + theta * out[t-1]
    return out

weeks = np.arange(20)
spend = np.zeros(20)
spend[3] = 100.0          # a single one-week burst of spend in week 3

ad_low  = adstock(spend, 0.10)   # paid search-like: almost no carryover
ad_high = adstock(spend, 0.75)   # TV-like: long carryover

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.bar(weeks, spend, color="#bbb", label="Raw spend (one burst, wk 3)")
ax.plot(weeks, ad_low,  marker="o", color="#1f77b4", label="Adstocked  θ=0.10 (search)")
ax.plot(weeks, ad_high, marker="o", color="#d62728", label="Adstocked  θ=0.75 (TV)")
ax.set_title("DECAY / ADSTOCK: how a single $100 burst echoes forward")
ax.set_xlabel("week"); ax.set_ylabel("effective media pressure")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# annotate the "total weight" intuition 1/(1-theta)
ax.text(0.02, 0.78, "total carryover weight = 1/(1-θ)\n θ=0.10 → 1.11×\n θ=0.75 → 4.0×",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", fc="#fff3cd", ec="#ccc"))

# ---------- SATURATION (Hill) ----------
def hill(x, half_sat, slope):
    return x**slope / (x**slope + half_sat**slope)

spend_axis = np.linspace(0, 300, 400)
ax = axes[1]
# vary half_sat
ax.plot(spend_axis, hill(spend_axis, half_sat=50,  slope=2.0), color="#2ca02c",
        label="half_sat=50, slope=2.0 (cheap, caps early)")
ax.plot(spend_axis, hill(spend_axis, half_sat=150, slope=2.0), color="#ff7f0e",
        label="half_sat=150, slope=2.0 (caps late, e.g. TV)")
# vary slope -> S-curve vs gentle
ax.plot(spend_axis, hill(spend_axis, half_sat=100, slope=4.0), color="#9467bd",
        ls="--", label="half_sat=100, slope=4.0 (sharp S-curve / threshold)")
ax.plot(spend_axis, hill(spend_axis, half_sat=100, slope=1.0), color="#8c564b",
        ls="--", label="half_sat=100, slope=1.0 (gentle, immediate DR)")
ax.axhline(0.5, color="#999", ls=":", lw=1)
ax.text(2, 0.52, "half of max effect", fontsize=8, color="#666")
ax.set_title("SATURATION (Hill): response per channel vs. spend")
ax.set_xlabel("spend in channel"); ax.set_ylabel("fraction of channel's max effect")
ax.set_ylim(0, 1.02); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("concepts.png", dpi=120, bbox_inches="tight")
print("saved concepts.png")

# A third plot: the two combined, in order, on a realistic wiggly spend series
fig2, axes2 = plt.subplots(1, 1, figsize=(13, 4.5))
rng = np.random.default_rng(7)
weeks2 = np.arange(60)
raw = np.clip(40 + 30*np.sin(weeks2/6) + rng.normal(0, 12, 60), 0, None)
raw[10:13] += 80   # a flight
step1 = adstock(raw, 0.6)                       # adstock first
# saturate the adstocked series (scale so it's visible alongside)
step2_frac = hill(step1, half_sat=np.median(step1)*1.2, slope=1.8)
step2 = step2_frac * step1.max()                # rescale to compare shapes
axes2.plot(weeks2, raw,  color="#bbb", label="1. raw spend")
axes2.plot(weeks2, step1, color="#1f77b4", label="2. after adstock (θ=0.6) — smoothed & shifted")
axes2.plot(weeks2, step2, color="#d62728", label="3. after saturation — peaks compressed toward ceiling")
axes2.set_title("ORDER OF OPERATIONS: raw → adstock → saturation")
axes2.set_xlabel("week"); axes2.set_ylabel("(rescaled for comparison)")
axes2.legend(fontsize=9); axes2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("concepts_pipeline.png", dpi=120, bbox_inches="tight")
print("saved concepts_pipeline.png")
