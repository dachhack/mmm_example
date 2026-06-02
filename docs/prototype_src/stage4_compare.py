import pickle, json, numpy as np, arviz as az, matplotlib.pyplot as plt
chans=["tv","search","social","affiliate","brand"]
colors={"tv":"#d62728","search":"#1f77b4","social":"#2ca02c","affiliate":"#ff7f0e","brand":"#9467bd"}
gt=json.load(open("ground_truth.json"))
freq=json.load(open("fit_frequentist.json"))
idata=pickle.load(open("bayes_idata.pkl","rb"))
post=idata.posterior.to_dataset()

# Bayesian implied avg contribution per channel: beta_c * mean(sat_c) — recompute sat from posterior draws is heavy;
# approximate contribution via posterior beta * (true-ish mean sat fraction). Simpler: use beta share proxy.
# For a fair contribution comparison, recompute transformed media at posterior MEAN params.
import pandas as pd
from transforms import geometric_adstock, hill_saturation
df=pd.read_csv("draftzone_mmm_data.csv")
imp={c:df[f"{c}_impressions"].values.astype(float) for c in chans}
imp_s={c:imp[c]/imp[c].mean() for c in chans}
def pm_mean(v): return float(post[v].mean())
bayes_contrib={}
for c in chans:
    th=pm_mean(f"theta_{c}"); sl=pm_mean(f"slope_{c}"); hs=pm_mean(f"hs_{c}"); be=pm_mean(f"beta_{c}")
    ad=geometric_adstock(imp_s[c],th,normalize=True)
    sat=hill_saturation(ad,hs,sl)
    bayes_contrib[c]=be*sat.mean()

# also get Bayesian contribution uncertainty band via draws (subsample)
draws=post.stack(s=("chain","draw"))
nd=min(300, draws.dims["s"])
sel=np.linspace(0,draws.dims["s"]-1,nd).astype(int)
bayes_contrib_draws={c:[] for c in chans}
for i in sel:
    for c in chans:
        th=float(draws[f"theta_{c}"][i]); sl=float(draws[f"slope_{c}"][i])
        hs=float(draws[f"hs_{c}"][i]); be=float(draws[f"beta_{c}"][i])
        ad=geometric_adstock(imp_s[c],th,normalize=True)
        sat=hill_saturation(ad,hs,sl)
        bayes_contrib_draws[c].append(be*sat.mean())

fig,ax=plt.subplots(figsize=(11,5.5))
x=np.arange(len(chans)); w=0.25
true_v=[gt["avg_contribution_decomposition"]["media_"+c] for c in chans]
freq_v=[freq["params"][c]["avg_contrib"] for c in chans]
bay_v =[bayes_contrib[c] for c in chans]
# clip freq social for display, annotate true value
freq_disp=[min(v, 600) for v in freq_v]
ax.bar(x-w, true_v, w, label="TRUE", color="#2ca02c")
b2=ax.bar(x, freq_disp, w, label="Frequentist", color="#d62728")
# error bars for bayesian from draws
bay_lo=[np.percentile(bayes_contrib_draws[c],5.5) for c in chans]
bay_hi=[np.percentile(bayes_contrib_draws[c],94.5) for c in chans]
err=np.array([[bay_v[i]-bay_lo[i] for i in range(5)],[bay_hi[i]-bay_v[i] for i in range(5)]])
ax.bar(x+w, bay_v, w, label="Bayesian (89% CI)", color="#1f77b4")
ax.errorbar(x+w, bay_v, yerr=err, fmt="none", ecolor="black", capsize=4, lw=1.2)
# annotate clipped freq social
for i,c in enumerate(chans):
    if freq_v[i]>600:
        ax.text(x[i], 605, f"{freq_v[i]:.0f}→", ha="center", fontsize=8, color="#d62728", rotation=0)
ax.set_xticks(x); ax.set_xticklabels(chans); ax.set_ylabel("avg conversions/wk")
ax.set_title("Stage 4: channel contribution — TRUE vs Frequentist vs Bayesian\n(Bayesian shows honest uncertainty; frequentist social=16512 off-chart)")
ax.legend(); ax.grid(alpha=0.3,axis="y")
plt.tight_layout(); plt.savefig("stage4_compare.png",dpi=110,bbox_inches="tight")
print("saved stage4_compare.png")
for c in chans:
    print(f"{c:10s} true={true_v[chans.index(c)]:7.1f}  freq={freq_v[chans.index(c)]:9.1f}  bayes={bayes_contrib[c]:7.1f}  [{bay_lo[chans.index(c)]:.0f},{bay_hi[chans.index(c)]:.0f}]")
