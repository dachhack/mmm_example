import numpy as np, json, matplotlib.pyplot as plt
ts=np.load("geo_treat_series.npy"); mask=np.load("geo_treat_mask.npy")
anchor=json.load(open("experiment_anchor.json"))
week=np.arange(ts.shape[1])
tmean=ts[mask].mean(0); cmean=ts[~mask].mean(0)
diff=tmean-cmean
fig,axes=plt.subplots(1,2,figsize=(13,4.6))
ax=axes[0]
ax.plot(week,tmean,"o-",color="#2ca02c",ms=3,label="Treatment (avg)")
ax.plot(week,cmean,"o-",color="#1f77b4",ms=3,label="Control (avg)")
ax.axvline(6,color="#999",ls=":"); ax.text(6.1,ax.get_ylim()[0]+5,"campaign on",fontsize=8)
ax.set_title("Treatment vs Control markets\n(both ride same season; gap opens at campaign)")
ax.set_xlabel("week"); ax.set_ylabel("avg conversions/market"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax=axes[1]
ax.plot(week,diff,"o-",color="#d62728",ms=4,label="Treatment − Control")
ax.axhline(anchor["true_increment"],color="#2ca02c",ls="--",label=f"true lift={anchor['true_increment']:.1f}")
ax.axhline(0,color="#999",lw=0.8); ax.axvspan(0,6,color="#eee",label="pre-period")
ax.fill_between([6,17],anchor["did_ci"][0],anchor["did_ci"][1],color="#d62728",alpha=0.15,
                label=f"DiD 89% CI")
ax.set_title(f"Causal lift recovered: {anchor['did_per_market_week']:.1f} (truth {anchor['true_increment']:.1f})")
ax.set_xlabel("week"); ax.set_ylabel("lift/market"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("stage8_experiment.png",dpi=110,bbox_inches="tight")
print("saved stage8_experiment.png")
