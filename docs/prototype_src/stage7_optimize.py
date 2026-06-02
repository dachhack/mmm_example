"""
Stage 7: BUDGET OPTIMIZATION under uncertainty (anchored model).
1) Optimize point-estimate allocation to maximize expected conversions at fixed total budget.
2) Show the 'equalize marginal ROI' principle before/after.
3) Robust view: optimize across posterior draws -> distribution of recommended allocations,
   separating CONFIDENT moves from TEST-FIRST moves.
"""
import pandas as pd, numpy as np, json, pickle, matplotlib.pyplot as plt
from scipy.optimize import minimize
from transforms import geometric_adstock, hill_saturation
chans=["tv","search","social","affiliate","brand"]
colors={"tv":"#d62728","search":"#1f77b4","social":"#2ca02c","affiliate":"#ff7f0e","brand":"#9467bd"}
df=pd.read_csv("draftzone_mmm_data.csv", parse_dates=["week"])
idata=pickle.load(open("bayes_idata_anchored.pkl","rb"))
post=idata.posterior.to_dataset(); draws=post.stack(s=("chain","draw"))
S=draws.sizes["s"]

imp={c:df[f"{c}_impressions"].values.astype(float) for c in chans}
imp_mean={c:imp[c].mean() for c in chans}
imp_s={c:imp[c]/imp_mean[c] for c in chans}
spend={c:df[f"{c}_spend"].values.astype(float) for c in chans}
cur_spend={c:spend[c].sum() for c in chans}
total_budget=sum(cur_spend.values())
# impressions-per-dollar (avg), to convert a spend allocation into impressions scale
ipd={c: imp[c].sum()/cur_spend[c] for c in chans}

def total_conv_for_alloc(alloc, params):
    """alloc: dict channel->total spend. params: dict channel->(theta,slope,hs,beta).
    Convert spend to a scaled-impression series proportional to current pattern, recompute conv."""
    tot=0.0
    for c in chans:
        scale=alloc[c]/cur_spend[c] if cur_spend[c]>0 else 0
        impr_series=imp[c]*scale          # scale the whole impression pattern
        s_series=impr_series/imp_mean[c]  # back to scaled units the params expect
        th,sl,hs,be=params[c]
        ad=geometric_adstock(s_series,th,normalize=True)
        sat=hill_saturation(ad,hs,sl)
        tot+=(be*sat).sum()
    return tot

def optimize_alloc(params, budget, bounds_frac=(0.2,3.0)):
    """maximize conv subject to sum(alloc)=budget, each channel within bounds_frac*current."""
    x0=np.array([cur_spend[c] for c in chans])
    def negconv(x):
        alloc={c:max(x[i],1) for i,c in enumerate(chans)}
        return -total_conv_for_alloc(alloc,params)
    cons=[{"type":"eq","fun":lambda x: x.sum()-budget}]
    bnds=[(cur_spend[c]*bounds_frac[0], cur_spend[c]*bounds_frac[1]) for c in chans]
    res=minimize(negconv,x0,method="SLSQP",bounds=bnds,constraints=cons,
                 options=dict(maxiter=300,ftol=1e-6))
    return {c:res.x[i] for i,c in enumerate(chans)}

# ---- point-estimate params (posterior mean) ----
pm_params={c:(float(draws[f"theta_{c}"].mean()),float(draws[f"slope_{c}"].mean()),
              float(draws[f"hs_{c}"].mean()),float(draws[f"beta_{c}"].mean())) for c in chans}
cur_conv=total_conv_for_alloc(cur_spend,pm_params)
opt=optimize_alloc(pm_params,total_budget)
opt_conv=total_conv_for_alloc(opt,pm_params)
print("="*72)
print(f"Total budget (fixed): ${total_budget:,.0f}")
print(f"Current total conversions (model): {cur_conv:,.0f}")
print(f"Optimized total conversions:       {opt_conv:,.0f}  (+{100*(opt_conv/cur_conv-1):.1f}%)")
print(f"\n{'channel':10s} {'current $':>13s} {'optimized $':>13s} {'change':>9s}")
for c in chans:
    print(f"{c:10s} {cur_spend[c]:13,.0f} {opt[c]:13,.0f} {100*(opt[c]/cur_spend[c]-1):+8.0f}%")

# ---- ROBUST: optimize per posterior draw -> distribution of recommended allocations ----
sel=np.linspace(0,S-1,120).astype(int)
rec=[ {c:[] for c in chans} for _ in range(1)][0]
for i in sel:
    pr={c:(float(draws[f"theta_{c}"][i]),float(draws[f"slope_{c}"][i]),
           float(draws[f"hs_{c}"][i]),float(draws[f"beta_{c}"][i])) for c in chans}
    a=optimize_alloc(pr,total_budget)
    for c in chans: rec[c].append(a[c]/cur_spend[c]-1)  # fractional change recommended

print("\nROBUST allocation recommendation (across posterior draws):")
print(f"{'channel':10s} {'median Δ':>9s} {'89% CI of Δ':>22s} {'verdict':>12s}")
verdicts={}
for c in chans:
    arr=np.array(rec[c])*100
    md=np.median(arr); lo=np.percentile(arr,5.5); hi=np.percentile(arr,94.5)
    # confident if CI doesn't straddle 0 by much
    if lo>5: v="INCREASE"
    elif hi<-5: v="DECREASE"
    else: v="TEST FIRST"
    verdicts[c]=v
    print(f"{c:10s} {md:+8.0f}% [{lo:+6.0f}%,{hi:+6.0f}%] {v:>12s}")

# ---- charts ----
fig,axes=plt.subplots(1,2,figsize=(13,5))
ax=axes[0]
x=np.arange(len(chans)); w=0.38
ax.bar(x-w/2,[cur_spend[c]/1e6 for c in chans],w,label="current",color="#bbb")
ax.bar(x+w/2,[opt[c]/1e6 for c in chans],w,label="optimized",
       color=[colors[c] for c in chans])
ax.set_xticks(x); ax.set_xticklabels(chans); ax.set_ylabel("spend ($M)")
ax.set_title(f"Point-estimate optimal reallocation\n(+{100*(opt_conv/cur_conv-1):.1f}% conversions, same budget)")
ax.legend(); ax.grid(alpha=0.3,axis="y")

ax=axes[1]
meds=[np.median(np.array(rec[c])*100) for c in chans]
los=[np.percentile(np.array(rec[c])*100,5.5) for c in chans]
his=[np.percentile(np.array(rec[c])*100,94.5) for c in chans]
err=np.array([[meds[i]-los[i] for i in range(5)],[his[i]-meds[i] for i in range(5)]])
vcolor={"INCREASE":"#2ca02c","DECREASE":"#d62728","TEST FIRST":"#888"}
bars=ax.barh(x,meds,color=[vcolor[verdicts[c]] for c in chans],alpha=0.85)
ax.errorbar(meds,x,xerr=err,fmt="none",ecolor="black",capsize=4)
ax.axvline(0,color="#333",lw=1)
ax.set_yticks(x); ax.set_yticklabels(chans); ax.set_xlabel("recommended spend change (%)")
ax.set_title("ROBUST recommendation across uncertainty\n(green=confident incr, red=confident decr, grey=test first)")
for i,c in enumerate(chans):
    ax.text(meds[i], x[i]+0.28, verdicts[c], fontsize=7, ha="center")
ax.grid(alpha=0.3,axis="x")
plt.tight_layout(); plt.savefig("stage7_optimize.png",dpi=110,bbox_inches="tight")
print("\nsaved stage7_optimize.png")

json.dump(dict(total_budget=total_budget,cur_conv=cur_conv,opt_conv=opt_conv,
               optimized={c:opt[c] for c in chans},verdicts=verdicts),
          open("optimization_result.json","w"),indent=2)
