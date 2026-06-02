"""
Stage 6: conversion -> revenue valuation and ROI, on the ANCHORED (experiment-calibrated) model.
Blended LTV with uncertainty. Computes per-channel:
  - revenue contribution = conversions_contrib * LTV
  - ROI = revenue / spend
  - marginal ROI = d(revenue)/d(spend) at current spend (depends on saturation slope/position)
All with posterior + LTV uncertainty propagated.
"""
import pandas as pd, numpy as np, json, pickle, matplotlib.pyplot as plt
from transforms import geometric_adstock, hill_saturation
chans=["tv","search","social","affiliate","brand"]
colors={"tv":"#d62728","search":"#1f77b4","social":"#2ca02c","affiliate":"#ff7f0e","brand":"#9467bd"}
df=pd.read_csv("draftzone_mmm_data.csv", parse_dates=["week"])
gt=json.load(open("ground_truth.json"))
idata=pickle.load(open("bayes_idata_anchored.pkl","rb"))
post=idata.posterior.to_dataset(); draws=post.stack(s=("chain","draw"))
S=draws.sizes["s"]; sel=np.linspace(0,S-1,min(600,S)).astype(int)

imp={c:df[f"{c}_impressions"].values.astype(float) for c in chans}
imp_s={c:imp[c]/imp[c].mean() for c in chans}
spend={c:df[f"{c}_spend"].values.astype(float) for c in chans}
mean_spend={c:spend[c].mean() for c in chans}
total_spend={c:spend[c].sum() for c in chans}

# LTV model: blended, with uncertainty
LTV_MU=220.0; LTV_LO=180.0; LTV_HI=260.0
LTV_SD=(LTV_HI-LTV_LO)/(2*1.6)   # ~89% within [180,260]
rng=np.random.default_rng(6)

# For each posterior draw: compute social-style contribution series for each channel,
# then avg weekly conversions, total conversions over period, revenue, ROI, marginal ROI.
def channel_metrics():
    out={c:dict(conv_avg=[],roi=[],mroi=[]) for c in chans}
    n_weeks=len(df)
    for j,i in enumerate(sel):
        ltv=rng.normal(LTV_MU,LTV_SD)
        for c in chans:
            th=float(draws[f"theta_{c}"][i]); sl=float(draws[f"slope_{c}"][i])
            hs=float(draws[f"hs_{c}"][i]); be=float(draws[f"beta_{c}"][i])
            ad=geometric_adstock(imp_s[c],th,normalize=True)
            sat=hill_saturation(ad,hs,sl)
            contrib=be*sat                      # conversions/wk attributable
            conv_total=contrib.sum()
            revenue=conv_total*ltv
            roi=revenue/total_spend[c]
            # marginal ROI: bump impressions +1% (proxy for +spend), recompute, finite diff
            bump=1.01
            ad2=geometric_adstock(imp_s[c]*bump,th,normalize=True)
            sat2=hill_saturation(ad2,hs,sl)
            dconv=(be*sat2 - contrib).sum()
            dspend=total_spend[c]*0.01
            mroi=(dconv*ltv)/dspend if dspend>0 else np.nan
            out[c]["conv_avg"].append(contrib.mean())
            out[c]["roi"].append(roi)
            out[c]["mroi"].append(mroi)
    return out

M=channel_metrics()
def pct(a,p): return np.percentile(a,p)
print("="*86)
print(f"Blended LTV = ${LTV_MU:.0f}  (89% range ${LTV_LO:.0f}-${LTV_HI:.0f})")
print(f"\n{'channel':10s} {'avg conv/wk':>12s} {'total $spend':>13s} {'ROI':>16s} {'marginal ROI':>18s}")
print("-"*86)
summary={}
for c in chans:
    roi=np.array(M[c]["roi"]); mroi=np.array(M[c]["mroi"]); conv=np.array(M[c]["conv_avg"])
    summary[c]=dict(roi=(roi.mean(),pct(roi,5.5),pct(roi,94.5)),
                    mroi=(mroi.mean(),pct(mroi,5.5),pct(mroi,94.5)),
                    conv=conv.mean())
    print(f"{c:10s} {conv.mean():12.0f} {total_spend[c]:13,.0f} "
          f"{roi.mean():6.2f} [{pct(roi,5.5):.2f},{pct(roi,94.5):.2f}] "
          f"{mroi.mean():7.2f} [{pct(mroi,5.5):.2f},{pct(mroi,94.5):.2f}]")
print("-"*86)
print("ROI = total revenue / total spend (average return). mROI = return on the NEXT dollar.")
print("Decision rule: shift budget toward HIGH marginal ROI, away from mROI<1 (next dollar loses money).")

# blended overall
tot_rev=sum(summary[c]["conv"]*len(df)*LTV_MU for c in chans)
tot_spend=sum(total_spend[c] for c in chans)
print(f"\nBlended media ROI (all channels): {tot_rev/tot_spend:.2f}")

json.dump({c:{k:(list(map(float,v)) if isinstance(v,tuple) else float(v))
              for k,v in summary[c].items()} for c in chans},
          open("roi_summary.json","w"),indent=2)

# ---- chart: ROI vs marginal ROI with uncertainty ----
fig,axes=plt.subplots(1,2,figsize=(13,5))
ax=axes[0]
x=np.arange(len(chans))
vals=[summary[c]["roi"][0] for c in chans]
err=np.array([[summary[c]["roi"][0]-summary[c]["roi"][1] for c in chans],
              [summary[c]["roi"][2]-summary[c]["roi"][0] for c in chans]])
ax.bar(x,vals,color=[colors[c] for c in chans],alpha=0.85)
ax.errorbar(x,vals,yerr=err,fmt="none",ecolor="black",capsize=5)
ax.axhline(1,color="#d62728",ls="--",label="ROI=1 (break-even)")
ax.set_xticks(x); ax.set_xticklabels(chans); ax.set_ylabel("ROI ($ rev / $ spend)")
ax.set_title("Average ROI by channel (89% CI)"); ax.legend(); ax.grid(alpha=0.3,axis="y")
ax=axes[1]
valsm=[summary[c]["mroi"][0] for c in chans]
errm=np.array([[summary[c]["mroi"][0]-summary[c]["mroi"][1] for c in chans],
               [summary[c]["mroi"][2]-summary[c]["mroi"][0] for c in chans]])
ax.bar(x,valsm,color=[colors[c] for c in chans],alpha=0.85)
ax.errorbar(x,valsm,yerr=errm,fmt="none",ecolor="black",capsize=5)
ax.axhline(1,color="#d62728",ls="--",label="mROI=1 (next $ break-even)")
ax.set_xticks(x); ax.set_xticklabels(chans); ax.set_ylabel("marginal ROI (next $)")
ax.set_title("MARGINAL ROI by channel (89% CI)\n— the number that drives budget shifts")
ax.legend(); ax.grid(alpha=0.3,axis="y")
plt.tight_layout(); plt.savefig("stage6_roi.png",dpi=110,bbox_inches="tight")
print("\nsaved stage6_roi.png")
