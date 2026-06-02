"""Stage 5: grade the Bayesian MMM against the sealed answer key.
Posterior predictive fit, full decomposition vs truth, and CALIBRATION:
do the stated credible intervals contain the truth at the claimed rate?"""
import pickle, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from transforms import geometric_adstock, hill_saturation
chans=["tv","search","social","affiliate","brand"]
colors={"tv":"#d62728","search":"#1f77b4","social":"#2ca02c","affiliate":"#ff7f0e","brand":"#9467bd"}
gt=json.load(open("ground_truth.json"))
idata=pickle.load(open("bayes_idata.pkl","rb"))
post=idata.posterior.to_dataset()
df=pd.read_csv("draftzone_mmm_data.csv", parse_dates=["week"])
T=len(df); y=df.conversions.values.astype(float); t=np.arange(T)
imp={c:df[f"{c}_impressions"].values.astype(float) for c in chans}
imp_s={c:imp[c]/imp[c].mean() for c in chans}

# rebuild control matrix exactly as in fit
trend=t/T
fourier=np.column_stack([f(2*np.pi*k*t/52) for k in range(1,4) for f in (np.sin,np.cos)])
promo=df.promo_flag.values.astype(float)
price=(df.price_index.values-100)/10
comp=(df.competitor_pressure.values-df.competitor_pressure.mean())/df.competitor_pressure.std()
holiday=df.holiday_flag.values.astype(float)
Xc=np.column_stack([trend,fourier,promo,price,comp,holiday]).astype(float)

draws=post.stack(s=("chain","draw"))
S=draws.sizes["s"]
sel=np.linspace(0,S-1,min(800,S)).astype(int)

# reconstruct full posterior predictive mu and per-channel contributions per draw
mu_draws=np.zeros((len(sel),T))
chan_contrib_draws={c:np.zeros((len(sel),T)) for c in chans}
ctrl_contrib_draws=np.zeros((len(sel),T))
base_draws=np.zeros(len(sel))
for j,i in enumerate(sel):
    base=float(draws["baseline"][i]); base_draws[j]=base
    cc=draws["ctrl_coef"][:,i].values
    ctrl=Xc@cc; ctrl_contrib_draws[j]=ctrl
    m=base+ctrl
    for c in chans:
        th=float(draws[f"theta_{c}"][i]); sl=float(draws[f"slope_{c}"][i])
        hs=float(draws[f"hs_{c}"][i]); be=float(draws[f"beta_{c}"][i])
        ad=geometric_adstock(imp_s[c],th,normalize=True)
        sat=hill_saturation(ad,hs,sl); contrib=be*sat
        chan_contrib_draws[c][j]=contrib; m=m+contrib
    mu_draws[j]=m

mu_mean=mu_draws.mean(0)
mu_lo=np.percentile(mu_draws,5.5,0); mu_hi=np.percentile(mu_draws,94.5,0)
r2=1-np.sum((y-mu_mean)**2)/np.sum((y-y.mean())**2)
mape=np.mean(np.abs((y-mu_mean)/y))*100
print(f"Posterior-predictive fit: R²={r2:.3f}  MAPE={mape:.1f}%")
# coverage of the observed y by 89% PP interval
cover_y=np.mean((y>=mu_lo)&(y<=mu_hi))
print(f"89% PP interval covers {100*cover_y:.0f}% of observed weeks (want ~89%)")

# ---- decomposition vs truth ----
print("\nAvg weekly contribution: TRUE vs POSTERIOR MEAN [89% CI]  -- does CI contain truth?")
gtd=gt["avg_contribution_decomposition"]
truth_media_total=sum(gtd["media_"+c] for c in chans)
hits=0
rows=[]
for c in chans:
    arr=chan_contrib_draws[c].mean(1)
    m=arr.mean(); lo=np.percentile(arr,5.5); hi=np.percentile(arr,94.5)
    tru=gtd["media_"+c]; hit=lo<=tru<=hi; hits+=hit
    rows.append((c,tru,m,lo,hi,hit))
    print(f"  {c:10s} true={tru:7.1f}  est={m:7.1f}  [{lo:6.1f},{hi:6.1f}]  {'HIT' if hit else 'MISS'}")
print(f"  channels whose 89% CI contains truth: {hits}/5")

# baseline+controls truth (organic-ish): baseline+trend+season+promo+price+comp+holiday
truth_nonmedia = (gtd["baseline"]+gtd["trend"]+gtd["seasonality"]+gtd["promo"]
                  +gtd["price"]+gtd["competitor"]+gtd["holiday"])
est_nonmedia=(base_draws.mean()+ctrl_contrib_draws.mean())
print(f"\nNon-media (baseline+controls): true={truth_nonmedia:.0f}  est={est_nonmedia:.0f}")
print(f"Total media: true={truth_media_total:.0f}  est={sum(r[2] for r in rows):.0f}")

# ---- theta calibration table ----
print("\nAdstock theta recovery:")
for c in chans:
    arr=draws[f"theta_{c}"].values
    m=arr.mean(); lo=np.percentile(arr,5.5); hi=np.percentile(arr,94.5)
    tru=gt["channels"][c]["theta"]; hit=lo<=tru<=hi
    print(f"  {c:10s} true={tru:.2f}  est={m:.2f}  [{lo:.2f},{hi:.2f}]  {'HIT' if hit else 'MISS'}")

# ========== FIGURE 1: posterior predictive fit ==========
fig,ax=plt.subplots(figsize=(13,4.5))
ax.fill_between(df.week,mu_lo,mu_hi,color="#1f77b4",alpha=0.25,label="89% predictive")
ax.plot(df.week,mu_mean,color="#1f77b4",lw=1.3,label="posterior mean")
ax.plot(df.week,y,color="#222",lw=1,alpha=0.8,label="observed")
ax.set_title(f"Stage 5: posterior predictive fit  (R²={r2:.3f}, MAPE={mape:.1f}%)")
ax.set_ylabel("conversions/wk"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("stage5_ppc.png",dpi=110,bbox_inches="tight")

# ========== FIGURE 2: stacked decomposition (posterior mean) vs truth lines ==========
fig,ax=plt.subplots(figsize=(13,5))
bottom=np.zeros(T)
base_series=base_draws.mean()*np.ones(T)
ax.fill_between(df.week,0,base_series,color="#cccccc",label="baseline(+ctrl mean)")
bottom=base_series.copy()
# add controls mean as part of nonmedia band
ctrl_mean_series=ctrl_contrib_draws.mean(0)
ax.fill_between(df.week,bottom,bottom+ctrl_mean_series,color="#999999",alpha=0.6,label="controls")
bottom=bottom+ctrl_mean_series
for c in chans:
    cm=chan_contrib_draws[c].mean(0)
    ax.fill_between(df.week,bottom,bottom+cm,color=colors[c],alpha=0.8,label=c)
    bottom=bottom+cm
ax.plot(df.week,y,color="black",lw=1,label="observed conversions")
ax.set_title("Stage 5: model decomposition of conversions (posterior mean)")
ax.set_ylabel("conversions/wk"); ax.legend(fontsize=8,ncol=4,loc="upper left"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("stage5_decomp.png",dpi=110,bbox_inches="tight")

# ========== FIGURE 3: calibration — recovered vs truth with CIs ==========
fig,axes=plt.subplots(1,2,figsize=(13,4.8))
ax=axes[0]
for i,c in enumerate(chans):
    arr=chan_contrib_draws[c].mean(1)
    m=arr.mean(); lo=np.percentile(arr,5.5); hi=np.percentile(arr,94.5); tru=gtd["media_"+c]
    ax.errorbar(tru,m,yerr=[[m-lo],[hi-m]],fmt="o",color=colors[c],capsize=4,label=c)
mx=max(gtd["media_"+c] for c in chans)*1.2
ax.plot([0,mx],[0,mx],"k--",alpha=0.5,label="perfect")
ax.set_xlabel("TRUE contribution"); ax.set_ylabel("estimated (89% CI)")
ax.set_title("Channel contribution: estimate vs truth"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax=axes[1]
for i,c in enumerate(chans):
    arr=draws[f"theta_{c}"].values
    m=arr.mean(); lo=np.percentile(arr,5.5); hi=np.percentile(arr,94.5); tru=gt["channels"][c]["theta"]
    ax.errorbar(tru,m,yerr=[[m-lo],[hi-m]],fmt="o",color=colors[c],capsize=4,label=c)
ax.plot([0,1],[0,1],"k--",alpha=0.5)
ax.set_xlabel("TRUE θ"); ax.set_ylabel("estimated θ (89% CI)")
ax.set_title("Adstock θ: estimate vs truth"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("stage5_calibration.png",dpi=110,bbox_inches="tight")
print("\nsaved stage5_ppc.png, stage5_decomp.png, stage5_calibration.png")
