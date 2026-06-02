"""
Stage 8a: GEO LIFT EXPERIMENT for paid social (units fixed).
All impressions in RAW units. Per-market half-sat scaled so markets sit on the
RESPONSIVE part of the curve (not saturated), so incremental campaign impressions
actually drive incremental conversions.
"""
import numpy as np, json, pandas as pd
from transforms import geometric_adstock, hill_saturation
gt=json.load(open("ground_truth.json"))
rng=np.random.default_rng(808)
p_social=gt["channels"]["social"]

N_MARKETS=80; T=18
week=np.arange(T)
season=250*(0.5+0.5*np.sin(2*np.pi*(week-3)/18))   # shared confounder

market_base=rng.normal(320,55,N_MARKETS)
perm=rng.permutation(N_MARKETS)
treat_idx=perm[:N_MARKETS//2]; ctrl_idx=perm[N_MARKETS//2:]
treat_mask=np.zeros(N_MARKETS,bool); treat_mask[treat_idx]=True

# --- coherent units: BAU social ~ 30k impr/wk/market, half-sat ~50k -> responsive region
BAU_BASE=30000.0
INCREMENT=22000.0           # campaign adds 22k impr/wk in treatment, weeks 6+
HS_MKT=50000.0
def social_impr(treated):
    bau=BAU_BASE + 900*season + rng.normal(0,4000,T)   # season-correlated (confound persists)
    bau=np.clip(bau,0,None)
    extra=np.where(week>=6, INCREMENT, 0.0) if treated else np.zeros(T)
    return bau+extra

def social_contrib(impr):
    ad=geometric_adstock(impr, p_social["theta"], normalize=True)
    sat=hill_saturation(ad, HS_MKT, p_social["slope"])
    return p_social["beta"]*sat

rows=[]; treat_series=np.zeros((N_MARKETS,T))
for m in range(N_MARKETS):
    treated=treat_mask[m]
    impr=social_impr(treated)
    conv=market_base[m]+season+social_contrib(impr)+rng.normal(0,15,T)
    treat_series[m]=conv
    for w in range(T):
        rows.append(dict(market=m,week=w,treated=int(treated),
                         conversions=conv[w],social_impr=impr[w]))
mdf=pd.DataFrame(rows); mdf.to_csv("geo_experiment_data.csv",index=False)

# --- DiD analysis ---
pre=mdf[mdf.week<6]; post=mdf[mdf.week>=6]
t_pre=pre[pre.treated==1].conversions.mean(); c_pre=pre[pre.treated==0].conversions.mean()
t_post=post[post.treated==1].conversions.mean(); c_post=post[post.treated==0].conversions.mean()
naive=t_post-c_post
did=(t_post-c_post)-(t_pre-c_pre)

# --- TRUE incremental (deterministic, mean bau) ---
mean_bau=BAU_BASE+900*season
inc_with=social_contrib(np.where(week>=6, mean_bau+INCREMENT, mean_bau))
inc_without=social_contrib(mean_bau)
true_inc=(inc_with-inc_without)[6:].mean()

causal_per_1k=did/(INCREMENT/1000)
print("="*64)
print(f"GEO EXPERIMENT — paid social, {N_MARKETS} markets, {T} wks, campaign wk6+")
print(f"  Pre-period treat-ctrl gap: {t_pre-c_pre:+.2f} (≈0 validates randomization)")
print(f"  Naive post diff:        {naive:+.2f}/market-wk")
print(f"  DiD causal lift:        {did:+.2f}/market-wk")
print(f"  TRUE incremental:       {true_inc:+.2f}/market-wk")
print(f"  => DiD recovered {100*did/true_inc:.0f}% of true incremental effect")
print(f"  Incremental impr: {INCREMENT/1000:.0f}k/market-wk")
print(f"  Causal conversions per 1k incremental impr: {causal_per_1k:.4f}")

# bootstrap CI on DiD (resample markets)
boot=[]
mkts=np.arange(N_MARKETS)
for _ in range(2000):
    bt=rng.choice(treat_idx,len(treat_idx),replace=True)
    bc=rng.choice(ctrl_idx,len(ctrl_idx),replace=True)
    tp=treat_series[bt][:,6:].mean(); cp=treat_series[bc][:,6:].mean()
    tpre=treat_series[bt][:,:6].mean(); cpre=treat_series[bc][:,:6].mean()
    boot.append((tp-cp)-(tpre-cpre))
boot=np.array(boot); ci=np.percentile(boot,[5.5,94.5])
print(f"  DiD 89% CI (bootstrap): [{ci[0]:.2f}, {ci[1]:.2f}]")

anchor=dict(channel="social", did_per_market_week=float(did),
            did_ci=[float(ci[0]),float(ci[1])], true_increment=float(true_inc),
            causal_per_1k_impr=float(causal_per_1k),
            causal_per_1k_ci=[float(ci[0]/(INCREMENT/1000)),float(ci[1]/(INCREMENT/1000))],
            n_markets=N_MARKETS)
json.dump(anchor,open("experiment_anchor.json","w"),indent=2)
np.save("geo_treat_series.npy",treat_series); np.save("geo_treat_mask.npy",treat_mask)
print("\nSaved geo_experiment_data.csv, experiment_anchor.json")
