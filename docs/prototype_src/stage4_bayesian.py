"""Stage 4b: BAYESIAN MMM (PyMC) — lean & sandbox-safe (single chain, small L)."""
import pandas as pd, numpy as np
import pymc as pm, pytensor.tensor as pt
import arviz as az

df = pd.read_csv("draftzone_mmm_data.csv", parse_dates=["week"])
chans=["tv","search","social","affiliate","brand"]
T=len(df); y=df.conversions.values.astype(float); t=np.arange(T)
trend=t/T
fourier=np.column_stack([f(2*np.pi*k*t/52) for k in range(1,4) for f in (np.sin,np.cos)])
promo=df.promo_flag.values.astype(float)
price=(df.price_index.values-100)/10
comp=(df.competitor_pressure.values-df.competitor_pressure.mean())/df.competitor_pressure.std()
holiday=df.holiday_flag.values.astype(float)
Xc=np.column_stack([trend,fourier,promo,price,comp,holiday]).astype(float)
imp={c:df[f"{c}_impressions"].values.astype(float) for c in chans}
imp_scale={c:imp[c].mean() for c in chans}
imp_s={c:imp[c]/imp_scale[c] for c in chans}
theta_prior={"tv":(6,2),"search":(1.5,6),"social":(3,4),"affiliate":(2,4),"brand":(4,3)}
L=10
lag_idx=(np.arange(T)[:,None]+np.arange(L)[None,:])
imp_padded={c:np.concatenate([np.zeros(L-1),imp_s[c]]) for c in chans}
wexp=np.arange(L)

def adstock_pt(xp,theta):
    w=theta**pt.as_tensor_variable(wexp.astype(float)); w=w/pt.sum(w)
    lagged=xp[lag_idx]              # T x L (col 0 = oldest)
    return pt.sum(lagged*w[None,:],axis=1)  # note: col0 oldest gets w[0]=theta^0; fine, symmetric label

with pm.Model() as model:
    baseline=pm.Normal("baseline",1300,300)
    ctrl_coef=pm.Normal("ctrl_coef",0,50,shape=Xc.shape[1])
    sigma=pm.HalfNormal("sigma",100)
    Xc_t=pt.as_tensor_variable(Xc)
    media=0
    for c in chans:
        a,b=theta_prior[c]
        theta=pm.Beta(f"theta_{c}",a,b)
        slope=pm.Gamma(f"slope_{c}",mu=1.3,sigma=0.5)
        hs=pm.Gamma(f"hs_{c}",mu=1.0,sigma=0.6)
        beta=pm.HalfNormal(f"beta_{c}",sigma=300)
        ad=adstock_pt(pt.as_tensor_variable(imp_padded[c]),theta)
        sat=ad**slope/(ad**slope+hs**slope+1e-9)
        media=media+beta*sat
    mu=baseline+pt.dot(Xc_t,ctrl_coef)+media
    pm.Normal("obs",mu=mu,sigma=sigma,observed=y)
    lp=model.compile_logp()(model.initial_point())
    print("LOGP_OK", float(lp), flush=True)
    idata=pm.sample(500,tune=500,chains=2,cores=1,target_accept=0.92,
                    random_seed=42,progressbar=False)

import pickle; pickle.dump(idata, open("bayes_idata.pkl","wb"))
summ=az.summary(idata,var_names=[f"theta_{c}" for c in chans]+[f"beta_{c}" for c in chans]+["baseline"])
print(summ[["mean","sd","hdi_3%","hdi_97%","r_hat"]].to_string())
print("MAXRHAT", float(summ["r_hat"].max()))
print("DONE_SAMPLING", flush=True)
