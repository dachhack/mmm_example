import pandas as pd, numpy as np
import pymc as pm, pytensor.tensor as pt
df = pd.read_csv("draftzone_mmm_data.csv")
chans=["tv","search","social","affiliate","brand"]
T=len(df); y=df.conversions.values.astype(float)
imp={c:df[f"{c}_impressions"].values.astype(float) for c in chans}
imp_s={c:imp[c]/imp[c].mean() for c in chans}
L=14
lag_idx=(np.arange(T)[:,None]+np.arange(L)[None,:])
imp_padded={c:np.concatenate([np.zeros(L-1),imp_s[c]]) for c in chans}
def adstock_pt(xp,theta):
    w=theta**pt.arange(L); w=w/pt.sum(w)
    wr=w[::-1]
    lagged=xp[lag_idx]
    return pt.sum(lagged*wr[None,:],axis=1)
with pm.Model() as m:
    base=pm.Normal("base",1300,300)
    beta=pm.HalfNormal("beta",300)
    theta=pm.Beta("theta",3,3)
    ad=adstock_pt(pt.as_tensor_variable(imp_padded["tv"]),theta)
    mu=base+beta*ad
    pm.Normal("obs",mu=mu,sigma=100,observed=y)
    # just test logp compiles & evaluates
    print("logp at test point:", float(m.compile_logp()(m.initial_point())))
    idata=pm.sample(50,tune=50,chains=1,cores=1,progressbar=False,random_seed=1)
print("SMOKE_OK")
