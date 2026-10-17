import sys; sys.path.insert(0,"/tmp/aud")
from audit_lib import *

print("="*100); print("TASK 2: SCENARIO C DECOMPOSITION"); print("="*100)

c = load("scenario_c_braking.csv")
c["sev"] = c.brake_intensity_t * c.brake_duration_ticks_t
c["inv_t1"] = np.maximum(0.0, 1.0 - c.speed_t1/15.0)
c["inv_contrib"] = 0.4*c.inv_t1
c["resid"] = c.rho_actual - c.inv_contrib     # = 0.4*lane_dev + 0.2*yaw

print("\n-- reconstruction sanity: residual must lie in [0,0.6] if rho sampled at same instant as speed_t1")
print(f"   resid min={c.resid.min():+.5f} max={c.resid.max():+.5f} mean={c.resid.mean():.5f} std={c.resid.std():.5f}")
print(f"   frac resid<0: {(c.resid<-1e-6).mean():.4f}")

print("\n-- VARIANCE SHARE of rho_actual")
vt = c.rho_actual.var()
print(f"   var(rho)={vt:.6f}  var(0.4*inv_t1)={c.inv_contrib.var():.6f} ({c.inv_contrib.var()/vt:.1%})"
      f"  var(resid)={c.resid.var():.6f} ({c.resid.var()/vt:.1%})"
      f"  2cov={2*np.cov(c.inv_contrib,c.resid)[0,1]:.6f} ({2*np.cov(c.inv_contrib,c.resid)[0,1]/vt:.1%})")

print("\n-- COVARIANCE ATTRIBUTION of cov(sev, rho) at pair level")
cv = np.cov(c.sev, c.rho_actual)[0,1]
cvi = np.cov(c.sev, c.inv_contrib)[0,1]
cvr = np.cov(c.sev, c.resid)[0,1]
print(f"   cov(sev,rho)={cv:.5f}   inv_speed channel={cvi:.5f} ({cvi/cv:.1%})   residual channel={cvr:.5f} ({cvr/cv:.1%})")

print("\n-- EPISODE-LEVEL correlations against each channel")
g = c.groupby("episode_id")[["sev","rho_actual","inv_contrib","resid","speed_t","speed_t1"]].mean()
for col in ["rho_actual","inv_contrib","resid","speed_t1","speed_t"]:
    full(f"   C episode r(sev, {col})", g.sev.values, g[col].values)

print("\n-- STOP SATURATION: which variable actually saturates rho? speed_t1 drives inv_t1, not speed_t")
for v,name in [(c.speed_t,"speed_t (state at T)"),(c.speed_t1,"speed_t1 (state at T+1)")]:
    print(f"   {name:26s} mean={v.mean():.3f} std={v.std():.3f} frac<0.1={( v<0.1).mean():.3f} frac<0.5={(v<0.5).mean():.3f} frac<1.0={(v<1.0).mean():.3f}")
print(f"   inv_t1 pinned at 1.0 (speed_t1==0): frac={(c.inv_t1>=0.999).mean():.3f}")

print("\n-- CLAIM CHECK: non-saturated subset (CLAUDE.md OPEN ISSUE used speed_t>0.5 -> claims +0.2284)")
for thr,col in [(0.5,"speed_t"),(0.5,"speed_t1"),(1.0,"speed_t1"),(2.0,"speed_t1")]:
    sub = c[c[col]>thr]
    gg = sub.groupby("episode_id")[["sev","rho_actual"]].mean()
    if len(gg)>10:
        full(f"   C episode r | {col}>{thr}  ({len(sub)}/{len(c)} pairs)", gg.sev.values, gg.rho_actual.values)

print("\n-- STOP/NO-STOP DICHOTOMY: does severity mostly just predict P(stop)?")
c["stopped"] = (c.speed_t1 < 0.5).astype(float)
gg = c.groupby("episode_id")[["sev","stopped","rho_actual"]].mean()
full("   C episode r(sev, P(stopped at T+1))", gg.sev.values, gg.stopped.values)

print("\n-- PARTIAL CORRELATION: sev vs rho controlling for speed_t1 (episode means)")
def partial(x,y,z):
    import numpy.linalg as la
    def resid(a,b):
        A = np.c_[np.ones(len(b)), b]
        return a - A@la.lstsq(A,a,rcond=None)[0]
    return np.corrcoef(resid(x,z), resid(y,z))[0,1]
g2 = c.groupby("episode_id")[["sev","rho_actual","speed_t1","resid"]].mean()
print(f"   r(sev,rho)                     = {np.corrcoef(g2.sev,g2.rho_actual)[0,1]:+.4f}")
print(f"   r(sev,rho | speed_t1)          = {partial(g2.sev.values,g2.rho_actual.values,g2.speed_t1.values):+.4f}")
print(f"   r(sev,speed_t1)                = {np.corrcoef(g2.sev,g2.speed_t1)[0,1]:+.4f}")
print(f"   r(sev,resid[lane+yaw])         = {np.corrcoef(g2.sev,g2.resid)[0,1]:+.4f}")
print(f"   r(sev, rho) using ONLY resid channel scaled: see above")

print("\n-- Is rho essentially a monotone function of speed_t1 alone?")
print(f"   pair r(speed_t1, rho_actual) = {np.corrcoef(c.speed_t1,c.rho_actual)[0,1]:+.4f}")
print(f"   pair r(inv_contrib, rho_actual) = {np.corrcoef(c.inv_contrib,c.rho_actual)[0,1]:+.4f}")
print(f"   pair r(resid, rho_actual) = {np.corrcoef(c.resid,c.rho_actual)[0,1]:+.4f}")
