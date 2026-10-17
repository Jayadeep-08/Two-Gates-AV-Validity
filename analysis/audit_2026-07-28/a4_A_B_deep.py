import sys; sys.path.insert(0,"/tmp/aud")
from audit_lib import *
import numpy.linalg as la
def partial(x,y,z):
    def r_(a,b):
        A=np.c_[np.ones(len(b)),b]; return a-A@la.lstsq(A,a,rcond=None)[0]
    return np.corrcoef(r_(x,z),r_(y,z))[0,1]

print("="*100); print("PART 4: SCENARIO A — sentinel rows, tautology test, robustness"); print("="*100)
a = load("scenario_a_lanechange.csv")
SENT_L, SENT_H = 1.5*3.0, 20.0*3.0
a["sentinel_t1"] = (np.isclose(a.lat_dev_m_t1, SENT_L) & np.isclose(a.heading_deg_t1, SENT_H))
a["sentinel_t"]  = (np.isclose(a.lat_dev_m_t, SENT_L) if "lat_dev_m_t" in a else False)
print(f"\n[A] 'target_wp is None' SENTINEL rows at T+1 (lat_dev_m=4.5 AND heading=60 exactly): "
      f"{a.sentinel_t1.sum()} / {len(a)} = {a.sentinel_t1.mean():.3%}")
print(f"    mean severity in sentinel rows = {a[a.sentinel_t1].severity_t.mean():.3f}  vs non-sentinel {a[~a.sentinel_t1].severity_t.mean():.3f}")
print(f"    mean rho in sentinel rows      = {a[a.sentinel_t1].rho_actual.mean():.3f}  vs non-sentinel {a[~a.sentinel_t1].rho_actual.mean():.3f}")
print(f"    episodes touched by sentinel   = {a[a.sentinel_t1].episode_id.nunique()} / {a.episode_id.nunique()}")

g=a.groupby("episode_id")[["severity_t","rho_actual"]].mean()
full("  A full", g.severity_t.values, g.rho_actual.values)
sub=a[~a.sentinel_t1]; gg=sub.groupby("episode_id")[["severity_t","rho_actual"]].mean()
full("  A excluding sentinel T+1 rows", gg.severity_t.values, gg.rho_actual.values)

print("\n[A] TAUTOLOGY TEST: severity = steer_mag x steer_ticks ~ integral of steering = lateral displacement.")
print("    If rho is a monotone transform of displacement-from-target, r(sev,rho) is a re-description of kinematics.")
print(f"    pair r(severity, lat_dev_m_t1)   = {np.corrcoef(a.severity_t,a.lat_dev_m_t1)[0,1]:+.4f}")
print(f"    pair r(severity, lat_dev_t1)     = {np.corrcoef(a.severity_t,a.lat_dev_t1)[0,1]:+.4f}")
print(f"    pair r(severity, heading_err_t1) = {np.corrcoef(a.severity_t,a.heading_err_t1)[0,1]:+.4f}")
print(f"    pair r(severity, speed_risk_t1)  = {np.corrcoef(a.severity_t,a.speed_risk_t1)[0,1]:+.4f}")
cv=np.cov(a.severity_t,a.rho_actual)[0,1]
for w,comp in [(0.5,"lat_dev_t1"),(0.3,"heading_err_t1"),(0.2,"speed_risk_t1")]:
    ci=w*np.cov(a.severity_t,a[comp])[0,1]; print(f"    cov share {comp:16s} = {ci/cv:+.1%}")
print(f"    r(sev, rho | lat_dev_t1) episode = {partial(g.severity_t.values, g.rho_actual.values, a.groupby('episode_id').lat_dev_t1.mean().values):+.4f}")

print("\n[A] completion split")
for k,v in a.groupby("lane_change_completed"):
    gg=v.groupby("episode_id")[["severity_t","rho_actual"]].mean()
    print(f"    completed={k}: n_pairs={len(v)} mean_sev={v.severity_t.mean():.2f} mean_rho={v.rho_actual.mean():.3f} "
          f"r_ep={np.corrcoef(gg.severity_t,gg.rho_actual)[0,1] if len(gg)>3 else float('nan'):+.4f} (n_ep={len(gg)})")
print(f"    r(severity, completion rate) episode = {np.corrcoef(a.groupby('episode_id').severity_t.mean(), a.groupby('episode_id').lane_change_completed.mean())[0,1]:+.4f}")

print("\n"+"="*100); print("PART 5: SCENARIO B — is the effect the same kinematic re-description?"); print("="*100)
bi = load("scenario_b_instrumented.csv")
print(f"[B-inst] n={len(bi)} episodes={bi.episode_id.nunique()}")
print(f"  junction_progress_t  pinned at 1.0: {(bi.junction_progress_t>=0.9999).mean():.1%}   (NOT disclosed in CLAUDE.md)")
print(f"  junction_progress_t1 pinned at 1.0: {(bi.junction_progress_t1>=0.9999).mean():.1%}   (disclosed: 47.3%)")
print(f"  speed_risk_t1 std={bi.speed_risk_t1.std():.4f} -> contributes {0.3*bi.speed_risk_t1.std():.5f} to rho sd {bi.rho_actual.std():.4f}")
cv=np.cov(bi.speed_t,bi.rho_actual)[0,1]
for w,comp in [(0.5,"junction_progress_t1"),(0.3,"speed_risk_t1"),(0.2,"lane_dev_t1")]:
    print(f"  cov share {comp:22s} = {w*np.cov(bi.speed_t,bi[comp])[0,1]/cv:+.1%}")
gb=bi.groupby("episode_id")[["speed_t","rho_actual","junction_progress_t1"]].mean()
print(f"  episode r(speed_t, rho)  = {np.corrcoef(gb.speed_t,gb.rho_actual)[0,1]:+.4f} (n={len(gb)})")
print(f"  episode r(speed_t, jp_t1)= {np.corrcoef(gb.speed_t,gb.junction_progress_t1)[0,1]:+.4f}")
print(f"  episode r(speed_t, rho | jp_t1) = {partial(gb.speed_t.values,gb.rho_actual.values,gb.junction_progress_t1.values):+.4f}")

print("\n  B banked: is the effect concentrated in a few junctions? (road_id_t1 as junction proxy)")
b = load("scenario_b_intersection.csv")
b["j"]=b.road_id_t1.astype(str)
rows=[]
for j,v in b.groupby("j"):
    if v.episode_id.nunique()>=8:
        gg=v.groupby("episode_id")[["speed_t","rho_actual"]].mean()
        if gg.speed_t.std()>0 and gg.rho_actual.std()>0:
            rows.append((j,len(v),gg.shape[0],np.corrcoef(gg.speed_t,gg.rho_actual)[0,1]))
rows.sort(key=lambda r:r[3])
print(f"   {len(rows)} junctions with >=8 episodes; per-junction episode-level r:")
for r in rows: print(f"     road {r[0]:>6s} pairs={r[1]:4d} eps={r[2]:3d} r={r[3]:+.3f}")
neg=sum(1 for r in rows if r[3]<0); print(f"   negative: {neg}/{len(rows)}   median r={np.median([r[3] for r in rows]):+.3f}")
