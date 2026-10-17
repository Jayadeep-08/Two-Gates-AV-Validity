import sys; sys.path.insert(0,"/tmp/aud")
from audit_lib import *
import numpy.linalg as la
def partial(x,y,z):
    def r_(a,b):
        A=np.c_[np.ones(len(b)),b]; return a-A@la.lstsq(A,a,rcond=None)[0]
    return np.corrcoef(r_(x,z),r_(y,z))[0,1]

print("="*100); print("PART 9: RSS selection-on-treatment check"); print("="*100)
cn=load("scenario_c_npc_rss.csv"); cn["sev"]=cn.brake_intensity_t*cn.brake_duration_ticks_t
cn["rear1"]=cn.rss_t1_rear_present.astype(str).str.lower().isin(["true","1"])
g=cn.groupby("episode_id")[["sev","rear1"]].mean()
full("  r(severity, P(rear vehicle detected at T+1))", g.sev.values, g.rear1.values.astype(float))
print("  ^ if positive, conditioning on rear_present selects on the treatment (harder brake -> follower closes -> detected)")
es=pd.read_csv(D+"scenario_c_npc_rss_episode_stats.csv")
print(f"  escorts_spawned distribution: {es.escorts_spawned.value_counts().to_dict()}  -> {'CAN' if es.escorts_spawned.nunique()>1 else 'CANNOT'} split episodes by escort")
if es.escorts_spawned.nunique()>1:
    m=cn.merge(es[["episode_id","escorts_spawned"]],on="episode_id",how="left")
    for k,v in m[m.rear1].groupby(m[m.rear1].escorts_spawned>0):
        gg=v.groupby("episode_id")[["sev","rss_t1_rear_margin"]].mean()
        if len(gg)>8: print(f"    escort>0={k}: eps={len(gg)} r={np.corrcoef(gg.sev,gg.rss_t1_rear_margin)[0,1]:+.4f}")

print("\n"+"="*100); print("PART 10: B — do the POSITIVE junctions correspond to the jp cap?"); print("="*100)
bi=load("scenario_b_instrumented.csv"); bi["j"]=bi.road_id_t1.astype(str)
rows=[]
for j,v in bi.groupby("j"):
    if len(v)>=6:
        gg=v.groupby("episode_id")[["speed_t","rho_actual"]].mean()
        if len(gg)>=5 and gg.speed_t.std()>0 and gg.rho_actual.std()>0:
            rows.append((j,len(v),(v.junction_progress_t1>=0.9999).mean(),np.corrcoef(gg.speed_t,gg.rho_actual)[0,1]))
print("  road  pairs  frac_jp_capped   r_episode")
for r in sorted(rows,key=lambda x:-x[2]): print(f"  {r[0]:>5s} {r[1]:5d}   {r[2]:10.2f}   {r[3]:+.3f}")
if len(rows)>3:
    fr=np.array([r[2] for r in rows]); rr=np.array([r[3] for r in rows])
    print(f"  r(frac_jp_capped, r_episode) across junctions = {np.corrcoef(fr,rr)[0,1]:+.3f}  (positive => cap causes the sign flip)")

print("\n"+"="*100); print("PART 11: is D actually a valid control, or vacuous?"); print("="*100)
d=load("scenario_d_baseline.csv")
print(f"  D has NO severity variable at all -> no ablation is even defined. Columns: {[c for c in d.columns if 'action' in c or 'sever' in c]}")
print(f"  D action_t values: {d.action_t.value_counts().to_dict()}")
print(f"  D speed_t std={d.speed_t.std():.3f}; using speed_t as a pseudo-severity (the closest analogue to B's):")
gd=d.groupby("episode_id")[["speed_t","rho_actual"]].mean()
full("  D: episode r(speed_t, rho_t1) [B's exact ablation applied to D]", gd.speed_t.values, gd.rho_actual.values)
print("  ^ If this is significant, D does NOT function as a null control for B's ablation.")

print("\n"+"="*100); print("PART 12: cross-check CLAUDE.md numeric claims against disk"); print("="*100)
a=load("scenario_a_lanechange.csv")
gg=a[a.outcome=="safe"].groupby("episode_id")[["severity_t","rho_actual"]].mean()
print(f"  A safe-only r  = {np.corrcoef(gg.severity_t,gg.rho_actual)[0,1]:+.4f}  [CLAUDE.md claims -0.457]")
ep_comp=a.groupby("episode_id").lane_change_completed.mean()
sub=a[a.lane_change_completed==True]; gg=sub.groupby("episode_id")[["severity_t","rho_actual"]].mean()
print(f"  A completed-only r = {np.corrcoef(gg.severity_t,gg.rho_actual)[0,1]:+.4f}  [CLAUDE.md claims -0.571]")
c=load("scenario_c_braking.csv"); c["sev"]=c.brake_intensity_t*c.brake_duration_ticks_t
print(f"  C pair-level r = {np.corrcoef(c.sev,c.rho_actual)[0,1]:+.4f}  [CLAUDE.md claims 0.584 / 0.770 episode]")
b=load("scenario_b_intersection.csv")
print(f"  B pair-level r = {np.corrcoef(b.speed_t,b.rho_actual)[0,1]:+.4f}  [CLAUDE.md claims -0.119]")
print(f"  B is_junction_t1 False count = {(b.is_junction_t1.astype(str).str.lower()=='false').sum()} [claims 1]")
bi2=load("scenario_b_instrumented.csv")
print(f"  B-inst within-junction r(speed_t,jp_t1) pair-level, non-capped subset:")
nc=bi2[bi2.junction_progress_t1<0.9999]
print(f"    n={len(nc)} r={np.corrcoef(nc.speed_t,nc.junction_progress_t1)[0,1]:+.4f} [CLAUDE.md claims -0.195 p=0.004 n=219]")
cap=bi2[bi2.junction_progress_t1>=0.9999]
print(f"    CAPPED subset n={len(cap)} r(speed_t,rho)={np.corrcoef(cap.speed_t,cap.rho_actual)[0,1]:+.4f} [CLAUDE.md claims +0.361 p=1.8e-05]")
