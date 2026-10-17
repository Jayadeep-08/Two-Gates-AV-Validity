import sys; sys.path.insert(0,"/tmp/aud")
from audit_lib import *

print("="*100); print("PART 6: LAYER 2 — THE RSS RESULT (Scenario C)"); print("="*100)
cn = load("scenario_c_npc_rss.csv")
cn["sev"]=cn.brake_intensity_t*cn.brake_duration_ticks_t
print(f"[C-NPC] pairs={len(cn)} episodes={cn.episode_id.nunique()}")
for col in ["rss_t1_rear_present","rss_t1_front_present","rss_t_rear_present","rss_t_front_present"]:
    print(f"  {col:24s} coverage = {cn[col].astype(str).str.lower().isin(['true','1']).mean():.3%}")
cn["rear1"]=cn.rss_t1_rear_present.astype(str).str.lower().isin(["true","1"])
cn["front1"]=cn.rss_t1_front_present.astype(str).str.lower().isin(["true","1"])

sub=cn[cn.rear1].copy()
print(f"\n  REAR-margin subset: {len(sub)} pairs, {sub.episode_id.nunique()} episodes")
g=sub.groupby("episode_id")[["sev","rss_t1_rear_margin","rss_t1_rear_gap","rss_t1_rear_v","rho_actual","speed_t1"]].mean()
full("  C RSS: episode r(sev, rear margin t1)  [claim -0.351, n=68]", g.sev.values, g.rss_t1_rear_margin.values)
full("  C RSS: episode r(sev, rear GAP t1)", g.sev.values, g.rss_t1_rear_gap.values)
full("  C RSS: episode r(sev, ego speed_t1) on same subset", g.sev.values, g.speed_t1.values)
print(f"  r(rear_margin_t1, speed_t1) episode = {np.corrcoef(g.rss_t1_rear_margin,g.speed_t1)[0,1]:+.4f}")
print(f"  r(rear_margin_t1, rho_actual) episode= {np.corrcoef(g.rss_t1_rear_margin,g.rho_actual)[0,1]:+.4f}")
import numpy.linalg as la
def partial(x,y,z):
    def r_(a,b):
        A=np.c_[np.ones(len(b)),b]; return a-A@la.lstsq(A,a,rcond=None)[0]
    return np.corrcoef(r_(x,z),r_(y,z))[0,1]
print(f"  PARTIAL r(sev, rear_margin_t1 | ego speed_t1) = {partial(g.sev.values,g.rss_t1_rear_margin.values,g.speed_t1.values):+.4f}   <-- is RSS independent of ego speed?")
sub2=cn[cn.front1]
if sub2.episode_id.nunique()>5:
    g2=sub2.groupby("episode_id")[["sev","rss_t1_front_margin"]].mean()
    full("  C RSS: episode r(sev, FRONT margin t1) [README: exploratory, CI incl 0]", g2.sev.values, g2.rss_t1_front_margin.values)
print("\n  ESCORT CONFOUND: does the CSV allow separating escort-driven from ambient-driven pairs?")
print(f"    columns containing 'escort': {[c for c in cn.columns if 'escort' in c.lower()]}")
es = pd.read_csv(D+"scenario_c_npc_rss_episode_stats.csv"); print(f"    episode_stats columns: {list(es.columns)}")

print("\n"+"="*100); print("PART 7: LAYER 4 — A's NPC replication and the empty-target-lane claim"); print("="*100)
an = load("scenario_a_lanechange_npc_rss.csv")
pres = an.rss_t1_present.astype(str).str.lower().isin(["true","1"])
pres_t = an.rss_t_present.astype(str).str.lower().isin(["true","1"])
print(f"[A-NPC] pairs={len(an)} eps={an.episode_id.nunique()}  target-lane vehicle present at T+1: {pres.mean():.2%}  at T: {pres_t.mean():.2%}")
print(f"    -> {1-pres.mean():.1%} of decisions had an EMPTY target lane (README claims 91.6%)")
g=an.groupby("episode_id")[["severity_t","rho_actual"]].mean()
full("  A-NPC full", g.severity_t.values, g.rho_actual.values)
if pres.sum()>20:
    s=an[pres]; gg=s.groupby("episode_id")[["severity_t","rho_actual"]].mean()
    full(f"  A-NPC pairs WITH target-lane vehicle ({pres.sum()} pairs)", gg.severity_t.values, gg.rho_actual.values)
print(f"  escort_longi_t non-null: {an.escort_longi_t.notna().sum()} / {len(an)}  (escorts claimed disabled)")

print("\n"+"="*100); print("PART 8: LAYER 3 — the D-mean secondary layer"); print("="*100)
d=load("scenario_d_baseline.csv"); dmean=d.rho_actual.mean()
c=load("scenario_c_braking.csv"); b=load("scenario_b_intersection.csv"); a=load("scenario_a_lanechange.csv")
print(f"  D global mean rho = {dmean:.4f}")
for nm,df in [("C",c),("B",b),("A",a)]:
    gap=(df.rho_actual.mean()-dmean)/dmean*100
    print(f"  {nm}: mean rho={df.rho_actual.mean():.4f}  gap vs D-mean = {gap:+.1f}%   [README claims C+209 B+377 A+322]")
print("\n  CONFOUND MAGNITUDE: what does each formula score for the SAME benign state?")
print("  D formula = 0.4*speed_norm + 0.6*lane_dev ; C = 0.4*inv_speed + 0.4*lane_dev + 0.2*yaw ;")
print("  A = 0.5*lat_dev(target lane!) + 0.3*heading + 0.2*speed_risk ; B = 0.5*jct_progress + 0.3*speed_risk + 0.2*lane_dev")
print(f"  A's rho for a PERFECTLY EXECUTED change at cruise: 0.5*~0 + 0.3*~0 + 0.2*0.10 = 0.020")
print(f"  A's rho floor from speed_risk alone = {0.2*a.speed_risk_t1.min():.4f}; observed A min rho = {a.rho_actual.min():.4f}")
print(f"  B's rho floor from jct_progress alone (jp=1 => 0.5) + speed_risk 0.3*0.21 = {0.5+0.3*0.207:.3f}; observed B min rho = {b.rho_actual.min():.4f}")
print(f"  -> B can NEVER score below ~{b.rho_actual.min():.3f} by construction. D's mean is {dmean:.3f}.")
print(f"  -> B's '+377% gap' is therefore mostly the constant offset of B's formula, not carry-forward.")
