import sys; sys.path.insert(0,"/tmp/aud")
from audit_lib import *
import numpy.linalg as la

def partial(x,y,z):
    def r_(a,b):
        A = np.c_[np.ones(len(b)), b]; return a - A@la.lstsq(A,a,rcond=None)[0]
    return np.corrcoef(r_(x,z), r_(y,z))[0,1]

print("="*100); print("PART 3a: SCENARIO C MEDIATION (completing crashed run)"); print("="*100)
c = load("scenario_c_braking.csv")
c["sev"]=c.brake_intensity_t*c.brake_duration_ticks_t
c["inv_t1"]=np.maximum(0,1-c.speed_t1/15.0); c["resid"]=c.rho_actual-0.4*c.inv_t1
g=c.groupby("episode_id")[["sev","rho_actual","speed_t1","resid","speed_t"]].mean()
print(f"  r(sev,rho)                = {np.corrcoef(g.sev,g.rho_actual)[0,1]:+.4f}")
print(f"  r(sev,rho | speed_t1)     = {partial(g.sev.values,g.rho_actual.values,g.speed_t1.values):+.4f}   <-- effect left after removing speed")
print(f"  r(sev,speed_t1)           = {np.corrcoef(g.sev,g.speed_t1)[0,1]:+.4f}")
print(f"  r(sev,resid lane+yaw)     = {np.corrcoef(g.sev,g.resid)[0,1]:+.4f}")
print(f"  r(speed_t1,rho)           = {np.corrcoef(g.speed_t1,g.rho_actual)[0,1]:+.4f}")
print("  NOTE: speed_t1 min over ALL 3200 pairs = %.3f m/s -> inverse_speed NEVER pins at 1.0 at T+1" % c.speed_t1.min())
print("  Range-restriction check (subsetting on speed_t removes the hardest brakes, i.e. restricts the TREATMENT):")
for thr in [0.0,0.5,1.0,2.0]:
    sub=c[c.speed_t>thr]; gg=sub.groupby("episode_id")[["sev","rho_actual"]].mean()
    print(f"    speed_t>{thr}: n_pairs={len(sub):5d} sd(sev)={sub.sev.std():.2f} sd(speed_t1)={sub.speed_t1.std():.2f} r_ep={np.corrcoef(gg.sev,gg.rho_actual)[0,1]:+.4f}")

print("\n"+"="*100); print("PART 3b: SIGN HETEROGENEITY — is it mechanism, or three different formulas?"); print("="*100)
print("Test: replace each scenario's bespoke rho with the SAME externally-defined outcome computable")
print("      from every CSV: inverse_speed at T+1 = max(0, 1 - speed_t1/15). If signs still differ,")
print("      heterogeneity is real. If they collapse, the signs were formula artifacts.\n")

a = load("scenario_a_lanechange.csv");  a["sev"]=a.severity_t
b = load("scenario_b_intersection.csv"); b["sev"]=b.speed_t
sets = [("C  (sev=int*dur)", c, "sev"), ("B  (sev=yield speed_t)", b, "sev"), ("A  (sev=steer*ticks)", a, "sev")]
for name, df, sv in sets:
    df["inv_t1"]=np.maximum(0,1-df.speed_t1/15.0)
    gg=df.groupby("episode_id")[[sv,"rho_actual","inv_t1","speed_t1"]].mean()
    r_own=np.corrcoef(gg[sv],gg.rho_actual)[0,1]
    r_com=np.corrcoef(gg[sv],gg.inv_t1)[0,1]
    r_spd=np.corrcoef(gg[sv],gg.speed_t1)[0,1]
    print(f"  {name:24s} r(sev, OWN rho)={r_own:+.4f}   r(sev, COMMON inv_speed_t1)={r_com:+.4f}   r(sev, speed_t1)={r_spd:+.4f}")

print("\n  Also: how much does each scenario's own rho depend on speed at all?")
for name, df, sv in sets:
    gg=df.groupby("episode_id")[["rho_actual","speed_t1"]].mean()
    print(f"  {name:24s} r(speed_t1, rho) episode = {np.corrcoef(gg.speed_t1,gg.rho_actual)[0,1]:+.4f}")

print("\n"+"="*100); print("PART 3c: COMPONENT AUDIT — hunting saturation instance #6"); print("="*100)
print("\n--- A banked (scenario_a_lanechange.csv) ---")
desc(a, ["lat_dev_t","heading_err_t","speed_risk_t","rho_t","lat_dev_t1","heading_err_t1","speed_risk_t1","rho_actual","lat_dev_m_t1","heading_deg_t1","severity_t","steer_mag_t","steer_ticks_t","speed_t1"])
print("\n--- B instrumented (scenario_b_instrumented.csv) ---")
bi = load("scenario_b_instrumented.csv")
desc(bi, ["junction_progress_t","speed_risk_t","lane_dev_t","rho_t","junction_progress_t1","speed_risk_t1","lane_dev_t1","rho_actual","speed_t","speed_t1","yield_speed_target_t"])
print("\n--- B banked ---")
desc(b, ["speed_t","speed_t1","yield_speed_target_t","rho_actual"])
print("\n--- D ---")
d = load("scenario_d_baseline.csv"); desc(d, ["speed_t","speed_t1","rho_actual"])
print("\n--- C ---")
desc(c, ["speed_t","speed_t1","rho_actual","brake_intensity_t","brake_duration_ticks_t","sev","resid"])
