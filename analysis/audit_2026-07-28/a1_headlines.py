import sys; sys.path.insert(0,"/tmp/aud")
from audit_lib import *

print("="*100)
print("TASK 1: INDEPENDENT REPLICATION OF TABLE I HEADLINE NUMBERS")
print("="*100)

# ---- C ----
c = load("scenario_c_braking.csv")
print(f"\n[C] rows={len(c)} episodes={c.episode_id.nunique()} outcomes={c.outcome.value_counts().to_dict()}")
c["sev"] = c.brake_intensity_t * c.brake_duration_ticks_t
x, y = ep_agg(c, "sev", "rho_actual")
res_c = full("C banked: episode r(sev, rho_t1)  [claim +0.7697]", x, y)
# pair level
full("C banked: PAIR-level (descriptive only)", c.sev.values, c.rho_actual.values, 2000)

# ---- B ----
b = load("scenario_b_intersection.csv")
print(f"\n[B] rows={len(b)} episodes={b.episode_id.nunique()} outcomes={b.outcome.value_counts().to_dict()}")
x, y = ep_agg(b, "speed_t", "rho_actual")
res_b = full("B banked: episode r(speed_t, rho_t1) [claim -0.150]", x, y)

# ---- A ----
a = load("scenario_a_lanechange.csv")
print(f"\n[A] rows={len(a)} episodes={a.episode_id.nunique()} outcomes={a.outcome.value_counts().to_dict()}")
print(f"    severity_t col present; check identity vs steer_mag*steer_ticks: "
      f"maxdiff={np.abs(a.severity_t - a.steer_mag_t*a.steer_ticks_t).max():.2e}")
x, y = ep_agg(a, "severity_t", "rho_actual")
res_a = full("A banked: episode r(sev, rho_t1)   [claim -0.4951]", x, y)

# ---- D ----
d = load("scenario_d_baseline.csv")
print(f"\n[D] rows={len(d)} episodes={d.episode_id.nunique()} "
      f"mean rho={d.rho_actual.mean():.4f} std={d.rho_actual.std():.4f} "
      f"outcomes={d.outcome.value_counts().to_dict()}  [claim mean 0.0963]")

# ---- NPC variants ----
cn = load("scenario_c_npc_rss.csv")
print(f"\n[C-NPC] rows={len(cn)} episodes={cn.episode_id.nunique()} outcomes={cn.outcome.value_counts().to_dict()}")
cn["sev"] = cn.brake_intensity_t * cn.brake_duration_ticks_t
x, y = ep_agg(cn, "sev", "rho_actual")
full("C NPC: episode r(sev, rho_t1)      [claim +0.4158]", x, y)

an = load("scenario_a_lanechange_npc_rss.csv")
print(f"\n[A-NPC] rows={len(an)} episodes={an.episode_id.nunique()} outcomes={an.outcome.value_counts().to_dict()}")
x, y = ep_agg(an, "severity_t", "rho_actual")
full("A NPC: episode r(sev, rho_t1)      [claim -0.5245]", x, y)
