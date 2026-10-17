"""
THE EXTRA-MARKOVIAN TEST.

Question: does severity at decision T predict risk at T+2 AFTER conditioning on the
logged state at T+1 (and on the new randomized severity of decision T+1)?

  null  -> state at T+1 fully mediates; RPI reduces to the transition kernel composed
           with the reward. No extra-Markovian propagation. Honest reframe required.
  non-null -> either genuine extra-Markovian propagation, OR the logged state vector is
           not a sufficient statistic. Scenario D discriminates: D has no scripted
           decision, so if D ALSO shows residual predictability the finding is generic
           state-vector incompleteness, not decision carry-forward.

Chain within an episode (all three scenarios log 8 sequential cycles, t = 0..7):
    cycle d:   [decision T, severity sev_d]  ->  state/risk at T+1 of cycle d
    cycle d+1: [decision T, severity sev_d+1] -> state/risk at T+1 of cycle d+1  (== "T+2")

Null distribution: permute sev_d WITHIN episode across cycles. Severity is randomized
per cycle by construction, so this is the exact randomization null and it automatically
absorbs all episode/location/spawn effects.
"""
import sys; sys.path.insert(0, "/tmp/aud")
from audit_lib import *
import numpy.linalg as la

def resid(y, X):
    X = np.c_[np.ones(len(y)), X]
    return y - X @ la.lstsq(X, y, rcond=None)[0]

def partial_r(x, y, Z):
    return np.corrcoef(resid(x, Z), resid(y, Z))[0, 1]

def within_ep_perm(df, xcol, ycol, Zcols, n=10000, seed=7):
    """Permute xcol within episode; recompute partial r each time."""
    Z = df[Zcols].values
    y = df[ycol].values
    ry = resid(y, Z)
    r0 = np.corrcoef(resid(df[xcol].values, Z), ry)[0, 1]
    rs = np.random.default_rng(seed)
    eps = df.episode_id.values
    order = np.argsort(eps, kind="stable")
    bounds = np.flatnonzero(np.diff(eps[order])) + 1
    groups = np.split(order, bounds)
    x = df[xcol].values.copy()
    cnt = 0
    for _ in range(n):
        xp = x.copy()
        for gidx in groups:
            xp[gidx] = rs.permutation(x[gidx])
        rp = np.corrcoef(resid(xp, Z), ry)[0, 1]
        if abs(rp) >= abs(r0) - 1e-15:
            cnt += 1
    return r0, (cnt + 1) / (n + 1)

def build_chain(df, sevcol, statecols):
    """Join cycle d to cycle d+1 within episode."""
    a = df.sort_values(["episode_id", "t"]).copy()
    b = a.copy()
    b["t"] = b["t"] - 1
    keep_b = ["episode_id", "t", "rho_actual", sevcol]
    b = b[keep_b].rename(columns={"rho_actual": "rho_next", sevcol: "sev_next"})
    m = a.merge(b, on=["episode_id", "t"], how="inner")
    return m.dropna(subset=["rho_next", "sev_next", "rho_actual", sevcol] + statecols)

def report(name, m, sevcol, statecols, extra_ctrl=("rho_actual", "sev_next")):
    Z_state = list(statecols) + list(extra_ctrl)
    print(f"\n{'='*96}\n{name}\n{'='*96}")
    print(f"  chained transitions n={len(m)}  episodes={m.episode_id.nunique()}")
    print(f"  controls conditioned on: {Z_state}")
    r_same = np.corrcoef(m["sev_next"], m["rho_next"])[0, 1]
    print(f"\n  [sanity] r(sev_next, rho_next)            = {r_same:+.4f}   <- the known one-step effect, should be large")
    r_uncond = np.corrcoef(m[sevcol], m["rho_next"])[0, 1]
    print(f"  [step 1] r(sev_T, rho_T+2)  UNCONDITIONAL = {r_uncond:+.4f}")
    r_p, p_p = within_ep_perm(m, sevcol, "rho_next", Z_state)
    print(f"  [step 2] r(sev_T, rho_T+2 | state_T+1, rho_T+1, sev_T+1) = {r_p:+.4f}   within-episode perm p = {p_p:.4f}")
    # incremental R^2
    y = m["rho_next"].values
    Zb = np.c_[np.ones(len(y)), m[Z_state].values]
    Zf = np.c_[Zb, m[sevcol].values]
    def r2(X):
        e = y - X @ la.lstsq(X, y, rcond=None)[0]
        return 1 - e.var() / y.var()
    print(f"  [step 3] R^2 without sev_T = {r2(Zb):.4f}   with sev_T = {r2(Zf):.4f}   "
          f"incremental = {r2(Zf)-r2(Zb):+.5f}")
    return dict(name=name, n=len(m), r_uncond=r_uncond, r_partial=r_p, p=p_p,
                dR2=r2(Zf) - r2(Zb))

out = []

# ---------------- Scenario C ----------------
c = load("scenario_c_braking.csv")
c["sev"] = c.brake_intensity_t * c.brake_duration_ticks_t
STATE_C = ["speed_t1", "yaw_t1", "x_t1", "y_t1"]
mc = build_chain(c, "sev", STATE_C)
out.append(report("SCENARIO C (braking) — extra-Markovian test", mc, "sev", STATE_C))

# ---------------- Scenario A ----------------
a = load("scenario_a_lanechange.csv")
# A logs the FULL risk-component vector at T+1 -> conditioning here is near-complete:
# rho is an exact deterministic function of these three columns.
STATE_A = ["lat_dev_t1", "heading_err_t1", "speed_risk_t1", "speed_t1", "yaw_t1"]
ma = build_chain(a, "severity_t", STATE_A)
out.append(report("SCENARIO A (lane change) — extra-Markovian test\n"
                  "  NOTE: rho is an EXACT function of the three logged components at T+1,\n"
                  "  so this conditions on a genuinely sufficient statistic for rho_T+1.",
                  ma, "severity_t", STATE_A))

# ---------------- Scenario D: the state-sufficiency control ----------------
# D has no scripted decision. Use speed_t (the only varying state feature at T) as a
# pseudo-severity. If D also shows residual predictability, the effect is generic
# state-vector incompleteness / autocorrelation, NOT decision carry-forward.
d = load("scenario_d_baseline.csv")
STATE_D = ["speed_t1", "yaw_t1", "x_t1", "y_t1"]
md = build_chain(d, "speed_t", STATE_D)
out.append(report("SCENARIO D (no decision) — STATE-SUFFICIENCY CONTROL\n"
                  "  pseudo-severity = speed_t. Non-null here means the logged state is\n"
                  "  simply not a sufficient statistic — nothing to do with decisions.",
                  md, "speed_t", STATE_D))

print(f"\n\n{'='*96}\nSUMMARY\n{'='*96}")
print(f"{'scenario':<52}{'n':>6}{'r uncond':>11}{'r partial':>11}{'perm p':>10}{'dR2':>10}")
for o in out:
    print(f"{o['name'].splitlines()[0]:<52}{o['n']:>6}{o['r_uncond']:>+11.4f}{o['r_partial']:>+11.4f}"
          f"{o['p']:>10.4f}{o['dR2']:>+10.5f}")
