"""
Robustness of the T -> T+2 conditional result. Three threats, tested in order:

  THREAT 1  Non-linear mediator. Conditioning is linear. If risk at T+2 depends
            non-linearly on speed/state at T+1, a straight line through a curve
            leaves residual variance that severity can pick up. Fix: condition on
            polynomial + interaction expansions of the state and see if it survives.

  THREAT 2  Permutation null is not exact here. Under the sharp null "sev_T does not
            affect rho_T+2", sev_T STILL affects the conditioning set (state at T+1).
            So the controls are mediators, not pre-treatment covariates, and
            exchangeability fails. Fix: episode-clustered bootstrap CI on the partial
            correlation, which needs no exchangeability assumption.

  THREAT 3  Conditioning-set incompleteness. C logs only (speed, yaw, x, y) at T+1 but
            its rho also depends on unlogged lane_dev and yaw_risk. A logs its complete
            component vector. If the effect is real it should be at least as strong in A.
"""
import sys; sys.path.insert(0, "/tmp/aud")
from audit_lib import *
import numpy.linalg as la

def resid(y, X):
    X = np.c_[np.ones(len(y)), X]
    return y - X @ la.lstsq(X, y, rcond=None)[0]

def partial_r(x, y, Z):
    return np.corrcoef(resid(x, Z), resid(y, Z))[0, 1]

def expand(Z, deg=3, inter=True):
    """Polynomial + pairwise-interaction expansion of the conditioning set."""
    cols = [Z]
    for d in range(2, deg + 1):
        cols.append(Z ** d)
    if inter:
        k = Z.shape[1]
        for i in range(k):
            for j in range(i + 1, k):
                cols.append((Z[:, i] * Z[:, j])[:, None])
    return np.hstack([c if c.ndim == 2 else c[:, None] for c in cols])

def cluster_boot(df, xcol, ycol, Zcols, n=4000, seed=99, deg=1):
    eps = df.episode_id.unique()
    rs = np.random.default_rng(seed)
    idx_by_ep = {e: np.flatnonzero(df.episode_id.values == e) for e in eps}
    out = []
    for _ in range(n):
        pick = rs.choice(eps, len(eps), replace=True)
        rows = np.concatenate([idx_by_ep[e] for e in pick])
        s = df.iloc[rows]
        Z = s[Zcols].values
        if deg > 1:
            Z = expand(Z, deg)
        try:
            out.append(partial_r(s[xcol].values, s[ycol].values, Z))
        except Exception:
            pass
    out = np.array([v for v in out if np.isfinite(v)])
    return np.percentile(out, [2.5, 97.5]), out.mean()

def build_chain(df, sevcol, statecols):
    a = df.sort_values(["episode_id", "t"]).copy()
    b = a[["episode_id", "t", "rho_actual", sevcol]].copy()
    b["t"] -= 1
    b = b.rename(columns={"rho_actual": "rho_next", sevcol: "sev_next"})
    m = a.merge(b, on=["episode_id", "t"], how="inner")
    return m.dropna(subset=["rho_next", "sev_next", "rho_actual", sevcol] + statecols).reset_index(drop=True)

CFG = [
    ("C  braking", "scenario_c_braking.csv", "sev", ["speed_t1", "yaw_t1", "x_t1", "y_t1"]),
    ("A  lanechange", "scenario_a_lanechange.csv", "severity_t",
     ["lat_dev_t1", "heading_err_t1", "speed_risk_t1", "speed_t1", "yaw_t1"]),
    ("D  control", "scenario_d_baseline.csv", "speed_t", ["speed_t1", "yaw_t1", "x_t1", "y_t1"]),
]

print("="*104)
print("THREAT 1 + 2: non-linear conditioning, and episode-clustered bootstrap CI (no exchangeability needed)")
print("="*104)
print(f"{'scenario':<16}{'deg':>4}{'partial r':>12}{'boot 95% CI':>26}{'n':>7}{'eps':>6}")
store = {}
for name, f, sev, st in CFG:
    df = load(f)
    if sev == "sev":
        df["sev"] = df.brake_intensity_t * df.brake_duration_ticks_t
    m = build_chain(df, sev, st)
    Zc = st + ["rho_actual", "sev_next"]
    for deg in (1, 2, 3):
        Z = m[Zc].values
        Zx = Z if deg == 1 else expand(Z, deg)
        r = partial_r(m[sev].values, m["rho_next"].values, Zx)
        ci, _ = cluster_boot(m, sev, "rho_next", Zc, deg=deg)
        flag = "" if (ci[0] < 0 < ci[1]) else "  <-- CI excludes 0"
        print(f"{name:<16}{deg:>4}{r:>+12.4f}   [{ci[0]:+.4f}, {ci[1]:+.4f}]{'':>4}{len(m):>7}{m.episode_id.nunique():>6}{flag}")
    store[name] = (m, sev, st)
    print()

print("="*104)
print("THREAT 1b: is the C residual just the cruise phase failing to reset speed?")
print("="*104)
mc, _, _ = store["C  braking"]
print("  If severity at T still predicts the speed the vehicle CARRIES INTO the next brake,")
print("  the propagation is through an ordinary (but unlogged-at-T+1) state variable.")
print(f"  r(sev_T, speed_t of cycle T+1)                     = {np.corrcoef(mc.sev, mc.speed_t_y if 'speed_t_y' in mc else mc.sev*0)[0,1]:+.4f}" if False else "")
c = load("scenario_c_braking.csv"); c["sev"] = c.brake_intensity_t*c.brake_duration_ticks_t
cc = c.sort_values(["episode_id","t"]).copy()
nxt = cc[["episode_id","t","speed_t"]].copy(); nxt["t"] -= 1
nxt = nxt.rename(columns={"speed_t":"speed_t_next_cycle"})
mm = cc.merge(nxt, on=["episode_id","t"], how="inner")
print(f"  r(sev_T, speed at start-of-brake of cycle T+1)      = {np.corrcoef(mm.sev, mm.speed_t_next_cycle)[0,1]:+.4f}")
print(f"  r(speed_t1_T, speed_t of cycle T+1)                 = {np.corrcoef(mm.speed_t1, mm.speed_t_next_cycle)[0,1]:+.4f}")
print(f"  partial r(sev_T, speed_t of T+1 | speed_t1_T)       = {partial_r(mm.sev.values, mm.speed_t_next_cycle.values, mm[['speed_t1']].values):+.4f}")
print("  ^ if this is non-zero, severity leaves a speed trace that survives past the logged T+1 speed")

print("\n" + "="*104)
print("THREAT 3: effect direction and magnitude, stated honestly")
print("="*104)
for name in ["C  braking", "A  lanechange", "D  control"]:
    m, sev, st = store[name]
    Zc = st + ["rho_actual", "sev_next"]
    y = m["rho_next"].values
    Zb = np.c_[np.ones(len(y)), m[Zc].values]
    Zf = np.c_[Zb, m[sev].values]
    def r2(X):
        e = y - X @ la.lstsq(X, y, rcond=None)[0]
        return 1 - e.var()/y.var()
    print(f"  {name:<16} incremental R^2 from severity_T = {r2(Zf)-r2(Zb):+.5f}"
          f"   (baseline R^2 = {r2(Zb):.3f})")
