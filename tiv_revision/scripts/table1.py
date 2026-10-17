"""Table I numbers + lag-sweep series for the T-IV revision.

READ-ONLY on the banked CSVs. Writes only under tiv_revision/.
Run:  py -3.7 tiv_revision\\scripts\\table1.py
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RPI_OUT") or os.path.abspath(os.path.join(HERE, ".."))
D = os.environ.get("RPI_DATA") or os.path.join(ROOT, "..", "data")
D = os.path.join(D, "")
OUT = os.path.join(ROOT, "")
os.makedirs(os.path.join(ROOT, "tables"), exist_ok=True)
RNG = np.random.default_rng(42)
NB, NP = 10000, 10000


def _r(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float(x @ y / np.sqrt((x @ x) * (y @ y)))


def resid(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def pcorr_np(x, y, C):
    return _r(resid(x, C), resid(y, C))


def pcorr(x, y, ctrls):
    C = np.column_stack([np.asarray(c, float) for c in ctrls])
    return pcorr_np(np.asarray(x, float), np.asarray(y, float), C)


def boot_ci(x, y, n=NB):
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    xs, ys = x[idx], y[idx]
    xs = xs - xs.mean(1, keepdims=True); ys = ys - ys.mean(1, keepdims=True)
    rs = (xs * ys).sum(1) / np.sqrt((xs ** 2).sum(1) * (ys ** 2).sum(1))
    return float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5))


def perm_p(x, y, n=NP):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r0 = abs(_r(x, y))
    xs = np.array([RNG.permutation(x) for _ in range(n)])
    xs = xs - xs.mean(1, keepdims=True)
    yc = y - y.mean()
    rs = np.abs((xs * yc).sum(1) / np.sqrt((xs ** 2).sum(1) * (yc @ yc)))
    return float((np.sum(rs >= r0) + 1) / (n + 1))


def common_outcome(speed):
    """The one outcome computable from every CSV: inverse speed at T+1."""
    return np.maximum(0.0, 1.0 - np.asarray(speed, float) / 15.0)


# ------------------------------------------------------------------ load
c = pd.read_csv(D + "scenario_c_braking.csv").dropna(subset=["rho_actual"])
c["sev"] = c.brake_intensity_t * c.brake_duration_ticks_t
a = pd.read_csv(D + "scenario_a_lanechange.csv").dropna(subset=["rho_actual"])
a["sev"] = a.severity_t
b = pd.read_csv(D + "scenario_b_intersection.csv").dropna(subset=["rho_actual"])
b["sev"] = b.yield_speed_target_t
d = pd.read_csv(D + "scenario_d_baseline.csv").dropna(subset=["rho_actual"])
d["sev"] = d.speed_t  # pseudo-severity: D has no scripted decision
for df in (c, a, b, d):
    df["common"] = common_outcome(df.speed_t1)

SCEN = [
    ("C", "Emergency braking", c, "brake intensity $\\times$ duration", "speed_t1", True),
    ("A", "Lane change", a, "steer magnitude $\\times$ ticks", "lat_dev_t1", True),
    ("B", "Intersection yield-cross", b, "realised yield speed", "speed_t", False),
    ("D", "Open-road lane keep", d, "none (speed$_t$ as pseudo-severity)", "speed_t1", False),
]

# ------------------------------------------------------------------ Table I
rows = []
print("=" * 104)
print("TABLE I  -  two treatments, two no-treatment conditions")
print("=" * 104)
for key, name, df, sevdesc, ctrl, is_treat in SCEN:
    g = df.groupby("episode_id").agg(
        s=("sev", "mean"), o=("rho_actual", "mean"),
        cm=("common", "mean"), ct=(ctrl, "mean")).dropna()
    r = _r(g.s.to_numpy(float), g.o.to_numpy(float))
    lo, hi = boot_ci(g.s, g.o)
    p = perm_p(g.s, g.o)
    r_common = _r(g.s.to_numpy(float), g.cm.to_numpy(float))
    r_part = pcorr(g.s, g.o, [g.ct])
    coll = _r(df.sev.to_numpy(float), df[ctrl].to_numpy(float))
    rows.append(dict(key=key, name=name, sev=sevdesc, treatment=is_treat,
                     n_ep=int(df.episode_id.nunique()), n_pairs=int(len(df)),
                     rho_mean=float(df.rho_actual.mean()), rho_sd=float(df.rho_actual.std()),
                     r=r, ci_lo=lo, ci_hi=hi, perm_p=p, r_common=r_common,
                     ctrl=ctrl, r_partial=r_part, collinearity=coll))
    print(f"\n {key}  {name}   [{'TREATMENT' if is_treat else 'NO INDEPENDENT TREATMENT'}]")
    print(f"   n                     {df.episode_id.nunique()} ep, {len(df)} pairs")
    print(f"   mean rho (sd)         {df.rho_actual.mean():.4f} ({df.rho_actual.std():.4f})")
    print(f"   r(sev, rho_T+1)       {r:+.4f}  CI[{lo:+.3f}, {hi:+.3f}]  perm p={p:.4f}")
    print(f"   r(sev, COMMON inv-sp) {r_common:+.4f}")
    print(f"   partial | {ctrl:<11} {r_part:+.4f}")
    print(f"   r(sev, {ctrl}) pair   {coll:+.4f}")

# ------------------------------------------------------------------ lag sweep
print("\n" + "=" * 104)
print("LAG SWEEP  -  r(severity at decision t, rho at decision t+k), episode level")
print("=" * 104)
lag = {}
for key, name, df, _, _, is_treat in SCEN:
    df = df.sort_values(["episode_id", "t"]).reset_index(drop=True)
    series = []
    for k in range(0, 8):
        sh = df.shift(-k)
        ok = (sh.episode_id == df.episode_id) & (sh.t == df.t + k)
        if ok.sum() < 60:
            break
        g = pd.DataFrame({"e": df.episode_id[ok].values, "s": df["sev"][ok].values,
                          "o": sh.rho_actual[ok].values}).groupby("e").mean()
        if len(g) < 20:
            break
        r = _r(g.s.to_numpy(float), g.o.to_numpy(float))
        lo, hi = boot_ci(g.s, g.o, 4000)
        series.append(dict(k=k, r=r, lo=lo, hi=hi,
                           n_pairs=int(ok.sum()), n_ep=int(len(g))))
    r0 = abs(series[0]["r"])
    for s in series:
        s["retention"] = abs(s["r"]) / r0
    lag[key] = series
    print(f"\n {key}  {name}  [{'treatment' if is_treat else 'no treatment'}]")
    for s in series:
        print(f"   k={s['k']}  r={s['r']:+.4f}  CI[{s['lo']:+.3f},{s['hi']:+.3f}]"
              f"  ret={s['retention']:.3f}  n_pairs={s['n_pairs']:>5}  n_ep={s['n_ep']}")

# ------------------------------------------------- episode-clustered conditional test
print("\n" + "=" * 104)
print("CONDITIONAL TEST  -  episode-clustered bootstrap on the pair-level partial")
print("=" * 104)
cc = c.sort_values(["episode_id", "t"]).reset_index(drop=True)
nx = cc.shift(-1)
m = (nx.episode_id == cc.episode_id) & (nx.t == cc.t + 1)
tt = pd.DataFrame({"episode_id": cc.episode_id[m].values, "sev_T": cc.sev[m].values,
                   "rho_T1": cc.rho_actual[m].values, "sev_T1": nx.sev[m].values,
                   "rho_T2": nx.rho_actual[m].values, "speed_T1": cc.speed_t1[m].values,
                   "yaw_T1": cc.yaw_t1[m].values, "lane_T1": cc.lane_id_t1[m].values,
                   "road_T1": cc.road_id_t1[m].values}).sort_values("episode_id").reset_index(drop=True)

SETS = [("kinematic state at T+1", ["speed_T1", "yaw_T1", "lane_T1", "road_T1"]),
        ("state + rho_T+1 + sev_T+1", ["speed_T1", "yaw_T1", "lane_T1", "road_T1", "rho_T1", "sev_T1"])]
epv = tt.episode_id.to_numpy()
ueps = np.unique(epv)
st = np.searchsorted(epv, ueps, "left"); en = np.searchsorted(epv, ueps, "right")
ep_idx = [np.arange(st[i], en[i]) for i in range(len(ueps))]
y_all = tt.rho_T2.to_numpy(float); x_all = tt.sev_T.to_numpy(float)
cond_out = {}
print(f"\n n={len(tt)} consecutive cycle pairs, {len(ueps)} episodes (clusters)")
print(f" unconditional r(sev_T, rho_T+2) = {_r(x_all, y_all):+.4f}")
for name, ks in SETS:
    C_all = tt[ks].to_numpy(float)
    pt = pcorr_np(x_all, y_all, C_all)
    reps = np.empty(2000)
    for i in range(2000):
        picks = RNG.integers(0, len(ueps), len(ueps))
        idx = np.concatenate([ep_idx[j] for j in picks])
        try:
            reps[i] = pcorr_np(x_all[idx], y_all[idx], C_all[idx])
        except Exception:
            reps[i] = np.nan
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    naive = 1.96 / np.sqrt(len(tt) - len(ks) - 3)
    cond_out[name] = dict(r=pt, lo=float(lo), hi=float(hi), n=len(tt), n_clusters=len(ueps),
                          naive_halfwidth=float(naive))
    print(f"   partial | {name:<28} r={pt:+.4f}   clustered 95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"   (naive +/-{naive:.4f})")

json.dump({"table1": rows, "lag": lag, "conditional": cond_out},
          open(os.path.join(ROOT, "numbers.json"), "w"), indent=1)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "tables", "table1.csv"), index=False)
pd.DataFrame([{**s, "scenario": k} for k, v in lag.items() for s in v]).to_csv(
    os.path.join(ROOT, "tables", "lag_sweep.csv"), index=False)
print("\nwrote numbers.json, tables/table1.csv, tables/lag_sweep.csv")
