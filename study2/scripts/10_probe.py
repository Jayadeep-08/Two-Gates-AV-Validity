"""STEP 5-S1 - real-data diagnostic probe for B-lite (Amendment J, S1; selection rule J.1).

Purpose: MEASURE the structure that the synthetic generator (S2) must reproduce, before any
coverage bridge is designed. Its estimates of the typical driver are design inputs, NOT
results, and are never reported as such.

Probe set: 300 location-1 E10,1.0 episodes drawn with seed 20260929 (fixed in Amendment J).

Steps
  1  B-lite fits for per = {T}, {T,s0}, {T,a}, {T,b} (J.1 step 2)
  2  J.1 rule: identifiable (<20% at bounds, T not pushed onto bounds) and useful (median
     per-episode RMS reduction >= 0.25 vs {T}); at most one parameter added
  3  S1 metrics under the SELECTED model:
       a per-driver parameter quantiles, IQR, fraction at bounds, rank correlation (if 2)
       b per-episode residual RMS: median, p90
       c residual autocorrelation at 1, 25, 50, 125 steps (0.04, 1, 2, 5 s), within episodes
       d growth of |residual| with time into the episode: mean |r| per time bin, slope
       e start disequilibrium: (s_obs(0) - s_eq(v_F(0); the driver's own parameters)) / s_eq
  4  option-A contrast: pooled theta_KL on the same probe, and its RMS

Writes code/results/step5_probe/probe.{json,md} (aggregates only) and, for S2, the per-driver
estimates of the selected model to derived/step5_probe_perdriver.npz (stays on this machine).
Usage (from AINN_ReWORK/scripts):  python 10_probe.py
                                   python 10_probe.py --quick   (40 episodes, smoke test only)
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # BLAS single-threaded; numba prange uses all cores

import glob
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import ARRAY_ROWS, DERIVED, RESULTS, SRC  # noqa: F401

import bridge as br                                     # noqa: E402

PROBE_SEED, PROBE_N = 20260929, 300
R = {k: i for i, k in enumerate(ARRAY_ROWS)}
LAGS = (1, 25, 50, 125)
TBINS = (0, 2, 5, 10, 20, 1e9)
MODELS = {"T": (1,), "T+s0": (1, 2), "T+a": (1, 3), "T+b": (1, 4)}
GAIN_MIN = 0.25                        # J.1 step 4; step 3 thresholds (20%, +5 pp) in j1_select


def load_probe(n=PROBE_N):
    d = os.path.join(DERIVED, "highd_episodes_defE")
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "??_index.csv"))):
        if os.path.getsize(f) <= 1:
            continue
        ix = pd.read_csv(f)
        ix = ix[(ix["leader_class"] == "car") & (ix["dur_s"] >= 10) & (ix["sd_vL"] >= 1.0)
                & (ix["location"] == 1)]
        rows += [(f, ep) for ep in ix["episode"]]
    pick = np.sort(np.random.default_rng(PROBE_SEED).choice(len(rows), n, replace=False))
    ul, vl, gap, v0, lens, cache = [], [], [], [], [], {}
    for i in pick:
        f, ep = rows[i]
        if f not in cache:
            cache = {f: np.load(f.replace("_index.csv", "_episodes.npz"))}
        a = cache[f][ep]
        ul.append(a[R["u_rL"]]); vl.append(a[R["vL"]]); gap.append(a[R["gap"]])
        v0.append(a[R["vF"], 0]); lens.append(a.shape[1])
    off = np.r_[0, np.cumsum(lens)].astype(np.int64)
    return (np.concatenate(ul), np.concatenate(vl), off, np.array(v0), np.concatenate(gap),
            len(rows))


def j1_select(fits):
    """Amendment J.1 steps 3-5. Returns (selected model name, per-candidate table)."""
    base = fits["T"]
    M = len(base["ep_rms"])
    nT0 = base["per_n_at_bound"]["T"]
    table = {}
    for name in ("T+s0", "T+a", "T+b"):
        f = fits[name]
        j = br.PARAM_NAMES[f["per_idx"][1]]
        gain = float(np.median(1.0 - f["ep_rms"] / np.maximum(base["ep_rms"], 1e-12)))
        # integer counts, so "below" is exact at the thresholds (S5 review finding 4);
        # 20 * n < 4 * M  <=>  n / M < 0.20 ;  20 * nT < max(4 * M, 20 * nT0 + M)  <=>  T slack rule
        ident = (20 * f["per_n_at_bound"][j] < 4 * M) and \
                (20 * f["per_n_at_bound"]["T"] < max(4 * M, 20 * nT0 + M))
        table[name] = {"param": j, "at_bound": f["per_at_bound"][j],
                       "T_at_bound": f["per_at_bound"]["T"], "identifiable": bool(ident),
                       "median_rms_reduction": gain, "useful": bool(gain >= GAIN_MIN)}
    ok = [n for n, t in table.items() if t["identifiable"] and t["useful"]]
    sel = max(ok, key=lambda n: table[n]["median_rms_reduction"]) if ok else "T"
    return sel, table


def main():
    quick = "--quick" in sys.argv
    out = os.path.join(RESULTS, "step5_probe_quick" if quick else "step5_probe")
    os.makedirs(out, exist_ok=True)
    ul, vl, off, v0, gap, n_pop = load_probe(40 if quick else PROBE_N)
    print(f"probe: {len(v0)} of {n_pop} location-1 E10,1.0 episodes, "
          f"{off[-1] * br.DT / 60:.1f} min", flush=True)
    br.BliteSample(ul[:off[2]], vl[:off[2]], off[:3], v0[:2], gap[:off[2]]).criterion(
        br.GENERIC_START[[0, 2, 3, 4]])                                   # compile once

    # 1  four B-lite fits
    fits, samples = {}, {}
    for name, per in MODELS.items():
        S = br.BliteSample(ul, vl, off, v0, gap, per=per)
        t0 = time.time()
        f = br.fit_blite(S, np.random.default_rng(PROBE_SEED), n_random=2)
        fits[name], samples[name] = f, S
        print(f"B-lite {name:5s} {time.time() - t0:6.0f} s  shared "
              f"{dict(zip([br.PARAM_NAMES[j] for j in f['shared_idx']], np.round(f['shared'], 3).tolist()))}"
              f"  rms {f['rms']:.3f} m  per at bound {f['per_at_bound']}"
              f"  shared at bound {f['shared_at_bound']}", flush=True)

    # 2  J.1 selection
    sel, table = j1_select(fits)
    print(f"J.1 table: {json.dumps(table)}", flush=True)
    print(f"J.1 SELECTED: per = {sel}", flush=True)
    t_flag = 20 * fits["T"]["per_n_at_bound"]["T"] >= 4 * len(v0)
    if t_flag:
        print("J.1 step 6 FLAG: >= 20% of per-driver T at a bound in the {T} model "
              "(reported as a finding; T stays per-driver)", flush=True)

    # 3  S1 metrics under the selected model
    S, f = samples[sel], fits[sel]
    th = br.blite_thetas(S, f)
    r = br.blite_residuals(S, f)
    ep_rms = f["ep_rms"]
    num = np.zeros(len(LAGS))
    den = 0.0
    tt, ar = [], []
    for e in range(S.M):
        x = r[S.off[e]:S.off[e + 1]]
        den += float(np.dot(x, x))
        for j, L in enumerate(LAGS):
            if len(x) > L:
                num[j] += float(np.dot(x[L:], x[:-L]))
        tt.append(np.arange(len(x)) * br.DT)
        ar.append(np.abs(x))
    tt, ar = np.concatenate(tt), np.concatenate(ar)
    bins = {(f"{TBINS[i]:g}-{TBINS[i + 1]:g}s" if TBINS[i + 1] < 1e8 else f">{TBINS[i]:g}s"): float(ar[(tt >= TBINS[i]) & (tt < TBINS[i + 1])].mean())
            for i in range(len(TBINS) - 1) if ((tt >= TBINS[i]) & (tt < TBINS[i + 1])).any()}
    slope = float(np.polyfit(tt, ar, 1)[0])
    q = np.minimum(v0 / th[:, 0], 0.95)
    s_eq = (th[:, 2] + v0 * th[:, 1]) / np.sqrt(1.0 - q ** 4)
    diseq = (gap[off[:-1]] - s_eq) / s_eq
    qq = lambda x: np.round(np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9]), 4).tolist()
    per_names = [br.PARAM_NAMES[j] for j in f["per_idx"]]
    perdrv = {n: {"q10_25_50_75_90": qq(f["P"][:, p]),
                  "iqr": float(np.subtract(*np.quantile(f["P"][:, p], [0.75, 0.25]))),
                  "log_mean": float(np.mean(np.log(f["P"][:, p]))),
                  "log_sd": float(np.std(np.log(f["P"][:, p]), ddof=1)),
                  "at_bound": f["per_at_bound"][n]} for p, n in enumerate(per_names)}
    rank_corr = (float(spearmanr(f["P"][:, 0], f["P"][:, 1]).correlation)
                 if len(per_names) == 2 else None)

    # 4  option-A contrast
    SA = br.Sample(ul, vl, off, v0, gap)
    SA.init_mode = "mse"
    t0 = time.time()
    fa = br.fit(SA, np.random.default_rng(PROBE_SEED), n_random=2)
    print(f"option A {time.time() - t0:.0f} s", flush=True)

    res = {"note": "S1 probe (Amendment J, J.1). Estimates are design inputs, NOT results.",
           "probe_seed": PROBE_SEED, "episodes": int(S.M), "population": n_pop,
           "quick_smoke_test": quick,
           "fits": {n: {"shared": dict(zip([br.PARAM_NAMES[j] for j in g["shared_idx"]],
                                           np.round(g["shared"], 4).tolist())),
                        "theta_typ": np.round(g["theta_typ"], 4).tolist(),
                        "rms": g["rms"], "ep_rms_median": float(np.median(g["ep_rms"])),
                        "per_at_bound": g["per_at_bound"], "shared_at_bound": g["shared_at_bound"],
                        "opt_message": g["opt_message"]}
                    for n, g in fits.items()},
           "j1_table": table, "j1_selected": sel, "j1_T_at_bound_flag": bool(t_flag),
           "selected_theta_typ": np.round(f["theta_typ"], 4).tolist(),
           "per_driver": perdrv, "per_driver_spearman": rank_corr,
           "ep_rms_median": float(np.median(ep_rms)), "ep_rms_p90": float(np.quantile(ep_rms, 0.9)),
           "acf_lags_steps": list(LAGS), "acf": np.round(num / den, 4).tolist(),
           "abs_resid_by_time": bins, "abs_resid_slope_m_per_s": slope,
           "start_diseq_q10_50_90": np.round(np.quantile(diseq, [0.1, 0.5, 0.9]), 4).tolist(),
           "optionA_theta_kl": np.round(fa["theta"], 4).tolist(),
           "optionA_rms": float(np.exp(-fa["ll"] / SA.M)), "optionA_at_bound": fa["at_bound"]}
    json.dump(res, open(os.path.join(out, "probe.json"), "w"), indent=2)
    L = ["# S1 probe (Amendment J, J.1) - design inputs, not results", ""]
    L += [f"- {k}: {json.dumps(v)}" for k, v in res.items()]
    open(os.path.join(out, "probe.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    if not quick:                                     # per-driver values for S2; never leaves derived/
        np.savez(os.path.join(DERIVED, "step5_probe_perdriver.npz"), per_idx=np.array(f["per_idx"]),
                 P=f["P"], theta_typ=f["theta_typ"], selected=sel)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
