"""STEP 4b - synthetic bridge with HETEROGENEOUS drivers (Amendment I.4).

Step 4 validated the estimator only for synthetic populations where every driver shares
one theta. Real highD drivers do not (DIARY Entry 29). This bridge validates the
Amendment I primary inference - the pooled spacing-MSE estimator theta_KL with a
driver-bootstrap-calibrated criterion region - on synthetic populations in which each
driver has their own time headway T_i and standstill gap s0_i, behind REAL highD leader
traces (location-1 E10,1.0 pool from 06_bridge_pools.py).

Per cell and rep: draw N episodes, give each a driver (T_i, s0_i), start at that driver's
own equilibrium gap, add AR(1) noise, fit theta_hat (MSE), compute D(theta_KL), and
calibrate the threshold by resampling whole episodes (B = 49). theta_KL per heterogeneity
level is fitted once on the full pool x 4 driver draws.

Order: theta_KL fits -> positive control H0-518 (stop if its bootstrap coverage CI excludes
0.95, I-D1) -> the rest, longest first. Resumable (JSONL per cell).

Usage (from AINN_ReWORK/scripts):
  python 09_bridge_het.py --workers 16
  python 09_bridge_het.py --summarise
  python 09_bridge_het.py --quick --workers 2
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from _common import DERIVED, RESULTS, SRC  # noqa: F401

import bridge as br                                    # noqa: E402
from idm import PARAM_NAMES                            # noqa: E402

SEED = 20260928
POOL_FILE = os.path.join(DERIVED, "bridge_pool_defE.npz")
BASE = np.array([38.0, 1.2, 2.0, 1.0, 1.5])
NOISE_SD, NOISE_RHO = 0.5, 0.99
KL_REPLICAS = 4

CELLS = [dict(id="H0-518", sT=0.0, ss=0.0, n=518, R=100, B=49),
         dict(id="HL-518", sT=0.25, ss=0.4, n=518, R=100, B=49),
         dict(id="HM-518", sT=0.45, ss=0.4, n=518, R=100, B=49),
         dict(id="HH-518", sT=0.65, ss=0.4, n=518, R=100, B=49),
         dict(id="HM-200", sT=0.45, ss=0.4, n=200, R=100, B=49),
         dict(id="HM-1000", sT=0.45, ss=0.4, n=1000, R=100, B=49),
         dict(id="HM-5691", sT=0.45, ss=0.4, n=5691, R=50, B=49)]
CELL = {c["id"]: c for c in CELLS}
_P = {}
_KL = {}


def level(c):
    return f"{c['sT']:g}_{c['ss']:g}"


def _apply_quick():
    for c in CELLS:
        c.update(R=2, B=2, n=min(c["n"], 60))


def _init(pool_file, kl, quick):
    if quick:
        _apply_quick()
    z = np.load(pool_file)
    _P.update({k: z[k] for k in ("ul", "vl", "off", "vF0", "pool_loc1")})
    _KL.update(kl or {})


def _gather(idx):
    off_all = _P["off"]
    lens = off_all[idx + 1] - off_all[idx]
    off = np.r_[0, np.cumsum(lens)].astype(np.int64)
    sel = np.concatenate([np.arange(off_all[i], off_all[i + 1]) for i in idx])
    return _P["ul"][sel], _P["vl"][sel], off, _P["vF0"][idx]


def _population(idx, sT, ss, rng):
    """Synthetic heterogeneous population on the given traces; returns an MSE Sample."""
    ul, vl, off, v0 = _gather(idx)
    M = len(idx)
    th = np.tile(BASE, (M, 1))
    th[:, 1] = np.clip(BASE[1] * np.exp(sT * rng.standard_normal(M)), br.LO[1], br.HI[1])
    th[:, 2] = np.clip(BASE[2] * np.exp(ss * rng.standard_normal(M)), br.LO[2], br.HI[2])
    q = np.minimum(v0 / th[:, 0], 0.95)
    s_eq = (th[:, 2] + v0 * th[:, 1]) / np.sqrt(1.0 - q ** 4)
    s_true = np.empty(off[-1])
    br._sim_idm_het(ul, vl, off, ul[off[:-1]] - s_eq, v0, th, br.DT, br.S_FLOOR, s_true)
    s_obs = s_true + br._ar1_filter(rng.standard_normal(off[-1]), off, NOISE_RHO, NOISE_SD)
    S = br.Sample(ul, vl, off, v0, s_obs)
    S.init_mode = "mse"
    return S


def task_kl(lev, sT, ss, n_cap):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 7, zlib.crc32(lev.encode())]))
    pool = _P["pool_loc1"][:n_cap]
    idx = np.concatenate([pool] * KL_REPLICAS)
    S = _population(idx, sT, ss, rng)
    f = br.fit(S, rng, n_random=4)
    return lev, {"theta_kl": list(f["theta"]), "at_bound": f["at_bound"], "M": S.M,
                 "rms": float(np.sqrt(np.exp(-2 * f["ll"] / S.M)))}


def run_task(cid, rep):
    c = CELL[cid]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32(cid.encode()), rep]))
    pool = _P["pool_loc1"]
    idx = pool if c["n"] >= len(pool) else np.sort(rng.choice(pool, c["n"], replace=False))
    S = _population(idx, c["sT"], c["ss"], rng)
    kl = np.asarray(_KL[level(c)]["theta_kl"])
    t0 = time.time()
    f = br.fit(S, rng, n_random=2)
    t_fit = time.time() - t0
    D = 2.0 * (f["ll"] - br.loglik(S, kl)[0])
    t0 = time.time()
    thr, lrs = br.bootstrap_threshold(S, f, c["B"], rng)
    return {"cell": cid, "rep": rep, "M": S.M, "theta_hat": [float(x) for x in f["theta"]],
            "at_bound": f["at_bound"], "D": float(D), "c_boot": thr,
            "cov_boot": bool(D <= thr), "cov_chi2": bool(D <= br.CHI2_K),
            "t_fit": round(t_fit, 2), "t_boot": round(time.time() - t0, 1)}


def cp(k, n, a=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def load_done(out):
    d = {}
    for f in glob.glob(os.path.join(out, "*.jsonl")):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                d.setdefault(r["cell"], {})[r["rep"]] = r
    return d


def pool_run(fn, args, workers, kl, quick, out, label, write=True):
    res = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(POOL_FILE, kl, quick)) as ex:
        futs = {ex.submit(fn, *x): x for x in args}
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except BrokenProcessPool:
                sys.exit("STOP: a worker died (memory?). Re-run with fewer --workers.")
            except Exception as e:
                print(f"  ERROR {futs[fu]}: {e!r}", flush=True)
                continue
            if write:
                with open(os.path.join(out, f"{r['cell']}.jsonl"), "a") as fh:
                    fh.write(json.dumps(r) + "\n")
            res.append(r)
            if n % 25 == 0 or n == len(args):
                el = time.time() - t0
                print(f"  [{label}] {n}/{len(args)}  {el / 60:.1f} min, "
                      f"~{el / n * (len(args) - n) / 60:.0f} min left", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 4)))
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = os.path.join(RESULTS, "step4b_bridge_het_quick" if a.quick else "step4b_bridge_het")
    os.makedirs(out, exist_ok=True)
    if a.quick:
        _apply_quick()
    if a.summarise:
        return summarise(out)

    # theta_KL per heterogeneity level
    klp = os.path.join(out, "theta_kl.json")
    kl = json.load(open(klp)) if os.path.exists(klp) else {}
    need = {level(c): (c["sT"], c["ss"]) for c in CELLS if level(c) not in kl}
    if need:
        n_cap = 150 if a.quick else 10 ** 9
        res = pool_run(task_kl, [(lv, sT, ss, n_cap) for lv, (sT, ss) in need.items()],
                       min(a.workers, len(need)), None, a.quick, out, "theta_KL", write=False)
        kl.update({lv: v for lv, v in res})
        json.dump(kl, open(klp, "w"), indent=2)
    for lv, v in kl.items():
        print(f"theta_KL[{lv}] = {np.round(v['theta_kl'], 4).tolist()}  rms {v['rms']:.2f} m  "
              f"at bound {v['at_bound']}", flush=True)

    done = load_done(out)
    todo = lambda cid: [(cid, r) for r in range(CELL[cid]["R"]) if r not in done.get(cid, {})]

    # positive control first (I-D1)
    pool_run(run_task, todo("H0-518"), a.workers, kl, a.quick, out, "H0-518")
    recs = list(load_done(out).get("H0-518", {}).values())
    k = sum(r["cov_boot"] for r in recs)
    ci = cp(k, len(recs))
    ok = ci[0] <= 0.95 <= ci[1]
    print(f"positive control H0-518: bootstrap coverage {k / len(recs):.3f} CI {ci}: "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    if not ok and not a.quick:
        sys.exit("STOP: positive control failed (I-D1).")

    tasks = [t for c in CELLS if c["id"] != "H0-518" for t in todo(c["id"])]
    tasks.sort(key=lambda t: CELL[t[0]]["n"], reverse=True)
    pool_run(run_task, tasks, a.workers, kl, a.quick, out, "bridge-het")
    return summarise(out)


def summarise(out):
    done = load_done(out)
    kl = json.load(open(os.path.join(out, "theta_kl.json")))
    rows = {}
    L = ["# Step 4b - heterogeneous-driver bridge (Amendment I.4)", "",
         f"base theta = {BASE.tolist()} (v0, T, s0, a, b); noise AR(1) sd {NOISE_SD} m, rho {NOISE_RHO}", "",
         "theta_KL by heterogeneity level (sigma_T_sigma_s) and its shift from base (I-D3):", ""]
    for lv, v in kl.items():
        L.append(f"- {lv}: {np.round(v['theta_kl'], 4).tolist()}  relative shift "
                 f"{np.round(np.asarray(v['theta_kl']) / BASE - 1, 3).tolist()}  at bound {v['at_bound']}")
    L += ["", "| cell | reps | N | boot coverage [CP] | chi2_5 coverage [CP] | median c_boot | "
              "at-bound fits | rel. bias vs theta_KL (v0,T,s0,a,b) |", "|---|---|---|---|---|---|---|---|"]
    for c in CELLS:
        recs = list(done.get(c["id"], {}).values())
        if not recs:
            continue
        n = len(recs)
        kb, kc = sum(r["cov_boot"] for r in recs), sum(r["cov_chi2"] for r in recs)
        th = np.array([r["theta_hat"] for r in recs])
        t_kl = np.asarray(kl[level(c)]["theta_kl"])
        ab = sum(bool(r["at_bound"]) for r in recs) / n
        rows[c["id"]] = {"reps": n, "boot_cov": kb / n, "boot_ci": cp(kb, n), "chi2_cov": kc / n,
                         "chi2_ci": cp(kc, n), "c_boot_median": float(np.median([r["c_boot"] for r in recs])),
                         "at_bound_rate": ab, "rel_bias": (np.median(th, 0) / t_kl - 1).tolist()}
        s = rows[c["id"]]
        L.append(f"| {c['id']} | {n} | {recs[0]['M']} | {s['boot_cov']:.3f} {s['boot_ci']} | "
                 f"{s['chi2_cov']:.3f} {s['chi2_ci']} | {s['c_boot_median']:.1f} | {ab:.2f} | "
                 f"{np.round(s['rel_bias'], 4).tolist()} |")
    json.dump({"theta_kl": kl, "cells": rows}, open(os.path.join(out, "summary.json"), "w"), indent=2)
    md = "\n".join(L) + "\n"
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
