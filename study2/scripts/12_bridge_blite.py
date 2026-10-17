"""STEP 5-S4 - heterogeneous synthetic bridge for B-lite (Amendment J.7).

Stages (each checkpointed; rerun the same command to resume):
  0  location-1 trace pool (5,691 episodes) -> derived/step5_loc1_pool.npz (LOCAL ONLY)
  1  pilot: bootstrap start-set rule (20 bootstrap samples of G0-300 rep 0) and timing
  2  pseudo-true theta* per arm: two fits on 4 x 5,691 synthetic drivers each
  3  PC-300 cell, then decision D5.1 (stops if it fails)
  4  all other cells: G0-300, G0-1000, G1-300, G2-300, G3-300, V-5691
  5  summary and decisions D5.1-D5.4 -> code/results/step5_bridge/summary.{json,md}

Replicates run in parallel worker processes; numba threads are split between them.
Only synthetic aggregates are written to code/results/step5_bridge/.

Usage (from AINN_ReWORK/scripts):
  python 12_bridge_blite.py                 # everything (resumable)
  python 12_bridge_blite.py --workers 8     # default: cores // 4 workers, cores // workers threads each
  python 12_bridge_blite.py --summary       # only recompute the summary from the saved lines
Keep the laptop plugged in with sleep off.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import multiprocessing
import hashlib
import json
import math
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd

from _common import ARRAY_ROWS, DERIVED, RESULTS, SRC  # noqa: F401

import bridge as br                                     # noqa: E402

OUT = os.path.join(RESULTS, "step5_bridge")
POOL_FILE = os.path.join(DERIVED, "step5_loc1_pool.npz")
GEN_FILE = os.path.join(RESULTS, "step5_generator", "generator.json")
RIDX = {k: i for i, k in enumerate(ARRAY_ROWS)}
B_BOOT, LEVEL, PT_COPIES = 99, 0.95, 4
PILOT_N, PILOT_TOL, PILOT_NEED, PILOT_CELL_N = 20, 1e-7, 19, 300   # pilot = G0-300 rep 0 draw
# cell id, arm, N, replicates, bootstrap size
CELLS = [("PC-300", "PC", 300, 100, B_BOOT),
         ("G0-300", "G0", 300, 100, B_BOOT), ("G0-1000", "G0", 1000, 60, B_BOOT),
         ("G1-300", "G1", 300, 60, B_BOOT), ("G2-300", "G2", 300, 60, B_BOOT),
         ("G3-300", "G3", 300, 60, B_BOOT), ("V-5691", "G0", 5691, 30, 0)]
PNAMES = list(br.PARAM_NAMES)
N_POOL = 5691
J6_G0 = {"sig_eta": 0.6, "tau": 4.0, "omega": 0.5, "kappa": 0.6}


def seed(label):
    return zlib.crc32(label.encode("utf-8"))


def arms_from_generator(gen):
    b = gen["best"]
    base = dict(sig_eta=b["sig_eta"], tau=b["tau"], omega=b["omega"], kappa=b["kappa"],
                t_df=None, sig_dT=0.0)
    return {"PC": dict(base, sig_eta=0.0),
            "G0": dict(base),
            "G1": dict(base, sig_eta=0.75),
            "G2": dict(base, t_df=3),
            "G3": dict(base, sig_dT=0.0447)}


# ---------------------------------------------------------------------------- pool
def build_pool():
    d = os.path.join(DERIVED, "highd_episodes_defE")
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "??_index.csv"))):
        if os.path.getsize(f) <= 1:
            continue
        ix = pd.read_csv(f)
        ix = ix[(ix["leader_class"] == "car") & (ix["dur_s"] >= 10) & (ix["sd_vL"] >= 1.0)
                & (ix["location"] == 1)]
        rows += [(f, ep) for ep in ix["episode"]]
    ul, vl, gap, gref, vf, lens, cache = [], [], [], [], [], [], {}
    for f, ep in rows:
        if f not in cache:
            cache = {f: np.load(f.replace("_index.csv", "_episodes.npz"))}
        a = cache[f][ep]
        ul.append(a[RIDX["u_rL"]]); vl.append(a[RIDX["vL"]]); gap.append(a[RIDX["gap"]])
        gref.append(a[RIDX["gap_ref"]]); vf.append(a[RIDX["vF"]]); lens.append(a.shape[1])
    off = np.r_[0, np.cumsum(lens)].astype(np.int64)
    gap, gref, vf = np.concatenate(gap), np.concatenate(gref), np.concatenate(vf)
    dmax = float(np.max(np.abs(gap - gref)))
    if dmax > 0.011:
        sys.exit(f"STOP - max |gap - dhw| = {dmax:.4f} m over the pool; F4 premise fails")
    np.savez(POOL_FILE, ul=np.concatenate(ul), vl=np.concatenate(vl), off=off,
             gap0=gap[off[:-1]], v0=vf[off[:-1]], dhw_add=br.dhw_add_from_real(gap, vf, off))
    return len(rows), dmax


def load_pool():
    z = np.load(POOL_FILE)
    return {k: z[k] for k in z.files}


def subset_pool(P, idx):
    lens = P["off"][idx + 1] - P["off"][idx]
    off = np.r_[0, np.cumsum(lens)].astype(np.int64)
    sel = np.concatenate([np.arange(P["off"][i], P["off"][i + 1]) for i in idx])
    return P["ul"][sel], P["vl"][sel], off, P["gap0"][idx], P["v0"][idx], P["dhw_add"][idx]


def generate(P, idx, arm, gen, label):
    ul, vl, off, gap0, v0, dadd = subset_pool(P, idx)
    s_obs, thetas, _, info = br.gen_synthetic(
        ul, vl, off, gap0, v0, gen["shared_theta"], gen["t_model"], arm["sig_eta"], arm["tau"],
        arm["omega"], arm["kappa"], seed(label), dadd, t_df=arm["t_df"], sig_dT=arm["sig_dT"])
    return br.BliteSample(ul, vl, off, v0, s_obs, per=(1,)), thetas, info


def bootstrap(S, f, B, label, mode):
    """Driver bootstrap (J.7). Resampling indices come from their own stream, so they are the
    same whichever start set is used; random starts (mode 'full') from a second stream."""
    r_idx = np.random.default_rng(seed(label + "/boot_idx"))
    r_st = np.random.default_rng(seed(label + "/boot_starts"))
    th_b, atb, Ls = [], [], []
    for _ in range(B):
        sb = S.subset(r_idx.integers(0, S.M, S.M))
        fb = br.fit_blite(sb, r_st, n_random=(2 if mode == "full" else 0), starts=(f["theta_typ"],))
        th_b.append(fb["theta_typ"]); atb.append(bool(fb["shared_at_bound"])); Ls.append(fb["L"])
    return np.array(th_b), np.array(atb), np.array(Ls)


def mahal(region, theta):
    d = np.asarray(theta, float) - region["center"]
    if np.any(np.abs(d[region["flat"]]) > 1e-9 * br.SPAN[region["flat"]]):
        return float("inf")
    return float(d @ np.linalg.pinv(region["cov"]) @ d)


# ---------------------------------------------------------------------------- worker
_P = None


def _init_worker(threads):
    global _P
    import numba
    numba.set_num_threads(threads)
    _P = load_pool()


def run_rep(cell, arm_name, arm, gen, N, r, B, mode, theta_star):
    t0 = time.time()
    lab = f"{cell}/rep{r}"
    M = len(_P["v0"])
    idx = np.sort(np.random.default_rng(seed(lab)).choice(M, N, replace=False))
    S, thetas, info = generate(_P, idx, arm, gen, lab + "/gen")
    f = br.fit_blite(S, np.random.default_rng(seed(lab + "/fit")), n_random=2)
    struct = np.array(gen["shared_theta"], float)
    struct[1] = float(np.median(thetas[:, 1]))
    rec = {"cell": cell, "arm": arm_name, "N": N, "rep": r, "theta_hat": f["theta_typ"].tolist(),
           "theta_star": list(theta_star), "theta_struct": struct.tolist(),
           "shared_at_bound": f["shared_at_bound"], "L": f["L"], "f4": info}
    if B > 0:
        th_b, atb, _ = bootstrap(S, f, B, lab, mode)
        reg = br.ellipsoid_region(f["theta_typ"], th_b, LEVEL)
        d_star = mahal(reg, theta_star)
        lo, hi = np.quantile(th_b, 0.025, axis=0), np.quantile(th_b, 0.975, axis=0)
        ts = np.asarray(theta_star, float)
        rec.update({"q": reg["q"], "q_emp": reg["q_emp"], "d_star": d_star,
                    "cover_ell": bool(d_star <= reg["q"]), "cover_ell_emp": bool(d_star <= reg["q_emp"]),
                    "theta_b": th_b.tolist(),
                    "flat": reg["flat_names"], "boot_sd": th_b.std(axis=0, ddof=1).tolist(),
                    "pct_lo": lo.tolist(), "pct_hi": hi.tolist(),
                    "cover_pct": [bool(a <= t <= b) for a, t, b in zip(lo, ts, hi)],
                    "boot_shared_at_bound_frac": float(atb.mean())})
    rec["seconds"] = time.time() - t0
    return rec


# ---------------------------------------------------------------------------- summary
def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def read_cell(cell):
    fn = os.path.join(OUT, "cells", f"{cell}.jsonl")
    if not os.path.exists(fn):
        return []
    recs = {}
    for line in open(fn, encoding="utf-8"):
        try:
            r = json.loads(line)
            recs[r["rep"]] = r
        except (json.JSONDecodeError, KeyError):
            continue                                  # a truncated last line
    return [recs[k] for k in sorted(recs)]


def cell_summary(cell, recs, theta_star, N, R):
    ts = np.asarray(theta_star, float)
    if any(not np.allclose(r["theta_star"], ts) for r in recs):
        raise SystemExit(f"STOP - {cell}: records were scored against a different theta*")
    H = np.array([r["theta_hat"] for r in recs])
    sd = H.std(axis=0, ddof=1) if len(H) > 1 else np.full(5, np.nan)
    bias = np.median(H, axis=0) - ts
    # J.8: model-based Monte Carlo error of theta* (2 x PT_COPIES copies of the 5,691 traces)
    mc = sd * math.sqrt(N / (2 * PT_COPIES * N_POOL))
    se = np.sqrt(1.57 * sd ** 2 / len(recs) + mc ** 2)       # SE of (median - theta*)
    with np.errstate(divide="ignore", invalid="ignore"):
        bos = np.abs(bias) / sd
    out = {"reps": len(recs), "planned": R, "complete": len(recs) == R,
           "median_theta_hat": np.median(H, 0).tolist(),
           "bias_median": bias.tolist(), "sd": sd.tolist(), "se_bias": se.tolist(),
           "mc_theta_star_model": mc.tolist(),
           "bias_over_sd": bos.tolist(),
           "rel_bias": (bias / ts).tolist(),
           "theta_struct_median": np.median([r["theta_struct"] for r in recs], 0).tolist(),
           "shared_at_bound_reps": int(sum(bool(r["shared_at_bound"]) for r in recs)),
           "f4_violating": int(sum(r["f4"]["f4_violating_final"] for r in recs)),
           "f4_accept_first_try": float(np.mean([r["f4"]["accept_first_try"] for r in recs])),
           "seconds_median": float(np.median([r["seconds"] for r in recs]))}
    if recs and "cover_ell" in recs[0]:
        k = sum(r["cover_ell"] for r in recs)
        ke = sum(r["cover_ell_emp"] for r in recs)
        out.update({"coverage_ell": k / len(recs), "coverage_ell_ci": wilson(k, len(recs)),
                    "coverage_ell_emp_rule": ke / len(recs),
                    "coverage_pct": np.mean([r["cover_pct"] for r in recs], 0).tolist(),
                    "boot_sd_over_sd": (np.mean([r["boot_sd"] for r in recs], 0) / sd).tolist(),
                    "reps_with_flat_direction": int(sum(bool(r["flat"]) for r in recs)),
                    "ratio_dstar_over_q": [r["d_star"] / r["q"] if r["q"] > 0 else float("inf")
                                           for r in recs]})
    return out


def includes95(ci):
    return ci[0] <= 0.95 <= ci[1]


def summarise(pt, pilot):
    S = {}
    for cell, arm, N, R, B in CELLS:
        recs = read_cell(cell)
        if recs:
            S[cell] = cell_summary(cell, recs, pt[arm]["theta_star"], N, R)
    D = {}
    fin = lambda v: bool(np.all(np.isfinite(np.asarray(v, float))))
    if "PC-300" in S:
        s = S["PC-300"]
        sd, se, bias = (np.array(s[k]) for k in ("sd", "se_bias", "bias_median"))
        tol = 0.25 * sd + 2.0 * se                               # J.8 D5.1
        D["D5.1"] = {"complete": s["complete"], "coverage_ci": s["coverage_ell_ci"],
                     "abs_bias": np.abs(bias).tolist(), "tolerance": tol.tolist(),
                     "pass": bool(s["complete"] and includes95(s["coverage_ell_ci"]) and fin(sd)
                                  and np.all(np.abs(bias) <= tol))}
    if "G0-300" in S and "G0-1000" in S:
        # "not validated" iff a CI excludes 0.95 from below; over-coverage is conservative
        ok = [not (S[c]["coverage_ell_ci"][1] < 0.95) for c in ("G0-300", "G0-1000")]
        d52 = {"complete": all(S[c]["complete"] for c in ("G0-300", "G0-1000")),
               "G0-300_ci": S["G0-300"]["coverage_ell_ci"], "G0-1000_ci": S["G0-1000"]["coverage_ell_ci"],
               "validated": bool(all(ok))}
        if not all(ok):
            rr = np.sort(np.array(S["G0-1000"]["ratio_dstar_over_q"], float))
            c = float(rr[int(math.ceil(0.95 * len(rr))) - 1])      # J.8: order statistic
            d52["calibration_factor_c"] = c
            d52["calibration_possible"] = bool(np.isfinite(c))
        D["D5.2"] = d52
        d53 = {}
        for c in ("G1-300", "G2-300", "G3-300"):
            if c not in S:
                d53[c] = {"holds": False, "note": "no records"}
                continue
            if d52["validated"]:
                d53[c] = {"complete": S[c]["complete"], "ci": S[c]["coverage_ell_ci"],
                          "holds": bool(not (S[c]["coverage_ell_ci"][1] < 0.95))}
            elif d52["calibration_possible"]:
                cov = float(np.mean(np.array(S[c]["ratio_dstar_over_q"], float) <= d52["calibration_factor_c"]))
                d53[c] = {"complete": S[c]["complete"], "calibrated_coverage": cov, "holds": bool(cov >= 0.90)}
            else:
                d53[c] = {"holds": False, "note": "calibration impossible (flat directions)"}
        d53["claim_holds_in_all_arms"] = bool(all(v["holds"] for v in d53.values() if isinstance(v, dict)))
        D["D5.3"] = d53
    if "V-5691" in S:
        s = S["V-5691"]
        sd, se, bias = (np.array(s[k]) for k in ("sd", "se_bias", "bias_median"))
        lower = (np.abs(bias) - 2.0 * se) / sd                   # J.8 D5.4
        upper = (np.abs(bias) + 2.0 * se) / sd
        D["D5.4"] = {"complete": s["complete"], "bias_over_sd": s["bias_over_sd"],
                     "ci_lower_over_sd": lower.tolist(), "ci_upper_over_sd": upper.tolist(),
                     "flag_bias_demonstrated": bool(fin(sd) and np.any(lower > 0.5)),
                     "theta_star_mc_empirical_over_sd": (np.array(pt["G0"]["mc_error"]) / sd).tolist()}
        trend = {c: S[c]["bias_over_sd"] for c in ("G0-300", "G0-1000", "V-5691") if c in S}
        D["D5.4"]["bias_over_sd_by_N"] = trend
    res = {"note": "S4 heterogeneous bridge (Amendment J.7). Synthetic aggregates only.",
           "pilot": pilot, "pseudo_true": pt, "cells": S, "decisions": D}
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=float)
    f3 = lambda v: "[" + ", ".join(f"{x:.3f}" for x in v) + "]"
    L = ["# S4 heterogeneous bridge for B-lite (Amendment J.7)", "",
         f"- bootstrap start set: **{pilot['mode']}** (pilot: {pilot['n_ok']}/{PILOT_N} within "
         f"{PILOT_TOL:g}; cheap {pilot['sec_cheap']:.1f} s vs full {pilot['sec_full']:.1f} s per refit)", "",
         "## Pseudo-true theta* per arm (v0, T, s0, a, b)", "",
         "| arm | theta* | MC error | structural | theta*/structural - 1 |", "|---|---|---|---|---|"]
    for a, p in pt.items():
        L.append(f"| {a} | {f3(p['theta_star'])} | {f3(p['mc_error'])} | {f3(p['theta_struct'])} | "
                 f"{f3(np.array(p['theta_star']) / np.array(p['theta_struct']) - 1)} |")
    L += ["", "## Cells", "",
          "| cell | reps | ellipsoid coverage [Wilson CI] | (J in-sample rule) | percentile coverage per param | abs(bias)/sd per param | boot sd / sd | shared at bound | s per rep |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c, s in S.items():
        cov = (f"{s['coverage_ell']:.3f} [{s['coverage_ell_ci'][0]:.3f}, {s['coverage_ell_ci'][1]:.3f}]"
               if "coverage_ell" in s else "-")
        L.append(f"| {c} | {s['reps']}/{s['planned']} | {cov} | "
                 f"{s['coverage_ell_emp_rule']:.3f} | " if "coverage_ell" in s else f"| {c} | {s['reps']}/{s['planned']} | - | - | ")
        L[-1] += (f"{f3(s['coverage_pct']) if 'coverage_pct' in s else '-'} | "
                  f"{f3(s['bias_over_sd'])} | {f3(s['boot_sd_over_sd']) if 'boot_sd_over_sd' in s else '-'} | "
                  f"{s['shared_at_bound_reps']} | {s['seconds_median']:.0f} |")
    L += ["", "## Decisions", ""] + [f"- **{k}**: {json.dumps(v, default=float)}" for k, v in D.items()]
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return D


# ---------------------------------------------------------------------------- main
def fingerprint(gen):
    h = hashlib.md5()
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in (os.path.join(SRC, "bridge.py"), os.path.join(SRC, "idm.py"),
               os.path.join(here, "_common.py"), os.path.abspath(__file__), POOL_FILE):
        h.update(open(fn, "rb").read())
    h.update(json.dumps(gen, sort_keys=True, default=float).encode())
    return h.hexdigest()


def run_cells(cells, arms, gen, pt, pilot, workers, threads):
    todo = []
    for cell, arm, N, R, B in cells:
        done = {r["rep"] for r in read_cell(cell)}
        todo += [(cell, arm, N, r, B) for r in range(R) if r not in done]
    if not todo:
        return
    todo.sort(key=lambda t: -t[2])                     # large-N tasks first: no long tail
    print(f"[run] {len(todo)} replicates on {workers} workers x {threads} threads", flush=True)
    t0, n, failed = time.time(), 0, []
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(threads,),
                                 mp_context=multiprocessing.get_context("spawn")) as ex:
            futs = {ex.submit(run_rep, c, a, arms[a], gen, N, r, B, pilot["mode"],
                              pt[a]["theta_star"]): (c, r) for c, a, N, r, B in todo}
            for fu in as_completed(futs):
                try:
                    rec = fu.result()
                except BrokenProcessPool:
                    raise
                except Exception as e:                       # one bad replicate: log, go on
                    failed.append({"cell": futs[fu][0], "rep": futs[fu][1], "error": repr(e)})
                    print(f"[run] FAILED {futs[fu]}: {e!r}", flush=True)
                    continue
                fn = os.path.join(OUT, "cells", f"{rec['cell']}.jsonl")
                lead = ""
                if os.path.exists(fn) and os.path.getsize(fn) > 0:
                    with open(fn, "rb") as fh:
                        fh.seek(-1, os.SEEK_END)
                        lead = "" if fh.read(1) == b"\n" else "\n"   # after a hard kill
                with open(fn, "a", encoding="utf-8") as fh:
                    fh.write(lead + json.dumps(rec, default=float) + "\n")
                n += 1
                if n % 10 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"[run] {n}/{len(todo)} done, {el / 60:.0f} min, about "
                          f"{el / n * (len(todo) - n) / 60:.0f} min left", flush=True)
    except BrokenProcessPool as e:
        sys.exit(f"worker pool broke ({e!r}); completed replicates are saved - rerun to resume")
    ffile = os.path.join(OUT, "failed_replicates.json")
    if not failed and os.path.exists(ffile):
        json.dump([], open(ffile, "w"))              # an earlier failure has been rerun
    if failed:
        json.dump(failed, open(ffile, "w"), indent=2)
        sys.exit(f"STOP - {len(failed)} replicate(s) failed (see failed_replicates.json); "
                 f"the others are saved. Report this before rerunning.")


def main():
    ap = argparse.ArgumentParser()
    ncpu = os.cpu_count() or 4
    ap.add_argument("--workers", type=int, default=max(1, ncpu // 4))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    threads = max(1, ncpu // a.workers)
    os.makedirs(os.path.join(OUT, "cells"), exist_ok=True)
    g = json.load(open(GEN_FILE))
    if not g.get("gate", {}).get("passed", False):
        sys.exit("STOP - the S3 gate did not pass; S4 may not run (J)")
    gen = {"shared_theta": g["shared_theta"], "t_model": g["t_model"], "best": g["best"]}
    if any(abs(g["best"][k] - v) > 1e-9 for k, v in J6_G0.items()):
        sys.exit(f"STOP - generator.json best {g['best']} is not the J.6 generator {J6_G0}")
    arms = arms_from_generator(g)
    pt_file, pilot_file = os.path.join(OUT, "pseudo_true.json"), os.path.join(OUT, "pilot.json")
    if a.summary:                                   # read-only: never touches the fingerprint
        summarise(json.load(open(pt_file)), json.load(open(pilot_file)))
        return 0

    # 0 pool
    if not os.path.exists(POOL_FILE):
        n, dmax = build_pool()
        print(f"[0] pool: {n} location-1 episodes, max |gap - dhw| {dmax:.4f} m", flush=True)
    P = load_pool()
    if len(P["v0"]) != N_POOL:
        sys.exit(f"STOP - pool has {len(P['v0'])} episodes, J.7 requires {N_POOL}")
    fp = fingerprint(gen)
    fpfile = os.path.join(OUT, "fingerprint.txt")
    if os.path.exists(fpfile):
        if open(fpfile).read().strip() != fp:
            sys.exit("STOP - saved S4 results come from different code, pool or generator. Move "
                     "code/results/step5_bridge aside and start again.")
    else:
        open(fpfile, "w").write(fp + "\n")
    global _P
    _P = P

    # 1 pilot
    if os.path.exists(pilot_file):
        pilot = json.load(open(pilot_file))
    else:
        lab = "G0-300/rep0"
        idx = np.sort(np.random.default_rng(seed(lab)).choice(len(P["v0"]), PILOT_CELL_N, replace=False))
        S, _, _ = generate(P, idx, arms["G0"], gen, lab + "/gen")
        f = br.fit_blite(S, np.random.default_rng(seed(lab + "/fit")), n_random=2)
        import numba
        nthr = numba.get_num_threads()
        numba.set_num_threads(threads)                # time it as a worker would run it
        t0 = time.time(); _, _, Lc = bootstrap(S, f, PILOT_N, "pilot", "cheap"); tc = time.time() - t0
        t0 = time.time(); _, _, Lf = bootstrap(S, f, PILOT_N, "pilot", "full"); tf = time.time() - t0
        numba.set_num_threads(nthr)
        rel = (Lc - Lf) / Lf
        n_ok = int(np.sum(rel <= PILOT_TOL))
        pilot = {"mode": "cheap" if n_ok >= PILOT_NEED else "full", "n_ok": n_ok,
                 "rel_diff": rel.tolist(), "sec_cheap": tc / PILOT_N, "sec_full": tf / PILOT_N}
        json.dump(pilot, open(pilot_file, "w"), indent=2)
    print(f"[1] pilot: bootstrap start set '{pilot['mode']}' ({pilot['n_ok']}/{PILOT_N}); "
          f"{pilot['sec_cheap']:.1f} s vs {pilot['sec_full']:.1f} s per refit at N = 300 on "
          f"{threads} threads", flush=True)
    per = pilot["sec_cheap" if pilot["mode"] == "cheap" else "sec_full"]
    work = sum(R * (B + 1.5) * per * N / 300 for _, _, N, R, B in CELLS if B > 0) + \
        sum(R * 1.5 * per * N / 300 for _, _, N, R, B in CELLS if B == 0)
    print(f"[1] rough estimate for the cells: {work / a.workers / 3600:.1f} h on {a.workers} workers "
          f"(+ pseudo-true fits, about 1 h)", flush=True)

    # 2 pseudo-true values
    pt = json.load(open(pt_file)) if os.path.exists(pt_file) else {}
    M = len(P["v0"])
    for arm in arms:
        if arm in pt and len(pt[arm].get("halves", [])) == 2:
            continue
        halves, structs = [], []
        for h in range(2):
            t0 = time.time()
            idx = np.tile(np.arange(M), PT_COPIES)
            S, thetas, info = generate(P, idx, arms[arm], gen, f"pt/{arm}/half{h}")
            f = br.fit_blite(S, np.random.default_rng(seed(f"pt/{arm}/half{h}/fit")), n_random=2)
            halves.append(f["theta_typ"].tolist())
            st = np.array(gen["shared_theta"], float); st[1] = float(np.median(thetas[:, 1]))
            structs.append(st.tolist())
            print(f"[2] pseudo-true {arm} half {h}: {np.round(f['theta_typ'], 4).tolist()} "
                  f"({(time.time() - t0) / 60:.1f} min, F4 first-try {info['accept_first_try']:.3f})", flush=True)
            del S
        H = np.array(halves)
        pt[arm] = {"halves": halves, "theta_star": H.mean(0).tolist(),
                   "mc_error": (np.abs(H[0] - H[1]) / 2).tolist(),
                   "theta_struct": np.mean(structs, 0).tolist()}
        json.dump(pt, open(pt_file, "w"), indent=2)

    # 3 positive control first, then D5.1
    run_cells([c for c in CELLS if c[0] == "PC-300"], arms, gen, pt, pilot, a.workers, threads)
    D = summarise(pt, pilot)
    if not D.get("D5.1", {}).get("pass", False):
        print("STOP - D5.1 (positive control) failed; the other cells do not run (J.7).")
        return 1

    # 4 the other cells
    run_cells([c for c in CELLS if c[0] != "PC-300"], arms, gen, pt, pilot, a.workers, threads)

    # 5 summary
    summarise(pt, pilot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
