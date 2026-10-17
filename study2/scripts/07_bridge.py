"""STEP 4 - the synthetic bridge (05_PREREGISTRATION_DRAFT.md Amendment G).

Known parameters theta* drive a synthetic IDM follower behind REAL highD leader traces
(position-driven, Step 3); synthetic gap observations get AR(1) noise (and, in arm D,
the real replay drift of the same episode); plain IDM is then fitted with the pooled
AR(1) likelihood exactly as it will be on real data (Step 5). Because the truth is
known, this measures - before any real fit -
  coverage of the chi2(0.95,5) region, identifiability (profile closure and width) as a
  function of N, the effect of real-data drift, the effect of model misspecification,
  the episode-bootstrap threshold, and the pure-information difference between sites.

Arms and cells are defined in CELLS below and mirrored in Amendment G. Order of work:
  1. arm P (positive control, designed traces). If its coverage CI excludes 0.95 the
     run STOPS: the machinery is broken and nothing else is interpretable.
  2. pseudo-true theta_KL for the misspecification arm (5 fits on the full loc-1 pool).
  3. everything else, longest tasks first, in parallel.
Each finished rep is appended to code/results/step4_bridge/<cell>.jsonl (aggregates
only - no trajectories), so the run is resumable: re-running skips finished reps.

Usage (from AINN_ReWORK/scripts):
  python 07_bridge.py --workers 28            # full run, resumable
  python 07_bridge.py --arm P --workers 28    # one arm only
  python 07_bridge.py --summarise             # tables from whatever has finished
  python 07_bridge.py --quick --workers 2     # sandbox smoke test (separate folder)
"""

import os

# One BLAS/OpenMP/numba thread per process, set BEFORE numpy is imported. Parallelism
# here comes from worker processes; letting each of 24 processes also start a 32-thread
# BLAS pool multiplies memory commitment and is the likely cause of the Windows
# "paging file is too small" failure (DIARY Entry 23). Results are unaffected.
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
# scipy.stats is imported lazily (summary only). On Windows every worker re-imports this
# module (spawn); keeping heavy imports out of module level keeps worker start-up memory
# down - 24 workers importing scipy.stats + numba at once exhausted the Windows commit
# limit ("paging file is too small", RUN REQUEST #4 first attempt, DIARY Entry 23).

from _common import DERIVED, RESULTS, SRC  # noqa: F401  (SRC puts code/src on sys.path)

import bridge as br                                     # noqa: E402
from idm import PARAM_NAMES, make_leader_trace          # noqa: E402

BASE_SEED = 20260926
POOL_FILE = os.path.join(DERIVED, "bridge_pool_defE.npz")
LAG_STEPS = 15                                          # 0.6 s reaction delay, arm M

THETA = {"A": (38.0, 1.2, 2.0, 1.0, 1.5),               # primary
         "B": (35.0, 1.6, 3.0, 0.8, 1.2),               # cautious
         "C": (42.0, 0.9, 1.5, 1.5, 2.5),               # assertive
         "P": (30.0, 1.2, 2.0, 1.5, 2.0)}               # A0-pre-2 value, positive control
NOISE = {"primary": (1.0, 0.99), "low": (0.3, 0.98), "high": (2.0, 0.995), "p": (0.3, 0.90)}


def _cell(cid, arm, theta, pool, n, noise, R, P, boot=0, drift=False, truth="idm",
          n_random=2, refit=False, seed_id=None):
    return dict(id=cid, arm=arm, theta=theta, pool=pool, n=n, noise=noise, R=R, P=P,
                boot=boot, drift=drift, truth=truth, n_random=n_random, refit=refit,
                seed_id=seed_id or cid)


CELLS = [
    _cell("P-16", "P", "P", "designed", 16, "p", 200, 20),
    # G.9: second positive control at the primary (high-rho) noise - run 1's P-16, at
    # rho = 0.9, could not detect the incidental-parameter bias that appears at rho ~ 0.99
    _cell("P2-64", "P", "P", "designed", 64, "primary", 200, 20),
    *[_cell(f"C-{t}-{n}", "C", t, "loc1", n, "primary", 200,
            40 if (t == "A" and n in (518, 1000)) else 20)
      for t in ("A", "B", "C") for n in (50, 200, 518, 1000)],
    _cell("C-A-5691", "C", "A", "loc1", 5691, "primary", 200, 10),
    *[_cell(f"N-A-{n}-{lv}", "N", "A", "loc1", n, lv, 200, 20)
      for lv in ("low", "high") for n in (518, 1000)],
    *[_cell(f"D-A-{n}", "D", "A", "loc1", n, "primary", 200, 20, drift=True) for n in (518, 1000)],
    _cell("D-A-518-low", "D", "A", "loc1", 518, "low", 200, 20, drift=True),
    _cell("M-518", "M", "A", "loc1", 518, "primary", 100, 20, boot=49, truth="delayed"),
    _cell("M-1000", "M", "A", "loc1", 1000, "primary", 100, 20, truth="delayed"),
    _cell("S-loc1ff-518", "S", "A", "loc1_ff", 518, "primary", 200, 40),
    _cell("S-loc5-518", "S", "A", "loc5", 518, "primary", 200, 40),
    _cell("X-A-518", "X", "A", "loc1", 518, "primary", 100, 0, boot=49),
    # G.9: the bootstrap under real-data drift - what Step 5 will actually rely on
    _cell("DX-A-518", "D", "A", "loc1", 518, "primary", 100, 0, boot=49, drift=True),
    _cell("DX-A-518-low", "D", "A", "loc1", 518, "low", 100, 0, boot=49, drift=True),
]
# D4.8 re-runs (Amendment H): cells where a profile found a better maximum in > 2% of
# profiled reps in run 3. Same seeds as the original cell (same synthetic data); only the
# optimiser changes: 5 random starts + refit-on-better-maximum.
for _orig in ("C-A-1000", "C-C-50", "D-A-518", "N-A-1000-high"):
    _c = dict([c for c in CELLS if c["id"] == _orig][0])
    _c.update(id=_orig + "-r", arm="R", n_random=5, refit=True, seed_id=_orig)
    CELLS.append(_c)
CELL = {c["id"]: c for c in CELLS}
# seed stream per cell from a stable hash of its id (independent of list order)
CELL_IDX = {c["id"]: zlib.crc32(c["seed_id"].encode()) for c in CELLS}

_POOL = {}
_KL = None


def _apply_quick():
    """Smoke-test sizes: tiny reps and small N, same code path."""
    for c in CELLS:
        c.update(R=2, P=min(c["P"], 1), boot=min(c["boot"], 2), n=min(c["n"], 60))


def _init_worker(pool_file, kl, quick=False):
    global _POOL, _KL
    if quick:
        _apply_quick()
    if pool_file and os.path.exists(pool_file):
        z = np.load(pool_file)
        _POOL = {k: z[k] for k in z.files if k != "episode"}
    _KL = None if kl is None else np.asarray(kl)


def _designed(rng, n=16):
    """n distinct designed traces as in A0-pre-2 (informative profile + jitter). The first
    16 are identical whatever n is, so P-16 is unchanged from run 1."""
    ul, vl, v0 = [], [], []
    for _ in range(n):
        _, v = make_leader_trace(br.DT, 20.0, kind="informative")
        v = np.maximum(v + (0.9 + 0.2 * rng.random()) * np.sin(
            2 * np.pi * np.arange(len(v)) * br.DT / (3.0 + 2.0 * rng.random())), 3.0)
        ul.append(100.0 + np.r_[0.0, np.cumsum(v[:-1] * br.DT)]); vl.append(v); v0.append(v[0])
    off = np.r_[0, np.cumsum([len(u) for u in ul])].astype(np.int64)
    return np.concatenate(ul), np.concatenate(vl), np.zeros(off[-1]), off, np.array(v0)


def _gather(idx):
    off_all = _POOL["off"]
    lens = off_all[idx + 1] - off_all[idx]
    off = np.r_[0, np.cumsum(lens)].astype(np.int64)
    sel = np.concatenate([np.arange(off_all[i], off_all[i + 1]) for i in idx])
    return _POOL["ul"][sel], _POOL["vl"][sel], _POOL["rf"][sel], off, _POOL["vF0"][idx]


def _synth(c, rng, truth_theta):
    if c["pool"] == "designed":
        ul, vl, rf, off, v0 = _designed(np.random.default_rng(BASE_SEED + 1), c["n"])  # fixed traces
    else:
        pool = _POOL[f"pool_{c['pool']}"]
        idx = pool if c["n"] >= len(pool) else np.sort(rng.choice(pool, c["n"], replace=False))
        ul, vl, rf, off, v0 = _gather(idx)
    th = np.asarray(truth_theta, dtype=float)
    x0 = ul[off[:-1]] - br.equilibrium_gap(v0, th)
    s_true = np.empty(off[-1])
    if c["truth"] == "delayed":
        hits = br._sim_delayed(ul, vl, off, x0, v0, th, br.DT, br.S_FLOOR, LAG_STEPS,
                               s_true, np.empty(off[-1]))
    else:
        hits = br._sim_idm(ul, vl, off, x0, v0, th, br.DT, br.S_FLOOR, s_true)
    sd, rho = NOISE[c["noise"]]
    s_obs = s_true + br._ar1_filter(rng.standard_normal(off[-1]), off, rho, sd)
    if c["drift"]:
        s_obs = s_obs + rf
    return br.Sample(ul, vl, off, v0, s_obs), int(hits)


def run_task(cid, rep):
    c = CELL[cid]
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, CELL_IDX[cid], rep]))
    truth = THETA[c["theta"]]
    target = np.asarray(truth) if c["truth"] == "idm" else _KL
    S, truth_hits = _synth(c, rng, truth)
    rec = {"cell": cid, "rep": rep, "M": S.M, "N_steps": S.N, "truth_floor_hits": truth_hits,
           "lik": S.init_mode}
    t0 = time.time()
    profs = None
    if c["refit"] and rep < c["P"]:
        f, profs, rounds, resolved = br.fit_and_profile(S, rng, c["n_random"])
        rec.update(refit_rounds=rounds, refit_resolved=resolved)
    else:
        f = br.fit(S, rng, n_random=c["n_random"])
    rec["t_fit"] = round(time.time() - t0, 3)
    ll_t = br.loglik(S, target)[0]
    lr = 2.0 * (f["ll"] - ll_t)
    rec.update(theta_hat=[round(float(x), 5) for x in f["theta"]], rho_hat=round(f["rho"], 5),
               at_bound=f["at_bound"], fit_floor_hits=f["floor_hits"], lr=float(lr),
               lr_negative=bool(lr < 0), covered=bool(lr <= br.CHI2_K), n_eval_fit=S.n_eval)
    if rep < c["P"]:
        t0 = time.time()
        rec["profiles"] = profs if profs is not None else [br.profile(S, f, i) for i in range(br.K)]
        rec["t_profile"] = round(time.time() - t0, 3)
    if rep < (c["R"] if c["boot"] else 0):
        t0 = time.time()
        thr, lrs = br.bootstrap_threshold(S, f, c["boot"], rng)
        rec.update(boot_thr=thr, boot_covered=bool(lr <= thr), t_boot=round(time.time() - t0, 3),
                   boot_lr_q=[round(float(q), 3) for q in np.quantile(lrs, [0.5, 0.9, 0.95])])
    return rec


def run_kl(draw, n=10 ** 9):
    """Pseudo-true theta_KL for arm M: plain IDM fitted to delayed-IDM data on the FULL
    location-1 pool with the primary noise; theta_KL = median over 5 draws."""
    c = dict(CELL["M-518"], n=n)
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, 999, draw]))
    S, _ = _synth(c, rng, THETA["A"])
    f = br.fit(S, rng)
    return {"draw": draw, "theta_hat": f["theta"].tolist(), "ll": f["ll"], "rho": f["rho"],
            "at_bound": f["at_bound"]}


# ------------------------------------------------------------------------------------
def clopper_pearson(k, n, a=0.05):
    from scipy.stats import beta as _beta
    lo = 0.0 if k == 0 else float(_beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - a / 2, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def load_done(outdir):
    done = {}
    for f in glob.glob(os.path.join(outdir, "*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                done.setdefault(r["cell"], {})[r["rep"]] = r
    return done


def cell_ok_positive_control(recs):
    if not recs:
        return None, None, None
    k = sum(r["covered"] for r in recs)
    lo, hi = clopper_pearson(k, len(recs))
    return lo <= 0.95 <= hi, k / len(recs), (lo, hi)


def run_parallel(tasks, workers, outdir, kl, label, quick=False):
    if not tasks:
        return
    print(f"[{label}] {len(tasks)} tasks on {workers} workers", flush=True)
    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(POOL_FILE, kl, quick)) as ex:
        futs = {ex.submit(run_task, cid, rep): (cid, rep) for cid, rep in tasks}
        for fu in as_completed(futs):
            n_done += 1
            try:
                r = fu.result()
            except BrokenProcessPool as e:
                with open(os.path.join(outdir, "errors.log"), "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} worker pool broke: {e!r}\n")
                sys.exit("STOP: a worker process died (usually memory: on Windows, 'paging file is "
                         "too small'). Re-run with fewer workers, e.g. --workers 8. Finished reps "
                         "are kept.")
            except Exception as e:                      # log and continue; rerun resumes it
                cid, rep = futs[fu]
                with open(os.path.join(outdir, "errors.log"), "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {cid} rep {rep}: {e!r}\n")
                print(f"  ERROR {cid} rep {rep}: {e!r} (logged; rerun resumes)", flush=True)
                continue
            with open(os.path.join(outdir, f"{r['cell']}.jsonl"), "a") as f:
                f.write(json.dumps(r) + "\n")
            if n_done % 25 == 0 or n_done == len(tasks):
                el = time.time() - t0
                print(f"  {n_done}/{len(tasks)} done, {el / 60:.1f} min elapsed, "
                      f"~{el / n_done * (len(tasks) - n_done) / 60:.0f} min left", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 4)))
    ap.add_argument("--arm", type=str, default="PCNDMSX")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    # run 1 (G.8) is kept in code/results/step4_bridge/, run 2 (G.9) in step4_bridge_int/;
    # run 3 (G.10 fixed-reference Jacobian) writes here
    outdir = os.path.join(RESULTS, "step4_bridge_quick" if a.quick else "step4_bridge_g10")
    os.makedirs(outdir, exist_ok=True)
    if a.quick:
        _apply_quick()
    if a.summarise:
        return summarise(outdir)
    if not os.path.exists(POOL_FILE):
        sys.exit(f"{POOL_FILE} missing - run 06_bridge_pools.py first")
    print(f"numba: {br.HAVE_NUMBA}", flush=True)
    _init_worker(POOL_FILE, None)                      # warm the numba cache in main
    _S0 = _synth(CELL["P-16"], np.random.default_rng(0), THETA["P"])[0]
    br.set_reference(_S0, THETA["P"])                   # compiles the G.10 kernels too
    br.loglik(_S0, THETA["P"])
    _synth(dict(CELL["M-518"], n=5), np.random.default_rng(0), THETA["A"])   # compiles _sim_delayed

    done = load_done(outdir)
    todo = lambda cid: [(cid, r) for r in range(CELL[cid]["R"]) if r not in done.get(cid, {})]

    # 1. positive controls first (P-16 and, since G.9, P2-64); stop if either fails
    if "P" in a.arm:
        for pc in ("P-16", "P2-64"):
            run_parallel(todo(pc), a.workers, outdir, None, f"positive control {pc}", a.quick)
            recs = list(load_done(outdir).get(pc, {}).values())
            ok, cov, ci = cell_ok_positive_control(recs)
            if ok is None:
                sys.exit(f"STOP: no {pc} reps finished - see the errors above.")
            if len(recs) < CELL[pc]["R"]:
                sys.exit(f"STOP: only {len(recs)} of {CELL[pc]['R']} {pc} reps finished; "
                         "re-run the same command to complete them.")
            print(f"positive control {pc}: coverage {cov:.3f} CI {ci}: {'PASS' if ok else 'FAIL'}",
                  flush=True)
            if not ok and not a.quick:
                sys.exit(f"STOP: positive control {pc} failed (Amendment G, D4.6). Nothing else is run.")

    # 2. pseudo-true for arm M
    kl = None
    klf = os.path.join(outdir, "theta_kl.json")
    if "M" in a.arm:
        if os.path.exists(klf):
            kl = json.load(open(klf))["theta_kl"]
        else:
            print("[theta_KL] 5 fits on the full location-1 pool", flush=True)
            with ProcessPoolExecutor(max_workers=min(5, a.workers), initializer=_init_worker,
                                     initargs=(POOL_FILE, None, a.quick)) as ex:
                nk = 200 if a.quick else 10 ** 9
                draws = [f.result() for f in [ex.submit(run_kl, d, nk) for d in range(5)]]
            kl = np.median([d["theta_hat"] for d in draws], axis=0).tolist()
            json.dump({"theta_kl": kl, "draws": draws, "lag_steps": LAG_STEPS,
                       "theta_true_delayed": THETA["A"]}, open(klf, "w"), indent=2)
            print(f"theta_KL = {np.round(kl, 3).tolist()}", flush=True)

    # 3. everything else, longest first
    tasks = []
    for c in CELLS:
        if c["arm"] in a.arm and c["arm"] != "P":
            tasks += todo(c["id"])
    cost = lambda t: CELL[t[0]]["n"] * (1 + 12 * (t[1] < CELL[t[0]]["P"])
                                        + 35 * (CELL[t[0]]["boot"] > 0))
    tasks.sort(key=cost, reverse=True)
    run_parallel(tasks, a.workers, outdir, kl, "bridge", a.quick)
    summarise(outdir)
    return 0


# ------------------------------------------------------------------------------------
def summarise(outdir):
    done = load_done(outdir)
    out = {"chi2_k": br.CHI2_K, "chi2_1": br.CHI2_1, "cells": {}}
    kl = None
    if os.path.exists(os.path.join(outdir, "theta_kl.json")):
        kl = json.load(open(os.path.join(outdir, "theta_kl.json")))["theta_kl"]
        out["theta_kl"] = kl
    for c in CELLS:
        recs = sorted(done.get(c["id"], {}).values(), key=lambda r: r["rep"])
        if not recs:
            continue
        n = len(recs)
        k = sum(r["covered"] for r in recs)
        th = np.array([r["theta_hat"] for r in recs])
        tgt = np.asarray(THETA[c["theta"]] if c["truth"] == "idm" else (kl or THETA["A"]))
        s = {"reps": n, "N_episodes": recs[0]["M"], "noise_sd_rho": NOISE[c["noise"]],
             "coverage": round(k / n, 4), "coverage_ci": clopper_pearson(k, n),
             "lr_median": round(float(np.median([r["lr"] for r in recs])), 3),
             "lr_median_chi2_ref": 4.351,                 # chi2(0.5, 5)
             "lr_negative": int(sum(r["lr_negative"] for r in recs)),
             "theta_target": list(tgt),
             "theta_hat_median": np.round(np.median(th, axis=0), 4).tolist(),
             "theta_hat_rel_bias": np.round(np.median(th, axis=0) / tgt - 1, 4).tolist(),
             "at_bound_rate": {p: round(sum(p in r["at_bound"] for r in recs) / n, 3) for p in PARAM_NAMES},
             "rho_hat_median": round(float(np.median([r["rho_hat"] for r in recs])), 5),
             "fit_floor_hit_reps": int(sum(r["fit_floor_hits"] > 0 for r in recs)),
             "t_fit_median_s": round(float(np.median([r["t_fit"] for r in recs])), 2)}
        pr = [r for r in recs if "profiles" in r]
        if pr:
            s["profile_reps"] = len(pr)
            s["t_profile_median_s"] = round(float(np.median([r["t_profile"] for r in pr])), 1)
            geo = {}
            for i, p in enumerate(PARAM_NAMES):
                g = {}
                for lvl in ("p1", "pk"):
                    cl = [r["profiles"][i][f"{lvl}_closed"] for r in pr]
                    w = [r["profiles"][i][f"{lvl}_width"] for r in pr]
                    kc = int(sum(cl))
                    wc = [x for x in w if x is not None]
                    wu = np.array([x if x is not None else np.inf for x in w])
                    g[lvl] = {"closure": round(kc / len(pr), 3), "closure_ci": clopper_pearson(kc, len(pr)),
                              "cond_width_median": round(float(np.median(wc)), 4) if wc else None,
                              "cond_width_iqr": [round(float(q), 4) for q in np.quantile(wc, [.25, .75])] if wc else None,
                              "uncond_width_median": (round(float(np.median(wu)), 4)
                                                      if np.isfinite(np.median(wu)) else "inf")}
                g["better_max_found_reps"] = int(sum(r["profiles"][i]["better_max_found"] > 1e-3 for r in pr))
                geo[p] = g
            s["geometry"] = geo
        bt = [r for r in recs if "boot_thr" in r]
        if bt:
            kb = sum(r["boot_covered"] for r in bt)
            s["boot"] = {"reps": len(bt), "coverage": round(kb / len(bt), 4),
                         "coverage_ci": clopper_pearson(kb, len(bt)),
                         "threshold_median": round(float(np.median([r["boot_thr"] for r in bt])), 3),
                         "t_boot_median_s": round(float(np.median([r["t_boot"] for r in bt])), 1)}
        out["cells"][c["id"]] = s
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    md = render(out)
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return 0


def render(out):
    L = ["# Step 4 - synthetic bridge summary", "",
         "| cell | reps | N | coverage [CP 95%] | LR med (ref 4.35) | boot cov | rel. bias v0,T,s0,a,b | at bound |",
         "|---|---|---|---|---|---|---|---|"]
    for cid, s in out["cells"].items():
        b = s.get("boot")
        ab = ",".join(f"{p}:{v}" for p, v in s["at_bound_rate"].items() if v > 0) or "-"
        L.append(f"| {cid} | {s['reps']} | {s['N_episodes']} | {s['coverage']} {s['coverage_ci']} | "
                 f"{s['lr_median']} | {(str(b['coverage']) + ' ' + str(b['coverage_ci'])) if b else '-'} | "
                 f"{s['theta_hat_rel_bias']} | {ab} |")
    L += ["", "Profile closure (chi2_1 interval | chi2_5 region projection) and conditional width:", "",
          "| cell | reps | " + " | ".join(PARAM_NAMES) + " |", "|---" * (len(PARAM_NAMES) + 2) + "|"]
    for cid, s in out["cells"].items():
        if "geometry" not in s:
            continue
        L.append(f"| {cid} | {s['profile_reps']} | " + " | ".join(
            f"{g['p1']['closure']} / {g['pk']['closure']} (w {g['pk']['cond_width_median']})"
            for g in s["geometry"].values()) + " |")
    if "theta_kl" in out:
        L.append(f"\ntheta_KL (arm M) = {np.round(out['theta_kl'], 3).tolist()}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
