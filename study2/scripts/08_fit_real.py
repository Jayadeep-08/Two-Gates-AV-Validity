"""STEP 5a - first real-data calibration: highD location 1, analysis set E10,1.0.

Pre-registration: 05_PREREGISTRATION_DRAFT.md Amendments F (F.1), G.10 and H (H.3, H.4).
Engine: code/src/bridge.py, unchanged from Step 4 run 3 apart from two additions that
do not alter any computation Step 4 validated (profile thresholds as a parameter;
residual_diagnostics).

Stages (each cached in code/results/step5_loc1/, so the run is resumable):
  1 fit        G.10 two-stage fit, 5 random starts (H.3)
  2 prof_chi2  profiles at chi2_1 / chi2_5 in parallel; if a profile beats the fit by
               > 0.01 log-lik, refit from that point and repeat (<= 3 rounds, D4.8)
  3 boot       episode bootstrap, B = 199 (H.3): resample whole episodes, refit,
               LR_b = 2[l_b(theta_b) - l_b(theta_hat)]; c_boot = 95th percentile
  4 diag       Gate L (innovation ACF), D4.7 noise bracket, H.4 drift statistic V
               against 20 parametric null replicates at (theta_hat, rho_hat, sigma_hat)
  5 drift      ONLY if V > 1.25 (H.4): scale k of the real replay drift RF matched to V by
               bisection (10 replicates per step), then 100 replicates -> c_drift
  6 prof_final profiles at chi2_1, chi2_5, c_boot (and c_drift if stage 5 ran)
  7 summary    code/results/step5_loc1/summary.{json,md}
Primary region (H.3/H.4): c_boot if V <= 1.25, c_drift if V > 1.25; the others are reported
alongside. That choice is made by the frozen rule inside this script, not afterwards.

LICENCE: the real-data sample is cached under derived/ (local only). Everything written
to code/results/ is aggregate.

Usage (from AINN_ReWORK/scripts):
  python 08_fit_real.py --workers 16
  python 08_fit_real.py --quick --workers 2     # sandbox smoke test, 150 episodes
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd

from _common import ARRAY_ROWS, DERIVED, RESULTS, SRC  # noqa: F401

import bridge as br                                   # noqa: E402
from idm import PARAM_NAMES                           # noqa: E402

SEED = 20260927
R = {k: i for i, k in enumerate(ARRAY_ROWS)}
B_BOOT, N_NULL, V_RULE = 199, 20, 1.25                # H.3, H.4 - frozen
DRIFT_STEP_REPS, DRIFT_FINAL_REPS = 10, 100           # H.4 - frozen
STAGE_CODE = {"boot": 1, "null": 2, "drift_k": 3, "drift": 4}

_G = {}                                               # per-process state


# ------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------
def replay_drift(a):
    uF, vF, aF = a[R["u_fF"]], a[R["vF"]], a[R["aF"]]
    x = np.empty(len(aF))
    x[0], v = uF[0], vF[0]
    for t in range(len(aF) - 1):
        x[t + 1] = x[t] + v * br.DT + 0.5 * aF[t] * br.DT * br.DT
        v = v + aF[t] * br.DT
    return x - uF


def build_sample_file(path, locations=(1,), sd_min=1.0, quick_n=None):
    """Select car-led episodes >= 10 s with sd(v_leader) >= sd_min at the given highD
    locations (definition E == C) and cache ul, vl, gap, v0, rf, offsets locally."""
    d = os.path.join(DERIVED, "highd_episodes_defE")
    ul, vl, gap, rf, v0, ids, lens = [], [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(d, "??_index.csv"))):
        if os.path.getsize(f) <= 1:
            continue
        ix = pd.read_csv(f)
        ix = ix[(ix["leader_class"] == "car") & (ix["dur_s"] >= 10) & (ix["sd_vL"] >= sd_min)
                & ix["location"].isin(list(locations))]
        if ix.empty:
            continue
        arrs = np.load(f.replace("_index.csv", "_episodes.npz"))
        for ep in ix["episode"]:
            a = arrs[ep]
            ul.append(a[R["u_rL"]]); vl.append(a[R["vL"]]); gap.append(a[R["gap"]])
            rf.append(replay_drift(a)); v0.append(a[R["vF"], 0]); ids.append(ep)
            lens.append(a.shape[1])
    idx = np.arange(len(ids))
    if quick_n:
        idx = np.sort(np.random.default_rng(SEED).choice(idx, quick_n, replace=False))
    sel = lambda L: np.concatenate([L[i] for i in idx])
    off = np.r_[0, np.cumsum([lens[i] for i in idx])].astype(np.int64)
    np.savez(path, ul=sel(ul), vl=sel(vl), gap=sel(gap), rf=sel(rf), off=off,
             v0=np.array(v0)[idx], episode=np.array(ids)[idx].astype(str))
    return len(idx), int(off[-1])


def load_sample(path):
    z = np.load(path)
    S = br.Sample(z["ul"], z["vl"], z["off"], z["v0"], z["gap"])
    return S, z["rf"], z["episode"]


def _init(path, theta_ref):
    S, rf, _ = load_sample(path)
    if theta_ref is not None:
        br.set_reference(S, theta_ref)
    _G.update(S=S, rf=rf)


# ------------------------------------------------------------------------------------
# worker tasks
# ------------------------------------------------------------------------------------
def task_profile(i, u, ll, rho, thresholds):
    f = {"u": np.asarray(u), "ll": ll, "rho": rho}
    t0 = time.time()
    p = br.profile(_G["S"], f, i, thresholds=thresholds)
    p["t"] = round(time.time() - t0, 1)
    return p


def task_boot(b, theta_hat):
    S = _G["S"]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, STAGE_CODE["boot"], b]))
    idx = rng.integers(0, S.M, S.M)
    Sb = S.subset(idx)                                     # inherits the full-data theta_ref
    fb = br.fit(Sb, rng, n_random=1, extra_starts=(theta_hat,))
    return {"b": b, "lr": float(2.0 * (fb["ll"] - br.loglik(Sb, theta_hat)[0])),
            "theta_b": [float(x) for x in fb["theta"]], "rho_b": fb["rho"]}


def task_replicate(stage, j, theta_hat, rho, sigma, k):
    """Parametric replicate at (theta_hat, rho, sigma), plus k x the real replay drift:
    the truth starts from each episode's OBSERVED initial state, exactly as the fit
    does. Refits with the same code and returns the LR at theta_hat and var(slopes)."""
    S = _G["S"]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, STAGE_CODE[stage], j,
                                                        int(round(k * 1e6))]))
    s_true = np.empty(S.N)
    br._sim_idm(S.ul, S.vl, S.off, S.x0, S.v0, np.asarray(theta_hat, float), br.DT,
                br.S_FLOOR, s_true)
    s_obs = s_true + br._ar1_filter(rng.standard_normal(S.N), S.off, rho, sigma) + k * _G["rf"]
    Sj = br.Sample(S.ul, S.vl, S.off, S.v0, s_obs)
    fj = br.fit(Sj, rng, n_random=1, extra_starts=(theta_hat,))
    dj = br.residual_diagnostics(Sj, fj["theta"], fj["rho"])
    return {"stage": stage, "j": j, "k": k,
            "lr": float(2.0 * (fj["ll"] - br.loglik(Sj, theta_hat)[0])),
            "var_slope": float(np.var(dj["slopes"])), "rho": fj["rho"],
            "sigma": dj["sigma_marginal"]}


# ------------------------------------------------------------------------------------
def run_pool(fn, argsets, workers, path, theta_ref, label):
    out = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(path, theta_ref)) as ex:
        futs = {ex.submit(fn, *a): a for a in argsets}
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                out.append(fu.result())
            except BrokenProcessPool:
                sys.exit("STOP: a worker process died (memory?). Re-run with fewer --workers; "
                         "finished stages are kept.")
            if n % max(1, len(argsets) // 10) == 0 or n == len(argsets):
                el = time.time() - t0
                print(f"  [{label}] {n}/{len(argsets)}  {el / 60:.1f} min, "
                      f"~{el / n * (len(argsets) - n) / 60:.0f} min left", flush=True)
    return out


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def jsave(p, obj):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 4)))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    global B_BOOT, N_NULL, DRIFT_STEP_REPS, DRIFT_FINAL_REPS
    tag = "step5_loc1_quick" if a.quick else "step5_loc1"
    if a.quick:
        B_BOOT, N_NULL, DRIFT_STEP_REPS, DRIFT_FINAL_REPS = 6, 3, 2, 4
    out = os.path.join(RESULTS, tag)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(DERIVED, f"{tag}_sample.npz")
    if not os.path.exists(path):
        M, N = build_sample_file(path, quick_n=150 if a.quick else None)
        print(f"sample: {M} episodes, {N} steps -> {path}", flush=True)
    S, rf, eps = load_sample(path)
    print(f"numba {br.HAVE_NUMBA}; location 1 E10,1.0: {S.M} episodes, {S.N} steps, "
          f"{S.N * br.DT / 3600:.2f} h", flush=True)

    # 1 fit --------------------------------------------------------------------------
    fp = os.path.join(out, "fit.json")
    fit = jload(fp)
    if fit is None:
        t0 = time.time()
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 0]))
        f = br.fit(S, rng, n_random=5)
        fit = {"theta": list(f["theta"]), "u": list(f["u"]), "ll": f["ll"], "rho": f["rho"],
               "theta_ref": list(S.theta_ref), "at_bound": f["at_bound"],
               "floor_hits": f["floor_hits"], "n_eval": S.n_eval, "refits": 0,
               "t_s": round(time.time() - t0, 1)}
        jsave(fp, fit)
    print(f"[fit] theta_hat {np.round(fit['theta'], 4).tolist()}  rho {fit['rho']:.5f}  "
          f"at bound {fit['at_bound']}", flush=True)
    br.set_reference(S, fit["theta_ref"])

    # 2 chi2 profiles, refit on better maximum (D4.8) --------------------------------------
    pp = os.path.join(out, "profiles_chi2.json")
    prof = jload(pp)
    thr_chi2 = {"p1": br.CHI2_1, "pk": br.CHI2_K}
    while prof is None:
        res = run_pool(task_profile, [(i, fit["u"], fit["ll"], fit["rho"], thr_chi2)
                                      for i in range(br.K)], min(a.workers, br.K), path,
                       fit["theta_ref"], "chi2 profiles")
        res = sorted(res, key=lambda p: PARAM_NAMES.index(p["param"]))
        worst = max(res, key=lambda p: p["better_max_found"])
        if worst["better_max_found"] > 0.02 and fit["refits"] < 3:
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 10, fit["refits"]]))
            f = br.fit(S, rng, n_random=0, extra_starts=(br.to_theta(worst["better_u"]),))
            print(f"  profile of {worst['param']} found a better maximum "
                  f"(+{worst['better_max_found']:.3f} LR); refit: ll {fit['ll']:.3f} -> {f['ll']:.3f}",
                  flush=True)
            if f["ll"] > fit["ll"]:
                fit.update(theta=list(f["theta"]), u=list(f["u"]), ll=f["ll"], rho=f["rho"],
                           at_bound=f["at_bound"])
            fit["refits"] += 1
            jsave(fp, fit)
            continue
        prof = res
        jsave(pp, prof)
    th = np.asarray(fit["theta"])

    # 3 bootstrap ----------------------------------------------------------------------
    bp = os.path.join(out, "boot.jsonl")
    done = [json.loads(l) for l in open(bp)] if os.path.exists(bp) else []
    todo = [(b, list(th)) for b in range(B_BOOT) if b not in {r["b"] for r in done}]
    if todo:
        for r in run_pool(task_boot, todo, a.workers, path, fit["theta_ref"], "bootstrap"):
            with open(bp, "a") as f:
                f.write(json.dumps(r) + "\n")
        done = [json.loads(l) for l in open(bp)]
    lrb = np.array([r["lr"] for r in done])
    c_boot = float(np.quantile(lrb, 0.95))
    print(f"[boot] B={len(lrb)}  c_boot = {c_boot:.3f}  (chi2_5 = {br.CHI2_K:.2f})", flush=True)

    # 4 diagnostics --------------------------------------------------------------------
    dp = os.path.join(out, "diag.json")
    diag = jload(dp)
    if diag is None:
        d = br.residual_diagnostics(S, th, fit["rho"])
        nulls = run_pool(task_replicate, [("null", j, list(th), fit["rho"], d["sigma_marginal"], 0.0)
                                          for j in range(N_NULL)], a.workers, path,
                         fit["theta_ref"], "null replicates")
        v_null = float(np.mean([r["var_slope"] for r in nulls]))
        V = float(np.var(d["slopes"]) / v_null)
        diag = {"rho_hat": fit["rho"], "sigma_marginal": d["sigma_marginal"],
                "sd_innov": d["sd_innov"], "resid_rms": d["resid_rms"],
                "innovation_acf_1_10": d["acf"],
                "delta_q": np.quantile(d["delta"], [0.05, 0.5, 0.95]).tolist(),
                "slope_sd_real": float(np.std(d["slopes"])),
                "slope_sd_null_mean": float(np.sqrt(v_null)), "V": V, "V_rule": V_RULE,
                "drift_present": bool(V > V_RULE),
                "null": nulls,
                "noise_bracket_D4.7": {"sigma_in_[0.3,2.0]": bool(0.3 <= d["sigma_marginal"] <= 2.0),
                                       "rho_in_[0.98,0.995]": bool(0.98 <= fit["rho"] <= 0.995)},
                "null_lr_coverage_chi2": float(np.mean([r["lr"] <= br.CHI2_K for r in nulls])),
                "null_lr_coverage_boot": float(np.mean([r["lr"] <= c_boot for r in nulls]))}
        jsave(dp, diag)
    print(f"[diag] sigma {diag['sigma_marginal']:.3f} m  rho {diag['rho_hat']:.4f}  "
          f"ACF1 {diag['innovation_acf_1_10'][0]:+.3f}  V = {diag['V']:.3f} "
          f"({'DRIFT' if diag['drift_present'] else 'no material drift'})", flush=True)

    # 5 drift calibration (H.4), only if triggered ----------------------------------------
    thresholds = {"p1": br.CHI2_1, "pk": br.CHI2_K, "pb": c_boot}
    drift = None
    if diag["drift_present"]:
        dfp = os.path.join(out, "drift.json")
        drift = jload(dfp) or {"steps": []}
        sig = diag["sigma_marginal"]
        v_null = diag["slope_sd_null_mean"] ** 2
        target = diag["V"]

        def V_at(k):
            for s_ in drift["steps"]:
                if abs(s_["k"] - k) < 1e-9:
                    return s_["V"]
            res = run_pool(task_replicate, [("drift_k", j, list(th), fit["rho"], sig, k)
                                            for j in range(DRIFT_STEP_REPS)], a.workers, path,
                           fit["theta_ref"], f"drift k={k:.3f}")
            v = float(np.mean([r["var_slope"] for r in res]) / v_null)
            drift["steps"].append({"k": k, "V": v})
            jsave(dfp, drift)
            return v

        lo, hi = 0.0, 1.0
        while V_at(hi) < target and hi < 16:
            lo, hi = hi, hi * 2
        for _ in range(6):
            mid = 0.5 * (lo + hi)
            if V_at(mid) < target:
                lo = mid
            else:
                hi = mid
        k_star = 0.5 * (lo + hi)
        if "c_drift" not in drift:
            res = run_pool(task_replicate, [("drift", j, list(th), fit["rho"], sig, k_star)
                                            for j in range(DRIFT_FINAL_REPS)], a.workers, path,
                           fit["theta_ref"], "drift calibration")
            drift.update(k_star=k_star, lrs=[r["lr"] for r in res],
                         c_drift=float(np.quantile([r["lr"] for r in res], 0.95)))
            jsave(dfp, drift)
        thresholds["pd"] = drift["c_drift"]
        print(f"[drift] k* = {drift['k_star']:.3f}  c_drift = {drift['c_drift']:.3f}", flush=True)

    # 6 final profiles -------------------------------------------------------------------
    fpp = os.path.join(out, "profiles_final.json")
    final = jload(fpp)
    if final is None or set(final[0]) < {f"{n}_closed" for n in thresholds}:
        final = run_pool(task_profile, [(i, fit["u"], fit["ll"], fit["rho"], thresholds)
                                        for i in range(br.K)], min(a.workers, br.K), path,
                         fit["theta_ref"], "final profiles")
        final = sorted(final, key=lambda p: PARAM_NAMES.index(p["param"]))
        jsave(fpp, final)

    # 7 summary --------------------------------------------------------------------------
    primary = "pd" if diag["drift_present"] else "pb"
    summ = {"population": "highD location 1, E10,1.0", "episodes": int(S.M), "steps": int(S.N),
            "hours": round(S.N * br.DT / 3600, 3), "theta_hat": fit["theta"], "rho_hat": fit["rho"],
            "sigma_marginal": diag["sigma_marginal"], "thresholds": thresholds,
            "primary_threshold": primary, "V": diag["V"], "drift_present": diag["drift_present"],
            "better_max_refits": fit["refits"], "at_bound": fit["at_bound"],
            "gate_L_innovation_acf1": diag["innovation_acf_1_10"][0],
            "noise_bracket_D4.7": diag["noise_bracket_D4.7"],
            "null_coverage": {"chi2": diag["null_lr_coverage_chi2"], "boot": diag["null_lr_coverage_boot"]},
            "profiles": final}
    jsave(os.path.join(out, "summary.json"), summ)
    L = ["# Step 5a - highD location 1, E10,1.0 (real data)", "",
         f"{S.M} episodes, {summ['hours']} h. theta_hat = {np.round(th, 4).tolist()} "
         f"(v0, T, s0, a, b); rho = {fit['rho']:.4f}; sigma = {diag['sigma_marginal']:.3f} m",
         f"Thresholds: chi2_1 {br.CHI2_1:.2f}, chi2_5 {br.CHI2_K:.2f}, c_boot {c_boot:.2f}"
         + (f", c_drift {thresholds['pd']:.2f}" if "pd" in thresholds else ""),
         f"Drift statistic V = {diag['V']:.3f} (rule {V_RULE}) -> primary = {primary}",
         f"Gate L innovation ACF lag 1-3: {np.round(diag['innovation_acf_1_10'][:3], 3).tolist()}; "
         f"D4.7 bracket: {diag['noise_bracket_D4.7']}", "",
         "| param | theta_hat | " + " | ".join(f"{n} interval" for n in thresholds) + " |",
         "|---" * (len(thresholds) + 2) + "|"]
    for p, t in zip(final, th):
        cells = []
        for n in thresholds:
            lo_, hi_ = p.get(f"{n}_lo"), p.get(f"{n}_hi")
            cells.append(f"[{'open' if lo_ is None else round(lo_, 4)}, {'open' if hi_ is None else round(hi_, 4)}]")
        L.append(f"| {p['param']} | {t:.4f} | " + " | ".join(cells) + " |")
    md = "\n".join(L) + "\n"
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
