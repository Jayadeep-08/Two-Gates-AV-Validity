"""A0-pre-2 : does pooling traces restore the coverage that A0-pre-1 lost?

WHY THIS EXPERIMENT EXISTS
--------------------------
A0-pre-1 (200 reps) produced an unplanned second finding. With the CORRECTLY specified
AR(1) likelihood, coverage of the chi2(0.95,5) region was:

    rho = 0.90 : 0.895   CI [0.844, 0.934]   - excludes 0.95
    rho = 0.98 : 0.735   CI [0.668, 0.795]   - excludes 0.95 badly

So the asymptotic chi-square threshold undercovers even when the likelihood is right.
The obvious suspect is effective sample size. For an AR(1) series,

    n_eff = n * (1 - rho) / (1 + rho)

and a 500-sample 20 s trace at 25 Hz carries n_eff = 26 at rho = 0.90 and n_eff = 5.1
at rho = 0.98 - fewer independent observations than there are parameters. Wilks'
theorem has no reason to hold there.

If that diagnosis is right it has a direct design consequence for the main study:
the confidence region must be built from MANY traces, because the real sample size is
the number of independent traces, not the number of timesteps. Reporting "n = 340
observations" for a single highD trace would be badly misleading.

DESIGN
------
  factor 1  n_traces pooled : 1 | 4 | 16   (all sharing one theta*, homogeneous by
                                            construction, so this isolates sample size
                                            from driver heterogeneity)
  factor 2  residual rho    : 0.90 | 0.98
  likelihood                : AR(1)-aware only (the naive one is already dead)
  outcome                   : coverage of chi2(0.95, 5)

Each pooled trace gets its own leader profile realisation, so pooling adds genuinely
new information rather than repeating the same experiment.

PRE-REGISTERED PREDICTION (written before the first run)
--------------------------------------------------------
Coverage rises toward 0.95 as n_traces grows, and the rise is faster at rho = 0.90
than at rho = 0.98. Specifically: at 16 traces, rho = 0.90 should reach nominal
coverage; rho = 0.98 may still fall short, because 16 x 5.1 = 82 effective
observations for 5 parameters plus 2 nuisance is still not a large sample.

If coverage does NOT improve with pooling, the diagnosis is wrong, the problem is not
sample size, and the bootstrap-calibrated threshold becomes mandatory rather than a
cross-check. That outcome is to be reported as such.

Usage:  python a0pre_02_pooling.py [--reps 60] [--quick]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as _beta
from scipy.stats import chi2

from _common import SRC  # noqa: E402,F401  (puts code/src on sys.path)

from idm import (PARAM_BOUNDS, PARAM_NAMES, make_leader_trace,  # noqa: E402
                 simulate_follower)
from likelihood import (ar1_loglik_multi, effective_n,  # noqa: E402
                        gen_ar1_noise, lr_statistic)

THETA_STAR = np.array([30.0, 1.2, 2.0, 1.5, 2.0])
DT = 0.04
DURATION = 20.0
S_INIT = 30.0
V_INIT = 28.0
NOISE_SD = 0.30
K_PARAMS = 5
CHI2_CRIT = float(chi2.ppf(0.95, K_PARAMS))

N_TRACE_CELLS = [1, 4, 16]
RHO_CELLS = [0.90, 0.98]


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else float(_beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def build_traces(n_traces, rng):
    """Distinct leader profiles, each perturbed in phase and amplitude so pooling adds
    information instead of repeating one experiment."""
    traces = []
    for _ in range(n_traces):
        _, v_lead = make_leader_trace(DT, DURATION, kind="informative")
        jitter = (0.9 + 0.2 * rng.random()) * np.sin(
            2 * np.pi * np.arange(len(v_lead)) * DT / (3.0 + 2.0 * rng.random()))
        v_lead = np.maximum(v_lead + jitter, 3.0)
        s_true, _, ok = simulate_follower(v_lead, DT, THETA_STAR, S_INIT, V_INIT)
        if not ok:
            continue
        traces.append((v_lead, s_true))
    return traces


def negloglik(theta, traces, s_obs_list):
    lo = np.array([b[0] for b in PARAM_BOUNDS])
    hi = np.array([b[1] for b in PARAM_BOUNDS])
    if np.any(theta < lo) or np.any(theta > hi):
        return 1e10
    resids = []
    for (v_lead, _), s_obs in zip(traces, s_obs_list):
        s_sim, _, ok = simulate_follower(v_lead, DT, theta, S_INIT, V_INIT)
        if not ok:
            return 1e10
        resids.append(s_obs - s_sim)
    ll = ar1_loglik_multi(resids)
    return 1e10 if not np.isfinite(ll) else -ll


def fit(traces, s_obs_list, rng, n_starts=2):
    best = None
    for j in range(n_starts):
        scale = 0.10 if j == 0 else 0.25
        x0 = THETA_STAR * (1.0 + scale * rng.standard_normal(K_PARAMS))
        x0 = np.clip(x0, [b[0] * 1.05 for b in PARAM_BOUNDS],
                     [b[1] * 0.95 for b in PARAM_BOUNDS])
        res = minimize(negloglik, x0, args=(traces, s_obs_list), method="L-BFGS-B",
                       bounds=PARAM_BOUNDS,
                       options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    th = np.array(best.x)
    bound_params = [PARAM_NAMES[i] for i in range(K_PARAMS)
                    if abs(th[i] - PARAM_BOUNDS[i][0]) < 1e-6
                    or abs(th[i] - PARAM_BOUNDS[i][1]) < 1e-6]
    return th, -best.fun, bound_params


def run(reps, seed=20260920):
    n_steps = int(round(DURATION / DT))
    print(f"trace length {n_steps} steps ({DURATION:.0f}s at {1/DT:.0f} Hz)")
    for rho in RHO_CELLS:
        print(f"  n_eff per trace at rho={rho:.2f} : {effective_n(n_steps, rho):.1f}")
    print(f"chi2(0.95,{K_PARAMS}) = {CHI2_CRIT:.4f}\n")

    results = {}
    for rho in RHO_CELLS:
        for n_tr in N_TRACE_CELLS:
            t0 = time.time()
            trace_rng = np.random.default_rng(seed + 11 * n_tr)
            traces = build_traces(n_tr, trace_rng)
            noise_rng = np.random.default_rng(seed + 500 + int(rho * 100) + n_tr)
            fit_rng = np.random.default_rng(seed + 77)

            covered, lrs, bound_hits = 0, [], 0
            for _ in range(reps):
                s_obs_list = [s_true + gen_ar1_noise(len(s_true), rho, NOISE_SD, noise_rng)
                              for (_, s_true) in traces]
                th, ll_hat, bp = fit(traces, s_obs_list, fit_rng)
                if bp:
                    bound_hits += 1
                ll_star = -negloglik(THETA_STAR, traces, s_obs_list)
                lr = lr_statistic(ll_hat, ll_star)
                lrs.append(lr)
                if lr <= CHI2_CRIT:
                    covered += 1
            lrs = np.array(lrs)
            lo, hi = clopper_pearson(covered, reps)
            cell = {
                "rho": rho, "n_traces": n_tr, "reps": reps,
                "n_eff_total": float(effective_n(n_steps, rho) * n_tr),
                "coverage": covered / reps, "coverage_ci95": [lo, hi],
                "lr_median": float(np.median(lrs)),
                "chi2_5_median_ref": float(chi2.ppf(0.5, K_PARAMS)),
                "bound_hits": bound_hits,
                "seconds": round(time.time() - t0, 1),
            }
            results[f"rho{rho}|traces{n_tr}"] = cell
            print(f"rho={rho:.2f} traces={n_tr:2d} (n_eff~{cell['n_eff_total']:6.1f}) : "
                  f"coverage {cell['coverage']:5.3f} CI[{lo:.3f},{hi:.3f}]  "
                  f"LRmed {np.median(lrs):6.2f}  bounds {bound_hits:3d}  "
                  f"[{cell['seconds']}s]")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    reps = 4 if args.quick else args.reps

    print("=" * 78)
    print("A0-pre-2 : coverage vs number of pooled traces")
    print("=" * 78)
    results = run(reps)

    print("\n--- READING ---")
    for rho in RHO_CELLS:
        covs = [results[f"rho{rho}|traces{n}"]["coverage"] for n in N_TRACE_CELLS]
        trend = "rises" if covs[-1] > covs[0] + 0.02 else "flat/falls"
        print(f"rho={rho:.2f}: coverage {covs} -> {trend}")
    print("If coverage does not rise with pooling, sample size is not the cause and "
          "the bootstrap-calibrated threshold becomes mandatory.")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": {
                "theta_star": THETA_STAR.tolist(), "dt": DT, "duration": DURATION,
                "noise_sd": NOISE_SD, "chi2_crit": CHI2_CRIT, "reps": reps},
                "results": results}, f, indent=2)
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
