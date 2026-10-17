"""A0-pre-1 : does the 95% confidence region actually cover theta* ?

WHY THIS EXPERIMENT EXISTS
--------------------------
The whole project rests on one claim: that a likelihood confidence region for the
background-driver parameters has the coverage it advertises, so that its image under
the safety functional inherits that coverage. `07_STATISTICAL_SPECIFICATION.md` (S2.1)
warns that a sum-of-squares objective treated as a log-likelihood assumes iid
residuals, while real car-following residuals are strongly autocorrelated - and that
if so, the region comes out several times too narrow and every downstream number is an
artefact.

That warning is currently an argument. This experiment turns it into a measurement.

DESIGN
------
Ground truth is known by construction: we generate the follower ourselves with a fixed
theta*, then add residual noise with a controlled correlation structure.

  factor 1  noise structure   : iid | AR(1) rho=0.90 | AR(1) rho=0.98
                                (marginal sd held CONSTANT across all three, so only
                                 the correlation changes)
  factor 2  likelihood used   : naive SSE | AR(1)-aware

  outcome   : fraction of replications in which
              2[ell(theta_hat) - ell(theta*)] <= chi2(0.95, 5) = 11.0705

POSITIVE CONTROL FOR THE EXPERIMENT ITSELF
------------------------------------------
Cell (iid noise, naive SSE likelihood) MUST return coverage near 0.95. It is the cell
where the naive likelihood's own assumption holds exactly. If that cell does not come
out near 0.95, the implementation is wrong and NO other cell may be interpreted. This
is checked and reported first, before anything else.

PRE-REGISTERED PREDICTION (written before the first run)
--------------------------------------------------------
  (iid,  naive) ~ 0.95      <- positive control
  (iid,  ar1)   ~ 0.95      <- AR(1) likelihood should not be harmed by rho_true = 0
  (0.90, naive) << 0.95     <- effective n inflated ~19x, region ~4.4x too narrow
  (0.98, naive) <<< 0.95
  (0.90, ar1)   ~ 0.95
  (0.98, ar1)   ~ 0.95

A result in which (0.90, naive) comes out near 0.95 would mean the hazard was
overstated, and that is to be reported as such rather than explained away.

Usage:  python a0pre_01_coverage.py [--reps 200] [--quick]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from _common import SRC  # noqa: E402,F401  (puts code/src on sys.path)

from idm import (PARAM_BOUNDS, PARAM_NAMES, make_leader_trace,  # noqa: E402
                 simulate_follower)
from likelihood import (ar1_loglik, ar1_rho_of, gen_ar1_noise,  # noqa: E402
                        lr_statistic, naive_sse_loglik)

# ----------------------------------------------------------------------------- config
THETA_STAR = np.array([30.0, 1.2, 2.0, 1.5, 2.0])   # v0, T, s0, a, b
DT = 0.04                 # 25 Hz, matching highD / exiD
DURATION = 20.0           # s  (highD median vehicle visibility is 13.6 s; 20 s is a
                          #     deliberately GENEROUS trace - see diary entry)
S_INIT = 30.0
V_INIT = 28.0
NOISE_SD = 0.30           # m, marginal sd of spacing residual
K_PARAMS = 5
CHI2_CRIT = float(chi2.ppf(0.95, K_PARAMS))   # 11.0705

NOISE_CELLS = [("iid", 0.0), ("ar1_rho0.90", 0.90), ("ar1_rho0.98", 0.98)]
LIK_CELLS = ["naive_sse", "ar1"]


def negloglik(theta, s_obs, v_lead, which):
    lo = np.array([b[0] for b in PARAM_BOUNDS])
    hi = np.array([b[1] for b in PARAM_BOUNDS])
    if np.any(theta < lo) or np.any(theta > hi):
        return 1e10
    s_sim, _, ok = simulate_follower(v_lead, DT, theta, S_INIT, V_INIT)
    if not ok:
        return 1e10                      # gap floor hit: unphysical, reject
    resid = s_obs - s_sim
    ll = naive_sse_loglik(resid) if which == "naive_sse" else ar1_loglik(resid)
    if not np.isfinite(ll):
        return 1e10
    return -ll


def fit(s_obs, v_lead, which, rng, n_starts=2):
    """MLE with a perturbed start plus restarts. Returns (theta_hat, ll_hat, info)."""
    best = None
    for j in range(n_starts):
        if j == 0:
            x0 = THETA_STAR * (1.0 + 0.10 * rng.standard_normal(K_PARAMS))
        else:
            x0 = THETA_STAR * (1.0 + 0.25 * rng.standard_normal(K_PARAMS))
        x0 = np.clip(x0,
                     [b[0] * 1.05 for b in PARAM_BOUNDS],
                     [b[1] * 0.95 for b in PARAM_BOUNDS])
        res = minimize(negloglik, x0, args=(s_obs, v_lead, which),
                       method="L-BFGS-B", bounds=PARAM_BOUNDS,
                       options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    theta_hat = np.array(best.x)
    ll_hat = -best.fun
    # a parameter resting on a bound invalidates the chi-square asymptotics, so it is
    # counted PER PARAMETER rather than as a single flag - run 1 of this experiment
    # showed every replication hitting a bound and the aggregate flag did not say which.
    which_bound = [
        PARAM_NAMES[i] for i in range(K_PARAMS)
        if abs(theta_hat[i] - PARAM_BOUNDS[i][0]) < 1e-6
        or abs(theta_hat[i] - PARAM_BOUNDS[i][1]) < 1e-6
    ]
    return theta_hat, ll_hat, {"bound_params": which_bound,
                               "on_bound": bool(which_bound),
                               "success": bool(best.success)}


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial CI. Needed because a normal-approximation interval has zero
    width at k == n, which makes any verdict rule built on it impossible to pass.
    Run 1 of this experiment failed its own positive control for exactly that reason -
    the gate could not pass for the right reason, which is the error class this
    project exists to avoid."""
    from scipy.stats import beta as _beta
    lo = 0.0 if k == 0 else float(_beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def run(reps, seed=20260920):
    rng = np.random.default_rng(seed)
    _, v_lead = make_leader_trace(DT, DURATION, rng, kind="informative")
    s_true, v_true, ok = simulate_follower(v_lead, DT, THETA_STAR, S_INIT, V_INIT)
    if not ok:
        raise RuntimeError("theta* trace hits the gap floor - change the scenario")

    n = len(v_lead)
    print(f"trace: n={n} steps, {DURATION:.0f}s at {1/DT:.0f} Hz, "
          f"gap range {s_true.min():.1f}-{s_true.max():.1f} m, "
          f"speed range {v_true.min():.1f}-{v_true.max():.1f} m/s")
    print(f"chi2(0.95, {K_PARAMS}) = {CHI2_CRIT:.4f}\n")

    results = {}
    for noise_name, rho_true in NOISE_CELLS:
        # noise realisations are shared across the two likelihoods so the comparison
        # is paired - the only thing that differs between the two cells is the
        # likelihood, not the data.
        noise_rng = np.random.default_rng(seed + 1000 + int(rho_true * 100))
        noises = [gen_ar1_noise(n, rho_true, NOISE_SD, noise_rng) for _ in range(reps)]

        for which in LIK_CELLS:
            t0 = time.time()
            covered, lrs, bound_hits, fails, rhos = 0, [], 0, 0, []
            bound_by_param = {p: 0 for p in PARAM_NAMES}
            fit_rng = np.random.default_rng(seed + 77)
            for r in range(reps):
                s_obs = s_true + noises[r]
                theta_hat, ll_hat, info = fit(s_obs, v_lead, which, fit_rng)
                if info["on_bound"]:
                    bound_hits += 1
                    for p in info["bound_params"]:
                        bound_by_param[p] += 1
                if not info["success"]:
                    fails += 1
                ll_star = -negloglik(THETA_STAR, s_obs, v_lead, which)
                lr = lr_statistic(ll_hat, ll_star)
                lrs.append(lr)
                if lr <= CHI2_CRIT:
                    covered += 1
                if which == "naive_sse":
                    s_sim, _, _ = simulate_follower(v_lead, DT, theta_hat, S_INIT, V_INIT)
                    rhos.append(ar1_rho_of(s_obs - s_sim))
            lrs = np.array(lrs)
            cov = covered / reps
            cp_lo, cp_hi = clopper_pearson(covered, reps)
            cell = {
                "noise": noise_name, "rho_true": rho_true, "likelihood": which,
                "reps": reps, "coverage": cov,
                "coverage_ci95": [cp_lo, cp_hi],
                "lr_median": float(np.median(lrs)),
                "lr_q90": float(np.quantile(lrs, 0.90)),
                "lr_mean": float(np.mean(lrs)),
                "chi2_5_median_ref": float(chi2.ppf(0.5, K_PARAMS)),
                # LR scales like effective sample size, so LR_median / chi2_median
                # estimates how far the likelihood is over-confident. Its square root
                # is roughly the factor by which the confidence region is too narrow
                # in each parameter direction. Theory predicts ~(1+rho)/(1-rho) for
                # the naive likelihood: 1, 19 and 99 for the three noise cells.
                "implied_information_inflation": float(
                    np.median(lrs) / chi2.ppf(0.5, K_PARAMS)),
                "implied_width_deflation": float(
                    np.sqrt(np.median(lrs) / chi2.ppf(0.5, K_PARAMS))),
                "bound_hits": bound_hits, "bound_by_param": bound_by_param,
                "optim_failures": fails,
                "resid_acf1_median": float(np.median(rhos)) if rhos else None,
                "seconds": round(time.time() - t0, 1),
            }
            results[f"{noise_name}|{which}"] = cell
            bstr = ",".join(f"{p}:{c}" for p, c in bound_by_param.items() if c) or "-"
            print(f"{noise_name:>12s} | {which:>9s} : coverage {cov:5.3f} "
                  f"CI[{cp_lo:.3f},{cp_hi:.3f}]  LRmed {np.median(lrs):8.2f}  "
                  f"bounds[{bstr}]  [{cell['seconds']}s]")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="5 reps, smoke test only")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    reps = 5 if args.quick else args.reps

    print("=" * 78)
    print("A0-pre-1 : coverage of the chi2(0.95,5) likelihood confidence region")
    print("=" * 78)
    results = run(reps)

    print("\n--- POSITIVE CONTROL FOR THE EXPERIMENT ---")
    pc = results["iid|naive_sse"]
    lo, hi = pc["coverage_ci95"]
    verdict = "PASS" if (lo <= 0.95 <= hi) else "FAIL"
    print(f"(iid, naive_sse) coverage = {pc['coverage']:.3f}, "
          f"exact 95% CI [{lo:.3f}, {hi:.3f}] -> {verdict}")
    print(f"  LR median {pc['lr_median']:.2f} vs chi2(5) median "
          f"{pc['chi2_5_median_ref']:.2f} (these should be close)")
    if verdict == "FAIL":
        print("  The implementation is wrong. Do NOT interpret any other cell.")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": {
                "theta_star": THETA_STAR.tolist(), "param_names": list(PARAM_NAMES),
                "dt": DT, "duration": DURATION, "noise_sd": NOISE_SD,
                "chi2_crit": CHI2_CRIT, "reps": reps},
                "results": results}, f, indent=2)
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
