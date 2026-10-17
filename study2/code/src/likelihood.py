"""Log-likelihoods for car-following calibration, and the likelihood-ratio machinery.

Two likelihoods are provided:

  naive_sse_loglik   - Gaussian iid residuals, sigma profiled out analytically.
                       This is what a sum-of-squares objective implicitly assumes,
                       and it is what most published car-following calibrations use.

  ar1_loglik         - Exact Gaussian AR(1) residuals, sigma profiled out
                       analytically and rho profiled out numerically.

The point of having both is that the difference between them is the subject of
experiment A0-pre-1. Neither is assumed correct; the experiment measures which one
delivers the coverage it advertises.

Dependencies: numpy, scipy.
"""

import numpy as np
from scipy.optimize import minimize_scalar


def naive_sse_loglik(resid):
    """Concentrated Gaussian iid log-likelihood, sigma profiled out.

    ell(theta) = -n/2 * log(SSE(theta)/n)   (additive constants dropped)

    Dropping constants is safe because every use here is a DIFFERENCE of
    log-likelihoods at the same n.
    """
    n = resid.size
    sse = float(np.dot(resid, resid))
    if sse <= 0.0:
        sse = 1e-300
    return -0.5 * n * np.log(sse / n)


def _ar1_concentrated(resid, rho):
    """Exact AR(1) Gaussian log-likelihood with sigma profiled out.

    Prais-Winsten transform:
        rt_1 = sqrt(1 - rho^2) * r_1
        rt_t = r_t - rho * r_{t-1},  t >= 2

    ell(rho) = 0.5*log(1 - rho^2) - n/2 * log(S(rho)/n)    (constants dropped)
    """
    n = resid.size
    if abs(rho) >= 0.999999:
        return -np.inf
    rt = np.empty(n)
    rt[0] = np.sqrt(1.0 - rho * rho) * resid[0]
    rt[1:] = resid[1:] - rho * resid[:-1]
    s = float(np.dot(rt, rt))
    if s <= 0.0:
        s = 1e-300
    return 0.5 * np.log(1.0 - rho * rho) - 0.5 * n * np.log(s / n)


def ar1_loglik(resid, return_rho=False):
    """AR(1) log-likelihood with BOTH sigma and rho profiled out.

    rho is a nuisance parameter: it is maximised over at every theta, which is what
    keeps the likelihood-ratio statistic for theta on 5 degrees of freedom.
    """
    res = minimize_scalar(lambda r: -_ar1_concentrated(resid, r),
                          bounds=(-0.9999, 0.9999), method="bounded",
                          options={"xatol": 1e-6})
    ll = -res.fun
    if return_rho:
        return ll, float(res.x)
    return ll


def _ar1_concentrated_multi(resid_list, rho):
    """Joint exact AR(1) log-likelihood over several independent traces, with a
    SHARED rho and a SHARED sigma, both profiled out.

    Shared rather than per-trace because in the synthetic experiments the traces are
    generated with a common residual process, so sharing is correct by construction
    and avoids paying M extra nuisance parameters. On real data this becomes an
    assumption and must be checked per driver.

        ell(rho) = (M/2) log(1 - rho^2) - (N/2) log( S(rho) / N )
    """
    if abs(rho) >= 0.999999:
        return -np.inf
    s_tot = 0.0
    n_tot = 0
    for resid in resid_list:
        n = resid.size
        rt = np.empty(n)
        rt[0] = np.sqrt(1.0 - rho * rho) * resid[0]
        rt[1:] = resid[1:] - rho * resid[:-1]
        s_tot += float(np.dot(rt, rt))
        n_tot += n
    if s_tot <= 0.0:
        s_tot = 1e-300
    m = len(resid_list)
    return 0.5 * m * np.log(1.0 - rho * rho) - 0.5 * n_tot * np.log(s_tot / n_tot)


def ar1_loglik_multi(resid_list, return_rho=False):
    """AR(1) log-likelihood over several traces, sigma and rho profiled out."""
    res = minimize_scalar(lambda r: -_ar1_concentrated_multi(resid_list, r),
                          bounds=(-0.9999, 0.9999), method="bounded",
                          options={"xatol": 1e-6})
    ll = -res.fun
    if return_rho:
        return ll, float(res.x)
    return ll


def effective_n(n, rho):
    """Effective sample size of an AR(1) series: n * (1-rho)/(1+rho).

    This is the number that matters for asymptotics, not n. At 25 Hz with rho = 0.98
    a 500-sample trace carries about 5 independent observations, which is fewer than
    the number of parameters being estimated.
    """
    return n * (1.0 - rho) / (1.0 + rho)


def lr_statistic(ll_hat, ll_theta):
    """2 * [ ell(theta_hat) - ell(theta) ].  Compared against chi2(alpha, k)."""
    return 2.0 * (ll_hat - ll_theta)


def ar1_rho_of(resid):
    """Lag-1 autocorrelation of a residual series. Diagnostic only."""
    r = resid - resid.mean()
    denom = float(np.dot(r, r))
    if denom <= 0:
        return 0.0
    return float(np.dot(r[1:], r[:-1]) / denom)


def gen_ar1_noise(n, rho, sd, rng):
    """AR(1) noise with the requested MARGINAL standard deviation.

    Marginal sd is held fixed across rho so that changing rho changes only the
    correlation structure and not the amount of noise. Without this the comparison
    in A0-pre-1 would confound the two.
    """
    innov_sd = sd * np.sqrt(1.0 - rho * rho)
    e = np.empty(n)
    e[0] = rng.normal(0.0, sd)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal(0.0, innov_sd)
    return e
