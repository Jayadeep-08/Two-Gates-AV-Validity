"""Step 4 engine: batched IDM simulation, pooled AR(1) likelihood, fitting, profiles
and the episode-bootstrap threshold. Used by the synthetic bridge (Step 4) and, later,
unchanged, by real fitting (Step 5). Pre-registration: 05_PREREGISTRATION_DRAFT.md
Amendment G.

Design choices that matter:
- Episodes are packed into flat arrays with offsets, so one compiled call simulates a
  whole pooled sample. numba is used when installed (about 50x faster than the Python
  loop); without it a pure-Python fallback gives identical numbers, slowly.
- The follower is driven by the leader's OBSERVED rear position (Step 3,
  simulate_follower_pos); the ballistic update and clipping are identical to
  code/src/idm.py and are checked against it in the tests.
- The pooled AR(1) likelihood (shared rho and sigma, both profiled out) is computed
  from four running sums, so the rho profile costs microseconds:
      S(rho) = (1-rho^2) F + Q - 2 rho P + rho^2 R
      F = sum_e r_e1^2, Q = sum_{t>=2} r_t^2, P = sum_{t>=2} r_t r_{t-1}, R = sum_{t>=2} r_{t-1}^2
      l(rho) = (M/2) log(1-rho^2) - (N/2) log(S(rho)/N)
  identical to likelihood.ar1_loglik_multi (tested).
- Each episode's initial gap is observed with noise. The simulation starts at the
  observed gap and the initial-gap error is a per-episode nuisance delta_e acting through
  the tangent-linear sensitivity h(t) = ds(t)/ds_init (Amendment G.8, found by the
  positive control). Since G.9 the delta_e are INTEGRATED out (flat prior; REML-type),
  not maximised over: maximising over M incidental offsets biased rho down and inflated
  the LR statistic at high rho (Step 4 run 1, DIARY Entry 25). 12 per-episode sums.
- Since G.10 the Jacobian term of that integration is evaluated at a FIXED reference
  theta_ref (a first-stage G.9 fit), as in Lindstrom & Bates' REML for nonlinear mixed
  models: with h(theta) in the Jacobian, -1/2 sum log A_hh(theta) biased a and b upward
  by a constant amount (Step 4 run 2, DIARY Entry 26). fit() does the two stages.
- Parameters are optimised in bound-normalised coordinates u = (theta-lo)/(hi-lo).
- A gap that hits the floor is clipped, not rejected: the likelihood stays continuous in
  theta, which keeps L-BFGS-B well behaved. Floor hits are counted and reported.
"""

import math
import time

import numpy as np
from scipy.optimize import minimize

from idm import PARAM_BOUNDS, PARAM_NAMES

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:                                  # pragma: no cover
    HAVE_NUMBA = False
    prange = range

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

LO = np.array([b[0] for b in PARAM_BOUNDS])
HI = np.array([b[1] for b in PARAM_BOUNDS])
SPAN = HI - LO
K = len(PARAM_NAMES)
DT = 0.04
S_FLOOR = 0.5
A_LO, A_HI = -9.0, 4.0
CHI2_1, CHI2_K = 3.841458820694124, 11.070497693516351      # chi2(0.95,1), chi2(0.95,5)


def to_theta(u):
    return LO + np.asarray(u, dtype=float) * SPAN


def to_u(theta):
    return (np.asarray(theta, dtype=float) - LO) / SPAN


# ------------------------------------------------------------------------------------
# simulation
# ------------------------------------------------------------------------------------
@njit(cache=True)
def _sim_idm(ul, vl, off, x0, v0, theta, dt, s_floor, s_out):
    """IDM follower against observed leader rear position; returns number of floor hits."""
    V0, T, S0, A, B = theta[0], theta[1], theta[2], theta[3], theta[4]
    inv_v0 = 1.0 / V0
    tsab = 2.0 * math.sqrt(A * B)
    hits = 0
    for e in range(off.shape[0] - 1):
        i0, i1 = off[e], off[e + 1]
        x = x0[e]
        v = v0[e]
        s = ul[i0] - x
        s_out[i0] = s
        for t in range(i0, i1 - 1):
            dv = v - vl[t]
            ss = S0 + v * T + (v * dv) / tsab
            if ss < S0:
                ss = S0
            se = s if s > 1e-3 else 1e-3
            ratio = ss / se
            q = v * inv_v0
            acc = A * (1.0 - q * q * q * q - ratio * ratio)
            if acc < -9.0:
                acc = -9.0
            elif acc > 4.0:
                acc = 4.0
            vn = v + acc * dt
            if vn < 0.0:
                vn = 0.0
            x = x + 0.5 * (v + vn) * dt
            sn = ul[t + 1] - x
            if sn < s_floor:
                x = ul[t + 1] - s_floor
                sn = s_floor
                hits += 1
            s_out[t + 1] = sn
            s = sn
            v = vn
    return hits


@njit(cache=True)
def _sim_delayed(ul, vl, off, x0, v0, theta, dt, s_floor, lag, s_out, v_buf):
    """Misspecification truth (Amendment G, arm M): IDM whose acceleration at step t is
    computed from the state at step t-lag (reaction delay, no compensation). History
    before the episode start is the initial state. lag = 0 reproduces _sim_idm."""
    V0, T, S0, A, B = theta[0], theta[1], theta[2], theta[3], theta[4]
    inv_v0 = 1.0 / V0
    tsab = 2.0 * math.sqrt(A * B)
    hits = 0
    for e in range(off.shape[0] - 1):
        i0, i1 = off[e], off[e + 1]
        x = x0[e]
        v = v0[e]
        s_out[i0] = ul[i0] - x
        v_buf[i0] = v
        for t in range(i0, i1 - 1):
            td = t - lag
            if td < i0:
                td = i0
            sd = s_out[td]
            vd = v_buf[td]
            dv = vd - vl[td]
            ss = S0 + vd * T + (vd * dv) / tsab
            if ss < S0:
                ss = S0
            se = sd if sd > 1e-3 else 1e-3
            ratio = ss / se
            q = vd * inv_v0
            acc = A * (1.0 - q * q * q * q - ratio * ratio)
            if acc < -9.0:
                acc = -9.0
            elif acc > 4.0:
                acc = 4.0
            vn = v + acc * dt
            if vn < 0.0:
                vn = 0.0
            x = x + 0.5 * (v + vn) * dt
            sn = ul[t + 1] - x
            if sn < s_floor:
                x = ul[t + 1] - s_floor
                sn = s_floor
                hits += 1
            s_out[t + 1] = sn
            v_buf[t + 1] = vn
            v = vn
    return hits


@njit(cache=True)
def _ar1_sums(r, off):
    F = 0.0
    Q = 0.0
    P = 0.0
    R = 0.0
    for e in range(off.shape[0] - 1):
        i0, i1 = off[e], off[e + 1]
        F += r[i0] * r[i0]
        for t in range(i0 + 1, i1):
            Q += r[t] * r[t]
            P += r[t] * r[t - 1]
            R += r[t - 1] * r[t - 1]
    return F, Q, P, R


@njit(cache=True)
def _ar1_ll(F, Q, P, R, M, N):
    """Maximise l(rho) over |rho| < 0.9999: grid in atanh space, then golden section."""
    best_z = 0.0
    best = -1e300
    for j in range(81):
        z = -4.95 + 9.9 * j / 80.0
        rho = math.tanh(z)
        S = (1.0 - rho * rho) * F + Q - 2.0 * rho * P + rho * rho * R
        if S <= 0.0:
            S = 1e-300
        ll = 0.5 * M * math.log(1.0 - rho * rho) - 0.5 * N * math.log(S / N)
        if ll > best:
            best = ll
            best_z = z
    a = best_z - 0.13
    b = best_z + 0.13
    g = 0.6180339887498949
    c = b - g * (b - a)
    d = a + g * (b - a)
    for _ in range(80):
        rc = math.tanh(c)
        rd = math.tanh(d)
        Sc = (1.0 - rc * rc) * F + Q - 2.0 * rc * P + rc * rc * R
        Sd = (1.0 - rd * rd) * F + Q - 2.0 * rd * P + rd * rd * R
        if Sc <= 0.0:
            Sc = 1e-300
        if Sd <= 0.0:
            Sd = 1e-300
        lc = 0.5 * M * math.log(1.0 - rc * rc) - 0.5 * N * math.log(Sc / N)
        ld = 0.5 * M * math.log(1.0 - rd * rd) - 0.5 * N * math.log(Sd / N)
        if lc > ld:
            b = d
        else:
            a = c
        c = b - g * (b - a)
        d = a + g * (b - a)
    z = 0.5 * (a + b)
    rho = math.tanh(z)
    S = (1.0 - rho * rho) * F + Q - 2.0 * rho * P + rho * rho * R
    if S <= 0.0:
        S = 1e-300
    ll = 0.5 * M * math.log(1.0 - rho * rho) - 0.5 * N * math.log(S / N)
    if best > ll:
        return best, math.tanh(best_z)
    return ll, rho


@njit(cache=True)
def _ar1_filter(innov, off, rho, sd):
    """AR(1) noise per episode with the given MARGINAL sd (as likelihood.gen_ar1_noise)."""
    out = np.empty_like(innov)
    k = math.sqrt(1.0 - rho * rho)
    for e in range(off.shape[0] - 1):
        i0, i1 = off[e], off[e + 1]
        out[i0] = sd * innov[i0]
        for t in range(i0 + 1, i1):
            out[t] = rho * out[t - 1] + sd * k * innov[t]
    return out


@njit(cache=True)
def _sim_idm_sens(ul, vl, off, x0, v0, theta, dt, s_floor, s_out, h_out):
    """_sim_idm plus the tangent-linear sensitivity h(t) = d s(t) / d s_init of the
    simulated gap to the episode's initial gap (Amendment G.8). Same update, same
    clipping; a clipped acceleration, speed or gap has zero derivative."""
    V0, T, S0, A, B = theta[0], theta[1], theta[2], theta[3], theta[4]
    inv_v0 = 1.0 / V0
    tsab = 2.0 * math.sqrt(A * B)
    hits = 0
    for e in range(off.shape[0] - 1):
        i0, i1 = off[e], off[e + 1]
        x = x0[e]
        v = v0[e]
        X = -1.0                  # dx/ds_init  (x = u_lead(0) - s_init)
        W = 0.0                   # dv/ds_init
        s = ul[i0] - x
        s_out[i0] = s
        h_out[i0] = 1.0
        for t in range(i0, i1 - 1):
            dv = v - vl[t]
            ss = S0 + v * T + (v * dv) / tsab
            dss_dv = T + (2.0 * v - vl[t]) / tsab
            if ss < S0:
                ss = S0
                dss_dv = 0.0
            if s > 1e-3:
                se = s
                dse = 1.0
            else:
                se = 1e-3
                dse = 0.0
            ratio = ss / se
            q = v * inv_v0
            acc = A * (1.0 - q * q * q * q - ratio * ratio)
            da_ds = 2.0 * A * ratio * ratio / se * dse
            da_dv = A * (-4.0 * q * q * q * inv_v0 - 2.0 * ratio / se * dss_dv)
            if acc < -9.0:
                acc = -9.0
                da_ds = 0.0
                da_dv = 0.0
            elif acc > 4.0:
                acc = 4.0
                da_ds = 0.0
                da_dv = 0.0
            dacc = da_ds * (-X) + da_dv * W
            vn = v + acc * dt
            Wn = W + dacc * dt
            if vn < 0.0:
                vn = 0.0
                Wn = 0.0
            x = x + 0.5 * (v + vn) * dt
            X = X + 0.5 * (W + Wn) * dt
            sn = ul[t + 1] - x
            if sn < s_floor:
                x = ul[t + 1] - s_floor
                X = 0.0
                sn = s_floor
                hits += 1
            s_out[t + 1] = sn
            h_out[t + 1] = -X
            s = sn
            v = vn
            W = Wn
    return hits


@njit(cache=True)
def _ep_sums(r, h, off):
    """Per-episode sums for the Prais-Winsten quadratic forms in rho of r and h:
    for each pair (a,b) in (rr, rh, hh): F = a0 b0, Q = sum_{t>=1} a_t b_t,
    P = sum_{t>=1} (a_t b_{t-1} + a_{t-1} b_t), R = sum_{t>=1} a_{t-1} b_{t-1}."""
    M = off.shape[0] - 1
    out = np.zeros((M, 12))
    for e in range(M):
        i0, i1 = off[e], off[e + 1]
        out[e, 0] = r[i0] * r[i0]
        out[e, 4] = r[i0] * h[i0]
        out[e, 8] = h[i0] * h[i0]
        for t in range(i0 + 1, i1):
            out[e, 1] += r[t] * r[t]
            out[e, 2] += 2.0 * r[t] * r[t - 1]
            out[e, 3] += r[t - 1] * r[t - 1]
            out[e, 5] += r[t] * h[t]
            out[e, 6] += r[t] * h[t - 1] + r[t - 1] * h[t]
            out[e, 7] += r[t - 1] * h[t - 1]
            out[e, 9] += h[t] * h[t]
            out[e, 10] += 2.0 * h[t] * h[t - 1]
            out[e, 11] += h[t - 1] * h[t - 1]
    return out


@njit(cache=True)
def _S_profiled(sums, rho):
    """sum over episodes of min_delta || PW_rho (r - delta h) ||^2."""
    S = 0.0
    c = 1.0 - rho * rho
    for e in range(sums.shape[0]):
        arr = c * sums[e, 0] + sums[e, 1] - rho * sums[e, 2] + rho * rho * sums[e, 3]
        arh = c * sums[e, 4] + sums[e, 5] - rho * sums[e, 6] + rho * rho * sums[e, 7]
        ahh = c * sums[e, 8] + sums[e, 9] - rho * sums[e, 10] + rho * rho * sums[e, 11]
        S += arr - (arh * arh / ahh if ahh > 1e-12 else 0.0)
    return S if S > 1e-300 else 1e-300


@njit(cache=True)
def _ar1_ll_profiled(sums, N):
    """AR(1) log-likelihood with sigma, rho AND one initial-gap offset per episode
    profiled out. l(rho) = (M/2) log(1-rho^2) - (N/2) log(S(rho)/N)."""
    M = float(sums.shape[0])
    best_z = 0.0
    best = -1e300
    for j in range(61):
        z = -4.95 + 9.9 * j / 60.0
        rho = math.tanh(z)
        ll = 0.5 * M * math.log(1.0 - rho * rho) - 0.5 * N * math.log(_S_profiled(sums, rho) / N)
        if ll > best:
            best = ll
            best_z = z
    a = best_z - 0.17
    b = best_z + 0.17
    g = 0.6180339887498949
    c = b - g * (b - a)
    d = a + g * (b - a)
    for _ in range(60):
        rc = math.tanh(c)
        rd = math.tanh(d)
        lc = 0.5 * M * math.log(1.0 - rc * rc) - 0.5 * N * math.log(_S_profiled(sums, rc) / N)
        ld = 0.5 * M * math.log(1.0 - rd * rd) - 0.5 * N * math.log(_S_profiled(sums, rd) / N)
        if lc > ld:
            b = d
        else:
            a = c
        c = b - g * (b - a)
        d = a + g * (b - a)
    rho = math.tanh(0.5 * (a + b))
    ll = 0.5 * M * math.log(1.0 - rho * rho) - 0.5 * N * math.log(_S_profiled(sums, rho) / N)
    if best > ll:
        return best, math.tanh(best_z)
    return ll, rho


@njit(cache=True)
def _ll_integrated_at(sums, rho, N):
    """Integrated (REML-type) log-likelihood at a given rho: each episode's initial-gap
    offset delta_e is integrated out under a flat prior instead of maximised over, and
    sigma is profiled. Up to a constant:
        l(rho) = (M/2) log(1-rho^2) - ((N-M)/2) log(S(rho)/(N-M)) - (1/2) sum_e log A_hh,e(rho)
    Maximising over M incidental offsets (the profile version) biases rho and sigma
    and inflates the LR statistic when episodes carry few effective observations
    (DIARY Entry 25); integrating them out is the standard remedy (REML)."""
    M = float(sums.shape[0])
    c = 1.0 - rho * rho
    S = 0.0
    L = 0.0
    for e in range(sums.shape[0]):
        arr = c * sums[e, 0] + sums[e, 1] - rho * sums[e, 2] + rho * rho * sums[e, 3]
        arh = c * sums[e, 4] + sums[e, 5] - rho * sums[e, 6] + rho * rho * sums[e, 7]
        ahh = c * sums[e, 8] + sums[e, 9] - rho * sums[e, 10] + rho * rho * sums[e, 11]
        if ahh < 1e-12:
            ahh = 1e-12
        S += arr - arh * arh / ahh
        L += math.log(ahh)
    if S < 1e-300:
        S = 1e-300
    return 0.5 * M * math.log(c) - 0.5 * (N - M) * math.log(S / (N - M)) - 0.5 * L


@njit(cache=True)
def _ar1_ll_integrated(sums, N):
    best_z = 0.0
    best = -1e300
    for j in range(61):
        z = -4.95 + 9.9 * j / 60.0
        ll = _ll_integrated_at(sums, math.tanh(z), N)
        if ll > best:
            best = ll
            best_z = z
    a = best_z - 0.17
    b = best_z + 0.17
    g = 0.6180339887498949
    c = b - g * (b - a)
    d = a + g * (b - a)
    for _ in range(60):
        if _ll_integrated_at(sums, math.tanh(c), N) > _ll_integrated_at(sums, math.tanh(d), N):
            b = d
        else:
            a = c
        c = b - g * (b - a)
        d = a + g * (b - a)
    rho = math.tanh(0.5 * (a + b))
    ll = _ll_integrated_at(sums, rho, N)
    if best > ll:
        return best, math.tanh(best_z)
    return ll, rho


@njit(cache=True)
def _ll_fixedjac_at(sums, jac, rho, N):
    """G.10: integrated (REML-type) log-likelihood whose Jacobian term uses the initial-gap
    sensitivity h at a FIXED reference theta_ref (jac = per-episode hh sums at theta_ref),
    while the residual projection uses h at the current theta. The Jacobian term then
    no longer depends on theta, so it cannot bias theta (run 2: -1/2 sum log A_hh(theta)
    pulled a, b upward by a constant amount, ~2 sd at N = 5,691, DIARY Entry 26), and rho
    keeps the REML correction that removed run 1's bias."""
    M = float(sums.shape[0])
    c = 1.0 - rho * rho
    L = 0.0
    for e in range(jac.shape[0]):
        ahh = c * jac[e, 0] + jac[e, 1] - rho * jac[e, 2] + rho * rho * jac[e, 3]
        if ahh < 1e-12:
            ahh = 1e-12
        L += math.log(ahh)
    return 0.5 * M * math.log(c) - 0.5 * (N - M) * math.log(_S_profiled(sums, rho) / (N - M)) - 0.5 * L


@njit(cache=True)
def _ar1_ll_fixedjac(sums, jac, N):
    best_z = 0.0
    best = -1e300
    for j in range(61):
        z = -4.95 + 9.9 * j / 60.0
        ll = _ll_fixedjac_at(sums, jac, math.tanh(z), N)
        if ll > best:
            best = ll
            best_z = z
    a = best_z - 0.17
    b = best_z + 0.17
    g = 0.6180339887498949
    c = b - g * (b - a)
    d = a + g * (b - a)
    for _ in range(60):
        if _ll_fixedjac_at(sums, jac, math.tanh(c), N) > _ll_fixedjac_at(sums, jac, math.tanh(d), N):
            b = d
        else:
            a = c
        c = b - g * (b - a)
        d = a + g * (b - a)
    rho = math.tanh(0.5 * (a + b))
    ll = _ll_fixedjac_at(sums, jac, rho, N)
    if best > ll:
        return best, math.tanh(best_z)
    return ll, rho


@njit(cache=True)
def _sim_idm_het(ul, vl, off, x0, v0, thetas, dt, s_floor, s_out):
    """_sim_idm with a separate parameter vector per episode (thetas: M x 5). Used to
    generate HETEROGENEOUS synthetic driver populations (Amendment I, Step 4b)."""
    hits = 0
    one = np.empty(off.shape[0] - 1 + 1, dtype=np.int64)
    for e in range(off.shape[0] - 1):
        i0, i1 = off[e], off[e + 1]
        o = np.empty(2, dtype=np.int64)
        o[0] = i0
        o[1] = i1
        xe = np.empty(1)
        ve = np.empty(1)
        xe[0] = x0[e]
        ve[0] = v0[e]
        hits += _sim_idm(ul, vl, o, xe, ve, thetas[e], dt, s_floor, s_out)
    return hits


@njit(cache=True)
def _ep_mse_mean(r, off):
    """Mean over episodes of each episode's mean squared residual (each driver one vote)."""
    tot = 0.0
    M = off.shape[0] - 1
    for e in range(M):
        i0, i1 = off[e], off[e + 1]
        acc = 0.0
        for t in range(i0, i1):
            acc += r[t] * r[t]
        tot += acc / (i1 - i0)
    return tot / M


# ------------------------------------------------------------------------------------
# B-lite (Amendment J): shared parameters + a per-driver subset (J.1 decides which)
# ------------------------------------------------------------------------------------
@njit(cache=True)
def _sim_ep_sens5(ul, vl, i0, i1, x0, v0, th, dt, s_floor, s_out, sens):
    """One episode: simulated gap and its exact tangent-linear sensitivities to all five
    parameters (v0, T, s0, a, b), sens[k, t] = d s(t) / d theta_k. Same update and clipping
    as _sim_idm; the initial state is observed, so all sensitivities start at zero."""
    V0, T, S0, A, B = th[0], th[1], th[2], th[3], th[4]
    inv_v0 = 1.0 / V0
    tsab = 2.0 * math.sqrt(A * B)
    x = x0
    v = v0
    X = np.zeros(5)
    W = np.zeros(5)
    Wn = np.zeros(5)
    dth = np.zeros(5)
    s = ul[i0] - x
    s_out[i0] = s
    for k in range(5):
        sens[k, i0] = 0.0
    hits = 0
    d_ss = np.zeros(5)
    for t in range(i0, i1 - 1):
        dv = v - vl[t]
        c = v * dv / tsab
        ss = S0 + v * T + c
        dss_dv = T + (2.0 * v - vl[t]) / tsab
        # direct derivatives of ss w.r.t. parameters (d_ss[0] is always 0)
        d_ss[1] = v
        d_ss[2] = 1.0
        d_ss[3] = -0.5 * c / A
        d_ss[4] = -0.5 * c / B
        if ss < S0:
            ss = S0
            dss_dv = 0.0
            d_ss[1] = 0.0
            d_ss[2] = 1.0
            d_ss[3] = 0.0
            d_ss[4] = 0.0
        if s > 1e-3:
            se = s
            dse = 1.0
        else:
            se = 1e-3
            dse = 0.0
        ratio = ss / se
        q = v * inv_v0
        core = 1.0 - q * q * q * q - ratio * ratio
        acc = A * core
        da_ds = 2.0 * A * ratio * ratio / se * dse
        da_dss = -2.0 * A * ratio / se
        da_dv = A * (-4.0 * q * q * q * inv_v0) + da_dss * dss_dv
        dth[0] = A * 4.0 * q * q * q * q * inv_v0
        dth[1] = da_dss * d_ss[1]
        dth[2] = da_dss * d_ss[2]
        dth[3] = core + da_dss * d_ss[3]
        dth[4] = da_dss * d_ss[4]
        clipped = False
        if acc < -9.0:
            acc = -9.0
            clipped = True
        elif acc > 4.0:
            acc = 4.0
            clipped = True
        vn = v + acc * dt
        for k in range(5):
            if clipped:
                dacc = 0.0
            else:
                dacc = da_ds * (-X[k]) + da_dv * W[k] + dth[k]
            Wn[k] = W[k] + dacc * dt
        if vn < 0.0:
            vn = 0.0
            for k in range(5):
                Wn[k] = 0.0
        x = x + 0.5 * (v + vn) * dt
        for k in range(5):
            X[k] = X[k] + 0.5 * (W[k] + Wn[k]) * dt
        sn = ul[t + 1] - x
        if sn < s_floor:
            x = ul[t + 1] - s_floor
            for k in range(5):
                X[k] = 0.0
            sn = s_floor
            hits += 1
        s_out[t + 1] = sn
        for k in range(5):
            sens[k, t + 1] = -X[k]
            W[k] = Wn[k]
        s = sn
        v = vn
    return hits


@njit(cache=True)
def _ep_lm(ul, vl, i0, i1, x0, v0, s_obs, th, per, lo, hi, max_iter, s_buf, sens):
    """Levenberg-Marquardt for ONE episode over the parameters listed in `per` (the rest
    of th fixed), within [lo, hi]. th is modified in place to the solution. Returns the
    episode's mean squared gap error."""
    k = per.shape[0]
    n = i1 - i0
    _sim_ep_sens5(ul, vl, i0, i1, x0, v0, th, DT, S_FLOOR, s_buf, sens)
    f = 0.0
    for t in range(i0, i1):
        r = s_obs[t] - s_buf[t]
        f += r * r
    f /= n
    lam = 1e-3
    trial = th.copy()
    for it in range(max_iter):
        g = np.zeros(k)
        H = np.zeros((k, k))
        for t in range(i0, i1):
            r = s_obs[t] - s_buf[t]
            for p in range(k):
                jp = -sens[per[p], t]
                g[p] += r * jp
                for q2 in range(k):
                    H[p, q2] += jp * (-sens[per[q2], t])
        improved = False
        rel = 1.0
        act = np.zeros(k, dtype=np.bool_)
        for p in range(k):
            j = per[p]
            if (th[j] <= lo[j] and g[p] > 0.0) or (th[j] >= hi[j] and g[p] < 0.0):
                act[p] = True
        for _ in range(10):
            A_ = H.copy()
            for p in range(k):
                A_[p, p] = H[p, p] * (1.0 + lam) + 1e-12
                if act[p]:
                    for q2 in range(k):
                        A_[p, q2] = 0.0
                        A_[q2, p] = 0.0
                    A_[p, p] = 1.0
            rhs = -g.copy()
            for p in range(k):
                if act[p]:
                    rhs[p] = 0.0
            d = np.linalg.solve(A_, rhs)
            for p in range(k):
                j = per[p]
                trial[j] = min(max(th[j] + d[p], lo[j]), hi[j])
            _sim_ep_sens5(ul, vl, i0, i1, x0, v0, trial, DT, S_FLOOR, s_buf, sens)
            fn = 0.0
            for t in range(i0, i1):
                r = s_obs[t] - s_buf[t]
                fn += r * r
            fn /= n
            if fn < f:
                rel = (f - fn) / (f + 1e-300)
                for p in range(k):
                    th[per[p]] = trial[per[p]]
                f = fn
                lam = max(lam / 3.0, 1e-9)
                improved = True
                break
            lam *= 5.0
        if not improved:
            _sim_ep_sens5(ul, vl, i0, i1, x0, v0, th, DT, S_FLOOR, s_buf, sens)
            break
        if rel < 1e-10:
            break
    return f


@njit(cache=True, parallel=True)
def _blite_all(ul, vl, off, x0, v0, s_obs, shared_th, per, starts, lo, hi, max_iter,
               s_buf, sens, P_out, f_out):
    """For every episode: from each start (rows of `starts`, values for the `per`
    parameters), run _ep_lm and keep the best. Writes the per-driver solution to P_out
    (M x len(per)) and the minimised per-episode MSE to f_out. Episodes are independent
    and each writes only its own slice of s_buf/sens, so the loop runs in parallel and
    the result does not depend on the thread count."""
    M = off.shape[0] - 1
    k = per.shape[0]
    for e in prange(M):
        th = shared_th.copy()
        best = 1e300
        for s_i in range(starts.shape[0]):
            for p in range(k):
                th[per[p]] = starts[s_i, p]
            fe = _ep_lm(ul, vl, off[e], off[e + 1], x0[e], v0[e], s_obs, th, per, lo, hi,
                        max_iter, s_buf, sens)
            if fe < best:
                best = fe
                for p in range(k):
                    P_out[e, p] = th[per[p]]
        f_out[e] = best
    return 0


START_VALUES = {1: (1.2, 0.7, 2.0), 2: (2.0, 1.0, 4.0), 3: (1.0, 0.5, 2.0), 4: (1.5, 0.8, 3.0)}


class BliteSample(object):
    """Episodes for B-lite. `per` = indices of the per-driver parameters (J.1 decides;
    e.g. (1,) = T only, (1, 3) = T and a). The initial state is the observed gap and
    speed: in real highD the gap is observed to centimetres and the discrepancy is
    dynamic (Amendment J, S2), so there is no initial-gap nuisance."""

    def __init__(self, ul, vl, off, v0, s_obs, per=(1,)):
        self.ul = np.ascontiguousarray(ul, dtype=np.float64)
        self.vl = np.ascontiguousarray(vl, dtype=np.float64)
        self.off = np.ascontiguousarray(off, dtype=np.int64)
        self.v0 = np.ascontiguousarray(v0, dtype=np.float64)
        self.s_obs = np.ascontiguousarray(s_obs, dtype=np.float64)
        for nm in ("ul", "vl", "v0", "s_obs"):
            if not np.isfinite(getattr(self, nm)).all():
                raise ValueError(f"BliteSample: non-finite values in {nm}")
        self.x0 = self.ul[self.off[:-1]] - self.s_obs[self.off[:-1]]
        self.M = self.off.shape[0] - 1
        self.N = int(self.off[-1])
        self.per = np.array(sorted(per), dtype=np.int64)
        if 0 in self.per:
            raise ValueError("BliteSample: v0 cannot be per-driver (not in J.1)")
        self.shared = np.array([j for j in range(K) if j not in per], dtype=np.int64)
        self.starts = np.array([[START_VALUES[j][s] for j in self.per] for s in range(3)])
        self.P = np.empty((self.M, len(self.per)))
        self.f_ep = np.empty(self.M)
        self._s = np.empty(self.N)
        self._sens = np.empty((5, self.N))
        self.n_eval = 0

    def subset(self, idx):
        idx = np.asarray(idx)
        lens = self.off[idx + 1] - self.off[idx]
        off = np.r_[0, np.cumsum(lens)].astype(np.int64)
        sel = np.concatenate([np.arange(self.off[i], self.off[i + 1]) for i in idx])
        return BliteSample(self.ul[sel], self.vl[sel], off, self.v0[idx], self.s_obs[sel],
                           per=tuple(self.per))

    def criterion(self, shared_vals, max_iter=40):
        """Mean over episodes of the per-episode MSE minimised over the per-driver
        parameters, from a FIXED start set every time (deterministic in the shared values).
        Leaves the per-driver solution in self.P."""
        th = np.array(GENERIC_START, dtype=np.float64)
        th[self.shared] = shared_vals
        self.n_eval += 1
        _blite_all(self.ul, self.vl, self.off, self.x0, self.v0, self.s_obs, th, self.per,
                   self.starts, LO, HI, max_iter, self._s, self._sens, self.P, self.f_ep)
        return float(self.f_ep.mean())


def fit_blite(sample, rng, n_random=2, starts=(), eps=1e-6):
    """B-lite fit over the shared parameters (bound-normalised, L-BFGS-B, fixed start set:
    generic + n_random uniform + optional extra starts given as full 5-vectors). Returns
    the shared estimate, the per-driver matrix, and the typical driver theta_typ (shared
    values, per-driver medians)."""
    sh = sample.shared
    lo, span = LO[sh], HI[sh] - LO[sh]
    to_sh = lambda u: lo + np.clip(u, 0, 1) * span
    us = [(GENERIC_START[sh] - lo) / span] + [rng.uniform(0.15, 0.85, len(sh)) for _ in range(n_random)]
    us += [(np.asarray(st, float)[sh] - lo) / span for st in starts]
    best = None
    for u0 in us:
        r = minimize(lambda u: sample.criterion(to_sh(u)), u0, method="L-BFGS-B",
                     bounds=[(0.0, 1.0)] * len(sh),
                     options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-9, "eps": eps})
        if best is None or r.fun < best[0]:
            best = (float(r.fun), r.x.copy(), str(r.message))
    L, u, msg = best
    L = sample.criterion(to_sh(u))                   # leaves P at the best solution
    P = sample.P.copy()
    typ = np.array(GENERIC_START, dtype=float)
    typ[sh] = to_sh(u)
    at_b, n_b = {}, {}
    for p, j in enumerate(sample.per):
        typ[j] = np.median(P[:, p])
        n_b[PARAM_NAMES[j]] = int(np.sum((P[:, p] <= LO[j] + 1e-9) | (P[:, p] >= HI[j] - 1e-9)))
        at_b[PARAM_NAMES[j]] = n_b[PARAM_NAMES[j]] / sample.M
    shared_at_bound = [PARAM_NAMES[j] for j, uu in zip(sh, u) if uu < 1e-6 or uu > 1 - 1e-6]
    return {"shared_idx": sh.tolist(), "shared": to_sh(u), "per_idx": sample.per.tolist(), "P": P,
            "L": float(L), "rms": float(np.sqrt(L)), "ep_rms": np.sqrt(sample.f_ep),
            "theta_typ": typ, "shared_at_bound": shared_at_bound, "per_at_bound": at_b,
            "per_n_at_bound": n_b, "opt_message": msg}


def blite_thetas(sample, fit_res):
    """M x 5 matrix of each driver's full parameter vector (shared values + own per-driver
    values) at a B-lite fit."""
    th = np.tile(np.asarray(fit_res["theta_typ"], float), (sample.M, 1))
    for p, j in enumerate(fit_res["per_idx"]):
        th[:, j] = fit_res["P"][:, p]
    return th


def blite_residuals(sample, fit_res):
    """Observed minus simulated gap for every sample of every episode at a B-lite fit."""
    s = np.empty(sample.N)
    _sim_idm_het(sample.ul, sample.vl, sample.off, sample.x0, sample.v0,
                 blite_thetas(sample, fit_res), DT, S_FLOOR, s)
    return sample.s_obs - s


def bootstrap_blite(sample, fit_res, B, rng, n_random=2):
    """Driver bootstrap of the whole B-lite procedure: resample episodes, refit with the SAME
    start set as the full-data fit (generic + n_random random) plus a warm start at the
    full-data estimate (S5 review finding 5). Returns (B x 5 theta_typ_b, B x n_shared
    at-bound flags of the shared parameters)."""
    th_b, atb = [], []
    for _ in range(B):
        idx = rng.integers(0, sample.M, sample.M)
        sb = sample.subset(idx)
        fb = fit_blite(sb, rng, n_random=n_random, starts=(fit_res["theta_typ"],))
        th_b.append(fb["theta_typ"])
        atb.append([PARAM_NAMES[j] in fb["shared_at_bound"] for j in fb["shared_idx"]])
    return np.array(th_b), np.array(atb, dtype=bool)


def ellipsoid_region(theta_hat, th_b, level=0.95):
    """Bootstrap confidence ellipsoid {theta : (theta-theta_hat)' S^+ (theta-theta_hat) <= q},
    S = bootstrap covariance (pseudo-inverse).

    q (J.8, primary) = the ceil(level (B+1))-th smallest LEAVE-ONE-OUT distance: each bootstrap
    point's distance uses the covariance of the other B-1 points, so it is an out-of-sample
    distance like the one of the target. The in-sample empirical quantile q_emp (the J rule)
    undercovers at small B (0.914 at B = 99 for an ideal Gaussian bootstrap in 5 dimensions;
    0.954 with the leave-one-out rule, 0.960 under t4 tails). q_emp is kept for reporting."""
    th_b = np.asarray(th_b, float)
    B = th_b.shape[0]
    sd = th_b.std(axis=0, ddof=1)
    flat = sd <= 1e-9 * SPAN                  # a direction the bootstrap never varied
    S = np.cov(th_b.T)
    Si = np.linalg.pinv(S)
    dd = th_b - theta_hat
    d = np.einsum("ij,jk,ik->i", dd, Si, dd)
    d_loo = np.empty(B)
    for b in range(B):
        Sb = np.linalg.pinv(np.cov(np.delete(th_b, b, axis=0).T))
        d_loo[b] = dd[b] @ Sb @ dd[b]
    k = min(int(math.ceil(level * (B + 1))), B) - 1
    return {"center": np.asarray(theta_hat, float), "cov": S, "q": float(np.sort(d_loo)[k]),
            "q_emp": float(np.quantile(d, level)),
            "flat": flat, "flat_names": [PARAM_NAMES[j] for j in range(K) if flat[j]]}


def in_region(region, theta):
    """Inside the ellipsoid AND equal to the centre along any direction the bootstrap never
    varied (a pseudo-inverse would otherwise accept any value there; S5 review finding 3).
    A non-empty region['flat_names'] must be reported: the ellipsoid is then degenerate."""
    theta = np.asarray(theta, float)
    dlt = theta - region["center"]
    if np.any(np.abs(dlt[region["flat"]]) > 1e-9 * SPAN[region["flat"]]):
        return False
    return float(dlt @ np.linalg.pinv(region["cov"]) @ dlt) <= region["q"]


# ------------------------------------------------------------------------------------
# S1 metrics and the S2 generator (Amendments J.3, J.4)
# ------------------------------------------------------------------------------------
S1_LAGS = (1, 25, 50, 125)
THW_MAX = 5.0            # F4: highD thw = dhw / |v|, dhw = bumper-to-bumper gap (DIARY Entry 10)


def s1_metrics(sample, fit_res):
    """The S1 structure metrics of a B-lite fit. The pooled ACF, pooled slope, RMS,
    disequilibrium and per-driver quantities are the SAME computation as 10_probe.py (J.3);
    the episode-weighted ACF and slope are the robust versions added in J.4. Unrounded."""
    th = blite_thetas(sample, fit_res)
    r = blite_residuals(sample, fit_res)
    ep_rms = np.asarray(fit_res["ep_rms"])
    num = np.zeros(len(S1_LAGS))
    den = 0.0
    tt, ar = [], []
    acf_ep = np.full((sample.M, len(S1_LAGS)), np.nan)
    slope_ep = np.empty(sample.M)
    for e in range(sample.M):
        x = r[sample.off[e]:sample.off[e + 1]]
        xx = float(np.dot(x, x))
        den += xx
        for j, L in enumerate(S1_LAGS):
            if len(x) > L:
                c = float(np.dot(x[L:], x[:-L]))
                num[j] += c
                acf_ep[e, j] = c / xx if xx > 0 else np.nan
        te = np.arange(len(x)) * DT
        tt.append(te)
        ar.append(np.abs(x))
        tc = te - te.mean()
        slope_ep[e] = float(np.dot(tc, np.abs(x)) / np.dot(tc, tc))
    tt, ar = np.concatenate(tt), np.concatenate(ar)
    slope = float(np.polyfit(tt, ar, 1)[0])
    v0 = sample.v0
    q = np.minimum(v0 / th[:, 0], 0.95)
    s_eq = (th[:, 2] + v0 * th[:, 1]) / np.sqrt(1.0 - q ** 4)
    diseq = (sample.s_obs[sample.off[:-1]] - s_eq) / s_eq
    out = {"ep_rms_median": float(np.median(ep_rms)), "ep_rms_p90": float(np.quantile(ep_rms, 0.9)),
           "acf": (num / den).tolist(), "abs_resid_slope_m_per_s": slope,
           "acf_ep_mean": np.nanmean(acf_ep, axis=0).tolist(),
           "slope_ep_median": float(np.median(slope_ep)),
           "start_diseq_q10_50_90": np.quantile(diseq, [0.1, 0.5, 0.9]).tolist()}
    for p, j in enumerate(fit_res["per_idx"]):
        n = PARAM_NAMES[j]
        P = fit_res["P"][:, p]
        out[n + "_q10_25_50_75_90"] = np.quantile(P, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
        out[n + "_iqr"] = float(np.subtract(*np.quantile(P, [0.75, 0.25])))
        out[n + "_at_bound"] = float(fit_res["per_at_bound"][n])
    return out


def inner_fit(sample, shared_vals):
    """The per-driver fit at KNOWN shared values, in fit_blite's format (diagnostics only)."""
    shared_vals = np.asarray(shared_vals, float)
    L = sample.criterion(shared_vals)
    P = sample.P.copy()
    typ = np.array(GENERIC_START, dtype=float)
    typ[sample.shared] = shared_vals
    at_b, n_b = {}, {}
    for p, j in enumerate(sample.per):
        typ[j] = np.median(P[:, p])
        n_b[PARAM_NAMES[j]] = int(np.sum((P[:, p] <= LO[j] + 1e-9) | (P[:, p] >= HI[j] - 1e-9)))
        at_b[PARAM_NAMES[j]] = n_b[PARAM_NAMES[j]] / sample.M
    return {"shared_idx": sample.shared.tolist(), "shared": shared_vals,
            "per_idx": sample.per.tolist(), "P": P, "L": float(L), "rms": float(np.sqrt(L)),
            "ep_rms": np.sqrt(sample.f_ep), "theta_typ": typ, "shared_at_bound": [],
            "per_at_bound": at_b, "per_n_at_bound": n_b, "opt_message": "inner only"}


@njit(cache=True)
def _sim_idm_procnoise(ul, vl, off, x0, v0, thetas, sig_e, phi, eps, dt, s_floor,
                       dhw_add, thw_max, s_out, ok_out, sig_dT, eps2, t_lo, t_hi):
    """_sim_idm with a per-episode parameter vector and an AR(1) acceleration perturbation
    (J.3, S2): acc = clip(a_IDM + eta, -9, 4); eta_{t+1} = phi eta_t + sig_e sqrt(1-phi^2)
    eps_{t+1}; eta_0 = sig_e eps_{i0} (stationary). eps: pre-drawn N(0,1), one per sample.
    ok_out[e] = the episode satisfies the following regime F4 at every sample (J.4):
    no gap-floor hit and (gap + dhw_add[e]) <= thw_max * v. dhw_add[e] <= 0 is a per-episode
    slack (J.5) that loosens the check only where the real episode itself exceeds 5 s through
    highD's thw rounding, so that the REAL trajectory always passes.
    sig_dT > 0 (J.7 arm G3): the driver's T drifts within the episode, T(t) = clip(T exp(l_t),
    t_lo, t_hi) with l_0 = 0 and l_{t+1} = l_t + sig_dT sqrt(dt) eps2_{t+1}. With sig_dT = 0
    the simulation is bit-identical to the J.3 version."""
    hits = 0
    c = math.sqrt(1.0 - phi * phi)
    for e in range(off.shape[0] - 1):
        i0, i1 = off[e], off[e + 1]
        V0, T, S0, A, B = thetas[e, 0], thetas[e, 1], thetas[e, 2], thetas[e, 3], thetas[e, 4]
        inv_v0 = 1.0 / V0
        tsab = 2.0 * math.sqrt(A * B)
        x = x0[e]
        v = v0[e]
        s = ul[i0] - x
        s_out[i0] = s
        ok = (s + dhw_add[e]) <= thw_max * v
        eta = sig_e[e] * eps[i0]
        lt = 0.0
        sq = math.sqrt(dt)
        for t in range(i0, i1 - 1):
            dv = v - vl[t]
            if sig_dT > 0.0:
                Tt = T * math.exp(lt)
                if Tt < t_lo:
                    Tt = t_lo
                elif Tt > t_hi:
                    Tt = t_hi
                lt = lt + sig_dT * sq * eps2[t + 1]
            else:
                Tt = T
            ss = S0 + v * Tt + (v * dv) / tsab
            if ss < S0:
                ss = S0
            se = s if s > 1e-3 else 1e-3
            ratio = ss / se
            q = v * inv_v0
            acc = A * (1.0 - q * q * q * q - ratio * ratio) + eta
            if acc < -9.0:
                acc = -9.0
            elif acc > 4.0:
                acc = 4.0
            vn = v + acc * dt
            if vn < 0.0:
                vn = 0.0
            x = x + 0.5 * (v + vn) * dt
            sn = ul[t + 1] - x
            if sn < s_floor:
                x = ul[t + 1] - s_floor
                sn = s_floor
                hits += 1
                ok = False
            if (sn + dhw_add[e]) > thw_max * vn:
                ok = False
            s_out[t + 1] = sn
            s = sn
            v = vn
            eta = phi * eta + sig_e[e] * c * eps[t + 1]
        ok_out[e] = ok
    return hits


def t_model_fit(T_hat, gap0, vF0):
    """OLS of log T_hat on log h0, h0 = gap0 / max(vF0, 1) (J.3 step 2)."""
    lh = np.log(np.asarray(gap0, float) / np.maximum(np.asarray(vF0, float), 1.0))
    y = np.log(np.asarray(T_hat, float))
    m_h = float(lh.mean())
    X = np.c_[np.ones_like(lh), lh - m_h]
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ coef
    return {"mu": float(coef[0]), "beta": float(coef[1]), "m_h": m_h,
            "s_res": float(np.sqrt(res @ res / (len(y) - 2))), "n": int(len(y))}


def dhw_add_from_real(gap, vF, off):
    """Per-episode F4 slack (J.5): min(0, min_t(5 vF - gap)) - 1e-6. The synthetic check is
    then the plain F4 rule gap <= 5 v, loosened only where the real episode itself exceeds it
    (thw rounding), so that each real trajectory passes the synthetic check."""
    out = np.empty(off.shape[0] - 1)
    for e in range(out.shape[0]):
        i0, i1 = off[e], off[e + 1]
        out[e] = min(0.0, float(np.min(THW_MAX * vF[i0:i1] - gap[i0:i1]))) - 1e-6
    return out


def gen_synthetic(ul, vl, off, gap0, vF0, shared_theta, tmod, sig_eta, tau, omega, kappa,
                  seed, dhw_add, sigma_obs=0.05, max_attempts=200, condition=True,
                  t_df=None, sig_dT=0.0):
    """One synthetic data set from generator G (J.3, S2; J.4) on the given real traces and
    real initial states. shared_theta: 5-vector whose T entry is ignored.

    Episode e, attempt k draws (z_T, z_sigma, eps[episode]) from default_rng([seed, e, k]),
    so the same seed gives the same base draws at every generator setting (common random
    numbers). With condition=True an episode is redrawn until it satisfies the following
    regime F4 at every sample - the synthetic analogue of selecting real episodes by the
    episode definition (J.4); after max_attempts the last draw is kept and counted.
    Observation noise comes from default_rng([seed, M, 0]).
    J.7 arms: t_df (G2) draws z_T from Student-t(t_df) scaled to the normal IQR instead of
    N(0,1); sig_dT > 0 (G3) adds the within-episode T drift, its increments drawn after eps
    from the same episode stream. With the defaults the draws are those of J.3/J.4.
    Returns (observed gap, true thetas, noise scales, info dict)."""
    ul = np.ascontiguousarray(ul, dtype=np.float64)
    vl = np.ascontiguousarray(vl, dtype=np.float64)
    off = np.ascontiguousarray(off, dtype=np.int64)
    gap0 = np.asarray(gap0, float)
    vF0 = np.ascontiguousarray(vF0, dtype=np.float64)
    M, N = off.shape[0] - 1, int(off[-1])
    lh = np.log(gap0 / np.maximum(vF0, 1.0))
    thetas = np.tile(np.asarray(shared_theta, float), (M, 1))
    sig_e = np.empty(M)
    eps = np.empty(N)
    phi = math.exp(-DT / tau)
    x0 = ul[off[:-1]] - gap0
    dhw_add = np.ascontiguousarray(np.broadcast_to(np.asarray(dhw_add, float), (M,)))
    s = np.empty(N)
    ok = np.zeros(M, dtype=np.bool_)
    attempts = np.zeros(M, dtype=np.int64)
    eps2 = np.zeros(N if sig_dT > 0.0 else 1)
    if t_df is not None:
        from scipy.stats import norm, t as student_t
        t_scale = float(norm.ppf(0.75) / student_t.ppf(0.75, t_df))
    todo = np.arange(M)
    for k in range(max_attempts):
        for e in todo:
            g = np.random.default_rng([seed, int(e), k])
            if t_df is None:
                zT, zs = g.standard_normal(2)
            else:
                zT = float(g.standard_t(t_df)) * t_scale
                zs = float(g.standard_normal())
            eps[off[e]:off[e + 1]] = g.standard_normal(off[e + 1] - off[e])
            if sig_dT > 0.0:
                eps2[off[e]:off[e + 1]] = g.standard_normal(off[e + 1] - off[e])
            thetas[e, 1] = min(max(math.exp(tmod["mu"] + tmod["beta"] * (lh[e] - tmod["m_h"])
                                             + kappa * tmod["s_res"] * zT), LO[1]), HI[1])
            sig_e[e] = sig_eta * math.exp(omega * zs - 0.5 * omega * omega)
        _sim_idm_procnoise(ul, vl, off, x0, vF0, thetas, sig_e, phi, eps, DT, S_FLOOR,
                           dhw_add, THW_MAX, s, ok, float(sig_dT), eps2, LO[1], HI[1])
        attempts[todo] = k + 1
        todo = np.flatnonzero(~ok) if condition else np.empty(0, dtype=np.int64)
        if todo.size == 0:
            break
    s_obs = s + sigma_obs * np.random.default_rng([seed, M, 0]).standard_normal(N)
    info = {"attempts_mean": float(attempts.mean()), "attempts_max": int(attempts.max()),
            "accept_first_try": float(np.mean(attempts == 1)),
            "f4_violating_final": int((~ok).sum())}
    return s_obs, thetas, sig_e, info


# ------------------------------------------------------------------------------------
# data container
# ------------------------------------------------------------------------------------
class Sample:
    """A pooled sample of episodes: observed leader rear position and speed, follower
    initial speed, observed gap, and the offsets. x_init is taken from the OBSERVED gap
    at t=0, exactly as on real data."""

    def __init__(self, ul, vl, off, v0, s_obs):
        self.ul = np.ascontiguousarray(ul, dtype=np.float64)
        self.vl = np.ascontiguousarray(vl, dtype=np.float64)
        self.off = np.ascontiguousarray(off, dtype=np.int64)
        self.v0 = np.ascontiguousarray(v0, dtype=np.float64)
        self.s_obs = np.ascontiguousarray(s_obs, dtype=np.float64)
        self.x0 = self.ul[self.off[:-1]] - self.s_obs[self.off[:-1]]
        self.M = self.off.shape[0] - 1
        self.N = int(self.off[-1])
        self._s = np.empty(self.N)
        self._h = np.empty(self.N)
        self.n_eval = 0
        self.profile_init = True      # Amendment G.8; False = legacy (initial gap as observed)
        self.init_mode = "fixedjac"   # G.10 default; "integrate" = G.9; "profile" = G.8
        self.jac = None               # per-episode hh sums at theta_ref (G.10)
        self.theta_ref = None

    def subset(self, idx):
        idx = np.asarray(idx)
        lens = self.off[idx + 1] - self.off[idx]
        off = np.r_[0, np.cumsum(lens)].astype(np.int64)
        sel = np.concatenate([np.arange(self.off[i], self.off[i + 1]) for i in idx])
        out = Sample(self.ul[sel], self.vl[sel], off, self.v0[idx], self.s_obs[sel])
        out.profile_init = self.profile_init
        out.init_mode = self.init_mode
        if self.jac is not None:
            out.jac = np.ascontiguousarray(self.jac[idx])
            out.theta_ref = self.theta_ref
        return out


def simulate(sample, theta, x0=None, v0=None):
    s = np.empty(sample.N)
    hits = _sim_idm(sample.ul, sample.vl, sample.off,
                    sample.x0 if x0 is None else x0, sample.v0 if v0 is None else v0,
                    np.asarray(theta, dtype=np.float64), DT, S_FLOOR, s)
    return s, hits


def loglik(sample, theta):
    """Pooled AR(1) log-likelihood of the observed gap; returns (ll, rho, floor hits).

    Default (Amendment G.8): the simulation starts at the observed gap, which contains
    observation noise, and each episode's initial-gap error is treated as a nuisance
    parameter delta_e and profiled out, using the tangent-linear sensitivity h(t) of the
    simulated gap to the initial gap: residual_e(t) = s_obs - s_sim - delta_e h_e(t).
    Conditioning on the noisy initial gap instead (profile_init=False) biases the LR
    statistic upward: the positive control undercovered at 0.755 [0.689, 0.813] that way,
    and covered at 0.94 when the true initial gap was supplied (DIARY Entry 24)."""
    th = np.asarray(theta, dtype=np.float64)
    if sample.init_mode == "mse":
        # Amendment I: M-estimator for theta_KL. Criterion L = mean over episodes of the
        # per-episode mean squared gap error, from the observed initial state. Returned
        # as the pseudo log-likelihood -(M/2) log L, so that 2 * difference = M log(L/L_hat),
        # the criterion-ratio statistic whose threshold is calibrated by the driver bootstrap.
        hits = _sim_idm(sample.ul, sample.vl, sample.off, sample.x0, sample.v0, th, DT,
                        S_FLOOR, sample._s)
        L = _ep_mse_mean(sample.s_obs - sample._s, sample.off)
        sample.n_eval += 1
        return -0.5 * sample.M * math.log(max(L, 1e-300)), 0.0, hits
    if not sample.profile_init:
        hits = _sim_idm(sample.ul, sample.vl, sample.off, sample.x0, sample.v0, th, DT,
                        S_FLOOR, sample._s)
        F, Q, P, R = _ar1_sums(sample.s_obs - sample._s, sample.off)
        ll, rho = _ar1_ll(F, Q, P, R, float(sample.M), float(sample.N))
    else:
        hits = _sim_idm_sens(sample.ul, sample.vl, sample.off, sample.x0, sample.v0, th, DT,
                             S_FLOOR, sample._s, sample._h)
        sums = _ep_sums(sample.s_obs - sample._s, sample._h, sample.off)
        if sample.init_mode == "integrate":
            ll, rho = _ar1_ll_integrated(sums, float(sample.N))
        elif sample.init_mode == "fixedjac":
            if sample.jac is None:
                raise RuntimeError("G.10 likelihood needs a Jacobian reference: call fit() "
                                   "or set_reference(sample, theta_ref) first")
            ll, rho = _ar1_ll_fixedjac(sums, sample.jac, float(sample.N))
        else:
            ll, rho = _ar1_ll_profiled(sums, float(sample.N))
    sample.n_eval += 1
    return ll, rho, hits


def set_reference(sample, theta_ref):
    """Fix the Jacobian reference of the G.10 likelihood at theta_ref."""
    _sim_idm_sens(sample.ul, sample.vl, sample.off, sample.x0, sample.v0,
                  np.asarray(theta_ref, dtype=np.float64), DT, S_FLOOR, sample._s, sample._h)
    sums = _ep_sums(sample.s_obs - sample._s, sample._h, sample.off)
    sample.jac = np.ascontiguousarray(sums[:, 8:12])
    sample.theta_ref = np.asarray(theta_ref, dtype=float)


# ------------------------------------------------------------------------------------
# fitting
# ------------------------------------------------------------------------------------
GENERIC_START = np.array([33.0, 1.5, 2.0, 1.0, 1.5])     # fixed; same on real data


def _nll_u(u, sample, fixed_i=None, fixed_u=None):
    if fixed_i is not None:
        u = np.insert(u, fixed_i, fixed_u)
    return -loglik(sample, to_theta(u))[0]


def fit(sample, rng, n_random=2, extra_starts=(), maxiter=500):
    """MLE by L-BFGS-B in normalised coordinates from a FIXED start set: the generic
    start, n_random uniform starts in [0.15, 0.85]^5, plus any extra starts (theta).
    Using no information about the true value keeps the synthetic test honest.

    G.10: with init_mode "fixedjac" and no reference yet, stage 1 fits the G.9 integrated
    likelihood, fixes the Jacobian reference at that estimate, and stage 2 refits from it."""
    if sample.init_mode == "fixedjac" and sample.jac is None:
        sample.init_mode = "integrate"
        f1 = fit(sample, rng, n_random, extra_starts, maxiter)
        set_reference(sample, f1["theta"])
        sample.init_mode = "fixedjac"
        extra_starts, n_random = tuple(extra_starts) + (f1["theta"],), 0
    starts = [to_u(GENERIC_START)] + [rng.uniform(0.15, 0.85, K) for _ in range(n_random)]
    starts += [to_u(s) for s in extra_starts]
    best = None
    for u0 in starts:
        r = minimize(_nll_u, u0, args=(sample,), method="L-BFGS-B",
                     bounds=[(0.0, 1.0)] * K,
                     options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    u = np.clip(best.x, 0.0, 1.0)
    ll, rho, hits = loglik(sample, to_theta(u))
    at_bound = [PARAM_NAMES[i] for i in range(K) if u[i] < 1e-6 or u[i] > 1 - 1e-6]
    return {"u": u, "theta": to_theta(u), "ll": ll, "rho": rho, "floor_hits": int(hits),
            "at_bound": at_bound}


def _inner(sample, i, ui, u_start):
    free0 = np.delete(u_start, i)
    r = minimize(_nll_u, free0, args=(sample, i, ui), method="L-BFGS-B",
                 bounds=[(0.0, 1.0)] * (K - 1),
                 options={"maxiter": 400, "ftol": 1e-13, "gtol": 1e-9})
    return -r.fun, np.insert(np.clip(r.x, 0, 1), i, ui)


def profile(sample, fit_res, i, steps=(0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0),
            tol=0.05, max_refine=5, thresholds=None):
    """Profile of parameter i outward from the MLE on both sides.

    Steps are in normalised units, warm-started, and stop once LR > chi2(0.95,5) or at the
    bound. Each threshold crossing (3.84 = per-parameter interval, 11.07 = projection of
    the 5-D region) is then located by secant iteration on the signed root sqrt(LR),
    which is near-linear in the parameter, until |LR - threshold| < tol * threshold.
    'open' (None) means the bound was reached with LR still below that threshold.

    thresholds: optional {name: LR threshold}; default {"p1": 3.84, "pk": 11.07}. Step 5
    adds e.g. "pb" (episode-bootstrap threshold). Stepping stops once LR exceeds the
    largest threshold. With the default the behaviour is identical to Step 4.
    """
    if thresholds is None:
        thresholds = {"p1": CHI2_1, "pk": CHI2_K}
    t_max = max(thresholds.values())
    ll_hat = fit_res["ll"]
    u_hat = fit_res["u"]
    out = {"param": PARAM_NAMES[i], "better_max_found": 0.0, "inner_fits": 0}

    def lr_at(ui, u_start):
        llp, u_new = _inner(sample, i, ui, u_start)
        out["inner_fits"] += 1
        lr = 2.0 * (ll_hat - llp)
        if lr < -1e-6:
            if -lr > out["better_max_found"]:
                out["better_max_found"] = -lr
                out["better_u"] = [float(x) for x in u_new]
        return max(lr, 0.0), u_new

    for side, sgn in (("lo", -1.0), ("hi", 1.0)):
        pts = [(u_hat[i], 0.0, u_hat.copy())]
        u_prev = u_hat.copy()
        reached = False
        for d in steps:
            ui = min(max(u_hat[i] + sgn * d, 0.0), 1.0)
            if pts and abs(ui - pts[-1][0]) < 1e-12:
                break
            lr, u_prev = lr_at(ui, u_prev)
            pts.append((ui, lr, u_prev.copy()))
            if lr > t_max:
                reached = True
                break
            if ui in (0.0, 1.0):
                break
        for name, thr in thresholds.items():
            br_ = None
            for pa, pb in zip(pts[:-1], pts[1:]):
                if pa[1] <= thr < pb[1]:
                    br_ = [pa, pb]
                    break
            if br_ is None:
                out[f"{name}_{side}"] = None                  # open at this threshold
                continue
            zt = math.sqrt(thr)
            best = br_[1]
            for _ in range(max_refine):
                (ua, la, uva), (ub, lb, uvb) = br_
                za, zb = math.sqrt(la), math.sqrt(lb)
                uc = ua + (zt - za) * (ub - ua) / (zb - za) if zb > za else 0.5 * (ua + ub)
                lc, uvc = lr_at(uc, uva)
                best = (uc, lc, uvc)
                if abs(lc - thr) < tol * thr:
                    break
                br_ = [br_[0], best] if lc > thr else [best, br_[1]]
            out[f"{name}_{side}"] = float(LO[i] + best[0] * SPAN[i])
            out[f"{name}_{side}_lr"] = float(best[1])
        out[f"reached_k_{side}"] = reached
    for name in thresholds:
        closed = out.get(f"{name}_lo") is not None and out.get(f"{name}_hi") is not None
        out[f"{name}_closed"] = closed
        out[f"{name}_width"] = (out[f"{name}_hi"] - out[f"{name}_lo"]) if closed else None
    return out


def fit_and_profile(sample, rng, n_random=5, max_rounds=3, tol=0.01):
    """D4.8 remedy (Amendment H): fit, profile all parameters, and if any profile point
    beats the fitted maximum by more than tol log-likelihood units, refit from that
    point and profile again (at most max_rounds). Returns (fit, profiles, rounds, resolved)."""
    f = fit(sample, rng, n_random=n_random)
    for rnd in range(max_rounds):
        profs = [profile(sample, f, i) for i in range(K)]
        worst = max(range(K), key=lambda i: profs[i]["better_max_found"])
        if profs[worst]["better_max_found"] <= 2.0 * tol:      # better_max is in LR units
            return f, profs, rnd, True
        th_b = to_theta(np.asarray(profs[worst]["better_u"]))
        f_new = fit(sample, rng, n_random=0, extra_starts=(th_b,))
        if f_new["ll"] > f["ll"]:
            f = f_new
    return f, profs, max_rounds, False


def bootstrap_threshold(sample, fit_res, B, rng, level=0.95):
    """Episode (block) bootstrap calibration of the LR threshold: resample whole episodes,
    refit, LR_b = 2[l_b(theta_b) - l_b(theta_hat)]. Returns the level-quantile."""
    lrs = []
    th_hat = fit_res["theta"]
    for _ in range(B):
        idx = rng.integers(0, sample.M, sample.M)
        sb = sample.subset(idx)
        fb = fit(sb, rng, n_random=1, extra_starts=(th_hat,))
        lrs.append(2.0 * (fb["ll"] - loglik(sb, th_hat)[0]))
    lrs = np.array(lrs)
    return float(np.quantile(lrs, level)), lrs


def residual_diagnostics(sample, theta, rho, max_lag=10):
    """Post-fit residual diagnostics at (theta, rho) (Step 5, Gate L and H.4).

    For each episode: the GLS initial-gap offset delta_e = A_rh/A_hh at rho, the residual
    e(t) = s_obs - s_sim - delta_e h(t), its OLS slope against time (m/s), and the AR(1)
    innovations e_t - rho e_{t-1}. Returns a dict with per-episode arrays and pooled
    summaries: innovation ACF (lags 1..max_lag, within episodes), innovation sd and the
    implied marginal sd sigma = sd_innov / sqrt(1 - rho^2)."""
    th = np.asarray(theta, dtype=np.float64)
    s = np.empty(sample.N)
    h = np.empty(sample.N)
    _sim_idm_sens(sample.ul, sample.vl, sample.off, sample.x0, sample.v0, th, DT, S_FLOOR, s, h)
    r = sample.s_obs - s
    sums = _ep_sums(r, h, sample.off)
    c = 1.0 - rho * rho
    arh = c * sums[:, 4] + sums[:, 5] - rho * sums[:, 6] + rho * rho * sums[:, 7]
    ahh = c * sums[:, 8] + sums[:, 9] - rho * sums[:, 10] + rho * rho * sums[:, 11]
    delta = arh / np.maximum(ahh, 1e-12)
    slopes = np.empty(sample.M)
    innov_all = []
    acf_num = np.zeros(max_lag + 1)
    acf_den = 0.0
    for e in range(sample.M):
        i0, i1 = sample.off[e], sample.off[e + 1]
        ee = r[i0:i1] - delta[e] * h[i0:i1]
        t = np.arange(i1 - i0) * DT
        tc = t - t.mean()
        slopes[e] = float(np.dot(tc, ee - ee.mean()) / np.dot(tc, tc))
        inn = ee[1:] - rho * ee[:-1]
        innov_all.append(inn)
        acf_den += float(np.dot(inn, inn))
        for L in range(1, max_lag + 1):
            if len(inn) > L:
                acf_num[L] += float(np.dot(inn[L:], inn[:-L]))
    inn = np.concatenate(innov_all)
    sd_innov = float(np.sqrt(np.mean(inn ** 2)))
    return {"delta": delta, "slopes": slopes,
            "acf": [float(acf_num[L] / acf_den) for L in range(1, max_lag + 1)],
            "sd_innov": sd_innov, "sigma_marginal": sd_innov / math.sqrt(max(1.0 - rho * rho, 1e-12)),
            "resid_rms": float(np.sqrt(np.mean((r - np.repeat(delta, np.diff(sample.off)) *
                                                h) ** 2)))}


def equilibrium_gap(v, theta, cap=0.95):
    v0, T, s0 = theta[0], theta[1], theta[2]
    q = np.minimum(np.asarray(v) / v0, cap)
    return (s0 + np.asarray(v) * T) / np.sqrt(1.0 - q ** 4)
