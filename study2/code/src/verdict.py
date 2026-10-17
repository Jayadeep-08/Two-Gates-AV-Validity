"""Step 7 simulator and verdict metrics (Amendment K).

A longitudinal kinematic string: a scripted lead, background drivers running IDM with the
calibrated parameters plus the calibrated AR(1) acceleration process noise, and one ego vehicle
running a constant-time-gap ACC that is NOT an IDM vehicle and carries no process noise.

    lead (scripted) -> B1 -> B2 -> EGO -> B3 -> B4 -> B5

Design points that matter for the study:
  * `simulate` returns trajectories only. The verdict metrics (`metrics_from_traces`) take
    positions, speeds and lengths and have no access to theta - the runner asserts this.
  * the noise draws are passed in, so the same seeds can be reused across theta (common random
    numbers), which is what makes the epistemic search a search over theta and not over noise.
  * accelerations, the speed floor and the gap are handled exactly as in the calibration
    simulator (`bridge._sim_idm`): clip to [-9, 4], v >= 0, trapezoidal position update.
"""

import math

import numpy as np

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

DT = 0.04
LEN_CAR = 4.5
A_MIN, A_MAX = -9.0, 4.0                 # background acceleration clip, as in the calibration
EGO_A_MIN, EGO_A_MAX = -8.0, 2.0
EGO_S0, EGO_LAG_S, EGO_VMAX = 2.0, 0.2, 36.0
EGO_K_GAP, EGO_K_RATE = 0.4, 0.8
RSS_RHO, RSS_AMAX, RSS_BMIN, RSS_BMAX = 0.2, 2.0, 3.0, 8.0
TTC_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)     # declared threshold grid for the conflict-rate curve
# Seed geometry (K.10). A stream's seeds are base + m*K_STRIDE + k, so one stream occupies a
# contiguous span below M_MAX*K_STRIDE. Callers space their bases K_BLOCK apart, and K_BLOCK
# exceeds that span, so no two streams can overlap. n_seeds must stay below K_STRIDE and the
# instance count below M_MAX; both are asserted by the caller.
K_STRIDE = 4096                          # seed indices per instance (n_seeds must be below it)
M_MAX = 256                              # instances per stream (the suite has 120)
K_BLOCK = M_MAX * K_STRIDE               # 1,048,576: one stream's whole seed block.
# The largest offset a stream can produce is (M_MAX-1)*K_STRIDE + (K_STRIDE-1) = K_BLOCK - 1,
# so bases at distinct multiples of K_BLOCK give disjoint blocks. With 65 streams the highest
# seed is about 6.8e7, far inside the 2^31 range numpy's legacy seeding accepts.
N_VEH = 7                                # lead + 5 background + ego
EGO_IDX = 3                              # lead=0, B1=1, B2=2, EGO=3, B3=4, B4=5, B5=6


@njit(cache=True)
def _idm_acc(v, dv, s, v0, T, s0, a, b):
    """IDM acceleration; s is the bumper-to-bumper gap, dv = v - v_lead."""
    ss = s0 + v * T + (v * dv) / (2.0 * math.sqrt(a * b))
    if ss < s0:
        ss = s0
    se = s if s > 1e-3 else 1e-3
    q = v / v0
    return a * (1.0 - q * q * q * q - (ss / se) * (ss / se))


@njit(cache=True)
def _ego_acc(v, v_lead, s, tau):
    """Constant-time-gap ACC: close the gap error and the rate error, cruise at EGO_VMAX."""
    s_des = EGO_S0 + tau * v
    acc = EGO_K_GAP * (s - s_des) + EGO_K_RATE * (v_lead - v)
    free = 0.6 * (EGO_VMAX - v)
    if free < acc:
        acc = free
    return acc


@njit(cache=True)
def simulate(lead_v, x0, v0, theta, T_i, sig_e, phi, eps, tau_ego, dt, bg_lag, cutin_step,
             cutin_gap, cutin_v, x_out, v_out):
    """One instance. lead_v: the scripted lead speed per step. x0, v0: initial front-bumper
    positions and speeds of all N_VEH vehicles, index 0 = lead. theta: the 5 shared IDM
    parameters; T_i: per-vehicle headway (the lead and ego entries are ignored). sig_e:
    per-vehicle noise scale. eps: pre-drawn N(0,1), shape (N_VEH, n).

    bg_lag > 0 is the delayed-IDM class arm: the WHOLE acceleration is computed from the state
    at t - bg_lag, as in bridge._sim_delayed (Amendment G arm M), not just the perceived gap.
    The ego carries a 0.2 s ACTUATION lag: the command is computed from the current state and
    applied bg steps later (K.1).
    cutin_step >= 0 moves vehicle EGO_IDX-1 into the gap ahead of the ego at that step.

    Returns (crash_step, first, second): crash_step < 0 if no collision; otherwise the step and
    the indices of the touching pair, `second` being the follower."""
    n = lead_v.shape[0]
    N = x0.shape[0]
    V0, S0, A, B = theta[0], theta[2], theta[3], theta[4]
    eta = np.zeros(N)
    c = math.sqrt(1.0 - phi * phi)
    lag = int(EGO_LAG_S / dt)
    cmd = np.zeros(n + 1)
    for i in range(N):
        x_out[i, 0] = x0[i]
        v_out[i, 0] = v0[i]
        eta[i] = sig_e[i] * eps[i, 0]
    cmd[0] = _ego_acc(v0[EGO_IDX], v0[EGO_IDX - 1],
                      x0[EGO_IDX - 1] - x0[EGO_IDX] - LEN_CAR, tau_ego)
    if cmd[0] < EGO_A_MIN:
        cmd[0] = EGO_A_MIN
    elif cmd[0] > EGO_A_MAX:
        cmd[0] = EGO_A_MAX
    for t in range(n - 1):
        if t == cutin_step:                     # the cut-in event (K.1 F2)
            x_out[EGO_IDX - 1, t] = x_out[EGO_IDX, t] + LEN_CAR + cutin_gap
            v_out[EGO_IDX - 1, t] = cutin_v
        for i in range(N):
            if i == 0:
                vn = lead_v[t + 1]
                acc = (vn - v_out[0, t]) / dt
            else:
                v = v_out[i, t]
                if i == EGO_IDX:
                    s = x_out[i - 1, t] - x_out[i, t] - LEN_CAR
                    cc = _ego_acc(v, v_out[i - 1, t], s, tau_ego)
                    if cc < EGO_A_MIN:
                        cc = EGO_A_MIN
                    elif cc > EGO_A_MAX:
                        cc = EGO_A_MAX
                    cmd[t] = cc
                    acc = cmd[t - lag] if t >= lag else cmd[0]      # actuation delay
                else:
                    k = t - bg_lag
                    if k < 0:
                        k = 0
                    vd = v_out[i, k]                                # arm-M delayed IDM
                    sb = x_out[i - 1, k] - x_out[i, k] - LEN_CAR
                    acc = _idm_acc(vd, vd - v_out[i - 1, k], sb, V0, T_i[i], S0, A, B) + eta[i]
                    if acc < A_MIN:
                        acc = A_MIN
                    elif acc > A_MAX:
                        acc = A_MAX
                vn = v + acc * dt
                if vn < 0.0:
                    vn = 0.0
            x_out[i, t + 1] = x_out[i, t] + 0.5 * (v_out[i, t] + vn) * dt
            v_out[i, t + 1] = vn
            if i > 0:
                eta[i] = phi * eta[i] + sig_e[i] * c * eps[i, t + 1]
        for i in range(1, N):
            if x_out[i - 1, t + 1] - x_out[i, t + 1] - LEN_CAR <= 0.0:
                for j in range(N):
                    for tt in range(t + 2, n):
                        x_out[j, tt] = x_out[j, t + 1]
                        v_out[j, tt] = v_out[j, t + 1]
                return t + 1, i - 1, i
    return -1, -1, -1


@njit(cache=True)
def _metrics(x, v, ego, n_valid):
    """min TTC and RSS longitudinal violation fraction for the ego as the following vehicle,
    computed from trajectories only."""
    min_ttc = 1e9
    viol = 0
    tot = 0
    f, l = ego, ego - 1                   # the ego as the following vehicle
    for t in range(n_valid):
        s = x[l, t] - x[f, t] - LEN_CAR
        dv = v[f, t] - v[l, t]
        if dv > 1e-6 and s > 0.0:
            ttc = s / dv
            if ttc < min_ttc:
                min_ttc = ttc
        vr, vf = v[f, t], v[l, t]
        d = (vr * RSS_RHO + 0.5 * RSS_AMAX * RSS_RHO * RSS_RHO
             + (vr + RSS_RHO * RSS_AMAX) ** 2 / (2.0 * RSS_BMIN)
             - vf * vf / (2.0 * RSS_BMAX))
        if d < 0.0:
            d = 0.0
        tot += 1
        if s < d:
            viol += 1
    return min_ttc, viol, tot


def metrics_from_traces(x, v, crash_step, ego=EGO_IDX):
    """Verdict metrics. Receives trajectories and nothing else - no theta, by design (K.1)."""
    n_valid = x.shape[1] if crash_step < 0 else crash_step + 1
    min_ttc, viol, tot = _metrics(x, v, ego, n_valid)
    return {"crash": bool(crash_step >= 0), "min_ttc": float(min(min_ttc, 1e9)),
            "rss_violation_frac": float(viol / tot) if tot else 0.0,
            "crash_step": int(crash_step)}


# ---------------------------------------------------------------------------- scenarios
def lead_profile(kind, n, p):
    """The scripted lead speed, per step. p holds the drawn instance parameters."""
    t = np.arange(n) * DT
    if kind == "brake":
        v = np.full(n, p["v_cruise"])
        t0, rate, vend = p["t_brake"], p["rate"], p["v_end"]
        dec = np.clip(p["v_cruise"] - rate * (t - t0), vend, p["v_cruise"])
        v = np.where(t >= t0, dec, v)
    elif kind == "stopgo":
        v = p["v_cruise"] + p["amp"] * np.sin(2 * np.pi * (t - p["phase"]) / p["period"])
        v = np.clip(v, 0.5, None)
    elif kind == "cutin":
        v = np.full(n, p["v_cruise"])
        t0 = p["t_brake"]
        v = np.where(t >= t0, np.clip(p["v_cruise"] - p["rate"] * (t - t0), p["v_end"],
                                      p["v_cruise"]), v)
    else:
        raise ValueError(kind)
    return np.ascontiguousarray(v, dtype=np.float64)


def draw_instances(n_per_family=40, duration_s=30.0, seed=20261001):
    """The frozen scenario suite (K.1): 3 families x n_per_family instances."""
    rng = np.random.default_rng(seed)
    n = int(duration_s / DT)
    out = []
    for kind in ("brake", "cutin", "stopgo"):
        for i in range(n_per_family):
            vc = float(rng.uniform(18.0, 33.0))
            # severity ranges follow standard AEB test protocols (Euro NCAP CCRb uses lead
            # decelerations of 2 and 6 m/s^2; cut-ins at short gaps with a speed deficit)
            p = {"v_cruise": vc, "t_brake": float(rng.uniform(4.0, 12.0)),
                 "rate": float(rng.uniform(2.0, 6.0)),
                 "v_end": float(rng.uniform(0.0, max(0.0, vc - 8.0))),
                 "amp": float(rng.uniform(2.0, 8.0)), "period": float(rng.uniform(8.0, 20.0)),
                 "phase": float(rng.uniform(0.0, 10.0)),
                 "gap_scale": float(rng.uniform(0.6, 1.4)),
                 "cutin_gap": float(rng.uniform(4.0, 20.0)),
                 "cutin_dv": float(rng.uniform(-12.0, 0.0)),
                 "t_cutin": float(rng.uniform(5.0, 15.0))}
            out.append({"family": kind, "i": i, "n": n, "p": p,
                        "lead_v": lead_profile(kind, n, p)})
    return out


def initial_state(inst, theta_typ):
    """Initial positions and speeds: the string starts near its equilibrium spacing at the lead's
    cruise speed, scaled by the instance's gap factor. For the cut-in family the vehicle ahead of
    the ego starts abnormally close and slower (the inserted vehicle)."""
    p = inst["p"]
    v = np.full(N_VEH, p["v_cruise"], dtype=np.float64)
    s_eq = theta_typ[2] + theta_typ[1] * p["v_cruise"]
    gaps = np.full(N_VEH - 1, max(s_eq * p["gap_scale"], 3.0))
    if inst["family"] == "cutin":
        gaps[EGO_IDX - 1] = max(s_eq * 4.0, 60.0)       # the vehicle that will cut in starts
        v[EGO_IDX - 1] = p["v_cruise"]                  # far ahead and moves in at t_cutin
    x = np.zeros(N_VEH)
    for i in range(1, N_VEH):
        x[i] = x[i - 1] - LEN_CAR - gaps[i - 1]
    return np.ascontiguousarray(x), np.ascontiguousarray(v)


# ------------------------------------------------------------------------------------
# fused suite runner (Amendment K): one kernel over (instance, seed), parallel over runs
# ------------------------------------------------------------------------------------
@njit(cache=True)
def _run_one(lead_v, i0, i1, x0, v0, theta, T_med, T_log_sd, het_T, sig_eta, omega, phi,
             tau_ego, dt, bg_lag, t_seed, run_seed, cutin_step, cutin_gap, cutin_v, x, v):
    """One (instance, seed) run. The per-vehicle headways come from t_seed and the noise from
    run_seed, so heterogeneity can be redrawn while the process noise is held fixed, and a
    change of theta leaves both untouched (common random numbers, K.2)."""
    N = x.shape[0]
    n = i1 - i0
    T_i = np.empty(N)
    np.random.seed(t_seed)
    for i in range(N):
        if het_T == 1:
            tt = T_med * math.exp(T_log_sd * np.random.normal(0.0, 1.0))
            T_i[i] = 0.3 if tt < 0.3 else (3.0 if tt > 3.0 else tt)
        else:
            T_i[i] = T_med
    np.random.seed(run_seed)
    sig = np.empty(N)
    for i in range(N):
        sig[i] = sig_eta * math.exp(omega * np.random.normal(0.0, 1.0) - 0.5 * omega * omega)
    sig[0] = 0.0
    sig[EGO_IDX] = 0.0
    eps = np.empty((N, n))
    for i in range(N):
        for t in range(n):
            eps[i, t] = np.random.normal(0.0, 1.0)
    return simulate(lead_v[i0:i1], x0, v0, theta, T_i, sig, phi, eps, tau_ego, dt, bg_lag,
                    cutin_step, cutin_gap, cutin_v, x, v)


@njit(parallel=True, cache=True)
def run_suite(lead_v, off, X0, V0, theta, T_med, T_log_sd, het_T, sig_eta, omega, phi,
              tau_ego, dt, bg_lag, n_seeds, base_seed, t_base, cut_step, cut_gap, cut_v,
              crash, ttc, rss, other):
    """Every (instance, seed) pair. Writes, per run: `crash` = 1 only for an EGO-INVOLVED
    collision (the pre-registered outcome, §6/K.1), `other` = 1 for a collision elsewhere in
    the string, and the ego's min TTC and RSS violation fraction. Instance-major order.
    Results do not depend on the thread count: each run seeds its own streams.

    SEEDING. Each run's seed is `base + m*K_STRIDE + k`, a DENSE block of at most
    K_STRIDE*M consecutive-ish values starting at `base`. Callers hand out bases as distinct
    multiples of a block size larger than that span, so two streams can never share a seed -
    by construction, not by luck. The previous scheme spread each stream over 11 million
    values with coprime multipliers, which looks safer and is not: with 40 heterogeneity
    bases drawn from CRC32s, two of them landed exactly 7919*95 + 104729*4 apart and 20% of
    their draws were literally the same numbers (K.10)."""
    M = off.shape[0] - 1
    N = X0.shape[1]
    for r in prange(M * n_seeds):
        m = r // n_seeds
        k = r - m * n_seeds
        i0, i1 = off[m], off[m + 1]
        x = np.empty((N, i1 - i0))
        v = np.empty((N, i1 - i0))
        cs, first, second = _run_one(
            lead_v, i0, i1, X0[m], V0[m], theta, T_med, T_log_sd, het_T, sig_eta, omega, phi,
            tau_ego, dt, bg_lag, t_base + m * K_STRIDE + k,
            base_seed + m * K_STRIDE + k, cut_step[m], cut_gap[m], cut_v[m], x, v)
        n_valid = (i1 - i0) if cs < 0 else cs + 1
        mt, viol, tot = _metrics(x, v, EGO_IDX, n_valid)
        ego_hit = cs >= 0 and (first == EGO_IDX or second == EGO_IDX)
        crash[r] = 1.0 if ego_hit else 0.0
        other[r] = 1.0 if (cs >= 0 and not ego_hit) else 0.0
        ttc[r] = mt if mt < 1e8 else 60.0
        rss[r] = viol / tot if tot > 0 else 0.0
    return 0


def pack_instances(instances, theta_typ):
    """Flatten the frozen suite into the arrays run_suite wants."""
    lead = np.concatenate([i["lead_v"] for i in instances])
    off = np.r_[0, np.cumsum([i["n"] for i in instances])].astype(np.int64)
    X0 = np.empty((len(instances), N_VEH))
    V0 = np.empty((len(instances), N_VEH))
    for j, inst in enumerate(instances):
        X0[j], V0[j] = initial_state(inst, theta_typ)
    fam = np.array([i["family"] for i in instances])
    cs = np.full(len(instances), -1, dtype=np.int64)
    cg = np.zeros(len(instances)); cv = np.zeros(len(instances))
    for j, inst in enumerate(instances):
        if inst["family"] == "cutin":
            cs[j] = int(inst["p"]["t_cutin"] / DT)
            cg[j] = inst["p"]["cutin_gap"]
            cv[j] = max(inst["p"]["v_cruise"] + inst["p"]["cutin_dv"], 1.0)
    return (np.ascontiguousarray(lead), off, np.ascontiguousarray(X0),
            np.ascontiguousarray(V0), fam, cs, np.ascontiguousarray(cg),
            np.ascontiguousarray(cv))
