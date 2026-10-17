"""Intelligent Driver Model: follower simulation given a leader trace.

Dependencies: numpy only (deliberate - this must run on the user's Windows Python
without a scientific stack beyond numpy/scipy).

Convention throughout this project:
    s      bumper-to-bumper gap [m], strictly positive
    v      follower speed [m/s], clipped at >= 0
    v_lead leader speed [m/s]
    dv     approach rate = v - v_lead  (positive when closing)
    theta  = (v0, T, s0, a, b);  delta is fixed at 4 and is NOT a free parameter.
"""

from math import sqrt as _sqrt

import numpy as np

DELTA = 4.0

PARAM_NAMES = ("v0", "T", "s0", "a", "b")

# Bounds used for every fit in this project. Recorded here so that a bound hit is
# always detectable and never silent - a parameter resting on a bound breaks the
# chi-square asymptotics the confidence region relies on.
PARAM_BOUNDS = (
    (15.0, 45.0),   # v0  [m/s]
    (0.30, 3.00),   # T   [s]
    (0.50, 8.00),   # s0  [m]
    (0.30, 4.00),   # a   [m/s^2]
    (0.30, 6.00),   # b   [m/s^2]
)


def idm_accel(s, v, v_lead, theta):
    """IDM acceleration. Scalar or array inputs."""
    v0, T, s0, a, b = theta
    dv = v - v_lead
    s_star = s0 + v * T + (v * dv) / (2.0 * np.sqrt(a * b))
    # s_star must not go below s0: negative desired gap is not meaningful and is a
    # known IDM edge case when the leader is much faster.
    s_star = np.maximum(s_star, s0)
    return a * (1.0 - (v / v0) ** DELTA - (s_star / np.maximum(s, 1e-3)) ** 2)


def simulate_follower(v_lead, dt, theta, s_init, v_init,
                      s_floor=0.5, a_clip=(-9.0, 4.0)):
    """Integrate the follower forward through a given leader speed trace.

    Ballistic (semi-implicit Euler) integration, which is the standard choice for
    IDM and is stable at 25 Hz.

    Written with plain Python floats rather than numpy scalars in the inner loop:
    the loop is inherently sequential and cannot be vectorised, and numpy scalar
    arithmetic carries ~5x the per-operation overhead of a Python float. This
    function is called a few hundred times per likelihood optimisation and a few
    hundred thousand times per experiment, so the constant factor decides whether a
    study runs overnight or over a weekend.

    Returns
    -------
    s : ndarray, shape (n,)   gap at each step
    v : ndarray, shape (n,)   follower speed at each step
    ok : bool                 False if the gap hit the floor (unphysical / collision)
    """
    v0, T, s0, a, b = (float(x) for x in theta)
    inv_v0 = 1.0 / v0
    two_sqrt_ab = 2.0 * _sqrt(a * b)
    a_lo, a_hi = a_clip

    n = len(v_lead)
    vl = [float(x) for x in v_lead]
    s_out = [0.0] * n
    v_out = [0.0] * n
    s = float(s_init)
    v = float(v_init)
    s_out[0], v_out[0] = s, v
    ok = True

    for t in range(n - 1):
        vlt = vl[t]
        dv = v - vlt
        s_star = s0 + v * T + (v * dv) / two_sqrt_ab
        if s_star < s0:
            s_star = s0
        s_eff = s if s > 1e-3 else 1e-3
        ratio = s_star / s_eff
        x = v * inv_v0
        acc = a * (1.0 - x * x * x * x - ratio * ratio)
        if acc < a_lo:
            acc = a_lo
        elif acc > a_hi:
            acc = a_hi
        v_next = v + acc * dt
        if v_next < 0.0:
            v_next = 0.0
        s_next = s + (vlt - 0.5 * (v + v_next)) * dt
        if s_next < s_floor:
            s_next = s_floor
            ok = False
        s_out[t + 1] = s_next
        v_out[t + 1] = v_next
        s, v = s_next, v_next

    return np.asarray(s_out), np.asarray(v_out), ok


def simulate_follower_pos(u_lead, v_lead, dt, theta, x_init, v_init,
                          s_floor=0.5, a_clip=(-9.0, 4.0)):
    """Integrate the follower forward against an OBSERVED leader rear position.

    Step 3 (05_PREREGISTRATION_DRAFT.md B.4). This is the integrator used on real
    data: the gap is s(t) = u_lead(t) - x_F(t), so the leader enters through its
    observed position and its own speed-vs-position drift (DIARY Entry 21, dL: median
    0.15 m, p99 0.52 m over an episode) never enters the simulated gap. The leader's
    observed speed is used only in the IDM approach-rate term.

    u_lead : leader rear bumper position along the lane [m], shape (n,)
    v_lead : leader speed [m/s], shape (n,)
    x_init : follower front bumper position at t=0 [m]  (s_init = u_lead[0] - x_init)
    Same ballistic update and clipping as simulate_follower; with
    u_lead[t+1] = u_lead[t] + v_lead[t]*dt the two functions agree to rounding.

    Returns s, v, ok as simulate_follower.
    """
    v0, T, s0, a, b = (float(x) for x in theta)
    inv_v0 = 1.0 / v0
    two_sqrt_ab = 2.0 * _sqrt(a * b)
    a_lo, a_hi = a_clip

    n = len(v_lead)
    ul = [float(x) for x in u_lead]
    vl = [float(x) for x in v_lead]
    s_out = [0.0] * n
    v_out = [0.0] * n
    x = float(x_init)
    v = float(v_init)
    s = ul[0] - x
    s_out[0], v_out[0] = s, v
    ok = True

    for t in range(n - 1):
        dv = v - vl[t]
        s_star = s0 + v * T + (v * dv) / two_sqrt_ab
        if s_star < s0:
            s_star = s0
        s_eff = s if s > 1e-3 else 1e-3
        ratio = s_star / s_eff
        q = v * inv_v0
        acc = a * (1.0 - q * q * q * q - ratio * ratio)
        if acc < a_lo:
            acc = a_lo
        elif acc > a_hi:
            acc = a_hi
        v_next = v + acc * dt
        if v_next < 0.0:
            v_next = 0.0
        x = x + 0.5 * (v + v_next) * dt
        s_next = ul[t + 1] - x
        if s_next < s_floor:
            # keep the follower where the floor puts it, as simulate_follower does
            x = ul[t + 1] - s_floor
            s_next = s_floor
            ok = False
        s_out[t + 1] = s_next
        v_out[t + 1] = v_next
        s, v = s_next, v_next

    return np.asarray(s_out), np.asarray(v_out), ok


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def make_leader_trace(dt, duration, rng=None, kind="informative"):
    """Leader speed profile.

    'informative' is deliberately NON-equilibrium. Wang et al. (2021) show that
    equilibrium initial conditions and constant inputs destroy the identifiability
    of car-following parameters, so a synthetic coverage check run at equilibrium
    would be measuring the wrong thing.

    The profile has three phases, each targeting different parameters. This is
    experiment design in the Wang et al. sense, and the phases exist because the
    first version of this trace left v0 unidentified and the optimiser drove it onto
    its bound in every replication (see DIARY 2026-09-20, A0-pre-1 run 1).

        phase 1  leader faster than v0, gap opens   -> follower reaches free flow
                                                       identifies  v0, a
        phase 2  leader decelerates, gap closes     -> identifies  T, s0
        phase 3  hard braking event                 -> identifies  b

    A persistent low-amplitude oscillation runs throughout so the input is never
    constant.
    """
    n = int(round(duration / dt))
    t = np.arange(n) * dt
    if kind == "informative":
        f1, f2 = 0.30 * duration, 0.62 * duration      # phase boundaries
        v = np.full(n, 34.0)
        # phase 2: ramp down to 20 over 3 s
        v = v - 14.0 * _smoothstep((t - f1) / 3.0)
        # phase 3: hard brake to 11 then recover
        v = v - 9.0 * _smoothstep((t - f2) / 1.5) + 9.0 * _smoothstep((t - f2 - 5.0) / 2.5)
        # persistent excitation
        v = v + 0.8 * np.sin(2 * np.pi * t / 3.7) + 0.5 * np.sin(2 * np.pi * t / 1.9)
        v = np.maximum(v, 3.0)
    elif kind == "equilibrium":
        v = np.full(n, 28.0)
    else:
        raise ValueError(kind)
    return t, v
